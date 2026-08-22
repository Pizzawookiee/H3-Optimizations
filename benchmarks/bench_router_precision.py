'''Compare BF16 and FP32 H3 sparse-router decisions on identical Q/K.'''

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACK_ROOT.parents[1]
BENCHMARK_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PACK_ROOT))


Q_TILE = 128
KV_TILE = 64
HEAD_DIM = 128
ARM_ORDER = (
    'bf16_pool_bf16_score',
    'bf16_pool_fp32_score',
    'fp32_pool_fp32_score',
)
DEFAULT_SEQUENCE = 54006
DEFAULT_HEADS = 56
DEFAULT_VIDEO_START = 256
DEFAULT_VIDEO_BUDGET = 0.3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Compare current BF16 H3 routing, FP32 scoring of BF16 summaries, '
            'and PlagueKind-style FP32 pooling plus FP32 scoring.'
        )
    )
    parser.add_argument('--sequence', type=int, default=DEFAULT_SEQUENCE)
    parser.add_argument('--heads', type=int, default=DEFAULT_HEADS)
    parser.add_argument('--video-start', type=int, default=DEFAULT_VIDEO_START)
    parser.add_argument('--video-budget', type=float, default=DEFAULT_VIDEO_BUDGET)
    parser.add_argument('--checkpoint')
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--projection-chunk-rows', type=int, default=4096)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if args.sequence <= 0 or args.heads <= 0:
        parser.error('sequence and heads must be positive')
    if not 0 < args.video_start < args.sequence:
        parser.error('--video-start must be inside the sequence')
    if not 0.01 <= args.video_budget < 1.0:
        parser.error('--video-budget must be in [0.01, 1)')
    if args.block < 0 or args.device < 0:
        parser.error('block and device must be non-negative')
    if args.projection_chunk_rows <= 0:
        parser.error('--projection-chunk-rows must be positive')
    if args.warmup < 0 or args.iterations <= 0:
        parser.error('iteration arguments are invalid')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU preflight'
        )
    return args


def geometry(sequence, video_start, video_budget):
    q_tiles = math.ceil(sequence / Q_TILE)
    kv_tiles = math.ceil(sequence / KV_TILE)
    pure_q_start = math.ceil(video_start / Q_TILE)
    pure_kv_start = math.ceil(video_start / KV_TILE)
    pure_q_tiles = q_tiles - pure_q_start
    pure_kv_tiles = kv_tiles - pure_kv_start
    retained = min(
        pure_kv_tiles,
        max(1, math.ceil(float(video_budget) * pure_kv_tiles)),
    )
    return {
        'q_tiles': q_tiles,
        'kv_tiles': kv_tiles,
        'pure_q_start': pure_q_start,
        'pure_kv_start': pure_kv_start,
        'pure_q_tiles': pure_q_tiles,
        'pure_kv_tiles': pure_kv_tiles,
        'retained': retained,
    }


def mean_pool(torch, x, block, dtype=None):
    sequence = x.shape[-2]
    full = sequence // block
    remainder = sequence % block
    pieces = []
    if full:
        pieces.append(
            x[..., :full * block, :]
            .reshape(*x.shape[:-2], full, block, x.shape[-1])
            .mean(dim=-2, dtype=dtype)
        )
    if remainder:
        pieces.append(
            x[..., full * block:, :].mean(
                dim=-2,
                keepdim=True,
                dtype=dtype,
            )
        )
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)


def route_arm(torch, q, k, route_geometry, name, *, cutoff_margin=False):
    if name not in ARM_ORDER:
        raise ValueError('unknown router precision arm: %s' % name)
    pool_dtype = torch.float32 if name == 'fp32_pool_fp32_score' else None
    q_summary = mean_pool(torch, q, Q_TILE, dtype=pool_dtype)
    k_summary = mean_pool(torch, k, KV_TILE, dtype=pool_dtype)
    q_video = q_summary[..., route_geometry['pure_q_start']:, :]
    k_video = k_summary[..., route_geometry['pure_kv_start']:, :]
    if name != 'bf16_pool_bf16_score':
        q_video = q_video.float()
        k_video = k_video.float()
    scores = torch.matmul(q_video, k_video.transpose(-1, -2))
    retained = route_geometry['retained']
    if cutoff_margin:
        ranked = torch.topk(scores, retained + 1, dim=-1, sorted=True)
        margin = ranked.values[..., retained - 1] - ranked.values[..., retained]
        return ranked.indices[..., :retained], margin
    return torch.topk(scores, retained, dim=-1).indices


def compare_routes(torch, left, right, kv_tiles):
    if left.shape != right.shape:
        raise ValueError('route shapes differ')
    left_mask = torch.zeros(
        (*left.shape[:-1], kv_tiles),
        dtype=torch.bool,
        device=left.device,
    )
    right_mask = torch.zeros_like(left_mask)
    left_mask.scatter_(-1, left.long(), True)
    right_mask.scatter_(-1, right.long(), True)
    substitutions = (left_mask ^ right_mask).sum(dim=-1) // 2
    changed = substitutions > 0
    retained = left.shape[-1]
    intersection = retained - substitutions
    union = retained + substitutions
    jaccard = intersection.float() / union.float()
    total_rows = substitutions.numel()
    total_selected = total_rows * retained
    substituted = int(substitutions.sum().item())
    return {
        'changed_rows': int(changed.sum().item()),
        'changed_row_fraction': float(changed.float().mean().item()),
        'total_rows': int(total_rows),
        'substituted_blocks': substituted,
        'selected_blocks': int(total_selected),
        'substituted_block_fraction': substituted / total_selected,
        'max_substitutions_per_row': int(substitutions.max().item()),
        'mean_jaccard': float(jaccard.mean().item()),
        'min_jaccard': float(jaccard.min().item()),
    }


def margin_metrics(torch, margin):
    values = margin.detach().float().flatten().cpu()
    return {
        'minimum': float(values.min().item()),
        'p01': float(torch.quantile(values, 0.01).item()),
        'median': float(values.median().item()),
        'mean': float(values.mean().item()),
    }


def benchmark_arms(torch, q, k, route_geometry, warmup, iterations, device):
    for _ in range(warmup):
        for name in ARM_ORDER:
            selected = route_arm(torch, q, k, route_geometry, name)
            torch.cuda.synchronize(device)
            del selected
    torch.cuda.empty_cache()

    samples = {name: [] for name in ARM_ORDER}
    peaks = {name: [] for name in ARM_ORDER}
    output_bytes = {name: [] for name in ARM_ORDER}
    for iteration in range(iterations):
        order = ARM_ORDER if iteration % 2 == 0 else tuple(reversed(ARM_ORDER))
        for name in order:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            selected = route_arm(torch, q, k, route_geometry, name)
            stop.record()
            stop.synchronize()
            samples[name].append(float(start.elapsed_time(stop)))
            peaks[name].append(
                int(torch.cuda.max_memory_allocated(device) - baseline)
            )
            output_bytes[name].append(
                int(torch.cuda.memory_allocated(device) - baseline)
            )
            del selected
            torch.cuda.synchronize(device)
    return {
        name: {
            'median_ms': statistics.median(samples[name]),
            'min_ms': min(samples[name]),
            'samples_ms': samples[name],
            'peak_allocated_bytes': max(peaks[name]),
            'output_live_bytes': max(output_bytes[name]),
        }
        for name in ARM_ORDER
    }


def random_qk(torch, args, device):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (1, args.heads, args.sequence, HEAD_DIM)
    q = torch.randn(
        shape,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        shape,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    return q, k, {'kind': 'random_bf16'}


def checkpoint_qk(torch, args, device):
    from bench_chunked_kitchen_qkv import (
        build_attention,
        make_rope,
        project_qkv,
        resolve_checkpoint,
    )

    checkpoint = resolve_checkpoint(args.checkpoint)
    module, hidden, prefix = build_attention(
        torch,
        checkpoint,
        args.block,
        args.epsilon,
        device,
    )
    if module.heads != args.heads:
        raise ValueError(
            '--heads=%d does not match checkpoint heads=%d'
            % (args.heads, module.heads)
        )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(
        (args.sequence, hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope = make_rope(torch, args.sequence, device)
    shape = (1, args.heads, args.sequence, HEAD_DIM)
    q_out = torch.empty(shape, dtype=torch.bfloat16, device=device)
    k_out = torch.empty_like(q_out)
    for start in range(0, args.sequence, args.projection_chunk_rows):
        stop = min(args.sequence, start + args.projection_chunk_rows)
        q, k, v = project_qkv(
            torch,
            module,
            x[start:stop],
            rope[:, start:stop],
        )
        q_out[0, :, start:stop].copy_(q.transpose(0, 1))
        k_out[0, :, start:stop].copy_(k.transpose(0, 1))
        del q, k, v
    del x, rope, module
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return q_out, k_out, {
        'kind': 'checkpoint_projected_synthetic_bf16',
        'checkpoint': str(checkpoint),
        'checkpoint_prefix': prefix,
        'block': args.block,
        'projection_chunk_rows': args.projection_chunk_rows,
    }


def theoretical_bytes(route_geometry, heads):
    score_elements = (
        heads
        * route_geometry['pure_q_tiles']
        * route_geometry['pure_kv_tiles']
    )
    summary_elements = (
        heads
        * (route_geometry['q_tiles'] + route_geometry['kv_tiles'])
        * HEAD_DIM
    )
    return {
        'bf16_score': score_elements * 2,
        'fp32_score': score_elements * 4,
        'bf16_summaries': summary_elements * 2,
        'fp32_summaries': summary_elements * 4,
        'fp32_increment_over_current': (score_elements + summary_elements) * 2,
    }


def main(argv=None):
    args = parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    device = torch.device('cuda', args.device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('highest')
    route_geometry = geometry(
        args.sequence,
        args.video_start,
        args.video_budget,
    )
    if route_geometry['retained'] >= route_geometry['pure_kv_tiles']:
        raise SystemExit('the selected geometry does not exercise sparse top-k')

    q, k, source = (
        checkpoint_qk(torch, args, device)
        if args.checkpoint
        else random_qk(torch, args, device)
    )
    torch.cuda.synchronize(device)
    input_allocated = int(torch.cuda.memory_allocated(device))
    timing = benchmark_arms(
        torch,
        q,
        k,
        route_geometry,
        args.warmup,
        args.iterations,
        device,
    )

    routes = {}
    margins = {}
    for name in ARM_ORDER:
        selected, margin = route_arm(
            torch,
            q,
            k,
            route_geometry,
            name,
            cutoff_margin=True,
        )
        routes[name] = selected
        margins[name] = margin_metrics(torch, margin)
        del margin
    comparisons = {
        '%s_vs_%s' % (left, right): compare_routes(
            torch,
            routes[left],
            routes[right],
            route_geometry['pure_kv_tiles'],
        )
        for left, right in (
            (ARM_ORDER[0], ARM_ORDER[1]),
            (ARM_ORDER[1], ARM_ORDER[2]),
            (ARM_ORDER[0], ARM_ORDER[2]),
        )
    }

    properties = torch.cuda.get_device_properties(device)
    result = {
        'environment': {
            'torch': torch.__version__,
            'device': properties.name,
            'compute_capability': [properties.major, properties.minor],
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
        },
        'input': source,
        'config': {
            'sequence': args.sequence,
            'heads': args.heads,
            'head_dim': HEAD_DIM,
            'q_tile': Q_TILE,
            'kv_tile': KV_TILE,
            'video_start': args.video_start,
            'video_budget': args.video_budget,
            'seed': args.seed,
            'warmup': args.warmup,
            'iterations': args.iterations,
        },
        'geometry': route_geometry,
        'input_allocated_bytes': input_allocated,
        'theoretical_bytes': theoretical_bytes(route_geometry, args.heads),
        'timing': timing,
        'cutoff_margin': margins,
        'comparisons': comparisons,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).resolve()
        if not output.parent.is_dir():
            raise SystemExit('output directory does not exist: %s' % output.parent)
        output.write_text(serialized + '\n', encoding='utf-8')
    if args.json:
        print(serialized)
    else:
        print(
            'H3 router precision: %d heads x %d rows, %.1f%% video KV'
            % (args.heads, args.sequence, args.video_budget * 100.0)
        )
        for name in ARM_ORDER:
            details = timing[name]
            print(
                '  %s: %.3f ms, peak %.1f MiB'
                % (
                    name,
                    details['median_ms'],
                    details['peak_allocated_bytes'] / (1024 ** 2),
                )
            )
        for name, details in comparisons.items():
            print(
                '  %s: %d/%d rows changed, %d block substitutions'
                % (
                    name,
                    details['changed_rows'],
                    details['total_rows'],
                    details['substituted_blocks'],
                )
            )
        if args.output:
            print('wrote %s' % Path(args.output).resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
