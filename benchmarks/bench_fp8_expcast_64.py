"""Compare native INT8 against 64x64 FP8, ExpCast, and V-Smooth arms."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import pathlib
import statistics
import sys
import time

import torch

PACK_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from h3_optimizations.native.fp8_expcast import (  # noqa: E402
    fp8_attention_64_from_prequantized,
    pack_fp8_values,
    pack_v_smooth_values,
)
from h3_optimizations.native import loader as native_loader  # noqa: E402
from h3_optimizations.native.int8_attention import (  # noqa: E402
    BlockSparseRoute,
    block_sparse_int8_attention_tiles_from_prequantized,
    prequantize_int8_attention,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=pathlib.Path)
    parser.add_argument('--library', type=pathlib.Path)
    parser.add_argument('--tokens', type=int, default=4096)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--clusters', type=int)
    parser.add_argument('--kmeans-iterations', type=int, default=4)
    parser.add_argument('--grouping', choices=('kmeans', 'known'), default='kmeans')
    parser.add_argument('--skip-reference', action='store_true')
    parser.add_argument('--arms', default='all')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', type=pathlib.Path)
    return parser.parse_args()


def load_inputs(args, device):
    if args.input is not None:
        saved = torch.load(args.input, map_location=device, weights_only=True)
        q, k, v = (saved[name].to(device=device, dtype=torch.bfloat16) for name in ('q', 'k', 'v'))
        if q.ndim != 4 or q.shape[-1] != 128 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError('input q/k/v must share [batch, heads, tokens, 128]')
        return q.contiguous(), k.contiguous(), v.contiguous(), 'capture', None

    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (1, args.heads, args.tokens, 128)
    q = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
    # A low-rank token component makes this a useful V-Smooth smoke test. A
    # captured post-RoPE tensor remains the evidence-producing mode.
    groups = max(1, math.ceil(args.tokens / 128))
    centers = torch.randn(
        1, args.heads, groups, 128,
        generator=generator, device=device, dtype=torch.bfloat16,
    )
    labels = torch.randint(
        groups, (1, args.heads, args.tokens), generator=generator, device=device
    )
    center_rows = centers.gather(
        2, labels.unsqueeze(-1).expand(-1, -1, -1, 128)
    )
    noise = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
    v = center_rows + noise * 0.35
    return q, k, v.contiguous(), 'synthetic_clustered', torch.argsort(
        labels, dim=2, stable=True
    )


def kmeans_permutation(values, clusters, iterations, chunk_rows=2048):
    """Deterministic Lloyd grouping; the paper leaves k and iteration count open."""
    batch, heads, tokens, width = values.shape
    clusters = max(1, min(int(clusters), tokens))
    permutations = []
    for b in range(batch):
        head_permutations = []
        for h in range(heads):
            points = values[b, h].float()
            seeds = torch.linspace(
                0, tokens - 1, clusters, device=values.device
            ).round().to(torch.int64)
            centroids = points.index_select(0, seeds).clone()
            labels = torch.empty(tokens, dtype=torch.int64, device=values.device)
            for _ in range(iterations):
                centroid_norm = centroids.square().sum(dim=1)
                for start in range(0, tokens, chunk_rows):
                    rows = points[start:start + chunk_rows]
                    distance = (
                        rows.square().sum(dim=1, keepdim=True)
                        + centroid_norm.unsqueeze(0)
                        - 2.0 * rows @ centroids.t()
                    )
                    labels[start:start + rows.shape[0]] = distance.argmin(dim=1)
                sums = torch.zeros_like(centroids)
                sums.scatter_add_(0, labels.unsqueeze(1).expand(-1, width), points)
                counts = torch.bincount(labels, minlength=clusters).float().unsqueeze(1)
                centroids = torch.where(counts > 0, sums / counts.clamp_min(1), centroids)
            head_permutations.append(torch.argsort(labels, stable=True))
        permutations.append(torch.stack(head_permutations))
    return torch.stack(permutations)


def permute_rows(tensor, permutation):
    return tensor.gather(2, permutation.unsqueeze(-1).expand_as(tensor)).contiguous()


def full_route(batch, heads, q_length, kv_length, device):
    q_tiles = math.ceil(q_length / 64)
    kv_tiles = math.ceil(kv_length / 64)
    indices = torch.arange(kv_tiles, dtype=torch.int32, device=device)
    indices = indices.view(1, 1, 1, kv_tiles).expand(
        batch, heads, q_tiles, kv_tiles
    ).contiguous()
    counts = torch.full(
        (batch, heads, q_tiles), kv_tiles, dtype=torch.int32, device=device
    )
    live = torch.full((kv_tiles,), 64, dtype=torch.int32, device=device)
    live[-1] = kv_length - (kv_tiles - 1) * 64
    return BlockSparseRoute(indices, counts, 64, 64), live


def event_ms(function):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = function()
    end.record()
    end.synchronize()
    return result, start.elapsed_time(end)


def benchmark_arms(functions, warmup, iterations, retain_outputs):
    for function in functions.values():
        for _ in range(warmup):
            function()
    torch.cuda.synchronize()
    names = tuple(functions)
    samples = {name: [] for name in names}
    outputs = {}
    for iteration in range(iterations):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            output, elapsed = event_ms(functions[name])
            samples[name].append(elapsed)
            if retain_outputs:
                outputs[name] = output
    return outputs, {
        name: {
            'median_ms': statistics.median(values),
            'min_ms': min(values),
            'samples_ms': values,
        }
        for name, values in samples.items()
    }


def error_metrics(output, reference):
    output = output.float()
    reference = reference.float()
    delta = output - reference
    return {
        'relative_l2': (delta.norm() / reference.norm()).item(),
        'norm_ratio': (output.norm() / reference.norm()).item(),
        'rmse': delta.square().mean().sqrt().item(),
        'max_abs': delta.abs().max().item(),
        'cosine': torch.nn.functional.cosine_similarity(
            output.flatten(), reference.flatten(), dim=0
        ).item(),
    }


def main():
    args = parse_args()
    if args.library is not None:
        native_loader._library = native_loader._bind(
            ctypes.CDLL(str(args.library.resolve()))
        )
        native_loader._load_error = None
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    device = torch.device('cuda')
    if torch.cuda.get_device_capability(device) != (8, 9):
        raise SystemExit('this experimental build requires SM89')

    q, k, v, source, known_permutation = load_inputs(args, device)
    batch, heads, tokens, _ = q.shape
    clusters = args.clusters or math.ceil(tokens / 128)
    scale = 128 ** -0.5

    prep_start = time.perf_counter()
    if args.grouping == 'known':
        if known_permutation is None:
            raise ValueError('--grouping known is available only for synthetic input')
        permutation, clustering_ms = event_ms(lambda: known_permutation.clone())
    else:
        permutation, clustering_ms = event_ms(
            lambda: kmeans_permutation(v, clusters, args.kmeans_iterations)
        )
    k_grouped = permute_rows(k, permutation)
    v_grouped = permute_rows(v, permutation)
    torch.cuda.synchronize()
    grouping_wall_ms = (time.perf_counter() - prep_start) * 1000.0

    preparers = {
        'int8_qkv_standard': lambda: prequantize_int8_attention(
            q, k, v, scale=scale, cta_k=64
        ),
        'int8_qkv_grouped': lambda: prequantize_int8_attention(
            q, k_grouped, v_grouped, scale=scale, cta_k=64
        ),
        'fp8_v_global': lambda: pack_fp8_values(v),
        'fp8_v_smooth': lambda: pack_v_smooth_values(v_grouped),
    }
    for prepare in preparers.values():
        prepare()
    torch.cuda.synchronize()
    prepared = {}
    preparation_ms = {}
    for name, prepare in preparers.items():
        prepared[name], preparation_ms[name] = event_ms(prepare)
    standard_qkv = prepared['int8_qkv_standard']
    grouped_qkv = prepared['int8_qkv_grouped']
    fp8_values = prepared['fp8_v_global']
    v_smooth_values = prepared['fp8_v_smooth']

    route, live = full_route(batch, heads, tokens, tokens, device)
    grouped_route, grouped_live = full_route(batch, heads, tokens, tokens, device)
    reference = None if args.skip_reference else torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=scale
    )

    functions = {
        'int8_64x64': lambda: block_sparse_int8_attention_tiles_from_prequantized(
            standard_qkv, route, live
        ),
        'fp8_exp2_64x64': lambda: fp8_attention_64_from_prequantized(
            standard_qkv, fp8_values, route, live, expcast=False
        ),
        'fp8_sage_denom_64x64': lambda: fp8_attention_64_from_prequantized(
            standard_qkv, fp8_values, route, live, expcast=False,
            pre_fp8_denom=True,
        ),
        'fp8_expcast_64x64': lambda: fp8_attention_64_from_prequantized(
            standard_qkv, fp8_values, route, live, expcast=True
        ),
        'fp8_v_smooth_exp2_64x64': lambda: fp8_attention_64_from_prequantized(
            grouped_qkv, v_smooth_values, grouped_route, grouped_live,
            expcast=False,
        ),
        'fp8_v_smooth_expcast_64x64': lambda: fp8_attention_64_from_prequantized(
            grouped_qkv, v_smooth_values, grouped_route, grouped_live,
            expcast=True,
        ),
    }
    if args.arms != 'all':
        selected = [name.strip() for name in args.arms.split(',') if name.strip()]
        unknown = [name for name in selected if name not in functions]
        if unknown:
            raise ValueError('unknown arms: %s' % ', '.join(unknown))
        functions = {name: functions[name] for name in selected}

    outputs, arms = benchmark_arms(
        functions, args.warmup, args.iterations, reference is not None
    )
    int8_output = outputs.get('int8_64x64')
    for name, output in outputs.items():
        if reference is not None:
            arms[name].update(error_metrics(output, reference))
        if int8_output is not None:
            arms[name]['relative_l2_vs_int8'] = (
                (output.float() - int8_output.float()).norm()
                / int8_output.float().norm()
            ).item()

    report = {
        'device': torch.cuda.get_device_name(device),
        'capability': list(torch.cuda.get_device_capability(device)),
        'source': source,
        'shape': list(q.shape),
        'tile': [64, 64],
        'v_smooth': {
            'block_rows': 128,
            'clusters': clusters,
            'kmeans_iterations': args.kmeans_iterations,
            'grouping': args.grouping,
            'paper_spec_note': 'The paper specifies online k-means but not k or Lloyd iterations.',
        },
        'preprocessing_ms': {
            'kmeans_cuda_events': clustering_ms,
            'kmeans_and_python_wall': grouping_wall_ms,
            **preparation_ms,
        },
        'arms': arms,
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
