'''Compare Sparse Kitchen kernels on retained full and chunked QKV carriers.'''

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = Path(__file__).resolve().parent
for _root in (str(BENCHMARK_ROOT), str(COMFY_ROOT), str(PACK_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from bench_chunked_kitchen_qkv import make_rope, resolve_checkpoint  # noqa: E402
from bench_h3_block import build_block  # noqa: E402
from bench_h3_block_allocator import make_layout  # noqa: E402


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Compare one sparse kernel on full and chunked Kitchen carriers.'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--qkv-chunk-rows', type=int, default=4096)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=54_006)
    parser.add_argument('--video-start', type=int, default=256)
    parser.add_argument('--video-budget', type=float, default=0.3)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--seed', type=int, default=9876)
    parser.add_argument('--output')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if args.sequence <= 0 or not 0 < args.video_start < args.sequence:
        parser.error('sequence/video-start arguments are invalid')
    if args.qkv_chunk_rows <= 0 or args.qkv_chunk_rows % 128:
        parser.error('--qkv-chunk-rows must be a positive multiple of 128')
    if args.warmup < 0 or args.iterations <= 0:
        parser.error('iteration arguments are invalid')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU preflight'
        )
    return args


def tensor_contract(torch, actual, reference):
    return {
        'exact': bool(torch.equal(actual, reference)),
        'shape': list(actual.shape),
        'stride': list(actual.stride()),
        'reference_stride': list(reference.stride()),
        'dtype': str(actual.dtype),
        'reference_dtype': str(reference.dtype),
        'data_ptr_mod_256': int(actual.data_ptr()) % 256,
        'reference_data_ptr_mod_256': int(reference.data_ptr()) % 256,
    }


def benchmark_kernel(torch, device, backend, prepared, iterations):
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        output = backend.execute(prepared)
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
        del output
    return samples


def summarize(samples):
    return {
        'median_ms': statistics.median(samples),
        'p10_ms': percentile(samples, 0.1),
        'p90_ms': percentile(samples, 0.9),
        'min_ms': min(samples),
        'max_ms': max(samples),
        'samples_ms': samples,
    }


def main(argv=None):
    args = parse_args(argv)

    import torch

    from h3_optimizations.attention.sparse.config import (
        HybridSparseConfig,
        MODE_SAGE128,
        MODE_SAGE128_FUSED_QKV,
    )
    from h3_optimizations.attention.sparse.kitchen_sparse import (
        SparseKitchenBackend,
        preflight_sparse_kitchen,
    )
    from h3_optimizations.attention_forward import project_qkv, to_hnd
    from h3_optimizations.kitchen_qkv import ChunkedKitchenQKVProjector
    from h3_optimizations.runtime.context import RUNTIME_KEY, RuntimeSnapshot

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    device = torch.device('cuda')
    checkpoint = resolve_checkpoint(args.checkpoint)
    block, prefix, hidden, _t_dim = build_block(
        torch, checkpoint, args.block, device
    )
    kitchen = preflight_sparse_kitchen(
        cuda_available=lambda: True,
        capability_getter=lambda: torch.cuda.get_device_capability(device),
    )
    layout = make_layout(args.sequence, args.video_start)
    options = {
        RUNTIME_KEY: RuntimeSnapshot(
            request_id=0,
            step_index=0,
            total_steps=1,
            layout=layout,
            compute_dtype=torch.bfloat16,
            device=device,
        )
    }
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(
        (args.sequence, hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope = make_rope(torch, args.sequence, device)

    full_backend = SparseKitchenBackend(
        HybridSparseConfig(
            mode=MODE_SAGE128,
            video_budget=args.video_budget,
        ),
        kitchen=kitchen,
    )
    q, k, v = to_hnd(*project_qkv(block.attn, x, rope))
    prepared_full = full_backend.prepare(
        q,
        k,
        v,
        layer_index=args.block,
        transformer_options=options,
    )
    del q, k, v
    torch.cuda.synchronize(device)

    projector = ChunkedKitchenQKVProjector(
        chunk_rows=args.qkv_chunk_rows,
        routing_summaries=True,
    )
    chunked_backend = SparseKitchenBackend(
        HybridSparseConfig(
            mode=MODE_SAGE128_FUSED_QKV,
            video_budget=args.video_budget,
        ),
        kitchen=kitchen,
        projector=projector,
    )
    projected = projector.try_project(
        block.attn,
        x,
        rope,
        layer_index=args.block,
        transformer_options=options,
    )
    if projected is None:
        raise SystemExit('the chunked Kitchen projector declined the test input')
    prepared_chunked = chunked_backend.prepare_projected(
        projected,
        layer_index=args.block,
        transformer_options=options,
    )
    del projected
    torch.cuda.synchronize(device)

    carrier_fields = ('q', 'k', 'v', 'q_scale', 'k_scale', 'v_scale')
    parity = {
        name: tensor_contract(
            torch,
            getattr(prepared_chunked.quantized, name),
            getattr(prepared_full.quantized, name),
        )
        for name in carrier_fields
    }
    parity['route_indices'] = tensor_contract(
        torch, prepared_chunked.route.indices, prepared_full.route.indices
    )
    parity['route_counts'] = tensor_contract(
        torch, prepared_chunked.route.counts, prepared_full.route.counts
    )
    parity['metadata_exact'] = (
        prepared_chunked.quantized.original_head_dim
        == prepared_full.quantized.original_head_dim
        and prepared_chunked.quantized.input_dtype
        == prepared_full.quantized.input_dtype
        and prepared_chunked.quantized.attention_scale
        == prepared_full.quantized.attention_scale
        and prepared_chunked.quantized.cta_k
        == prepared_full.quantized.cta_k
        and prepared_chunked.route.q_tile == prepared_full.route.q_tile
        and prepared_chunked.route.kv_tile == prepared_full.route.kv_tile
        and prepared_chunked.route.encoding == prepared_full.route.encoding
    )

    for _ in range(args.warmup):
        for backend, prepared in (
            (full_backend, prepared_full),
            (chunked_backend, prepared_chunked),
        ):
            output = backend.execute(prepared)
            torch.cuda.synchronize(device)
            del output

    samples = {'full_qkv': [], 'chunked_qkv': []}
    for iteration in range(args.iterations):
        order = (
            ('full_qkv', 'chunked_qkv')
            if iteration % 2 == 0
            else ('chunked_qkv', 'full_qkv')
        )
        for name in order:
            backend, prepared = (
                (full_backend, prepared_full)
                if name == 'full_qkv'
                else (chunked_backend, prepared_chunked)
            )
            samples[name].extend(
                benchmark_kernel(torch, device, backend, prepared, 1)
            )

    result = {
        'checkpoint': str(checkpoint),
        'checkpoint_prefix': prefix,
        'block': int(args.block),
        'sequence': int(args.sequence),
        'video_start': int(args.video_start),
        'video_budget': float(args.video_budget),
        'qkv_chunk_rows': int(args.qkv_chunk_rows),
        'parity': parity,
        'kernel': {name: summarize(values) for name, values in samples.items()},
        'gpu': {
            'name': torch.cuda.get_device_name(device),
            'capability': list(torch.cuda.get_device_capability(device)),
        },
        'torch_version': torch.__version__,
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
        for name, details in result['kernel'].items():
            print(
                '%s: %.3f ms (p10 %.3f, p90 %.3f)'
                % (
                    name,
                    details['median_ms'],
                    details['p10_ms'],
                    details['p90_ms'],
                )
            )
        exact = all(
            value['exact']
            for key, value in parity.items()
            if key != 'metadata_exact'
        ) and parity['metadata_exact']
        print('carrier and route exact: %s' % exact)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
