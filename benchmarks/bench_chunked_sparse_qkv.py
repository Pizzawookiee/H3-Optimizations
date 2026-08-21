'''Compare standard, fused, and chunked Sparse Sage H3 QKV production.'''

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace


PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PACK_ROOT))

from bench_chunked_kitchen_qkv import (  # noqa: E402
    benchmark_case,
    build_attention,
    cuda_kernel_names,
    make_rope,
    output_error_metrics,
    resolve_checkpoint,
    weight_contract,
)


DEFAULT_SEQUENCE = 54006
DEFAULT_CHUNK_ROWS = 4096
DEFAULT_PARITY_SEQUENCE = 5003


class BenchmarkAttention:
    pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Compare standard H3 QKV, legacy native-carrier fused QKV, and '
            'chunked Kitchen-backed QKV through real Sparse Sage attention.'
        )
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--kitchen-source', required=True)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=DEFAULT_SEQUENCE)
    parser.add_argument('--chunk-rows', type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument(
        '--projector-sweep',
        default='',
        help='comma-separated chunk rows for a projected-carrier-only round robin',
    )
    parser.add_argument('--parity-sequence', type=int, default=DEFAULT_PARITY_SEQUENCE)
    parser.add_argument('--video-start', type=int, default=256)
    parser.add_argument('--video-budget', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--skip-parity', action='store_true')
    parser.add_argument('--skip-kernel-trace', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if args.sequence <= 0 or args.parity_sequence <= 0:
        parser.error('sequence arguments must be positive')
    if args.chunk_rows <= 0 or args.chunk_rows % 128:
        parser.error('--chunk-rows must be a positive multiple of 128')
    try:
        args.projector_sweep = tuple(
            int(value.strip())
            for value in args.projector_sweep.split(',')
            if value.strip()
        )
    except ValueError:
        parser.error('--projector-sweep must contain integers')
    if any(value <= 0 or value % 128 for value in args.projector_sweep):
        parser.error('--projector-sweep rows must be positive multiples of 128')
    if len(set(args.projector_sweep)) != len(args.projector_sweep):
        parser.error('--projector-sweep must not contain duplicates')
    if args.warmup < 0 or args.iterations <= 0:
        parser.error('iteration arguments are invalid')
    if not 0 < args.video_start < args.sequence:
        parser.error('--video-start must be inside the sequence')
    if not 0.01 <= args.video_budget <= 1.0:
        parser.error('--video-budget must be in [0.01, 1]')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU preflight'
        )
    return args


def make_layout(sequence, video_start):
    video_start = min(int(video_start), int(sequence) - 1)
    text_stop = min(128, video_start)
    return SimpleNamespace(
        seq_len=int(sequence),
        video_range=(video_start, int(sequence)),
        segments=(
            (0, text_stop, 'text'),
            (text_stop, video_start, 'audio'),
            (video_start, int(sequence), 'video'),
        ),
        video_shape=(1, 1, int(sequence) - video_start),
        audio_t=max(0, (video_start - text_stop) // 2),
    )


def mask_metadata(mask, layer_index, heads):
    metadata = mask.as_dict()
    metadata.update(
        {
            'layer': int(layer_index),
            'sparse_sage_heads': int(heads),
            'total_q_video_tiles': int(mask.pure_video_q_tiles) * int(heads),
        }
    )
    return metadata


def project_standard(module, x, rope):
    from h3_optimizations.attention_forward import project_qkv, to_hnd

    return to_hnd(*project_qkv(module, x, rope))


def prepare_standard(
    module,
    x,
    rope,
    *,
    layer_index,
    executor,
    router,
    layout,
    video_budget,
):
    q, k, v = project_standard(module, x, rope)
    lut, valid, mask = router.build_lut(q, k, layout, video_budget)
    try:
        return executor.prepare(
            q,
            k,
            v,
            lut,
            valid,
            layer_index=layer_index,
            metadata=mask_metadata(mask, layer_index, q.shape[1]),
        )
    finally:
        del q, k, v, lut, valid, mask


def prepare_projected(
    projected,
    *,
    layer_index,
    executor,
    router,
    layout,
    video_budget,
):
    lut, valid, mask = router.build_lut_from_summaries(
        projected.q_summary,
        projected.k_summary,
        layout,
        video_budget,
    )
    try:
        return executor.prepare_projected(
            projected,
            lut,
            valid,
            metadata=mask_metadata(mask, layer_index, projected.heads),
        )
    finally:
        del lut, valid, mask


def execute_prepared(executor, prepare):
    prepared = prepare()
    try:
        return executor.execute(prepared)
    finally:
        del prepared


def benchmark_round_robin(torch, cases, warmup, iterations, device):
    names = tuple(cases)
    for _ in range(warmup):
        for name in names:
            result = cases[name]()
            torch.cuda.synchronize(device)
            del result
    torch.cuda.empty_cache()

    samples = {name: [] for name in names}
    peaks = {name: [] for name in names}
    output_live = {name: [] for name in names}
    for iteration in range(iterations):
        order = names if iteration % 2 == 0 else tuple(reversed(names))
        for name in order:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            result = cases[name]()
            stop.record()
            stop.synchronize()
            samples[name].append(float(start.elapsed_time(stop)))
            peaks[name].append(int(torch.cuda.max_memory_allocated(device) - before))
            output_live[name].append(int(torch.cuda.memory_allocated(device) - before))
            del result
            torch.cuda.synchronize(device)
    return {
        name: {
            'median_ms': statistics.median(samples[name]),
            'min_ms': min(samples[name]),
            'samples_ms': samples[name],
            'peak_allocated_bytes': max(peaks[name]),
            'output_live_bytes': max(output_live[name]),
        }
        for name in names
    }


def tensor_metrics(torch, actual, reference):
    if actual.shape != reference.shape:
        return {
            'shape_match': False,
            'actual_shape': list(actual.shape),
            'reference_shape': list(reference.shape),
        }
    actual_float = actual.float()
    reference_float = reference.float()
    delta = actual_float - reference_float
    squared = float(delta.square().sum().item())
    reference_squared = float(reference_float.square().sum().item())
    return {
        'shape_match': True,
        'exact': bool(torch.equal(actual, reference)),
        'max_abs': float(delta.abs().max().item()),
        'rmse': (squared / actual.numel()) ** 0.5,
        'relative_rmse': (squared / max(reference_squared, 1e-20)) ** 0.5,
    }


def carrier_metrics(torch, actual, reference):
    return {
        name: tensor_metrics(torch, getattr(actual, name), getattr(reference, name))
        for name in (
            'q_int8',
            'q_scale',
            'k_int8',
            'k_scale',
            'v',
            'q_summary',
            'k_summary',
        )
    }


def route_carrier(router, projected, layout, video_budget):
    return router.build_lut_from_summaries(
        projected.q_summary,
        projected.k_summary,
        layout,
        video_budget,
    )[:2]


def route_metrics(torch, router, actual, reference, layout, video_budget):
    actual_lut, actual_valid = route_carrier(
        router, actual, layout, video_budget
    )
    reference_lut, reference_valid = route_carrier(
        router, reference, layout, video_budget
    )
    result = {
        'lut_exact': bool(torch.equal(actual_lut, reference_lut)),
        'lut_different_elements': int((actual_lut != reference_lut).sum().item()),
        'valid_exact': bool(torch.equal(actual_valid, reference_valid)),
    }
    del actual_lut, actual_valid, reference_lut, reference_valid
    return result


def parity_report(
    torch,
    module,
    x,
    rope,
    *,
    layer_index,
    spec,
    executor,
    router,
    video_start,
    video_budget,
    chunk_rows,
):
    from h3_optimizations.attention.sparse.chunked_qkv import (
        run_chunked_sparse_qkv,
    )
    from h3_optimizations.attention.sparse.fused_qkv import run_fused_qkv

    layout = make_layout(x.shape[0], min(video_start, x.shape[0] - 1))
    one_chunk_rows = math.ceil(int(x.shape[0]) / 128) * 128
    full_kitchen = run_chunked_sparse_qkv(
        module,
        x,
        rope,
        layer_index=layer_index,
        spec=spec,
        chunk_rows=one_chunk_rows,
    )
    chunked = run_chunked_sparse_qkv(
        module,
        x,
        rope,
        layer_index=layer_index,
        spec=spec,
        chunk_rows=chunk_rows,
    )
    fused = run_fused_qkv(module, x, rope, layer_index=layer_index)
    report = {
        'chunked_vs_one_chunk_kitchen': carrier_metrics(
            torch, chunked, full_kitchen
        ),
        'chunked_vs_legacy_fused': carrier_metrics(torch, chunked, fused),
        'routes': {
            'chunked_vs_one_chunk_kitchen': route_metrics(
                torch,
                router,
                chunked,
                full_kitchen,
                layout,
                video_budget,
            ),
            'chunked_vs_legacy_fused': route_metrics(
                torch,
                router,
                chunked,
                fused,
                layout,
                video_budget,
            ),
        },
    }

    def projected_output(projected):
        prepared = prepare_projected(
            projected,
            layer_index=layer_index,
            executor=executor,
            router=router,
            layout=layout,
            video_budget=video_budget,
        )
        try:
            return executor.execute(prepared)
        finally:
            del prepared

    fused_output = projected_output(fused)
    chunked_output = projected_output(chunked)
    report['attention_output_chunked_vs_legacy_fused'] = output_error_metrics(
        torch,
        chunked_output,
        fused_output,
    )
    del (
        full_kitchen,
        chunked,
        fused,
        fused_output,
        chunked_output,
    )
    torch.cuda.synchronize(x.device)
    return report


def main(argv=None):
    args = parse_args(argv)
    kitchen_root = Path(args.kitchen_source).resolve()
    if not (kitchen_root / 'comfy_kitchen' / '__init__.py').is_file():
        raise SystemExit('--kitchen-source is not a Comfy Kitchen source checkout')
    sys.path.insert(0, str(kitchen_root))

    import torch

    from h3_optimizations.attention.sparse.chunked_qkv import (
        run_chunked_sparse_qkv,
    )
    from h3_optimizations.attention.sparse.config import HybridSparseConfig
    from h3_optimizations.attention.sparse.fused_qkv import (
        FusedQKVProjector,
        run_fused_qkv,
        sparse_fused_qkv_contract_mismatch,
    )
    from h3_optimizations.attention.sparse.router import SparseTileRouter
    from h3_optimizations.attention.sparse.sparse_sage import (
        SparseSageExecutor,
        load_sparse_sage_spec,
    )

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    device = torch.device('cuda')
    checkpoint = resolve_checkpoint(args.checkpoint)
    loaded, hidden, prefix = build_attention(
        torch,
        checkpoint,
        args.block,
        args.epsilon,
        device,
    )
    module = BenchmarkAttention()
    module.__dict__.update(vars(loaded))
    del loaded
    contract = weight_contract(module)
    if not (
        contract['quantized']
        and contract['layout'] == 'TensorWiseINT8Layout'
        and contract['convrot']
        and contract['convrot_groupsize'] == 256
    ):
        raise SystemExit('checkpoint QKV is not ConvRot-256 TensorWise INT8')

    spec = load_sparse_sage_spec()
    mismatch = sparse_fused_qkv_contract_mismatch(spec)
    if mismatch is not None:
        raise SystemExit('Sparse Sage projected-QKV contract mismatch: %s' % mismatch)
    projector = FusedQKVProjector()
    projector.bind(module)
    executor = SparseSageExecutor(spec)
    config = HybridSparseConfig(video_budget=args.video_budget)
    router = SparseTileRouter(config, spec=spec)
    layout = make_layout(args.sequence, args.video_start)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(
        (args.sequence, hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope = make_rope(torch, args.sequence, device)

    if args.projector_sweep:
        cases = {
            'chunked_sparse_%d' % chunk_rows: (
                lambda rows=chunk_rows: run_chunked_sparse_qkv(
                    module,
                    x,
                    rope,
                    layer_index=args.block,
                    spec=spec,
                    chunk_rows=rows,
                )
            )
            for chunk_rows in args.projector_sweep
        }
        measured = benchmark_round_robin(
            torch,
            cases,
            args.warmup,
            args.iterations,
            device,
        )
        result = {
            'boundary': 'projected Sparse Sage carrier only',
            'order': list(args.projector_sweep),
            'sequence': args.sequence,
            'gpu': {
                'name': torch.cuda.get_device_name(device),
                'capability': list(torch.cuda.get_device_capability(device)),
            },
            'sparse_contract': {
                'architecture': spec.architecture,
                'q_tile': spec.q_tile,
                'kv_tile': spec.kv_tile,
                'v_format': spec.v_format,
            },
            'cases': measured,
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
            for name, details in measured.items():
                print(
                    '%s: %.3f ms, peak %.3f GiB'
                    % (
                        name,
                        details['median_ms'],
                        details['peak_allocated_bytes'] / 2**30,
                    )
                )
            if args.output:
                print('wrote %s' % Path(args.output).resolve())
        return 0

    standard_project = lambda: project_standard(module, x, rope)
    fused_project = lambda: run_fused_qkv(
        module, x, rope, layer_index=args.block
    )
    chunked_project = lambda: run_chunked_sparse_qkv(
        module,
        x,
        rope,
        layer_index=args.block,
        spec=spec,
        chunk_rows=args.chunk_rows,
    )

    standard_prepare = lambda: prepare_standard(
        module,
        x,
        rope,
        layer_index=args.block,
        executor=executor,
        router=router,
        layout=layout,
        video_budget=args.video_budget,
    )
    fused_prepare = lambda: prepare_projected(
        fused_project(),
        layer_index=args.block,
        executor=executor,
        router=router,
        layout=layout,
        video_budget=args.video_budget,
    )
    chunked_prepare = lambda: prepare_projected(
        chunked_project(),
        layer_index=args.block,
        executor=executor,
        router=router,
        layout=layout,
        video_budget=args.video_budget,
    )

    projected_cases = {
        'standard_floating_qkv': benchmark_case(
            torch, standard_project, args.warmup, args.iterations, device
        ),
        'legacy_fused_sparse': benchmark_case(
            torch, fused_project, args.warmup, args.iterations, device
        ),
        'chunked_sparse_%d' % args.chunk_rows: benchmark_case(
            torch, chunked_project, args.warmup, args.iterations, device
        ),
    }
    preparation_cases = {
        'standard': benchmark_case(
            torch, standard_prepare, args.warmup, args.iterations, device
        ),
        'legacy_fused_sparse': benchmark_case(
            torch, fused_prepare, args.warmup, args.iterations, device
        ),
        'chunked_sparse_%d' % args.chunk_rows: benchmark_case(
            torch, chunked_prepare, args.warmup, args.iterations, device
        ),
    }
    attention_cases = {
        'standard': benchmark_case(
            torch,
            lambda: execute_prepared(executor, standard_prepare),
            args.warmup,
            args.iterations,
            device,
        ),
        'legacy_fused_sparse': benchmark_case(
            torch,
            lambda: execute_prepared(executor, fused_prepare),
            args.warmup,
            args.iterations,
            device,
        ),
        'chunked_sparse_%d' % args.chunk_rows: benchmark_case(
            torch,
            lambda: execute_prepared(executor, chunked_prepare),
            args.warmup,
            args.iterations,
            device,
        ),
    }

    parity = None
    if not args.skip_parity:
        parity_sequence = min(args.sequence, args.parity_sequence)
        parity = parity_report(
            torch,
            module,
            x[:parity_sequence],
            rope[:, :parity_sequence],
            layer_index=args.block,
            spec=spec,
            executor=executor,
            router=router,
            video_start=min(args.video_start, parity_sequence - 1),
            video_budget=args.video_budget,
            chunk_rows=args.chunk_rows,
        )

    kernel_trace = None
    if not args.skip_kernel_trace:
        kernel_trace = {
            'legacy_fused_projector': cuda_kernel_names(
                torch, fused_project, device
            ),
            'chunked_sparse_projector': cuda_kernel_names(
                torch, chunked_project, device
            ),
            'standard_sparse_attention': cuda_kernel_names(
                torch,
                lambda: execute_prepared(executor, standard_prepare),
                device,
            ),
            'legacy_fused_sparse_attention': cuda_kernel_names(
                torch,
                lambda: execute_prepared(executor, fused_prepare),
                device,
            ),
            'chunked_sparse_attention': cuda_kernel_names(
                torch,
                lambda: execute_prepared(executor, chunked_prepare),
                device,
            ),
        }

    result = {
        'boundary': {
            'projected_carrier': (
                'QKV projection through post-RMSNorm/RoPE Q/K and the native '
                'carrier returned by each producer; standard retains floating Q/K/V'
            ),
            'complete_preparation': (
                'projection, routing, Q/K packing, and architecture-native Sparse '
                'Sage V preparation'
            ),
            'complete_attention': (
                'complete preparation followed by the real Sparse Sage kernel'
            ),
        },
        'versions': {
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'comfy_kitchen': importlib.metadata.version('comfy-kitchen'),
            'comfy_kitchen_source': str(
                Path(sys.modules['comfy_kitchen'].__file__).resolve()
            ),
            'sparse_sage': spec.version,
        },
        'gpu': {
            'name': torch.cuda.get_device_name(device),
            'capability': list(torch.cuda.get_device_capability(device)),
        },
        'checkpoint': str(checkpoint),
        'checkpoint_prefix': prefix,
        'block': args.block,
        'sequence': args.sequence,
        'chunk_rows': args.chunk_rows,
        'parity_sequence': min(args.sequence, args.parity_sequence),
        'video_start': args.video_start,
        'video_budget': args.video_budget,
        'hidden': hidden,
        'heads': module.heads,
        'head_dim': module.head_dim,
        'weight_contract': contract,
        'sparse_contract': {
            'architecture': spec.architecture,
            'q_tile': spec.q_tile,
            'kv_tile': spec.kv_tile,
            'v_format': spec.v_format,
            'accumulator': spec.accumulator,
            'kernel': spec.kernel_name,
        },
        'projected_carrier': projected_cases,
        'complete_preparation': preparation_cases,
        'complete_attention': attention_cases,
        'parity': parity,
        'kernel_trace': kernel_trace,
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
        for boundary, cases in (
            ('projected carrier', projected_cases),
            ('complete preparation', preparation_cases),
            ('complete attention', attention_cases),
        ):
            print(boundary + ':')
            for name, details in cases.items():
                print(
                    '  %s: %.3f ms, peak %.3f GiB, output live %.3f GiB'
                    % (
                        name,
                        details['median_ms'],
                        details['peak_allocated_bytes'] / 2**30,
                        details['output_live_bytes'] / 2**30,
                    )
                )
        if parity is not None:
            print('parity:')
            print(json.dumps(parity, indent=2, sort_keys=True))
        if args.output:
            print('wrote %s' % Path(args.output).resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
