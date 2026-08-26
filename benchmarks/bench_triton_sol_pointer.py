'''Numerically gate and optionally time the full-route Sol-shaped Triton kernel.

The benchmark starts from one already-prequantized Kitchen carrier. This keeps
the comparison focused on attention execution:

* native Kitchen dense attention;
* native Kitchen 64Q x 64KV attention on the same full route;
* PR #41's Kitchen-parity Triton kernel on an explicit full route;
* the experimental one-program-per-64Q Sol-shaped pointer kernel;
* all four launch geometries of PlagueKind's BF16 SLA Triton kernel.

By default only the small numerical gate runs. Pass ``--benchmark`` to add the
larger timing case. No model or ComfyUI server is involved.
'''

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


PACK = Path(__file__).resolve().parents[1]
ROOT = next(parent for parent in PACK.parents if (parent / 'comfy').is_dir())
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))


ARM_ORDER = (
    'kitchen_dense',
    'kitchen_64x64_full',
    'pr41_triton',
    'sol_pointer',
    'sol_bf16_hnd_64x64',
    'production_bf16_64x64',
    'plaguekind_64x64',
    'plaguekind_64x128',
    'plaguekind_128x64',
    'plaguekind_128x128',
)
PLAGUEKIND_GEOMETRIES = {
    'plaguekind_64x64': (64, 64),
    'plaguekind_64x128': (64, 128),
    'plaguekind_128x64': (128, 64),
    'plaguekind_128x128': (128, 128),
}
KITCHEN_INT8_RELATIVE_RMSE_LIMIT = 3e-4
KITCHEN_INT8_MAX_ABS_LIMIT = 0.002


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    parser.add_argument('--parity-sequence', type=int, default=257)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument(
        '--benchmark-inexact',
        action='store_true',
        help='time kernels even when the Kitchen bit-exact numerical gate fails',
    )
    parser.add_argument('--benchmark-sequence', type=int, default=4096)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260826)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not args.i_understand_this_uses_gpu:
        parser.error('pass --i-understand-this-uses-gpu to run CUDA kernels')
    for name in ('parity_sequence', 'benchmark_sequence', 'heads', 'iterations'):
        if getattr(args, name) <= 0:
            parser.error('--%s must be positive' % name.replace('_', '-'))
    if args.warmup < 0:
        parser.error('--warmup cannot be negative')
    if args.benchmark_inexact and not args.benchmark:
        parser.error('--benchmark-inexact requires --benchmark')
    return args


def dense_delta_route(torch, sequence, heads, device):
    q_tiles = (int(sequence) + 63) // 64
    kv_tiles = (int(sequence) + 63) // 64
    lut = torch.zeros(
        (1, int(heads), q_tiles, kv_tiles),
        dtype=torch.int32,
        device=device,
    )
    if kv_tiles > 1:
        lut[..., 1:] = 1
    valid = torch.full(
        (1, int(heads), q_tiles),
        kv_tiles,
        dtype=torch.int32,
        device=device,
    )
    return lut, valid


def dense_absolute_route(torch, sequence, heads, q_tile, kv_tile, device):
    q_tiles = (int(sequence) + int(q_tile) - 1) // int(q_tile)
    kv_tiles = (int(sequence) + int(kv_tile) - 1) // int(kv_tile)
    lut = torch.arange(kv_tiles, dtype=torch.int32, device=device)
    lut = lut.reshape(1, 1, 1, kv_tiles).expand(
        1, int(heads), q_tiles, kv_tiles
    ).contiguous()
    return lut, kv_tiles


def mixed_delta_route(torch, sequence, heads, device):
    q_tiles = (int(sequence) + 63) // 64
    kv_tiles = (int(sequence) + 63) // 64
    if q_tiles < 2 or kv_tiles < 3:
        raise ValueError('mixed route needs at least two Q and three KV tiles')
    retained = max(1, (kv_tiles - 1) // 2)
    selected = 1 + retained
    absolute = torch.cat(
        (
            torch.zeros((1,), dtype=torch.int32, device=device),
            torch.linspace(
                1,
                kv_tiles - 1,
                retained,
                dtype=torch.float32,
                device=device,
            ).round().to(torch.int32),
        )
    ).unique(sorted=True)
    selected = int(absolute.numel())
    delta = torch.cat((absolute[:1], absolute[1:] - absolute[:-1]))
    lut, valid = dense_delta_route(torch, sequence, heads, device)
    lut[..., 1:, :].zero_()
    lut[..., 1:, :selected] = delta.reshape(1, 1, 1, selected)
    valid[..., 1:] = selected
    metadata = {
        'dense_q_tiles': 1,
        'sparse_q_tiles': q_tiles - 1,
        'pure_video_kv_tiles': kv_tiles - 1,
        'retained_video_kv_tiles': selected - 1,
    }
    return lut, valid, metadata


def prepare_production_bf16(
    torch, triton_bf16, q, k, v, lut=None, valid=None, metadata=None
):
    sequence = int(q.shape[-2])
    heads = int(q.shape[1])
    q_tiles = (sequence + 63) // 64
    if lut is None:
        sparse_lut = torch.empty(
            (1, heads, 0, 0), dtype=torch.int32, device=q.device
        )
        dense_q_tiles = q_tiles
        sparse_q_tiles = 0
        sparse_selected = 0
        route_metadata = {'route_format': 'implicit_full'}
    else:
        sparse_lut, dense_q_tiles, sparse_q_tiles, sparse_selected = (
            triton_bf16._compact_route(lut, valid, metadata)
        )
        route_metadata = dict(metadata)
    return triton_bf16.PreparedTritonBF16(
        q=q,
        k=k,
        v=v,
        sparse_lut=sparse_lut,
        dense_q_tiles=dense_q_tiles,
        sparse_q_tiles=sparse_q_tiles,
        sparse_selected=sparse_selected,
        layer_index=0,
        metadata=route_metadata,
    )


def tensor_metrics(torch, actual, expected):
    delta = actual.float() - expected.float()
    denominator = expected.float().square().mean().sqrt().clamp_min(1.0e-12)
    return {
        'exact': bool(torch.equal(actual, expected)),
        'rmse': float(delta.square().mean().sqrt().item()),
        'relative_rmse': float(
            (delta.square().mean().sqrt() / denominator).item()
        ),
        'max_abs': float(delta.abs().max().item()),
    }


def dense_fp32_reference(torch, q, k, v):
    q_cpu = q.float().cpu()
    k_cpu = k.float().cpu()
    v_cpu = v.float().cpu()
    logits = torch.matmul(q_cpu, k_cpu.transpose(-1, -2)) * (128**-0.5)
    output = torch.matmul(torch.softmax(logits, dim=-1), v_cpu)
    return output.to(dtype=q.dtype, device=q.device)


def make_inputs(torch, native, sequence, heads, seed, device):
    generator = torch.Generator(device=device).manual_seed(int(seed))

    def sample():
        return torch.randn(
            (1, int(heads), int(sequence), 128),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )

    q, k, v = sample(), sample(), sample()
    carrier = native.prequantize_int8_attention(q, k, v, cta_k=64)
    return q, k, v, carrier


def plaguekind_inputs(q, k, v):
    return tuple(value.transpose(1, 2).contiguous() for value in (q, k, v))


def plaguekind_routes(torch, sequence, heads, device):
    return {
        name: dense_absolute_route(
            torch, sequence, heads, q_tile, kv_tile, device
        )
        for name, (q_tile, kv_tile) in PLAGUEKIND_GEOMETRIES.items()
    }


def plaguekind_arms(plaguekind, q, k, v, routes):
    arms = {}
    for name, (q_tile, kv_tile) in PLAGUEKIND_GEOMETRIES.items():
        lut, topk = routes[name]
        arms[name] = lambda lut=lut, topk=topk, q_tile=q_tile, kv_tile=kv_tile: (
            plaguekind.block_sparse_attention(
                q, k, v, lut, topk, q_tile, kv_tile
            )
        )
    return arms


def numerical_gate(
    torch, native, pr41_launch, sol, triton_bf16, bf16_control, plaguekind, args, device
):
    q, k, v, carrier = make_inputs(
        torch,
        native,
        args.parity_sequence,
        args.heads,
        args.seed,
        device,
    )
    lut, valid = dense_delta_route(
        torch, args.parity_sequence, args.heads, device
    )
    route = native.BlockSparseRoute(
        indices=lut,
        counts=valid,
        q_tile=64,
        kv_tile=64,
        encoding='delta',
    )
    prepared = sol.prepare_carrier(carrier)
    production_bf16 = prepare_production_bf16(torch, triton_bf16, q, k, v)
    q_blhd, k_blhd, v_blhd = plaguekind_inputs(q, k, v)
    production_bf16_strided = prepare_production_bf16(
        torch,
        triton_bf16,
        q_blhd.transpose(1, 2),
        k_blhd.transpose(1, 2),
        v_blhd.transpose(1, 2),
    )
    plague_routes = plaguekind_routes(
        torch, args.parity_sequence, args.heads, device
    )
    plague_arms = plaguekind_arms(
        plaguekind, q_blhd, k_blhd, v_blhd, plague_routes
    )
    outputs = {
        'kitchen_dense': native.int8_attention_from_prequantized(carrier),
        'kitchen_64x64_full': (
            native.block_sparse_int8_attention_from_prequantized(carrier, route)
        ),
        'pr41_triton': pr41_launch(carrier, lut, valid),
        'sol_pointer': sol.launch_prepared(prepared),
        'sol_bf16_hnd_64x64': bf16_control.launch(q, k, v),
        'production_bf16_64x64': triton_bf16._launch(production_bf16),
        'production_bf16_strided_hnd_64x64': triton_bf16._launch(
            production_bf16_strided
        ),
    }
    outputs.update(
        (name, function().transpose(1, 2))
        for name, function in plague_arms.items()
    )
    torch.cuda.synchronize(device)
    references = {
        'kitchen_64x64_full': outputs['kitchen_64x64_full'],
        'dense_fp32_softmax_bf16_output': dense_fp32_reference(torch, q, k, v),
    }
    result = {
        reference_name: {
            name: tensor_metrics(torch, value, reference)
            for name, value in outputs.items()
        }
        for reference_name, reference in references.items()
    }
    result['production_bf16_full_route_against_plaguekind_64x64'] = {
        'production_bf16_64x64': tensor_metrics(
            torch, outputs['production_bf16_64x64'], outputs['plaguekind_64x64']
        ),
        'production_bf16_strided_hnd_64x64': tensor_metrics(
            torch,
            outputs['production_bf16_strided_hnd_64x64'],
            outputs['plaguekind_64x64'],
        ),
        'plaguekind_64x64': tensor_metrics(
            torch, outputs['plaguekind_64x64'], outputs['plaguekind_64x64']
        ),
    }
    sparse_lut, sparse_valid, sparse_metadata = mixed_delta_route(
        torch, args.parity_sequence, args.heads, device
    )
    sparse_pr41 = pr41_launch(carrier, sparse_lut, sparse_valid)
    sparse_prepared = sol.prepare_routed_carrier(
        carrier,
        sparse_lut,
        sparse_valid,
        layer_index=0,
        metadata=sparse_metadata,
    )
    sparse_sol = sol.launch_prepared(sparse_prepared)
    sparse_bf16_prepared = prepare_production_bf16(
        torch,
        triton_bf16,
        q,
        k,
        v,
        sparse_lut,
        sparse_valid,
        sparse_metadata,
    )
    sparse_bf16 = triton_bf16._launch(sparse_bf16_prepared)
    absolute = torch.cumsum(sparse_lut, dim=-1, dtype=torch.int32)
    dense_q_tiles = int(sparse_metadata['dense_q_tiles'])
    sparse_selected = int(sparse_bf16_prepared.sparse_selected)
    dense_rows = dense_q_tiles * 64
    sparse_bf16_reference = torch.empty_like(q_blhd)
    plaguekind._block_sparse_attention_into(
        q_blhd[:, :dense_rows],
        k_blhd,
        v_blhd,
        absolute[..., :dense_q_tiles, :].contiguous(),
        absolute.shape[-1],
        64,
        64,
        sparse_bf16_reference[:, :dense_rows],
    )
    plaguekind._block_sparse_attention_into(
        q_blhd[:, dense_rows:],
        k_blhd,
        v_blhd,
        absolute[..., dense_q_tiles:, :sparse_selected].contiguous(),
        sparse_selected,
        64,
        64,
        sparse_bf16_reference[:, dense_rows:],
    )
    sparse_bf16_reference = sparse_bf16_reference.transpose(1, 2)
    torch.cuda.synchronize(device)
    result['mixed_sparse_route_against_pr41'] = {
        'pr41_triton': tensor_metrics(torch, sparse_pr41, sparse_pr41),
        'sol_pointer': tensor_metrics(torch, sparse_sol, sparse_pr41),
    }
    result['production_bf16_mixed_route_against_plaguekind_64x64'] = {
        'production_bf16_64x64': tensor_metrics(
            torch, sparse_bf16, sparse_bf16_reference
        ),
        'plaguekind_64x64': tensor_metrics(
            torch, sparse_bf16_reference, sparse_bf16_reference
        ),
    }
    del outputs, references, prepared, carrier, lut, valid
    del q, k, v, q_blhd, k_blhd, v_blhd, plague_routes, plague_arms
    del production_bf16, production_bf16_strided
    del sparse_lut, sparse_valid, sparse_prepared, sparse_pr41, sparse_sol
    del sparse_bf16_prepared, sparse_bf16, sparse_bf16_reference, absolute
    return result


def elapsed_ms(torch, function, device):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    value = function()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)), value


def benchmark(
    torch, native, pr41_launch, sol, triton_bf16, bf16_control, plaguekind, args, device
):
    q, k, v, carrier = make_inputs(
        torch,
        native,
        args.benchmark_sequence,
        args.heads,
        args.seed + 1,
        device,
    )
    lut, valid = dense_delta_route(
        torch, args.benchmark_sequence, args.heads, device
    )
    route = native.BlockSparseRoute(
        indices=lut,
        counts=valid,
        q_tile=64,
        kv_tile=64,
        encoding='delta',
    )
    prepared = sol.prepare_carrier(carrier)
    production_bf16 = prepare_production_bf16(torch, triton_bf16, q, k, v)
    q_blhd, k_blhd, v_blhd = plaguekind_inputs(q, k, v)
    plague_routes = plaguekind_routes(
        torch, args.benchmark_sequence, args.heads, device
    )
    arms = {
        'kitchen_dense': lambda: native.int8_attention_from_prequantized(carrier),
        'kitchen_64x64_full': lambda: (
            native.block_sparse_int8_attention_from_prequantized(carrier, route)
        ),
        'pr41_triton': lambda: pr41_launch(carrier, lut, valid),
        'sol_pointer': lambda: sol.launch_prepared(prepared),
        'sol_bf16_hnd_64x64': lambda: bf16_control.launch(q, k, v),
        'production_bf16_64x64': lambda: triton_bf16._launch(production_bf16),
    }
    arms.update(
        plaguekind_arms(
            plaguekind, q_blhd, k_blhd, v_blhd, plague_routes
        )
    )

    for _ in range(args.warmup):
        for name in ARM_ORDER:
            output = arms[name]()
            torch.cuda.synchronize(device)
            del output

    samples = {name: [] for name in ARM_ORDER}
    for iteration in range(args.iterations):
        order = ARM_ORDER if iteration % 2 == 0 else tuple(reversed(ARM_ORDER))
        for name in order:
            milliseconds, output = elapsed_ms(torch, arms[name], device)
            samples[name].append(milliseconds)
            del output

    prepare_samples = []
    complete_samples = []
    for _ in range(args.warmup):
        transient = sol.prepare_carrier(carrier)
        output = sol.launch_prepared(transient)
        torch.cuda.synchronize(device)
        del output, transient
    for _ in range(args.iterations):
        milliseconds, transient = elapsed_ms(
            torch, lambda: sol.prepare_carrier(carrier), device
        )
        prepare_samples.append(milliseconds)
        del transient

        def complete():
            current = sol.prepare_carrier(carrier)
            return sol.launch_prepared(current)

        milliseconds, output = elapsed_ms(torch, complete, device)
        complete_samples.append(milliseconds)
        del output

    result = {
        'kernel_execution': {
            name: {
                'median_ms': statistics.median(samples[name]),
                'min_ms': min(samples[name]),
                'samples_ms': samples[name],
            }
            for name in ARM_ORDER
        },
        'sol_v_sum_preparation': {
            'median_ms': statistics.median(prepare_samples),
            'min_ms': min(prepare_samples),
            'samples_ms': prepare_samples,
        },
        'sol_complete_from_kitchen_carrier': {
            'median_ms': statistics.median(complete_samples),
            'min_ms': min(complete_samples),
            'samples_ms': complete_samples,
        },
    }
    del prepared, carrier, lut, valid
    del q, k, v, q_blhd, k_blhd, v_blhd, plague_routes
    return result


def git_value(*args):
    completed = subprocess.run(
        ('git', *args),
        cwd=PACK,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main(argv=None):
    args = parse_args(argv)

    import torch

    from h3_optimizations.attention.sparse import triton_sol_pointer as sol
    from h3_optimizations.attention.sparse import triton_bf16
    from h3_optimizations.attention.sparse.triton_kitchen import (
        _launch as pr41_launch,
    )
    from h3_optimizations.native import carrier_selftest
    from h3_optimizations.native import int8_attention as native
    import sol_bf16_control as bf16_control
    import plaguekind_sla as plaguekind

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    if not sol.TRITON_AVAILABLE:
        raise SystemExit('Triton is required')
    device = torch.device('cuda')
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    if capability[0] < 8:
        raise SystemExit('INT8 Triton requires compute capability >= 8.0')
    if not carrier_selftest.check(device):
        raise SystemExit('Kitchen carrier self-test failed')
    if not native.int8_attention_is_available(device):
        raise SystemExit('native Kitchen 64x64 sparse attention is unavailable')

    parity = numerical_gate(
        torch, native, pr41_launch, sol, triton_bf16, bf16_control, plaguekind, args, device
    )
    kitchen_parity = parity['kitchen_64x64_full']
    failed = []
    for name in ('pr41_triton', 'sol_pointer'):
        metrics = kitchen_parity[name]
        if (
            metrics['relative_rmse'] > KITCHEN_INT8_RELATIVE_RMSE_LIMIT
            or metrics['max_abs'] > KITCHEN_INT8_MAX_ABS_LIMIT
        ):
            failed.append(name)
    for section in (
        'production_bf16_full_route_against_plaguekind_64x64',
        'production_bf16_mixed_route_against_plaguekind_64x64',
    ):
        for name, metrics in parity[section].items():
            if name.startswith('production_bf16') and not metrics['exact']:
                failed.append(name)
    timings = (
        benchmark(
            torch,
            native,
            pr41_launch,
            sol,
            triton_bf16,
            bf16_control,
            plaguekind,
            args,
            device,
        )
        if args.benchmark and (not failed or args.benchmark_inexact)
        else None
    )
    result = {
        'boundary': (
            'attention execution from identical source BF16 Q/K/V; Kitchen '
            'carrier preparation, PlagueKind BLHD packing, QKV projection, '
            'and routing are excluded'
        ),
        'source': {
            'path': str(PACK),
            'branch': git_value('branch', '--show-current'),
            'commit': git_value('rev-parse', 'HEAD'),
            'dirty': bool(git_value('status', '--porcelain')),
        },
        'gpu': {
            'name': torch.cuda.get_device_name(device),
            'capability': list(capability),
        },
        'contract': sol.kernel_contract(),
        'bf16_control': bf16_control.contract(),
        'plaguekind': {
            'source_commit': plaguekind.PLAGUEKIND_SOURCE_COMMIT,
            'math': 'BF16 QK and BF16 P-by-V with FP32 online softmax',
            'route': 'full absolute fixed-topk',
            'geometries': {
                name: {'q_tile': geometry[0], 'kv_tile': geometry[1]}
                for name, geometry in PLAGUEKIND_GEOMETRIES.items()
            },
        },
        'parity_shape': {
            'batch': 1,
            'heads': args.heads,
            'sequence': args.parity_sequence,
            'head_dim': 128,
            'dtype': 'bfloat16',
        },
        'numerics_against': parity,
        'parity_passed': not failed,
        'timing_shape': (
            {
                'batch': 1,
                'heads': args.heads,
                'sequence': args.benchmark_sequence,
                'head_dim': 128,
                'dtype': 'bfloat16',
            }
            if timings is not None
            else None
        ),
        'timings': timings,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = args.output.resolve()
        if not output.parent.is_dir():
            raise SystemExit('output directory does not exist: %s' % output.parent)
        output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)
    if failed:
        raise SystemExit(
            'numerical gate failed against Kitchen: %s' % ', '.join(failed)
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
