'''Numerically gate and optionally time the full-route Sol-shaped Triton kernel.

The benchmark starts from one already-prequantized Kitchen carrier. This keeps
the comparison focused on attention execution:

* native Kitchen dense attention;
* PR #41's Kitchen-parity Triton kernel on an explicit full route;
* the experimental one-program-per-64Q Sol-shaped pointer kernel.

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


ARM_ORDER = ('kitchen', 'pr41_triton', 'sol_pointer')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    parser.add_argument('--parity-sequence', type=int, default=257)
    parser.add_argument('--benchmark', action='store_true')
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


def make_carrier(torch, native, sequence, heads, seed, device):
    generator = torch.Generator(device=device).manual_seed(int(seed))

    def sample():
        return torch.randn(
            (1, int(heads), int(sequence), 128),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )

    return native.prequantize_int8_attention(
        sample(), sample(), sample(), cta_k=64
    )


def numerical_gate(torch, native, pr41_launch, sol, args, device):
    carrier = make_carrier(
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
    prepared = sol.prepare_carrier(carrier)
    outputs = {
        'kitchen': native.int8_attention_from_prequantized(carrier),
        'pr41_triton': pr41_launch(carrier, lut, valid),
        'sol_pointer': sol.launch_prepared(prepared),
    }
    torch.cuda.synchronize(device)
    reference = outputs['kitchen']
    result = {
        name: tensor_metrics(torch, value, reference)
        for name, value in outputs.items()
    }
    del outputs, reference, prepared, carrier, lut, valid
    return result


def elapsed_ms(torch, function, device):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    value = function()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)), value


def benchmark(torch, native, pr41_launch, sol, args, device):
    carrier = make_carrier(
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
    prepared = sol.prepare_carrier(carrier)
    arms = {
        'kitchen': lambda: native.int8_attention_from_prequantized(carrier),
        'pr41_triton': lambda: pr41_launch(carrier, lut, valid),
        'sol_pointer': lambda: sol.launch_prepared(prepared),
    }

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
    from h3_optimizations.attention.sparse.triton_kitchen import (
        _launch as pr41_launch,
    )
    from h3_optimizations.native import carrier_selftest
    from h3_optimizations.native import int8_attention as native

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

    parity = numerical_gate(torch, native, pr41_launch, sol, args, device)
    failed = [name for name, metrics in parity.items() if not metrics['exact']]
    timings = (
        benchmark(torch, native, pr41_launch, sol, args, device)
        if args.benchmark and not failed
        else None
    )
    result = {
        'boundary': (
            'attention execution from one identical, already-prequantized '
            'Kitchen carrier; QKV projection and routing are excluded'
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
        'parity_shape': {
            'batch': 1,
            'heads': args.heads,
            'sequence': args.parity_sequence,
            'head_dim': 128,
            'dtype': 'bfloat16',
        },
        'parity_against_kitchen': parity,
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
