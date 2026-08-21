'''Test whether CUTLASS config-0 H3 QKV optima follow CTA-wave alignment.'''

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import random
import statistics
import sys
from pathlib import Path

from bench_chunked_kitchen_qkv import (
    DEFAULT_SEQUENCE,
    build_attention,
    chunk_ranges,
    resolve_checkpoint,
    weight_contract,
)
from bench_config0_preallocated import Config0Experiment


M_TILE = 128
N_TILE = 256
DEFAULT_ROWS = (
    2560, 2688, 2816, 2944, 3072, 3200, 3328,
    3584, 3712, 3840, 3968, 4096, 4224, 4352, 4480, 4608,
    5632, 5760, 5888, 6016, 6144,
    7936, 8064, 8192, 8320, 8448,
)
DEFAULT_AGGREGATE_ROWS = (
    2816, 2944, 3072,
    3968, 4096, 4224,
    5760, 5888, 6016,
    8064, 8192, 8320,
)
DEFAULT_CONTROL_ROWS = (2944, 4096, 5888, 8192)


def parse_rows(value):
    rows = tuple(int(item.strip()) for item in str(value).split(',') if item.strip())
    if not rows:
        raise ValueError('at least one row count is required')
    if len(set(rows)) != len(rows):
        raise ValueError('row counts must not contain duplicates')
    if any(row <= 0 or row % M_TILE for row in rows):
        raise ValueError('row counts must be positive multiples of 128')
    return rows


def wave_geometry(rows, output_width, multiprocessors):
    m_tiles = math.ceil(int(rows) / M_TILE)
    n_tiles = math.ceil(int(output_width) / N_TILE)
    ctas = m_tiles * n_tiles
    waves = math.ceil(ctas / int(multiprocessors))
    tail_ctas = ctas - (waves - 1) * int(multiprocessors)
    return {
        'm_tiles': m_tiles,
        'n_tiles': n_tiles,
        'ctas': ctas,
        'waves': waves,
        'tail_ctas': tail_ctas,
        'tail_fraction': tail_ctas / int(multiprocessors),
        'wave_efficiency': ctas / (waves * int(multiprocessors)),
        'exact_waves': ctas % int(multiprocessors) == 0,
    }


def aggregate_geometry(sequence, chunk_size, output_width, multiprocessors):
    parts = []
    for start, stop in chunk_ranges(int(sequence), int(chunk_size)):
        geometry = wave_geometry(stop - start, output_width, multiprocessors)
        parts.append(geometry)
    total_ctas = sum(part['ctas'] for part in parts)
    total_waves = sum(part['waves'] for part in parts)
    return {
        'chunks': len(parts),
        'final_rows': int(sequence) % int(chunk_size) or int(chunk_size),
        'total_ctas': total_ctas,
        'total_waves': total_waves,
        'wave_efficiency': total_ctas / (total_waves * int(multiprocessors)),
    }


def summarize(samples, rows=None):
    median = statistics.median(samples)
    result = {
        'median_ms': median,
        'min_ms': min(samples),
        'max_ms': max(samples),
        'samples_ms': samples,
    }
    if rows is not None:
        result['ms_per_1000_rows'] = median * 1000.0 / int(rows)
    return result


def measure_interleaved(torch, cases, warmup, iterations, seed, device):
    for _ in range(warmup):
        for _name, (_rows, fn) in cases.items():
            fn()
    torch.cuda.synchronize(device)

    samples = {name: [] for name in cases}
    names = list(cases)
    for iteration in range(iterations):
        order = list(names)
        random.Random(int(seed) + iteration).shuffle(order)
        for name in order:
            _rows, fn = cases[name]
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            stop.record()
            stop.synchronize()
            samples[name].append(float(start.elapsed_time(stop)))
    return {
        name: summarize(samples[name], rows=cases[name][0])
        for name in names
    }


class WaveSweep:
    def __init__(self, torch, module, x, max_rows, ring_size):
        self.torch = torch
        self.experiment = Config0Experiment(torch, module, x, max_rows)
        self.x = x
        self.max_rows = int(max_rows)
        self.output_width = int(self.experiment.weight_qdata.shape[0])
        self.ring_size = int(ring_size)
        self.ring_output = torch.empty(
            (self.ring_size, self.max_rows, self.output_width),
            dtype=torch.bfloat16,
            device=x.device,
        )
        self.experiment.full_quant()
        torch.cuda.synchronize(x.device)

    def single_gemm(self, rows):
        rows = int(rows)
        self.experiment.gemm_into(
            self.experiment.full_qdata[:rows],
            self.experiment.full_scale[:rows],
            self.experiment.chunk_output[:rows],
        )
        return self.experiment.chunk_output[:rows]

    def _output_slice(self, mode, slot, start, stop):
        rows = stop - start
        if mode == 'reuse':
            return self.experiment.chunk_output[:rows]
        if mode == 'full':
            return self.experiment.full_output[start:stop]
        if mode == 'ring':
            return self.ring_output[slot % self.ring_size, :rows]
        raise ValueError('unknown output mode: %s' % mode)

    def aggregate_gemm(self, chunk_size, sequence, output_mode):
        result = None
        for slot, (start, stop) in enumerate(chunk_ranges(sequence, chunk_size)):
            output = self._output_slice(output_mode, slot, start, stop)
            self.experiment.gemm_into(
                self.experiment.full_qdata[start:stop],
                self.experiment.full_scale[start:stop],
                output,
            )
            result = output
        return result

    def aggregate_combined(self, chunk_size, sequence, output_mode):
        result = None
        for slot, (start, stop) in enumerate(chunk_ranges(sequence, chunk_size)):
            rows = stop - start
            qdata = self.experiment.chunk_qdata[:rows]
            scale = self.experiment.chunk_scale[:rows]
            output = self._output_slice(output_mode, slot, start, stop)
            self.experiment.quantize_into(self.x[start:stop], qdata, scale)
            self.experiment.gemm_into(qdata, scale, output)
            result = output
        return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=DEFAULT_SEQUENCE)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--rows', default=','.join(map(str, DEFAULT_ROWS)))
    parser.add_argument(
        '--aggregate-rows', default=','.join(map(str, DEFAULT_AGGREGATE_ROWS))
    )
    parser.add_argument(
        '--control-rows', default=','.join(map(str, DEFAULT_CONTROL_ROWS))
    )
    parser.add_argument('--ring-size', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--single-iterations', type=int, default=15)
    parser.add_argument('--aggregate-iterations', type=int, default=7)
    parser.add_argument('--seed', type=int, default=20260821)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    try:
        args.rows = parse_rows(args.rows)
        args.aggregate_rows = parse_rows(args.aggregate_rows)
        args.control_rows = parse_rows(args.control_rows)
    except ValueError as exc:
        parser.error(str(exc))
    all_rows = set(args.rows) | set(args.aggregate_rows) | set(args.control_rows)
    if max(all_rows) > args.sequence:
        parser.error('row counts must not exceed the sequence')
    if args.ring_size <= 0 or args.warmup < 0:
        parser.error('ring-size/warmup are invalid')
    if args.single_iterations <= 0 or args.aggregate_iterations <= 0:
        parser.error('iteration counts must be positive')
    if not args.i_understand_this_uses_gpu:
        parser.error('pass --i-understand-this-uses-gpu after the idle preflight')
    return args


def compact_samples(value):
    if isinstance(value, dict):
        return {
            key: compact_samples(item)
            for key, item in value.items()
            if key != 'samples_ms'
        }
    if isinstance(value, list):
        return [compact_samples(item) for item in value]
    return value


def main(argv=None):
    args = parse_args(argv)
    comfy_root = Path(__file__).resolve().parents[3]
    pack_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(pack_root))

    import torch

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    device = torch.device('cuda')
    checkpoint = resolve_checkpoint(args.checkpoint)
    module, hidden, prefix = build_attention(
        torch, checkpoint, args.block, args.epsilon, device
    )
    contract = weight_contract(module)
    if not (
        contract['quantized']
        and contract['layout'] == 'TensorWiseINT8Layout'
        and contract['convrot']
        and contract['convrot_groupsize'] == 256
    ):
        raise SystemExit('checkpoint QKV is not ConvRot-256 TensorWise INT8')

    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(
        (args.sequence, hidden), generator=generator,
        dtype=torch.bfloat16, device=device,
    )
    max_rows = max(set(args.rows) | set(args.aggregate_rows) | set(args.control_rows))
    sweep = WaveSweep(torch, module, x, max_rows, args.ring_size)
    properties = torch.cuda.get_device_properties(device)
    multiprocessors = int(properties.multi_processor_count)

    single_cases = {
        str(rows): (rows, lambda count=rows: sweep.single_gemm(count))
        for rows in args.rows
    }
    single = measure_interleaved(
        torch, single_cases, args.warmup, args.single_iterations, args.seed, device
    )

    aggregate_cases = {
        str(rows): (
            args.sequence,
            lambda size=rows: sweep.aggregate_gemm(size, args.sequence, 'reuse'),
        )
        for rows in args.aggregate_rows
    }
    aggregate = measure_interleaved(
        torch, aggregate_cases, args.warmup, args.aggregate_iterations,
        args.seed + 1000, device,
    )

    output_cases = {}
    for rows in args.control_rows:
        for mode in ('reuse', 'ring', 'full'):
            name = '%d_%s' % (rows, mode)
            output_cases[name] = (
                args.sequence,
                lambda size=rows, selected=mode: sweep.aggregate_gemm(
                    size, args.sequence, selected
                ),
            )
    output_modes = measure_interleaved(
        torch, output_cases, args.warmup, args.aggregate_iterations,
        args.seed + 2000, device,
    )

    combined_cases = {
        str(rows): (
            args.sequence,
            lambda size=rows: sweep.aggregate_combined(
                size, args.sequence, 'reuse'
            ),
        )
        for rows in args.control_rows
    }
    combined = measure_interleaved(
        torch, combined_cases, args.warmup, args.aggregate_iterations,
        args.seed + 3000, device,
    )

    exact_cases = {}
    exact_lengths = {}
    for rows in args.control_rows:
        exact = args.sequence // rows * rows
        exact_lengths[str(rows)] = exact
        exact_cases[str(rows)] = (
            exact,
            lambda size=rows, length=exact: sweep.aggregate_gemm(
                size, length, 'reuse'
            ),
        )
    exact = measure_interleaved(
        torch, exact_cases, args.warmup, args.aggregate_iterations,
        args.seed + 4000, device,
    )

    result = {
        'versions': {
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'comfy_kitchen': importlib.metadata.version('comfy-kitchen'),
        },
        'gpu': {
            'name': torch.cuda.get_device_name(device),
            'capability': list(torch.cuda.get_device_capability(device)),
            'multiprocessors': multiprocessors,
        },
        'checkpoint': str(checkpoint),
        'checkpoint_prefix': prefix,
        'sequence': args.sequence,
        'config': 0,
        'tile': {'m': M_TILE, 'n': N_TILE},
        'weight_contract': contract,
        'single_kernel': {
            rows: {
                **details,
                'geometry': wave_geometry(
                    int(rows), sweep.output_width, multiprocessors
                ),
            }
            for rows, details in single.items()
        },
        'aggregate_reuse': {
            rows: {
                **details,
                'geometry': aggregate_geometry(
                    args.sequence, int(rows), sweep.output_width, multiprocessors
                ),
            }
            for rows, details in aggregate.items()
        },
        'output_modes': output_modes,
        'combined_reuse': combined,
        'exact_divisibility': {
            rows: {**details, 'sequence': exact_lengths[rows]}
            for rows, details in exact.items()
        },
    }
    if args.compact:
        result = compact_samples(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
