'''Sweep shipped Kitchen QKV chunk sizes in fresh Python processes.'''

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parent
ARM_SCRIPT = BENCHMARK_ROOT / 'bench_kitchen_qkv_arm.py'
DEFAULT_CHUNKS = (
    '512,768,1024,1536,2048,3072,4096,6144,8192,12288,16384,'
    '24576,32768,full'
)


def aligned_full(sequence, alignment=128):
    return ((int(sequence) + int(alignment) - 1) // int(alignment)) * int(alignment)


def parse_chunks(value, sequence):
    chunks = []
    for item in str(value).split(','):
        item = item.strip().lower()
        if not item:
            continue
        rows = aligned_full(sequence) if item in ('full', 'sequence') else int(item)
        if rows <= 0 or rows % 128:
            raise ValueError('chunk rows must be positive multiples of 128')
        if rows not in chunks:
            chunks.append(rows)
    if not chunks:
        raise ValueError('at least one chunk size is required')
    return tuple(chunks)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Sweep Kitchen QKV chunks through a real H3 block.'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--chunks', default=DEFAULT_CHUNKS)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=54_006)
    parser.add_argument('--video-start', type=int, default=256)
    parser.add_argument('--video-budget', type=float, default=0.3)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--forwards', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--result-dir', required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    try:
        args.chunk_sizes = parse_chunks(args.chunks, args.sequence)
    except ValueError as exc:
        parser.error(str(exc))
    if args.repeats <= 0:
        parser.error('--repeats must be positive')
    if not Path(args.result_dir).is_dir():
        parser.error('--result-dir must be an existing directory')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU preflight'
        )
    return args


def arm_order(chunks, repeat):
    chunks = list(chunks)
    if repeat % 2:
        chunks.reverse()
    if not chunks:
        return chunks
    rotation = (repeat // 2) % len(chunks)
    return chunks[rotation:] + chunks[:rotation]


def run_arm(args, rows, repeat):
    label = 'qkv%d_r%d' % (rows, repeat)
    output = Path(args.result_dir).resolve() / ('%s.json' % label)
    command = [
        args.python,
        str(ARM_SCRIPT),
        '--checkpoint', str(args.checkpoint),
        '--block', str(args.block),
        '--sequence', str(args.sequence),
        '--video-start', str(args.video_start),
        '--video-budget', str(args.video_budget),
        '--qkv-chunk-rows', str(rows),
        '--warmup', str(args.warmup),
        '--forwards', str(args.forwards),
        '--seed', str(args.seed + repeat),
        '--label', label,
        '--output', str(output),
        '--i-understand-this-uses-gpu',
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit('%s failed with exit code %d' % (label, completed.returncode))
    result = json.loads(output.read_text(encoding='utf-8'))
    route = result.get('route', {})
    if (
        route.get('sparse_architecture') != 'comfy_kitchen_int8'
        or route.get('qkv_projector') != 'chunked_kitchen_qkv'
        or route.get('fused_qkv') is not True
    ):
        raise SystemExit('%s did not reach the required Kitchen route' % label)
    expected_calls = int(result['qkv_chunks_per_block']) + 1
    actual_calls = {
        int(forward['counts'].get('qkv_linear', 0))
        for forward in result['timing']['forwards']
    }
    if actual_calls != {expected_calls}:
        raise SystemExit(
            '%s made QKV calls %r, expected %d'
            % (label, sorted(actual_calls), expected_calls)
        )
    return result


def median_metric(results, stage):
    return statistics.median(
        result['timing']['stages'][stage]['median_gpu_ms'] for result in results
    )


def steady_stage_samples(results, stage):
    values = []
    for result in results:
        forwards = result['timing']['forwards']
        for forward in forwards[len(forwards) // 2:]:
            values.append(float(forward['stages'][stage]['gpu_ms']))
    return values


def median_absolute_deviation(values):
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def aggregate(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result['qkv_chunk_rows'], []).append(result)
    rows = []
    for chunk_rows, samples in sorted(grouped.items()):
        producer_samples = steady_stage_samples(samples, 'qkv_producer_total')
        block_samples = steady_stage_samples(samples, 'block')
        rows.append({
            'qkv_chunk_rows': int(chunk_rows),
            'qkv_chunks_per_block': samples[0]['qkv_chunks_per_block'],
            'repeats': len(samples),
            'qkv_linear_calls': statistics.median(
                statistics.median(
                    forward['counts'].get('qkv_linear', 0)
                    for forward in sample['timing']['forwards']
                )
                for sample in samples
            ),
            'qkv_producer_median_ms': median_metric(samples, 'qkv_producer_total'),
            'qkv_producer_mad_ms': median_absolute_deviation(producer_samples),
            'sparse_prepare_median_ms': median_metric(samples, 'sparse_route_prepare'),
            'attention_kernel_median_ms': median_metric(samples, 'attention_kernel'),
            'attention_total_median_ms': median_metric(samples, 'attention_total'),
            'block_median_ms': median_metric(samples, 'block'),
            'block_mad_ms': median_absolute_deviation(block_samples),
            'block_peak_allocated_bytes': max(
                sample['memory']['uninstrumented_block']['peak_allocated_bytes']
                for sample in samples
            ),
            'block_peak_reserved_bytes': max(
                sample['memory']['uninstrumented_block']['peak_reserved_bytes']
                for sample in samples
            ),
            'qkv_phase_peak_allocated_bytes': max(
                sample['memory']['phases']['qkv_producer_total']['peak_allocated_bytes']
                for sample in samples
            ),
            'qkv_phase_incremental_allocated_bytes': max(
                sample['memory']['phases']['qkv_producer_total']['incremental_allocated_bytes']
                for sample in samples
            ),
        })
    return rows


def fingerprint_parity(results):
    grouped = {}
    for result in results:
        grouped.setdefault(int(result['seed']), []).append(result)
    reports = []
    for seed, samples in sorted(grouped.items()):
        reference = samples[0]
        reference_values = [
            value
            for row in reference['output_fingerprint']
            for value in row
        ]
        comparisons = []
        for sample in samples[1:]:
            values = [
                value
                for row in sample['output_fingerprint']
                for value in row
            ]
            delta = max(
                abs(float(actual) - float(expected))
                for actual, expected in zip(values, reference_values)
            )
            comparisons.append({
                'qkv_chunk_rows': int(sample['qkv_chunk_rows']),
                'max_abs': delta,
                'exact': delta == 0.0,
            })
        reports.append({
            'seed': seed,
            'reference_qkv_chunk_rows': int(reference['qkv_chunk_rows']),
            'comparisons': comparisons,
            'exact': all(item['exact'] for item in comparisons),
        })
    return reports


def format_summary(rows):
    header = (
        '%-8s %6s %9s %10s %10s %10s %10s %11s %11s'
        % (
            'rows', 'chunks', 'qkv_calls', 'producer', 'kernel', 'attention',
            'block', 'alloc_MiB', 'reserve_MiB'
        )
    )
    lines = [header, '-' * len(header)]
    for row in rows:
        lines.append(
            '%-8d %6d %9.1f %10.3f %10.3f %10.3f %10.3f %11.1f %11.1f'
            % (
                row['qkv_chunk_rows'],
                row['qkv_chunks_per_block'],
                row['qkv_linear_calls'],
                row['qkv_producer_median_ms'],
                row['attention_kernel_median_ms'],
                row['attention_total_median_ms'],
                row['block_median_ms'],
                row['block_peak_allocated_bytes'] / 2**20,
                row['block_peak_reserved_bytes'] / 2**20,
            )
        )
    return '\n'.join(lines)


def main(argv=None):
    args = parse_args(argv)
    if not ARM_SCRIPT.is_file():
        raise SystemExit('missing %s' % ARM_SCRIPT)
    results = []
    for repeat in range(args.repeats):
        for rows in arm_order(args.chunk_sizes, repeat):
            results.append(run_arm(args, rows, repeat))
    summary = aggregate(results)
    parity = fingerprint_parity(results)
    output = Path(args.result_dir).resolve() / 'sweep_summary.json'
    output.write_text(
        json.dumps(
            {
                'sequence': int(args.sequence),
                'video_start': int(args.video_start),
                'video_budget': float(args.video_budget),
                'arms': summary,
                'output_fingerprint_parity': parity,
                'runs': results,
            },
            indent=2,
            sort_keys=True,
        ) + '\n',
        encoding='utf-8',
    )
    print(format_summary(summary))
    if not all(report['exact'] for report in parity):
        raise SystemExit('output fingerprint changed between chunk-size arms')
    print('wrote %s' % output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
