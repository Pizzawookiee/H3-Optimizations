'''Compare current and route-before-V Kitchen producer ordering.'''

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parent
ARM_SCRIPT = BENCHMARK_ROOT / 'bench_kitchen_qkv_arm.py'
ORDERS = ('full_qkv', 'current', 'route_before_v')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Compare Kitchen QKV V-finalization ordering in a real H3 block.'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--qkv-chunk-rows', type=int, default=4096)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=54_006)
    parser.add_argument('--video-start', type=int, default=256)
    parser.add_argument('--video-budget', type=float, default=0.3)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--forwards', type=int, default=24)
    parser.add_argument('--seed', type=int, default=4321)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--result-dir', required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if args.qkv_chunk_rows <= 0 or args.qkv_chunk_rows % 128:
        parser.error('--qkv-chunk-rows must be a positive multiple of 128')
    if args.repeats <= 0:
        parser.error('--repeats must be positive')
    if not Path(args.result_dir).is_dir():
        parser.error('--result-dir must be an existing directory')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU preflight'
        )
    return args


def run_order(args, order, repeat):
    label = '%s_r%d' % (order, repeat)
    output = Path(args.result_dir).resolve() / ('%s.json' % label)
    command = [
        args.python,
        str(ARM_SCRIPT),
        '--checkpoint', str(args.checkpoint),
        '--qkv-chunk-rows', str(args.qkv_chunk_rows),
        '--producer-order', order,
        '--block', str(args.block),
        '--sequence', str(args.sequence),
        '--video-start', str(args.video_start),
        '--video-budget', str(args.video_budget),
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
    expected_projector = {
        'full_qkv': None,
        'current': 'chunked_kitchen_qkv',
        'route_before_v': 'benchmark_chunked_kitchen_qkv_route_before_v',
    }[order]
    expected_fused = order != 'full_qkv'
    if (
        route.get('sparse_architecture') != 'comfy_kitchen_int8'
        or route.get('fused_qkv') is not expected_fused
        or route.get('qkv_projector') != expected_projector
    ):
        raise SystemExit('%s did not reach the required route' % label)
    expected_calls = (
        1
        if order == 'full_qkv'
        else int(result['qkv_chunks_per_block']) + 1
    )
    calls = {
        int(forward['counts'].get('qkv_linear', 0))
        for forward in result['timing']['forwards']
    }
    if calls != {expected_calls}:
        raise SystemExit('%s made unexpected QKV calls %r' % (label, sorted(calls)))
    return result


def stage_median(result, name):
    return float(result['timing']['stages'][name]['median_gpu_ms'])


def order_summary(samples):
    stages = (
        'attention_pre_kernel',
        'attention_kernel',
        'attention_out',
        'attention_total',
        'block',
    )
    return {
        '%s_median_ms' % stage: statistics.median(
            stage_median(sample, stage) for sample in samples
        )
        for stage in stages
    }


def fingerprint_delta(left, right):
    left_values = [value for row in left for value in row]
    right_values = [value for row in right for value in row]
    return max(
        abs(float(actual) - float(expected))
        for actual, expected in zip(left_values, right_values)
    )


def main(argv=None):
    args = parse_args(argv)
    results = []
    paired = []
    for repeat in range(args.repeats):
        order = ORDERS if repeat % 2 == 0 else tuple(reversed(ORDERS))
        pair = {name: run_order(args, name, repeat) for name in order}
        paired.append({
            'repeat': repeat,
            'seed': int(args.seed + repeat),
            'current_fingerprint_max_abs_vs_full': fingerprint_delta(
                pair['full_qkv']['output_fingerprint'],
                pair['current']['output_fingerprint'],
            ),
            'route_before_v_fingerprint_max_abs_vs_full': fingerprint_delta(
                pair['full_qkv']['output_fingerprint'],
                pair['route_before_v']['output_fingerprint'],
            ),
            'current_attention_pre_kernel_ms': stage_median(
                pair['current'], 'attention_pre_kernel'
            ),
            'route_before_v_attention_pre_kernel_ms': stage_median(
                pair['route_before_v'], 'attention_pre_kernel'
            ),
            'current_attention_kernel_ms': stage_median(
                pair['current'], 'attention_kernel'
            ),
            'route_before_v_attention_kernel_ms': stage_median(
                pair['route_before_v'], 'attention_kernel'
            ),
            'current_attention_total_ms': stage_median(
                pair['current'], 'attention_total'
            ),
            'route_before_v_attention_total_ms': stage_median(
                pair['route_before_v'], 'attention_total'
            ),
            'current_block_ms': stage_median(pair['current'], 'block'),
            'route_before_v_block_ms': stage_median(
                pair['route_before_v'], 'block'
            ),
        })
        results.extend(pair.values())

    grouped = {
        name: [result for result in results if result['producer_order'] == name]
        for name in ORDERS
    }
    summary = {name: order_summary(samples) for name, samples in grouped.items()}
    output = Path(args.result_dir).resolve() / 'ordering_summary.json'
    output.write_text(
        json.dumps(
            {
                'qkv_chunk_rows': int(args.qkv_chunk_rows),
                'sequence': int(args.sequence),
                'summary': summary,
                'paired': paired,
                'runs': results,
            },
            indent=2,
            sort_keys=True,
        ) + '\n',
        encoding='utf-8',
    )
    for name in ORDERS:
        item = summary[name]
        print(
            '%-14s pre-kernel %.3f ms, kernel %.3f ms, attention %.3f ms, '
            'block %.3f ms'
            % (
                name,
                item['attention_pre_kernel_median_ms'],
                item['attention_kernel_median_ms'],
                item['attention_total_median_ms'],
                item['block_median_ms'],
            )
        )
    print('wrote %s' % output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
