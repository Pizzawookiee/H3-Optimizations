'''Drive bench_h3_block_allocator arms, one per fresh Python process.

Each arm must start from a cold CUDA caching allocator. Running two arms in one
interpreter would let the first arm's cached segments become the second arm's
starting state, which is exactly the variable under study, so every arm gets its
own interpreter here.

The default arms are:

  4096:4096  production control
  2816:4096  near match, QKV segment larger than the MLP request
  2688:4096  near match, QKV segment equal to the MLP request
  3072:4608  exact request match, positive control
  2048:3072  exact request match at a smaller size, positive control

The last two change the MLP rows away from the measured MLP optimum on purpose.
They are mechanism probes, not production candidates: if an exactly matched pair
does not produce reuse, no near match will either.
'''

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parent
ARM_SCRIPT = BENCHMARK_ROOT / 'bench_h3_block_allocator.py'
DEFAULT_ARMS = '4096:4096,2816:4096,2688:4096,3072:4608,2048:3072'


def parse_arms(value):
    arms = []
    for item in str(value).split(','):
        item = item.strip()
        if not item:
            continue
        qkv, _, mlp = item.partition(':')
        if not mlp:
            raise ValueError('each arm must be written as QKVROWS:MLPROWS')
        arm = (int(qkv), int(mlp))
        if arm in arms:
            raise ValueError('duplicate arm %d:%d' % arm)
        arms.append(arm)
    if not arms:
        raise ValueError('at least one arm is required')
    return tuple(arms)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run one allocator-lifetime arm per fresh Python process.'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--arms', default=DEFAULT_ARMS)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=54006)
    parser.add_argument('--video-start', type=int, default=256)
    parser.add_argument('--video-budget', type=float, default=0.5)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--forwards', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument(
        '--result-dir',
        required=True,
        help='existing directory that receives one JSON file per arm run',
    )
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    try:
        args.arms = parse_arms(args.arms)
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


def run_arm(args, qkv_rows, mlp_rows, repeat):
    label = 'qkv%d_mlp%d_r%d' % (qkv_rows, mlp_rows, repeat)
    output = Path(args.result_dir).resolve() / ('%s.json' % label)
    command = [
        args.python,
        str(ARM_SCRIPT),
        '--checkpoint', str(args.checkpoint),
        '--block', str(args.block),
        '--sequence', str(args.sequence),
        '--video-start', str(args.video_start),
        '--video-budget', str(args.video_budget),
        '--qkv-chunk-rows', str(qkv_rows),
        '--mlp-chunk-rows', str(mlp_rows),
        '--warmup', str(args.warmup),
        '--forwards', str(args.forwards),
        # Repeats must differ, or a repeat only re-measures one fixed input.
        '--seed', str(args.seed + repeat),
        '--label', label,
        '--output', str(output),
        '--i-understand-this-uses-gpu',
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(
            'arm %s failed with exit code %d' % (label, completed.returncode)
        )
    return json.loads(output.read_text(encoding='utf-8'))


def summarize(results):
    header = (
        '%-22s %-18s %-7s %10s %10s %8s %10s'
        % (
            'arm',
            'predicted',
            'reuse',
            'reserved',
            'inactive',
            'segments',
            'median_ms',
        )
    )
    lines = [header, '-' * len(header)]
    for result in results:
        last = result['stats']['after_last_forward']
        lines.append(
            '%-22s %-18s %-7s %9.3fG %9.1fM %8d %10.3f'
            % (
                result['label'],
                result['prediction']['verdict'],
                'yes' if result['reuse']['reused_every_forward'] else (
                    'part' if result['reuse']['reused_any_forward'] else 'no'
                ),
                last.get('reserved_bytes.all.current', 0) / 2**30,
                last.get('inactive_split_bytes.all.current', 0) / 2**20,
                last.get('segment.large_pool.current', 0),
                result['timing']['steady_median_ms'],
            )
        )
    return '\n'.join(lines)


def main(argv=None):
    args = parse_args(argv)
    if not ARM_SCRIPT.is_file():
        raise SystemExit('missing %s' % ARM_SCRIPT)
    results = []
    for repeat in range(args.repeats):
        for qkv_rows, mlp_rows in args.arms:
            results.append(run_arm(args, qkv_rows, mlp_rows, repeat))
    summary_path = Path(args.result_dir).resolve() / 'sweep_summary.json'
    summary_path.write_text(
        json.dumps(
            [
                {
                    'label': result['label'],
                    'qkv_chunk_rows': result['qkv_chunk_rows'],
                    'mlp_chunk_rows': result['mlp_chunk_rows'],
                    'prediction': result['prediction'],
                    'reuse': result['reuse'],
                    'timing': result['timing'],
                    'stats': result['stats'],
                }
                for result in results
            ],
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )
    print(summarize(results))
    print('wrote %s' % summary_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
