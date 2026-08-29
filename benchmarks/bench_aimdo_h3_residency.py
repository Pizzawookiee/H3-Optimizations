'''Compare AIMDO residency limits on a real five-frame MiniMax H3 request.

This specializes ``bench_attention_arms`` without duplicating its live-schema
graph builder, prompt monitoring, or whole-device VRAM sampler. Every arm uses
the same real H3 checkpoint, blank conditioning, seed, schedule, and sampler;
only the H3 AIMDO Residency Limiter setting changes.

Use ``--unload-between-arms`` so each arm reloads the diffusion weights onto a
mostly free card while the cached text conditioning survives. That is the
condition in which stock AIMDO is free to retain as much of the model as it
finds useful.
'''

from __future__ import annotations

import sys

import bench_attention_arms as benchmark


benchmark.WORKLOADS = {'5f': 5}
benchmark.DEFAULT_PROMPT = ''
benchmark.COMMON_SUFFIX = [
    entry for entry in benchmark.COMMON_SUFFIX
    if entry[0] != 'H3AIMDOResidencyLimiter'
]
benchmark.ARMS = {
    'aimdo_stock': [
        ('H3AIMDOResidencyLimiter', {'residency': 'stock'}),
    ],
    'aimdo_0': [
        ('H3AIMDOResidencyLimiter', {'residency': '0 blocks'}),
    ],
    'aimdo_1': [
        ('H3AIMDOResidencyLimiter', {'residency': '1 block'}),
    ],
    'aimdo_2': [
        ('H3AIMDOResidencyLimiter', {'residency': '2 blocks'}),
    ],
    'aimdo_4': [
        ('H3AIMDOResidencyLimiter', {'residency': '4 blocks'}),
    ],
}
benchmark.DEFAULT_ARMS = ','.join(benchmark.ARMS)
benchmark.ARM_LABELS = {
    'aimdo_stock': 'AIMDO stock',
    'aimdo_0': 'AIMDO 0 block-equivalents',
    'aimdo_1': 'AIMDO 1 block-equivalent',
    'aimdo_2': 'AIMDO 2 block-equivalents',
    'aimdo_4': 'AIMDO 4 block-equivalents',
}


def benchmark_argv(argv):
    argv = list(argv)
    if '--unique-arm-seeds' not in argv:
        argv.append('--unique-arm-seeds')
    return argv


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    return benchmark.main(benchmark_argv(argv))


if __name__ == '__main__':
    sys.exit(main())
