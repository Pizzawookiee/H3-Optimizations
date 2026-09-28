"""Run the focused Nsight Compute comparison for the FP8 experiment."""

from __future__ import annotations

import pathlib
import os
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
NCU = pathlib.Path(
    r'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1'
    r'\target\windows-desktop-win7-x64\ncu.exe'
)

ARMS = (
    'int8_64x64',
    'fp8_exp2_64x64',
    'fp8_sage_denom_64x64',
    'fp8_expcast_64x64',
    'fp8_v_smooth_exp2_64x64',
    'fp8_v_smooth_expcast_64x64',
)

METRICS = ','.join((
    'gpu__time_duration.sum',
    'launch__registers_per_thread',
    'launch__shared_mem_per_block',
    'launch__stack_size',
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'smsp__pcsamp_sample_count',
    'smsp__pcsamp_warps_issue_stalled_barrier',
    'smsp__pcsamp_warps_issue_stalled_long_scoreboard',
    'smsp__pcsamp_warps_issue_stalled_short_scoreboard',
    'smsp__pcsamp_warps_issue_stalled_math_pipe_throttle',
    'smsp__pcsamp_warps_issue_stalled_mio_throttle',
    'smsp__pcsamp_warps_issue_stalled_wait',
    'smsp__pcsamp_warps_issue_stalled_not_selected',
    'smsp__inst_executed_op_ldgsts.sum',
    'smsp__sass_inst_executed_op_local_ld.sum',
    'smsp__sass_inst_executed_op_local_st.sum',
))


def main():
    output = COMFY_ROOT / '.agent' / 'artifacts' / 'nsight-fp8-expcast-64'
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONNOUSERSITE'] = '1'
    environment.pop('PYTHONHOME', None)
    environment.pop('PYTHONPATH', None)
    for arm in ARMS:
        command = [
            str(NCU),
            '--force-overwrite',
            '--target-processes', 'all',
            '--kernel-name', 'regex:qk_int_sv_i8_sparse_attn_kernel.*',
            '--launch-count', '1',
            '--metrics', METRICS,
            '--export', str(output / arm),
            sys.executable,
            str(ROOT / 'benchmarks' / 'bench_fp8_expcast_64.py'),
            '--library', str(
                ROOT / 'native' / 'build-opt-ninja' / 'h3_int8_attention.dll'
            ),
            '--tokens', '4096',
            '--heads', '56',
            '--grouping', 'known',
            '--skip-reference',
            '--warmup', '0',
            '--iterations', '1',
            '--arms', arm,
        ]
        completed = subprocess.run(
            command, cwd=COMFY_ROOT, env=environment, check=False
        )
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
