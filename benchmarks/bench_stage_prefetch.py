'''Isolate whole-block versus stage-local AIMDO/VBAR prefetch residency.

The four synthetic linear stages use H3-sized page footprints for QKV,
attention output, MLP expansion, and MLP reduction.  Each arm runs in a fresh
process with a five-page watermark: enough for any one stage, not the whole
block.  This tests the residency mechanism only, not end-to-end H3 latency.
'''

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys


PACK_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = Path(__file__).resolve().parent
COMFY_ROOT = Path(__file__).resolve().parents[3]
for _root in (str(BENCHMARK_ROOT), str(COMFY_ROOT), str(PACK_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import bench_aimdo_residency as base


STAGES = ('qkv', 'attention_out', 'mlp_expand', 'mlp_reduce')
PAGE_FOOTPRINTS = (4, 2, 5, 3)
ARMS = ('whole_block', 'stage_local')
RESULT_PREFIX = 'H3_STAGE_PREFETCH_RESULT='


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Compare whole-block and stage-local AIMDO prefetch scopes.'
    )
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--driver-sample-ms', type=int, default=10)
    parser.add_argument('--output')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    parser.add_argument('--_child-arm', choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.arm_names = tuple(name.strip() for name in args.arms.split(',') if name.strip())
    unknown = [name for name in args.arm_names if name not in ARMS]
    if unknown or not args.arm_names:
        parser.error('--arms must be a non-empty subset of %s' % ','.join(ARMS))
    if args.device < 0 or args.driver_sample_ms <= 0:
        parser.error('device and sampling interval are invalid')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU preflight'
        )
    return args


def expected_stage_bytes():
    return dict(zip(STAGES, (pages * base.PAGE_SIZE for pages in PAGE_FOOTPRINTS)))


def _run_child(args):
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', base.PRODUCTION_ALLOC_CONF)
    sys.argv = [sys.argv[0], '--enable-dynamic-vram', '--async-offload', '2']
    import comfy.options
    comfy.options.enable_args_parsing()

    import comfy_aimdo.control as aimdo_control
    if not aimdo_control.init():
        raise RuntimeError('AIMDO initialization failed')
    import torch
    import comfy.memory_management
    import comfy.model_management as model_management
    import comfy.model_prefetch as model_prefetch
    import comfy.ops as ops
    import comfy_aimdo.host_buffer as host_buffer
    import comfy_aimdo.model_vbar as model_vbar

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    device = torch.device('cuda', args.device)
    torch.cuda.set_device(device)
    if not aimdo_control.init_device(args.device):
        raise RuntimeError('AIMDO device initialization failed')
    comfy.memory_management.aimdo_enabled = True

    base.PAGE_FOOTPRINTS = PAGE_FOOTPRINTS
    total_bytes = sum(PAGE_FOOTPRINTS) * base.PAGE_SIZE
    pin_state = base._make_pin_state(host_buffer, total_bytes)
    vbar, layers, factors = base._make_layers(
        torch, ops, model_vbar, args.device, pin_state
    )
    vbar.set_watermark(max(PAGE_FOOTPRINTS) * base.PAGE_SIZE)
    context, events, original_fault, original_unpin = base._instrument_faults(
        model_vbar, layers
    )
    torch.cuda.synchronize(device)
    baseline_free, total = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    sampler = base.DriverVramSampler(args.device, args.driver_sample_ms)
    sampler.start()
    observations = []

    def resolve_and_check(index):
        layer = layers[index]
        context['pass_index'] = 0
        context['module_index'] = index
        weight, bias = ops.resolve_cast_module_with_vbar(
            layer, torch.bfloat16, device, torch.bfloat16, None, False
        )
        x = torch.zeros((1, layer.in_features), dtype=torch.bfloat16, device=device)
        expected_base = torch.linspace(-1, 1, base.OUT_FEATURES, dtype=torch.bfloat16, device=device)
        x[:, :base.OUT_FEATURES] = expected_base
        output = torch.nn.functional.linear(x, weight, bias).squeeze(0)
        torch.cuda.synchronize(device)
        error = float((output.float() - expected_base.float() * factors[index]).abs().max().item())
        observations.append({
            'stage': STAGES[index],
            'max_abs_error': error,
            'memory': base._memory_snapshot(
                torch, device, vbar, model_management, aimdo_control
            ),
        })
        del weight, bias, x, expected_base, output

    try:
        if args._child_arm == 'whole_block':
            stream = ops.cast_modules_with_vbar(
                layers, None, device, None, True
            )
            model_management.sync_stream(device, stream)
            for index in range(len(layers)):
                resolve_and_check(index)
            model_prefetch.cleanup_prefetched_modules(object(), layers)
        else:
            for index, layer in enumerate(layers):
                stream = ops.cast_modules_with_vbar(
                    [layer], None, device, None, True
                )
                model_management.sync_stream(device, stream)
                resolve_and_check(index)
                model_prefetch.cleanup_prefetched_modules(object(), [layer])
        torch.cuda.synchronize(device)
        final = base._memory_snapshot(
            torch, device, vbar, model_management, aimdo_control
        )
    finally:
        sampler.stop()
        model_vbar.vbar_fault = original_fault
        model_vbar.vbar_unpin = original_unpin

    peak_driver = sampler.peak_mib()
    driver_baseline = sampler.samples[0] if sampler.samples else None
    failures = [
        '%s output max_abs_error %.6g' % (row['stage'], row['max_abs_error'])
        for row in observations if row['max_abs_error'] != 0
    ]
    if final['pinned_pages']:
        failures.append('cleanup left %d VBAR pages pinned' % final['pinned_pages'])
    result = {
        'arm': args._child_arm,
        'status': 'pass' if not failures else 'fail',
        'failures': failures,
        'stages': STAGES,
        'page_footprints': PAGE_FOOTPRINTS,
        'watermark_pages': int(vbar.get_watermark()),
        'events': events,
        'observations': observations,
        'peak_torch_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
        'peak_torch_reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
        'driver_peak_delta_mib': (
            peak_driver - driver_baseline if peak_driver is not None and driver_baseline is not None
            else (
                max(row['memory']['whole_device_used_bytes'] for row in observations)
                - (total - baseline_free)
            ) / 2 ** 20
        ),
        'final_memory': final,
    }
    model_management.reset_cast_buffers()
    del layers, vbar, pin_state
    gc.collect()
    torch.cuda.synchronize(device)
    aimdo_control.deinit()
    return result


def _run_arm(args, arm):
    command = [
        sys.executable, str(Path(__file__).resolve()), '--_child-arm', arm,
        '--device', str(args.device), '--driver-sample-ms', str(args.driver_sample_ms),
        '--i-understand-this-uses-gpu',
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    marker = next((
        line[len(RESULT_PREFIX):] for line in reversed(completed.stdout.splitlines())
        if line.startswith(RESULT_PREFIX)
    ), None)
    if completed.returncode or marker is None:
        raise RuntimeError(
            '%s child failed (%d)\nstdout:\n%s\nstderr:\n%s'
            % (arm, completed.returncode, completed.stdout, completed.stderr)
        )
    return json.loads(marker)


def main(argv=None):
    args = parse_args(argv)
    if args._child_arm:
        result = _run_child(args)
        print(RESULT_PREFIX + json.dumps(result, separators=(',', ':')))
        return 0 if result['status'] == 'pass' else 1
    results = [_run_arm(args, arm) for arm in args.arm_names]
    report = {
        'schema': 1,
        'experiment': 'h3_stage_prefetch_residency',
        'scope': 'synthetic Comfy AIMDO/VBAR mechanism, not end-to-end H3 timing',
        'stage_bytes': expected_stage_bytes(),
        'results': results,
    }
    serialized = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(serialized + '\n', encoding='utf-8')
    print(serialized)
    return 0 if all(row['status'] == 'pass' for row in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
