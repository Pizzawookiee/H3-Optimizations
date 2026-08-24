'''Measure AIMDO residency limits through ComfyUI's real weight-cast path.

The benchmark uses four synthetic Comfy linear modules whose weights occupy
one, two, three, and four AIMDO pages. Each residency arm runs in a fresh
process. Two passes distinguish initial VBAR population from warm resident
hits, while weights rejected by the watermark are allowed to follow ComfyUI's
normal temporary-buffer streaming path.

This is intentionally not a raw ``vbar.fault()`` test. A rejected fault is a
valid offload signal; correctness is established only after Comfy has streamed
the weight, executed the linear, and released the cast.
'''

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


PAGE_SIZE = 32 * 1024 * 1024
PAGE_FOOTPRINTS = (1, 2, 3, 4)
LEVELS = ('stock', '0', '1', '2', '4')
LEVEL_BLOCKS = {'0': 0, '1': 1, '2': 2, '4': 4}
OUT_FEATURES = 4096
DTYPE_BYTES = 2
RESULT_PREFIX = 'H3_AIMDO_RESULT='
PRODUCTION_ALLOC_CONF = (
    'backend:native,garbage_collection_threshold:0.95,expandable_segments:False'
)

PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = next(
    (parent for parent in PACK_ROOT.parents if (parent / 'comfy').is_dir()),
    PACK_ROOT.parents[1],
)
for _root in (str(COMFY_ROOT), str(PACK_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)


class BenchError(RuntimeError):
    pass


def parse_levels(value):
    levels = []
    for item in str(value).split(','):
        item = item.strip().lower().removesuffix(' blocks').removesuffix(' block')
        if item not in LEVELS:
            raise argparse.ArgumentTypeError(
                'levels must be a comma-separated subset of stock,0,1,2,4'
            )
        if item not in levels:
            levels.append(item)
    if not levels:
        raise argparse.ArgumentTypeError('at least one residency level is required')
    return tuple(levels)


def cap_pages(level, footprints=PAGE_FOOTPRINTS):
    if level == 'stock':
        return sum(footprints)
    blocks = LEVEL_BLOCKS[level]
    return sum(sorted(footprints, reverse=True)[:blocks])


def expected_routes(level, footprints=PAGE_FOOTPRINTS):
    watermark = cap_pages(level, footprints)
    offset = 0
    routes = []
    for pages in footprints:
        stop = offset + pages
        routes.append('vbar' if stop <= watermark else 'stream')
        offset = stop
    return tuple(routes)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Measure AIMDO VBAR fills, resident hits, Comfy streaming fallback, '
            'pin cleanup, parity, and whole-device VRAM by residency level.'
        )
    )
    parser.add_argument('--levels', type=parse_levels, default=parse_levels(','.join(LEVELS)))
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--driver-sample-ms', type=int, default=25)
    parser.add_argument('--output')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--disable-pinned-memory', action='store_true')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    parser.add_argument('--_child-level', choices=LEVELS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error('--device must be non-negative')
    if args.passes < 2:
        parser.error('--passes must be at least 2 to observe warm residency')
    if args.driver_sample_ms <= 0:
        parser.error('--driver-sample-ms must be positive')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU preflight'
        )
    return args


class DriverVramSampler:
    '''Sample whole-device VRAM with nvidia-smi when it is available.'''

    def __init__(self, device, interval_ms):
        self.device = int(device)
        self.interval_ms = int(interval_ms)
        self.samples = []
        self.process = None
        self.thread = None

    def start(self):
        if shutil.which('nvidia-smi') is None:
            return
        self.process = subprocess.Popen(
            [
                'nvidia-smi',
                '--id=%d' % self.device,
                '--query-gpu=memory.used',
                '--format=csv,noheader,nounits',
                '-lms',
                str(self.interval_ms),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 2.0
        while not self.samples and self.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)

    def _read(self):
        for line in self.process.stdout:
            try:
                self.samples.append(float(line.strip()))
            except ValueError:
                continue

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.process = None
        self.thread = None

    def peak_mib(self):
        return max(self.samples, default=None)


def _memory_snapshot(torch, device, vbar, model_management, aimdo_control):
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    buffers = {
        str(index): int(buffer.size())
        for index, buffer in enumerate(model_management.STREAM_AIMDO_CAST_BUFFERS.values())
    }
    residency = tuple(int(value) for value in vbar.get_residency())
    return {
        'whole_device_used_bytes': int(total_bytes - free_bytes),
        'torch_allocated_bytes': int(torch.cuda.memory_allocated(device)),
        'torch_reserved_bytes': int(torch.cuda.memory_reserved(device)),
        'aimdo_total_bytes': int(aimdo_control.get_total_vram_usage()),
        'vbar_loaded_bytes': int(vbar.loaded_size()),
        'resident_pages': sum(bool(value & 1) for value in residency),
        'pinned_pages': sum(bool(value & 2) for value in residency),
        'residency': residency,
        'cast_buffer_bytes': sum(buffers.values()),
        'cast_buffers': buffers,
    }


def _make_pin_state(host_buffer, total_bytes):
    def bucket(prewarm):
        return (
            host_buffer.HostBuffer(0, prewarm, total_bytes),
            [],
            [-1],
            [0],
            [0],
            {},
        )

    return {
        'weights': bucket(64 * 1024 * 1024),
        'weights-loaded': bucket(64 * 1024 * 1024),
    }


def _make_layers(torch, ops, model_vbar, device_index, pin_state):
    total_pages = sum(PAGE_FOOTPRINTS)
    vbar = model_vbar.ModelVBAR(total_pages * PAGE_SIZE, device_index)
    vbar.prioritize()
    layers = []
    diagonal = torch.arange(OUT_FEATURES)
    factors = (1.0, 0.5, -1.0, 2.0)

    for index, (pages, factor) in enumerate(zip(PAGE_FOOTPRINTS, factors)):
        in_features = pages * PAGE_SIZE // (OUT_FEATURES * DTYPE_BYTES)
        weight = torch.zeros((OUT_FEATURES, in_features), dtype=torch.bfloat16)
        weight[diagonal, diagonal] = factor
        layer = ops.manual_cast.Linear(
            in_features,
            OUT_FEATURES,
            bias=False,
            device='cpu',
            dtype=torch.bfloat16,
        )
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.bias = None
        layer.seed_key = 'synthetic_%d' % index
        layer._v = vbar.alloc(weight.numel() * weight.element_size())
        layer._v_signature = None
        if pin_state is not None:
            layer._pin_state = pin_state
        layers.append(layer)
    return vbar, layers, factors


class _DynamicPatcher:
    @staticmethod
    def is_dynamic():
        return True


def _apply_level(level, vbar, layers, limiter, mock):
    if level == 'stock':
        return
    with mock.patch.object(limiter, 'get_h3_blocks', return_value=tuple(layers)):
        limiter._apply_residency_cap(_DynamicPatcher(), LEVEL_BLOCKS[level])


def _instrument_faults(model_vbar, layers):
    allocations = {int(layer._v[1]): (index, layer) for index, layer in enumerate(layers)}
    events = []
    context = {'pass_index': None, 'module_index': None}
    original_fault = model_vbar.vbar_fault
    original_unpin = model_vbar.vbar_unpin

    def fault(allocation):
        signature = original_fault(allocation)
        index, layer = allocations[int(allocation[1])]
        resident = model_vbar.vbar_signature_compare(signature, layer._v_signature)
        events.append(
            {
                'event': 'fault',
                'pass': context['pass_index'],
                'module': index,
                'pages': PAGE_FOOTPRINTS[index],
                'outcome': (
                    'streamed' if signature is None else
                    'resident_hit' if resident else
                    'resident_fill'
                ),
            }
        )
        return signature

    def unpin(allocation):
        result = original_unpin(allocation)
        if allocation is not None:
            index, _layer = allocations[int(allocation[1])]
            events.append(
                {
                    'event': 'unpin',
                    'pass': context['pass_index'],
                    'module': index,
                }
            )
        return result

    model_vbar.vbar_fault = fault
    model_vbar.vbar_unpin = unpin
    return context, events, original_fault, original_unpin


def _run_passes(
    torch,
    device,
    vbar,
    layers,
    factors,
    passes,
    ops,
    model_management,
    aimdo_control,
    context,
):
    observations = []
    expected_base = torch.linspace(
        -1.0,
        1.0,
        OUT_FEATURES,
        dtype=torch.bfloat16,
        device=device,
    )

    for pass_index in range(passes):
        for module_index, (layer, factor) in enumerate(zip(layers, factors)):
            context['pass_index'] = pass_index
            context['module_index'] = module_index
            x = torch.zeros((1, layer.in_features), dtype=torch.bfloat16, device=device)
            x[:, :OUT_FEATURES] = expected_base
            expected = expected_base * factor
            weight = bias = release = None
            try:
                weight, bias, release = ops.cast_bias_weight(
                    layer,
                    x,
                    offloadable=True,
                )
                after_cast = _memory_snapshot(
                    torch, device, vbar, model_management, aimdo_control
                )
                output = torch.nn.functional.linear(x, weight, bias).squeeze(0)
                torch.cuda.synchronize(device)
                error = (output.float() - expected.float()).abs()
                max_abs_error = float(error.max().item())
                output_sum = float(output.float().sum().item())
            finally:
                if release is not None:
                    ops.uncast_bias_weight(layer, weight, bias, release)
            del weight, bias, release
            torch.cuda.synchronize(device)
            after_release = _memory_snapshot(
                torch, device, vbar, model_management, aimdo_control
            )
            observations.append(
                {
                    'pass': pass_index,
                    'module': module_index,
                    'pages': PAGE_FOOTPRINTS[module_index],
                    'max_abs_error': max_abs_error,
                    'output_sum': output_sum,
                    'after_cast': after_cast,
                    'after_release': after_release,
                }
            )
            del x, expected, output, error
    return observations


def _validate_result(level, watermark, events, observations):
    failures = []
    routes = expected_routes(level)
    fault_events = [event for event in events if event['event'] == 'fault']
    for event in fault_events:
        expected = routes[event['module']]
        if expected == 'stream' and event['outcome'] != 'streamed':
            failures.append(
                'module %d pass %d expected streaming, got %s'
                % (event['module'], event['pass'], event['outcome'])
            )
        if expected == 'vbar':
            expected_outcome = 'resident_fill' if event['pass'] == 0 else 'resident_hit'
            if event['outcome'] != expected_outcome:
                failures.append(
                    'module %d pass %d expected %s, got %s'
                    % (event['module'], event['pass'], expected_outcome, event['outcome'])
                )

    for observation in observations:
        if observation['max_abs_error'] != 0.0:
            failures.append(
                'module %d pass %d parity error %.6g'
                % (
                    observation['module'],
                    observation['pass'],
                    observation['max_abs_error'],
                )
            )
        expected_pinned = (
            PAGE_FOOTPRINTS[observation['module']]
            if routes[observation['module']] == 'vbar'
            else 0
        )
        if observation['after_cast']['pinned_pages'] != expected_pinned:
            failures.append(
                'module %d pass %d pinned %d pages during cast; expected %d'
                % (
                    observation['module'],
                    observation['pass'],
                    observation['after_cast']['pinned_pages'],
                    expected_pinned,
                )
            )
        if observation['after_release']['pinned_pages']:
            failures.append(
                'module %d pass %d left %d pinned pages after release'
                % (
                    observation['module'],
                    observation['pass'],
                    observation['after_release']['pinned_pages'],
                )
            )

    if watermark != cap_pages(level):
        failures.append('watermark %d pages; expected %d' % (watermark, cap_pages(level)))
    return failures


def run_child(args):
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', PRODUCTION_ALLOC_CONF)
    comfy_argv = [
        sys.argv[0],
        '--enable-dynamic-vram',
        '--async-offload',
        '2',
    ]
    if args.disable_pinned_memory:
        comfy_argv.append('--disable-pinned-memory')
    sys.argv = comfy_argv

    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy_aimdo.control as aimdo_control

    if not aimdo_control.init():
        raise BenchError('AIMDO native library initialization failed')

    import torch
    from unittest import mock
    import comfy.memory_management
    import comfy.model_management as model_management
    import comfy.ops as ops
    import comfy_aimdo.host_buffer as host_buffer
    import comfy_aimdo.model_vbar as model_vbar
    from h3_optimizations import aimdo_limiter as limiter

    if not torch.cuda.is_available():
        raise BenchError('CUDA is unavailable')
    if args.device >= torch.cuda.device_count():
        raise BenchError(
            'CUDA device %d is unavailable; found %d device(s)'
            % (args.device, torch.cuda.device_count())
        )
    device = torch.device('cuda', args.device)
    if not aimdo_control.init_device(args.device):
        raise BenchError('AIMDO device %d initialization failed' % args.device)
    comfy.memory_management.aimdo_enabled = True
    if model_management.NUM_STREAMS != 2:
        raise BenchError(
            'expected two Comfy offload streams, got %d' % model_management.NUM_STREAMS
        )

    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)
    baseline_free, baseline_total = torch.cuda.mem_get_info(device)
    baseline_used = int(baseline_total - baseline_free)
    torch.cuda.reset_peak_memory_stats(device)

    sampler = DriverVramSampler(args.device, args.driver_sample_ms)
    sampler.start()
    driver_baseline_mib = sampler.samples[0] if sampler.samples else None
    vbar = layers = pin_state = None
    original_fault = original_unpin = None
    try:
        total_weight_bytes = sum(PAGE_FOOTPRINTS) * PAGE_SIZE
        pin_state = (
            None if args.disable_pinned_memory else
            _make_pin_state(host_buffer, total_weight_bytes)
        )
        vbar, layers, factors = _make_layers(
            torch, ops, model_vbar, args.device, pin_state
        )
        _apply_level(args._child_level, vbar, layers, limiter, mock)
        watermark = int(vbar.get_watermark())
        context, events, original_fault, original_unpin = _instrument_faults(
            model_vbar, layers
        )
        observations = _run_passes(
            torch,
            device,
            vbar,
            layers,
            factors,
            args.passes,
            ops,
            model_management,
            aimdo_control,
            context,
        )
        torch.cuda.synchronize(device)
        final_memory = _memory_snapshot(
            torch, device, vbar, model_management, aimdo_control
        )
        failures = _validate_result(
            args._child_level, watermark, events, observations
        )
        result = {
            'level': args._child_level,
            'status': 'pass' if not failures else 'fail',
            'failures': failures,
            'page_size_bytes': PAGE_SIZE,
            'module_page_footprints': PAGE_FOOTPRINTS,
            'expected_routes': expected_routes(args._child_level),
            'watermark_pages': watermark,
            'native_pages': int(vbar.get_nr_pages()),
            'passes': args.passes,
            'pinned_memory_enabled': not args.disable_pinned_memory,
            'baseline_whole_device_used_bytes': baseline_used,
            'peak_torch_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
            'peak_torch_reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
            'events': events,
            'observations': observations,
            'final_memory': final_memory,
        }
    finally:
        sampler.stop()
        if original_fault is not None:
            model_vbar.vbar_fault = original_fault
        if original_unpin is not None:
            model_vbar.vbar_unpin = original_unpin

    result['driver_peak_mib'] = sampler.peak_mib()
    if result['driver_peak_mib'] is not None and driver_baseline_mib is not None:
        result['driver_baseline_mib'] = driver_baseline_mib
        result['driver_peak_delta_mib'] = (
            result['driver_peak_mib'] - driver_baseline_mib
        )
    else:
        result['driver_baseline_mib'] = None
        peak_used = max(
            observation[phase]['whole_device_used_bytes']
            for observation in observations
            for phase in ('after_cast', 'after_release')
        )
        result['driver_peak_delta_mib'] = (peak_used - baseline_used) / (1024 ** 2)

    for layer in layers:
        for name in ('_v_weight', '_v_bias', '_prefetch'):
            if hasattr(layer, name):
                delattr(layer, name)
    model_management.reset_cast_buffers()
    del layers, vbar, pin_state
    gc.collect()
    torch.cuda.synchronize(device)
    aimdo_control.deinit()
    return result


def _run_level(args, level):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        '--_child-level',
        level,
        '--device',
        str(args.device),
        '--passes',
        str(args.passes),
        '--driver-sample-ms',
        str(args.driver_sample_ms),
        '--i-understand-this-uses-gpu',
    ]
    if args.disable_pinned_memory:
        command.append('--disable-pinned-memory')
    completed = subprocess.run(command, capture_output=True, text=True)
    marker = next(
        (
            line[len(RESULT_PREFIX):]
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(RESULT_PREFIX)
        ),
        None,
    )
    if completed.returncode or marker is None:
        raise BenchError(
            'level %s child failed with exit code %d\nstdout:\n%s\nstderr:\n%s'
            % (level, completed.returncode, completed.stdout, completed.stderr)
        )
    return json.loads(marker)


def summarize(result):
    fault_events = [event for event in result['events'] if event['event'] == 'fault']
    outcomes = {
        name: sum(event['outcome'] == name for event in fault_events)
        for name in ('resident_fill', 'resident_hit', 'streamed')
    }
    peak_cast_buffer = max(
        observation[phase]['cast_buffer_bytes']
        for observation in result['observations']
        for phase in ('after_cast', 'after_release')
    )
    peak_vbar_loaded = max(
        observation[phase]['vbar_loaded_bytes']
        for observation in result['observations']
        for phase in ('after_cast', 'after_release')
    )
    return {
        'level': result['level'],
        'status': result['status'],
        'watermark_pages': result['watermark_pages'],
        'final_resident_pages': result['final_memory']['resident_pages'],
        'resident_fills': outcomes['resident_fill'],
        'resident_hits': outcomes['resident_hit'],
        'streamed_faults': outcomes['streamed'],
        'peak_vbar_mib': peak_vbar_loaded / (1024 ** 2),
        'peak_cast_buffer_mib': peak_cast_buffer / (1024 ** 2),
        'whole_device_peak_delta_mib': result['driver_peak_delta_mib'],
        'max_abs_error': max(
            observation['max_abs_error'] for observation in result['observations']
        ),
    }


def render_table(results):
    rows = [summarize(result) for result in results]
    lines = [
        '| level | status | watermark | resident | fills | hits | streamed | '
        'VBAR peak MiB | cast buffers MiB | whole GPU delta MiB | max error |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            '| {level} | {status} | {watermark_pages} | {final_resident_pages} | '
            '{resident_fills} | {resident_hits} | {streamed_faults} | '
            '{peak_vbar_mib:.0f} | {peak_cast_buffer_mib:.0f} | '
            '{whole_device_peak_delta_mib:.0f} | {max_abs_error:.6g} |'.format(**row)
        )
    return '\n'.join(lines)


def main(argv=None):
    args = parse_args(argv)
    if args._child_level is not None:
        result = run_child(args)
        print(RESULT_PREFIX + json.dumps(result, separators=(',', ':')))
        return 0 if result['status'] == 'pass' else 1

    results = [_run_level(args, level) for level in args.levels]
    report = {
        'schema': 1,
        'description': (
            'Synthetic Comfy cast-path AIMDO residency and streaming behavior'
        ),
        'levels': list(args.levels),
        'page_footprints': PAGE_FOOTPRINTS,
        'passes': args.passes,
        'results': results,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_table(results))
    return 0 if all(result['status'] == 'pass' for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
