'''Benchmark synthetic non-ConvRot H3 MLP checkpoint representations.

One real ConvRot block supplies the source tensors. The benchmark dequantizes
them to BF16, requantizes a second copy to native Comfy FP8, and then exercises
the production non-ConvRot provider primitives beside the checkpoint-native
ConvRot two-slice path. This is a performance and activation-lifetime benchmark;
the synthetic representations are intentionally too lossy to provide
model-quality evidence.
'''

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace


HIDDEN = 5376
FFN = 14336
EXPANDED = FFN * 2
DOMAIN = 256
DEFAULT_ROWS = (2048, 4096)
FP8_LAYOUT = 'TensorCoreFP8E4M3Layout'

PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = next(
    (parent for parent in PACK_ROOT.parents if (parent / 'comfy').is_dir()),
    PACK_ROOT.parents[1],
)


def parse_rows(value):
    rows = tuple(
        int(item.strip())
        for item in str(value).split(',')
        if item.strip()
    )
    if not rows or any(item <= 0 or item % 256 for item in rows):
        raise argparse.ArgumentTypeError(
            'rows must be positive multiples of 256'
        )
    if len(rows) != len(set(rows)):
        raise argparse.ArgumentTypeError('rows must not contain duplicates')
    return rows


def validate_h3_shapes(fc1_shape, fc2_shape):
    fc1_shape = tuple(int(value) for value in fc1_shape)
    fc2_shape = tuple(int(value) for value in fc2_shape)
    if fc1_shape != (EXPANDED, HIDDEN):
        raise ValueError(
            'fc1 must have the exact H3 shape (%d, %d), got %s'
            % (EXPANDED, HIDDEN, fc1_shape)
        )
    if fc2_shape != (HIDDEN, FFN):
        raise ValueError(
            'fc2 must have the exact H3 shape (%d, %d), got %s'
            % (HIDDEN, FFN, fc2_shape)
        )
    return HIDDEN, FFN, EXPANDED


def activation_contract(rows, element_size=2):
    rows = int(rows)
    element_size = int(element_size)
    if rows <= 0 or element_size <= 0:
        raise ValueError('rows and element size must be positive')
    return {
        'input': {
            'shape': [rows, HIDDEN],
            'format': 'BF16',
            'bytes': rows * HIDDEN * element_size,
        },
        'full_fc1_expansion': {
            'shape': [rows, EXPANDED],
            'format': 'BF16',
            'bytes': rows * EXPANDED * element_size,
        },
        'swiglu_activation': {
            'shape': [rows, FFN],
            'format': 'BF16',
            'bytes': rows * FFN * element_size,
        },
        'output': {
            'shape': [rows, HIDDEN],
            'format': 'BF16',
            'bytes': rows * HIDDEN * element_size,
        },
        'convrot_two_slice_fc1_target': {
            'shape': [rows, FFN],
            'format': 'BF16',
            'bytes': rows * FFN * element_size,
        },
        'logical_gemms': [
            {
                'name': 'fc1',
                'm': rows,
                'n': EXPANDED,
                'k': HIDDEN,
            },
            {
                'name': 'fc2',
                'm': rows,
                'n': HIDDEN,
                'k': FFN,
            },
        ],
        'convrot_two_slice_gemms': [
            gemm
            for tile in range(2)
            for gemm in (
                {
                    'name': 'fc1_tile_%d' % tile,
                    'm': rows,
                    'n': FFN,
                    'k': HIDDEN,
                },
                {
                    'name': 'fc2_tile_%d' % tile,
                    'm': rows,
                    'n': HIDDEN,
                    'k': FFN // 2,
                },
            )
        ],
    }


def _decode_quant_config(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return json.loads(value)
    raw = value.detach().cpu().numpy().tobytes()
    return json.loads(raw)


def _mlp_prefixes(block_index):
    stem = 'blocks.%d.mlp.' % int(block_index)
    return (
        'model.diffusion_model.' + stem,
        'diffusion_model.' + stem,
        stem,
    )


def load_convrot_mlp(checkpoint, block_index=0, safe_open_fn=None):
    if safe_open_fn is None:
        from safetensors import safe_open as safe_open_fn

    required = (
        'fc1.weight',
        'fc1.weight_scale',
        'fc1.comfy_quant',
        'fc2.weight',
        'fc2.weight_scale',
        'fc2.comfy_quant',
    )
    with safe_open_fn(str(checkpoint), framework='pt', device='cpu') as handle:
        keys = set(handle.keys())
        prefix = next(
            (
                candidate
                for candidate in _mlp_prefixes(block_index)
                if all(candidate + suffix in keys for suffix in required)
            ),
            None,
        )
        if prefix is None:
            raise KeyError(
                'checkpoint has no complete blocks.%d.mlp ConvRot state'
                % int(block_index)
            )
        state = {
            suffix: handle.get_tensor(prefix + suffix)
            for suffix in required
        }

    layers = {}
    for name in ('fc1', 'fc2'):
        weight = state[name + '.weight']
        scale = state[name + '.weight_scale']
        quant = _decode_quant_config(state[name + '.comfy_quant'])
        if (
            quant.get('format') != 'int8_tensorwise'
            or quant.get('convrot') is not True
            or quant.get('transposed', False)
            or int(quant.get('convrot_groupsize', DOMAIN)) != DOMAIN
        ):
            raise ValueError(
                '%s must be non-transposed ConvRot-256 TensorWise INT8'
                % name
            )
        if getattr(weight, 'ndim', None) != 2 or str(weight.dtype) != 'torch.int8':
            raise ValueError('%s weight must be a rank-2 INT8 tensor' % name)
        if scale.numel() not in (1, int(weight.shape[0])):
            raise ValueError(
                '%s scale must be scalar or per-output-channel' % name
            )
        layers[name] = {
            'weight': weight,
            'scale': scale,
            'quant': quant,
        }

    validate_h3_shapes(
        layers['fc1']['weight'].shape,
        layers['fc2']['weight'].shape,
    )
    return {
        'prefix': prefix,
        'block_index': int(block_index),
        'fc1': layers['fc1'],
        'fc2': layers['fc2'],
    }


def resolve_checkpoint(value):
    candidate = Path(value)
    if candidate.is_absolute():
        if candidate.suffix.lower() != '.safetensors' or not candidate.is_file():
            raise FileNotFoundError(str(candidate))
        return candidate.resolve()

    import folder_paths

    resolved = Path(
        folder_paths.get_full_path_or_raise('diffusion_models', value)
    )
    if resolved.suffix.lower() != '.safetensors' or not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return resolved.resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--block-index', type=int, default=0)
    parser.add_argument(
        '--rows',
        type=parse_rows,
        default=DEFAULT_ROWS,
        metavar='N[,N...]',
    )
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if args.block_index < 0 or args.device < 0:
        parser.error('block index and device must be non-negative')
    if args.warmup < 0 or args.iterations <= 0:
        parser.error('warmup must be non-negative and iterations positive')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required GPU preflight'
        )
    return args


def _linear_module(weight):
    return SimpleNamespace(
        weight=weight,
        bias=None,
        weight_function=[],
        bias_function=[],
        _full_precision_mm=False,
    )


def _mlp_module(fc1_weight, fc2_weight):
    return SimpleNamespace(
        fc1=_linear_module(fc1_weight),
        fc2=_linear_module(fc2_weight),
    )


def _convrot_weight(torch, QuantizedTensor, TensorWiseINT8Layout, layer):
    params = TensorWiseINT8Layout.Params(
        scale=layer['scale'].to(dtype=torch.float32),
        orig_dtype=torch.bfloat16,
        orig_shape=tuple(layer['weight'].shape),
        is_weight=True,
        convrot=True,
        convrot_groupsize=DOMAIN,
        transposed=False,
    )
    return QuantizedTensor(
        layer['weight'],
        'TensorWiseINT8Layout',
        params,
    )


def _inventory(inspect_h3_linears, mlp):
    block = SimpleNamespace(
        attn=SimpleNamespace(qkv_proj=mlp.fc1),
        mlp=mlp,
    )
    return inspect_h3_linears((block,))


def _resolution_payload(resolution):
    return {
        'provider_id': resolution.provider_id,
        'activation_mode': resolution.activation_mode,
        'reason': resolution.reason,
    }


def _tensor_storage_bytes(torch, tensors):
    storages = {}

    def add(tensor):
        if not torch.is_tensor(tensor):
            return
        storage = tensor.untyped_storage()
        key = (str(tensor.device), int(storage.data_ptr()))
        storages.setdefault(key, int(storage.nbytes()))

    for tensor in tensors:
        qdata = getattr(tensor, '_qdata', None)
        if torch.is_tensor(qdata):
            add(qdata)
            params = getattr(tensor, '_params', None)
            if params is not None:
                for name in ('scale', 'block_scale'):
                    add(getattr(params, name, None))
        else:
            add(tensor)
    return sum(storages.values())


def _measure_preparation(torch, factory, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = int(torch.cuda.memory_allocated(device))
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    session = factory()
    session.__enter__()
    stop.record()
    stop.synchronize()
    return session, {
        'elapsed_ms': float(start.elapsed_time(stop)),
        'peak_allocated_delta_bytes': int(
            torch.cuda.max_memory_allocated(device) - baseline
        ),
        'live_allocated_delta_bytes': int(
            torch.cuda.memory_allocated(device) - baseline
        ),
    }


def _measure_fp8_synthesis(
    torch,
    QuantizedTensor,
    fc1_weight,
    fc2_weight,
    device,
):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = int(torch.cuda.memory_allocated(device))
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    fp8_fc1 = QuantizedTensor.from_float(
        fc1_weight,
        FP8_LAYOUT,
        scale='recalculate',
    )
    fp8_fc2 = QuantizedTensor.from_float(
        fc2_weight,
        FP8_LAYOUT,
        scale='recalculate',
    )
    stop.record()
    stop.synchronize()
    return (fp8_fc1, fp8_fc2), {
        'elapsed_ms': float(start.elapsed_time(stop)),
        'peak_allocated_delta_bytes': int(
            torch.cuda.max_memory_allocated(device) - baseline
        ),
        'live_allocated_delta_bytes': int(
            torch.cuda.memory_allocated(device) - baseline
        ),
        'included_in_execution_timing': False,
    }


def _measure_bf16_synthesis(torch, source, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = int(torch.cuda.memory_allocated(device))
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    fc1_weight = _dequantize_source(torch, source['fc1'], device)
    fc2_weight = _dequantize_source(torch, source['fc2'], device)
    stop.record()
    stop.synchronize()
    return (fc1_weight, fc2_weight), {
        'elapsed_ms': float(start.elapsed_time(stop)),
        'peak_allocated_delta_bytes': int(
            torch.cuda.max_memory_allocated(device) - baseline
        ),
        'live_allocated_delta_bytes': int(
            torch.cuda.memory_allocated(device) - baseline
        ),
        'included_in_execution_timing': False,
    }


def _measure_cases(torch, cases, warmup, iterations, device):
    names = tuple(cases)
    for iteration in range(warmup):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            output = cases[name]()
            del output
    torch.cuda.synchronize(device)

    samples = {name: [] for name in names}
    peaks = {name: [] for name in names}
    live = {name: [] for name in names}
    for iteration in range(iterations):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline = int(torch.cuda.memory_allocated(device))
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            output = cases[name]()
            stop.record()
            stop.synchronize()
            samples[name].append(float(start.elapsed_time(stop)))
            peaks[name].append(
                int(torch.cuda.max_memory_allocated(device) - baseline)
            )
            live[name].append(
                int(torch.cuda.memory_allocated(device) - baseline)
            )
            del output

    return {
        name: {
            'samples_ms': samples[name],
            'median_ms': statistics.median(samples[name]),
            'min_ms': min(samples[name]),
            'mean_ms': statistics.mean(samples[name]),
            'peak_allocated_delta_bytes': max(peaks[name]),
            'output_live_delta_bytes': max(live[name]),
        }
        for name in names
    }


def _measure_stages(torch, fc1, activate, fc2, x, device):
    events = tuple(
        torch.cuda.Event(enable_timing=True)
        for _ in range(4)
    )
    events[0].record()
    expanded = fc1(x)
    events[1].record()
    activated = activate(expanded)
    events[2].record()
    output = fc2(activated)
    events[3].record()
    events[3].synchronize()
    result = {
        'fc1_ms': float(events[0].elapsed_time(events[1])),
        'swiglu_ms': float(events[1].elapsed_time(events[2])),
        'fc2_ms': float(events[2].elapsed_time(events[3])),
        'single_diagnostic_pass': True,
    }
    del expanded, activated, output
    return result


def _measure_convrot_stages(torch, session, x):
    records = []

    @contextmanager
    def stage(name):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            stop.record()
            records.append((name, start, stop))

    output, path = session.fc1_fc2(x, stage)
    if path != 'held_convrot_2slice':
        raise RuntimeError('unexpected ConvRot path %s' % path)
    torch.cuda.synchronize(output.device)
    totals = {'mlp_fc1': 0.0, 'mlp_swiglu_fc2': 0.0}
    launches = {'mlp_fc1': 0, 'mlp_swiglu_fc2': 0}
    for name, start, stop in records:
        totals[name] += float(start.elapsed_time(stop))
        launches[name] += 1
    del output
    return {
        'fc1_ms': totals['mlp_fc1'],
        'swiglu_fc2_ms': totals['mlp_swiglu_fc2'],
        'fc1_launches': launches['mlp_fc1'],
        'swiglu_fc2_launches': launches['mlp_swiglu_fc2'],
        'swiglu_is_fused_into_fc2': True,
        'single_diagnostic_pass': True,
    }


def _validate_output(torch, fn, rows):
    output = fn()
    torch.cuda.synchronize(output.device)
    if tuple(output.shape) != (int(rows), HIDDEN):
        raise RuntimeError(
            'MLP output shape is %s, expected (%d, %d)'
            % (tuple(output.shape), int(rows), HIDDEN)
        )
    if output.dtype != torch.bfloat16:
        raise RuntimeError('MLP output dtype is %s, expected BF16' % output.dtype)
    finite = bool(torch.isfinite(output).all().item())
    checksum = float(output.reshape(-1)[:64].float().sum().item())
    del output
    if not finite:
        raise RuntimeError('MLP output contains non-finite values')
    return {
        'shape': [int(rows), HIDDEN],
        'dtype': 'torch.bfloat16',
        'finite': True,
        'sample_checksum': checksum,
        'accuracy_comparison': 'intentionally_not_measured',
    }


def _dequantize_source(torch, layer, device):
    qdata = layer['weight'].to(device=device)
    scale = layer['scale'].to(device=device, dtype=torch.float32)
    weight = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
        qdata,
        scale,
        DOMAIN,
        2,
    ).to(dtype=torch.bfloat16)
    del qdata, scale
    return weight


def run(args):
    for path in (str(COMFY_ROOT), str(PACK_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import torch
    import torch.nn.functional as F

    import comfy.model_management
    import comfy.quant_ops
    from comfy.quant_ops import (
        QuantizedTensor,
        TensorWiseINT8Layout,
        get_layout_class,
    )
    from h3_optimizations.memory.linear import (
        ConvRotTwoSliceMLP,
        HeldMLP,
        swiglu_eager,
    )
    from h3_optimizations.qkv.formats import inspect_h3_linears
    from h3_optimizations.qkv.fp8 import HeldFP8MLP
    from h3_optimizations.qkv.providers import (
        MLP_CONVROT_INT8_TWO_SLICE,
        MLP_FLOAT_CHUNKED,
        MLP_FP8_CHUNKED,
        resolve_mlp_provider,
    )

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    device = torch.device('cuda', int(args.device))
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability < (8, 9):
        raise RuntimeError('synthetic FP8 provider benchmark requires SM89 or newer')
    if not bool(getattr(comfy.quant_ops, '_CK_AVAILABLE', False)):
        raise RuntimeError('Comfy Kitchen is required')
    if get_layout_class(FP8_LAYOUT) is None:
        raise RuntimeError('%s is unavailable' % FP8_LAYOUT)
    if not comfy.model_management.supports_fp8_compute(device):
        raise RuntimeError('Comfy reports accelerated FP8 execution unavailable')

    checkpoint = resolve_checkpoint(args.checkpoint)
    source = load_convrot_mlp(checkpoint, args.block_index)
    convrot_mlp = _mlp_module(
        _convrot_weight(
            torch,
            QuantizedTensor,
            TensorWiseINT8Layout,
            source['fc1'],
        ),
        _convrot_weight(
            torch,
            QuantizedTensor,
            TensorWiseINT8Layout,
            source['fc2'],
        ),
    )
    (fc1_weight, fc2_weight), bf16_synthesis = _measure_bf16_synthesis(
        torch,
        source,
        device,
    )
    bf16_mlp = _mlp_module(fc1_weight, fc2_weight)

    (fp8_fc1, fp8_fc2), fp8_synthesis = _measure_fp8_synthesis(
        torch,
        QuantizedTensor,
        fc1_weight,
        fc2_weight,
        device,
    )
    native_fp8_mlp = _mlp_module(fp8_fc1, fp8_fc2)

    convrot_inventory = _inventory(inspect_h3_linears, convrot_mlp)
    bf16_inventory = _inventory(inspect_h3_linears, bf16_mlp)
    native_fp8_inventory = _inventory(inspect_h3_linears, native_fp8_mlp)
    convrot_resolution = resolve_mlp_provider(
        convrot_inventory,
        request='auto',
        fp8_available=True,
    )
    bf16_float_resolution = resolve_mlp_provider(
        bf16_inventory,
        request='auto',
        fp8_available=False,
    )
    bf16_auto_resolution = resolve_mlp_provider(
        bf16_inventory,
        request='auto',
        fp8_available=True,
    )
    native_fp8_resolution = resolve_mlp_provider(
        native_fp8_inventory,
        request='auto',
        fp8_available=True,
    )
    if convrot_resolution.provider_id != MLP_CONVROT_INT8_TWO_SLICE:
        raise RuntimeError('ConvRot auto did not resolve convrot_int8_two_slice')
    if bf16_float_resolution.provider_id != MLP_FLOAT_CHUNKED:
        raise RuntimeError('BF16 float control did not resolve float_chunked')
    if bf16_auto_resolution.provider_id != MLP_FP8_CHUNKED:
        raise RuntimeError('BF16 auto did not resolve fp8_chunked')
    if native_fp8_resolution.provider_id != MLP_FP8_CHUNKED:
        raise RuntimeError('native FP8 auto did not resolve fp8_chunked')

    convrot_session = None
    float_session = None
    bf16_fp8_session = None
    native_fp8_session = None
    try:
        convrot_session, convrot_preparation = _measure_preparation(
            torch,
            lambda: ConvRotTwoSliceMLP(
                convrot_mlp,
                fc1_weight.new_empty((1, HIDDEN)),
            ),
            device,
        )
        float_session, float_preparation = _measure_preparation(
            torch,
            lambda: HeldMLP(bf16_mlp, fc1_weight.new_empty((1, HIDDEN))),
            device,
        )
        bf16_fp8_session, bf16_fp8_preparation = _measure_preparation(
            torch,
            lambda: HeldFP8MLP(
                bf16_mlp,
                fc1_weight.new_empty((1, HIDDEN)),
                allow_float_conversion=True,
            ),
            device,
        )
        native_fp8_session, native_fp8_preparation = _measure_preparation(
            torch,
            lambda: HeldFP8MLP(
                native_fp8_mlp,
                fc1_weight.new_empty((1, HIDDEN)),
                allow_float_conversion=False,
            ),
            device,
        )

        if not (
            bf16_fp8_session.fc1_binding.converted_from_float
            and bf16_fp8_session.fc2_binding.converted_from_float
        ):
            raise RuntimeError('BF16 auto did not convert both weights to FP8')
        if (
            native_fp8_session.fc1_binding.converted_from_float
            or native_fp8_session.fc2_binding.converted_from_float
        ):
            raise RuntimeError('native FP8 auto unexpectedly converted its weights')

        generator = torch.Generator(device=device).manual_seed(args.seed)
        row_results = []
        for rows in args.rows:
            x = torch.randn(
                (int(rows), HIDDEN),
                generator=generator,
                dtype=torch.bfloat16,
                device=device,
            )

            def bf16_upstream():
                expanded = F.linear(x, fc1_weight)
                activated = swiglu_eager(expanded)
                return F.linear(activated, fc2_weight)

            def convrot_two_slice():
                output, path = convrot_session.fc1_fc2(x)
                if path != 'held_convrot_2slice':
                    raise RuntimeError('unexpected ConvRot path %s' % path)
                return output

            def bf16_float_chunked():
                expanded = float_session.fc1(x)
                output, path = float_session.fc2_swiglu(
                    expanded,
                    native=True,
                )
                if path != 'held_bf16_swiglu':
                    raise RuntimeError('unexpected BF16 held path %s' % path)
                return output

            def bf16_auto_fp8():
                output, path = bf16_fp8_session.fc1_fc2(x, swiglu_eager)
                if path != 'held_fp8':
                    raise RuntimeError('unexpected BF16-to-FP8 path %s' % path)
                return output

            def native_fp8_auto():
                output, path = native_fp8_session.fc1_fc2(x, swiglu_eager)
                if path != 'held_fp8':
                    raise RuntimeError('unexpected native FP8 path %s' % path)
                return output

            cases = {
                'convrot_two_slice_production': convrot_two_slice,
                'bf16_upstream': bf16_upstream,
                'bf16_float_chunked_control': bf16_float_chunked,
                'bf16_auto_fp8': bf16_auto_fp8,
                'native_fp8_auto': native_fp8_auto,
            }
            validation = {
                name: _validate_output(torch, fn, rows)
                for name, fn in cases.items()
            }
            timing = _measure_cases(
                torch,
                cases,
                args.warmup,
                args.iterations,
                device,
            )
            stages = {
                'convrot_two_slice_production': _measure_convrot_stages(
                    torch,
                    convrot_session,
                    x,
                ),
                'bf16_upstream': _measure_stages(
                    torch,
                    lambda value: F.linear(value, fc1_weight),
                    swiglu_eager,
                    lambda value: F.linear(value, fc2_weight),
                    x,
                    device,
                ),
                'bf16_float_chunked_control': _measure_stages(
                    torch,
                    float_session.fc1,
                    swiglu_eager,
                    float_session.fc2_weight.linear,
                    x,
                    device,
                ),
                'bf16_auto_fp8': _measure_stages(
                    torch,
                    bf16_fp8_session.fc1_binding.linear,
                    swiglu_eager,
                    bf16_fp8_session.fc2_binding.linear,
                    x,
                    device,
                ),
                'native_fp8_auto': _measure_stages(
                    torch,
                    native_fp8_session.fc1_binding.linear,
                    swiglu_eager,
                    native_fp8_session.fc2_binding.linear,
                    x,
                    device,
                ),
            }
            baseline_ms = timing['bf16_upstream']['median_ms']
            bf16_comparisons = {
                name: {
                    'latency_ratio': timing[name]['median_ms'] / baseline_ms,
                    'speedup': baseline_ms / timing[name]['median_ms'],
                    'peak_allocated_delta_bytes_difference': (
                        timing[name]['peak_allocated_delta_bytes']
                        - timing['bf16_upstream']['peak_allocated_delta_bytes']
                    ),
                }
                for name in cases
                if name != 'bf16_upstream'
            }
            convrot_ms = timing['convrot_two_slice_production']['median_ms']
            convrot_comparisons = {
                name: {
                    'latency_ratio': timing[name]['median_ms'] / convrot_ms,
                    'speedup': convrot_ms / timing[name]['median_ms'],
                    'peak_allocated_delta_bytes_difference': (
                        timing[name]['peak_allocated_delta_bytes']
                        - timing['convrot_two_slice_production'][
                            'peak_allocated_delta_bytes'
                        ]
                    ),
                }
                for name in cases
                if name != 'convrot_two_slice_production'
            }
            row_results.append(
                {
                    'rows': int(rows),
                    'activation_contract': activation_contract(rows),
                    'comparison_to_bf16_upstream': bf16_comparisons,
                    'comparison_to_convrot_two_slice': convrot_comparisons,
                    'cases': {
                        name: {
                            **timing[name],
                            'stage_timing': stages[name],
                            'validation': validation[name],
                            'logical_gemm_launches': (
                                4
                                if name == 'convrot_two_slice_production'
                                else 2
                            ),
                            'gemm_shape_contract': (
                                'convrot_two_slice_gemms'
                                if name == 'convrot_two_slice_production'
                                else 'logical_gemms'
                            ),
                        }
                        for name in cases
                    },
                }
            )
            del x

        result = {
            'benchmark': 'h3_synthetic_nonconvrot_mlp_provider_parity',
            'scope': 'benchmark-only; shipping implementation unchanged',
            'evidence_boundary': (
                'ConvRot weights are dequantized to BF16 and requantized to '
                'FP8 solely to exercise provider performance and activation '
                'lifetime. Results are not checkpoint-quality evidence.'
            ),
            'checkpoint': str(checkpoint),
            'checkpoint_prefix': source['prefix'],
            'block_index': int(args.block_index),
            'shape': {
                'hidden': HIDDEN,
                'ffn': FFN,
                'fc1_output': EXPANDED,
            },
            'device': {
                'index': int(args.device),
                'name': torch.cuda.get_device_name(device),
                'capability': list(capability),
            },
            'torch': torch.__version__,
            'warmup': int(args.warmup),
            'iterations': int(args.iterations),
            'stage_timing_semantics': {
                'convrot_two_slice_production': (
                    'exact production pass; SwiGLU is fused into each fc2 launch'
                ),
                'bf16_paths': 'fc1, eager SwiGLU, and fc2 are timed separately',
                'fp8_paths': (
                    'fc1 includes input quantization; fc2 includes activation '
                    'quantization'
                ),
            },
            'representations': {
                'convrot_int8': {
                    'source': 'checkpoint-native ConvRot-256 TensorWise INT8',
                    'weight_storage_bytes': _tensor_storage_bytes(
                        torch,
                        (convrot_mlp.fc1.weight, convrot_mlp.fc2.weight),
                    ),
                    'prepacked_tile_storage_bytes': _tensor_storage_bytes(
                        torch,
                        tuple(
                            value
                            for tile in convrot_session.tiles
                            for value in tile.values()
                        ),
                    ),
                },
                'bf16': {
                    'source': 'dequantized ConvRot-256 block',
                    'weight_storage_bytes': _tensor_storage_bytes(
                        torch,
                        (fc1_weight, fc2_weight),
                    ),
                    'synthesis': bf16_synthesis,
                },
                'native_fp8': {
                    'source': 'BF16 reconstruction requantized to E4M3',
                    'layout': FP8_LAYOUT,
                    'weight_storage_bytes': _tensor_storage_bytes(
                        torch,
                        (fp8_fc1, fp8_fc2),
                    ),
                    'synthesis': fp8_synthesis,
                },
            },
            'provider_resolution': {
                'convrot_two_slice_production': _resolution_payload(
                    convrot_resolution
                ),
                'bf16_float_chunked_control': _resolution_payload(
                    bf16_float_resolution
                ),
                'bf16_auto_fp8': _resolution_payload(
                    bf16_auto_resolution
                ),
                'native_fp8_auto': _resolution_payload(
                    native_fp8_resolution
                ),
            },
            'provider_preparation': {
                'convrot_two_slice_production': convrot_preparation,
                'bf16_float_chunked_control': float_preparation,
                'bf16_auto_fp8': bf16_fp8_preparation,
                'native_fp8_auto': native_fp8_preparation,
            },
            'results': row_results,
        }
    finally:
        if native_fp8_session is not None:
            native_fp8_session.__exit__(None, None, None)
        if bf16_fp8_session is not None:
            bf16_fp8_session.__exit__(None, None, None)
        if float_session is not None:
            float_session.__exit__(None, None, None)
        if convrot_session is not None:
            convrot_session.__exit__(None, None, None)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
    if args.json or args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
