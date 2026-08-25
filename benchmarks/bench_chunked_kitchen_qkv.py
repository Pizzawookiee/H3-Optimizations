'''Benchmark full and sequence-chunked Kitchen QKV preparation for H3.'''

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace


HEAD_DIM = 128
QK_TILE = 128
DEFAULT_SEQUENCE = 54006
DEFAULT_CHUNKS = '512,768,1024,1536,2048,3072,4096'


def parse_chunk_sizes(value):
    chunks = []
    for item in str(value).split(','):
        item = item.strip()
        if not item:
            continue
        chunk = int(item)
        if chunk <= 0 or chunk % QK_TILE:
            raise ValueError('chunk sizes must be positive multiples of 128')
        if chunk in chunks:
            raise ValueError('chunk sizes must not contain duplicates')
        chunks.append(chunk)
    if not chunks:
        raise ValueError('at least one chunk size is required')
    return tuple(chunks)


def chunk_ranges(sequence, chunk_size):
    if sequence <= 0:
        raise ValueError('sequence must be positive')
    if chunk_size <= 0:
        raise ValueError('chunk size must be positive')
    return tuple(
        (start, min(sequence, start + chunk_size))
        for start in range(0, sequence, chunk_size)
    )


def resolve_checkpoint(value):
    candidate = Path(value)
    if candidate.is_absolute():
        if candidate.suffix.lower() != '.safetensors' or not candidate.is_file():
            raise FileNotFoundError(str(candidate))
        return candidate.resolve()

    import folder_paths

    resolved = Path(folder_paths.get_full_path_or_raise('diffusion_models', value))
    if resolved.suffix.lower() != '.safetensors' or not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return resolved.resolve()


def _prefixes(block_index):
    stem = 'blocks.%d.attn.' % int(block_index)
    return ('model.diffusion_model.' + stem, 'diffusion_model.' + stem, stem)


def load_attention_tensors(checkpoint, block_index):
    from safetensors import safe_open

    required = ('qkv_proj.weight', 'q_norm.weight', 'k_norm.weight')
    with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
        keys = set(handle.keys())
        prefix = next(
            (
                candidate
                for candidate in _prefixes(block_index)
                if all(candidate + suffix in keys for suffix in required)
            ),
            None,
        )
        if prefix is None:
            raise KeyError(
                'checkpoint has no complete blocks.%d.attn QKV state' % block_index
            )
        state = {
            key[len(prefix):]: handle.get_tensor(key)
            for key in keys
            if key.startswith(prefix)
            and key[len(prefix):].startswith(('qkv_proj.', 'q_norm.', 'k_norm.'))
        }
    return prefix, state


def build_attention(torch, checkpoint, block_index, epsilon, device):
    import comfy.ops

    prefix, state = load_attention_tensors(checkpoint, block_index)
    weight = state['qkv_proj.weight']
    if weight.ndim != 2 or int(weight.shape[0]) % (3 * HEAD_DIM):
        raise ValueError('QKV storage shape is not H3-compatible: %s' % (tuple(weight.shape),))
    hidden = int(weight.shape[1])
    heads = int(weight.shape[0]) // (3 * HEAD_DIM)
    ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    qkv_proj = ops.Linear(hidden, heads * HEAD_DIM * 3, bias=False)
    qkv_state = {
        key[len('qkv_proj.'):]: value
        for key, value in state.items()
        if key.startswith('qkv_proj.')
    }
    qkv_proj.load_state_dict(qkv_state, strict=True)
    qkv_proj.to(device)

    module = SimpleNamespace(
        qkv_proj=qkv_proj,
        q_norm=SimpleNamespace(
            weight=state['q_norm.weight'].to(device=device, dtype=torch.bfloat16),
            eps=float(epsilon),
        ),
        k_norm=SimpleNamespace(
            weight=state['k_norm.weight'].to(device=device, dtype=torch.bfloat16),
            eps=float(epsilon),
        ),
        heads=heads,
        head_dim=HEAD_DIM,
    )
    return module, hidden, prefix


def validate_mlp_shapes(fc1_shape, fc2_shape, hidden):
    fc1_shape = tuple(int(value) for value in fc1_shape)
    fc2_shape = tuple(int(value) for value in fc2_shape)
    hidden = int(hidden)
    if len(fc1_shape) != 2 or len(fc2_shape) != 2:
        raise ValueError('MLP weights must be matrices')
    if fc1_shape[1] != hidden or fc1_shape[0] % 2:
        raise ValueError('fc1 shape is incompatible with the H3 hidden width')
    ffn = fc1_shape[0] // 2
    if fc2_shape != (hidden, ffn):
        raise ValueError('fc1/fc2 shapes are not an H3 SwiGLU pair')
    return ffn


def load_mlp_contract(checkpoint, block_index, hidden):
    from safetensors import safe_open

    suffixes = ('fc1.weight', 'fc2.weight')
    prefixes = tuple(
        prefix[:-len('attn.')] + 'mlp.'
        for prefix in _prefixes(block_index)
    )
    with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
        keys = set(handle.keys())
        prefix = next(
            (
                candidate
                for candidate in prefixes
                if all(candidate + suffix in keys for suffix in suffixes)
            ),
            None,
        )
        if prefix is None:
            raise KeyError(
                'checkpoint has no complete blocks.%d.mlp state' % block_index
            )
        shapes = {
            suffix: tuple(handle.get_slice(prefix + suffix).get_shape())
            for suffix in suffixes
        }
    ffn = validate_mlp_shapes(
        shapes['fc1.weight'],
        shapes['fc2.weight'],
        hidden,
    )
    return {
        'prefix': prefix,
        'fc1_shape': list(shapes['fc1.weight']),
        'fc2_shape': list(shapes['fc2.weight']),
        'ffn': ffn,
    }


def two_slice_visible_peak_bytes(rows, hidden, ffn, element_size=2):
    rows = int(rows)
    hidden = int(hidden)
    ffn = int(ffn)
    element_size = int(element_size)
    if min(rows, hidden, ffn, element_size) <= 0:
        raise ValueError('two-slice MLP dimensions must be positive')
    return rows * (ffn + 3 * hidden) * element_size


def make_rope(torch, sequence, device):
    angles = torch.arange(
        sequence * 48,
        device=device,
        dtype=torch.float32,
    ).reshape(sequence, 48)
    angles = angles * (1.0 / 8192.0)
    c = torch.cos(angles)
    s = torch.sin(angles)
    return torch.stack((c, -s, s, c), dim=-1).reshape(
        1, sequence, 1, 48, 2, 2
    ).to(torch.bfloat16)


def project_qkv(torch, module, x, rope):
    import comfy.quant_ops

    rows = int(x.shape[0])
    inner = int(module.heads) * int(module.head_dim)
    q, k, v = module.qkv_proj(x).split(inner, dim=-1)
    q = q.view(1, rows, module.heads, module.head_dim)
    k = k.view(1, rows, module.heads, module.head_dim)
    comfy.quant_ops.ck.rms_rope_split_half_(
        q,
        k,
        rope,
        module.q_norm.weight,
        module.k_norm.weight,
        epsilon=module.q_norm.eps,
        rot_dim=int(rope.shape[-3]) * 2,
    )
    return q[0], k[0], v.view(rows, module.heads, module.head_dim)


def projection_only(module, x):
    return module.qkv_proj(x)


def chunked_projection_only(module, x, chunk_size):
    result = None
    for start, stop in chunk_ranges(int(x.shape[0]), int(chunk_size)):
        result = module.qkv_proj(x[start:stop])
    return result


def projection_rope_preparation(torch, module, x, rope):
    return project_qkv(torch, module, x, rope)


def chunked_projection_rope_preparation(torch, module, x, rope, chunk_size):
    result = None
    for start, stop in chunk_ranges(int(x.shape[0]), int(chunk_size)):
        result = project_qkv(
            torch,
            module,
            x[start:stop],
            rope[:, start:stop],
        )
    return result


def full_kitchen_preparation(torch, module, x, rope):
    q, k, v = project_qkv(torch, module, x, rope)
    retained_v = v.clone()
    checksum = q[0, 0, 0].float() + k[-1, -1, -1].float()
    return retained_v, checksum


def chunked_kitchen_preparation(torch, module, x, rope, chunk_size):
    retained_v = torch.empty(
        (x.shape[0], module.heads, module.head_dim),
        dtype=x.dtype,
        device=x.device,
    )
    samples = []
    for start, stop in chunk_ranges(int(x.shape[0]), int(chunk_size)):
        q, k, v = project_qkv(
            torch,
            module,
            x[start:stop],
            rope[:, start:stop],
        )
        retained_v[start:stop].copy_(v)
        samples.extend((q[0, 0, 0].float(), k[-1, -1, -1].float()))
    return retained_v, torch.stack(samples).sum()


def _to_hnd(q, k, v):
    return (
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
    )


def full_kitchen_carrier(torch, module, x, rope):
    import comfy.quant_ops

    q, k, v = _to_hnd(*project_qkv(torch, module, x, rope))
    carrier = comfy.quant_ops.ck.prequantize_int8_attention(q, k, v)
    del q, k, v
    return carrier


def chunked_kitchen_carrier(torch, module, x, rope, chunk_size, block_index):
    import comfy.quant_ops
    from h3_optimizations.kitchen_qkv import run_chunked_kitchen_qkv

    shape = (1, int(module.heads), int(x.shape[0]), int(module.head_dim))
    spec = comfy.quant_ops.ck.int8_attention_producer_spec(
        shape,
        shape,
        dtype=x.dtype,
        device=x.device,
    )
    prepared = run_chunked_kitchen_qkv(
        module,
        x,
        rope,
        layer_index=block_index,
        transformer_options={},
        spec=spec,
        chunk_rows=chunk_size,
    )
    return prepared.carrier


def full_kitchen_attention(torch, module, x, rope):
    import comfy.quant_ops

    carrier = full_kitchen_carrier(torch, module, x, rope)
    try:
        return comfy.quant_ops.ck.int8_attention_from_prequantized(carrier)
    finally:
        del carrier


def chunked_kitchen_attention(torch, module, x, rope, chunk_size, block_index):
    import comfy.quant_ops

    carrier = chunked_kitchen_carrier(
        torch,
        module,
        x,
        rope,
        chunk_size,
        block_index,
    )
    try:
        return comfy.quant_ops.ck.int8_attention_from_prequantized(carrier)
    finally:
        del carrier


def current_fused_preparation(module, x, rope, block_index):
    from h3_optimizations.dense_fused_qkv import run_dense_fused_qkv

    return run_dense_fused_qkv(
        module,
        x,
        rope,
        layer_index=block_index,
    )


def build_legacy_fused_backend():
    import h3_optimizations.attention.v_snapshot_compat  # noqa: F401
    from h3_optimizations.attention.sage_mem_eff import (
        SM89SageMemoryEfficientBackend,
    )
    from h3_optimizations.dense_backend import ProjectedSM89SageBackend

    return ProjectedSM89SageBackend(SM89SageMemoryEfficientBackend())


def current_fused_attention(module, x, rope, block_index, backend):
    projected = current_fused_preparation(module, x, rope, block_index)
    prepared = backend.prepare_projected(
        projected,
        layer_index=block_index,
        transformer_options={},
    )
    del projected
    try:
        return backend.execute(prepared)
    finally:
        del prepared


def convrot_quantization_only(torch, module, x):
    from comfy_kitchen.backends import cuda

    weight = module.qkv_proj.weight
    x_qdata = torch.empty_like(x, dtype=torch.int8)
    x_scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
    cuda._C.quantize_int8_rowwise_convrot64(
        cuda._wrap_for_dlpack(x),
        cuda._wrap_for_dlpack(x_qdata),
        cuda._wrap_for_dlpack(x_scale),
        int(weight._params.convrot_groupsize),
        False,
        0,
        0,
        stream_ptr,
    )
    return x_qdata, x_scale


def chunked_convrot_quantization_only(torch, module, x, chunk_size):
    result = None
    for start, stop in chunk_ranges(int(x.shape[0]), int(chunk_size)):
        result = convrot_quantization_only(torch, module, x[start:stop])
    return result


def forced_cutlass_projection(torch, module, x, config):
    """Run the private Kitchen config hook for benchmark diagnosis only."""
    from comfy_kitchen.backends import cuda

    extension = cuda._C
    runner = getattr(extension, 'cutlass_int8_dequant_config', None)
    if runner is None:
        raise RuntimeError('Kitchen extension has no forced-config binding')
    weight = module.qkv_proj.weight
    x_qdata, x_scale = convrot_quantization_only(torch, module, x)
    weight_qdata = weight._qdata.contiguous()
    weight_scale = weight._params.scale.to(torch.float32).contiguous()
    output = torch.empty(
        (x.shape[0], weight_qdata.shape[0]),
        dtype=torch.bfloat16,
        device=x.device,
    )
    used = runner(
        cuda._wrap_for_dlpack(x_qdata),
        cuda._wrap_for_dlpack(weight_qdata),
        cuda._wrap_for_dlpack(x_scale),
        cuda._wrap_for_dlpack(weight_scale),
        cuda._wrap_for_dlpack(output),
        cuda.DTYPE_TO_CODE[torch.bfloat16],
        int(config),
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if not used:
        raise RuntimeError('Kitchen declined CUTLASS config %d' % int(config))
    return output


def benchmark_cutlass_configs(torch, module, x, row_counts, configs, iterations):
    """Benchmark Kitchen's private diagnostic binding without using it in runtime code."""
    from comfy_kitchen.backends import cuda

    extension = cuda._C
    benchmark = getattr(extension, 'benchmark_cutlass_int8_dequant_config', None)
    if benchmark is None:
        return {
            'available': False,
            'reason': 'Kitchen extension has no config benchmark binding',
        }

    weight = module.qkv_proj.weight
    weight_qdata = weight._qdata.contiguous()
    weight_scale = weight._params.scale.to(torch.float32).contiguous()
    output_dtype_code = cuda.DTYPE_TO_CODE[torch.bfloat16]
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
    results = {}

    for rows in row_counts:
        if rows <= 0 or rows > int(x.shape[0]):
            raise ValueError('CUTLASS row count is outside the input sequence')
        x_rows = x[:rows].contiguous()
        x_qdata = torch.empty(
            (rows, x.shape[-1]),
            dtype=torch.int8,
            device=x.device,
        )
        x_scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
        output = torch.empty(
            (rows, weight_qdata.shape[0]),
            dtype=torch.bfloat16,
            device=x.device,
        )
        extension.quantize_int8_rowwise_convrot64(
            cuda._wrap_for_dlpack(x_rows),
            cuda._wrap_for_dlpack(x_qdata),
            cuda._wrap_for_dlpack(x_scale),
            int(weight._params.convrot_groupsize),
            False,
            0,
            0,
            stream_ptr,
        )
        torch.cuda.synchronize(x.device)
        per_config = {}
        for config in configs:
            total_ms = float(benchmark(
                cuda._wrap_for_dlpack(x_qdata),
                cuda._wrap_for_dlpack(weight_qdata),
                cuda._wrap_for_dlpack(x_scale),
                cuda._wrap_for_dlpack(weight_scale),
                cuda._wrap_for_dlpack(output),
                output_dtype_code,
                int(config),
                int(iterations),
                stream_ptr,
            ))
            per_config[str(config)] = {
                'total_ms': total_ms,
                'per_iteration_ms': total_ms / int(iterations),
            }
        results[str(rows)] = per_config
        del x_rows, x_qdata, x_scale, output
        torch.cuda.synchronize(x.device)

    return {
        'available': True,
        'boundary': 'fused CUTLASS INT8 GEMM/dequant only; ConvRot quantization excluded',
        'iterations': int(iterations),
        'rows': results,
    }


def tensor_bytes(tensor):
    return int(tensor.numel()) * int(tensor.element_size())


def kitchen_carrier_bytes(carrier):
    return sum(
        tensor_bytes(value)
        for name in ('q', 'k', 'v', 'q_scale', 'k_scale', 'v_scale', 'attn_mask')
        if (value := getattr(carrier, name, None)) is not None
    )


class PhasePeakRecorder:
    def __init__(self, torch, device, model_baseline_allocated):
        self.torch = torch
        self.device = device
        self.model_baseline_allocated = int(model_baseline_allocated)
        self.case_baseline_allocated = int(torch.cuda.memory_allocated(device))
        self.case_baseline_reserved = int(torch.cuda.memory_reserved(device))
        self.phases = {}
        self._phase = None

    def begin(self, name):
        if self._phase is not None:
            raise RuntimeError('memory phases must not overlap')
        self.torch.cuda.synchronize(self.device)
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self._phase = (
            str(name),
            int(self.torch.cuda.memory_allocated(self.device)),
            int(self.torch.cuda.memory_reserved(self.device)),
        )

    def end(self):
        if self._phase is None:
            raise RuntimeError('no memory phase is active')
        self.torch.cuda.synchronize(self.device)
        name, start_allocated, start_reserved = self._phase
        peak_allocated = int(self.torch.cuda.max_memory_allocated(self.device))
        peak_reserved = int(self.torch.cuda.max_memory_reserved(self.device))
        self.phases[name] = {
            'start_allocated_bytes': start_allocated,
            'end_allocated_bytes': int(
                self.torch.cuda.memory_allocated(self.device)
            ),
            'peak_allocated_bytes': peak_allocated,
            'peak_above_case_baseline_bytes': (
                peak_allocated - self.case_baseline_allocated
            ),
            'peak_above_model_baseline_bytes': (
                peak_allocated - self.model_baseline_allocated
            ),
            'start_reserved_bytes': start_reserved,
            'peak_reserved_bytes': peak_reserved,
        }
        self._phase = None

    def result(self):
        if self._phase is not None:
            raise RuntimeError('cannot finish with an active memory phase')
        peak_allocated = max(
            phase['peak_allocated_bytes'] for phase in self.phases.values()
        )
        peak_reserved = max(
            phase['peak_reserved_bytes'] for phase in self.phases.values()
        )
        return {
            'model_baseline_allocated_bytes': self.model_baseline_allocated,
            'case_baseline_allocated_bytes': self.case_baseline_allocated,
            'case_baseline_reserved_bytes': self.case_baseline_reserved,
            'peak_allocated_bytes': peak_allocated,
            'peak_reserved_bytes': peak_reserved,
            'activation_peak_bytes': (
                peak_allocated - self.model_baseline_allocated
            ),
            'incremental_peak_bytes': (
                peak_allocated - self.case_baseline_allocated
            ),
            'end_allocated_bytes': int(
                self.torch.cuda.memory_allocated(self.device)
            ),
            'phases': self.phases,
        }


def _noop_convrot_two_slice_mlp(torch, x, hidden, ffn, chunk_rows):
    for start, stop in chunk_ranges(int(x.shape[0]), int(chunk_rows)):
        rows = stop - start
        h = torch.empty(
            (rows, hidden), dtype=x.dtype, device=x.device
        )
        output = None
        for _tile in range(2):
            expanded = torch.empty(
                (rows, ffn), dtype=x.dtype, device=x.device
            )
            partial = torch.empty(
                (rows, hidden), dtype=x.dtype, device=x.device
            )
            if output is None:
                output = partial
            del expanded, partial
        del h, output


def synthetic_block_lifetime(
    torch,
    x,
    carrier_factory,
    *,
    heads,
    head_dim,
    hidden,
    ffn,
    mlp_chunk_rows,
    model_baseline_allocated,
    modulation_rows=3,
):
    recorder = PhasePeakRecorder(torch, x.device, model_baseline_allocated)

    recorder.begin('adaln_and_attention_norm')
    modulation = torch.empty(
        (6, int(modulation_rows), int(hidden)),
        dtype=x.dtype,
        device=x.device,
    )
    h = x.clone()
    recorder.end()

    recorder.begin('qkv_and_kitchen_carrier')
    carrier = carrier_factory(h)
    carrier_live_bytes = kitchen_carrier_bytes(carrier)
    recorder.end()

    recorder.begin('noop_attention_output')
    out_hnd = torch.empty(
        (1, int(heads), int(x.shape[0]), int(head_dim)),
        dtype=x.dtype,
        device=x.device,
    )
    del carrier
    recorder.end()

    recorder.begin('noop_attention_out_projection')
    attention_out = torch.empty(
        (int(x.shape[0]), int(hidden)),
        dtype=x.dtype,
        device=x.device,
    )
    del out_hnd
    recorder.end()

    recorder.begin('attention_residual_release')
    del h, attention_out
    recorder.end()

    recorder.begin('noop_convrot_two_slice_mlp')
    _noop_convrot_two_slice_mlp(
        torch,
        x,
        int(hidden),
        int(ffn),
        int(mlp_chunk_rows),
    )
    recorder.end()

    del modulation
    result = recorder.result()
    result['carrier_live_bytes'] = carrier_live_bytes
    result['mlp_visible_peak_bytes'] = two_slice_visible_peak_bytes(
        min(int(x.shape[0]), int(mlp_chunk_rows)),
        hidden,
        ffn,
        element_size=x.element_size(),
    )
    return result


def benchmark_synthetic_block(
    torch,
    fn,
    warmup,
    iterations,
    device,
):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    samples = []
    for _ in range(iterations):
        torch.cuda.empty_cache()
        samples.append(fn())
        torch.cuda.synchronize(device)

    phase_names = tuple(samples[0]['phases'])
    return {
        'iterations': int(iterations),
        'peak_allocated_bytes': max(
            sample['peak_allocated_bytes'] for sample in samples
        ),
        'peak_reserved_bytes': max(
            sample['peak_reserved_bytes'] for sample in samples
        ),
        'activation_peak_bytes': max(
            sample['activation_peak_bytes'] for sample in samples
        ),
        'incremental_peak_bytes': max(
            sample['incremental_peak_bytes'] for sample in samples
        ),
        'carrier_live_bytes': max(
            sample['carrier_live_bytes'] for sample in samples
        ),
        'mlp_visible_peak_bytes': max(
            sample['mlp_visible_peak_bytes'] for sample in samples
        ),
        'end_allocated_bytes': max(
            sample['end_allocated_bytes'] for sample in samples
        ),
        'phases': {
            name: {
                key: max(sample['phases'][name][key] for sample in samples)
                for key in samples[0]['phases'][name]
            }
            for name in phase_names
        },
    }


def benchmark_case(torch, fn, warmup, iterations, device):
    for _ in range(warmup):
        result = fn()
        torch.cuda.synchronize(device)
        del result
    torch.cuda.empty_cache()

    samples = []
    peaks = []
    output_live = []
    for _ in range(iterations):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
        peaks.append(int(torch.cuda.max_memory_allocated(device) - before))
        output_live.append(int(torch.cuda.memory_allocated(device) - before))
        del result
        torch.cuda.synchronize(device)
    return {
        'median_ms': statistics.median(samples),
        'min_ms': min(samples),
        'samples_ms': samples,
        'peak_allocated_bytes': max(peaks),
        'output_live_bytes': max(output_live),
    }


def error_metrics(torch, module, x, rope, chunk_size):
    reference_q, reference_k, reference_v = project_qkv(torch, module, x, rope)
    squared = {'q': 0.0, 'k': 0.0, 'v': 0.0}
    reference_squared = {'q': 0.0, 'k': 0.0, 'v': 0.0}
    max_abs = {'q': 0.0, 'k': 0.0, 'v': 0.0}
    exact = {'q': True, 'k': True, 'v': True}
    count = {'q': 0, 'k': 0, 'v': 0}
    for start, stop in chunk_ranges(int(x.shape[0]), int(chunk_size)):
        actual = project_qkv(torch, module, x[start:stop], rope[:, start:stop])
        references = (
            reference_q[start:stop],
            reference_k[start:stop],
            reference_v[start:stop],
        )
        for name, value, reference in zip(('q', 'k', 'v'), actual, references):
            delta = value.float() - reference.float()
            max_abs[name] = max(max_abs[name], float(delta.abs().max().item()))
            squared[name] += float(delta.square().sum().item())
            reference_squared[name] += float(reference.float().square().sum().item())
            exact[name] = exact[name] and bool(torch.equal(value, reference))
            count[name] += int(value.numel())
        del actual
    del reference_q, reference_k, reference_v
    return {
        name: {
            'exact': exact[name],
            'max_abs': max_abs[name],
            'rmse': (squared[name] / count[name]) ** 0.5,
            'relative_rmse': (
                squared[name] / max(reference_squared[name], 1e-20)
            ) ** 0.5,
        }
        for name in ('q', 'k', 'v')
    }


def carrier_parity(torch, module, x, rope, chunk_size, block_index):
    reference = full_kitchen_carrier(torch, module, x, rope)
    actual = chunked_kitchen_carrier(
        torch,
        module,
        x,
        rope,
        chunk_size,
        block_index,
    )
    fields = ('q', 'k', 'v', 'q_scale', 'k_scale', 'v_scale')
    result = {
        name: {
            'exact': bool(torch.equal(getattr(actual, name), getattr(reference, name))),
            'shape': list(getattr(actual, name).shape),
            'dtype': str(getattr(actual, name).dtype),
        }
        for name in fields
    }
    result['metadata_exact'] = all(
        getattr(actual, name) == getattr(reference, name)
        for name in (
            'original_head_dim',
            'input_dtype',
            'attention_scale',
            'cta_k',
        )
    )
    del actual, reference
    torch.cuda.synchronize(x.device)
    return result


def output_error_metrics(torch, actual, reference, chunk_rows=2048):
    if actual.shape != reference.shape:
        raise ValueError('attention output shapes differ')
    squared = 0.0
    reference_squared = 0.0
    max_abs = 0.0
    exact = True
    count = 0
    sequence = int(actual.shape[2])
    for start, stop in chunk_ranges(sequence, chunk_rows):
        actual_chunk = actual[:, :, start:stop]
        reference_chunk = reference[:, :, start:stop]
        delta = actual_chunk.float() - reference_chunk.float()
        max_abs = max(max_abs, float(delta.abs().max().item()))
        squared += float(delta.square().sum().item())
        reference_squared += float(reference_chunk.float().square().sum().item())
        exact = exact and bool(torch.equal(actual_chunk, reference_chunk))
        count += int(actual_chunk.numel())
        del delta
    return {
        'exact': exact,
        'max_abs': max_abs,
        'rmse': (squared / count) ** 0.5,
        'relative_rmse': (squared / max(reference_squared, 1e-20)) ** 0.5,
    }


def complete_output_parity(
    torch,
    module,
    x,
    rope,
    chunk_size,
    block_index,
    legacy_backend,
):
    reference = full_kitchen_attention(torch, module, x, rope)
    chunked = chunked_kitchen_attention(
        torch,
        module,
        x,
        rope,
        chunk_size,
        block_index,
    )
    chunked_metrics = output_error_metrics(torch, chunked, reference)
    del chunked
    legacy = current_fused_attention(
        module,
        x,
        rope,
        block_index,
        legacy_backend,
    )
    legacy_metrics = output_error_metrics(torch, legacy, reference)
    del legacy, reference
    torch.cuda.synchronize(x.device)
    return {
        'chunked_vs_full_kitchen': chunked_metrics,
        'legacy_fused_vs_full_kitchen': legacy_metrics,
    }


def cuda_kernel_names(torch, fn, device):
    result = fn()
    torch.cuda.synchronize(device)
    del result
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as profile:
        result = fn()
        torch.cuda.synchronize(device)
    del result
    names = {
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    }
    return sorted(names)


def weight_contract(module):
    from comfy.quant_ops import QuantizedTensor

    weight = module.qkv_proj.weight
    params = getattr(weight, '_params', None)
    return {
        'quantized': isinstance(weight, QuantizedTensor),
        'layout': getattr(weight, '_layout_cls', None),
        'convrot': bool(getattr(params, 'convrot', False)),
        'convrot_groupsize': int(getattr(params, 'convrot_groupsize', 0)),
        'device': str(weight.device),
        'shape': list(weight.shape),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Compare complete full Kitchen, chunked Kitchen, and legacy fused '
            'Sage H3 QKV-through-attention boundaries.'
        )
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--kitchen-source',
        required=True,
        help='isolated Comfy Kitchen checkout containing the producer API build',
    )
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=DEFAULT_SEQUENCE)
    parser.add_argument('--chunks', default=DEFAULT_CHUNKS)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument(
        '--synthetic-block-vram-only',
        action='store_true',
        help=(
            'run real Kitchen carrier production followed by shape-faithful '
            'no-op attention and ConvRot two-slice MLP lifetimes'
        ),
    )
    parser.add_argument('--mlp-chunk-rows', type=int, default=2048)
    parser.add_argument('--modulation-rows', type=int, default=3)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--skip-kernel-trace', action='store_true')
    parser.add_argument('--skip-parity', action='store_true')
    parser.add_argument(
        '--cutlass-configs',
        default='',
        help='private diagnostic only: comma-separated Kitchen CUTLASS configs',
    )
    parser.add_argument(
        '--cutlass-rows',
        default='',
        help='row counts for --cutlass-configs (defaults to 16384 and full)',
    )
    parser.add_argument('--cutlass-iterations', type=int, default=20)
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if args.sequence <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error('sequence/iteration arguments are invalid')
    try:
        args.chunk_sizes = parse_chunk_sizes(args.chunks)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.cutlass_configs = tuple(
            int(item.strip())
            for item in args.cutlass_configs.split(',')
            if item.strip()
        )
        args.cutlass_rows = tuple(
            int(item.strip())
            for item in args.cutlass_rows.split(',')
            if item.strip()
        )
    except ValueError:
        parser.error('CUTLASS configs and row counts must be integers')
    if args.cutlass_iterations <= 0:
        parser.error('--cutlass-iterations must be positive')
    if args.mlp_chunk_rows <= 0 or args.mlp_chunk_rows % 256:
        parser.error('--mlp-chunk-rows must be a positive multiple of 256')
    if args.modulation_rows <= 0:
        parser.error('--modulation-rows must be positive')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU preflight'
        )
    return args


def main(argv=None):
    args = parse_args(argv)

    comfy_root = Path(__file__).resolve().parents[3]
    pack_root = Path(__file__).resolve().parents[1]
    kitchen_root = Path(args.kitchen_source).resolve()
    if not (kitchen_root / 'comfy_kitchen' / '__init__.py').is_file():
        raise SystemExit('--kitchen-source is not a Comfy Kitchen source checkout')
    sys.path.insert(0, str(kitchen_root))
    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(pack_root))

    import torch

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    device = torch.device('cuda')
    checkpoint = resolve_checkpoint(args.checkpoint)
    module, hidden, prefix = build_attention(
        torch,
        checkpoint,
        args.block,
        args.epsilon,
        device,
    )
    contract = weight_contract(module)
    if not (
        contract['quantized']
        and contract['layout'] == 'TensorWiseINT8Layout'
        and contract['convrot']
        and contract['convrot_groupsize'] == 256
    ):
        raise SystemExit('checkpoint QKV is not ConvRot-256 TensorWise INT8')

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    model_baseline_allocated = int(torch.cuda.memory_allocated(device))

    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(
        (args.sequence, hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope = make_rope(torch, args.sequence, device)

    if args.synthetic_block_vram_only:
        mlp_contract = load_mlp_contract(
            checkpoint,
            args.block,
            hidden,
        )
        common = {
            'heads': module.heads,
            'head_dim': module.head_dim,
            'hidden': hidden,
            'ffn': mlp_contract['ffn'],
            'mlp_chunk_rows': args.mlp_chunk_rows,
            'model_baseline_allocated': model_baseline_allocated,
            'modulation_rows': args.modulation_rows,
        }
        block_cases = {
            'full_kitchen': benchmark_synthetic_block(
                torch,
                lambda: synthetic_block_lifetime(
                    torch,
                    x,
                    lambda h: full_kitchen_carrier(torch, module, h, rope),
                    **common,
                ),
                args.warmup,
                args.iterations,
                device,
            ),
        }
        for chunk_size in args.chunk_sizes:
            block_cases['chunk_%d' % chunk_size] = benchmark_synthetic_block(
                torch,
                lambda size=chunk_size: synthetic_block_lifetime(
                    torch,
                    x,
                    lambda h: chunked_kitchen_carrier(
                        torch,
                        module,
                        h,
                        rope,
                        size,
                        args.block,
                    ),
                    **common,
                ),
                args.warmup,
                args.iterations,
                device,
            )
        result = {
            'boundary': (
                'synthetic H3 block activation lifetime: real QKV projection, '
                'RMSNorm/RoPE, and Kitchen carrier production; shape-correct '
                'no-op attention output/out projection and ConvRot two-slice '
                'MLP allocations; residual and modulation tensors remain live'
            ),
            'excludes': (
                'attention kernel workspace, out-projection quantization and '
                'GEMM workspace, MLP weights/prepacked tiles and linear '
                'workspaces, numerical behavior, and usable latency'
            ),
            'versions': {
                'torch': torch.__version__,
                'torch_cuda': torch.version.cuda,
                'comfy_kitchen': importlib.metadata.version('comfy-kitchen'),
                'comfy_kitchen_source': str(
                    Path(sys.modules['comfy_kitchen'].__file__).resolve()
                ),
            },
            'gpu': {
                'name': torch.cuda.get_device_name(device),
                'capability': list(torch.cuda.get_device_capability(device)),
            },
            'checkpoint': str(checkpoint),
            'checkpoint_prefix': prefix,
            'block': args.block,
            'sequence': args.sequence,
            'hidden': hidden,
            'heads': module.heads,
            'head_dim': module.head_dim,
            'mlp': mlp_contract,
            'mlp_chunk_rows': args.mlp_chunk_rows,
            'modulation_rows': args.modulation_rows,
            'weight_contract': contract,
            'cases': block_cases,
        }
        torch.cuda.synchronize(device)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(result['boundary'])
            print('Excluded: %s' % result['excludes'])
            for name, details in block_cases.items():
                dominant = max(
                    details['phases'],
                    key=lambda phase: details['phases'][phase][
                        'peak_above_model_baseline_bytes'
                    ],
                )
                print(
                    '%s: activation peak %.3f GiB; case increment %.3f GiB; '
                    'carrier %.3f GiB; dominant phase %s'
                    % (
                        name,
                        details['activation_peak_bytes'] / 2**30,
                        details['incremental_peak_bytes'] / 2**30,
                        details['carrier_live_bytes'] / 2**30,
                        dominant,
                    )
                )
        return 0

    legacy_backend = build_legacy_fused_backend()
    full_projection_fn = lambda: projection_only(module, x)
    full_rope_fn = lambda: projection_rope_preparation(torch, module, x, rope)
    full_preparation_fn = lambda: full_kitchen_preparation(torch, module, x, rope)
    fused_preparation_fn = lambda: current_fused_preparation(
        module, x, rope, args.block
    )
    full_attention_fn = lambda: full_kitchen_attention(torch, module, x, rope)
    legacy_attention_fn = lambda: current_fused_attention(
        module,
        x,
        rope,
        args.block,
        legacy_backend,
    )
    stage_cases = {
        'convrot_quantization_only': {
            'full': benchmark_case(
                torch,
                lambda: convrot_quantization_only(torch, module, x),
                args.warmup,
                args.iterations,
                device,
            ),
        },
        'projection_only': {
            'full': benchmark_case(
                torch, full_projection_fn, args.warmup, args.iterations, device
            ),
        },
        'projection_plus_rope': {
            'full': benchmark_case(
                torch, full_rope_fn, args.warmup, args.iterations, device
            ),
        },
    }
    preparation_cases = {
        'full_kitchen': benchmark_case(
            torch, full_preparation_fn, args.warmup, args.iterations, device
        ),
        'current_fused': benchmark_case(
            torch, fused_preparation_fn, args.warmup, args.iterations, device
        ),
    }
    cases = {
        'full_kitchen': benchmark_case(
            torch, full_attention_fn, args.warmup, args.iterations, device
        ),
        'current_fused': benchmark_case(
            torch, legacy_attention_fn, args.warmup, args.iterations, device
        ),
    }
    parity = {}
    for chunk_size in args.chunk_sizes:
        name = 'chunk_%d' % chunk_size
        projection_fn = lambda size=chunk_size: chunked_projection_only(
            module, x, size
        )
        rope_fn = lambda size=chunk_size: chunked_projection_rope_preparation(
            torch, module, x, rope, size
        )
        chunk_fn = lambda size=chunk_size: chunked_kitchen_preparation(
            torch, module, x, rope, size
        )
        quantizer_fn = lambda size=chunk_size: chunked_convrot_quantization_only(
            torch, module, x, size
        )
        stage_cases['convrot_quantization_only'][name] = benchmark_case(
            torch, quantizer_fn, args.warmup, args.iterations, device
        )
        stage_cases['projection_only'][name] = benchmark_case(
            torch, projection_fn, args.warmup, args.iterations, device
        )
        stage_cases['projection_plus_rope'][name] = benchmark_case(
            torch, rope_fn, args.warmup, args.iterations, device
        )
        preparation_cases[name] = benchmark_case(
            torch, chunk_fn, args.warmup, args.iterations, device
        )
        cases[name] = benchmark_case(
            torch,
            lambda size=chunk_size: chunked_kitchen_attention(
                torch,
                module,
                x,
                rope,
                size,
                args.block,
            ),
            args.warmup,
            args.iterations,
            device,
        )
        if not args.skip_parity:
            parity[name] = error_metrics(torch, module, x, rope, chunk_size)

    stage_cases['preparation_with_retained_v'] = preparation_cases
    primary_chunk = 4096 if 4096 in args.chunk_sizes else args.chunk_sizes[0]
    carrier_comparison = None
    output_comparison = None
    if not args.skip_parity:
        carrier_comparison = carrier_parity(
            torch,
            module,
            x,
            rope,
            primary_chunk,
            args.block,
        )
        output_comparison = complete_output_parity(
            torch,
            module,
            x,
            rope,
            primary_chunk,
            args.block,
            legacy_backend,
        )

    cutlass_configs = None
    cutlass_config_e2e = None
    if args.cutlass_configs:
        cutlass_rows = args.cutlass_rows or (16384, args.sequence)
        cutlass_configs = benchmark_cutlass_configs(
            torch,
            module,
            x,
            cutlass_rows,
            args.cutlass_configs,
            args.cutlass_iterations,
        )
        cutlass_config_e2e = {}
        for rows in cutlass_rows:
            per_config = {}
            for config in args.cutlass_configs:
                per_config[str(config)] = benchmark_case(
                    torch,
                    lambda count=rows, selected=config: forced_cutlass_projection(
                        torch, module, x[:count], selected
                    ),
                    args.warmup,
                    args.iterations,
                    device,
                )
            cutlass_config_e2e[str(rows)] = per_config

    kernel_trace = None
    if not args.skip_kernel_trace:
        trace_chunk = primary_chunk
        kernel_trace = {
            'full_kitchen': cuda_kernel_names(torch, full_attention_fn, device),
            'chunk_%d' % trace_chunk: cuda_kernel_names(
                torch,
                lambda: chunked_kitchen_attention(
                    torch,
                    module,
                    x,
                    rope,
                    trace_chunk,
                    args.block,
                ),
                device,
            ),
            'current_fused': cuda_kernel_names(
                torch,
                legacy_attention_fn,
                device,
            ),
        }

    full_ms = cases['full_kitchen']['median_ms']
    full_peak = cases['full_kitchen']['peak_allocated_bytes']
    comparisons = {
        name: {
            'time_over_full': details['median_ms'] / full_ms,
            'peak_reduction_vs_full_bytes': full_peak - details['peak_allocated_bytes'],
        }
        for name, details in cases.items()
        if name.startswith('chunk_')
    }
    result = {
        'boundary': {
            'full_kitchen': (
                'full Kitchen ConvRot QKV, Q/K RMSNorm/RoPE, complete INT8 '
                'carrier preparation, and Kitchen INT8 attention'
            ),
            'chunked_kitchen': (
                '4K ConvRot QKV chunks into final Kitchen Q/K carriers, one '
                'retained BF16 V, Kitchen V packing, and Kitchen INT8 attention'
            ),
            'current_fused': (
                'legacy fused ConvRot QKV into dense-Sage Q/K carriers, Sage '
                'FP8 V preparation, and dense Sage attention'
            ),
        },
        'versions': {
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'comfy_kitchen': importlib.metadata.version('comfy-kitchen'),
            'comfy_kitchen_source': str(
                Path(sys.modules['comfy_kitchen'].__file__).resolve()
            ),
        },
        'gpu': {
            'name': torch.cuda.get_device_name(device),
            'capability': list(torch.cuda.get_device_capability(device)),
        },
        'checkpoint': str(checkpoint),
        'checkpoint_prefix': prefix,
        'block': args.block,
        'sequence': args.sequence,
        'hidden': hidden,
        'heads': module.heads,
        'head_dim': module.head_dim,
        'weight_contract': contract,
        'theoretical_bytes': {
            'full_bf16_qkv': args.sequence * module.heads * HEAD_DIM * 3 * 2,
            'full_bf16_v': args.sequence * module.heads * HEAD_DIM * 2,
        },
        'cases': cases,
        'stages': stage_cases,
        'cutlass_configs': cutlass_configs,
        'cutlass_config_e2e': cutlass_config_e2e,
        'comparisons': comparisons,
        'projection_parity': parity,
        'carrier_parity': carrier_comparison,
        'complete_output_parity': output_comparison,
        'parity_chunk': primary_chunk,
        'kernel_trace': kernel_trace,
    }
    torch.cuda.synchronize(device)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print('full Kitchen attention: %.3f ms, peak %.3f GiB' % (
            cases['full_kitchen']['median_ms'],
            cases['full_kitchen']['peak_allocated_bytes'] / 2**30,
        ))
        for chunk_size in args.chunk_sizes:
            name = 'chunk_%d' % chunk_size
            print('%s: %.3f ms (%.3fx), peak %.3f GiB, saves %.3f GiB' % (
                name,
                cases[name]['median_ms'],
                comparisons[name]['time_over_full'],
                cases[name]['peak_allocated_bytes'] / 2**30,
                comparisons[name]['peak_reduction_vs_full_bytes'] / 2**30,
            ))
        print('stage medians (ms):')
        for stage_name, stage in stage_cases.items():
            print('  %s: %s' % (
                stage_name,
                ', '.join(
                    '%s=%.3f' % (name, details['median_ms'])
                    for name, details in stage.items()
                ),
            ))
        if cutlass_configs is not None:
            print('CUTLASS config microbenchmark:')
            print(json.dumps(cutlass_configs, indent=2, sort_keys=True))
        print('legacy fused Sage attention: %.3f ms, peak %.3f GiB' % (
            cases['current_fused']['median_ms'],
            cases['current_fused']['peak_allocated_bytes'] / 2**30,
        ))
        print('carrier parity:')
        print(json.dumps(carrier_comparison, indent=2, sort_keys=True))
        print('complete output parity:')
        print(json.dumps(output_comparison, indent=2, sort_keys=True))
        if kernel_trace is not None:
            print('kernel trace:')
            print(json.dumps(kernel_trace, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
