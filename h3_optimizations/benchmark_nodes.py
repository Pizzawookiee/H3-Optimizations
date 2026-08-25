'''Benchmark-only H3 controls and full-forward memory experiments.

These nodes are intentionally absent from normal ComfyUI registration. They are
only exposed when H3_OPTIMIZATIONS_BENCHMARK_NODES=1 is present at server
startup. They exist to make repository benchmarks reproducible without turning
diagnostic controls into supported user-facing features.
'''

from __future__ import annotations

import hashlib
import json
import logging

import torch

import comfy.ops
from comfy_api.latest import io, ui
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

from .apply import (
    ATTENTION_KITCHEN_SPARSE,
    RuntimeEnvironment,
    _resolve_attention,
)
from .model import get_h3_blocks, get_minimax_h3_model, is_minimax_h3
from .patch import configure_backend
from .plan import STATUS_KEY, read_plan
from .qkv.formats import inspect_h3_linears


# ---------------------------------------------------------------------------
# Forced Comfy Kitchen CUTLASS config 0 (benchmark fairness control)
# ---------------------------------------------------------------------------

CONFIG = 0
PATCH_PREFIX = 'diffusion_model.blocks'


class BenchmarkConfig0Error(RuntimeError):
    pass


def _forced_config0_projection(linear, x):
    '''Execute one compatible ConvRot-256 INT8 linear through CUTLASS config 0.'''
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16:
        raise BenchmarkConfig0Error(
            'benchmark config 0 requires rank-2 CUDA BF16 activations'
        )

    from comfy_kitchen.backends import cuda

    runner = getattr(cuda._C, 'cutlass_int8_dequant_config', None)
    if runner is None:
        raise BenchmarkConfig0Error(
            'Comfy Kitchen does not expose cutlass_int8_dequant_config'
        )

    weight, bias, handle = comfy.ops.cast_bias_weight(
        linear,
        x,
        offloadable=True,
        compute_dtype=torch.bfloat16,
        want_requant=True,
    )
    try:
        if bias is not None:
            raise BenchmarkConfig0Error('benchmark config 0 does not support bias')
        if not isinstance(weight, QuantizedTensor):
            raise BenchmarkConfig0Error(
                'benchmark config 0 requires quantized ConvRot INT8 QKV weights'
            )
        if getattr(weight, '_layout_cls', None) != 'TensorWiseINT8Layout':
            raise BenchmarkConfig0Error(
                'benchmark config 0 requires TensorWiseINT8Layout QKV weights'
            )
        params = weight._params
        if (
            getattr(params, 'transposed', False)
            or not getattr(params, 'convrot', False)
            or int(getattr(params, 'convrot_groupsize', 0)) != 256
        ):
            raise BenchmarkConfig0Error(
                'benchmark config 0 requires non-transposed ConvRot-256 QKV weights'
            )

        weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
        weight_qdata = weight_qdata.contiguous()
        weight_scale = weight_scale.to(torch.float32).contiguous()
        x_qdata = torch.empty_like(x, dtype=torch.int8)
        x_scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
        output = torch.empty(
            (x.shape[0], weight_qdata.shape[0]),
            dtype=torch.bfloat16,
            device=x.device,
        )
        stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
        cuda._C.quantize_int8_rowwise_convrot64(
            cuda._wrap_for_dlpack(x),
            cuda._wrap_for_dlpack(x_qdata),
            cuda._wrap_for_dlpack(x_scale),
            256,
            False,
            0,
            0,
            stream_ptr,
        )
        used = runner(
            cuda._wrap_for_dlpack(x_qdata),
            cuda._wrap_for_dlpack(weight_qdata),
            cuda._wrap_for_dlpack(x_scale),
            cuda._wrap_for_dlpack(weight_scale),
            cuda._wrap_for_dlpack(output),
            cuda.DTYPE_TO_CODE[torch.bfloat16],
            CONFIG,
            stream_ptr,
        )
        if not used:
            raise BenchmarkConfig0Error('Comfy Kitchen declined CUTLASS config 0')
        return output
    finally:
        comfy.ops.uncast_bias_weight(linear, weight, bias, handle)


def _make_forward(linear):
    def forward(x):
        return _forced_config0_projection(linear, x)

    forward._h3_benchmark_forced_kitchen_config = CONFIG
    return forward


class H3BenchmarkForceQKVConfig0(io.ComfyNode):
    '''Force compatible H3 QKV projections through Kitchen CUTLASS config 0.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3BenchmarkForceQKVConfig0',
            display_name='H3 Benchmark Force QKV Config 0',
            category='H3-Optimizations/Benchmarks',
            description=(
                'Diagnostic benchmark control. Forces ConvRot-256 INT8 H3 QKV '
                'linears through Comfy Kitchen CUTLASS config 0. Not a public '
                'production node.'
            ),
            inputs=[io.Model.Input('model')],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model):
        if not is_minimax_h3(model):
            return io.NodeOutput(model)
        patched = model.clone()
        blocks = get_h3_blocks(patched)
        if not blocks:
            raise BenchmarkConfig0Error('MiniMax H3 has no main blocks')
        for index, block in enumerate(blocks):
            linear = block.attn.qkv_proj
            patched.add_object_patch(
                '%s.%d.attn.qkv_proj.forward' % (PATCH_PREFIX, index),
                _make_forward(linear),
            )
        return io.NodeOutput(patched)


class H3BenchmarkAssertRoute(io.ComfyNode):
    '''Fail before sampling when a benchmark arm resolved the wrong route.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3BenchmarkAssertRoute',
            display_name='H3 Benchmark Assert Route',
            category='H3-Optimizations/Benchmarks',
            description='Benchmark-only fail-closed route assertion.',
            inputs=[
                io.Model.Input('model'),
                io.String.Input('attention'),
                io.String.Input('backend'),
                io.String.Input('qkv'),
                io.String.Input('projector'),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, attention, backend, qkv, projector):
        options = (
            getattr(model, 'model_options', {})
            .get('transformer_options', {})
            or {}
        )
        status = options.get(STATUS_KEY) or {}
        attention_status = status.get('attention') or {}
        qkv_status = status.get('fused_qkv') or {}
        backend_status = attention_status.get('backend_details') or {}
        actual = {
            'attention': attention_status.get('selected'),
            'backend': backend_status.get('backend'),
            'qkv': qkv_status.get('provider'),
            'projector': qkv_status.get('projector'),
        }
        expected = {
            'attention': str(attention),
            'backend': str(backend),
            'qkv': str(qkv),
            'projector': str(projector),
        }
        if actual != expected:
            raise RuntimeError(
                'benchmark route mismatch: expected %s, actual %s'
                % (json.dumps(expected, sort_keys=True), json.dumps(actual, sort_keys=True))
            )
        logging.info(
            '[H3 Benchmark] validated route: %s',
            json.dumps(actual, sort_keys=True),
        )
        return io.NodeOutput(model)


# ---------------------------------------------------------------------------
# Full-forward memory experiment nodes
# ---------------------------------------------------------------------------

LOG_PREFIX = '[H3 Full Forward Experiment]'
VARIANTS = (
    'final_layer_chunking',
    'streamed_kitchen_output',
    'combined',
)
_STATS = {}
_REFERENCES = {}


def _selector(selected, start, stop):
    return selected if selected.ndim == 1 else selected[start:stop]


def chunked_final_layer(layer, x, t_emb, video_seg, audio_seg, chunk_rows):
    shift, scale = layer.adaln_proj(t_emb)

    def project(segment, output):
        first, last, row = segment
        pieces = []
        selected_scale = scale[row]
        selected_shift = shift[row]
        for start in range(first, last, int(chunk_rows)):
            stop = min(start + int(chunk_rows), last)
            local_start = start - first
            local_stop = stop - first
            normalized = layer.norm(x[start:stop])
            value = (
                normalized * (1.0 + _selector(selected_scale, local_start, local_stop))
                + _selector(selected_shift, local_start, local_stop)
            ).to(torch.float32)
            pieces.append(output(value))
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    return project(video_seg, layer.video_out), project(audio_seg, layer.audio_out)


def make_chunked_final_forward(layer, chunk_rows, stats):
    def forward(x, t_emb, video_seg, audio_seg):
        result = chunked_final_layer(
            layer, x, t_emb, video_seg, audio_seg, chunk_rows
        )
        stats['final_layer_calls'] += 1
        logging.info(
            '%s chunked FinalLayer completed forward #%d',
            LOG_PREFIX,
            stats['final_layer_calls'],
        )
        return result

    return forward


def _cpu_streams(samples):
    value = samples['samples']
    streams = value.unbind() if getattr(value, 'is_nested', False) else (value,)
    return tuple(stream.detach().cpu().contiguous() for stream in streams)


def _digest(stream):
    raw = stream.view(torch.uint8).numpy().tobytes()
    return {
        'shape': list(stream.shape),
        'dtype': str(stream.dtype),
        'sha256': hashlib.sha256(raw).hexdigest(),
    }


def _compare(reference, actual):
    if len(reference) != len(actual):
        raise RuntimeError('full-forward output stream count changed')
    total_squared = 0.0
    total_reference_squared = 0.0
    total_values = 0
    max_abs = 0.0
    exact = True
    for expected, observed in zip(reference, actual):
        if expected.shape != observed.shape:
            raise RuntimeError('full-forward output shape changed')
        exact = exact and bool(torch.equal(expected, observed))
        delta = observed.float() - expected.float()
        max_abs = max(max_abs, float(delta.abs().max().item()))
        total_squared += float(delta.square().sum().item())
        total_reference_squared += float(expected.float().square().sum().item())
        total_values += delta.numel()
    rmse = (total_squared / max(1, total_values)) ** 0.5
    reference_rms = (total_reference_squared / max(1, total_values)) ** 0.5
    return {
        'exact': exact,
        'max_abs': max_abs,
        'rmse': rmse,
        'relative_rmse': rmse / max(reference_rms, 1e-12),
    }


class CapturedProjection:
    def __init__(self, projected, input):
        self.projected = projected
        self.input = input


class CapturingProjector:
    name = 'captured_chunked_kitchen_qkv'

    def __init__(self, projector):
        self.projector = projector

    @property
    def installation_signature(self):
        return (self.name, self.projector.installation_signature)

    def bind(self, module):
        bind = getattr(self.projector, 'bind', None)
        if bind is not None:
            bind(module)

    def try_project(self, module, x, rope_freqs, **kwargs):
        projected = self.projector.try_project(module, x, rope_freqs, **kwargs)
        return None if projected is None else CapturedProjection(projected, x)


class PreparedStreamedOutput:
    def __init__(self, prepared, input, layer_index):
        self.prepared = prepared
        self.input = input
        self.layer_index = int(layer_index)


class StreamedOutputBackend:
    name = 'sparse_kitchen_streamed_output_full_forward_experiment'

    def __init__(self, backend, query_chunk_rows, stats, layer_count):
        q_tile = int(backend.executor.q_tile)
        if q_tile not in (64, 128):
            raise ValueError('streamed output requires 64- or 128-row Q tiles')
        if query_chunk_rows <= 0 or query_chunk_rows % 128:
            raise ValueError('query chunk rows must be a positive 128-row multiple')
        self.backend = backend
        self.query_chunk_rows = int(query_chunk_rows)
        self.stats = stats
        self.layer_count = int(layer_count)
        self.requires_runtime_context = backend.requires_runtime_context
        self.approximate = backend.approximate

    @property
    def installation_signature(self):
        return (
            self.name,
            self.query_chunk_rows,
            self.backend.installation_signature,
        )

    def prepare_projected(self, projected, *, layer_index, **kwargs):
        if not isinstance(projected, CapturedProjection):
            raise TypeError('streamed output requires CapturedProjection')
        prepared = self.backend.prepare_projected(
            projected.projected, layer_index=layer_index, **kwargs
        )
        return PreparedStreamedOutput(prepared, projected.input, layer_index)

    def execute_projected(self, module, wrapped):
        from dataclasses import replace

        prepared = wrapped.prepared
        quantized = prepared.quantized
        route = prepared.route
        output = wrapped.input
        route_q_tile = int(route.q_tile)
        sequence = int(quantized.q.shape[-2])
        if int(quantized.q.shape[-1]) != 128:
            raise RuntimeError('streamed output requires head_dim 128')
        packed_q_tiles = (sequence + 127) // 128
        if quantized.q_scale.shape[-1] % packed_q_tiles:
            raise RuntimeError('streamed output received invalid Q scales')
        scales_per_packed_q_tile = quantized.q_scale.shape[-1] // packed_q_tiles

        for start in range(0, sequence, self.query_chunk_rows):
            stop = min(start + self.query_chunk_rows, sequence)
            first_route_tile = start // route_q_tile
            stop_route_tile = (stop + route_q_tile - 1) // route_q_tile
            first_scale_tile = start // 128
            stop_scale_tile = (stop + 127) // 128
            q = quantized.q[..., start:stop, :].contiguous()
            q_scale = quantized.q_scale[
                ...,
                first_scale_tile * scales_per_packed_q_tile:
                stop_scale_tile * scales_per_packed_q_tile,
            ].contiguous()
            chunk_carrier = replace(quantized, q=q, q_scale=q_scale)
            chunk_route = replace(
                route,
                indices=route.indices[
                    ..., first_route_tile:stop_route_tile, :
                ].contiguous(),
                counts=route.counts[
                    ..., first_route_tile:stop_route_tile
                ].contiguous(),
            )
            raw = self.backend.executor.kitchen.block_sparse_int8_attention_from_prequantized(
                chunk_carrier, chunk_route, output_layout='nhd'
            )
            flat = raw.transpose(1, 2).reshape(
                raw.shape[0], raw.shape[2], module.heads * module.head_dim
            )
            output[start:stop].copy_(module.out_proj(flat.squeeze(0)))
            del raw, flat, chunk_carrier, chunk_route, q, q_scale

        prepared.release()
        wrapped.prepared = None
        self.stats['streamed_blocks'] += 1
        if wrapped.layer_index == self.layer_count - 1:
            self.stats['streamed_complete_forwards'] += 1
            logging.info(
                '%s streamed Kitchen output completed forward #%d (%d/%d layers)',
                LOG_PREFIX,
                self.stats['streamed_complete_forwards'],
                self.layer_count,
                self.layer_count,
            )
        return output


class H3FullForwardExperiment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3FullForwardExperiment',
            display_name='H3 Full Forward Experiment',
            category='H3-Optimizations/Benchmarks',
            description='Benchmark-only FinalLayer and streamed Kitchen output patch.',
            is_experimental=True,
            inputs=[
                io.Model.Input('model'),
                io.Combo.Input('variant', options=list(VARIANTS)),
                io.Int.Input('chunk_rows', default=4096, min=128, step=128),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, variant, chunk_rows=4096):
        if not is_minimax_h3(model):
            return io.NodeOutput(model)
        if variant not in VARIANTS:
            raise ValueError('unknown full-forward experiment %r' % variant)

        patched = model.clone()
        blocks = get_h3_blocks(patched)
        stats = {
            'variant': variant,
            'attention_layers': len(blocks),
            'streamed_blocks': 0,
            'streamed_complete_forwards': 0,
            'final_layer_calls': 0,
        }
        _STATS[variant] = stats

        if variant in ('streamed_kitchen_output', 'combined'):
            plan = read_plan(patched)
            inventory = inspect_h3_linears(blocks)
            attention, _qkv = _resolve_attention(
                plan, patched, inventory, RuntimeEnvironment.detect()
            )
            if attention.backend_kind != ATTENTION_KITCHEN_SPARSE:
                raise RuntimeError(
                    'streamed output requires explicit Kitchen INT8, resolved %s'
                    % attention.backend_kind
                )
            if attention.projector is None:
                raise RuntimeError('streamed output requires chunked Kitchen QKV')
            backend = StreamedOutputBackend(
                attention.backend, int(chunk_rows), stats, len(blocks)
            )
            projector = CapturingProjector(attention.projector)
            _backend, installed = configure_backend(
                patched, backend, projector=projector
            )
            if installed != len(blocks):
                raise RuntimeError(
                    'streamed output patched %d of %d attention layers'
                    % (installed, len(blocks))
                )
            stats.update({
                'route_q_tile': int(attention.backend.executor.q_tile),
                'route_kv_tile': int(attention.backend.executor.kv_tile),
                'query_chunk_rows': int(chunk_rows),
            })

        if variant in ('final_layer_chunking', 'combined'):
            diffusion_model = get_minimax_h3_model(patched)
            patched.add_object_patch(
                'diffusion_model.final_layer.forward',
                make_chunked_final_forward(
                    diffusion_model.final_layer, int(chunk_rows), stats
                ),
            )
            stats['final_layer_chunk_rows'] = int(chunk_rows)

        logging.info('%s armed %s: %s', LOG_PREFIX, variant, json.dumps(stats))
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText('Armed benchmark-only %s' % variant),
        )


class H3FullForwardDigest(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3FullForwardDigest',
            display_name='H3 Full Forward Digest',
            category='H3-Optimizations/Benchmarks',
            description='Benchmark-only output parity sink for MiniMax H3 AV latents.',
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Latent.Input('samples'),
                io.String.Input('arm'),
                io.String.Input('reference_key'),
            ],
        )

    @classmethod
    def execute(cls, samples, arm, reference_key):
        streams = _cpu_streams(samples)
        record = {
            'arm': str(arm),
            'reference_key': str(reference_key),
            'streams': [_digest(stream) for stream in streams],
            'execution': dict(_STATS.get(str(arm), {})),
        }
        if arm == 'baseline':
            _REFERENCES[str(reference_key)] = streams
            record['comparison'] = {
                'exact': True,
                'max_abs': 0.0,
                'rmse': 0.0,
                'relative_rmse': 0.0,
            }
        else:
            reference = _REFERENCES.get(str(reference_key))
            if reference is None:
                raise RuntimeError(
                    'full-forward baseline %r was not captured' % reference_key
                )
            record['comparison'] = _compare(reference, streams)
        serialized = json.dumps(record, sort_keys=True)
        logging.info('H3_FULL_FORWARD_DIGEST=%s', serialized)
        return io.NodeOutput(ui=ui.PreviewText(serialized))
