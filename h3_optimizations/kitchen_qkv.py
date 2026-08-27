'''Chunked H3 QKV production through Comfy Kitchen's public carrier API.'''

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

import comfy.model_management
import comfy.quant_ops

from . import diagnostics
from .attention_forward import project_qkv
from .qkv.bf16 import HeldBF16QKV
from .qkv.chunked import project_chunk_hnd
from .qkv.formats import describe_linear
from .qkv.fp8 import FP8BindingError, HeldFP8QKV
from .qkv.int8 import ConvRotINT8BindingError, HeldConvRotINT8QKV
from .qkv.streamed import (
    PROJECTION_FORCE_BF16,
    PROJECTION_FORCE_FP8,
    PROJECTION_FORCE_INT8,
    PROJECTION_NATIVE,
    create_held_qkv,
    project_kv_hnd,
    project_q_hnd,
    project_v_hnd,
)
from .qkv.w4a8 import HeldW4A8QKV, W4A8BindingError


CHUNK_ROWS = 4096
STRIDED_QK_CAPABILITY = 'SUPPORTS_STRIDED_QK_CHUNK'
PRODUCER_ABI_VERSION = 1
PRODUCER_ABI = 'external_v%d' % PRODUCER_ABI_VERSION
PRODUCER_API = (
    'INT8_ATTENTION_PRODUCER_ABI_VERSION',
    'Int8AttentionProducerUnavailableError',
    'int8_attention_producer_is_available',
    'int8_attention_producer_spec',
    'select_int8_attention_k_anchor',
    'create_int8_attention_producer',
    'quantize_int8_attention_qk_chunk',
    'quantize_int8_attention_v',
    'finalize_int8_attention_producer',
    'int8_attention_from_prequantized',
)
STREAMED_PRODUCER_API = (
    'quantize_int8_attention_k_chunk',
    'quantize_int8_attention_q_chunk',
)
H3_ATTENTION_BACKEND_KEY = 'h3_optimizations_attention_backend'
DENSE_KITCHEN_BACKEND = 'comfy_kitchen_int8_prequantized'


class FusedQKVError(RuntimeError):
    pass


def resolve_kitchen(device=None):
    """The module providing the producer API, vendored first.

    The vendored library ships with this pack. comfy_kitchen is whatever pip
    installed, and ComfyUI pins 0.2.31, which has no producer API at all --
    which is why this whole path was integrated and then never ran.
    """
    from .native import int8_attention as _  # noqa: F401 - import check

    from . import native

    if _supports_producer(native, device):
        return native
    kitchen = comfy.quant_ops.ck
    return kitchen if _supports_producer(kitchen, device) else None


def _supports_producer(module, device):
    if module is None:
        return False
    if not all(hasattr(module, name) for name in PRODUCER_API):
        return False
    if module.INT8_ATTENTION_PRODUCER_ABI_VERSION != PRODUCER_ABI_VERSION:
        return False
    try:
        return bool(module.int8_attention_producer_is_available(device))
    except Exception:  # noqa: BLE001 - an unavailable backend is not an error
        return False


def _supports_streamed_producer(module, device):
    return bool(
        _supports_producer(module, device)
        and all(hasattr(module, name) for name in STREAMED_PRODUCER_API)
    )


def producer_api_available(kitchen=None, device=None):
    if kitchen is not None:
        return _supports_producer(kitchen, device)
    return resolve_kitchen(device) is not None


def _native_bf16_format(fmt):
    dtype = str(getattr(fmt, 'logical_dtype', '')).lower()
    return bool(
        getattr(fmt, 'plain_float', False)
        and ('bfloat16' in dtype or 'bf16' in dtype)
    )


@dataclass
class PreparedChunkedKitchenQKV:
    carrier: object
    q_summary: torch.Tensor | None = None
    k_summary: torch.Tensor | None = None
    output_buffer: torch.Tensor | None = None

    def release(self):
        self.carrier = None
        self.q_summary = None
        self.k_summary = None
        self.output_buffer = None


@dataclass
class PreparedStreamedKitchenQKV:
    module: object
    x: torch.Tensor
    rope_freqs: torch.Tensor | None
    carrier: object
    projection_mode: str
    output_buffer: torch.Tensor | None
    owned_q_projection: object | None = None
    q_int8_workspace: torch.Tensor | None = None
    q_scale_workspace: torch.Tensor | None = None
    q_summary: torch.Tensor | None = None
    k_summary: torch.Tensor | None = None

    def release(self):
        self.module = None
        self.x = None
        self.rope_freqs = None
        self.carrier = None
        self.output_buffer = None
        if self.owned_q_projection is not None:
            self.owned_q_projection.release()
        self.owned_q_projection = None
        self.q_int8_workspace = None
        self.q_scale_workspace = None
        self.q_summary = None
        self.k_summary = None


def _rope_rows(rope_freqs, rows):
    if rope_freqs is None:
        return None
    return rope_freqs.index_select(1, rows)


def _project_anchor_samples(module, x, rope_freqs, positions, projector=None):
    rows = torch.tensor(positions, dtype=torch.int64, device=x.device)
    if projector is not None:
        _q, k, _v = projector.project_rows(x, rope_freqs, rows)
        return k
    sample_x = x.index_select(0, rows)
    sample_rope = _rope_rows(rope_freqs, rows)
    _q, k, _v = project_qkv(module, sample_x, sample_rope)
    return k.transpose(0, 1).unsqueeze(0)


def _qk_chunk_kwargs(kitchen, strided_qk_input):
    '''Ask for strided Q/K only from a producer that advertises it.

    `resolve_kitchen` can return an installed comfy-kitchen whose chunk
    quantizer has no such parameter. Passing the keyword blindly would raise a
    TypeError that the projector's own except clause would then swallow into a
    silent fallback -- the exact failure mode this pack keeps trying to
    eliminate.
    '''
    if not strided_qk_input:
        return {}
    if not getattr(kitchen, STRIDED_QK_CAPABILITY, False):
        raise FusedQKVError(
            'strided Q/K chunk input was requested but %r does not support it'
            % getattr(kitchen, '__name__', kitchen)
        )
    return {'allow_strided_input': True}


def _tile_mean(x, tile):
    full = x.shape[-2] // tile
    remainder = x.shape[-2] % tile
    pieces = []
    if full:
        pieces.append(
            x[..., :full * tile, :]
            .reshape(*x.shape[:-2], full, tile, x.shape[-1])
            .mean(dim=-2)
        )
    if remainder:
        pieces.append(x[..., full * tile:, :].mean(dim=-2, keepdim=True))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)


def run_chunked_kitchen_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    transformer_options,
    spec,
    chunk_rows=CHUNK_ROWS,
    force_weights_bf16=False,
    fp8_projection=False,
    convrot_int8_projection=False,
    routing_summaries=False,
    routing_q_tile=None,
    routing_kv_tile=None,
    strided_qk_input=False,
    v_optimization=False,
):
    del layer_index, transformer_options
    routing_q_tile = int(
        spec.q_tile if routing_q_tile is None else routing_q_tile
    )
    routing_kv_tile = int(
        spec.k_tile if routing_kv_tile is None else routing_kv_tile
    )
    kitchen = resolve_kitchen(x.device)
    if kitchen is None:
        raise FusedQKVError('no INT8 attention producer is available')
    held = None
    try:
        fmt = describe_linear(module.qkv_proj)
        if force_weights_bf16:
            held = HeldBF16QKV(module, x[:1], allow_quantized_source=True)
            held.__enter__()
        elif convrot_int8_projection:
            held = HeldConvRotINT8QKV(
                module,
                x[:1],
                allow_float_conversion=getattr(fmt, 'plain_float', False),
            )
            held.__enter__()
        elif fp8_projection:
            held = HeldFP8QKV(
                module,
                x[:1],
                allow_float_conversion=getattr(fmt, 'plain_float', False),
            )
            held.__enter__()
        elif getattr(fmt, 'w4a8', False):
            held = HeldW4A8QKV(module, x[:1])
            held.__enter__()
        elif _native_bf16_format(fmt):
            held = HeldBF16QKV(module, x[:1])
            held.__enter__()

        if v_optimization and held is None:
            # Native ConvRot normally lets project_chunk_hnd call the module
            # directly. Two-pass V needs to reuse the exact same full fused QKV
            # weight for V pass 2, so acquire the upstream native held binding.
            held = create_held_qkv(module, x[:1], PROJECTION_NATIVE)
            held.__enter__()

        with diagnostics.stage('anchor_projection'):
            samples = _project_anchor_samples(
                module, x, rope_freqs, spec.k_anchor_positions, projector=held
            )
        anchor = kitchen.select_int8_attention_k_anchor(spec, samples)
        del samples
        with diagnostics.stage('producer_create'):
            producer = kitchen.create_int8_attention_producer(spec, anchor)
        del anchor

        v_staging = None
        if v_optimization:
            from .native.v_staging import TwoPassVCarrier
            v_staging = TwoPassVCarrier(spec, backend='native')

        sequence = int(x.shape[0])
        chunk_kwargs = _qk_chunk_kwargs(kitchen, strided_qk_input)
        retained_v = None
        q_summaries = []
        k_summaries = []
        for start in range(0, sequence, int(chunk_rows)):
            end = min(start + int(chunk_rows), sequence)
            q, k, v = project_chunk_hnd(
                module, x, rope_freqs, start, end, projector=held
            )
            if routing_summaries:
                with diagnostics.stage('routing_summary_generation'):
                    q_summaries.append(_tile_mean(q, routing_q_tile))
                    k_summaries.append(_tile_mean(k, routing_kv_tile))
            kitchen.quantize_int8_attention_qk_chunk(
                producer, q, k, q_start=start, k_start=start, **chunk_kwargs
            )
            if v_staging is not None:
                with diagnostics.stage('v_amax_update'):
                    v_staging.update(v)
            else:
                if retained_v is None:
                    retained_v = v.new_empty(
                        (1, int(module.heads), sequence, int(module.head_dim))
                    )
                with diagnostics.stage('v_retention_copy'):
                    retained_v[:, :, start:end, :].copy_(v)
            del q, k, v

        if v_staging is not None:
            with diagnostics.stage('v_scale_finalize'):
                v_staging.finalize_scale()
            for start in range(0, sequence, int(chunk_rows)):
                end = min(start + int(chunk_rows), sequence)
                with diagnostics.stage('v_reprojection'):
                    v = project_v_hnd(held, x, rope_freqs, start, end)
                with diagnostics.stage('v_carrier_pack'):
                    v_staging.quantize(v, start)
                del v
            producer.v, producer.v_scale = v_staging.finish()
        else:
            kitchen.quantize_int8_attention_v(producer, retained_v)
            del retained_v

        with diagnostics.stage('carrier_finalize'):
            return PreparedChunkedKitchenQKV(
                kitchen.finalize_int8_attention_producer(producer),
                q_summary=torch.cat(q_summaries, dim=-2) if q_summaries else None,
                k_summary=torch.cat(k_summaries, dim=-2) if k_summaries else None,
            )
    finally:
        if held is not None:
            held.__exit__(None, None, None)


def run_streamed_kitchen_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    transformer_options,
    spec,
    chunk_rows,
    projection_mode,
    strided_qk_input=False,
    routing_summaries=False,
    routing_q_tile=None,
    routing_kv_tile=None,
    v_optimization=False,
):
    del layer_index, transformer_options
    kitchen = resolve_kitchen(x.device)
    if kitchen is None:
        raise FusedQKVError('no streamed INT8 attention producer is available')

    routing_q_tile = int(spec.q_tile if routing_q_tile is None else routing_q_tile)
    routing_kv_tile = int(spec.k_tile if routing_kv_tile is None else routing_kv_tile)

    held = create_held_qkv(module, x[:1], projection_mode)
    held.__enter__()
    owned_q_projection = None
    try:
        with diagnostics.stage('anchor_projection'):
            samples = _project_anchor_samples(
                module, x, rope_freqs, spec.k_anchor_positions, projector=held
            )
        anchor = kitchen.select_int8_attention_k_anchor(spec, samples)
        del samples
        producer = kitchen.create_int8_attention_producer(spec, anchor)
        del anchor

        v_staging = None
        if v_optimization:
            from .native.v_staging import TwoPassVCarrier
            v_staging = TwoPassVCarrier(spec, backend='native')

        sequence = int(x.shape[0])
        retained_v = None
        k_summaries = []
        chunk_kwargs = _qk_chunk_kwargs(kitchen, strided_qk_input)

        for start in range(0, sequence, int(chunk_rows)):
            end = min(start + int(chunk_rows), sequence)
            k, v = project_kv_hnd(held, x, rope_freqs, start, end)

            if routing_summaries:
                with diagnostics.stage('routing_summary_generation'):
                    k_summaries.append(_tile_mean(k, routing_kv_tile))

            kitchen.quantize_int8_attention_k_chunk(
                producer, k, k_start=start, **chunk_kwargs
            )

            if v_staging is not None:
                with diagnostics.stage('v_amax_update'):
                    v_staging.update(v)
            else:
                if retained_v is None:
                    retained_v = v.new_empty(
                        (1, int(module.heads), sequence, int(module.head_dim))
                    )
                with diagnostics.stage('v_retention_copy'):
                    retained_v[..., start:end, :].copy_(v)

            del k, v

        if v_staging is not None:
            with diagnostics.stage('v_scale_finalize'):
                v_staging.finalize_scale()

            for start in range(0, sequence, int(chunk_rows)):
                end = min(start + int(chunk_rows), sequence)
                with diagnostics.stage('v_reprojection'):
                    v = project_v_hnd(held, x, rope_freqs, start, end)
                with diagnostics.stage('v_carrier_pack'):
                    v_staging.quantize(v, start)
                del v

            producer.v, producer.v_scale = v_staging.finish()
        else:
            kitchen.quantize_int8_attention_v(producer, retained_v)
            del retained_v

        carrier = kitchen.finalize_int8_attention_producer(producer)

        clone_owned_q = getattr(held, 'clone_owned_q_projection', None)
        if callable(clone_owned_q):
            with diagnostics.stage('owned_q_weight_clone'):
                owned_q_projection = clone_owned_q()
            if (
                owned_q_projection is not None
                and 'lora_cache' in locals()
                and lora_cache is not None
            ):
                owned_q_projection.attach_sliced_lora_cache(lora_cache)
    finally:
        held.__exit__(None, None, None)

    max_q_rows = min(int(chunk_rows), int(x.shape[0]))
    q_int8_workspace = torch.empty(
        int(module.heads) * max_q_rows * int(module.head_dim),
        dtype=torch.int8,
        device=x.device,
    )
    q_scale_workspace = torch.empty(
        int(module.heads)
        * ((max_q_rows + 127) // 128)
        * 32,
        dtype=torch.float32,
        device=x.device,
    )

    return PreparedStreamedKitchenQKV(
        module=module,
        x=x,
        rope_freqs=rope_freqs,
        carrier=carrier,
        projection_mode=projection_mode,
        output_buffer=x,
        owned_q_projection=owned_q_projection,
        q_int8_workspace=q_int8_workspace,
        q_scale_workspace=q_scale_workspace,
        q_summary=None,
        k_summary=torch.cat(k_summaries, dim=-2) if k_summaries else None,
    )


class ChunkedKitchenQKVProjector:
    name = 'chunked_kitchen_qkv'

    def __init__(
        self,
        chunk_rows=CHUNK_ROWS,
        force_weights_bf16=False,
        fp8_projection=False,
        convrot_int8_projection=False,
        routing_summaries=False,
        q_tile=None,
        kv_tile=None,
        strided_qk_input=False,
        stream_output=False,
        streamed_q=False,
        v_optimization=False,
    ):
        self.chunk_rows = int(chunk_rows)
        self.force_weights_bf16 = bool(force_weights_bf16)
        self.fp8_projection = bool(fp8_projection)
        self.convrot_int8_projection = bool(convrot_int8_projection)
        if sum(
            (
                self.force_weights_bf16,
                self.fp8_projection,
                self.convrot_int8_projection,
            )
        ) > 1:
            raise ValueError(
                'Kitchen QKV projection cannot force more than one weight format'
            )
        self.routing_summaries = bool(routing_summaries)
        self.q_tile = None if q_tile is None else int(q_tile)
        self.kv_tile = None if kv_tile is None else int(kv_tile)
        if self.q_tile is not None and self.q_tile <= 0:
            raise ValueError('q_tile must be positive')
        if self.kv_tile is not None and self.kv_tile <= 0:
            raise ValueError('kv_tile must be positive')
        self.strided_qk_input = bool(strided_qk_input)
        self.stream_output = bool(stream_output)
        self.streamed_q = bool(streamed_q)
        self.v_optimization = bool(v_optimization)
        if self.streamed_q and not self.stream_output:
            raise ValueError('streamed Kitchen Q requires streamed output')

    @property
    def installation_signature(self):
        return (
            self.name,
            self.chunk_rows,
            self.force_weights_bf16,
            self.fp8_projection,
            self.convrot_int8_projection,
            self.routing_summaries,
            self.q_tile,
            self.kv_tile,
            self.strided_qk_input,
            self.stream_output,
            self.streamed_q,
            self.v_optimization,
        )

    def try_project(
        self,
        module,
        x,
        rope_freqs,
        *,
        layer_index,
        transformer_options,
    ):
        kitchen = resolve_kitchen(x.device)
        if kitchen is None:
            return None
        fmt = describe_linear(module.qkv_proj)
        native_bf16 = _native_bf16_format(fmt)
        if self.force_weights_bf16:
            format_ok = bool(
                getattr(fmt, 'plain_float', False)
                or getattr(fmt, 'convrot_int8_256', False)
                or getattr(fmt, 'w4a8', False)
                or getattr(fmt, 'fp8', False)
            )
        elif self.convrot_int8_projection:
            format_ok = bool(
                getattr(fmt, 'convrot_int8_256', False)
                or getattr(fmt, 'plain_float', False)
            )
        elif self.fp8_projection:
            format_ok = bool(
                getattr(fmt, 'fp8', False)
                or getattr(fmt, 'plain_float', False)
            )
        else:
            format_ok = bool(
                getattr(fmt, 'convrot_int8_256', False)
                or getattr(fmt, 'w4a8', False)
                or native_bf16
            )
        owns_dense_h3 = (
            (transformer_options or {}).get(H3_ATTENTION_BACKEND_KEY)
            == DENSE_KITCHEN_BACKEND
        )
        if (
            (not self.routing_summaries and not owns_dense_h3)
            or comfy.model_management.in_training
            or x.ndim != 2
            or not x.is_cuda
            or not format_ok
            or not producer_api_available(kitchen, x.device)
        ):
            return None

        use_streamed_q = bool(
            self.streamed_q
            and _supports_streamed_producer(kitchen, x.device)
        )
        shape = (1, int(module.heads), int(x.shape[0]), int(module.head_dim))
        q_shape = (
            (1, int(module.heads), 1, int(module.head_dim))
            if use_streamed_q
            else shape
        )
        try:
            spec_kwargs = {}
            if self.kv_tile is not None:
                spec_kwargs['cta_k'] = self.kv_tile
            spec = kitchen.int8_attention_producer_spec(
                q_shape,
                shape,
                dtype=x.dtype,
                device=x.device,
                **spec_kwargs,
            )
        except kitchen.Int8AttentionProducerUnavailableError:
            return None
        if (
            getattr(spec, 'abi_version', None) != PRODUCER_ABI_VERSION
            or self.chunk_rows % int(spec.sequence_alignment)
        ):
            return None
        try:
            with diagnostics.stage('qkv_producer_total'):
                if use_streamed_q:
                    projection_mode = PROJECTION_NATIVE
                    if self.force_weights_bf16:
                        projection_mode = PROJECTION_FORCE_BF16
                    elif self.fp8_projection:
                        projection_mode = PROJECTION_FORCE_FP8
                    elif self.convrot_int8_projection:
                        projection_mode = PROJECTION_FORCE_INT8
                    projected = run_streamed_kitchen_qkv(
                        module,
                        x,
                        rope_freqs,
                        layer_index=layer_index,
                        transformer_options=transformer_options,
                        spec=spec,
                        chunk_rows=self.chunk_rows,
                        projection_mode=projection_mode,
                        strided_qk_input=self.strided_qk_input,
                        routing_summaries=self.routing_summaries,
                        routing_q_tile=self.q_tile,
                        routing_kv_tile=self.kv_tile,
                        v_optimization=self.v_optimization,
                    )
                else:
                    projected = run_chunked_kitchen_qkv(
                        module,
                        x,
                        rope_freqs,
                        layer_index=layer_index,
                        transformer_options=transformer_options,
                        spec=spec,
                        chunk_rows=self.chunk_rows,
                        force_weights_bf16=self.force_weights_bf16,
                        fp8_projection=self.fp8_projection,
                        convrot_int8_projection=self.convrot_int8_projection,
                        routing_summaries=self.routing_summaries,
                        routing_q_tile=self.q_tile,
                        routing_kv_tile=self.kv_tile,
                        strided_qk_input=self.strided_qk_input,
                        v_optimization=self.v_optimization,
                    )
                if self.stream_output:
                    projected = replace(projected, output_buffer=x)
                return projected
        except (
            FP8BindingError,
            ConvRotINT8BindingError,
            W4A8BindingError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            if self.force_weights_bf16 or self.convrot_int8_projection:
                raise
            if (
                not self.fp8_projection
                and not getattr(fmt, 'w4a8', False)
            ):
                raise
            return None


class ChunkedKitchenAttentionBackend:
    name = DENSE_KITCHEN_BACKEND

    def __init__(self, *, stream_output=False, query_chunk_rows=CHUNK_ROWS):
        self.stream_output = bool(stream_output)
        self.query_chunk_rows = int(query_chunk_rows)
        if self.stream_output and (
            self.query_chunk_rows <= 0 or self.query_chunk_rows % 128
        ):
            raise ValueError(
                'query_chunk_rows must be a positive 128-row multiple'
            )

    @property
    def installation_signature(self):
        return (self.name, self.stream_output, self.query_chunk_rows)

    def prepare(self, *_args, **_kwargs):
        raise RuntimeError('chunked Kitchen attention requires its QKV producer')

    def prepare_projected(
        self,
        projected,
        *,
        layer_index,
        transformer_options,
    ):
        del layer_index, transformer_options
        if not isinstance(
            projected,
            (PreparedChunkedKitchenQKV, PreparedStreamedKitchenQKV),
        ):
            raise TypeError('chunked Kitchen attention received an invalid carrier')
        return projected

    def execute(self, prepared):
        if not isinstance(prepared, PreparedChunkedKitchenQKV):
            raise TypeError('chunked Kitchen attention received an invalid carrier')
        # The carrier has to be consumed by the module that produced it:
        # the two implementations agree byte for byte, but only one of them is
        # guaranteed to be present.
        kitchen = resolve_kitchen()
        if kitchen is None:
            raise FusedQKVError('no INT8 attention producer is available')
        return kitchen.int8_attention_from_prequantized(prepared.carrier)

    def execute_projected(self, module, prepared):
        if not self.stream_output or prepared.output_buffer is None:
            return None
        if not isinstance(
            prepared,
            (PreparedChunkedKitchenQKV, PreparedStreamedKitchenQKV),
        ):
            raise TypeError('chunked Kitchen attention received an invalid carrier')

        kitchen = resolve_kitchen()
        if kitchen is None:
            raise FusedQKVError('no INT8 attention producer is available')
        from .native.int8_attention import OUTPUT_NHD

        quantized = prepared.carrier
        output = prepared.output_buffer
        streamed_q = isinstance(prepared, PreparedStreamedKitchenQKV)
        sequence = int(prepared.x.shape[0]) if streamed_q else int(quantized.q.shape[-2])
        if not streamed_q:
            packed_q_tiles = (sequence + 127) // 128
            if quantized.q_scale.shape[-1] % packed_q_tiles:
                raise FusedQKVError(
                    'streamed dense Kitchen output received invalid Q scales'
                )
            scales_per_packed_q_tile = (
                quantized.q_scale.shape[-1] // packed_q_tiles
            )

        try:
            for start in range(0, sequence, self.query_chunk_rows):
                stop = min(start + self.query_chunk_rows, sequence)
                if streamed_q:
                    held = create_held_qkv(
                        prepared.module,
                        prepared.x[start:start + 1],
                        prepared.projection_mode,
                    )
                    held.__enter__()
                    try:
                        q = project_q_hnd(
                            held,
                            prepared.x,
                            prepared.rope_freqs,
                            start,
                            stop,
                        )
                    finally:
                        held.__exit__(None, None, None)
                    q_shape = tuple(q.shape)
                    k_shape = (
                        q_shape[0],
                        quantized.k.shape[1],
                        1,
                        q_shape[3],
                    )
                    spec = kitchen.int8_attention_producer_spec(
                        q_shape,
                        k_shape,
                        dtype=q.dtype,
                        device=q.device,
                        cta_k=quantized.cta_k,
                    )
                    samples = q.new_zeros(
                        k_shape[0],
                        k_shape[1],
                        len(spec.k_anchor_positions),
                        k_shape[3],
                    )
                    anchor = kitchen.select_int8_attention_k_anchor(spec, samples)
                    del samples
                    producer = kitchen.create_int8_attention_producer(spec, anchor)
                    kitchen.quantize_int8_attention_q_chunk(
                        producer,
                        q,
                        q_start=0,
                        allow_strided_input=True,
                    )
                    kitchen.quantize_int8_attention_v(producer, q[..., :1, :])
                    q_carrier = kitchen.finalize_int8_attention_producer(producer)
                    chunk_carrier = replace(
                        q_carrier,
                        k=quantized.k,
                        v=quantized.v,
                        k_scale=quantized.k_scale,
                        v_scale=quantized.v_scale,
                        input_dtype=quantized.input_dtype,
                        attention_scale=quantized.attention_scale,
                        cta_k=quantized.cta_k,
                        anchor_indices=quantized.anchor_indices,
                    )
                    q_scale = q_carrier.q_scale
                    del q_carrier
                else:
                    first_scale_tile = start // 128
                    stop_scale_tile = (stop + 127) // 128
                    q = quantized.q[..., start:stop, :].contiguous()
                    q_scale = quantized.q_scale[
                        ...,
                        first_scale_tile * scales_per_packed_q_tile:
                        stop_scale_tile * scales_per_packed_q_tile,
                    ].contiguous()
                    chunk_carrier = replace(quantized, q=q, q_scale=q_scale)
                raw = kitchen.int8_attention_from_prequantized(
                    chunk_carrier,
                    output_layout=OUTPUT_NHD,
                )
                flat = raw.transpose(1, 2).reshape(
                    raw.shape[0],
                    raw.shape[2],
                    module.heads * module.head_dim,
                )
                with diagnostics.stage('attention_out'):
                    output[start:stop].copy_(module.out_proj(flat.squeeze(0)))
                del raw, flat, chunk_carrier, q, q_scale
            return output
        finally:
            prepared.release()
