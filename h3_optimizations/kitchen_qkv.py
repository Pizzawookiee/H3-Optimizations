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
from .qkv.w4a8 import HeldW4A8QKV, W4A8BindingError


CHUNK_ROWS = 4096
STRIDED_QK_CAPABILITY = 'SUPPORTS_STRIDED_QK_CHUNK'
V_MODE_RETAIN = 'retain'
V_MODE_TWO_PASS = 'two_pass'
V_MODES = (V_MODE_RETAIN, V_MODE_TWO_PASS)
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


@dataclass(frozen=True)
class PreparedChunkedKitchenQKV:
    carrier: object
    q_summary: torch.Tensor | None = None
    k_summary: torch.Tensor | None = None
    output_buffer: torch.Tensor | None = None


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


def _project_v_chunk(module, x, rope_freqs, start, end, projector):
    """One chunk's V, with Q and K discarded.

    The prototype deliberately re-runs the whole QKV projection and throws two
    thirds of it away. That is the wrong shape for production -- V is the last
    `heads * head_dim` rows of the fused weight and a row slice of a
    TensorWise INT8 ConvRot weight carries the same scalar scale, so a V-only
    projection should cost a third of this -- but it is the honest way to
    measure what removing `retained_v` buys before building that slice. The
    time it costs is reported, not hidden.
    """
    _q, _k, v = project_chunk_hnd(
        module, x, rope_freqs, start, end, projector=projector
    )
    del _q, _k
    return v


def run_two_pass_v_kitchen_qkv(
    module,
    x,
    rope_freqs,
    *,
    kitchen,
    held,
    spec,
    chunk_rows,
    routing_summaries,
    routing_q_tile,
    routing_kv_tile,
    v_backend,
    strided_qk_input,
):
    """Produce the carrier without ever holding a full-sequence BF16 V.

    Pass one streams Q and K into their carriers and folds each chunk's V into
    a [B, H, D] maximum. Pass two regenerates V and writes it straight into the
    final INT8 carrier under the finalized scale. Q and K are packed exactly
    once, in pass one, so nothing about their carrier changes.
    """
    from .native.v_staging import TwoPassVCarrier

    with diagnostics.stage('anchor_projection'):
        samples = _project_anchor_samples(
            module, x, rope_freqs, spec.k_anchor_positions, projector=held
        )
    with diagnostics.stage('anchor_selection'):
        anchor = kitchen.select_int8_attention_k_anchor(spec, samples)
    del samples
    with diagnostics.stage('producer_create'):
        producer = kitchen.create_int8_attention_producer(spec, anchor)
    del anchor

    sequence = int(x.shape[0])
    chunk_kwargs = _qk_chunk_kwargs(kitchen, strided_qk_input)
    staging = TwoPassVCarrier(spec, backend=v_backend)
    q_summaries = []
    k_summaries = []
    for start in range(0, sequence, int(chunk_rows)):
        end = min(start + int(chunk_rows), sequence)
        q, k, v = project_chunk_hnd(
            module, x, rope_freqs, start, end, projector=held
        )
        if routing_summaries:
            with diagnostics.stage('routing_summary_generation'):
                q_summaries.append(_tile_mean(q, int(routing_q_tile)))
                k_summaries.append(_tile_mean(k, int(routing_kv_tile)))
        kitchen.quantize_int8_attention_qk_chunk(
            producer, q, k, q_start=start, k_start=start, **chunk_kwargs
        )
        with diagnostics.stage('v_amax_update'):
            staging.update(v)
        del q, k, v

    with diagnostics.stage('v_scale_finalize'):
        staging.finalize_scale()
    for start in range(0, sequence, int(chunk_rows)):
        end = min(start + int(chunk_rows), sequence)
        with diagnostics.stage('v_reprojection'):
            v = _project_v_chunk(module, x, rope_freqs, start, end, held)
        with diagnostics.stage('v_carrier_pack'):
            staging.quantize(v, start)
        del v

    with diagnostics.stage('carrier_finalize'):
        v_int8, v_scale = staging.finish()
        producer.v = v_int8
        producer.v_scale = v_scale
        return PreparedChunkedKitchenQKV(
            kitchen.finalize_int8_attention_producer(producer),
            q_summary=torch.cat(q_summaries, dim=-2) if q_summaries else None,
            k_summary=torch.cat(k_summaries, dim=-2) if k_summaries else None,
        )


def run_chunked_kitchen_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    transformer_options,
    spec,
    chunk_rows=CHUNK_ROWS,
    fp8_projection=False,
    routing_summaries=False,
    routing_q_tile=None,
    routing_kv_tile=None,
    v_mode=V_MODE_RETAIN,
    v_backend=None,
    strided_qk_input=False,
):
    del layer_index, transformer_options
    if v_mode not in V_MODES:
        raise FusedQKVError('unknown V mode %r' % v_mode)
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
        if fp8_projection:
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

        if v_mode == V_MODE_TWO_PASS:
            return run_two_pass_v_kitchen_qkv(
                module,
                x,
                rope_freqs,
                kitchen=kitchen,
                held=held,
                spec=spec,
                chunk_rows=chunk_rows,
                routing_summaries=routing_summaries,
                routing_q_tile=routing_q_tile,
                routing_kv_tile=routing_kv_tile,
                v_backend=v_backend,
                strided_qk_input=strided_qk_input,
            )

        with diagnostics.stage('anchor_projection'):
            samples = _project_anchor_samples(
                module,
                x,
                rope_freqs,
                spec.k_anchor_positions,
                projector=held,
            )
        anchor = kitchen.select_int8_attention_k_anchor(spec, samples)
        del samples
        with diagnostics.stage('producer_create'):
            producer = kitchen.create_int8_attention_producer(spec, anchor)
        del anchor

        sequence = int(x.shape[0])
        chunk_kwargs = _qk_chunk_kwargs(kitchen, strided_qk_input)
        retained_v = None
        q_summaries = []
        k_summaries = []
        for start in range(0, sequence, int(chunk_rows)):
            end = min(start + int(chunk_rows), sequence)
            q, k, v = project_chunk_hnd(
                module,
                x,
                rope_freqs,
                start,
                end,
                projector=held,
            )
            if retained_v is None:
                retained_v = v.new_empty(
                    (1, int(module.heads), sequence, int(module.head_dim))
                )
            if routing_summaries:
                with diagnostics.stage('routing_summary_generation'):
                    q_summaries.append(_tile_mean(q, routing_q_tile))
                    k_summaries.append(_tile_mean(k, routing_kv_tile))
            kitchen.quantize_int8_attention_qk_chunk(
                producer,
                q,
                k,
                q_start=start,
                k_start=start,
                **chunk_kwargs,
            )
            with diagnostics.stage('v_retention_copy'):
                retained_v[:, :, start:end, :].copy_(v)
            del q, k, v

        kitchen.quantize_int8_attention_v(producer, retained_v)
        del retained_v
        with diagnostics.stage('carrier_finalize'):
            return PreparedChunkedKitchenQKV(
                kitchen.finalize_int8_attention_producer(producer),
                q_summary=(
                    torch.cat(q_summaries, dim=-2) if q_summaries else None
                ),
                k_summary=(
                    torch.cat(k_summaries, dim=-2) if k_summaries else None
                ),
            )
    finally:
        if held is not None:
            held.__exit__(None, None, None)


class ChunkedKitchenQKVProjector:
    name = 'chunked_kitchen_qkv'

    def __init__(
        self,
        chunk_rows=CHUNK_ROWS,
        fp8_projection=False,
        routing_summaries=False,
        q_tile=None,
        kv_tile=None,
        v_mode=V_MODE_RETAIN,
        v_backend=None,
        strided_qk_input=False,
        stream_output=False,
    ):
        self.chunk_rows = int(chunk_rows)
        self.fp8_projection = bool(fp8_projection)
        self.routing_summaries = bool(routing_summaries)
        self.q_tile = None if q_tile is None else int(q_tile)
        self.kv_tile = None if kv_tile is None else int(kv_tile)
        if self.q_tile is not None and self.q_tile <= 0:
            raise ValueError('q_tile must be positive')
        if self.kv_tile is not None and self.kv_tile <= 0:
            raise ValueError('kv_tile must be positive')
        if v_mode not in V_MODES:
            raise ValueError('v_mode must be one of %s' % ', '.join(V_MODES))
        self.v_mode = str(v_mode)
        self.v_backend = v_backend
        self.strided_qk_input = bool(strided_qk_input)
        self.stream_output = bool(stream_output)

    @property
    def installation_signature(self):
        return (
            self.name,
            self.chunk_rows,
            self.fp8_projection,
            self.routing_summaries,
            self.q_tile,
            self.kv_tile,
            self.v_mode,
            self.v_backend,
            self.strided_qk_input,
            self.stream_output,
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
        format_ok = (
            getattr(fmt, 'fp8', False) or getattr(fmt, 'plain_float', False)
            if self.fp8_projection
            else (
                getattr(fmt, 'convrot_int8_256', False)
                or getattr(fmt, 'w4a8', False)
                or native_bf16
            )
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

        shape = (1, int(module.heads), int(x.shape[0]), int(module.head_dim))
        try:
            spec_kwargs = {}
            if self.kv_tile is not None:
                spec_kwargs['cta_k'] = self.kv_tile
            spec = kitchen.int8_attention_producer_spec(
                shape,
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
                projected = run_chunked_kitchen_qkv(
                    module,
                    x,
                    rope_freqs,
                    layer_index=layer_index,
                    transformer_options=transformer_options,
                    spec=spec,
                    chunk_rows=self.chunk_rows,
                    fp8_projection=self.fp8_projection,
                    routing_summaries=self.routing_summaries,
                    routing_q_tile=self.q_tile,
                    routing_kv_tile=self.kv_tile,
                    v_mode=self.v_mode,
                    v_backend=self.v_backend,
                    strided_qk_input=self.strided_qk_input,
                )
                if self.stream_output:
                    projected = replace(projected, output_buffer=x)
                return projected
        except (
            FP8BindingError,
            W4A8BindingError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            if (
                not self.fp8_projection
                and not getattr(fmt, 'w4a8', False)
            ):
                raise
            return None


class ChunkedKitchenAttentionBackend:
    name = DENSE_KITCHEN_BACKEND

    @property
    def installation_signature(self):
        return (self.name,)

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
        if not isinstance(projected, PreparedChunkedKitchenQKV):
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
