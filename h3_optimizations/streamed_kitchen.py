'''Explicit low-VRAM streamed-Q execution for native Kitchen sparse attention.

This first implementation intentionally leaves the native ABI unchanged.
Preparation retains full K/V INT8 carriers and routing summaries but never
keeps a full Q INT8 carrier. Execution reprojects Q in bounded chunks, runs
the existing short-Q/full-KV sparse kernel, immediately applies out_proj, and
writes the hidden-size result into its final sequence buffer.

The current producer API couples Q and K quantization. During streamed query
execution we therefore quantize each real Q chunk beside a bounded 128-row
throwaway K chunk. This preserves the exact Kitchen Q carrier transform while
keeping the workaround small. A later speed patch can replace it with a native
Q-only entry point without changing this backend contract.
'''

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
import comfy.memory_management
import comfy.model_management
import comfy_aimdo.torch as aimdo_torch
from comfy_aimdo.vram_buffer import VRAMBuffer

import comfy.quant_ops
from comfy.quant_ops import QuantizedTensor

from . import diagnostics, native
from .native.int8_attention import quantize_int8_attention_q
from .attention.sparse.config import resolve_video_budget
from .attention.sparse.kitchen_sparse import (
    OUTPUT_HND,
    Q_TILE,
    SparseKitchenBackend,
    SparseKitchenError,
    route_metadata,
    snapshot_for,
)
from .attention.sparse.router import SparseRouterError
from .kitchen_qkv import (
    H3_ATTENTION_BACKEND_KEY,
    ChunkedKitchenQKVProjector,
    FusedQKVError,
    _project_anchor_samples,
    _qk_chunk_kwargs,
    _tile_mean,
    resolve_kitchen,
)
from .mlp_sharing.route import router_kwargs as _route_kwargs
from .qkv.chunked import project_chunk_hnd
from .qkv.formats import describe_linear
from .qkv.w4a8 import HeldW4A8QKV

def abandon_stage_prefetch(_ticket):
    return None


def begin_stage_prefetch(*_args, **_kwargs):
    return None


def release_stage_prefetch(_ticket):
    return None


def wait_stage_prefetch(_ticket):
    return None


def stage_prefetch_enabled(_transformer_options):
    return False

ATTENTION_MEMORY_MODE_KEY = 'h3_optimizations_attention_memory_mode'
QUERY_CHUNK_ROWS_KEY = 'h3_optimizations_query_chunk_rows'
ATTENTION_MEMORY_STANDARD = 'standard'
ATTENTION_MEMORY_STREAMED = 'streamed'
DEFAULT_QUERY_CHUNK_ROWS = 4096
QUERY_CHUNK_ALIGNMENT = Q_TILE
SPARSE_KITCHEN_BACKEND = 'sparse_kitchen_int8'


@dataclass
class PreparedStreamedKitchenQKV:
    x: torch.Tensor
    rope_freqs: torch.Tensor | None
    kitchen: object
    k: torch.Tensor
    v: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    q_summary: torch.Tensor | None
    k_summary: torch.Tensor | None
    original_head_dim: int
    input_dtype: torch.dtype
    attention_scale: float
    cta_k: int
    sequence: int
    query_chunk_rows: int
    stage_prefetch: bool = False

    def release_summaries(self):
        self.q_summary = None
        self.k_summary = None

    def release_carriers(self):
        self.k = None
        self.v = None
        self.k_scale = None
        self.v_scale = None

    def release(self):
        self.release_carriers()
        self.release_summaries()
        self.x = None
        self.rope_freqs = None


@dataclass
class PreparedStreamedSparseKitchen:
    projected: PreparedStreamedKitchenQKV
    route: object
    layer_index: int
    metadata: dict

    def release(self):
        if self.projected is not None:
            self.projected.release()
        self.projected = None
        self.route = None


def normalize_query_chunk_rows(value):
    rows = int(value)
    if rows < QUERY_CHUNK_ALIGNMENT:
        raise ValueError(
            'query chunk rows must be at least %d'
            % QUERY_CHUNK_ALIGNMENT
        )
    if rows % QUERY_CHUNK_ALIGNMENT:
        raise ValueError(
            'query chunk rows must be a multiple of %d'
            % QUERY_CHUNK_ALIGNMENT
        )
    return rows


def requested_memory_mode(transformer_options):
    return (transformer_options or {}).get(
        ATTENTION_MEMORY_MODE_KEY,
        ATTENTION_MEMORY_STANDARD,
    )


def requested_query_chunk_rows(transformer_options):
    return normalize_query_chunk_rows(
        (transformer_options or {}).get(
            QUERY_CHUNK_ROWS_KEY,
            DEFAULT_QUERY_CHUNK_ROWS,
        )
    )


def _query_stub(q, rows=Q_TILE):
    if q.shape[-2] >= rows:
        return q[..., :rows, :]
    stub = q.new_zeros(*q.shape[:-2], rows, q.shape[-1])
    stub[..., : q.shape[-2], :].copy_(q)
    return stub


def _held_projector(module, x):
    fmt = describe_linear(module.qkv_proj)
    if fmt.w4a8:
        held = HeldW4A8QKV(module, x[:1])
        held.__enter__()
        return held
    if fmt.convrot_int8_256:
        return None
    raise FusedQKVError(
        'streamed Kitchen currently requires ConvRot-256 TensorWise INT8 '
        'or W4A8 QKV; got %s' % fmt.label
    )


_STREAMED_QKV_WORKSPACES = {}
_STREAMED_QKV_WORKSPACE_MIN_BYTES = 64 * 1024 * 1024


def _workspace_device_index(device):
    if device.type != 'cuda':
        raise FusedQKVError('streamed ConvRot weight workspace requires CUDA')
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _streamed_weight_workspace(device, required_size):
    """Return one reusable AIMDO VRAMBuffer-backed gathered byte tensor."""
    required_size = int(required_size)
    device_index = _workspace_device_index(device)
    entry = _STREAMED_QKV_WORKSPACES.get(device_index)
    if entry is None or int(entry.max_size) < required_size:
        max_size = max(
            int(_STREAMED_QKV_WORKSPACE_MIN_BYTES),
            required_size,
        )
        entry = VRAMBuffer(max_size, device_index)
        _STREAMED_QKV_WORKSPACES[device_index] = entry
    alloc = entry.get(required_size)
    return aimdo_torch.aimdo_to_tensor(alloc, device)


def _convrot_slice_source(module, index, label):
    linear = module.qkv_proj
    if getattr(linear, 'weight_function', None):
        raise FusedQKVError(
            'streamed %s-only ConvRot projection does not support patched '
            'qkv_proj weights' % label
        )
    source = linear.weight
    if (
        not isinstance(source, QuantizedTensor)
        or getattr(source, '_layout_cls', None) != 'TensorWiseINT8Layout'
    ):
        raise FusedQKVError(
            'streamed %s-only projection requires TensorWiseINT8Layout source'
            % label
        )
    params = source._params
    if (
        getattr(params, 'transposed', False)
        or not getattr(params, 'convrot', False)
        or int(getattr(params, 'convrot_groupsize', 0)) != 256
    ):
        raise FusedQKVError(
            'streamed %s-only projection requires non-transposed ConvRot-256'
            % label
        )

    inner = int(module.heads) * int(module.head_dim)
    hidden = int(source.shape[-1])
    if int(source.shape[0]) != inner * 3:
        raise FusedQKVError(
            'fused QKV source has %d rows; expected %d'
            % (int(source.shape[0]), inner * 3)
        )

    start = int(index) * inner
    stop = start + inner
    if not source._qdata.is_contiguous():
        raise FusedQKVError(
            'streamed %s-only ConvRot qdata must be contiguous' % label
        )
    qdata = source._qdata[start:stop]

    scale = params.scale
    if scale.ndim == 0:
        sliced_scale = scale
    elif int(scale.shape[0]) == int(source.shape[0]):
        if not scale.is_contiguous():
            raise FusedQKVError(
                'streamed %s-only ConvRot scale must be contiguous' % label
            )
        sliced_scale = scale[start:stop]
    else:
        raise FusedQKVError(
            'ConvRot QKV scale is not scalar or per-output-row'
        )

    return source, params, inner, hidden, qdata, sliced_scale


@contextmanager
def _convrot_qkv_slice_weight(module, device, index, label):
    """Stage one packed Q/K/V third into one reusable AIMDO VRAMBuffer."""
    (
        source,
        params,
        inner,
        hidden,
        source_qdata,
        source_scale,
    ) = _convrot_slice_source(module, index, label)

    geometry = [source_qdata, source_scale]
    required_size = int(comfy.memory_management.vram_aligned_size(geometry))
    gathered = _streamed_weight_workspace(device, required_size)

    with diagnostics.stage('streamed_%s_weight_transfer' % label.lower()):
        comfy.model_management.cast_to_gathered(
            geometry,
            gathered,
            non_blocking=False,
        )

    qdata_view, scale_view = comfy.memory_management.interpret_gathered_like(
        geometry,
        gathered,
    )

    sliced_params = replace(
        params,
        scale=scale_view,
        orig_shape=(inner, hidden),
    )
    weight = QuantizedTensor(
        qdata_view,
        source._layout_cls,
        sliced_params,
    )
    try:
        yield weight
    finally:
        # Views become invalid as soon as another Q/K/V slice overwrites the
        # single shared workspace, so never retain them beyond this context.
        weight = None
        qdata_view = None
        scale_view = None
        gathered = None


def _q_only_convrot_weight(module, device):
    return _convrot_qkv_slice_weight(module, device, 0, 'Q')


def _k_only_convrot_weight(module, device):
    return _convrot_qkv_slice_weight(module, device, 1, 'K')


def _v_only_convrot_weight(module, device):
    return _convrot_qkv_slice_weight(module, device, 2, 'V')


def _apply_single_rope(x, rope_freqs):
    if rope_freqs is None:
        return x
    rot = int(rope_freqs.shape[-3]) * 2
    x4 = x.unsqueeze(0)
    x_rot = x4[..., :rot].contiguous()
    comfy.quant_ops.ck.apply_rope_split_half1_(x_rot, rope_freqs)
    x4[..., :rot].copy_(x_rot)
    del x_rot
    return x4[0]


def _project_q_only_hnd(module, x, rope_freqs, start, end, q_weight):
    rows = x[start:end]
    with diagnostics.stage('streamed_q_only_linear'):
        q = F.linear(rows, q_weight, None)
    count = int(end - start)
    heads = int(module.heads)
    head_dim = int(module.head_dim)
    q = module.q_norm(q.view(count, heads, head_dim))
    if rope_freqs is not None:
        q = _apply_single_rope(q, rope_freqs[:, start:end])
    return q.transpose(0, 1).unsqueeze(0)


def _project_k_only_hnd(module, x, rope_freqs, start, end, k_weight):
    rows = x[start:end]
    with diagnostics.stage('streamed_k_only_linear'):
        k = F.linear(rows, k_weight, None)
    count = int(end - start)
    heads = int(module.heads)
    head_dim = int(module.head_dim)
    k = module.k_norm(k.view(count, heads, head_dim))
    if rope_freqs is not None:
        k = _apply_single_rope(k, rope_freqs[:, start:end])
    return k.transpose(0, 1).unsqueeze(0)


def _project_k_anchor_rows(module, x, rope_freqs, positions, k_weight):
    rows = torch.tensor(positions, dtype=torch.int64, device=x.device)
    sample_x = x.index_select(0, rows)
    with diagnostics.stage('streamed_k_anchor_linear'):
        k = F.linear(sample_x, k_weight, None)
    count = int(sample_x.shape[0])
    heads = int(module.heads)
    head_dim = int(module.head_dim)
    k = module.k_norm(k.view(count, heads, head_dim))
    if rope_freqs is not None:
        sample_rope = rope_freqs.index_select(1, rows)
        k = _apply_single_rope(k, sample_rope)
    return k.transpose(0, 1).unsqueeze(0)


def _project_v_only_hnd(module, x, start, end, v_weight):
    rows = x[start:end]
    with diagnostics.stage('streamed_v_only_linear'):
        v = F.linear(rows, v_weight, None)
    count = int(end - start)
    heads = int(module.heads)
    head_dim = int(module.head_dim)
    return v.view(count, heads, head_dim).transpose(0, 1).unsqueeze(0)


def run_streamed_kitchen_qkv(
    module,
    x,
    rope_freqs,
    *,
    chunk_rows,
    query_chunk_rows,
    strided_qk_input,
    routing_q_tile,
    routing_kv_tile,
    stage_prefetch=False,
    qkv_prefetch_ticket=None,
):
    """Prepare streamed carriers with a one-third reusable ConvRot workspace."""
    kitchen = resolve_kitchen(x.device)
    if kitchen is None:
        raise FusedQKVError('no INT8 attention producer is available')
    if kitchen is not native:
        raise FusedQKVError(
            'streamed Kitchen requires the vendored native producer in '
            'this first implementation'
        )

    sequence = int(x.shape[0])
    heads = int(module.heads)
    head_dim = int(module.head_dim)
    full_shape = (1, heads, sequence, head_dim)
    stub_shape = (1, heads, Q_TILE, head_dim)
    try:
        spec = kitchen.int8_attention_producer_spec(
            stub_shape,
            full_shape,
            dtype=x.dtype,
            device=x.device,
            cta_k=int(routing_kv_tile),
        )
    except kitchen.Int8AttentionProducerUnavailableError as exc:
        raise FusedQKVError(
            'native Kitchen producer is unavailable'
        ) from exc
    if int(chunk_rows) % int(spec.sequence_alignment):
        raise FusedQKVError(
            'QKV prep chunk rows %d are not aligned to producer '
            'requirement %d'
            % (int(chunk_rows), int(spec.sequence_alignment))
        )

    fmt = describe_linear(module.qkv_proj)
    staged_v = bool(fmt.convrot_int8_256)

    # ConvRot streamed mode deliberately bypasses full-qkv model-VBAR
    # prefetch. If an older caller handed us a ticket, release it before the
    # dedicated one-third workspace starts consuming VRAM.
    if staged_v:
        release_stage_prefetch(qkv_prefetch_ticket)
        qkv_prefetch_ticket = None

    if not staged_v:
        # Preserve the existing W4A8 behavior. Its held projector owns the
        # fused projection and this patch only specializes ConvRot TensorWise.
        held = _held_projector(module, x)
        retained_v = None
        try:
            with diagnostics.stage('streamed_anchor_projection'):
                samples = _project_anchor_samples(
                    module,
                    x,
                    rope_freqs,
                    spec.k_anchor_positions,
                    projector=held,
                )
            with diagnostics.stage('streamed_anchor_selection'):
                anchor = kitchen.select_int8_attention_k_anchor(spec, samples)
            del samples
            producer = kitchen.create_int8_attention_producer(spec, anchor)
            del anchor

            q_summaries = []
            k_summaries = []
            chunk_kwargs = _qk_chunk_kwargs(kitchen, strided_qk_input)
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
                q_summaries.append(_tile_mean(q, int(routing_q_tile)))
                k_summaries.append(_tile_mean(k, int(routing_kv_tile)))
                q_stub = _query_stub(q)
                kitchen.quantize_int8_attention_qk_chunk(
                    producer,
                    q_stub,
                    k,
                    q_start=0,
                    k_start=start,
                    **chunk_kwargs,
                )
                del q_stub
                if retained_v is None:
                    retained_v = v.new_empty(
                        (1, heads, sequence, head_dim)
                    )
                retained_v[:, :, start:end, :].copy_(v)
                del q, k, v

            with diagnostics.stage('streamed_v_carrier_pack'):
                kitchen.quantize_int8_attention_v(producer, retained_v)
            del retained_v
            retained_v = None
            if producer.v is None or producer.v_scale is None:
                raise FusedQKVError(
                    'streamed Kitchen V carrier was not produced'
                )
            return PreparedStreamedKitchenQKV(
                x=x,
                rope_freqs=rope_freqs,
                kitchen=kitchen,
                k=producer.k,
                v=producer.v,
                k_scale=producer.k_scale,
                v_scale=producer.v_scale,
                q_summary=torch.cat(q_summaries, dim=-2),
                k_summary=torch.cat(k_summaries, dim=-2),
                original_head_dim=int(spec.original_head_dim),
                input_dtype=spec.input_dtype,
                attention_scale=float(spec.attention_scale),
                cta_k=int(spec.cta_k),
                sequence=sequence,
                query_chunk_rows=normalize_query_chunk_rows(
                    query_chunk_rows
                ),
                stage_prefetch=bool(stage_prefetch),
            )
        finally:
            retained_v = None
            release_stage_prefetch(qkv_prefetch_ticket)
            if held is not None:
                held.__exit__(None, None, None)

    q_summaries = []
    k_summaries = []
    q_stub = None
    chunk_kwargs = _qk_chunk_kwargs(kitchen, strided_qk_input)

    # Q pass: routing summaries only. No full-sequence Q INT8 carrier.
    with _q_only_convrot_weight(module, x.device) as q_weight:
        for start in range(0, sequence, int(chunk_rows)):
            end = min(start + int(chunk_rows), sequence)
            q = _project_q_only_hnd(module, x, rope_freqs, start, end, q_weight)
            with diagnostics.stage('streamed_routing_q_summary_generation'):
                q_summaries.append(_tile_mean(q, int(routing_q_tile)))
            if q_stub is None:
                q_stub = _query_stub(q).clone()
            del q

    # K pass: one-third weight workspace + persistent full K carrier.
    with _k_only_convrot_weight(module, x.device) as k_weight:
        with diagnostics.stage('streamed_anchor_projection'):
            samples = _project_k_anchor_rows(
                module, x, rope_freqs, spec.k_anchor_positions, k_weight
            )
        with diagnostics.stage('streamed_anchor_selection'):
            anchor = kitchen.select_int8_attention_k_anchor(spec, samples)
        del samples
        producer = kitchen.create_int8_attention_producer(spec, anchor)
        del anchor

        for start in range(0, sequence, int(chunk_rows)):
            end = min(start + int(chunk_rows), sequence)
            k = _project_k_only_hnd(module, x, rope_freqs, start, end, k_weight)
            with diagnostics.stage('streamed_routing_k_summary_generation'):
                k_summaries.append(_tile_mean(k, int(routing_kv_tile)))
            kitchen.quantize_int8_attention_qk_chunk(
                producer, q_stub, k, q_start=0, k_start=start, **chunk_kwargs
            )
            del k
    del q_stub

    # V pass: exact one-pass V quantization, but with only the V weight third staged.
    retained_v = None
    with _v_only_convrot_weight(module, x.device) as v_weight:
        for start in range(0, sequence, int(chunk_rows)):
            end = min(start + int(chunk_rows), sequence)
            v = _project_v_only_hnd(module, x, start, end, v_weight)
            if retained_v is None:
                retained_v = v.new_empty(1, heads, sequence, head_dim)
            retained_v[..., start:end, :].copy_(v)
            del v

    kitchen.quantize_int8_attention_v(producer, retained_v)
    del retained_v

    return PreparedStreamedKitchenQKV(
        x=x,
        rope_freqs=rope_freqs,
        kitchen=kitchen,
        k=producer.k,
        v=producer.v,
        k_scale=producer.k_scale,
        v_scale=producer.v_scale,
        q_summary=torch.cat(q_summaries, dim=-2),
        k_summary=torch.cat(k_summaries, dim=-2),
        original_head_dim=int(spec.original_head_dim),
        input_dtype=spec.input_dtype,
        attention_scale=float(spec.attention_scale),
        cta_k=int(spec.cta_k),
        sequence=sequence,
        query_chunk_rows=normalize_query_chunk_rows(query_chunk_rows),
        stage_prefetch=bool(stage_prefetch),
    )


def _local_query_carrier(projected, q):
    '''Pack one bounded Q chunk using the native Q-only Kitchen transform.'''
    q_int8, q_scale = quantize_int8_attention_q(
        q,
        full_k_length=projected.sequence,
    )
    return native.PrequantizedInt8Attention(
        q=q_int8,
        k=projected.k,
        v=projected.v,
        q_scale=q_scale,
        k_scale=projected.k_scale,
        v_scale=projected.v_scale,
        original_head_dim=projected.original_head_dim,
        input_dtype=projected.input_dtype,
        attention_scale=projected.attention_scale,
        cta_k=projected.cta_k,
        anchor_indices=None,
    )


def _route_chunk(route, start, end):
    route_q_tile = int(route.q_tile)
    if start % route_q_tile:
        raise ValueError(
            'streamed query start must align to %d rows' % route_q_tile
        )
    first = start // route_q_tile
    stop = (end + route_q_tile - 1) // route_q_tile
    return type(route)(
        indices=route.indices[..., first:stop, :].contiguous(),
        counts=route.counts[..., first:stop].contiguous(),
        q_tile=route.q_tile,
        kv_tile=route.kv_tile,
        encoding=route.encoding,
    )


def execute_streamed_projected(backend, module, prepared):
    projected = prepared.projected
    sequence = int(projected.sequence)
    query_rows = int(projected.query_chunk_rows)
    # Reuse the full attention input buffer as the full attention output.
    # For a given query chunk, projected.x[start:end] has no remaining uses
    # after Q projection/packing. K/V and routing state are already persistent.
    # This removes the second full [sequence, hidden] BF16 allocation.
    result = projected.x
    if (
        result.ndim != 2
        or int(result.shape[0]) != sequence
    ):
        raise SparseKitchenError(
            'streamed in-place attention output requires projected.x shape '
            '[sequence, hidden]; got %r' % (tuple(result.shape),)
        )
    if not result.is_contiguous():
        raise SparseKitchenError(
            'streamed in-place attention output requires contiguous projected.x'
        )

    fmt = describe_linear(module.qkv_proj)
    held = None
    q_weight_hold = None
    q_weight = None

    if fmt.convrot_int8_256:
        with diagnostics.stage('streamed_q_weight_workspace_stage'):
            q_weight_hold = _q_only_convrot_weight(
                module, projected.x.device
            )
            q_weight = q_weight_hold.__enter__()
    else:
        held = _held_projector(module, projected.x)

    # out_proj remains on the normal stage-aware Comfy/AIMDO path. The
    # streamed Q workspace is a separate VRAMBuffer, so no shared cast-buffer
    # alias exists between the two stages.
    out_ticket = begin_stage_prefetch(
        module.out_proj,
        projected.x.device,
        enabled=projected.stage_prefetch,
    )
    try:
        for start in range(0, sequence, query_rows):
            end = min(start + query_rows, sequence)
            with diagnostics.stage('streamed_query_projection'):
                if q_weight is not None:
                    q = _project_q_only_hnd(
                        module,
                        projected.x,
                        projected.rope_freqs,
                        start,
                        end,
                        q_weight,
                    )
                else:
                    q, k_unused, v_unused = project_chunk_hnd(
                        module,
                        projected.x,
                        projected.rope_freqs,
                        start,
                        end,
                        projector=held,
                    )
                    del k_unused, v_unused

            with diagnostics.stage('streamed_query_carrier_pack'):
                quantized = _local_query_carrier(projected, q)
            del q

            route = _route_chunk(prepared.route, start, end)
            with diagnostics.stage('streamed_sparse_attention_kernel'):
                if backend.output_layout == OUTPUT_HND:
                    raw = (
                        backend.executor.kitchen
                        .block_sparse_int8_attention_from_prequantized(
                            quantized,
                            route,
                        )
                    )
                else:
                    raw = (
                        backend.executor.kitchen
                        .block_sparse_int8_attention_from_prequantized(
                            quantized,
                            route,
                            output_layout=backend.output_layout,
                        )
                    )
            del quantized, route

            out = raw.transpose(1, 2).reshape(
                raw.shape[0],
                raw.shape[2],
                int(module.heads) * int(module.head_dim),
            )
            del raw
            if out_ticket is not None:
                wait_stage_prefetch(out_ticket)
            with diagnostics.stage('streamed_attention_out'):
                local = module.out_proj(out.squeeze(0))
            del out

            if int(local.shape[0]) != int(end - start):
                raise SparseKitchenError(
                    'streamed out_proj returned %d rows for query chunk %d:%d'
                    % (int(local.shape[0]), int(start), int(end))
                )
            if int(local.shape[-1]) != int(result.shape[-1]):
                raise SparseKitchenError(
                    'streamed out_proj hidden size %d does not match reusable '
                    'attention buffer hidden size %d'
                    % (
                        int(local.shape[-1]),
                        int(result.shape[-1]),
                    )
                )
            if local.dtype != result.dtype:
                raise SparseKitchenError(
                    'streamed out_proj dtype %s does not match reusable '
                    'attention buffer dtype %s'
                    % (local.dtype, result.dtype)
                )

            # Q for this range was projected and quantized above, so the
            # original input rows are dead. Overwrite them with final out_proj
            # rows and progressively turn projected.x into attention output.
            result[start:end].copy_(local)
            del local
        # Drop carriers/summaries/rope while preserving the aliased result
        # tensor until the caller receives it. Prepared ownership is cleared
        # manually here because PreparedStreamedKitchenQKV.release() also sets
        # projected.x = None.
        projected.release_carriers()
        projected.release_summaries()
        projected.rope_freqs = None
        prepared.projected = None
        prepared.route = None
        release_stage_prefetch(out_ticket)
        out_ticket = None
        return result
    finally:
        abandon_stage_prefetch(out_ticket)
        q_weight = None
        if q_weight_hold is not None:
            q_weight_hold.__exit__(None, None, None)
        if held is not None:
            held.__exit__(None, None, None)
        if prepared.projected is not None:
            prepared.release()


_ORIGINAL_PROJECT = ChunkedKitchenQKVProjector.try_project
_ORIGINAL_PREPARE_PROJECTED = (
    SparseKitchenBackend.prepare_projected
)
_ORIGINAL_EXECUTE_PROJECTED = SparseKitchenBackend.execute_projected
_INSTALLED = False


def _stream_aware_project(
    self,
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    transformer_options,
):
    mode = requested_memory_mode(transformer_options)
    backend_name = (transformer_options or {}).get(
        H3_ATTENTION_BACKEND_KEY
    )
    if (
        mode != ATTENTION_MEMORY_STREAMED
        or backend_name != SPARSE_KITCHEN_BACKEND
    ):
        return _ORIGINAL_PROJECT(
            self,
            module,
            x,
            rope_freqs,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )
    if not self.routing_summaries:
        raise FusedQKVError(
            'streamed Kitchen requires routing summaries'
        )
    if self.fp8_projection:
        raise FusedQKVError(
            'streamed Kitchen FP8 projection is not implemented yet'
        )

    if comfy.model_management.in_training:
        raise FusedQKVError('streamed Kitchen is inference-only')
    if x.ndim != 2 or not x.is_cuda:
        raise FusedQKVError(
            'streamed Kitchen requires rank-2 CUDA activations'
        )

    qkv_ticket = getattr(
        module,
        '_h3_optimizations_qkv_prefetch_ticket',
        None,
    )
    if hasattr(module, '_h3_optimizations_qkv_prefetch_ticket'):
        delattr(module, '_h3_optimizations_qkv_prefetch_ticket')
    try:
        wait_stage_prefetch(qkv_ticket)
        return run_streamed_kitchen_qkv(
            module,
            x,
            rope_freqs,
            chunk_rows=self.chunk_rows,
            query_chunk_rows=requested_query_chunk_rows(
                transformer_options
            ),
            strided_qk_input=self.strided_qk_input,
            routing_q_tile=(64 if self.q_tile is None else int(self.q_tile)),
            routing_kv_tile=(64 if self.kv_tile is None else int(self.kv_tile)),
            stage_prefetch=stage_prefetch_enabled(transformer_options),
            qkv_prefetch_ticket=qkv_ticket,
        )
    finally:
        release_stage_prefetch(qkv_ticket)


def _stream_aware_prepare_projected(
    self,
    projected,
    *,
    layer_index,
    transformer_options,
):
    if not isinstance(projected, PreparedStreamedKitchenQKV):
        return _ORIGINAL_PREPARE_PROJECTED(
            self,
            projected,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )
    if (
        projected.q_summary is None
        or projected.k_summary is None
    ):
        raise SparseKitchenError(
            'streamed Kitchen carrier has no routing summaries'
        )

    snapshot = snapshot_for(
        transformer_options, projected.sequence
    )
    video_budget = resolve_video_budget(
        self.config,
        snapshot.step_index,
        snapshot.total_steps,
        layer_index,
    )
    try:
        with diagnostics.stage('streamed_sparse_route'):
            lut, valid_block_num, mask_metadata = (
                self.router.build_lut_from_summaries(
                    projected.q_summary,
                    projected.k_summary,
                    snapshot.layout,
                    video_budget,
                    **_route_kwargs(
                        transformer_options, layer_index
                    ),
                )
            )
    except SparseRouterError as exc:
        raise SparseKitchenError(
            'streamed sparse routing failed: %s' % exc
        ) from exc

    route = self.executor.kitchen.BlockSparseRoute(
        indices=lut,
        counts=valid_block_num,
        q_tile=self.executor.q_tile,
        kv_tile=self.executor.kv_tile,
        encoding='delta',
    )
    metadata = route_metadata(
        mask_metadata,
        layer_index,
        projected.q_summary.shape[1],
    )
    projected.release_summaries()
    return PreparedStreamedSparseKitchen(
        projected=projected,
        route=route,
        layer_index=int(layer_index),
        metadata=metadata,
    )


def _execute_projected(self, module, prepared):
    if not isinstance(prepared, PreparedStreamedSparseKitchen):
        return _ORIGINAL_EXECUTE_PROJECTED(self, module, prepared)
    return execute_streamed_projected(
        self, module, prepared
    )


def install_streaming_support():
    '''Install the branch-local streamed contract on package-owned classes.'''
    global _INSTALLED
    if _INSTALLED:
        return
    ChunkedKitchenQKVProjector.try_project = (
        _stream_aware_project
    )
    SparseKitchenBackend.prepare_projected = (
        _stream_aware_prepare_projected
    )
    SparseKitchenBackend.execute_projected = _execute_projected
    _INSTALLED = True
