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
import comfy.quant_ops
from comfy.quant_ops import QuantizedTensor
import comfy_aimdo.model_vbar
import comfy_aimdo.torch as aimdo_torch

from . import diagnostics, native
from .native.int8_attention import quantize_int8_attention_q
from .native.v_staging import BACKEND_NATIVE, TwoPassVCarrier, VStagingError
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
from .runtime.stage_prefetch import (
    abandon_stage_prefetch,
    begin_stage_prefetch,
    release_stage_prefetch,
    wait_stage_prefetch,
    stage_prefetch_enabled,
)

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


@dataclass
class PreparedStreamedSparseKitchen:
    projected: PreparedStreamedKitchenQKV
    route: object
    layer_index: int
    metadata: dict

    def release(self):
        if self.projected is not None:
            self.projected.release_carriers()
            self.projected.x = None
            self.projected.rope_freqs = None
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


def _quantized_component_offsets(source):
    '''Return Comfy gathered-buffer byte offsets for a QuantizedTensor.'''
    inner_tensors, ctx = source.__tensor_flatten__()
    offsets = {}
    offset = 0
    for attr in inner_tensors:
        tensor = getattr(source, attr)
        offsets[attr] = offset
        offset += comfy.memory_management.vram_aligned_size(tensor)
    return offsets, ctx


def _fault_vbar_tensor_slice(linear, source_tensor, byte_offset, device, label):
    '''Fault, populate, and expose one tensor slice from a module AIMDO VBAR.'''
    alloc = getattr(linear, '_v', None)
    if alloc is None:
        raise FusedQKVError(
            'streamed %s-only ConvRot projection requires AIMDO VBAR backing '
            'for an offloaded qkv_proj' % label
        )
    try:
        vbar, base_ptr, alloc_size = alloc
    except (TypeError, ValueError) as exc:
        raise FusedQKVError('qkv_proj AIMDO VBAR allocation is invalid') from exc

    size = int(source_tensor.numel()) * int(source_tensor.element_size())
    byte_offset = int(byte_offset)
    if byte_offset < 0 or byte_offset + size > int(alloc_size):
        raise FusedQKVError(
            'streamed %s-only ConvRot VBAR slice is outside qkv_proj allocation'
            % label
        )
    suballoc = (vbar, int(base_ptr) + byte_offset, size)
    signature = comfy_aimdo.model_vbar.vbar_fault(suballoc)
    if signature is None:
        raise FusedQKVError(
            'streamed %s-only ConvRot VBAR fault ran out of VRAM' % label
        )
    try:
        gathered = aimdo_torch.aimdo_to_tensor(suballoc, device)
        comfy.model_management.cast_to_gathered(
            [source_tensor],
            gathered,
            non_blocking=False,
        )
        view = gathered.view(dtype=source_tensor.dtype).view(source_tensor.shape)
        return suballoc, view
    except Exception:
        comfy_aimdo.model_vbar.vbar_unpin(suballoc)
        raise


@contextmanager
def _convrot_qkv_slice_weight(module, device, index, label):
    '''Hold one third of ConvRot QKV using ranged AIMDO VBAR residency.'''
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
        sliced_scale = scale[start:stop]
    else:
        raise FusedQKVError('ConvRot QKV scale is not scalar or per-output-row')

    # Already-resident weights need no VBAR fault or copy.
    if qdata.device == device and sliced_scale.device == device:
        sliced_params = replace(
            params,
            scale=sliced_scale,
            orig_shape=(inner, hidden),
        )
        yield QuantizedTensor(qdata, source._layout_cls, sliced_params)
        return

    if device.type != 'cuda':
        raise FusedQKVError(
            'streamed %s-only ranged ConvRot projection requires CUDA' % label
        )

    offsets, ctx = _quantized_component_offsets(source)
    scale_attr = ctx.get('tensor_fields', {}).get('scale')
    if '_qdata' not in offsets or scale_attr not in offsets:
        raise FusedQKVError(
            'streamed %s-only ConvRot source has unsupported flattened storage'
            % label
        )

    qdata_row_bytes = int(source._qdata.shape[1]) * int(source._qdata.element_size())
    qdata_offset = int(offsets['_qdata']) + start * qdata_row_bytes

    if scale.ndim == 0:
        scale_offset = int(offsets[scale_attr])
    else:
        if not scale.is_contiguous():
            raise FusedQKVError(
                'streamed %s-only ConvRot scale must be contiguous' % label
            )
        scale_row_bytes = int(scale[0].numel()) * int(scale.element_size())
        scale_offset = int(offsets[scale_attr]) + start * scale_row_bytes

    pinned = []
    qdata_view = None
    scale_view = None
    try:
        qdata_alloc, qdata_view = _fault_vbar_tensor_slice(
            linear, qdata, qdata_offset, device, label
        )
        pinned.append(qdata_alloc)
        scale_alloc, scale_view = _fault_vbar_tensor_slice(
            linear, sliced_scale, scale_offset, device, label
        )
        pinned.append(scale_alloc)
        sliced_params = replace(
            params,
            scale=scale_view,
            orig_shape=(inner, hidden),
        )
        yield QuantizedTensor(qdata_view, source._layout_cls, sliced_params)
    finally:
        qdata_view = None
        scale_view = None
        for alloc in reversed(pinned):
            comfy_aimdo.model_vbar.vbar_unpin(alloc)


def _q_only_convrot_weight(module, device):
    return _convrot_qkv_slice_weight(module, device, 0, 'Q')


def _v_only_convrot_weight(module, device):
    return _convrot_qkv_slice_weight(module, device, 2, 'V')


def _project_q_only_hnd(module, x, rope_freqs, start, end, q_weight):
    rows = x[start:end]
    with diagnostics.stage('streamed_q_only_linear'):
        q = F.linear(rows, q_weight, None)
    count = int(end - start)
    heads = int(module.heads)
    head_dim = int(module.head_dim)
    q = module.q_norm(q.view(count, heads, head_dim))

    if rope_freqs is not None:
        rope = rope_freqs[:, start:end]
        rot = int(rope.shape[-3]) * 2
        q4 = q.unsqueeze(0)
        q_rot = q4[..., :rot].contiguous()
        comfy.quant_ops.ck.apply_rope_split_half1_(q_rot, rope)
        q4[..., :rot].copy_(q_rot)
        del q_rot
        q = q4[0]

    return q.transpose(0, 1).unsqueeze(0)


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
    stage_prefetch=False,
    qkv_prefetch_ticket=None,
):
    '''Prepare streamed K/V carriers without full-sequence BF16 Q or V.'''
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
    v_stager = None
    if staged_v:
        try:
            v_stager = TwoPassVCarrier(spec, backend=BACKEND_NATIVE)
        except VStagingError as exc:
            raise FusedQKVError(
                'streamed ConvRot V staging requires the native V staging '
                'API; rebuild the H3 native library from this revision'
            ) from exc

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
            anchor = kitchen.select_int8_attention_k_anchor(
                spec, samples
            )
        del samples
        producer = kitchen.create_int8_attention_producer(spec, anchor)
        del anchor

        q_summaries = []
        k_summaries = []
        chunk_kwargs = _qk_chunk_kwargs(
            kitchen, strided_qk_input
        )
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
            with diagnostics.stage(
                'streamed_routing_summary_generation'
            ):
                q_summaries.append(
                    _tile_mean(q, int(spec.q_tile))
                )
                k_summaries.append(
                    _tile_mean(k, int(spec.k_tile))
                )

            # The current producer couples Q and K. Repeatedly overwrite
            # the tiny Q scratch while K advances through the real sequence.
            q_stub = _query_stub(q)
            with diagnostics.stage('streamed_k_carrier_pack'):
                kitchen.quantize_int8_attention_qk_chunk(
                    producer,
                    q_stub,
                    k,
                    q_start=0,
                    k_start=start,
                    **chunk_kwargs,
                )
            del q_stub

            if staged_v:
                with diagnostics.stage('streamed_v_amax_update'):
                    v_stager.update(v)
            else:
                if retained_v is None:
                    retained_v = v.new_empty(
                        (1, heads, sequence, head_dim)
                    )
                retained_v[:, :, start:end, :].copy_(v)
            del q, k, v

        if held is not None:
            held.__exit__(None, None, None)
            held = None
        release_stage_prefetch(qkv_prefetch_ticket)
        qkv_prefetch_ticket = None

        if staged_v:
            try:
                with diagnostics.stage('streamed_v_scale_finalize'):
                    v_stager.finalize_scale()
                v_weight_hold = None
                with diagnostics.stage('streamed_v_weight_slice'):
                    v_weight_hold = _v_only_convrot_weight(module, x.device)
                    v_weight = v_weight_hold.__enter__()
                try:
                    for start in range(0, sequence, int(chunk_rows)):
                        end = min(start + int(chunk_rows), sequence)
                        with diagnostics.stage('streamed_v_reprojection'):
                            v = _project_v_only_hnd(
                                module, x, start, end, v_weight
                            )
                        with diagnostics.stage('streamed_v_chunk_pack'):
                            v_stager.quantize(v, start)
                        del v
                finally:
                    v_weight = None
                    if v_weight_hold is not None:
                        v_weight_hold.__exit__(None, None, None)
                v_carrier, v_scale = v_stager.finish()
            except VStagingError as exc:
                raise FusedQKVError('streamed native V staging failed: %s' % exc) from exc
        else:
            with diagnostics.stage('streamed_v_carrier_pack'):
                kitchen.quantize_int8_attention_v(producer, retained_v)
            del retained_v
            retained_v = None
            if producer.v is None or producer.v_scale is None:
                raise FusedQKVError('streamed Kitchen V carrier was not produced')
            v_carrier = producer.v
            v_scale = producer.v_scale
        return PreparedStreamedKitchenQKV(
            x=x,
            rope_freqs=rope_freqs,
            kitchen=kitchen,
            k=producer.k,
            v=v_carrier,
            k_scale=producer.k_scale,
            v_scale=v_scale,
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


def _local_query_carrier(projected, q):
    '''Pack one bounded Q chunk using the native Q-only Kitchen transform.'''
    q_int8, q_scale = quantize_int8_attention_q(q)
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
    if start % Q_TILE:
        raise ValueError(
            'streamed query start must align to %d rows' % Q_TILE
        )
    first = start // Q_TILE
    stop = (end + Q_TILE - 1) // Q_TILE
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
    result = None
    fmt = describe_linear(module.qkv_proj)
    held = None
    q_weight_hold = None
    q_weight = None
    if fmt.convrot_int8_256:
        with diagnostics.stage('streamed_q_weight_slice'):
            q_weight_hold = _q_only_convrot_weight(
                module, projected.x.device
            )
            q_weight = q_weight_hold.__enter__()
    else:
        held = _held_projector(module, projected.x)

    # out_proj is the next stage after the Q-only slice. Fault it while the
    # first local-Q projection / carrier pack / sparse kernel is running.
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
            with diagnostics.stage(
                'streamed_query_carrier_pack'
            ):
                quantized = _local_query_carrier(projected, q)
            del q

            route = _route_chunk(prepared.route, start, end)
            with diagnostics.stage(
                'streamed_sparse_attention_kernel'
            ):
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

            if result is None:
                result = local.new_empty(
                    sequence, local.shape[-1]
                )
            result[start:end].copy_(local)
            del local

        if result is None:
            raise SparseKitchenError(
                'streamed Kitchen received an empty sequence'
            )
        prepared.release()
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


_ORIGINAL_PROJECT = ChunkedKitchenQKVProjector.try_project
_ORIGINAL_PREPARE_PROJECTED = (
    SparseKitchenBackend.prepare_projected
)
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
    import comfy.model_management

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
        q_tile=Q_TILE,
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
        return None
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
