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

from dataclasses import dataclass

import torch

from . import diagnostics, native
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


def run_streamed_kitchen_qkv(
    module,
    x,
    rope_freqs,
    *,
    chunk_rows,
    query_chunk_rows,
    strided_qk_input,
):
    '''Prepare full K/V INT8 and route summaries without a full Q carrier.'''
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

    held = _held_projector(module, x)
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

        retained_v = None
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

            if retained_v is None:
                retained_v = v.new_empty(
                    (1, heads, sequence, head_dim)
                )
            retained_v[:, :, start:end, :].copy_(v)
            del q, k, v

        with diagnostics.stage('streamed_v_carrier_pack'):
            kitchen.quantize_int8_attention_v(
                producer, retained_v
            )
        del retained_v

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
        )
    finally:
        if held is not None:
            held.__exit__(None, None, None)


def _local_query_carrier(projected, q):
    '''Pack one bounded Q chunk using the current exact Kitchen transform.'''
    kitchen = projected.kitchen
    dummy_k = _query_stub(q)
    q_shape = tuple(int(value) for value in q.shape)
    k_shape = tuple(int(value) for value in dummy_k.shape)
    spec = kitchen.int8_attention_producer_spec(
        q_shape,
        k_shape,
        dtype=q.dtype,
        device=q.device,
    )
    anchor = kitchen.Int8AttentionKAnchor(
        values=torch.zeros(
            k_shape[0],
            k_shape[1],
            int(spec.kernel_head_dim),
            dtype=q.dtype,
            device=q.device,
        ),
        indices=torch.full(
            (k_shape[0], k_shape[1]),
            -1,
            dtype=torch.int32,
            device=q.device,
        ),
    )
    producer = kitchen.create_int8_attention_producer(
        spec, anchor
    )
    kitchen.quantize_int8_attention_qk_chunk(
        producer,
        q,
        dummy_k,
        q_start=0,
        k_start=0,
        allow_strided_input=True,
    )
    del dummy_k, anchor
    return native.PrequantizedInt8Attention(
        q=producer.q,
        k=projected.k,
        v=projected.v,
        q_scale=producer.q_scale,
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
    held = _held_projector(module, projected.x)
    try:
        for start in range(0, sequence, query_rows):
            end = min(start + query_rows, sequence)
            with diagnostics.stage('streamed_query_projection'):
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
        return result
    finally:
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
    return run_streamed_kitchen_qkv(
        module,
        x,
        rope_freqs,
        chunk_rows=self.chunk_rows,
        query_chunk_rows=requested_query_chunk_rows(
            transformer_options
        ),
        strided_qk_input=self.strided_qk_input,
    )


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
