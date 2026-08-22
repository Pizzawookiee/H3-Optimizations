"""Streamed-query Sparse Sage for low-VRAM MiniMax H3 inference.

Keeps full K and final FP8 V carriers, but does not materialize a full Q
carrier or a full attention output. Query tiles are reprojected and
quantized in bounded chunks and passed to SpargeAttn using its native
qo_len != kv_len support.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ...qkv.chunked import project_chunk_hnd
from .backend import HybridSparseBackend
from .chunked_qkv import pack_sparse_qk_chunk_into
from .config import resolve_video_budget
from .fp8_v_stream import (
    allocate_v_carrier,
    finalize_v_scale,
    pack_v_chunk,
    update_v_amax,
)
from .fused_qkv import FusedQKVError, HEAD_DIM, sparse_fused_qkv_contract_mismatch
from .router import SparseRouterError
from .sparse_sage import SparseSageError
from ...mlp_sharing.route import router_kwargs as _route_kwargs


DEFAULT_PROJECT_CHUNK_ROWS = 1024
DEFAULT_QUERY_CHUNK_ROWS = 1024


@dataclass
class StreamedSparseSageQKV:
    module: object
    x: torch.Tensor
    rope_freqs: torch.Tensor | None
    k_int8: torch.Tensor
    k_scale: torch.Tensor
    v_carrier: torch.Tensor
    v_scale: torch.Tensor
    q_summary: torch.Tensor
    k_summary: torch.Tensor
    output_dtype: torch.dtype
    sequence: int
    heads: int
    head_dim: int
    layer_index: int
    project_chunk_rows: int
    query_chunk_rows: int


@dataclass
class PreparedStreamedSparseSage:
    projected: StreamedSparseSageQKV
    lut: torch.Tensor
    valid_block_num: torch.Tensor
    metadata: dict


@dataclass
class PreparedStreamedHybrid:
    sparse: PreparedStreamedSparseSage


def _validate_chunk_rows(value, q_tile, kv_tile, *, name):
    value = int(value)
    alignment = math.lcm(int(q_tile), int(kv_tile))
    if value <= 0 or value % alignment:
        raise SparseSageError(
            "%s must be a positive multiple of %d" % (name, alignment)
        )
    return value


def _validate_projected(projected, spec):
    if not isinstance(projected, StreamedSparseSageQKV):
        raise SparseSageError("expected StreamedSparseSageQKV")
    mismatch = sparse_fused_qkv_contract_mismatch(spec)
    if mismatch is not None:
        raise SparseSageError(
            "streamed Sparse Sage carrier contract mismatch: %s" % mismatch
        )
    if not spec.uses_fp8_v:
        raise SparseSageError("streamed Sparse Sage currently requires FP8 V")
    sequence = int(projected.sequence)
    heads = int(projected.heads)
    head_dim = int(projected.head_dim)
    if sequence <= 0 or heads <= 0 or head_dim != HEAD_DIM:
        raise SparseSageError("streamed Sparse Sage metadata is invalid")
    if (
        not projected.x.is_cuda
        or projected.x.dtype != torch.bfloat16
        or projected.x.ndim != 2
        or int(projected.x.shape[0]) != sequence
    ):
        raise SparseSageError(
            "streamed Sparse Sage requires rank-2 CUDA BF16 attention input"
        )

    q_blocks = (sequence + int(spec.q_tile) - 1) // int(spec.q_tile)
    k_blocks = (sequence + int(spec.kv_tile) - 1) // int(spec.kv_tile)
    padded = (sequence + 127) // 128 * 128
    device = projected.x.device

    contracts = (
        (
            "k_int8",
            projected.k_int8,
            (1, heads, sequence, head_dim),
            torch.int8,
        ),
        (
            "k_scale",
            projected.k_scale,
            (1, heads, k_blocks),
            torch.float32,
        ),
        (
            "v_carrier",
            projected.v_carrier,
            (1, heads, head_dim, padded),
            torch.float8_e4m3fn,
        ),
        (
            "v_scale",
            projected.v_scale,
            (1, heads, head_dim),
            torch.float32,
        ),
        (
            "q_summary",
            projected.q_summary,
            (1, heads, q_blocks, head_dim),
            projected.output_dtype,
        ),
        (
            "k_summary",
            projected.k_summary,
            (1, heads, k_blocks, head_dim),
            projected.output_dtype,
        ),
    )
    for name, tensor, shape, dtype in contracts:
        if tuple(tensor.shape) != tuple(shape):
            raise SparseSageError(
                "%s shape %s does not match %s"
                % (name, tuple(tensor.shape), tuple(shape))
            )
        if tensor.dtype != dtype:
            raise SparseSageError(
                "%s dtype %s does not match %s" % (name, tensor.dtype, dtype)
            )
        if tensor.device != device or not tensor.is_contiguous():
            raise SparseSageError(
                "%s must be contiguous on the attention device" % name
            )

    _validate_chunk_rows(
        projected.project_chunk_rows,
        spec.q_tile,
        spec.kv_tile,
        name="project_chunk_rows",
    )
    _validate_chunk_rows(
        projected.query_chunk_rows,
        spec.q_tile,
        spec.kv_tile,
        name="query_chunk_rows",
    )
    return projected


def _collect_q_summary_chunk(q, destination, *, row_start, q_tile):
    """Pack only a temporary local Q carrier to obtain exact tile summaries."""
    rows = int(q.shape[-2])
    heads = int(q.shape[1])
    blocks = (rows + int(q_tile) - 1) // int(q_tile)

    q_tmp = torch.empty(
        (1, heads, rows, HEAD_DIM),
        dtype=torch.int8,
        device=q.device,
    )
    scale_tmp = torch.empty(
        (1, heads, blocks),
        dtype=torch.float32,
        device=q.device,
    )
    summary_tmp = torch.empty(
        (1, heads, blocks, HEAD_DIM),
        dtype=q.dtype,
        device=q.device,
    )
    pack_sparse_qk_chunk_into(
        q,
        q_tmp,
        scale_tmp,
        summary_tmp,
        row_start=0,
        block_size=q_tile,
    )
    block_start = int(row_start) // int(q_tile)
    destination[..., block_start:block_start + blocks, :].copy_(summary_tmp)
    del q_tmp, scale_tmp, summary_tmp


def run_streamed_sparse_sage_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    spec,
    project_chunk_rows=DEFAULT_PROJECT_CHUNK_ROWS,
    query_chunk_rows=DEFAULT_QUERY_CHUNK_ROWS,
):
    import comfy.model_management

    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise SparseSageError(
            "streamed Sparse Sage QKV requires rank-2 CUDA BF16 input"
        )
    if comfy.model_management.in_training:
        raise SparseSageError("streamed Sparse Sage QKV is inference-only")
    if int(module.head_dim) != HEAD_DIM:
        raise SparseSageError("streamed Sparse Sage QKV requires head_dim 128")
    if not spec.uses_fp8_v or spec.fused_v_ops is None:
        raise SparseSageError(
            "streamed Sparse Sage requires the FP8-V fused Sparse Sage ABI"
        )

    mismatch = sparse_fused_qkv_contract_mismatch(spec)
    if mismatch is not None:
        raise SparseSageError(
            "streamed Sparse Sage QKV contract mismatch: %s" % mismatch
        )

    if rope_freqs is not None and (
        rope_freqs.ndim != 6
        or tuple(rope_freqs.shape[:3]) != (1, x.shape[0], 1)
        or rope_freqs.device != x.device
    ):
        raise SparseSageError("streamed Sparse Sage QKV received invalid RoPE")

    project_chunk_rows = _validate_chunk_rows(
        project_chunk_rows,
        spec.q_tile,
        spec.kv_tile,
        name="project_chunk_rows",
    )
    query_chunk_rows = _validate_chunk_rows(
        query_chunk_rows,
        spec.q_tile,
        spec.kv_tile,
        name="query_chunk_rows",
    )

    sequence = int(x.shape[0])
    heads = int(module.heads)
    q_blocks = (sequence + int(spec.q_tile) - 1) // int(spec.q_tile)
    k_blocks = (sequence + int(spec.kv_tile) - 1) // int(spec.kv_tile)

    # Full K remains necessary because every Q chunk may select arbitrary K
    # tiles. Full Q is intentionally absent.
    k_int8 = torch.empty(
        (1, heads, sequence, HEAD_DIM),
        dtype=torch.int8,
        device=x.device,
    )
    k_scale = torch.empty(
        (1, heads, k_blocks),
        dtype=torch.float32,
        device=x.device,
    )
    q_summary = torch.empty(
        (1, heads, q_blocks, HEAD_DIM),
        dtype=x.dtype,
        device=x.device,
    )
    k_summary = torch.empty(
        (1, heads, k_blocks, HEAD_DIM),
        dtype=x.dtype,
        device=x.device,
    )
    v_amax = torch.zeros(
        (1, heads, HEAD_DIM),
        dtype=torch.float32,
        device=x.device,
    )

    # Pass 1: collect Q routing summaries, build full K, and obtain the exact
    # global per-[head,dim] V scale required by SpargeAttn.
    for start in range(0, sequence, project_chunk_rows):
        end = min(start + project_chunk_rows, sequence)
        q, k, v = project_chunk_hnd(
            module,
            x,
            rope_freqs,
            start,
            end,
        )
        try:
            _collect_q_summary_chunk(
                q,
                q_summary,
                row_start=start,
                q_tile=spec.q_tile,
            )
            pack_sparse_qk_chunk_into(
                k,
                k_int8,
                k_scale,
                k_summary,
                row_start=start,
                block_size=spec.kv_tile,
            )
            update_v_amax(v_amax, v)
        finally:
            del q, k, v

    v_scale = finalize_v_scale(v_amax, spec.v_quant_bound)
    del v_amax

    v_carrier = allocate_v_carrier(
        sequence=sequence,
        heads=heads,
        head_dim=HEAD_DIM,
        device=x.device,
    )

    # Pass 2: reproject bounded slabs and write only V into the permanent
    # Sparse Sage FP8 carrier. Q/K from this pass are temporary.
    for start in range(0, sequence, project_chunk_rows):
        end = min(start + project_chunk_rows, sequence)
        q_unused, k_unused, v = project_chunk_hnd(
            module,
            x,
            rope_freqs,
            start,
            end,
        )
        try:
            pack_v_chunk(
                v,
                v_carrier,
                v_scale,
                row_start=start,
                sequence=sequence,
                fused_v_ops=spec.fused_v_ops,
                scale_max=spec.v_quant_bound,
            )
        finally:
            del q_unused, k_unused, v

    return _validate_projected(
        StreamedSparseSageQKV(
            module=module,
            x=x,
            rope_freqs=rope_freqs,
            k_int8=k_int8,
            k_scale=k_scale,
            v_carrier=v_carrier,
            v_scale=v_scale,
            q_summary=q_summary,
            k_summary=k_summary,
            output_dtype=x.dtype,
            sequence=sequence,
            heads=heads,
            head_dim=HEAD_DIM,
            layer_index=int(layer_index),
            project_chunk_rows=project_chunk_rows,
            query_chunk_rows=query_chunk_rows,
        ),
        spec,
    )


class StreamedSparseSageQKVProjector:
    name = "streamed_sparse_sage_qkv"
    qk_format = "streamed_q_sparge_block_int8"

    def __init__(
        self,
        spec,
        *,
        project_chunk_rows=DEFAULT_PROJECT_CHUNK_ROWS,
        query_chunk_rows=DEFAULT_QUERY_CHUNK_ROWS,
    ):
        self.spec = spec
        self.chunk_rows = _validate_chunk_rows(
            project_chunk_rows,
            spec.q_tile,
            spec.kv_tile,
            name="project_chunk_rows",
        )
        self.query_chunk_rows = _validate_chunk_rows(
            query_chunk_rows,
            spec.q_tile,
            spec.kv_tile,
            name="query_chunk_rows",
        )

    @property
    def installation_signature(self):
        return (
            self.name,
            self.qk_format,
            self.chunk_rows,
            self.query_chunk_rows,
            self.spec.signature,
        )

    def project(
        self,
        module,
        x,
        rope_freqs,
        *,
        layer_index,
        transformer_options,
    ):
        del transformer_options
        return run_streamed_sparse_sage_qkv(
            module,
            x,
            rope_freqs,
            layer_index=layer_index,
            spec=self.spec,
            project_chunk_rows=self.chunk_rows,
            query_chunk_rows=self.query_chunk_rows,
        )


def _make_local_q(projected, spec, row_start, row_end):
    q, k_unused, v_unused = project_chunk_hnd(
        projected.module,
        projected.x,
        projected.rope_freqs,
        row_start,
        row_end,
    )
    try:
        rows = row_end - row_start
        heads = projected.heads
        q_blocks = (rows + int(spec.q_tile) - 1) // int(spec.q_tile)
        q_int8 = torch.empty(
            (1, heads, rows, HEAD_DIM),
            dtype=torch.int8,
            device=projected.x.device,
        )
        q_scale = torch.empty(
            (1, heads, q_blocks),
            dtype=torch.float32,
            device=projected.x.device,
        )
        q_summary_unused = torch.empty(
            (1, heads, q_blocks, HEAD_DIM),
            dtype=projected.output_dtype,
            device=projected.x.device,
        )
        pack_sparse_qk_chunk_into(
            q,
            q_int8,
            q_scale,
            q_summary_unused,
            row_start=0,
            block_size=spec.q_tile,
        )
    finally:
        del q, k_unused, v_unused

    del q_summary_unused
    return q_int8, q_scale


def _execute_streamed(module, backend, prepared):
    projected = _validate_projected(
        prepared.projected,
        backend.executor.spec,
    )
    spec = backend.executor.spec

    if module is not projected.module:
        raise SparseSageError(
            "streamed Sparse Sage module changed between prepare and execute"
        )

    sequence = projected.sequence
    q_tile = int(spec.q_tile)
    kv_tiles = (sequence + int(spec.kv_tile) - 1) // int(spec.kv_tile)
    result = projected.x

    pv_threshold = torch.full(
        (projected.heads,),
        50.0,
        dtype=torch.float32,
        device=projected.x.device,
    )

    for row_start in range(0, sequence, projected.query_chunk_rows):
        row_end = min(
            row_start + projected.query_chunk_rows,
            sequence,
        )
        if row_start % q_tile:
            raise SparseSageError(
                "streamed Sparse Sage Q chunk is not Q-tile aligned"
            )

        rows = row_end - row_start
        q_int8, q_scale = _make_local_q(
            projected,
            spec,
            row_start,
            row_end,
        )

        tile_start = row_start // q_tile
        local_q_tiles = (rows + q_tile - 1) // q_tile
        tile_end = tile_start + local_q_tiles

        # SpargeAttn interprets bx locally using qo_len. Slicing these route
        # rows makes local bx=0 consume the original global tile's route.
        lut_chunk = prepared.lut[
            ...,
            tile_start:tile_end,
            :,
        ].contiguous()
        valid_chunk = prepared.valid_block_num[
            ...,
            tile_start:tile_end,
        ].contiguous()

        expected_lut = (
            1,
            projected.heads,
            local_q_tiles,
            kv_tiles,
        )
        if tuple(lut_chunk.shape) != expected_lut:
            raise SparseSageError(
                "streamed Sparse Sage LUT slice %s does not match %s"
                % (tuple(lut_chunk.shape), expected_lut)
            )

        output = torch.empty(
            (1, projected.heads, rows, HEAD_DIM),
            dtype=projected.output_dtype,
            device=projected.x.device,
        )

        # This is the normal compiled SpargeAttn kernel. Its wrapper derives
        # qo_len from q_int8 and kv_len from full K, so qo_len != kv_len is
        # a supported native execution mode.
        spec.dispatch(
            q_int8,
            projected.k_int8,
            projected.v_carrier,
            output,
            lut_chunk,
            valid_chunk,
            pv_threshold,
            q_scale,
            projected.k_scale,
            projected.v_scale,
            projected.output_dtype,
        )

        attention_rows = output.transpose(1, 2).reshape(
            rows,
            projected.heads * HEAD_DIM,
        )
        projected_rows = module.out_proj(attention_rows)
        expected_output = (rows, projected.x.shape[1])
        if tuple(projected_rows.shape) != expected_output:
            raise SparseSageError(
                "streamed Sparse Sage out_proj shape %s does not match %s"
                % (tuple(projected_rows.shape), expected_output)
            )

        # norm1(x) is a temporary separate from the residual stream. By the
        # time a Q chunk is consumed its source rows are no longer needed, so
        # reuse them for the final out-projected attention result.
        result[row_start:row_end].copy_(projected_rows)

        del (
            q_int8,
            q_scale,
            lut_chunk,
            valid_chunk,
            output,
            attention_rows,
            projected_rows,
        )

    return result


class StreamedSparseSageBackend(HybridSparseBackend):
    """Sparse Sage backend using global K/V with streamed Q/output."""

    def prepare_projected(
        self,
        projected,
        *,
        layer_index,
        transformer_options,
    ):
        projected = _validate_projected(projected, self.executor.spec)
        if int(projected.layer_index) != int(layer_index):
            raise SparseSageError(
                "streamed QKV layer %d does not match attention layer %d"
                % (projected.layer_index, layer_index)
            )

        snapshot = self._snapshot(
            transformer_options,
            projected.sequence,
        )
        video_budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
            layer_index,
        )
        try:
            lut, valid_block_num, mask_metadata = (
                self.router.build_lut_from_summaries(
                    projected.q_summary,
                    projected.k_summary,
                    snapshot.layout,
                    video_budget,
                    **_route_kwargs(transformer_options, layer_index),
                )
            )
        except SparseRouterError as exc:
            raise SparseSageError("sparse routing failed: %s" % exc) from exc

        metadata = self._metadata(
            mask_metadata,
            layer_index,
            projected.heads,
        )
        metadata.update(
            {
                "qkv_lifetime": "streamed_q_global_k_fp8v",
                "attention_output": "chunked_out_proj_inplace",
                "project_chunk_rows": projected.project_chunk_rows,
                "query_chunk_rows": projected.query_chunk_rows,
            }
        )

        return PreparedStreamedHybrid(
            sparse=PreparedStreamedSparseSage(
                projected=projected,
                lut=lut,
                valid_block_num=valid_block_num,
                metadata=metadata,
            )
        )

    def execute_projected(self, module, prepared):
        if not isinstance(prepared, PreparedStreamedHybrid):
            return None
        return _execute_streamed(
            module,
            self,
            prepared.sparse,
        )

    def execute(self, prepared):
        if isinstance(prepared, PreparedStreamedHybrid):
            raise SparseSageError(
                "streamed Sparse Sage must execute through execute_projected"
            )
        return super().execute(prepared)

    def as_status(self):
        status = super().as_status()
        status.update(
            {
                "streamed_q": True,
                "chunked_attention_output": True,
                "reuses_attention_input_for_output": True,
                "query_chunk_rows": getattr(
                    self.projector,
                    "query_chunk_rows",
                    None,
                ),
            }
        )
        return status
