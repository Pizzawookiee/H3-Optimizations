"""Low-VRAM streamed-query execution for BF16 Triton sparse attention."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

import comfy.model_management

from ... import diagnostics
from ...qkv.streamed import (
    PROJECTION_FORCE_INT8,
    PROJECTION_NATIVE,
    create_held_qkv,
    project_kv_hnd,
    project_q_hnd,
)
from .config import resolve_video_budget
from .router import SparseRouterError
from .triton_route import (
    TritonRouteError,
    build_compact_absolute_route_chunk,
    prepare_compact_absolute_route_chunks,
)


Q_TILE = 64
KV_TILE = 64
HEAD_DIM = 128
OUT_PROJ_CHUNK_ROWS = 2048


@dataclass
class StreamedTritonBF16QKV:
    module: object
    x: torch.Tensor
    rope_freqs: torch.Tensor | None
    held: object
    k: torch.Tensor | None
    v: torch.Tensor | None
    k_summary: torch.Tensor | None
    sequence: int
    heads: int
    head_dim: int
    layer_index: int
    chunk_rows: int
    projection_mode: str = PROJECTION_NATIVE

    def project_q(self, start, end):
        if self.held is not None:
            return project_q_hnd(
                self.held,
                self.x,
                self.rope_freqs,
                int(start),
                int(end),
            )
        held = create_held_qkv(
            self.module,
            self.x[int(start):int(start) + 1],
            self.projection_mode,
        )
        held.__enter__()
        try:
            return project_q_hnd(
                held,
                self.x,
                self.rope_freqs,
                int(start),
                int(end),
            )
        finally:
            held.__exit__(None, None, None)

    def release_weight(self):
        held, self.held = self.held, None
        if held is not None:
            held.__exit__(None, None, None)

    def release(self):
        self.release_weight()
        self.x = self.rope_freqs = None
        self.k = self.v = None
        self.k_summary = None


@dataclass
class PreparedStreamedTritonBF16:
    projected: StreamedTritonBF16QKV
    route_plan: object
    dense_q_tiles: int
    sparse_q_tiles: int
    sparse_selected: int
    metadata: dict

    def release(self):
        self.projected.release()
        if self.route_plan is not None:
            self.route_plan.release()
        self.route_plan = None


def _validate_chunk_rows(chunk_rows):
    chunk_rows = int(chunk_rows)
    alignment = math.lcm(Q_TILE, KV_TILE)
    if chunk_rows <= 0 or chunk_rows % alignment:
        raise RuntimeError(
            "streamed Triton chunk_rows must be a positive multiple of %d"
            % alignment
        )
    return chunk_rows


def _tile_mean(x, tile):
    sequence = int(x.shape[-2])
    full = sequence // int(tile)
    remainder = sequence % int(tile)
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


def _uses_reusable_cast_buffer(held):
    binding = getattr(held, "binding", held)
    return getattr(binding, "handle", None) is not None


def _assemble_streamed_triton_qkv(
    module,
    x,
    rope_freqs,
    held,
    *,
    layer_index,
    chunk_rows,
    projection_mode=PROJECTION_NATIVE,
):
    chunk_rows = _validate_chunk_rows(chunk_rows)
    sequence = int(x.shape[0])
    heads = int(module.heads)
    head_dim = int(module.head_dim)
    if sequence <= 0 or head_dim != HEAD_DIM:
        raise RuntimeError("streamed Triton QKV requires head_dim 128")

    shape = (1, heads, sequence, head_dim)
    kv_tiles = (sequence + KV_TILE - 1) // KV_TILE
    k = torch.empty(shape, dtype=torch.bfloat16, device=x.device)
    v = torch.empty(shape, dtype=torch.bfloat16, device=x.device)
    k_summary = torch.empty(
        (1, heads, kv_tiles, head_dim),
        dtype=torch.bfloat16,
        device=x.device,
    )

    try:
        for start in range(0, sequence, chunk_rows):
            end = min(start + chunk_rows, sequence)
            k_chunk, v_chunk = project_kv_hnd(
                held,
                x,
                rope_freqs,
                start,
                end,
            )
            try:
                k[..., start:end, :].copy_(k_chunk)
                v[..., start:end, :].copy_(v_chunk)
                k_block = start // KV_TILE
                k_mean = _tile_mean(k_chunk, KV_TILE)
                k_summary[..., k_block:k_block + k_mean.shape[-2], :].copy_(
                    k_mean
                )
                del k_mean
            finally:
                del k_chunk, v_chunk
        projected = StreamedTritonBF16QKV(
            module=module,
            x=x,
            rope_freqs=rope_freqs,
            held=held,
            k=k,
            v=v,
            k_summary=k_summary,
            sequence=sequence,
            heads=heads,
            head_dim=head_dim,
            layer_index=int(layer_index),
            chunk_rows=chunk_rows,
            projection_mode=projection_mode,
        )
        if _uses_reusable_cast_buffer(held):
            projected.release_weight()
        return projected
    except Exception:
        held.__exit__(None, None, None)
        raise


def _validate_streamed_input(x, rope_freqs):
    if (
        x.ndim != 2
        or not x.is_cuda
        or x.dtype != torch.bfloat16
    ):
        raise RuntimeError(
            "streamed Triton QKV requires rank-2 CUDA BF16 input"
        )
    if comfy.model_management.in_training:
        raise RuntimeError("streamed Triton QKV is inference-only")
    if rope_freqs is not None and (
        rope_freqs.ndim != 6
        or tuple(rope_freqs.shape[:3]) != (1, x.shape[0], 1)
        or rope_freqs.device != x.device
    ):
        raise RuntimeError("streamed Triton QKV received invalid RoPE")


def run_streamed_triton_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    chunk_rows,
):
    _validate_streamed_input(x, rope_freqs)

    return run_streamed_source_triton_qkv(
        module,
        x,
        rope_freqs,
        layer_index=layer_index,
        chunk_rows=chunk_rows,
        projection_mode=PROJECTION_FORCE_INT8,
    )


def run_streamed_bf16_triton_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    chunk_rows,
):
    return run_streamed_source_triton_qkv(
        module,
        x,
        rope_freqs,
        layer_index=layer_index,
        chunk_rows=chunk_rows,
        projection_mode=PROJECTION_NATIVE,
    )


def run_streamed_source_triton_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    chunk_rows,
    projection_mode=PROJECTION_NATIVE,
):
    _validate_streamed_input(x, rope_freqs)

    held = create_held_qkv(module, x[:1], projection_mode)
    held.__enter__()
    return _assemble_streamed_triton_qkv(
        module,
        x,
        rope_freqs,
        held,
        layer_index=layer_index,
        chunk_rows=chunk_rows,
        projection_mode=projection_mode,
    )


def prepare_streamed_triton_bf16(
    backend,
    projected,
    *,
    layer_index,
    transformer_options,
):
    if not isinstance(projected, StreamedTritonBF16QKV):
        raise RuntimeError("expected StreamedTritonBF16QKV")
    if int(projected.layer_index) != int(layer_index):
        projected.release()
        raise RuntimeError("streamed Triton QKV layer changed")
    snapshot = backend._snapshot(transformer_options, projected.sequence)
    budget = resolve_video_budget(
        backend.config,
        snapshot.step_index,
        snapshot.total_steps,
        layer_index,
    )
    try:
        with diagnostics.stage("sparse_route"):
            route_plan = prepare_compact_absolute_route_chunks(
                backend.router,
                projected.k_summary,
                snapshot.layout,
                budget,
            )
    except (SparseRouterError, TritonRouteError):
        projected.release()
        raise

    projected.k_summary = None
    mask_metadata = route_plan.metadata
    metadata = mask_metadata.as_dict()
    metadata.update(
        {
            "layer": int(layer_index),
            "sparse_backend": backend.name,
            "route_format": "dense_implicit_plus_sparse_absolute_int32",
            "program_shape": "one_64q_tile_x_one_head_x_full_d128",
            "qkv_lifetime": "streamed_q_global_bf16_kv",
            "router_lifetime": "k_summary_until_final_q_route",
            "attention_output": "q_slab_reused_then_chunked_out_proj_inplace",
            "q_output_alias": True,
            "query_chunk_rows": projected.chunk_rows,
            "out_proj_chunk_rows": OUT_PROJ_CHUNK_ROWS,
        }
    )
    return PreparedStreamedTritonBF16(
        projected=projected,
        route_plan=route_plan,
        dense_q_tiles=int(mask_metadata.dense_q_tiles),
        sparse_q_tiles=int(mask_metadata.sparse_q_tiles),
        sparse_selected=(
            0
            if route_plan.retained == route_plan.geometry.pure_video_kv_tiles
            else int(route_plan.geometry.pure_video_kv_start) + route_plan.retained
        ),
        metadata=metadata,
    )


def execute_streamed_triton_bf16(module, backend, prepared):
    if not isinstance(prepared, PreparedStreamedTritonBF16):
        return None
    from .triton_bf16 import _launch_streamed_chunk

    projected = prepared.projected
    if getattr(module, "_module", module) is not projected.module:
        prepared.release()
        raise RuntimeError("streamed Triton attention module changed")

    result = projected.x
    sequence = int(projected.sequence)
    try:
        for start in range(0, sequence, projected.chunk_rows):
            end = min(start + projected.chunk_rows, sequence)
            q = projected.project_q(start, end)
            if end == sequence:
                projected.release_weight()
            q_summary = _tile_mean(q, Q_TILE)
            with diagnostics.stage("sparse_route"):
                sparse_lut = build_compact_absolute_route_chunk(
                    backend.router,
                    q_summary,
                    prepared.route_plan,
                    q_tile_start=start // Q_TILE,
                )
            del q_summary

            # After the last query slab has selected its route, the global K
            # summary has no remaining consumer. Drop it before attention so
            # the same-stream allocator may reuse those bytes immediately.
            if end == sequence and prepared.route_plan is not None:
                prepared.route_plan.release()
                prepared.route_plan = None

            with diagnostics.stage("sparse_attention_kernel"):
                output = _launch_streamed_chunk(
                    q,
                    projected.k,
                    projected.v,
                    sparse_lut,
                    dense_q_tiles=prepared.dense_q_tiles,
                    sparse_q_tiles=prepared.sparse_q_tiles,
                    sparse_selected=prepared.sparse_selected,
                    sequence=sequence,
                    q_row_start=start,
                    sparse_lut_q_start=max(
                        0,
                        start // Q_TILE - prepared.dense_q_tiles,
                    ),
                )
            del q, sparse_lut
            if end == sequence:
                projected.k = None
                projected.v = None

            with diagnostics.stage("attention_out"):
                rows = end - start
                for local_start in range(0, rows, OUT_PROJ_CHUNK_ROWS):
                    local_end = min(local_start + OUT_PROJ_CHUNK_ROWS, rows)
                    count = local_end - local_start
                    attention_rows = (
                        output[..., local_start:local_end, :]
                        .transpose(1, 2)
                        .reshape(count, projected.heads * projected.head_dim)
                    )
                    projected_rows = module.out_proj(attention_rows)
                    result[
                        start + local_start:start + local_end
                    ].copy_(projected_rows)
                    del attention_rows, projected_rows
            del output
        return result
    finally:
        prepared.release()


__all__ = [
    "PreparedStreamedTritonBF16",
    "StreamedTritonBF16QKV",
    "execute_streamed_triton_bf16",
    "prepare_streamed_triton_bf16",
    "run_streamed_bf16_triton_qkv",
    "run_streamed_source_triton_qkv",
    "run_streamed_triton_qkv",
]
