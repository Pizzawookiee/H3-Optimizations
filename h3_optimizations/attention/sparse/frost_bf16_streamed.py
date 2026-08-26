'''Low-VRAM streamed-query execution for the FROST BF16 backend.'''

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ... import diagnostics
from ...qkv.streamed import (
    PROJECTION_NATIVE,
    create_held_qkv,
    project_q_hnd,
)
from .config import resolve_video_budget
from .frost_route import build_full_absolute_route_from_summaries
from .router import SparseRouterError
from .triton_route import TritonRouteError


OUT_PROJ_CHUNK_ROWS = 2048


@dataclass
class StreamedFrostBF16QKV:
    module: object
    x: torch.Tensor
    rope_freqs: torch.Tensor | None
    projection_mode: str
    held_factory: object
    k: torch.Tensor | None
    v: torch.Tensor | None
    q_summary: torch.Tensor | None
    k_summary: torch.Tensor | None
    sequence: int
    heads: int
    head_dim: int
    layer_index: int
    chunk_rows: int

    def project_q(self, start, end):
        if self.held_factory is None:
            raise RuntimeError('streamed FROST QKV projector was released')
        held = self.held_factory(
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

    def release(self):
        self.held_factory = None
        self.x = self.rope_freqs = None
        self.k = self.v = None
        self.q_summary = self.k_summary = None


@dataclass
class PreparedStreamedFrostBF16:
    projected: StreamedFrostBF16QKV
    route: torch.Tensor | None
    counts: torch.Tensor | None
    metadata: dict

    def release(self):
        self.projected.release()
        self.route = self.counts = None


def _validate_chunk_rows(chunk_rows, q_tile, kv_tile):
    chunk_rows = int(chunk_rows)
    alignment = math.lcm(int(q_tile), int(kv_tile))
    if chunk_rows <= 0 or chunk_rows % alignment:
        raise RuntimeError(
            'streamed FROST chunk_rows must be a positive multiple of %d'
            % alignment
        )
    return chunk_rows


def _tile_mean(x, tile):
    rows = int(x.shape[-2])
    full = rows // int(tile)
    remainder = rows % int(tile)
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


def _assemble_streamed_frost_qkv(
    module,
    x,
    rope_freqs,
    *,
    spec,
    layer_index,
    chunk_rows,
    projection_mode=PROJECTION_NATIVE,
    held_factory=create_held_qkv,
):
    sequence = int(x.shape[0])
    heads = int(module.heads)
    head_dim = int(module.head_dim)
    chunk_rows = _validate_chunk_rows(
        chunk_rows,
        spec.q_tile,
        spec.kv_tile,
    )
    if sequence <= 0 or heads != int(spec.heads) or head_dim != int(spec.head_dim):
        raise RuntimeError(
            'streamed FROST QKV requires %d heads with head_dim %d'
            % (spec.heads, spec.head_dim)
        )

    storage_shape = (1, sequence, heads, head_dim)
    k_storage = torch.empty(storage_shape, dtype=torch.bfloat16, device=x.device)
    v_storage = torch.empty(storage_shape, dtype=torch.bfloat16, device=x.device)
    k_full = k_storage.permute(0, 2, 1, 3)
    v_full = v_storage.permute(0, 2, 1, 3)
    q_tiles = (sequence + int(spec.q_tile) - 1) // int(spec.q_tile)
    kv_tiles = (sequence + int(spec.kv_tile) - 1) // int(spec.kv_tile)
    q_summary = torch.empty(
        (1, heads, q_tiles, head_dim),
        dtype=torch.bfloat16,
        device=x.device,
    )
    k_summary = torch.empty(
        (1, heads, kv_tiles, head_dim),
        dtype=torch.bfloat16,
        device=x.device,
    )

    def consume(start, end, q, k, v):
        start = int(start)
        end = int(end)
        if start % int(spec.q_tile) or start % int(spec.kv_tile):
            raise RuntimeError('streamed FROST QKV chunk is not tile-aligned')
        expected = (1, heads, end - start, head_dim)
        if tuple(q.shape) != expected or tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise RuntimeError('streamed FROST QKV chunk shape is invalid')
        k_full[..., start:end, :].copy_(k)
        v_full[..., start:end, :].copy_(v)
        q_means = _tile_mean(q, spec.q_tile)
        k_means = _tile_mean(k, spec.kv_tile)
        q_start = start // int(spec.q_tile)
        k_start = start // int(spec.kv_tile)
        q_summary[..., q_start:q_start + q_means.shape[-2], :].copy_(q_means)
        k_summary[..., k_start:k_start + k_means.shape[-2], :].copy_(k_means)

    held = held_factory(module, x[:1], projection_mode)
    held.__enter__()
    try:
        for start in range(0, sequence, chunk_rows):
            end = min(start + chunk_rows, sequence)
            q, k, v = held.project_hnd(x, rope_freqs, start, end)
            try:
                consume(start, end, q, k, v)
            finally:
                del q, k, v
    finally:
        held.__exit__(None, None, None)
    return StreamedFrostBF16QKV(
        module=module,
        x=x,
        rope_freqs=rope_freqs,
        projection_mode=projection_mode,
        held_factory=held_factory,
        k=k_full,
        v=v_full,
        q_summary=q_summary,
        k_summary=k_summary,
        sequence=sequence,
        heads=heads,
        head_dim=head_dim,
        layer_index=int(layer_index),
        chunk_rows=chunk_rows,
    )


def prepare_streamed_frost_bf16(
    backend,
    projected,
    *,
    layer_index,
    transformer_options,
):
    if not isinstance(projected, StreamedFrostBF16QKV):
        raise RuntimeError('expected StreamedFrostBF16QKV')
    if int(projected.layer_index) != int(layer_index):
        projected.release()
        raise RuntimeError('streamed FROST QKV layer changed')
    snapshot = backend._snapshot(transformer_options, projected.sequence)
    budget = resolve_video_budget(
        backend.config,
        snapshot.step_index,
        snapshot.total_steps,
        layer_index,
    )
    try:
        with diagnostics.stage('sparse_route'):
            route, counts, route_metadata = build_full_absolute_route_from_summaries(
                backend.router,
                projected.q_summary,
                projected.k_summary,
                snapshot.layout,
                budget,
            )
    except (SparseRouterError, TritonRouteError):
        projected.release()
        raise

    projected.q_summary = None
    projected.k_summary = None
    metadata = route_metadata.as_dict()
    metadata.update(
        {
            'layer': int(layer_index),
            'sparse_backend': backend.name,
            'route_format': 'absolute_full_int32_direct',
            'qkv_lifetime': 'streamed_q_global_sequence_major_bf16_kv',
            'query_chunk_rows': projected.chunk_rows,
            'out_proj_chunk_rows': OUT_PROJ_CHUNK_ROWS,
        }
    )
    return PreparedStreamedFrostBF16(
        projected=projected,
        route=route,
        counts=counts,
        metadata=metadata,
    )


def execute_streamed_frost_bf16(module, backend, prepared):
    if not isinstance(prepared, PreparedStreamedFrostBF16):
        return None
    projected = prepared.projected
    if getattr(module, '_module', module) is not projected.module:
        prepared.release()
        raise RuntimeError('streamed FROST attention module changed')

    result = projected.x
    sequence = int(projected.sequence)
    q_tile = int(backend.spec.q_tile)
    try:
        for start in range(0, sequence, projected.chunk_rows):
            end = min(start + projected.chunk_rows, sequence)
            rows = end - start
            q = projected.project_q(start, end)
            q_storage = torch.empty(
                (1, rows, projected.heads, projected.head_dim),
                dtype=torch.bfloat16,
                device=result.device,
            )
            q_sequence_major = q_storage.permute(0, 2, 1, 3)
            q_sequence_major.copy_(q)
            del q

            tile_start = start // q_tile
            tile_end = (end + q_tile - 1) // q_tile
            route = prepared.route[..., tile_start:tile_end, :].contiguous()
            counts = prepared.counts[..., tile_start:tile_end].contiguous()
            with diagnostics.stage('sparse_attention_kernel'):
                chunk = backend.executor.prepare(
                    q_sequence_major,
                    projected.k,
                    projected.v,
                    route,
                    counts,
                    layer_index=projected.layer_index,
                    metadata=dict(prepared.metadata, q_row_start=start),
                )
                output = backend.executor.execute(chunk)
            del chunk, q_sequence_major, q_storage, route, counts

            if end == sequence:
                projected.k = None
                projected.v = None

            with diagnostics.stage('attention_out'):
                for local_start in range(0, rows, OUT_PROJ_CHUNK_ROWS):
                    local_end = min(local_start + OUT_PROJ_CHUNK_ROWS, rows)
                    count = local_end - local_start
                    attention_rows = (
                        output[..., local_start:local_end, :]
                        .transpose(1, 2)
                        .reshape(count, projected.heads * projected.head_dim)
                    )
                    projected_rows = module.out_proj(attention_rows)
                    if tuple(projected_rows.shape) != (count, result.shape[1]):
                        raise RuntimeError(
                            'streamed FROST out_proj shape %s is invalid'
                            % (tuple(projected_rows.shape),)
                        )
                    result[start + local_start:start + local_end].copy_(projected_rows)
                    del attention_rows, projected_rows
            del output
        return result
    finally:
        prepared.release()


__all__ = [
    'PreparedStreamedFrostBF16',
    'StreamedFrostBF16QKV',
    '_assemble_streamed_frost_qkv',
    'execute_streamed_frost_bf16',
    'prepare_streamed_frost_bf16',
]
