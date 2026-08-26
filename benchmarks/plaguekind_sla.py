'''PlagueKind SLA kernel and a benchmark-only H3 route adapter.

The Triton kernel and launch ladder are copied unchanged from:

    PlagueKind/ComfyUI-PlagueKind-Nodes
    ComfyUI-H3-SLA-Attention/sla/kernel.py
    commit a05db58981fec697bd3644229b215ee04c126ea7

Copyright (c) 2026 PlagueKind, distributed under the MIT License. The source
declares that its forward kernel is vendored and reduced from LightX2V's
``sla_kernel_ar.py``, distributed under the Apache License, Version 2.0.

Only the adapter below ``block_sparse_attention`` is new. It preserves the H3
benchmark's shared route by launching the unchanged fixed-topk kernel once for
the dense prefix Q tiles and once for the sparse video Q tiles.
'''

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from h3_optimizations.attention.sparse.config import resolve_video_budget
from h3_optimizations.attention.sparse.router import SparseTileRouter
from h3_optimizations.runtime.context import get_runtime_snapshot


PLAGUEKIND_SOURCE_COMMIT = 'a05db58981fec697bd3644229b215ee04c126ea7'


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT,
    OS,
    H: tl.constexpr,
    LQ: tl.constexpr,
    LK: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    idx_b = idx_bh // H
    idx_h = idx_bh % H

    HD: tl.constexpr = H * D

    q_offset = idx_b * LQ * HD + idx_h * D
    kv_offset = idx_b * LK * HD + idx_h * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    OS_ptrs = OS + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    LUT_ptr = LUT + lut_offset

    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < LQ, other=0.0)
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx).to(tl.int64)
        k_start = idx_n * BLOCK_N
        k_mask = (k_start + offs_n) < LK

        K_ptrs = K + kv_offset + (k_start + offs_n)[None, :] * HD + offs_d[:, None]
        V_ptrs = V + kv_offset + (k_start + offs_n)[:, None] * HD + offs_d[None, :]

        k = tl.load(K_ptrs, mask=k_mask[None, :], other=0.0)
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)
        qk = tl.where(k_mask[None, :], qk, float('-inf'))

        v = tl.load(V_ptrs, mask=k_mask[:, None], other=0.0)
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < LQ)


_LADDER = {
    (128, 64): ((8, 3), (4, 3), (8, 2), (4, 1)),
    (128, 128): ((8, 2), (4, 2), (8, 1), (4, 1)),
    (64, 128): ((4, 2), (8, 2), (4, 1)),
    (64, 64): ((4, 1), (4, 3), (8, 3), (8, 1)),
}
_CHOSEN: dict = {}


def block_sparse_attention(q, k, v, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
    '''Attend each query block to only the absolute key blocks in ``lut``.'''
    output = torch.empty_like(q)
    _block_sparse_attention_into(
        q, k, v, lut, topk, BLOCK_M, BLOCK_N, output, qk_scale
    )
    return output


def _block_sparse_attention_into(
    q,
    k,
    v,
    lut,
    topk,
    BLOCK_M,
    BLOCK_N,
    output,
    qk_scale=None,
):
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert lut.is_contiguous() and output.is_contiguous()
    assert output.shape == q.shape and output.dtype == q.dtype
    assert BLOCK_M in (64, 128) and BLOCK_N in (64, 128)

    B, LQ, H, D = q.shape
    LK = k.shape[1]
    if qk_scale is None:
        qk_scale = D**-0.5

    M_BLOCKS = triton.cdiv(LQ, BLOCK_M)
    grid = (M_BLOCKS, B * H)

    key = (BLOCK_M, BLOCK_N, D)
    ladder = (_CHOSEN[key],) if key in _CHOSEN else _LADDER[(BLOCK_M, BLOCK_N)]

    last = None
    for cfg in ladder:
        num_warps, num_stages = cfg
        try:
            _attn_fwd[grid](
                q, k, v, qk_scale, topk, lut, output,
                H, LQ, LK, M_BLOCKS, D, BLOCK_M, BLOCK_N,
                num_warps=num_warps, num_stages=num_stages,
            )
        except triton.runtime.errors.OutOfResources as exc:
            last = exc
            continue
        _CHOSEN[key] = cfg
        return output

    raise last if last is not None else RuntimeError('no viable launch config')


@dataclass(frozen=True)
class PlagueKindSLASpec:
    q_tile: int = 128
    kv_tile: int = 64
    head_dim: int = 128
    implementation: str = 'plaguekind_sla_float_blhd'
    source_commit: str = PLAGUEKIND_SOURCE_COMMIT


@dataclass
class PreparedPlagueKindSLA:
    q_blhd: torch.Tensor
    k_blhd: torch.Tensor
    v_blhd: torch.Tensor
    dense_lut: torch.Tensor
    kv_indices: torch.Tensor
    valid_block_num: torch.Tensor
    sequence: int
    q_tiles: int
    kv_tiles: int
    dense_q_tiles: int
    sparse_q_tiles: int
    sparse_selected: int
    metadata: dict


class PlagueKindSLABackend:
    name = 'plaguekind_sla'

    def __init__(self, config, spec=None, kernel=None):
        self.config = config
        self.spec = spec or PlagueKindSLASpec()
        self.kernel = kernel or _block_sparse_attention_into
        self.router = SparseTileRouter(
            config,
            q_tile=self.spec.q_tile,
            kv_tile=self.spec.kv_tile,
        )

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
            raise ValueError('PlagueKind SLA requires equal HND rank-4 Q/K/V')
        if q.shape[0] != 1 or q.shape[-1] != self.spec.head_dim:
            raise ValueError('PlagueKind SLA requires batch 1 and head_dim 128')
        if q.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError('PlagueKind SLA requires BF16 or FP16 Q/K/V')
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError('PlagueKind SLA Q/K/V dtypes differ')

        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None or not snapshot.valid_layout:
            raise ValueError('PlagueKind SLA benchmark requires a valid H3 layout')
        budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
        )
        delta_lut, valid, metadata = self.router.build_lut(
            q,
            k,
            snapshot.layout,
            budget,
        )
        absolute = torch.cumsum(delta_lut, dim=-1, dtype=torch.int32)
        dense_q_tiles = int(metadata.dense_q_tiles)
        sparse_q_tiles = int(metadata.sparse_q_tiles)
        selected = (
            int(metadata.kv_tiles)
            - int(metadata.pure_video_kv_tiles)
            + int(metadata.retained_video_kv_tiles)
            if sparse_q_tiles
            else 0
        )
        dense_lut = absolute[..., :dense_q_tiles, :].contiguous()
        sparse_lut = absolute[
            ...,
            dense_q_tiles:dense_q_tiles + sparse_q_tiles,
            :selected,
        ].contiguous()
        return PreparedPlagueKindSLA(
            q_blhd=q.transpose(1, 2).contiguous(),
            k_blhd=k.transpose(1, 2).contiguous(),
            v_blhd=v.transpose(1, 2).contiguous(),
            dense_lut=dense_lut,
            kv_indices=sparse_lut,
            valid_block_num=valid,
            sequence=int(q.shape[-2]),
            q_tiles=int(metadata.q_tiles),
            kv_tiles=int(metadata.kv_tiles),
            dense_q_tiles=dense_q_tiles,
            sparse_q_tiles=sparse_q_tiles,
            sparse_selected=selected,
            metadata={
                **metadata.as_dict(),
                'sparse_backend': self.name,
                'source_commit': self.spec.source_commit,
                'input_layout': 'BLHD contiguous',
                'shared_route_launches': int(bool(dense_q_tiles)) + int(bool(sparse_q_tiles)),
            },
        )

    def execute(self, prepared):
        dense_rows = min(
            prepared.sequence,
            prepared.dense_q_tiles * self.spec.q_tile,
        )
        output = torch.empty_like(prepared.q_blhd)
        if prepared.dense_q_tiles:
            self.kernel(
                prepared.q_blhd[:, :dense_rows],
                prepared.k_blhd,
                prepared.v_blhd,
                prepared.dense_lut,
                prepared.kv_tiles,
                self.spec.q_tile,
                self.spec.kv_tile,
                output[:, :dense_rows],
            )
        if prepared.sparse_q_tiles:
            self.kernel(
                prepared.q_blhd[:, dense_rows:],
                prepared.k_blhd,
                prepared.v_blhd,
                prepared.kv_indices,
                prepared.sparse_selected,
                self.spec.q_tile,
                self.spec.kv_tile,
                output[:, dense_rows:],
            )
        return output.transpose(1, 2)
