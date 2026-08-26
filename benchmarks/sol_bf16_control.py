'''BF16 HND control for the Sol-shaped full-route benchmark.'''

from __future__ import annotations

import torch
import triton
import triton.language as tl


Q_TILE = 64
KV_TILE = 64
HEAD_DIM = 128
CONFIGS = ((4, 1), (8, 1), (4, 2), (8, 2))


_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=warps, num_stages=stages)
    for warps, stages in CONFIGS
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=['KV_BLOCKS'])
@triton.jit
def _bf16_attention_kernel(
    Q,
    K,
    V,
    O,
    sequence,
    KV_BLOCKS: tl.constexpr,
    softmax_scale: tl.constexpr,
    Q_TILE_: tl.constexpr,
    KV_TILE_: tl.constexpr,
    D: tl.constexpr,
):
    q_block = tl.program_id(0)
    bh = tl.program_id(1)

    q_rows = q_block * Q_TILE_ + tl.arange(0, Q_TILE_)
    kv_rows = tl.arange(0, KV_TILE_)
    dims = tl.arange(0, D)
    q_mask = q_rows < sequence
    hnd_base = bh * sequence * D

    q = tl.load(
        Q + hnd_base + q_rows[:, None] * D + dims[None, :],
        mask=q_mask[:, None],
        other=0.0,
    )
    row_max = tl.full((Q_TILE_,), -float('inf'), dtype=tl.float32)
    row_sum = tl.zeros((Q_TILE_,), dtype=tl.float32)
    output = tl.zeros((Q_TILE_, D), dtype=tl.float32)

    for key_block in tl.range(0, KV_BLOCKS):
        k_rows = key_block * KV_TILE_ + kv_rows
        k_mask = k_rows < sequence
        k = tl.load(
            K + hnd_base + k_rows[None, :] * D + dims[:, None],
            mask=k_mask[None, :],
            other=0.0,
        )
        logits = tl.dot(q, k) * (softmax_scale * 1.4426950408889634)
        logits = tl.where(k_mask[None, :], logits, -float('inf'))
        v = tl.load(
            V + hnd_base + k_rows[:, None] * D + dims[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )

        tile_max = tl.max(logits, axis=1)
        new_row_max = tl.maximum(row_max, tile_max)
        probability = tl.math.exp2(logits - new_row_max[:, None])
        tile_sum = tl.sum(probability, axis=1)
        old_scale = tl.math.exp2(row_max - new_row_max)

        output = output * old_scale[:, None]
        output += tl.dot(probability.to(v.dtype), v)
        row_sum = row_sum * old_scale + tile_sum
        row_max = new_row_max

    tl.store(
        O + hnd_base + q_rows[:, None] * D + dims[None, :],
        (output / row_sum[:, None]).to(O.type.element_ty),
        mask=q_mask[:, None],
    )


def launch(q, k, v):
    '''Run ordinary BF16 attention in the Sol HND 64x64 program shape.'''
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError('BF16 control requires equal HND rank-4 Q/K/V')
    if q.shape[0] != 1 or q.shape[-1] != HEAD_DIM:
        raise ValueError('BF16 control requires batch 1 and head dimension 128')
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('BF16 control requires BF16 Q/K/V')
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError('BF16 control requires contiguous HND Q/K/V')
    sequence = int(q.shape[2])
    heads = int(q.shape[1])
    q_blocks = triton.cdiv(sequence, Q_TILE)
    kv_blocks = triton.cdiv(sequence, KV_TILE)
    output = torch.empty_like(q)
    _bf16_attention_kernel[(q_blocks, heads)](
        q,
        k,
        v,
        output,
        sequence,
        KV_BLOCKS=kv_blocks,
        softmax_scale=HEAD_DIM**-0.5,
        Q_TILE_=Q_TILE,
        KV_TILE_=KV_TILE,
        D=HEAD_DIM,
    )
    return output


def contract():
    return {
        'dtype': 'BF16 QK and BF16 P-by-V with FP32 online state',
        'layout': 'HND contiguous',
        'q_tile': Q_TILE,
        'kv_tile': KV_TILE,
        'route': 'implicit full',
        'autotune': CONFIGS,
    }
