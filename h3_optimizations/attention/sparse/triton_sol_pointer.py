'''Minimal Sol-shaped Triton attention over the Kitchen INT8 carrier.

This experiment deliberately answers one narrow question: can Triton execute a
single 64-query x one-head x full-D128 program efficiently on the Kitchen
carrier? It is full-route only, has no router, and is not selected by the
production backend.

The program shape follows Sol-Attn's non-TMA pointer reference. Kitchen's Q/K
scale layouts, U8 probability rounding, signed-dot correction, permuted V
layout, online-softmax state, and final V scaling remain unchanged. V sums are
prepared once per carrier KV tile so they are not reduced again for every Q
tile.
'''

from __future__ import annotations

from dataclasses import dataclass

import torch

from .triton_kitchen import (
    HEAD_DIM,
    KV_TILE,
    LOG2E,
    Q_TILE,
    S_U8_OFFSET,
    _validate_carrier,
)

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    triton = None
    tl = None
    TRITON_AVAILABLE = False


SOL_POINTER_CONFIGS = ((4, 1), (8, 1), (4, 2), (8, 2))


class TritonSolPointerError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedSolKitchenCarrier:
    carrier: object
    v_sum: torch.Tensor


def kernel_contract():
    '''Return the static experiment contract without requiring CUDA.'''
    return {
        'q_tile': Q_TILE,
        'kv_tile': KV_TILE,
        'head_dim': HEAD_DIM,
        'program': 'one_64q_tile_x_one_head_x_full_d128',
        'route': 'full_absolute_compile_time',
        'v_sum': 'precomputed_once_per_carrier_kv_tile',
        'autotune': SOL_POINTER_CONFIGS,
        'production_backend': False,
    }


def _shape(carrier):
    carrier = _validate_carrier(carrier)
    sequence = int(carrier.q.shape[2])
    heads = int(carrier.q.shape[1])
    q_blocks = (sequence + Q_TILE - 1) // Q_TILE
    kv_blocks = (sequence + KV_TILE - 1) // KV_TILE
    padded_sequence = kv_blocks * KV_TILE
    return carrier, sequence, heads, q_blocks, kv_blocks, padded_sequence


if TRITON_AVAILABLE:
    @triton.jit
    def _rni_s32(x):
        return tl.inline_asm_elementwise(
            asm='cvt.rni.s32.f32 $0, $1;',
            constraints='=r,f',
            args=[x],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _v_perm16(position):
        low = position & 15
        perm = (
            (low & 1)
            | (((low >> 3) & 1) << 1)
            | (((low >> 1) & 1) << 2)
            | (((low >> 2) & 1) << 3)
        )
        return (position & ~15) | perm

    @triton.jit
    def _prepare_v_sum_kernel(
        V,
        V_SUM,
        padded_sequence,
        KV_BLOCKS: tl.constexpr,
        KV_TILE_: tl.constexpr,
        D: tl.constexpr,
    ):
        key_block = tl.program_id(0)
        bh = tl.program_id(1)
        kv_rows = tl.arange(0, KV_TILE_)
        dims = tl.arange(0, D)
        v_positions = _v_perm16(key_block * KV_TILE_ + kv_rows)
        v = tl.load(
            V
            + (bh * D + dims[None, :]) * padded_sequence
            + v_positions[:, None],
        ).to(tl.int32)
        tl.store(
            V_SUM + (bh * KV_BLOCKS + key_block) * D + dims,
            tl.sum(v, axis=0),
        )

    _CONFIGS = [
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps, stages in SOL_POINTER_CONFIGS
    ]

    def _autotune(configs, key):
        try:
            return triton.autotune(
                configs=configs,
                key=key,
                cache_results=True,
            )
        except TypeError:  # older Triton without persistent autotune cache
            return triton.autotune(configs=configs, key=key)

    @_autotune(_CONFIGS, key=['KV_BLOCKS', 'OUTPUT_BF16'])
    @triton.jit
    def _sol_kitchen_full_kernel(
        Q,
        K,
        V,
        Q_SCALE,
        K_SCALE,
        V_SCALE,
        V_SUM,
        O,
        sequence,
        padded_sequence,
        KV_BLOCKS: tl.constexpr,
        softmax_scale: tl.constexpr,
        Q_TILE_: tl.constexpr,
        KV_TILE_: tl.constexpr,
        D: tl.constexpr,
        LOG2E_: tl.constexpr,
        S_U8_OFFSET_: tl.constexpr,
        OUTPUT_BF16: tl.constexpr,
    ):
        # Sol pointer shape: one value tile, one Q block, one batch/head.
        _value_tile = tl.program_id(0)
        q_block = tl.program_id(1)
        bh = tl.program_id(2)

        q_rows = q_block * Q_TILE_ + tl.arange(0, Q_TILE_)
        kv_rows = tl.arange(0, KV_TILE_)
        dims = tl.arange(0, D)
        q_mask = q_rows < sequence
        hnd_base = bh * sequence * D

        q = tl.load(
            Q + hnd_base + q_rows[:, None] * D + dims[None, :],
            mask=q_mask[:, None],
            other=0,
        ).to(tl.int8)
        q_scale_count = ((sequence + 127) // 128) * 32
        q_scale_index = (q_rows // 32) * 8 + (q_rows & 7)
        q_scale = tl.load(
            Q_SCALE + bh * q_scale_count + q_scale_index,
            mask=q_mask,
            other=1.0,
        ).to(tl.float32)

        row_max = tl.full((Q_TILE_,), -float('inf'), dtype=tl.float32)
        row_sum = tl.zeros((Q_TILE_,), dtype=tl.float32)
        output = tl.zeros((Q_TILE_, D), dtype=tl.float32)
        k_scale_count = KV_BLOCKS * 4

        # Full route only. The absolute KV block is the compile-time loop index;
        # there is no LUT load or loop-carried delta dependency.
        for key_block in tl.range(0, KV_BLOCKS):
            k_positions = key_block * KV_TILE_ + kv_rows
            k_valid = k_positions < sequence
            k = tl.load(
                K + hnd_base + k_positions[None, :] * D + dims[:, None],
                mask=k_valid[None, :],
                other=0,
            ).to(tl.int8)
            score_i32 = tl.dot(q, k, out_dtype=tl.int32)

            k_scale_index = key_block * 4 + ((kv_rows & 7) >> 1)
            k_scale = tl.load(
                K_SCALE + bh * k_scale_count + k_scale_index
            ).to(tl.float32)
            logits = score_i32.to(tl.float32) * (
                q_scale[:, None]
                * k_scale[None, :]
                * softmax_scale
                * LOG2E_
            )
            logits = tl.where(k_valid[None, :], logits, -float('inf'))

            tile_max = tl.max(logits, axis=1)
            tile_row_max = tile_max - S_U8_OFFSET_
            new_row_max = tl.maximum(row_max, tile_row_max)
            old_scale = tl.math.exp2(row_max - new_row_max)
            tile_scale = tl.math.exp2(tile_row_max - new_row_max)

            probability = tl.math.exp2(logits - tile_row_max[:, None])
            probability = tl.where(k_valid[None, :], probability, 0.0)
            p_code = _rni_s32(probability)
            p_code = tl.minimum(tl.maximum(p_code, 0), 255)
            p_signed = (p_code - 128).to(tl.int8)

            v_positions = _v_perm16(k_positions)
            v = tl.load(
                V
                + (bh * D + dims[None, :]) * padded_sequence
                + v_positions[:, None],
            ).to(tl.int8)
            v_sum = tl.load(
                V_SUM + (bh * KV_BLOCKS + key_block) * D + dims
            ).to(tl.int32)
            pv_i32 = tl.dot(p_signed, v, out_dtype=tl.int32)
            pv_i32 += 128 * v_sum[None, :]

            output = output * old_scale[:, None]
            output += pv_i32.to(tl.float32) * tile_scale[:, None]
            row_sum = row_sum * old_scale
            row_sum += tl.sum(p_code, axis=1).to(tl.float32) * tile_scale
            row_max = new_row_max

        v_scale = tl.load(V_SCALE + bh * D + dims).to(tl.float32)
        normalized = (output / row_sum[:, None]) * v_scale[None, :]
        tl.store(
            O + hnd_base + q_rows[:, None] * D + dims[None, :],
            normalized.to(O.type.element_ty),
            mask=q_mask[:, None],
        )


def prepare_carrier(carrier):
    '''Precompute the per-KV-tile signed V sums used by logical U8 P x S8 V.'''
    if not TRITON_AVAILABLE:
        raise TritonSolPointerError('Sol-shaped Kitchen experiment requires Triton')
    carrier, _sequence, heads, _q_blocks, kv_blocks, padded = _shape(carrier)
    if not carrier.q.is_cuda:
        raise TritonSolPointerError('Sol-shaped Kitchen experiment requires CUDA')
    v_sum = torch.empty(
        (heads, kv_blocks, HEAD_DIM),
        dtype=torch.int32,
        device=carrier.q.device,
    )
    _prepare_v_sum_kernel[(kv_blocks, heads)](
        carrier.v,
        v_sum,
        padded,
        KV_BLOCKS=kv_blocks,
        KV_TILE_=KV_TILE,
        D=HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )
    return PreparedSolKitchenCarrier(carrier=carrier, v_sum=v_sum)


def launch_prepared(prepared):
    '''Execute the full-route pointer kernel from a prepared Kitchen carrier.'''
    if not TRITON_AVAILABLE:
        raise TritonSolPointerError('Sol-shaped Kitchen experiment requires Triton')
    if not isinstance(prepared, PreparedSolKitchenCarrier):
        raise TritonSolPointerError('invalid Sol-shaped Kitchen payload')
    carrier, sequence, heads, q_blocks, kv_blocks, padded = _shape(
        prepared.carrier
    )
    expected_v_sum = (heads, kv_blocks, HEAD_DIM)
    if tuple(prepared.v_sum.shape) != expected_v_sum:
        raise TritonSolPointerError(
            'V sum shape %s does not match %s'
            % (tuple(prepared.v_sum.shape), expected_v_sum)
        )
    if prepared.v_sum.dtype != torch.int32 or not prepared.v_sum.is_contiguous():
        raise TritonSolPointerError('V sum must be contiguous INT32')
    if prepared.v_sum.device != carrier.q.device:
        raise TritonSolPointerError('V sum and Kitchen carrier devices differ')
    if not carrier.q.is_cuda:
        raise TritonSolPointerError('Sol-shaped Kitchen experiment requires CUDA')

    output = torch.empty(
        (1, heads, sequence, HEAD_DIM),
        dtype=carrier.input_dtype,
        device=carrier.q.device,
    )
    _sol_kitchen_full_kernel[(1, q_blocks, heads)](
        carrier.q,
        carrier.k,
        carrier.v,
        carrier.q_scale,
        carrier.k_scale,
        carrier.v_scale,
        prepared.v_sum,
        output,
        sequence,
        padded,
        KV_BLOCKS=kv_blocks,
        softmax_scale=float(carrier.attention_scale),
        Q_TILE_=Q_TILE,
        KV_TILE_=KV_TILE,
        D=HEAD_DIM,
        LOG2E_=LOG2E,
        S_U8_OFFSET_=S_U8_OFFSET,
        OUTPUT_BF16=carrier.input_dtype == torch.bfloat16,
    )
    return output


def launch_full_route(carrier):
    '''Prepare V sums and execute the full-route pointer kernel.'''
    return launch_prepared(prepare_carrier(carrier))


__all__ = [
    'PreparedSolKitchenCarrier',
    'SOL_POINTER_CONFIGS',
    'TRITON_AVAILABLE',
    'TritonSolPointerError',
    'kernel_contract',
    'launch_full_route',
    'launch_prepared',
    'prepare_carrier',
]
