'''Minimal Sol-shaped Triton attention over the Kitchen INT8 carrier.

Each program owns one 64-query x one-head x full-D128 tile. Dense query tiles
walk KV implicitly; sparse query tiles consume a compact absolute route. This
module remains a benchmark control and is not selected by production.

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


@dataclass
class PreparedSolKitchenCarrier:
    carrier: object
    v_sum: torch.Tensor
    sparse_lut: torch.Tensor
    dense_q_tiles: int
    sparse_q_tiles: int
    sparse_selected: int
    layer_index: int = -1
    metadata: dict | None = None


def kernel_contract():
    '''Return the static experiment contract without requiring CUDA.'''
    return {
        'q_tile': Q_TILE,
        'kv_tile': KV_TILE,
        'head_dim': HEAD_DIM,
        'program': 'one_64q_tile_x_one_head_x_full_d128',
        'route': 'dense_implicit_plus_sparse_absolute',
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

    @_autotune(
        _CONFIGS,
        key=['KV_BLOCKS', 'N_SELECTED', 'USE_ROUTE', 'OUTPUT_BF16'],
    )
    @triton.jit
    def _sol_kitchen_kernel(
        Q,
        K,
        V,
        Q_SCALE,
        K_SCALE,
        V_SCALE,
        V_SUM,
        LUT,
        O,
        sequence,
        padded_sequence,
        Q_BLOCK_START,
        Q_BLOCK_COUNT,
        KV_BLOCKS: tl.constexpr,
        N_SELECTED: tl.constexpr,
        USE_ROUTE: tl.constexpr,
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
        local_q_block = tl.program_id(1)
        q_block = Q_BLOCK_START + local_q_block
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

        for route_position in tl.range(0, N_SELECTED):
            if USE_ROUTE:
                route_offset = (
                    (bh * Q_BLOCK_COUNT + local_q_block) * N_SELECTED
                    + route_position
                )
                key_block = tl.load(LUT + route_offset)
            else:
                key_block = route_position
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
            v_positions = _v_perm16(k_positions)
            v = tl.load(
                V
                + (bh * D + dims[None, :]) * padded_sequence
                + v_positions[:, None],
            ).to(tl.int8)
            v_sum = tl.load(
                V_SUM + (bh * KV_BLOCKS + key_block) * D + dims
            ).to(tl.int32)

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


def _prepare_v_sum(carrier, heads, kv_blocks, padded):
    if not TRITON_AVAILABLE:
        raise TritonSolPointerError('Sol-shaped Kitchen backend requires Triton')
    if not carrier.q.is_cuda:
        raise TritonSolPointerError('Sol-shaped Kitchen backend requires CUDA')
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
    return v_sum


def _compact_route(lut, valid, metadata):
    if lut.ndim != 4 or valid.ndim != 3:
        raise TritonSolPointerError('Sol-shaped Kitchen route ranks are invalid')
    if lut.dtype != torch.int32 or valid.dtype != torch.int32:
        raise TritonSolPointerError('Sol-shaped Kitchen route must be INT32')
    if not lut.is_contiguous() or not valid.is_contiguous():
        raise TritonSolPointerError('Sol-shaped Kitchen route must be contiguous')
    batch, heads, q_tiles, kv_tiles = (int(value) for value in lut.shape)
    if tuple(valid.shape) != (batch, heads, q_tiles):
        raise TritonSolPointerError('Sol-shaped Kitchen route shapes differ')
    dense_q_tiles = int(metadata['dense_q_tiles'])
    sparse_q_tiles = int(metadata['sparse_q_tiles'])
    if dense_q_tiles + sparse_q_tiles != q_tiles:
        raise TritonSolPointerError('Sol-shaped Kitchen Q route geometry differs')
    if sparse_q_tiles:
        pure_video_kv_tiles = int(metadata['pure_video_kv_tiles'])
        retained_video_kv_tiles = int(metadata['retained_video_kv_tiles'])
        sparse_selected = (
            kv_tiles - pure_video_kv_tiles + retained_video_kv_tiles
        )
        if not 0 < sparse_selected < kv_tiles:
            raise TritonSolPointerError('Sol-shaped Kitchen sparse route is invalid')
        sparse_delta = lut[
            ...,
            dense_q_tiles:dense_q_tiles + sparse_q_tiles,
            :sparse_selected,
        ]
        sparse_lut = torch.cumsum(
            sparse_delta, dim=-1, dtype=torch.int32
        ).contiguous()
    else:
        sparse_selected = 0
        sparse_lut = lut.new_empty((batch, heads, 0, 0))
    return sparse_lut, dense_q_tiles, sparse_q_tiles, sparse_selected


def prepare_carrier(carrier):
    '''Prepare a full-route Kitchen carrier for direct benchmarks.'''
    carrier, _sequence, heads, q_blocks, kv_blocks, padded = _shape(carrier)
    v_sum = _prepare_v_sum(carrier, heads, kv_blocks, padded)
    return PreparedSolKitchenCarrier(
        carrier=carrier,
        v_sum=v_sum,
        sparse_lut=torch.empty(
            (1, heads, 0, 0), dtype=torch.int32, device=carrier.q.device
        ),
        dense_q_tiles=q_blocks,
        sparse_q_tiles=0,
        sparse_selected=0,
        metadata={'route_format': 'implicit_full'},
    )


def prepare_routed_carrier(carrier, lut, valid, *, layer_index, metadata):
    '''Prepare dense and compact sparse launch groups from a delta route.'''
    carrier, _sequence, heads, _q_blocks, kv_blocks, padded = _shape(carrier)
    if lut.device != carrier.q.device or valid.device != carrier.q.device:
        raise TritonSolPointerError('Sol-shaped Kitchen route device differs')
    sparse_lut, dense_q_tiles, sparse_q_tiles, sparse_selected = _compact_route(
        lut, valid, metadata
    )
    route_metadata = dict(metadata)
    route_metadata.update(
        {
            'sparse_backend': 'triton_sol_pointer_int8',
            'route_format': 'dense_implicit_plus_sparse_absolute_int32',
            'program_shape': 'one_64q_tile_x_one_head_x_full_d128',
        }
    )
    return PreparedSolKitchenCarrier(
        carrier=carrier,
        v_sum=_prepare_v_sum(carrier, heads, kv_blocks, padded),
        sparse_lut=sparse_lut,
        dense_q_tiles=dense_q_tiles,
        sparse_q_tiles=sparse_q_tiles,
        sparse_selected=sparse_selected,
        layer_index=int(layer_index),
        metadata=route_metadata,
    )


def launch_prepared(prepared):
    '''Execute the dense and sparse pointer launch groups.'''
    if not TRITON_AVAILABLE:
        raise TritonSolPointerError('Sol-shaped Kitchen backend requires Triton')
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
        raise TritonSolPointerError('Sol-shaped Kitchen backend requires CUDA')
    if prepared.dense_q_tiles + prepared.sparse_q_tiles != q_blocks:
        raise TritonSolPointerError('prepared Sol-shaped Q geometry differs')
    expected_sparse_lut = (
        1,
        heads,
        prepared.sparse_q_tiles,
        prepared.sparse_selected,
    )
    if tuple(prepared.sparse_lut.shape) != expected_sparse_lut:
        raise TritonSolPointerError('prepared Sol-shaped sparse LUT shape differs')
    if (
        prepared.sparse_lut.dtype != torch.int32
        or not prepared.sparse_lut.is_contiguous()
        or prepared.sparse_lut.device != carrier.q.device
    ):
        raise TritonSolPointerError('prepared Sol-shaped sparse LUT is invalid')

    output = torch.empty(
        (1, heads, sequence, HEAD_DIM),
        dtype=carrier.input_dtype,
        device=carrier.q.device,
    )
    def launch_group(q_start, q_count, selected, use_route):
        if not q_count:
            return
        _sol_kitchen_kernel[(1, q_count, heads)](
            carrier.q,
            carrier.k,
            carrier.v,
            carrier.q_scale,
            carrier.k_scale,
            carrier.v_scale,
            prepared.v_sum,
            prepared.sparse_lut,
            output,
            sequence,
            padded,
            q_start,
            q_count,
            KV_BLOCKS=kv_blocks,
            N_SELECTED=selected,
            USE_ROUTE=use_route,
            softmax_scale=float(carrier.attention_scale),
            Q_TILE_=Q_TILE,
            KV_TILE_=KV_TILE,
            D=HEAD_DIM,
            LOG2E_=LOG2E,
            S_U8_OFFSET_=S_U8_OFFSET,
            OUTPUT_BF16=carrier.input_dtype == torch.bfloat16,
        )

    launch_group(0, prepared.dense_q_tiles, kv_blocks, False)
    launch_group(
        prepared.dense_q_tiles,
        prepared.sparse_q_tiles,
        prepared.sparse_selected,
        True,
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
    '_compact_route',
    'kernel_contract',
    'launch_full_route',
    'launch_prepared',
    'prepare_carrier',
    'prepare_routed_carrier',
]
