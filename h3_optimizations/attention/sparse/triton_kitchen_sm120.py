'''Conservative SM120 execution for Kitchen-parity Triton attention.

The generic Kitchen-parity kernel uses a runtime-loaded ``valid`` value as the
upper bound of a loop containing INT8 ``tl.dot`` operations. The pre-PR39
Triton path that is known to run on consumer Blackwell instead specializes the
selected KV count at compile time. This module keeps the exact Kitchen carrier
and probability/value math, but restores compile-time loop bounds on SM120.
'''

from __future__ import annotations

import torch

from .triton_kitchen import (
    HEAD_DIM,
    KV_TILE,
    LOG2E,
    Q_TILE,
    S_U8_OFFSET,
    TRITON_AVAILABLE,
    TritonKitchenBackend,
    TritonKitchenError,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - environment dependent
    triton = None
    tl = None


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

    # Keep the Blackwell fallback deliberately conservative. The old Triton
    # implementation is known to run on this hardware, but PR39 introduced a
    # different loop shape. One pipeline stage avoids making the workaround
    # depend on Blackwell's more aggressive dot-loop scheduling.
    _SM120_CONFIGS = [
        triton.Config({'BLOCK_M': 16}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=1),
    ]

    def _autotune(configs, key):
        try:
            return triton.autotune(configs=configs, key=key, cache_results=True)
        except TypeError:  # older Triton
            return triton.autotune(configs=configs, key=key)

    @_autotune(
        _SM120_CONFIGS,
        key=['N_SELECTED', 'KV_BLOCKS', 'OUTPUT_BF16'],
    )
    @triton.jit
    def _kitchen_sparse_static_kernel(
        Q,
        K,
        V,
        Q_SCALE,
        K_SCALE,
        V_SCALE,
        LUT,
        O,
        sequence,
        heads,
        padded_sequence,
        q_tiles,
        Q_BLOCK_START: tl.constexpr,
        Q_BLOCK_COUNT: tl.constexpr,
        N_SELECTED: tl.constexpr,
        KV_BLOCKS: tl.constexpr,
        softmax_scale: tl.constexpr,
        Q_TILE_: tl.constexpr,
        KV_TILE_: tl.constexpr,
        D: tl.constexpr,
        LOG2E_: tl.constexpr,
        S_U8_OFFSET_: tl.constexpr,
        OUTPUT_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        work = tl.program_id(0).to(tl.int64)
        bh = tl.program_id(1).to(tl.int64)
        subblocks = Q_TILE_ // BLOCK_M
        local_q_block = work // subblocks
        q_sub = work - local_q_block * subblocks
        if local_q_block >= Q_BLOCK_COUNT:
            return
        q_block = Q_BLOCK_START + local_q_block

        q_rows = q_block * Q_TILE_ + q_sub * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, D)
        kv_rows = tl.arange(0, KV_TILE_)
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

        m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
        d_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

        route_row = bh * q_tiles + q_block
        lut_base = LUT + route_row * KV_BLOCKS
        key_block = tl.zeros((), dtype=tl.int32)
        k_scale_count = KV_BLOCKS * 4

        # N_SELECTED is constexpr. This is the critical SM120 difference from
        # PR39's runtime-loaded ``valid`` loop bound.
        for slot in tl.range(0, N_SELECTED):
            key_block += tl.load(lut_base + slot).to(tl.int32)
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
            tile_m = tile_max - S_U8_OFFSET_
            new_m = tl.maximum(m_i, tile_m)
            o_scale = tl.math.exp2(m_i - new_m)
            tile_scale = tl.math.exp2(tile_m - new_m)

            probability = tl.math.exp2(logits - tile_m[:, None])
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
            v_sum = tl.sum(v.to(tl.int32), axis=0)
            pv_i32 = tl.dot(p_signed, v, out_dtype=tl.int32)
            pv_i32 += 128 * v_sum[None, :]

            acc = acc * o_scale[:, None]
            acc += pv_i32.to(tl.float32) * tile_scale[:, None]
            d_i = d_i * o_scale
            d_i += tl.sum(p_code, axis=1).to(tl.float32) * tile_scale
            m_i = new_m

        v_scale = tl.load(V_SCALE + bh * D + dims).to(tl.float32)
        output = (acc / d_i[:, None]) * v_scale[None, :]
        tl.store(
            O + hnd_base + q_rows[:, None] * D + dims[None, :],
            output.to(O.type.element_ty),
            mask=q_mask[:, None],
        )


def _route_groups(metadata, q_tiles, kv_blocks):
    try:
        dense_q_tiles = int(metadata['dense_q_tiles'])
        sparse_q_tiles = int(metadata['sparse_q_tiles'])
        pure_video_kv_tiles = int(metadata['pure_video_kv_tiles'])
        retained_video_kv_tiles = int(metadata['retained_video_kv_tiles'])
    except (KeyError, TypeError, ValueError) as exc:
        raise TritonKitchenError(
            'SM120 Kitchen-parity Triton requires fixed-density route metadata'
        ) from exc

    if dense_q_tiles < 0 or sparse_q_tiles < 0:
        raise TritonKitchenError('SM120 route has negative query-tile counts')
    if dense_q_tiles + sparse_q_tiles != int(q_tiles):
        raise TritonKitchenError(
            'SM120 route query-tile counts do not cover the sequence'
        )
    if not 0 <= pure_video_kv_tiles <= int(kv_blocks):
        raise TritonKitchenError('SM120 route has invalid pure-video KV count')
    if not 0 <= retained_video_kv_tiles <= pure_video_kv_tiles:
        raise TritonKitchenError('SM120 route has invalid retained-video KV count')

    groups = []
    if dense_q_tiles:
        groups.append((0, dense_q_tiles, int(kv_blocks)))
    if sparse_q_tiles:
        selected = (
            int(kv_blocks) - pure_video_kv_tiles + retained_video_kv_tiles
        )
        if not 0 < selected <= int(kv_blocks):
            raise TritonKitchenError('SM120 route has invalid sparse KV count')
        groups.append((dense_q_tiles, sparse_q_tiles, selected))
    return groups


def _launch_sm120(carrier, lut, metadata):
    if not TRITON_AVAILABLE:
        raise TritonKitchenError('Kitchen-parity Triton requires Triton')
    sequence = int(carrier.q.shape[2])
    heads = int(carrier.q.shape[1])
    q_tiles = (sequence + Q_TILE - 1) // Q_TILE
    kv_blocks = (sequence + KV_TILE - 1) // KV_TILE
    padded = ((sequence + KV_TILE - 1) // KV_TILE) * KV_TILE
    output = torch.empty(
        (1, heads, sequence, HEAD_DIM),
        dtype=carrier.input_dtype,
        device=carrier.q.device,
    )

    for q_start, q_count, selected in _route_groups(
        metadata, q_tiles, kv_blocks
    ):
        def grid(meta, q_count=q_count):
            return (q_count * (Q_TILE // meta['BLOCK_M']), heads)

        _kitchen_sparse_static_kernel[grid](
            carrier.q,
            carrier.k,
            carrier.v,
            carrier.q_scale,
            carrier.k_scale,
            carrier.v_scale,
            lut,
            output,
            sequence,
            heads,
            padded,
            q_tiles,
            Q_BLOCK_START=q_start,
            Q_BLOCK_COUNT=q_count,
            N_SELECTED=selected,
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


class SM120TritonKitchenBackend(TritonKitchenBackend):
    '''Kitchen-parity Triton with static fixed-density loops for SM120.'''

    def execute(self, prepared):
        from .triton_kitchen import PreparedTritonKitchen

        if not isinstance(prepared, PreparedTritonKitchen):
            raise TritonKitchenError('invalid Kitchen-parity Triton payload')
        try:
            return _launch_sm120(
                prepared.carrier,
                prepared.lut,
                prepared.metadata,
            )
        except Exception as exc:
            raise TritonKitchenError(
                'SM120 Kitchen-parity Triton kernel failed: '
                'layer=%d sequence=%d heads=%d; cause=%s: %s'
                % (
                    prepared.layer_index,
                    prepared.carrier.q.shape[2],
                    prepared.carrier.q.shape[1],
                    type(exc).__name__,
                    exc,
                )
            ) from exc

    def as_status(self):
        status = super().as_status()
        status['sm120_static_selected_loop'] = True
        status['sm120_num_stages'] = 1
        return status
