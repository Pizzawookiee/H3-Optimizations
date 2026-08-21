'''Autotuned fixed-density INT8 Triton sparse attention kernels.'''

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    _AUTOTUNE_CONFIGS = [
        triton.Config({'BLOCK_M': 16, 'PIPE_STAGES': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'PIPE_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'PIPE_STAGES': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'PIPE_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'PIPE_STAGES': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'PIPE_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'PIPE_STAGES': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'PIPE_STAGES': 2}, num_warps=8, num_stages=2),
    ]

    def _autotune(configs, key):
        try:
            return triton.autotune(configs=configs, key=key, cache_results=True)
        except TypeError:  # older Triton
            return triton.autotune(configs=configs, key=key)

    @triton.jit
    def _probability_int8_pv(probabilities, v_int8, v_sum, v_scale):
        p_max = tl.max(probabilities, axis=1)
        p_scale = tl.maximum(p_max / 255.0, 1.0e-8)
        p_code = probabilities / p_scale[:, None] + 0.5
        p_code = tl.minimum(tl.maximum(p_code, 0.0), 255.0)
        p_signed = (p_code - 128.0).to(tl.int8)
        pv_i32 = tl.dot(p_signed, v_int8, out_dtype=tl.int32)
        pv_i32 += 128 * v_sum[None, :]
        return pv_i32.to(tl.float32) * p_scale[:, None] * v_scale[None, :]

    @_autotune(_AUTOTUNE_CONFIGS, key=['KV_BLOCKS', 'V_SCALE_GROUP'])
    @triton.jit
    def _dense_kernel(
        Q, K, V, Q_SCALE, K_SCALE, V_SCALE, V_SUM, O,
        sequence, heads,
        KV_BLOCKS: tl.constexpr,
        softmax_scale: tl.constexpr,
        Q_TILE_: tl.constexpr,
        KV_TILE_: tl.constexpr,
        D: tl.constexpr,
        V_SCALE_GROUP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
    ):
        work = tl.program_id(0).to(tl.int64)
        bh = tl.program_id(1).to(tl.int64)
        subblocks = Q_TILE_ // BLOCK_M
        q_block = work // subblocks
        q_sub = work - q_block * subblocks
        batch = bh // heads
        head = bh - batch * heads
        q_rows = q_block * Q_TILE_ + q_sub * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        kv_rows = tl.arange(0, KV_TILE_).to(tl.int64)
        dims = tl.arange(0, D).to(tl.int64)
        q_mask = q_rows < sequence
        hnd_base = (batch * heads + head) * sequence * D
        q_tiles = (sequence + Q_TILE_ - 1) // Q_TILE_
        q = tl.load(
            Q + hnd_base + q_rows[:, None] * D + dims[None, :],
            mask=q_mask[:, None], other=0,
        ).to(tl.int8)
        q_scale = tl.load(Q_SCALE + bh * q_tiles + q_block).to(tl.float32)
        m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, D), tl.float32)
        v_groups = D // V_SCALE_GROUP

        for key_block in tl.range(0, KV_BLOCKS, num_stages=PIPE_STAGES):
            k_positions = key_block * KV_TILE_ + kv_rows
            k_mask = k_positions < sequence
            k = tl.load(
                K + hnd_base + k_positions[None, :] * D + dims[:, None],
                mask=k_mask[None, :], other=0,
            ).to(tl.int8)
            score_i32 = tl.dot(q, k, out_dtype=tl.int32)
            k_scale = tl.load(K_SCALE + bh * KV_BLOCKS + key_block).to(tl.float32)
            logits = score_i32.to(tl.float32) * (
                q_scale * k_scale * softmax_scale * 1.4426950408889634
            )
            logits = tl.where(k_mask[None, :], logits, -float('inf'))
            local_m = tl.max(logits, axis=1)
            new_m = tl.maximum(m_i, local_m)
            p = tl.math.exp2(logits - new_m[:, None])
            alpha = tl.math.exp2(m_i - new_m)
            v_int8 = tl.load(
                V + hnd_base + k_positions[:, None] * D + dims[None, :],
                mask=k_mask[:, None], other=0,
            ).to(tl.int8)
            v_sum = tl.load(
                V_SUM + (bh * KV_BLOCKS + key_block) * D + dims
            ).to(tl.int32)
            v_scale = tl.load(
                V_SCALE
                + (bh * KV_BLOCKS + key_block) * v_groups
                + dims // V_SCALE_GROUP
            ).to(tl.float32)
            acc = acc * alpha[:, None]
            acc += _probability_int8_pv(p, v_int8, v_sum, v_scale)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = new_m

        tl.store(
            O + hnd_base + q_rows[:, None] * D + dims[None, :],
            (acc / l_i[:, None]).to(O.type.element_ty),
            mask=q_mask[:, None],
        )

    @_autotune(
        _AUTOTUNE_CONFIGS,
        key=['N_SELECTED', 'KV_BLOCKS', 'V_SCALE_GROUP'],
    )
    @triton.jit
    def _sparse_kernel(
        Q, K, V, Q_SCALE, K_SCALE, V_SCALE, V_SUM, KV_INDICES, O,
        sequence, heads, SPARSE_Q_START, SPARSE_Q_TILES,
        N_SELECTED: tl.constexpr,
        KV_BLOCKS: tl.constexpr,
        softmax_scale: tl.constexpr,
        Q_TILE_: tl.constexpr,
        KV_TILE_: tl.constexpr,
        D: tl.constexpr,
        V_SCALE_GROUP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
    ):
        work = tl.program_id(0).to(tl.int64)
        bh = tl.program_id(1).to(tl.int64)
        subblocks = Q_TILE_ // BLOCK_M
        sparse_local = work // subblocks
        q_sub = work - sparse_local * subblocks
        q_block = SPARSE_Q_START + sparse_local
        batch = bh // heads
        head = bh - batch * heads
        q_rows = q_block * Q_TILE_ + q_sub * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        kv_rows = tl.arange(0, KV_TILE_).to(tl.int64)
        dims = tl.arange(0, D).to(tl.int64)
        q_mask = q_rows < sequence
        hnd_base = (batch * heads + head) * sequence * D
        q_tiles = (sequence + Q_TILE_ - 1) // Q_TILE_
        q = tl.load(
            Q + hnd_base + q_rows[:, None] * D + dims[None, :],
            mask=q_mask[:, None], other=0,
        ).to(tl.int8)
        q_scale = tl.load(Q_SCALE + bh * q_tiles + q_block).to(tl.float32)
        m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, D), tl.float32)
        v_groups = D // V_SCALE_GROUP
        route_base = (bh * SPARSE_Q_TILES + sparse_local) * N_SELECTED

        for slot in tl.range(0, N_SELECTED, num_stages=PIPE_STAGES):
            key_block = tl.load(KV_INDICES + route_base + slot).to(tl.int32)
            k_positions = key_block * KV_TILE_ + kv_rows
            k_mask = k_positions < sequence
            k = tl.load(
                K + hnd_base + k_positions[None, :] * D + dims[:, None],
                mask=k_mask[None, :], other=0,
            ).to(tl.int8)
            score_i32 = tl.dot(q, k, out_dtype=tl.int32)
            k_scale = tl.load(K_SCALE + bh * KV_BLOCKS + key_block).to(tl.float32)
            logits = score_i32.to(tl.float32) * (
                q_scale * k_scale * softmax_scale * 1.4426950408889634
            )
            logits = tl.where(k_mask[None, :], logits, -float('inf'))
            local_m = tl.max(logits, axis=1)
            new_m = tl.maximum(m_i, local_m)
            p = tl.math.exp2(logits - new_m[:, None])
            alpha = tl.math.exp2(m_i - new_m)
            v_int8 = tl.load(
                V + hnd_base + k_positions[:, None] * D + dims[None, :],
                mask=k_mask[:, None], other=0,
            ).to(tl.int8)
            v_sum = tl.load(
                V_SUM + (bh * KV_BLOCKS + key_block) * D + dims
            ).to(tl.int32)
            v_scale = tl.load(
                V_SCALE
                + (bh * KV_BLOCKS + key_block) * v_groups
                + dims // V_SCALE_GROUP
            ).to(tl.float32)
            acc = acc * alpha[:, None]
            acc += _probability_int8_pv(p, v_int8, v_sum, v_scale)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = new_m

        tl.store(
            O + hnd_base + q_rows[:, None] * D + dims[None, :],
            (acc / l_i[:, None]).to(O.type.element_ty),
            mask=q_mask[:, None],
        )


def launch_int8_sparse(prepared, spec, output):
    if not TRITON_AVAILABLE:
        raise RuntimeError('INT8 Triton sparse attention requires Triton')
    q_tile = int(spec.q_tile)
    bh = int(prepared.q_int8.shape[0] * prepared.heads)
    kv_blocks = int(prepared.kv_tiles)
    if prepared.dense_q_tiles:
        def dense_grid(meta):
            return (
                int(prepared.dense_q_tiles) * (q_tile // meta['BLOCK_M']),
                bh,
            )
        _dense_kernel[dense_grid](
            prepared.q_int8, prepared.k_int8, prepared.v_int8,
            prepared.q_scale, prepared.k_scale, prepared.v_scale,
            prepared.v_sum, output,
            prepared.sequence, prepared.heads,
            KV_BLOCKS=kv_blocks,
            softmax_scale=spec.head_dim ** -0.5,
            Q_TILE_=spec.q_tile, KV_TILE_=spec.kv_tile, D=spec.head_dim,
            V_SCALE_GROUP=spec.v_scale_group_size,
        )
    if prepared.sparse_q_tiles:
        def sparse_grid(meta):
            return (
                int(prepared.sparse_q_tiles) * (q_tile // meta['BLOCK_M']),
                bh,
            )
        _sparse_kernel[sparse_grid](
            prepared.q_int8, prepared.k_int8, prepared.v_int8,
            prepared.q_scale, prepared.k_scale, prepared.v_scale,
            prepared.v_sum, prepared.kv_indices, output,
            prepared.sequence, prepared.heads,
            prepared.dense_q_tiles, prepared.sparse_q_tiles,
            N_SELECTED=prepared.sparse_selected,
            KV_BLOCKS=kv_blocks,
            softmax_scale=spec.head_dim ** -0.5,
            Q_TILE_=spec.q_tile, KV_TILE_=spec.kv_tile, D=spec.head_dim,
            V_SCALE_GROUP=spec.v_scale_group_size,
        )
    return output
