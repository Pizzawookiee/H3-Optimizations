"""Triton tensor core for dense SM89 fused-QKV projection."""

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None
    TRITON_AVAILABLE = False

from .dense_fused_qkv_contract import (
    DenseFusedQKVError,
    HEAD_DIM,
    Q_TILE,
    K_TILE,
    Q_SCALES_PER_TILE,
    K_SCALES_PER_TILE,
)

if TRITON_AVAILABLE:

    @triton.jit
    def _dense_fused_qkv_kernel(
        x_ptr,
        weight_ptr,
        x_scale_ptr,
        weight_scale_ptr,
        q_norm_ptr,
        k_norm_ptr,
        rope_ptr,
        q_ptr,
        q_scale_ptr,
        k_ptr,
        k_scale_ptr,
        v_ptr,
        sequence: tl.constexpr,
        hidden: tl.constexpr,
        heads: tl.constexpr,
        rope_stride_seq: tl.constexpr,
        rope_stride_dim: tl.constexpr,
        rope_stride_rot: tl.constexpr,
        rope_stride_pair: tl.constexpr,
        epsilon: tl.constexpr,
        has_rope: tl.constexpr,
        KIND: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        block_m = tl.program_id(0)
        head = tl.program_id(1)

        rows = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, BLOCK_N)
        output_col = KIND * heads * BLOCK_N + head * BLOCK_N + dims
        row_mask = rows < sequence

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for start in range(0, hidden, BLOCK_K):
            inner = start + tl.arange(0, BLOCK_K)
            inner_mask = inner < hidden
            x_offsets = rows[:, None].to(tl.int64) * hidden + inner[None, :]
            w_offsets = output_col[:, None].to(tl.int64) * hidden + inner[None, :]
            x = tl.load(
                x_ptr + x_offsets,
                mask=row_mask[:, None] & inner_mask[None, :],
                other=0,
            )
            weight = tl.load(
                weight_ptr + w_offsets,
                mask=inner_mask[None, :],
                other=0,
            )
            accumulator += tl.dot(x, tl.trans(weight), out_dtype=tl.int32)

        row_scale = tl.load(x_scale_ptr + rows, mask=row_mask, other=0.0)
        column_scale = tl.load(weight_scale_ptr + output_col)
        value = accumulator.to(tl.float32)
        value *= row_scale[:, None] * column_scale[None, :]
        value = value.to(tl.bfloat16).to(tl.float32)

        output_offsets = (
            head.to(tl.int64) * sequence * BLOCK_N
            + rows[:, None].to(tl.int64) * BLOCK_N
            + dims[None, :]
        )
        if KIND == 2:
            tl.store(v_ptr + output_offsets, value, mask=row_mask[:, None])
            return

        if KIND == 0:
            norm_weight = tl.load(q_norm_ptr + dims).to(tl.float32)
        else:
            norm_weight = tl.load(k_norm_ptr + dims).to(tl.float32)
        inverse_rms = tl.rsqrt(
            tl.sum(value * value, axis=1) / BLOCK_N + epsilon
        )
        value = value * inverse_rms[:, None] * norm_weight[None, :]
        value = value.to(tl.bfloat16).to(tl.float32)

        if has_rope:
            groups = tl.arange(0, 8)
            pairs = tl.arange(0, 16)
            grouped = tl.reshape(value, (BLOCK_M, 8, 16))
            group = dims // 16
            for rope_group in tl.static_range(0, 3):
                first = tl.sum(
                    grouped * (groups[None, :, None] == rope_group), axis=1
                )
                second = tl.sum(
                    grouped * (groups[None, :, None] == rope_group + 3), axis=1
                )
                rope_base = (
                    rope_ptr
                    + rows[:, None].to(tl.int64) * rope_stride_seq
                    + (rope_group * 16 + pairs[None, :]) * rope_stride_dim
                )
                rope_mask = row_mask[:, None]
                f00 = tl.load(rope_base, mask=rope_mask, other=0.0).to(tl.float32)
                f01 = tl.load(
                    rope_base + rope_stride_pair,
                    mask=rope_mask,
                    other=0.0,
                ).to(tl.float32)
                f10 = tl.load(
                    rope_base + rope_stride_rot,
                    mask=rope_mask,
                    other=0.0,
                ).to(tl.float32)
                f11 = tl.load(
                    rope_base + rope_stride_rot + rope_stride_pair,
                    mask=rope_mask,
                    other=0.0,
                ).to(tl.float32)
                rotated_first = f00 * first + f01 * second
                rotated_second = f10 * first + f11 * second
                first_full = tl.reshape(
                    tl.broadcast_to(
                        rotated_first[:, None, :], (BLOCK_M, 8, 16)
                    ),
                    (BLOCK_M, BLOCK_N),
                )
                second_full = tl.reshape(
                    tl.broadcast_to(
                        rotated_second[:, None, :], (BLOCK_M, 8, 16)
                    ),
                    (BLOCK_M, BLOCK_N),
                )
                value = tl.where(
                    group[None, :] == rope_group,
                    first_full,
                    tl.where(
                        group[None, :] == rope_group + 3,
                        second_full,
                        value,
                    ),
                )
            value = value.to(tl.bfloat16).to(tl.float32)

        absolute = tl.where(row_mask[:, None], tl.abs(value), 0.0)
        if KIND == 0:
            # Sage per-thread Q: each 32-row region has eight scales. Scale
            # lane L covers rows L, L+8, L+16, and L+24 across all 128 dims.
            q_groups = tl.reshape(absolute, (4, 4, 8, BLOCK_N))
            q_scales = (
                tl.max(tl.max(q_groups, axis=3), axis=1) / 127.0 + 1e-7
            )
            q_grid = tl.broadcast_to(
                q_scales[:, None, :, None], (4, 4, 8, BLOCK_N)
            )
            row_scales = tl.reshape(q_grid, (BLOCK_M, BLOCK_N))
            quantized = value / row_scales
            quantized += 0.5 * tl.where(quantized >= 0, 1.0, -1.0)
            tl.store(
                q_ptr + output_offsets,
                quantized.to(tl.int8),
                mask=row_mask[:, None],
            )
            q_scale_count = tl.cdiv(sequence, BLOCK_M) * 32
            q_offsets = (
                head * q_scale_count
                + block_m * 32
                + tl.arange(0, 32)
            )
            tl.store(q_scale_ptr + q_offsets, tl.reshape(q_scales, (32,)))
        else:
            # Sage per-thread K: each 64-row region has four scales. Scale
            # lane L covers adjacent row pair 2L/2L+1 in each 8-row group.
            k_groups = tl.reshape(absolute, (2, 8, 4, 2, BLOCK_N))
            k_scales = (
                tl.max(
                    tl.max(tl.max(k_groups, axis=4), axis=3),
                    axis=1,
                )
                / 127.0
                + 1e-7
            )
            k_grid = tl.broadcast_to(
                k_scales[:, None, :, None, None],
                (2, 8, 4, 2, BLOCK_N),
            )
            row_scales = tl.reshape(k_grid, (BLOCK_M, BLOCK_N))
            quantized = value / row_scales
            quantized += 0.5 * tl.where(quantized >= 0, 1.0, -1.0)
            tl.store(
                k_ptr + output_offsets,
                quantized.to(tl.int8),
                mask=row_mask[:, None],
            )
            k_scale_count = tl.cdiv(sequence, 64) * 4
            k_local = block_m * 8 + tl.arange(0, 8)
            tl.store(
                k_scale_ptr + head * k_scale_count + k_local,
                tl.reshape(k_scales, (8,)),
                mask=k_local < k_scale_count,
            )


def _dense_fused_qkv_tensor_core(
    x_int8,
    qdata,
    x_scale,
    weight_scale,
    q_norm,
    k_norm,
    rope,
    *,
    heads,
    sequence,
    hidden,
    epsilon,
    has_rope,
    rope_strides,
    output_dtype,
):
    if not TRITON_AVAILABLE:
        raise DenseFusedQKVError("dense fused H3 QKV requires Triton")

    shape = (1, heads, sequence, HEAD_DIM)
    q_int8 = torch.empty(shape, dtype=torch.int8, device=x_int8.device)
    k_int8 = torch.empty(shape, dtype=torch.int8, device=x_int8.device)
    v = torch.empty(shape, dtype=output_dtype, device=x_int8.device)
    q_scales = ((sequence + Q_TILE - 1) // Q_TILE) * Q_SCALES_PER_TILE
    k_scales = ((sequence + K_TILE - 1) // K_TILE) * K_SCALES_PER_TILE
    q_scale = torch.empty(
        (1, heads, q_scales), dtype=torch.float32, device=x_int8.device
    )
    k_scale = torch.empty(
        (1, heads, k_scales), dtype=torch.float32, device=x_int8.device
    )

    grid = (triton.cdiv(sequence, Q_TILE), heads)
    for kind in range(3):
        _dense_fused_qkv_kernel[grid](
            x_int8,
            qdata,
            x_scale,
            weight_scale,
            q_norm,
            k_norm,
            rope,
            q_int8,
            q_scale,
            k_int8,
            k_scale,
            v,
            sequence=sequence,
            hidden=hidden,
            heads=heads,
            rope_stride_seq=rope_strides[0],
            rope_stride_dim=rope_strides[1],
            rope_stride_rot=rope_strides[2],
            rope_stride_pair=rope_strides[3],
            epsilon=epsilon,
            has_rope=has_rope,
            KIND=kind,
            BLOCK_M=Q_TILE,
            BLOCK_N=HEAD_DIM,
            BLOCK_K=64,
            num_warps=8,
            num_stages=3,
        )
    return q_int8, q_scale, k_int8, k_scale, v


# Public tensor-only seam for focused CUDA parity and benchmark tests.
dense_fused_qkv_tensor_core = _dense_fused_qkv_tensor_core
