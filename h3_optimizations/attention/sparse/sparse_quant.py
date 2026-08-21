"""Stride-aware per-block Q/K quantization for prepared Sparse Sage."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_blocks(
    x_ptr,
    mean_ptr,
    out_ptr,
    scale_ptr,
    stride_b,
    stride_h,
    stride_n,
    out_b,
    out_h,
    out_n,
    scale_b,
    scale_h,
    mean_b,
    mean_h,
    sequence: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    subtract_mean: tl.constexpr,
):
    block = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    rows = block * block_size + tl.arange(0, block_size).to(tl.int64)
    cols = tl.arange(0, head_dim).to(tl.int64)
    row_mask = rows < sequence
    source = (
        x_ptr
        + batch * stride_b
        + head * stride_h
        + rows[:, None] * stride_n
        + cols[None, :]
    )
    value = tl.load(source, mask=row_mask[:, None], other=0.0).to(tl.float32)
    if subtract_mean:
        mean = tl.load(
            mean_ptr
            + batch * mean_b
            + head * mean_h
            + cols
        )
        value -= mean[None, :]
        value = tl.where(row_mask[:, None], value, 0.0)

    scale = tl.max(tl.max(tl.abs(value), axis=1), axis=0) / 127.0 + 1e-7
    quantized = value / scale
    quantized += 0.5 * tl.where(quantized >= 0, 1, -1)
    destination = (
        out_ptr
        + batch * out_b
        + head * out_h
        + rows[:, None] * out_n
        + cols[None, :]
    )
    tl.store(destination, quantized.to(tl.int8), mask=row_mask[:, None])
    tl.store(
        scale_ptr
        + batch * scale_b
        + head * scale_h
        + block,
        scale,
    )


def _run(x, block_size, mean=None):
    batch, heads, sequence, head_dim = x.shape
    blocks = (sequence + block_size - 1) // block_size
    output = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty((batch, heads, blocks), dtype=torch.float32, device=x.device)
    subtract_mean = mean is not None
    mean = mean if mean is not None else x.new_empty((batch, heads, head_dim))
    _quantize_blocks[(blocks, heads, batch)](
        x,
        mean,
        output,
        scales,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        scales.stride(0),
        scales.stride(1),
        mean.stride(0),
        mean.stride(1),
        sequence=sequence,
        head_dim=head_dim,
        block_size=block_size,
        subtract_mean=subtract_mean,
    )
    return output, scales


def quantize_qk(q, k, q_block, kv_block):
    q_int8, q_scale = _run(q, q_block)
    k_mean = k.mean(dim=-2)
    k_int8, k_scale = _run(k, kv_block, k_mean)
    return q_int8, q_scale, k_int8, k_scale
