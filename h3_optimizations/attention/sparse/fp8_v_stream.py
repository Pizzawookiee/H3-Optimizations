"""Low-VRAM construction of Sparse-Sage FP8 V carriers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


class StreamingVError(RuntimeError):
    pass


@triton.jit
def _quantize_transposed_v_kernel(
    src_ptr,
    dst_ptr,
    scale_ptr,
    src_stride_h,
    src_stride_d,
    src_stride_s,
    dst_stride_h,
    dst_stride_d,
    dst_stride_s,
    scale_stride_h,
    scale_stride_d,
    row_start,
    chunk_padded: tl.constexpr,
    head_dim: tl.constexpr,
    scale_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channel = tl.program_id(0)
    block = tl.program_id(1)
    head = channel // head_dim
    dim = channel % head_dim
    rows = block * BLOCK + tl.arange(0, BLOCK)
    mask = rows < chunk_padded
    src_offsets = (
        head * src_stride_h
        + dim * src_stride_d
        + rows * src_stride_s
    )
    dst_offsets = (
        head * dst_stride_h
        + dim * dst_stride_d
        + (row_start + rows) * dst_stride_s
    )
    scale_offset = head * scale_stride_h + dim * scale_stride_d
    value = tl.load(src_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + scale_offset).to(tl.float32)
    scale = tl.maximum(scale, 1.0e-20)
    value = value / scale
    value = tl.maximum(tl.minimum(value, scale_max), -scale_max)
    tl.store(dst_ptr + dst_offsets, value, mask=mask)


def update_v_amax(amax: torch.Tensor, v: torch.Tensor):
    if v.ndim != 4:
        raise StreamingVError("V must be an HND rank-4 tensor")
    chunk_amax = v.abs().amax(dim=2).to(torch.float32)
    torch.maximum(amax, chunk_amax, out=amax)


def finalize_v_scale(amax, scale_max):
    return amax.div(float(scale_max)).clamp_min_(1.0e-20)


def allocate_v_carrier(*, sequence, heads, head_dim, device):
    padded = (int(sequence) + 127) // 128 * 128
    return torch.zeros(
        (1, int(heads), int(head_dim), padded),
        dtype=torch.float8_e4m3fn,
        device=device,
    )


def pack_v_chunk(
    v,
    carrier,
    scale,
    *,
    row_start,
    sequence,
    fused_v_ops,
    scale_max,
):
    if not v.is_cuda or v.ndim != 4:
        raise StreamingVError("streaming V requires a rank-4 CUDA HND tensor")
    batch, heads, rows, head_dim = v.shape
    if batch != 1:
        raise StreamingVError("released H3 requires V batch size 1")

    row_start = int(row_start)
    sequence = int(sequence)
    if row_start % 128:
        raise StreamingVError("streaming V row_start must be 128 aligned")
    if row_start + rows > sequence:
        raise StreamingVError("streaming V chunk exceeds sequence")

    chunk_padded = (int(rows) + 127) // 128 * 128
    global_padded = (sequence + 127) // 128 * 128
    if row_start + chunk_padded > global_padded:
        raise StreamingVError("streaming V padded chunk exceeds carrier")

    expected = (1, heads, head_dim, global_padded)
    if tuple(carrier.shape) != expected:
        raise StreamingVError("streaming V carrier shape mismatch")
    if carrier.dtype != torch.float8_e4m3fn:
        raise StreamingVError("streaming V destination must be FP8 E4M3")
    if tuple(scale.shape) != (1, heads, head_dim):
        raise StreamingVError("streaming V scale shape mismatch")

    transposed = torch.empty(
        (1, heads, head_dim, chunk_padded),
        dtype=v.dtype,
        device=v.device,
    )
    fused_v_ops.transpose_pad_permute_cuda(v, transposed, 1)

    grid = (heads * head_dim, triton.cdiv(chunk_padded, 256))
    _quantize_transposed_v_kernel[grid](
        transposed,
        carrier,
        scale,
        transposed.stride(1),
        transposed.stride(2),
        transposed.stride(3),
        carrier.stride(1),
        carrier.stride(2),
        carrier.stride(3),
        scale.stride(1),
        scale.stride(2),
        row_start,
        chunk_padded=chunk_padded,
        head_dim=head_dim,
        scale_max=float(scale_max),
        BLOCK=256,
    )
    del transposed
