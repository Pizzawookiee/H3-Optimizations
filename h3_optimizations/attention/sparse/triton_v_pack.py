'''Fast V packing overrides for Triton sparse carriers.'''

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False

from . import triton_qkv_fast as _qkv
from .fused_qkv import HEAD_DIM
from .router import KV_TILE

TritonSparseQKVError = _qkv.TritonSparseQKVError
normalize_v_scale_group_size = _qkv.normalize_v_scale_group_size


if TRITON_AVAILABLE:
    @triton.jit
    def _pack_v_int8_channel_chunk_kernel(
        x_ptr, output_ptr, scale_ptr, sum_ptr,
        x_b, x_h, x_n,
        output_b, output_h, output_n,
        scale_b, scale_h, scale_n,
        sum_b, sum_h, sum_n,
        row_start, block_start,
        chunk_sequence: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        block = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1).to(tl.int64)
        batch = tl.program_id(2).to(tl.int64)
        rows = block * block_size + tl.arange(0, block_size).to(tl.int64)
        columns = tl.arange(0, head_dim).to(tl.int64)
        row_mask = rows < chunk_sequence
        source = (
            x_ptr + batch * x_b + head * x_h
            + rows[:, None] * x_n + columns[None, :]
        )
        value = tl.load(source, mask=row_mask[:, None], other=0.0).to(tl.float32)
        scale = (
            tl.max(tl.where(row_mask[:, None], tl.abs(value), 0.0), axis=0)
            / 127.0 + 1e-7
        )
        quantized_f = value / scale[None, :]
        quantized_f += 0.5 * tl.where(quantized_f >= 0, 1.0, -1.0)
        quantized = quantized_f.to(tl.int8)
        destination_rows = row_start + rows
        destination = (
            output_ptr + batch * output_b + head * output_h
            + destination_rows[:, None] * output_n + columns[None, :]
        )
        tl.store(destination, quantized, mask=row_mask[:, None])
        destination_block = block_start + block
        tl.store(
            scale_ptr + batch * scale_b + head * scale_h
            + destination_block * scale_n + columns,
            scale,
        )
        channel_sum = tl.sum(
            tl.where(row_mask[:, None], quantized.to(tl.int32), 0), axis=0
        )
        tl.store(
            sum_ptr + batch * sum_b + head * sum_h
            + destination_block * sum_n + columns,
            channel_sum,
        )


def pack_triton_v_chunk_into(
    x, output, scales, sums, *, row_start,
    block_size=KV_TILE, v_scale_group_size=1,
):
    if not TRITON_AVAILABLE:
        raise TritonSparseQKVError('chunked Triton sparse QKV requires Triton')
    if not x.is_cuda or x.ndim != 4 or x.stride(-1) != 1:
        raise TritonSparseQKVError('Triton sparse V input must be a CUDA HND tensor')
    batch, heads, chunk_sequence, head_dim = x.shape
    row_start = int(row_start)
    block_size = int(block_size)
    group = normalize_v_scale_group_size(v_scale_group_size)
    groups = head_dim // group
    if batch != 1 or head_dim != HEAD_DIM or chunk_sequence <= 0:
        raise TritonSparseQKVError('Triton sparse V input shape is invalid')
    if row_start < 0 or block_size <= 0 or row_start % block_size:
        raise TritonSparseQKVError('Triton sparse V tile offset is invalid')
    if (
        output.dtype != torch.int8
        or tuple(output.shape[:2]) != (batch, heads)
        or output.shape[-1] != head_dim
        or not output.is_contiguous()
        or output.device != x.device
        or row_start + chunk_sequence > output.shape[-2]
    ):
        raise TritonSparseQKVError('Triton sparse V destination is invalid')
    total_blocks = (output.shape[-2] + block_size - 1) // block_size
    if (
        tuple(scales.shape) != (batch, heads, total_blocks, groups)
        or scales.dtype != torch.float32
        or not scales.is_contiguous()
        or scales.device != x.device
    ):
        raise TritonSparseQKVError('Triton sparse V scale destination is invalid')
    if (
        tuple(sums.shape) != (batch, heads, total_blocks, head_dim)
        or sums.dtype != torch.int32
        or not sums.is_contiguous()
        or sums.device != x.device
    ):
        raise TritonSparseQKVError('Triton sparse V sum destination is invalid')
    if chunk_sequence % block_size and row_start + chunk_sequence != output.shape[-2]:
        raise TritonSparseQKVError(
            'only the final Triton sparse V chunk may contain a partial tile'
        )

    blocks = (chunk_sequence + block_size - 1) // block_size
    common = (
        x, output, scales, sums,
        x.stride(0), x.stride(1), x.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        scales.stride(0), scales.stride(1), scales.stride(2),
        sums.stride(0), sums.stride(1), sums.stride(2),
        row_start, row_start // block_size,
    )
    if group == 1:
        _pack_v_int8_channel_chunk_kernel[(blocks, heads, batch)](
            *common,
            chunk_sequence=chunk_sequence,
            head_dim=head_dim,
            block_size=block_size,
        )
    else:
        _qkv._pack_v_int8_grouped_chunk_kernel[(blocks, heads, batch * groups)](
            *common,
            groups=groups,
            chunk_sequence=chunk_sequence,
            head_dim=head_dim,
            block_size=block_size,
            group_size=group,
        )
