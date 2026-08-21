'''Chunked Kitchen-backed H3 QKV production for Sparse Sage carriers.'''

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False

from ...qkv.chunked import project_chunk_hnd
from .fused_qkv import (
    FusedQKVError,
    HEAD_DIM,
    PreparedFusedQKV,
    sparse_fused_qkv_contract_mismatch,
    validate_prepared_fused_qkv,
)


CHUNK_ROWS = 4096


if TRITON_AVAILABLE:

    @triton.jit
    def _pack_sparse_qk_chunk_kernel(
        x_ptr,
        output_ptr,
        scale_ptr,
        summary_ptr,
        x_b,
        x_h,
        x_n,
        output_b,
        output_h,
        output_n,
        scale_b,
        scale_h,
        summary_b,
        summary_h,
        summary_n,
        row_start,
        block_start,
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
            x_ptr
            + batch * x_b
            + head * x_h
            + rows[:, None] * x_n
            + columns[None, :]
        )
        value = tl.load(source, mask=row_mask[:, None], other=0.0).to(tl.float32)
        scale = (
            tl.max(
                tl.max(
                    tl.where(row_mask[:, None], tl.abs(value), 0.0),
                    axis=1,
                ),
                axis=0,
            )
            / 127.0
            + 1e-7
        )
        quantized = value / scale
        quantized += 0.5 * tl.where(quantized >= 0, 1.0, -1.0)

        destination_rows = row_start + rows
        destination = (
            output_ptr
            + batch * output_b
            + head * output_h
            + destination_rows[:, None] * output_n
            + columns[None, :]
        )
        tl.store(destination, quantized.to(tl.int8), mask=row_mask[:, None])

        destination_block = block_start + block
        tl.store(
            scale_ptr
            + batch * scale_b
            + head * scale_h
            + destination_block,
            scale,
        )
        count = tl.maximum(
            tl.minimum(chunk_sequence - block * block_size, block_size),
            1,
        )
        summary = tl.sum(
            tl.where(row_mask[:, None], value, 0.0), axis=0
        ) / count
        summary_destination = (
            summary_ptr
            + batch * summary_b
            + head * summary_h
            + destination_block * summary_n
            + columns
        )
        tl.store(summary_destination, summary)


def pack_sparse_qk_chunk_into(
    x,
    output,
    scales,
    summaries,
    *,
    row_start,
    block_size,
):
    if not TRITON_AVAILABLE:
        raise FusedQKVError('chunked Sparse Sage QKV requires Triton')
    if not x.is_cuda or x.ndim != 4 or x.stride(-1) != 1:
        raise FusedQKVError(
            'chunked Sparse Sage Q/K input must be a CUDA HND tensor'
        )
    batch, heads, chunk_sequence, head_dim = x.shape
    row_start = int(row_start)
    block_size = int(block_size)
    if batch != 1 or head_dim != HEAD_DIM or chunk_sequence <= 0:
        raise FusedQKVError('chunked Sparse Sage Q/K input shape is invalid')
    if row_start < 0 or block_size <= 0 or row_start % block_size:
        raise FusedQKVError('chunked Sparse Sage Q/K tile offset is invalid')
    expected_output = (batch, heads, output.shape[-2], head_dim)
    if (
        tuple(output.shape) != expected_output
        or output.dtype != torch.int8
        or not output.is_contiguous()
        or output.device != x.device
        or row_start + chunk_sequence > output.shape[-2]
    ):
        raise FusedQKVError('chunked Sparse Sage Q/K destination is invalid')
    total_blocks = (output.shape[-2] + block_size - 1) // block_size
    if (
        tuple(scales.shape) != (batch, heads, total_blocks)
        or scales.dtype != torch.float32
        or not scales.is_contiguous()
        or scales.device != x.device
        or tuple(summaries.shape) != (batch, heads, total_blocks, head_dim)
        or summaries.dtype != x.dtype
        or not summaries.is_contiguous()
        or summaries.device != x.device
    ):
        raise FusedQKVError(
            'chunked Sparse Sage Q/K scale or summary destination is invalid'
        )
    if chunk_sequence % block_size and row_start + chunk_sequence != output.shape[-2]:
        raise FusedQKVError(
            'only the final Sparse Sage Q/K chunk may contain a partial tile'
        )

    blocks = (chunk_sequence + block_size - 1) // block_size
    _pack_sparse_qk_chunk_kernel[(blocks, heads, batch)](
        x,
        output,
        scales,
        summaries,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        scales.stride(0),
        scales.stride(1),
        summaries.stride(0),
        summaries.stride(1),
        summaries.stride(2),
        row_start,
        row_start // block_size,
        chunk_sequence=chunk_sequence,
        head_dim=head_dim,
        block_size=block_size,
    )


def _validate_chunk_rows(chunk_rows, q_tile, kv_tile):
    chunk_rows = int(chunk_rows)
    alignment = math.lcm(int(q_tile), int(kv_tile))
    if chunk_rows <= 0 or chunk_rows % alignment:
        raise FusedQKVError(
            'chunked Sparse Sage QKV rows must be a positive multiple of %d'
            % alignment
        )
    return chunk_rows


def _assemble_chunked_sparse_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    spec,
    chunk_rows,
    packer,
):
    mismatch = sparse_fused_qkv_contract_mismatch(spec)
    if mismatch is not None:
        raise FusedQKVError(
            'chunked QKV does not match the Sparse Sage carrier contract: %s'
            % mismatch
        )
    chunk_rows = _validate_chunk_rows(
        chunk_rows,
        spec.q_tile,
        spec.kv_tile,
    )
    sequence = int(x.shape[0])
    heads = int(module.heads)
    head_dim = int(module.head_dim)
    if sequence <= 0 or head_dim != HEAD_DIM:
        raise FusedQKVError('chunked Sparse Sage QKV requires H3 head_dim 128')

    shape = (1, heads, sequence, head_dim)
    q_blocks = (sequence + int(spec.q_tile) - 1) // int(spec.q_tile)
    k_blocks = (sequence + int(spec.kv_tile) - 1) // int(spec.kv_tile)
    q_int8 = torch.empty(shape, dtype=torch.int8, device=x.device)
    k_int8 = torch.empty(shape, dtype=torch.int8, device=x.device)
    v = torch.empty(shape, dtype=x.dtype, device=x.device)
    q_scale = torch.empty((1, heads, q_blocks), dtype=torch.float32, device=x.device)
    k_scale = torch.empty((1, heads, k_blocks), dtype=torch.float32, device=x.device)
    q_summary = torch.empty(
        (1, heads, q_blocks, head_dim), dtype=x.dtype, device=x.device
    )
    k_summary = torch.empty(
        (1, heads, k_blocks, head_dim), dtype=x.dtype, device=x.device
    )

    for start in range(0, sequence, chunk_rows):
        end = min(start + chunk_rows, sequence)
        q, k, chunk_v = project_chunk_hnd(
            module,
            x,
            rope_freqs,
            start,
            end,
        )
        packer(
            q,
            q_int8,
            q_scale,
            q_summary,
            row_start=start,
            block_size=spec.q_tile,
        )
        packer(
            k,
            k_int8,
            k_scale,
            k_summary,
            row_start=start,
            block_size=spec.kv_tile,
        )
        v[:, :, start:end, :].copy_(chunk_v)
        del q, k, chunk_v

    return validate_prepared_fused_qkv(
        PreparedFusedQKV(
            q_int8=q_int8,
            q_scale=q_scale,
            k_int8=k_int8,
            k_scale=k_scale,
            v=v,
            q_summary=q_summary,
            k_summary=k_summary,
            output_dtype=x.dtype,
            sequence=sequence,
            heads=heads,
            head_dim=head_dim,
            layer_index=int(layer_index),
            smooth_k=False,
        )
    )


def run_chunked_sparse_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    spec,
    chunk_rows=CHUNK_ROWS,
):
    import comfy.model_management

    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise FusedQKVError(
            'chunked Sparse Sage QKV requires a rank-2 CUDA BF16 input'
        )
    if comfy.model_management.in_training:
        raise FusedQKVError('chunked Sparse Sage QKV is inference-only')
    if rope_freqs is not None and (
        rope_freqs.ndim != 6
        or tuple(rope_freqs.shape[:3]) != (1, x.shape[0], 1)
        or rope_freqs.device != x.device
    ):
        raise FusedQKVError('chunked Sparse Sage QKV received invalid RoPE')
    return _assemble_chunked_sparse_qkv(
        module,
        x,
        rope_freqs,
        layer_index=layer_index,
        spec=spec,
        chunk_rows=chunk_rows,
        packer=pack_sparse_qk_chunk_into,
    )


class ChunkedSparseQKVProjector:
    name = 'chunked_sparse_sage_qkv'
    qk_format = 'sparge_block_int8'

    def __init__(self, spec, chunk_rows=CHUNK_ROWS):
        self.spec = spec
        self.chunk_rows = _validate_chunk_rows(
            chunk_rows,
            spec.q_tile,
            spec.kv_tile,
        )

    @property
    def installation_signature(self):
        return (
            self.name,
            self.qk_format,
            self.chunk_rows,
            self.spec.signature,
        )

    def project(
        self,
        module,
        x,
        rope_freqs,
        *,
        layer_index,
        transformer_options,
    ):
        del transformer_options
        return run_chunked_sparse_qkv(
            module,
            x,
            rope_freqs,
            layer_index=layer_index,
            spec=self.spec,
            chunk_rows=self.chunk_rows,
        )
