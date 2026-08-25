'''Optimized chunked INT8 Q/K/V production for the H3 Triton sparse fallback.'''

from __future__ import annotations

from dataclasses import dataclass
import math
import os

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
from .chunked_qkv import pack_sparse_qk_chunk_into
from .fused_qkv import HEAD_DIM


CHUNK_ROWS = 4096
Q_TILE = 64
KV_TILE = 64
CARRIER_VERSION = 3
V_SCALE_GROUPS = (1, 8, 16, 32, 128)
V_SCALE_GROUP_ENV = 'H3_TRITON_V_SCALE_GROUP'


class TritonSparseQKVError(RuntimeError):
    pass


def normalize_v_scale_group_size(value=None):
    if value is None:
        value = os.environ.get(V_SCALE_GROUP_ENV, '1')
    try:
        group = int(value)
    except (TypeError, ValueError) as exc:
        raise TritonSparseQKVError(
            '%s must be one of %s' % (V_SCALE_GROUP_ENV, V_SCALE_GROUPS)
        ) from exc
    if group not in V_SCALE_GROUPS or HEAD_DIM % group:
        raise TritonSparseQKVError(
            'V scale group must be one of %s; got %r'
            % (V_SCALE_GROUPS, value)
        )
    return group


@dataclass
class PreparedTritonSparseQKV:
    q_int8: torch.Tensor
    q_scale: torch.Tensor
    k_int8: torch.Tensor
    k_scale: torch.Tensor
    v_int8: torch.Tensor
    v_scale: torch.Tensor
    v_sum: torch.Tensor
    q_summary: torch.Tensor
    k_summary: torch.Tensor
    output_dtype: torch.dtype
    sequence: int
    heads: int
    head_dim: int
    layer_index: int
    v_scale_group_size: int = 1
    carrier_version: int = CARRIER_VERSION


def validate_prepared_triton_sparse_qkv(prepared):
    if not isinstance(prepared, PreparedTritonSparseQKV):
        raise TritonSparseQKVError(
            'Triton sparse QKV must be PreparedTritonSparseQKV'
        )
    sequence = int(prepared.sequence)
    heads = int(prepared.heads)
    head_dim = int(prepared.head_dim)
    group = normalize_v_scale_group_size(prepared.v_scale_group_size)
    if sequence <= 0 or heads <= 0 or head_dim != HEAD_DIM:
        raise TritonSparseQKVError('Triton sparse QKV metadata is invalid')
    if int(prepared.carrier_version) != CARRIER_VERSION:
        raise TritonSparseQKVError(
            'unsupported Triton sparse QKV carrier version %r'
            % prepared.carrier_version
        )
    if prepared.output_dtype not in (torch.float16, torch.bfloat16):
        raise TritonSparseQKVError(
            'Triton sparse QKV output dtype must be fp16 or bf16'
        )

    q_blocks = (sequence + Q_TILE - 1) // Q_TILE
    kv_blocks = (sequence + KV_TILE - 1) // KV_TILE
    v_groups = head_dim // group
    hnd = (1, heads, sequence, head_dim)
    contracts = (
        ('q_int8', prepared.q_int8, hnd, torch.int8),
        ('k_int8', prepared.k_int8, hnd, torch.int8),
        ('v_int8', prepared.v_int8, hnd, torch.int8),
        ('q_scale', prepared.q_scale, (1, heads, q_blocks), torch.float32),
        ('k_scale', prepared.k_scale, (1, heads, kv_blocks), torch.float32),
        ('v_scale', prepared.v_scale, (1, heads, kv_blocks, v_groups), torch.float32),
        ('v_sum', prepared.v_sum, (1, heads, kv_blocks, head_dim), torch.int32),
        (
            'q_summary',
            prepared.q_summary,
            (1, heads, q_blocks, head_dim),
            prepared.output_dtype,
        ),
        (
            'k_summary',
            prepared.k_summary,
            (1, heads, kv_blocks, head_dim),
            prepared.output_dtype,
        ),
    )
    device = prepared.q_int8.device
    for name, tensor, shape, dtype in contracts:
        if tuple(tensor.shape) != shape:
            raise TritonSparseQKVError(
                '%s shape %s does not match %s'
                % (name, tuple(tensor.shape), shape)
            )
        if tensor.dtype != dtype:
            raise TritonSparseQKVError(
                '%s dtype %s does not match %s' % (name, tensor.dtype, dtype)
            )
        if tensor.device != device:
            raise TritonSparseQKVError('Triton sparse QKV devices differ')
        if not tensor.is_contiguous():
            raise TritonSparseQKVError('%s must be contiguous' % name)
    return prepared


def _empty_carrier(
    *,
    device,
    dtype,
    sequence,
    heads,
    layer_index,
    v_scale_group_size,
):
    sequence = int(sequence)
    heads = int(heads)
    group = normalize_v_scale_group_size(v_scale_group_size)
    q_blocks = (sequence + Q_TILE - 1) // Q_TILE
    kv_blocks = (sequence + KV_TILE - 1) // KV_TILE
    v_groups = HEAD_DIM // group
    shape = (1, heads, sequence, HEAD_DIM)
    return PreparedTritonSparseQKV(
        q_int8=torch.empty(shape, dtype=torch.int8, device=device),
        q_scale=torch.empty((1, heads, q_blocks), dtype=torch.float32, device=device),
        k_int8=torch.empty(shape, dtype=torch.int8, device=device),
        k_scale=torch.empty((1, heads, kv_blocks), dtype=torch.float32, device=device),
        v_int8=torch.empty(shape, dtype=torch.int8, device=device),
        v_scale=torch.empty(
            (1, heads, kv_blocks, v_groups), dtype=torch.float32, device=device
        ),
        v_sum=torch.empty(
            (1, heads, kv_blocks, HEAD_DIM), dtype=torch.int32, device=device
        ),
        q_summary=torch.empty(
            (1, heads, q_blocks, HEAD_DIM), dtype=dtype, device=device
        ),
        k_summary=torch.empty(
            (1, heads, kv_blocks, HEAD_DIM), dtype=dtype, device=device
        ),
        output_dtype=dtype,
        sequence=sequence,
        heads=heads,
        head_dim=HEAD_DIM,
        layer_index=int(layer_index),
        v_scale_group_size=group,
    )


if TRITON_AVAILABLE:

    @triton.jit
    def _pack_v_int8_grouped_chunk_kernel(
        x_ptr,
        output_ptr,
        scale_ptr,
        sum_ptr,
        x_b,
        x_h,
        x_n,
        output_b,
        output_h,
        output_n,
        scale_b,
        scale_h,
        scale_n,
        sum_b,
        sum_h,
        sum_n,
        row_start,
        block_start,
        groups: tl.constexpr,
        chunk_sequence: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        group_size: tl.constexpr,
    ):
        block = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1).to(tl.int64)
        batch_group = tl.program_id(2).to(tl.int64)
        batch = batch_group // groups
        group = batch_group - batch * groups
        rows = block * block_size + tl.arange(0, block_size).to(tl.int64)
        local_columns = tl.arange(0, group_size).to(tl.int64)
        columns = group * group_size + local_columns
        row_mask = rows < chunk_sequence
        source = (
            x_ptr
            + batch * x_b
            + head * x_h
            + rows[:, None] * x_n
            + columns[None, :]
        )
        value = tl.load(
            source,
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        abs_value = tl.where(row_mask[:, None], tl.abs(value), 0.0)
        row_max = tl.max(abs_value, axis=1)
        scale = tl.max(row_max, axis=0) / 127.0 + 1e-7
        quantized_f = value / scale
        quantized_f += 0.5 * tl.where(quantized_f >= 0, 1.0, -1.0)
        quantized = quantized_f.to(tl.int8)

        destination_rows = row_start + rows
        destination = (
            output_ptr
            + batch * output_b
            + head * output_h
            + destination_rows[:, None] * output_n
            + columns[None, :]
        )
        tl.store(destination, quantized, mask=row_mask[:, None])

        destination_block = block_start + block
        tl.store(
            scale_ptr
            + batch * scale_b
            + head * scale_h
            + destination_block * scale_n
            + group,
            scale,
        )
        quantized_i32 = quantized.to(tl.int32)
        channel_sum = tl.sum(
            tl.where(row_mask[:, None], quantized_i32, 0), axis=0
        )
        sum_destination = (
            sum_ptr
            + batch * sum_b
            + head * sum_h
            + destination_block * sum_n
            + columns
        )
        tl.store(sum_destination, channel_sum)


def pack_triton_v_chunk_into(
    x,
    output,
    scales,
    sums,
    *,
    row_start,
    block_size=KV_TILE,
    v_scale_group_size=1,
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
    _pack_v_int8_grouped_chunk_kernel[(blocks, heads, batch * groups)](
        x,
        output,
        scales,
        sums,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        scales.stride(0),
        scales.stride(1),
        scales.stride(2),
        sums.stride(0),
        sums.stride(1),
        sums.stride(2),
        row_start,
        row_start // block_size,
        groups=groups,
        chunk_sequence=chunk_sequence,
        head_dim=head_dim,
        block_size=block_size,
        group_size=group,
    )


def _pack_reference_blocks(x, block_size):
    batch, heads, sequence, head_dim = x.shape
    blocks = (sequence + block_size - 1) // block_size
    output = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((batch, heads, blocks), dtype=torch.float32)
    summaries = torch.empty((batch, heads, blocks, head_dim), dtype=x.dtype)
    for block in range(blocks):
        start = block * block_size
        end = min(start + block_size, sequence)
        value = x[..., start:end, :].float()
        scale = value.abs().amax(dim=(-2, -1)) / 127.0 + 1e-7
        quantized = value / scale[..., None, None]
        quantized += 0.5 * torch.where(quantized >= 0, 1.0, -1.0)
        output[..., start:end, :].copy_(quantized.to(torch.int8))
        scales[..., block].copy_(scale)
        summaries[..., block, :].copy_(value.mean(dim=-2).to(x.dtype))
    return output, scales, summaries


def _pack_reference_v(x, block_size, v_scale_group_size):
    batch, heads, sequence, head_dim = x.shape
    group = normalize_v_scale_group_size(v_scale_group_size)
    groups = head_dim // group
    blocks = (sequence + block_size - 1) // block_size
    output = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((batch, heads, blocks, groups), dtype=torch.float32)
    sums = torch.empty((batch, heads, blocks, head_dim), dtype=torch.int32)
    for block in range(blocks):
        start = block * block_size
        end = min(start + block_size, sequence)
        value = x[..., start:end, :].float()
        for group_index in range(groups):
            col_start = group_index * group
            col_end = col_start + group
            group_value = value[..., col_start:col_end]
            scale = group_value.abs().amax(dim=(-2, -1)) / 127.0 + 1e-7
            quantized = group_value / scale[..., None, None]
            quantized += 0.5 * torch.where(quantized >= 0, 1.0, -1.0)
            quantized_i8 = quantized.to(torch.int8)
            output[..., start:end, col_start:col_end].copy_(quantized_i8)
            scales[..., block, group_index].copy_(scale)
            sums[..., block, col_start:col_end].copy_(
                quantized_i8.to(torch.int32).sum(dim=-2)
            )
    return output, scales, sums


def pack_float_qkv(q, k, v, *, layer_index, v_scale_group_size=1):
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise TritonSparseQKVError(
            'Triton sparse Q/K/V must have equal HND rank-4 shapes'
        )
    if q.shape[0] != 1 or q.shape[-1] != HEAD_DIM:
        raise TritonSparseQKVError('Triton sparse Q/K/V shape is invalid')
    if q.dtype not in (torch.float16, torch.bfloat16) or not (
        q.dtype == k.dtype == v.dtype
    ):
        raise TritonSparseQKVError(
            'Triton sparse Q/K/V require matching fp16 or bf16 dtypes'
        )
    if q.device != k.device or q.device != v.device:
        raise TritonSparseQKVError('Triton sparse Q/K/V devices differ')
    group = normalize_v_scale_group_size(v_scale_group_size)

    if not q.is_cuda:
        q_i8, q_scale, q_summary = _pack_reference_blocks(q, Q_TILE)
        k_i8, k_scale, k_summary = _pack_reference_blocks(k, KV_TILE)
        v_i8, v_scale, v_sum = _pack_reference_v(v, KV_TILE, group)
        return validate_prepared_triton_sparse_qkv(
            PreparedTritonSparseQKV(
                q_int8=q_i8.contiguous(),
                q_scale=q_scale.contiguous(),
                k_int8=k_i8.contiguous(),
                k_scale=k_scale.contiguous(),
                v_int8=v_i8.contiguous(),
                v_scale=v_scale.contiguous(),
                v_sum=v_sum.contiguous(),
                q_summary=q_summary.contiguous(),
                k_summary=k_summary.contiguous(),
                output_dtype=q.dtype,
                sequence=q.shape[-2],
                heads=q.shape[1],
                head_dim=q.shape[-1],
                layer_index=int(layer_index),
                v_scale_group_size=group,
            )
        )

    carrier = _empty_carrier(
        device=q.device,
        dtype=q.dtype,
        sequence=q.shape[-2],
        heads=q.shape[1],
        layer_index=layer_index,
        v_scale_group_size=group,
    )
    try:
        pack_sparse_qk_chunk_into(
            q,
            carrier.q_int8,
            carrier.q_scale,
            carrier.q_summary,
            row_start=0,
            block_size=Q_TILE,
        )
        pack_sparse_qk_chunk_into(
            k,
            carrier.k_int8,
            carrier.k_scale,
            carrier.k_summary,
            row_start=0,
            block_size=KV_TILE,
        )
    except Exception as exc:
        raise TritonSparseQKVError(str(exc)) from exc
    pack_triton_v_chunk_into(
        v,
        carrier.v_int8,
        carrier.v_scale,
        carrier.v_sum,
        row_start=0,
        block_size=KV_TILE,
        v_scale_group_size=group,
    )
    return validate_prepared_triton_sparse_qkv(carrier)


def _validate_chunk_rows(chunk_rows):
    chunk_rows = int(chunk_rows)
    alignment = math.lcm(Q_TILE, KV_TILE)
    if chunk_rows <= 0 or chunk_rows % alignment:
        raise TritonSparseQKVError(
            'chunked Triton sparse QKV rows must be a positive multiple of %d'
            % alignment
        )
    return chunk_rows


def run_chunked_triton_sparse_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    chunk_rows=CHUNK_ROWS,
    v_scale_group_size=1,
):
    import comfy.model_management

    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise TritonSparseQKVError(
            'chunked Triton sparse QKV requires a rank-2 CUDA BF16 input'
        )
    if comfy.model_management.in_training:
        raise TritonSparseQKVError('chunked Triton sparse QKV is inference-only')
    if rope_freqs is not None and (
        rope_freqs.ndim != 6
        or tuple(rope_freqs.shape[:3]) != (1, x.shape[0], 1)
        or rope_freqs.device != x.device
    ):
        raise TritonSparseQKVError('chunked Triton sparse QKV received invalid RoPE')
    chunk_rows = _validate_chunk_rows(chunk_rows)
    group = normalize_v_scale_group_size(v_scale_group_size)
    sequence = int(x.shape[0])
    heads = int(module.heads)
    if int(module.head_dim) != HEAD_DIM:
        raise TritonSparseQKVError('chunked Triton sparse QKV requires head_dim 128')
    carrier = _empty_carrier(
        device=x.device,
        dtype=x.dtype,
        sequence=sequence,
        heads=heads,
        layer_index=layer_index,
        v_scale_group_size=group,
    )

    for start in range(0, sequence, chunk_rows):
        end = min(start + chunk_rows, sequence)
        q, k, v = project_chunk_hnd(module, x, rope_freqs, start, end)
        try:
            pack_sparse_qk_chunk_into(
                q,
                carrier.q_int8,
                carrier.q_scale,
                carrier.q_summary,
                row_start=start,
                block_size=Q_TILE,
            )
            pack_sparse_qk_chunk_into(
                k,
                carrier.k_int8,
                carrier.k_scale,
                carrier.k_summary,
                row_start=start,
                block_size=KV_TILE,
            )
            pack_triton_v_chunk_into(
                v,
                carrier.v_int8,
                carrier.v_scale,
                carrier.v_sum,
                row_start=start,
                block_size=KV_TILE,
                v_scale_group_size=group,
            )
        except Exception as exc:
            if isinstance(exc, TritonSparseQKVError):
                raise
            raise TritonSparseQKVError(str(exc)) from exc
        finally:
            del q, k, v
    return validate_prepared_triton_sparse_qkv(carrier)


class ChunkedTritonSparseQKVProjector:
    name = 'chunked_triton_sparse_qkv'
    qk_format = 'block_int8'

    def __init__(self, chunk_rows=CHUNK_ROWS, v_scale_group_size=1):
        self.chunk_rows = _validate_chunk_rows(chunk_rows)
        self.v_scale_group_size = normalize_v_scale_group_size(v_scale_group_size)

    @property
    def v_format(self):
        return 'per_kv_tile_group%d_int8' % self.v_scale_group_size

    @property
    def installation_signature(self):
        return (
            self.name,
            self.qk_format,
            self.v_format,
            self.chunk_rows,
            self.v_scale_group_size,
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
        return run_chunked_triton_sparse_qkv(
            module,
            x,
            rope_freqs,
            layer_index=layer_index,
            chunk_rows=self.chunk_rows,
            v_scale_group_size=self.v_scale_group_size,
        )
