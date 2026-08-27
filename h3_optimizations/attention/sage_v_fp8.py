"""Direct HND V conversion for SageAttention FP8 kernels.

The carrier layout and scaling follow SageAttention 2.2's Apache-2.0
``TransposePadPermuteKernel`` and ``MeanScaleKernel``. This version reduces the
source V directly, then quantizes into the final permuted carrier without the
full-size floating-point staging tensor.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _reduce_channel_amax(
        v,
        amax,
        heads: tl.constexpr,
        sequence: tl.constexpr,
        head_dim: tl.constexpr,
        stride_b: tl.constexpr,
        stride_h: tl.constexpr,
        stride_n: tl.constexpr,
        stride_d: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        GROUP_N: tl.constexpr,
        USE_I64: tl.constexpr,
    ):
        flat_head = tl.program_id(0)
        batch = flat_head // heads
        head = flat_head % heads
        n_block = tl.program_id(1)
        d_block = tl.program_id(2)
        n = n_block * BLOCK_N * GROUP_N + tl.arange(0, BLOCK_N)
        d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        if USE_I64:
            batch = batch.to(tl.int64)
            flat_head = flat_head.to(tl.int64)
            head = head.to(tl.int64)
            n = n.to(tl.int64)
            d = d.to(tl.int64)
        block_amax = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for group_index in range(GROUP_N):
            group_n = n + group_index * BLOCK_N
            offsets = (
                batch * stride_b
                + head * stride_h
                + group_n[:, None] * stride_n
                + d[None, :] * stride_d
            )
            values = tl.load(
                v + offsets,
                mask=(group_n[:, None] < sequence) & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            block_amax = tl.maximum(block_amax, tl.max(tl.abs(values), axis=0))
        tl.atomic_max(
            amax + flat_head * head_dim + d,
            block_amax,
            mask=d < head_dim,
        )


    @triton.jit
    def _quantize_transpose_permute(
        v,
        amax,
        carrier,
        scale,
        heads: tl.constexpr,
        sequence: tl.constexpr,
        padded_sequence: tl.constexpr,
        head_dim: tl.constexpr,
        stride_b: tl.constexpr,
        stride_h: tl.constexpr,
        stride_n: tl.constexpr,
        stride_d: tl.constexpr,
        scale_max: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_I64: tl.constexpr,
    ):
        flat_head = tl.program_id(0)
        batch = flat_head // heads
        head = flat_head % heads
        n_block = tl.program_id(1)
        n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        d = tl.arange(0, BLOCK_D)
        if USE_I64:
            batch = batch.to(tl.int64)
            flat_head = flat_head.to(tl.int64)
            head = head.to(tl.int64)
            n = n.to(tl.int64)
            d = d.to(tl.int64)
        source_offsets = (
            batch * stride_b
            + head * stride_h
            + n[:, None] * stride_n
            + d[None, :] * stride_d
        )
        values = tl.load(
            v + source_offsets,
            mask=(n[:, None] < sequence) & (d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        channel_amax = tl.load(
            amax + flat_head * head_dim + d,
            mask=d < head_dim,
            other=0.0,
        )
        quantized = values * (scale_max / channel_amax[None, :])

        local = n % 16
        permuted_n = (
            (n // 16) * 16
            + (local // 8) * 2
            + ((local // 2) % 4) * 4
            + local % 2
        )
        if USE_I64:
            permuted_n = permuted_n.to(tl.int64)
        destination_offsets = (
            flat_head * head_dim * padded_sequence
            + d[:, None] * padded_sequence
            + permuted_n[None, :]
        )
        tl.store(
            carrier + destination_offsets,
            tl.trans(quantized),
            mask=(d[:, None] < head_dim) & (permuted_n[None, :] < padded_sequence),
        )
        tl.store(
            scale + flat_head * head_dim + d,
            channel_amax / scale_max,
            mask=(n_block == 0) & (d < head_dim),
        )


    @triton.jit
    def _quantize_chunk_transpose_permute(
        v,
        amax,
        carrier,
        row_start,
        heads: tl.constexpr,
        rows: tl.constexpr,
        padded_sequence: tl.constexpr,
        head_dim: tl.constexpr,
        stride_b: tl.constexpr,
        stride_h: tl.constexpr,
        stride_n: tl.constexpr,
        stride_d: tl.constexpr,
        scale_max: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_I64: tl.constexpr,
    ):
        """Quantize one V chunk into its slice of the final permuted carrier.

        Identical arithmetic to _quantize_transpose_permute, except the source
        row index is chunk-local while the permuted destination row is global.
        The permutation is defined within 16-row groups, so a chunk whose
        row_start is 16-aligned lands in a disjoint span of the carrier.
        """
        flat_head = tl.program_id(0)
        batch = flat_head // heads
        head = flat_head % heads
        n_block = tl.program_id(1)
        n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        d = tl.arange(0, BLOCK_D)
        if USE_I64:
            batch = batch.to(tl.int64)
            flat_head = flat_head.to(tl.int64)
            head = head.to(tl.int64)
            n = n.to(tl.int64)
            d = d.to(tl.int64)
        source_offsets = (
            batch * stride_b
            + head * stride_h
            + n[:, None] * stride_n
            + d[None, :] * stride_d
        )
        values = tl.load(
            v + source_offsets,
            mask=(n[:, None] < rows) & (d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        channel_amax = tl.load(
            amax + flat_head * head_dim + d,
            mask=d < head_dim,
            other=0.0,
        )
        quantized = values * (scale_max / channel_amax[None, :])

        global_n = n + row_start
        local = global_n % 16
        permuted_n = (
            (global_n // 16) * 16
            + (local // 8) * 2
            + ((local // 2) % 4) * 4
            + local % 2
        )
        if USE_I64:
            permuted_n = permuted_n.to(tl.int64)
        destination_offsets = (
            flat_head * head_dim * padded_sequence
            + d[:, None] * padded_sequence
            + permuted_n[None, :]
        )
        tl.store(
            carrier + destination_offsets,
            tl.trans(quantized),
            mask=(
                (d[:, None] < head_dim)
                & (n[None, :] < rows)
                & (permuted_n[None, :] < padded_sequence)
            ),
        )


def direct_per_channel_fp8(v, *, scale_max, pad_to=64):
    if not TRITON_AVAILABLE:
        raise RuntimeError("direct Sage FP8 V conversion requires Triton")
    if (
        not v.is_cuda
        or v.ndim != 4
        or v.dtype not in (torch.float16, torch.bfloat16)
        or v.stride(-1) != 1
    ):
        raise ValueError("expected CUDA HND fp16/bf16 V with contiguous head dimension")
    if int(v.shape[-1]) != 128:
        raise ValueError("direct Sage FP8 V conversion requires head_dim 128")

    batch, heads, sequence, head_dim = (int(value) for value in v.shape)
    padded_sequence = (sequence + pad_to - 1) // pad_to * pad_to
    source_max_offset = sum(
        (int(size) - 1) * int(stride)
        for size, stride in zip(v.shape, v.stride())
    )
    carrier_max_offset = batch * heads * head_dim * padded_sequence - 1
    use_i64 = max(source_max_offset, carrier_max_offset) > (1 << 31) - 1
    amax = torch.zeros((batch, heads, head_dim), dtype=torch.float32, device=v.device)
    scale = torch.empty_like(amax)
    carrier = torch.empty(
        (batch, heads, head_dim, padded_sequence),
        dtype=torch.float8_e4m3fn,
        device=v.device,
    )
    _reduce_channel_amax[
        (batch * heads, triton.cdiv(sequence, 256 * 8), triton.cdiv(head_dim, 64))
    ](
        v,
        amax,
        heads=heads,
        sequence=sequence,
        head_dim=head_dim,
        stride_b=int(v.stride(0)),
        stride_h=int(v.stride(1)),
        stride_n=int(v.stride(2)),
        stride_d=int(v.stride(3)),
        BLOCK_N=256,
        BLOCK_D=64,
        GROUP_N=8,
        USE_I64=use_i64,
        num_warps=8,
    )
    _quantize_transpose_permute[
        (batch * heads, triton.cdiv(padded_sequence, 128))
    ](
        v,
        amax,
        carrier,
        scale,
        heads=heads,
        sequence=sequence,
        padded_sequence=padded_sequence,
        head_dim=head_dim,
        stride_b=int(v.stride(0)),
        stride_h=int(v.stride(1)),
        stride_n=int(v.stride(2)),
        stride_d=int(v.stride(3)),
        scale_max=float(scale_max),
        BLOCK_N=128,
        BLOCK_D=128,
        USE_I64=use_i64,
        num_warps=8,
    )
    return carrier, scale


def prepare_sage_v_fp8(v, stock_quantizer, *, scale_max, pad_to=64):
    if v.is_cuda and TRITON_AVAILABLE:
        return direct_per_channel_fp8(v, scale_max=scale_max, pad_to=pad_to)

    pad_rows = (-int(v.shape[-2])) % pad_to if pad_to > 64 else 0
    source = F.pad(v, (0, 0, 0, pad_rows)) if pad_rows else v
    carrier, scale, _ = stock_quantizer(
        source,
        tensor_layout="HND",
        scale_max=scale_max,
        smooth_v=False,
    )
    return carrier, scale
