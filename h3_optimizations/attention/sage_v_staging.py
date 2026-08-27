"""Two-pass V carrier staging for the SageAttention FP8 kernels.

The Kitchen INT8 equivalent lives in ``h3_optimizations.native.v_staging``.
This is the same idea against a different carrier: pass one reduces projected
V chunks to a per-channel amax, pass two re-projects each chunk and quantizes
it straight into the final permuted FP8 carrier, so a full-sequence BF16 V
tensor is never materialized.

The arithmetic and the carrier layout must stay identical to the one-pass
``direct_per_channel_fp8`` path: any divergence shows up as a quality delta
rather than an error. That includes the zero-amax behaviour, which is left
exactly as the one-pass kernel has it instead of being clamped here.
"""

from __future__ import annotations

import torch

from .sage_v_fp8 import TRITON_AVAILABLE


BACKEND_TRITON = 'triton'
BACKEND_TORCH = 'torch_reference'

# The carrier permutation is defined within 16-row groups, so every chunk
# boundary must be 16-aligned for chunks to occupy disjoint carrier spans.
ROW_GROUP = 16


class SageVStagingError(RuntimeError):
    pass


def _permuted_rows(rows):
    local = rows % ROW_GROUP
    return (
        (rows // ROW_GROUP) * ROW_GROUP
        + (local // 8) * 2
        + ((local // 2) % 4) * 4
        + local % 2
    )


def _torch_update(amax, v_chunk):
    chunk = v_chunk.to(torch.float32).abs().amax(dim=-2)
    amax.copy_(torch.maximum(amax, chunk))


def _torch_quantize(v_chunk, carrier, amax, scale_max, row_start):
    rows = int(v_chunk.shape[2])
    source = torch.arange(
        int(row_start),
        int(row_start) + rows,
        dtype=torch.int64,
        device=v_chunk.device,
    )
    destination = _permuted_rows(source)
    quantized = v_chunk.to(torch.float32) * (
        scale_max / amax.to(torch.float32).unsqueeze(-2)
    )
    packed = quantized.permute(0, 1, 3, 2).contiguous().to(carrier.dtype)
    # index_copy_ has no float8 CPU kernel, so move the bit patterns instead.
    # The reference path exists to be runnable without a GPU.
    carrier.view(torch.uint8).index_copy_(3, destination, packed.view(torch.uint8))


class TwoPassSageVCarrier:
    """Build a Sage FP8 V carrier from chunks without a full BF16 V tensor."""

    def __init__(
        self,
        batch,
        heads,
        sequence,
        head_dim,
        *,
        scale_max,
        device,
        dtype,
        pad_to=64,
        backend=BACKEND_TRITON,
    ):
        if int(head_dim) != 128:
            raise SageVStagingError(
                'Sage FP8 V staging requires head_dim 128; got %d' % head_dim
            )
        if backend not in (BACKEND_TRITON, BACKEND_TORCH):
            raise SageVStagingError('unknown Sage V staging backend %r' % backend)
        if backend == BACKEND_TRITON and not TRITON_AVAILABLE:
            raise SageVStagingError('two-pass Sage V staging requires Triton')
        self.batch = int(batch)
        self.heads = int(heads)
        self.sequence = int(sequence)
        self.head_dim = int(head_dim)
        self.scale_max = float(scale_max)
        self.pad_to = int(pad_to)
        self.padded = (
            (self.sequence + self.pad_to - 1) // self.pad_to * self.pad_to
        )
        self.device = device
        self.dtype = dtype
        self.backend = backend
        self.amax = torch.zeros(
            (self.batch, self.heads, self.head_dim),
            dtype=torch.float32,
            device=device,
        )
        self.scale = None
        self.carrier = None
        self._covered = []

    def _check_chunk(self, v_chunk):
        if v_chunk.ndim != 4:
            raise SageVStagingError('V chunks must be [batch, heads, rows, dim]')
        if tuple(v_chunk.shape[:2]) != (self.batch, self.heads):
            raise SageVStagingError('V chunk batch/head shape does not match')
        if int(v_chunk.shape[-1]) != self.head_dim:
            raise SageVStagingError('V chunk head dimension does not match')
        if v_chunk.dtype != self.dtype:
            raise SageVStagingError('V chunk dtype does not match the source')
        if v_chunk.stride(-1) != 1:
            raise SageVStagingError('V chunk head dimension must be contiguous')

    def _use_i64(self, v_chunk):
        source_max_offset = sum(
            (int(size) - 1) * int(stride)
            for size, stride in zip(v_chunk.shape, v_chunk.stride())
        )
        carrier_max_offset = (
            self.batch * self.heads * self.head_dim * self.padded - 1
        )
        return max(source_max_offset, carrier_max_offset) > (1 << 31) - 1

    def update(self, v_chunk):
        if self.scale is not None:
            raise SageVStagingError('V scale is already finalized')
        self._check_chunk(v_chunk)
        if self.backend == BACKEND_TORCH:
            _torch_update(self.amax, v_chunk)
            return
        import triton

        from .sage_v_fp8 import _reduce_channel_amax

        batch, heads, rows, head_dim = (int(value) for value in v_chunk.shape)
        _reduce_channel_amax[
            (batch * heads, triton.cdiv(rows, 256 * 8), triton.cdiv(head_dim, 64))
        ](
            v_chunk,
            self.amax,
            heads=heads,
            sequence=rows,
            head_dim=head_dim,
            stride_b=int(v_chunk.stride(0)),
            stride_h=int(v_chunk.stride(1)),
            stride_n=int(v_chunk.stride(2)),
            stride_d=int(v_chunk.stride(3)),
            BLOCK_N=256,
            BLOCK_D=64,
            GROUP_N=8,
            USE_I64=self._use_i64(v_chunk),
            num_warps=8,
        )

    def finalize_scale(self):
        if self.scale is not None:
            return self.scale
        self.scale = self.amax / self.scale_max
        self.carrier = torch.empty(
            (self.batch, self.heads, self.head_dim, self.padded),
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        # Chunks only cover [0, sequence). The one-pass kernel writes zeros
        # across the padded tail because its grid spans the padded sequence;
        # reproduce that here. Rows in the final 16-group interleave with real
        # rows under the permutation, so zero from the group boundary up and
        # let the last chunk overwrite its own rows afterwards.
        tail = (self.sequence // ROW_GROUP) * ROW_GROUP
        if tail < self.padded:
            self.carrier[..., tail:].zero_()
        return self.scale

    def quantize(self, v_chunk, row_start):
        if self.scale is None:
            raise SageVStagingError(
                'V scale must be finalized before quantization'
            )
        self._check_chunk(v_chunk)
        row_start = int(row_start)
        rows = int(v_chunk.shape[2])
        if row_start % ROW_GROUP:
            raise SageVStagingError(
                'V chunk row start %d is not %d-aligned' % (row_start, ROW_GROUP)
            )
        if row_start < 0 or row_start + rows > self.sequence:
            raise SageVStagingError('V chunk falls outside the carrier sequence')
        if self.backend == BACKEND_TORCH:
            _torch_quantize(
                v_chunk, self.carrier, self.amax, self.scale_max, row_start
            )
        else:
            import triton

            from .sage_v_fp8 import _quantize_chunk_transpose_permute

            batch, heads, _, head_dim = (int(value) for value in v_chunk.shape)
            _quantize_chunk_transpose_permute[
                (batch * heads, triton.cdiv(rows, 128))
            ](
                v_chunk,
                self.amax,
                self.carrier,
                row_start,
                heads=heads,
                rows=rows,
                padded_sequence=self.padded,
                head_dim=head_dim,
                stride_b=int(v_chunk.stride(0)),
                stride_h=int(v_chunk.stride(1)),
                stride_n=int(v_chunk.stride(2)),
                stride_d=int(v_chunk.stride(3)),
                scale_max=self.scale_max,
                BLOCK_N=128,
                BLOCK_D=128,
                USE_I64=self._use_i64(v_chunk),
                num_warps=8,
            )
        self._covered.append((row_start, row_start + rows))

    def finish(self):
        covered = 0
        for start, end in sorted(self._covered):
            if start != covered:
                raise SageVStagingError('V chunks contain a gap or overlap')
            covered = end
        if covered != self.sequence:
            raise SageVStagingError('V chunks do not cover the carrier sequence')
        return self.carrier, self.scale
