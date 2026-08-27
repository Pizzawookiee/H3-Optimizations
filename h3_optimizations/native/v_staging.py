"""Exact two-pass Kitchen V carrier staging."""

from __future__ import annotations

import ctypes
import torch

from . import loader
from .int8_attention import _DTYPE_TO_CODE, _pad_to, _ptr, _stream


class VStagingError(RuntimeError):
    pass


def _bind():
    try:
        library = loader.load()
    except loader.NativeUnavailableError as exc:
        raise VStagingError("vendored H3 INT8 library is unavailable") from exc
    for symbol in ("h3_int8_v_amax_chunk", "h3_int8_quantize_v_chunk_into"):
        if not hasattr(library, symbol):
            raise VStagingError(
                "native library lacks exact two-pass V staging; rebuild native/"
            )
    p, i, i64, sz = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_size_t
    library.h3_int8_v_amax_chunk.restype = i
    library.h3_int8_v_amax_chunk.argtypes = [p, p] + [i] * 4 + [i64] * 3 + [i, sz]
    library.h3_int8_quantize_v_chunk_into.restype = i
    library.h3_int8_quantize_v_chunk_into.argtypes = [p, p, p] + [i] * 6 + [i64] * 3 + [i, sz]
    if hasattr(library, "h3_v_staging_last_error"):
        library.h3_v_staging_last_error.restype = ctypes.c_char_p
        library.h3_v_staging_last_error.argtypes = []
    return library


def _check(library, status, what):
    if int(status) == 0:
        return
    reporter = getattr(library, "h3_v_staging_last_error", None) or library.h3_int8_last_error
    detail = reporter()
    detail = detail.decode("utf-8", "replace") if detail else "no detail"
    raise VStagingError("%s failed (status %d): %s" % (what, int(status), detail))


def _inverse_permutation_16(device):
    w = torch.arange(16, dtype=torch.int64, device=device)
    return (w & 1) | (((w >> 3) & 1) << 1) | (((w >> 1) & 1) << 2) | (((w >> 2) & 1) << 3)


def _zero_padding(v_int8, sequence):
    padded = int(v_int8.shape[-1])
    if padded == int(sequence):
        return
    source = torch.arange(int(sequence), padded, dtype=torch.int64, device=v_int8.device)
    destination = (source & ~15) | _inverse_permutation_16(v_int8.device)[source & 15]
    v_int8.index_fill_(1, destination, 0)


class TwoPassVCarrier:
    def __init__(self, spec, *, backend="native"):
        if backend != "native":
            raise VStagingError("production V optimization requires native backend")
        self.library = _bind()
        self.spec = spec
        self.batch = int(spec.k_input_shape[0])
        self.heads = int(spec.k_input_shape[1])
        self.sequence = int(spec.k_input_shape[2])
        self.head_dim = int(spec.kernel_head_dim)
        self.padded = int(_pad_to(self.sequence, spec.cta_k))
        self.amax = torch.zeros(
            self.batch, self.heads, self.head_dim,
            dtype=torch.float32, device=spec.device,
        )
        self.scale = None
        self.v_int8 = None
        self._covered = []

    def _check_chunk(self, v):
        if v.ndim != 4:
            raise VStagingError("V chunk must be [B,H,N,D]")
        if int(v.shape[0]) != self.batch or int(v.shape[1]) != self.heads:
            raise VStagingError("V chunk batch/head mismatch")
        if int(v.shape[3]) != self.head_dim:
            raise VStagingError("V chunk head_dim mismatch")
        if v.dtype != self.spec.input_dtype:
            raise VStagingError("V chunk dtype mismatch")
        if int(v.stride(-1)) != 1:
            raise VStagingError("V head dimension must be contiguous")

    def update(self, v):
        self._check_chunk(v)
        if self.scale is not None:
            raise VStagingError("V scale is already finalized")
        b, h, rows, d = map(int, v.shape)
        _check(
            self.library,
            self.library.h3_int8_v_amax_chunk(
                _ptr(v), _ptr(self.amax), b, h, rows, d,
                int(v.stride(0)), int(v.stride(1)), int(v.stride(2)),
                _DTYPE_TO_CODE[v.dtype], _stream(),
            ),
            "v_amax_chunk",
        )

    def finalize_scale(self):
        if self.scale is None:
            self.scale = torch.clamp(self.amax * (1.0 / 127.0), min=1e-12).reshape(-1).contiguous()
            self.v_int8 = torch.empty(
                self.batch * self.heads * self.head_dim,
                self.padded,
                dtype=torch.int8,
                device=self.spec.device,
            )
            _zero_padding(self.v_int8, self.sequence)
        return self.scale

    def quantize(self, v, row_start):
        self._check_chunk(v)
        if self.scale is None:
            raise VStagingError("finalize_scale() must run before quantize()")
        b, h, rows, d = map(int, v.shape)
        row_start = int(row_start)
        if row_start < 0 or row_start + rows > self.sequence:
            raise VStagingError("V chunk lies outside sequence")
        _check(
            self.library,
            self.library.h3_int8_quantize_v_chunk_into(
                _ptr(v), _ptr(self.v_int8), _ptr(self.scale),
                b, h, rows, row_start, d, self.padded,
                int(v.stride(0)), int(v.stride(1)), int(v.stride(2)),
                _DTYPE_TO_CODE[v.dtype], _stream(),
            ),
            "quantize_v_chunk_into",
        )
        self._covered.append((row_start, row_start + rows))

    def finish(self):
        if self.scale is None:
            raise VStagingError("V was never finalized")
        covered = 0
        for start, stop in sorted(self._covered):
            if start != covered:
                raise VStagingError("V chunks leave a gap/overlap at row %d" % covered)
            covered = stop
        if covered != self.sequence:
            raise VStagingError("V chunks cover %d/%d rows" % (covered, self.sequence))
        return self.v_int8, self.scale
