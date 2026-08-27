"""64-token tile-local INT8 V carrier for sparse H3 attention."""

from __future__ import annotations

import ctypes
import torch

from . import loader
from .int8_attention import _DTYPE_TO_CODE, _pad_to, _ptr, _stream


class TileLocalVError(RuntimeError):
    pass


_BOUND = False


def _bind():
    global _BOUND
    lib = loader.load()
    if _BOUND:
        return lib

    required = (
        "h3_int8_quantize_v_tile_local",
        "h3_int8_sparse_attention_tile_v",
        "h3_int8_sparse_attention_tile_v_lse",
    )
    for name in required:
        if not hasattr(lib, name):
            raise TileLocalVError(
                "native DLL lacks tile-local V support; rebuild native/"
            )

    p, i, i64, sz = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_size_t,
    )
    lib.h3_int8_quantize_v_tile_local.restype = i
    lib.h3_int8_quantize_v_tile_local.argtypes = (
        [p, p, p] + [i] * 7 + [i64] * 3 + [i, sz]
    )

    lib.h3_int8_sparse_attention_tile_v.restype = i
    lib.h3_int8_sparse_attention_tile_v.argtypes = (
        lib.h3_int8_sparse_attention.argtypes
    )
    lib.h3_int8_sparse_attention_tile_v_lse.restype = i
    lib.h3_int8_sparse_attention_tile_v_lse.argtypes = (
        lib.h3_int8_sparse_attention_lse.argtypes
    )
    _BOUND = True
    return lib


class TileLocalVCarrier:
    TILE = 64

    def __init__(self, spec):
        if int(spec.cta_k) != self.TILE:
            raise TileLocalVError(
                "tile-local V currently requires sparse CTA_K=64"
            )
        self.spec = spec
        self.batch = int(spec.k_input_shape[0])
        self.heads = int(spec.k_input_shape[1])
        self.sequence = int(spec.k_input_shape[2])
        self.head_dim = int(spec.kernel_head_dim)
        self.padded = int(_pad_to(self.sequence, self.TILE))
        self.tiles = self.padded // self.TILE
        self.v_int8 = torch.zeros(
            self.batch * self.heads * self.head_dim,
            self.padded,
            dtype=torch.int8,
            device=spec.device,
        )
        self.scale = torch.empty(
            self.tiles,
            self.batch,
            self.heads,
            self.head_dim,
            dtype=torch.float32,
            device=spec.device,
        )
        self._covered = []

    def quantize(self, v, row_start):
        if v.ndim != 4:
            raise TileLocalVError("V chunk must be [B,H,N,D]")
        if tuple(v.shape[:2]) != (self.batch, self.heads):
            raise TileLocalVError("V chunk batch/head mismatch")
        if int(v.shape[-1]) != self.head_dim:
            raise TileLocalVError("V chunk head_dim mismatch")
        if v.dtype != self.spec.input_dtype:
            raise TileLocalVError("V chunk dtype mismatch")
        if int(v.stride(-1)) != 1:
            raise TileLocalVError("V head dimension must be contiguous")

        row_start = int(row_start)
        rows = int(v.shape[2])
        if row_start < 0 or row_start + rows > self.sequence:
            raise TileLocalVError("V chunk lies outside sequence")
        if row_start % self.TILE:
            raise TileLocalVError("V chunk start must be 64-row aligned")
        if row_start + rows != self.sequence and rows % self.TILE:
            raise TileLocalVError(
                "non-final V chunks must contain whole 64-row tiles"
            )

        lib = _bind()
        status = lib.h3_int8_quantize_v_tile_local(
            _ptr(v),
            _ptr(self.v_int8),
            _ptr(self.scale),
            self.batch,
            self.heads,
            rows,
            self.head_dim,
            self.padded,
            row_start,
            self.TILE,
            int(v.stride(0)),
            int(v.stride(1)),
            int(v.stride(2)),
            _DTYPE_TO_CODE[v.dtype],
            _stream(),
        )
        loader.check(status, "tile-local V quantization")
        self._covered.append((row_start, row_start + rows))

    def finish(self):
        covered = 0
        for start, stop in sorted(self._covered):
            if start != covered:
                raise TileLocalVError(
                    "V chunks leave a gap/overlap at row %d" % covered
                )
            covered = stop
        if covered != self.sequence:
            raise TileLocalVError(
                "V chunks cover %d/%d rows" % (covered, self.sequence)
            )
        return self.v_int8, self.scale


def sparse_symbol(with_lse=False):
    lib = _bind()
    return (
        lib.h3_int8_sparse_attention_tile_v_lse
        if with_lse
        else lib.h3_int8_sparse_attention_tile_v
    )
