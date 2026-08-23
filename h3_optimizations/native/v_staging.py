"""Two-phase V carrier production, without retaining V in BF16.

The shipped producer streams Q and K straight into their INT8 carriers but
still builds a full-sequence BF16 V, because Kitchen's V scale is a global
per-channel absmax and a chunk cannot know it. At the production H3 shape that
buffer is 738 MiB -- the largest single allocation the producer makes.

It does not have to exist. Splitting the whole-V quantizer along the seam it
already has costs nothing in accuracy:

  phase 1  ``update_v_amax_chunk``  accumulate |v| maxima per (b, h, d)
           ``finalize_v_scale``     scale = max(amax / 127, 1e-12)
  phase 2  ``quantize_v_chunk_into`` write one chunk into the final carrier

Both halves are exactly separable, which is why byte parity is the acceptance
test rather than a tolerance:

* Phase 1 is a maximum of absolute values. ``max`` is associative and exact on
  floats, so chunk-wise partials reduced with ``max`` give the identical
  result for any chunk boundaries. (A NaN would make the result order
  dependent -- but it already is, across warps, in the fused kernel.)
* Phase 2's destination index is ``(src & ~15) | inv_perm16(src & 15)``, a
  pure function of the *global* row. It does not depend on the sequence
  length, on the chunk, or on where the chunk starts, so a chunked pass writes
  the same bytes to the same places. Given a finalized scale the arithmetic is
  identical too.

Two implementations sit behind the same interface. The native one is what
would ship, and is byte-identical to the whole-V quantizer -- measured, not
asserted. The Torch one is a reference for the design: it reproduces the
permutation, the scale and the rounding mode, so the seam above can be
validated with no build at all.

The Torch reference is NOT bit-identical to the CUDA kernels, and cannot be.
The library is built with ``--use_fast_math``, so the kernel's ``1.f / sc`` is
an approximate reciprocal differing from an exactly-rounded one by an ulp.
That tips values landing exactly on a rounding boundary: measured at the
production dtype, 142 of 2,097,152 INT8 values differ by one, every one of
them at ``|v / scale - round(v / scale)| == 0.5``. The scales themselves are
bit-identical. This is a property of the shipped whole-V quantizer too, which
uses the same expression -- so the native path matches the carrier exactly and
the reference is close, which is the right way round.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import platform
import threading

import torch

from . import loader
from .int8_attention import _DTYPE_TO_CODE, _pad_to, _ptr, _stream

V_STAGING_ABI_VERSION = 1
_NATIVE_SYMBOLS = ('h3_int8_v_amax_chunk', 'h3_int8_quantize_v_chunk_into')

BACKEND_NATIVE = 'native'
BACKEND_TORCH = 'torch_reference'


class VStagingError(RuntimeError):
    pass


_SIDECAR_NAMES = {
    'Windows': 'h3_v_staging.dll',
    'Linux': 'libh3_v_staging.so',
    'Darwin': 'libh3_v_staging.dylib',
}
_PACK_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_sidecar_lock = threading.Lock()
_sidecar = None
_sidecar_searched = False


def _exports(library):
    for symbol in _NATIVE_SYMBOLS:
        if not hasattr(library, symbol):
            return False
    return True


def _bind_v_staging(library):
    """Declare argument types so ctypes cannot truncate a pointer or stride."""
    p, i, i64, sz = (
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_size_t
    )
    library.h3_int8_v_amax_chunk.restype = i
    library.h3_int8_v_amax_chunk.argtypes = (
        [p, p] + [i] * 4 + [i64] * 3 + [i, sz]
    )
    library.h3_int8_quantize_v_chunk_into.restype = i
    library.h3_int8_quantize_v_chunk_into.argtypes = (
        [p, p, p] + [i] * 6 + [i64] * 3 + [i, sz]
    )
    return library


def _sidecar_paths():
    name = _SIDECAR_NAMES.get(platform.system())
    if name is None:
        return []
    override = os.environ.get('H3_V_STAGING_LIBRARY')
    candidates = [pathlib.Path(override)] if override else []
    candidates.extend(
        [
            _PACK_ROOT / 'native' / 'bin' / name,
            _PACK_ROOT / 'native' / 'build' / name,
        ]
    )
    return candidates


def _load_sidecar():
    """The experimental side-car, built by native/build_v_staging.ps1."""
    global _sidecar, _sidecar_searched
    with _sidecar_lock:
        if _sidecar_searched:
            return _sidecar
        _sidecar_searched = True
        path = next((p for p in _sidecar_paths() if p.is_file()), None)
        if path is None:
            return None
        library = ctypes.CDLL(str(path))
        if not _exports(library):
            raise VStagingError(
                'the V staging library at %s does not export %s'
                % (path, ', '.join(_NATIVE_SYMBOLS))
            )
        version = library.h3_v_staging_abi_version()
        if version != V_STAGING_ABI_VERSION:
            raise VStagingError(
                'the V staging library at %s reports ABI %d; this build '
                'expects %d' % (path, version, V_STAGING_ABI_VERSION)
            )
        library.h3_v_staging_last_error.restype = ctypes.c_char_p
        library.h3_v_staging_last_error.argtypes = []
        _sidecar = _bind_v_staging(library)
        return _sidecar


def _library_with_v_staging():
    """Whichever library carries the staging kernels, shipped one first."""
    try:
        library = loader.load()
    except loader.NativeUnavailableError:
        library = None
    if library is not None and _exports(library):
        return library
    return _load_sidecar()


def _check(library, status, what):
    if status == 0:
        return
    reporter = getattr(
        library, 'h3_v_staging_last_error', None
    ) or library.h3_int8_last_error
    detail = reporter()
    detail = detail.decode('utf-8', 'replace') if detail else 'no detail'
    raise VStagingError('%s failed (status %d): %s' % (what, status, detail))


def available_backend():
    return BACKEND_NATIVE if _library_with_v_staging() else BACKEND_TORCH


def native_library_path():
    """Which binary the native backend would use, for the result record."""
    try:
        library = loader.load()
    except loader.NativeUnavailableError:
        library = None
    if library is not None and _exports(library):
        return 'vendored'
    return next(
        (str(path) for path in _sidecar_paths() if path.is_file()), None
    )


def new_v_amax(batch, heads, head_dim, *, device):
    """The [B, H, D] accumulator phase 1 folds every chunk into."""
    return torch.zeros(
        int(batch), int(heads), int(head_dim),
        dtype=torch.float32,
        device=device,
    )


def finalize_v_scale(amax):
    """Kitchen's exact scale: max(amax / 127, 1e-12), flattened per channel."""
    if amax.dtype != torch.float32:
        raise VStagingError('the V amax accumulator must be float32')
    return torch.clamp(amax * (1.0 / 127.0), min=1e-12).reshape(-1).contiguous()


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------

def _inverse_permutation_16(device):
    """inv(s): {b3,b2,b1,b0} -> {b2,b0,b3,b1}, matching quant_v_int8.cu."""
    w = torch.arange(16, dtype=torch.int64, device=device)
    return (
        (w & 1)
        | (((w >> 3) & 1) << 1)
        | (((w >> 1) & 1) << 2)
        | (((w >> 2) & 1) << 3)
    )


def _torch_update_v_amax(amax, v_chunk):
    # float32 before abs so a bf16 chunk reduces exactly the way the kernel's
    # `static_cast<float>` then `fabsf` does.
    chunk = v_chunk.to(torch.float32).abs().amax(dim=-2)
    amax.copy_(torch.maximum(amax, chunk))


def _torch_quantize_v_chunk_into(v_chunk, v_int8, scale, row_start):
    batch, heads, rows, head_dim = v_chunk.shape
    padded = v_int8.shape[-1]
    source = torch.arange(
        int(row_start), int(row_start) + int(rows),
        dtype=torch.int64,
        device=v_chunk.device,
    )
    destination = (source & ~15) | _inverse_permutation_16(v_chunk.device)[
        source & 15
    ]
    inverse = (
        1.0 / scale.reshape(batch, heads, head_dim).to(torch.float32)
    ).unsqueeze(-2)
    # round-half-to-even then saturate: `cvt.rni.sat.s8.f32`.
    quantized = torch.clamp(
        torch.round(v_chunk.to(torch.float32) * inverse), -128.0, 127.0
    ).to(torch.int8)
    rows_view = v_int8.view(batch, heads, head_dim, padded)
    rows_view.index_copy_(
        3, destination, quantized.permute(0, 1, 3, 2).contiguous()
    )


def _torch_zero_v_padding(v_int8, sequence):
    padded = v_int8.shape[-1]
    if padded == sequence:
        return
    source = torch.arange(
        int(sequence), int(padded), dtype=torch.int64, device=v_int8.device
    )
    destination = (source & ~15) | _inverse_permutation_16(v_int8.device)[
        source & 15
    ]
    v_int8.index_fill_(1, destination, 0)


# ---------------------------------------------------------------------------
# Native
# ---------------------------------------------------------------------------

def _native_update_v_amax(library, amax, v_chunk):
    batch, heads, rows, head_dim = v_chunk.shape
    _check(
        library,
        library.h3_int8_v_amax_chunk(
            _ptr(v_chunk), _ptr(amax),
            batch, heads, rows, head_dim,
            v_chunk.stride(0), v_chunk.stride(1), v_chunk.stride(2),
            _DTYPE_TO_CODE[v_chunk.dtype], _stream(),
        ),
        'v_amax_chunk',
    )


def _native_quantize_v_chunk_into(library, v_chunk, v_int8, scale, row_start):
    batch, heads, rows, head_dim = v_chunk.shape
    padded = v_int8.shape[-1]
    _check(
        library,
        library.h3_int8_quantize_v_chunk_into(
            _ptr(v_chunk), _ptr(v_int8), _ptr(scale),
            batch, heads, rows, int(row_start), head_dim, padded,
            v_chunk.stride(0), v_chunk.stride(1), v_chunk.stride(2),
            _DTYPE_TO_CODE[v_chunk.dtype], _stream(),
        ),
        'quantize_v_chunk_into',
    )


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------

class TwoPassVCarrier:
    """Accumulate V maxima over one pass, then quantize over a second.

    ``backend`` is deliberately explicit rather than negotiated per call: an
    experiment that silently ran the Torch reference where it meant to measure
    the native kernel would report a number about nothing.
    """

    def __init__(self, spec, *, backend=None):
        self.spec = spec
        self.batch, self.heads = spec.k_input_shape[0], spec.k_input_shape[1]
        self.sequence = spec.k_input_shape[2]
        self.head_dim = spec.kernel_head_dim
        self.padded = _pad_to(self.sequence, spec.cta_k)
        resolved = available_backend() if backend is None else str(backend)
        if resolved not in (BACKEND_NATIVE, BACKEND_TORCH):
            raise VStagingError('unknown V staging backend %r' % resolved)
        self.library = _library_with_v_staging()
        if resolved == BACKEND_NATIVE and self.library is None:
            raise VStagingError(
                'the native V staging kernels are not present in this build; '
                'build native/ with v_staging.cu or ask for the torch '
                'reference explicitly'
            )
        self.backend = resolved
        self.amax = new_v_amax(
            self.batch, self.heads, self.head_dim, device=spec.device
        )
        self.scale = None
        self.v_int8 = None
        self._covered = []

    def update(self, v_chunk):
        """Phase 1: fold one chunk's per-channel maxima in, then forget it."""
        if self.scale is not None:
            raise VStagingError('the V scale is already finalized')
        self._check_chunk(v_chunk)
        if self.backend == BACKEND_NATIVE:
            _native_update_v_amax(self.library, self.amax, v_chunk)
        else:
            _torch_update_v_amax(self.amax, v_chunk)

    def finalize_scale(self):
        """Phase 1 -> 2: fix the exact Kitchen scale for the whole sequence."""
        if self.scale is None:
            self.scale = finalize_v_scale(self.amax)
            self.v_int8 = torch.empty(
                self.batch * self.heads * self.head_dim, self.padded,
                dtype=torch.int8, device=self.spec.device,
            )
            # The tail is at most cta_k - 1 columns and its destinations are
            # disjoint from every real row's, so zeroing it once here keeps
            # both chunk kernels free of any padding special case.
            _torch_zero_v_padding(self.v_int8, self.sequence)
        return self.scale

    def quantize(self, v_chunk, row_start):
        """Phase 2: write one chunk into the final carrier at its own rows."""
        if self.scale is None:
            raise VStagingError(
                'finalize_scale() must run before any chunk is quantized'
            )
        self._check_chunk(v_chunk)
        rows = int(v_chunk.shape[2])
        row_start = int(row_start)
        if row_start < 0 or row_start + rows > self.sequence:
            raise VStagingError(
                'V chunk [%d, %d) falls outside the sequence of %d'
                % (row_start, row_start + rows, self.sequence)
            )
        if self.backend == BACKEND_NATIVE:
            _native_quantize_v_chunk_into(
                self.library, v_chunk, self.v_int8, self.scale, row_start
            )
        else:
            _torch_quantize_v_chunk_into(
                v_chunk, self.v_int8, self.scale, row_start
            )
        self._covered.append((row_start, row_start + rows))

    def finish(self):
        """The carrier fields, once the chunks tile the sequence exactly."""
        if self.scale is None:
            raise VStagingError('V was never quantized')
        covered = 0
        for start, stop in sorted(self._covered):
            if start != covered:
                raise VStagingError(
                    'V chunks leave a gap or overlap at row %d' % covered
                )
            covered = stop
        if covered != self.sequence:
            raise VStagingError(
                'V chunks cover %d of %d rows' % (covered, self.sequence)
            )
        return self.v_int8, self.scale

    def _check_chunk(self, v_chunk):
        if v_chunk.ndim != 4:
            raise VStagingError('V chunks must be [batch, heads, rows, dim]')
        if v_chunk.shape[0] != self.batch or v_chunk.shape[1] != self.heads:
            raise VStagingError('V chunk batch/head shape does not match spec')
        if v_chunk.shape[3] != self.head_dim:
            raise VStagingError(
                'V chunk head_dim %d does not match the carrier %d'
                % (v_chunk.shape[3], self.head_dim)
            )
        if v_chunk.dtype != self.spec.input_dtype:
            raise VStagingError(
                'V chunks must have dtype %r' % self.spec.input_dtype
            )
        if v_chunk.stride(-1) != 1:
            raise VStagingError('the V head dimension must be contiguous')

