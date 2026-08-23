"""Prove the native kernels work on *this* GPU before trusting them.

The library is compiled for several architectures and validated directly on
one. A prebuilt binary makes "it was built for this card" an assumption rather
than something anyone checked, so check it here once and cache the verdict.

Two legs, because either alone passes while broken:

    dense INT8 vs FP32 SDPA   catches a dense kernel that is wrong on this
                              architecture. Without it, a sparse kernel that
                              faithfully reproduces a broken dense kernel
                              passes leg two and ships corrupt video.

    100% route vs dense       catches a broken traversal. Bit-identical at the
                              128-wide KV tile used by H3's production sparse
                              Kitchen backend, rather than an auto-selected
                              short-sequence tile that may dispatch different
                              dense and sparse math on some architectures.

Then synchronize and look for asynchronous faults, because a bad launch on an
unseen architecture surfaces later at an unrelated point. "It did not raise"
is not a result.
"""

from __future__ import annotations

import json
import logging
import pathlib
import threading

import torch

from . import loader

LOG_PREFIX = '[H3 Optimizations]'

_CACHE = pathlib.Path(__file__).resolve().parent.parent.parent / 'native' / 'selftest.json'
_lock = threading.Lock()
_result = None

# Bump whenever the meaning of a cached pass/fail changes without changing the
# native binary build ID. Otherwise an old failure can survive a Python-only fix.
_SELFTEST_REVISION = 'v2'

# Small enough to be free, large enough for a full q tile, a ragged tail,
# several heads and more than one KV tile.
_BATCH, _HEADS, _Q_LEN, _KV_LEN, _HEAD_DIM = 1, 4, 300, 300, 128

# H3's sparse Kitchen backend is deliberately pinned to 128-wide KV tiles.
# Keep this local to the native startup layer: importing the high-level sparse
# backend here would pull Comfy runtime modules into pre-startup initialization.
_SPARSE_PARITY_CTA_K = 128

# Healthy INT8 error is ~0.016 relative L2; the mildest corruption injected in
# testing produced 0.42. This sits between, well clear of both.
_INT8_TOLERANCE = 0.15


def _relative_l2(actual, expected):
    error = (actual.float() - expected.float()).norm()
    return (error / expected.float().norm().clamp_min(1e-12)).item()


def _cache_key(device):
    major, minor = torch.cuda.get_device_capability(device)
    from .bootstrap import installed_build_id

    return 'sm%d%d|%s|%s|%s' % (
        major,
        minor,
        installed_build_id() or 'local',
        _SELFTEST_REVISION,
        torch.cuda.get_device_name(device),
    )


def _read_cache():
    try:
        return json.loads(_CACHE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def _write_cache(cache):
    try:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE.write_text(json.dumps(cache, indent=2), encoding='utf-8')
    except OSError:
        pass


def run(device=None, *, verbose=False):
    """Return (passed, detail). Never raises on a kernel fault; reports it."""
    from . import int8_attention as native

    device = torch.device('cuda' if device is None else device)
    detail = {}
    try:
        generator = torch.Generator(device=device).manual_seed(20260823)
        q, k, v = (
            torch.randn(
                _BATCH, _Q_LEN, _HEADS, _HEAD_DIM,
                device=device, dtype=torch.bfloat16, generator=generator,
            ).transpose(1, 2)
            for _ in range(3)
        )

        # Leg one intentionally keeps the normal tile heuristic: this is the
        # broad dense-kernel sanity check, not an H3-production-path check.
        dense_carrier = native.prequantize_int8_attention(q, k, v)
        dense = native.int8_attention_from_prequantized(dense_carrier)

        reference = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float()
        ).to(torch.bfloat16)
        int8_error = _relative_l2(dense, reference)
        detail['int8_vs_sdpa_rel_l2'] = round(int8_error, 6)
        leg1 = int8_error < _INT8_TOLERANCE and bool(torch.isfinite(dense).all())

        # Leg two must compare like with like. H3 production sparse attention
        # uses CTA_K=128. On Blackwell and any future architecture that chooses
        # architecture-specific probability math, the auto-selected CTA_K=64
        # short-sequence dense path may legitimately use a different kernel
        # specialization from sparse. Pinning both sides to production geometry
        # keeps bit identity a valid traversal invariant on every architecture.
        parity_carrier = native.prequantize_int8_attention(
            q, k, v, cta_k=_SPARSE_PARITY_CTA_K
        )
        parity_dense = native.int8_attention_from_prequantized(parity_carrier)
        detail['sparse_parity_cta_k'] = int(parity_carrier.cta_k)

        kv_tiles = (_KV_LEN + parity_carrier.cta_k - 1) // parity_carrier.cta_k
        q_tiles = (_Q_LEN + native.Q_TILE - 1) // native.Q_TILE
        indices = torch.arange(kv_tiles, dtype=torch.int32, device=device)
        route = native.BlockSparseRoute(
            indices=indices.view(1, 1, 1, -1)
            .expand(_BATCH, _HEADS, q_tiles, -1)
            .contiguous(),
            counts=torch.full(
                (_BATCH, _HEADS, q_tiles), kv_tiles,
                dtype=torch.int32, device=device,
            ),
            q_tile=native.Q_TILE,
            kv_tile=parity_carrier.cta_k,
            encoding='absolute',
        )
        routed = native.block_sparse_int8_attention_from_prequantized(
            parity_carrier, route
        )
        leg2 = torch.equal(routed, parity_dense)
        detail['full_route_bit_identical'] = leg2

        # Asynchronous faults land here, not at the call that caused them.
        torch.cuda.synchronize(device)
    except Exception as error:  # noqa: BLE001 - reporting is the job
        detail['error'] = '%s: %s' % (type(error).__name__, error)
        return False, detail

    passed = bool(leg1 and leg2)
    if verbose:
        print('  dense INT8 vs FP32 SDPA : rel_l2 %.6f (tolerance %s) %s'
              % (int8_error, _INT8_TOLERANCE, 'ok' if leg1 else 'FAIL'))
        print('  100%% route vs dense     : CTA_K=%d bit-identical %s'
              % (parity_carrier.cta_k, leg2))
    return passed, detail


def check(device=None, *, force=False):
    """Cached gate. Runs once per (architecture, build, revision, device)."""
    global _result
    with _lock:
        if _result is not None and not force:
            return _result
        if not torch.cuda.is_available() or not loader.is_available():
            _result = False
            return _result

        key = _cache_key(device)
        cache = _read_cache()
        if not force and key in cache:
            _result = bool(cache[key].get('passed'))
            return _result

        passed, detail = run(device)
        detail['passed'] = passed
        cache[key] = detail
        _write_cache(cache)
        if not passed:
            logging.warning(
                '%s NATIVE SELF-TEST FAILED on %s - refusing the native '
                'kernels and falling back. Detail: %s',
                LOG_PREFIX,
                key,
                detail,
            )
        _result = passed
        return _result
