"""Prove the Kitchen INT8 carrier path is healthy without trusting 64x64 traversal.

The native backend-wide self-test already separates optional sparse geometries, but
this smaller probe is useful to the Triton consumer itself: it exercises the exact
CTA_K=64 carrier that Triton consumes and does not depend on any sparse traversal.
"""

from __future__ import annotations

import logging
import threading

import torch

from . import loader


LOG_PREFIX = '[H3 Optimizations]'
_INT8_TOLERANCE = 0.15
_BATCH, _HEADS, _SEQUENCE, _HEAD_DIM = 1, 2, 129, 128

_lock = threading.Lock()
_cache = {}


def _relative_l2(actual, expected):
    error = (actual.float() - expected.float()).norm()
    return (error / expected.float().norm().clamp_min(1e-12)).item()


def _device_key(device):
    from .bootstrap import installed_build_id

    device = torch.device('cuda' if device is None else device)
    major, minor = torch.cuda.get_device_capability(device)
    return (
        device.index,
        int(major),
        int(minor),
        installed_build_id() or 'local',
        torch.cuda.get_device_name(device),
    )


def run(device=None):
    """Return ``(passed, detail)`` for the 64-wide Kitchen carrier."""
    from . import int8_attention as native

    device = torch.device('cuda' if device is None else device)
    detail = {}
    try:
        generator = torch.Generator(device=device).manual_seed(20260825)

        def sample():
            return torch.randn(
                _BATCH,
                _HEADS,
                _SEQUENCE,
                _HEAD_DIM,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )

        q, k, v = sample(), sample(), sample()
        carrier = native.prequantize_int8_attention(q, k, v, cta_k=64)
        dense = native.int8_attention_from_prequantized(carrier)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float()
        ).to(torch.bfloat16)
        error = _relative_l2(dense, reference)
        finite = bool(torch.isfinite(dense).all())
        torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001 - health probe
        detail['error'] = '%s: %s' % (type(exc).__name__, exc)
        return False, detail

    detail['int8_vs_sdpa_rel_l2'] = round(error, 6)
    detail['finite'] = finite
    detail['cta_k'] = 64
    return bool(finite and error < _INT8_TOLERANCE), detail


def check(device=None, *, force=False):
    if not torch.cuda.is_available() or not loader.is_available():
        return False
    key = _device_key(device)
    with _lock:
        if not force and key in _cache:
            return bool(_cache[key]['passed'])
        passed, detail = run(device)
        detail['passed'] = passed
        _cache[key] = detail
        if not passed:
            logging.warning(
                '%s KITCHEN CTA64 CARRIER SELF-TEST FAILED on %s. Detail: %s',
                LOG_PREFIX,
                key,
                detail,
            )
        return passed
