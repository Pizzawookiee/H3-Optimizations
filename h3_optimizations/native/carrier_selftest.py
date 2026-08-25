"""Prove the Kitchen INT8 carrier path is healthy without trusting sparse traversal.

The full native self-test deliberately rejects the whole native attention library when
any shipped sparse geometry is wrong. That is the right gate for executing the native
sparse kernel, but it is too strong for the Triton fallback: Triton only needs the
Kitchen Q/K/V packers (and their carrier ABI), not the native sparse traversal.

This check is therefore the old dense leg in isolation. It exercises anchor selection,
anchor subtraction, ConvRot/Hadamard Q/K packing, per-thread Q/K scales, per-channel V
packing, and the dense consumer on the current GPU. A sparse-only architecture bug can
fail the normal native self-test while this remains healthy.
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
    """Return ``(passed, detail)`` for Kitchen carrier + dense consumption."""
    from . import int8_attention as native

    device = torch.device('cuda' if device is None else device)
    detail = {}
    try:
        generator = torch.Generator(device=device).manual_seed(20260825)
        q = torch.randn(
            _BATCH,
            _HEADS,
            _SEQUENCE,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        k = torch.randn_like(q, generator=generator)
        v = torch.randn_like(q, generator=generator)

        # Force the carrier geometry used by the parity Triton path.
        carrier = native.prequantize_int8_attention(q, k, v, cta_k=64)
        dense = native.int8_attention_from_prequantized(carrier)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float()
        ).to(torch.bfloat16)
        error = _relative_l2(dense, reference)
        finite = bool(torch.isfinite(dense).all())
        torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001 - this function is a health probe
        detail['error'] = '%s: %s' % (type(exc).__name__, exc)
        return False, detail

    detail['int8_vs_sdpa_rel_l2'] = round(error, 6)
    detail['finite'] = finite
    detail['cta_k'] = 64
    return bool(finite and error < _INT8_TOLERANCE), detail


def check(device=None, *, force=False):
    """Cached carrier-only gate used by the Triton Kitchen-parity producer."""
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
                '%s KITCHEN CARRIER SELF-TEST FAILED on %s - refusing the '
                'Kitchen-parity Triton carrier. Detail: %s',
                LOG_PREFIX,
                key,
                detail,
            )
        return passed
