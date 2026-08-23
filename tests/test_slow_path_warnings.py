"""A degraded fast path has to announce itself.

The chunked Comfy Kitchen QKV producer and the native sparse kernel are the
two paths this pack exists to provide. Both degrade gracefully when their
dependency is missing -- they set an honest reason and carry on -- and that
politeness is exactly how a missing dependency came to look like a working
install for a long time. Roughly half the speed, no message.

So: warn when a fast path was asked for and something slower was used, stay
quiet when the slower thing was chosen deliberately.
"""

import logging
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_optimizations import apply as apply_module  # noqa: E402
from h3_optimizations.qkv.providers import (  # noqa: E402
    QKV_DENSE_KITCHEN_CHUNKED,
    QKV_STANDARD,
    QKVProviderResolution,
)


def _attention(requested, selected, reason="because"):
    return apply_module.ResolvedAttention(
        requested=requested,
        selected=selected,
        backend=None,
        reason=reason,
        backend_kind=selected,
    )


def _warnings(caplog, attention, qkv):
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        apply_module._warn_about_slow_paths(attention, qkv)
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_sparse_fallback_to_triton_is_loud(caplog):
    messages = _warnings(
        caplog,
        _attention(
            apply_module.ATTENTION_SPARSE,
            apply_module.ATTENTION_TRITON_SPARSE,
            reason="Sparse Sage unavailable: no compiled kernel",
        ),
        QKVProviderResolution(QKV_DENSE_KITCHEN_CHUNKED, False, "fine"),
    )
    assert len(messages) == 1
    assert "FELL BACK" in messages[0]
    assert "half the speed" in messages[0]
    assert "no compiled kernel" in messages[0]


def test_sparse_fallback_to_flex_is_loud(caplog):
    messages = _warnings(
        caplog,
        _attention(
            apply_module.ATTENTION_SPARSE, apply_module.ATTENTION_FP8_FLEX
        ),
        QKVProviderResolution(QKV_DENSE_KITCHEN_CHUNKED, False, "fine"),
    )
    assert len(messages) == 1
    assert "far slower" in messages[0]


def test_kitchen_sparse_fallback_is_loud(caplog):
    messages = _warnings(
        caplog,
        _attention(
            apply_module.ATTENTION_KITCHEN_SPARSE,
            apply_module.ATTENTION_TRITON_SPARSE,
        ),
        QKVProviderResolution(QKV_DENSE_KITCHEN_CHUNKED, False, "fine"),
    )
    assert len(messages) == 1


def test_missing_kitchen_producer_is_loud(caplog):
    """The exact failure that made a missing dependency look like a working install."""
    messages = _warnings(
        caplog,
        _attention(apply_module.ATTENTION_SPARSE, apply_module.ATTENTION_SPARSE),
        QKVProviderResolution(
            QKV_STANDARD,
            False,
            "Comfy Kitchen external producer API is unavailable",
        ),
    )
    assert len(messages) == 1
    assert "FUSED QKV IS NOT RUNNING" in messages[0]
    assert "producer API is unavailable" in messages[0]


def test_both_degradations_warn_separately(caplog):
    messages = _warnings(
        caplog,
        _attention(
            apply_module.ATTENTION_SPARSE, apply_module.ATTENTION_TRITON_SPARSE
        ),
        QKVProviderResolution(
            QKV_STANDARD, False, "Comfy Kitchen external producer API is unavailable"
        ),
    )
    assert len(messages) == 2


def test_a_healthy_resolution_is_silent(caplog):
    messages = _warnings(
        caplog,
        _attention(apply_module.ATTENTION_SPARSE, apply_module.ATTENTION_SPARSE),
        QKVProviderResolution(QKV_DENSE_KITCHEN_CHUNKED, False, "fine"),
    )
    assert messages == []


def test_deliberately_choosing_a_slower_backend_is_silent(caplog):
    """Explicitly asking for Triton is a choice, not a degradation."""
    messages = _warnings(
        caplog,
        _attention(
            apply_module.ATTENTION_TRITON_SPARSE,
            apply_module.ATTENTION_TRITON_SPARSE,
        ),
        QKVProviderResolution(QKV_DENSE_KITCHEN_CHUNKED, False, "fine"),
    )
    assert messages == []


def test_standard_qkv_for_an_unrelated_reason_is_silent(caplog):
    """Only the missing Kitchen producer is worth shouting about here."""
    messages = _warnings(
        caplog,
        _attention(apply_module.ATTENTION_SPARSE, apply_module.ATTENTION_SPARSE),
        QKVProviderResolution(
            QKV_STANDARD, False, "QKV projection optimization was disabled"
        ),
    )
    assert messages == []
