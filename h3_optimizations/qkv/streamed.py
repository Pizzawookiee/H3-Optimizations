"""Source-aware held QKV bindings for streamed attention consumers."""

from __future__ import annotations

from .bf16 import HeldBF16QKV
from .formats import describe_linear
from .fp8 import HeldFP8QKV
from .int8 import HeldConvRotINT8QKV
from .w4a8 import HeldW4A8QKV


PROJECTION_NATIVE = "native"
PROJECTION_FORCE_BF16 = "force_bf16"
PROJECTION_FORCE_FP8 = "force_fp8"
PROJECTION_FORCE_INT8 = "force_int8"
PROJECTION_MODES = frozenset(
    (
        PROJECTION_NATIVE,
        PROJECTION_FORCE_BF16,
        PROJECTION_FORCE_FP8,
        PROJECTION_FORCE_INT8,
    )
)


class StreamedQKVBindingError(RuntimeError):
    pass


def _native_binding(module, sample):
    actual = describe_linear(module.qkv_proj)
    if actual.convrot_int8_256:
        return HeldConvRotINT8QKV(module, sample)
    if actual.w4a8:
        return HeldW4A8QKV(module, sample)
    if actual.fp8:
        return HeldFP8QKV(module, sample)
    dtype = str(getattr(actual, "logical_dtype", "")).lower()
    if actual.plain_float and ("bfloat16" in dtype or "bf16" in dtype):
        return HeldBF16QKV(module, sample)
    raise StreamedQKVBindingError(
        "native streamed QKV does not support %s" % actual.label
    )


def create_held_qkv(module, sample, projection_mode=PROJECTION_NATIVE):
    """Create, but do not enter, the requested execution-scoped QKV binding."""
    if projection_mode not in PROJECTION_MODES:
        raise ValueError("unknown streamed QKV projection mode %r" % projection_mode)
    if projection_mode == PROJECTION_FORCE_BF16:
        return HeldBF16QKV(module, sample, allow_quantized_source=True)
    if projection_mode == PROJECTION_FORCE_FP8:
        return HeldFP8QKV(module, sample, allow_float_conversion=True)
    if projection_mode == PROJECTION_FORCE_INT8:
        return HeldConvRotINT8QKV(module, sample, allow_float_conversion=True)
    return _native_binding(module, sample)


def project_q_hnd(held, x, rope_freqs, start, end):
    """Project bounded Q, using a true Q-only path when the source exposes one."""
    project_q = getattr(held, "project_q_hnd", None)
    if callable(project_q):
        return project_q(x, rope_freqs, start, end)
    q, k, v = held.project_hnd(x, rope_freqs, start, end)
    del k, v
    return q


def project_kv_hnd(held, x, rope_freqs, start, end):
    """Project K/V without Q when the source binding exposes a split path."""
    project_kv = getattr(held, "project_kv_hnd", None)
    if callable(project_kv):
        return project_kv(x, rope_freqs, start, end)
    q, k, v = held.project_hnd(x, rope_freqs, start, end)
    del q
    return k, v



def project_k_head_rows(held, x, rope_freqs, rows, head):
    """Project one K head for arbitrary absolute sequence rows."""
    project = getattr(held, "project_k_head_rows", None)
    if callable(project):
        return project(x, rope_freqs, rows, head)
    q, k, v = held.project_rows(x, rope_freqs, rows)
    result = k[0, int(head)]
    del q, k, v
    return result


def project_v_head_rows(held, x, rope_freqs, rows, head):
    """Project one V head for arbitrary absolute sequence rows."""
    project = getattr(held, "project_v_head_rows", None)
    if callable(project):
        return project(x, rope_freqs, rows, head)
    q, k, v = held.project_rows(x, rope_freqs, rows)
    result = v[0, int(head)]
    del q, k, v
    return result


def project_grouped_kv_hnd(held, x, rope_freqs, rows):
    """Project per-head arbitrary rows; use a fused provider path when available."""
    project = getattr(held, "project_grouped_kv_hnd", None)
    if callable(project):
        return project(x, rope_freqs, rows)
    heads, count = (int(rows.shape[0]), int(rows.shape[1]))
    k_out = v_out = None
    for head in range(heads):
        k_head = project_k_head_rows(held, x, rope_freqs, rows[head], head)
        v_head = project_v_head_rows(held, x, rope_freqs, rows[head], head)
        if k_out is None:
            dim = int(k_head.shape[-1])
            k_out = k_head.new_empty((1, heads, count, dim))
            v_out = v_head.new_empty((1, heads, count, dim))
        k_out[0, head].copy_(k_head)
        v_out[0, head].copy_(v_head)
    return k_out, v_out


def project_grouped_v_hnd(held, x, rope_freqs, rows):
    """Project grouped V; use a fused provider path when available."""
    project = getattr(held, "project_grouped_v_hnd", None)
    if callable(project):
        return project(x, rope_freqs, rows)
    heads, count = (int(rows.shape[0]), int(rows.shape[1]))
    v_out = None
    for head in range(heads):
        v_head = project_v_head_rows(held, x, rope_freqs, rows[head], head)
        if v_out is None:
            v_out = v_head.new_empty((1, heads, count, int(v_head.shape[-1])))
        v_out[0, head].copy_(v_head)
    return v_out

def project_v_hnd(held, x, rope_freqs, start, end):
    """Project V alone; two-pass staging requires this bounded row slice."""
    project_v = getattr(held, "project_v_hnd", None)
    if not callable(project_v):
        raise StreamedQKVBindingError(
            '%s does not expose V-only projection' % type(held).__name__
        )
    return project_v(x, rope_freqs, start, end)


__all__ = [
    "PROJECTION_FORCE_BF16",
    "PROJECTION_FORCE_FP8",
    "PROJECTION_FORCE_INT8",
    "PROJECTION_MODES",
    "PROJECTION_NATIVE",
    "StreamedQKVBindingError",
    "create_held_qkv",
    "project_grouped_kv_hnd",
    "project_grouped_v_hnd",
    "project_k_head_rows",
    "project_kv_hnd",
    "project_q_hnd",
    "project_v_head_rows",
    "project_v_hnd",
]
