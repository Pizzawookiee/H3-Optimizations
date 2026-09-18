"""Direct grouped ConvRot-INT8 K/V producer for H3V-Smooth.

The CUTLASS epilogue writes Kitchen K bytes/scales/summaries directly and only
materializes the matching V head in BF16.  K never exists as a BF16 global
memory slab.
"""
from __future__ import annotations

import math
import torch

from . import loader

_SYMBOL = "h3_int8_fused_kv"
_HEAD_DIM = 128
_OUTPUTS = 256
_ROPE_PAIRS = 48
_MAX_INT = 2**31 - 1


def fused_h3_kv_is_available(device=None):
    if not torch.cuda.is_available():
        return False
    try:
        if tuple(torch.cuda.get_device_capability(device)) < (8, 0):
            return False
        library = loader.load()
    except (loader.NativeUnavailableError, RuntimeError):
        return False
    if getattr(library, _SYMBOL, None) is None:
        return False
    from . import selftest
    return selftest.fused_kv_check(device)


def _tensor(name, value, *, dtype, device, dimensions=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if not value.is_cuda or value.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on {device}")
    if dimensions is not None and value.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def fused_h3_kv_from_int8(
    activation,
    weight,
    activation_scale,
    weight_scale,
    norm,
    freqs,
    anchor,
    anchor_index,
    k_out,
    k_scale,
    summary,
    *,
    full_rows,
    k_start,
    cta_k,
    full_k_length,
    epsilon,
):
    """Produce final Kitchen K carrier data plus one BF16 V head."""
    if not isinstance(activation, torch.Tensor) or activation.ndim != 2:
        raise ValueError("activation must be a 2D tensor")
    device = activation.device
    _tensor("activation", activation, dtype=torch.int8, device=device, dimensions=2)
    _tensor("weight", weight, dtype=torch.int8, device=device, dimensions=2)
    _tensor("activation_scale", activation_scale, dtype=torch.float32, device=device, dimensions=2)
    _tensor("weight_scale", weight_scale, dtype=torch.float32, device=device, dimensions=1)
    _tensor("norm", norm, dtype=torch.bfloat16, device=device)
    _tensor("freqs", freqs, dtype=torch.bfloat16, device=device)
    _tensor("anchor", anchor, dtype=torch.bfloat16, device=device)
    _tensor("anchor_index", anchor_index, dtype=torch.int32, device=device)
    _tensor("k_out", k_out, dtype=torch.int8, device=device, dimensions=2)
    _tensor("k_scale", k_scale, dtype=torch.float32, device=device, dimensions=1)
    _tensor("summary", summary, dtype=torch.bfloat16, device=device, dimensions=2)

    rows, hidden = map(int, activation.shape)
    if tuple(weight.shape) != (_OUTPUTS, hidden):
        raise ValueError("weight must contain one 128-row K head followed by one 128-row V head")
    if tuple(activation_scale.shape) != (rows, 1):
        raise ValueError("activation_scale must have shape [rows, 1]")
    if int(weight_scale.numel()) != _OUTPUTS:
        raise ValueError("weight_scale must contain 256 values")
    if int(norm.numel()) != _HEAD_DIM or int(anchor.numel()) != _HEAD_DIM:
        raise ValueError("norm and anchor must each contain 128 values")
    if int(anchor_index.numel()) != 1:
        raise ValueError("anchor_index must contain one int32 value")
    if int(freqs.numel()) != rows * _ROPE_PAIRS * 4:
        raise ValueError("freqs must contain rows*48*2*2 BF16 values")
    full_rows, k_start, cta_k = int(full_rows), int(k_start), int(cta_k)
    full_k_length = int(full_k_length)
    if cta_k not in (64, 128):
        raise ValueError("cta_k must be 64 or 128")
    if not (0 <= k_start < full_rows and k_start + rows <= full_rows):
        raise ValueError("K destination range is invalid")
    if k_start % cta_k:
        raise ValueError("k_start must be aligned to cta_k")
    if tuple(k_out.shape) != (full_rows, _HEAD_DIM):
        raise ValueError("k_out must be the complete [full_rows, 128] head carrier")
    blocks = (full_rows + cta_k - 1) // cta_k
    if int(k_scale.numel()) != blocks * 4 or tuple(summary.shape) != (blocks, _HEAD_DIM):
        raise ValueError("K scale/summary geometry does not match the carrier")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not 0 < full_k_length <= _MAX_INT:
        raise ValueError("full_k_length must be positive")

    v_out = torch.empty((rows, _HEAD_DIM), dtype=torch.bfloat16, device=device)
    library = loader.load()
    function = getattr(library, _SYMBOL, None)
    if function is None:
        raise loader.NativeUnavailableError("native library has no fused H3 K/V producer; rebuild native/")
    loader.check(
        function(
            activation.data_ptr(), weight.data_ptr(), activation_scale.data_ptr(),
            weight_scale.data_ptr(), norm.data_ptr(), freqs.data_ptr(),
            anchor.data_ptr(), anchor_index.data_ptr(), k_out.data_ptr(),
            k_scale.data_ptr(), summary.data_ptr(), v_out.data_ptr(), rows, hidden,
            full_rows, k_start, cta_k, full_k_length, epsilon,
            torch.cuda.current_stream(device).cuda_stream,
        ),
        "fused_h3_kv_exact_128x256",
    )
    return v_out


__all__ = ["fused_h3_kv_from_int8", "fused_h3_kv_is_available"]
