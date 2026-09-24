"""H3-owned SM89 Sparse SageAttention2++ consumer."""

from __future__ import annotations

import torch

from . import loader


_SYMBOL = "h3_sparse_sage_sa2pp_sm89"
_OUTPUT_DTYPE = {torch.float16: 1, torch.bfloat16: 2}
_MAX_INT = 2**31 - 1


def sparse_sage_sa2pp_is_available(device=None, capability=None):
    if not torch.cuda.is_available() or getattr(torch, "float8_e4m3fn", None) is None:
        return False
    if capability is None:
        capability = torch.cuda.get_device_capability(device)
    if tuple(capability) != (8, 9):
        return False
    try:
        library = loader.load()
    except (loader.NativeUnavailableError, RuntimeError):
        return False
    return callable(getattr(library, _SYMBOL, None))


def _tensor(name, value, *, dtype, device, dimensions):
    if not isinstance(value, torch.Tensor):
        raise TypeError("%s must be a torch.Tensor" % name)
    if value.dtype != dtype:
        raise TypeError("%s must have dtype %s, got %s" % (name, dtype, value.dtype))
    if not value.is_cuda:
        raise ValueError("%s must be a CUDA tensor" % name)
    if value.device != device:
        raise ValueError("%s must be on %s" % (name, device))
    if value.ndim != dimensions:
        raise ValueError("%s must have %d dimensions" % (name, dimensions))
    if not value.is_contiguous():
        raise ValueError("%s must be contiguous" % name)


def sparse_sage_sa2pp(
    q, k, v, output, lut, valid, pv_threshold, q_scale, k_scale, v_scale,
):
    device = q.device
    _tensor("q", q, dtype=torch.int8, device=device, dimensions=4)
    _tensor("k", k, dtype=torch.int8, device=device, dimensions=4)
    _tensor("v", v, dtype=torch.float8_e4m3fn, device=device, dimensions=4)
    _tensor("output", output, dtype=output.dtype, device=device, dimensions=4)
    _tensor("lut", lut, dtype=torch.int32, device=device, dimensions=4)
    _tensor("valid", valid, dtype=torch.int32, device=device, dimensions=3)
    _tensor("pv_threshold", pv_threshold, dtype=torch.float32, device=device, dimensions=1)
    _tensor("q_scale", q_scale, dtype=torch.float32, device=device, dimensions=3)
    _tensor("k_scale", k_scale, dtype=torch.float32, device=device, dimensions=3)
    _tensor("v_scale", v_scale, dtype=torch.float32, device=device, dimensions=3)
    if output.dtype not in _OUTPUT_DTYPE:
        raise TypeError("Sparse Sage SA2++ output must be FP16 or BF16")

    batch, q_heads, q_length, head_dim = (int(value) for value in q.shape)
    k_batch, kv_heads, kv_length, k_dim = (int(value) for value in k.shape)
    if head_dim != 128 or k_dim != head_dim or k_batch != batch:
        raise ValueError("Sparse Sage SA2++ requires matching 128-wide Q and K")
    if q_heads % kv_heads:
        raise ValueError("Q heads must be divisible by KV heads")
    if tuple(output.shape) != tuple(q.shape):
        raise ValueError("output must have the same shape as Q")
    padded_k = ((kv_length + 127) // 128) * 128
    if tuple(v.shape) != (batch, kv_heads, head_dim, padded_k):
        raise ValueError("V must be the padded [B, H, D, N] FP8 carrier")
    q_tiles = (q_length + 127) // 128
    kv_tiles = (kv_length + 63) // 64
    if tuple(q_scale.shape) != (batch, q_heads, q_tiles):
        raise ValueError("Q scales do not match 128-row tiles")
    if tuple(k_scale.shape) != (batch, kv_heads, kv_tiles):
        raise ValueError("K scales do not match 64-row tiles")
    if tuple(v_scale.shape) != (batch, kv_heads, head_dim):
        raise ValueError("V scales must have shape [B, Hkv, 128]")
    if tuple(lut.shape) != (batch, q_heads, q_tiles, kv_tiles):
        raise ValueError("route LUT does not match 128Q x 64KV geometry")
    if tuple(valid.shape) != tuple(lut.shape[:-1]):
        raise ValueError("valid counts must match the route rows")
    if int(pv_threshold.numel()) != q_heads:
        raise ValueError("PV threshold must contain one value per Q head")
    strides = (
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(2), output.stride(1),
    )
    if max(batch, q_length, kv_length, q_heads, kv_heads, head_dim, *strides) > _MAX_INT:
        raise ValueError("Sparse Sage SA2++ dimensions exceed its 32-bit ABI")
    if tuple(torch.cuda.get_device_capability(device)) != (8, 9):
        raise loader.NativeUnavailableError("the H3 Sparse Sage SA2++ kernel requires SM89")

    library = loader.load()
    function = getattr(library, _SYMBOL, None)
    if function is None:
        raise loader.NativeUnavailableError(
            "the loaded ABI-4 native library has no Sparse Sage SA2++ kernel; rebuild native/"
        )
    loader.check(
        function(
            q.data_ptr(), k.data_ptr(), v.data_ptr(), output.data_ptr(),
            lut.data_ptr(), valid.data_ptr(), pv_threshold.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
            batch, q_length, kv_length, q_heads, kv_heads, head_dim,
            *strides, 128 ** -0.5, _OUTPUT_DTYPE[output.dtype],
            torch.cuda.current_stream(device).cuda_stream,
        ),
        "sparse_sage_sa2pp_sm89",
    )


__all__ = ["sparse_sage_sa2pp", "sparse_sage_sa2pp_is_available"]
