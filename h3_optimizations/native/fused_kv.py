"""H3-owned exact 128x256 fused Kitchen K/V producer."""

from __future__ import annotations

import math

import torch

from . import loader


_SYMBOL = "h3_int8_fused_kv"
_HEAD_DIM = 128
_TILE_M = 128
_TILE_N = 256
_ROPE_PAIRS = 48
_MAX_INT = 2**31 - 1


def fused_h3_kv_is_available(device=None):
    """Whether this CUDA device can use the fixed CUTLASS producer."""
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
        raise TypeError("%s must be a torch.Tensor" % name)
    if value.dtype != dtype:
        raise TypeError("%s must have dtype %s, got %s" % (name, dtype, value.dtype))
    if not value.is_cuda:
        raise ValueError("%s must be a CUDA tensor" % name)
    if value.device != device:
        raise ValueError("%s must be on %s" % (name, device))
    if dimensions is not None and value.ndim != dimensions:
        raise ValueError("%s must have %d dimensions" % (name, dimensions))
    if not value.is_contiguous():
        raise ValueError("%s must be contiguous" % name)


def fused_h3_kv_from_int8(
    activation,
    weight,
    activation_scale,
    weight_scale,
    norm,
    freqs,
    producer,
    *,
    row_start,
    epsilon,
):
    """Write exact Kitchen K bytes/scales and return BF16 V plus K summaries."""
    if not isinstance(activation, torch.Tensor) or activation.ndim != 2:
        raise ValueError("activation must be a two-dimensional torch.Tensor")
    if getattr(producer, "_finalized", False):
        raise RuntimeError("the producer has already been finalized")
    device = activation.device
    _tensor("activation", activation, dtype=torch.int8, device=device, dimensions=2)
    _tensor("weight", weight, dtype=torch.int8, device=device, dimensions=2)
    _tensor(
        "activation_scale",
        activation_scale,
        dtype=torch.float32,
        device=device,
        dimensions=2,
    )
    _tensor(
        "weight_scale",
        weight_scale,
        dtype=torch.float32,
        device=device,
        dimensions=1,
    )
    _tensor("norm", norm, dtype=torch.bfloat16, device=device)
    _tensor("freqs", freqs, dtype=torch.bfloat16, device=device)

    spec = producer.spec
    _tensor(
        "producer.k", producer.k, dtype=torch.int8, device=device, dimensions=4
    )
    _tensor(
        "producer.k_scale",
        producer.k_scale,
        dtype=torch.float32,
        device=device,
        dimensions=3,
    )
    _tensor(
        "producer.anchor.values",
        producer.anchor.values,
        dtype=torch.bfloat16,
        device=device,
        dimensions=3,
    )
    _tensor(
        "producer.anchor.indices",
        producer.anchor.indices,
        dtype=torch.int32,
        device=device,
        dimensions=2,
    )

    rows, hidden = (int(value) for value in activation.shape)
    outputs, weight_hidden = (int(value) for value in weight.shape)
    heads = outputs // (2 * _HEAD_DIM)
    full_k_length = int(spec.k_input_shape[2])
    row_start = int(row_start)
    if not 0 < rows <= _MAX_INT or not 0 < hidden <= _MAX_INT:
        raise ValueError("activation dimensions exceed the fused-KV kernel limits")
    if not 0 < outputs <= _MAX_INT or weight_hidden != hidden:
        raise ValueError("weight must be [2*H*128, K] with the activation K dimension")
    if outputs % (2 * _TILE_N):
        raise ValueError("KV output width must be divisible by %d" % (2 * _TILE_N))
    if tuple(activation_scale.shape) != (rows, 1):
        raise ValueError("activation_scale must have shape [M, 1]")
    if int(weight_scale.numel()) != outputs:
        raise ValueError("weight_scale must contain one value per output row")
    if int(norm.numel()) != _HEAD_DIM:
        raise ValueError("norm must contain %d BF16 values" % _HEAD_DIM)
    if int(freqs.numel()) != rows * _ROPE_PAIRS * 2 * 2:
        raise ValueError("freqs must contain M*48*2*2 BF16 values")
    if tuple(spec.k_input_shape) != (1, heads, full_k_length, _HEAD_DIM):
        raise ValueError("producer K geometry does not match the fused-KV outputs")
    if int(spec.cta_k) != 64:
        raise ValueError("the exact fused-KV producer requires 64-row K tiles")
    if tuple(producer.anchor.values.shape) != (1, heads, _HEAD_DIM):
        raise ValueError("producer anchor values have the wrong shape")
    if tuple(producer.anchor.indices.shape) != (1, heads):
        raise ValueError("producer anchor indices have the wrong shape")
    if row_start < 0 or row_start % _TILE_M or row_start + rows > full_k_length:
        raise ValueError("fused-KV rows must start on a 128-row K boundary")
    if row_start + rows < full_k_length and rows % _TILE_M:
        raise ValueError("non-final fused-KV chunks must contain complete 128-row tiles")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if tuple(torch.cuda.get_device_capability(device)) < (8, 0):
        raise loader.NativeUnavailableError(
            "the exact H3 fused-KV producer requires CUDA capability 8.0 or newer"
        )

    library = loader.load()
    function = getattr(library, _SYMBOL, None)
    if function is None:
        raise loader.NativeUnavailableError(
            "the loaded ABI-4 native library has no exact H3 fused-KV producer; "
            "rebuild native/"
        )

    summary_tiles = (rows + 63) // 64
    v_rows = torch.empty(
        (rows, heads, _HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    summary = torch.empty(
        (1, heads, summary_tiles, _HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    loader.check(
        function(
            activation.data_ptr(),
            weight.data_ptr(),
            activation_scale.data_ptr(),
            weight_scale.data_ptr(),
            norm.data_ptr(),
            freqs.data_ptr(),
            producer.anchor.values.data_ptr(),
            producer.anchor.indices.data_ptr(),
            summary.data_ptr(),
            v_rows.data_ptr(),
            producer.k.data_ptr(),
            producer.k_scale.data_ptr(),
            rows,
            outputs,
            hidden,
            full_k_length,
            row_start,
            epsilon,
            torch.cuda.current_stream(device).cuda_stream,
        ),
        "fused_h3_kv_exact_128x256",
    )
    if not producer._q_ranges:
        producer.q.zero_()
        producer.q_scale.fill_(1e-7)
        producer._q_ranges.append((0, 1))
    producer._k_ranges.append((row_start, row_start + rows))
    return v_rows.transpose(0, 1).unsqueeze(0), summary


__all__ = ["fused_h3_kv_from_int8", "fused_h3_kv_is_available"]
