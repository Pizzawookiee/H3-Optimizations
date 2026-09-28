"""Experimental 64Q x 64KV FP8 attention and V-Smooth carriers.

This module is intentionally outside the production attention dispatcher. It
shares the native INT8 Q/K carrier so benchmark differences are confined to
the probability cast and the value path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..attention.sage_v_fp8 import direct_per_channel_fp8
from . import loader
from .int8_attention import (
    OUTPUT_HND,
    _DTYPE_TO_CODE,
    _attention_geometry,
    _finish,
    _ptr,
    _stream,
    _validate_sparse_route,
)


FP8_MAX = 448.0
V_SMOOTH_BLOCK = 128


@dataclass(frozen=True)
class Fp8ValueCarrier:
    values: torch.Tensor
    scale: torch.Tensor
    mean: torch.Tensor | None
    padded_length: int
    block_size: int | None


def _pad_to(value, multiple):
    return ((int(value) + multiple - 1) // multiple) * multiple


def _fp8_source_index(length, device):
    destination = torch.arange(length, dtype=torch.int64, device=device)
    low = destination & 15
    source_low = (
        (low & 1)
        | ((low & 4) >> 1)
        | ((low & 8) >> 1)
        | ((low & 2) << 2)
    )
    return (destination & ~15) | source_low


def _pack_permuted_fp8(values, padded_length):
    batch, heads, length, width = values.shape
    if padded_length != length:
        values = torch.nn.functional.pad(values, (0, 0, 0, padded_length - length))
    values = values.permute(0, 1, 3, 2)
    source_index = _fp8_source_index(padded_length, values.device)
    return values.index_select(-1, source_index).contiguous().reshape(
        batch * heads * width, padded_length
    )


def pack_fp8_values(values):
    """Per-channel E4M3 value carrier matching the native MMA layout."""
    if values.ndim != 4 or values.shape[-1] != 128:
        raise ValueError('values must be [batch, heads, sequence, 128]')
    batch, heads, length, width = values.shape
    padded_length = _pad_to(length, 64)
    quantized, scale = direct_per_channel_fp8(
        values, scale_max=FP8_MAX, pad_to=64
    )
    return Fp8ValueCarrier(
        values=quantized.reshape(batch * heads * width, padded_length),
        scale=scale.contiguous(),
        mean=None,
        padded_length=padded_length,
        block_size=None,
    )


def pack_v_smooth_values(values, block_size=V_SMOOTH_BLOCK):
    """Block-demean and quantize values after their k-means permutation."""
    if block_size != V_SMOOTH_BLOCK:
        raise ValueError('the experimental kernel is fixed to 128-row V blocks')
    if values.ndim != 4 or values.shape[-1] != 128:
        raise ValueError('values must be [batch, heads, sequence, 128]')

    batch, heads, length, width = values.shape
    padded_length = _pad_to(length, block_size)
    padded = torch.nn.functional.pad(
        values.float(), (0, 0, 0, padded_length - length)
    ).reshape(batch, heads, padded_length // block_size, block_size, width)
    live = torch.arange(padded_length, device=values.device) < length
    live = live.reshape(1, 1, -1, block_size, 1)
    count = live.sum(dim=3).clamp_min_(1)
    mean = (padded * live).sum(dim=3) / count
    mean = mean.to(torch.bfloat16).float()
    residual = (padded - mean.unsqueeze(3)) * live
    scale = residual.abs().amax(dim=3).clamp_min_(1.0e-12) / FP8_MAX
    quantized = (residual / scale.unsqueeze(3)).to(torch.float8_e4m3fn)
    quantized = quantized.reshape(batch, heads, padded_length, width)
    # VC-Attention stores the 16-bit block mean divided by the per-channel V
    # scale, allowing the attention epilogue to scale the tensor-core and mean
    # terms together.
    scaled_mean = (mean / scale).to(torch.bfloat16).float()
    return Fp8ValueCarrier(
        values=_pack_permuted_fp8(quantized, padded_length),
        scale=scale.contiguous(),
        mean=scaled_mean.contiguous(),
        padded_length=padded_length,
        block_size=block_size,
    )


def fp8_attention_64_from_prequantized(
    quantized,
    values,
    route,
    kv_tile_len,
    *,
    expcast,
    pre_fp8_denom=False,
    output_layout=OUTPUT_HND,
):
    """Run the optional SM89 FP8 kernel over prepared Q/K and value carriers."""
    library = loader.load()
    if not hasattr(library, 'h3_fp8_attention_64'):
        raise loader.NativeUnavailableError(
            'the loaded native library has no FP8 ExpCast experiment'
        )
    if torch.cuda.get_device_capability(quantized.q.device) != (8, 9):
        raise RuntimeError('the FP8 ExpCast experiment is built only for SM89')
    if quantized.cta_k != 64 or (int(route.q_tile), int(route.kv_tile)) != (64, 64):
        raise ValueError('FP8 ExpCast is fixed to 64Q x 64KV carriers')
    if values.padded_length < quantized.k.shape[2]:
        raise ValueError('the FP8 value carrier is shorter than K')
    if pre_fp8_denom and (expcast or values.mean is not None):
        raise ValueError(
            'pre-FP8 denominator supports only ordinary FP8 probabilities'
        )

    kernel_route = _validate_sparse_route(quantized, route)
    kv_tile_len = kv_tile_len.to(
        device=quantized.q.device, dtype=torch.int32
    ).contiguous()
    kv_tiles = (int(quantized.k.shape[2]) + 63) // 64
    if kv_tile_len.numel() != kv_tiles:
        raise ValueError('kv_tile_len must have one entry per 64-row KV tile')

    output, output_dtype, geometry, strides = _attention_geometry(
        quantized, output_layout
    )
    loader.check(
        library.h3_fp8_attention_64(
            _ptr(quantized.q), _ptr(quantized.k), _ptr(values.values),
            _ptr(output), _ptr(quantized.q_scale), _ptr(quantized.k_scale),
            _ptr(values.scale), _ptr(values.mean), _ptr(kernel_route.indices),
            _ptr(kernel_route.counts), _ptr(kv_tile_len),
            kernel_route.indices.shape[-1], *geometry, *strides,
            quantized.q_scale.stride(0), quantized.q_scale.stride(1),
            quantized.attention_scale, _DTYPE_TO_CODE[output_dtype],
            int(bool(expcast)), int(bool(pre_fp8_denom)),
            int(values.mean is not None), _stream(),
        ),
        'FP8 ExpCast attention',
    )
    return _finish(quantized, output)
