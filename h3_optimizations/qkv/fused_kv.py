"""Exact H3 fused-KV binding for the native Kitchen carrier."""

from __future__ import annotations

import torch

import comfy.model_management
from comfy.quant_ops import QuantizedTensor

from .. import diagnostics
from ..native import (
    fused_h3_kv_from_int8,
    fused_h3_kv_is_available,
    int8_rowwise_convrot256_is_available,
    quantize_int8_rowwise_convrot256,
)
from .formats import describe_linear
from .int8 import HeldConvRotINT8QKV
from .streamed import PROJECTION_NATIVE


HEAD_DIM = 128
CONVROT_GROUP = 256


class FusedH3KVUnavailableError(RuntimeError):
    pass


def fused_h3_kv_supported(module, x, rope_freqs, projection_mode):
    """Whether the exact measured producer matches this execution contract."""
    if projection_mode != PROJECTION_NATIVE:
        return False
    if (
        not isinstance(x, torch.Tensor)
        or x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or rope_freqs is None
        or not isinstance(rope_freqs, torch.Tensor)
        or rope_freqs.dtype != torch.bfloat16
        or rope_freqs.device != x.device
        or int(x.shape[1]) <= 0
        or int(x.shape[1]) % CONVROT_GROUP
        or int(getattr(module, "head_dim", 0)) != HEAD_DIM
        or (int(getattr(module, "heads", 0)) * HEAD_DIM) % 256
        or not describe_linear(module.qkv_proj).convrot_int8_256
    ):
        return False
    return bool(
        int8_rowwise_convrot256_is_available()
        and fused_h3_kv_is_available(x.device)
    )


class HeldExactH3FusedKV:
    """Hold native ConvRot K/V weights while filling global K carriers."""

    def __init__(self, module, sample, rope_freqs, projection_mode):
        self.module = module
        self.sample = sample
        self.rope_freqs = rope_freqs
        self.projection_mode = projection_mode
        self.held = None
        self.weight_qdata = None
        self.weight_scale = None
        self.norm = None

    def __enter__(self):
        if not fused_h3_kv_supported(
            self.module,
            self.sample,
            self.rope_freqs,
            self.projection_mode,
        ):
            raise FusedH3KVUnavailableError(
                "the exact H3 fused-KV producer does not support this projection"
            )

        self.held = HeldConvRotINT8QKV(self.module, self.sample)
        self.held.__enter__()
        try:
            weight = self.held.binding.weight
            if not isinstance(weight, QuantizedTensor):
                raise FusedH3KVUnavailableError(
                    "the exact H3 fused-KV producer requires ConvRot INT8 QKV weights"
                )
            inner = int(self.module.heads) * int(self.module.head_dim)
            if int(weight._qdata.shape[0]) < 3 * inner:
                raise FusedH3KVUnavailableError(
                    "the H3 QKV weight has no complete K/V slice"
                )
            self.weight_qdata = weight._qdata[inner : 3 * inner].contiguous()
            scale = weight._params.scale.to(torch.float32).reshape(-1)
            if int(scale.numel()) != int(weight._qdata.shape[0]):
                raise FusedH3KVUnavailableError(
                    "the H3 QKV weight must have one scale per output row"
                )
            self.weight_scale = scale[inner : 3 * inner].contiguous()
            self.norm = comfy.model_management.cast_to(
                self.module.k_norm.weight,
                dtype=torch.bfloat16,
                device=self.weight_qdata.device,
            ).contiguous()
            if int(self.norm.numel()) != HEAD_DIM:
                raise FusedH3KVUnavailableError(
                    "the exact H3 fused-KV producer requires a 128-value K norm"
                )
            return self
        except Exception:
            self.release()
            raise

    def release(self):
        self.weight_qdata = None
        self.weight_scale = None
        self.norm = None
        self.sample = None
        self.rope_freqs = None
        if self.held is not None:
            self.held.__exit__(None, None, None)
            self.held = None

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def project_rows(self, x, rope_freqs, rows):
        if self.held is None:
            raise RuntimeError("the exact H3 fused-KV binding is not active")
        return self.held.project_rows(x, rope_freqs, rows)

    def project_v_hnd(self, x, rope_freqs, start, end):
        if self.held is None:
            raise RuntimeError("the exact H3 fused-KV binding is not active")
        return self.held.project_v_hnd(x, rope_freqs, start, end)

    def project(self, x, rope_freqs, start, stop, producer):
        if self.held is None:
            raise RuntimeError("the exact H3 fused-KV binding is not active")
        rows = x[start:stop].contiguous()
        freqs = rope_freqs[0, start:stop, 0].contiguous()
        with diagnostics.stage('kv_activation_quant'):
            activation, activation_scale = quantize_int8_rowwise_convrot256(rows)
        with diagnostics.stage('fused_kv_projection'):
            return fused_h3_kv_from_int8(
                activation,
                self.weight_qdata,
                activation_scale,
                self.weight_scale,
                self.norm,
                freqs,
                producer,
                row_start=start,
                epsilon=float(self.module.k_norm.eps),
            )


__all__ = [
    "FusedH3KVUnavailableError",
    "HeldExactH3FusedKV",
    "fused_h3_kv_supported",
]
