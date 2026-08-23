"""Reusable held-weight BF16 H3 QKV projection.

This module deliberately stops at normalized/post-RoPE BF16 Q/K/V.  It does
not know whether the consumer is Sage, Sol, dense attention, or a future
backend.  Consumers that can ingest one sequence slab at a time should use
``stream`` so no full-sequence fused QKV temporary is ever materialized.
Consumers that still require complete Q/K/V tensors can use ``project``;
that path keeps the projection chunked but necessarily allocates the final
full-sequence Q/K/V buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
from comfy.quant_ops import QuantizedTensor

from .. import diagnostics
from ..attention_forward import finish_qkv_projection, to_hnd
from .formats import describe_linear


CHUNK_ROWS = 4096


class BF16QKVBindingError(RuntimeError):
    pass


@dataclass
class PreparedBF16QKV:
    """Complete post-RoPE HND Q/K/V for consumers that cannot stream yet."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor

    @property
    def sequence(self):
        return int(self.q.shape[-2])

    @property
    def heads(self):
        return int(self.q.shape[1])

    @property
    def head_dim(self):
        return int(self.q.shape[-1])


class HeldBF16QKV:
    """Acquire one floating H3 QKV weight once across all token chunks."""

    def __init__(self, attention, sample):
        self.attention = attention
        self.sample = sample
        self.weight = None
        self.bias = None
        self.handle = None

    def __enter__(self):
        if self.sample.ndim != 2 or self.sample.dtype != torch.bfloat16:
            raise BF16QKVBindingError(
                "held BF16 QKV requires a rank-2 BF16 activation sample"
            )
        fmt = describe_linear(self.attention.qkv_proj)
        if not fmt.plain_float:
            raise BF16QKVBindingError(
                "held BF16 QKV requires floating QKV weights, got %s" % fmt.label
            )
        weight, bias, handle = comfy.ops.cast_bias_weight(
            self.attention.qkv_proj,
            self.sample,
            offloadable=True,
            compute_dtype=torch.bfloat16,
            want_requant=False,
        )
        try:
            if isinstance(weight, QuantizedTensor):
                raise BF16QKVBindingError(
                    "held BF16 QKV unexpectedly acquired a quantized weight"
                )
            if getattr(weight, "dtype", None) != torch.bfloat16:
                raise BF16QKVBindingError(
                    "held BF16 QKV acquired %s instead of bfloat16"
                    % getattr(weight, "dtype", None)
                )
            if bias is not None and getattr(bias, "dtype", None) != torch.bfloat16:
                bias = bias.to(dtype=torch.bfloat16)
            self.weight = weight
            self.bias = bias
            self.handle = handle
            return self
        except Exception:
            comfy.ops.uncast_bias_weight(
                self.attention.qkv_proj,
                weight,
                bias,
                handle,
            )
            raise

    def release(self):
        if self.handle is not None:
            comfy.ops.uncast_bias_weight(
                self.attention.qkv_proj,
                self.weight,
                self.bias,
                self.handle,
            )
        self.weight = None
        self.bias = None
        self.handle = None
        self.sample = None

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def _finish(self, rows, rope):
        if self.weight is None:
            raise RuntimeError("held BF16 QKV binding is not active")
        comfy.ops.run_every_op()
        with diagnostics.stage("qkv_linear"):
            projected = F.linear(rows, self.weight, self.bias)
        with diagnostics.stage("qk_norm_rope"):
            return to_hnd(
                *finish_qkv_projection(self.attention, projected, rope)
            )

    def project_hnd(self, x, rope_freqs, start, end):
        chunk_rope = None if rope_freqs is None else rope_freqs[:, start:end]
        return self._finish(x[start:end], chunk_rope)

    def project_rows(self, x, rope_freqs, rows):
        sample_x = x.index_select(0, rows)
        sample_rope = (
            None if rope_freqs is None else rope_freqs.index_select(1, rows)
        )
        return self._finish(sample_x, sample_rope)


class ChunkedBF16QKVProjector:
    """Project native floating H3 QKV in bounded BF16 sequence slabs.

    ``stream`` is the preferred interface: each projected chunk is handed to a
    backend-owned callback and then released.  ``project`` is a compatibility
    helper for backends that need complete HND Q/K/V tensors.
    """

    name = "chunked_bf16_qkv"

    def __init__(self, chunk_rows=CHUNK_ROWS):
        chunk_rows = int(chunk_rows)
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        self.chunk_rows = chunk_rows

    @property
    def installation_signature(self):
        return (self.name, self.chunk_rows)

    def _validate(self, module, x, rope_freqs):
        if comfy.model_management.in_training:
            raise BF16QKVBindingError("chunked BF16 QKV is inference-only")
        if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16:
            raise BF16QKVBindingError(
                "chunked BF16 QKV requires a rank-2 CUDA BF16 activation"
            )
        fmt = describe_linear(module.qkv_proj)
        if not fmt.plain_float:
            raise BF16QKVBindingError(
                "chunked BF16 QKV requires floating QKV weights, got %s" % fmt.label
            )
        if rope_freqs is not None and (
            rope_freqs.ndim != 6
            or tuple(rope_freqs.shape[:3]) != (1, x.shape[0], 1)
            or rope_freqs.device != x.device
        ):
            raise BF16QKVBindingError("chunked BF16 QKV received invalid RoPE")

    def stream(
        self,
        module,
        x,
        rope_freqs,
        consume_chunk: Callable[[int, int, torch.Tensor, torch.Tensor, torch.Tensor], None],
    ):
        """Project each post-RoPE HND chunk and immediately hand it to a consumer."""
        self._validate(module, x, rope_freqs)
        if not callable(consume_chunk):
            raise TypeError("consume_chunk must be callable")
        sequence = int(x.shape[0])
        held = HeldBF16QKV(module, x[:1])
        held.__enter__()
        try:
            for start in range(0, sequence, self.chunk_rows):
                end = min(start + self.chunk_rows, sequence)
                q, k, v = held.project_hnd(x, rope_freqs, start, end)
                consume_chunk(start, end, q, k, v)
                del q, k, v
        finally:
            held.__exit__(None, None, None)

    def project(self, module, x, rope_freqs):
        """Return complete HND Q/K/V while keeping the fused projection bounded."""
        self._validate(module, x, rope_freqs)
        sequence = int(x.shape[0])
        heads = int(module.heads)
        head_dim = int(module.head_dim)
        shape = (1, heads, sequence, head_dim)
        q_full = torch.empty(shape, dtype=torch.bfloat16, device=x.device)
        k_full = torch.empty(shape, dtype=torch.bfloat16, device=x.device)
        v_full = torch.empty(shape, dtype=torch.bfloat16, device=x.device)

        def consume(start, end, q, k, v):
            q_full[:, :, start:end, :].copy_(q)
            k_full[:, :, start:end, :].copy_(k)
            v_full[:, :, start:end, :].copy_(v)

        self.stream(module, x, rope_freqs, consume)
        return PreparedBF16QKV(q=q_full, k=k_full, v=v_full)

    def try_project(
        self,
        module,
        x,
        rope_freqs,
        *,
        layer_index=None,
        transformer_options=None,
    ):
        del layer_index, transformer_options
        try:
            return self.project(module, x, rope_freqs)
        except BF16QKVBindingError:
            return None
