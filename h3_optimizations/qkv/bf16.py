"""Reusable bounded BF16 H3 QKV projection.

The compatibility contract is about the projected activation dtype, not the
checkpoint weight dtype. Supported H3 QKV weights are projected in bounded
sequence slabs and normalized/RoPE-applied as BF16 Q/K/V. Consumers that cannot
stream may still require complete Q/K/V tensors; in that case only the fused
projection temporary is bounded.
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
from ..attention_forward import finish_qkv_projection, project_qkv, to_hnd
from .fp8 import HeldFP8QKV
from .formats import describe_linear


CHUNK_ROWS = 4096


class BF16QKVBindingError(RuntimeError):
    pass


@dataclass
class PreparedBF16QKV:
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

    def __init__(self, attention, sample, *, allow_quantized_source=False):
        self.attention = attention
        self.sample = sample
        self.allow_quantized_source = bool(allow_quantized_source)
        self.weight = None
        self.bias = None
        self.acquired_weight = None
        self.acquired_bias = None
        self.handle = None

    def __enter__(self):
        if self.sample.ndim != 2 or self.sample.dtype != torch.bfloat16:
            raise BF16QKVBindingError(
                'held BF16 QKV requires a rank-2 BF16 activation sample'
            )
        fmt = describe_linear(self.attention.qkv_proj)
        if not fmt.plain_float and not self.allow_quantized_source:
            raise BF16QKVBindingError(
                'held BF16 QKV requires floating QKV weights, got %s' % fmt.label
            )
        weight, bias, handle = comfy.ops.cast_bias_weight(
            self.attention.qkv_proj,
            self.sample,
            offloadable=True,
            compute_dtype=torch.bfloat16,
            want_requant=False,
        )
        self.acquired_weight = weight
        self.acquired_bias = bias
        self.handle = handle
        try:
            if isinstance(weight, QuantizedTensor):
                raise BF16QKVBindingError(
                    'held BF16 QKV unexpectedly acquired a quantized weight'
                )
            if getattr(weight, 'dtype', None) != torch.bfloat16:
                raise BF16QKVBindingError(
                    'held BF16 QKV acquired %s instead of bfloat16'
                    % getattr(weight, 'dtype', None)
                )
            if bias is not None and getattr(bias, 'dtype', None) != torch.bfloat16:
                raise BF16QKVBindingError('held BF16 QKV bias is not bfloat16')
            self.weight = weight
            self.bias = bias
            return self
        except Exception:
            self.release()
            raise

    def release(self):
        if self.handle is not None:
            comfy.ops.uncast_bias_weight(
                self.attention.qkv_proj,
                self.acquired_weight,
                self.acquired_bias,
                self.handle,
            )
        self.weight = self.bias = None
        self.acquired_weight = self.acquired_bias = None
        self.handle = None
        self.sample = None

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def _finish(self, rows, rope):
        if self.weight is None:
            raise RuntimeError('held BF16 QKV binding is not active')
        comfy.ops.run_every_op()
        with diagnostics.stage('qkv_linear'):
            projected = F.linear(rows, self.weight, self.bias)
        with diagnostics.stage('qk_norm_rope'):
            return to_hnd(*finish_qkv_projection(self.attention, projected, rope))

    def project_hnd(self, x, rope_freqs, start, end):
        rope = None if rope_freqs is None else rope_freqs[:, start:end]
        return self._finish(x[start:end], rope)

    def project_rows(self, x, rope_freqs, rows):
        sample_x = x.index_select(0, rows)
        sample_rope = None if rope_freqs is None else rope_freqs.index_select(1, rows)
        return self._finish(sample_x, sample_rope)


def _streamable_format(fmt):
    return bool(
        fmt.plain_float
        or fmt.convrot_int8_256
        or fmt.w4a8
        or fmt.fp8
    )


class ChunkedBF16QKVProjector:
    """Produce post-RoPE BF16 Q/K/V in bounded sequence slabs."""

    name = 'chunked_bf16_qkv'

    def __init__(
        self,
        chunk_rows=CHUNK_ROWS,
        *,
        force_weights_bf16=False,
        force_weights_fp8=False,
    ):
        self.chunk_rows = int(chunk_rows)
        self.force_weights_bf16 = bool(force_weights_bf16)
        self.force_weights_fp8 = bool(force_weights_fp8)
        if self.force_weights_bf16 and self.force_weights_fp8:
            raise ValueError('QKV weights cannot be forced to both BF16 and FP8')
        if self.chunk_rows <= 0:
            raise ValueError('chunk_rows must be positive')

    @property
    def installation_signature(self):
        return (
            self.name,
            self.chunk_rows,
            self.force_weights_bf16,
            self.force_weights_fp8,
        )

    def _validate(self, module, x, rope_freqs):
        if comfy.model_management.in_training:
            raise BF16QKVBindingError('chunked BF16 QKV is inference-only')
        if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16:
            raise BF16QKVBindingError(
                'chunked BF16 QKV requires rank-2 CUDA BF16 activations'
            )
        fmt = describe_linear(module.qkv_proj)
        if not _streamable_format(fmt):
            raise BF16QKVBindingError(
                'chunked BF16 QKV does not support %s' % fmt.label
            )
        if rope_freqs is not None and (
            rope_freqs.ndim != 6
            or tuple(rope_freqs.shape[:3]) != (1, x.shape[0], 1)
            or rope_freqs.device != x.device
        ):
            raise BF16QKVBindingError('chunked BF16 QKV received invalid RoPE')
        return fmt

    def stream(
        self,
        module,
        x,
        rope_freqs,
        consume_chunk: Callable[[int, int, torch.Tensor, torch.Tensor, torch.Tensor], None],
    ):
        fmt = self._validate(module, x, rope_freqs)
        if not callable(consume_chunk):
            raise TypeError('consume_chunk must be callable')
        sequence = int(x.shape[0])

        if self.force_weights_fp8:
            held = HeldFP8QKV(
                module,
                x[:1],
                allow_float_conversion=True,
            )
        else:
            held = (
                HeldBF16QKV(
                    module,
                    x[:1],
                    allow_quantized_source=self.force_weights_bf16,
                )
                if fmt.plain_float or self.force_weights_bf16
                else None
            )
        if held is not None:
            held.__enter__()
        try:
            for start in range(0, sequence, self.chunk_rows):
                end = min(start + self.chunk_rows, sequence)
                rope = None if rope_freqs is None else rope_freqs[:, start:end]
                if held is not None:
                    q, k, v = held.project_hnd(x, rope_freqs, start, end)
                else:
                    q, k, v = to_hnd(*project_qkv(module, x[start:end], rope))
                consume_chunk(start, end, q, k, v)
                del q, k, v
        finally:
            if held is not None:
                held.__exit__(None, None, None)

    def project(self, module, x, rope_freqs):
        self._validate(module, x, rope_freqs)
        sequence = int(x.shape[0])
        shape = (1, int(module.heads), sequence, int(module.head_dim))
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
            if self.force_weights_bf16 or self.force_weights_fp8:
                raise
            return None
