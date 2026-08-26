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
from .int8 import HeldConvRotINT8QKV


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


@dataclass
class PreparedStreamedDenseBF16QKV:
    x: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    rope_freqs: torch.Tensor | None
    held: object
    binding_factory: Callable[[], object] | None
    chunk_rows: int
    projection_mode: str

    @property
    def sequence(self):
        return int(self.k.shape[-2])

    def stream_q(self):
        from .streamed import project_q_hnd

        for start in range(0, self.sequence, self.chunk_rows):
            end = min(start + self.chunk_rows, self.sequence)
            if self.held is not None:
                q = project_q_hnd(
                    self.held,
                    self.x,
                    self.rope_freqs,
                    start,
                    end,
                )
            else:
                held = self.binding_factory()
                held.__enter__()
                try:
                    q = project_q_hnd(
                        held,
                        self.x,
                        self.rope_freqs,
                        start,
                        end,
                    )
                finally:
                    held.__exit__(None, None, None)
            yield start, end, q

    def release(self):
        held, self.held = self.held, None
        if held is not None:
            held.__exit__(None, None, None)
        self.x = None
        self.k = self.v = self.rope_freqs = None
        self.binding_factory = None


def _held_cast_handle(held):
    binding = getattr(held, 'binding', held)
    return getattr(binding, 'handle', None)


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

    def _project_slice(self, rows, start, end):
        if self.weight is None:
            raise RuntimeError('held BF16 QKV binding is not active')
        bias = None if self.bias is None else self.bias[start:end]
        comfy.ops.run_every_op()
        with diagnostics.stage('qkv_linear'):
            return F.linear(rows, self.weight[start:end], bias)

    def _finish_single_qk(self, projected, rope, norm):
        seq = int(projected.shape[0])
        projected = projected.view(1, seq, self.attention.heads, self.attention.head_dim)
        if rope is None:
            return norm(projected[0])
        scale = comfy.model_management.cast_to(
            norm.weight,
            device=projected.device,
        )
        projected = F.rms_norm(
            projected,
            (self.attention.head_dim,),
            weight=scale,
            eps=norm.eps,
        )
        rot_dim = int(rope.shape[-3]) * 2
        comfy.quant_ops.ck.apply_rope_split_half1_(
            projected[..., :rot_dim],
            rope,
        )
        return projected[0]

    def project_kv_hnd(self, x, rope_freqs, start, end):
        inner = int(self.attention.heads) * int(self.attention.head_dim)
        rope = None if rope_freqs is None else rope_freqs[:, start:end]
        projected = self._project_slice(x[start:end], inner, inner * 3)
        k, v = projected.split(inner, dim=-1)
        with diagnostics.stage('qk_norm_rope'):
            k = self._finish_single_qk(k, rope, self.attention.k_norm)
        v = v.view(end - start, self.attention.heads, self.attention.head_dim)
        return (
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
        )

    def project_q_hnd(self, x, rope_freqs, start, end):
        inner = int(self.attention.heads) * int(self.attention.head_dim)
        rope = None if rope_freqs is None else rope_freqs[:, start:end]
        q = self._project_slice(x[start:end], 0, inner)
        with diagnostics.stage('qk_norm_rope'):
            q = self._finish_single_qk(q, rope, self.attention.q_norm)
        return q.transpose(0, 1).unsqueeze(0)

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
        force_weights_int8=False,
    ):
        self.chunk_rows = int(chunk_rows)
        self.force_weights_bf16 = bool(force_weights_bf16)
        self.force_weights_fp8 = bool(force_weights_fp8)
        self.force_weights_int8 = bool(force_weights_int8)
        if sum(
            (
                self.force_weights_bf16,
                self.force_weights_fp8,
                self.force_weights_int8,
            )
        ) > 1:
            raise ValueError(
                'QKV weights cannot be forced to more than one execution dtype'
            )
        if self.chunk_rows <= 0:
            raise ValueError('chunk_rows must be positive')

    @property
    def installation_signature(self):
        return (
            self.name,
            self.chunk_rows,
            self.force_weights_bf16,
            self.force_weights_fp8,
            self.force_weights_int8,
        )

    def _validate(self, module, x, rope_freqs, *, allow_cpu=False):
        if comfy.model_management.in_training:
            raise BF16QKVBindingError('chunked BF16 QKV is inference-only')
        if (
            x.ndim != 2
            or (not x.is_cuda and not allow_cpu)
            or x.dtype != torch.bfloat16
        ):
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

        if self.force_weights_int8:
            held = HeldConvRotINT8QKV(
                module,
                x[:1],
                allow_float_conversion=True,
            )
        elif self.force_weights_fp8:
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
            if (
                self.force_weights_bf16
                or self.force_weights_fp8
                or self.force_weights_int8
            ):
                raise
            return None


class StreamedDenseBF16QKVProjector(ChunkedBF16QKVProjector):
    """Keep full BF16 K/V while consuming source-aware Q in bounded slabs."""

    name = 'streamed_dense_bf16_qkv'

    def __init__(
        self,
        chunk_rows=CHUNK_ROWS,
        *,
        projection_mode="native",
        allow_cpu_for_tests=False,
    ):
        super().__init__(chunk_rows)
        self.projection_mode = projection_mode
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)
        self.streamed_q = True

    @property
    def installation_signature(self):
        return (self.name, self.chunk_rows, self.projection_mode)

    def _validate(self, module, x, rope_freqs):
        fmt = super()._validate(
            module,
            x,
            rope_freqs,
            allow_cpu=self.allow_cpu_for_tests,
        )
        if rope_freqs is not None and not callable(
            getattr(comfy.quant_ops.ck, 'apply_rope_split_half1_', None)
        ):
            raise BF16QKVBindingError(
                'streamed dense BF16 QKV requires apply_rope_split_half1_'
            )
        from .streamed import (
            PROJECTION_FORCE_BF16,
            PROJECTION_FORCE_FP8,
            PROJECTION_FORCE_INT8,
            PROJECTION_MODES,
            PROJECTION_NATIVE,
        )

        if self.projection_mode not in PROJECTION_MODES:
            raise ValueError(
                'unknown streamed dense QKV projection mode %r'
                % self.projection_mode
            )
        dtype = str(getattr(fmt, 'logical_dtype', '')).lower()
        native_supported = bool(
            fmt.convrot_int8_256
            or fmt.w4a8
            or fmt.fp8
            or (
                fmt.plain_float
                and ('bfloat16' in dtype or 'bf16' in dtype)
            )
        )
        if self.projection_mode == PROJECTION_NATIVE and not native_supported:
            raise BF16QKVBindingError(
                'streamed dense QKV does not support native %s weights'
                % fmt.label
            )
        if self.projection_mode in (
            PROJECTION_FORCE_FP8,
            PROJECTION_FORCE_INT8,
        ) and not fmt.plain_float:
            raise BF16QKVBindingError(
                '%s streamed dense QKV requires floating source weights'
                % self.projection_mode
            )
        if self.projection_mode == PROJECTION_FORCE_BF16 and not _streamable_format(fmt):
            raise BF16QKVBindingError(
                'BF16 streamed dense QKV does not support %s' % fmt.label
            )
        return fmt

    def project(self, module, x, rope_freqs):
        fmt = self._validate(module, x, rope_freqs)
        from .streamed import PROJECTION_NATIVE, create_held_qkv

        sequence = int(x.shape[0])
        shape = (1, int(module.heads), sequence, int(module.head_dim))
        k_full = torch.empty(shape, dtype=torch.bfloat16, device=x.device)
        v_full = torch.empty(shape, dtype=torch.bfloat16, device=x.device)
        dtype = str(getattr(fmt, 'logical_dtype', '')).lower()
        def binding_factory():
            return (
                HeldBF16QKV(module, x[:1])
                if (
                    self.projection_mode == PROJECTION_NATIVE
                    and fmt.plain_float
                    and ('bfloat16' in dtype or 'bf16' in dtype)
                )
                else create_held_qkv(module, x[:1], self.projection_mode)
            )

        held = binding_factory()
        held.__enter__()
        try:
            for start in range(0, sequence, self.chunk_rows):
                end = min(start + self.chunk_rows, sequence)
                project_kv = getattr(held, 'project_kv_hnd', None)
                if callable(project_kv):
                    k, v = project_kv(x, rope_freqs, start, end)
                else:
                    q, k, v = held.project_hnd(
                        x,
                        rope_freqs,
                        start,
                        end,
                    )
                    del q
                k_full[:, :, start:end, :].copy_(k)
                v_full[:, :, start:end, :].copy_(v)
                del k, v
            if _held_cast_handle(held) is not None:
                held.__exit__(None, None, None)
                held = None
            return PreparedStreamedDenseBF16QKV(
                x=x,
                k=k_full,
                v=v_full,
                rope_freqs=rope_freqs,
                held=held,
                binding_factory=(binding_factory if held is None else None),
                chunk_rows=self.chunk_rows,
                projection_mode=self.projection_mode,
            )
        except Exception:
            if held is not None:
                held.__exit__(None, None, None)
            raise
