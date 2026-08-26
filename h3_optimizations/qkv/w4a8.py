"""Held native W4A8 bindings for H3 QKV projection."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

import comfy.ops
from comfy.quant_ops import QuantizedTensor

from .. import diagnostics
from .formats import W4A8_LAYOUT, describe_linear


class W4A8BindingError(RuntimeError):
    pass


class HeldW4A8Linear:
    """Acquire one effective W4A8 weight once and preserve Kitchen dispatch."""

    def __init__(self, module, sample):
        self.module = module
        self.sample = sample
        self.weight = None
        self.bias = None
        self.handle = None

    def __enter__(self):
        source = describe_linear(self.module)
        if not source.w4a8:
            raise W4A8BindingError(
                "W4A8 provider received source format %s" % source.label
            )
        weight, bias, handle = comfy.ops.cast_bias_weight(
            self.module,
            self.sample,
            offloadable=True,
            compute_dtype=self.sample.dtype,
            want_requant=True,
        )
        self.weight = weight
        self.bias = bias
        self.handle = handle
        try:
            if not isinstance(weight, QuantizedTensor):
                raise W4A8BindingError(
                    "W4A8 acquisition materialized an unquantized weight"
                )
            if getattr(weight, "_layout_cls", None) != W4A8_LAYOUT:
                raise W4A8BindingError(
                    "W4A8 acquisition returned layout %r"
                    % getattr(weight, "_layout_cls", None)
                )
            if bool(getattr(getattr(weight, "_params", None), "transposed", False)):
                raise W4A8BindingError("W4A8 H3 projection weight is transposed")
            return self
        except Exception:
            self.release()
            raise

    def release(self):
        if self.handle is not None:
            comfy.ops.uncast_bias_weight(
                self.module,
                self.weight,
                self.bias,
                self.handle,
            )
            self.handle = None
        self.weight = None
        self.bias = None
        self.sample = None

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def linear(self, x):
        if self.weight is None:
            raise RuntimeError("W4A8 binding is not active")
        return F.linear(x, self.weight, self.bias)

    def linear_range(self, x, start, end):
        if self.weight is None:
            raise RuntimeError("W4A8 binding is not active")
        start = int(start)
        end = int(end)
        if not 0 <= start < end <= int(self.weight.shape[0]):
            raise W4A8BindingError("W4A8 output slice is invalid")
        params = self.weight._params
        correction = params.correction
        if correction is not None:
            correction = correction[:, start:end]
        weight = QuantizedTensor(
            self.weight._qdata[start:end],
            self.weight._layout_cls,
            replace(
                params,
                scale=params.scale[start:end],
                s_channel=params.s_channel[start:end],
                correction=correction,
                orig_shape=(end - start, int(self.weight.shape[1])),
            ),
        )
        bias = None if self.bias is None else self.bias[start:end]
        return F.linear(x, weight, bias)


class HeldW4A8QKV:
    """Hold native W4A8 QKV across every sequence chunk."""

    def __init__(self, attention, sample):
        self.attention = attention
        self.binding = HeldW4A8Linear(attention.qkv_proj, sample)

    def __enter__(self):
        self.binding.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.binding.__exit__(exc_type, exc, tb)

    def _finish(self, rows, rope):
        from ..attention_forward import finish_qkv_projection, to_hnd

        with diagnostics.stage('qkv_linear'):
            projected = self.binding.linear(rows)
        with diagnostics.stage('qk_norm_rope'):
            return to_hnd(
                *finish_qkv_projection(self.attention, projected, rope)
            )

    def _finish_single_qk(self, projected, rope, norm):
        seq = int(projected.shape[0])
        projected = projected.view(
            1,
            seq,
            self.attention.heads,
            self.attention.head_dim,
        )
        if rope is None:
            return norm(projected[0])
        scale = comfy.ops.cast_to_input(norm.weight, projected)
        projected = F.rms_norm(
            projected,
            (self.attention.head_dim,),
            weight=scale,
            eps=norm.eps,
        )
        rot_dim = int(rope.shape[-3]) * 2
        comfy.quant_ops.ck.apply_rope_split_half1_(projected[..., :rot_dim], rope)
        return projected[0]

    def project_hnd(self, x, rope_freqs, start, end):
        chunk_rope = None if rope_freqs is None else rope_freqs[:, start:end]
        return self._finish(x[start:end], chunk_rope)

    def project_q_hnd(self, x, rope_freqs, start, end):
        inner = int(self.attention.heads) * int(self.attention.head_dim)
        rope = None if rope_freqs is None else rope_freqs[:, start:end]
        with diagnostics.stage("qkv_linear"):
            q = self.binding.linear_range(x[start:end], 0, inner)
        with diagnostics.stage("qk_norm_rope"):
            q = self._finish_single_qk(q, rope, self.attention.q_norm)
        return q.transpose(0, 1).unsqueeze(0)

    def project_kv_hnd(self, x, rope_freqs, start, end):
        inner = int(self.attention.heads) * int(self.attention.head_dim)
        rope = None if rope_freqs is None else rope_freqs[:, start:end]
        with diagnostics.stage("qkv_linear"):
            projected = self.binding.linear_range(x[start:end], inner, inner * 3)
        k, v = projected.split(inner, dim=-1)
        with diagnostics.stage("qk_norm_rope"):
            k = self._finish_single_qk(k, rope, self.attention.k_norm)
        v = v.view(end - start, self.attention.heads, self.attention.head_dim)
        return (
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
        )

    def project_rows(self, x, rope_freqs, rows):
        sample_x = x.index_select(0, rows)
        sample_rope = (
            None if rope_freqs is None else rope_freqs.index_select(1, rows)
        )
        return self._finish(sample_x, sample_rope)
