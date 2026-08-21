"""Held FP8 linear bindings for H3 QKV and MLP execution."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.ops
from comfy.quant_ops import QuantizedTensor

from .formats import FP8_LAYOUTS

DEFAULT_FP8_LAYOUT = "TensorCoreFP8E4M3Layout"


class FP8BindingError(RuntimeError):
    pass


class HeldFP8Linear:
    """Acquire one H3 linear once and execute it through FP8 matmul.

    Existing FP8 checkpoints retain their layout. Ordinary FP16/BF16 weights are
    deliberately converted to E4M3 for the lifetime of the binding. Other
    quantized layouts are rejected rather than dequantized behind the user's back.
    """

    def __init__(self, module, sample, *, allow_float_conversion=False):
        self.module = module
        self.sample = sample
        self.allow_float_conversion = bool(allow_float_conversion)
        self.weight = None
        self.bias = None
        self.handle = None
        self.layout_type = None
        self.input_scale = None
        self.converted_from_float = False

    def __enter__(self):
        if getattr(self.module, "_full_precision_mm", False):
            raise FP8BindingError("module explicitly requests full-precision matmul")
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
            if isinstance(weight, QuantizedTensor):
                layout = getattr(weight, "_layout_cls", None)
                if layout not in FP8_LAYOUTS:
                    raise FP8BindingError(
                        "FP8 provider received quantized layout %r" % layout
                    )
                self.layout_type = layout
            else:
                if not self.allow_float_conversion:
                    raise FP8BindingError(
                        "FP8 provider received a floating weight without conversion enabled"
                    )
                if getattr(weight, "dtype", None) not in (
                    torch.bfloat16,
                    torch.float16,
                ):
                    raise FP8BindingError(
                        "FP8 conversion requires BF16/FP16 weights, got %s"
                        % getattr(weight, "dtype", None)
                    )
                converted = QuantizedTensor.from_float(
                    weight,
                    DEFAULT_FP8_LAYOUT,
                    scale="recalculate",
                )
                comfy.ops.uncast_bias_weight(
                    self.module,
                    weight,
                    bias,
                    handle,
                )
                self.weight = converted
                self.handle = None
                self.layout_type = DEFAULT_FP8_LAYOUT
                self.converted_from_float = True
            scale = getattr(self.module, "input_scale", None)
            if scale is not None:
                scale = scale.to(device=self.sample.device, non_blocking=True)
            self.input_scale = scale
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
        self.weight = None
        self.bias = None
        self.handle = None
        self.sample = None

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def linear(self, x):
        if self.weight is None or self.layout_type is None:
            raise RuntimeError("FP8 binding is not active")
        input_shape = x.shape
        x2d = x.reshape(-1, input_shape[-1]) if x.ndim != 2 else x
        qx = QuantizedTensor.from_float(
            x2d,
            self.layout_type,
            scale=self.input_scale,
        )
        out = F.linear(qx, self.weight, self.bias)
        if x.ndim != 2:
            out = out.reshape((*input_shape[:-1], self.weight.shape[0]))
        return out


class HeldFP8QKV:
    """Hold an FP8 QKV projection weight across all sequence chunks."""

    def __init__(self, attention, sample, *, allow_float_conversion=False):
        self.attention = attention
        self.binding = HeldFP8Linear(
            attention.qkv_proj,
            sample,
            allow_float_conversion=allow_float_conversion,
        )

    def __enter__(self):
        self.binding.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.binding.__exit__(exc_type, exc, tb)

    def _finish(self, rows, rope):
        from ..attention_forward import finish_qkv_projection, to_hnd

        projected = self.binding.linear(rows)
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


class HeldFP8MLP:
    """Hold fc1/fc2 FP8 bindings across all bounded token slabs."""

    def __init__(self, mlp, sample, *, allow_float_conversion=False):
        self.mlp = mlp
        self.sample = sample
        self.allow_float_conversion = bool(allow_float_conversion)
        self.fc1_binding = None
        self.fc2_binding = None

    def __enter__(self):
        try:
            self.fc1_binding = HeldFP8Linear(
                self.mlp.fc1,
                self.sample,
                allow_float_conversion=self.allow_float_conversion,
            )
            self.fc1_binding.__enter__()
            self.fc2_binding = HeldFP8Linear(
                self.mlp.fc2,
                self.sample,
                allow_float_conversion=self.allow_float_conversion,
            )
            self.fc2_binding.__enter__()
            return self
        except Exception:
            self.release()
            raise

    def release(self):
        if self.fc2_binding is not None:
            self.fc2_binding.__exit__(None, None, None)
            self.fc2_binding = None
        if self.fc1_binding is not None:
            self.fc1_binding.__exit__(None, None, None)
            self.fc1_binding = None

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def fc1_fc2(self, x, swiglu):
        expanded = self.fc1_binding.linear(x)
        activated = swiglu(expanded)
        out = self.fc2_binding.linear(activated)
        return out, "held_fp8"
