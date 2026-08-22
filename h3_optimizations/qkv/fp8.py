"""Held FP8 linear bindings for H3 QKV and MLP execution."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.ops
from comfy.quant_ops import QuantizedTensor, get_layout_class

from .formats import FP8_LAYOUTS, describe_linear

DEFAULT_FP8_LAYOUT = "TensorCoreFP8E4M3Layout"


class FP8BindingError(RuntimeError):
    pass


def _stream_from_handle(handle):
    if isinstance(handle, tuple) and handle:
        return handle[0]
    return None


def _has_weight_functions(module):
    return bool(getattr(module, "weight_function", ())) or bool(
        getattr(module, "bias_function", ())
    )


def _wrap_raw_fp8(weight, layout_type, output_dtype):
    layout_cls = get_layout_class(layout_type)
    if layout_cls is None:
        raise FP8BindingError("FP8 layout %s is unavailable" % layout_type)
    params = layout_cls.Params(
        scale=torch.ones((), dtype=torch.float32, device=weight.device),
        orig_dtype=output_dtype,
        orig_shape=tuple(weight.shape),
    )
    return QuantizedTensor(weight, layout_type, params)


class HeldFP8Linear:
    """Acquire one H3 linear once and execute it through FP8 matmul.

    Comfy quant-metadata FP8 checkpoints retain their layout. Raw torch FP8
    checkpoints are acquired at their native dtype and wrapped with scale 1 so
    their stored values are not round-tripped through BF16. Ordinary FP16/BF16
    weights are deliberately converted to E4M3 for the lifetime of the binding.
    Other quantized layouts are rejected rather than dequantized behind the
    user's back.
    """

    def __init__(self, module, sample, *, allow_float_conversion=False):
        self.module = module
        self.sample = sample
        self.allow_float_conversion = bool(allow_float_conversion)
        self.weight = None
        self.bias = None
        self.acquired_weight = None
        self.acquired_bias = None
        self.handle = None
        self.layout_type = None
        self.input_scale = None
        self.converted_from_float = False
        self.normalized_raw_fp8 = False

    def _acquire(self, source_format):
        if source_format.raw_fp8 and not _has_weight_functions(self.module):
            source_weight = getattr(self.module, "weight", None)
            raw_dtype = getattr(source_weight, "dtype", None)
            if raw_dtype is None:
                raise FP8BindingError("raw FP8 module has no weight dtype")
            return comfy.ops.cast_bias_weight(
                self.module,
                input=None,
                dtype=raw_dtype,
                device=self.sample.device,
                bias_dtype=self.sample.dtype,
                offloadable=True,
                compute_dtype=self.sample.dtype,
                want_requant=True,
            )
        return comfy.ops.cast_bias_weight(
            self.module,
            self.sample,
            offloadable=True,
            compute_dtype=self.sample.dtype,
            want_requant=True,
        )

    def _release_acquired(self):
        if self.handle is not None:
            comfy.ops.uncast_bias_weight(
                self.module,
                self.acquired_weight,
                self.acquired_bias,
                self.handle,
            )
            self.handle = None
        self.acquired_weight = None
        self.acquired_bias = None

    def __enter__(self):
        if getattr(self.module, "_full_precision_mm", False):
            raise FP8BindingError("module explicitly requests full-precision matmul")

        source_format = describe_linear(self.module)
        weight, bias, handle = self._acquire(source_format)
        self.acquired_weight = weight
        self.acquired_bias = bias
        self.handle = handle
        self.bias = bias

        try:
            if isinstance(weight, QuantizedTensor):
                layout = getattr(weight, "_layout_cls", None)
                if layout not in FP8_LAYOUTS:
                    raise FP8BindingError(
                        "FP8 provider received quantized layout %r" % layout
                    )
                self.weight = weight
                self.layout_type = layout
            elif source_format.raw_fp8:
                layout = source_format.fp8_layout_name
                if layout is None:
                    raise FP8BindingError(
                        "raw FP8 dtype %s has no supported Kitchen layout"
                        % source_format.logical_dtype
                    )
                if getattr(weight, "dtype", None) == getattr(
                    self.module.weight, "dtype", None
                ):
                    self.weight = _wrap_raw_fp8(
                        weight,
                        layout,
                        self.sample.dtype,
                    )
                    self.layout_type = layout
                    self.normalized_raw_fp8 = True
                elif getattr(weight, "dtype", None) in (
                    torch.bfloat16,
                    torch.float16,
                ):
                    # Runtime weight functions can require Comfy to materialize
                    # the effective patched weight in compute dtype. Re-quantize
                    # that effective value back into the checkpoint's FP8 family.
                    self.weight = QuantizedTensor.from_float(
                        weight,
                        layout,
                        scale="recalculate",
                    )
                    self.layout_type = layout
                    self.converted_from_float = True
                    self._release_acquired()
                else:
                    raise FP8BindingError(
                        "raw FP8 acquisition returned unsupported dtype %s"
                        % getattr(weight, "dtype", None)
                    )
            else:
                if not self.allow_float_conversion:
                    raise FP8BindingError(
                        "FP8 provider received a floating weight without conversion enabled"
                    )
                if bias is not None:
                    raise FP8BindingError(
                        "floating FP8 conversion currently requires bias-free H3 linears"
                    )
                if getattr(weight, "dtype", None) not in (
                    torch.bfloat16,
                    torch.float16,
                ):
                    raise FP8BindingError(
                        "FP8 conversion requires BF16/FP16 weights, got %s"
                        % getattr(weight, "dtype", None)
                    )
                self.weight = QuantizedTensor.from_float(
                    weight,
                    DEFAULT_FP8_LAYOUT,
                    scale="recalculate",
                )
                self.layout_type = DEFAULT_FP8_LAYOUT
                self.converted_from_float = True
                self._release_acquired()

            scale = getattr(self.module, "input_scale", None)
            if scale is not None:
                scale = scale.to(device=self.sample.device, non_blocking=True)
            self.input_scale = scale
            return self
        except Exception:
            self.release()
            raise

    def release(self):
        self._release_acquired()
        self.weight = None
        self.bias = None
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

    def project_v_hnd(self, x, start, end):
        rows = x[start:end]
        projected = self.binding.linear(rows)
        heads = int(self.attention.heads)
        head_dim = int(self.attention.head_dim)
        inner = heads * head_dim
        if projected.shape[-1] != 3 * inner:
            raise FP8BindingError(
                "H3 QKV projection width does not match 3 * heads * head_dim"
            )
        v = projected[:, 2 * inner:3 * inner]
        return (
            v.view(end - start, heads, head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
            .contiguous()
        )

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
            stream1 = _stream_from_handle(self.fc1_binding.handle)
            stream2 = _stream_from_handle(self.fc2_binding.handle)
            if stream1 is not None and stream1 is stream2:
                raise FP8BindingError(
                    "fc1 and fc2 were acquired from the same async cast stream; "
                    "the second reusable cast buffer can overwrite the first"
                )
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