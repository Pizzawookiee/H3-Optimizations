"""Execution-scoped ConvRot INT8 bindings for floating H3 linears."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
import comfy.quant_ops
from comfy.quant_ops import QuantizedTensor
from comfy.weight_adapter.lora import LoRAAdapter

from .. import diagnostics
from .formats import describe_linear, describe_weight


LAYOUT = "TensorWiseINT8Layout"
GROUP_SIZE = 256


class ConvRotINT8BindingError(RuntimeError):
    pass


class _SlicedLoRA:
    __slots__ = ("up", "down", "scale", "key")
    def __init__(self, up, down, scale, key):
        self.up, self.down, self.scale, self.key = up, down, float(scale), str(key)
    def apply(self, x, base, row_start, row_end):
        if self.scale == 0.0:
            return base
        down = comfy.model_management.cast_to_device(self.down, x.device, x.dtype)
        up = comfy.model_management.cast_to_device(self.up[int(row_start):int(row_end)], x.device, x.dtype)
        hidden = F.linear(x, down, None)
        delta = F.linear(hidden, up, None)
        base.add_(delta, alpha=self.scale)
        del hidden, delta, up, down
        return base


def _lowvram_patch_objects(module):
    result, seen = [], set()
    direct = getattr(module, "weight_lowvram_function", None)
    if direct is not None:
        if not getattr(direct, "is_lowvram_patch", False):
            raise ConvRotINT8BindingError("unsupported weight_lowvram_function")
        result.append(direct); seen.add(id(direct))
    for fn in getattr(module, "weight_function", []) or []:
        if getattr(fn, "is_lowvram_patch", False):
            if id(fn) not in seen:
                result.append(fn); seen.add(id(fn))
        else:
            raise ConvRotINT8BindingError("sliced-LoRA bypass does not support additional weight wrappers")
    return result


def _extract_plain_qkv_loras(module):
    lows = _lowvram_patch_objects(module)
    if not lows:
        return ()
    output_rows, input_cols = int(module.weight.shape[0]), int(module.weight.shape[1])
    result = []
    for low in lows:
        key, patch_map = getattr(low, "key", None), getattr(low, "patches", None)
        if key is None or patch_map is None or key not in patch_map:
            raise ConvRotINT8BindingError("QKV LowVramPatch does not expose original patch list")
        for patch in patch_map[key]:
            if len(patch) != 5:
                raise ConvRotINT8BindingError("unsupported QKV patch tuple")
            strength, adapter, strength_model, offset, function = patch
            if not isinstance(adapter, LoRAAdapter):
                raise ConvRotINT8BindingError("sliced-LoRA supports only standard Comfy LoRAAdapter")
            if float(strength_model) != 1.0 or offset is not None or function is not None:
                raise ConvRotINT8BindingError("unsupported QKV LoRA patch semantics")
            up, down, alpha, mid, dora_scale, reshape = adapter.weights
            if mid is not None or dora_scale is not None or reshape is not None:
                raise ConvRotINT8BindingError("LoCon/DoRA/reshape QKV patches are not supported")
            if up.ndim != 2 or down.ndim != 2:
                raise ConvRotINT8BindingError("QKV LoRA factors must be 2D")
            rank = int(down.shape[0])
            if rank <= 0 or int(up.shape[1]) != rank or int(up.shape[0]) != output_rows or int(down.shape[1]) != input_cols:
                raise ConvRotINT8BindingError("QKV LoRA dimensions do not match fused QKV")
            alpha_scale = float(alpha) / rank if alpha is not None else 1.0
            result.append(_SlicedLoRA(up, down, float(strength) * alpha_scale, key))
    return tuple(result)


def _temporarily_disable_qkv_lowvram_patch(module):
    had_low = hasattr(module, "weight_lowvram_function")
    saved_low = getattr(module, "weight_lowvram_function", None)
    had_functions = hasattr(module, "weight_function")
    saved_functions = getattr(module, "weight_function", None)
    if had_low: module.weight_lowvram_function = None
    if had_functions: module.weight_function = []
    return had_low, saved_low, had_functions, saved_functions


def _restore_qkv_lowvram_patch(module, state):
    had_low, saved_low, had_functions, saved_functions = state
    if had_low: module.weight_lowvram_function = saved_low
    elif hasattr(module, "weight_lowvram_function"): delattr(module, "weight_lowvram_function")
    if had_functions: module.weight_function = saved_functions
    elif hasattr(module, "weight_function"): delattr(module, "weight_function")


class OwnedConvRotINT8QProjection:
    def __init__(self, attention, weight, sliced_loras=()):
        self.attention, self.weight, self.sliced_loras = attention, weight, tuple(sliced_loras)
    def release(self):
        self.weight, self.sliced_loras, self.attention = None, (), None
    def _finish_q(self, projected, rope):
        seq = int(projected.shape[0])
        projected = projected.view(1, seq, self.attention.heads, self.attention.head_dim)
        norm = self.attention.q_norm
        if rope is None:
            return norm(projected[0])
        scale = comfy.model_management.cast_to(norm.weight, device=projected.device)
        projected = F.rms_norm(projected, (self.attention.head_dim,), weight=scale, eps=norm.eps)
        rot_dim = int(rope.shape[-3]) * 2
        comfy.quant_ops.ck.apply_rope_split_half1_(projected[..., :rot_dim], rope)
        return projected[0]
    def project_q_hnd(self, x, rope_freqs, start, end):
        if self.weight is None:
            raise RuntimeError("owned ConvRot INT8 Q projection was released")
        start, end = int(start), int(end)
        rows = x[start:end]
        rope = None if rope_freqs is None else rope_freqs[:, start:end]
        with diagnostics.stage("qkv_linear"):
            comfy.ops.run_every_op()
            q = F.linear(rows, self.weight, None)
            for lora in self.sliced_loras:
                q = lora.apply(rows, q, 0, int(self.weight.shape[0]))
        with diagnostics.stage("qk_norm_rope"):
            q = self._finish_q(q, rope)
        return q.transpose(0, 1).unsqueeze(0)


class HeldConvRotINT8Linear:
    """Hold one native or runtime-quantized ConvRot-256 linear weight."""

    def __init__(self, module, sample, *, allow_float_conversion=False):
        self.module = module
        self.sample = sample
        self.allow_float_conversion = bool(allow_float_conversion)
        self.weight = None
        self.bias = None
        self.acquired_weight = None
        self.acquired_bias = None
        self.handle = None
        self.converted_from_float = False
        self.allow_sliced_lora = False
        self.sliced_loras = ()

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
            raise ConvRotINT8BindingError(
                "module explicitly requests full-precision matmul"
            )
        if self.sample.ndim < 2 or self.sample.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            raise ConvRotINT8BindingError(
                "ConvRot INT8 execution requires BF16/FP16 activations"
            )

        source = describe_linear(self.module)
        bypass_state = None
        if self.allow_sliced_lora:
            self.sliced_loras = _extract_plain_qkv_loras(self.module)
            if self.sliced_loras:
                bypass_state = _temporarily_disable_qkv_lowvram_patch(self.module)
        try:
            weight, bias, handle = comfy.ops.cast_bias_weight(
                self.module, self.sample, offloadable=True,
                compute_dtype=self.sample.dtype, want_requant=True,
            )
        finally:
            if bypass_state is not None:
                _restore_qkv_lowvram_patch(self.module, bypass_state)
        self.acquired_weight = weight
        self.acquired_bias = bias
        self.handle = handle
        try:
            if bias is not None:
                raise ConvRotINT8BindingError(
                    "ConvRot INT8 conversion requires bias-free H3 linears"
                )
            if isinstance(weight, QuantizedTensor):
                actual = describe_weight(weight, bias=bias)
                if not actual.convrot_int8_256:
                    raise ConvRotINT8BindingError(
                        "ConvRot INT8 provider received quantized layout %r"
                        % getattr(weight, "_layout_cls", None)
                    )
                self.weight = weight
            else:
                if not self.allow_float_conversion or not source.plain_float:
                    raise ConvRotINT8BindingError(
                        "ConvRot INT8 provider received a floating weight without conversion enabled"
                    )
                if getattr(weight, "dtype", None) not in (
                    torch.bfloat16,
                    torch.float16,
                ):
                    raise ConvRotINT8BindingError(
                        "ConvRot INT8 conversion requires BF16/FP16 weights, got %s"
                        % getattr(weight, "dtype", None)
                    )
                self.weight = QuantizedTensor.from_float(
                    weight,
                    LAYOUT,
                    scale="recalculate",
                    is_weight=True,
                    per_channel=True,
                    convrot=True,
                    convrot_groupsize=GROUP_SIZE,
                )
                self.converted_from_float = True
                self._release_acquired()
            return self
        except Exception:
            self.release()
            raise

    def release(self):
        self._release_acquired()
        self.weight = None
        self.bias = None
        self.sample = None
        self.sliced_loras = ()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def _linear(self, x, weight):
        override = getattr(
            self.module,
            "_h3_benchmark_convrot_linear",
            None,
        )
        if override is not None:
            if not callable(override):
                raise ConvRotINT8BindingError(
                    "benchmark ConvRot linear override is not callable"
                )
            return override(x, weight, self.bias)
        return F.linear(x, weight, self.bias)

    def linear(self, x):
        if self.weight is None:
            raise RuntimeError("ConvRot INT8 binding is not active")
        comfy.ops.run_every_op()
        out = self._linear(x, self.weight)
        for lora in self.sliced_loras:
            out = lora.apply(x, out, 0, int(self.weight.shape[0]))
        return out

    def linear_range(self, x, start, end):
        if self.weight is None:
            raise RuntimeError("ConvRot INT8 binding is not active")
        if self.bias is not None:
            raise ConvRotINT8BindingError(
                "ConvRot INT8 output slicing requires a bias-free linear"
            )
        start = int(start)
        end = int(end)
        if not 0 <= start < end <= int(self.weight.shape[0]):
            raise ConvRotINT8BindingError("ConvRot INT8 output slice is invalid")
        params = self.weight._params
        scale = params.scale
        if scale.numel() != 1:
            scale = scale[start:end]
        sliced = QuantizedTensor(
            self.weight._qdata[start:end],
            self.weight._layout_cls,
            replace(
                params,
                scale=scale,
                orig_shape=(end - start, int(self.weight.shape[1])),
            ),
        )
        comfy.ops.run_every_op()
        out = self._linear(x, sliced)
        for lora in self.sliced_loras:
            out = lora.apply(x, out, start, end)
        return out


class HeldConvRotINT8QKV:
    """Hold a ConvRot INT8 QKV weight across all projection chunks."""

    def __init__(self, attention, sample, *, allow_float_conversion=False):
        self.attention = attention
        self.binding = HeldConvRotINT8Linear(
            attention.qkv_proj, sample, allow_float_conversion=allow_float_conversion,
        )
        self.binding.allow_sliced_lora = True

    def __enter__(self):
        self.binding.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.binding.__exit__(exc_type, exc, tb)

    def clone_owned_q_projection(self):
        weight = self.binding.weight
        if weight is None:
            raise RuntimeError("ConvRot INT8 QKV binding is not active")
        inner = int(self.attention.heads) * int(self.attention.head_dim)
        params = weight._params
        scale = params.scale
        if scale.numel() != 1:
            scale = scale[:inner]
        owned_qdata = weight._qdata[:inner].clone()
        owned_scale = scale.clone()
        owned_weight = QuantizedTensor(
            owned_qdata, weight._layout_cls,
            replace(params, scale=owned_scale, orig_shape=(inner, int(weight.shape[1]))),
        )
        return OwnedConvRotINT8QProjection(self.attention, owned_weight, self.binding.sliced_loras)

    def _finish(self, rows, rope):
        from ..attention_forward import finish_qkv_projection, to_hnd

        with diagnostics.stage("qkv_linear"):
            projected = self.binding.linear(rows)
        with diagnostics.stage("qk_norm_rope"):
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

    def project_hnd(self, x, rope_freqs, start, end):
        rope = None if rope_freqs is None else rope_freqs[:, start:end]
        return self._finish(x[start:end], rope)

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
            projected = self.binding.linear_range(
                x[start:end],
                inner,
                inner * 3,
            )
        k, v = projected.split(inner, dim=-1)
        with diagnostics.stage("qk_norm_rope"):
            k = self._finish_single_qk(k, rope, self.attention.k_norm)
        v = v.view(
            end - start,
            self.attention.heads,
            self.attention.head_dim,
        )
        return (
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
        )

    def project_v_hnd(self, x, rope_freqs, start, end):
        del rope_freqs
        inner = int(self.attention.heads) * int(self.attention.head_dim)
        with diagnostics.stage("qkv_linear"):
            v = self.binding.linear_range(x[start:end], inner * 2, inner * 3)
        v = v.view(end - start, self.attention.heads, self.attention.head_dim)
        return v.transpose(0, 1).unsqueeze(0)

    def project_rows(self, x, rope_freqs, rows):
        sample_x = x.index_select(0, rows)
        sample_rope = (
            None if rope_freqs is None else rope_freqs.index_select(1, rows)
        )
        return self._finish(sample_x, sample_rope)


class HeldConvRotINT8MLP:
    """Hold runtime ConvRot INT8 fc1/fc2 weights across bounded token slabs."""

    def __init__(self, mlp, sample, *, allow_float_conversion=False):
        self.mlp = mlp
        self.sample = sample
        self.allow_float_conversion = bool(allow_float_conversion)
        self.fc1_binding = None
        self.fc2_binding = None

    def __enter__(self):
        try:
            self.fc1_binding = HeldConvRotINT8Linear(
                self.mlp.fc1,
                self.sample,
                allow_float_conversion=self.allow_float_conversion,
            )
            self.fc1_binding.__enter__()
            self.fc2_binding = HeldConvRotINT8Linear(
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
        return out, "held_convrot_int8"


class LazyConvRotINT8Linear:
    """Delay runtime quantization until the first output-projection slab."""

    def __init__(self, module):
        self.module = module
        self.binding = None

    def linear(self, x):
        if self.binding is None:
            sample = x.reshape(-1, x.shape[-1])[:1]
            self.binding = HeldConvRotINT8Linear(
                self.module,
                sample,
                allow_float_conversion=True,
            )
            self.binding.__enter__()
        return self.binding.linear(x)

    def release(self):
        if self.binding is not None:
            self.binding.__exit__(None, None, None)
            self.binding = None
