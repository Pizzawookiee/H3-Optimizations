"""Execution-scoped ConvRot INT8 bindings for floating H3 linears."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
import comfy.quant_ops
from comfy.quant_ops import QuantizedTensor

from .. import diagnostics
from ..native.convrot import quantize_int8_rowwise_convrot256
from ..native.fused_kv import fused_h3_kv_from_int8, fused_h3_kv_is_available
from .formats import describe_linear, describe_weight


LAYOUT = "TensorWiseINT8Layout"
GROUP_SIZE = 256


class ConvRotINT8BindingError(RuntimeError):
    pass


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
        weight, bias, handle = comfy.ops.cast_bias_weight(
            self.module,
            self.sample,
            offloadable=True,
            compute_dtype=self.sample.dtype,
            want_requant=True,
        )
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
        return self._linear(x, self.weight)

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
        return self._linear(x, sliced)

    def linear_ranges(self, x, ranges):
        """Execute disjoint output-channel ranges in one native INT8 GEMM.

        H3V-Smooth needs one K-head slice and one V-head slice for the same
        gathered activation rows.  Concatenating the already-quantized weight
        rows lets Kitchen perform ConvRot + dynamic activation quantization only
        once instead of once for K and again for V.
        """
        if self.weight is None:
            raise RuntimeError("ConvRot INT8 binding is not active")
        if self.bias is not None:
            raise ConvRotINT8BindingError(
                "ConvRot INT8 multi-range slicing requires a bias-free linear"
            )
        normalized = [(int(start), int(end)) for start, end in ranges]
        if not normalized or any(
            not 0 <= start < end <= int(self.weight.shape[0])
            for start, end in normalized
        ):
            raise ConvRotINT8BindingError("ConvRot INT8 output ranges are invalid")
        params = self.weight._params
        qdata = torch.cat(
            [self.weight._qdata[start:end] for start, end in normalized], dim=0
        ).contiguous()
        scale = params.scale
        if scale.numel() != 1:
            scale = torch.cat(
                [scale[start:end] for start, end in normalized], dim=0
            ).contiguous()
        rows = sum(end - start for start, end in normalized)
        sliced = QuantizedTensor(
            qdata,
            self.weight._layout_cls,
            replace(
                params,
                scale=scale,
                orig_shape=(rows, int(self.weight.shape[1])),
            ),
        )
        comfy.ops.run_every_op()
        return self._linear(x, sliced)


class HeldConvRotINT8QKV:
    """Hold a ConvRot INT8 QKV weight across all projection chunks."""

    def __init__(self, attention, sample, *, allow_float_conversion=False):
        self.attention = attention
        self.binding = HeldConvRotINT8Linear(
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


    def project_v_features_hnd(self, x, start, end, feature_dim):
        """Project only a few evenly-spaced V channels per head.

        H3V-Smooth clustering only needs a compact value-space signal.  Packing
        these disjoint output rows into one provider-native linear avoids the
        full heads*head_dim V projection used by the previous clustering path.
        """
        heads = int(self.attention.heads)
        dim = int(self.attention.head_dim)
        feature_dim = max(1, min(int(feature_dim), dim))
        inner = heads * dim
        channels = [(i * dim) // feature_dim for i in range(feature_dim)]
        ranges = []
        for head in range(heads):
            base = inner * 2 + head * dim
            ranges.extend((base + channel, base + channel + 1) for channel in channels)
        with diagnostics.stage('h3v_skinny_v_projection'):
            projected = self.binding.linear_ranges(x[start:end], ranges)
        return projected.view(end - start, heads, feature_dim).permute(1, 0, 2).contiguous()


    def _finish_k_head(self, projected, rope):
        seq = int(projected.shape[0])
        projected = projected.view(1, seq, 1, self.attention.head_dim)
        norm = self.attention.k_norm
        if rope is None:
            return norm(projected[0])[:, 0, :]
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
        return projected[0, :, 0, :]

    def project_grouped_kv_into_carrier(
        self, x, rope_freqs, rows, *, producer, k_start, k_summary, cta_k
    ):
        """Direct grouped K -> Kitchen carrier, materializing only BF16 V.

        This is the faithful H3V-Smooth hot path for native ConvRot INT8. Each
        head has its own row permutation, so we still execute one GEMM per
        head, but its CUTLASS epilogue performs K RMSNorm/RoPE, routing-summary
        reduction and Kitchen INT8 packing before K ever reaches global BF16.
        """
        if not fused_h3_kv_is_available(x.device):
            return None
        if x.dtype != torch.bfloat16 or rope_freqs is None:
            return None
        heads, count = int(rows.shape[0]), int(rows.shape[1])
        dim = int(self.attention.head_dim)
        if heads != int(self.attention.heads) or dim != 128:
            return None
        weight = self.binding.weight
        if not isinstance(weight, QuantizedTensor):
            return None
        params = weight._params
        if not getattr(params, 'convrot', False) or int(getattr(params, 'convrot_groupsize', 0)) != 256:
            return None
        scales = params.scale.to(torch.float32).reshape(-1)
        if int(scales.numel()) != int(weight._qdata.shape[0]):
            return None
        inner = heads * dim
        norm = comfy.model_management.cast_to(
            self.attention.k_norm.weight, dtype=torch.bfloat16, device=x.device
        ).contiguous()
        v_out = x.new_empty((1, heads, count, dim))
        for head in range(heads):
            head_rows = rows[head].to(dtype=torch.long)
            sample_x = x.index_select(0, head_rows).contiguous()
            freqs = rope_freqs[0, :, 0].index_select(0, head_rows).contiguous()
            with diagnostics.stage('h3v_kv_activation_quant'):
                activation, activation_scale = quantize_int8_rowwise_convrot256(sample_x)
            k0 = inner + head * dim
            v0 = inner * 2 + head * dim
            with diagnostics.stage('h3v_fused_kv_weight_pack'):
                kv_weight = torch.cat(
                    (weight._qdata[k0:k0 + dim], weight._qdata[v0:v0 + dim]), dim=0
                ).contiguous()
                kv_scale = torch.cat(
                    (scales[k0:k0 + dim], scales[v0:v0 + dim]), dim=0
                ).contiguous()
            with diagnostics.stage('h3v_fused_kv_projection'):
                v_head = fused_h3_kv_from_int8(
                    activation, kv_weight, activation_scale, kv_scale, norm, freqs,
                    producer.anchor.values[0, head].contiguous(),
                    producer.anchor.indices[0, head:head + 1].contiguous(),
                    producer.k[0, head], producer.k_scale[0, head],
                    k_summary[0, head], full_rows=int(producer.spec.k_input_shape[2]),
                    k_start=int(k_start), cta_k=int(cta_k),
                    full_k_length=int(producer.spec.k_input_shape[2]),
                    epsilon=float(self.attention.k_norm.eps),
                )
            v_out[0, head].copy_(v_head)
            del sample_x, activation, activation_scale, kv_weight, kv_scale, v_head
        return v_out


    def project_grouped_kv_hnd(self, x, rope_freqs, rows):
        """Grouped K/V with one gather + one native INT8 GEMM per head."""
        heads, count = int(rows.shape[0]), int(rows.shape[1])
        dim = int(self.attention.head_dim)
        inner = int(self.attention.heads) * dim
        if heads != int(self.attention.heads):
            raise ConvRotINT8BindingError('grouped row-map head count mismatch')
        k_out = x.new_empty((1, heads, count, dim))
        v_out = x.new_empty((1, heads, count, dim))
        for head in range(heads):
            head_rows = rows[head].to(dtype=torch.long)
            sample_x = x.index_select(0, head_rows)
            rope = None if rope_freqs is None else rope_freqs.index_select(1, head_rows)
            k_start = inner + head * dim
            v_start = inner * 2 + head * dim
            with diagnostics.stage('qkv_linear'):
                projected = self.binding.linear_ranges(
                    sample_x,
                    ((k_start, k_start + dim), (v_start, v_start + dim)),
                )
            k_head, v_head = projected.split(dim, dim=-1)
            with diagnostics.stage('qk_norm_rope'):
                k_head = self._finish_k_head(k_head, rope)
            k_out[0, head].copy_(k_head)
            v_out[0, head].copy_(v_head)
        return k_out, v_out

    def project_k_head_rows(self, x, rope_freqs, rows, head):
        head = int(head)
        dim = int(self.attention.head_dim)
        inner = int(self.attention.heads) * dim
        if not 0 <= head < int(self.attention.heads):
            raise ConvRotINT8BindingError('K head index is out of range')
        sample_x = x.index_select(0, rows)
        rope = None if rope_freqs is None else rope_freqs.index_select(1, rows)
        start = inner + head * dim
        with diagnostics.stage('qkv_linear'):
            projected = self.binding.linear_range(sample_x, start, start + dim)
        with diagnostics.stage('qk_norm_rope'):
            return self._finish_k_head(projected, rope)

    def project_v_head_rows(self, x, rope_freqs, rows, head):
        del rope_freqs
        head = int(head)
        dim = int(self.attention.head_dim)
        inner = int(self.attention.heads) * dim
        if not 0 <= head < int(self.attention.heads):
            raise ConvRotINT8BindingError('V head index is out of range')
        sample_x = x.index_select(0, rows)
        start = inner * 2 + head * dim
        with diagnostics.stage('qkv_linear'):
            return self.binding.linear_range(sample_x, start, start + dim)

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
