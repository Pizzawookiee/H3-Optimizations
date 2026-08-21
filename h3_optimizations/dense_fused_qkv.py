"""Fused H3 QKV projection for the dense SM89 SageAttention ABI.

Sparse Sage consumes one scale per 128Q/64KV tile. Dense SageAttention's
quantization mode 3 consumes its per-thread scale layout instead. This module
keeps projection, RMSNorm, RoPE, and Q/K quantization in one Triton path while
emitting the exact dense carrier layout expected by the existing SM89 kernel.
"""

from __future__ import annotations

import torch

from .dense_fused_qkv_contract import (
    DENSE_QK_FORMAT,
    DenseFusedQKVError,
    HEAD_DIM,
    ROT_DIM,
    PreparedDenseFusedQKV,
    validate_prepared_dense_fused_qkv,
)
from .dense_fused_qkv_kernel import (
    TRITON_AVAILABLE,
    dense_fused_qkv_tensor_core,
)

def _plain_qkv_weight(module, x):
    import comfy.ops
    from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

    weight, bias, handle = comfy.ops.cast_bias_weight(
        module.qkv_proj,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=True,
    )
    try:
        if bias is not None:
            raise DenseFusedQKVError(
                "dense fused H3 QKV does not support projection bias"
            )
        if not isinstance(weight, QuantizedTensor):
            raise DenseFusedQKVError(
                "dense fused H3 QKV requires a quantized projection weight"
            )
        if getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout":
            raise DenseFusedQKVError(
                "dense fused H3 QKV requires TensorWise INT8 weights"
            )
        params = weight._params
        if getattr(params, "transposed", False):
            raise DenseFusedQKVError(
                "dense fused H3 QKV does not support transposed weights"
            )
        if not getattr(params, "convrot", False) or int(
            getattr(params, "convrot_groupsize", 0)
        ) != 256:
            raise DenseFusedQKVError(
                "dense fused H3 QKV requires ConvRot-256 weights"
            )
        qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
        return qdata, scale, handle, weight, bias
    except Exception:
        comfy.ops.uncast_bias_weight(module.qkv_proj, weight, bias, handle)
        raise


def _quantize_projection_input(x):
    try:
        from comfy_kitchen.backends.cuda import quantize_int8_rowwise_convrot64
    except ImportError as exc:  # pragma: no cover - runtime compatibility error
        raise DenseFusedQKVError(
            "dense fused H3 QKV requires Comfy Kitchen's CUDA ConvRot quantizer"
        ) from exc
    return quantize_int8_rowwise_convrot64(x, 256)

def run_dense_fused_qkv(module, x, rope_freqs, *, layer_index, tensor_core=None):
    import comfy.model_management
    import comfy.ops

    if not TRITON_AVAILABLE:
        raise DenseFusedQKVError("dense fused H3 QKV requires Triton")
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise DenseFusedQKVError(
            "dense fused H3 QKV requires a rank-2 CUDA BF16 input"
        )
    if comfy.model_management.in_training:
        raise DenseFusedQKVError("dense fused H3 QKV is inference-only")
    if int(module.head_dim) != HEAD_DIM:
        raise DenseFusedQKVError("dense fused H3 QKV requires head_dim 128")
    if rope_freqs is not None and (
        rope_freqs.ndim != 6
        or tuple(rope_freqs.shape[:3]) != (1, x.shape[0], 1)
        or int(rope_freqs.shape[-3]) * 2 != ROT_DIM
        or tuple(rope_freqs.shape[-2:]) != (2, 2)
        or rope_freqs.device != x.device
    ):
        raise DenseFusedQKVError(
            "dense fused H3 QKV requires H3's 96-wide split-half RoPE"
        )
    if float(module.q_norm.eps) != float(module.k_norm.eps):
        raise DenseFusedQKVError(
            "dense fused H3 QKV requires matching Q/K RMSNorm epsilon"
        )

    sequence, hidden = x.shape
    if sequence <= 0 or hidden % 256:
        raise DenseFusedQKVError(
            "dense fused H3 QKV requires a non-empty ConvRot-256 hidden dimension"
        )
    heads = int(module.heads)
    inner = heads * HEAD_DIM
    expected_weight = (inner * 3, hidden)
    qdata, weight_scale, handle, held_weight, bias = _plain_qkv_weight(module, x)
    try:
        if (
            tuple(qdata.shape) != expected_weight
            or qdata.dtype != torch.int8
            or qdata.device != x.device
        ):
            raise DenseFusedQKVError(
                "dense fused H3 QKV weight shape is %s; expected %s"
                % (tuple(qdata.shape), expected_weight)
            )
        qdata = qdata.contiguous()
        weight_scale = weight_scale.reshape(-1).contiguous()
        if (
            weight_scale.numel() != inner * 3
            or weight_scale.dtype != torch.float32
            or weight_scale.device != x.device
        ):
            raise DenseFusedQKVError(
                "dense fused H3 QKV weight scale shape is invalid"
            )

        q_norm = comfy.model_management.cast_to(
            module.q_norm.weight, device=x.device, dtype=x.dtype
        ).contiguous()
        k_norm = comfy.model_management.cast_to(
            module.k_norm.weight, device=x.device, dtype=x.dtype
        ).contiguous()
        if (
            q_norm.numel() != HEAD_DIM
            or k_norm.numel() != HEAD_DIM
            or q_norm.dtype != x.dtype
            or k_norm.dtype != x.dtype
        ):
            raise DenseFusedQKVError(
                "dense fused H3 QKV RMSNorm weights are invalid"
            )

        x_int8, x_scale = _quantize_projection_input(x)
        x_scale = x_scale.reshape(-1).contiguous()
        if (
            tuple(x_int8.shape) != tuple(x.shape)
            or x_int8.dtype != torch.int8
            or x_int8.device != x.device
            or not x_int8.is_contiguous()
            or x_scale.numel() != sequence
            or x_scale.dtype != torch.float32
            or x_scale.device != x.device
        ):
            raise DenseFusedQKVError(
                "Comfy Kitchen returned an invalid ConvRot activation carrier"
            )

        if rope_freqs is None:
            rope = x.new_empty((1, 1, 1, 16, 2, 2))
            rope_strides = (0, 0, 0, 0)
        else:
            rope = rope_freqs
            rope_strides = (
                rope.stride(1),
                rope.stride(3),
                rope.stride(4),
                rope.stride(5),
            )

        core = tensor_core or dense_fused_qkv_tensor_core
        carriers = core(
            x_int8,
            qdata,
            x_scale,
            weight_scale,
            q_norm,
            k_norm,
            rope,
            heads=heads,
            sequence=sequence,
            hidden=hidden,
            epsilon=float(module.q_norm.eps),
            has_rope=rope_freqs is not None,
            rope_strides=rope_strides,
            output_dtype=x.dtype,
        )
        return validate_prepared_dense_fused_qkv(
            PreparedDenseFusedQKV(
                q_int8=carriers[0],
                q_scale=carriers[1],
                k_int8=carriers[2],
                k_scale=carriers[3],
                v=carriers[4],
                output_dtype=x.dtype,
                sequence=int(sequence),
                heads=heads,
                head_dim=HEAD_DIM,
                layer_index=int(layer_index),
            )
        )
    finally:
        comfy.ops.uncast_bias_weight(
            module.qkv_proj, held_weight, bias, handle
        )


class DenseFusedQKVProjector:
    """Project H3 directly into the dense SM89 Sage per-thread carrier."""

    name = "h3_fused_qkv_dense_sage"
    qk_format = DENSE_QK_FORMAT

    def __init__(self, tensor_core=None):
        self.tensor_core = tensor_core

    @property
    def installation_signature(self):
        function = getattr(self.tensor_core, "__func__", self.tensor_core)
        core = None if function is None else (
            getattr(function, "__module__", type(function).__module__),
            getattr(function, "__qualname__", type(function).__qualname__),
            id(function),
        )
        return (self.name, self.qk_format, core)

    def bind(self, module):
        return None

    def project(self, module, x, rope_freqs, *, layer_index, transformer_options):
        return run_dense_fused_qkv(
            module,
            x,
            rope_freqs,
            layer_index=layer_index,
            tensor_core=self.tensor_core,
        )
