"""Dense SM89 Sage projected-QKV carrier contract."""

from dataclasses import dataclass

import torch

HEAD_DIM = 128
ROT_DIM = 96
Q_TILE = 128
K_TILE = 64
Q_SCALES_PER_TILE = 32
K_SCALES_PER_TILE = 4
DENSE_QK_FORMAT = "sage_per_thread_int8"


class DenseFusedQKVError(RuntimeError):
    pass


@dataclass
class PreparedDenseFusedQKV:
    q_int8: torch.Tensor
    q_scale: torch.Tensor
    k_int8: torch.Tensor
    k_scale: torch.Tensor
    v: torch.Tensor
    output_dtype: torch.dtype
    sequence: int
    heads: int
    head_dim: int
    layer_index: int
    qk_format: str = DENSE_QK_FORMAT
    smooth_k: bool = False


def validate_prepared_dense_fused_qkv(prepared):
    """Validate the complete dense Sage projected-QKV carrier contract."""

    sequence = int(prepared.sequence)
    heads = int(prepared.heads)
    head_dim = int(prepared.head_dim)
    shape = (1, heads, sequence, head_dim)
    q_scales = ((sequence + Q_TILE - 1) // Q_TILE) * Q_SCALES_PER_TILE
    k_scales = ((sequence + K_TILE - 1) // K_TILE) * K_SCALES_PER_TILE

    if getattr(prepared, "qk_format", None) != DENSE_QK_FORMAT:
        raise DenseFusedQKVError("dense fused QKV carrier has the wrong Q/K format")
    if head_dim != HEAD_DIM:
        raise DenseFusedQKVError("dense fused H3 QKV requires head_dim 128")
    if tuple(prepared.q_int8.shape) != shape or tuple(prepared.k_int8.shape) != shape:
        raise DenseFusedQKVError("dense fused H3 Q/K carrier shape is invalid")
    if prepared.q_int8.dtype != torch.int8 or prepared.k_int8.dtype != torch.int8:
        raise DenseFusedQKVError("dense fused H3 Q/K carriers must be INT8")
    if tuple(prepared.q_scale.shape) != (1, heads, q_scales):
        raise DenseFusedQKVError("dense fused H3 Q scale shape is invalid")
    if tuple(prepared.k_scale.shape) != (1, heads, k_scales):
        raise DenseFusedQKVError("dense fused H3 K scale shape is invalid")
    if prepared.q_scale.dtype != torch.float32 or prepared.k_scale.dtype != torch.float32:
        raise DenseFusedQKVError("dense fused H3 Q/K scales must be float32")
    if tuple(prepared.v.shape) != shape or prepared.v.dtype != prepared.output_dtype:
        raise DenseFusedQKVError("dense fused H3 V carrier is invalid")
    tensors = (
        prepared.q_int8,
        prepared.q_scale,
        prepared.k_int8,
        prepared.k_scale,
        prepared.v,
    )
    if any(t.device != prepared.q_int8.device for t in tensors):
        raise DenseFusedQKVError("dense fused H3 QKV carrier devices differ")
    if any(not t.is_contiguous() for t in tensors):
        raise DenseFusedQKVError("dense fused H3 QKV carriers must be contiguous")
    return prepared
