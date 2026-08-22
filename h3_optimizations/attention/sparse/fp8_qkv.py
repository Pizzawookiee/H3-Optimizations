"""Held FP8 or native W4A8 H3 QKV projection into Sparse Sage-native carriers."""

from __future__ import annotations

import torch

import comfy.model_management

from ...qkv.formats import describe_linear
from ...qkv.fp8 import FP8BindingError, HeldFP8QKV
from ...qkv.w4a8 import HeldW4A8QKV, W4A8BindingError
from .chunked_qkv import _validate_chunk_rows, pack_sparse_qk_chunk_into
from .fused_qkv import (
    FusedQKVError,
    HEAD_DIM,
    PreparedFusedQKV,
    sparse_fused_qkv_contract_mismatch,
    validate_prepared_fused_qkv,
)

CHUNK_ROWS = 4096


def run_fp8_sparse_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    spec,
    chunk_rows=CHUNK_ROWS,
):
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise FusedQKVError(
            "chunked Sparse Sage QKV requires a rank-2 CUDA BF16 input"
        )
    if comfy.model_management.in_training:
        raise FusedQKVError("chunked Sparse Sage QKV is inference-only")
    mismatch = sparse_fused_qkv_contract_mismatch(spec)
    if mismatch is not None:
        raise FusedQKVError(
            "QKV does not match the Sparse Sage carrier contract: %s"
            % mismatch
        )
    if rope_freqs is not None and (
        rope_freqs.ndim != 6
        or tuple(rope_freqs.shape[:3]) != (1, x.shape[0], 1)
        or rope_freqs.device != x.device
    ):
        raise FusedQKVError("chunked Sparse Sage QKV received invalid RoPE")

    chunk_rows = _validate_chunk_rows(
        chunk_rows,
        spec.q_tile,
        spec.kv_tile,
    )
    sequence = int(x.shape[0])
    heads = int(module.heads)
    head_dim = int(module.head_dim)
    if sequence <= 0 or head_dim != HEAD_DIM:
        raise FusedQKVError("chunked Sparse Sage QKV requires H3 head_dim 128")

    fmt = describe_linear(module.qkv_proj)
    if fmt.w4a8:
        held = HeldW4A8QKV(module, x[:1])
    else:
        held = HeldFP8QKV(
            module,
            x[:1],
            allow_float_conversion=fmt.plain_float,
        )
    held.__enter__()
    try:
        shape = (1, heads, sequence, head_dim)
        q_blocks = (sequence + int(spec.q_tile) - 1) // int(spec.q_tile)
        k_blocks = (sequence + int(spec.kv_tile) - 1) // int(spec.kv_tile)
        q_int8 = torch.empty(shape, dtype=torch.int8, device=x.device)
        k_int8 = torch.empty(shape, dtype=torch.int8, device=x.device)
        v = torch.empty(shape, dtype=x.dtype, device=x.device)
        q_scale = torch.empty(
            (1, heads, q_blocks), dtype=torch.float32, device=x.device
        )
        k_scale = torch.empty(
            (1, heads, k_blocks), dtype=torch.float32, device=x.device
        )
        q_summary = torch.empty(
            (1, heads, q_blocks, head_dim), dtype=x.dtype, device=x.device
        )
        k_summary = torch.empty(
            (1, heads, k_blocks, head_dim), dtype=x.dtype, device=x.device
        )

        for start in range(0, sequence, chunk_rows):
            end = min(start + chunk_rows, sequence)
            q, k, chunk_v = held.project_hnd(x, rope_freqs, start, end)
            pack_sparse_qk_chunk_into(
                q,
                q_int8,
                q_scale,
                q_summary,
                row_start=start,
                block_size=spec.q_tile,
            )
            pack_sparse_qk_chunk_into(
                k,
                k_int8,
                k_scale,
                k_summary,
                row_start=start,
                block_size=spec.kv_tile,
            )
            v[:, :, start:end, :].copy_(chunk_v)
            del q, k, chunk_v

        return validate_prepared_fused_qkv(
            PreparedFusedQKV(
                q_int8=q_int8,
                q_scale=q_scale,
                k_int8=k_int8,
                k_scale=k_scale,
                v=v,
                q_summary=q_summary,
                k_summary=k_summary,
                output_dtype=x.dtype,
                sequence=sequence,
                heads=heads,
                head_dim=head_dim,
                layer_index=int(layer_index),
                smooth_k=False,
            )
        )
    finally:
        held.__exit__(None, None, None)


class FP8SparseQKVProjector:
    """Project FP8, float, or native W4A8 H3 QKV into Sparse Sage carriers."""

    name = "chunked_native_sparse_sage_qkv"
    qk_format = "sparge_block_int8"

    def __init__(self, spec, required=False, chunk_rows=CHUNK_ROWS):
        self.spec = spec
        self.required = bool(required)
        self.chunk_rows = _validate_chunk_rows(
            chunk_rows,
            spec.q_tile,
            spec.kv_tile,
        )

    @property
    def installation_signature(self):
        return (
            self.name,
            self.qk_format,
            self.chunk_rows,
            bool(self.required),
            self.spec.signature,
        )

    def try_project(
        self,
        module,
        x,
        rope_freqs,
        *,
        layer_index,
        transformer_options,
    ):
        del transformer_options
        fmt = describe_linear(module.qkv_proj)
        if not (fmt.fp8 or fmt.plain_float or fmt.w4a8):
            if self.required:
                raise RuntimeError(
                    "required chunked sparse QKV optimization received format %s"
                    % fmt.label
                )
            return None
        try:
            return run_fp8_sparse_qkv(
                module,
                x,
                rope_freqs,
                layer_index=layer_index,
                spec=self.spec,
                chunk_rows=self.chunk_rows,
            )
        except (
            FP8BindingError,
            W4A8BindingError,
            FusedQKVError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            if self.required:
                raise
            return None
