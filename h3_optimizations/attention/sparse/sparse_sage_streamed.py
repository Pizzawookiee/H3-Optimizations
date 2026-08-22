"""Streamed-query Sparse Sage for low-VRAM MiniMax H3 inference.

Keeps full K and final FP8 V carriers, but does not materialize a full Q
carrier or a full attention output. Query tiles are reprojected and
quantized in bounded chunks and passed to SpargeAttn using its native
qo_len != kv_len support.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ...qkv.chunked import project_chunk_hnd
from .backend import HybridSparseBackend
from .chunked_qkv import pack_sparse_qk_chunk_into
from .config import resolve_video_budget
from .fp8_v_stream import (
    allocate_v_carrier,
    finalize_v_scale,
    pack_v_chunk,
    update_v_amax,
)
from . import fused_qkv as _fused_qkv_mod
from .fused_qkv import (
    FusedQKVError,
    HEAD_DIM,
    run_fused_q_only_into,
    sparse_fused_qkv_contract_mismatch,
)
from .router import SparseRouterError
from .sparse_sage import SparseSageError
from ...mlp_sharing.route import router_kwargs as _route_kwargs


DEFAULT_PROJECT_CHUNK_ROWS = 1024
DEFAULT_QUERY_CHUNK_ROWS = 4096
OUT_PROJ_CHUNK_ROWS = 2048


@dataclass
class _HeldQOnlyState:
    module: object
    qdata: torch.Tensor
    weight_scale: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    handle: object
    held_weight: object
    bias: object
    heads: int
    hidden: int
    epsilon: float
    released: bool = False


def _acquire_q_only_state(module, x):
    """Hold existing ConvRot QKV state across every streamed Q chunk."""
    import comfy.model_management
    import comfy.ops

    qdata, weight_scale, handle, held_weight, bias = (
        _fused_qkv_mod._plain_qkv_weight(module, x)
    )
    try:
        heads = int(module.heads)
        hidden = int(x.shape[1])
        inner = heads * HEAD_DIM
        expected_weight = (inner * 3, hidden)
        if (
            tuple(qdata.shape) != expected_weight
            or qdata.dtype != torch.int8
            or qdata.device != x.device
        ):
            raise FusedQKVError(
                "held Q-only fused H3 weight shape is %s; expected %s"
                % (tuple(qdata.shape), expected_weight)
            )
        weight_scale = weight_scale.reshape(-1).contiguous()
        if (
            weight_scale.numel() != inner * 3
            or weight_scale.dtype != torch.float32
            or weight_scale.device != x.device
        ):
            raise FusedQKVError("held Q-only fused H3 weight scale is invalid")
        q_norm = comfy.model_management.cast_to(
            module.q_norm.weight,
            device=x.device,
            dtype=x.dtype,
        ).contiguous()
        k_norm = comfy.model_management.cast_to(
            module.k_norm.weight,
            device=x.device,
            dtype=x.dtype,
        ).contiguous()
        if (
            q_norm.numel() != HEAD_DIM
            or k_norm.numel() != HEAD_DIM
            or q_norm.dtype != x.dtype
            or k_norm.dtype != x.dtype
        ):
            raise FusedQKVError("held fused H3 RMSNorm weights are invalid")
        return _HeldQOnlyState(
            module=module,
            qdata=qdata,
            weight_scale=weight_scale,
            q_norm=q_norm,
            k_norm=k_norm,
            handle=handle,
            held_weight=held_weight,
            bias=bias,
            heads=heads,
            hidden=hidden,
            epsilon=float(module.q_norm.eps),
        )
    except Exception:
        comfy.ops.uncast_bias_weight(
            module.qkv_proj,
            held_weight,
            bias,
            handle,
        )
        raise


def _release_q_only_state(state):
    if state is None or state.released:
        return
    import comfy.ops

    comfy.ops.uncast_bias_weight(
        state.module.qkv_proj,
        state.held_weight,
        state.bias,
        state.handle,
    )
    state.released = True


def _quantize_projection_input_into(x, q_out, scale_out):
    """Reuse caller-owned buffers with Kitchen's already-installed CUDA op."""
    if (
        tuple(q_out.shape) != tuple(x.shape)
        or q_out.dtype != torch.int8
        or q_out.device != x.device
        or not q_out.is_contiguous()
        or tuple(scale_out.shape) != (x.shape[0], 1)
        or scale_out.dtype != torch.float32
        or scale_out.device != x.device
        or not scale_out.is_contiguous()
    ):
        raise FusedQKVError("reused ConvRot activation scratch is invalid")

    try:
        import comfy_kitchen.backends.cuda as cuda_backend

        extension = getattr(cuda_backend, "_C", None)
        wrap = getattr(cuda_backend, "_wrap_for_dlpack", None)
        op = getattr(extension, "quantize_int8_rowwise_convrot64", None)
        if callable(op) and callable(wrap):
            stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
            op(
                wrap(x),
                wrap(q_out),
                wrap(scale_out),
                256,
                False,
                0,
                0,
                stream_ptr,
            )
            return q_out, scale_out
    except (AttributeError, ImportError, RuntimeError, TypeError):
        pass

    q_tmp, scale_tmp = _fused_qkv_mod._quantize_projection_input(x)
    q_out.copy_(q_tmp)
    scale_out.copy_(scale_tmp.reshape_as(scale_out))
    del q_tmp, scale_tmp
    return q_out, scale_out


def _run_q_only_held(
    state,
    x,
    rope_freqs,
    q_int8,
    q_scale,
    q_summary_scratch,
    x_int8_scratch,
    x_scale_scratch,
):
    """Run the existing Q-only Triton kernel with held state and reused scratch."""
    sequence = int(x.shape[0])
    heads = int(state.heads)

    if rope_freqs is None:
        rope = x.new_empty((1, 1, 1, 16, 2, 2))
        rope_strides = (0, 0, 0, 0)
        has_rope = False
    else:
        rope = rope_freqs
        rope_strides = (
            rope.stride(1),
            rope.stride(3),
            rope.stride(4),
            rope.stride(5),
        )
        has_rope = True

    x_int8, x_scale = _quantize_projection_input_into(
        x,
        x_int8_scratch,
        x_scale_scratch,
    )
    grid = (_fused_qkv_mod.triton.cdiv(sequence, _fused_qkv_mod.Q_TILE), heads)
    _fused_qkv_mod._fused_qk_kernel[grid](
        x_int8,
        state.qdata,
        x_scale,
        state.weight_scale,
        state.q_norm,
        state.q_norm,
        rope,
        q_int8,
        q_scale,
        q_int8,
        q_scale,
        q_summary_scratch,
        q_summary_scratch,
        sequence=sequence,
        hidden=state.hidden,
        heads=heads,
        weight_stride_output=state.qdata.stride(0),
        weight_stride_inner=state.qdata.stride(1),
        rope_stride_seq=rope_strides[0],
        rope_stride_dim=rope_strides[1],
        rope_stride_rot=rope_strides[2],
        rope_stride_pair=rope_strides[3],
        epsilon=state.epsilon,
        has_rope=has_rope,
        KIND=0,
        BLOCK_M=_fused_qkv_mod.Q_TILE,
        BLOCK_N=HEAD_DIM,
        BLOCK_K=128,
        num_warps=8,
        num_stages=3,
    )


def _rope_args(x, rope_freqs):
    if rope_freqs is None:
        return x.new_empty((1, 1, 1, 16, 2, 2)), (0, 0, 0, 0), False
    return (
        rope_freqs,
        (
            rope_freqs.stride(1),
            rope_freqs.stride(3),
            rope_freqs.stride(4),
            rope_freqs.stride(5),
        ),
        True,
    )


def _run_prep_qkv_held(
    state,
    x,
    rope_freqs,
    *,
    q_int8,
    q_scale,
    q_summary,
    k_int8,
    k_scale,
    k_summary,
    v,
    x_int8_scratch,
    x_scale_scratch,
):
    """Project Q/K/V for prep with one activation quantization and held weights."""
    sequence = int(x.shape[0])
    heads = int(state.heads)
    rope, rope_strides, has_rope = _rope_args(x, rope_freqs)
    x_int8, x_scale = _quantize_projection_input_into(
        x,
        x_int8_scratch,
        x_scale_scratch,
    )

    qk_grid = (
        _fused_qkv_mod.triton.cdiv(sequence, _fused_qkv_mod.Q_TILE),
        heads,
    )
    for kind in (0, 1):
        _fused_qkv_mod._fused_qk_kernel[qk_grid](
            x_int8,
            state.qdata,
            x_scale,
            state.weight_scale,
            state.q_norm,
            state.k_norm,
            rope,
            q_int8,
            q_scale,
            k_int8,
            k_scale,
            q_summary,
            k_summary,
            sequence=sequence,
            hidden=state.hidden,
            heads=heads,
            weight_stride_output=state.qdata.stride(0),
            weight_stride_inner=state.qdata.stride(1),
            rope_stride_seq=rope_strides[0],
            rope_stride_dim=rope_strides[1],
            rope_stride_rot=rope_strides[2],
            rope_stride_pair=rope_strides[3],
            epsilon=state.epsilon,
            has_rope=has_rope,
            KIND=kind,
            BLOCK_M=_fused_qkv_mod.Q_TILE,
            BLOCK_N=HEAD_DIM,
            BLOCK_K=128,
            num_warps=8,
            num_stages=3,
        )

    v_grid = (
        _fused_qkv_mod.triton.cdiv(sequence, 128),
        _fused_qkv_mod.triton.cdiv(heads * HEAD_DIM, 256),
    )
    _fused_qkv_mod._fused_v_kernel[v_grid](
        x_int8,
        state.qdata,
        x_scale,
        state.weight_scale,
        v,
        sequence=sequence,
        hidden=state.hidden,
        output_features=heads * HEAD_DIM,
        head_dim=HEAD_DIM,
        weight_stride_output=state.qdata.stride(0),
        weight_stride_inner=state.qdata.stride(1),
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=128,
        num_warps=8,
        num_stages=3,
    )


def _run_v_only_held(
    state,
    x,
    *,
    v,
    x_int8_scratch,
    x_scale_scratch,
):
    """Run only the existing fused V projection for streamed prep pass two."""
    sequence = int(x.shape[0])
    heads = int(state.heads)
    x_int8, x_scale = _quantize_projection_input_into(
        x,
        x_int8_scratch,
        x_scale_scratch,
    )
    v_grid = (
        _fused_qkv_mod.triton.cdiv(sequence, 128),
        _fused_qkv_mod.triton.cdiv(heads * HEAD_DIM, 256),
    )
    _fused_qkv_mod._fused_v_kernel[v_grid](
        x_int8,
        state.qdata,
        x_scale,
        state.weight_scale,
        v,
        sequence=sequence,
        hidden=state.hidden,
        output_features=heads * HEAD_DIM,
        head_dim=HEAD_DIM,
        weight_stride_output=state.qdata.stride(0),
        weight_stride_inner=state.qdata.stride(1),
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=128,
        num_warps=8,
        num_stages=3,
    )


@dataclass
class StreamedSparseSageQKV:
    module: object
    x: torch.Tensor
    rope_freqs: torch.Tensor | None
    k_int8: torch.Tensor
    k_scale: torch.Tensor
    v_carrier: torch.Tensor
    v_scale: torch.Tensor
    q_summary: torch.Tensor
    k_summary: torch.Tensor
    output_dtype: torch.dtype
    sequence: int
    heads: int
    head_dim: int
    layer_index: int
    project_chunk_rows: int
    query_chunk_rows: int
    q_only_convrot: bool


@dataclass
class PreparedStreamedSparseSage:
    projected: StreamedSparseSageQKV
    route_plan: list
    metadata: dict


@dataclass
class PreparedStreamedHybrid:
    sparse: PreparedStreamedSparseSage


def _validate_chunk_rows(value, q_tile, kv_tile, *, name):
    value = int(value)
    alignment = math.lcm(int(q_tile), int(kv_tile))
    if value <= 0 or value % alignment:
        raise SparseSageError(
            "%s must be a positive multiple of %d" % (name, alignment)
        )
    return value


def _validate_projected(projected, spec):
    if not isinstance(projected, StreamedSparseSageQKV):
        raise SparseSageError("expected StreamedSparseSageQKV")
    mismatch = sparse_fused_qkv_contract_mismatch(spec)
    if mismatch is not None:
        raise SparseSageError(
            "streamed Sparse Sage carrier contract mismatch: %s" % mismatch
        )
    if not spec.uses_fp8_v:
        raise SparseSageError("streamed Sparse Sage currently requires FP8 V")
    sequence = int(projected.sequence)
    heads = int(projected.heads)
    head_dim = int(projected.head_dim)
    if sequence <= 0 or heads <= 0 or head_dim != HEAD_DIM:
        raise SparseSageError("streamed Sparse Sage metadata is invalid")
    if (
        not projected.x.is_cuda
        or projected.x.dtype != torch.bfloat16
        or projected.x.ndim != 2
        or int(projected.x.shape[0]) != sequence
    ):
        raise SparseSageError(
            "streamed Sparse Sage requires rank-2 CUDA BF16 attention input"
        )

    q_blocks = (sequence + int(spec.q_tile) - 1) // int(spec.q_tile)
    k_blocks = (sequence + int(spec.kv_tile) - 1) // int(spec.kv_tile)
    padded = (sequence + 127) // 128 * 128
    device = projected.x.device

    contracts = (
        (
            "k_int8",
            projected.k_int8,
            (1, heads, sequence, head_dim),
            torch.int8,
        ),
        (
            "k_scale",
            projected.k_scale,
            (1, heads, k_blocks),
            torch.float32,
        ),
        (
            "v_carrier",
            projected.v_carrier,
            (1, heads, head_dim, padded),
            torch.float8_e4m3fn,
        ),
        (
            "v_scale",
            projected.v_scale,
            (1, heads, head_dim),
            torch.float32,
        ),
        (
            "q_summary",
            projected.q_summary,
            (1, heads, q_blocks, head_dim),
            projected.output_dtype,
        ),
        (
            "k_summary",
            projected.k_summary,
            (1, heads, k_blocks, head_dim),
            projected.output_dtype,
        ),
    )
    for name, tensor, shape, dtype in contracts:
        if tuple(tensor.shape) != tuple(shape):
            raise SparseSageError(
                "%s shape %s does not match %s"
                % (name, tuple(tensor.shape), tuple(shape))
            )
        if tensor.dtype != dtype:
            raise SparseSageError(
                "%s dtype %s does not match %s" % (name, tensor.dtype, dtype)
            )
        if tensor.device != device or not tensor.is_contiguous():
            raise SparseSageError(
                "%s must be contiguous on the attention device" % name
            )

    _validate_chunk_rows(
        projected.project_chunk_rows,
        spec.q_tile,
        spec.kv_tile,
        name="project_chunk_rows",
    )
    _validate_chunk_rows(
        projected.query_chunk_rows,
        spec.q_tile,
        spec.kv_tile,
        name="query_chunk_rows",
    )
    return projected


def _collect_q_summary_chunk(q, destination, *, row_start, q_tile):
    """Pack only a temporary local Q carrier to obtain exact tile summaries."""
    rows = int(q.shape[-2])
    heads = int(q.shape[1])
    blocks = (rows + int(q_tile) - 1) // int(q_tile)

    q_tmp = torch.empty(
        (1, heads, rows, HEAD_DIM),
        dtype=torch.int8,
        device=q.device,
    )
    scale_tmp = torch.empty(
        (1, heads, blocks),
        dtype=torch.float32,
        device=q.device,
    )
    summary_tmp = torch.empty(
        (1, heads, blocks, HEAD_DIM),
        dtype=q.dtype,
        device=q.device,
    )
    pack_sparse_qk_chunk_into(
        q,
        q_tmp,
        scale_tmp,
        summary_tmp,
        row_start=0,
        block_size=q_tile,
    )
    block_start = int(row_start) // int(q_tile)
    destination[..., block_start:block_start + blocks, :].copy_(summary_tmp)
    del q_tmp, scale_tmp, summary_tmp


def run_streamed_sparse_sage_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    spec,
    project_chunk_rows=DEFAULT_PROJECT_CHUNK_ROWS,
    query_chunk_rows=DEFAULT_QUERY_CHUNK_ROWS,
    q_only_convrot=False,
):
    import comfy.model_management

    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise SparseSageError(
            "streamed Sparse Sage QKV requires rank-2 CUDA BF16 input"
        )
    if comfy.model_management.in_training:
        raise SparseSageError("streamed Sparse Sage QKV is inference-only")
    if int(module.head_dim) != HEAD_DIM:
        raise SparseSageError("streamed Sparse Sage QKV requires head_dim 128")
    if not spec.uses_fp8_v or spec.fused_v_ops is None:
        raise SparseSageError(
            "streamed Sparse Sage requires the FP8-V fused Sparse Sage ABI"
        )

    mismatch = sparse_fused_qkv_contract_mismatch(spec)
    if mismatch is not None:
        raise SparseSageError(
            "streamed Sparse Sage QKV contract mismatch: %s" % mismatch
        )

    if rope_freqs is not None and (
        rope_freqs.ndim != 6
        or tuple(rope_freqs.shape[:3]) != (1, x.shape[0], 1)
        or rope_freqs.device != x.device
    ):
        raise SparseSageError("streamed Sparse Sage QKV received invalid RoPE")

    project_chunk_rows = _validate_chunk_rows(
        project_chunk_rows,
        spec.q_tile,
        spec.kv_tile,
        name="project_chunk_rows",
    )
    query_chunk_rows = _validate_chunk_rows(
        query_chunk_rows,
        spec.q_tile,
        spec.kv_tile,
        name="query_chunk_rows",
    )

    sequence = int(x.shape[0])
    heads = int(module.heads)
    q_blocks = (sequence + int(spec.q_tile) - 1) // int(spec.q_tile)
    k_blocks = (sequence + int(spec.kv_tile) - 1) // int(spec.kv_tile)

    # Full K remains necessary because every Q chunk may select arbitrary K
    # tiles. Full Q is intentionally absent.
    k_int8 = torch.empty(
        (1, heads, sequence, HEAD_DIM),
        dtype=torch.int8,
        device=x.device,
    )
    k_scale = torch.empty(
        (1, heads, k_blocks),
        dtype=torch.float32,
        device=x.device,
    )
    q_summary = torch.empty(
        (1, heads, q_blocks, HEAD_DIM),
        dtype=x.dtype,
        device=x.device,
    )
    k_summary = torch.empty(
        (1, heads, k_blocks, HEAD_DIM),
        dtype=x.dtype,
        device=x.device,
    )
    v_amax = torch.zeros(
        (1, heads, HEAD_DIM),
        dtype=torch.float32,
        device=x.device,
    )

    if q_only_convrot:
        # ConvRot fast path: hold the quantized QKV projection once, quantize
        # each activation slab once in pass one, and never materialize BF16 Q/K.
        max_rows = min(project_chunk_rows, sequence)
        max_q_blocks = (max_rows + int(spec.q_tile) - 1) // int(spec.q_tile)
        max_k_blocks = (max_rows + int(spec.kv_tile) - 1) // int(spec.kv_tile)
        hidden = int(x.shape[1])

        x_int8_scratch = torch.empty(
            (max_rows, hidden),
            dtype=torch.int8,
            device=x.device,
        )
        x_scale_scratch = torch.empty(
            (max_rows, 1),
            dtype=torch.float32,
            device=x.device,
        )
        q_int8_scratch = torch.empty(
            (1, heads, max_rows, HEAD_DIM),
            dtype=torch.int8,
            device=x.device,
        )
        q_scale_scratch = torch.empty(
            (1, heads, max_q_blocks),
            dtype=torch.float32,
            device=x.device,
        )
        q_summary_scratch = torch.empty(
            (1, heads, max_q_blocks, HEAD_DIM),
            dtype=x.dtype,
            device=x.device,
        )
        k_int8_scratch = torch.empty(
            (1, heads, max_rows, HEAD_DIM),
            dtype=torch.int8,
            device=x.device,
        )
        k_scale_scratch = torch.empty(
            (1, heads, max_k_blocks),
            dtype=torch.float32,
            device=x.device,
        )
        k_summary_scratch = torch.empty(
            (1, heads, max_k_blocks, HEAD_DIM),
            dtype=x.dtype,
            device=x.device,
        )
        v_scratch = torch.empty(
            (1, heads, max_rows, HEAD_DIM),
            dtype=x.dtype,
            device=x.device,
        )

        held_prep = _acquire_q_only_state(module, x)
        try:
            for start in range(0, sequence, project_chunk_rows):
                end = min(start + project_chunk_rows, sequence)
                rows = end - start
                q_local_blocks = (
                    rows + int(spec.q_tile) - 1
                ) // int(spec.q_tile)
                k_local_blocks = (
                    rows + int(spec.kv_tile) - 1
                ) // int(spec.kv_tile)
                full_chunk = rows == max_rows

                if full_chunk:
                    q_local = q_int8_scratch
                    q_scale_local = q_scale_scratch
                    q_summary_local = q_summary_scratch
                    k_local = k_int8_scratch
                    k_scale_local = k_scale_scratch
                    k_summary_local = k_summary_scratch
                    v_local = v_scratch
                else:
                    q_local = torch.empty(
                        (1, heads, rows, HEAD_DIM),
                        dtype=torch.int8,
                        device=x.device,
                    )
                    q_scale_local = torch.empty(
                        (1, heads, q_local_blocks),
                        dtype=torch.float32,
                        device=x.device,
                    )
                    q_summary_local = torch.empty(
                        (1, heads, q_local_blocks, HEAD_DIM),
                        dtype=x.dtype,
                        device=x.device,
                    )
                    k_local = torch.empty(
                        (1, heads, rows, HEAD_DIM),
                        dtype=torch.int8,
                        device=x.device,
                    )
                    k_scale_local = torch.empty(
                        (1, heads, k_local_blocks),
                        dtype=torch.float32,
                        device=x.device,
                    )
                    k_summary_local = torch.empty(
                        (1, heads, k_local_blocks, HEAD_DIM),
                        dtype=x.dtype,
                        device=x.device,
                    )
                    v_local = torch.empty(
                        (1, heads, rows, HEAD_DIM),
                        dtype=x.dtype,
                        device=x.device,
                    )

                chunk_rope = (
                    None if rope_freqs is None else rope_freqs[:, start:end]
                )
                _run_prep_qkv_held(
                    held_prep,
                    x[start:end],
                    chunk_rope,
                    q_int8=q_local,
                    q_scale=q_scale_local,
                    q_summary=q_summary_local,
                    k_int8=k_local,
                    k_scale=k_scale_local,
                    k_summary=k_summary_local,
                    v=v_local,
                    x_int8_scratch=x_int8_scratch[:rows],
                    x_scale_scratch=x_scale_scratch[:rows],
                )

                q_block_start = start // int(spec.q_tile)
                k_block_start = start // int(spec.kv_tile)
                q_summary[
                    ...,
                    q_block_start:q_block_start + q_local_blocks,
                    :,
                ].copy_(q_summary_local)
                k_int8[..., start:end, :].copy_(k_local)
                k_scale[
                    ...,
                    k_block_start:k_block_start + k_local_blocks,
                ].copy_(k_scale_local)
                k_summary[
                    ...,
                    k_block_start:k_block_start + k_local_blocks,
                    :,
                ].copy_(k_summary_local)
                update_v_amax(v_amax, v_local)

                if not full_chunk:
                    del (
                        q_local,
                        q_scale_local,
                        q_summary_local,
                        k_local,
                        k_scale_local,
                        k_summary_local,
                        v_local,
                    )

            v_scale = finalize_v_scale(v_amax, spec.v_quant_bound)
            del v_amax
            v_carrier = allocate_v_carrier(
                sequence=sequence,
                heads=heads,
                head_dim=HEAD_DIM,
                device=x.device,
            )

            # Pass two is genuinely V-only: no Q/K GEMMs or BF16 Q/K outputs.
            for start in range(0, sequence, project_chunk_rows):
                end = min(start + project_chunk_rows, sequence)
                rows = end - start
                if rows == max_rows:
                    v_local = v_scratch
                else:
                    v_local = torch.empty(
                        (1, heads, rows, HEAD_DIM),
                        dtype=x.dtype,
                        device=x.device,
                    )
                _run_v_only_held(
                    held_prep,
                    x[start:end],
                    v=v_local,
                    x_int8_scratch=x_int8_scratch[:rows],
                    x_scale_scratch=x_scale_scratch[:rows],
                )
                pack_v_chunk(
                    v_local,
                    v_carrier,
                    v_scale,
                    row_start=start,
                    sequence=sequence,
                    fused_v_ops=spec.fused_v_ops,
                    scale_max=spec.v_quant_bound,
                )
                if rows != max_rows:
                    del v_local
        finally:
            _release_q_only_state(held_prep)
            del (
                x_int8_scratch,
                x_scale_scratch,
                q_int8_scratch,
                q_scale_scratch,
                q_summary_scratch,
                k_int8_scratch,
                k_scale_scratch,
                k_summary_scratch,
                v_scratch,
            )
    else:
        # Compatibility path for non-ConvRot projected providers.
        for start in range(0, sequence, project_chunk_rows):
            end = min(start + project_chunk_rows, sequence)
            q, k, v = project_chunk_hnd(
                module,
                x,
                rope_freqs,
                start,
                end,
            )
            try:
                _collect_q_summary_chunk(
                    q,
                    q_summary,
                    row_start=start,
                    q_tile=spec.q_tile,
                )
                pack_sparse_qk_chunk_into(
                    k,
                    k_int8,
                    k_scale,
                    k_summary,
                    row_start=start,
                    block_size=spec.kv_tile,
                )
                update_v_amax(v_amax, v)
            finally:
                del q, k, v

        v_scale = finalize_v_scale(v_amax, spec.v_quant_bound)
        del v_amax
        v_carrier = allocate_v_carrier(
            sequence=sequence,
            heads=heads,
            head_dim=HEAD_DIM,
            device=x.device,
        )
        for start in range(0, sequence, project_chunk_rows):
            end = min(start + project_chunk_rows, sequence)
            q_unused, k_unused, v = project_chunk_hnd(
                module,
                x,
                rope_freqs,
                start,
                end,
            )
            try:
                pack_v_chunk(
                    v,
                    v_carrier,
                    v_scale,
                    row_start=start,
                    sequence=sequence,
                    fused_v_ops=spec.fused_v_ops,
                    scale_max=spec.v_quant_bound,
                )
            finally:
                del q_unused, k_unused, v

    return _validate_projected(
        StreamedSparseSageQKV(
            module=module,
            x=x,
            rope_freqs=rope_freqs,
            k_int8=k_int8,
            k_scale=k_scale,
            v_carrier=v_carrier,
            v_scale=v_scale,
            q_summary=q_summary,
            k_summary=k_summary,
            output_dtype=x.dtype,
            sequence=sequence,
            heads=heads,
            head_dim=HEAD_DIM,
            layer_index=int(layer_index),
            project_chunk_rows=project_chunk_rows,
            query_chunk_rows=query_chunk_rows,
            q_only_convrot=bool(q_only_convrot),
        ),
        spec,
    )


class StreamedSparseSageQKVProjector:
    name = "streamed_sparse_sage_qkv"
    qk_format = "streamed_q_sparge_block_int8"

    def __init__(
        self,
        spec,
        *,
        project_chunk_rows=DEFAULT_PROJECT_CHUNK_ROWS,
        query_chunk_rows=DEFAULT_QUERY_CHUNK_ROWS,
        q_only_convrot=False,
    ):
        self.spec = spec
        self.chunk_rows = _validate_chunk_rows(
            project_chunk_rows,
            spec.q_tile,
            spec.kv_tile,
            name="project_chunk_rows",
        )
        self.query_chunk_rows = _validate_chunk_rows(
            query_chunk_rows,
            spec.q_tile,
            spec.kv_tile,
            name="query_chunk_rows",
        )
        self.q_only_convrot = bool(q_only_convrot)

    @property
    def installation_signature(self):
        return (
            self.name,
            self.qk_format,
            self.chunk_rows,
            self.query_chunk_rows,
            self.q_only_convrot,
            self.spec.signature,
        )

    def project(
        self,
        module,
        x,
        rope_freqs,
        *,
        layer_index,
        transformer_options,
    ):
        del transformer_options
        return run_streamed_sparse_sage_qkv(
            module,
            x,
            rope_freqs,
            layer_index=layer_index,
            spec=self.spec,
            project_chunk_rows=self.chunk_rows,
            query_chunk_rows=self.query_chunk_rows,
            q_only_convrot=self.q_only_convrot,
        )


def _make_local_q(
    projected,
    spec,
    row_start,
    row_end,
    *,
    q_int8,
    q_scale,
    q_summary_scratch,
):
    rows = row_end - row_start
    if projected.q_only_convrot:
        chunk_rope = (
            None
            if projected.rope_freqs is None
            else projected.rope_freqs[:, row_start:row_end]
        )
        run_fused_q_only_into(
            projected.module,
            projected.x[row_start:row_end],
            chunk_rope,
            q_int8,
            q_scale,
            q_summary_scratch,
        )
        return

    # Compatibility fallback for non-ConvRot projected providers.
    q, k_unused, v_unused = project_chunk_hnd(
        projected.module,
        projected.x,
        projected.rope_freqs,
        row_start,
        row_end,
    )
    try:
        pack_sparse_qk_chunk_into(
            q,
            q_int8,
            q_scale,
            q_summary_scratch,
            row_start=0,
            block_size=spec.q_tile,
        )
    finally:
        del q, k_unused, v_unused

def _execute_streamed_legacy(module, backend, prepared):
    projected = _validate_projected(
        prepared.projected,
        backend.executor.spec,
    )
    spec = backend.executor.spec

    if module is not projected.module:
        raise SparseSageError(
            "streamed Sparse Sage module changed between prepare and execute"
        )

    sequence = projected.sequence
    heads = projected.heads
    q_tile = int(spec.q_tile)
    kv_tiles = (sequence + int(spec.kv_tile) - 1) // int(spec.kv_tile)
    max_rows = min(int(projected.query_chunk_rows), sequence)
    max_q_tiles = (max_rows + q_tile - 1) // q_tile
    result = projected.x

    pv_threshold = torch.full(
        (heads,),
        50.0,
        dtype=torch.float32,
        device=projected.x.device,
    )

    # Reused by every full-size query chunk. This removes the allocator churn
    # from Q, Q-scale, routing slices, valid counts, and attention output.
    q_int8_buffer = torch.empty(
        (1, heads, max_rows, HEAD_DIM),
        dtype=torch.int8,
        device=projected.x.device,
    )
    q_scale_buffer = torch.empty(
        (1, heads, max_q_tiles),
        dtype=torch.float32,
        device=projected.x.device,
    )
    q_summary_buffer = torch.empty(
        (1, heads, max_q_tiles, HEAD_DIM),
        dtype=projected.output_dtype,
        device=projected.x.device,
    )
    output_buffer = torch.empty(
        (1, heads, max_rows, HEAD_DIM),
        dtype=projected.output_dtype,
        device=projected.x.device,
    )
    lut_buffer = torch.empty(
        (1, heads, max_q_tiles, kv_tiles),
        dtype=torch.int32,
        device=projected.x.device,
    )
    valid_buffer = torch.empty(
        (1, heads, max_q_tiles),
        dtype=torch.int32,
        device=projected.x.device,
    )

    for row_start in range(0, sequence, projected.query_chunk_rows):
        row_end = min(
            row_start + projected.query_chunk_rows,
            sequence,
        )
        if row_start % q_tile:
            raise SparseSageError(
                "streamed Sparse Sage Q chunk is not Q-tile aligned"
            )

        rows = row_end - row_start
        tile_start = row_start // q_tile
        local_q_tiles = (rows + q_tile - 1) // q_tile
        tile_end = tile_start + local_q_tiles
        full_chunk = rows == max_rows and local_q_tiles == max_q_tiles

        if full_chunk:
            q_int8 = q_int8_buffer
            q_scale = q_scale_buffer
            q_summary_scratch = q_summary_buffer
            output = output_buffer
            lut_chunk = lut_buffer
            valid_chunk = valid_buffer
        else:
            # One exact-size tail allocation is preferable to presenting a
            # strided Q carrier to the Q-only Triton projection kernel.
            q_int8 = torch.empty(
                (1, heads, rows, HEAD_DIM),
                dtype=torch.int8,
                device=projected.x.device,
            )
            q_scale = torch.empty(
                (1, heads, local_q_tiles),
                dtype=torch.float32,
                device=projected.x.device,
            )
            q_summary_scratch = torch.empty(
                (1, heads, local_q_tiles, HEAD_DIM),
                dtype=projected.output_dtype,
                device=projected.x.device,
            )
            output = output_buffer[..., :rows, :]
            lut_chunk = torch.empty(
                (1, heads, local_q_tiles, kv_tiles),
                dtype=torch.int32,
                device=projected.x.device,
            )
            valid_chunk = torch.empty(
                (1, heads, local_q_tiles),
                dtype=torch.int32,
                device=projected.x.device,
            )

        _make_local_q(
            projected,
            spec,
            row_start,
            row_end,
            q_int8=q_int8,
            q_scale=q_scale,
            q_summary_scratch=q_summary_scratch,
        )

        # SpargeAttn requires contiguous LUT/valid carriers. Copy into reusable
        # fixed-size buffers instead of allocating .contiguous() every chunk.
        lut_chunk.copy_(
            prepared.lut[
                ...,
                tile_start:tile_end,
                :,
            ]
        )
        valid_chunk.copy_(
            prepared.valid_block_num[
                ...,
                tile_start:tile_end,
            ]
        )

        expected_lut = (
            1,
            heads,
            local_q_tiles,
            kv_tiles,
        )
        if tuple(lut_chunk.shape) != expected_lut:
            raise SparseSageError(
                "streamed Sparse Sage LUT slice %s does not match %s"
                % (tuple(lut_chunk.shape), expected_lut)
            )

        # Normal compiled SpargeAttn kernel: qo_len is the local Q chunk,
        # while kv_len remains the full sequence.
        spec.dispatch(
            q_int8,
            projected.k_int8,
            projected.v_carrier,
            output,
            lut_chunk,
            valid_chunk,
            pv_threshold,
            q_scale,
            projected.k_scale,
            projected.v_scale,
            projected.output_dtype,
        )

        attention_rows = output.transpose(1, 2).reshape(
            rows,
            heads * HEAD_DIM,
        )
        projected_rows = module.out_proj(attention_rows)
        expected_output = (rows, projected.x.shape[1])
        if tuple(projected_rows.shape) != expected_output:
            raise SparseSageError(
                "streamed Sparse Sage out_proj shape %s does not match %s"
                % (tuple(projected_rows.shape), expected_output)
            )

        result[row_start:row_end].copy_(projected_rows)

        del attention_rows, projected_rows
        if not full_chunk:
            del q_int8, q_scale, q_summary_scratch, lut_chunk, valid_chunk

    return result


def _execute_streamed(module, backend, prepared):
    """Low-VRAM streamed Sparse Sage with held Q state and lazy route chunks."""
    projected = _validate_projected(
        prepared.projected,
        backend.executor.spec,
    )
    spec = backend.executor.spec
    if module is not projected.module:
        raise SparseSageError(
            "streamed Sparse Sage module changed between prepare and execute"
        )

    sequence = int(projected.sequence)
    heads = int(projected.heads)
    hidden = int(projected.x.shape[1])
    q_tile = int(spec.q_tile)
    max_rows = min(int(projected.query_chunk_rows), sequence)
    max_q_tiles = (max_rows + q_tile - 1) // q_tile
    result = projected.x

    pv_threshold = torch.full(
        (heads,),
        50.0,
        dtype=torch.float32,
        device=projected.x.device,
    )
    q_int8_buffer = torch.empty(
        (1, heads, max_rows, HEAD_DIM),
        dtype=torch.int8,
        device=projected.x.device,
    )
    q_scale_buffer = torch.empty(
        (1, heads, max_q_tiles),
        dtype=torch.float32,
        device=projected.x.device,
    )
    q_summary_buffer = torch.empty(
        (1, heads, max_q_tiles, HEAD_DIM),
        dtype=projected.output_dtype,
        device=projected.x.device,
    )
    output_buffer = torch.empty(
        (1, heads, max_rows, HEAD_DIM),
        dtype=projected.output_dtype,
        device=projected.x.device,
    )

    held_q_state = None
    x_int8_buffer = None
    x_scale_buffer = None
    if projected.q_only_convrot:
        held_q_state = _acquire_q_only_state(
            projected.module,
            projected.x,
        )
        x_int8_buffer = torch.empty(
            (max_rows, hidden),
            dtype=torch.int8,
            device=projected.x.device,
        )
        x_scale_buffer = torch.empty(
            (max_rows, 1),
            dtype=torch.float32,
            device=projected.x.device,
        )

    route_iter = backend.router.iter_lut_chunks(
        prepared.route_plan,
        q_chunk_tiles=max_q_tiles,
    )
    try:
        for tile_start, tile_end, lut_chunk, valid_chunk in route_iter:
            row_start = int(tile_start) * q_tile
            row_end = min(int(tile_end) * q_tile, sequence)
            rows = row_end - row_start
            local_q_tiles = int(tile_end) - int(tile_start)
            if rows <= 0:
                raise SparseSageError("streamed Sparse Sage produced empty Q chunk")

            full_chunk = rows == max_rows and local_q_tiles == max_q_tiles
            if full_chunk:
                q_int8 = q_int8_buffer
                q_scale = q_scale_buffer
                q_summary_scratch = q_summary_buffer
                output = output_buffer
            else:
                q_int8 = torch.empty(
                    (1, heads, rows, HEAD_DIM),
                    dtype=torch.int8,
                    device=projected.x.device,
                )
                q_scale = torch.empty(
                    (1, heads, local_q_tiles),
                    dtype=torch.float32,
                    device=projected.x.device,
                )
                q_summary_scratch = torch.empty(
                    (1, heads, local_q_tiles, HEAD_DIM),
                    dtype=projected.output_dtype,
                    device=projected.x.device,
                )
                output = torch.empty(
                    (1, heads, rows, HEAD_DIM),
                    dtype=projected.output_dtype,
                    device=projected.x.device,
                )

            if held_q_state is not None:
                chunk_rope = (
                    None
                    if projected.rope_freqs is None
                    else projected.rope_freqs[:, row_start:row_end]
                )
                _run_q_only_held(
                    held_q_state,
                    projected.x[row_start:row_end],
                    chunk_rope,
                    q_int8,
                    q_scale,
                    q_summary_scratch,
                    x_int8_buffer[:rows],
                    x_scale_buffer[:rows],
                )
            else:
                _make_local_q(
                    projected,
                    spec,
                    row_start,
                    row_end,
                    q_int8=q_int8,
                    q_scale=q_scale,
                    q_summary_scratch=q_summary_scratch,
                )

            spec.dispatch(
                q_int8,
                projected.k_int8,
                projected.v_carrier,
                output,
                lut_chunk,
                valid_chunk,
                pv_threshold,
                q_scale,
                projected.k_scale,
                projected.v_scale,
                projected.output_dtype,
            )
            del lut_chunk, valid_chunk

            for local_start in range(0, rows, OUT_PROJ_CHUNK_ROWS):
                local_end = min(local_start + OUT_PROJ_CHUNK_ROWS, rows)
                proj_rows = local_end - local_start
                attention_rows = (
                    output[..., local_start:local_end, :]
                    .transpose(1, 2)
                    .reshape(proj_rows, heads * HEAD_DIM)
                )
                projected_rows = module.out_proj(attention_rows)
                expected_output = (proj_rows, hidden)
                if tuple(projected_rows.shape) != expected_output:
                    raise SparseSageError(
                        "streamed Sparse Sage out_proj shape %s does not match %s"
                        % (tuple(projected_rows.shape), expected_output)
                    )
                result[
                    row_start + local_start:row_start + local_end
                ].copy_(projected_rows)
                del attention_rows, projected_rows

            if not full_chunk:
                del q_int8, q_scale, q_summary_scratch, output
    finally:
        close = getattr(route_iter, "close", None)
        if callable(close):
            close()
        _release_q_only_state(held_q_state)
        if prepared.route_plan is not None and len(prepared.route_plan) >= 2:
            prepared.route_plan[0] = None
            prepared.route_plan[1] = None
        prepared.route_plan = None

        # The attention result already lives in projected.x. Drop every large
        # carrier before returning so the following MLP weight acquisition does
        # not overlap K/V or streamed-attention scratch residency.
        projected.q_summary = None
        projected.k_summary = None
        projected.k_int8 = None
        projected.k_scale = None
        projected.v_carrier = None
        projected.v_scale = None
        del (
            q_int8_buffer,
            q_scale_buffer,
            q_summary_buffer,
            output_buffer,
            pv_threshold,
        )
        if x_int8_buffer is not None:
            del x_int8_buffer, x_scale_buffer

    return result


class StreamedSparseSageBackend(HybridSparseBackend):
    """Sparse Sage backend using global K/V with streamed Q/output."""

    def prepare_projected(
        self,
        projected,
        *,
        layer_index,
        transformer_options,
    ):
        projected = _validate_projected(projected, self.executor.spec)
        if int(projected.layer_index) != int(layer_index):
            raise SparseSageError(
                "streamed QKV layer %d does not match attention layer %d"
                % (projected.layer_index, layer_index)
            )

        snapshot = self._snapshot(
            transformer_options,
            projected.sequence,
        )
        video_budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
            layer_index,
        )
        try:
            route_plan, mask_metadata = (
                self.router.prepare_lut_chunks_from_summaries(
                    projected.q_summary,
                    projected.k_summary,
                    snapshot.layout,
                    video_budget,
                    **_route_kwargs(transformer_options, layer_index),
                )
            )
        except SparseRouterError as exc:
            raise SparseSageError("sparse routing failed: %s" % exc) from exc

        metadata = self._metadata(
            mask_metadata,
            layer_index,
            projected.heads,
        )
        metadata.update(
            {
                "qkv_lifetime": "streamed_q_global_k_fp8v",
                "attention_output": "chunked_out_proj_inplace",
                "router_lifetime": "lazy_contiguous_query_chunks",
                "project_chunk_rows": projected.project_chunk_rows,
                "query_chunk_rows": projected.query_chunk_rows,
                "out_proj_chunk_rows": OUT_PROJ_CHUNK_ROWS,
            }
        )

        return PreparedStreamedHybrid(
            sparse=PreparedStreamedSparseSage(
                projected=projected,
                route_plan=route_plan,
                metadata=metadata,
            )
        )

    def execute_projected(self, module, prepared):
        if not isinstance(prepared, PreparedStreamedHybrid):
            return None
        return _execute_streamed(
            module,
            self,
            prepared.sparse,
        )

    def execute(self, prepared):
        if isinstance(prepared, PreparedStreamedHybrid):
            raise SparseSageError(
                "streamed Sparse Sage must execute through execute_projected"
            )
        return super().execute(prepared)

    def as_status(self):
        status = super().as_status()
        status.update(
            {
                "streamed_q": True,
                "chunked_attention_output": True,
                "reuses_attention_input_for_output": True,
                "query_chunk_rows": getattr(
                    self.projector,
                    "query_chunk_rows",
                    None,
                ),
                "q_only_second_pass": bool(
                    getattr(self.projector, "q_only_convrot", False)
                ),
            }
        )
        return status
