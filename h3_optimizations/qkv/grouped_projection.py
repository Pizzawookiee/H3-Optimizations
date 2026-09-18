"""Bounded multi-head arbitrary-row projection helpers for H3V-Smooth.

The important property is that the per-head row map is consumed inside the
projection kernel.  We never materialize [heads, rows, hidden] gathered input
or a full reordered BF16 K/V carrier.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


class GroupedProjectionError(RuntimeError):
    pass


if TRITON_AVAILABLE:
    @triton.jit
    def _grouped_kv_bf16_kernel(
        x_ptr, w_ptr, bias_ptr, rows_ptr, k_ptr, v_ptr,
        hidden: tl.constexpr, heads: tl.constexpr, head_dim: tl.constexpr,
        count: tl.constexpr, weight_stride_o: tl.constexpr,
        weight_stride_i: tl.constexpr, x_stride_n: tl.constexpr,
        rows_stride_h: tl.constexpr, rows_stride_n: tl.constexpr,
        has_bias: tl.constexpr, DO_K: tl.constexpr, DO_V: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d = tl.arange(0, head_dim)
        mask_m = m < count
        src = tl.load(rows_ptr + h * rows_stride_h + m * rows_stride_n,
                      mask=mask_m, other=0).to(tl.int64)
        acc_k = tl.zeros((BLOCK_M, head_dim), tl.float32)
        acc_v = tl.zeros((BLOCK_M, head_dim), tl.float32)
        k_base = heads * head_dim + h * head_dim
        v_base = 2 * heads * head_dim + h * head_dim
        for kk0 in range(0, hidden, BLOCK_K):
            kk = kk0 + tl.arange(0, BLOCK_K)
            kmask = kk < hidden
            x = tl.load(x_ptr + src[:, None] * x_stride_n + kk[None, :],
                        mask=mask_m[:, None] & kmask[None, :], other=0.0)
            if DO_K:
                wk = tl.load(w_ptr + (k_base + d[:, None]) * weight_stride_o
                             + kk[None, :] * weight_stride_i,
                             mask=kmask[None, :], other=0.0)
                acc_k += tl.dot(x, tl.trans(wk))
            if DO_V:
                wv = tl.load(w_ptr + (v_base + d[:, None]) * weight_stride_o
                             + kk[None, :] * weight_stride_i,
                             mask=kmask[None, :], other=0.0)
                acc_v += tl.dot(x, tl.trans(wv))
        if has_bias:
            if DO_K:
                acc_k += tl.load(bias_ptr + k_base + d)[None, :]
            if DO_V:
                acc_v += tl.load(bias_ptr + v_base + d)[None, :]
        out = (h * count + m)[:, None] * head_dim + d[None, :]
        omask = mask_m[:, None]
        if DO_K:
            tl.store(k_ptr + out, acc_k, mask=omask)
        if DO_V:
            tl.store(v_ptr + out, acc_v, mask=omask)


def grouped_bf16_kv_linear(x, weight, bias, rows, *, heads, head_dim, want_k=True, want_v=True):
    """Project arbitrary rows for every head in one Triton launch family.

    rows is [heads, count] and contains absolute sequence rows.  The returned
    tensors are [heads, count, head_dim].  This path is intentionally limited
    to effective BF16 weights; quantized providers retain their native kernels.
    """
    if not TRITON_AVAILABLE:
        raise GroupedProjectionError('Triton is unavailable')
    if not (x.is_cuda and weight.is_cuda and rows.is_cuda):
        raise GroupedProjectionError('grouped BF16 projection requires CUDA')
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise GroupedProjectionError('grouped BF16 projection requires BF16 activation and weight')
    if x.ndim != 2 or weight.ndim != 2 or rows.ndim != 2:
        raise GroupedProjectionError('invalid grouped BF16 projection rank')
    heads, head_dim = int(heads), int(head_dim)
    if int(rows.shape[0]) != heads:
        raise GroupedProjectionError('row map head count mismatch')
    count = int(rows.shape[1])
    if count <= 0:
        raise GroupedProjectionError('row map is empty')
    hidden = int(x.shape[1])
    if int(weight.shape[1]) != hidden or int(weight.shape[0]) < 3 * heads * head_dim:
        raise GroupedProjectionError('QKV weight geometry mismatch')
    rows = rows.to(dtype=torch.int32).contiguous()
    k = torch.empty((heads, count, head_dim), dtype=x.dtype, device=x.device) if want_k else torch.empty((1,), dtype=x.dtype, device=x.device)
    v = torch.empty((heads, count, head_dim), dtype=x.dtype, device=x.device) if want_v else torch.empty((1,), dtype=x.dtype, device=x.device)
    bias_ptr = bias if bias is not None else weight
    block_m = 16 if count < 64 else 32
    grid = (triton.cdiv(count, block_m), heads)
    _grouped_kv_bf16_kernel[grid](
        x, weight, bias_ptr, rows, k, v,
        hidden=hidden, heads=heads, head_dim=head_dim, count=count,
        weight_stride_o=int(weight.stride(0)), weight_stride_i=int(weight.stride(1)),
        x_stride_n=int(x.stride(0)), rows_stride_h=int(rows.stride(0)),
        rows_stride_n=int(rows.stride(1)), has_bias=bias is not None,
        DO_K=bool(want_k), DO_V=bool(want_v), BLOCK_M=block_m, BLOCK_K=32,
        num_warps=4,
    )
    return (k if want_k else None), (v if want_v else None)


__all__ = ['GroupedProjectionError', 'TRITON_AVAILABLE', 'grouped_bf16_kv_linear']
