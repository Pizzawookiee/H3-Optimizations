"""Pooled-tail helpers for H3 sparse attention.

This intentionally does not implement Sol routing or token augmentation. It
keeps the existing H3 route and approximates only the route complement with
one K centroid and one summed V per KV tile, merged through the exact sparse
softmax normalizer.
"""

from __future__ import annotations

import math
import torch

_LOG2E = math.log2(math.e)


def tile_mean(x, tile):
    sequence = int(x.shape[-2])
    tile = int(tile)
    full, remainder = divmod(sequence, tile)
    pieces = []
    if full:
        pieces.append(
            x[..., : full * tile, :]
            .reshape(*x.shape[:-2], full, tile, x.shape[-1])
            .mean(dim=-2)
        )
    if remainder:
        pieces.append(x[..., full * tile :, :].mean(dim=-2, keepdim=True))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)


def tile_sum(x, tile):
    sequence = int(x.shape[-2])
    tile = int(tile)
    full, remainder = divmod(sequence, tile)
    pieces = []
    if full:
        pieces.append(
            x[..., : full * tile, :]
            .reshape(*x.shape[:-2], full, tile, x.shape[-1])
            .float()
            .sum(dim=-2)
        )
    if remainder:
        pieces.append(x[..., full * tile :, :].float().sum(dim=-2, keepdim=True))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)


def block_lengths(sequence, tile, device):
    blocks = (int(sequence) + int(tile) - 1) // int(tile)
    lengths = torch.full((blocks,), float(tile), dtype=torch.float32, device=device)
    remainder = int(sequence) - (blocks - 1) * int(tile)
    if remainder != int(tile):
        lengths[-1] = float(remainder)
    return lengths


def exact_mask_from_route(route, kv_tiles):
    absolute = route.to_absolute()
    indices = absolute.indices.to(torch.int64)
    counts = absolute.counts.to(torch.int64)
    live = torch.arange(indices.shape[-1], device=indices.device).view(1, 1, 1, -1)
    live = live < counts.unsqueeze(-1)
    hits = torch.zeros(
        *indices.shape[:-1], int(kv_tiles), dtype=torch.int32, device=indices.device
    )
    hits.scatter_add_(
        -1,
        indices.clamp(0, int(kv_tiles) - 1),
        live.to(torch.int32),
    )
    return hits > 0


def exact_mask_from_compact(
    sparse_lut,
    *,
    heads,
    q_tiles,
    kv_tiles,
    dense_q_tiles,
):
    """Expand H3's dense-implicit + absolute sparse LUT to a bool block mask."""
    mask = torch.zeros(
        1, int(heads), int(q_tiles), int(kv_tiles),
        dtype=torch.bool, device=sparse_lut.device,
    )
    dense = min(int(q_tiles), max(0, int(dense_q_tiles)))
    if dense:
        mask[..., :dense, :] = True
    sparse_count = min(
        int(q_tiles) - dense,
        int(sparse_lut.shape[-2]),
    )
    if sparse_count > 0 and sparse_lut.shape[-1] > 0:
        rows = sparse_lut[..., :sparse_count, :].to(torch.int64)
        mask[..., dense:dense + sparse_count, :].scatter_(
            -1, rows.clamp(0, int(kv_tiles) - 1), True
        )
    return mask


def merge_pooled_tail(
    exact_output,
    exact_lse2,
    *,
    q_summary,
    k_summary,
    v_sum,
    exact_mask,
    sequence,
    q_tile,
    kv_tile,
    scale,
    k_offset=None,
):
    """Merge exact sparse attention with a Sol-style pooled complement.

    ``exact_lse2`` is log2(sum(exp2(logits))) for the exact branch. The pooled
    branch uses one query-tile centroid score per omitted KV tile, V sums, and
    the live token count in each tile. Both branches are merged in the same
    base-2 softmax normalization.
    """
    if exact_output.ndim != 4 or exact_lse2.ndim != 3:
        raise ValueError('pooled tail expects HND output and BHT LSE')
    if q_summary.ndim != 4 or k_summary.ndim != 4 or v_sum.ndim != 4:
        raise ValueError('pooled tail summaries must be rank-4 HND block tensors')
    qf = q_summary.float()
    kf = k_summary.float()
    if k_offset is not None:
        kf = kf - k_offset.float().unsqueeze(-2)
    vf = v_sum.float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * (float(scale) * _LOG2E)
    if tuple(exact_mask.shape) != tuple(scores.shape):
        raise ValueError(
            'pooled tail exact-mask shape %s differs from score shape %s'
            % (tuple(exact_mask.shape), tuple(scores.shape))
        )
    neg = torch.tensor(float('-inf'), device=scores.device, dtype=scores.dtype)
    scores = scores.masked_fill(exact_mask, neg)
    lengths = block_lengths(sequence, kv_tile, scores.device)

    tail_max = scores.amax(dim=-1)
    finite = torch.isfinite(tail_max)
    safe_max = torch.where(finite, tail_max, torch.zeros_like(tail_max))
    weights = torch.exp2(scores - safe_max.unsqueeze(-1))
    weights = torch.where(torch.isfinite(scores), weights, torch.zeros_like(weights))
    tail_den_scaled = (weights * lengths.view(1, 1, 1, -1)).sum(dim=-1)
    tail_num_scaled = torch.matmul(weights, vf)
    tail_out = tail_num_scaled / tail_den_scaled.clamp_min(1e-30).unsqueeze(-1)
    tail_lse2 = torch.where(
        tail_den_scaled > 0,
        safe_max + torch.log2(tail_den_scaled.clamp_min(1e-30)),
        torch.full_like(safe_max, float('-inf')),
    )

    rows = int(exact_output.shape[-2])
    tail_out_rows = tail_out.repeat_interleave(int(q_tile), dim=-2)[..., :rows, :]
    tail_lse_rows = tail_lse2.repeat_interleave(int(q_tile), dim=-1)[..., :rows]

    ln2 = math.log(2.0)
    total_lse2 = torch.logaddexp(exact_lse2.float() * ln2, tail_lse_rows * ln2) / ln2
    exact_weight = torch.exp2(exact_lse2.float() - total_lse2)
    tail_weight = torch.exp2(tail_lse_rows - total_lse2)
    merged = (
        exact_output.float() * exact_weight.unsqueeze(-1)
        + tail_out_rows * tail_weight.unsqueeze(-1)
    )
    # Preserve the caller's physical output layout (notably Kitchen's NHD
    # backing storage) instead of returning a fresh contiguous HND tensor.
    exact_output.copy_(merged.to(exact_output.dtype))
    return exact_output, total_lse2
