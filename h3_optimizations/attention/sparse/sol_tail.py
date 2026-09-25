"""Sol-style complement helpers for H3 sparse attention.

The existing H3 router remains authoritative. These helpers improve only the
route complement: pooled tail preserves approximate mass from omitted KV tiles,
and the optional token augmentation stage rescues a small shared set of
individual KV tokens from blocks that neighbouring H3 query tiles both omitted.
Sol's own block router is intentionally not implemented here.
"""

from __future__ import annotations

import math
import torch

_LOG2E = math.log2(math.e)
SOL_TOKEN_AUG_BUDGET = 64
SOL_TOKEN_GROUP = 2
_TOKEN_SCORE_GROUP_CHUNK = 4


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


def _pooled_tail_block_state(
    *,
    q_summary,
    k_summary,
    v_sum,
    exact_mask,
    sequence,
    kv_tile,
    scale,
    k_offset=None,
):
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
    scores = scores.masked_fill(exact_mask, float('-inf'))
    lengths = block_lengths(sequence, kv_tile, scores.device)

    tail_max = scores.amax(dim=-1)
    finite = torch.isfinite(tail_max)
    safe_max = torch.where(finite, tail_max, torch.zeros_like(tail_max))
    weights = torch.exp2(scores - safe_max.unsqueeze(-1))
    weights = torch.where(torch.isfinite(scores), weights, torch.zeros_like(weights))
    tail_den_scaled = (weights * lengths.view(1, 1, 1, -1)).sum(dim=-1)
    tail_num_scaled = torch.matmul(weights, vf)
    return scores, tail_max, tail_num_scaled, tail_den_scaled


def pooled_tail_state(
    *,
    q_summary,
    k_summary,
    v_sum,
    exact_mask,
    sequence,
    q_tile,
    kv_tile,
    scale,
    rows,
    k_offset=None,
):
    """Return normalized pooled-tail output and base-2 LSE per query row."""
    _scores, tail_max, tail_num_scaled, tail_den_scaled = _pooled_tail_block_state(
        q_summary=q_summary,
        k_summary=k_summary,
        v_sum=v_sum,
        exact_mask=exact_mask,
        sequence=sequence,
        kv_tile=kv_tile,
        scale=scale,
        k_offset=k_offset,
    )
    safe_den = tail_den_scaled.clamp_min(1e-30)
    tail_out = tail_num_scaled / safe_den.unsqueeze(-1)
    tail_lse2 = torch.where(
        tail_den_scaled > 0,
        tail_max + torch.log2(safe_den),
        torch.full_like(tail_max, float('-inf')),
    )
    tail_out_rows = tail_out.repeat_interleave(int(q_tile), dim=-2)[..., :int(rows), :]
    tail_lse_rows = tail_lse2.repeat_interleave(int(q_tile), dim=-1)[..., :int(rows)]
    return tail_out_rows, tail_lse_rows


def merge_attention_state(exact_output, exact_lse2, branch_output, branch_lse2):
    """Merge a second normalized attention state without touching empty rows."""
    if exact_output.ndim != 4 or exact_lse2.ndim != 3:
        raise ValueError('attention merge expects HND output and BHT exact LSE')
    if branch_output.shape != exact_output.shape or branch_lse2.shape != exact_lse2.shape:
        raise ValueError('attention merge branch shapes differ from exact branch')

    active = torch.isfinite(branch_lse2)

    ln2 = math.log(2.0)
    total_calc = torch.logaddexp(
        exact_lse2.float() * ln2,
        branch_lse2.float() * ln2,
    ) / ln2
    exact_weight = torch.exp2(exact_lse2.float() - total_calc)
    branch_weight = torch.exp2(branch_lse2.float() - total_calc)
    merged = (
        exact_output.float() * exact_weight.unsqueeze(-1)
        + branch_output.float() * branch_weight.unsqueeze(-1)
    )
    exact_output.copy_(
        torch.where(active.unsqueeze(-1), merged, exact_output.float()).to(exact_output.dtype)
    )
    total_lse2 = torch.where(active, total_calc, exact_lse2.float())
    return exact_output, total_lse2


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
    """Merge exact sparse attention with a Sol-style pooled complement."""
    if q_summary.ndim != 4 or k_summary.ndim != 4 or v_sum.ndim != 4:
        raise ValueError('pooled tail summaries must be rank-4 HND block tensors')
    branch_output, branch_lse2 = pooled_tail_state(
        q_summary=q_summary,
        k_summary=k_summary,
        v_sum=v_sum,
        exact_mask=exact_mask,
        sequence=sequence,
        q_tile=q_tile,
        kv_tile=kv_tile,
        scale=scale,
        rows=int(exact_output.shape[-2]),
        k_offset=k_offset,
    )
    return merge_attention_state(
        exact_output, exact_lse2, branch_output, branch_lse2
    )


def quantize_token_rows(x):
    """Cheap per-token INT8 carrier used only by Sol token augmentation.

    This is deliberately separate from Kitchen's execution carrier. It keeps
    one scale per token, so individual-token scores can be reconstructed
    without retaining full BF16 K/Q tensors.
    """
    if x.ndim != 4:
        raise ValueError('token carrier expects rank-4 HND rows')
    xf = x.float()
    scale = xf.abs().amax(dim=-1).div(127.0).clamp_min_(1e-8)
    packed = torch.round(xf / scale.unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
    return packed.contiguous(), scale.contiguous()


def _dequant_rows(x, scale):
    if scale is None:
        return x.float()
    return x.float() * scale.float().unsqueeze(-1)


def _token_groups(q_summary, exact_mask, group_size):
    q_tiles = int(q_summary.shape[-2])
    centroids = []
    candidates = []
    ranges = []
    start = 0
    while start < q_tiles:
        stop = min(q_tiles, start + int(group_size))
        centroids.append(q_summary[..., start:stop, :].float().mean(dim=-2))
        candidates.append((~exact_mask[..., start:stop, :]).all(dim=-2))
        ranges.append((start, stop))
        start = stop
    return torch.stack(centroids, dim=-2), torch.stack(candidates, dim=-2), ranges


_HIST_LOW = 8.0
_HIST_FINE_WIDTH = 0.25
_HIST_FINE_BINS = 96
_HIST_COARSE_WIDTH = 2.0
_HIST_BINS = 128
_TOKEN_SCORE_GROUP_CHUNK = 4
_TOKEN_SCORE_K_CHUNK = 4096


def _histogram_bins(relative):
    fine_span = _HIST_FINE_BINS * _HIST_FINE_WIDTH
    fine = torch.floor(relative / _HIST_FINE_WIDTH).to(torch.int64)
    coarse = _HIST_FINE_BINS + torch.floor(
        (relative - fine_span) / _HIST_COARSE_WIDTH
    ).to(torch.int64)
    bins = torch.where(relative < fine_span, fine, coarse)
    return bins.clamp_(0, _HIST_BINS - 1)


def _hist_edge(index):
    index = int(index)
    if index <= _HIST_FINE_BINS:
        return float(index) * _HIST_FINE_WIDTH
    return (
        _HIST_FINE_BINS * _HIST_FINE_WIDTH
        + (index - _HIST_FINE_BINS) * _HIST_COARSE_WIDTH
    )


def _histogram_thresholds(hist, reference, budget):
    """Kitchen/Sol threshold rule: admit whole bins, never split a bin."""
    flat = hist.reshape(-1, _HIST_BINS)
    out = torch.empty(flat.shape[0], dtype=torch.float32, device=hist.device)
    refs = reference.reshape(-1).float()
    # Only 128 bins and a small B*H*group count. Keeping this control logic on
    # CPU would synchronize; the loop stays on-device through tensor ops.
    acc = torch.zeros(flat.shape[0], dtype=torch.int64, device=hist.device)
    boundary = torch.full(
        (flat.shape[0],), -1, dtype=torch.int64, device=hist.device
    )
    open_rows = torch.ones(flat.shape[0], dtype=torch.bool, device=hist.device)
    for b in range(_HIST_BINS - 1, -1, -1):
        proposed = acc + flat[:, b].to(torch.int64)
        overflow = open_rows & (proposed > int(budget))
        boundary = torch.where(overflow, torch.full_like(boundary, b), boundary)
        open_rows = open_rows & ~overflow
        acc = torch.where(open_rows, proposed, acc)
    edges = torch.tensor(
        [_hist_edge(i) for i in range(_HIST_BINS + 1)],
        dtype=torch.float32,
        device=hist.device,
    )
    # If everything fits boundary remains -1, matching the CUDA b+1 == 0 case.
    edge_index = (boundary + 1).clamp_(0, _HIST_BINS)
    out.copy_(refs - _HIST_LOW + edges.index_select(0, edge_index))
    return out.reshape(reference.shape)


def _select_augmented_tokens_histogram(
    *,
    q_summary,
    k,
    exact_mask,
    kv_tile,
    scale,
    token_budget,
    k_scale=None,
    block_scores=None,
    group_size=SOL_TOKEN_GROUP,
):
    """Sol token selection using the two-pass histogram threshold.

    The H3 block route remains authoritative. A token is eligible only when its
    KV block was omitted by every Q tile in the neighbouring-Q group. Scores
    are shared-group-centroid dot individual-K-token scores. The threshold uses
    the same 128-bin policy as Comfy Kitchen: 0.25 log2 bins from ref-8 through
    ref+16, then 2.0 bins above that. Whole bins are admitted, so a result may
    contain fewer than ``token_budget`` tokens rather than splitting a tie bin.
    """
    centroids, candidates, ranges = _token_groups(q_summary, exact_mask, group_size)
    batch, heads, groups = centroids.shape[:3]
    sequence = int(k.shape[-2])
    budget = min(int(token_budget), sequence)
    if budget <= 0:
        empty_idx = torch.empty(
            (batch, heads, groups, 0), dtype=torch.int64, device=k.device
        )
        return empty_idx, torch.empty_like(empty_idx, dtype=torch.bool), ranges

    token_blocks = torch.div(
        torch.arange(sequence, device=k.device, dtype=torch.int64),
        int(kv_tile),
        rounding_mode='floor',
    )
    if block_scores is None:
        # The reference is only a threshold origin. Computing it from centroid
        # vs block centroid is equivalent to taking the best omitted pooled
        # score for the group when no per-Q score slab was retained.
        raise ValueError('histogram token augmentation requires pooled block scores')

    group_refs = []
    for start, stop in ranges:
        scores = block_scores[..., start:stop, :]
        cand = candidates[..., len(group_refs), :]
        masked = scores.masked_fill(~cand.unsqueeze(-2), float('-inf'))
        group_refs.append(masked.amax(dim=(-2, -1)))
    reference = torch.stack(group_refs, dim=-1)

    hist = torch.zeros(
        batch, heads, groups, _HIST_BINS,
        dtype=torch.int32, device=k.device,
    )
    scale_log2 = float(scale) * _LOG2E

    # Pass 1: score every eligible token and build the relative-score histogram.
    for g0 in range(0, groups, _TOKEN_SCORE_GROUP_CHUNK):
        g1 = min(groups, g0 + _TOKEN_SCORE_GROUP_CHUNK)
        c = centroids[..., g0:g1, :]
        cand_blocks = candidates[..., g0:g1, :]
        ref = reference[..., g0:g1]
        for k0 in range(0, sequence, _TOKEN_SCORE_K_CHUNK):
            k1 = min(sequence, k0 + _TOKEN_SCORE_K_CHUNK)
            kc = _dequant_rows(k[..., k0:k1, :], None if k_scale is None else k_scale[..., k0:k1])
            scores = torch.matmul(c, kc.transpose(-1, -2)) * scale_log2
            blocks = token_blocks[k0:k1]
            eligible = cand_blocks.index_select(-1, blocks)
            rel = scores - ref.unsqueeze(-1) + _HIST_LOW
            live = eligible & torch.isfinite(scores) & (rel >= 0)
            bins = _histogram_bins(rel.clamp_min(0))
            hist[..., g0:g1, :].scatter_add_(
                -1, bins, live.to(torch.int32)
            )
            del kc, scores, eligible, rel, live, bins

    threshold = _histogram_thresholds(hist, reference, budget)
    selected_idx = torch.zeros(
        (batch, heads, groups, budget), dtype=torch.int64, device=k.device
    )
    selected_valid = torch.zeros(
        (batch, heads, groups, budget), dtype=torch.bool, device=k.device
    )
    counts = torch.zeros(
        (batch, heads, groups), dtype=torch.int64, device=k.device
    )

    # Pass 2: rescore, admit complete bins through the derived threshold, and
    # fill deterministic sequence-ordered lists. Histogram semantics guarantee
    # the count does not exceed the budget.
    for g0 in range(0, groups, _TOKEN_SCORE_GROUP_CHUNK):
        g1 = min(groups, g0 + _TOKEN_SCORE_GROUP_CHUNK)
        c = centroids[..., g0:g1, :]
        cand_blocks = candidates[..., g0:g1, :]
        thr = threshold[..., g0:g1]
        for k0 in range(0, sequence, _TOKEN_SCORE_K_CHUNK):
            k1 = min(sequence, k0 + _TOKEN_SCORE_K_CHUNK)
            kc = _dequant_rows(k[..., k0:k1, :], None if k_scale is None else k_scale[..., k0:k1])
            scores = torch.matmul(c, kc.transpose(-1, -2)) * scale_log2
            eligible = cand_blocks.index_select(-1, token_blocks[k0:k1])
            take = eligible & (scores >= thr.unsqueeze(-1))
            # Append in absolute token order without host synchronization.
            flat = take.reshape(-1, k1 - k0)
            base = counts[..., g0:g1].reshape(-1)
            rank = flat.to(torch.int64).cumsum(dim=-1) - 1
            slot = rank + base.unsqueeze(-1)
            emit = flat & (slot < budget)
            row, col = torch.nonzero(emit, as_tuple=True)
            if row.numel():
                target_idx = selected_idx[..., g0:g1, :].reshape(-1, budget)
                target_valid = selected_valid[..., g0:g1, :].reshape(-1, budget)
                dst = slot[row, col]
                target_idx[row, dst] = col.to(torch.int64) + int(k0)
                target_valid[row, dst] = True
            counts[..., g0:g1].copy_(
                (base + flat.sum(dim=-1, dtype=torch.int64))
                .clamp_max(budget)
                .reshape(batch, heads, g1 - g0)
            )
            del kc, scores, eligible, take, flat, base, rank, slot, emit
    return selected_idx, selected_valid, ranges


def _gather_hnd_rows(x, idx, scale=None):
    d = int(x.shape[-1])
    gather_idx = idx.unsqueeze(-1).expand(*idx.shape, d)
    out = torch.gather(x, 2, gather_idx).float()
    if scale is not None:
        s = torch.gather(scale, 2, idx).float()
        out.mul_(s.unsqueeze(-1))
    return out


def _gather_kitchen_v(v, v_scale, idx, *, batch, heads, head_dim):
    if v.ndim != 2 or v_scale is None:
        raise ValueError('Kitchen V carrier must be [B*H*D,Tpad] with channel scales')
    padded = int(v.shape[-1])
    view = v.view(int(batch), int(heads), int(head_dim), padded)
    scale = v_scale.view(int(batch), int(heads), int(head_dim))
    # Gather selected sequence columns independently for every head/channel.
    expanded = idx.unsqueeze(-2).expand(*idx.shape[:-1], int(head_dim), idx.shape[-1])
    gathered = torch.gather(view, 3, expanded).float()
    gathered.mul_(scale.unsqueeze(-1))
    return gathered.transpose(-1, -2).contiguous()


def token_augmented_tail_state(
    q,
    k,
    v,
    *,
    q_summary,
    k_summary,
    v_sum,
    exact_mask,
    sequence,
    q_tile,
    kv_tile,
    scale,
    token_budget=SOL_TOKEN_AUG_BUDGET,
    q_scale=None,
    k_scale=None,
    kitchen_v_scale=None,
    k_offset=None,
):
    """Return pooled-tail + histogram-selected exact-token rescue.

    ``q``/``k`` may be BF16/FP16 HND tensors or simple per-token INT8 carriers
    accompanied by ``q_scale``/``k_scale``. ``v`` may be HND, or Kitchen's
    channel-major INT8 V carrier when ``kitchen_v_scale`` is supplied.
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError('token augmentation expects rank-4 HND Q/K')
    if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError('token augmentation Q/K shapes differ')
    if int(k.shape[-2]) != int(sequence):
        raise ValueError('token augmentation K sequence differs from global sequence')
    if int(token_budget) <= 0:
        return pooled_tail_state(
            q_summary=q_summary,
            k_summary=k_summary,
            v_sum=v_sum,
            exact_mask=exact_mask,
            sequence=sequence,
            q_tile=q_tile,
            kv_tile=kv_tile,
            scale=scale,
            rows=int(q.shape[-2]),
            k_offset=k_offset,
        )

    block_scores, tail_max, tail_num_scaled, tail_den_scaled = _pooled_tail_block_state(
        q_summary=q_summary,
        k_summary=k_summary,
        v_sum=v_sum,
        exact_mask=exact_mask,
        sequence=sequence,
        kv_tile=kv_tile,
        scale=scale,
        k_offset=k_offset,
    )
    selected_idx, selected_valid, ranges = _select_augmented_tokens_histogram(
        q_summary=q_summary,
        k=k,
        k_scale=k_scale,
        exact_mask=exact_mask,
        kv_tile=kv_tile,
        scale=scale,
        token_budget=token_budget,
        block_scores=block_scores,
    )

    rows = int(q.shape[-2])
    branch_output = torch.zeros(
        q.shape, dtype=(torch.bfloat16 if q.dtype == torch.int8 else q.dtype), device=q.device
    )
    branch_lse2 = torch.full(
        (q.shape[0], q.shape[1], rows),
        float('-inf'), dtype=torch.float32, device=q.device,
    )
    scale_log2 = float(scale) * _LOG2E
    d = int(q.shape[-1])

    for group_index, (tile_start, tile_stop) in enumerate(ranges):
        row_start = int(tile_start) * int(q_tile)
        row_stop = min(rows, int(tile_stop) * int(q_tile))
        if row_start >= row_stop:
            continue

        idx = selected_idx[..., group_index, :]
        valid = selected_valid[..., group_index, :]
        k_sel = _gather_hnd_rows(k, idx, k_scale)
        if kitchen_v_scale is None:
            v_sel = _gather_hnd_rows(v, idx)
        else:
            v_sel = _gather_kitchen_v(
                v, kitchen_v_scale, idx,
                batch=q.shape[0], heads=q.shape[1], head_dim=d,
            )

        q_rows = _dequant_rows(
            q[..., row_start:row_stop, :],
            None if q_scale is None else q_scale[..., row_start:row_stop],
        )
        exact_scores = torch.einsum('bhrd,bhnd->bhrn', q_rows, k_sel) * scale_log2
        exact_scores = exact_scores.masked_fill(~valid.unsqueeze(-2), float('-inf'))

        block_idx = torch.div(idx, int(kv_tile), rounding_mode='floor')
        group_block_scores = block_scores[..., tile_start:tile_stop, :]
        approx = torch.gather(
            group_block_scores,
            -1,
            block_idx.unsqueeze(-2).expand(
                *block_idx.shape[:-1], int(tile_stop - tile_start), block_idx.shape[-1]
            ),
        )
        approx_rows = approx.repeat_interleave(int(q_tile), dim=-2)
        approx_rows = approx_rows[..., :row_stop - row_start, :]
        approx_rows = approx_rows.masked_fill(~valid.unsqueeze(-2), float('-inf'))

        group_tail_max = tail_max[..., tile_start:tile_stop]
        tail_max_rows = group_tail_max.repeat_interleave(int(q_tile), dim=-1)
        tail_max_rows = tail_max_rows[..., :row_stop - row_start]
        exact_max = exact_scores.amax(dim=-1)
        merged_max = torch.maximum(tail_max_rows, exact_max)
        finite_max = torch.isfinite(merged_max)
        safe_max = torch.where(finite_max, merged_max, torch.zeros_like(merged_max))

        base_num = tail_num_scaled[..., tile_start:tile_stop, :]
        base_num = base_num.repeat_interleave(int(q_tile), dim=-2)
        base_num = base_num[..., :row_stop - row_start, :]
        base_den = tail_den_scaled[..., tile_start:tile_stop]
        base_den = base_den.repeat_interleave(int(q_tile), dim=-1)
        base_den = base_den[..., :row_stop - row_start]
        base_scale = torch.exp2(tail_max_rows - safe_max)
        base_scale = torch.where(torch.isfinite(tail_max_rows), base_scale, torch.zeros_like(base_scale))
        num = base_num * base_scale.unsqueeze(-1)
        den = base_den * base_scale

        approx_w = torch.exp2(approx_rows - safe_max.unsqueeze(-1))
        exact_w = torch.exp2(exact_scores - safe_max.unsqueeze(-1))
        approx_w = torch.where(torch.isfinite(approx_rows), approx_w, torch.zeros_like(approx_w))
        exact_w = torch.where(torch.isfinite(exact_scores), exact_w, torch.zeros_like(exact_w))
        correction = exact_w - approx_w
        num = num + torch.einsum('bhrn,bhnd->bhrd', correction, v_sel)
        den = den + correction.sum(dim=-1)

        active = finite_max & (den > 0)
        safe_den = den.clamp_min(1e-30)
        out = num / safe_den.unsqueeze(-1)
        lse = safe_max + torch.log2(safe_den)
        branch_output[..., row_start:row_stop, :].copy_(
            torch.where(active.unsqueeze(-1), out, torch.zeros_like(out)).to(branch_output.dtype)
        )
        branch_lse2[..., row_start:row_stop].copy_(
            torch.where(active, lse, torch.full_like(lse, float('-inf')))
        )
        del k_sel, v_sel, q_rows, exact_scores, approx, approx_rows

    return branch_output, branch_lse2


def merge_pooled_tail_token_aug(
    exact_output,
    exact_lse2,
    q,
    k,
    v,
    *,
    q_summary,
    k_summary,
    v_sum,
    exact_mask,
    sequence,
    q_tile,
    kv_tile,
    scale,
    token_budget=SOL_TOKEN_AUG_BUDGET,
    q_scale=None,
    k_scale=None,
    kitchen_v_scale=None,
    k_offset=None,
):
    branch_output, branch_lse2 = token_augmented_tail_state(
        q, k, v,
        q_summary=q_summary,
        k_summary=k_summary,
        v_sum=v_sum,
        exact_mask=exact_mask,
        sequence=sequence,
        q_tile=q_tile,
        kv_tile=kv_tile,
        scale=scale,
        token_budget=token_budget,
        q_scale=q_scale,
        k_scale=k_scale,
        kitchen_v_scale=kitchen_v_scale,
        k_offset=k_offset,
    )
    return merge_attention_state(exact_output, exact_lse2, branch_output, branch_lse2)
