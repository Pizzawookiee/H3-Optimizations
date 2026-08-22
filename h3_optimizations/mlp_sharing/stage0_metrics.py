'''Stage 0 measurement math for attention-selected and cached H3 MLP work.

Every function here is a pure tensor computation with no ComfyUI or model
dependency, so the whole statistic set is exercised on CPU.

Three families of question are answered:

A. Does an attention signal identify disposable MLP work? Selection is a top-k
   decision over 64-row KV tiles, so the statistic is oracle mass captured by a
   selector's top-k, not a correlation.
B. Is the cross-step SwiGLU delta concentrated in few ConvRot groups, and is
   the group set shared between tokens?
C. Which cache representation preserves the cached output and the group
   ranking - BF16 or FP8 E4M3?
'''

from __future__ import annotations

import torch

CONVROT_GROUP = 256
FFN_GROUPS = 56
FP8_MAX = 448.0
TILE_BUDGETS = (0.125, 0.25, 0.5, 0.75)
GROUP_BUDGETS = (7, 14, 28)
CACHE_DTYPES = ('bfloat16', 'float8_e4m3fn')
GRANULARITIES = ('token', 'tile64', 'tile128', 'video')
TILE_SELECTORS = (
    'hit_rate',
    'attention_update_energy',
    'combined',
    'random',
    'oracle',
)


class Stage0Error(RuntimeError):
    pass


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def deterministic_uniform(count, seed, device):
    '''Reproducible [0, 1) values; the random selector control.'''
    values = torch.arange(int(count), dtype=torch.int64, device=device)
    values.mul_(1_103_515_245).add_(int(seed) & 0x7FFFFFFF)
    values.bitwise_and_(0x7FFFFFFF)
    values.mul_(1_103_515_245).add_(12_345)
    values.bitwise_and_(0x7FFFFFFF)
    return values.to(torch.float64).div_(float(0x7FFFFFFF)).to(torch.float32)


def rank_normalize(values):
    '''Map a score vector onto [0, 1] by rank so scales become comparable.'''
    values = values.float().reshape(-1)
    count = int(values.numel())
    if count <= 1:
        return torch.zeros_like(values)
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(
        count,
        dtype=values.dtype,
        device=values.device,
    )
    return ranks / float(count - 1)


def top_k_mask(scores, k):
    '''Boolean mask of the k highest scores along the last dimension.'''
    k = int(k)
    width = int(scores.shape[-1])
    if k <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    if k >= width:
        return torch.ones_like(scores, dtype=torch.bool)
    indices = torch.topk(scores, k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter_(-1, indices, True)


def captured_fraction(weights, mask):
    '''Fraction of total ``weights`` mass that ``mask`` retains.'''
    total = weights.double().sum()
    if float(total) <= 0.0:
        return float('nan')
    return float(weights.double()[mask].sum() / total)


def jaccard(mask_a, mask_b):
    union = float((mask_a | mask_b).sum())
    if union <= 0.0:
        return float('nan')
    return float((mask_a & mask_b).sum()) / union


def fp8_roundtrip(values, *, per_row=True):
    '''Row-scaled E4M3 store/load, the representation a cache would use.'''
    source = values.float()
    if per_row:
        scale = source.abs().amax(dim=-1, keepdim=True)
    else:
        scale = source.abs().amax()
    scale = torch.where(
        scale > 0.0,
        scale / FP8_MAX,
        torch.ones_like(scale),
    )
    stored = (source / scale).to(torch.float8_e4m3fn)
    return (stored.to(torch.float32) * scale).to(values.dtype)


def store_roundtrip(values, dtype):
    '''Store and reload through one candidate cache representation.'''
    if dtype == 'bfloat16':
        return values.to(torch.bfloat16).to(values.dtype)
    if dtype == 'float8_e4m3fn':
        return fp8_roundtrip(values)
    raise Stage0Error('unknown cache dtype %r' % dtype)


def relative_norm(delta, reference, *, dim=-1):
    delta_norm = torch.linalg.vector_norm(delta.float(), dim=dim)
    reference_norm = torch.linalg.vector_norm(reference.float(), dim=dim)
    return delta_norm / reference_norm.clamp_min(1.0e-12)


def cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denominator = (
        torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    ).clamp_min(1.0e-12)
    return float(torch.dot(a, b) / denominator)


def summarize(values):
    '''Mean/median/p99/max of a 1-D statistic, as plain floats.'''
    values = values.float().reshape(-1)
    values = values[torch.isfinite(values)]
    if not int(values.numel()):
        return {'count': 0}
    return {
        'count': int(values.numel()),
        'mean': float(values.mean()),
        'median': float(values.median()),
        'p99': float(values.quantile(0.99)),
        'max': float(values.max()),
    }


# --------------------------------------------------------------------------
# A. attention selectors against the MLP-update oracle
# --------------------------------------------------------------------------

def row_update_energy(y, gate, selector):
    '''Per-row ``|gate_mlp * y|^2``: the update a skipped row would lose.'''
    if torch.is_tensor(selector) and selector.device != gate.device:
        selector = selector.to(device=gate.device)
    gate_rows = gate[selector].to(y.dtype)
    return (y * gate_rows).float().pow(2).sum(dim=-1)


def accumulate_tile_energy(destination, row_energy, first_row, kv_tile,
                           video_kv_start, video_kv_tiles):
    '''Add chunk-local row energy into a global pure-video tile accumulator.

    Rows outside the pure-video KV tiles - context and the mixed boundary tile
    - are dropped, matching the rows Sparse Sage never routes sparsely.
    '''
    rows = int(row_energy.shape[0])
    absolute = torch.arange(
        int(first_row),
        int(first_row) + rows,
        dtype=torch.int64,
        device=row_energy.device,
    )
    relative = torch.div(
        absolute,
        int(kv_tile),
        rounding_mode='floor',
    ) - int(video_kv_start)
    inside = (relative >= 0) & (relative < int(video_kv_tiles))
    if not bool(inside.any()):
        return destination
    destination.scatter_add_(
        0,
        relative[inside],
        row_energy[inside].double(),
    )
    return destination


def selector_scores(*, hit_rate, attention_update_energy, oracle, seed):
    '''Every candidate tile score, including the two controls.'''
    tiles = int(oracle.shape[0])
    device = oracle.device
    scores = {'oracle': oracle.float()}
    scores['random'] = deterministic_uniform(tiles, seed, device)
    if hit_rate is not None:
        scores['hit_rate'] = hit_rate.float()
    if attention_update_energy is not None:
        scores['attention_update_energy'] = attention_update_energy.float()
    if 'hit_rate' in scores and 'attention_update_energy' in scores:
        # Rank-normalized sum: the two signals have unrelated units.
        scores['combined'] = (
            rank_normalize(scores['hit_rate'])
            + rank_normalize(scores['attention_update_energy'])
        )
    return scores


def selector_capture_table(scores, oracle, budgets=TILE_BUDGETS):
    '''Oracle mass captured, and error incurred, per selector and budget.'''
    tiles = int(oracle.shape[0])
    oracle = oracle.float()
    rows = []
    for budget in budgets:
        retained = max(1, min(tiles, int(round(float(budget) * tiles))))
        oracle_mask = top_k_mask(oracle, retained)
        for name, score in scores.items():
            mask = top_k_mask(score, retained)
            captured = captured_fraction(oracle, mask)
            rows.append({
                'selector': name,
                'tile_budget': float(budget),
                'retained_tiles': retained,
                'total_tiles': tiles,
                'oracle_mass_captured': captured,
                'skipped_update_error': (
                    float('nan') if captured != captured else 1.0 - captured
                ),
                'oracle_topk_overlap': jaccard(mask, oracle_mask),
            })
    return rows


# --------------------------------------------------------------------------
# B. cross-step SwiGLU delta structure
# --------------------------------------------------------------------------

def group_magnitudes(delta, group=CONVROT_GROUP):
    '''Per-token, per-ConvRot-group squared delta magnitude.'''
    rows, width = delta.shape
    if width % int(group):
        raise Stage0Error(
            'SwiGLU width %d is not a multiple of ConvRot group %d'
            % (width, int(group))
        )
    return delta.float().reshape(rows, width // int(group), int(group)).pow(
        2
    ).sum(dim=-1)


def _block_masks(group_energy, block_rows, k):
    '''Top-k group mask shared by every row inside a block.'''
    rows, groups = group_energy.shape
    if block_rows <= 0 or rows % block_rows:
        raise Stage0Error(
            'sampled rows %d do not divide into %s-row blocks'
            % (rows, block_rows)
        )
    blocks = rows // block_rows
    pooled = group_energy.reshape(blocks, block_rows, groups).sum(dim=1)
    mask = top_k_mask(pooled, k)
    return mask.repeat_interleave(block_rows, dim=0)


def group_mask_for(group_energy, granularity, k):
    '''Top-k ConvRot group mask at one sharing granularity.'''
    rows = int(group_energy.shape[0])
    if granularity == 'token':
        return top_k_mask(group_energy, k)
    if granularity == 'tile64':
        return _block_masks(group_energy, min(64, rows), k)
    if granularity == 'tile128':
        return _block_masks(group_energy, min(128, rows), k)
    if granularity == 'video':
        return _block_masks(group_energy, rows, k)
    raise Stage0Error('unknown group granularity %r' % granularity)


def group_concentration(group_energy, budgets=GROUP_BUDGETS,
                        granularities=GRANULARITIES):
    '''How much delta mass each sharing granularity keeps at each budget.'''
    rows = []
    for k in budgets:
        per_token = group_mask_for(group_energy, 'token', k)
        token_capture = captured_fraction(group_energy, per_token)
        for granularity in granularities:
            mask = group_mask_for(group_energy, granularity, k)
            captured = captured_fraction(group_energy, mask)
            rows.append({
                'granularity': granularity,
                'groups': int(k),
                'total_groups': int(group_energy.shape[1]),
                'delta_mass_captured': captured,
                # 1.0 means a shared mask is as good as per-token selection.
                'fraction_of_token_oracle': (
                    float('nan')
                    if token_capture != token_capture or token_capture <= 0.0
                    else captured / token_capture
                ),
            })
    return rows


def pairwise_group_jaccard(group_energy, k, *, pairs=256, seed=0):
    '''Mean top-k group overlap between randomly drawn token pairs.'''
    rows = int(group_energy.shape[0])
    if rows < 2:
        return float('nan')
    mask = top_k_mask(group_energy, k)
    draw = deterministic_uniform(2 * int(pairs), seed, group_energy.device)
    index = (draw * rows).long().clamp_(0, rows - 1)
    left = mask.index_select(0, index[:pairs])
    right = mask.index_select(0, index[pairs:])
    keep = index[:pairs] != index[pairs:]
    if not bool(keep.any()):
        return float('nan')
    intersection = (left & right).sum(dim=-1).float()
    union = (left | right).sum(dim=-1).float().clamp_min(1.0)
    return float((intersection / union)[keep].mean())


def delta_decomposition(*, delta_total, delta_modulation, delta_state):
    '''Split the cross-step SwiGLU delta into AdaLN and state contributions.'''
    total_energy = float(delta_total.float().pow(2).sum())
    if total_energy <= 0.0:
        return {'total_delta_energy': 0.0}
    residual = delta_total - delta_modulation
    return {
        'total_delta_energy': total_energy,
        'modulation_only_fraction': float(
            delta_modulation.float().pow(2).sum()
        ) / total_energy,
        'state_only_fraction': float(
            delta_state.float().pow(2).sum()
        ) / total_energy,
        'modulation_cosine': cosine(delta_modulation, delta_total),
        'state_cosine': cosine(delta_state, delta_total),
        # What a shared modulation-only correction would leave behind.
        'residual_after_modulation_fraction': float(
            residual.float().pow(2).sum()
        ) / total_energy,
    }


# --------------------------------------------------------------------------
# C. FP8 cache viability
# --------------------------------------------------------------------------

def output_cache_error(y_cached, gate_rows, dtype):
    """Gated relative error of holding a cached MLP output in ``dtype``.

    The cached value is kept exactly during the run, so this is the cost of
    the representation alone rather than a floor baked into the measurement.
    """
    stored = store_roundtrip(y_cached, dtype)
    gated_reference = y_cached.float() * gate_rows.float()
    gated_error = (stored.float() - y_cached.float()) * gate_rows.float()
    return relative_norm(gated_error, gated_reference)


def activation_cache_error(a_current, a_cached, dtype, *, apply_fc2=None):
    """Delta and group-ranking damage from holding the activation in ``dtype``."""
    delta = a_current.float() - a_cached.float()
    stored_delta = a_current.float() - store_roundtrip(a_cached, dtype).float()
    report = {
        'delta_relative_error': summarize(
            relative_norm(stored_delta - delta, delta)
        ),
    }
    exact_groups = group_magnitudes(delta)
    stored_groups = group_magnitudes(stored_delta)
    for k in GROUP_BUDGETS:
        report['group_rank_jaccard_top%d' % k] = jaccard(
            top_k_mask(exact_groups, k),
            top_k_mask(stored_groups, k),
        )
    if apply_fc2 is not None:
        exact_out = apply_fc2(delta.to(a_current.dtype))
        stored_out = apply_fc2(stored_delta.to(a_current.dtype))
        report['fc2_relative_error'] = summarize(
            relative_norm(stored_out - exact_out, exact_out)
        )
    return report


def cache_representation_report(*, y_cached, gate_rows, a_current, a_cached,
                                apply_fc2=None, dtypes=CACHE_DTYPES):
    """Every candidate cache representation measured against exact state."""
    return {
        dtype: {
            'output': summarize(
                output_cache_error(y_cached, gate_rows, dtype)
            ),
            'activation': activation_cache_error(
                a_current,
                a_cached,
                dtype,
                apply_fc2=apply_fc2,
            ),
        }
        for dtype in dtypes
    }
