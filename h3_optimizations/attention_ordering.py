'''Post-RoPE target-video token ordering experiment for H3 attention.'''

from __future__ import annotations

import math

import torch


Q_TILE = 128
KV_TILE = 64
ORDERINGS = ('native', 'row_major', 'time_major', 'morton', 'hilbert')


def _shape(video_shape):
    shape = tuple(int(value) for value in video_shape)
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError('video_shape must contain positive T, H, and W values')
    return shape


def _native_index(t, y, x, height, width):
    return (int(t) * int(height) + int(y)) * int(width) + int(x)


def _hilbert_point(distance, bits, dimensions=3):
    bit_string = format(int(distance), '0%db' % (bits * dimensions))
    point = [int(bit_string[index::dimensions], 2) for index in range(dimensions)]
    limit = 2 << (bits - 1)
    value = point[dimensions - 1] >> 1
    for index in range(dimensions - 1, 0, -1):
        point[index] ^= point[index - 1]
    point[0] ^= value

    mask = 2
    while mask != limit:
        lower = mask - 1
        for index in range(dimensions - 1, -1, -1):
            if point[index] & mask:
                point[0] ^= lower
            else:
                value = (point[0] ^ point[index]) & lower
                point[0] ^= value
                point[index] ^= value
        mask <<= 1
    return tuple(point)


def _morton_code(t, y, x, bits):
    code = 0
    for bit in range(bits):
        code |= ((int(t) >> bit) & 1) << (3 * bit + 2)
        code |= ((int(y) >> bit) & 1) << (3 * bit + 1)
        code |= ((int(x) >> bit) & 1) << (3 * bit)
    return code


def video_permutation(video_shape, ordering):
    '''Return native video-row indices in the requested traversal order.'''
    t_count, height, width = _shape(video_shape)
    ordering = str(ordering).strip().lower()
    if ordering not in ORDERINGS:
        raise ValueError('unknown token ordering %r' % ordering)
    tokens = t_count * height * width
    if ordering in ('native', 'row_major'):
        return tuple(range(tokens))
    if ordering == 'time_major':
        return tuple(
            _native_index(t, y, x, height, width)
            for y in range(height)
            for x in range(width)
            for t in range(t_count)
        )

    side = 1
    while side < max(t_count, height, width):
        side <<= 1
    bits = max(1, side.bit_length() - 1)
    if ordering == 'morton':
        coordinates = (
            (_morton_code(t, y, x, bits), _native_index(t, y, x, height, width))
            for t in range(t_count)
            for y in range(height)
            for x in range(width)
        )
        return tuple(index for _code, index in sorted(coordinates))

    traversal = []
    for distance in range(side ** 3):
        t, y, x = _hilbert_point(distance, bits)
        if t < t_count and y < height and x < width:
            traversal.append(_native_index(t, y, x, height, width))
    if len(traversal) != tokens or len(set(traversal)) != tokens:
        raise RuntimeError('Hilbert traversal did not cover the video lattice exactly')
    return tuple(traversal)


def inverse_permutation(permutation):
    inverse = [0] * len(permutation)
    for ordered, native in enumerate(permutation):
        inverse[int(native)] = int(ordered)
    return tuple(inverse)


def packed_permutation(layout, ordering):
    '''Keep packed context fixed and reorder only the final target-video span.'''
    video_start, video_stop = (int(value) for value in layout.video_range)
    sequence = int(layout.seq_len)
    if video_stop != sequence:
        raise ValueError('the ordering experiment requires target video to be final')
    video = video_permutation(layout.video_shape, ordering)
    if len(video) != video_stop - video_start:
        raise ValueError('video lattice size does not match the packed video span')
    return tuple(range(video_start)) + tuple(video_start + index for index in video)


def apply_permutation(x, permutation):
    index = torch.tensor(permutation, dtype=torch.long, device=x.device)
    return x.index_select(-2, index)


def restore_permutation(x, permutation):
    return apply_permutation(x, inverse_permutation(permutation))


def _geometry(sequence, video_start, q_tile, kv_tile):
    return {
        'sequence': int(sequence),
        'q_tile': int(q_tile),
        'kv_tile': int(kv_tile),
        'q_tiles': math.ceil(int(sequence) / int(q_tile)),
        'kv_tiles': math.ceil(int(sequence) / int(kv_tile)),
        'pure_q_start': math.ceil(int(video_start) / int(q_tile)),
        'pure_kv_start': math.ceil(int(video_start) / int(kv_tile)),
    }


def _mean_pool(x, block):
    sequence = int(x.shape[-2])
    full = sequence // int(block)
    remainder = sequence % int(block)
    pieces = []
    if full:
        pieces.append(
            x[..., :full * block, :]
            .reshape(*x.shape[:-2], full, block, x.shape[-1])
            .mean(dim=-2, dtype=torch.float32)
        )
    if remainder:
        pieces.append(x[..., full * block:, :].mean(dim=-2, keepdim=True, dtype=torch.float32))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)


def _block_locality(x, start, block):
    similarities = []
    variances = []
    for row in range(int(start), int(x.shape[-2]), int(block)):
        value = x[..., row:min(row + block, x.shape[-2]), :].float()
        center = value.mean(dim=-2, keepdim=True)
        similarity = torch.nn.functional.cosine_similarity(value, center, dim=-1).mean(dim=-1)
        variance = (value - center).square().mean(dim=(-2, -1))
        similarities.append(similarity.detach().cpu())
        variances.append(variance.detach().cpu())
    return torch.cat(similarities), torch.cat(variances)


def _sample_queries(layout, permutations, count, q_tile):
    video_start, video_stop = (int(value) for value in layout.video_range)
    pure_start = math.ceil(video_start / int(q_tile)) * int(q_tile)
    eligible = []
    inverses = {
        name: inverse_permutation(permutation)
        for name, permutation in permutations.items()
    }
    for native in range(video_start, video_stop):
        if all(inverse[native] >= pure_start for inverse in inverses.values()):
            eligible.append(native)
    if not eligible:
        raise ValueError('no target-video queries are pure Q-tile rows in every ordering')
    count = min(max(1, int(count)), len(eligible))
    if count == 1:
        return (eligible[len(eligible) // 2],), inverses
    positions = [index * (len(eligible) - 1) // (count - 1) for index in range(count)]
    return tuple(eligible[index] for index in positions), inverses


def _budget(value, pure_kv_tiles):
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError('video budgets must lie in (0, 1]')
    return min(int(pure_kv_tiles), max(1, math.ceil(value * pure_kv_tiles)))


def _distribution(values):
    values = torch.cat(values).float()
    return {
        'mean': float(values.mean()),
        'minimum': float(values.min()),
        'median': float(values.median()),
    }


def analyze_orderings(
    q,
    k,
    v,
    layout,
    *,
    budgets=(0.2, 0.3, 0.5),
    query_samples=64,
    q_tile=Q_TILE,
    kv_tile=KV_TILE,
    head_chunk=2,
    permutations=None,
):
    '''Compare fixed-density routes on identical post-RoPE H3 Q/K/V.'''
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1:
        raise ValueError('expected matching Q/K/V shaped [1, heads, sequence, dim]')
    if int(q.shape[-2]) != int(layout.seq_len):
        raise ValueError('Q/K/V sequence does not match the packed layout')
    if q.device != k.device or q.device != v.device:
        raise ValueError('Q/K/V devices differ')

    budgets = tuple(sorted(set(float(value) for value in budgets)))
    if not budgets:
        raise ValueError('at least one video budget is required')
    if permutations is None:
        permutations = {
            name: packed_permutation(layout, name)
            for name in ORDERINGS
        }
    elif set(permutations) != set(ORDERINGS):
        raise ValueError('permutations must contain every ordering arm')
    query_native, inverses = _sample_queries(
        layout,
        permutations,
        query_samples,
        q_tile,
    )
    video_start = int(layout.video_range[0])
    geometry = _geometry(layout.seq_len, video_start, q_tile, kv_tile)
    pure_q_tiles = geometry['q_tiles'] - geometry['pure_q_start']
    pure_kv_tiles = geometry['kv_tiles'] - geometry['pure_kv_start']
    if pure_q_tiles <= 0 or pure_kv_tiles <= 0:
        raise ValueError('packed layout has no pure target-video tiles')

    results = {}
    heads = int(q.shape[1])
    head_chunk = max(1, int(head_chunk))
    scale = float(q.shape[-1]) ** -0.5
    kv_token_tiles = torch.div(
        torch.arange(layout.seq_len, device=q.device),
        int(kv_tile),
        rounding_mode='floor',
    )
    for name in ORDERINGS:
        if name == 'row_major':
            results[name] = results['native']
            continue
        permutation = permutations[name]
        order = torch.tensor(permutation, dtype=torch.long, device=q.device)
        ordered_queries = torch.tensor(
            [inverses[name][native] for native in query_native],
            dtype=torch.long,
            device=q.device,
        )
        q_rows = torch.div(ordered_queries, int(q_tile), rounding_mode='floor') - geometry['pure_q_start']
        locality_q = []
        locality_k = []
        variance_q = []
        variance_k = []
        accum = {
            budget: {
                'l1_numerator': 0.0,
                'l1_denominator': 0.0,
                'l2_numerator': 0.0,
                'l2_denominator': 0.0,
                'retained': [],
            }
            for budget in budgets
        }

        for h0 in range(0, heads, head_chunk):
            h1 = min(heads, h0 + head_chunk)
            q_ordered = q[0, h0:h1].index_select(1, order)
            k_ordered = k[0, h0:h1].index_select(1, order)
            v_ordered = v[0, h0:h1].index_select(1, order)
            q_summary = _mean_pool(q_ordered, q_tile)
            k_summary = _mean_pool(k_ordered, kv_tile)
            route_scores = torch.matmul(
                q_summary[:, geometry['pure_q_start']:, :],
                k_summary[:, geometry['pure_kv_start']:, :].transpose(-1, -2),
            )
            q_similarity, q_variance = _block_locality(
                q_ordered,
                geometry['pure_q_start'] * int(q_tile),
                q_tile,
            )
            k_similarity, k_variance = _block_locality(
                k_ordered,
                geometry['pure_kv_start'] * int(kv_tile),
                kv_tile,
            )
            locality_q.append(q_similarity)
            locality_k.append(k_similarity)
            variance_q.append(q_variance)
            variance_k.append(k_variance)

            selected_q = q_ordered.index_select(1, ordered_queries).float()
            scores = torch.matmul(selected_q, k_ordered.float().transpose(-1, -2)) * scale
            dense_prob = torch.softmax(scores, dim=-1)
            dense_output = torch.matmul(dense_prob, v_ordered.float())
            for budget in budgets:
                retained_tiles = _budget(budget, pure_kv_tiles)
                selected = torch.topk(route_scores, retained_tiles, dim=-1).indices
                routed = selected.index_select(1, q_rows) + geometry['pure_kv_start']
                tile_keep = torch.zeros(
                    h1 - h0,
                    len(query_native),
                    geometry['kv_tiles'],
                    dtype=torch.bool,
                    device=q.device,
                )
                tile_keep[..., :geometry['pure_kv_start']] = True
                tile_keep.scatter_(2, routed, True)
                keep = tile_keep.index_select(2, kv_token_tiles)
                retained_mass = (dense_prob * keep).sum(dim=-1)
                sparse_prob = torch.softmax(scores.masked_fill(~keep, -torch.inf), dim=-1)
                sparse_output = torch.matmul(sparse_prob, v_ordered.float())
                difference = sparse_output - dense_output
                bucket = accum[budget]
                bucket['l1_numerator'] += float(difference.abs().sum())
                bucket['l1_denominator'] += float(dense_output.abs().sum())
                bucket['l2_numerator'] += float(difference.square().sum())
                bucket['l2_denominator'] += float(dense_output.square().sum())
                bucket['retained'].append(retained_mass.detach().cpu().flatten())
            del scores, dense_prob, dense_output, q_ordered, k_ordered, v_ordered

        rows = []
        for budget in budgets:
            retained_tiles = _budget(budget, pure_kv_tiles)
            bucket = accum[budget]
            rows.append(
                {
                    'video_budget': float(budget),
                    'retained_video_kv_tiles': int(retained_tiles),
                    'pure_video_kv_tiles': int(pure_kv_tiles),
                    'actual_video_tile_density': float(retained_tiles / pure_kv_tiles),
                    'retained_dense_attention_mass': _distribution(bucket['retained']),
                    'relative_l1_error': float(
                        bucket['l1_numerator'] / max(bucket['l1_denominator'], 1.0e-12)
                    ),
                    'relative_l2_error': float(
                        math.sqrt(bucket['l2_numerator'] / max(bucket['l2_denominator'], 1.0e-12))
                    ),
                }
            )
        results[name] = {
            'q_block_cosine_similarity': float(torch.cat(locality_q).mean()),
            'k_block_cosine_similarity': float(torch.cat(locality_k).mean()),
            'q_block_variance': float(torch.cat(variance_q).mean()),
            'k_block_variance': float(torch.cat(variance_k).mean()),
            'budgets': rows,
        }

    return {
        'post_rope': True,
        'visual_tokens_only': True,
        'output_alignment': 'native query indices via inverse permutation',
        'q_tile': int(q_tile),
        'kv_tile': int(kv_tile),
        'video_shape': [int(value) for value in layout.video_shape],
        'video_range': [int(value) for value in layout.video_range],
        'query_samples': len(query_native),
        'query_native_indices': list(query_native),
        'orderings': results,
    }
