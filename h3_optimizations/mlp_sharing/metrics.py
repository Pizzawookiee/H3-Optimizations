'''Local-cell geometry, oracle metrics, and merge-fraction summaries.'''

import math

import numpy as np
import torch

from .config import SELECTORS, TARGET_FRACTIONS

PAIR_LEFT = (0, 0, 0, 1, 1, 2)
PAIR_RIGHT = (1, 2, 3, 2, 3, 3)
PAIR_COMPLEMENT = np.asarray((5, 4, 3, 2, 1, 0), dtype=np.int64)

VALUE_COLUMNS = (
    'output_distance',
    'input_cosine_distance',
    'input_l2_distance',
    'representative_residual_error',
    'representative_block_error',
    'mean_left_residual_error',
    'mean_right_residual_error',
    'mean_left_block_error',
    'mean_right_block_error',
)


def spatial_cells(layout, chunk_start, chunk_stop, cell_height, cell_width):
    '''Return chunk-local indices for complete target-video spatial cells.'''

    video_start, video_stop = (int(value) for value in layout.video_range)
    latent_t, patch_h, patch_w = (
        int(value) for value in layout.video_shape
    )
    chunk_start = int(chunk_start)
    chunk_stop = int(chunk_stop)
    cell_height = int(cell_height)
    cell_width = int(cell_width)
    if cell_height < 1 or cell_width < 1:
        raise ValueError('cell dimensions must be positive')
    if chunk_stop <= video_start or chunk_start >= video_stop:
        return (), video_stop - video_start

    frame_rows = patch_h * patch_w
    cells = []
    for t in range(latent_t):
        frame_start = video_start + t * frame_rows
        if frame_start >= chunk_stop or frame_start + frame_rows <= chunk_start:
            continue
        for y in range(0, patch_h - cell_height + 1, cell_height):
            row_start = frame_start + y * patch_w
            for x in range(0, patch_w - cell_width + 1, cell_width):
                first = row_start + x
                indices = tuple(
                    first + row * patch_w + column
                    for row in range(cell_height)
                    for column in range(cell_width)
                )
                if all(chunk_start <= index < chunk_stop for index in indices):
                    cells.append(tuple(index - chunk_start for index in indices))
    return tuple(cells), video_stop - video_start


def spatial_cells_1x2x2(layout, chunk_start, chunk_stop):
    '''Return chunk-local indices for complete target-video 1T x 2Y x 2X cells.'''

    return spatial_cells(layout, chunk_start, chunk_stop, 2, 2)


def spatial_cells_1x2x4(layout, chunk_start, chunk_stop):
    '''Return chunk-local indices for complete target-video 1T x 2Y x 4X cells.'''

    return spatial_cells(layout, chunk_start, chunk_stop, 2, 4)


def filter_cells_by_modulation(cells, selector, device):
    if not cells:
        return torch.empty((0, 0), dtype=torch.long, device=device)
    cell_indices = torch.tensor(cells, dtype=torch.long, device=device)
    if torch.is_tensor(selector):
        selected = selector.index_select(0, cell_indices.reshape(-1))
        selected = selected.reshape(cell_indices.shape)
        cell_indices = cell_indices[(selected == selected[:, :1]).all(dim=1)]
    return cell_indices


def _relative_norm(delta, reference):
    delta_norm = torch.linalg.vector_norm(delta.float(), dim=-1)
    reference_norm = torch.linalg.vector_norm(reference.float(), dim=-1)
    return delta_norm / reference_norm.clamp_min(1.0e-12)


def _resolved_cell_gates(gate, selector, cell_indices, dtype):
    if torch.is_tensor(selector):
        selected = selector.index_select(0, cell_indices.reshape(-1))
        selected = selected.reshape(cell_indices.shape)
        same_modulation = (selected == selected[:, :1]).all(dim=1)
        cell_indices = cell_indices[same_modulation]
        selected = selected[same_modulation]
        gates = gate.index_select(0, selected.reshape(-1).long())
        gates = gates.reshape(*selected.shape, gate.shape[-1]).to(dtype)
        return cell_indices, gates

    gates = gate[int(selector)].to(dtype)
    return cell_indices, gates


def measure_chunk(
    *,
    h,
    y,
    residual,
    gate,
    selector,
    cells,
    evaluate_mlp,
    include_mean_input,
    mean_batch_rows,
):
    '''Measure all six candidate edges without modifying model tensors.'''

    if not cells:
        return None
    cell_indices = torch.tensor(cells, dtype=torch.long, device=h.device)
    cell_indices, cell_gates = _resolved_cell_gates(
        gate,
        selector,
        cell_indices,
        y.dtype,
    )
    if not int(cell_indices.shape[0]):
        return None

    flat = cell_indices.reshape(-1)
    hidden = int(h.shape[-1])
    h_cells = h.index_select(0, flat).reshape(-1, 4, hidden)
    y_cells = y.index_select(0, flat).reshape(-1, 4, hidden)
    x_cells = residual.index_select(0, flat).reshape(-1, 4, hidden)

    left = torch.tensor(PAIR_LEFT, dtype=torch.long, device=h.device)
    right = torch.tensor(PAIR_RIGHT, dtype=torch.long, device=h.device)
    h_float = h_cells.float()
    y_float = y_cells.float()
    x_float = x_cells.float()
    h_left = h_float.index_select(1, left)
    h_right = h_float.index_select(1, right)
    y_left = y_float.index_select(1, left)
    y_right = y_float.index_select(1, right)

    h_left_norm = torch.linalg.vector_norm(h_left, dim=-1)
    h_right_norm = torch.linalg.vector_norm(h_right, dim=-1)
    cosine = 1.0 - (h_left * h_right).sum(dim=-1) / (
        h_left_norm * h_right_norm
    ).clamp_min(1.0e-12)
    input_l2 = torch.linalg.vector_norm(h_left - h_right, dim=-1) / (
        0.5 * (h_left_norm + h_right_norm)
    ).clamp_min(1.0e-12)

    y_left_norm = torch.linalg.vector_norm(y_left, dim=-1)
    y_right_norm = torch.linalg.vector_norm(y_right, dim=-1)
    output_distance = torch.linalg.vector_norm(
        y_left - y_right,
        dim=-1,
    ) / (0.5 * (y_left_norm + y_right_norm)).clamp_min(1.0e-12)

    if cell_gates.ndim == 1:
        gate_left = gate_right = cell_gates.float()
    else:
        gate_left = cell_gates.float().index_select(1, left)
        gate_right = cell_gates.float().index_select(1, right)

    representative_delta = (y_left - y_right) * gate_right
    representative_update = y_right * gate_right
    representative_residual_error = _relative_norm(
        representative_delta,
        representative_update,
    )
    representative_block_error = _relative_norm(
        representative_delta,
        x_float.index_select(1, right) + representative_update,
    )

    shape = output_distance.shape
    nan = torch.full(shape, float('nan'), device=h.device, dtype=torch.float32)
    mean_left_residual_error = nan
    mean_right_residual_error = nan
    mean_left_block_error = nan
    mean_right_block_error = nan
    if include_mean_input:
        mean_h = (
            h_cells.index_select(1, left) + h_cells.index_select(1, right)
        ).mul(0.5).reshape(-1, hidden)
        mean_parts = []
        for start in range(0, int(mean_h.shape[0]), int(mean_batch_rows)):
            mean_parts.append(evaluate_mlp(mean_h[start:start + mean_batch_rows]))
        mean_y = torch.cat(mean_parts, dim=0).reshape(
            h_cells.shape[0],
            len(PAIR_LEFT),
            hidden,
        ).float()
        mean_left_delta = (mean_y - y_left) * gate_left
        mean_right_delta = (mean_y - y_right) * gate_right
        mean_left_update = y_left * gate_left
        mean_right_update = y_right * gate_right
        mean_left_residual_error = _relative_norm(
            mean_left_delta,
            mean_left_update,
        )
        mean_right_residual_error = _relative_norm(
            mean_right_delta,
            mean_right_update,
        )
        mean_left_block_error = _relative_norm(
            mean_left_delta,
            x_float.index_select(1, left) + mean_left_update,
        )
        mean_right_block_error = _relative_norm(
            mean_right_delta,
            x_float.index_select(1, right) + mean_right_update,
        )

    values = torch.stack(
        (
            output_distance,
            cosine,
            input_l2,
            representative_residual_error,
            representative_block_error,
            mean_left_residual_error,
            mean_right_residual_error,
            mean_left_block_error,
            mean_right_block_error,
        ),
        dim=-1,
    )
    return values


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {'median': None, 'p95': None, 'p99': None}
    median, p95, p99 = np.quantile(values, (0.5, 0.95, 0.99))
    return {
        'median': float(median),
        'p95': float(p95),
        'p99': float(p99),
    }


def _pearson(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def summarize_measurements(
    values,
    *,
    request_id,
    step_index,
    total_steps,
    sigma,
    layer_index,
    evaluation_index,
    total_video_tokens,
):
    '''Build selector curves at 5%-50% requested row reduction.'''

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (6, len(VALUE_COLUMNS)):
        raise ValueError('values must have shape [cells, 6, %d]' % len(VALUE_COLUMNS))
    cells = int(values.shape[0])
    if not cells:
        return []

    random_seed = (
        (int(request_id) + 1) * 1_000_003
        + (int(step_index) + 1) * 10_007
        + (int(layer_index) + 1) * 101
        + int(evaluation_index)
    ) & 0xFFFFFFFF
    random_scores = np.random.default_rng(random_seed).random((cells, 6))
    score_by_selector = {
        'output_oracle': values[:, :, 0],
        'input_cosine': values[:, :, 1],
        'input_l2': values[:, :, 2],
        'random_local': random_scores,
    }
    correlations = {
        'input_cosine_vs_output': _pearson(values[:, :, 1], values[:, :, 0]),
        'input_l2_vs_output': _pearson(values[:, :, 2], values[:, :, 0]),
    }
    cell_rows = np.arange(cells, dtype=np.int64)
    rows = []
    for selector_name in SELECTORS:
        scores = score_by_selector[selector_name]
        first_edges = np.argmin(scores, axis=1)
        second_edges = PAIR_COMPLEMENT[first_edges]
        candidate_cells = np.concatenate((cell_rows, cell_rows))
        candidate_edges = np.concatenate((first_edges, second_edges))
        candidate_rank = np.concatenate((np.zeros(cells), np.ones(cells)))
        candidate_scores = scores[candidate_cells, candidate_edges]
        order = np.lexsort((candidate_rank, candidate_scores))
        maximum_pairs = int(order.size)

        for target_fraction in TARGET_FRACTIONS:
            requested_pairs = int(math.ceil(float(target_fraction) * total_video_tokens))
            selected_count = min(requested_pairs, maximum_pairs)
            if selected_count <= 0:
                continue
            selected = order[:selected_count]
            selected_cells = candidate_cells[selected]
            selected_edges = candidate_edges[selected]
            threshold = float(candidate_scores[selected].max())
            merge_fraction = float(selected_count) / float(total_video_tokens)

            output_distance = values[selected_cells, selected_edges, 0]
            input_cosine = values[selected_cells, selected_edges, 1]
            input_l2 = values[selected_cells, selected_edges, 2]
            common = {
                'request_id': int(request_id),
                'step_index': int(step_index),
                'total_steps': int(total_steps),
                'sigma': None if sigma is None else float(sigma),
                'layer': int(layer_index),
                'evaluation_index': int(evaluation_index),
                'selector': selector_name,
                'target_merge_fraction': float(target_fraction),
                'merge_fraction': merge_fraction,
                'selector_distance_threshold': threshold,
                'selected_pairs': int(selected_count),
                'candidate_cells': cells,
                'total_video_tokens': int(total_video_tokens),
                'eligible_video_fraction': min(
                    1.0,
                    float(cells * 4) / float(total_video_tokens),
                ),
                'selected_output_distance': _quantiles(output_distance),
                'selected_input_cosine': _quantiles(input_cosine),
                'selected_input_l2': _quantiles(input_l2),
                'correlations': correlations,
            }

            representative = dict(common)
            representative.update(
                reconstruction='representative',
                changed_tokens=int(selected_count),
                residual_error=_quantiles(
                    values[selected_cells, selected_edges, 3]
                ),
                block_error=_quantiles(
                    values[selected_cells, selected_edges, 4]
                ),
            )
            rows.append(representative)

            mean_residual = values[selected_cells, selected_edges, 5:7].reshape(-1)
            if np.isfinite(mean_residual).any():
                mean_input = dict(common)
                mean_input.update(
                    reconstruction='mean_input',
                    changed_tokens=int(selected_count * 2),
                    residual_error=_quantiles(mean_residual),
                    block_error=_quantiles(
                        values[selected_cells, selected_edges, 7:9].reshape(-1)
                    ),
                )
                rows.append(mean_input)
    return rows
