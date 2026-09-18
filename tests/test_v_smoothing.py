"""Pure CPU tests for H3V-Smooth grouping and mean preprocessing."""

from types import SimpleNamespace
from pathlib import Path
import importlib.util
import sys

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'h3_optimizations' / 'attention' / 'sparse' / 'v_smoothing.py'
)
SPEC = importlib.util.spec_from_file_location('h3_v_smoothing_test_module', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
clear_h3v_grouping_cache = MODULE.clear_h3v_grouping_cache
demean_v_blocks_ = MODULE.demean_v_blocks_
h3v_grouping_refresh_due = MODULE.h3v_grouping_refresh_due
h3v_smooth_active = MODULE.h3v_smooth_active
normalized_v_means = MODULE.normalized_v_means
resolve_h3v_grouping = MODULE.resolve_h3v_grouping
subtract_v_means_ = MODULE.subtract_v_means_


def _snapshot(step, total, request_id=7):
    return SimpleNamespace(
        request_id=request_id,
        step_index=step,
        total_steps=total,
    )


def test_schedule_scales_with_arbitrary_step_counts():
    expected = {
        1: [True],
        2: [True, False],
        4: [True, False, False, False],
        8: [True, True, False, False, False, False, False, False],
        20: [True] * 5 + [False] * 15,
        32: [True] * 8 + [False] * 24,
    }
    for total, mask in expected.items():
        assert [h3v_smooth_active(_snapshot(i, total)) for i in range(total)] == mask


def test_grouping_refresh_uses_four_step_windows_inside_first_quarter():
    snap0 = _snapshot(0, 20)
    assert h3v_grouping_refresh_due(snap0, None)
    assert h3v_grouping_refresh_due(snap0, -1)
    assert not h3v_grouping_refresh_due(_snapshot(1, 20), 0)
    assert not h3v_grouping_refresh_due(_snapshot(3, 20), 0)
    assert h3v_grouping_refresh_due(_snapshot(4, 20), 0)
    assert not h3v_grouping_refresh_due(_snapshot(5, 20), 4)
    assert not h3v_grouping_refresh_due(_snapshot(8, 20), 4)


def test_schedule_rejects_unknown_or_out_of_range_progress():
    assert not h3v_smooth_active(None)
    assert not h3v_smooth_active(_snapshot(-1, 4))
    assert not h3v_smooth_active(_snapshot(4, 4))
    assert not h3v_smooth_active(_snapshot(0, 0))


def _exercise_block_width(block_rows):
    torch.manual_seed(1234 + block_rows)
    original = torch.randn(2, 3, block_rows * 2 + 7, 16, dtype=torch.float32)
    residual = original.clone()
    means = demean_v_blocks_(residual, block_rows)

    assert means.shape == (2, 3, 3, 16)
    for block in range(3):
        start = block * block_rows
        stop = min(start + block_rows, residual.shape[-2])
        assert torch.allclose(
            residual[..., start:stop, :].mean(dim=-2),
            torch.zeros_like(means[..., block, :]),
            atol=2e-7,
            rtol=0,
        )
        reconstructed = residual[..., start:stop, :] + means[..., block, :].unsqueeze(-2)
        assert torch.allclose(reconstructed, original[..., start:stop, :], atol=2e-6, rtol=0)


def test_backend_native_64_row_blocks():
    _exercise_block_width(64)


def test_backend_native_128_row_blocks():
    _exercise_block_width(128)


def test_two_pass_subtraction_matches_retained_v_demeaning():
    block_rows = 64
    torch.manual_seed(42)
    original = torch.randn(1, 2, 192, 8, dtype=torch.float32)
    retained = original.clone()
    means = demean_v_blocks_(retained, block_rows)

    second_pass = original.clone()
    for start in range(0, second_pass.shape[-2], 128):
        chunk = second_pass[..., start:start + 128, :]
        subtract_v_means_(chunk, means, block_rows, start)

    assert torch.equal(second_pass, retained)


def test_normalized_means_are_bf16_and_restore_with_channel_scale():
    means = torch.tensor(
        [[[[1.0, -2.0, 0.5], [3.0, 4.0, -1.5]]]], dtype=torch.float32
    )
    scale = torch.tensor([0.25, 0.5, 0.125], dtype=torch.float32)
    normalized = normalized_v_means(means, scale)
    assert normalized.dtype == torch.bfloat16
    assert normalized.is_contiguous()
    restored = normalized.float() * scale.reshape(1, 1, 1, 3)
    assert torch.allclose(restored, means, atol=1e-3, rtol=1e-3)


class _Module:
    pass


def test_grouping_is_per_head_bijective_and_reused_between_refreshes():
    clear_h3v_grouping_cache()
    heads, sequence, dim = 2, 32, 4
    values = torch.empty(heads, sequence, dim)
    # Four distinct value families, deliberately interleaved in sequence order.
    bases = torch.tensor(
        [[-6.0, 0.0, 0.0, 0.0], [0.0, -6.0, 0.0, 0.0],
         [0.0, 0.0, 6.0, 0.0], [0.0, 0.0, 0.0, 6.0]]
    )
    for row in range(sequence):
        values[0, row] = bases[row % 4]
        values[1, row] = bases[(row * 3) % 4]

    calls = []

    def project(head, rows):
        calls.append((int(head), int(rows.numel())))
        return values[int(head)].index_select(0, rows.to(torch.int64))

    layout = SimpleNamespace(
        seq_len=sequence,
        video_range=(0, sequence),
        video_shape=(2, 4, 4),
    )
    module = _Module()
    plan0 = resolve_h3v_grouping(
        module,
        _snapshot(0, 20),
        layout,
        block_rows=8,
        heads=heads,
        head_dim=dim,
        device=torch.device('cpu'),
        project_v_head_rows=project,
        cluster_rows=8,
    )
    assert plan0.refreshed
    assert plan0.demean
    assert plan0.permutation.shape == (heads, sequence)
    expected = torch.arange(sequence, dtype=torch.int32)
    for head in range(heads):
        assert torch.equal(torch.sort(plan0.permutation[head]).values, expected)

    calls_after_refresh = len(calls)
    plan1 = resolve_h3v_grouping(
        module,
        _snapshot(1, 20),
        layout,
        block_rows=8,
        heads=heads,
        head_dim=dim,
        device=torch.device('cpu'),
        project_v_head_rows=project,
        cluster_rows=8,
    )
    assert not plan1.refreshed
    assert torch.equal(plan1.permutation, plan0.permutation)
    assert len(calls) == calls_after_refresh

    plan4 = resolve_h3v_grouping(
        module,
        _snapshot(4, 20),
        layout,
        block_rows=8,
        heads=heads,
        head_dim=dim,
        device=torch.device('cpu'),
        project_v_head_rows=project,
        cluster_rows=8,
    )
    assert plan4.refreshed
    assert len(calls) > calls_after_refresh

    calls_after_step4 = len(calls)
    plan5 = resolve_h3v_grouping(
        module,
        _snapshot(5, 20),
        layout,
        block_rows=8,
        heads=heads,
        head_dim=dim,
        device=torch.device('cpu'),
        project_v_head_rows=project,
        cluster_rows=8,
    )
    assert not plan5.refreshed
    assert not plan5.demean
    assert torch.equal(plan5.permutation, plan4.permutation)
    assert len(calls) == calls_after_step4



def test_grouping_materially_reduces_block_residual_on_interleaved_value_families():
    clear_h3v_grouping_cache()
    heads, sequence, dim, block_rows = 1, 64, 4, 16
    bases = torch.tensor(
        [[-8.0, 0.0, 0.0, 0.0], [0.0, -8.0, 0.0, 0.0],
         [0.0, 0.0, 8.0, 0.0], [0.0, 0.0, 0.0, 8.0]],
        dtype=torch.float32,
    )
    values = torch.stack([bases[row % 4] for row in range(sequence)])

    def project(_head, rows):
        return values.index_select(0, rows.to(torch.int64))

    layout = SimpleNamespace(
        seq_len=sequence,
        video_range=(0, sequence),
        video_shape=(4, 4, 4),
    )
    plan = resolve_h3v_grouping(
        _Module(), _snapshot(0, 20), layout,
        block_rows=block_rows, heads=heads, head_dim=dim,
        device=torch.device('cpu'), project_v_head_rows=project,
        cluster_rows=16,
    )
    grouped = values.index_select(0, plan.permutation[0].to(torch.int64))
    raw = values.reshape(-1, block_rows, dim)
    packed = grouped.reshape(-1, block_rows, dim)
    raw_residual = raw - raw.mean(dim=1, keepdim=True)
    packed_residual = packed - packed.mean(dim=1, keepdim=True)
    assert packed_residual.square().mean() < raw_residual.square().mean() * 0.25


def test_grouping_leaves_mixed_context_video_tile_outside_permutation():
    clear_h3v_grouping_cache()
    sequence, heads, dim = 26, 1, 2
    values = torch.arange(sequence * dim, dtype=torch.float32).reshape(sequence, dim)

    def project(_head, rows):
        return values.index_select(0, rows.to(torch.int64))

    layout = SimpleNamespace(
        seq_len=sequence,
        video_range=(10, sequence),
        video_shape=(2, 2, 4),
    )
    plan = resolve_h3v_grouping(
        _Module(),
        _snapshot(0, 4),
        layout,
        block_rows=8,
        heads=heads,
        head_dim=dim,
        device=torch.device('cpu'),
        project_v_head_rows=project,
        cluster_rows=4,
    )
    assert plan.group_start == 16
    assert plan.group_stop == 26
    assert torch.equal(
        torch.sort(plan.permutation[0]).values,
        torch.arange(16, 26, dtype=torch.int32),
    )


def test_per_head_kv_permutation_preserves_dense_attention():
    """The same per-head permutation on K and V is exact for dense attention."""
    torch.manual_seed(9876)
    batch, heads, q_rows, kv_rows, dim = 1, 3, 7, 19, 8
    q = torch.randn(batch, heads, q_rows, dim, dtype=torch.float64)
    k = torch.randn(batch, heads, kv_rows, dim, dtype=torch.float64)
    v = torch.randn(batch, heads, kv_rows, dim, dtype=torch.float64)
    scale = dim ** -0.5

    reference = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1) @ v

    k_grouped = torch.empty_like(k)
    v_grouped = torch.empty_like(v)
    for head in range(heads):
        permutation = torch.randperm(kv_rows)
        k_grouped[:, head] = k[:, head].index_select(1, permutation)
        v_grouped[:, head] = v[:, head].index_select(1, permutation)

    grouped = torch.softmax(
        torch.matmul(q, k_grouped.transpose(-2, -1)) * scale, dim=-1
    ) @ v_grouped
    assert torch.allclose(grouped, reference, atol=1e-12, rtol=1e-12)


def test_grouping_all_head_projection_reuses_one_v_projection_per_slab():
    clear_h3v_grouping_cache()
    heads, sequence, dim, block_rows, cluster_rows = 3, 48, 4, 8, 12
    torch.manual_seed(2026)
    values = torch.randn(heads, sequence, dim)
    head_calls = []
    slab_calls = []

    def project_head(head, rows):
        head_calls.append((int(head), int(rows.numel())))
        return values[int(head)].index_select(0, rows.to(torch.int64))

    def project_rows(start, end):
        slab_calls.append((int(start), int(end)))
        return values[:, int(start):int(end), :]

    layout = SimpleNamespace(
        seq_len=sequence,
        video_range=(0, sequence),
        video_shape=(3, 4, 4),
    )
    plan = resolve_h3v_grouping(
        _Module(), _snapshot(0, 20), layout,
        block_rows=block_rows, heads=heads, head_dim=dim,
        device=torch.device('cpu'),
        project_v_head_rows=project_head,
        project_v_rows=project_rows,
        cluster_rows=cluster_rows,
    )
    assert plan.refreshed
    # Initial centroid seeds remain head-specific, but the expensive full scan is
    # one all-head projection per bounded source slab.
    assert len(head_calls) == heads
    assert len(slab_calls) == (sequence + cluster_rows - 1) // cluster_rows
    expected = torch.arange(sequence, dtype=torch.int32)
    for head in range(heads):
        assert torch.equal(torch.sort(plan.permutation[head]).values, expected)
