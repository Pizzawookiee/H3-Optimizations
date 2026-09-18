'''H3V-Smooth: value-guided K/V grouping plus block demeaning for Sparse Kitchen.

The VC-Attention V-Smooth paper groups value tokens independently for every
batch/head, sorts by cluster label, applies the same permutation to K and V,
then subtracts one mean per hardware V block before low-bit quantization.  The
mean is restored inside online attention.

H3 Sparse Kitchen has one additional constraint: its router selects physical KV
blocks.  H3V-Smooth therefore establishes the grouped K/V physical layout
*before* K summaries are built and before sparse routing runs.  The non-video
prefix stays in stock order; only the pure-video tail (the region that H3 may
sparsify) is grouped.  Q stays in stock order.

To keep the low-VRAM streamed path, grouping never materializes a full BF16 V
or a reordered BF16 K/V carrier.  It projects one V head in bounded row chunks,
keeps only centroids/labels, stores the resulting permutation on CPU between
steps, and lets the producer re-project K/V directly in grouped order.
'''

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import weakref

import torch


V_SMOOTH_ACTIVE_FRACTION = 0.25
V_SMOOTH_REUSE_STEPS = 4
# Bounded assignment slab.  Distances are [rows, clusters] FP32, so keeping this
# well below the normal QKV chunk size avoids replacing QKV intermediates with a
# large clustering intermediate.
V_SMOOTH_CLUSTER_ROWS = 1024


@dataclass(frozen=True)
class H3VGroupingPlan:
    '''CPU-resident per-head physical-row permutation for one attention layer.'''

    permutation: torch.Tensor  # [heads, grouped_rows], absolute int32 rows, CPU
    group_start: int
    group_stop: int
    refreshed: bool
    demean: bool
    refresh_step: int

    @property
    def grouped_rows(self):
        return int(self.group_stop - self.group_start)

    def to_device(self, device):
        return self.permutation.to(device=device, dtype=torch.int32)


@dataclass
class _GroupingState:
    signature: tuple
    centroids: torch.Tensor  # [heads, clusters, dim], FP32 CPU
    permutation: torch.Tensor  # [heads, rows], absolute int32 CPU
    refresh_step: int


_CACHE = weakref.WeakKeyDictionary()
_CACHE_LOCK = threading.RLock()


def clear_h3v_grouping_cache():
    '''Drop cached centroids/permutations (mainly useful for tests).'''
    with _CACHE_LOCK:
        _CACHE.clear()


def h3v_smooth_active(snapshot) -> bool:
    '''Whether grouping-window demeaning is active at this denoising step.'''
    if snapshot is None:
        return False
    step_index = int(getattr(snapshot, 'step_index', -1))
    total_steps = int(getattr(snapshot, 'total_steps', 0))
    if step_index < 0 or total_steps <= 0 or step_index >= total_steps:
        return False
    active_steps = max(1, int(math.ceil(total_steps * V_SMOOTH_ACTIVE_FRACTION)))
    return step_index < active_steps


def h3v_grouping_refresh_due(snapshot, previous_refresh_step=None) -> bool:
    '''Paper schedule: refresh in the first quarter, reuse for four steps.

    A missing cached layout must be established even if execution first reaches
    this layer outside the nominal grouping window.
    '''
    if snapshot is None:
        return previous_refresh_step is None
    step_index = int(getattr(snapshot, 'step_index', -1))
    total_steps = int(getattr(snapshot, 'total_steps', 0))
    if step_index < 0 or total_steps <= 0:
        return previous_refresh_step is None
    if previous_refresh_step is None:
        return True
    return bool(
        h3v_smooth_active(snapshot)
        and step_index % V_SMOOTH_REUSE_STEPS == 0
        and step_index != int(previous_refresh_step)
    )


def _validate_v(v, block_rows):
    if v.ndim != 4:
        raise ValueError('H3V-Smooth expects V as [batch, heads, rows, dim]')
    block_rows = int(block_rows)
    if block_rows <= 0:
        raise ValueError('H3V-Smooth block_rows must be positive')
    return block_rows


def demean_v_blocks_(v, block_rows):
    '''Subtract one FP32 mean per physical KV block, in place.'''
    block_rows = _validate_v(v, block_rows)
    sequence = int(v.shape[-2])
    full_blocks = sequence // block_rows
    remainder = sequence % block_rows
    means = []

    if full_blocks:
        prefix = v[..., : full_blocks * block_rows, :]
        shaped = prefix.reshape(
            *v.shape[:-2], full_blocks, block_rows, int(v.shape[-1])
        )
        mean = shaped.to(torch.float32).mean(dim=-2)
        shaped.sub_(mean.to(v.dtype).unsqueeze(-2))
        means.append(mean)

    if remainder:
        tail = v[..., full_blocks * block_rows :, :]
        mean = tail.to(torch.float32).mean(dim=-2, keepdim=True)
        tail.sub_(mean.to(v.dtype))
        means.append(mean)

    if not means:
        raise ValueError('H3V-Smooth cannot process an empty V sequence')
    return torch.cat(means, dim=-2).contiguous()


def subtract_v_means_(v, means, block_rows, row_start):
    '''Subtract previously measured block means from one aligned V chunk.'''
    block_rows = _validate_v(v, block_rows)
    row_start = int(row_start)
    if row_start < 0 or row_start % block_rows:
        raise ValueError(
            'H3V-Smooth V chunks must start on a backend KV-tile boundary'
        )
    sequence_rows = int(v.shape[-2])
    first_block = row_start // block_rows
    block_count = (sequence_rows + block_rows - 1) // block_rows
    selected = means[..., first_block : first_block + block_count, :]
    if int(selected.shape[-2]) != block_count:
        raise ValueError('H3V-Smooth mean carrier does not cover the V chunk')

    for local_block in range(block_count):
        start = local_block * block_rows
        stop = min(start + block_rows, sequence_rows)
        v[..., start:stop, :].sub_(
            selected[..., local_block, :].to(v.dtype).unsqueeze(-2)
        )
    return v


def normalized_v_means(means, v_scale):
    '''Store means divided by the INT8 per-channel V scale, as BF16.'''
    if means.ndim != 4:
        raise ValueError('H3V-Smooth means must be [batch, heads, blocks, dim]')
    batch, heads, _blocks, dim = (int(value) for value in means.shape)
    expected = batch * heads * dim
    if int(v_scale.numel()) != expected:
        raise ValueError(
            'H3V-Smooth V scale shape does not match block-mean channels'
        )
    scale = v_scale.reshape(batch, heads, dim).to(torch.float32).unsqueeze(-2)
    return (means.to(torch.float32) / scale).to(torch.bfloat16).contiguous()


def _group_bounds(layout, block_rows):
    sequence = int(layout.seq_len)
    video_start, video_stop = (int(value) for value in layout.video_range)
    if video_stop != sequence:
        raise ValueError('H3V-Smooth requires target video to be the final segment')
    # Match SparseTileRouter.pure_video_kv_start: the mixed prefix/video tile is
    # left untouched and remains in the always-dense context region.
    group_start = ((video_start + block_rows - 1) // block_rows) * block_rows
    return min(group_start, video_stop), video_stop


def _state_signature(snapshot, layout, block_rows, heads, head_dim):
    group_start, group_stop = _group_bounds(layout, block_rows)
    return (
        int(getattr(snapshot, 'request_id', -1)),
        int(getattr(snapshot, 'total_steps', 0)),
        int(layout.seq_len),
        tuple(int(value) for value in layout.video_range),
        tuple(int(value) for value in layout.video_shape),
        int(block_rows),
        int(heads),
        int(head_dim),
        int(group_start),
        int(group_stop),
    )


def _initial_centroid_rows(group_start, group_stop, clusters, device, head=0):
    rows = int(group_stop - group_start)
    if clusters <= 1:
        return torch.tensor([group_start], device=device, dtype=torch.int32)
    # Deterministic low-discrepancy seeds.  A stride coprime with the row count
    # visits unique sequence locations while avoiding the aliasing that evenly
    # spaced seeds can have with repeating spatial/value patterns.
    stride = max(1, rows // clusters)
    while math.gcd(stride, rows) != 1:
        stride += 1
    offset = (int(head) * 17 + clusters * 13) % rows
    positions = (
        offset
        + torch.arange(clusters, device=device, dtype=torch.int64) * stride
    ) % rows
    return (positions + group_start).to(torch.int32)


def _nearest_labels(values, centroids):
    '''Nearest-centroid labels with one bounded FP32 distance slab.'''
    x = values.to(torch.float32)
    c = centroids.to(torch.float32)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    c2 = (c * c).sum(dim=-1).unsqueeze(0)
    distances = x2 + c2 - 2.0 * torch.matmul(x, c.transpose(0, 1))
    labels = torch.argmin(distances, dim=-1)
    del distances, x2, c2, x
    return labels


def _online_kmeans_heads(
    *,
    heads,
    group_start,
    group_stop,
    clusters,
    head_dim,
    device,
    project_v_head_rows,
    project_v_rows,
    warm_centroids=None,
    cluster_rows=V_SMOOTH_CLUSTER_ROWS,
):
    """Low-memory per-head k-means with one all-head V projection per slab.

    Clustering remains independent for every attention head.  Only projection is
    shared: contiguous source rows are projected to all V heads once, then each
    head's bounded [rows, clusters] distance matrix is processed sequentially.
    This preserves the faithful per-head assignments while removing H separate
    V GEMMs for every clustering slab.
    """
    rows = int(group_stop - group_start)
    heads = int(heads)
    cluster_rows = max(1, int(cluster_rows))

    centroids = torch.empty(
        heads, clusters, head_dim, dtype=torch.float32, device=device
    )
    if warm_centroids is None:
        # Seeds are head-specific, so the inexpensive initialization remains
        # arbitrary-row.  The expensive full scan below is shared across heads.
        for head in range(heads):
            seed_rows = _initial_centroid_rows(
                group_start, group_stop, clusters, device, head=head
            )
            centroids[head].copy_(
                project_v_head_rows(head, seed_rows).to(torch.float32)
            )
    else:
        if tuple(warm_centroids.shape) != (heads, clusters, head_dim):
            raise ValueError('cached H3V-Smooth centroid geometry changed')
        centroids.copy_(warm_centroids.to(device=device, dtype=torch.float32))

    sums = torch.zeros_like(centroids)
    counts = torch.zeros(heads, clusters, device=device, dtype=torch.int64)
    labels_cpu = torch.empty(heads, rows, dtype=torch.int32, device='cpu')

    for relative_start in range(0, rows, cluster_rows):
        relative_stop = min(relative_start + cluster_rows, rows)
        absolute_start = group_start + relative_start
        absolute_stop = group_start + relative_stop
        values_all = project_v_rows(absolute_start, absolute_stop)
        if values_all.ndim == 4:
            if int(values_all.shape[0]) != 1:
                raise ValueError('H3V-Smooth all-head V projection batch must be 1')
            values_all = values_all[0]
        expected = (heads, relative_stop - relative_start, head_dim)
        if tuple(values_all.shape) != expected:
            raise ValueError(
                'H3V-Smooth all-head V projection has shape %s, expected %s'
                % (tuple(values_all.shape), expected)
            )

        # Deliberately process heads one at a time so the FP32 distance workspace
        # remains [cluster_rows, clusters], not [heads, cluster_rows, clusters].
        for head in range(heads):
            values = values_all[head]
            labels = _nearest_labels(values, centroids[head])
            values_f = values.to(torch.float32)
            sums[head].index_add_(0, labels, values_f)
            counts[head].index_add_(
                0, labels, torch.ones_like(labels, dtype=torch.int64)
            )
            nonzero = counts[head] > 0
            centroids[head, nonzero] = (
                sums[head, nonzero] / counts[head, nonzero].unsqueeze(-1)
            )
            labels_cpu[head, relative_start:relative_stop].copy_(
                labels.to(device='cpu', dtype=torch.int32)
            )
            del values_f, labels, nonzero
        del values_all

    permutation_cpu = torch.empty(
        heads, rows, dtype=torch.int32, device='cpu'
    )
    for head in range(heads):
        permutation_cpu[head].copy_(
            torch.argsort(labels_cpu[head], stable=True).to(torch.int32)
            + int(group_start)
        )
    return centroids.to(device='cpu', dtype=torch.float32), permutation_cpu


def _online_kmeans_head(
    *,
    head,
    group_start,
    group_stop,
    clusters,
    head_dim,
    device,
    project_v_head_rows,
    warm_centroids=None,
    cluster_rows=V_SMOOTH_CLUSTER_ROWS,
):
    '''Low-memory online k-means for one value head.

    The working set is centroids plus one V slab and one [slab, clusters]
    distance matrix.  Labels are retained only as int32 so the permutation can
    be constructed without retaining V.
    '''
    rows = int(group_stop - group_start)
    cluster_rows = max(1, int(cluster_rows))
    if warm_centroids is None:
        seed_rows = _initial_centroid_rows(
            group_start, group_stop, clusters, device, head=head
        )
        centroids = project_v_head_rows(int(head), seed_rows).to(torch.float32)
    else:
        if tuple(warm_centroids.shape) != (clusters, head_dim):
            raise ValueError('cached H3V-Smooth centroid geometry changed')
        centroids = warm_centroids.to(device=device, dtype=torch.float32)

    sums = torch.zeros_like(centroids)
    counts = torch.zeros(clusters, device=device, dtype=torch.int64)
    labels_cpu = torch.empty(rows, dtype=torch.int32, device='cpu')

    for relative_start in range(0, rows, cluster_rows):
        relative_stop = min(relative_start + cluster_rows, rows)
        absolute = torch.arange(
            group_start + relative_start,
            group_start + relative_stop,
            device=device,
            dtype=torch.int32,
        )
        values = project_v_head_rows(int(head), absolute)
        labels = _nearest_labels(values, centroids)
        values_f = values.to(torch.float32)
        sums.index_add_(0, labels, values_f)
        counts.index_add_(
            0,
            labels,
            torch.ones_like(labels, dtype=torch.int64),
        )
        nonzero = counts > 0
        centroids[nonzero] = sums[nonzero] / counts[nonzero].unsqueeze(-1)
        labels_cpu[relative_start:relative_stop].copy_(
            labels.to(device='cpu', dtype=torch.int32)
        )
        del absolute, values, values_f, labels, nonzero

    # Stable ordering keeps original row order inside one cluster.
    permutation_rel = torch.argsort(labels_cpu, stable=True).to(torch.int32)
    permutation_abs = permutation_rel + int(group_start)
    return centroids.to(device='cpu', dtype=torch.float32), permutation_abs


def resolve_h3v_grouping(
    module,
    snapshot,
    layout,
    *,
    block_rows,
    heads,
    head_dim,
    device,
    project_v_head_rows,
    project_v_rows=None,
    cluster_rows=V_SMOOTH_CLUSTER_ROWS,
):
    '''Build/reuse the per-head V-guided permutation for this H3 layer.

    `project_v_head_rows(head, rows)` must return [len(rows), head_dim] values
    for arbitrary absolute sequence rows.  Cached state stays on CPU so every
    transformer layer does not retain a large per-head permutation in VRAM.
    '''
    block_rows = int(block_rows)
    heads = int(heads)
    head_dim = int(head_dim)
    group_start, group_stop = _group_bounds(layout, block_rows)
    grouped_rows = int(group_stop - group_start)
    if grouped_rows <= 0:
        identity = torch.empty((heads, 0), dtype=torch.int32)
        return H3VGroupingPlan(
            identity, group_start, group_stop, False,
            h3v_smooth_active(snapshot), -1,
        )

    signature = _state_signature(
        snapshot, layout, block_rows, heads, head_dim
    )
    with _CACHE_LOCK:
        state = _CACHE.get(module)
        if state is not None and state.signature != signature:
            state = None
        previous_refresh = None if state is None else state.refresh_step
        refresh = h3v_grouping_refresh_due(snapshot, previous_refresh)

    if refresh:
        clusters = max(1, int(math.ceil(grouped_rows / block_rows)))
        centroids_cpu = torch.empty(
            heads, clusters, head_dim, dtype=torch.float32, device='cpu'
        )
        permutation_cpu = torch.empty(
            heads, grouped_rows, dtype=torch.int32, device='cpu'
        )
        if project_v_rows is not None:
            warm = None if state is None else state.centroids
            centroids, permutation = _online_kmeans_heads(
                heads=heads,
                group_start=group_start,
                group_stop=group_stop,
                clusters=clusters,
                head_dim=head_dim,
                device=device,
                project_v_head_rows=project_v_head_rows,
                project_v_rows=project_v_rows,
                warm_centroids=warm,
                cluster_rows=cluster_rows,
            )
            centroids_cpu.copy_(centroids)
            permutation_cpu.copy_(permutation)
        else:
            # Compatibility/reference path used by CPU tests and non-streamed
            # callers that only expose arbitrary one-head projection.
            for head in range(heads):
                warm = None if state is None else state.centroids[head]
                centroids, permutation = _online_kmeans_head(
                    head=head,
                    group_start=group_start,
                    group_stop=group_stop,
                    clusters=clusters,
                    head_dim=head_dim,
                    device=device,
                    project_v_head_rows=project_v_head_rows,
                    warm_centroids=warm,
                    cluster_rows=cluster_rows,
                )
                centroids_cpu[head].copy_(centroids)
                permutation_cpu[head].copy_(permutation)
        refresh_step = int(getattr(snapshot, 'step_index', -1))
        state = _GroupingState(
            signature=signature,
            centroids=centroids_cpu,
            permutation=permutation_cpu,
            refresh_step=refresh_step,
        )
        with _CACHE_LOCK:
            _CACHE[module] = state
    elif state is None:
        raise RuntimeError('H3V-Smooth grouping cache unexpectedly unavailable')

    return H3VGroupingPlan(
        permutation=state.permutation,
        group_start=group_start,
        group_stop=group_stop,
        refreshed=bool(refresh),
        demean=h3v_smooth_active(snapshot),
        refresh_step=int(state.refresh_step),
    )


__all__ = [
    'H3VGroupingPlan',
    'V_SMOOTH_ACTIVE_FRACTION',
    'V_SMOOTH_CLUSTER_ROWS',
    'V_SMOOTH_REUSE_STEPS',
    'clear_h3v_grouping_cache',
    'demean_v_blocks_',
    'h3v_grouping_refresh_due',
    'h3v_smooth_active',
    'normalized_v_means',
    'resolve_h3v_grouping',
    'subtract_v_means_',
]
