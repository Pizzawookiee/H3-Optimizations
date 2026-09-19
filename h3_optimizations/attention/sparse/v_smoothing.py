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
or a reordered BF16 K/V carrier.  The production path projects one bounded
all-head V slab at a time, clusters in a compact feature space, stores only the
resulting permutation/centroids between refreshes, and lets the producer
re-project K/V directly in grouped order.
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
# Fast clustering path used when an all-head V projector is available.  The
# permutation only needs a value-space neighborhood signal; doing nearest-
# centroid assignment in the full 128-D head space against one centroid per
# hardware block is prohibitively expensive on long video sequences.
V_SMOOTH_FEATURE_DIM = 8
# Skinny feature projection chunks can be much larger than full-V clustering
# slabs because they produce only heads * feature_dim outputs.
V_SMOOTH_FEATURE_ROWS = 8192
V_SMOOTH_MAX_COARSE_CLUSTERS = 256
V_SMOOTH_HEAD_BATCH = 8
V_SMOOTH_CLUSTER_ALGO_VERSION = 3

_FEATURE_PROJECTION_CACHE = {}
_FEATURE_PROJECTION_LOCK = threading.RLock()


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
    centroids: torch.Tensor  # clustering centroids, FP32 CPU
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
    '''Build one grouping per layer per sampling request, then reuse it.

    The cache signature contains the request id / sequence geometry, so a new
    generation naturally invalidates the previous grouping.  Re-clustering every
    four denoising steps made the optimization slower than the attention itself
    on long H3 video sequences and provides little value relative to the much
    larger variation between transformer layers.
    '''
    del snapshot
    return previous_refresh_step is None


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


def _group_bounds(layout, block_rows, group_alignment=None):
    sequence = int(layout.seq_len)
    video_start, video_stop = (int(value) for value in layout.video_range)
    if video_stop != sequence:
        raise ValueError('H3V-Smooth requires target video to be the final segment')
    # Match SparseTileRouter.pure_video_kv_start, but also honor the producer's
    # chunk alignment.  Kitchen's INT8 producer currently requires 128-row K
    # chunks even for a 64-row KV tile, so starting grouping on only a 64-row
    # boundary can split the prefix at e.g. 320 rows and make that non-final K
    # chunk illegal.  Align to both constraints while keeping `block_rows` as
    # the actual clustering / V-mean block size.
    alignment = int(block_rows)
    if group_alignment is not None:
        group_alignment = int(group_alignment)
        if group_alignment <= 0:
            raise ValueError('H3V-Smooth group_alignment must be positive')
        alignment = math.lcm(alignment, group_alignment)
    group_start = ((video_start + alignment - 1) // alignment) * alignment
    return min(group_start, video_stop), video_stop


def _state_signature(snapshot, layout, block_rows, heads, head_dim, group_alignment=None):
    group_start, group_stop = _group_bounds(layout, block_rows, group_alignment)
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
        None if group_alignment is None else int(group_alignment),
        int(V_SMOOTH_CLUSTER_ALGO_VERSION),
        int(V_SMOOTH_FEATURE_DIM),
        int(V_SMOOTH_FEATURE_ROWS),
        int(V_SMOOTH_MAX_COARSE_CLUSTERS),
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



def _feature_projection(head_dim, feature_dim, device):
    """Deterministic Rademacher projection used only for clustering.

    A small random-sign projection preserves broad value-space neighborhoods
    while making nearest-centroid assignment far cheaper than the full head
    dimension.  A private CPU generator keeps it deterministic without changing
    PyTorch's global RNG state.
    """
    head_dim = int(head_dim)
    feature_dim = max(1, min(int(feature_dim), head_dim))
    key = (head_dim, feature_dim, str(device))
    with _FEATURE_PROJECTION_LOCK:
        cached = _FEATURE_PROJECTION_CACHE.get(key)
        if cached is not None:
            return cached

    generator = torch.Generator(device='cpu')
    generator.manual_seed(0x48335653 + head_dim * 131 + feature_dim * 17)
    projection = torch.empty(
        head_dim, feature_dim, dtype=torch.float32, device='cpu'
    )
    projection.bernoulli_(0.5, generator=generator)
    projection.mul_(2.0).sub_(1.0).div_(math.sqrt(float(feature_dim)))
    projection = projection.to(device=device, non_blocking=True).contiguous()
    with _FEATURE_PROJECTION_LOCK:
        _FEATURE_PROJECTION_CACHE[key] = projection
    return projection


def _cluster_features(values, feature_dim):
    """Compress [..., head_dim] values into a small FP32 clustering space."""
    if int(feature_dim) >= int(values.shape[-1]):
        return values.to(torch.float32)
    projection = _feature_projection(values.shape[-1], feature_dim, values.device)
    return torch.matmul(values.to(torch.float32), projection)

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
    project_v_features_rows=None,
    warm_centroids=None,
    cluster_rows=V_SMOOTH_CLUSTER_ROWS,
):
    """Fast low-memory per-head grouping for the streamed production path.

    The preferred path projects only a tiny set of V channels for every head
    (8-D by default), so clustering never materializes a full 128-D V slab.
    Providers that cannot expose a skinny projection fall back to the previous
    full-V projection plus deterministic feature compression.  Labels stay
    on the GPU for the full scan; only the final permutation and compact cached
    centroids cross to CPU once per refresh.

    Coarse clusters are intentionally allowed to contain multiple final 64/128-
    row hardware blocks.  Stable sorting by coarse label still packs similar V
    rows together, while avoiding one full k-means centroid per physical block.
    """
    rows = int(group_stop - group_start)
    heads = int(heads)
    feature_dim = max(1, min(V_SMOOTH_FEATURE_DIM, int(head_dim)))
    if project_v_features_rows is not None:
        cluster_rows = max(1, int(V_SMOOTH_FEATURE_ROWS))
    else:
        cluster_rows = max(1, int(cluster_rows))
    coarse_clusters = max(
        1, min(int(clusters), V_SMOOTH_MAX_COARSE_CLUSTERS, cluster_rows, rows)
    )
    head_batch = max(1, min(V_SMOOTH_HEAD_BATCH, heads))

    centroids = None
    if warm_centroids is not None:
        expected = (heads, coarse_clusters, feature_dim)
        if tuple(warm_centroids.shape) == expected:
            centroids = warm_centroids.to(device=device, dtype=torch.float32)

    sums = torch.zeros(
        heads, coarse_clusters, feature_dim, dtype=torch.float32, device=device
    )
    counts = torch.zeros(
        heads, coarse_clusters, dtype=torch.int32, device=device
    )
    labels_gpu = torch.empty(heads, rows, dtype=torch.int32, device=device)

    for relative_start in range(0, rows, cluster_rows):
        relative_stop = min(relative_start + cluster_rows, rows)
        absolute_start = group_start + relative_start
        absolute_stop = group_start + relative_stop
        if project_v_features_rows is not None:
            features = project_v_features_rows(
                absolute_start, absolute_stop, feature_dim
            )
            if features.ndim == 4:
                if int(features.shape[0]) != 1:
                    raise ValueError('H3V-Smooth skinny V projection batch must be 1')
                features = features[0]
            expected = (heads, relative_stop - relative_start, feature_dim)
            if tuple(features.shape) != expected:
                raise ValueError(
                    'H3V-Smooth skinny V projection has shape %s, expected %s'
                    % (tuple(features.shape), expected)
                )
            features = features.to(torch.float32)
        else:
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
            features = _cluster_features(values_all, feature_dim)
            del values_all

        if centroids is None:
            slab_rows = int(features.shape[1])
            if slab_rows < coarse_clusters:
                raise ValueError('H3V-Smooth clustering slab is smaller than centroid count')
            base = torch.linspace(
                0, slab_rows - 1, coarse_clusters, device=device
            ).round().to(torch.int64)
            head_offsets = (
                torch.arange(heads, device=device, dtype=torch.int64) * 17
            ) % slab_rows
            seed_idx = (base.unsqueeze(0) + head_offsets.unsqueeze(1)) % slab_rows
            centroids = torch.gather(
                features,
                1,
                seed_idx.unsqueeze(-1).expand(-1, -1, feature_dim),
            ).contiguous()

        for head_start in range(0, heads, head_batch):
            head_stop = min(head_start + head_batch, heads)
            values_f = features[head_start:head_stop]
            center_f = centroids[head_start:head_stop]

            x2 = (values_f * values_f).sum(dim=-1, keepdim=True)
            c2 = (center_f * center_f).sum(dim=-1).unsqueeze(1)
            distances = x2 + c2 - 2.0 * torch.bmm(
                values_f, center_f.transpose(1, 2)
            )
            labels = torch.argmin(distances, dim=-1)
            labels_gpu[head_start:head_stop, relative_start:relative_stop] = (
                labels.to(torch.int32)
            )
            del distances, x2, c2

            batch_heads = head_stop - head_start
            offsets = (
                torch.arange(batch_heads, device=device, dtype=torch.int64)
                * coarse_clusters
            ).unsqueeze(1)
            flat_labels = (labels + offsets).reshape(-1)
            sums_flat = sums[head_start:head_stop].reshape(
                batch_heads * coarse_clusters, feature_dim
            )
            counts_flat = counts[head_start:head_stop].reshape(
                batch_heads * coarse_clusters
            )
            sums_flat.index_add_(0, flat_labels, values_f.reshape(-1, feature_dim))
            counts_flat.index_add_(
                0, flat_labels, torch.ones_like(flat_labels, dtype=torch.int32)
            )

            batch_counts = counts[head_start:head_stop]
            updated = sums[head_start:head_stop] / batch_counts.clamp_min(1).unsqueeze(-1)
            nonzero = batch_counts > 0
            centroids[head_start:head_stop] = torch.where(
                nonzero.unsqueeze(-1), updated, center_f
            )
            del labels, flat_labels, updated, nonzero

        del features

    # One GPU sort and one GPU->CPU transfer per refresh, rather than one transfer
    # for every head of every source slab.
    permutation_gpu = torch.argsort(labels_gpu, dim=-1, stable=True).to(torch.int32)
    permutation_gpu.add_(int(group_start))
    permutation_cpu = permutation_gpu.to(device='cpu', dtype=torch.int32)
    centroids_cpu = centroids.to(device='cpu', dtype=torch.float32)
    return centroids_cpu, permutation_cpu

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
    project_v_features_rows=None,
    cluster_rows=V_SMOOTH_CLUSTER_ROWS,
    group_alignment=None,
):
    '''Build/reuse the per-head V-guided permutation for this H3 layer.

    `project_v_head_rows(head, rows)` must return [len(rows), head_dim] values
    for arbitrary absolute sequence rows.  Cached state stays on CPU so every
    transformer layer does not retain a large per-head permutation in VRAM.
    '''
    block_rows = int(block_rows)
    heads = int(heads)
    head_dim = int(head_dim)
    group_start, group_stop = _group_bounds(layout, block_rows, group_alignment)
    grouped_rows = int(group_stop - group_start)
    if grouped_rows <= 0:
        identity = torch.empty((heads, 0), dtype=torch.int32)
        return H3VGroupingPlan(
            identity, group_start, group_stop, False,
            h3v_smooth_active(snapshot), -1,
        )

    signature = _state_signature(
        snapshot, layout, block_rows, heads, head_dim, group_alignment
    )
    with _CACHE_LOCK:
        state = _CACHE.get(module)
        if state is not None and state.signature != signature:
            state = None
        previous_refresh = None if state is None else state.refresh_step
        refresh = h3v_grouping_refresh_due(snapshot, previous_refresh)

    if refresh:
        clusters = max(1, int(math.ceil(grouped_rows / block_rows)))
        if project_v_rows is not None or project_v_features_rows is not None:
            warm = None if state is None else state.centroids
            centroids_cpu, permutation_cpu = _online_kmeans_heads(
                heads=heads,
                group_start=group_start,
                group_stop=group_stop,
                clusters=clusters,
                head_dim=head_dim,
                device=device,
                project_v_head_rows=project_v_head_rows,
                project_v_rows=project_v_rows,
                project_v_features_rows=project_v_features_rows,
                warm_centroids=warm,
                cluster_rows=cluster_rows,
            )
        else:
            # Compatibility/reference path used by CPU tests and non-streamed
            # callers that only expose arbitrary one-head projection.
            centroids_cpu = torch.empty(
                heads, clusters, head_dim, dtype=torch.float32, device='cpu'
            )
            permutation_cpu = torch.empty(
                heads, grouped_rows, dtype=torch.int32, device='cpu'
            )
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
    'V_SMOOTH_FEATURE_DIM',
    'V_SMOOTH_FEATURE_ROWS',
    'V_SMOOTH_MAX_COARSE_CLUSTERS',
    'V_SMOOTH_REUSE_STEPS',
    'clear_h3v_grouping_cache',
    'demean_v_blocks_',
    'h3v_grouping_refresh_due',
    'h3v_smooth_active',
    'normalized_v_means',
    'resolve_h3v_grouping',
    'subtract_v_means_',
]
