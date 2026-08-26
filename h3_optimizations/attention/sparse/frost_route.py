'''Direct full-width absolute routes for the BF16 FROST backend.'''

import torch

from .triton_route import (
    TritonRouteError,
    build_compact_absolute_route,
)


def build_full_absolute_route(
    router,
    q,
    k,
    layout,
    video_budget,
    *,
    sink=None,
):
    if q.ndim != 4 or k.ndim != 4 or q.shape != k.shape:
        raise TritonRouteError('FROST route expects equal rank-4 HND Q/K')
    try:
        q_summary = router._mean_pool(q, router.q_tile)
        k_summary = router._mean_pool(k, router.kv_tile)
    except Exception as error:
        raise TritonRouteError('FROST route mean pooling failed: %s' % error) from error
    compact, metadata = build_compact_absolute_route(
        router,
        q_summary,
        k_summary,
        layout,
        video_budget,
    )
    geometry = router.geometry(layout)
    batch, heads = q.shape[:2]
    dense = torch.arange(
        geometry.kv_tiles,
        dtype=torch.int32,
        device=q.device,
    )
    route = dense.view(1, 1, 1, -1).expand(
        batch,
        heads,
        geometry.q_tiles,
        -1,
    ).clone()
    counts = torch.full(
        (batch, heads, geometry.q_tiles),
        geometry.kv_tiles,
        dtype=torch.int32,
        device=q.device,
    )
    if compact.numel():
        start = geometry.pure_video_q_start
        route[..., start:, :compact.shape[-1]].copy_(compact)
        counts[..., start:] = compact.shape[-1]
        selected = (
            compact[..., geometry.pure_video_kv_start:]
            - geometry.pure_video_kv_start
        )
    else:
        selected = None
    router._notify(sink, geometry, selected)
    return route.contiguous(), counts.contiguous(), metadata
