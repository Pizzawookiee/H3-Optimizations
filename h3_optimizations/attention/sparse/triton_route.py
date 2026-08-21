'''Compact fixed-density routes for the INT8 Triton sparse backend.'''

import torch

from .router import SparseRouterError, sort_selected_indices


class TritonRouteError(RuntimeError):
    pass


def _validate_summaries(router, q_summary, k_summary, layout, video_budget):
    if q_summary.ndim != 4 or k_summary.ndim != 4:
        raise TritonRouteError('Triton route summaries must be rank-4 HND tensors')
    if q_summary.shape[:2] != k_summary.shape[:2]:
        raise TritonRouteError('Triton route Q/K batch and head shapes differ')
    if q_summary.shape[-1] != k_summary.shape[-1]:
        raise TritonRouteError('Triton route Q/K summary dimensions differ')
    if q_summary.device != k_summary.device:
        raise TritonRouteError('Triton route Q/K summary devices differ')
    if not 0.01 <= float(video_budget) <= 1.0:
        raise TritonRouteError('video_budget must be in [0.01, 1]')
    try:
        geometry = router.geometry(layout)
    except SparseRouterError as exc:
        raise TritonRouteError(str(exc)) from exc
    expected_q = (geometry.q_tiles, q_summary.shape[-1])
    expected_k = (geometry.kv_tiles, k_summary.shape[-1])
    if tuple(q_summary.shape[-2:]) != expected_q:
        raise TritonRouteError(
            'Q router summary shape %s does not match %s'
            % (tuple(q_summary.shape[-2:]), expected_q)
        )
    if tuple(k_summary.shape[-2:]) != expected_k:
        raise TritonRouteError(
            'K router summary shape %s does not match %s'
            % (tuple(k_summary.shape[-2:]), expected_k)
        )
    return geometry


def build_compact_absolute_route(
    router,
    q_summary,
    k_summary,
    layout,
    video_budget,
):
    '''Return only the absolute KV indices consumed by pure-video Q rows.'''
    geometry = _validate_summaries(
        router, q_summary, k_summary, layout, video_budget
    )
    retained = router._retained(video_budget, geometry)
    metadata = router._metadata(geometry, video_budget, retained)
    batch, heads = q_summary.shape[:2]

    if retained == geometry.pure_video_kv_tiles:
        return (
            torch.empty(
                (batch, heads, 0, 0),
                dtype=torch.int32,
                device=q_summary.device,
            ),
            metadata,
        )

    scores = torch.matmul(
        q_summary[..., geometry.pure_video_q_start:, :],
        k_summary[
            ..., geometry.pure_video_kv_start:, :
        ].transpose(-1, -2),
    )
    selected = torch.topk(scores, retained, dim=-1).indices.to(torch.int32)
    selected = sort_selected_indices(
        selected + int(geometry.pure_video_kv_start)
    )

    context_count = int(geometry.pure_video_kv_start)
    if context_count:
        context = torch.arange(
            context_count,
            dtype=torch.int32,
            device=q_summary.device,
        ).view(1, 1, 1, -1).expand(
            batch,
            heads,
            geometry.pure_video_q_tiles,
            -1,
        )
        route = torch.cat((context, selected), dim=-1)
    else:
        route = selected
    return route.contiguous(), metadata


def build_compact_absolute_route_from_qk(
    router,
    q,
    k,
    layout,
    video_budget,
):
    if q.ndim != 4 or k.ndim != 4 or q.shape != k.shape:
        raise TritonRouteError('Triton route expects equal rank-4 HND Q/K')
    try:
        q_summary = router._mean_pool(q, router.q_tile)
        k_summary = router._mean_pool(k, router.kv_tile)
    except Exception as exc:
        raise TritonRouteError('Triton route mean pooling failed: %s' % exc) from exc
    return build_compact_absolute_route(
        router,
        q_summary,
        k_summary,
        layout,
        video_budget,
    )
