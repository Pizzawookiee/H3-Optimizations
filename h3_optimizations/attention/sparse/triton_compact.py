'''Direct compact-route preparation for optimized INT8 Triton sparse attention.'''

import torch

from .config import resolve_video_budget
from .triton_qkv import (
    TritonSparseQKVError,
    validate_prepared_triton_sparse_qkv,
)
from .triton_route import (
    TritonRouteError,
    build_compact_absolute_route,
    build_compact_absolute_route_from_qk,
)


def _prepare_compact(executor, projected, kv_indices, metadata):
    from .triton_sparse_fast import PreparedTritonSparse, TritonSparseError, _route_plan

    try:
        validate_prepared_triton_sparse_qkv(projected)
    except TritonSparseQKVError as exc:
        raise TritonSparseError(str(exc)) from exc
    if int(projected.v_scale_group_size) != int(executor.spec.v_scale_group_size):
        raise TritonSparseError(
            'projected Triton V scale group %d does not match backend group %d'
            % (projected.v_scale_group_size, executor.spec.v_scale_group_size)
        )
    if not executor.allow_cpu_for_tests and not projected.q_int8.is_cuda:
        raise TritonSparseError('INT8 Triton sparse attention requires CUDA')

    q_tiles = (projected.sequence + executor.spec.q_tile - 1) // executor.spec.q_tile
    kv_tiles = (projected.sequence + executor.spec.kv_tile - 1) // executor.spec.kv_tile
    dense_q_tiles, sparse_q_tiles, selected = _route_plan(
        metadata,
        q_tiles=q_tiles,
        kv_tiles=kv_tiles,
    )
    expected_route = (
        1,
        projected.heads,
        sparse_q_tiles,
        selected if sparse_q_tiles else 0,
    )
    if tuple(kv_indices.shape) != expected_route:
        raise TritonSparseError(
            'compact Triton route shape %s does not match %s'
            % (tuple(kv_indices.shape), expected_route)
        )
    if kv_indices.dtype != torch.int32 or not kv_indices.is_contiguous():
        raise TritonSparseError('compact Triton route must be contiguous int32')
    if kv_indices.device != projected.q_int8.device:
        raise TritonSparseError('compact Triton route device differs from QKV')

    valid = torch.full(
        (1, projected.heads, q_tiles),
        kv_tiles,
        dtype=torch.int32,
        device=projected.q_int8.device,
    )
    if sparse_q_tiles:
        valid[..., dense_q_tiles:] = selected

    details = dict(metadata)
    details.update(
        {
            'sparse_backend': 'triton_int8',
            'sparse_kernel': executor.spec.implementation,
            'qkv_lifetime': 'independent_int8_carriers',
            'v_format': 'per_kv_tile_group%d_int8'
            % executor.spec.v_scale_group_size,
            'route_format': 'absolute_compact_int32_direct',
            'probability_value_path': 'u8_logical_x_int8_tensorcore',
            'dense_q_kernel': bool(dense_q_tiles),
            'fixed_sparse_kv_blocks': int(selected),
        }
    )
    return PreparedTritonSparse(
        q_int8=projected.q_int8,
        q_scale=projected.q_scale,
        k_int8=projected.k_int8,
        k_scale=projected.k_scale,
        v_int8=projected.v_int8,
        v_scale=projected.v_scale,
        v_sum=projected.v_sum,
        kv_indices=kv_indices,
        valid_block_num=valid,
        output_dtype=projected.output_dtype,
        layer_index=int(projected.layer_index),
        sequence=int(projected.sequence),
        heads=int(projected.heads),
        q_tiles=q_tiles,
        kv_tiles=kv_tiles,
        dense_q_tiles=dense_q_tiles,
        sparse_q_tiles=sparse_q_tiles,
        sparse_selected=selected,
        metadata=details,
    )


def executor_prepare_compact(
    self,
    q,
    k,
    v,
    kv_indices,
    *,
    layer_index,
    metadata,
):
    from .triton_sparse_fast import TritonSparseError

    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise TritonSparseError(
            'INT8 Triton sparse requires equal HND rank-4 Q/K/V shapes'
        )
    try:
        projected = self.float_packer(
            q,
            k,
            v,
            layer_index=layer_index,
            v_scale_group_size=self.spec.v_scale_group_size,
        )
    except TritonSparseQKVError as exc:
        raise TritonSparseError(str(exc)) from exc
    prepared = _prepare_compact(self, projected, kv_indices, metadata)
    prepared.metadata['qkv_projection'] = 'standard_qkv_then_int8_pack'
    return prepared


def executor_prepare_projected_compact(
    self,
    projected,
    kv_indices,
    *,
    layer_index,
    metadata,
):
    from .triton_sparse_fast import TritonSparseError

    if int(projected.layer_index) != int(layer_index):
        raise TritonSparseError(
            'projected Triton QKV layer %d does not match attention layer %d'
            % (projected.layer_index, layer_index)
        )
    prepared = _prepare_compact(self, projected, kv_indices, metadata)
    prepared.metadata['qkv_projection'] = 'chunked_convrot_int8'
    return prepared


def backend_prepare(self, q, k, v, *, layer_index, transformer_options):
    from .triton_sparse_fast import PreparedTritonHybrid, TritonSparseError

    snapshot = self._snapshot(transformer_options, q.shape[-2])
    video_budget = resolve_video_budget(
        self.config,
        snapshot.step_index,
        snapshot.total_steps,
        layer_index,
    )
    try:
        route, mask_metadata = build_compact_absolute_route_from_qk(
            self.router,
            q,
            k,
            snapshot.layout,
            video_budget,
        )
    except TritonRouteError as exc:
        raise TritonSparseError('sparse routing failed: %s' % exc) from exc
    metadata = self._metadata(mask_metadata, layer_index, q.shape[1])
    return PreparedTritonHybrid(
        sparse=self.executor.prepare_compact(
            q,
            k,
            v,
            route,
            layer_index=layer_index,
            metadata=metadata,
        )
    )


def backend_prepare_projected(
    self,
    projected,
    *,
    layer_index,
    transformer_options,
):
    from .triton_sparse_fast import PreparedTritonHybrid, TritonSparseError

    try:
        validate_prepared_triton_sparse_qkv(projected)
    except TritonSparseQKVError as exc:
        raise TritonSparseError(str(exc)) from exc
    snapshot = self._snapshot(transformer_options, projected.sequence)
    video_budget = resolve_video_budget(
        self.config,
        snapshot.step_index,
        snapshot.total_steps,
        layer_index,
    )
    try:
        route, mask_metadata = build_compact_absolute_route(
            self.router,
            projected.q_summary,
            projected.k_summary,
            snapshot.layout,
            video_budget,
        )
    except TritonRouteError as exc:
        raise TritonSparseError('sparse routing failed: %s' % exc) from exc
    metadata = self._metadata(mask_metadata, layer_index, projected.heads)
    return PreparedTritonHybrid(
        sparse=self.executor.prepare_projected_compact(
            projected,
            route,
            layer_index=layer_index,
            metadata=metadata,
        )
    )
