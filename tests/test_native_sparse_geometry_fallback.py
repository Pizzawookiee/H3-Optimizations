"""CPU contracts for native sparse geometry validation and fallback."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from h3_optimizations.native import int8_attention as native


def _route(rows):
    """Build an absolute 64Q x 64KV route from variable-length row lists."""
    slots = max(len(row) for row in rows)
    indices = torch.zeros((1, 1, len(rows), slots), dtype=torch.int32)
    counts = torch.zeros((1, 1, len(rows)), dtype=torch.int32)
    for row_index, values in enumerate(rows):
        counts[0, 0, row_index] = len(values)
        if values:
            indices[0, 0, row_index, :len(values)] = torch.tensor(
                values, dtype=torch.int32
            )
    return native.BlockSparseRoute(
        indices=indices,
        counts=counts,
        q_tile=64,
        kv_tile=64,
        encoding='absolute',
    )


def test_coarsen_64q_route_unions_each_pair_of_query_tiles():
    route = _route([
        [0, 2],
        [1, 2],
        [3],
        [4, 5],
    ])

    coarsened = native._coarsen_64q_route_to_128q(route, kv_tiles=6)

    assert (coarsened.q_tile, coarsened.kv_tile) == (128, 64)
    assert coarsened.encoding == 'absolute'
    assert coarsened.counts.tolist() == [[[3, 3]]]
    assert coarsened.indices[0, 0, 0, :3].tolist() == [0, 1, 2]
    assert coarsened.indices[0, 0, 1, :3].tolist() == [3, 4, 5]


def test_coarsen_64q_route_handles_an_unpaired_tail_query_tile():
    route = _route([
        [0],
        [1],
        [2, 3],
    ])

    coarsened = native._coarsen_64q_route_to_128q(route, kv_tiles=4)

    assert coarsened.counts.tolist() == [[[2, 2]]]
    assert coarsened.indices[0, 0, 0, :2].tolist() == [0, 1]
    assert coarsened.indices[0, 0, 1, :2].tolist() == [2, 3]


def test_runtime_geometry_prefers_128x64_kitchen_when_64x64_fails():
    route = _route([[0, 1], [1, 2]])
    quantized = SimpleNamespace(
        q=torch.empty((1, 1, 128, 128)),
        k=torch.empty((1, 1, 192, 128)),
    )

    def geometry_ok(q_tile, kv_tile, _device):
        return (q_tile, kv_tile) == (128, 64)

    with mock.patch(
        'h3_optimizations.native.selftest.sparse_geometry_check',
        side_effect=geometry_ok,
    ):
        selected = native._runtime_sparse_route(
            quantized,
            route,
            validate_geometry=True,
        )

    assert (selected.q_tile, selected.kv_tile) == (128, 64)
    assert selected.counts.tolist() == [[[3]]]
    assert selected.indices[0, 0, 0, :3].tolist() == [0, 1, 2]


def test_runtime_geometry_keeps_64x64_when_it_passes():
    route = _route([[0], [1]])
    quantized = SimpleNamespace(
        q=torch.empty((1, 1, 128, 128)),
        k=torch.empty((1, 1, 128, 128)),
    )

    with mock.patch(
        'h3_optimizations.native.selftest.sparse_geometry_check',
        return_value=True,
    ):
        selected = native._runtime_sparse_route(
            quantized,
            route,
            validate_geometry=True,
        )

    assert selected is route


def test_runtime_geometry_never_substitutes_an_unproven_backend():
    route = _route([[0], [1]])
    quantized = SimpleNamespace(
        q=torch.empty((1, 1, 128, 128)),
        k=torch.empty((1, 1, 128, 128)),
    )

    with mock.patch(
        'h3_optimizations.native.selftest.sparse_geometry_check',
        return_value=False,
    ):
        with pytest.raises(
            RuntimeError,
            match='no carrier-compatible Kitchen fallback geometry',
        ):
            native._runtime_sparse_route(
                quantized,
                route,
                validate_geometry=True,
            )


def test_selftest_bypass_does_not_query_geometry_cache():
    route = _route([[0], [1]])
    quantized = SimpleNamespace(
        q=torch.empty((1, 1, 128, 128)),
        k=torch.empty((1, 1, 128, 128)),
    )

    with mock.patch(
        'h3_optimizations.native.selftest.sparse_geometry_check',
    ) as geometry_check:
        selected = native._runtime_sparse_route(
            quantized,
            route,
            validate_geometry=False,
        )

    assert selected is route
    geometry_check.assert_not_called()
