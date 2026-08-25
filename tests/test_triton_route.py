'''Route parity for direct compact INT8 Triton sparse routing.'''

from types import SimpleNamespace
import unittest

import torch

from h3_optimizations.attention.sparse.router import SparseTileRouter
from h3_optimizations.attention.sparse.triton_route import (
    build_compact_absolute_route,
)


def layout(sequence=512, video_start=192):
    return SimpleNamespace(
        seq_len=sequence,
        video_range=(video_start, sequence),
        segments=((0, video_start, 'context'), (video_start, sequence, 'video')),
        video_shape=(1, 1, 1),
        audio_t=0,
    )


class TritonRouteTests(unittest.TestCase):
    def test_compact_absolute_route_matches_delta_router_selection(self):
        torch.manual_seed(17)
        router = SparseTileRouter(q_tile=64, kv_tile=64)
        packed = layout()
        geometry = router.geometry(packed)
        q = torch.randn(1, 2, geometry.q_tiles, 128, dtype=torch.float32)
        k = torch.randn(1, 2, geometry.kv_tiles, 128, dtype=torch.float32)
        budget = 0.3

        lut, valid, metadata = router.build_lut_from_summaries(
            q, k, packed, budget
        )
        route, compact_metadata = build_compact_absolute_route(
            router, q, k, packed, budget
        )

        self.assertEqual(metadata, compact_metadata)
        sparse_start = geometry.pure_video_q_start
        selected = int(valid[0, 0, sparse_start])
        expected = torch.cumsum(
            lut[..., sparse_start:, :selected], dim=-1, dtype=torch.int32
        )
        self.assertTrue(torch.equal(route, expected))
        self.assertEqual(
            tuple(route.shape),
            (1, 2, geometry.pure_video_q_tiles, selected),
        )

    def test_full_density_returns_no_sparse_route(self):
        router = SparseTileRouter(q_tile=64, kv_tile=64)
        packed = layout()
        geometry = router.geometry(packed)
        q = torch.zeros(1, 1, geometry.q_tiles, 128)
        k = torch.zeros(1, 1, geometry.kv_tiles, 128)
        route, metadata = build_compact_absolute_route(
            router, q, k, packed, 1.0
        )
        self.assertEqual(route.numel(), 0)
        self.assertEqual(metadata.sparse_q_tiles, 0)
        self.assertEqual(metadata.dense_q_tiles, geometry.q_tiles)


if __name__ == '__main__':
    unittest.main()
