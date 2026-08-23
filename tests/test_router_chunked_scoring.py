'''Chunked router scoring must change the route by not one byte.

"Each query tile's top-K is independent" is a statement about mathematics, not
about floating point. Scoring a slice of the query tiles is a matmul with a
different M, and a different M can pick a different algorithm with a different
accumulation order, which can reorder near-ties inside topk. So the contract
these tests hold is byte identity of the emitted LUT, not equivalence of the
selection rule.
'''

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from h3_optimizations.attention.sparse.router import (  # noqa: E402
    SparseTileRouter,
)

Q_TILE = 128
KV_TILE = 128


def layout(sequence, video_start):
    return SimpleNamespace(
        seq_len=sequence,
        video_range=(video_start, sequence),
        segments=(
            (0, min(128, video_start), 'text'),
            (min(128, video_start), video_start, 'audio'),
            (video_start, sequence, 'video'),
        ),
        video_shape=(1, 1, sequence - video_start),
        audio_t=max(0, (video_start - min(128, video_start)) // 2),
    )


def summaries(sequence, heads=4, head_dim=32, seed=0):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    tiles = (sequence + Q_TILE - 1) // Q_TILE
    shape = (1, heads, tiles, head_dim)
    return (
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator),
    )


def route(q_summary, k_summary, packed, budget, chunk):
    router = SparseTileRouter(
        q_tile=Q_TILE, kv_tile=KV_TILE, score_chunk_tiles=chunk
    )
    return router.build_lut_from_summaries(q_summary, k_summary, packed, budget)


class ChunkedScoringTests(unittest.TestCase):
    SEQUENCE = 128 * 40
    VIDEO_START = 256

    def setUp(self):
        self.packed = layout(self.SEQUENCE, self.VIDEO_START)
        self.q_summary, self.k_summary = summaries(self.SEQUENCE, seed=17)

    def test_every_chunk_size_emits_the_identical_lut(self):
        reference_lut, reference_valid, reference_meta = route(
            self.q_summary, self.k_summary, self.packed, 0.3, None
        )
        for chunk in (1, 2, 5, 8, 17, 38, 64):
            lut, valid, meta = route(
                self.q_summary, self.k_summary, self.packed, 0.3, chunk
            )
            self.assertTrue(
                torch.equal(lut, reference_lut),
                'LUT differs at score_chunk_tiles=%d' % chunk,
            )
            self.assertTrue(torch.equal(valid, reference_valid))
            self.assertEqual(meta.as_dict(), reference_meta.as_dict())

    def test_it_holds_across_densities(self):
        for budget in (0.1, 0.3, 0.5, 0.9):
            reference, _, _ = route(
                self.q_summary, self.k_summary, self.packed, budget, None
            )
            chunked, _, _ = route(
                self.q_summary, self.k_summary, self.packed, budget, 7
            )
            self.assertTrue(
                torch.equal(chunked, reference), 'budget %.2f' % budget
            )

    def test_a_fully_dense_budget_still_short_circuits(self):
        lut, valid, meta = route(
            self.q_summary, self.k_summary, self.packed, 1.0, 4
        )
        self.assertEqual(meta.sparse_q_tiles, 0)
        self.assertTrue(bool((valid == meta.kv_tiles).all()))
        self.assertEqual(lut.shape[-1], meta.kv_tiles)

    def test_the_unchunked_path_is_kept_verbatim(self):
        '''Production must not change shape when the option is off.'''
        source = (
            PACK
            / 'h3_optimizations'
            / 'attention'
            / 'sparse'
            / 'router.py'
        ).read_text(encoding='utf-8')
        self.assertIn(
            'if self.score_chunk_tiles is None:\n'
            '            return torch.topk(torch.matmul(q_video, keys), '
            'retained, dim=-1).indices',
            source,
        )

    def test_a_non_positive_chunk_is_refused(self):
        for chunk in (0, -1):
            with self.assertRaises(ValueError):
                SparseTileRouter(
                    q_tile=Q_TILE, kv_tile=KV_TILE, score_chunk_tiles=chunk
                )

    def test_a_chunk_larger_than_the_tile_count_is_harmless(self):
        reference, _, _ = route(
            self.q_summary, self.k_summary, self.packed, 0.3, None
        )
        chunked, _, _ = route(
            self.q_summary, self.k_summary, self.packed, 0.3, 10_000
        )
        self.assertTrue(torch.equal(chunked, reference))

    def test_the_default_router_does_not_chunk(self):
        self.assertIsNone(
            SparseTileRouter(q_tile=Q_TILE, kv_tile=KV_TILE).score_chunk_tiles
        )


if __name__ == '__main__':
    unittest.main()
