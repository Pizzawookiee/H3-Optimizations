'''CPU contracts for attention-derived target-video importance.'''

import os
from pathlib import Path
import sys
import unittest

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.attention.sparse.router import SparseTileRouter  # noqa: E402
from h3_optimizations.mlp_sharing.route import (  # noqa: E402
    ROUTE_KEY,
    AttentionRouteError,
    AttentionRouteRecorder,
    get_route_recorder,
    route_sink,
    row_scores,
)
from h3_optimizations.runtime.layout import TokenLayout  # noqa: E402

Q_TILE = 128
KV_TILE = 64


def make_layout(context_rows=192, latent_t=2, patch_h=8, patch_w=8):
    '''Packed layout with target video as the final segment.'''
    frame_rows = patch_h * patch_w
    video_rows = latent_t * frame_rows
    seq_len = context_rows + video_rows
    return TokenLayout(
        seq_len=seq_len,
        text_range=(0, context_rows - 64),
        audio_range=(context_rows - 64, context_rows),
        video_range=(context_rows, seq_len),
        video_shape=(latent_t, patch_h, patch_w),
        audio_t=32,
        reference_ranges=[],
        segments=[
            (0, context_rows - 64, 'text'),
            (context_rows - 64, context_rows, 'audio'),
            (context_rows, seq_len, 'video'),
        ],
    )


class AttentionRouteRecorderTests(unittest.TestCase):
    def setUp(self):
        self.layout = make_layout()
        self.router = SparseTileRouter(q_tile=Q_TILE, kv_tile=KV_TILE)
        self.geometry = self.router.geometry(self.layout)

    def _summaries(self, heads=4, dim=16, seed=0):
        generator = torch.Generator().manual_seed(seed)
        q = torch.randn(
            1, heads, self.geometry.q_tiles, dim, generator=generator
        )
        k = torch.randn(
            1, heads, self.geometry.kv_tiles, dim, generator=generator
        )
        return q, k

    def test_router_reports_selection_counts(self):
        recorder = AttentionRouteRecorder(kv_tile=KV_TILE)
        q, k = self._summaries()
        self.router.build_lut_from_summaries(
            q,
            k,
            self.layout,
            0.5,
            sink=recorder.sink(7),
        )
        route = recorder.route(7)
        self.assertIsNotNone(route)
        self.assertFalse(route.dense)
        self.assertEqual(
            route.video_kv_tiles,
            self.geometry.pure_video_kv_tiles,
        )
        self.assertEqual(
            route.video_kv_start,
            self.geometry.pure_video_kv_start,
        )
        retained = self.router._retained(0.5, self.geometry)
        expected_draws = q.shape[0] * q.shape[1] * self.geometry.pure_video_q_tiles
        self.assertEqual(route.draws, expected_draws)
        self.assertEqual(
            int(route.counts.sum()),
            expected_draws * retained,
        )
        rate = route.hit_rate()
        self.assertEqual(rate.shape, (self.geometry.pure_video_kv_tiles,))
        self.assertTrue(bool(((rate >= 0.0) & (rate <= 1.0)).all()))
        # A fixed per-query-tile budget makes the mean hit rate the density.
        self.assertAlmostEqual(
            float(rate.mean()),
            retained / self.geometry.pure_video_kv_tiles,
            places=5,
        )

    def test_counts_match_a_reference_histogram(self):
        recorder = AttentionRouteRecorder(kv_tile=KV_TILE)
        q, k = self._summaries(seed=3)
        self.router.build_lut_from_summaries(
            q,
            k,
            self.layout,
            0.25,
            sink=recorder.sink(0),
        )
        scores = torch.matmul(
            q[..., self.geometry.pure_video_q_start:, :],
            k[..., self.geometry.pure_video_kv_start:, :].transpose(-1, -2),
        )
        retained = self.router._retained(0.25, self.geometry)
        indices = torch.topk(scores, retained, dim=-1).indices
        expected = torch.bincount(
            indices.reshape(-1),
            minlength=self.geometry.pure_video_kv_tiles,
        )
        self.assertTrue(
            torch.equal(recorder.route(0).counts.long(), expected.long())
        )

    def test_dense_route_reports_full_retention(self):
        recorder = AttentionRouteRecorder(kv_tile=KV_TILE)
        q, k = self._summaries()
        self.router.build_lut_from_summaries(
            q,
            k,
            self.layout,
            1.0,
            sink=recorder.sink(1),
        )
        route = recorder.route(1)
        self.assertTrue(route.dense)
        self.assertIsNone(route.counts)
        self.assertTrue(bool((route.hit_rate() == 1.0).all()))

    def test_router_is_unchanged_without_a_sink(self):
        q, k = self._summaries(seed=11)
        with_sink = self.router.build_lut_from_summaries(
            q,
            k,
            self.layout,
            0.5,
            sink=AttentionRouteRecorder(kv_tile=KV_TILE).sink(0),
        )
        without = self.router.build_lut_from_summaries(q, k, self.layout, 0.5)
        self.assertTrue(torch.equal(with_sink[0], without[0]))
        self.assertTrue(torch.equal(with_sink[1], without[1]))
        self.assertEqual(with_sink[2].as_dict(), without[2].as_dict())

    def test_kv_tile_mismatch_is_rejected(self):
        recorder = AttentionRouteRecorder(kv_tile=32)
        q, k = self._summaries()
        with self.assertRaises(AttentionRouteError):
            self.router.build_lut_from_summaries(
                q,
                k,
                self.layout,
                0.5,
                sink=recorder.sink(0),
            )

    def test_route_sink_is_none_without_a_recorder(self):
        self.assertIsNone(route_sink({}, 0))
        self.assertIsNone(route_sink(None, 0))
        self.assertIsNone(get_route_recorder({'other': object()}))
        recorder = AttentionRouteRecorder(kv_tile=KV_TILE)
        self.assertIs(
            get_route_recorder({ROUTE_KEY: recorder}),
            recorder,
        )
        self.assertIsNotNone(route_sink({ROUTE_KEY: recorder}, 4))


class AttentionEnergyTests(unittest.TestCase):
    def setUp(self):
        self.layout = make_layout()
        self.segments = [
            (start, stop, index)
            for index, (start, stop, _kind) in enumerate(self.layout.segments)
        ]

    def test_energy_is_the_gated_residual_tile_mean(self):
        recorder = AttentionRouteRecorder(kv_tile=KV_TILE)
        rows = self.layout.seq_len
        generator = torch.Generator().manual_seed(5)
        attn_out = torch.randn(rows, 12, generator=generator)
        gate = torch.randn(3, 12, generator=generator)
        recorder.record_attention_energy(
            0,
            attn_out,
            gate,
            self.segments,
            4096,
        )
        energy = recorder.route(0).tile_energy

        reference = torch.empty(rows, dtype=torch.float32)
        for start, stop, row in self.segments:
            gated = attn_out[start:stop] * gate[row]
            reference[start:stop] = gated.pow(2).sum(dim=-1)
        tiles = rows // KV_TILE
        self.assertEqual(rows % KV_TILE, 0)
        expected = reference.view(tiles, KV_TILE).mean(dim=1)
        self.assertEqual(energy.shape, (tiles,))
        self.assertTrue(torch.allclose(energy, expected, atol=1e-5))

    def test_chunking_does_not_change_energy(self):
        rows = self.layout.seq_len
        generator = torch.Generator().manual_seed(9)
        attn_out = torch.randn(rows, 12, generator=generator)
        gate = torch.randn(rows, 12, generator=generator)
        segments = [
            (start, stop, torch.arange(start, stop, dtype=torch.long))
            for start, stop, _kind in self.layout.segments
        ]
        whole = AttentionRouteRecorder(kv_tile=KV_TILE)
        whole.record_attention_energy(0, attn_out, gate, segments, 4096)
        split = AttentionRouteRecorder(kv_tile=KV_TILE)
        split.record_attention_energy(0, attn_out, gate, segments, 37)
        self.assertTrue(
            torch.allclose(
                whole.route(0).tile_energy,
                split.route(0).tile_energy,
                atol=1e-6,
            )
        )

    def test_partial_final_tile_uses_its_real_occupancy(self):
        recorder = AttentionRouteRecorder(kv_tile=KV_TILE)
        rows = KV_TILE + 10
        attn_out = torch.ones(rows, 4)
        gate = torch.ones(1, 4)
        recorder.record_attention_energy(
            0,
            attn_out,
            gate,
            [(0, rows, 0)],
            4096,
        )
        energy = recorder.route(0).tile_energy
        self.assertEqual(energy.shape, (2,))
        self.assertTrue(torch.allclose(energy, torch.full((2,), 4.0)))


class RowScoreTests(unittest.TestCase):
    def setUp(self):
        self.layout = make_layout()
        self.router = SparseTileRouter(q_tile=Q_TILE, kv_tile=KV_TILE)
        self.geometry = self.router.geometry(self.layout)
        self.recorder = AttentionRouteRecorder(kv_tile=KV_TILE)
        generator = torch.Generator().manual_seed(1)
        q = torch.randn(1, 2, self.geometry.q_tiles, 8, generator=generator)
        k = torch.randn(1, 2, self.geometry.kv_tiles, 8, generator=generator)
        self.router.build_lut_from_summaries(
            q,
            k,
            self.layout,
            0.5,
            sink=self.recorder.sink(0),
        )
        self.route = self.recorder.route(0)

    def test_context_rows_are_not_scored(self):
        scores = row_scores(self.route, 0, self.layout.seq_len)
        video_start = self.geometry.pure_video_kv_start * KV_TILE
        self.assertTrue(bool(torch.isnan(scores[:video_start]).all()))
        self.assertFalse(bool(torch.isnan(scores[video_start:]).any()))

    def test_rows_inherit_their_kv_tile_score(self):
        scores = row_scores(self.route, 0, self.layout.seq_len)
        rate = self.route.hit_rate()
        video_start = self.geometry.pure_video_kv_start * KV_TILE
        for tile in range(self.geometry.pure_video_kv_tiles):
            start = video_start + tile * KV_TILE
            stop = min(start + KV_TILE, self.layout.seq_len)
            block = scores[start:stop]
            self.assertTrue(
                torch.allclose(block, torch.full_like(block, float(rate[tile])))
            )

    def test_chunk_slices_agree_with_the_whole_sequence(self):
        whole = row_scores(self.route, 0, self.layout.seq_len)
        for start in range(0, self.layout.seq_len, 100):
            stop = min(start + 100, self.layout.seq_len)
            part = row_scores(self.route, start, stop)
            self.assertTrue(
                torch.equal(
                    torch.nan_to_num(part, nan=-1.0),
                    torch.nan_to_num(whole[start:stop], nan=-1.0),
                )
            )

    def test_energy_signal_shares_the_row_mapping(self):
        rows = self.layout.seq_len
        attn_out = torch.randn(rows, 6, generator=torch.Generator().manual_seed(2))
        gate = torch.ones(3, 6)
        segments = [
            (start, stop, index)
            for index, (start, stop, _kind) in enumerate(self.layout.segments)
        ]
        self.recorder.record_attention_energy(0, attn_out, gate, segments, 4096)
        scores = row_scores(self.route, 0, rows, signal='attention_update_energy')
        video_start = self.geometry.pure_video_kv_start * KV_TILE
        self.assertTrue(bool(torch.isnan(scores[:video_start]).all()))
        self.assertTrue(bool((scores[video_start:] > 0.0).all()))

    def test_unknown_signal_is_rejected(self):
        with self.assertRaises(AttentionRouteError):
            row_scores(self.route, 0, 64, signal='nope')

    def test_missing_route_returns_none(self):
        self.assertIsNone(row_scores(None, 0, 64))
        self.assertIsNone(
            row_scores(self.recorder.route(0), 0, 64, signal='attention_update_energy')
            if self.recorder.route(0).tile_energy is not None
            else None
        )


if __name__ == '__main__':
    sys.argv = [sys.argv[0], *TEST_ARGS]
    unittest.main()
