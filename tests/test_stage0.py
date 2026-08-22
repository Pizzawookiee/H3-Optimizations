'''CPU contracts for the Stage 0 dense-MLP diagnostic.'''

import math
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

from h3_optimizations.mlp_sharing import stage0_metrics as metrics  # noqa: E402
from h3_optimizations.mlp_sharing.config import Stage0Config  # noqa: E402
from h3_optimizations.mlp_sharing.stage0 import (  # noqa: E402
    Stage0Session,
    _modulate,
    _unmodulate,
    pool_summary,
    sample_blocks,
)
from test_attention_route import make_layout  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]

GROUPS = metrics.FFN_GROUPS
WIDTH = GROUPS * metrics.CONVROT_GROUP


class SelectorMetricTests(unittest.TestCase):
    def test_row_update_energy_matches_the_gated_norm(self):
        y = torch.randn(6, 5)
        gate = torch.randn(2, 5)
        selector = torch.tensor([0, 1, 0, 1, 1, 0])
        energy = metrics.row_update_energy(y, gate, selector)
        expected = torch.stack([
            (y[row] * gate[int(selector[row])]).pow(2).sum()
            for row in range(6)
        ])
        self.assertTrue(torch.allclose(energy, expected, atol=1e-6))

    def test_scalar_selector_broadcasts(self):
        y = torch.randn(4, 5)
        gate = torch.randn(3, 5)
        energy = metrics.row_update_energy(y, gate, 2)
        expected = (y * gate[2]).pow(2).sum(dim=-1)
        self.assertTrue(torch.allclose(energy, expected, atol=1e-6))

    def test_tile_accumulation_drops_unrouted_rows(self):
        kv_tile, video_kv_start, video_kv_tiles = 4, 2, 3
        energy = torch.arange(20, dtype=torch.float32)
        destination = torch.zeros(video_kv_tiles, dtype=torch.float64)
        metrics.accumulate_tile_energy(
            destination,
            energy,
            0,
            kv_tile,
            video_kv_start,
            video_kv_tiles,
        )
        # rows 0-7 sit before the first pure-video tile, rows 20+ do not exist
        self.assertAlmostEqual(float(destination[0]), sum(range(8, 12)))
        self.assertAlmostEqual(float(destination[1]), sum(range(12, 16)))
        self.assertAlmostEqual(float(destination[2]), sum(range(16, 20)))

    def test_accumulation_is_chunk_split_invariant(self):
        kv_tile, video_kv_start, video_kv_tiles = 4, 1, 4
        energy = torch.rand(24)
        whole = torch.zeros(video_kv_tiles, dtype=torch.float64)
        metrics.accumulate_tile_energy(
            whole, energy, 0, kv_tile, video_kv_start, video_kv_tiles
        )
        split = torch.zeros(video_kv_tiles, dtype=torch.float64)
        for start, stop in ((0, 7), (7, 13), (13, 24)):
            metrics.accumulate_tile_energy(
                split,
                energy[start:stop],
                start,
                kv_tile,
                video_kv_start,
                video_kv_tiles,
            )
        self.assertTrue(torch.allclose(whole, split, atol=1e-9))

    def test_oracle_selector_captures_everything_it_can(self):
        oracle = torch.rand(32)
        scores = metrics.selector_scores(
            hit_rate=torch.rand(32),
            attention_update_energy=torch.rand(32) * 1e6,
            oracle=oracle,
            seed=1,
        )
        rows = metrics.selector_capture_table(scores, oracle)
        by_key = {(row['selector'], row['tile_budget']): row for row in rows}
        for budget in metrics.TILE_BUDGETS:
            best = by_key[('oracle', budget)]
            self.assertAlmostEqual(best['oracle_topk_overlap'], 1.0)
            for name in ('hit_rate', 'attention_update_energy', 'random'):
                self.assertLessEqual(
                    by_key[(name, budget)]['oracle_mass_captured'],
                    best['oracle_mass_captured'] + 1e-9,
                )
            self.assertAlmostEqual(
                best['oracle_mass_captured'] + best['skipped_update_error'],
                1.0,
                places=6,
            )

    def test_a_selector_that_agrees_with_the_oracle_captures_it_all(self):
        oracle = torch.rand(16)
        scores = metrics.selector_scores(
            hit_rate=oracle.clone(),
            attention_update_energy=None,
            oracle=oracle,
            seed=0,
        )
        rows = metrics.selector_capture_table(scores, oracle, budgets=(0.5,))
        hit = next(row for row in rows if row['selector'] == 'hit_rate')
        best = next(row for row in rows if row['selector'] == 'oracle')
        self.assertAlmostEqual(
            hit['oracle_mass_captured'],
            best['oracle_mass_captured'],
        )
        self.assertAlmostEqual(hit['oracle_topk_overlap'], 1.0)

    def test_combined_appears_only_with_both_signals(self):
        oracle = torch.rand(8)
        only_hits = metrics.selector_scores(
            hit_rate=torch.rand(8),
            attention_update_energy=None,
            oracle=oracle,
            seed=0,
        )
        self.assertNotIn('combined', only_hits)
        both = metrics.selector_scores(
            hit_rate=torch.rand(8),
            attention_update_energy=torch.rand(8),
            oracle=oracle,
            seed=0,
        )
        self.assertIn('combined', both)

    def test_random_selector_is_reproducible_and_spread(self):
        first = metrics.deterministic_uniform(512, 7, torch.device('cpu'))
        second = metrics.deterministic_uniform(512, 7, torch.device('cpu'))
        third = metrics.deterministic_uniform(512, 8, torch.device('cpu'))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))
        self.assertTrue(bool(((first >= 0.0) & (first < 1.0)).all()))
        self.assertGreater(float(first.std()), 0.2)

    def test_rank_normalize_spans_the_unit_interval(self):
        values = torch.tensor([5.0, -1.0, 3.0, 100.0])
        ranked = metrics.rank_normalize(values)
        self.assertAlmostEqual(float(ranked.min()), 0.0)
        self.assertAlmostEqual(float(ranked.max()), 1.0)
        self.assertEqual(int(ranked.argmax()), 3)
        self.assertEqual(int(ranked.argmin()), 1)


class GroupStructureTests(unittest.TestCase):
    def test_group_magnitudes_split_the_convrot_groups(self):
        delta = torch.zeros(2, WIDTH)
        delta[0, :metrics.CONVROT_GROUP] = 1.0
        delta[1, metrics.CONVROT_GROUP:2 * metrics.CONVROT_GROUP] = 2.0
        energy = metrics.group_magnitudes(delta)
        self.assertEqual(tuple(energy.shape), (2, GROUPS))
        self.assertAlmostEqual(float(energy[0, 0]), float(metrics.CONVROT_GROUP))
        self.assertAlmostEqual(
            float(energy[1, 1]),
            4.0 * metrics.CONVROT_GROUP,
        )
        self.assertAlmostEqual(float(energy[0, 1]), 0.0)

    def test_misaligned_width_is_rejected(self):
        with self.assertRaises(metrics.Stage0Error):
            metrics.group_magnitudes(torch.zeros(2, WIDTH + 1))

    def test_shared_mask_matches_token_mask_when_delta_is_shared(self):
        energy = torch.zeros(128, GROUPS)
        energy[:, :14] = 1.0
        rows = metrics.group_concentration(energy, budgets=(14,))
        for row in rows:
            self.assertAlmostEqual(row['delta_mass_captured'], 1.0, places=6)
            self.assertAlmostEqual(row['fraction_of_token_oracle'], 1.0, places=6)

    def test_shared_mask_loses_ground_when_tokens_disagree(self):
        generator = torch.Generator().manual_seed(4)
        energy = torch.rand(128, GROUPS, generator=generator) * 1e-3
        for row in range(128):
            energy[row, torch.randperm(GROUPS, generator=generator)[:14]] = 1.0
        rows = {
            item['granularity']: item
            for item in metrics.group_concentration(energy, budgets=(14,))
        }
        self.assertAlmostEqual(rows['token']['fraction_of_token_oracle'], 1.0)
        self.assertLess(rows['video']['fraction_of_token_oracle'], 0.5)
        self.assertGreaterEqual(
            rows['tile64']['delta_mass_captured'],
            rows['video']['delta_mass_captured'] - 1e-9,
        )

    def test_tile_granularity_beats_video_granularity_on_tiled_structure(self):
        energy = torch.full((128, GROUPS), 1e-4)
        energy[:64, :14] = 1.0
        energy[64:, 20:34] = 1.0
        rows = {
            item['granularity']: item
            for item in metrics.group_concentration(energy, budgets=(14,))
        }
        self.assertAlmostEqual(
            rows['tile64']['fraction_of_token_oracle'],
            1.0,
            places=3,
        )
        self.assertLess(rows['video']['delta_mass_captured'], 0.6)

    def test_pairwise_jaccard_is_one_for_a_shared_mask(self):
        energy = torch.zeros(64, GROUPS)
        energy[:, :7] = 1.0
        self.assertAlmostEqual(
            metrics.pairwise_group_jaccard(energy, 7, pairs=64),
            1.0,
            places=6,
        )

    def test_pairwise_jaccard_falls_for_disjoint_masks(self):
        energy = torch.full((64, GROUPS), 1e-6)
        for row in range(64):
            start = (row * 7) % (GROUPS - 7)
            energy[row, start:start + 7] = 1.0
        value = metrics.pairwise_group_jaccard(energy, 7, pairs=256)
        self.assertLess(value, 0.6)

    def test_block_granularity_requires_divisible_rows(self):
        with self.assertRaises(metrics.Stage0Error):
            metrics.group_mask_for(torch.rand(100, GROUPS), 'tile64', 7)

    def test_decomposition_splits_a_purely_modulation_delta(self):
        delta = torch.randn(8, 32)
        report = metrics.delta_decomposition(
            delta_total=delta,
            delta_modulation=delta,
            delta_state=torch.zeros_like(delta),
        )
        self.assertAlmostEqual(report['modulation_only_fraction'], 1.0, places=5)
        self.assertAlmostEqual(report['state_only_fraction'], 0.0, places=6)
        self.assertAlmostEqual(report['modulation_cosine'], 1.0, places=5)
        self.assertAlmostEqual(
            report['residual_after_modulation_fraction'],
            0.0,
            places=6,
        )

    def test_decomposition_reports_a_state_driven_delta(self):
        delta = torch.randn(8, 32)
        report = metrics.delta_decomposition(
            delta_total=delta,
            delta_modulation=torch.zeros_like(delta),
            delta_state=delta,
        )
        self.assertAlmostEqual(report['state_only_fraction'], 1.0, places=5)
        self.assertAlmostEqual(
            report['residual_after_modulation_fraction'],
            1.0,
            places=5,
        )


class FP8CacheTests(unittest.TestCase):
    def test_roundtrip_is_close_and_uses_row_scales(self):
        values = torch.randn(8, 64)
        values[3] *= 1e-4
        restored = metrics.fp8_roundtrip(values)
        error = metrics.relative_norm(restored - values, values)
        # E4M3 keeps 3 mantissa bits: ~2-3% relative error per row vector.
        self.assertTrue(bool((error < 0.04).all()))
        self.assertTrue(bool((error > 0.0).all()))
        # A tiny row keeps its own scale instead of collapsing to zero.
        self.assertGreater(float(restored[3].abs().max()), 0.0)

    def test_zero_rows_survive(self):
        values = torch.zeros(2, 16)
        restored = metrics.fp8_roundtrip(values)
        self.assertTrue(torch.equal(restored, values))

    def test_output_cache_error_is_gated(self):
        y = torch.randn(4, 32)
        blind = torch.zeros(4, 32)
        # A closed gate discards the row, so cache error cannot reach the output.
        self.assertTrue(bool(
            (metrics.output_cache_error(y, blind, 'float8_e4m3fn') == 0.0).all()
        ))
        gate = torch.ones(4, 32)
        error = metrics.output_cache_error(y, gate, 'float8_e4m3fn')
        self.assertTrue(bool((error < 0.04).all()))

    def test_bf16_beats_fp8_on_every_cached_quantity(self):
        generator = torch.Generator().manual_seed(8)
        y = torch.randn(8, 64, generator=generator)
        gate = torch.ones(8, 64)
        wide = torch.randn(8, WIDTH, generator=generator)
        current = wide + torch.randn(8, WIDTH, generator=generator)
        report = metrics.cache_representation_report(
            y_cached=y,
            gate_rows=gate,
            a_current=current,
            a_cached=wide,
        )
        self.assertEqual(sorted(report), sorted(metrics.CACHE_DTYPES))
        self.assertLess(
            report['bfloat16']['output']['mean'],
            report['float8_e4m3fn']['output']['mean'],
        )
        self.assertLess(
            report['bfloat16']['activation']['delta_relative_error']['mean'],
            report['float8_e4m3fn']['activation']['delta_relative_error']['mean'],
        )

    def test_unknown_cache_dtype_is_rejected(self):
        with self.assertRaises(metrics.Stage0Error):
            metrics.store_roundtrip(torch.randn(2, 4), 'int4')

    def test_activation_cache_reports_ranking_stability(self):
        generator = torch.Generator().manual_seed(2)
        cached = torch.randn(16, WIDTH, generator=generator)
        current = cached + 0.1 * torch.randn(16, WIDTH, generator=generator)
        report = metrics.activation_cache_error(
            current,
            cached,
            'float8_e4m3fn',
        )
        self.assertIn('delta_relative_error', report)
        for k in metrics.GROUP_BUDGETS:
            self.assertIn('group_rank_jaccard_top%d' % k, report)
        self.assertNotIn('fc2_relative_error', report)

    def test_activation_cache_uses_the_fc2_seam_when_given_one(self):
        generator = torch.Generator().manual_seed(3)
        cached = torch.randn(8, WIDTH, generator=generator)
        current = cached + torch.randn(8, WIDTH, generator=generator)
        weight = torch.randn(24, WIDTH, generator=generator)
        report = metrics.activation_cache_error(
            current,
            cached,
            'float8_e4m3fn',
            apply_fc2=lambda value: value @ weight.t(),
        )
        self.assertIn('fc2_relative_error', report)
        self.assertLess(report['fc2_relative_error']['mean'], 0.1)

    def test_cache_quantization_dominates_a_small_delta(self):
        # The metric that matters: FP8 error is fixed to the cached magnitude,
        # so it swamps the delta once the step-to-step change gets small.
        generator = torch.Generator().manual_seed(6)
        cached = torch.randn(8, WIDTH, generator=generator)
        errors = []
        for size in (1.0, 0.1, 0.01):
            current = cached + size * torch.randn(
                8, WIDTH, generator=generator
            )
            report = metrics.activation_cache_error(
                current,
                cached,
                'float8_e4m3fn',
            )
            errors.append(report['delta_relative_error']['mean'])
        self.assertLess(errors[0], errors[1])
        self.assertLess(errors[1], errors[2])


class ModulationRecoveryTests(unittest.TestCase):
    def test_unmodulate_inverts_modulate(self):
        normalized = torch.randn(16, 32)
        shift = torch.randn(16, 32)
        scale = torch.randn(16, 32) * 0.1
        h = _modulate(normalized, shift, scale, torch.float32)
        recovered = _unmodulate(h, shift, scale)
        self.assertTrue(torch.allclose(recovered, normalized, atol=1e-4))

    def test_degenerate_scale_does_not_produce_infinities(self):
        shift = torch.zeros(4, 8)
        scale = torch.full((4, 8), -1.0)
        recovered = _unmodulate(torch.ones(4, 8), shift, scale)
        self.assertTrue(bool(torch.isfinite(recovered).all()))


class SampleBlockTests(unittest.TestCase):
    def setUp(self):
        self.layout = make_layout(
            context_rows=192,
            latent_t=8,
            patch_h=16,
            patch_w=16,
        )

    def test_blocks_are_aligned_and_inside_the_video(self):
        video_start, video_stop = self.layout.video_range
        blocks = sample_blocks(self.layout, 64, 4)
        self.assertEqual(len(blocks), 4)
        for start in blocks:
            self.assertEqual(start % 128, 0)
            self.assertGreaterEqual(start, video_start)
            self.assertLessEqual(start + 128, video_stop)

    def test_blocks_are_stable_and_spread(self):
        first = sample_blocks(self.layout, 64, 4)
        self.assertEqual(first, sample_blocks(self.layout, 64, 4))
        self.assertEqual(len(set(first)), len(first))
        self.assertGreater(max(first) - min(first), 128)

    def test_request_is_clamped_to_what_the_video_holds(self):
        small = make_layout(context_rows=192, latent_t=1, patch_h=4, patch_w=4)
        blocks = sample_blocks(small, 64, 8)
        self.assertLessEqual(len(blocks), 8)
        for start in blocks:
            self.assertLessEqual(start + 128, small.video_range[1])


class ConfigTests(unittest.TestCase):
    def test_defaults_and_derived_rows(self):
        config = Stage0Config()
        self.assertEqual(config.sample_rows, 4 * 128)
        self.assertTrue(config.measure_cache)
        self.assertEqual(config.signature, Stage0Config().signature)

    def test_invalid_values_are_rejected(self):
        for kwargs in (
            {'sample_blocks': 0},
            {'sample_blocks': True},
            {'cache_step_stride': 0},
            {'start_step': -1},
            {'run_tag': 'bad tag'},
            {'layers': '0,0'},
            {'layers': '50'},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    Stage0Config(**kwargs)


class PoolSummaryTests(unittest.TestCase):
    def test_pooling_averages_across_layers_and_steps(self):
        selection = [
            {
                'selectors': [
                    {
                        'selector': 'hit_rate',
                        'tile_budget': 0.5,
                        'oracle_mass_captured': value,
                        'oracle_topk_overlap': value / 2.0,
                    }
                ]
            }
            for value in (0.6, 0.8)
        ]
        pooled = pool_summary(selection, [])
        row = pooled['selection'][0]
        self.assertEqual(row['selector'], 'hit_rate')
        self.assertAlmostEqual(row['oracle_mass_captured'], 0.7)
        self.assertAlmostEqual(row['oracle_topk_overlap'], 0.35)
        self.assertEqual(row['samples'], 2)

    def test_pooling_tolerates_a_selection_only_run(self):
        pooled = pool_summary([], [])
        self.assertEqual(pooled['selection'], [])
        self.assertEqual(pooled['group_sharing'], [])
        for dtype in metrics.CACHE_DTYPES:
            self.assertIsNone(pooled['cache'][dtype]['output_relative_error'])

    def test_pooling_carries_cache_statistics(self):
        cache = [{
            'group_concentration': [{
                'granularity': 'video',
                'groups': 14,
                'delta_mass_captured': 0.5,
                'fraction_of_token_oracle': 0.9,
            }],
            'pairwise_group_jaccard': {'7': 0.4, '14': 0.5, '28': 0.6},
            'delta_decomposition': {
                'modulation_only_fraction': 0.8,
                'state_only_fraction': 0.3,
                'modulation_cosine': 0.9,
                'state_cosine': 0.2,
                'residual_after_modulation_fraction': 0.25,
            },
            'cache_representations': {
                'float8_e4m3fn': {
                    'output': {'mean': 0.01},
                    'activation': {
                        'delta_relative_error': {'mean': 0.02},
                        'group_rank_jaccard_top7': 0.95,
                        'group_rank_jaccard_top14': 0.96,
                        'group_rank_jaccard_top28': 0.97,
                    },
                },
            },
        }]
        pooled = pool_summary([], cache)
        self.assertAlmostEqual(
            pooled['group_sharing'][0]['fraction_of_token_oracle'],
            0.9,
        )
        self.assertAlmostEqual(
            pooled['delta_decomposition']['modulation_only_fraction'],
            0.8,
        )
        fp8 = pooled['cache']['float8_e4m3fn']
        self.assertAlmostEqual(fp8['output_relative_error'], 0.01)
        self.assertAlmostEqual(fp8['group_rank_jaccard_top7'], 0.95)
        self.assertIsNone(fp8['fc2_relative_error'])
        self.assertIsNone(
            pooled['cache']['bfloat16']['output_relative_error']
        )
        self.assertAlmostEqual(
            pooled['token_pair_group_overlap']['group_rank_jaccard_top7'],
            0.4,
        )


class SessionTests(unittest.TestCase):
    def test_concurrent_requests_are_refused(self):
        session = Stage0Session(Stage0Config())
        session.begin_request(torch.linspace(1.0, 0.0, 5))
        with self.assertRaises(Exception):
            session.begin_request(torch.linspace(1.0, 0.0, 5))

    def test_schedule_is_captured_for_sigma_lookup(self):
        session = Stage0Session(Stage0Config())
        session.begin_request(torch.tensor([1.0, 0.5, 0.0]))
        self.assertEqual(session.schedule, (1.0, 0.5, 0.0))
        self.assertAlmostEqual(session._sigma(1), 0.5)
        self.assertIsNone(session._sigma(9))
        self.assertIsNone(session._sigma(-1))

    def test_notes_are_deduplicated(self):
        session = Stage0Session(Stage0Config())
        session._note('same')
        session._note('same')
        session._note('other')
        self.assertEqual(session.notes, ['same', 'other'])


if __name__ == '__main__':
    unittest.main()
