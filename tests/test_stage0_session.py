'''End-to-end Stage 0 session flow driven with synthetic block tensors.'''

import json
import os
from pathlib import Path
import sys
import tempfile
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
from h3_optimizations.mlp_sharing import stage0_metrics as metrics  # noqa: E402
from h3_optimizations.mlp_sharing.config import Stage0Config  # noqa: E402
from h3_optimizations.mlp_sharing.route import (  # noqa: E402
    ROUTE_KEY,
    AttentionRouteRecorder,
)
from h3_optimizations.mlp_sharing.stage0 import Stage0Session  # noqa: E402
from h3_optimizations.runtime.context import (  # noqa: E402
    RUNTIME_KEY,
    RuntimeSnapshot,
)
from test_attention_route import make_layout  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]

HIDDEN = 16
FFN = 4 * metrics.CONVROT_GROUP
KV_TILE = 64
LAYER = 5


class FakeMLP:
    '''A tiny but structurally faithful stand-in for the H3 gated MLP.'''

    def __init__(self, seed):
        generator = torch.Generator().manual_seed(seed)
        self.fc1 = torch.randn(2 * FFN, HIDDEN, generator=generator) * 0.1
        self.fc2 = torch.randn(HIDDEN, FFN, generator=generator) * 0.1

    def activation(self, h):
        expanded = h.float() @ self.fc1.t()
        gate, up = expanded.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    def forward(self, h):
        return self.activation(h) @ self.fc2.t()

    def apply_fc2(self, activated):
        return activated.float() @ self.fc2.t()


class Stage0FlowTests(unittest.TestCase):
    def setUp(self):
        self.layout = make_layout(
            context_rows=192,
            latent_t=8,
            patch_h=16,
            patch_w=16,
        )
        self.rows = self.layout.seq_len
        self.router = SparseTileRouter(q_tile=128, kv_tile=KV_TILE)
        self.geometry = self.router.geometry(self.layout)
        self.mlp = FakeMLP(seed=11)
        self.generator = torch.Generator().manual_seed(3)
        # Two modulation rows so scalar and per-token selectors both exercise.
        self.shift = torch.randn(2, HIDDEN, generator=self.generator) * 0.05
        self.scale = torch.randn(2, HIDDEN, generator=self.generator) * 0.05
        self.gate = torch.randn(2, HIDDEN, generator=self.generator) * 0.5
        self.state = torch.randn(
            self.rows,
            HIDDEN,
            generator=self.generator,
        )

    def _options(self, step_index, video_budget=0.5):
        recorder = AttentionRouteRecorder(kv_tile=KV_TILE)
        q = torch.randn(
            1,
            4,
            self.geometry.q_tiles,
            8,
            generator=self.generator,
        )
        k = torch.randn(
            1,
            4,
            self.geometry.kv_tiles,
            8,
            generator=self.generator,
        )
        self.router.build_lut_from_summaries(
            q,
            k,
            self.layout,
            video_budget,
            sink=recorder.sink(LAYER),
        )
        attn_out = torch.randn(
            self.rows,
            HIDDEN,
            generator=self.generator,
        )
        recorder.record_attention_energy(
            LAYER,
            attn_out,
            self.gate,
            [(0, self.rows, 0)],
            4096,
        )
        snapshot = RuntimeSnapshot(
            request_id=0,
            step_index=int(step_index),
            total_steps=4,
            layout=self.layout,
            compute_dtype=torch.float32,
            device=torch.device('cpu'),
            error=None,
        )
        return {RUNTIME_KEY: snapshot, ROUTE_KEY: recorder}

    def _drive_step(self, session, step_index, chunk_rows=None, **kwargs):
        '''Run one whole block forward's worth of chunks through the session.'''
        options = self._options(step_index, **kwargs)
        # The state drifts between steps the way a denoised latent would.
        self.state = self.state + 0.2 * torch.randn(
            self.rows,
            HIDDEN,
            generator=self.generator,
        )
        chunk_rows = chunk_rows or self.rows
        for start in range(0, self.rows, chunk_rows):
            stop = min(start + chunk_rows, self.rows)
            normalized = self.state[start:stop]
            h = normalized * (1.0 + self.scale[0]) + self.shift[0]
            y = self.mlp.forward(h)
            session.observe_exact_mlp(
                LAYER,
                options,
                h=h,
                y=y,
                residual=self.state[start:stop],
                gate=self.gate,
                shift=self.shift,
                scale=self.scale,
                selector=0,
                chunk_start=start,
                chunk_stop=stop,
                evaluate_mlp=lambda value: self.mlp.forward(value),
                evaluate_activation=self.mlp.activation,
                apply_fc2=self.mlp.apply_fc2,
                mlp_path='module',
            )
        session.end_mlp_block(LAYER, options)
        return options

    def _run(self, config, steps=3, **kwargs):
        session = Stage0Session(config)
        session.begin_request(torch.linspace(1.0, 0.0, steps + 1))
        for step in range(steps):
            self._drive_step(session, step, **kwargs)
        return session

    # -- Tier A -----------------------------------------------------------

    def test_selection_records_cover_every_step(self):
        session = self._run(
            Stage0Config(layers=(LAYER,), measure_cache=False),
            steps=3,
        )
        self.assertEqual(len(session.selection_records), 3)
        for index, record in enumerate(session.selection_records):
            self.assertEqual(record['step_index'], index)
            self.assertEqual(record['layer'], LAYER)
            self.assertFalse(record['route_dense'])
            self.assertEqual(
                record['video_kv_tiles'],
                self.geometry.pure_video_kv_tiles,
            )
            self.assertGreater(record['oracle_total_energy'], 0.0)
            self.assertEqual(
                sorted(record['available_signals']),
                [
                    'attention_update_energy',
                    'combined',
                    'hit_rate',
                    'oracle',
                    'random',
                ],
            )
            self.assertEqual(
                len(record['selectors']),
                len(metrics.TILE_BUDGETS) * 5,
            )

    def test_oracle_bounds_every_other_selector(self):
        session = self._run(
            Stage0Config(layers=(LAYER,), measure_cache=False),
            steps=2,
        )
        for record in session.selection_records:
            by_budget = {}
            for row in record['selectors']:
                by_budget.setdefault(row['tile_budget'], []).append(row)
            for budget, rows in by_budget.items():
                best = next(
                    row for row in rows if row['selector'] == 'oracle'
                )['oracle_mass_captured']
                for row in rows:
                    self.assertLessEqual(
                        row['oracle_mass_captured'],
                        best + 1e-9,
                        '%s exceeded the oracle at budget %s'
                        % (row['selector'], budget),
                    )

    def test_chunking_does_not_change_the_selection_result(self):
        whole = self._run(
            Stage0Config(layers=(LAYER,), measure_cache=False, selector_seed=1),
            steps=1,
        )
        self.setUp()
        split = self._run(
            Stage0Config(layers=(LAYER,), measure_cache=False, selector_seed=1),
            steps=1,
            chunk_rows=320,
        )
        self.assertAlmostEqual(
            whole.selection_records[0]['oracle_total_energy'],
            split.selection_records[0]['oracle_total_energy'],
            places=3,
        )
        for left, right in zip(
            whole.selection_records[0]['selectors'],
            split.selection_records[0]['selectors'],
        ):
            self.assertEqual(left['selector'], right['selector'])
            self.assertAlmostEqual(
                left['oracle_mass_captured'],
                right['oracle_mass_captured'],
                places=6,
            )

    def test_a_dense_route_takes_no_selector_statistics(self):
        session = self._run(
            Stage0Config(layers=(LAYER,), measure_cache=False),
            steps=1,
            video_budget=1.0,
        )
        record = session.selection_records[0]
        self.assertTrue(record['route_dense'])
        self.assertEqual(record['selectors'], [])
        self.assertEqual(record['available_signals'], [])

    def test_unmeasured_layers_are_ignored(self):
        session = self._run(
            Stage0Config(layers=(LAYER + 1,), measure_cache=False),
            steps=2,
        )
        self.assertEqual(session.selection_records, [])

    # -- Tier B/C ---------------------------------------------------------

    def test_cache_records_start_after_the_first_held_step(self):
        session = self._run(
            Stage0Config(
                layers=(LAYER,),
                measure_cache=True,
                sample_blocks=2,
                start_step=0,
            ),
            steps=3,
        )
        # Step 0 only seeds the cache; steps 1 and 2 can measure a delta.
        self.assertEqual(len(session.cache_records), 2)
        for record in session.cache_records:
            self.assertEqual(record['layer'], LAYER)
            self.assertEqual(record['previous_step'], record['step_index'] - 1)
            self.assertEqual(record['sampled_rows'], 256)
            self.assertEqual(len(record['sampled_block_starts']), 2)

    def test_cache_record_carries_every_headline_statistic(self):
        session = self._run(
            Stage0Config(
                layers=(LAYER,),
                measure_cache=True,
                sample_blocks=2,
                start_step=0,
            ),
            steps=2,
        )
        record = session.cache_records[0]
        granularities = {
            row['granularity'] for row in record['group_concentration']
        }
        self.assertEqual(set(metrics.GRANULARITIES), granularities)
        for row in record['group_concentration']:
            self.assertGreaterEqual(row['delta_mass_captured'], 0.0)
            self.assertLessEqual(row['delta_mass_captured'], 1.0 + 1e-9)
            self.assertLessEqual(row['fraction_of_token_oracle'], 1.0 + 1e-9)
        self.assertEqual(
            sorted(record['pairwise_group_jaccard']),
            ['14', '28', '7'],
        )
        decomposition = record['delta_decomposition']
        self.assertGreater(decomposition['total_delta_energy'], 0.0)
        for field in ('modulation_only_fraction', 'state_only_fraction'):
            self.assertGreaterEqual(decomposition[field], 0.0)
        representations = record['cache_representations']
        self.assertEqual(sorted(representations), sorted(metrics.CACHE_DTYPES))
        for dtype, entry in representations.items():
            self.assertIn('fc2_relative_error', entry['activation'])
            self.assertGreater(entry['output']['mean'], 0.0, dtype)

    def test_state_dominates_when_modulation_is_frozen(self):
        self.scale = self.scale * 0.0
        self.shift = self.shift * 0.0
        session = self._run(
            Stage0Config(
                layers=(LAYER,),
                measure_cache=True,
                sample_blocks=2,
                start_step=0,
            ),
            steps=2,
        )
        decomposition = session.cache_records[0]['delta_decomposition']
        # The held sample is exact, so a frozen AdaLN leaves no residue.
        self.assertLess(decomposition['modulation_only_fraction'], 1e-9)
        self.assertAlmostEqual(
            decomposition['state_only_fraction'],
            1.0,
            places=4,
        )

    def test_cache_measurement_can_be_switched_off(self):
        session = self._run(
            Stage0Config(layers=(LAYER,), measure_cache=False),
            steps=3,
        )
        self.assertEqual(session.cache_records, [])

    def test_sampled_blocks_are_the_same_rows_every_step(self):
        session = self._run(
            Stage0Config(
                layers=(LAYER,),
                measure_cache=True,
                sample_blocks=2,
                start_step=0,
            ),
            steps=3,
        )
        starts = {
            tuple(record['sampled_block_starts'])
            for record in session.cache_records
        }
        self.assertEqual(len(starts), 1)

    def test_a_missing_activation_seam_is_noted_not_fatal(self):
        session = Stage0Session(
            Stage0Config(layers=(LAYER,), measure_cache=True, start_step=0)
        )
        session.begin_request(torch.linspace(1.0, 0.0, 3))
        options = self._options(0)
        h = self.state * (1.0 + self.scale[0]) + self.shift[0]
        session.observe_exact_mlp(
            LAYER,
            options,
            h=h,
            y=self.mlp.forward(h),
            residual=self.state,
            gate=self.gate,
            selector=0,
            chunk_start=0,
            chunk_stop=self.rows,
            evaluate_mlp=lambda value: self.mlp.forward(value),
            mlp_path='convrot',
        )
        session.end_mlp_block(LAYER, options)
        self.assertEqual(session.cache_records, [])
        self.assertEqual(len(session.selection_records), 1)
        self.assertTrue(
            any('activation seam' in note for note in session.notes)
        )

    # -- report -----------------------------------------------------------

    def test_end_request_writes_a_complete_report(self):
        session = self._run(
            Stage0Config(
                layers=(LAYER,),
                measure_cache=True,
                sample_blocks=2,
                start_step=0,
                run_tag='flow-test',
            ),
            steps=3,
        )
        with tempfile.TemporaryDirectory() as root:
            session._output_directory = lambda: os.path.join(root, 'report')
            session.end_request()
            directory = Path(session.last_report_directory)
            selection = [
                json.loads(line)
                for line in (directory / 'selection.jsonl').read_text(
                    encoding='utf-8'
                ).splitlines()
            ]
            cache = [
                json.loads(line)
                for line in (directory / 'cache.jsonl').read_text(
                    encoding='utf-8'
                ).splitlines()
            ]
            summary = json.loads(
                (directory / 'summary.json').read_text(encoding='utf-8')
            )
        self.assertEqual(len(selection), 3)
        self.assertEqual(len(cache), 2)
        self.assertTrue(summary['completed'])
        self.assertTrue(summary['output_exact'])
        self.assertIsNone(summary['error'])
        pooled = summary['pooled']
        self.assertEqual(
            len(pooled['selection']),
            len(metrics.TILE_BUDGETS) * 5,
        )
        self.assertTrue(pooled['group_sharing'])
        self.assertIsNotNone(
            pooled['delta_decomposition']['modulation_only_fraction']
        )
        for dtype in metrics.CACHE_DTYPES:
            self.assertIsNotNone(
                pooled['cache'][dtype]['output_relative_error']
            )

    def test_an_unfinished_block_is_reported_as_an_error(self):
        session = Stage0Session(
            Stage0Config(layers=(LAYER,), measure_cache=False)
        )
        session.begin_request(torch.linspace(1.0, 0.0, 3))
        options = self._options(0)
        h = self.state * (1.0 + self.scale[0]) + self.shift[0]
        session.observe_exact_mlp(
            LAYER,
            options,
            h=h,
            y=self.mlp.forward(h),
            residual=self.state,
            gate=self.gate,
            shift=self.shift,
            scale=self.scale,
            selector=0,
            chunk_start=0,
            chunk_stop=self.rows,
            evaluate_mlp=lambda value: self.mlp.forward(value),
            evaluate_activation=self.mlp.activation,
            mlp_path='module',
        )
        with tempfile.TemporaryDirectory() as root:
            session._output_directory = lambda: os.path.join(root, 'report')
            session.end_request()
            summary = json.loads(
                (
                    Path(session.last_report_directory) / 'summary.json'
                ).read_text(encoding='utf-8')
            )
        self.assertFalse(summary['completed'])
        self.assertIn('unfinished', summary['error'])


if __name__ == '__main__':
    unittest.main()
