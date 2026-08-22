'''CPU contracts for executable target-video MLP output sharing.'''

from pathlib import Path
from types import SimpleNamespace
import os
import sys
import unittest

import torch

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.mlp_sharing.config import MLPSharingConfig  # noqa: E402
from h3_optimizations.mlp_sharing.execution import MLPSharingSession  # noqa: E402
from h3_optimizations.runtime.context import RUNTIME_KEY, RuntimeSnapshot  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class RecordingMLP:
    def __init__(self):
        self.rows = []

    def __call__(self, value):
        self.rows.append(int(value.shape[0]))
        return value * 10.0, None, 'recording'


def options(step, layout, total_steps=20):
    return {
        RUNTIME_KEY: RuntimeSnapshot(
            request_id=0,
            step_index=step,
            total_steps=total_steps,
            layout=layout,
            compute_dtype=torch.float32,
            device=torch.device('cpu'),
        )
    }


class MLPSharingExecutionTests(unittest.TestCase):
    @staticmethod
    def _layout(video_start=0, temporal=1, height=4, width=4):
        rows = temporal * height * width
        return SimpleNamespace(
            video_range=(video_start, video_start + rows),
            video_shape=(temporal, height, width),
        )

    @staticmethod
    def _hidden(rows):
        values = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
        return values.add_(1.0)

    def _run(self, fraction, *, step=3, selector='random_local', mod_selector=0):
        config = MLPSharingConfig(
            selector=selector,
            removal_fraction=fraction,
            start_after_step=3,
            selector_seed=91,
        )
        session = MLPSharingSession(config)
        h = self._hidden(16)
        evaluator = RecordingMLP()
        transformer_options = options(step, self._layout())
        out, _expanded, path = session.evaluate_chunk(
            0,
            transformer_options,
            h=h,
            selector=mod_selector,
            chunk_start=0,
            chunk_stop=16,
            evaluate_mlp=evaluator,
        )
        session.end_mlp_block(0, transformer_options)
        return h, out, path, evaluator, session.records[0]

    def test_zero_percent_and_protected_prefix_are_exact(self):
        h, out, path, evaluator, record = self._run(0.0, step=3)
        self.assertTrue(torch.equal(out, h * 10.0))
        self.assertEqual(evaluator.rows, [16])
        self.assertEqual(path, 'recording')
        self.assertEqual(record['removed_video_rows'], 0)
        self.assertFalse(record['active'])

        h, out, _path, evaluator, record = self._run(0.75, step=2)
        self.assertTrue(torch.equal(out, h * 10.0))
        self.assertEqual(evaluator.rows, [16])
        self.assertEqual(record['removed_video_rows'], 0)
        self.assertFalse(record['active'])

    def test_expected_rows_are_removed_for_each_geometry(self):
        expected = {
            0.25: (12, 4),
            0.5: (8, 8),
            0.75: (4, 12),
            0.875: (2, 14),
        }
        for fraction, (evaluated, removed) in expected.items():
            with self.subTest(fraction=fraction):
                _h, _out, _path, evaluator, record = self._run(fraction)
                self.assertEqual(evaluator.rows, [evaluated])
                self.assertEqual(record['removed_video_rows'], removed)
                self.assertEqual(record['evaluated_video_rows'], evaluated)
                self.assertAlmostEqual(
                    record['realized_target_video_removal_fraction'],
                    fraction,
                )
                self.assertTrue(record['active'])

    def test_random_selector_is_reproducible(self):
        first = self._run(0.5)[1]
        second = self._run(0.5)[1]
        self.assertTrue(torch.equal(first, second))

    def test_per_token_modulation_excludes_incompatible_cells(self):
        selectors = torch.arange(16, dtype=torch.long)
        h, out, _path, evaluator, record = self._run(
            0.75,
            mod_selector=selectors,
        )
        self.assertTrue(torch.equal(out, h * 10.0))
        self.assertEqual(evaluator.rows, [16])
        self.assertEqual(record['eligible_video_rows'], 0)
        self.assertEqual(record['removed_video_rows'], 0)

    def test_non_video_rows_remain_exact(self):
        config = MLPSharingConfig(
            selector='random_local',
            removal_fraction=0.75,
            start_after_step=3,
        )
        session = MLPSharingSession(config)
        h = self._hidden(24)
        evaluator = RecordingMLP()
        transformer_options = options(3, self._layout(video_start=4))
        out, _expanded, _path = session.evaluate_chunk(
            0,
            transformer_options,
            h=h,
            selector=0,
            chunk_start=0,
            chunk_stop=24,
            evaluate_mlp=evaluator,
        )
        self.assertTrue(torch.equal(out[:4], h[:4] * 10.0))
        self.assertTrue(torch.equal(out[20:], h[20:] * 10.0))
        self.assertEqual(evaluator.rows, [12])


if __name__ == '__main__':
    unittest.main()
