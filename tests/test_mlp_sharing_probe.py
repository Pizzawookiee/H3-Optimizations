import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h3_optimizations.mlp_sharing.config import (  # noqa: E402
    DEFAULT_LAYERS,
    MLPSharingConfig,
    MLPSharingProbeConfig,
    parse_execution_layers,
    parse_layers,
)
from h3_optimizations.mlp_sharing.metrics import (  # noqa: E402
    measure_chunk,
    spatial_cells_1x2x4,
    spatial_cells_1x2x2,
    summarize_measurements,
)


class MLPSharingProbeTests(unittest.TestCase):
    def test_config_parses_and_validates_layers(self):
        self.assertEqual(parse_layers('49,0,10'), (0, 10, 49))
        self.assertEqual(MLPSharingProbeConfig().layers, DEFAULT_LAYERS)
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            parse_layers('1,1')
        with self.assertRaisesRegex(ValueError, r'\[0, 49\]'):
            parse_layers('50')
        with self.assertRaisesRegex(ValueError, 'run_tag'):
            MLPSharingProbeConfig(run_tag='../escape')

        self.assertEqual(parse_execution_layers('all'), tuple(range(50)))
        sharing = MLPSharingConfig(removal_fraction='87.5%', start_after_step=3)
        self.assertEqual(sharing.removal_fraction, 0.875)
        self.assertEqual(sharing.geometry, (1, 2, 4))
        with self.assertRaisesRegex(ValueError, 'removal_fraction'):
            MLPSharingConfig(removal_fraction=0.6)

    def test_geometry_uses_real_target_video_coordinates(self):
        layout = SimpleNamespace(
            video_range=(10, 26),
            video_shape=(1, 4, 4),
        )
        cells, video_tokens = spatial_cells_1x2x2(layout, 10, 18)
        self.assertEqual(video_tokens, 16)
        self.assertEqual(cells, ((0, 1, 4, 5), (2, 3, 6, 7)))
        split_cells, _ = spatial_cells_1x2x2(layout, 10, 15)
        self.assertEqual(split_cells, ())
        wide_cells, _ = spatial_cells_1x2x4(layout, 10, 26)
        self.assertEqual(
            wide_cells,
            ((0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11, 12, 13, 14, 15)),
        )

    def test_measurement_is_output_exact_and_filters_modulation(self):
        h = torch.tensor(
            [[1.0, 0.0], [1.1, 0.0], [0.0, 1.0], [0.0, 1.2]],
            dtype=torch.float32,
        )
        y = h * 2.0
        residual = torch.ones_like(h)
        gate = torch.tensor([[0.5, 0.25], [0.25, 0.5]])
        h_before = h.clone()
        y_before = y.clone()
        residual_before = residual.clone()

        measured = measure_chunk(
            h=h,
            y=y,
            residual=residual,
            gate=gate,
            selector=0,
            cells=((0, 1, 2, 3),),
            evaluate_mlp=lambda value: value * 2.0,
            include_mean_input=True,
            mean_batch_rows=64,
        )
        self.assertEqual(tuple(measured.shape), (1, 6, 9))
        self.assertTrue(torch.equal(h, h_before))
        self.assertTrue(torch.equal(y, y_before))
        self.assertTrue(torch.equal(residual, residual_before))
        self.assertTrue(torch.isfinite(measured).all())

        filtered = measure_chunk(
            h=h,
            y=y,
            residual=residual,
            gate=gate,
            selector=torch.tensor([0, 1, 0, 0]),
            cells=((0, 1, 2, 3),),
            evaluate_mlp=lambda value: value * 2.0,
            include_mean_input=True,
            mean_batch_rows=64,
        )
        self.assertIsNone(filtered)

    def test_summary_contains_all_selectors_and_main_ratios(self):
        values = np.full((3, 6, 9), 0.01, dtype=np.float32)
        values[:, :, 0] = np.arange(18, dtype=np.float32).reshape(3, 6) / 100.0
        rows = summarize_measurements(
            values,
            request_id=0,
            step_index=2,
            total_steps=10,
            sigma=0.5,
            layer_index=5,
            evaluation_index=0,
            total_video_tokens=12,
        )
        self.assertEqual({row['selector'] for row in rows}, {
            'output_oracle', 'input_cosine', 'input_l2', 'random_local'
        })
        self.assertEqual({row['reconstruction'] for row in rows}, {
            'representative', 'mean_input'
        })
        self.assertIn(0.25, {row['target_merge_fraction'] for row in rows})
        self.assertIn(0.5, {row['target_merge_fraction'] for row in rows})
        maximum = max(row['merge_fraction'] for row in rows)
        self.assertEqual(maximum, 0.5)


if __name__ == '__main__':
    unittest.main()
