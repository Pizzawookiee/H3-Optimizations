'''CPU contracts for budget-selected Sparse Kitchen geometry.'''

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.attention.sparse.config import HybridSparseConfig  # noqa: E402
from h3_optimizations.attention.sparse.kitchen_hybrid import (  # noqa: E402
    FASTER_GEOMETRY,
    QUALITY_GEOMETRY,
    HybridKitchenProjected,
    HybridKitchenQKVProjector,
    HybridSparseKitchenBackend,
)
import h3_optimizations.attention.sparse.kitchen_hybrid as hybrid  # noqa: E402


class _Projector:
    routing_summaries = True
    stream_output = True
    streamed_q = True
    strided_qk_input = True
    chunk_rows = 4096
    v_mode = 'retain'

    def __init__(self, name):
        self.name = name
        self.installation_signature = (name,)
        self.calls = []

    def try_project(self, module, x, rope_freqs, **kwargs):
        self.calls.append((module, x, rope_freqs, kwargs))
        return self.name


class _Backend:
    stream_output = True
    release_carrier_before_out_proj = True
    output_layout = 'nhd'

    def __init__(self, name):
        self.name = name
        self.installation_signature = (name,)
        self.calls = []

    def prepare_projected(self, projected, **kwargs):
        self.calls.append(('prepare_projected', projected, kwargs))
        return '%s-prepared' % self.name

    def prepare(self, q, k, v, **kwargs):
        self.calls.append(('prepare', q, k, v, kwargs))
        return '%s-prepared' % self.name

    def execute_projected(self, module, prepared):
        self.calls.append(('execute_projected', module, prepared))
        return '%s-output' % self.name

    def as_status(self):
        return {'backend': self.name}


class KitchenHybridTests(unittest.TestCase):
    def setUp(self):
        self.config = HybridSparseConfig(
            step_video_budgets=(0.19, 0.20),
        )
        self.quality_projector = _Projector('quality-projector')
        self.faster_projector = _Projector('faster-projector')
        self.projector = HybridKitchenQKVProjector(
            self.config,
            self.quality_projector,
            self.faster_projector,
        )

    def _snapshot(self, step):
        return mock.patch.object(
            hybrid,
            'snapshot_for',
            return_value=SimpleNamespace(step_index=step, total_steps=2),
        )

    def test_projector_uses_64kv_below_twenty_percent(self):
        x = SimpleNamespace(shape=(1024, 3072))
        with self._snapshot(0):
            projected = self.projector.try_project(
                'module',
                x,
                'rope',
                layer_index=3,
                transformer_options={},
            )
        self.assertEqual(projected.geometry, QUALITY_GEOMETRY)
        self.assertEqual(projected.projected, 'quality-projector')
        self.assertEqual(len(self.quality_projector.calls), 1)
        self.assertEqual(len(self.faster_projector.calls), 0)
        self.assertTrue(self.projector.strided_qk_input)

    def test_projector_uses_128kv_at_twenty_percent(self):
        x = SimpleNamespace(shape=(1024, 3072))
        with self._snapshot(1):
            projected = self.projector.try_project(
                'module',
                x,
                'rope',
                layer_index=3,
                transformer_options={},
            )
        self.assertEqual(projected.geometry, FASTER_GEOMETRY)
        self.assertEqual(projected.projected, 'faster-projector')
        self.assertEqual(len(self.quality_projector.calls), 0)
        self.assertEqual(len(self.faster_projector.calls), 1)

    def test_backend_consumes_with_the_projectors_selected_geometry(self):
        quality = _Backend('quality')
        faster = _Backend('faster')
        backend = HybridSparseKitchenBackend(
            self.config,
            quality,
            faster,
            self.projector,
        )
        prepared = backend.prepare_projected(
            HybridKitchenProjected(FASTER_GEOMETRY, 'carrier'),
            layer_index=5,
            transformer_options={},
        )
        self.assertEqual(prepared.geometry, FASTER_GEOMETRY)
        self.assertEqual(prepared.prepared, 'faster-prepared')
        self.assertEqual(len(quality.calls), 0)
        self.assertEqual(faster.calls[0][0], 'prepare_projected')
        self.assertEqual(
            backend.execute_projected('module', prepared),
            'faster-output',
        )

    def test_backend_rejects_unknown_wrapped_geometry(self):
        backend = HybridSparseKitchenBackend(
            self.config,
            _Backend('quality'),
            _Backend('faster'),
            self.projector,
        )
        with self.assertRaisesRegex(ValueError, 'unknown hybrid Kitchen geometry'):
            backend.prepare_projected(
                HybridKitchenProjected('unknown', 'carrier'),
                layer_index=5,
                transformer_options={},
            )


if __name__ == '__main__':
    unittest.main()
