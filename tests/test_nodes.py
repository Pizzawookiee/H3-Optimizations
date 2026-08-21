'''CPU-only schema, disabled-node, and non-H3 no-op tests.'''

import asyncio
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.apply import apply_plan  # noqa: E402
from h3_optimizations.environment import RuntimeEnvironment  # noqa: E402
from h3_optimizations.nodes import (  # noqa: E402
    H3MemoryOptimization,
    H3OptimizationsExtension,
    H3SparseAttention,
)
from h3_optimizations.plan import H3OptimizationPlan  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def input_by_id(schema, input_id):
    return next(item for item in schema.inputs if item.id == input_id)


class NodeTests(unittest.TestCase):
    def test_public_schemas_are_small_and_stable(self):
        memory = H3MemoryOptimization.define_schema()
        sparse = H3SparseAttention.define_schema()
        self.assertEqual(memory.node_id, 'H3MemoryOptimization')
        self.assertEqual(memory.display_name, 'H3 Memory Optimization')
        self.assertEqual(
            memory.category,
            'H3-Optimizations/Model Patches',
        )
        self.assertEqual(
            [item.id for item in memory.inputs],
            [
                'model',
                'fused_qkv',
                'mlp_memory',
                'chunk_rows',
            ],
        )
        self.assertEqual(
            input_by_id(memory, 'fused_qkv').options,
            ['auto', 'off'],
        )
        self.assertEqual(
            input_by_id(memory, 'mlp_memory').options,
            ['auto', 'off'],
        )
        self.assertEqual(sparse.node_id, 'H3SparseAttention')
        self.assertEqual(sparse.display_name, 'H3 Sparse Attention')
        self.assertEqual(
            sparse.category,
            'H3-Optimizations/Model Patches',
        )
        self.assertEqual(
            [item.id for item in sparse.inputs],
            [
                'model',
                'video_budget',
                'denser_early_late_steps',
            ],
        )
        self.assertFalse(
            input_by_id(sparse, 'denser_early_late_steps').default
        )
        self.assertNotIn('Experimental', memory.display_name)
        self.assertNotIn('Experimental', sparse.display_name)

    def test_extension_exposes_only_the_two_production_nodes(self):
        nodes = asyncio.run(H3OptimizationsExtension().get_node_list())
        self.assertEqual(nodes, [H3MemoryOptimization, H3SparseAttention])

    def test_non_h3_models_do_not_probe_the_runtime(self):
        class OtherModel:
            model_options = {}

            @staticmethod
            def get_model_object(_name):
                raise KeyError('not a diffusion model')

        model = OtherModel()
        with patch.object(
            RuntimeEnvironment,
            'detect',
            side_effect=AssertionError('CUDA detection must not run'),
        ):
            self.assertIs(
                apply_plan(model, H3OptimizationPlan()),
                model,
            )


if __name__ == '__main__':
    unittest.main()
