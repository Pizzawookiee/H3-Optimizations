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
    H3SparseAttentionAdvanced,
)
from h3_optimizations.plan import H3OptimizationPlan  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def input_by_id(schema, input_id):
    return next(item for item in schema.inputs if item.id == input_id)


class NodeTests(unittest.TestCase):
    def test_public_schemas_are_small_and_stable(self):
        memory = H3MemoryOptimization.define_schema()
        sparse = H3SparseAttention.define_schema()
        advanced = H3SparseAttentionAdvanced.define_schema()
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
        fused_qkv_tooltip = input_by_id(memory, 'fused_qkv').tooltip
        self.assertIn('chunked QKV projection providers', fused_qkv_tooltip)
        self.assertIn('BF16/FP16', fused_qkv_tooltip)
        self.assertEqual(
            input_by_id(memory, 'mlp_memory').options,
            ['auto', 'off'],
        )
        self.assertEqual(input_by_id(memory, 'chunk_rows').default, 4096)
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
        self.assertEqual(input_by_id(sparse, 'video_budget').default, 0.3)
        self.assertFalse(
            input_by_id(sparse, 'denser_early_late_steps').default
        )

        self.assertEqual(advanced.node_id, 'H3SparseAttentionAdvanced')
        self.assertEqual(
            advanced.display_name,
            'H3 Sparse Attention (Advanced)',
        )
        self.assertEqual(
            advanced.category,
            'H3-Optimizations/Model Patches',
        )
        self.assertEqual(
            [item.id for item in advanced.inputs],
            [
                'model',
                'video_budget',
                'early_steps',
                'early_kv',
                'late_steps',
                'late_kv',
                'backend',
            ],
        )
        self.assertEqual(
            [item.id for item in advanced.inputs[:6]],
            [
                'model',
                'video_budget',
                'early_steps',
                'early_kv',
                'late_steps',
                'late_kv',
            ],
        )
        backend = input_by_id(advanced, 'backend')
        self.assertEqual(backend.default, 'auto')
        self.assertEqual(
            backend.options,
            ['auto', 'Sparse Sage', 'INT8 Triton', 'FP8 FlexAttention'],
        )
        self.assertIn('hard requirements', advanced.description)
        self.assertIn('Bypass this node', backend.tooltip)
        self.assertEqual(input_by_id(advanced, 'video_budget').default, 0.3)
        self.assertEqual(input_by_id(advanced, 'early_steps').default, 2)
        self.assertEqual(input_by_id(advanced, 'early_kv').default, 0.5)
        self.assertEqual(input_by_id(advanced, 'late_steps').default, 2)
        self.assertEqual(input_by_id(advanced, 'late_kv').default, 0.5)
        self.assertNotIn('Experimental', memory.display_name)
        self.assertNotIn('Experimental', sparse.display_name)
        self.assertNotIn('Experimental', advanced.display_name)

    def test_extension_exposes_three_production_nodes(self):
        nodes = asyncio.run(H3OptimizationsExtension().get_node_list())
        self.assertEqual(
            nodes,
            [
                H3MemoryOptimization,
                H3SparseAttention,
                H3SparseAttentionAdvanced,
            ],
        )

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
