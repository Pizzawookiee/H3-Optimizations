'''CPU-only schema, disabled-node, and non-H3 no-op tests.'''

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
    H3AttentionOrderingProbe,
    H3MLPSharing,
    H3MLPSharingProbe,
    H3MLPSharingProbeOutput,
    H3MLPStage0,
    H3MemoryOptimization,
    H3SparseAttention,
    H3SparseAttentionAdvanced,
)
from h3_optimizations.plan import H3OptimizationPlan  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def input_by_id(schema, input_id):
    return next(item for item in schema.inputs if item.id == input_id)


class NodeTests(unittest.TestCase):
    def test_node_schemas_are_small_and_stable(self):
        memory = H3MemoryOptimization.define_schema()
        ordering = H3AttentionOrderingProbe.define_schema()
        sharing = H3MLPSharing.define_schema()
        probe = H3MLPSharingProbe.define_schema()
        probe_output = H3MLPSharingProbeOutput.define_schema()
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
                'preserve_precision',
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
        preserve_precision = input_by_id(memory, 'preserve_precision')
        self.assertTrue(preserve_precision.default)
        self.assertIn('Do not introduce new quantization', preserve_precision.tooltip)
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
                'layer_video_budgets',
            ],
        )
        self.assertEqual(input_by_id(sparse, 'video_budget').default, 0.3)
        self.assertFalse(
            input_by_id(sparse, 'denser_early_late_steps').default
        )
        layer_budgets = input_by_id(sparse, 'layer_video_budgets')
        self.assertEqual(layer_budgets.default, '')
        self.assertTrue(layer_budgets.optional)

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
        self.assertEqual(backend.default, 'Kitchen INT8')
        self.assertEqual(
            backend.options,
            [
                'Kitchen INT8',
                'Sparse Sage',
                'INT8 Triton',
                'FP8 FlexAttention',
            ],
        )
        self.assertIn('hard requirements', advanced.description)
        self.assertIn('Kitchen INT8 (64Q x 64KV)', advanced.description)
        self.assertIn('Bypass this node', backend.tooltip)
        self.assertIn(
            'INT8 Triton and FP8 FlexAttention also use 64Q x 64KV',
            backend.tooltip,
        )
        self.assertNotIn('experimental', ' '.join(backend.options).lower())
        self.assertTrue(H3SparseAttentionAdvanced.validate_inputs('auto'))
        self.assertTrue(
            H3SparseAttentionAdvanced.validate_inputs(
                'Native INT8 128x128 + Sol residual 64x64'
            )
        )
        self.assertIsInstance(
            H3SparseAttentionAdvanced.validate_inputs('unknown backend'),
            str,
        )
        self.assertEqual(input_by_id(advanced, 'video_budget').default, 0.3)
        self.assertEqual(input_by_id(advanced, 'early_steps').default, 2)
        self.assertEqual(input_by_id(advanced, 'early_kv').default, 0.5)
        self.assertEqual(input_by_id(advanced, 'late_steps').default, 2)
        self.assertEqual(input_by_id(advanced, 'late_kv').default, 0.5)
        self.assertNotIn('Experimental', memory.display_name)
        self.assertNotIn('Experimental', sparse.display_name)
        self.assertNotIn('Experimental', advanced.display_name)
        self.assertEqual(probe.node_id, 'H3MLPSharingProbe')
        self.assertEqual(probe.category, 'H3-Optimizations/Experiments')
        self.assertEqual(sharing.node_id, 'H3MLPSharing')
        self.assertEqual(sharing.category, 'H3-Optimizations/Experiments')
        self.assertEqual(
            [item.id for item in sharing.inputs],
            [
                'model',
                'enabled',
                'removal_fraction',
                'selector',
                'start_after_step',
                'layers',
                'selector_seed',
                'run_tag',
            ],
        )
        self.assertEqual(input_by_id(sharing, 'removal_fraction').default, '50%')
        self.assertEqual(input_by_id(sharing, 'start_after_step').default, 3)
        self.assertEqual(probe_output.node_id, 'H3MLPSharingProbeOutput')
        self.assertTrue(probe_output.is_output_node)
        self.assertEqual([item.id for item in probe_output.inputs], ['samples'])
        self.assertEqual(
            [item.id for item in probe.inputs],
            [
                'model',
                'enabled',
                'layers',
                'include_mean_input',
                'mean_batch_rows',
                'run_tag',
            ],
        )
        self.assertEqual(ordering.node_id, 'H3AttentionOrderingProbe')
        self.assertEqual(ordering.category, 'H3-Optimizations/Experiments')
        self.assertEqual(
            [item.id for item in ordering.inputs],
            [
                'model',
                'enabled',
                'layers',
                'steps',
                'video_budgets',
                'query_samples',
                'head_chunk',
                'capture_uncond',
                'run_tag',
            ],
        )
        self.assertEqual(input_by_id(ordering, 'video_budgets').default, '20,30,50')
        self.assertEqual(input_by_id(ordering, 'query_samples').default, 64)

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
