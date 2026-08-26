'''CPU-only tests for the production ComfyUI node registry.'''

import asyncio
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.aimdo_limiter import H3AIMDOResidencyLimiter  # noqa: E402
from h3_optimizations.memory_migration_node import (  # noqa: E402
    H3MemoryOptimization,
    PRECISION_MODE_ALLOW_FP8,
    PRECISION_MODE_AUTO,
    PRECISION_MODE_BF16,
    PRECISION_MODE_FORCE_QUANT,
    PRECISION_MODE_OPTIONS,
    PRECISION_MODE_PRESERVE,
    PRECISION_MODE_PRESERVE_NATIVE,
    QKV_STREAMING_MODE_AUTO,
    QKV_STREAMING_MODE_FORCED,
    QKV_STREAMING_MODE_OFF,
    _memory_request_for_modes,
    _normalize_precision_mode,
    _qkv_streaming_request,
)
from h3_optimizations.nodes import (  # noqa: E402
    H3SparseAttention,
    H3SparseAttentionAdvanced,
    _memory_request,
)
from h3_optimizations.plan import (  # noqa: E402
    ATTENTION_AUTO,
    ATTENTION_EXISTING,
    FUSED_QKV_AUTO,
    FUSED_QKV_FORCE_BF16,
    FUSED_QKV_FORCE_QUANT,
    FUSED_QKV_OFF,
    FUSED_QKV_PRESERVE_BF16,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_BF16,
    MLP_MEMORY_FORCE_QUANT,
    MLP_MEMORY_OFF,
    MLP_MEMORY_PRESERVE,
    QKV_STREAMING_AUTO,
    QKV_STREAMING_FORCED,
    QKV_STREAMING_OFF,
)
from h3_optimizations.public_nodes import H3OptimizationsExtension  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class PublicNodeTests(unittest.TestCase):
    def test_public_registry_contains_only_production_nodes(self):
        nodes = asyncio.run(H3OptimizationsExtension().get_node_list())
        self.assertEqual(
            nodes,
            [
                H3MemoryOptimization,
                H3AIMDOResidencyLimiter,
                H3SparseAttention,
                H3SparseAttentionAdvanced,
            ],
        )
        self.assertFalse(any('MLPSharing' in node.__name__ for node in nodes))

    def test_explicit_preserve_precision_off_keeps_quantizing_auto_policy(self):
        request = _memory_request(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=2048,
            preserve_precision=False,
        )
        self.assertEqual(request.attention, 'auto')
        self.assertEqual(request.fused_qkv, 'auto')
        self.assertEqual(request.mlp_memory, 'auto')
        self.assertEqual(request.chunk_rows, 2048)

    def test_preserve_precision_defaults_on(self):
        request = _memory_request(chunk_rows=2048)
        self.assertEqual(request.attention, ATTENTION_EXISTING)
        self.assertEqual(request.fused_qkv, FUSED_QKV_PRESERVE_BF16)
        self.assertEqual(request.mlp_memory, MLP_MEMORY_PRESERVE)
        self.assertEqual(request.chunk_rows, 2048)

    def test_preserve_precision_keeps_bf16_qkv_chunking_without_requantization(self):
        request = _memory_request(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=2048,
            preserve_precision=True,
        )
        self.assertEqual(request.attention, ATTENTION_EXISTING)
        self.assertEqual(request.fused_qkv, FUSED_QKV_PRESERVE_BF16)
        self.assertEqual(request.mlp_memory, MLP_MEMORY_PRESERVE)
        self.assertEqual(request.chunk_rows, 2048)

    def test_preserve_precision_respects_explicit_mlp_off(self):
        request = _memory_request(
            fused_qkv='off',
            mlp_memory=MLP_MEMORY_OFF,
            preserve_precision=True,
        )
        self.assertEqual(request.fused_qkv, FUSED_QKV_PRESERVE_BF16)
        self.assertEqual(request.mlp_memory, MLP_MEMORY_OFF)

    def test_legacy_precision_modes_map_to_new_policies(self):
        self.assertEqual(
            _normalize_precision_mode(PRECISION_MODE_PRESERVE),
            PRECISION_MODE_PRESERVE_NATIVE,
        )
        self.assertEqual(
            _normalize_precision_mode(PRECISION_MODE_ALLOW_FP8),
            PRECISION_MODE_AUTO,
        )

    def test_qkv_streaming_mode_maps_to_plan_values(self):
        self.assertEqual(_qkv_streaming_request(QKV_STREAMING_MODE_OFF), QKV_STREAMING_OFF)
        self.assertEqual(_qkv_streaming_request(QKV_STREAMING_MODE_AUTO), QKV_STREAMING_AUTO)
        self.assertEqual(_qkv_streaming_request(QKV_STREAMING_MODE_FORCED), QKV_STREAMING_FORCED)

    def test_streaming_auto_claims_unselected_attention(self):
        request = _memory_request_for_modes(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=2048,
            precision_mode=PRECISION_MODE_PRESERVE_NATIVE,
            qkv_streaming_mode=QKV_STREAMING_MODE_AUTO,
            explicit_attention_selected=False,
        )
        self.assertEqual(request.attention, ATTENTION_AUTO)
        self.assertEqual(request.fused_qkv, FUSED_QKV_PRESERVE_BF16)
        self.assertEqual(request.qkv_streaming, QKV_STREAMING_AUTO)

    def test_streaming_auto_yields_to_explicit_attention(self):
        request = _memory_request_for_modes(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=2048,
            precision_mode=PRECISION_MODE_PRESERVE_NATIVE,
            qkv_streaming_mode=QKV_STREAMING_MODE_AUTO,
            explicit_attention_selected=True,
        )
        self.assertEqual(request.attention, ATTENTION_EXISTING)
        self.assertEqual(request.fused_qkv, FUSED_QKV_PRESERVE_BF16)
        self.assertEqual(request.qkv_streaming, QKV_STREAMING_AUTO)

    def test_streaming_forced_claims_attention_even_when_explicit(self):
        request = _memory_request_for_modes(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=2048,
            precision_mode=PRECISION_MODE_PRESERVE_NATIVE,
            qkv_streaming_mode=QKV_STREAMING_MODE_FORCED,
            explicit_attention_selected=True,
        )
        self.assertEqual(request.attention, ATTENTION_AUTO)
        self.assertEqual(request.qkv_streaming, QKV_STREAMING_FORCED)

    def test_streaming_off_preserves_attention_and_native_qkv_policy(self):
        request = _memory_request_for_modes(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=2048,
            precision_mode=PRECISION_MODE_PRESERVE_NATIVE,
            qkv_streaming_mode=QKV_STREAMING_MODE_OFF,
        )
        self.assertEqual(request.attention, ATTENTION_EXISTING)
        self.assertEqual(request.fused_qkv, FUSED_QKV_PRESERVE_BF16)
        self.assertEqual(request.qkv_streaming, QKV_STREAMING_OFF)

    def test_precision_modes_map_to_distinct_execution_policies(self):
        expected = {
            PRECISION_MODE_AUTO: (FUSED_QKV_AUTO, MLP_MEMORY_AUTO, False),
            PRECISION_MODE_BF16: (FUSED_QKV_FORCE_BF16, MLP_MEMORY_BF16, True),
            PRECISION_MODE_PRESERVE_NATIVE: (
                FUSED_QKV_PRESERVE_BF16,
                MLP_MEMORY_PRESERVE,
                False,
            ),
            PRECISION_MODE_FORCE_QUANT: (
                FUSED_QKV_FORCE_QUANT,
                MLP_MEMORY_FORCE_QUANT,
                True,
            ),
        }
        for mode, values in expected.items():
            with self.subTest(mode=mode):
                request = _memory_request_for_modes(
                    fused_qkv='auto',
                    mlp_memory='auto',
                    chunk_rows=2048,
                    precision_mode=mode,
                    qkv_streaming_mode=QKV_STREAMING_MODE_AUTO,
                )
                self.assertEqual(
                    (request.fused_qkv, request.mlp_memory, request.mlp_strict),
                    values,
                )

    def test_streaming_off_allows_bounded_qkv_when_precision_is_not_preserved(self):
        request = _memory_request_for_modes(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=2048,
            precision_mode=PRECISION_MODE_AUTO,
            qkv_streaming_mode=QKV_STREAMING_MODE_OFF,
        )
        self.assertEqual(request.attention, ATTENTION_EXISTING)
        self.assertEqual(request.fused_qkv, FUSED_QKV_AUTO)
        self.assertEqual(request.qkv_streaming, QKV_STREAMING_OFF)

    def test_memory_schema_appends_streaming_after_precision_mode(self):
        schema = H3MemoryOptimization.define_schema()
        inputs = schema.inputs
        ids = [item.id for item in inputs]
        legacy_index = ids.index('preserve_precision')
        precision_index = ids.index('precision_mode')
        streaming_index = ids.index('qkv_streaming_mode')
        self.assertEqual(precision_index, legacy_index + 1)
        self.assertEqual(streaming_index, precision_index + 1)

        legacy = inputs[legacy_index]
        precision = inputs[precision_index]
        streaming = inputs[streaming_index]
        self.assertTrue(legacy.extra_dict.get('hidden'))
        self.assertEqual(precision.options, list(PRECISION_MODE_OPTIONS))
        self.assertEqual(precision.default, PRECISION_MODE_AUTO)
        self.assertEqual(streaming.default, QKV_STREAMING_MODE_AUTO)

    def test_memory_node_accepts_legacy_precision_values(self):
        self.assertTrue(H3MemoryOptimization.validate_inputs(PRECISION_MODE_PRESERVE))
        self.assertTrue(H3MemoryOptimization.validate_inputs(PRECISION_MODE_ALLOW_FP8))


if __name__ == '__main__':
    unittest.main()
