'''CPU contracts for treating Comfy Kitchen as a compatible streaming choice.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
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

from comfy.model_patcher import ModelPatcher  # noqa: E402
import h3_optimizations.apply as apply_module  # noqa: E402
from h3_optimizations.memory_migration_node import (  # noqa: E402
    PRECISION_MODE_PRESERVE,
    QKV_STREAMING_MODE_AUTO,
    _memory_request_for_modes,
)
from h3_optimizations.plan import (  # noqa: E402
    ATTENTION_AUTO,
    ATTENTION_EXISTING,
    H3OptimizationPlan,
)
from h3_optimizations.qkv.providers import QKV_DENSE_KITCHEN_CHUNKED  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakePatcher:
    def __init__(self):
        self.model_options = {'transformer_options': {}}

    def set_model_optimized_attention(self, attention):
        ModelPatcher.set_model_optimized_attention(self, attention)


def kitchen_attention(*_args, **_kwargs):
    return None


def kitchen_containers(*_args, **_kwargs):
    return None


kitchen_attention.container_function = kitchen_containers


def pytorch_attention(*_args, **_kwargs):
    return None


def convrot_inventory():
    return SimpleNamespace(
        qkv=(object(),),
        qkv_convrot_int8_256=True,
        qkv_w4a8=False,
        qkv_fp8=False,
        qkv_plain_float=False,
        homogeneous=lambda name: name == 'qkv',
        labels=lambda _name: ('TensorWiseINT8Layout+convrot256',),
    )


class ComfyKitchenSelectorStreamingTests(unittest.TestCase):
    def test_official_comfy_kitchen_selector_does_not_block_streaming_auto(self):
        model = FakePatcher()
        model.set_model_optimized_attention(kitchen_attention)

        request = _memory_request_for_modes(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=4096,
            precision_mode=PRECISION_MODE_PRESERVE,
            qkv_streaming_mode=QKV_STREAMING_MODE_AUTO,
        )
        self.assertEqual(request.attention, ATTENTION_EXISTING)

        plan = H3OptimizationPlan().with_memory(request)
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=kitchen_attention,
        ), patch.object(
            apply_module,
            'producer_api_available',
            return_value=True,
        ):
            attention, qkv = apply_module._resolve_dense(
                plan,
                model,
                convrot_inventory(),
            )

        self.assertEqual(qkv.provider_id, QKV_DENSE_KITCHEN_CHUNKED)
        self.assertEqual(attention.projector.name, 'chunked_kitchen_qkv')
        self.assertEqual(
            attention.backend.name,
            'comfy_kitchen_int8_prequantized',
        )
        self.assertIn('preserved the external Comfy Kitchen', attention.reason)

    def test_non_kitchen_explicit_selector_gets_generic_streaming(self):
        model = FakePatcher()
        model.set_model_optimized_attention(pytorch_attention)
        request = _memory_request_for_modes(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=4096,
            precision_mode=PRECISION_MODE_PRESERVE,
            qkv_streaming_mode=QKV_STREAMING_MODE_AUTO,
        )
        self.assertEqual(request.attention, ATTENTION_EXISTING)
        plan = H3OptimizationPlan().with_memory(request)
        with patch.object(
            apply_module,
            'producer_api_available',
            return_value=True,
        ):
            attention, _qkv = apply_module._resolve_dense(
                plan,
                model,
                convrot_inventory(),
            )
        self.assertEqual(attention.selected, ATTENTION_EXISTING)
        self.assertEqual(attention.projector.name, 'streamed_dense_bf16_qkv')


if __name__ == '__main__':
    unittest.main()
