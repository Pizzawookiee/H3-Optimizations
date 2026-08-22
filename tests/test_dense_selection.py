'''CPU-only contracts for public dense-attention backend selection.'''

import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
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
from h3_optimizations.dense_resolver import (  # noqa: E402
    ATTENTION_COMFY_KITCHEN_INT8,
    ATTENTION_EXISTING,
    install_dense_attention,
    resolve_dense_attention,
)
from h3_optimizations.plan import H3OptimizationPlan, MemoryRequest  # noqa: E402
from h3_optimizations.qkv.providers import (  # noqa: E402
    QKV_DENSE_KITCHEN_CHUNKED,
    QKV_STANDARD,
)

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


class DenseSelectionTests(unittest.TestCase):
    @staticmethod
    def _convrot_inventory():
        return SimpleNamespace(
            qkv=(object(),),
            qkv_convrot_int8_256=True,
            homogeneous=lambda name: name == 'qkv',
            labels=lambda _name: ('TensorWiseINT8Layout+convrot256',),
        )

    def test_public_lookup_and_setter_retain_container_function(self):
        model = FakePatcher()
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=kitchen_attention,
        ) as lookup:
            resolution = resolve_dense_attention(model)

        lookup.assert_called_once_with('comfy_kitchen_int8', None)
        self.assertEqual(
            resolution.selected,
            ATTENTION_COMFY_KITCHEN_INT8,
        )
        self.assertTrue(install_dense_attention(model, resolution))
        override = model.model_options[
            'transformer_options'
        ]['optimized_attention_override']
        self.assertIs(override.container_function, kitchen_containers)

    def test_existing_explicit_override_wins(self):
        model = FakePatcher()
        model.set_model_optimized_attention(pytorch_attention)
        original = model.model_options[
            'transformer_options'
        ]['optimized_attention_override']

        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            side_effect=AssertionError('explicit override must skip lookup'),
        ):
            resolution = resolve_dense_attention(model)

        self.assertEqual(resolution.selected, ATTENTION_EXISTING)
        self.assertFalse(install_dense_attention(model, resolution))
        self.assertIs(
            model.model_options['transformer_options'][
                'optimized_attention_override'
            ],
            original,
        )

    def test_unavailable_kitchen_leaves_normal_comfy_selection(self):
        model = FakePatcher()
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=None,
        ):
            resolution = resolve_dense_attention(model)

        self.assertEqual(resolution.selected, ATTENTION_EXISTING)
        self.assertFalse(install_dense_attention(model, resolution))
        self.assertNotIn(
            'optimized_attention_override',
            model.model_options['transformer_options'],
        )

    def test_official_override_composes_before_and_after_h3_selection(self):
        before = FakePatcher()
        before.set_model_optimized_attention(pytorch_attention)
        before_override = before.model_options[
            'transformer_options'
        ]['optimized_attention_override']
        self.assertEqual(
            resolve_dense_attention(before).selected,
            ATTENTION_EXISTING,
        )
        self.assertIs(
            before.model_options['transformer_options'][
                'optimized_attention_override'
            ],
            before_override,
        )

        after = FakePatcher()
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=kitchen_attention,
        ):
            automatic = resolve_dense_attention(after)
        install_dense_attention(after, automatic)
        after.set_model_optimized_attention(pytorch_attention)
        official_override = after.model_options[
            'transformer_options'
        ]['optimized_attention_override']
        self.assertEqual(
            resolve_dense_attention(after).selected,
            ATTENTION_EXISTING,
        )
        self.assertIs(
            after.model_options['transformer_options'][
                'optimized_attention_override'
            ],
            official_override,
        )

    def test_h3_dense_auto_installs_chunked_producer_only_for_kitchen(self):
        model = FakePatcher()
        plan = H3OptimizationPlan().with_memory(MemoryRequest())
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
                self._convrot_inventory(),
            )
        self.assertEqual(qkv.provider_id, QKV_DENSE_KITCHEN_CHUNKED)
        self.assertEqual(attention.projector.name, 'chunked_kitchen_qkv')
        self.assertEqual(
            attention.backend.name,
            'comfy_kitchen_int8_prequantized',
        )

        model.set_model_optimized_attention(pytorch_attention)
        with patch.object(
            apply_module,
            'producer_api_available',
            return_value=True,
        ):
            attention, qkv = apply_module._resolve_dense(
                plan,
                model,
                self._convrot_inventory(),
            )
        self.assertEqual(qkv.provider_id, QKV_STANDARD)
        self.assertIsNone(attention.projector)

    def test_legacy_existing_request_preserves_incoming_attention(self):
        model = FakePatcher()
        plan = H3OptimizationPlan().with_memory(
            MemoryRequest(attention='existing')
        )
        with patch.object(
            apply_module,
            'resolve_dense_attention',
            side_effect=AssertionError('existing must skip dense selection'),
        ):
            attention, qkv = apply_module._resolve_dense(
                plan,
                model,
                self._convrot_inventory(),
            )
        self.assertEqual(attention.selected, ATTENTION_EXISTING)
        self.assertEqual(qkv.provider_id, QKV_STANDARD)


if __name__ == '__main__':
    unittest.main()
