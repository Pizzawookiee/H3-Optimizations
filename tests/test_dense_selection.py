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
    ATTENTION_SAGE,
    ATTENTION_SAGE_SM89,
    DenseResolution,
    has_explicit_dense_attention,
    install_dense_attention,
    resolve_dense_attention,
)
from h3_optimizations.plan import (  # noqa: E402
    FUSED_QKV_AUTO,
    FUSED_QKV_PRESERVE_BF16,
    H3OptimizationPlan,
    MemoryRequest,
    QKV_STREAMING_OFF,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    QKV_DENSE_CONVROT_INT8,
    QKV_DENSE_KITCHEN_CHUNKED,
    QKV_STANDARD,
)
from h3_optimizations.qkv.policy import (  # noqa: E402
    resolve_qkv_provider as resolve_qkv_policy,
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
            qkv_w4a8=False,
            qkv_fp8=False,
            qkv_plain_float=False,
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
        self.assertEqual(resolution.selected, ATTENTION_COMFY_KITCHEN_INT8)
        self.assertTrue(install_dense_attention(model, resolution))
        override = model.model_options['transformer_options']['optimized_attention_override']
        self.assertIs(override.container_function, kitchen_containers)
        self.assertFalse(has_explicit_dense_attention(model))

    def test_existing_explicit_override_is_detected(self):
        model = FakePatcher()
        model.set_model_optimized_attention(pytorch_attention)
        self.assertTrue(has_explicit_dense_attention(model))

    def test_existing_explicit_override_is_preserved(self):
        model = FakePatcher()
        model.set_model_optimized_attention(pytorch_attention)
        original = model.model_options['transformer_options']['optimized_attention_override']

        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=kitchen_attention,
        ):
            resolution = resolve_dense_attention(model)

        self.assertEqual(resolution.selected, ATTENTION_COMFY_KITCHEN_INT8)
        self.assertFalse(install_dense_attention(model, resolution))
        self.assertIs(
            model.model_options['transformer_options']['optimized_attention_override'],
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
        before_override = before.model_options['transformer_options']['optimized_attention_override']
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=kitchen_attention,
        ):
            before_resolution = resolve_dense_attention(before)
        self.assertEqual(before_resolution.selected, ATTENTION_COMFY_KITCHEN_INT8)
        self.assertIs(
            before.model_options['transformer_options']['optimized_attention_override'],
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
        official_override = after.model_options['transformer_options']['optimized_attention_override']
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=kitchen_attention,
        ):
            after_resolution = resolve_dense_attention(after)
        self.assertEqual(after_resolution.selected, ATTENTION_COMFY_KITCHEN_INT8)
        self.assertIs(
            after.model_options['transformer_options']['optimized_attention_override'],
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
        self.assertEqual(attention.backend.name, 'comfy_kitchen_int8_prequantized')

        model.set_model_optimized_attention(pytorch_attention)
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

    def test_preserve_precision_convrot_can_stream_through_dense_kitchen(self):
        model = FakePatcher()
        plan = H3OptimizationPlan().with_memory(
            MemoryRequest(fused_qkv=FUSED_QKV_PRESERVE_BF16)
        )
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

    def test_streaming_off_preserve_native_uses_fused_dense_sage(self):
        model = FakePatcher()
        plan = H3OptimizationPlan().with_memory(
            MemoryRequest(
                attention=ATTENTION_EXISTING,
                fused_qkv=FUSED_QKV_PRESERVE_BF16,
                qkv_streaming=QKV_STREAMING_OFF,
            )
        )
        delegate = SimpleNamespace(
            name='sage_mem_eff',
            api=SimpleNamespace(version='2.2.test', kernel_name='fake'),
            allow_cpu_for_tests=True,
            runtime_listeners=(),
        )
        resolution = DenseResolution(
            ATTENTION_SAGE,
            ATTENTION_SAGE,
            delegate,
            'test command-line Sage route',
            ATTENTION_SAGE_SM89,
        )
        environment = SimpleNamespace(
            capability=(8, 9),
            device_index=None,
            cuda_available=False,
        )
        with patch.object(
            apply_module,
            'resolve_command_line_sage_fused_attention',
            return_value=resolution,
        ), patch(
            'h3_optimizations.dense_fused_qkv.TRITON_AVAILABLE',
            True,
        ), patch.object(
            apply_module,
            'resolve_qkv_provider',
            resolve_qkv_policy,
        ):
            attention, qkv = apply_module._resolve_dense(
                plan,
                model,
                self._convrot_inventory(),
                environment,
            )

        self.assertEqual(qkv.provider_id, QKV_DENSE_CONVROT_INT8)
        self.assertTrue(qkv.fused)
        self.assertEqual(attention.selected, ATTENTION_SAGE)
        self.assertEqual(attention.backend.name, 'sage_mem_eff')
        self.assertEqual(attention.projector.name, 'chunked_kitchen_dense_sage_qkv')

    def test_legacy_existing_request_preserves_incoming_attention(self):
        model = FakePatcher()
        plan = H3OptimizationPlan().with_memory(MemoryRequest(attention='existing'))
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
