'''Dense H3 resolver preserves arbitrary upstream attention overrides.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import h3_optimizations.dense_resolver as dense_resolver  # noqa: E402
import h3_optimizations.kitchen_qkv as kitchen_qkv  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakePatcher:
    def __init__(self, override=None):
        transformer_options = {}
        if override is not None:
            transformer_options['optimized_attention_override'] = override
        self.model_options = {'transformer_options': transformer_options}
        self.set_calls = []

    def set_model_optimized_attention(self, backend):
        self.set_calls.append(backend)
        self.model_options['transformer_options'][
            'optimized_attention_override'
        ] = SimpleNamespace()


class FakeKitchen:
    INT8_ATTENTION_PRODUCER_ABI_VERSION = 1

    class Int8AttentionProducerUnavailableError(RuntimeError):
        pass

    @staticmethod
    def int8_attention_producer_is_available(_device=None):
        return True

    @staticmethod
    def int8_attention_producer_spec(*_args, **_kwargs):
        return SimpleNamespace(
            abi_version=1,
            k_anchor_positions=(),
            sequence_alignment=256,
            q_tile=128,
            k_tile=128,
        )

    @staticmethod
    def select_int8_attention_k_anchor(*_args, **_kwargs):
        return object()

    @staticmethod
    def create_int8_attention_producer(*_args, **_kwargs):
        return object()

    @staticmethod
    def quantize_int8_attention_qk_chunk(*_args, **_kwargs):
        return None

    @staticmethod
    def quantize_int8_attention_v(*_args, **_kwargs):
        return None

    @staticmethod
    def finalize_int8_attention_producer(*_args, **_kwargs):
        return object()

    @staticmethod
    def int8_attention_from_prequantized(*_args, **_kwargs):
        return object()


class DenseResolverTests(unittest.TestCase):
    def test_opted_in_external_consumer_gets_streamed_q(self):
        def override(*_args, **_kwargs):
            return None

        override.supports_streamed_h3_qkv = True
        override.consume = lambda **_kwargs: None
        patcher = FakePatcher(override)

        with mock.patch.object(
            dense_resolver,
            'is_comfy_kitchen_dense_attention',
            return_value=False,
        ), mock.patch.object(
            dense_resolver,
            'resolve_sage_fused_attention',
        ) as resolve_sage:
            resolution = dense_resolver.resolve_current_dense_attention(
                patcher,
                SimpleNamespace(capability=(8, 9)),
            )

        self.assertEqual(resolution.backend_kind, dense_resolver.ATTENTION_EXISTING)
        self.assertIn('streamed-H3 QKV', resolution.reason)
        resolve_sage.assert_not_called()

    def test_unknown_external_consumer_keeps_full_q_single_call(self):
        patcher = FakePatcher(lambda *_args, **_kwargs: None)

        with mock.patch.object(
            dense_resolver,
            'is_comfy_kitchen_dense_attention',
            return_value=False,
        ), mock.patch.object(
            dense_resolver,
            'resolve_sage_fused_attention',
            return_value=None,
        ), mock.patch.object(
            dense_resolver,
            'is_known_comfy_dense_attention',
            return_value=False,
        ):
            resolution = dense_resolver.resolve_current_dense_attention(
                patcher,
                SimpleNamespace(capability=(8, 9)),
            )

        self.assertEqual(
            resolution.backend_kind,
            dense_resolver.ATTENTION_EXISTING_FULL_Q,
        )
        self.assertIn('full-Q single-call', resolution.reason)

    def test_advertised_consumer_requires_consume_callable(self):
        def override(*_args, **_kwargs):
            return None

        override.supports_streamed_h3_qkv = True
        patcher = FakePatcher(override)

        with mock.patch.object(
            dense_resolver,
            'is_comfy_kitchen_dense_attention',
            return_value=False,
        ), self.assertRaisesRegex(TypeError, 'callable consume'):
            dense_resolver.resolve_current_dense_attention(
                patcher,
                SimpleNamespace(capability=(8, 9)),
            )

    def test_arbitrary_upstream_override_is_preserved_for_private_h3_kitchen(self):
        upstream = object()
        kitchen = object()
        patcher = FakePatcher(upstream)

        with mock.patch.object(
            dense_resolver,
            'get_attention_function',
            return_value=kitchen,
        ):
            resolution = dense_resolver.resolve_dense_attention(patcher)

        self.assertEqual(
            resolution.selected,
            dense_resolver.ATTENTION_COMFY_KITCHEN_INT8,
        )
        self.assertEqual(
            resolution.backend_kind,
            dense_resolver.ATTENTION_COMFY_KITCHEN_INT8,
        )
        self.assertIsNone(resolution.backend)
        self.assertIs(
            patcher.model_options['transformer_options'][
                'optimized_attention_override'
            ],
            upstream,
        )
        self.assertIn('preserved', resolution.reason)
        self.assertIn('private H3 memory path', resolution.reason)

    def test_preserved_override_is_not_reinstalled(self):
        upstream = object()
        patcher = FakePatcher(upstream)
        resolution = dense_resolver.DenseResolution(
            dense_resolver.ATTENTION_AUTO,
            dense_resolver.ATTENTION_COMFY_KITCHEN_INT8,
            None,
            'private H3 Kitchen with upstream override preserved',
            dense_resolver.ATTENTION_COMFY_KITCHEN_INT8,
        )

        installed = dense_resolver.install_dense_attention(patcher, resolution)

        self.assertFalse(installed)
        self.assertEqual(patcher.set_calls, [])
        self.assertIs(
            patcher.model_options['transformer_options'][
                'optimized_attention_override'
            ],
            upstream,
        )

    def test_override_falls_back_to_existing_when_kitchen_is_unavailable(self):
        upstream = object()
        patcher = FakePatcher(upstream)

        with mock.patch.object(
            dense_resolver,
            'get_attention_function',
            return_value=None,
        ):
            resolution = dense_resolver.resolve_dense_attention(patcher)

        self.assertEqual(resolution.selected, dense_resolver.ATTENTION_EXISTING)
        self.assertEqual(resolution.backend_kind, dense_resolver.ATTENTION_EXISTING)
        self.assertIsNone(resolution.backend)
        self.assertIs(
            patcher.model_options['transformer_options'][
                'optimized_attention_override'
            ],
            upstream,
        )
        self.assertIn('unavailable', resolution.reason)

    def test_no_override_still_installs_registered_kitchen(self):
        kitchen = object()
        patcher = FakePatcher()

        with mock.patch.object(
            dense_resolver,
            'get_attention_function',
            return_value=kitchen,
        ):
            resolution = dense_resolver.resolve_dense_attention(patcher)

        self.assertIs(resolution.backend, kitchen)
        self.assertEqual(
            resolution.backend_kind,
            dense_resolver.ATTENTION_COMFY_KITCHEN_INT8,
        )

    def test_private_h3_projector_ignores_arbitrary_upstream_override(self):
        upstream = object()
        fake = FakeKitchen()
        expected = object()
        module = SimpleNamespace(qkv_proj=object(), heads=2, head_dim=128)
        x = SimpleNamespace(
            ndim=2,
            is_cuda=True,
            shape=(256, 256),
            dtype=torch.bfloat16,
            device=torch.device('cuda:0'),
        )
        projector = kitchen_qkv.ChunkedKitchenQKVProjector(chunk_rows=4096)
        options = {
            'optimized_attention_override': upstream,
            kitchen_qkv.H3_ATTENTION_BACKEND_KEY:
                kitchen_qkv.DENSE_KITCHEN_BACKEND,
        }

        with mock.patch.object(
            kitchen_qkv,
            'resolve_kitchen',
            return_value=fake,
        ), mock.patch.object(
            kitchen_qkv.comfy.model_management,
            'in_training',
            False,
        ), mock.patch.object(
            kitchen_qkv,
            'describe_linear',
            return_value=SimpleNamespace(
                fp8=False,
                plain_float=False,
                convrot_int8_256=True,
                w4a8=False,
            ),
        ), mock.patch.object(
            kitchen_qkv,
            'run_chunked_kitchen_qkv',
            return_value=expected,
        ):
            actual = projector.try_project(
                module,
                x,
                None,
                layer_index=0,
                transformer_options=options,
            )

        self.assertIs(actual, expected)
        self.assertIs(options['optimized_attention_override'], upstream)


if __name__ == '__main__':
    unittest.main()
