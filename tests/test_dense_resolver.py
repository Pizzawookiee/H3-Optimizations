'''Dense H3 resolver preserves arbitrary upstream attention overrides.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

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


class DenseResolverTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
