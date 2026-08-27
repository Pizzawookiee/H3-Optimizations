'''CPU contracts for preserving and rebuilding ModelPatcher object patches.'''

from pathlib import Path
import os
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations import patch as attention_patch  # noqa: E402
from h3_optimizations.memory import patch as memory_patch  # noqa: E402
from h3_optimizations.memory.config import ActivationMemoryConfig  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class Patcher:
    def __init__(self):
        self.object_patches = {}
        self.model_options = {'transformer_options': {}}

    def add_object_patch(self, key, value):
        self.object_patches[key] = value


def attention_forward(module, _index, **_kwargs):
    def forward(value):
        return module.forward(value) + 10

    return forward


def memory_forward(block, _index, config, original_forward):
    def forward(value):
        return original_forward(value) + config.chunk_rows

    setattr(forward, memory_patch.OWNER_MARKER, True)
    setattr(forward, memory_patch.SIGNATURE_MARKER, config.signature)
    setattr(forward, memory_patch.ORIGINAL_MARKER, original_forward)
    return forward


class ObjectPatchCompositionTests(unittest.TestCase):
    def test_attention_preserves_foreign_block_and_rebuilds_owned_sibling(self):
        patcher = Patcher()
        modules = (
            SimpleNamespace(forward=lambda value: value),
            SimpleNamespace(forward=lambda value: value * 2),
        )
        foreign = lambda value: value - 5
        patcher.object_patches[attention_patch.key_for(0)] = foreign
        backend = SimpleNamespace(name='synthetic')

        with mock.patch.object(
            attention_patch, 'validate', return_value=modules
        ), mock.patch(
            'h3_optimizations.attention_forward.make_forward',
            side_effect=attention_forward,
        ):
            _backend, installed = attention_patch.configure_backend(
                patcher,
                backend,
            )
            first_owned = patcher.object_patches[attention_patch.key_for(1)]
            _backend, rebuilt = attention_patch.configure_backend(
                patcher,
                backend,
                force_rebuild=True,
            )

        self.assertEqual(installed, 1)
        self.assertEqual(rebuilt, 1)
        self.assertIs(
            patcher.object_patches[attention_patch.key_for(0)],
            foreign,
        )
        self.assertIsNot(
            patcher.object_patches[attention_patch.key_for(1)],
            first_owned,
        )
        self.assertEqual(
            patcher.model_options['transformer_options'][
                'h3_optimizations_preserved_attention_patches'
            ],
            [attention_patch.key_for(0)],
        )

    def test_memory_preserves_foreign_block_and_rebuilds_owned_sibling(self):
        patcher = Patcher()
        blocks = (
            SimpleNamespace(forward=lambda value: value),
            SimpleNamespace(forward=lambda value: value * 2),
        )
        foreign = lambda value: value - 7
        patcher.object_patches[memory_patch.key_for(0)] = foreign
        config = ActivationMemoryConfig(chunk_rows=4096)

        with mock.patch.object(
            memory_patch, 'validate', return_value=blocks
        ), mock.patch.object(
            memory_patch, 'make_forward', side_effect=memory_forward
        ):
            installed = memory_patch.install(patcher, config)
            first_owned = patcher.object_patches[memory_patch.key_for(1)]
            rebuilt = memory_patch.install(
                patcher,
                config,
                force_rebuild=True,
            )

        self.assertEqual(installed, 1)
        self.assertEqual(rebuilt, 1)
        self.assertIs(
            patcher.object_patches[memory_patch.key_for(0)],
            foreign,
        )
        self.assertIsNot(
            patcher.object_patches[memory_patch.key_for(1)],
            first_owned,
        )
        self.assertEqual(
            patcher.model_options['transformer_options'][
                'h3_optimizations_preserved_block_patches'
            ],
            [memory_patch.key_for(0)],
        )

    def test_clear_attention_removes_only_owned_and_refreshes_diagnostics(self):
        patcher = Patcher()
        foreign = lambda value: value - 5
        owned = lambda value: value + 10
        setattr(owned, attention_patch.OWNER_MARKER, True)
        patcher.object_patches[attention_patch.key_for(0)] = foreign
        patcher.object_patches[attention_patch.key_for(1)] = owned
        patcher.model_options['transformer_options'][
            'h3_optimizations_preserved_attention_patches'
        ] = ['stale']

        removed = attention_patch.clear_backend(patcher)

        self.assertEqual(removed, 1)
        self.assertIs(
            patcher.object_patches[attention_patch.key_for(0)],
            foreign,
        )
        self.assertNotIn(
            attention_patch.key_for(1),
            patcher.object_patches,
        )
        self.assertEqual(
            patcher.model_options['transformer_options'][
                'h3_optimizations_preserved_attention_patches'
            ],
            [attention_patch.key_for(0)],
        )


if __name__ == '__main__':
    unittest.main()
