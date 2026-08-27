'''CPU contracts for non-BF16 ConvRot MLP routing.'''

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

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

from h3_optimizations.memory.config import (  # noqa: E402
    MODE_CONVROT_2SLICE,
    ActivationMemoryConfig,
)
import h3_optimizations.memory.forward as forward_module  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class NonBF16MLPRoutingTests(unittest.TestCase):
    @staticmethod
    def _block():
        mlp = type('MLP', (), {'fc1': object(), 'fc2': object()})()
        return type('Block', (), {'mlp': mlp})()

    def test_fp16_convrot_routes_directly_to_held_fallback(self):
        block = self._block()
        config = ActivationMemoryConfig(
            mode=MODE_CONVROT_2SLICE,
            strict=False,
        )
        sample = torch.empty((1, 8), dtype=torch.float16)
        held = object()

        with patch.object(
            forward_module,
            '_open_generic_held',
            return_value=(held, None),
        ) as generic, patch.object(
            forward_module,
            'ConvRotTwoSliceMLP',
        ) as convrot:
            actual = forward_module._open_mlp(block, sample, config)

        self.assertEqual(actual, (held, 'held', None))
        generic.assert_called_once_with(block, sample, config)
        convrot.assert_not_called()

    def test_fp16_convrot_strict_mode_still_rejects_input(self):
        block = self._block()
        config = ActivationMemoryConfig(
            mode=MODE_CONVROT_2SLICE,
            strict=True,
        )
        sample = torch.empty((1, 8), dtype=torch.float16)

        with self.assertRaisesRegex(TypeError, 'requires BF16 input'):
            forward_module._open_mlp(block, sample, config)


if __name__ == '__main__':
    unittest.main()
