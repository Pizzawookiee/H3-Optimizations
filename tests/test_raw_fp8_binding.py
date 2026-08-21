'''CPU contracts for raw torch FP8 H3 weight normalization.'''

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

import comfy.ops  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402
from h3_optimizations.qkv.fp8 import HeldFP8Linear  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class RawFP8BindingTests(unittest.TestCase):
    def test_raw_e4m3_is_wrapped_without_requantizing_weight_values(self):
        dtype = getattr(torch, 'float8_e4m3fn', None)
        if dtype is None:
            self.skipTest('this PyTorch build has no float8_e4m3fn')

        raw = torch.tensor(
            [[1.0, -0.5], [0.25, 2.0]],
            dtype=dtype,
        )
        module = SimpleNamespace(
            weight=raw,
            bias=None,
            weight_function=[],
            bias_function=[],
            _full_precision_mm=False,
        )
        sample = torch.zeros((1, 2), dtype=torch.bfloat16)

        def acquire(_module, input=None, **kwargs):
            self.assertIsNone(input)
            self.assertEqual(kwargs['dtype'], dtype)
            self.assertEqual(kwargs['device'], sample.device)
            return raw, None, None

        with mock.patch.object(
            comfy.ops,
            'cast_bias_weight',
            side_effect=acquire,
        ):
            with HeldFP8Linear(module, sample) as binding:
                self.assertTrue(binding.normalized_raw_fp8)
                self.assertFalse(binding.converted_from_float)
                self.assertIsInstance(binding.weight, QuantizedTensor)
                self.assertEqual(
                    binding.weight._layout_cls,
                    'TensorCoreFP8E4M3Layout',
                )
                self.assertIs(binding.weight._qdata, raw)
                self.assertEqual(float(binding.weight._params.scale), 1.0)
                self.assertEqual(
                    binding.weight._params.orig_dtype,
                    torch.bfloat16,
                )


if __name__ == '__main__':
    unittest.main()