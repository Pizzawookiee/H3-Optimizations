"""CPU contracts for runtime ConvRot-256 INT8 H3 linears."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import comfy.ops  # noqa: E402
from comfy.quant_ops import TensorWiseINT8Layout  # noqa: E402
from h3_optimizations.qkv.int8 import (  # noqa: E402
    HeldConvRotINT8Linear,
    LazyConvRotINT8Linear,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


def floating_linear(weight, bias=None):
    return SimpleNamespace(
        weight=weight,
        bias=bias,
        weight_function=[],
        bias_function=[],
        _full_precision_mm=False,
    )


class RuntimeConvRotINT8Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.weight = torch.randn((256, 256), dtype=torch.bfloat16)
        self.sample = torch.randn((2, 256), dtype=torch.bfloat16)
        self.module = floating_linear(self.weight)

    def test_float_weight_becomes_reference_convrot_layout(self):
        handle = object()
        with mock.patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(self.weight, None, handle),
        ), mock.patch.object(comfy.ops, "uncast_bias_weight") as release:
            with HeldConvRotINT8Linear(
                self.module,
                self.sample[:1],
                allow_float_conversion=True,
            ) as binding:
                self.assertTrue(binding.converted_from_float)
                self.assertEqual(binding.weight._layout_cls, "TensorWiseINT8Layout")
                self.assertTrue(binding.weight._params.convrot)
                self.assertEqual(binding.weight._params.convrot_groupsize, 256)
                qdata, scale = TensorWiseINT8Layout.get_plain_tensors(
                    binding.weight
                )
                self.assertEqual(qdata.dtype, torch.int8)
                self.assertEqual(scale.dtype, torch.float32)
                self.assertEqual(tuple(scale.shape), (256, 1))
                output = binding.linear(self.sample)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(tuple(output.shape), (2, 256))

            release.assert_called_once_with(
                self.module,
                self.weight,
                None,
                handle,
            )

    def test_lazy_binding_quantizes_once_for_multiple_output_slabs(self):
        handle = object()
        with mock.patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(self.weight, None, handle),
        ) as acquire, mock.patch.object(
            comfy.ops,
            "uncast_bias_weight",
        ) as release:
            session = LazyConvRotINT8Linear(self.module)
            first = session.linear(self.sample[:1])
            second = session.linear(self.sample[1:])
            self.assertEqual(tuple(first.shape), (1, 256))
            self.assertEqual(tuple(second.shape), (1, 256))
            self.assertEqual(acquire.call_count, 1)
            session.release()
            self.assertEqual(release.call_count, 1)

    def test_output_slices_match_one_full_convrot_linear(self):
        handle = object()
        with mock.patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(self.weight, None, handle),
        ), mock.patch.object(comfy.ops, "uncast_bias_weight"):
            with HeldConvRotINT8Linear(
                self.module,
                self.sample[:1],
                allow_float_conversion=True,
            ) as binding:
                full = binding.linear(self.sample)
                sliced = torch.cat(
                    (
                        binding.linear_range(self.sample, 0, 96),
                        binding.linear_range(self.sample, 96, 256),
                    ),
                    dim=-1,
                )

        torch.testing.assert_close(sliced, full, rtol=0, atol=0)

    def test_bias_is_not_silently_quantized(self):
        bias = torch.zeros((256,), dtype=torch.bfloat16)
        module = floating_linear(self.weight, bias)
        with mock.patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(self.weight, bias, object()),
        ), mock.patch.object(comfy.ops, "uncast_bias_weight"):
            with self.assertRaisesRegex(RuntimeError, "bias-free"):
                HeldConvRotINT8Linear(
                    module,
                    self.sample[:1],
                    allow_float_conversion=True,
                ).__enter__()


if __name__ == "__main__":
    unittest.main()
