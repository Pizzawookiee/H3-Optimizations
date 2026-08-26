"""CPU contracts for source-aware streamed QKV weight selection."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import h3_optimizations.qkv.streamed as streamed  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def weight_format(**values):
    defaults = {
        "convrot_int8_256": False,
        "w4a8": False,
        "fp8": False,
        "plain_float": False,
        "logical_dtype": "torch.bfloat16",
        "label": "test",
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


class StreamedQKVBindingTests(unittest.TestCase):
    def test_native_mode_selects_each_checkpoint_binding(self):
        cases = (
            (weight_format(plain_float=True), "HeldBF16QKV"),
            (weight_format(convrot_int8_256=True), "HeldConvRotINT8QKV"),
            (weight_format(w4a8=True), "HeldW4A8QKV"),
            (weight_format(fp8=True), "HeldFP8QKV"),
        )
        module = SimpleNamespace(qkv_proj=object())
        sample = object()
        for actual, constructor_name in cases:
            with self.subTest(constructor=constructor_name), mock.patch.object(
                streamed,
                "describe_linear",
                return_value=actual,
            ), mock.patch.object(
                streamed,
                constructor_name,
                return_value=constructor_name,
            ) as constructor:
                binding = streamed.create_held_qkv(module, sample)

            self.assertEqual(binding, constructor_name)
            constructor.assert_called_once_with(module, sample)

    def test_forced_modes_select_the_requested_execution_precision(self):
        module = SimpleNamespace(qkv_proj=object())
        sample = object()
        cases = (
            (streamed.PROJECTION_FORCE_BF16, "HeldBF16QKV", "allow_quantized_source"),
            (streamed.PROJECTION_FORCE_FP8, "HeldFP8QKV", "allow_float_conversion"),
            (streamed.PROJECTION_FORCE_INT8, "HeldConvRotINT8QKV", "allow_float_conversion"),
        )
        for mode, constructor_name, flag in cases:
            with self.subTest(mode=mode), mock.patch.object(
                streamed,
                constructor_name,
                return_value=constructor_name,
            ) as constructor:
                binding = streamed.create_held_qkv(module, sample, mode)

            self.assertEqual(binding, constructor_name)
            constructor.assert_called_once_with(module, sample, **{flag: True})


if __name__ == "__main__":
    unittest.main()
