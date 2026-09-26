"""CPU contracts for selecting and holding the exact H3 fused-KV path."""

import os
from pathlib import Path
from types import SimpleNamespace
import sys
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

from h3_optimizations.qkv import fused_kv  # noqa: E402
from h3_optimizations.qkv.streamed import PROJECTION_NATIVE  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class FusedKVBindingTests(unittest.TestCase):
    def test_selection_is_native_bf16_convrot_sm80_only(self):
        module = SimpleNamespace(heads=2, head_dim=128, qkv_proj=object())
        x = torch.zeros(64, 256, dtype=torch.bfloat16)
        rope = torch.zeros(1, 64, 1, 48, 2, 2, dtype=torch.bfloat16)
        with (
            mock.patch.object(
                torch.Tensor,
                "is_cuda",
                new_callable=mock.PropertyMock,
                return_value=True,
            ),
            mock.patch.object(
                fused_kv,
                "describe_linear",
                return_value=SimpleNamespace(convrot_int8_256=True),
            ),
            mock.patch.object(
                fused_kv, "int8_rowwise_convrot256_is_available", return_value=True
            ),
            mock.patch.object(fused_kv, "fused_h3_kv_is_available", return_value=True),
        ):
            self.assertTrue(
                fused_kv.fused_h3_kv_supported(module, x, rope, PROJECTION_NATIVE)
            )
            self.assertFalse(
                fused_kv.fused_h3_kv_supported(module, x.float(), rope, PROJECTION_NATIVE)
            )
            self.assertFalse(
                fused_kv.fused_h3_kv_supported(module, x, rope, "force_bf16")
            )

    def test_project_reuses_held_binding_and_writes_the_native_producer(self):
        class FakeQuantizedTensor:
            def __init__(self):
                self._qdata = (
                    torch.arange(768 * 256, dtype=torch.int32)
                    .to(torch.int8)
                    .reshape(768, 256)
                )
                self._params = SimpleNamespace(scale=torch.ones(768, dtype=torch.float32))

        class Held:
            def __init__(self):
                self.binding = SimpleNamespace(weight=FakeQuantizedTensor())
                self.exited = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.exited = True
                return False

            def project_rows(self, *_args):
                return "rows"

            def project_v_hnd(self, *_args):
                return "v"

        k_norm = SimpleNamespace(weight=torch.ones(128, dtype=torch.bfloat16), eps=1e-6)
        module = SimpleNamespace(
            heads=2,
            head_dim=128,
            qkv_proj=object(),
            k_norm=k_norm,
        )
        x = torch.zeros(128, 256, dtype=torch.bfloat16)
        rope = torch.zeros(1, 128, 1, 48, 2, 2, dtype=torch.bfloat16)
        held_binding = Held()
        activation = torch.zeros(128, 256, dtype=torch.int8)
        activation_scale = torch.ones(128, 1)
        producer = object()
        expected = (object(), object())

        with (
            mock.patch.object(fused_kv, "QuantizedTensor", FakeQuantizedTensor),
            mock.patch.object(fused_kv, "fused_h3_kv_supported", return_value=True),
            mock.patch.object(fused_kv, "HeldConvRotINT8QKV", return_value=held_binding),
            mock.patch.object(
                fused_kv.comfy.model_management,
                "cast_to",
                return_value=k_norm.weight,
            ),
            mock.patch.object(
                fused_kv,
                "quantize_int8_rowwise_convrot256",
                return_value=(activation, activation_scale),
            ) as quantize,
            mock.patch.object(
                fused_kv, "fused_h3_kv_from_int8", return_value=expected
            ) as produce,
        ):
            binding = fused_kv.HeldExactH3FusedKV(
                module, x[:1], rope, PROJECTION_NATIVE
            )
            with binding:
                actual = binding.project(x, rope, 0, 128, producer)
                self.assertEqual(binding.project_rows(None, None, None), "rows")
                self.assertEqual(binding.project_v_hnd(None, None, 0, 1), "v")

        self.assertIs(actual, expected)
        quantize.assert_called_once()
        self.assertEqual(tuple(produce.call_args.args[1].shape), (512, 256))
        self.assertEqual(int(produce.call_args.args[3].numel()), 512)
        self.assertIs(produce.call_args.args[6], producer)
        self.assertEqual(produce.call_args.kwargs["row_start"], 0)
        self.assertEqual(produce.call_args.kwargs["epsilon"], 1e-6)
        self.assertTrue(held_binding.exited)


if __name__ == "__main__":
    unittest.main()
