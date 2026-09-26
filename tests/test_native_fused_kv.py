"""CPU-visible contracts for the exact native H3 fused-KV producer."""

import ctypes
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.native import fused_kv, loader  # noqa: E402


class _Function:
    def __init__(self):
        self.args = None

    def __call__(self, *args):
        self.args = args
        return 0


class _Library:
    def __init__(self, include_fused):
        self.include_fused = include_fused

    def __getattr__(self, name):
        if name == fused_kv._SYMBOL and not self.include_fused:
            raise AttributeError(name)
        function = _Function()
        setattr(self, name, function)
        return function


class NativeFusedKVTests(unittest.TestCase):
    def test_loader_treats_fused_kv_as_an_additive_abi4_symbol(self):
        old_library = _Library(False)
        self.assertIs(loader._bind(old_library), old_library)

        new_library = _Library(True)
        loader._bind(new_library)
        function = getattr(new_library, fused_kv._SYMBOL)
        self.assertIs(function.restype, ctypes.c_int)
        self.assertEqual(
            function.argtypes,
            [ctypes.c_void_p] * 12
            + [ctypes.c_int64] * 3
            + [ctypes.c_int] * 2
            + [ctypes.c_float, ctypes.c_size_t],
        )

    def test_fake_call_writes_global_k_and_returns_v_and_summary(self):
        call = _Function()
        library = SimpleNamespace(**{fused_kv._SYMBOL: call})
        activation = torch.zeros(128, 256, dtype=torch.int8)
        weight = torch.zeros(512, 256, dtype=torch.int8)
        activation_scale = torch.ones(128, 1, dtype=torch.float32)
        weight_scale = torch.ones(512, dtype=torch.float32)
        norm = torch.ones(128, dtype=torch.bfloat16)
        freqs = torch.zeros(128, 48, 2, 2, dtype=torch.bfloat16)
        producer = SimpleNamespace(
            spec=SimpleNamespace(k_input_shape=(1, 2, 256, 128), cta_k=64),
            anchor=SimpleNamespace(
                values=torch.ones(1, 2, 128, dtype=torch.bfloat16),
                indices=torch.full((1, 2), -1, dtype=torch.int32),
            ),
            q=torch.empty(1, 2, 1, 128, dtype=torch.int8),
            q_scale=torch.empty(1, 2, 32, dtype=torch.float32),
            k=torch.empty(1, 2, 256, 128, dtype=torch.int8),
            k_scale=torch.empty(1, 2, 16, dtype=torch.float32),
            _q_ranges=[],
            _k_ranges=[],
            _finalized=False,
        )
        with (
            mock.patch.object(
                torch.Tensor,
                "is_cuda",
                new_callable=mock.PropertyMock,
                return_value=True,
            ),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(8, 9)
            ),
            mock.patch.object(
                torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=1234),
            ),
            mock.patch.object(fused_kv.loader, "load", return_value=library),
            mock.patch.object(fused_kv.loader, "check") as check,
        ):
            v, summary = fused_kv.fused_h3_kv_from_int8(
                activation,
                weight,
                activation_scale,
                weight_scale,
                norm,
                freqs,
                producer,
                row_start=128,
                epsilon=1e-6,
            )

        self.assertEqual(tuple(v.shape), (1, 2, 128, 128))
        self.assertEqual(v.dtype, torch.bfloat16)
        self.assertEqual(tuple(summary.shape), (1, 2, 2, 128))
        self.assertEqual(summary.dtype, torch.bfloat16)
        self.assertEqual(call.args[12:18], (128, 512, 256, 256, 128, 1e-6))
        self.assertEqual(call.args[-1], 1234)
        self.assertEqual(producer._q_ranges, [(0, 1)])
        self.assertEqual(producer._k_ranges, [(128, 256)])
        self.assertTrue(torch.equal(producer.q, torch.zeros_like(producer.q)))
        self.assertTrue(torch.equal(producer.q_scale, torch.full_like(producer.q_scale, 1e-7)))
        check.assert_called_once_with(0, "fused_h3_kv_exact_128x256")

    def test_source_owns_only_the_measured_exact_configuration(self):
        source = (PACK / "native" / "src" / "h3_fused_k_cutlass.cu").read_text(
            encoding="utf-8"
        )
        api = (PACK / "native" / "src" / "h3_int8_attention_api.cu").read_text(
            encoding="utf-8"
        )
        self.assertIn("GemmShape<128, 256, 64>", source)
        self.assertIn("cutlass::arch::Sm80", source)
        self.assertIn(fused_kv._SYMBOL, api)
        self.assertNotIn("h3_int8_fused_kv_experiment", api)


if __name__ == "__main__":
    unittest.main()
