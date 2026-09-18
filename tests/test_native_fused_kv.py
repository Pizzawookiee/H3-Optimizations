"""CPU-visible contracts for the direct native H3 grouped K/V producer."""

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

from h3_optimizations.native import fused_kv, loader, selftest  # noqa: E402


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
    def test_availability_requires_device_selftest(self):
        library = SimpleNamespace(**{fused_kv._SYMBOL: object()})
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
            mock.patch.object(fused_kv.loader, "load", return_value=library),
            mock.patch.object(selftest, "fused_kv_check", return_value=False) as check,
        ):
            self.assertFalse(fused_kv.fused_h3_kv_is_available("cuda"))
            check.return_value = True
            self.assertTrue(fused_kv.fused_h3_kv_is_available("cuda"))
        self.assertEqual(check.call_args_list, [mock.call("cuda"), mock.call("cuda")])

    def test_loader_binds_additive_symbol(self):
        old_library = _Library(False)
        self.assertIs(loader._bind(old_library), old_library)
        new_library = _Library(True)
        loader._bind(new_library)
        function = getattr(new_library, fused_kv._SYMBOL)
        self.assertIs(function.restype, ctypes.c_int)
        self.assertEqual(
            function.argtypes,
            [ctypes.c_void_p] * 12
            + [ctypes.c_int64] * 4
            + [ctypes.c_int] * 2
            + [ctypes.c_float, ctypes.c_size_t],
        )

    def test_fake_call_writes_existing_k_carrier_and_returns_only_v(self):
        call = _Function()
        library = SimpleNamespace(**{fused_kv._SYMBOL: call})
        rows, hidden, full_rows, cta = 128, 256, 384, 64
        activation = torch.zeros(rows, hidden, dtype=torch.int8)
        weight = torch.zeros(256, hidden, dtype=torch.int8)
        activation_scale = torch.ones(rows, 1, dtype=torch.float32)
        weight_scale = torch.ones(256, dtype=torch.float32)
        norm = torch.ones(128, dtype=torch.bfloat16)
        freqs = torch.zeros(rows, 48, 2, 2, dtype=torch.bfloat16)
        anchor = torch.zeros(128, dtype=torch.bfloat16)
        anchor_index = torch.tensor([-1], dtype=torch.int32)
        k_out = torch.empty(full_rows, 128, dtype=torch.int8)
        k_scale = torch.empty(((full_rows + cta - 1) // cta) * 4, dtype=torch.float32)
        summary = torch.empty((full_rows + cta - 1) // cta, 128, dtype=torch.bfloat16)
        with (
            mock.patch.object(torch.Tensor, "is_cuda", new_callable=mock.PropertyMock, return_value=True),
            mock.patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=77)),
            mock.patch.object(fused_kv.loader, "load", return_value=library),
            mock.patch.object(fused_kv.loader, "check") as check,
        ):
            v = fused_kv.fused_h3_kv_from_int8(
                activation, weight, activation_scale, weight_scale, norm, freqs,
                anchor, anchor_index, k_out, k_scale, summary,
                full_rows=full_rows, k_start=128, cta_k=cta,
                full_k_length=full_rows, epsilon=1e-6,
            )
        self.assertEqual(tuple(v.shape), (rows, 128))
        self.assertEqual(v.dtype, torch.bfloat16)
        self.assertEqual(call.args[12:19], (rows, hidden, full_rows, 128, cta, full_rows, 1e-6))
        self.assertEqual(call.args[-1], 77)
        check.assert_called_once_with(0, "fused_h3_kv_exact_128x256")

    def test_source_is_cutlass_epilogue_and_ships_in_build(self):
        source = (PACK / "native" / "src" / "h3_fused_kv_cutlass.cu").read_text(encoding="utf-8")
        cmake = (PACK / "native" / "CMakeLists.txt").read_text(encoding="utf-8")
        api = (PACK / "native" / "src" / "h3_int8_attention_api.cu").read_text(encoding="utf-8")
        self.assertIn("VisitorH3KVStore", source)
        self.assertIn("GemmShape<128, 256, 64>", source)
        self.assertIn("params_ptr->k_scale", source)
        self.assertIn("params_ptr->summary", source)
        self.assertIn("params_ptr->v", source)
        self.assertIn("src/h3_fused_kv_cutlass.cu", cmake)
        self.assertIn(fused_kv._SYMBOL, api)


if __name__ == "__main__":
    unittest.main()
