"""CPU-visible contracts for the H3-owned SM89 Sparse Sage SA2++ kernel."""

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
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.attention.sparse import sparse_sage as integration  # noqa: E402
from h3_optimizations.native import loader  # noqa: E402
from h3_optimizations.native import sparse_sage as native  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class _Function:
    def __init__(self):
        self.args = None

    def __call__(self, *args):
        self.args = args
        return 0


class _Library:
    def __init__(self, include_sa2pp):
        self.include_sa2pp = include_sa2pp

    def __getattr__(self, name):
        if name == native._SYMBOL and not self.include_sa2pp:
            raise AttributeError(name)
        function = _Function()
        setattr(self, name, function)
        return function


class NativeSparseSageSA2PPTests(unittest.TestCase):
    def test_loader_treats_sa2pp_as_an_additive_abi4_symbol(self):
        old_library = _Library(False)
        self.assertIs(loader._bind(old_library), old_library)

        new_library = _Library(True)
        loader._bind(new_library)
        function = getattr(new_library, native._SYMBOL)
        self.assertIs(function.restype, ctypes.c_int)
        self.assertEqual(
            function.argtypes,
            [ctypes.c_void_p] * 10
            + [ctypes.c_int] * 18
            + [ctypes.c_float, ctypes.c_int, ctypes.c_size_t],
        )

    def test_availability_is_exactly_sm89_and_requires_the_symbol(self):
        library = SimpleNamespace(**{native._SYMBOL: _Function()})
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
            mock.patch.object(native.loader, "load", return_value=library),
        ):
            self.assertTrue(native.sparse_sage_sa2pp_is_available("cuda"))
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            mock.patch.object(native.loader, "load") as load,
        ):
            self.assertFalse(native.sparse_sage_sa2pp_is_available("cuda"))
            load.assert_not_called()

    def test_fake_call_forwards_shapes_strides_dtype_and_stream(self):
        call = _Function()
        library = SimpleNamespace(**{native._SYMBOL: call})
        q = torch.zeros((1, 2, 129, 128), dtype=torch.int8)
        k = torch.zeros_like(q)
        v = torch.zeros((1, 2, 128, 256), dtype=torch.float8_e4m3fn)
        output = torch.zeros(q.shape, dtype=torch.bfloat16)
        lut = torch.zeros((1, 2, 2, 3), dtype=torch.int32)
        valid = torch.ones((1, 2, 2), dtype=torch.int32)
        threshold = torch.full((2,), 50.0)
        q_scale = torch.ones((1, 2, 2))
        k_scale = torch.ones((1, 2, 3))
        v_scale = torch.ones((1, 2, 128))
        with (
            mock.patch.object(
                torch.Tensor, "is_cuda", new_callable=mock.PropertyMock,
                return_value=True,
            ),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
            mock.patch.object(
                torch.cuda, "current_stream",
                return_value=SimpleNamespace(cuda_stream=1234),
            ),
            mock.patch.object(native.loader, "load", return_value=library),
            mock.patch.object(native.loader, "check") as check,
        ):
            native.sparse_sage_sa2pp(
                q, k, v, output, lut, valid, threshold,
                q_scale, k_scale, v_scale,
            )

        self.assertEqual(call.args[10:16], (1, 129, 129, 2, 2, 128))
        self.assertEqual(call.args[16:28], (
            q.stride(0), q.stride(2), q.stride(1),
            k.stride(0), k.stride(2), k.stride(1),
            v.stride(0), v.stride(1), v.stride(2),
            output.stride(0), output.stride(2), output.stride(1),
        ))
        self.assertAlmostEqual(call.args[28], 128 ** -0.5)
        self.assertEqual(call.args[29:], (2, 1234))
        check.assert_called_once_with(0, "sparse_sage_sa2pp_sm89")

    def test_sm89_spec_prefers_native_sa2pp_and_old_binary_falls_back(self):
        old_kernel = _Function()
        qattn = SimpleNamespace(**{integration._SM89_F16_KERNEL: old_kernel})
        fused = object()
        with (
            mock.patch.dict(sys.modules, {"spas_sage_attn": SimpleNamespace()}),
            mock.patch.object(
                integration, "_load_qattn_surface", return_value=(qattn, "split")
            ),
            mock.patch.object(integration, "_load_fused_surface", return_value=fused),
            mock.patch.object(
                integration.importlib.metadata, "version", return_value="test"
            ),
            mock.patch.object(
                integration, "sparse_sage_sa2pp_is_available", return_value=True
            ) as available,
        ):
            spec = integration.load_sparse_sage_spec(
                capability=(8, 9), cuda_version=(13, 0)
            )
            available.assert_called_once_with(capability=(8, 9))
            available.reset_mock()
            available.return_value = False
            fallback = integration.load_sparse_sage_spec(
                capability=(8, 9), cuda_version=(13, 0)
            )
            available.assert_called_once_with(capability=(8, 9))
        self.assertIs(spec.kernel, integration._native_sm89_sa2pp_kernel)
        self.assertEqual(spec.kernel_name, integration._NATIVE_SM89_SA2PP_KERNEL)
        self.assertEqual(spec.accumulator, "fp32+fp16")
        self.assertEqual(spec.extension_layout, "split+h3-native")
        self.assertIs(fallback.kernel, old_kernel)
        self.assertEqual(fallback.accumulator, "f16")

    def test_source_owns_only_the_measured_sm89_geometry(self):
        source = (
            PACK / "native" / "src" / "sparge_attention"
            / "sm89_sparse_sage_sa2pp.cu"
        ).read_text(encoding="utf-8")
        cmake = (PACK / "native" / "CMakeLists.txt").read_text(encoding="utf-8")
        api = (PACK / "native" / "src" / "h3_int8_attention_api.cu").read_text(
            encoding="utf-8"
        )
        provenance = (PACK / "native" / "src" / "PROVENANCE").read_text(
            encoding="utf-8"
        )
        self.assertIn("constexpr uint32_t CTA_Q = 128", source)
        self.assertIn("constexpr uint32_t CTA_K = 64", source)
        self.assertIn("true, PVThresholdMode::kPerBlock", source)
        self.assertIn("CUDA_ARCHITECTURES 89-real", cmake)
        self.assertIn(native._SYMBOL, api)
        self.assertIn("569addefbd1dd8b7b412eae877bd6a23a2c1ccf4", provenance)


if __name__ == "__main__":
    unittest.main()
