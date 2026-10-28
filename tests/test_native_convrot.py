"""CPU-visible contracts for the standalone ConvRot-256 quantizer."""

import ctypes
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.native import convrot, loader  # noqa: E402


class _Function:
    def __call__(self, *_args):
        return 0


class _Library:
    def __init__(self, *, include_convrot):
        self.include_convrot = include_convrot

    def __getattr__(self, name):
        if name == convrot._SYMBOL and not self.include_convrot:
            raise AttributeError(name)
        function = _Function()
        setattr(self, name, function)
        return function


class NativeConvRotTests(unittest.TestCase):
    def test_loader_treats_the_symbol_as_an_additive_abi4_capability(self):
        old_library = _Library(include_convrot=False)
        self.assertIs(loader._bind(old_library), old_library)

        new_library = _Library(include_convrot=True)
        loader._bind(new_library)
        function = getattr(new_library, convrot._SYMBOL)
        self.assertIs(function.restype, ctypes.c_int)
        self.assertEqual(
            function.argtypes,
            [ctypes.c_void_p] * 3
            + [ctypes.c_int64] * 2
            + [ctypes.c_size_t],
        )
        fused = getattr(new_library, convrot._RMS_ADALN_SYMBOL)
        self.assertIs(fused.restype, ctypes.c_int)
        self.assertEqual(
            fused.argtypes,
            [ctypes.c_void_p] * 6
            + [ctypes.c_int64] * 2
            + [ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_size_t],
        )

    def test_fake_native_call_preserves_kitchen_tensor_contract(self):
        class NativeCall:
            def __init__(self):
                self.args = None

            def __call__(self, *args):
                self.args = args
                return 0

        call = NativeCall()
        library = SimpleNamespace(**{convrot._SYMBOL: call})
        input = torch.zeros(3, 512, dtype=torch.bfloat16)
        with (
            mock.patch.object(convrot.loader, 'load', return_value=library),
            mock.patch.object(convrot.loader, 'check') as check,
            mock.patch.object(
                torch.Tensor, 'is_cuda', new_callable=mock.PropertyMock,
                return_value=True,
            ),
            mock.patch.object(
                torch.cuda, 'current_stream',
                return_value=SimpleNamespace(cuda_stream=1234),
            ),
        ):
            output, scales = convrot.quantize_int8_rowwise_convrot256(input)

        self.assertEqual(tuple(output.shape), (3, 512))
        self.assertEqual(output.dtype, torch.int8)
        self.assertEqual(tuple(scales.shape), (3, 1))
        self.assertEqual(scales.dtype, torch.float32)
        self.assertEqual(call.args[3:], (3, 512, 1234))
        check.assert_called_once_with(0, 'quantize_bf16_rowwise_convrot256')

    def test_python_boundary_rejects_non_bf16_and_bad_group_width(self):
        with self.assertRaisesRegex(TypeError, 'torch.bfloat16'):
            convrot.quantize_int8_rowwise_convrot256(torch.zeros(2, 256))

        input = torch.zeros(2, 255, dtype=torch.bfloat16)
        with mock.patch.object(
            torch.Tensor, 'is_cuda', new_callable=mock.PropertyMock,
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, 'multiple of 256'):
                convrot.quantize_int8_rowwise_convrot256(input)

    def test_fused_norm_adaln_fake_call_exposes_both_modes(self):
        class NativeCall:
            def __init__(self):
                self.calls = []

            def __call__(self, *args):
                self.calls.append(args)
                return 0

        call = NativeCall()
        library = SimpleNamespace(**{convrot._RMS_ADALN_SYMBOL: call})
        input = torch.zeros(3, 512, dtype=torch.bfloat16)
        norm = torch.ones(512, dtype=torch.bfloat16)
        shift = torch.zeros(512, dtype=torch.float32)
        scale = torch.zeros(512, dtype=torch.float32)
        with (
            mock.patch.object(convrot.loader, 'load', return_value=library),
            mock.patch.object(convrot.loader, 'check') as check,
            mock.patch.object(
                torch.Tensor, 'is_cuda', new_callable=mock.PropertyMock,
                return_value=True,
            ),
            mock.patch.object(
                torch.cuda, 'current_stream',
                return_value=SimpleNamespace(cuda_stream=1234),
            ),
        ):
            for mode in (
                convrot.RMS_ADALN_BF16_STAGED,
                convrot.RMS_ADALN_FP32_THROUGH,
            ):
                output, output_scale = (
                    convrot.quantize_rms_adaln_int8_rowwise_convrot256(
                        input,
                        norm,
                        shift,
                        scale,
                        1e-5,
                        mode=mode,
                    )
                )
                self.assertEqual(tuple(output.shape), (3, 512))
                self.assertEqual(tuple(output_scale.shape), (3, 1))

        self.assertEqual(len(call.calls), 2)
        self.assertEqual(call.calls[0][6:8], (3, 512))
        self.assertEqual(call.calls[0][9:12], (1, 0, 1234))
        self.assertEqual(call.calls[1][9:12], (1, 1, 1234))
        self.assertEqual(check.call_count, 2)

    def test_fused_norm_adaln_boundary_rejects_mismatched_inputs(self):
        input = torch.zeros(2, 512, dtype=torch.bfloat16)
        norm = torch.ones(512, dtype=torch.bfloat16)
        shift = torch.zeros(512, dtype=torch.float32)
        scale = torch.zeros(512, dtype=torch.float32)
        with mock.patch.object(
            torch.Tensor, 'is_cuda', new_callable=mock.PropertyMock,
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, 'shape'):
                convrot.quantize_rms_adaln_int8_rowwise_convrot256(
                    input, norm[:-1], shift, scale, 1e-5,
                    mode=convrot.RMS_ADALN_BF16_STAGED,
                )
            with self.assertRaisesRegex(TypeError, 'same BF16 or FP32'):
                convrot.quantize_rms_adaln_int8_rowwise_convrot256(
                    input, norm, shift.to(torch.bfloat16), scale, 1e-5,
                    mode=convrot.RMS_ADALN_BF16_STAGED,
                )
            with self.assertRaisesRegex(ValueError, 'mode must be one of'):
                convrot.quantize_rms_adaln_int8_rowwise_convrot256(
                    input, norm, shift, scale, 1e-5, mode='unknown',
                )

    def test_old_abi4_binary_reports_the_optional_capability_absent(self):
        with mock.patch.object(
            convrot.loader, 'load', return_value=SimpleNamespace()
        ):
            self.assertFalse(convrot.int8_rowwise_convrot256_is_available())

    def test_native_source_owns_the_exact_bf16_64_lane_path(self):
        source = (PACK / 'native' / 'src' / 'convrot_quantize.cu').read_text(
            encoding='utf-8'
        )
        cmake = (PACK / 'native' / 'CMakeLists.txt').read_text(encoding='utf-8')
        api = (
            PACK / 'native' / 'src' / 'h3_int8_attention_api.cu'
        ).read_text(encoding='utf-8')

        self.assertIn('src/convrot_quantize.cu', cmake)
        self.assertIn('constexpr int kGroupThreads = 64;', source)
        self.assertIn('__float2bfloat16_rn(value)', source)
        self.assertIn(
            'quantize_bf16_rms_adaln_rowwise_convrot256_kernel', source
        )
        self.assertIn('if constexpr (StagedBF16)', source)
        self.assertIn('columns == 6144', source)
        self.assertNotIn('#include <cutlass', source)
        self.assertIn(convrot._SYMBOL, api)
        self.assertIn(convrot._RMS_ADALN_SYMBOL, api)
        self.assertIn('h3_int8_abi_version() noexcept { return 4; }', api)


if __name__ == '__main__':
    unittest.main()
