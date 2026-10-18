'''CPU contracts for native sparse self-test numerical health gates.'''

import os
from pathlib import Path
import sys
import unittest
from unittest import mock
from types import SimpleNamespace

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

from h3_optimizations.native import selftest, int8_attention  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class NativeSelftestNumericsTests(unittest.TestCase):
    def test_packed_dense_dispatch_requires_export_and_numerical_pass(self):
        for capability, exported, passed, expected in (((8, 9), True, True, True), ((8, 9), False, True, False), ((8, 9), True, False, False), ((8, 0), True, True, False)):
            with self.subTest(capability=capability, exported=exported, passed=passed):
                library = SimpleNamespace(**({'h3_int8_packed_dense_attention': object()} if exported else {}))
                with mock.patch.object(torch.cuda, 'get_device_capability', return_value=capability), \
                     mock.patch.object(int8_attention.loader, 'load', return_value=library), \
                     mock.patch.object(selftest, '_load_result', return_value=(True, {'packed_dense_passed': passed})):
                    self.assertEqual(int8_attention.packed_dense_attention_is_available('cuda:0'), expected)

    def test_packed_dense_rejects_128kv_carrier_without_loading_binary(self):
        carrier = SimpleNamespace(cta_k=128, q=torch.zeros(1, 1, 64, 128))
        with mock.patch.object(int8_attention.loader, 'load') as load:
            with self.assertRaisesRegex(ValueError, '64KV carrier'):
                int8_attention.packed_int8_attention_from_prequantized(carrier)
        load.assert_not_called()

    @staticmethod
    def _reference():
        return torch.linspace(-0.5, 0.5, 4096, dtype=torch.bfloat16)

    def test_one_local_bf16_ulp_is_healthy_without_bit_identity(self):
        expected = self._reference()
        actual = expected.clone()
        actual[3072] = torch.nextafter(
            actual[3072],
            torch.tensor(float('inf'), dtype=torch.bfloat16),
        )

        passed, detail = selftest._sparse_output_health(actual, expected)

        self.assertTrue(passed)
        self.assertFalse(torch.equal(actual, expected))
        self.assertTrue(detail['finite'])
        self.assertLess(detail['rel_l2'], selftest._SPARSE_REL_L2_TOLERANCE)
        self.assertLess(detail['max_abs'], selftest._SPARSE_MAX_ABS_TOLERANCE)

    def test_distributed_corruption_fails_relative_l2_gate(self):
        expected = self._reference()
        actual = expected.float() * 1.01

        passed, detail = selftest._sparse_output_health(actual, expected)

        self.assertFalse(passed)
        self.assertGreaterEqual(
            detail['rel_l2'], selftest._SPARSE_REL_L2_TOLERANCE
        )
        self.assertLess(detail['max_abs'], selftest._SPARSE_MAX_ABS_TOLERANCE)

    def test_local_corruption_fails_max_absolute_gate(self):
        expected = self._reference()
        actual = expected.float()
        actual[3072] += 0.02

        passed, detail = selftest._sparse_output_health(actual, expected)

        self.assertFalse(passed)
        self.assertLess(detail['rel_l2'], selftest._SPARSE_REL_L2_TOLERANCE)
        self.assertGreaterEqual(
            detail['max_abs'], selftest._SPARSE_MAX_ABS_TOLERANCE
        )

    def test_nonfinite_output_fails_before_error_metrics(self):
        expected = self._reference()
        actual = expected.clone()
        actual[0] = float('nan')

        passed, detail = selftest._sparse_output_health(actual, expected)

        self.assertFalse(passed)
        self.assertEqual(
            detail,
            {
                'finite': False,
                'rel_l2': None,
                'max_abs': None,
                'passed': False,
            },
        )


class PackedSelftestNumericsTests(unittest.TestCase):
    @staticmethod
    def _reference():
        generator = torch.Generator().manual_seed(1234)
        return (torch.randn(4096, generator=generator) * 0.1).to(torch.bfloat16)

    @staticmethod
    def _noisy(expected, relative):
        generator = torch.Generator().manual_seed(5678)
        noise = torch.randn(expected.shape, generator=generator)
        noise *= relative * expected.float().norm() / noise.norm()
        return expected.float() + noise

    def test_packed_geometries_are_the_64x64_and_64x32_routes(self):
        self.assertEqual(selftest._PACKED_GEOMETRIES, ((64, 64), (64, 32)))
        self.assertIn((64, 32), selftest._SPARSE_PARITY_GEOMETRIES)
        self.assertNotIn((64, 32), selftest._PRODUCTION_GEOMETRIES)

    def test_packed_quantization_noise_is_healthy(self):
        expected = self._reference()
        actual = self._noisy(expected, 0.006)

        passed, detail = selftest._packed_output_health(actual, expected)

        self.assertTrue(passed)
        self.assertFalse(selftest._sparse_output_health(actual, expected)[0])
        self.assertLess(abs(detail['gain'] - 1.0), selftest._PACKED_GAIN_TOLERANCE)

    def test_packed_gate_accepts_bit_identical_output(self):
        expected = self._reference()
        passed, detail = selftest._packed_output_health(expected.clone(), expected)
        self.assertTrue(passed)
        self.assertEqual(detail['gain'], 1.0)

    def test_packed_distributed_scaling_fails_gain_gate(self):
        expected = self._reference()
        actual = expected.float() * 1.01

        passed, detail = selftest._packed_output_health(actual, expected)

        self.assertFalse(passed)
        self.assertLess(detail['rel_l2'], selftest._PACKED_REL_L2_TOLERANCE)
        self.assertGreaterEqual(
            abs(detail['gain'] - 1.0), selftest._PACKED_GAIN_TOLERANCE
        )

    def test_packed_excess_noise_fails_relative_l2_gate(self):
        expected = self._reference()
        actual = self._noisy(expected, 0.02)

        passed, detail = selftest._packed_output_health(actual, expected)

        self.assertFalse(passed)
        self.assertGreaterEqual(detail['rel_l2'], selftest._PACKED_REL_L2_TOLERANCE)

    def test_packed_local_corruption_fails_max_absolute_gate(self):
        expected = self._reference()
        actual = expected.float()
        actual[3072] += 0.02

        passed, detail = selftest._packed_output_health(actual, expected)

        self.assertFalse(passed)
        self.assertGreaterEqual(
            detail['max_abs'], selftest._SPARSE_MAX_ABS_TOLERANCE
        )

    def test_packed_nonfinite_output_fails(self):
        expected = self._reference()
        actual = expected.clone()
        actual[0] = float('nan')

        passed, detail = selftest._packed_output_health(actual, expected)

        self.assertFalse(passed)
        self.assertFalse(detail['finite'])


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
