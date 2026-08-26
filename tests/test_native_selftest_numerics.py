'''CPU contracts for native sparse self-test numerical health gates.'''

import os
from pathlib import Path
import sys
import unittest

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

from h3_optimizations.native import selftest  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class NativeSelftestNumericsTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
