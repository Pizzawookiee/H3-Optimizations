'''CPU contracts for H3 FP8 runtime capability selection.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import h3_optimizations.apply as apply_module  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class FP8CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.ada = SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
            device_index=0,
        )

    def test_comfy_rejection_wins_even_on_sm89(self):
        with mock.patch.object(
            apply_module.comfy.quant_ops,
            '_CK_AVAILABLE',
            True,
        ), mock.patch.object(
            apply_module.comfy.model_management,
            'supports_fp8_compute',
            return_value=False,
        ) as probe:
            self.assertFalse(
                apply_module._fp8_execution_available(self.ada)
            )
        probe.assert_called_once()

    def test_comfy_acceptance_enables_sm89(self):
        with mock.patch.object(
            apply_module.comfy.quant_ops,
            '_CK_AVAILABLE',
            True,
        ), mock.patch.object(
            apply_module.comfy.model_management,
            'supports_fp8_compute',
            return_value=True,
        ):
            self.assertTrue(
                apply_module._fp8_execution_available(self.ada)
            )

    def test_kitchen_is_required_by_our_fp8_provider(self):
        with mock.patch.object(
            apply_module.comfy.quant_ops,
            '_CK_AVAILABLE',
            False,
        ), mock.patch.object(
            apply_module.comfy.model_management,
            'supports_fp8_compute',
            side_effect=AssertionError('must stop before probing FP8 compute'),
        ):
            self.assertFalse(
                apply_module._fp8_execution_available(self.ada)
            )


if __name__ == '__main__':
    unittest.main()