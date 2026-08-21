'''CPU-only tests for backend-aware runtime classification.'''

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

import torch  # noqa: E402

import h3_optimizations.environment as environment  # noqa: E402


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_nvidia_cuda_reports_capability(self):
        with mock.patch.object(
            environment,
            '_selected_device',
            return_value=torch.device('cuda', 1),
        ), mock.patch.object(
            torch.version,
            'hip',
            None,
        ), mock.patch.object(
            torch.version,
            'cuda',
            '12.8',
        ), mock.patch.object(
            torch.cuda,
            'is_available',
            return_value=True,
        ), mock.patch.object(
            torch.cuda,
            'get_device_name',
            return_value='fake NVIDIA',
        ), mock.patch.object(
            torch.cuda,
            'get_device_capability',
            return_value=(8, 9),
        ):
            detected = environment.RuntimeEnvironment.detect()

        self.assertEqual(detected.backend, environment.BACKEND_NVIDIA_CUDA)
        self.assertTrue(detected.cuda_available)
        self.assertEqual(detected.capability, (8, 9))
        self.assertEqual(detected.architecture, 'sm89')

    def test_rocm_is_not_sparse_cuda(self):
        with mock.patch.object(
            environment,
            '_selected_device',
            return_value=torch.device('cuda', 0),
        ), mock.patch.object(
            torch.version,
            'hip',
            '6.4',
        ), mock.patch.object(
            torch.version,
            'cuda',
            None,
        ), mock.patch.object(
            torch.cuda,
            'get_device_name',
            return_value='fake AMD',
        ):
            detected = environment.RuntimeEnvironment.detect()

        self.assertEqual(detected.backend, environment.BACKEND_ROCM)
        self.assertFalse(detected.cuda_available)
        self.assertIsNone(detected.capability)
        self.assertEqual(detected.architecture, 'rocm')

    def test_non_cuda_devices_remain_available_to_dense_h3(self):
        cases = (
            ('cpu', environment.BACKEND_CPU),
            ('mps', environment.BACKEND_MPS),
            ('xpu', environment.BACKEND_XPU),
        )
        for device, backend in cases:
            with self.subTest(device=device), mock.patch.object(
                environment,
                '_selected_device',
                return_value=torch.device(device),
            ):
                detected = environment.RuntimeEnvironment.detect()
            self.assertEqual(detected.backend, backend)
            self.assertFalse(detected.cuda_available)
            self.assertIsNone(detected.capability)


if __name__ == '__main__':
    unittest.main()
