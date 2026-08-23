'''ROCm must never enter the vendored CUDA native backend.'''

import unittest
from unittest import mock

from h3_optimizations.native import loader


class RocmNativeGateTests(unittest.TestCase):
    def setUp(self):
        self.library = loader._library
        self.load_error = loader._load_error
        loader._library = None
        loader._load_error = None

    def tearDown(self):
        loader._library = self.library
        loader._load_error = self.load_error

    def test_rocm_is_rejected_before_ctypes_load(self):
        with (
            mock.patch.object(loader, '_is_rocm_runtime', return_value=True),
            mock.patch.object(loader.ctypes, 'CDLL') as cdll,
        ):
            self.assertFalse(loader.is_available())
            cdll.assert_not_called()
            self.assertIn('ROCm/HIP detected', loader.unavailable_reason())

    def test_rocm_rejection_is_not_reported_as_an_sm_architecture(self):
        with mock.patch.object(loader, '_is_rocm_runtime', return_value=True):
            with self.assertRaisesRegex(loader.NativeUnavailableError, 'ROCm/HIP'):
                loader.load()


if __name__ == '__main__':
    unittest.main()
