'''CPU contracts for packaged native backend shipping.'''

import os
from pathlib import Path
import sys
import tomllib
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

from h3_optimizations import __version__  # noqa: E402
from h3_optimizations.native import artifacts, int8_attention  # noqa: E402


BIN_DIR = PACK / 'native' / 'bin'


class NativeShippingTests(unittest.TestCase):
    def test_native_binaries_are_packaged(self):
        windows_binary = BIN_DIR / 'h3_int8_attention.dll'
        linux_binary = BIN_DIR / 'libh3_int8_attention.so'

        self.assertGreater(windows_binary.stat().st_size, 1_000_000)
        self.assertEqual(windows_binary.read_bytes()[:2], b'MZ')
        self.assertGreater(linux_binary.stat().st_size, 1_000_000)
        self.assertEqual(linux_binary.read_bytes()[:4], b'\x7fELF')
        self.assertEqual(
            (BIN_DIR / 'BUILD_ID').read_text(encoding='utf-8').strip(),
            artifacts.NATIVE_BUILD,
        )

    def test_package_versions_match(self):
        metadata = tomllib.loads((PACK / 'pyproject.toml').read_text(encoding='utf-8'))
        self.assertEqual(metadata['project']['version'], __version__)

    def test_native_availability_requires_selftest(self):
        with (
            mock.patch.object(int8_attention.torch.cuda, 'is_available', return_value=True),
            mock.patch.object(int8_attention.loader, 'is_available', return_value=True),
            mock.patch.object(int8_attention.torch.cuda, 'get_device_capability', return_value=(8, 9)),
            mock.patch('h3_optimizations.native.selftest.check', return_value=True) as selftest,
        ):
            self.assertTrue(int8_attention.int8_attention_is_available('cuda'))
            selftest.assert_called_once_with('cuda')

    def test_native_availability_rejects_unsupported_capability_before_selftest(self):
        with (
            mock.patch.object(int8_attention.torch.cuda, 'is_available', return_value=True),
            mock.patch.object(int8_attention.loader, 'is_available', return_value=True),
            mock.patch.object(int8_attention.torch.cuda, 'get_device_capability', return_value=(7, 5)),
            mock.patch('h3_optimizations.native.selftest.check') as selftest,
        ):
            self.assertFalse(int8_attention.int8_attention_is_available('cuda'))
            selftest.assert_not_called()


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
