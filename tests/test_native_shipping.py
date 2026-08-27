'''CPU contracts for packaged native backend shipping.'''

import os
from pathlib import Path
import re
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
from h3_optimizations.native import artifacts, int8_attention, loader, selftest  # noqa: E402


BIN_DIR = PACK / 'native' / 'bin'


class NativeShippingTests(unittest.TestCase):
    def test_native_binaries_are_packaged(self):
        windows_binary = BIN_DIR / 'h3_int8_attention_v4.dll'
        linux_binary = BIN_DIR / 'libh3_int8_attention.so'

        self.assertEqual(loader._LIBRARY_NAMES['Windows'], windows_binary.name)
        self.assertGreater(windows_binary.stat().st_size, 1_000_000)
        self.assertEqual(windows_binary.read_bytes()[:2], b'MZ')
        self.assertGreater(linux_binary.stat().st_size, 1_000_000)
        self.assertEqual(linux_binary.read_bytes()[:4], b'\x7fELF')
        self.assertEqual(
            (BIN_DIR / 'BUILD_ID').read_text(encoding='utf-8').strip(),
            artifacts.NATIVE_BUILD,
        )

    def test_linux_binary_keeps_old_libstdcxx_compatibility(self):
        contents = (BIN_DIR / 'libh3_int8_attention.so').read_bytes()
        versions = {
            tuple(int(part) for part in match.split(b'.'))
            for match in re.findall(rb'GLIBCXX_(\d+\.\d+\.\d+)', contents)
        }

        self.assertTrue(versions)
        self.assertLessEqual(max(versions), (3, 4, 21))

    def test_package_versions_match(self):
        metadata = tomllib.loads((PACK / 'pyproject.toml').read_text(encoding='utf-8'))
        self.assertEqual(metadata['project']['version'], __version__)

    def test_frost_artifact_has_reproducible_source_and_license(self):
        frost = PACK / 'native' / 'frost'
        for name in (
            'h3_frost_bf16_sm89.cubin',
            'h3_frost_bf16_sm89.symbol',
            'frost_h3.patch',
            'compile_sm89.py',
            'Dockerfile',
            'PROVENANCE',
            'LICENSE.txt',
        ):
            self.assertTrue((frost / name).is_file(), name)
        provenance = (frost / 'PROVENANCE').read_text(encoding='utf-8')
        self.assertIn('ae8705effeea3804585b6aca554beaca1a76a3da', provenance)
        self.assertIn('64690d05f52335bd252c6ecd9ad5d470ad5cff1df0d48f59c35396d0f775188c', provenance)

    def test_native_availability_requires_selftest(self):
        with (
            mock.patch.object(int8_attention.torch.cuda, 'is_available', return_value=True),
            mock.patch.object(int8_attention.loader, 'is_available', return_value=True),
            mock.patch.object(int8_attention.torch.cuda, 'get_device_capability', return_value=(8, 9)),
            mock.patch('h3_optimizations.native.selftest.check', return_value=True) as selftest_check,
        ):
            self.assertTrue(int8_attention.int8_attention_is_available('cuda'))
            selftest_check.assert_called_once_with('cuda')

    def test_native_availability_allows_sm75_only_after_selftest(self):
        with (
            mock.patch.object(int8_attention.torch.cuda, 'is_available', return_value=True),
            mock.patch.object(int8_attention.loader, 'is_available', return_value=True),
            mock.patch.object(int8_attention.torch.cuda, 'get_device_capability', return_value=(7, 5)),
            mock.patch('h3_optimizations.native.selftest.check', return_value=True) as selftest_check,
        ):
            self.assertTrue(int8_attention.int8_attention_is_available('cuda'))
            selftest_check.assert_called_once_with('cuda')

    def test_native_availability_rejects_unsupported_capability_before_selftest(self):
        with (
            mock.patch.object(int8_attention.torch.cuda, 'is_available', return_value=True),
            mock.patch.object(int8_attention.loader, 'is_available', return_value=True),
            mock.patch.object(int8_attention.torch.cuda, 'get_device_capability', return_value=(7, 0)),
            mock.patch('h3_optimizations.native.selftest.check') as selftest_check,
        ):
            self.assertFalse(int8_attention.int8_attention_is_available('cuda'))
            selftest_check.assert_not_called()

    def test_selftest_covers_every_shipped_sparse_geometry(self):
        self.assertEqual(
            selftest._SPARSE_PARITY_GEOMETRIES,
            int8_attention.SPARSE_GEOMETRIES,
        )

    def test_selftest_cache_key_includes_revision(self):
        with (
            mock.patch.object(
                selftest.torch.cuda, 'get_device_capability', return_value=(12, 0)
            ),
            mock.patch.object(
                selftest.torch.cuda,
                'get_device_name',
                return_value='NVIDIA GeForce RTX 5070 Ti',
            ),
            mock.patch(
                'h3_optimizations.native.bootstrap.installed_build_id',
                return_value='native-v5',
            ),
        ):
            key = selftest._cache_key('cuda')

        self.assertEqual(
            key,
            'sm120|native-v5|%s|NVIDIA GeForce RTX 5070 Ti'
            % selftest._SELFTEST_REVISION,
        )


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
