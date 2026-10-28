"""CPU checks for quiet Windows DLL load failures."""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(PACK.parents[1]))
ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
from h3_optimizations.native import loader
sys.argv = [sys.argv[0], *ARGS]


class NativeLoadErrorTests(unittest.TestCase):
    def test_windows_suppresses_dialog_and_restores_flags_on_failure(self):
        kernel = mock.Mock()
        kernel.GetThreadErrorMode.return_value = 0x8000
        kernel.SetThreadErrorMode.return_value = 1
        with mock.patch.object(loader.platform, 'system', return_value='Windows'), \
                mock.patch.object(loader.ctypes, 'WinDLL', return_value=kernel, create=True), \
                mock.patch.object(loader.ctypes, 'CDLL', side_effect=OSError('bad image')):
            with self.assertRaisesRegex(OSError, 'bad image'):
                loader._load_shared_library(Path('broken.dll'))
        self.assertEqual(kernel.SetThreadErrorMode.call_args_list,
                         [mock.call(0x8001, None), mock.call(0x8000, None)])

    def test_windows_restores_flags_on_success(self):
        kernel = mock.Mock()
        kernel.GetThreadErrorMode.return_value = 2
        kernel.SetThreadErrorMode.return_value = 1
        library = object()
        with mock.patch.object(loader.platform, 'system', return_value='Windows'), \
                mock.patch.object(loader.ctypes, 'WinDLL', return_value=kernel, create=True), \
                mock.patch.object(loader.ctypes, 'CDLL', return_value=library):
            self.assertIs(loader._load_shared_library(Path('valid.dll')), library)
        self.assertEqual(kernel.SetThreadErrorMode.call_args_list,
                         [mock.call(3, None), mock.call(2, None)])

    def test_non_windows_load_is_unchanged(self):
        with mock.patch.object(loader.platform, 'system', return_value='Linux'), \
                mock.patch.object(loader.ctypes, 'CDLL') as cdll:
            loader._load_shared_library(Path('kernel.so'))
        cdll.assert_called_once_with('kernel.so')


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *ARGS])
