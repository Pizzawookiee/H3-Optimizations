'''CPU-only tests for the Sparse Sage startup installer.'''

from contextlib import nullcontext
import os
from pathlib import Path
import runpy
import subprocess
import sys
import unittest
from unittest.mock import patch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations import sparse_install  # noqa: E402


def nvidia_runtime(**overrides):
    runtime = {
        'ok': True,
        'torch_version': '2.14.0.dev20260809+cu130',
        'cuda_version': '13.0',
        'hip_version': None,
        'backend': 'nvidia_cuda',
        'accelerator_available': True,
        'capability': [12, 0],
        'device_name': 'fake SM120',
        'sparse_compatible': False,
        'sparse_error': None,
    }
    runtime.update(overrides)
    return runtime


class SparseInstallTests(unittest.TestCase):
    def test_sparse_abi_module_imports_without_comfyui(self):
        if sparse_install.importlib.util.find_spec('torch') is None:
            self.skipTest('Torch is not installed in this test interpreter')
        code = (
            'import sys; '
            'sys.path.insert(0, sys.argv[1]); '
            'import h3_optimizations.attention.sparse.sparse_sage'
        )
        result = subprocess.run(
            [sys.executable, '-I', '-c', code, str(PACK)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_every_verified_wheel_is_hash_locked(self):
        cases = (
            (
                '2.5.1+cu124',
                '12.4',
                'e2bfa39fd31443124839c0273ad06845393daf9d3450449748b4bab899bf029e',
            ),
            (
                '2.6.0+cu126',
                '12.6',
                'd847d09ea0b053bd3c2581b83c48113f235269363918bc8c3e76c5ff0f2a6d87',
            ),
            (
                '2.7.1+cu128',
                '12.8',
                '425afea0544f948f5236199f1e23f13e821a5d13cf761035ab9723b5de25e802',
            ),
            (
                '2.8.0+cu128',
                '12.8',
                '6e28ecf05956fcfdcfa87af064e5ba282b8c721246702f78b36a7ba0aa2b76c3',
            ),
            (
                '2.9.0+cu128',
                '12.8',
                '258eadcd4433892ac445ab7e7a7f07de1ede3897c71334b4ec08b26da894bca3',
            ),
            (
                '2.9.0+cu130',
                '13.0',
                '2b34a5abe45c0ea51872f78dc1d0b407893ac5ecab3bf71425a73e769e47f27a',
            ),
        )
        for torch_version, cuda_version, digest in cases:
            with self.subTest(torch=torch_version, cuda=cuda_version):
                wheel = sparse_install._wheel_for(
                    'Windows',
                    'AMD64',
                    torch_version,
                    cuda_version,
                )
                self.assertTrue(wheel.endswith('#sha256=' + digest))

    def test_stable_abi_wheel_supports_newer_torch(self):
        wheel = sparse_install._wheel_for(
            'Windows',
            'AMD64',
            '2.14.0.dev20260809+cu130',
            '13.0',
        )
        self.assertIn('cu130torch2.9.0andhigher.post4', wheel)
        self.assertIn(
            'sha256=2b34a5abe45c0ea51872f78dc1d0b407893ac5ecab3bf71425a73e769e47f27a',
            wheel,
        )

    def test_exact_wheel_matches_older_torch(self):
        wheel = sparse_install._wheel_for(
            'Windows',
            'x86_64',
            '2.8.0+cu128',
            '12.8',
        )
        self.assertIn('cu128torch2.8.0.post3', wheel)

    def test_unsupported_runtime_has_no_wheel(self):
        self.assertIsNone(
            sparse_install._wheel_for(
                'Linux',
                'x86_64',
                '2.9.0+cu128',
                '12.8',
            )
        )
        self.assertIsNone(
            sparse_install._wheel_for(
                'Windows',
                'AMD64',
                '2.8.0+cu126',
                '12.6',
            )
        )

    def test_linux_uses_pinned_source_build(self):
        self.assertEqual(
            sparse_install._linux_source_for(
                'Linux',
                'x86_64',
                '2.9.0+cu128',
                '12.8',
            ),
            'git+https://github.com/woct0rdho/SpargeAttn.git@'
            '067d80cb6b76345c7b8be40e86c7d19a3cf7c4eb',
        )
        self.assertIsNone(
            sparse_install._linux_source_for(
                'Linux',
                'aarch64',
                '2.9.0+cu128',
                '12.8',
            )
        )
        self.assertEqual(
            sparse_install._linux_source_for(
                'Linux',
                'x86_64',
                '2.9.0',
                '',
            ),
            sparse_install.SPARGE_SOURCE,
        )
        self.assertIsNone(
            sparse_install._linux_source_for(
                'Linux',
                'x86_64',
                '2.2.0+cu118',
                '11.8',
            )
        )

    def test_pinned_source_patch_enables_ninja_and_bounds_nvcc_threads(self):
        class SetupFile:
            def __init__(self):
                self.source = '\n'.join(
                    (
                        sparse_install._SPARGE_NINJA_DISABLED,
                        sparse_install._SPARGE_NVCC_ALL_CORES,
                    )
                )

            def read_text(self, *, encoding):
                self.assert_encoding = encoding
                return self.source

            def write_text(self, source, *, encoding):
                self.source = source
                self.assert_encoding = encoding

        setup = SetupFile()
        self.assertTrue(sparse_install._patch_sparge_setup(setup))
        self.assertNotIn(sparse_install._SPARGE_NINJA_DISABLED, setup.source)
        self.assertNotIn(sparse_install._SPARGE_NVCC_ALL_CORES, setup.source)
        self.assertIn(sparse_install._SPARGE_NINJA_ENABLED, setup.source)
        self.assertIn(sparse_install._SPARGE_NVCC_CONFIGURED, setup.source)
        self.assertEqual(setup.assert_encoding, 'utf-8')
        compile(setup.source, 'setup.py', 'exec')

    def test_pinned_source_patch_rejects_unreviewed_setup(self):
        class SetupFile:
            def __init__(self):
                self.written = False

            def read_text(self, *, encoding):
                return 'changed upstream setup'

            def write_text(self, source, *, encoding):
                self.written = True

        setup = SetupFile()
        with self.assertLogs(level='ERROR'):
            self.assertFalse(sparse_install._patch_sparge_setup(setup))
        self.assertFalse(setup.written)

    @patch.object(sparse_install.shutil, 'which', return_value='/usr/bin/git')
    @patch.object(sparse_install, '_run_command')
    def test_source_checkout_verifies_exact_pinned_commit(self, run, _which):
        run.side_effect = (
            subprocess.CompletedProcess([], 0, stdout=''),
            subprocess.CompletedProcess([], 0, stdout=''),
            subprocess.CompletedProcess([], 0, stdout=sparse_install.SPARGE_SOURCE_REF + '\n'),
        )
        destination = Path('build') / 'SpargeAttn'
        self.assertTrue(sparse_install._checkout_pinned_sparge(destination))
        self.assertEqual(run.call_count, 3)
        self.assertEqual(
            run.call_args_list[0].args[0],
            [
                '/usr/bin/git',
                'clone',
                '--quiet',
                '--no-checkout',
                sparse_install.SPARGE_REPOSITORY,
                str(destination),
            ],
        )
        self.assertIn(sparse_install.SPARGE_SOURCE_REF, run.call_args_list[1].args[0])

    def test_linux_build_defaults_to_half_the_logical_cores(self):
        with patch.object(os, 'cpu_count', return_value=16), patch.dict(
            os.environ,
            {'NVCC_APPEND_FLAGS': '--keep-this-flag'},
            clear=True,
        ):
            environment = sparse_install._linux_build_environment(
                '/usr/local/cuda/bin/nvcc'
            )
        self.assertEqual(environment['MAX_JOBS'], '8')
        self.assertEqual(environment['NVCC_THREADS'], '2')
        self.assertEqual(environment['NVCC_APPEND_FLAGS'], '--keep-this-flag -DNDEBUG')

    def test_linux_build_respects_explicit_parallelism(self):
        with patch.object(os, 'cpu_count', return_value=16), patch.dict(
            os.environ,
            {'MAX_JOBS': '3', 'NVCC_THREADS': '1'},
            clear=True,
        ):
            environment = sparse_install._linux_build_environment(
                '/usr/local/cuda/bin/nvcc'
            )
        self.assertEqual(environment['MAX_JOBS'], '3')
        self.assertEqual(environment['NVCC_THREADS'], '1')

    @patch.object(sparse_install.subprocess, 'run')
    def test_runtime_probe_does_not_import_torch_in_parent(self, run):
        run.return_value = subprocess.CompletedProcess(
            [],
            0,
            stdout=(
                sparse_install.PROBE_RESULT_PREFIX
                + '{"ok": true, "backend": "nvidia_cuda"}'
            ),
        )
        with patch.dict('sys.modules', {'torch': None}):
            result = sparse_install._runtime_probe(validate_sparse=False)
        self.assertTrue(result['ok'])
        self.assertEqual(result['backend'], 'nvidia_cuda')
        self.assertNotIn('--validate-sparse', run.call_args.args[0])

    @patch.object(sparse_install, '_is_installed', return_value=True)
    @patch.object(
        sparse_install,
        '_runtime_probe',
        return_value=nvidia_runtime(
            sparse_compatible=True,
            sparse_version='0.1.0',
            sparse_architecture='sm120',
        ),
    )
    @patch.object(sparse_install, '_run_pip')
    def test_compatible_existing_install_is_untouched(
        self,
        run_pip,
        probe,
        _installed,
    ):
        self.assertTrue(sparse_install.ensure_sparse_sage())
        probe.assert_called_once_with(validate_sparse=True)
        run_pip.assert_not_called()

    @patch.object(sparse_install, '_is_installed', return_value=False)
    @patch.object(sparse_install.subprocess, 'run')
    def test_skip_environment_disables_install(self, run, _installed):
        with patch.dict(
            os.environ,
            {sparse_install.SKIP_ENV: '1'},
            clear=False,
        ):
            self.assertFalse(sparse_install.ensure_sparse_sage())
        run.assert_not_called()

    @patch.object(sparse_install, '_is_installed', side_effect=[False, True])
    @patch.object(sparse_install.platform, 'system', return_value='Windows')
    @patch.object(sparse_install.platform, 'machine', return_value='AMD64')
    @patch.object(
        sparse_install,
        '_runtime_probe',
        side_effect=[
            nvidia_runtime(),
            nvidia_runtime(
                sparse_compatible=True,
                sparse_version='0.1.0',
                sparse_architecture='sm120',
            ),
        ],
    )
    @patch.object(sparse_install, '_run_pip', return_value=True)
    def test_installer_uses_pinned_wheel_without_dependencies(
        self,
        run_pip,
        _probe,
        _machine,
        _system,
        _installed,
    ):
        self.assertTrue(sparse_install.ensure_sparse_sage())
        arguments = run_pip.call_args.args[0]
        self.assertIn('--no-deps', arguments)
        self.assertIn('--only-binary=:all:', arguments)
        self.assertNotIn('--force-reinstall', arguments)
        self.assertTrue(arguments[-1].startswith('https://github.com/woct0rdho/'))
        self.assertIn('#sha256=', arguments[-1])

    @patch.object(sparse_install, '_is_installed', side_effect=[True, True])
    @patch.object(sparse_install.platform, 'system', return_value='Windows')
    @patch.object(sparse_install.platform, 'machine', return_value='AMD64')
    @patch.object(
        sparse_install,
        '_runtime_probe',
        side_effect=[
            nvidia_runtime(sparse_error='compiled extension ABI mismatch'),
            nvidia_runtime(
                sparse_compatible=True,
                sparse_version='0.1.0',
                sparse_architecture='sm120',
            ),
        ],
    )
    @patch.object(sparse_install, '_run_pip', return_value=True)
    def test_stale_install_is_replaced_only_with_verified_match(
        self,
        run_pip,
        _probe,
        _machine,
        _system,
        _installed,
    ):
        with self.assertLogs(level='WARNING'):
            self.assertTrue(sparse_install.ensure_sparse_sage())
        self.assertIn('--force-reinstall', run_pip.call_args.args[0])
        self.assertIn('--no-deps', run_pip.call_args.args[0])

    @patch.object(sparse_install, '_is_installed', return_value=False)
    @patch.object(
        sparse_install,
        '_runtime_probe',
        return_value={
            'ok': True,
            'backend': 'rocm',
            'accelerator_available': True,
            'capability': None,
            'sparse_compatible': False,
            'sparse_error': 'Sparse Sage requires NVIDIA CUDA, not ROCm',
        },
    )
    @patch.object(sparse_install, '_run_pip')
    def test_rocm_is_left_on_dense_without_install(
        self,
        run_pip,
        _probe,
        _installed,
    ):
        with self.assertLogs(level='INFO'):
            self.assertFalse(sparse_install.ensure_sparse_sage())
        run_pip.assert_not_called()

    @patch.object(sparse_install.shutil, 'which', return_value='/usr/bin/git')
    @patch.object(sparse_install, '_linux_nvcc', return_value='/usr/local/cuda/bin/nvcc')
    @patch.object(sparse_install, '_nvcc_version', return_value=(12, 8))
    @patch.object(sparse_install, '_missing_linux_build_requirements', return_value=[])
    @patch.object(sparse_install, '_run_pip', return_value=True)
    def test_linux_build_is_pinned_and_does_not_resolve_dependencies(
        self,
        run_pip,
        _missing,
        _nvcc_version,
        _nvcc,
        _which,
    ):
        prepared_source = Path('verified') / 'SpargeAttn'
        with patch.dict(
            os.environ,
            {'NVCC_APPEND_FLAGS': '--keep-this-flag'},
            clear=False,
        ), patch.object(
            sparse_install,
            '_prepared_linux_source',
            return_value=nullcontext(prepared_source),
        ) as prepared, patch.object(os, 'cpu_count', return_value=16):
            self.assertTrue(
                sparse_install._install_linux_source(
                    sparse_install.SPARGE_SOURCE
                )
            )
        arguments = run_pip.call_args.args[0]
        self.assertIn('--no-build-isolation', arguments)
        self.assertIn('--no-deps', arguments)
        self.assertEqual(arguments[-1], str(prepared_source))
        prepared.assert_called_once_with(sparse_install.SPARGE_SOURCE)
        self.assertEqual(
            run_pip.call_args.kwargs['environment']['CUDA_HOME'],
            str(Path('/usr/local/cuda/bin/nvcc').resolve().parent.parent),
        )
        self.assertEqual(
            run_pip.call_args.kwargs['environment']['NVCC_APPEND_FLAGS'],
            '--keep-this-flag -DNDEBUG',
        )
        self.assertEqual(run_pip.call_args.kwargs['environment']['MAX_JOBS'], '8')
        self.assertEqual(run_pip.call_args.kwargs['environment']['NVCC_THREADS'], '2')

    @patch.object(sparse_install.subprocess, 'run')
    def test_nvcc_version_is_validated_without_torch(self, run):
        run.return_value = subprocess.CompletedProcess(
            [],
            0,
            stdout='Cuda compilation tools, release 12.8, V12.8.93',
        )
        with patch.dict('sys.modules', {'torch': None}):
            self.assertEqual(
                sparse_install._nvcc_version('/usr/local/cuda/bin/nvcc'),
                (12, 8),
            )

    @patch.object(sparse_install.shutil, 'which', return_value='/usr/bin/git')
    @patch.object(sparse_install, '_linux_nvcc', return_value=None)
    @patch.object(sparse_install, '_run_pip')
    def test_linux_missing_nvcc_does_not_run_pip(
        self,
        run_pip,
        _nvcc,
        _which,
    ):
        with self.assertLogs(level='WARNING'):
            self.assertFalse(
                sparse_install._install_linux_source(
                    sparse_install.SPARGE_SOURCE
                )
            )
        run_pip.assert_not_called()

    @patch.object(sparse_install.subprocess, 'run')
    def test_failed_pip_command_returns_false(self, run):
        run.return_value = subprocess.CompletedProcess(
            [],
            1,
            stdout='build failed',
        )
        with self.assertLogs(level='ERROR'):
            self.assertFalse(
                sparse_install._run_pip(
                    ['--no-deps', sparse_install.SPARGE_SOURCE],
                    timeout=1,
                )
            )

    def test_prestartup_script_respects_skip_environment(self):
        # Assert the behaviour rather than a leaked module global: prestartup
        # must reach the installer and must not raise when the skip flag is
        # set. It also prepares the native backend now, which must not be able
        # to stop startup either.
        with patch.dict(
            os.environ,
            {sparse_install.SKIP_ENV: '1'},
            clear=False,
        ), patch.object(sparse_install, '_is_installed', return_value=False):
            result = runpy.run_path(str(PACK / 'prestartup_script.py'))
        self.assertIn('_prepare_sparse_sage', result)
        self.assertIn('_prepare_native_backend', result)


if __name__ == '__main__':
    unittest.main()
