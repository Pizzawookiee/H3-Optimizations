'''Install a verified Sparse Sage wheel before ComfyUI loads this pack.'''

import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


LOG_PREFIX = '[H3 Optimizations]'
SKIP_ENV = 'H3_OPTIMIZATIONS_SKIP_SPARSE_INSTALL'
INSTALL_TIMEOUT_SECONDS = 600
BUILD_TIMEOUT_SECONDS = 1800
PROBE_TIMEOUT_SECONDS = 120
PROBE_RESULT_PREFIX = 'H3_SPARSE_PROBE='
RELEASE_ROOT = (
    'https://github.com/woct0rdho/SpargeAttn/releases/download'
)
SPARGE_SOURCE_REF = '067d80cb6b76345c7b8be40e86c7d19a3cf7c4eb'
SPARGE_SOURCE = (
    'git+https://github.com/woct0rdho/SpargeAttn.git@' + SPARGE_SOURCE_REF
)
_LINUX_BUILD_REQUIREMENTS = {
    'ninja': 'ninja',
    'packaging': 'packaging',
    'setuptools': 'setuptools',
    'wheel': 'wheel',
}
_SUPPORTED_CAPABILITIES = {
    (8, 0),
    (8, 6),
    (8, 7),
    (8, 9),
    (9, 0),
    (12, 0),
}

_EXACT_WINDOWS_WHEELS = {
    ((2, 5, 1), (12, 4)): (
        'v0.1.0-windows.post3',
        'spas_sage_attn-0.1.0%2Bcu124torch2.5.1.post3-cp39-abi3-win_amd64.whl',
        'e2bfa39fd31443124839c0273ad06845393daf9d3450449748b4bab899bf029e',
    ),
    ((2, 6, 0), (12, 6)): (
        'v0.1.0-windows.post3',
        'spas_sage_attn-0.1.0%2Bcu126torch2.6.0.post3-cp39-abi3-win_amd64.whl',
        'd847d09ea0b053bd3c2581b83c48113f235269363918bc8c3e76c5ff0f2a6d87',
    ),
    ((2, 7, 1), (12, 8)): (
        'v0.1.0-windows.post3',
        'spas_sage_attn-0.1.0%2Bcu128torch2.7.1.post3-cp39-abi3-win_amd64.whl',
        '425afea0544f948f5236199f1e23f13e821a5d13cf761035ab9723b5de25e802',
    ),
    ((2, 8, 0), (12, 8)): (
        'v0.1.0-windows.post3',
        'spas_sage_attn-0.1.0%2Bcu128torch2.8.0.post3-cp39-abi3-win_amd64.whl',
        '6e28ecf05956fcfdcfa87af064e5ba282b8c721246702f78b36a7ba0aa2b76c3',
    ),
}

_STABLE_ABI_WINDOWS_WHEELS = {
    (12, 8): (
        'v0.1.0-windows.post4',
        'spas_sage_attn-0.1.0%2Bcu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl',
        '258eadcd4433892ac445ab7e7a7f07de1ede3897c71334b4ec08b26da894bca3',
    ),
    (13, 0): (
        'v0.1.0-windows.post4',
        'spas_sage_attn-0.1.0%2Bcu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl',
        '2b34a5abe45c0ea51872f78dc1d0b407893ac5ecab3bf71425a73e769e47f27a',
    ),
}


def _version_tuple(value):
    match = re.match(r'^(\d+)\.(\d+)(?:\.(\d+))?', str(value))
    if match is None:
        return None
    return tuple(int(part or 0) for part in match.groups())


def _wheel_for(system, machine, torch_version, cuda_version):
    if system != 'Windows' or machine.lower() not in ('amd64', 'x86_64'):
        return None
    torch_key = _version_tuple(torch_version)
    cuda_key = _version_tuple(cuda_version)
    if torch_key is None or cuda_key is None:
        return None
    cuda_key = cuda_key[:2]
    if torch_key >= (2, 9, 0):
        wheel = _STABLE_ABI_WINDOWS_WHEELS.get(cuda_key)
    else:
        wheel = _EXACT_WINDOWS_WHEELS.get((torch_key, cuda_key))
    if wheel is None:
        return None
    release, filename, digest = wheel
    return '%s/%s/%s#sha256=%s' % (
        RELEASE_ROOT,
        release,
        filename,
        digest,
    )


def _linux_source_for(system, machine, torch_version, cuda_version):
    if system != 'Linux' or machine.lower() not in ('amd64', 'x86_64'):
        return None
    torch_key = _version_tuple(torch_version)
    cuda_key = _version_tuple(cuda_version)
    if torch_key is None:
        return None
    if torch_key < (2, 3, 0):
        return None
    if cuda_key is not None and cuda_key[:2] < (12, 0):
        return None
    return SPARGE_SOURCE


def _is_installed():
    try:
        return importlib.util.find_spec('spas_sage_attn') is not None
    except (ImportError, ValueError):
        return False


def _skip_requested():
    return os.environ.get(SKIP_ENV, '').strip().lower() in (
        '1',
        'true',
        'yes',
        'on',
    )


def _runtime_probe(*, validate_sparse):
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name('sparse_probe.py')),
    ]
    if validate_sparse:
        command.append('--validate-sparse')
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logging.warning('%s Sparse Sage compatibility probe failed: %s', LOG_PREFIX, exc)
        return None
    for line in reversed((result.stdout or '').splitlines()):
        if not line.startswith(PROBE_RESULT_PREFIX):
            continue
        try:
            return json.loads(line[len(PROBE_RESULT_PREFIX):])
        except (TypeError, ValueError):
            break
    detail = (result.stdout or '').strip().splitlines()
    logging.warning(
        '%s Sparse Sage compatibility probe failed: %s',
        LOG_PREFIX,
        detail[-1] if detail else 'no result',
    )
    return None


def _run_pip(arguments, *, timeout, environment=None):
    command = [
        sys.executable,
        '-m',
        'pip',
        'install',
        '--no-input',
        '--disable-pip-version-check',
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logging.error('%s Sparse Sage installation failed: %s', LOG_PREFIX, exc)
        return False
    if result.returncode == 0:
        return True
    output = (result.stdout or '').strip().splitlines()
    detail = output[-1] if output else 'pip exited with code %d' % result.returncode
    logging.error('%s Sparse Sage installation failed: %s', LOG_PREFIX, detail)
    return False


def _install_windows_wheel(wheel, *, force_reinstall=False):
    arguments = [
        '--no-deps',
        '--only-binary=:all:',
    ]
    if force_reinstall:
        arguments.append('--force-reinstall')
    arguments.append(wheel)
    return _run_pip(
        arguments,
        timeout=INSTALL_TIMEOUT_SECONDS,
    )


def _missing_linux_build_requirements():
    missing = []
    for module, package in _LINUX_BUILD_REQUIREMENTS.items():
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            missing.append(package)
    return missing


def _linux_nvcc():
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home:
        candidate = Path(cuda_home) / 'bin' / 'nvcc'
        if candidate.is_file():
            return str(candidate)
    return shutil.which('nvcc')


def _nvcc_version(nvcc):
    try:
        result = subprocess.run(
            [nvcc, '--version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r'release\s+(\d+)\.(\d+)', result.stdout or '')
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _install_linux_source(source, *, force_reinstall=False):
    if shutil.which('git') is None:
        logging.warning('%s Linux Sparse Sage installation requires git', LOG_PREFIX)
        return False
    nvcc = _linux_nvcc()
    if nvcc is None:
        logging.warning(
            '%s Linux Sparse Sage installation requires the CUDA nvcc compiler',
            LOG_PREFIX,
        )
        return False
    nvcc_version = _nvcc_version(nvcc)
    if nvcc_version is None or nvcc_version < (12, 0):
        logging.warning('%s Linux Sparse Sage installation requires nvcc 12.0 or newer', LOG_PREFIX)
        return False
    missing = _missing_linux_build_requirements()
    if missing and not _run_pip(
        ['--no-deps', *missing],
        timeout=INSTALL_TIMEOUT_SECONDS,
    ):
        return False
    environment = os.environ.copy()
    environment['CUDA_HOME'] = str(Path(nvcc).resolve().parent.parent)
    # Avoid cuda_fp*.hpp's unresolved __assert_fail path in Linux nvcc builds.
    nvcc_flags = environment.get('NVCC_APPEND_FLAGS', '').split()
    if '-DNDEBUG' not in nvcc_flags:
        nvcc_flags.append('-DNDEBUG')
    environment['NVCC_APPEND_FLAGS'] = ' '.join(nvcc_flags)
    arguments = [
        '--no-build-isolation',
        '--no-deps',
    ]
    if force_reinstall:
        arguments.append('--force-reinstall')
    arguments.append(source)
    return _run_pip(
        arguments,
        timeout=BUILD_TIMEOUT_SECONDS,
        environment=environment,
    )


def ensure_sparse_sage():
    if _skip_requested():
        logging.info('%s automatic Sparse Sage installation is disabled', LOG_PREFIX)
        return False

    package_present = _is_installed()
    runtime = _runtime_probe(validate_sparse=package_present)
    if runtime is None or not runtime.get('ok'):
        detail = 'compatibility probe failed'
        if runtime is not None:
            detail = runtime.get('error') or detail
        logging.warning('%s Sparse Sage was not installed because %s', LOG_PREFIX, detail)
        return False
    if package_present and runtime.get('sparse_compatible'):
        logging.info(
            '%s Sparse Sage %s validated for %s',
            LOG_PREFIX,
            runtime.get('sparse_version') or 'installation',
            runtime.get('sparse_architecture') or runtime.get('device_name'),
        )
        return True

    backend = runtime.get('backend')
    capability = tuple(runtime.get('capability') or ())
    if (
        backend != 'nvidia_cuda'
        or not runtime.get('accelerator_available')
        or capability not in _SUPPORTED_CAPABILITIES
    ):
        logging.info(
            '%s Sparse Sage acceleration is unavailable: %s; dense H3 remains available',
            LOG_PREFIX,
            runtime.get('sparse_error') or 'unsupported runtime',
        )
        return False

    torch_version = runtime.get('torch_version')
    cuda_version = runtime.get('cuda_version')
    system = platform.system()
    machine = platform.machine()
    wheel = _wheel_for(
        system,
        machine,
        torch_version,
        cuda_version or '',
    )
    source = _linux_source_for(
        system,
        machine,
        torch_version,
        cuda_version or '',
    )
    if wheel is None and source is None:
        logging.warning(
            '%s Sparse Sage is not installed and no verified automatic install '
            'matches %s %s, Torch %s, CUDA %s',
            LOG_PREFIX,
            system,
            machine,
            torch_version,
            cuda_version or 'unknown',
        )
        return False

    force_reinstall = package_present
    if force_reinstall:
        logging.warning(
            '%s existing Sparse Sage is incompatible (%s); installing a verified replacement',
            LOG_PREFIX,
            runtime.get('sparse_error') or 'ABI validation failed',
        )
    if wheel is not None:
        logging.info(
            '%s installing Sparse Sage wheel for Torch %s, CUDA %s',
            LOG_PREFIX,
            torch_version,
            cuda_version,
        )
        installed = _install_windows_wheel(
            wheel,
            force_reinstall=force_reinstall,
        )
    else:
        logging.info(
            '%s building pinned Sparse Sage %s for Torch %s, CUDA %s',
            LOG_PREFIX,
            SPARGE_SOURCE_REF,
            torch_version,
            cuda_version or 'detected by nvcc',
        )
        installed = _install_linux_source(
            source,
            force_reinstall=force_reinstall,
        )
    if not installed:
        return False

    importlib.invalidate_caches()
    if not _is_installed():
        logging.error('%s pip completed but spas_sage_attn is still unavailable', LOG_PREFIX)
        return False
    runtime = _runtime_probe(validate_sparse=True)
    if runtime is None or not runtime.get('sparse_compatible'):
        logging.error(
            '%s Sparse Sage installed but failed compatibility validation: %s',
            LOG_PREFIX,
            'no probe result'
            if runtime is None
            else runtime.get('sparse_error') or runtime.get('error') or 'unknown error',
        )
        return False
    logging.info('%s Sparse Sage installed successfully', LOG_PREFIX)
    return True
