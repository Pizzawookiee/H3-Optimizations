'''Install a verified Sparse Sage build before ComfyUI loads this pack.'''

from contextlib import contextmanager
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
from tempfile import TemporaryDirectory


LOG_PREFIX = '[H3 Optimizations]'
SKIP_ENV = 'H3_OPTIMIZATIONS_SKIP_SPARSE_INSTALL'
INSTALL_TIMEOUT_SECONDS = 600
BUILD_TIMEOUT_SECONDS = 1800
PROBE_TIMEOUT_SECONDS = 120
PROBE_RESULT_PREFIX = 'H3_SPARSE_PROBE='
WINDOWS_RELEASE_ROOT = (
    'https://github.com/woct0rdho/SpargeAttn/releases/download'
)
UPSTREAM_SPARGE_SOURCE_REF = 'ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a'
UPSTREAM_SPARGE_REPOSITORY = 'https://github.com/thu-ml/SpargeAttn.git'
UPSTREAM_SPARGE_SOURCE = (
    'git+' + UPSTREAM_SPARGE_REPOSITORY + '@' + UPSTREAM_SPARGE_SOURCE_REF
)
FORK_SPARGE_SOURCE_REF = '067d80cb6b76345c7b8be40e86c7d19a3cf7c4eb'
FORK_SPARGE_REPOSITORY = 'https://github.com/woct0rdho/SpargeAttn.git'
FORK_SPARGE_SOURCE = (
    'git+' + FORK_SPARGE_REPOSITORY + '@' + FORK_SPARGE_SOURCE_REF
)
_SPARGE_NINJA_DISABLED = 'BuildExtension.with_options(use_ninja=False)'
_SPARGE_NINJA_ENABLED = 'BuildExtension.with_options(use_ninja=True)'
_SPARGE_NVCC_ALL_CORES = 'f"--threads={os.cpu_count()}"'
_SPARGE_NVCC_CONFIGURED = 'f"--threads={os.environ.get(\'NVCC_THREADS\', \'2\')}"'
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
_UPSTREAM_LINUX_CAPABILITIES = {
    (8, 0),
    (8, 6),
    (8, 7),
    (8, 9),
    (9, 0),
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
        WINDOWS_RELEASE_ROOT,
        release,
        filename,
        digest,
    )


def _linux_source_for(system, machine, torch_version, cuda_version, capability):
    if system != 'Linux' or machine.lower() not in ('amd64', 'x86_64'):
        return None
    torch_key = _version_tuple(torch_version)
    cuda_key = _version_tuple(cuda_version)
    capability = tuple(capability or ())
    if torch_key is None or torch_key < (2, 3, 0):
        return None
    if cuda_key is not None and cuda_key[:2] < (12, 0):
        return None
    if capability == (8, 9) and cuda_key is not None and cuda_key[:2] < (12, 4):
        return None
    if capability == (9, 0) and cuda_key is not None and cuda_key[:2] < (12, 4):
        return None
    if capability in _UPSTREAM_LINUX_CAPABILITIES:
        return UPSTREAM_SPARGE_SOURCE
    if capability == (12, 0) and cuda_key is not None and cuda_key[:2] >= (12, 8):
        return FORK_SPARGE_SOURCE
    return None


def _is_installed():
    try:
        return importlib.util.find_spec('spas_sage_attn') is not None
    except (ImportError, ValueError):
        return False


def _skip_requested():
    return os.environ.get(SKIP_ENV, '').strip().lower() in (
        '1', 'true', 'yes', 'on',
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


def _run_command(command, *, timeout, operation):
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logging.error('%s %s failed: %s', LOG_PREFIX, operation, exc)
        return None
    if result.returncode == 0:
        return result
    output = (result.stdout or '').strip().splitlines()
    detail = output[-1] if output else 'command exited with code %d' % result.returncode
    logging.error('%s %s failed: %s', LOG_PREFIX, operation, detail)
    return None


def _install_windows_wheel(wheel, *, force_reinstall=False):
    arguments = ['--no-deps', '--only-binary=:all:']
    if force_reinstall:
        arguments.append('--force-reinstall')
    arguments.append(wheel)
    return _run_pip(arguments, timeout=INSTALL_TIMEOUT_SECONDS)


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


def _source_details(source):
    if source == UPSTREAM_SPARGE_SOURCE:
        return UPSTREAM_SPARGE_REPOSITORY, UPSTREAM_SPARGE_SOURCE_REF, False
    if source == FORK_SPARGE_SOURCE:
        return FORK_SPARGE_REPOSITORY, FORK_SPARGE_SOURCE_REF, True
    return None


def _checkout_pinned_sparge(destination, source):
    details = _source_details(source)
    git = shutil.which('git')
    if details is None or git is None:
        return False
    repository, source_ref, _ = details
    if _run_command(
        [git, 'clone', '--quiet', '--no-checkout', repository, str(destination)],
        timeout=INSTALL_TIMEOUT_SECONDS,
        operation='Sparse Sage source checkout',
    ) is None:
        return False
    if _run_command(
        [git, '-C', str(destination), 'checkout', '--quiet', '--detach', source_ref],
        timeout=INSTALL_TIMEOUT_SECONDS,
        operation='Sparse Sage source checkout',
    ) is None:
        return False
    result = _run_command(
        [git, '-C', str(destination), 'rev-parse', 'HEAD'],
        timeout=30,
        operation='Sparse Sage source verification',
    )
    if result is None:
        return False
    if (result.stdout or '').strip().lower() != source_ref:
        logging.error('%s Sparse Sage source checkout did not resolve the pinned commit', LOG_PREFIX)
        return False
    return True


def _patch_fork_setup(path):
    try:
        source = path.read_text(encoding='utf-8')
    except OSError as exc:
        logging.error('%s Sparse Sage build setup could not be read: %s', LOG_PREFIX, exc)
        return False
    replacements = (
        (_SPARGE_NINJA_DISABLED, _SPARGE_NINJA_ENABLED),
        (_SPARGE_NVCC_ALL_CORES, _SPARGE_NVCC_CONFIGURED),
    )
    for original, replacement in replacements:
        if source.count(original) != 1:
            logging.error('%s pinned Sparse Sage fork build setup no longer matches', LOG_PREFIX)
            return False
        source = source.replace(original, replacement)
    try:
        path.write_text(source, encoding='utf-8')
    except OSError as exc:
        logging.error('%s Sparse Sage build setup could not be updated: %s', LOG_PREFIX, exc)
        return False
    return True


@contextmanager
def _prepared_linux_source(source):
    details = _source_details(source)
    if details is None:
        logging.error('%s refused an unpinned Sparse Sage source', LOG_PREFIX)
        yield None
        return
    _, _, needs_fork_patch = details
    with TemporaryDirectory(prefix='h3-sparge-build-') as temporary:
        checkout = Path(temporary) / 'SpargeAttn'
        if not _checkout_pinned_sparge(checkout, source):
            yield None
            return
        if needs_fork_patch and not _patch_fork_setup(checkout / 'setup.py'):
            yield None
            return
        yield checkout


def _linux_build_environment(nvcc, capability, source):
    environment = os.environ.copy()
    environment['CUDA_HOME'] = str(Path(nvcc).resolve().parent.parent)
    logical_cores = max(1, os.cpu_count() or 1)
    environment.setdefault('MAX_JOBS', str(max(1, logical_cores // 2)))
    capability = tuple(capability or ())
    if capability:
        environment.setdefault('TORCH_CUDA_ARCH_LIST', '%d.%d' % capability)
    if source == FORK_SPARGE_SOURCE:
        environment.setdefault('NVCC_THREADS', str(min(2, logical_cores)))
        nvcc_flags = environment.get('NVCC_APPEND_FLAGS', '').split()
        if '-DNDEBUG' not in nvcc_flags:
            nvcc_flags.append('-DNDEBUG')
        environment['NVCC_APPEND_FLAGS'] = ' '.join(nvcc_flags)
    return environment


def _minimum_nvcc_for(capability):
    capability = tuple(capability or ())
    if capability in ((8, 9), (9, 0)):
        return (12, 4)
    if capability == (12, 0):
        return (12, 8)
    return (12, 0)


def _install_linux_source(source, *, capability, force_reinstall=False):
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
    minimum_nvcc = _minimum_nvcc_for(capability)
    if nvcc_version is None or nvcc_version < minimum_nvcc:
        logging.warning(
            '%s Linux Sparse Sage installation for SM%d%d requires nvcc %d.%d or newer',
            LOG_PREFIX,
            capability[0], capability[1], minimum_nvcc[0], minimum_nvcc[1],
        )
        return False
    missing = _missing_linux_build_requirements()
    if missing and not _run_pip(
        ['--no-deps', *missing],
        timeout=INSTALL_TIMEOUT_SECONDS,
    ):
        return False
    environment = _linux_build_environment(nvcc, capability, source)
    logging.info(
        '%s building Sparse Sage for SM%d%d with %s parallel jobs',
        LOG_PREFIX,
        capability[0], capability[1], environment['MAX_JOBS'],
    )
    arguments = ['--no-build-isolation', '--no-deps']
    if force_reinstall:
        arguments.append('--force-reinstall')
    with _prepared_linux_source(source) as prepared_source:
        if prepared_source is None:
            return False
        arguments.append(str(prepared_source))
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
    wheel = _wheel_for(system, machine, torch_version, cuda_version or '')
    source = _linux_source_for(
        system,
        machine,
        torch_version,
        cuda_version or '',
        capability,
    )
    if wheel is None and source is None:
        logging.warning(
            '%s Sparse Sage is not installed and no verified automatic install '
            'matches %s %s, Torch %s, CUDA %s, SM%s',
            LOG_PREFIX,
            system,
            machine,
            torch_version,
            cuda_version or 'unknown',
            ''.join(str(part) for part in capability) or 'unknown',
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
        details = _source_details(source)
        source_ref = details[1]
        source_name = 'upstream' if source == UPSTREAM_SPARGE_SOURCE else 'SM120 fork'
        logging.info(
            '%s building pinned %s Sparse Sage %s for Torch %s, CUDA %s',
            LOG_PREFIX,
            source_name,
            source_ref,
            torch_version,
            cuda_version or 'detected by nvcc',
        )
        installed = _install_linux_source(
            source,
            capability=capability,
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