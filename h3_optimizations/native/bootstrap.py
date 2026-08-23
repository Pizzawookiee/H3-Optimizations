"""Make the native backend present, or say clearly why it is not.

Runs from prestartup_script.py before node registration, which is the same
slot sparse_install.py used to occupy for Sparge. It is much smaller than that
was, because the ctypes boundary removed everything that made Sparge's
installer hard: there is no torch version, CUDA wheel suffix, Python version
or foreign package ABI to match. Just OS and machine, one file.

    present and valid?  -> use it
    absent?             -> fetch the pinned release asset
                           -> SHA-256
                           -> load
                           -> ABI check
                           -> runtime self-test
    anything failed?    -> warn loudly, leave the backend unavailable

Nothing here raises into ComfyUI's startup. A missing accelerator is not a
reason to stop a server from booting; it is a reason to say so in a way nobody
can miss, which is what the earlier silent fallback failed to do.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import shutil
import tempfile
import urllib.error
import urllib.request

from . import artifacts, loader

LOG_PREFIX = '[H3 Optimizations]'

_PACK_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_BIN_DIR = _PACK_ROOT / 'native' / 'bin'
_MARKER = _BIN_DIR / 'BUILD_ID'
_DOWNLOAD_TIMEOUT = 120


def installed_build_id():
    """Which release the installed binary came from, or None for a local build."""
    try:
        return _MARKER.read_text(encoding='utf-8').strip() or None
    except OSError:
        return None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _download(artifact, destination):
    """Fetch to a temporary file, verify, then move into place.

    Verifying before the move means an interrupted or corrupted download can
    never be mistaken for a usable binary on the next start.
    """
    _BIN_DIR.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(_BIN_DIR), suffix='.part')
    os.close(handle)
    temporary = pathlib.Path(temporary)
    try:
        logging.info(
            '%s fetching native backend %s for %s',
            LOG_PREFIX, artifacts.NATIVE_BUILD, artifacts.describe_platform(),
        )
        with urllib.request.urlopen(artifact.url, timeout=_DOWNLOAD_TIMEOUT) as response:
            with temporary.open('wb') as output:
                shutil.copyfileobj(response, output)

        actual = _sha256(temporary)
        if actual != artifact.sha256:
            raise RuntimeError(
                'checksum mismatch for %s: expected %s, got %s'
                % (artifact.filename, artifact.sha256, actual)
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_native_backend(*, allow_download=True):
    """Return True when the native backend is usable. Never raises."""
    from . import selftest

    try:
        if loader.is_available():
            return _verify(selftest)

        artifact = artifacts.artifact_for_this_platform()
        if artifact is None:
            logging.warning(
                '%s NATIVE BACKEND UNAVAILABLE - no prebuilt binary is '
                'published for %s. Sparse attention will fall back to a '
                'slower path. Build it locally with: cmake -S native -B '
                'native/build && cmake --build native/build --config Release',
                LOG_PREFIX, artifacts.describe_platform(),
            )
            return False

        if not allow_download:
            logging.warning(
                '%s NATIVE BACKEND UNAVAILABLE - not downloaded (downloads '
                'disabled). Sparse attention will fall back to a slower path.',
                LOG_PREFIX,
            )
            return False

        destination = _BIN_DIR / artifact.filename
        try:
            _download(artifact, destination)
        except (urllib.error.URLError, OSError, RuntimeError) as error:
            logging.warning(
                '%s NATIVE BACKEND UNAVAILABLE - could not fetch %s: %s. '
                'Sparse attention will fall back to a slower path.',
                LOG_PREFIX, artifact.filename, error,
            )
            return False

        _MARKER.write_text(artifacts.NATIVE_BUILD, encoding='utf-8')
        os.environ.setdefault('H3_INT8_ATTENTION_LIBRARY', str(destination))
        return _verify(selftest, force_reload=True)
    except Exception as error:  # noqa: BLE001 - startup must not die here
        logging.warning(
            '%s NATIVE BACKEND UNAVAILABLE - unexpected error during setup: '
            '%s: %s. Sparse attention will fall back to a slower path.',
            LOG_PREFIX, type(error).__name__, error,
        )
        return False


def _verify(selftest, *, force_reload=False):
    """Load, check the ABI, then prove the kernels on this actual GPU."""
    try:
        loader.load(force_reload=force_reload)
    except loader.NativeUnavailableError as error:
        logging.warning(
            '%s NATIVE BACKEND UNAVAILABLE - %s. Sparse attention will fall '
            'back to a slower path.',
            LOG_PREFIX, error,
        )
        return False

    if not selftest.check():
        # selftest.check already logged what failed and why.
        return False
    logging.info('%s native backend ready (%s)', LOG_PREFIX, describe())
    return True


def describe():
    """One line for a bug report."""
    try:
        library = loader.load()
        abi = library.h3_int8_abi_version()
        encoding = loader.route_encoding()
    except loader.NativeUnavailableError as error:
        return 'unavailable: %s' % str(error).splitlines()[0]
    return 'abi=%d build=%s platform=%s route=%s' % (
        abi,
        installed_build_id() or 'local',
        artifacts.describe_platform(),
        encoding,
    )


def diagnostics():
    """A block a bug report can paste, with no GPU work beyond the cached test."""
    from . import selftest

    lines = ['native backend:']
    try:
        library = loader.load()
    except loader.NativeUnavailableError as error:
        lines.append('  status   : unavailable')
        for line in str(error).splitlines():
            lines.append('  %s' % line)
        return '\n'.join(lines)

    lines.append('  ABI      : %d (expected %d)' % (
        library.h3_int8_abi_version(), artifacts.REQUIRED_ABI
    ))
    lines.append('  build    : %s' % (installed_build_id() or 'local'))
    lines.append('  platform : %s' % artifacts.describe_platform())
    lines.append('  route    : %s' % loader.route_encoding())
    lines.append('  self-test: %s' % ('passed' if selftest.check() else 'FAILED'))
    return '\n'.join(lines)
