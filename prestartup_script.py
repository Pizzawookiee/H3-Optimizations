'''Prepare the optional accelerated backends before node registration.

Two independent things, in order of how much they matter:

  * the pack's own INT8 attention library, which is what makes the sparse
    kernel and the chunked QKV producer available at all;
  * the Sparse Sage backend, still the default until the native path has been
    measured against it on real work.

Neither may stop ComfyUI from starting. A missing accelerator means slower
generation, not a broken server -- so failures here warn loudly and leave the
backend unavailable rather than raising.
'''

from pathlib import Path
import logging
import runpy
import sys


_HERE = Path(__file__).resolve().parent


def _prepare_native_backend():
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    try:
        from h3_optimizations.native.bootstrap import ensure_native_backend
    except Exception as error:  # noqa: BLE001 - startup must survive this
        logging.warning(
            '[H3 Optimizations] could not prepare the native INT8 backend: '
            '%s: %s', type(error).__name__, error,
        )
        return
    ensure_native_backend()


def _prepare_sparse_sage():
    try:
        installer = runpy.run_path(
            str(_HERE / 'h3_optimizations' / 'sparse_install.py')
        )
        installer['ensure_sparse_sage']()
    except Exception as error:  # noqa: BLE001 - startup must survive this
        logging.warning(
            '[H3 Optimizations] could not prepare Sparse Sage: %s: %s',
            type(error).__name__, error,
        )


_prepare_native_backend()
_prepare_sparse_sage()
