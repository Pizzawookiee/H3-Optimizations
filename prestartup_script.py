'''Verify the shipped native backend before node registration.

The binary self-test may not stop ComfyUI from starting. A missing or invalid
accelerator means the normal fallback chain is used instead.
'''

import logging
from pathlib import Path
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


_prepare_native_backend()
