'''Stable public surface for the INT8 Triton sparse backend.

The production backend consumes the exact Kitchen carrier and mirrors the
Kitchen pure-INT8 probability/value math at 64Q x 64KV. The old carrier,
executor, and spec remain importable so existing tests/benchmarks and saved
integrations keep their low-level ABI.
'''

from . import triton_sparse_fast as _legacy
from .triton_sparse_fast import (  # legacy low-level compatibility
    PreparedTritonSparse,
    PreparedTritonHybrid,
    TritonSparseExecutor,
    TritonSparseSpec,
)
from .triton_kitchen import (
    PreparedTritonKitchen,
    TritonKitchenBackend,
    TritonKitchenError,
)


TritonSparseBackend = TritonKitchenBackend
TritonSparseError = TritonKitchenError


def preflight_triton_sparse(**kwargs):
    '''Keep the historical spec ABI but translate preflight errors uniformly.'''
    try:
        return _legacy.preflight_triton_sparse(**kwargs)
    except _legacy.TritonSparseError as exc:
        raise TritonKitchenError(str(exc)) from exc


__all__ = [
    'PreparedTritonHybrid',
    'PreparedTritonKitchen',
    'PreparedTritonSparse',
    'TritonSparseBackend',
    'TritonSparseError',
    'TritonSparseExecutor',
    'TritonSparseSpec',
    'preflight_triton_sparse',
]
