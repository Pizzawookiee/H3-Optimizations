'''Stable public surface for the INT8 Triton sparse backend.

The production backend consumes the exact Kitchen carrier and mirrors the
Kitchen pure-INT8 probability/value math at 64Q x 64KV. The old carrier,
executor, and spec remain importable so existing tests/benchmarks and saved
integrations keep their low-level ABI.
'''

import torch

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
from .triton_kitchen_sm120 import SM120TritonKitchenBackend


TritonSparseError = TritonKitchenError


def TritonSparseBackend(config=None, **kwargs):
    '''Construct the Kitchen-parity backend appropriate for this CUDA target.'''
    capability = tuple(int(value) for value in torch.cuda.get_device_capability())
    backend = (
        SM120TritonKitchenBackend
        if capability == (12, 0)
        else TritonKitchenBackend
    )
    return backend(config, **kwargs)


def preflight_triton_sparse(**kwargs):
    '''Keep the historical spec ABI while translating preflight errors uniformly.'''
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
