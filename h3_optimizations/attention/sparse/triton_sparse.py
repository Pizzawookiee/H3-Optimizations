'''Stable public surface for the INT8 Triton sparse backend.

The production backend now consumes the exact Kitchen carrier and mirrors the
Kitchen pure-INT8 probability/value math at 64Q x 64KV. The old Triton carrier
and executor remain importable for focused legacy tests and benchmarks, but are
no longer selected by the normal resolver.
'''

from .triton_sparse_fast import (  # legacy low-level compatibility
    PreparedTritonSparse,
    PreparedTritonHybrid,
    TritonSparseExecutor,
)
from .triton_kitchen import (
    PreparedTritonKitchen,
    TritonKitchenBackend,
    TritonKitchenError,
    TritonKitchenSpec,
    preflight_triton_kitchen,
)

# Preserve the public names used by apply.py and saved integrations. Only the
# implementation behind them changes.
TritonSparseBackend = TritonKitchenBackend
TritonSparseError = TritonKitchenError
TritonSparseSpec = TritonKitchenSpec
preflight_triton_sparse = preflight_triton_kitchen

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
