'''Stable public surface for the optimized INT8 Triton sparse backend.'''

from . import triton_sparse_fast as _impl
from .triton_compact import (
    backend_prepare as _backend_prepare,
    backend_prepare_projected as _backend_prepare_projected,
    executor_prepare_compact as _executor_prepare_compact,
    executor_prepare_projected_compact as _executor_prepare_projected_compact,
)
from .triton_sparse_kernels import launch_int8_sparse as _fast_launch

# Keep the public backend surface stable while replacing the expensive pieces:
# the attention launcher and Sparge-format route preparation.
_impl._launch_int8_sparse = _fast_launch
_impl.TritonSparseExecutor.prepare_compact = _executor_prepare_compact
_impl.TritonSparseExecutor.prepare_projected_compact = (
    _executor_prepare_projected_compact
)
_impl.TritonSparseBackend.prepare = _backend_prepare
_impl.TritonSparseBackend.prepare_projected = _backend_prepare_projected

_original_as_status = _impl.TritonSparseBackend.as_status


def _optimized_as_status(self):
    status = _original_as_status(self)
    status['autotune_block_m'] = [16, 32, 64]
    status['autotune_warps'] = [4, 8]
    status['route_format'] = 'absolute_compact_int32_direct'
    return status


_impl.TritonSparseBackend.as_status = _optimized_as_status

from .triton_sparse_fast import *  # noqa: E402,F401,F403
