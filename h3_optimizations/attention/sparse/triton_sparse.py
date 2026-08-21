'''Stable public surface for the optimized INT8 Triton sparse backend.'''

from . import triton_sparse_fast as _impl
from .triton_sparse_kernels import launch_int8_sparse as _fast_launch

# Executor.execute resolves this module global at call time. Keep the public
# backend API stable while isolating the aggressively tuned kernel surface.
_impl._launch_int8_sparse = _fast_launch

from .triton_sparse_fast import *  # noqa: E402,F401,F403
