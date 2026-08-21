'''Stable public surface for the optimized INT8 Triton sparse QKV carrier.'''

from . import triton_qkv_fast as _impl
from .triton_v_pack import pack_triton_v_chunk_into as _fast_v_pack

# The implementation resolves this global at call time, so patching it here
# keeps the public ABI stable while using the lower-launch-count group=1 packer.
_impl.pack_triton_v_chunk_into = _fast_v_pack

from .triton_qkv_fast import *  # noqa: E402,F401,F403

pack_triton_v_chunk_into = _fast_v_pack
