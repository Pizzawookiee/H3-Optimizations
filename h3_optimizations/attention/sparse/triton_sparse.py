'''Stable public surface for the BF16 Triton sparse backend.

The production backend keeps projected Q/K/V in BF16 and uses ordinary BF16
tensor-core dots with FP32 online-softmax state at 64Q x 64KV.
'''

from .triton_bf16 import (
    PreparedTritonBF16,
    TritonBF16Backend,
    TritonBF16Error,
    TritonBF16Spec,
    preflight_triton_bf16,
)


TritonSparseError = TritonBF16Error
TritonSparseSpec = TritonBF16Spec


def TritonSparseBackend(config=None, **kwargs):
    '''Construct the portable BF16 Triton backend.'''
    return TritonBF16Backend(config, **kwargs)


TritonSparseBackend.name = TritonBF16Backend.name


def preflight_triton_sparse(**kwargs):
    '''Validate the portable BF16 Triton fallback.'''
    return preflight_triton_bf16(**kwargs)


__all__ = [
    'PreparedTritonBF16',
    'TritonSparseBackend',
    'TritonSparseError',
    'TritonSparseSpec',
    'preflight_triton_sparse',
]
