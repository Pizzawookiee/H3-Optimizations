'''Fixed-density Sparse Sage execution for MiniMax H3.'''


_EXPORT_MODULES = {
    'HybridSparseBackend': 'backend',
    'PreparedHybrid': 'backend',
    'HybridSparseConfig': 'config',
    'DENSITY_FIXED': 'config',
    'IMPLEMENTED_MODES': 'config',
    'MODE_SAGE128': 'config',
    'MODE_SAGE128_FUSED_QKV': 'config',
    'ChunkedSparseQKVProjector': 'chunked_qkv',
    'pack_sparse_qk_chunk_into': 'chunked_qkv',
    'run_chunked_sparse_qkv': 'chunked_qkv',
    'FusedQKVError': 'fused_qkv',
    'FusedQKVProjector': 'fused_qkv',
    'PreparedFusedQKV': 'fused_qkv',
    'validate_prepared_fused_qkv': 'fused_qkv',
    'FP8FlexBackend': 'fp8_flex',
    'FP8FlexError': 'fp8_flex',
    'FP8FlexSpec': 'fp8_flex',
    'PreparedFP8Flex': 'fp8_flex',
    'block_mask_from_delta_lut': 'fp8_flex',
    'load_fp8_flex_spec': 'fp8_flex',
    'preflight_fp8_flex': 'fp8_flex',
    'FrostBF16Backend': 'frost_bf16',
    'FrostBF16Error': 'frost_bf16',
    'FrostBF16Executor': 'frost_bf16',
    'FrostBF16Spec': 'frost_bf16',
    'PreparedFrostBF16': 'frost_bf16',
    'preflight_frost_bf16': 'frost_bf16',
    'KV_TILE': 'router',
    'Q_TILE': 'router',
    'SparseMaskMetadata': 'router',
    'SparseRouterError': 'router',
    'SparseTileGeometry': 'router',
    'SparseTileRouter': 'router',
    'PreparedSparseSage': 'sparse_sage',
    'SparseSageKernelSpec': 'sparse_sage',
    'SparseSageError': 'sparse_sage',
    'SparseSageExecutor': 'sparse_sage',
    'load_sparse_sage_spec': 'sparse_sage',
    'preflight_sparse_sage': 'sparse_sage',
    'resolve_sparse_sage_spec': 'sparse_sage',
    'TritonSparseBackend': 'triton_sparse',
    'TritonSparseError': 'triton_sparse',
    'TritonSparseSpec': 'triton_sparse',
    'preflight_triton_sparse': 'triton_sparse',
}


def _load(module_name):
    """Import one backend module by name, lazily and explicitly.

    Deferred so that importing this package does not drag in every optional
    backend, and spelled out per module so no import target is built at
    runtime -- the Registry package scanner reads a computed module name as
    import obfuscation.
    """
    if module_name == 'backend':
        from . import backend as module
    elif module_name == 'config':
        from . import config as module
    elif module_name == 'chunked_qkv':
        from . import chunked_qkv as module
    elif module_name == 'fused_qkv':
        from . import fused_qkv as module
    elif module_name == 'fp8_flex':
        from . import fp8_flex as module
    elif module_name == 'frost_bf16':
        from . import frost_bf16 as module
    elif module_name == 'router':
        from . import router as module
    elif module_name == 'sparse_sage':
        from . import sparse_sage as module
    elif module_name == 'triton_sparse':
        from . import triton_sparse as module
    else:
        raise AttributeError(module_name)
    return module


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(_load(module_name), name)


# Sparse Kitchen is the production sparse backend and has a first-class
# streamed-Q composition. Install its adapter explicitly when the sparse
# package loads; optional Sage/FROST/Triton modules remain lazy.
from .kitchen_streamed_q import install as _install_sparse_kitchen_streamed_q

_install_sparse_kitchen_streamed_q()
del _install_sparse_kitchen_streamed_q


__all__ = list(_EXPORT_MODULES)
