'''Fixed-density Sparse Sage execution for MiniMax H3.'''

from importlib import import_module


_EXPORT_MODULES = {
    'HybridSparseBackend': '.backend',
    'PreparedHybrid': '.backend',
    'HybridSparseConfig': '.config',
    'DENSITY_FIXED': '.config',
    'IMPLEMENTED_MODES': '.config',
    'MODE_SAGE128': '.config',
    'MODE_SAGE128_FUSED_QKV': '.config',
    'ChunkedSparseQKVProjector': '.chunked_qkv',
    'pack_sparse_qk_chunk_into': '.chunked_qkv',
    'run_chunked_sparse_qkv': '.chunked_qkv',
    'FusedQKVError': '.fused_qkv',
    'FusedQKVProjector': '.fused_qkv',
    'PreparedFusedQKV': '.fused_qkv',
    'validate_prepared_fused_qkv': '.fused_qkv',
    'FP8FlexBackend': '.fp8_flex',
    'FP8FlexError': '.fp8_flex',
    'FP8FlexSpec': '.fp8_flex',
    'PreparedFP8Flex': '.fp8_flex',
    'block_mask_from_delta_lut': '.fp8_flex',
    'load_fp8_flex_spec': '.fp8_flex',
    'preflight_fp8_flex': '.fp8_flex',
    'KV_TILE': '.router',
    'Q_TILE': '.router',
    'SparseMaskMetadata': '.router',
    'SparseRouterError': '.router',
    'SparseTileGeometry': '.router',
    'SparseTileRouter': '.router',
    'PreparedSparseSage': '.sparse_sage',
    'SparseSageKernelSpec': '.sparse_sage',
    'SparseSageError': '.sparse_sage',
    'SparseSageExecutor': '.sparse_sage',
    'load_sparse_sage_spec': '.sparse_sage',
    'preflight_sparse_sage': '.sparse_sage',
    'resolve_sparse_sage_spec': '.sparse_sage',
    'ChunkedTritonSparseQKVProjector': '.triton_qkv',
    'PreparedTritonSparseQKV': '.triton_qkv',
    'TritonSparseQKVError': '.triton_qkv',
    'pack_float_qkv': '.triton_qkv',
    'pack_triton_v_chunk_into': '.triton_qkv',
    'run_chunked_triton_sparse_qkv': '.triton_qkv',
    'validate_prepared_triton_sparse_qkv': '.triton_qkv',
    'PreparedTritonSparse': '.triton_sparse',
    'TritonSparseBackend': '.triton_sparse',
    'TritonSparseError': '.triton_sparse',
    'TritonSparseExecutor': '.triton_sparse',
    'TritonSparseSpec': '.triton_sparse',
    'preflight_triton_sparse': '.triton_sparse',
}


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name, __name__), name)


__all__ = list(_EXPORT_MODULES)
