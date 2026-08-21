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
    'FusedQKVError': '.fused_qkv',
    'FusedQKVProjector': '.fused_qkv',
    'PreparedFusedQKV': '.fused_qkv',
    'validate_prepared_fused_qkv': '.fused_qkv',
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
}


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name, __name__), name)


__all__ = list(_EXPORT_MODULES)
