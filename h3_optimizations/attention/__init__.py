'''Production dense and sparse attention backends owned by this package.'''

from importlib import import_module

_ARCHITECTURE_BACKENDS = {
    'SageSM80MemoryEfficientBackend',
    'SageSM86MemoryEfficientBackend',
    'SageSM90MemoryEfficientBackend',
    'SageSM12xMemoryEfficientBackend',
}
_SPARSE_EXPORTS = {
    'HybridSparseBackend',
    'HybridSparseConfig',
    'SparseSageError',
    'preflight_sparse_sage',
}


def __getattr__(name):
    if name == 'SM89SageMemoryEfficientBackend':
        implementation = import_module(
            '.sage_mem_eff',
            __name__,
        )
        import_module('.sm89_compat', __name__)
        import_module('.v_snapshot_compat', __name__)
        return implementation.SM89SageMemoryEfficientBackend
    if name in _ARCHITECTURE_BACKENDS:
        return getattr(import_module('.sage_arch', __name__), name)
    if name in _SPARSE_EXPORTS:
        return getattr(import_module('.sparse', __name__), name)
    raise AttributeError(name)


__all__ = [
    'SageSM80MemoryEfficientBackend',
    'SageSM86MemoryEfficientBackend',
    'SM89SageMemoryEfficientBackend',
    'SageSM90MemoryEfficientBackend',
    'SageSM12xMemoryEfficientBackend',
    'HybridSparseBackend',
    'HybridSparseConfig',
    'SparseSageError',
    'preflight_sparse_sage',
]
