'''Production Sparse Sage attention exports owned by this package.'''


class AttentionBackendUnavailable(RuntimeError):
    pass


_SPARSE_EXPORTS = {
    'HybridSparseBackend',
    'HybridSparseConfig',
    'SparseSageError',
    'preflight_sparse_sage',
}


def __getattr__(name):
    if name in _SPARSE_EXPORTS:
        # Imported here, not at module scope, so importing this package does
        # not pull in every optional sparse backend and its dependencies.
        from . import sparse
        return getattr(sparse, name)
    raise AttributeError(name)


__all__ = [
    'AttentionBackendUnavailable',
    'HybridSparseBackend',
    'HybridSparseConfig',
    'SparseSageError',
    'preflight_sparse_sage',
]
