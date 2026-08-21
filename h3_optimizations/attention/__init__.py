'''Production Sparse Sage attention exports owned by this package.'''

from importlib import import_module

_SPARSE_EXPORTS = {
    'HybridSparseBackend',
    'HybridSparseConfig',
    'SparseSageError',
    'preflight_sparse_sage',
}


def __getattr__(name):
    if name in _SPARSE_EXPORTS:
        return getattr(import_module('.sparse', __name__), name)
    raise AttributeError(name)


__all__ = [
    'HybridSparseBackend',
    'HybridSparseConfig',
    'SparseSageError',
    'preflight_sparse_sage',
]
