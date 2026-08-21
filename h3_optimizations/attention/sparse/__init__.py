'''Fixed-density Sparse Sage execution for MiniMax H3.'''

from .backend import HybridSparseBackend, PreparedHybrid
from .config import (
    DENSITY_FIXED,
    HybridSparseConfig,
    IMPLEMENTED_MODES,
    MODE_SAGE128,
    MODE_SAGE128_FUSED_QKV,
)
from .fused_qkv import (
    FusedQKVError,
    FusedQKVProjector,
    PreparedFusedQKV,
    validate_prepared_fused_qkv,
)
from .router import (
    KV_TILE,
    Q_TILE,
    SparseMaskMetadata,
    SparseRouterError,
    SparseTileGeometry,
    SparseTileRouter,
)
from .sparse_sage import (
    PreparedSparseSage,
    SparseSageError,
    SparseSageExecutor,
    SparseSageKernelSpec,
    load_sparse_sage_spec,
    preflight_sparse_sage,
    resolve_sparse_sage_spec,
)

__all__ = [
    'HybridSparseBackend',
    'PreparedHybrid',
    'HybridSparseConfig',
    'DENSITY_FIXED',
    'IMPLEMENTED_MODES',
    'MODE_SAGE128',
    'MODE_SAGE128_FUSED_QKV',
    'FusedQKVError',
    'FusedQKVProjector',
    'PreparedFusedQKV',
    'validate_prepared_fused_qkv',
    'KV_TILE',
    'Q_TILE',
    'SparseMaskMetadata',
    'SparseRouterError',
    'SparseTileGeometry',
    'SparseTileRouter',
    'PreparedSparseSage',
    'SparseSageKernelSpec',
    'SparseSageError',
    'SparseSageExecutor',
    'load_sparse_sage_spec',
    'preflight_sparse_sage',
    'resolve_sparse_sage_spec',
]
