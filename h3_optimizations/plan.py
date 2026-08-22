'''Immutable, order-independent configuration for H3 optimizations.'''

from __future__ import annotations

from dataclasses import dataclass, replace
import math

PLAN_KEY = 'h3_optimizations_plan'
STATUS_KEY = 'h3_optimizations_status'
PLAN_VERSION = 1

FUSED_QKV_AUTO = 'auto'
FUSED_QKV_OFF = 'off'
FUSED_QKV_REQUESTS = (FUSED_QKV_AUTO, FUSED_QKV_OFF)

MLP_MEMORY_AUTO = 'auto'
MLP_MEMORY_OFF = 'off'
MLP_MEMORY_REQUESTS = (MLP_MEMORY_AUTO, MLP_MEMORY_OFF)

MIN_CHUNK_ROWS = 256
MAX_CHUNK_ROWS = 65_536
CHUNK_ALIGNMENT = 256
DENSITY_FIXED = 'fixed'
DEFAULT_VIDEO_BUDGET = 0.3
DEFAULT_EDGE_STEPS = 2
DEFAULT_EDGE_KV = 0.5


def _validate_sparse_budget(name, value):
    budget = float(value)
    if not math.isfinite(budget) or not 0.01 <= budget <= 1.0:
        raise ValueError('%s must be finite and in [0.01, 1]' % name)


def _validate_edge_schedule(early_steps, early_kv, late_steps, late_kv):
    values = (early_steps, early_kv, late_steps, late_kv)
    if not any(value is not None for value in values):
        return
    if not all(value is not None for value in values):
        raise ValueError(
            'early_steps, early_kv, late_steps, and late_kv must be set together'
        )
    for name, value in (
        ('early_steps', early_steps),
        ('late_steps', late_steps),
    ):
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError('%s must be a non-negative integer' % name)
    _validate_sparse_budget('early_kv', early_kv)
    _validate_sparse_budget('late_kv', late_kv)


@dataclass(frozen=True)
class MemoryRequest:
    '''Execution and activation-memory options owned by the memory node.'''

    fused_qkv: str = FUSED_QKV_AUTO
    mlp_memory: str = MLP_MEMORY_AUTO
    chunk_rows: int = 4096

    def __post_init__(self):
        if self.fused_qkv not in FUSED_QKV_REQUESTS:
            raise ValueError('unknown fused QKV request %r' % self.fused_qkv)
        if self.mlp_memory not in MLP_MEMORY_REQUESTS:
            raise ValueError('unknown MLP memory request %r' % self.mlp_memory)
        chunk_rows = int(self.chunk_rows)
        if not MIN_CHUNK_ROWS <= chunk_rows <= MAX_CHUNK_ROWS:
            raise ValueError(
                'chunk_rows must be between %d and %d'
                % (MIN_CHUNK_ROWS, MAX_CHUNK_ROWS)
            )
        if chunk_rows % CHUNK_ALIGNMENT:
            raise ValueError(
                'chunk_rows must be a multiple of %d' % CHUNK_ALIGNMENT
            )

    @property
    def signature(self):
        return (
            self.fused_qkv,
            self.mlp_memory,
            int(self.chunk_rows),
        )


@dataclass(frozen=True)
class SparseRequest:
    '''Fixed-density sparse attention request.'''

    video_budget: float = DEFAULT_VIDEO_BUDGET
    denser_early_late_steps: bool = False
    early_steps: int | None = None
    early_kv: float | None = None
    late_steps: int | None = None
    late_kv: float | None = None

    def __post_init__(self):
        _validate_sparse_budget('video_budget', self.video_budget)
        _validate_edge_schedule(
            self.early_steps,
            self.early_kv,
            self.late_steps,
            self.late_kv,
        )
        if self.advanced_schedule and self.denser_early_late_steps:
            raise ValueError(
                'explicit early/late budgets cannot be combined with the '
                'legacy denser early/late toggle'
            )

    @property
    def advanced_schedule(self):
        return self.early_steps is not None

    @property
    def signature(self):
        return (
            float(self.video_budget),
            DENSITY_FIXED,
            bool(self.denser_early_late_steps),
            None if self.early_steps is None else int(self.early_steps),
            None if self.early_kv is None else float(self.early_kv),
            None if self.late_steps is None else int(self.late_steps),
            None if self.late_kv is None else float(self.late_kv),
        )


@dataclass(frozen=True)
class H3OptimizationPlan:
    '''Complete composable request carried by one cloned ModelPatcher.'''

    version: int = PLAN_VERSION
    memory: MemoryRequest | None = None
    sparse: SparseRequest | None = None

    def __post_init__(self):
        if int(self.version) != PLAN_VERSION:
            raise ValueError(
                'unsupported H3 optimization plan version %r' % self.version
            )

    def with_memory(self, request: MemoryRequest):
        if not isinstance(request, MemoryRequest):
            raise TypeError('request must be MemoryRequest')
        if self.memory is not None and self.memory != request:
            raise ValueError(
                'a different H3 Memory Optimization node is already present; '
                'remove one instead of relying on node order'
            )
        return replace(self, memory=request)

    def with_sparse(self, request: SparseRequest):
        if not isinstance(request, SparseRequest):
            raise TypeError('request must be SparseRequest')
        if self.sparse is not None and self.sparse != request:
            raise ValueError(
                'a different H3 Sparse Attention node is already present; '
                'remove one instead of relying on node order'
            )
        return replace(self, sparse=request)

    @property
    def signature(self):
        return (
            int(self.version),
            None if self.memory is None else self.memory.signature,
            None if self.sparse is None else self.sparse.signature,
        )


def read_plan(model):
    options = getattr(model, 'model_options', {}) or {}
    plan = options.get(PLAN_KEY)
    if plan is None:
        return H3OptimizationPlan()
    if not isinstance(plan, H3OptimizationPlan):
        raise TypeError('%s does not contain an H3OptimizationPlan' % PLAN_KEY)
    return plan
