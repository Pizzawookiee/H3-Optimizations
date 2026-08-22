'''Immutable, order-independent configuration for H3 optimizations.'''

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .mlp_sharing.config import MLPSharingConfig

PLAN_KEY = 'h3_optimizations_plan'
STATUS_KEY = 'h3_optimizations_status'
PLAN_VERSION = 3

ATTENTION_AUTO = 'auto'
ATTENTION_EXISTING = 'existing'
ATTENTION_REQUESTS = (ATTENTION_AUTO, ATTENTION_EXISTING)

FUSED_QKV_AUTO = 'auto'
FUSED_QKV_OFF = 'off'
FUSED_QKV_REQUIRED = 'required'
FUSED_QKV_REQUESTS = (FUSED_QKV_AUTO, FUSED_QKV_OFF, FUSED_QKV_REQUIRED)

MLP_MEMORY_AUTO = 'auto'
MLP_MEMORY_OFF = 'off'
MLP_MEMORY_LEGACY_BF16 = 'legacy_bf16'
MLP_MEMORY_LEGACY_NATIVE = 'legacy_native'
MLP_MEMORY_LEGACY_CONVROT_REQUIRED = 'legacy_convrot_2slice_required'
MLP_MEMORY_REQUESTS = (
    MLP_MEMORY_AUTO,
    MLP_MEMORY_OFF,
    MLP_MEMORY_LEGACY_BF16,
    MLP_MEMORY_LEGACY_NATIVE,
    MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
)

SPARSE_BACKEND_AUTO = 'auto'
SPARSE_BACKEND_SAGE = 'Sparse Sage'
SPARSE_BACKEND_TRITON = 'INT8 Triton'
SPARSE_BACKEND_FLEX = 'FP8 FlexAttention'
SPARSE_BACKEND_REQUESTS = (
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_SAGE,
    SPARSE_BACKEND_TRITON,
    SPARSE_BACKEND_FLEX,
)

MIN_CHUNK_ROWS = 256
MAX_CHUNK_ROWS = 65_536
CHUNK_ALIGNMENT = 256
DENSITY_FIXED = 'fixed'
DEFAULT_VIDEO_BUDGET = 0.3
DEFAULT_EDGE_STEPS = 2
DEFAULT_EDGE_KV = 0.5
H3_LAYER_COUNT = 50


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


def parse_layer_video_budgets(value):
    text = str(value).strip()
    if not text:
        return None
    budgets = tuple(float(item.strip()) for item in text.split(','))
    if len(budgets) != H3_LAYER_COUNT:
        raise ValueError(
            'layer_video_budgets must contain exactly %d comma-separated values'
            % H3_LAYER_COUNT
        )
    for layer_index, budget in enumerate(budgets):
        _validate_sparse_budget('layer_video_budgets[%d]' % layer_index, budget)
    return budgets


@dataclass(frozen=True)
class MemoryRequest:
    '''Execution and activation-memory options owned by the memory node.'''

    attention: str = ATTENTION_AUTO
    fused_qkv: str = FUSED_QKV_AUTO
    mlp_memory: str = MLP_MEMORY_AUTO
    chunk_rows: int = 4096
    prefer_held_weights: bool = True
    mlp_strict: bool = False

    def __post_init__(self):
        if self.attention not in ATTENTION_REQUESTS:
            raise ValueError('unknown dense attention request %r' % self.attention)
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
            self.attention,
            self.fused_qkv,
            self.mlp_memory,
            int(self.chunk_rows),
            bool(self.prefer_held_weights),
            bool(self.mlp_strict),
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
    backend: str = SPARSE_BACKEND_AUTO
    layer_video_budgets: tuple[float, ...] | None = None

    def __post_init__(self):
        _validate_sparse_budget('video_budget', self.video_budget)
        if self.backend not in SPARSE_BACKEND_REQUESTS:
            raise ValueError('unknown sparse backend request %r' % self.backend)
        _validate_edge_schedule(
            self.early_steps,
            self.early_kv,
            self.late_steps,
            self.late_kv,
        )
        if self.layer_video_budgets is not None:
            budgets = tuple(float(value) for value in self.layer_video_budgets)
            if len(budgets) != H3_LAYER_COUNT:
                raise ValueError(
                    'layer_video_budgets must contain exactly %d values'
                    % H3_LAYER_COUNT
                )
            for layer_index, budget in enumerate(budgets):
                _validate_sparse_budget(
                    'layer_video_budgets[%d]' % layer_index,
                    budget,
                )
            object.__setattr__(self, 'layer_video_budgets', budgets)
        if self.advanced_schedule and self.denser_early_late_steps:
            raise ValueError(
                'explicit early/late budgets cannot be combined with the '
                'legacy denser early/late toggle'
            )
        if self.layer_video_budgets is not None and (
            self.denser_early_late_steps or self.advanced_schedule
        ):
            raise ValueError(
                'static per-layer budgets cannot be combined with early/late schedules'
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
            self.backend,
            None if self.early_steps is None else int(self.early_steps),
            None if self.early_kv is None else float(self.early_kv),
            None if self.late_steps is None else int(self.late_steps),
            None if self.late_kv is None else float(self.late_kv),
            self.layer_video_budgets,
        )


@dataclass(frozen=True)
class H3OptimizationPlan:
    '''Complete composable request carried by one cloned ModelPatcher.'''

    version: int = PLAN_VERSION
    memory: MemoryRequest | None = None
    sparse: SparseRequest | None = None
    mlp_sharing: MLPSharingConfig | None = None

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

    def with_mlp_sharing(self, request: MLPSharingConfig):
        if not isinstance(request, MLPSharingConfig):
            raise TypeError('request must be MLPSharingConfig')
        if self.mlp_sharing is not None and self.mlp_sharing != request:
            raise ValueError(
                'a different H3 MLP Sharing node is already present; '
                'remove one instead of relying on node order'
            )
        return replace(self, mlp_sharing=request)

    @property
    def signature(self):
        return (
            int(self.version),
            None if self.memory is None else self.memory.signature,
            None if self.sparse is None else self.sparse.signature,
            None if self.mlp_sharing is None else self.mlp_sharing.signature,
        )


def read_plan(model):
    options = getattr(model, 'model_options', {}) or {}
    plan = options.get(PLAN_KEY)
    if plan is None:
        return H3OptimizationPlan()
    if not isinstance(plan, H3OptimizationPlan):
        raise TypeError('%s does not contain an H3OptimizationPlan' % PLAN_KEY)
    return plan
