'''Resolve and apply the complete two-node H3 optimization plan.'''

from __future__ import annotations

from dataclasses import dataclass
import logging

from .attention.sparse import (
    HybridSparseBackend,
    HybridSparseConfig,
    MODE_SAGE128,
    MODE_SAGE128_FUSED_QKV,
    preflight_sparse_sage,
)
from .attention.sparse.fused_qkv import (
    TRITON_AVAILABLE as SPARSE_TRITON_AVAILABLE,
)
from .dense_resolver import resolve_dense_attention
from .environment import RuntimeEnvironment
from .memory.config import ActivationMemoryConfig
from .memory.patch import install as install_memory_patch
from .model import get_h3_blocks, is_minimax_h3
from .patch import configure_backend
from .plan import (
    ATTENTION_EXISTING,
    DENSITY_FIXED,
    FUSED_QKV_OFF,
    H3OptimizationPlan,
    PLAN_KEY,
    STATUS_KEY,
)
from .qkv.formats import inspect_h3_linears
from .qkv.providers import (
    MLP_OFF,
    QKV_DENSE_CONVROT_INT8,
    QKV_SPARSE_CONVROT_INT8,
    resolve_mlp_provider,
    resolve_qkv_provider,
)
from .runtime.context import (
    H3RuntimeSession,
    RUNTIME_SESSION_KEY,
    install_runtime_wrapper,
)

LOG_PREFIX = '[H3 Optimizations]'
ATTENTION_SPARSE = 'sparse_sage'


@dataclass(frozen=True)
class ResolvedAttention:
    requested: str
    selected: str
    backend: object | None
    reason: str
    backend_kind: str
    projector: object | None = None


def _fused_request(plan):
    return FUSED_QKV_OFF if plan.memory is None else plan.memory.fused_qkv


def _dense_triton_available():
    try:
        from .dense_fused_qkv import TRITON_AVAILABLE
    except Exception:
        return False
    return bool(TRITON_AVAILABLE)


def _resolve_dense(plan, environment, inventory):
    memory = plan.memory
    requested = ATTENTION_EXISTING if memory is None else memory.attention
    dense = resolve_dense_attention(requested, environment)
    qkv = resolve_qkv_provider(
        inventory,
        request=_fused_request(plan),
        backend_kind=dense.backend_kind,
        capability=environment.capability,
        triton_available=_dense_triton_available(),
    )
    backend = dense.backend
    projector = None
    if qkv.provider_id == QKV_DENSE_CONVROT_INT8:
        from .dense_backend import ProjectedSM89SageBackend
        from .qkv.projectors import DenseFusedQKVProjector

        backend = ProjectedSM89SageBackend(backend)
        projector = DenseFusedQKVProjector()
    return (
        ResolvedAttention(
            requested=dense.requested,
            selected=dense.selected,
            backend=backend,
            reason=dense.reason,
            backend_kind=dense.backend_kind,
            projector=projector,
        ),
        qkv,
    )


def _resolve_sparse(plan, environment, inventory):
    kernel_spec = preflight_sparse_sage(
        cuda_available=lambda: environment.cuda_available,
        capability_getter=lambda: environment.capability,
    )
    qkv = resolve_qkv_provider(
        inventory,
        request=_fused_request(plan),
        backend_kind=ATTENTION_SPARSE,
        capability=environment.capability,
        triton_available=bool(SPARSE_TRITON_AVAILABLE),
        sparse_spec=kernel_spec,
    )
    use_fused = qkv.provider_id == QKV_SPARSE_CONVROT_INT8
    config = HybridSparseConfig(
        mode=MODE_SAGE128_FUSED_QKV if use_fused else MODE_SAGE128,
        video_budget=float(plan.sparse.video_budget),
        density_mode=DENSITY_FIXED,
        denser_early_late_steps=bool(plan.sparse.denser_early_late_steps),
        strict=True,
    )
    projector = None
    if use_fused:
        from .qkv.projectors import SparseFusedQKVProjector

        projector = SparseFusedQKVProjector()
    backend = HybridSparseBackend(
        config,
        kernel_spec=kernel_spec,
        projector=projector,
    )
    return (
        ResolvedAttention(
            requested=ATTENTION_SPARSE,
            selected=ATTENTION_SPARSE,
            backend=backend,
            reason='explicit fixed-density Sparse Sage attention',
            backend_kind=ATTENTION_SPARSE,
            projector=projector,
        ),
        qkv,
    )


def _install_mlp(model_patcher, plan, inventory):
    memory = plan.memory
    if memory is None:
        return resolve_mlp_provider(inventory, request='off'), 0

    resolution = resolve_mlp_provider(
        inventory,
        request=memory.mlp_memory,
    )
    if resolution.provider_id == MLP_OFF:
        return resolution, 0

    config = ActivationMemoryConfig(
        mode=resolution.activation_mode,
        chunk_rows=int(memory.chunk_rows),
        strict=False,
        prefer_held_weights=bool(memory.prefer_held_weights),
    )
    return resolution, int(install_memory_patch(model_patcher, config))


def _ensure_sparse_runtime(model_patcher):
    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    session = options.get(RUNTIME_SESSION_KEY)
    if session is not None:
        if not isinstance(session, H3RuntimeSession):
            raise TypeError(
                '%s is not an H3 Optimizations runtime session'
                % RUNTIME_SESSION_KEY
            )
        session.strict_layout = True
        return session, False

    session = H3RuntimeSession(strict_layout=True)
    install_runtime_wrapper(model_patcher, session)
    return session, True


def _inventory_status(inventory):
    return {
        'qkv': list(inventory.labels('qkv')),
        'fc1': list(inventory.labels('fc1')),
        'fc2': list(inventory.labels('fc2')),
    }


def _status(
    plan,
    environment,
    attention,
    qkv,
    mlp,
    *,
    attention_blocks,
    mlp_blocks,
    runtime_installed,
    inventory,
):
    return {
        'plan_version': int(plan.version),
        'plan_signature': plan.signature,
        'attention': {
            'requested': attention.requested,
            'selected': attention.selected,
            'reason': attention.reason,
            'patched_blocks': int(attention_blocks),
        },
        'sparse': (
            None
            if plan.sparse is None
            else {
                'video_budget': float(plan.sparse.video_budget),
                'denser_early_late_steps': bool(
                    plan.sparse.denser_early_late_steps
                ),
            }
        ),
        'fused_qkv': {
            'requested': _fused_request(plan),
            'provider': qkv.provider_id,
            'fused': bool(qkv.fused),
            'reason': qkv.reason,
            'projector': getattr(attention.projector, 'name', None),
        },
        'mlp': {
            'requested': 'off' if plan.memory is None else plan.memory.mlp_memory,
            'provider': mlp.provider_id,
            'activation_mode': mlp.activation_mode,
            'reason': mlp.reason,
            'chunk_rows': (
                None if plan.memory is None else int(plan.memory.chunk_rows)
            ),
            'patched_blocks': int(mlp_blocks),
        },
        'weight_formats': _inventory_status(inventory),
        'runtime_installed': bool(runtime_installed),
        'device': {
            'name': environment.device_name,
            'architecture': environment.architecture,
            'capability': (
                None
                if environment.capability is None
                else [int(value) for value in environment.capability]
            ),
        },
    }


def apply_plan(model, plan: H3OptimizationPlan):
    '''Apply compatible H3 features; other model families are exact no-ops.'''

    if not isinstance(plan, H3OptimizationPlan):
        raise TypeError('plan must be H3OptimizationPlan')
    if not is_minimax_h3(model):
        return model

    blocks = get_h3_blocks(model)
    inventory = inspect_h3_linears(blocks)
    environment = RuntimeEnvironment.detect()

    if plan.sparse is not None:
        attention, qkv = _resolve_sparse(plan, environment, inventory)
    else:
        attention, qkv = _resolve_dense(plan, environment, inventory)

    patched = model.clone()
    attention_blocks = 0
    if attention.backend is not None:
        _backend, attention_blocks = configure_backend(
            patched,
            attention.backend,
            projector=attention.projector,
        )

    mlp, mlp_blocks = _install_mlp(patched, plan, inventory)
    runtime_installed = False
    if plan.sparse is not None:
        _session, _created = _ensure_sparse_runtime(patched)
        runtime_installed = True

    patched.model_options[PLAN_KEY] = plan
    options = patched.model_options['transformer_options'] = (
        patched.model_options.get('transformer_options', {}).copy()
    )
    options[STATUS_KEY] = _status(
        plan,
        environment,
        attention,
        qkv,
        mlp,
        attention_blocks=attention_blocks,
        mlp_blocks=mlp_blocks,
        runtime_installed=runtime_installed,
        inventory=inventory,
    )
    logging.info(
        '%s armed: attention=%s qkv=%s mlp=%s device=%s',
        LOG_PREFIX,
        attention.selected,
        qkv.provider_id,
        mlp.provider_id,
        environment.device_name,
    )
    return patched
