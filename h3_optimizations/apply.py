'''Resolve and apply the complete two-node H3 optimization plan.'''

from __future__ import annotations

from dataclasses import dataclass
import logging

from .attention.sparse import (
    FP8FlexBackend,
    FP8FlexError,
    HybridSparseBackend,
    HybridSparseConfig,
    MODE_SAGE128,
    MODE_SAGE128_FUSED_QKV,
    SparseSageError,
    TritonSparseBackend,
    TritonSparseError,
    preflight_fp8_flex,
    preflight_sparse_sage,
    preflight_triton_sparse,
)
from .attention.sparse.fused_qkv import (
    TRITON_AVAILABLE as SPARSE_TRITON_AVAILABLE,
)
from .dense_resolver import (
    install_dense_attention,
    preserve_dense_attention,
    resolve_dense_attention,
)
from .environment import RuntimeEnvironment
from .kitchen_qkv import (
    PRODUCER_ABI as KITCHEN_PRODUCER_ABI,
    ChunkedKitchenAttentionBackend,
    ChunkedKitchenQKVProjector,
    producer_api_available,
)
from .memory.config import ActivationMemoryConfig
from .memory.patch import install as install_memory_patch
from .model import get_h3_blocks, is_minimax_h3
from .patch import configure_backend
from .plan import (
    DENSITY_FIXED,
    FUSED_QKV_OFF,
    H3OptimizationPlan,
    PLAN_KEY,
    STATUS_KEY,
)
from .qkv.formats import inspect_h3_linears
from .qkv.providers import (
    MLP_OFF,
    QKV_DENSE_KITCHEN_CHUNKED,
    QKV_SPARSE_CONVROT_INT8,
    QKV_TRITON_SPARSE_CHUNKED,
    resolve_mlp_provider,
    resolve_qkv_provider,
)
from .runtime.context import (
    H3RuntimeSession,
    RUNTIME_SESSION_KEY,
    install_runtime_wrapper,
)
from .v_layout_compat import (
    install_v_layout_compat,
    not_applicable_v_layout,
)

LOG_PREFIX = '[H3 Optimizations]'
ATTENTION_SPARSE = 'sparse_sage'
ATTENTION_TRITON_SPARSE = 'triton_sparse_int8'
ATTENTION_FP8_FLEX = 'flex_attention_fp8'


@dataclass(frozen=True)
class ResolvedAttention:
    requested: str
    selected: str
    backend: object | None
    reason: str
    backend_kind: str
    projector: object | None = None
    dense_resolution: object | None = None


def _fused_request(plan):
    return FUSED_QKV_OFF if plan.memory is None else plan.memory.fused_qkv


def _resolve_dense(plan, model, inventory, environment=None):
    memory = plan.memory
    dense = (
        preserve_dense_attention('no memory optimization requested')
        if memory is None
        else resolve_dense_attention(model)
    )
    qkv = resolve_qkv_provider(
        inventory,
        request=_fused_request(plan),
        backend_kind=dense.backend_kind,
        kitchen_producer_available=producer_api_available(
            device=getattr(environment, 'device_index', None)
        ),
    )
    backend = None
    projector = None
    if qkv.provider_id == QKV_DENSE_KITCHEN_CHUNKED:
        backend = ChunkedKitchenAttentionBackend()
        projector = ChunkedKitchenQKVProjector()
    return (
        ResolvedAttention(
            requested=dense.requested,
            selected=dense.selected,
            backend=backend,
            reason=dense.reason,
            backend_kind=dense.backend_kind,
            projector=projector,
            dense_resolution=dense,
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

        projector = SparseFusedQKVProjector(
            kernel_spec,
            chunk_rows=4096,
        )
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


def _resolve_fp8_flex(
    plan,
    environment,
    inventory,
    fallback_reason,
    dense_attention,
):
    spec = preflight_fp8_flex(
        cuda_available=lambda: environment.cuda_available,
        capability_getter=lambda: environment.capability,
        device=getattr(environment, 'device_index', None),
    )
    qkv = resolve_qkv_provider(
        inventory,
        request=_fused_request(plan),
        backend_kind=ATTENTION_FP8_FLEX,
    )
    config = HybridSparseConfig(
        mode=MODE_SAGE128,
        video_budget=float(plan.sparse.video_budget),
        density_mode=DENSITY_FIXED,
        denser_early_late_steps=bool(plan.sparse.denser_early_late_steps),
        strict=True,
    )
    backend = FP8FlexBackend(config, spec=spec)
    return (
        ResolvedAttention(
            requested=ATTENTION_SPARSE,
            selected=ATTENTION_FP8_FLEX,
            backend=backend,
            reason=(
                '%s; using FP8 FlexAttention'
                % fallback_reason
            ),
            backend_kind=ATTENTION_FP8_FLEX,
            dense_resolution=dense_attention.dense_resolution,
        ),
        qkv,
    )


def _resolve_triton_sparse(plan, environment, inventory, sparse_error):
    spec = preflight_triton_sparse(
        cuda_available=lambda: environment.cuda_available,
        capability_getter=lambda: environment.capability,
    )
    qkv = resolve_qkv_provider(
        inventory,
        request=_fused_request(plan),
        backend_kind=ATTENTION_TRITON_SPARSE,
        triton_available=True,
    )
    config = HybridSparseConfig(
        mode=MODE_SAGE128,
        video_budget=float(plan.sparse.video_budget),
        density_mode=DENSITY_FIXED,
        denser_early_late_steps=bool(plan.sparse.denser_early_late_steps),
        strict=True,
    )
    projector = None
    if qkv.provider_id == QKV_TRITON_SPARSE_CHUNKED:
        from .qkv.projectors import TritonSparseQKVProjector

        projector = TritonSparseQKVProjector(chunk_rows=4096)
    backend = TritonSparseBackend(
        config,
        spec=spec,
        projector=projector,
    )
    return (
        ResolvedAttention(
            requested=ATTENTION_SPARSE,
            selected=ATTENTION_TRITON_SPARSE,
            backend=backend,
            reason=(
                'Sparse Sage unavailable: %s; using INT8 Triton sparse attention'
                % sparse_error
            ),
            backend_kind=ATTENTION_TRITON_SPARSE,
            projector=projector,
        ),
        qkv,
    )


def _resolve_attention(plan, model, inventory, environment):
    dense_attention, dense_qkv = _resolve_dense(
        plan,
        model,
        inventory,
        environment,
    )
    if plan.sparse is None:
        return dense_attention, dense_qkv
    try:
        return _resolve_sparse(plan, environment, inventory)
    except SparseSageError as sparse_exc:
        try:
            return _resolve_triton_sparse(
                plan,
                environment,
                inventory,
                sparse_exc,
            )
        except TritonSparseError as triton_exc:
            fallback_reason = (
                'Sparse Sage unavailable: %s; INT8 Triton unavailable: %s'
                % (sparse_exc, triton_exc)
            )
            try:
                return _resolve_fp8_flex(
                    plan,
                    environment,
                    inventory,
                    fallback_reason,
                    dense_attention,
                )
            except FP8FlexError as flex_exc:
                return (
                    ResolvedAttention(
                        requested=ATTENTION_SPARSE,
                        selected=dense_attention.selected,
                        backend=dense_attention.backend,
                        reason=(
                            '%s; FP8 FlexAttention unavailable: %s; %s'
                            % (
                                fallback_reason,
                                flex_exc,
                                dense_attention.reason,
                            )
                        ),
                        backend_kind=dense_attention.backend_kind,
                        projector=dense_attention.projector,
                        dense_resolution=dense_attention.dense_resolution,
                    ),
                    dense_qkv,
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
    v_layout,
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
        'v_layout': {
            'state': v_layout.state,
            'reason': v_layout.reason,
            'patched_blocks': int(v_layout.patched_blocks),
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
            'chunk_rows': getattr(attention.projector, 'chunk_rows', None),
            'producer_abi': (
                KITCHEN_PRODUCER_ABI
                if qkv.provider_id == QKV_DENSE_KITCHEN_CHUNKED
                else None
            ),
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
            'backend': environment.backend,
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

    attention, qkv = _resolve_attention(
        plan,
        model,
        inventory,
        environment,
    )

    patched = model.clone()
    attention_blocks = 0
    sparse_execution_selected = attention.backend_kind in (
        ATTENTION_SPARSE,
        ATTENTION_TRITON_SPARSE,
        ATTENTION_FP8_FLEX,
    )
    if sparse_execution_selected:
        if attention.backend_kind == ATTENTION_FP8_FLEX:
            _backend, attention_blocks = configure_backend(
                patched,
                attention.backend,
                projector=attention.projector,
                backend_fallback_to_dense=True,
            )
        else:
            _backend, attention_blocks = configure_backend(
                patched,
                attention.backend,
                projector=attention.projector,
            )
        if (
            attention.backend_kind == ATTENTION_FP8_FLEX
            and attention.dense_resolution is not None
        ):
            install_dense_attention(patched, attention.dense_resolution)
        v_layout = not_applicable_v_layout(
            '%s owns the main H3 forward' % attention.selected
        )
    elif plan.memory is not None:
        if qkv.provider_id == QKV_DENSE_KITCHEN_CHUNKED:
            _backend, attention_blocks = configure_backend(
                patched,
                attention.backend,
                projector=attention.projector,
                projector_fallback_to_original=True,
            )
            v_layout = not_applicable_v_layout(
                'chunked Kitchen QKV owns the main H3 forward'
            )
        else:
            v_layout = install_v_layout_compat(patched)
            attention_blocks = int(v_layout.patched_blocks)
        install_dense_attention(patched, attention.dense_resolution)
    else:
        v_layout = not_applicable_v_layout(
            'no H3 Memory Optimization request'
        )

    mlp, mlp_blocks = _install_mlp(patched, plan, inventory)
    runtime_installed = False
    if sparse_execution_selected:
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
        v_layout=v_layout,
    )
    logging.info(
        '%s armed: attention=%s v_layout=%s qkv=%s mlp=%s device=%s',
        LOG_PREFIX,
        attention.selected,
        v_layout.state,
        qkv.provider_id,
        mlp.provider_id,
        environment.device_name,
    )
    return patched
