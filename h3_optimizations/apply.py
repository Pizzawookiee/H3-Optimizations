'''Resolve and apply the complete H3 optimization plan.'''

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch

import comfy.model_management
import comfy.quant_ops

from .attention.sparse import (
    FP8FlexBackend,
    FP8FlexError,
    FrostBF16Backend,
    HybridSparseBackend,
    HybridSparseConfig,
    MODE_SAGE128,
    MODE_SAGE128_FUSED_QKV,
    SparseSageError,
    TritonSparseBackend,
    TritonSparseError,
    preflight_fp8_flex,
    preflight_frost_bf16,
    preflight_sparse_sage,
    preflight_triton_sparse,
)
from .attention.sparse.fused_qkv import (
    TRITON_AVAILABLE as SPARSE_TRITON_AVAILABLE,
)
from .attention.sparse.kitchen_sparse import (
    OUTPUT_NHD,
    PRODUCTION_KV_TILE as KITCHEN_KV_TILE,
    PRODUCTION_Q_TILE as KITCHEN_Q_TILE,
    SparseKitchenBackend,
    SparseKitchenError,
    preflight_sparse_kitchen,
)
from .dense_resolver import (
    ATTENTION_EXISTING_FULL_Q,
    ATTENTION_SAGE_PREFIX,
    ATTENTION_SAGE_SM89,
    install_dense_attention,
    preserve_dense_attention,
    resolve_current_dense_attention,
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
from .memory.final_layer import install as install_final_layer
from .memory.patch import install as install_memory_patch
from .model import get_h3_blocks, is_minimax_h3
from .patch import configure_backend
from .plan import (
    ATTENTION_EXISTING,
    DENSITY_FIXED,
    FUSED_QKV_AUTO,
    FUSED_QKV_FORCE_QUANT,
    FUSED_QKV_OFF,
    H3OptimizationPlan,
    PLAN_KEY,
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_FLEX,
    SPARSE_BACKEND_FROST,
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_SAGE,
    SPARSE_BACKEND_TRITON,
    STATUS_KEY,
    QKV_STREAMING_OFF,
    QKV_STREAMING_FORCED,
)
from .qkv.bf16 import (
    ChunkedBF16QKVProjector,
    StreamedDenseBF16QKVProjector,
)
from .qkv.formats import inspect_h3_linears
from .qkv.streamed import (
    PROJECTION_FORCE_BF16,
    PROJECTION_FORCE_FP8,
    PROJECTION_FORCE_INT8,
    PROJECTION_NATIVE,
)
from .qkv.providers import (
    MLP_OFF,
    MLP_PRESERVE_UPSTREAM,
    QKV_BF16_CHUNKED,
    QKV_DENSE_CONVROT_INT8,
    QKV_FORCE_BF16_CHUNKED,
    QKV_FORCE_BF16_STREAMED_KITCHEN,
    QKV_FORCE_CONVROT_INT8_CHUNKED,
    QKV_FORCE_CONVROT_INT8_FROST,
    QKV_FORCE_CONVROT_INT8_KITCHEN,
    QKV_FORCE_CONVROT_INT8_TRITON,
    QKV_FORCE_FP8_CHUNKED,
    QKV_FROST_STREAMED,
    QKV_DENSE_FP8_CHUNKED,
    QKV_DENSE_KITCHEN_CHUNKED,
    QKV_SPARSE_CONVROT_INT8,
    QKV_SPARSE_FP8_CHUNKED,
    QKV_STANDARD,
    QKV_STREAMED_BF16_KITCHEN,
    QKV_TRITON_SPARSE_CHUNKED,
    resolve_mlp_provider,
    resolve_qkv_provider,
)
from .runtime.context import (
    H3RuntimeSession,
    RUNTIME_SESSION_KEY,
    install_runtime_wrapper,
)
from .status import format_qkv_execution

LOG_PREFIX = '[H3 Optimizations]'
ATTENTION_SPARSE = 'sparse_sage'
ATTENTION_TRITON_SPARSE = 'triton_sparse_bf16'
ATTENTION_FP8_FLEX = 'flex_attention_fp8'
ATTENTION_FROST_BF16 = 'frost_bf16_sm89'
ATTENTION_KITCHEN_SPARSE = 'sparse_kitchen_int8'
SPARSE_EXECUTION_BACKENDS = (
    ATTENTION_SPARSE,
    ATTENTION_TRITON_SPARSE,
    ATTENTION_FP8_FLEX,
    ATTENTION_FROST_BF16,
    ATTENTION_KITCHEN_SPARSE,
)

_BOUNDED_QKV_PROVIDERS = (
    QKV_BF16_CHUNKED,
    QKV_FORCE_BF16_CHUNKED,
    QKV_FORCE_CONVROT_INT8_CHUNKED,
    QKV_FORCE_FP8_CHUNKED,
)


def _bounded_qkv_projector(qkv, chunk_rows=4096):
    return ChunkedBF16QKVProjector(
        chunk_rows=chunk_rows,
        force_weights_bf16=qkv.provider_id == QKV_FORCE_BF16_CHUNKED,
        force_weights_fp8=qkv.provider_id == QKV_FORCE_FP8_CHUNKED,
        force_weights_int8=(
            qkv.provider_id == QKV_FORCE_CONVROT_INT8_CHUNKED
        ),
    )

def _frost_qkv_projector(qkv, inventory, spec):
    from .qkv.projectors import FrostStreamedQKVProjector

    return FrostStreamedQKVProjector(
        spec,
        required=bool(qkv.fused),
        chunk_rows=4096,
        projection_mode=_streamed_projection_mode(qkv, inventory),
    )


def _force_out_proj_int8(plan, inventory):
    return bool(
        _qkv_request(plan) == FUSED_QKV_FORCE_QUANT
        and getattr(inventory, 'out_proj_plain_float', False)
    )


def _streamed_projection_mode(qkv, inventory):
    if qkv.provider_id == QKV_FORCE_BF16_CHUNKED:
        return PROJECTION_FORCE_BF16
    if qkv.provider_id in (
        QKV_FORCE_CONVROT_INT8_CHUNKED,
        QKV_FORCE_CONVROT_INT8_FROST,
        QKV_FORCE_CONVROT_INT8_TRITON,
    ):
        return PROJECTION_FORCE_INT8
    if (
        qkv.provider_id == QKV_SPARSE_FP8_CHUNKED
        and getattr(inventory, 'qkv_plain_float', False)
    ):
        return PROJECTION_FORCE_FP8
    return PROJECTION_NATIVE

@dataclass(frozen=True)
class ResolvedAttention:
    requested: str
    selected: str
    backend: object | None
    reason: str
    backend_kind: str
    projector: object | None = None
    dense_resolution: object | None = None


# Fallbacks with a measured performance penalty relative to Kitchen INT8.
# Unlisted fallbacks remain visible without making an unsupported speed claim.
_SLOW_SPARSE_FALLBACKS = {
    ATTENTION_TRITON_SPARSE: 'BF16 Triton attention is slower than Kitchen INT8 in measured H3 runs',
    ATTENTION_FP8_FLEX: 'FP8 FlexAttention is slower than Kitchen INT8 in measured H3 runs',
}


def _warn_about_slow_paths(attention, qkv):
    """Say loudly when a fast path was wanted and something slower was used.

    Only fires when the resolution actually degraded. Choosing a backend
    explicitly is not a fallback and stays quiet.
    """
    requested_sparse = attention.requested in (
        ATTENTION_SPARSE,
        ATTENTION_KITCHEN_SPARSE,
        ATTENTION_FROST_BF16,
        ATTENTION_TRITON_SPARSE,
        ATTENTION_FP8_FLEX,
    )
    if requested_sparse and attention.selected != attention.requested:
        cost = _SLOW_SPARSE_FALLBACKS.get(attention.selected)
        if cost is None:
            logging.warning(
                '%s SPARSE ATTENTION FELL BACK to %s. Reason: %s',
                LOG_PREFIX,
                attention.selected,
                attention.reason or 'unknown',
            )
        else:
            logging.warning(
                '%s SPARSE ATTENTION FELL BACK to %s. %s. Reason: %s',
                LOG_PREFIX,
                attention.selected,
                cost,
                attention.reason or 'unknown',
            )

    # The chunked Comfy Kitchen QKV producer is the fast QKV path. When its
    # Kitchen-side API is missing the pack silently projects the slow way,
    # which is what made a missing dependency look like a working install.
    if qkv.provider_id == QKV_STANDARD and 'Kitchen' in (qkv.reason or ''):
        logging.warning(
            '%s FUSED QKV IS NOT RUNNING - falling back to standard projection, '
            'which is roughly half the speed. Reason: %s',
            LOG_PREFIX,
            qkv.reason,
        )


def _qkv_request(plan):
    if plan.memory is not None:
        if plan.memory.qkv_streaming == QKV_STREAMING_OFF:
            return FUSED_QKV_OFF
        return plan.memory.fused_qkv
    if plan.sparse is not None:
        return FUSED_QKV_AUTO
    return FUSED_QKV_OFF


def _fp8_execution_available(environment):
    if not bool(getattr(environment, 'cuda_available', False)):
        return False
    capability = getattr(environment, 'capability', None)
    if capability is None or tuple(capability) < (8, 9):
        return False
    if not bool(getattr(comfy.quant_ops, '_CK_AVAILABLE', False)):
        return False
    index = getattr(environment, 'device_index', None)
    device = torch.device('cuda', index) if index is not None else torch.device('cuda')
    try:
        return bool(comfy.model_management.supports_fp8_compute(device))
    except (AttributeError, RuntimeError, TypeError):
        return False


def _sparse_config_kwargs(plan):
    sparse = plan.sparse
    return {
        'video_budget': float(sparse.video_budget),
        'density_mode': DENSITY_FIXED,
        'denser_early_late_steps': bool(sparse.denser_early_late_steps),
        'early_steps': sparse.early_steps,
        'early_kv': sparse.early_kv,
        'late_steps': sparse.late_steps,
        'late_kv': sparse.late_kv,
        'layer_video_budgets': sparse.layer_video_budgets,
        'strict': True,
    }


def describe_memory_options(attention):
    """The opt-in memory behaviour actually installed, short enough to log.

    Without this the plan summary looks identical whether or not the memory
    defaults are in force, which is exactly the question someone asks after
    pulling: did I get the new code? Report the behaviour, not the version.
    """
    backend = getattr(attention, 'backend', None)
    projector = getattr(attention, 'projector', None)
    installed = []
    if getattr(backend, 'output_layout', 'hnd') == 'nhd':
        installed.append('seq_major_out')
    if getattr(backend, 'release_carrier_before_out_proj', False):
        installed.append('early_carrier_release')
    if getattr(backend, 'stream_output', False):
        installed.append('streamed_out')
    if getattr(projector, 'strided_qk_input', False):
        installed.append('strided_qk')
    if getattr(projector, 'streamed_q', False):
        installed.append('streamed_q')
    if getattr(projector, 'consumer_native_carrier', False):
        installed.append('consumer_native_carrier')
    score_chunk = getattr(
        getattr(backend, 'router', None), 'score_chunk_tiles', None
    )
    if score_chunk:
        installed.append('score_chunk%d' % int(score_chunk))
    return '+'.join(installed) if installed else 'baseline'


def _resolve_dense(plan, model, inventory, environment=None):
    memory = plan.memory
    dense = (
        preserve_dense_attention('no memory optimization requested')
        if memory is None
        else (
            resolve_current_dense_attention(model, environment)
            if memory.attention == ATTENTION_EXISTING
            else resolve_dense_attention(model)
        )
    )
    dense_sage = dense.backend_kind.startswith(ATTENTION_SAGE_PREFIX)
    dense_carrier_available = False
    if dense_sage:
        requires_h3_triton = (
            dense.backend_kind == ATTENTION_SAGE_SM89
            or bool(getattr(dense.backend, 'requires_h3_triton', False))
        )
        if requires_h3_triton:
            from .attention.triton_i64 import TRITON_AVAILABLE

            dense_carrier_available = bool(TRITON_AVAILABLE)
        else:
            dense_carrier_available = True
    qkv = resolve_qkv_provider(
        inventory,
        request=_qkv_request(plan),
        backend_kind=dense.backend_kind,
        kitchen_producer_available=producer_api_available(
            device=getattr(environment, 'device_index', None)
        ),
        triton_available=dense_carrier_available,
        memory_optimize=memory is not None,
        fp8_available=_fp8_execution_available(environment),
    )
    backend = None
    projector = None
    if (
        dense_sage
        and qkv.provider_id != QKV_STANDARD
    ):
        from .dense_streamed_sage import (
            StreamedDenseSageBackend,
            StreamedDenseSageQKVProjector,
        )

        backend = StreamedDenseSageBackend(dense.backend)
        projector = StreamedDenseSageQKVProjector(
            dense.backend,
            chunk_rows=memory.chunk_rows,
            projection_mode=_streamed_projection_mode(qkv, inventory),
        )
    elif qkv.provider_id in _BOUNDED_QKV_PROVIDERS:
        backend = ChunkedKitchenAttentionBackend()
        if (
            dense.backend_kind != ATTENTION_EXISTING_FULL_Q
            and dense.backend_kind != 'sage'
        ) or memory.qkv_streaming == QKV_STREAMING_FORCED:
            projector = StreamedDenseBF16QKVProjector(
                chunk_rows=memory.chunk_rows,
                projection_mode=_streamed_projection_mode(qkv, inventory),
            )
        else:
            projector = _bounded_qkv_projector(
                qkv,
                chunk_rows=memory.chunk_rows,
            )
    elif qkv.provider_id in (
        QKV_DENSE_KITCHEN_CHUNKED,
        QKV_DENSE_FP8_CHUNKED,
        QKV_FORCE_CONVROT_INT8_KITCHEN,
        QKV_FORCE_BF16_STREAMED_KITCHEN,
    ):
        backend = ChunkedKitchenAttentionBackend(
            stream_output=True,
            query_chunk_rows=memory.chunk_rows,
        )
        # The chunk quantizer is handed the same strided Q/K views here as on
        # the sparse path, and takes them through the same guarded predicate.
        projector = ChunkedKitchenQKVProjector(
            chunk_rows=memory.chunk_rows,
            force_weights_bf16=(
                qkv.provider_id == QKV_FORCE_BF16_STREAMED_KITCHEN
            ),
            fp8_projection=qkv.provider_id == QKV_DENSE_FP8_CHUNKED,
            convrot_int8_projection=(
                qkv.provider_id == QKV_FORCE_CONVROT_INT8_KITCHEN
            ),
            strided_qk_input=True,
            stream_output=True,
            streamed_q=True,
        )
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
        request=_qkv_request(plan),
        backend_kind=ATTENTION_SPARSE,
        triton_available=bool(SPARSE_TRITON_AVAILABLE),
        sparse_spec=kernel_spec,
        memory_optimize=plan.memory is not None,
        fp8_available=_fp8_execution_available(environment),
    )
    use_projected = qkv.provider_id in (
        QKV_FORCE_BF16_CHUNKED,
        QKV_SPARSE_CONVROT_INT8,
        QKV_SPARSE_FP8_CHUNKED,
    )
    config = HybridSparseConfig(
        mode=MODE_SAGE128_FUSED_QKV if use_projected else MODE_SAGE128,
        **_sparse_config_kwargs(plan),
    )
    projector = None
    if qkv.provider_id in (
        QKV_FORCE_BF16_CHUNKED,
        QKV_SPARSE_CONVROT_INT8,
        QKV_SPARSE_FP8_CHUNKED,
    ):
        from .qkv.projectors import SparseFusedQKVProjector

        projector = SparseFusedQKVProjector(
            kernel_spec,
            required=bool(qkv.fused),
            chunk_rows=4096,
            projection_mode=_streamed_projection_mode(qkv, inventory),
        )
    elif qkv.provider_id in _BOUNDED_QKV_PROVIDERS:
        projector = _bounded_qkv_projector(qkv)
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


def _resolve_frost_bf16(plan, environment, inventory):
    spec = preflight_frost_bf16(
        cuda_available=lambda: environment.cuda_available,
        capability_getter=lambda: environment.capability,
    )
    qkv = resolve_qkv_provider(
        inventory,
        request=_qkv_request(plan),
        backend_kind=ATTENTION_FROST_BF16,
        memory_optimize=plan.memory is not None,
        fp8_available=_fp8_execution_available(environment),
    )
    projector = None
    if qkv.provider_id in (
        QKV_FROST_STREAMED,
        QKV_FORCE_BF16_CHUNKED,
        QKV_FORCE_CONVROT_INT8_FROST,
    ):
        projector = _frost_qkv_projector(qkv, inventory, spec)
    config = HybridSparseConfig(
        mode=MODE_SAGE128,
        **_sparse_config_kwargs(plan),
    )
    backend = FrostBF16Backend(
        config,
        spec=spec,
        projector=projector,
    )
    return (
        ResolvedAttention(
            requested=ATTENTION_FROST_BF16,
            selected=ATTENTION_FROST_BF16,
            backend=backend,
            reason='explicit FROST BF16 SM89 64Q x 64KV sparse attention',
            backend_kind=ATTENTION_FROST_BF16,
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
        request=_qkv_request(plan),
        backend_kind=ATTENTION_FP8_FLEX,
        memory_optimize=plan.memory is not None,
        fp8_available=_fp8_execution_available(environment),
    )
    config = HybridSparseConfig(
        mode=MODE_SAGE128,
        **_sparse_config_kwargs(plan),
    )
    backend = FP8FlexBackend(config, spec=spec)
    projector = (
        _bounded_qkv_projector(qkv)
        if qkv.provider_id in _BOUNDED_QKV_PROVIDERS
        else None
    )
    if fallback_reason is None:
        reason = 'explicit FP8 FlexAttention 64Q x 64KV selection'
        dense_resolution = None
    else:
        reason = '%s; using FP8 FlexAttention 64Q x 64KV' % fallback_reason
        dense_resolution = dense_attention.dense_resolution
    return (
        ResolvedAttention(
            requested=(
                ATTENTION_FP8_FLEX
                if fallback_reason is None
                else ATTENTION_SPARSE
            ),
            selected=ATTENTION_FP8_FLEX,
            backend=backend,
            reason=reason,
            backend_kind=ATTENTION_FP8_FLEX,
            projector=projector,
            dense_resolution=dense_resolution,
        ),
        qkv,
    )


def _resolve_triton_sparse(plan, environment, inventory, fallback_reason):
    spec = preflight_triton_sparse(
        cuda_available=lambda: environment.cuda_available,
        capability_getter=lambda: environment.capability,
    )
    qkv = resolve_qkv_provider(
        inventory,
        request=_qkv_request(plan),
        backend_kind=ATTENTION_TRITON_SPARSE,
        triton_available=True,
        memory_optimize=plan.memory is not None,
        fp8_available=_fp8_execution_available(environment),
    )
    config = HybridSparseConfig(
        mode=MODE_SAGE128,
        **_sparse_config_kwargs(plan),
    )
    projector = None
    if qkv.provider_id in (
        QKV_BF16_CHUNKED,
        QKV_FORCE_BF16_CHUNKED,
        QKV_TRITON_SPARSE_CHUNKED,
        QKV_FORCE_CONVROT_INT8_TRITON,
    ):
        from .qkv.projectors import TritonSparseQKVProjector

        projector = TritonSparseQKVProjector(
            required=bool(qkv.fused),
            chunk_rows=4096,
            projection_mode=_streamed_projection_mode(qkv, inventory),
        )
    elif qkv.provider_id in _BOUNDED_QKV_PROVIDERS:
        projector = _bounded_qkv_projector(qkv)
    backend = TritonSparseBackend(
        config,
        spec=spec,
        projector=projector,
    )
    reason = (
        'explicit BF16 Triton 64Q x 64KV sparse attention selection'
        if fallback_reason is None
        else '%s; using BF16 Triton 64Q x 64KV sparse attention'
        % fallback_reason
    )
    return (
        ResolvedAttention(
            requested=(
                ATTENTION_TRITON_SPARSE
                if fallback_reason is None
                else ATTENTION_SPARSE
            ),
            selected=ATTENTION_TRITON_SPARSE,
            backend=backend,
            reason=reason,
            backend_kind=ATTENTION_TRITON_SPARSE,
            projector=projector,
        ),
        qkv,
    )


def _resolve_kitchen_sparse(plan, environment, inventory):
    """Explicit Kitchen block-sparse INT8, with no Sparge anywhere in it."""
    kitchen = preflight_sparse_kitchen(
        cuda_available=lambda: environment.cuda_available,
        capability_getter=lambda: environment.capability,
        q_tile=KITCHEN_Q_TILE,
        kv_tile=KITCHEN_KV_TILE,
    )
    qkv = resolve_qkv_provider(
        inventory,
        request=_qkv_request(plan),
        backend_kind=ATTENTION_KITCHEN_SPARSE,
        kitchen_producer_available=producer_api_available(
            device=getattr(environment, 'device_index', None),
        ),
        memory_optimize=plan.memory is not None,
        fp8_available=_fp8_execution_available(environment),
    )
    use_projected = qkv.provider_id in (
        QKV_DENSE_KITCHEN_CHUNKED,
        QKV_STREAMED_BF16_KITCHEN,
        QKV_FORCE_CONVROT_INT8_KITCHEN,
        QKV_FORCE_BF16_STREAMED_KITCHEN,
    )
    config = HybridSparseConfig(
        mode=MODE_SAGE128_FUSED_QKV if use_projected else MODE_SAGE128,
        **_sparse_config_kwargs(plan),
    )
    if qkv.provider_id in _BOUNDED_QKV_PROVIDERS:
        projector = _bounded_qkv_projector(qkv)
    elif use_projected:
        projector = ChunkedKitchenQKVProjector(
            force_weights_bf16=(
                qkv.provider_id == QKV_FORCE_BF16_STREAMED_KITCHEN
            ),
            routing_summaries=True,
            q_tile=KITCHEN_Q_TILE,
            kv_tile=KITCHEN_KV_TILE,
            strided_qk_input=True,
            stream_output=True,
            convrot_int8_projection=(
                qkv.provider_id == QKV_FORCE_CONVROT_INT8_KITCHEN
            ),
        )
    else:
        projector = None
    # Reuse the disposable normalized input as the output buffer and project
    # query slices immediately. The non-projected fallback still uses the
    # sequence-major output and early carrier release below.
    backend = SparseKitchenBackend(
        config,
        kitchen=kitchen,
        projector=projector,
        q_tile=KITCHEN_Q_TILE,
        kv_tile=KITCHEN_KV_TILE,
        output_layout=OUTPUT_NHD,
        release_carrier_before_out_proj=True,
        stream_output=use_projected,
    )
    return (
        ResolvedAttention(
            requested=ATTENTION_KITCHEN_SPARSE,
            selected=ATTENTION_KITCHEN_SPARSE,
            backend=backend,
            reason='native Kitchen INT8 64Q x 64KV sparse attention',
            backend_kind=ATTENTION_KITCHEN_SPARSE,
            projector=projector,
        ),
        qkv,
    )


def _resolve_attention(plan, model, inventory, environment):
    if plan.sparse is not None:
        backend_request = plan.sparse.backend
        if backend_request == SPARSE_BACKEND_SAGE:
            return _resolve_sparse(plan, environment, inventory)
        if backend_request == SPARSE_BACKEND_TRITON:
            return _resolve_triton_sparse(plan, environment, inventory, None)
        if backend_request == SPARSE_BACKEND_KITCHEN:
            return _resolve_kitchen_sparse(plan, environment, inventory)
        if backend_request == SPARSE_BACKEND_FLEX:
            return _resolve_fp8_flex(
                plan,
                environment,
                inventory,
                None,
                None,
            )
        if backend_request == SPARSE_BACKEND_FROST:
            return _resolve_frost_bf16(plan, environment, inventory)
        if backend_request != SPARSE_BACKEND_AUTO:
            raise ValueError('unknown sparse backend request %r' % backend_request)

    dense_attention, dense_qkv = _resolve_dense(
        plan,
        model,
        inventory,
        environment,
    )
    if plan.sparse is None:
        return dense_attention, dense_qkv

    try:
        return _resolve_kitchen_sparse(plan, environment, inventory)
    except SparseKitchenError as kitchen_exc:
        try:
            return _resolve_sparse(plan, environment, inventory)
        except SparseSageError as sparse_exc:
            fallback_reason = (
                'Kitchen INT8 unavailable: %s; Sparse Sage unavailable: %s'
                % (kitchen_exc, sparse_exc)
            )
            try:
                return _resolve_triton_sparse(
                    plan,
                    environment,
                    inventory,
                    fallback_reason,
                )
            except TritonSparseError as triton_exc:
                fallback_reason = (
                    '%s; BF16 Triton unavailable: %s'
                    % (fallback_reason, triton_exc)
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


def _install_mlp(model_patcher, plan, inventory, environment):
    memory = plan.memory
    if memory is None:
        return resolve_mlp_provider(inventory, request='off'), 0
    install_final_layer(model_patcher, int(memory.chunk_rows))
    resolution = resolve_mlp_provider(
        inventory,
        request=memory.mlp_memory,
        fp8_available=_fp8_execution_available(environment),
    )
    if resolution.provider_id in (MLP_OFF, MLP_PRESERVE_UPSTREAM):
        return resolution, 0
    config = ActivationMemoryConfig(
        mode=resolution.activation_mode,
        chunk_rows=int(memory.chunk_rows),
        strict=bool(memory.mlp_strict),
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
        'out_proj': list(inventory.labels('out_proj')),
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
            'backend_details': (
                attention.backend.as_status()
                if callable(getattr(attention.backend, 'as_status', None))
                else None
            ),
        },
        'sparse': (
            None
            if plan.sparse is None
            else {
                'backend': plan.sparse.backend,
                'video_budget': float(plan.sparse.video_budget),
                'denser_early_late_steps': bool(
                    plan.sparse.denser_early_late_steps
                ),
                'early_steps': plan.sparse.early_steps,
                'early_kv': plan.sparse.early_kv,
                'late_steps': plan.sparse.late_steps,
                'late_kv': plan.sparse.late_kv,
                'layer_video_budgets': (
                    None
                    if plan.sparse.layer_video_budgets is None
                    else list(plan.sparse.layer_video_budgets)
                ),
            }
        ),
        'fused_qkv': {
            'requested': _qkv_request(plan),
            'provider': qkv.provider_id,
            'fused': bool(qkv.fused),
            'reason': qkv.reason,
            'projector': getattr(attention.projector, 'name', None),
            'chunk_rows': getattr(attention.projector, 'chunk_rows', None),
            'streamed_q': bool(
                getattr(attention.projector, 'streamed_q', False)
            ),
            'strided_qk_input': getattr(
                attention.projector, 'strided_qk_input', None
            ),
            'output_streamed': bool(
                getattr(attention.projector, 'stream_output', False)
                or getattr(attention.backend, 'stream_output', False)
            ),
            'producer_abi': (
                KITCHEN_PRODUCER_ABI
                if qkv.provider_id in (
                    QKV_DENSE_KITCHEN_CHUNKED,
                    QKV_DENSE_FP8_CHUNKED,
                    QKV_STREAMED_BF16_KITCHEN,
                    QKV_FORCE_CONVROT_INT8_KITCHEN,
                    QKV_FORCE_BF16_STREAMED_KITCHEN,
                )
                else None
            ),
            'out_proj_runtime_convrot_int8': _force_out_proj_int8(
                plan,
                inventory,
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
        'final_layer': (
            None
            if plan.memory is None
            else {
                'chunked': True,
                'chunk_rows': int(plan.memory.chunk_rows),
            }
        ),
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
    attention, qkv = _resolve_attention(plan, model, inventory, environment)
    previous_options = model.model_options.get('transformer_options', {})
    previous_status = previous_options.get(STATUS_KEY)
    previous_attention = previous_options.get(
        'h3_optimizations_attention_backend'
    )
    if previous_attention is None and previous_status is not None:
        previous_attention = previous_status['attention']['selected']

    patched = model.clone()
    attention_blocks = 0
    out_proj_kwargs = (
        {'force_out_proj_int8': True}
        if _force_out_proj_int8(plan, inventory)
        else {}
    )
    sparse_execution_selected = attention.backend_kind in SPARSE_EXECUTION_BACKENDS
    flex_dense_fallback = (
        attention.backend_kind == ATTENTION_FP8_FLEX
        and plan.sparse is not None
        and plan.sparse.backend == SPARSE_BACKEND_AUTO
    )
    if sparse_execution_selected:
        if attention.backend_kind == ATTENTION_FP8_FLEX:
            _backend, attention_blocks = configure_backend(
                patched,
                attention.backend,
                projector=attention.projector,
                backend_fallback_to_dense=flex_dense_fallback,
                **out_proj_kwargs,
            )
        else:
            _backend, attention_blocks = configure_backend(
                patched,
                attention.backend,
                projector=attention.projector,
                **out_proj_kwargs,
            )
        if flex_dense_fallback and attention.dense_resolution is not None:
            install_dense_attention(patched, attention.dense_resolution)
    elif plan.memory is not None:
        if qkv.provider_id in (
            QKV_DENSE_CONVROT_INT8,
            QKV_BF16_CHUNKED,
            QKV_FORCE_BF16_CHUNKED,
            QKV_FORCE_CONVROT_INT8_CHUNKED,
            QKV_FORCE_CONVROT_INT8_KITCHEN,
            QKV_FORCE_BF16_STREAMED_KITCHEN,
            QKV_FORCE_FP8_CHUNKED,
            QKV_DENSE_KITCHEN_CHUNKED,
            QKV_DENSE_FP8_CHUNKED,
        ):
            _backend, attention_blocks = configure_backend(
                patched,
                attention.backend,
                projector=attention.projector,
                projector_fallback_to_original=True,
                **out_proj_kwargs,
            )
        install_dense_attention(patched, attention.dense_resolution)

    mlp, mlp_blocks = _install_mlp(
        patched,
        plan,
        inventory,
        environment,
    )
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
    )
    options[STATUS_KEY]['memory_options'] = describe_memory_options(attention)
    _warn_about_slow_paths(attention, qkv)
    qkv_labels = inventory.labels('qkv')
    features = '+'.join(
        name
        for name, request in (
            ('memory', plan.memory),
            ('sparse', plan.sparse),
        )
        if request is not None
    ) or 'none'
    attention_name = getattr(
        attention.backend,
        'name',
        attention.selected,
    )
    replacement = (
        ' replaces_attention=%s' % previous_attention
        if previous_attention is not None
        and previous_attention != attention_name
        else ''
    )
    logging.info(
        '%s applied plan: features=%s attention=%s%s qkv="%s" qkv_provider=%s qkv_weights=%s qkv_layers=%d out_proj=%s mlp=%s memory=%s device=%s',
        LOG_PREFIX,
        features,
        attention_name,
        replacement,
        format_qkv_execution(options[STATUS_KEY]),
        qkv.provider_id,
        ','.join(sorted(set(qkv_labels))) or 'unknown',
        len(qkv_labels),
        (
            'runtime_convrot_int8'
            if options[STATUS_KEY]['fused_qkv'][
                'out_proj_runtime_convrot_int8'
            ]
            else 'checkpoint_native'
        ),
        mlp.provider_id,
        describe_memory_options(attention),
        environment.device_name,
    )
    return patched
