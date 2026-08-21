'''Resolve validated QKV and MLP providers from checkpoint formats.'''

from dataclasses import dataclass

from ..plan import FUSED_QKV_OFF, MLP_MEMORY_AUTO, MLP_MEMORY_OFF

QKV_STANDARD = 'standard_h3_qkv'
QKV_DENSE_KITCHEN_CHUNKED = 'chunked_kitchen_qkv'
QKV_DENSE_FP8_CHUNKED = 'chunked_fp8_kitchen_qkv'
QKV_SPARSE_CONVROT_INT8 = 'convrot_int8_sparse_sage'
QKV_SPARSE_FP8_CHUNKED = 'chunked_fp8_sparse_sage'
QKV_TRITON_SPARSE_CHUNKED = 'chunked_triton_int8_sparse'

MLP_OFF = 'off'
MLP_PRESERVE_UPSTREAM = 'preserve_upstream_mlp'
MLP_FLOAT_CHUNKED = 'float_chunked'
MLP_FP8_CHUNKED = 'fp8_chunked'
MLP_CONVROT_INT8_TWO_SLICE = 'convrot_int8_two_slice'


@dataclass(frozen=True)
class QKVProviderResolution:
    provider_id: str
    fused: bool
    reason: str


@dataclass(frozen=True)
class MLPProviderResolution:
    provider_id: str
    activation_mode: str
    reason: str


def _standard_qkv(reason):
    return QKVProviderResolution(QKV_STANDARD, False, reason)


def _sparse_contract_ok(sparse_spec):
    from ..attention.sparse.fused_qkv import sparse_fused_qkv_contract_mismatch

    return sparse_fused_qkv_contract_mismatch(sparse_spec) is None


def resolve_qkv_provider(
    inventory,
    *,
    request,
    backend_kind,
    triton_available=False,
    sparse_spec=None,
    kitchen_producer_available=False,
    memory_optimize=False,
    fp8_available=False,
):
    if request == FUSED_QKV_OFF:
        return _standard_qkv('QKV projection optimization was disabled')
    if not inventory.qkv:
        return _standard_qkv('the H3 model has no QKV projection inventory')
    if not inventory.homogeneous('qkv'):
        return _standard_qkv('H3 QKV layers use mixed weight formats')

    fp8_memory_candidate = (
        memory_optimize
        and fp8_available
        and (inventory.qkv_fp8 or inventory.qkv_plain_float)
    )
    if fp8_memory_candidate:
        source = 'checkpoint-native FP8' if inventory.qkv_fp8 else 'BF16/FP16 converted to FP8 E4M3'
        if backend_kind == 'comfy_kitchen_int8':
            if not kitchen_producer_available:
                return _standard_qkv('Comfy Kitchen external producer API is unavailable')
            return QKVProviderResolution(
                QKV_DENSE_FP8_CHUNKED,
                False,
                '%s QKV uses held FP8 projection into chunked Kitchen carriers' % source,
            )
        if backend_kind == 'sparse_sage':
            if not triton_available:
                return _standard_qkv('Triton is unavailable')
            if not _sparse_contract_ok(sparse_spec):
                return _standard_qkv('the selected Sparse Sage ABI cannot consume chunked projected QKV')
            return QKVProviderResolution(
                QKV_SPARSE_FP8_CHUNKED,
                True,
                '%s QKV uses held FP8 projection into Sparse Sage carriers' % source,
            )

    if not inventory.qkv_convrot_int8_256:
        labels = ', '.join(sorted(set(inventory.labels('qkv'))))
        return _standard_qkv(
            'native checkpoint projection preserves QKV format %s'
            % (labels or 'unknown')
        )
    if backend_kind == 'comfy_kitchen_int8':
        if not kitchen_producer_available:
            return _standard_qkv(
                'Comfy Kitchen external producer API is unavailable'
            )
        return QKVProviderResolution(
            QKV_DENSE_KITCHEN_CHUNKED,
            False,
            '4K ConvRot QKV chunks into Comfy Kitchen INT8 carriers',
        )
    if backend_kind == 'triton_sparse_int8':
        if not triton_available:
            return _standard_qkv('Triton is unavailable')
        return QKVProviderResolution(
            QKV_TRITON_SPARSE_CHUNKED,
            True,
            '4K ConvRot QKV chunks into Triton INT8 sparse carriers',
        )
    if backend_kind == 'sparse_sage':
        if not triton_available:
            return _standard_qkv('Triton is unavailable')
        from ..attention.sparse.fused_qkv import (
            sparse_fused_qkv_contract_mismatch,
        )

        mismatch = sparse_fused_qkv_contract_mismatch(sparse_spec)
        if mismatch is not None:
            return _standard_qkv(
                'the selected Sparse Sage ABI cannot consume fused QKV: %s'
                % mismatch
            )
        return QKVProviderResolution(
            QKV_SPARSE_CONVROT_INT8,
            True,
            '4K ConvRot QKV chunks into Sparse Sage-native carriers',
        )
    return _standard_qkv(
        'the resolved attention backend has no fused-QKV consumer'
    )


def _convrot_compatible(inventory):
    return (
        bool(inventory.fc1)
        and bool(inventory.fc2)
        and inventory.homogeneous('fc1')
        and inventory.homogeneous('fc2')
        and inventory.mlp_convrot_int8_256
    )


def resolve_mlp_provider(inventory, *, request, fp8_available=False):
    if request == MLP_MEMORY_OFF:
        return MLPProviderResolution(
            MLP_OFF,
            'off',
            'MLP memory optimization was disabled',
        )
    if request != MLP_MEMORY_AUTO:
        raise ValueError('unknown MLP memory request %r' % request)
    if not inventory.fc1 or not inventory.fc2:
        return MLPProviderResolution(
            MLP_OFF,
            'off',
            'the H3 model has no MLP inventory',
        )
    if _convrot_compatible(inventory):
        return MLPProviderResolution(
            MLP_CONVROT_INT8_TWO_SLICE,
            'mlp_chunked_convrot_2slice',
            (
                'ConvRot-256 TensorWise INT8 MLP uses token chunks and two '
                'feature slices'
            ),
        )
    if inventory.mlp_fp8:
        if fp8_available:
            return MLPProviderResolution(
                MLP_FP8_CHUNKED,
                'mlp_chunked_fp8',
                'checkpoint-native FP8 MLP uses held weights and bounded token chunks',
            )
        return MLPProviderResolution(
            MLP_PRESERVE_UPSTREAM,
            'off',
            'FP8 checkpoint detected but accelerated FP8 execution is unavailable',
        )
    if inventory.mlp_plain_float:
        if fp8_available:
            return MLPProviderResolution(
                MLP_FP8_CHUNKED,
                'mlp_chunked_fp8',
                'floating H3 MLP weights are converted to FP8 E4M3 for Memory auto',
            )
        return MLPProviderResolution(
            MLP_FLOAT_CHUNKED,
            'mlp_chunked_native',
            'floating H3 MLP uses bounded held-weight token chunking',
        )

    labels = sorted(
        set(inventory.labels('fc1')) | set(inventory.labels('fc2'))
    )
    return MLPProviderResolution(
        MLP_PRESERVE_UPSTREAM,
        'off',
        (
            'no H3 memory provider supports MLP format %s; preserving upstream Comfy execution'
            % (', '.join(labels) or 'unknown')
        ),
    )
