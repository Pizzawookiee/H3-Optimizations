'''Resolve validated QKV and MLP providers from checkpoint formats.'''

from dataclasses import dataclass

from ..plan import FUSED_QKV_OFF, MLP_MEMORY_AUTO, MLP_MEMORY_OFF

QKV_STANDARD = 'standard_h3_qkv'
QKV_DENSE_KITCHEN_CHUNKED = 'chunked_kitchen_qkv'
QKV_SPARSE_CONVROT_INT8 = 'convrot_int8_sparse_sage'

MLP_OFF = 'off'
MLP_GENERIC_CHUNKED = 'generic_chunked_quantized'
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


def resolve_qkv_provider(
    inventory,
    *,
    request,
    backend_kind,
    triton_available=False,
    sparse_spec=None,
    kitchen_producer_available=False,
):
    if request == FUSED_QKV_OFF:
        return _standard_qkv('fused QKV was disabled')
    if not inventory.qkv:
        return _standard_qkv('the H3 model has no QKV projection inventory')
    if not inventory.homogeneous('qkv'):
        return _standard_qkv('H3 QKV layers use mixed weight formats')
    if not inventory.qkv_convrot_int8_256:
        labels = ', '.join(sorted(set(inventory.labels('qkv'))))
        return _standard_qkv(
            'no fused provider supports QKV format %s'
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


def resolve_mlp_provider(inventory, *, request):
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

    labels = sorted(
        set(inventory.labels('fc1')) | set(inventory.labels('fc2'))
    )
    return MLPProviderResolution(
        MLP_GENERIC_CHUNKED,
        'mlp_chunked_native',
        (
            'generic token chunking preserves the model linear formats: %s'
            % (', '.join(labels) or 'unknown')
        ),
    )
