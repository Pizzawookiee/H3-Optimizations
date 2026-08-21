'''Resolve validated QKV and MLP providers from checkpoint formats.'''

from dataclasses import dataclass

from ..plan import FUSED_QKV_OFF, MLP_MEMORY_AUTO, MLP_MEMORY_OFF

QKV_STANDARD = 'standard_h3_qkv'
QKV_DENSE_CONVROT_INT8 = 'convrot_int8_dense_sage'
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
    capability,
    triton_available,
    sparse_spec=None,
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
    if tuple(capability or ()) != (8, 9):
        return _standard_qkv('the current fused QKV providers require SM89')
    if not triton_available:
        return _standard_qkv('Triton is unavailable')

    if backend_kind == 'dense_sage_sm89':
        return QKVProviderResolution(
            QKV_DENSE_CONVROT_INT8,
            True,
            'ConvRot-256 TensorWise INT8 QKV into dense Sage carriers',
        )
    if backend_kind == 'sparse_sage':
        if sparse_spec is None:
            return _standard_qkv('Sparse Sage ABI was not resolved')
        if (
            tuple(getattr(sparse_spec, 'capability', ())) != (8, 9)
            or int(getattr(sparse_spec, 'q_tile', 0)) != 128
            or int(getattr(sparse_spec, 'kv_tile', 0)) != 64
        ):
            return _standard_qkv(
                'the selected Sparse Sage ABI is not SM89 128Q x 64KV'
            )
        return QKVProviderResolution(
            QKV_SPARSE_CONVROT_INT8,
            True,
            'ConvRot-256 TensorWise INT8 QKV into Sparse Sage carriers',
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
