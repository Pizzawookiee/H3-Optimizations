'''Resolve validated QKV and MLP providers from checkpoint formats.'''

from dataclasses import dataclass

from ..plan import (
    FUSED_QKV_OFF,
    FUSED_QKV_PRESERVE_BF16,
    FUSED_QKV_REQUIRED,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_BF16,
    MLP_MEMORY_FORCE_QUANT,
    MLP_MEMORY_LEGACY_BF16,
    MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
    MLP_MEMORY_LEGACY_NATIVE,
    MLP_MEMORY_OFF,
    MLP_MEMORY_PRESERVE,
)

QKV_STANDARD = 'standard_h3_qkv'
QKV_BF16_CHUNKED = 'chunked_bf16_qkv'
QKV_FORCE_BF16_CHUNKED = 'force_bf16_qkv'
QKV_FORCE_BF16_STREAMED_KITCHEN = 'force_bf16_streamed_kitchen_qkv'
QKV_FORCE_FP8_CHUNKED = 'force_fp8_qkv'
QKV_FORCE_QUANT_CHUNKED = QKV_FORCE_FP8_CHUNKED
QKV_FORCE_CONVROT_INT8_CHUNKED = 'force_convrot_int8_qkv'
QKV_FORCE_CONVROT_INT8_KITCHEN = 'force_convrot_int8_kitchen_qkv'
QKV_FORCE_CONVROT_INT8_TRITON = 'force_convrot_int8_triton_qkv'
QKV_DENSE_CONVROT_INT8 = 'convrot_int8_dense_sage'
QKV_DENSE_KITCHEN_CHUNKED = 'chunked_kitchen_qkv'
QKV_STREAMED_BF16_KITCHEN = 'streamed_bf16_kitchen_qkv'
QKV_DENSE_FP8_CHUNKED = 'chunked_fp8_kitchen_qkv'
QKV_DENSE_W4A8_CHUNKED = QKV_DENSE_KITCHEN_CHUNKED
QKV_SPARSE_CONVROT_INT8 = 'convrot_int8_sparse_sage'
QKV_SPARSE_FP8_CHUNKED = 'chunked_fp8_sparse_sage'
QKV_SPARSE_W4A8_CHUNKED = QKV_SPARSE_FP8_CHUNKED
QKV_TRITON_SPARSE_CHUNKED = 'chunked_triton_bf16_sparse'
QKV_TRITON_W4A8_CHUNKED = QKV_TRITON_SPARSE_CHUNKED

MLP_OFF = 'off'
MLP_PRESERVE_UPSTREAM = 'preserve_upstream_mlp'
MLP_FLOAT_CHUNKED = 'float_chunked'
MLP_FP8_CHUNKED = 'fp8_chunked'
MLP_RUNTIME_CONVROT_INT8_CHUNKED = 'runtime_convrot_int8_chunked'
MLP_W4A8_CHUNKED = 'w4a8_chunked'
MLP_CONVROT_INT8_TWO_SLICE = 'convrot_int8_two_slice'

# These backends already consume the package's Kitchen INT8 carrier. A native
# BF16 checkpoint can therefore stream projection/norm/RoPE slabs straight
# into that existing carrier instead of materializing complete BF16 Q/K/V.
_BF16_KITCHEN_CARRIER_CONSUMERS = {
    'comfy_kitchen_int8',
    'sparse_kitchen_int8',
}


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


def _required_or_standard(request, reason):
    if request == FUSED_QKV_REQUIRED:
        raise RuntimeError('required fused QKV is unavailable: %s' % reason)
    return _standard_qkv(reason)


def standard_qkv_provider(request, reason):
    """The unfused QKV path, for a backend with no fused producer.

    Raises when the plan explicitly required fused QKV, so a backend that
    cannot consume it fails loudly instead of silently downgrading.
    """
    return _required_or_standard(request, reason)


def _sparse_contract_ok(sparse_spec):
    from ..attention.sparse.fused_qkv import sparse_fused_qkv_contract_mismatch

    return sparse_fused_qkv_contract_mismatch(sparse_spec) is None


def _qkv_is_native_bf16(inventory):
    if not inventory.qkv:
        return False
    for item in inventory.qkv:
        dtype = str(item.logical_dtype).lower()
        if not item.plain_float or not (
            'bfloat16' in dtype or 'bf16' in dtype
        ):
            return False
    return True


def _resolve_preserve_precision_qkv(
    inventory, backend_kind, kitchen_producer_available
):
    if (
        inventory.qkv_convrot_int8_256
        and backend_kind in ('comfy_kitchen_int8', 'sparse_kitchen_int8')
    ):
        if not kitchen_producer_available:
            return _standard_qkv(
                'the Kitchen QKV producer is unavailable; Preserve precision keeps ConvRot-256 INT8 QKV on upstream Comfy execution'
            )
        return QKVProviderResolution(
            (
                QKV_DENSE_KITCHEN_CHUNKED
                if backend_kind == 'comfy_kitchen_int8'
                else QKV_STREAMED_BF16_KITCHEN
            ),
            backend_kind != 'comfy_kitchen_int8',
            'checkpoint-native ConvRot-256 INT8 QKV streams BF16 projection chunks into the Kitchen INT8 carrier',
        )
    if not _qkv_is_native_bf16(inventory):
        labels = ', '.join(sorted(set(inventory.labels('qkv'))))
        return _standard_qkv(
            'Preserve precision keeps non-BF16 QKV format %s on upstream Comfy execution'
            % (labels or 'unknown')
        )

    if backend_kind in _BF16_KITCHEN_CARRIER_CONSUMERS:
        if not kitchen_producer_available:
            return _standard_qkv(
                'the Kitchen QKV producer is unavailable; Preserve precision keeps native BF16 QKV on upstream Comfy execution'
            )
        return QKVProviderResolution(
            QKV_DENSE_KITCHEN_CHUNKED,
            backend_kind != 'comfy_kitchen_int8',
            (
                'checkpoint-native BF16 QKV streams held 4K BF16 projection, '
                'norm, and RoPE chunks directly into the Kitchen INT8 carrier '
                'without weight requantization'
            ),
        )

    consumers = {
        'existing',
        'sparse_sage',
        'triton_sparse_bf16',
        'flex_attention_fp8',
        'frost_bf16_sm89',
    }
    if backend_kind not in consumers:
        return _standard_qkv(
            'the resolved attention backend has no Preserve-precision BF16 QKV consumer'
        )
    return QKVProviderResolution(
        QKV_BF16_CHUNKED,
        False,
        'checkpoint-native BF16 QKV uses held 4K projection chunks and preserves BF16 values for the selected attention backend',
    )


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
        return _required_or_standard(
            request, 'the H3 model has no QKV projection inventory'
        )
    if not inventory.homogeneous('qkv'):
        return _required_or_standard(
            request, 'H3 QKV layers use mixed weight formats'
        )
    if backend_kind == 'dense_sage_sm89':
        if not inventory.qkv_convrot_int8_256:
            return _required_or_standard(
                request,
                'native dense fused Sage requires ConvRot-256 INT8 QKV',
            )
        if not triton_available:
            return _required_or_standard(
                request, 'native dense fused Sage requires Triton'
            )
        return QKVProviderResolution(
            QKV_DENSE_CONVROT_INT8,
            True,
            'native ConvRot-256 INT8 QKV projects directly into dense Sage carriers',
        )
    if request == FUSED_QKV_PRESERVE_BF16:
        return _resolve_preserve_precision_qkv(
            inventory, backend_kind, kitchen_producer_available
        )

    if inventory.qkv_w4a8:
        if backend_kind == 'comfy_kitchen_int8' and memory_optimize:
            if not kitchen_producer_available:
                return _required_or_standard(
                    request, 'Comfy Kitchen external producer API is unavailable'
                )
            return QKVProviderResolution(
                QKV_DENSE_W4A8_CHUNKED,
                False,
                'checkpoint-native W4A8 QKV is projected in bounded token chunks into Kitchen carriers',
            )
        if backend_kind == 'sparse_kitchen_int8':
            if not kitchen_producer_available:
                return _required_or_standard(
                    request, 'Comfy Kitchen external producer API is unavailable'
                )
            return QKVProviderResolution(
                QKV_DENSE_W4A8_CHUNKED,
                True,
                'checkpoint-native W4A8 QKV is projected in bounded token chunks into routed Kitchen INT8 carriers',
            )
        if backend_kind == 'sparse_sage':
            if not triton_available:
                return _required_or_standard(request, 'Triton is unavailable')
            if not _sparse_contract_ok(sparse_spec):
                return _required_or_standard(
                    request,
                    'the selected Sparse Sage ABI cannot consume chunked projected QKV',
                )
            return QKVProviderResolution(
                QKV_SPARSE_W4A8_CHUNKED,
                True,
                'checkpoint-native W4A8 QKV is projected in bounded token chunks into Sparse Sage carriers',
            )
        if backend_kind == 'triton_sparse_bf16':
            if not triton_available:
                return _required_or_standard(request, 'Triton is unavailable')
            return QKVProviderResolution(
                QKV_TRITON_W4A8_CHUNKED,
                True,
                'checkpoint-native W4A8 QKV is projected in bounded BF16 chunks into the Triton sparse carrier',
            )

    fp8_memory_candidate = (
        request != FUSED_QKV_REQUIRED
        and memory_optimize
        and fp8_available
        and (inventory.qkv_fp8 or inventory.qkv_plain_float)
    )
    if fp8_memory_candidate:
        source = 'checkpoint-native FP8' if inventory.qkv_fp8 else 'BF16/FP16 converted to FP8 E4M3'
        if backend_kind == 'comfy_kitchen_int8':
            if not kitchen_producer_available:
                return _required_or_standard(
                    request, 'Comfy Kitchen external producer API is unavailable'
                )
            return QKVProviderResolution(
                QKV_DENSE_FP8_CHUNKED,
                False,
                '%s QKV uses held FP8 projection into chunked Kitchen carriers' % source,
            )
        if backend_kind == 'sparse_sage':
            if not triton_available:
                return _required_or_standard(request, 'Triton is unavailable')
            if not _sparse_contract_ok(sparse_spec):
                return _required_or_standard(
                    request,
                    'the selected Sparse Sage ABI cannot consume chunked projected QKV',
                )
            return QKVProviderResolution(
                QKV_SPARSE_FP8_CHUNKED,
                True,
                '%s QKV uses held FP8 projection into Sparse Sage carriers' % source,
            )

    if not inventory.qkv_convrot_int8_256:
        labels = ', '.join(sorted(set(inventory.labels('qkv'))))
        return _required_or_standard(
            request,
            'native checkpoint projection preserves QKV format %s'
            % (labels or 'unknown')
        )
    if backend_kind == 'comfy_kitchen_int8':
        if not kitchen_producer_available:
            return _required_or_standard(
                request, 'Comfy Kitchen external producer API is unavailable'
            )
        return QKVProviderResolution(
            QKV_DENSE_KITCHEN_CHUNKED,
            False,
            '4K ConvRot QKV chunks into Comfy Kitchen INT8 carriers',
        )
    if backend_kind == 'sparse_kitchen_int8':
        if not kitchen_producer_available:
            return _required_or_standard(
                request, 'Comfy Kitchen external producer API is unavailable'
            )
        return QKVProviderResolution(
            QKV_DENSE_KITCHEN_CHUNKED,
            True,
            '4K ConvRot QKV chunks into routed Comfy Kitchen INT8 carriers',
        )
    if backend_kind == 'triton_sparse_bf16':
        if not triton_available:
            return _required_or_standard(request, 'Triton is unavailable')
        return QKVProviderResolution(
            QKV_TRITON_SPARSE_CHUNKED,
            True,
            '4K ConvRot QKV chunks into the Triton BF16 sparse carrier',
        )
    if backend_kind == 'sparse_sage':
        if not triton_available:
            return _required_or_standard(request, 'Triton is unavailable')
        from ..attention.sparse.fused_qkv import (
            sparse_fused_qkv_contract_mismatch,
        )

        mismatch = sparse_fused_qkv_contract_mismatch(sparse_spec)
        if mismatch is not None:
            return _required_or_standard(
                request,
                'the selected Sparse Sage ABI cannot consume fused QKV: %s'
                % mismatch
            )
        return QKVProviderResolution(
            QKV_SPARSE_CONVROT_INT8,
            True,
            '4K ConvRot QKV chunks into Sparse Sage-native carriers',
        )
    return _required_or_standard(
        request,
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


def _resolve_preserve_precision_mlp(inventory, fp8_available):
    """Use bounded MLP execution without introducing new quantization."""
    if _convrot_compatible(inventory):
        return MLPProviderResolution(
            MLP_CONVROT_INT8_TWO_SLICE,
            'mlp_chunked_convrot_2slice',
            (
                'checkpoint-native ConvRot-256 TensorWise INT8 MLP uses token '
                'chunks and two feature slices without requantization'
            ),
        )
    if inventory.mlp_w4a8:
        return MLPProviderResolution(
            MLP_W4A8_CHUNKED,
            'mlp_chunked_native',
            (
                'checkpoint-native W4A8 MLP uses held quantized weights and '
                'bounded token chunks without requantization'
            ),
        )
    if inventory.mlp_fp8:
        if fp8_available:
            return MLPProviderResolution(
                MLP_FP8_CHUNKED,
                'mlp_chunked_fp8',
                (
                    'checkpoint-native FP8 MLP uses held weights and bounded '
                    'token chunks without changing checkpoint precision'
                ),
            )
        return MLPProviderResolution(
            MLP_PRESERVE_UPSTREAM,
            'off',
            (
                'checkpoint-native FP8 MLP is preserved upstream because '
                'accelerated FP8 execution is unavailable'
            ),
        )
    if inventory.mlp_plain_float:
        return MLPProviderResolution(
            MLP_FLOAT_CHUNKED,
            'mlp_chunked_native',
            (
                'floating H3 MLP keeps its checkpoint precision and uses '
                'bounded held-weight token chunking'
            ),
        )

    labels = sorted(
        set(inventory.labels('fc1')) | set(inventory.labels('fc2'))
    )
    return MLPProviderResolution(
        MLP_PRESERVE_UPSTREAM,
        'off',
        (
            'preserve-precision mode has no chunked provider for MLP format %s; '
            'preserving upstream Comfy execution'
            % (', '.join(labels) or 'unknown')
        ),
    )


def resolve_mlp_provider(inventory, *, request, fp8_available=False):
    if request == MLP_MEMORY_OFF:
        return MLPProviderResolution(
            MLP_OFF,
            'off',
            'MLP memory optimization was disabled',
        )
    if not inventory.fc1 or not inventory.fc2:
        if request == MLP_MEMORY_LEGACY_CONVROT_REQUIRED:
            raise RuntimeError(
                'required ConvRot two-slice MLP is unavailable: '
                'the H3 model has no MLP inventory'
            )
        return MLPProviderResolution(
            MLP_OFF,
            'off',
            'the H3 model has no MLP inventory',
        )
    if request == MLP_MEMORY_PRESERVE:
        return _resolve_preserve_precision_mlp(inventory, fp8_available)
    if request == MLP_MEMORY_BF16:
        return MLPProviderResolution(
            MLP_FLOAT_CHUNKED,
            'mlp_chunked_bf16',
            'MLP weights are materialized as BF16 for bounded BF16 execution',
        )
    if request == MLP_MEMORY_FORCE_QUANT:
        if inventory.mlp_plain_float:
            return MLPProviderResolution(
                MLP_RUNTIME_CONVROT_INT8_CHUNKED,
                'mlp_chunked_convrot_int8_runtime',
                (
                    'floating H3 MLP weights are converted to execution-scoped '
                    'ConvRot-256 INT8 while activations remain BF16'
                ),
            )
        resolved = _resolve_preserve_precision_mlp(inventory, fp8_available)
        if resolved.provider_id == MLP_PRESERVE_UPSTREAM:
            raise RuntimeError(
                'Force quant has no executable provider for the checkpoint MLP format'
            )
        return resolved
    if _convrot_compatible(inventory):
        if request == MLP_MEMORY_LEGACY_CONVROT_REQUIRED:
            return MLPProviderResolution(
                MLP_CONVROT_INT8_TWO_SLICE,
                'mlp_chunked_convrot_2slice',
                'deprecated compatibility request requires ConvRot two-slice execution',
            )
        if request in (MLP_MEMORY_LEGACY_BF16, MLP_MEMORY_LEGACY_NATIVE):
            return MLPProviderResolution(
                MLP_FLOAT_CHUNKED,
                (
                    'mlp_chunked_bf16'
                    if request == MLP_MEMORY_LEGACY_BF16
                    else 'mlp_chunked_native'
                ),
                'deprecated compatibility request preserves explicit chunked MLP math',
            )
        if request != MLP_MEMORY_AUTO:
            raise ValueError('unknown MLP memory request %r' % request)
        return MLPProviderResolution(
            MLP_CONVROT_INT8_TWO_SLICE,
            'mlp_chunked_convrot_2slice',
            (
                'ConvRot-256 TensorWise INT8 MLP uses token chunks and two '
                'feature slices'
            ),
        )
    if request == MLP_MEMORY_LEGACY_CONVROT_REQUIRED:
        labels = sorted(
            set(inventory.labels('fc1')) | set(inventory.labels('fc2'))
        )
        raise RuntimeError(
            'required ConvRot two-slice MLP is unavailable for %s'
            % (', '.join(labels) or 'unknown formats')
        )
    if request in (MLP_MEMORY_LEGACY_BF16, MLP_MEMORY_LEGACY_NATIVE):
        return MLPProviderResolution(
            MLP_FLOAT_CHUNKED,
            (
                'mlp_chunked_bf16'
                if request == MLP_MEMORY_LEGACY_BF16
                else 'mlp_chunked_native'
            ),
            'deprecated compatibility request preserves explicit chunked MLP math',
        )
    if request != MLP_MEMORY_AUTO:
        raise ValueError('unknown MLP memory request %r' % request)
    if inventory.mlp_w4a8:
        return MLPProviderResolution(
            MLP_W4A8_CHUNKED,
            'mlp_chunked_native',
            'checkpoint-native W4A8 MLP uses held quantized weights and bounded token chunks',
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
