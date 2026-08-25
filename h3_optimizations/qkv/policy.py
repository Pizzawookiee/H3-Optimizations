'''QKV execution policy layered over format-specific providers.

Streaming is an activation-memory policy, not a request to quantize QKV
weights. Whenever a compatible consumer exists we keep the checkpoint's weight
format for the linear, produce post-projection Q/K/V as BF16 chunks, and feed
those chunks directly into the consumer carrier. BF16/FP16 -> FP8 weight
conversion is only a fallback after streaming and native/bounded execution have
both been ruled out. Explicit BF16 and Force quant requests bypass that Auto
priority and require their requested weight representation.
'''

from . import providers as base
from ..plan import (
    FUSED_QKV_FORCE_BF16,
    FUSED_QKV_FORCE_QUANT,
    FUSED_QKV_OFF,
    FUSED_QKV_PRESERVE_BF16,
    FUSED_QKV_REQUIRED,
)

QKV_STREAMED_SPARSE_SAGE = base.QKV_SPARSE_CONVROT_INT8
DENSE_SAGE_SM89 = 'dense_sage_sm89'

_KITCHEN_CARRIER_CONSUMERS = {
    'comfy_kitchen_int8',
    'sparse_kitchen_int8',
    'native_int8_128x64',
    'native_int8_128x64_sol_residual_64x64',
    'native_int8_64x64',
    'native_int8_64x64_sol_residual_64x64',
    'native_int8_128x128_hard_control',
    'native_int8_128x128_sol_residual_64x64',
}


def _native_stream_format(inventory):
    if inventory.qkv_convrot_int8_256:
        return 'ConvRot-256 INT8'
    if inventory.qkv_w4a8:
        return 'W4A8'
    if inventory.qkv_fp8:
        return 'FP8'
    if base._qkv_is_native_bf16(inventory):
        return 'BF16'
    return None


def is_dense_streamed_provider(provider_id):
    return provider_id in (
        base.QKV_DENSE_KITCHEN_CHUNKED,
        base.QKV_DENSE_FP8_CHUNKED,
    )


def _stream_kitchen(
    inventory,
    *,
    request,
    backend_kind,
    kitchen_producer_available,
):
    fmt = _native_stream_format(inventory)
    if fmt is None or backend_kind not in _KITCHEN_CARRIER_CONSUMERS:
        return None
    if not kitchen_producer_available:
        if request == FUSED_QKV_REQUIRED:
            raise RuntimeError(
                'required fused QKV is unavailable: the Kitchen QKV producer '
                'is unavailable for streamed %s QKV' % fmt
            )
        return None
    return base.QKVProviderResolution(
        base.QKV_DENSE_KITCHEN_CHUNKED,
        backend_kind != 'comfy_kitchen_int8',
        (
            'checkpoint-native %s weights project into bounded BF16 Q/K/V '
            'chunks streamed directly into the Kitchen carrier'
        ) % fmt,
    )


def _stream_sparse_sage(
    inventory,
    *,
    backend_kind,
    triton_available,
    sparse_spec,
):
    if backend_kind != 'sparse_sage' or _native_stream_format(inventory) is None:
        return None
    if not triton_available or not base._sparse_contract_ok(sparse_spec):
        return None
    return base.QKVProviderResolution(
        QKV_STREAMED_SPARSE_SAGE,
        True,
        (
            'checkpoint-native %s weights project into bounded BF16 Q/K/V '
            'chunks for streamed Sparse Sage execution'
        ) % _native_stream_format(inventory),
    )


def _stream_triton(inventory, *, backend_kind, triton_available):
    if (
        backend_kind != 'triton_sparse_int8'
        or _native_stream_format(inventory) is None
        or not triton_available
    ):
        return None
    return base.QKVProviderResolution(
        base.QKV_TRITON_SPARSE_CHUNKED,
        True,
        (
            'checkpoint-native %s weights project into bounded BF16 Q/K/V '
            'chunks for the Triton sparse carrier'
        ) % _native_stream_format(inventory),
    )


def _native_bounded_fallback(
    inventory,
    *,
    backend_kind,
    triton_available,
    sparse_spec,
    fp8_available,
):
    '''Best bounded provider when streaming is unavailable, without new quantization.'''
    if inventory.qkv_convrot_int8_256:
        if backend_kind == 'sparse_sage' and triton_available and base._sparse_contract_ok(sparse_spec):
            return base.QKVProviderResolution(
                base.QKV_SPARSE_CONVROT_INT8,
                True,
                'checkpoint-native ConvRot QKV uses its native Sparse Sage provider',
            )
        if backend_kind == 'triton_sparse_int8' and triton_available:
            return base.QKVProviderResolution(
                base.QKV_TRITON_SPARSE_CHUNKED,
                True,
                'checkpoint-native ConvRot QKV uses its native Triton provider',
            )

    if inventory.qkv_w4a8:
        if backend_kind == 'sparse_sage' and triton_available and base._sparse_contract_ok(sparse_spec):
            return base.QKVProviderResolution(
                base.QKV_SPARSE_W4A8_CHUNKED,
                True,
                'checkpoint-native W4A8 QKV uses its native Sparse Sage provider',
            )
        if backend_kind == 'triton_sparse_int8' and triton_available:
            return base.QKVProviderResolution(
                base.QKV_TRITON_W4A8_CHUNKED,
                True,
                'checkpoint-native W4A8 QKV uses its native Triton provider',
            )

    if inventory.qkv_fp8 and (
        backend_kind == 'sparse_sage'
        and fp8_available
        and triton_available
        and base._sparse_contract_ok(sparse_spec)
    ):
        return base.QKVProviderResolution(
            base.QKV_SPARSE_FP8_CHUNKED,
            True,
            'checkpoint-native FP8 QKV uses held FP8 projection without changing checkpoint precision',
        )

    fmt = _native_stream_format(inventory)
    if fmt is not None:
        return base.QKVProviderResolution(
            base.QKV_BF16_CHUNKED,
            False,
            (
                'checkpoint-native %s QKV projects in bounded token chunks and '
                'materializes complete BF16 Q/K/V for the selected attention consumer'
            ) % fmt,
        )

    labels = ', '.join(sorted(set(inventory.labels('qkv'))))
    return base._standard_qkv(
        'checkpoint-native QKV format %s has no bounded provider for the selected attention backend'
        % (labels or 'unknown')
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
    '''Resolve QKV with streamed BF16 activations as the first preference.'''
    if request == FUSED_QKV_OFF:
        return base._standard_qkv('QKV streaming was disabled')
    if not inventory.qkv:
        return base._required_or_standard(request, 'the H3 model has no QKV projection inventory')
    if not inventory.homogeneous('qkv'):
        return base._required_or_standard(request, 'H3 QKV layers use mixed weight formats')
    if backend_kind == DENSE_SAGE_SM89:
        return base.resolve_qkv_provider(
            inventory,
            request=request,
            backend_kind=backend_kind,
            triton_available=triton_available,
            memory_optimize=memory_optimize,
            fp8_available=fp8_available,
        )
    if request == FUSED_QKV_FORCE_BF16:
        if not (
            inventory.qkv_plain_float
            or inventory.qkv_convrot_int8_256
            or inventory.qkv_w4a8
            or inventory.qkv_fp8
        ):
            raise RuntimeError(
                'BF16 mode cannot materialize the checkpoint QKV format as BF16'
            )
        return base.QKVProviderResolution(
            base.QKV_FORCE_BF16_CHUNKED,
            False,
            'QKV weights are materialized as BF16 for bounded BF16 projection',
        )
    if request == FUSED_QKV_FORCE_QUANT and inventory.qkv_plain_float:
        if not fp8_available:
            raise RuntimeError(
                'Force quant requires accelerated FP8 execution for floating QKV weights'
            )
        return base.QKVProviderResolution(
            base.QKV_FORCE_QUANT_CHUNKED,
            False,
            'floating QKV weights are forced to FP8 E4M3 for bounded projection',
        )

    streamed = _stream_kitchen(
        inventory,
        request=request,
        backend_kind=backend_kind,
        kitchen_producer_available=kitchen_producer_available,
    )
    if streamed is not None:
        return streamed
    streamed = _stream_sparse_sage(
        inventory,
        backend_kind=backend_kind,
        triton_available=triton_available,
        sparse_spec=sparse_spec,
    )
    if streamed is not None:
        return streamed
    streamed = _stream_triton(
        inventory,
        backend_kind=backend_kind,
        triton_available=triton_available,
    )
    if streamed is not None:
        return streamed

    native = _native_bounded_fallback(
        inventory,
        backend_kind=backend_kind,
        triton_available=triton_available,
        sparse_spec=sparse_spec,
        fp8_available=fp8_available,
    )

    if request == FUSED_QKV_REQUIRED:
        if native.provider_id != base.QKV_STANDARD and native.fused:
            return native
        raise RuntimeError(
            'required fused QKV is unavailable: %s'
            % (native.reason or 'no compatible projected carrier')
        )

    if native.provider_id != base.QKV_STANDARD:
        return native
    if request == FUSED_QKV_PRESERVE_BF16:
        return native

    return base.resolve_qkv_provider(
        inventory,
        request=request,
        backend_kind=backend_kind,
        triton_available=triton_available,
        sparse_spec=sparse_spec,
        kitchen_producer_available=kitchen_producer_available,
        memory_optimize=memory_optimize,
        fp8_available=fp8_available,
    )
