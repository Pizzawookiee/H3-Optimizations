'''Install the streamed-QKV policy used by every public optimization node.

The core apply module deliberately stays backend/mechanism focused. This layer
makes QKV streaming policy explicit without duplicating the optimizer: it swaps
in the policy resolver once, teaches the generic Kitchen projector to stream
native FP8 weights as BF16 Q/K/V chunks, and leaves plan application in the
owning apply module.
'''

from . import apply as _base
from .environment import RuntimeEnvironment
from .plan import (
    FUSED_QKV_AUTO,
    FUSED_QKV_FORCE_QUANT,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_FORCE_QUANT,
)
from .qkv.formats import describe_linear
from .qkv import policy as _qkv_policy
from .qkv import providers as _providers

_BASE_KITCHEN_PROJECTOR = _base.ChunkedKitchenQKVProjector
_BASE_MLP_RESOLVER = _base.resolve_mlp_provider


def _current_capability():
    '''Best-effort capability for ComfyUI's selected NVIDIA device.'''
    environment = RuntimeEnvironment.detect()
    if not environment.cuda_available or environment.capability is None:
        return None
    return tuple(int(value) for value in environment.capability)


def resolve_qkv_provider(inventory, *, request, backend_kind, **kwargs):
    '''Apply architecture legality before the ordinary QKV preference policy.

    Turing can execute the package's ConvRot INT8 kernels with FP16 activations,
    but it cannot execute the BF16 carrier paths used by the normal floating
    Auto policy. Keep the public immutable plan as Auto; only the effective
    provider request is coerced here.

    Plain floating weights are runtime-quantized only when a direct Kitchen
    carrier is selected. Other consumers stay on upstream Comfy QKV so Turing
    receives its normal FP16/dequantized execution instead of entering one of
    H3's BF16-only bounded projectors. Existing ConvRot checkpoints retain their
    native provider. Unknown/unsupported quantized layouts likewise stay
    upstream rather than being requantized blindly.
    '''
    capability = _current_capability()
    if capability == (7, 5) and request == FUSED_QKV_AUTO:
        if getattr(inventory, 'qkv_plain_float', False):
            if backend_kind in ('comfy_kitchen_int8', 'sparse_kitchen_int8'):
                return _qkv_policy.resolve_qkv_provider(
                    inventory,
                    request=FUSED_QKV_FORCE_QUANT,
                    backend_kind=backend_kind,
                    **kwargs,
                )
            return _providers.QKVProviderResolution(
                _providers.QKV_STANDARD,
                False,
                'SM75 Auto keeps floating QKV on upstream FP16 execution '
                'unless a direct Kitchen INT8 carrier is selected',
            )
        if not getattr(inventory, 'qkv_convrot_int8_256', False):
            return _providers.QKVProviderResolution(
                _providers.QKV_STANDARD,
                False,
                'SM75 Auto leaves non-ConvRot quantized QKV on upstream Comfy '
                'execution so unsupported storage formats can dequantize safely',
            )
    return _qkv_policy.resolve_qkv_provider(
        inventory,
        request=request,
        backend_kind=backend_kind,
        **kwargs,
    )


def resolve_mlp_provider(inventory, *, request, **kwargs):
    '''Use Turing's executable INT8+FP16 MLP route for floating Auto weights.'''
    if (
        _current_capability() == (7, 5)
        and request == MLP_MEMORY_AUTO
        and getattr(inventory, 'mlp_plain_float', False)
    ):
        request = MLP_MEMORY_FORCE_QUANT
    return _BASE_MLP_RESOLVER(inventory, request=request, **kwargs)


class PolicyChunkedKitchenQKVProjector(_BASE_KITCHEN_PROJECTOR):
    '''Generic Kitchen carrier with checkpoint-native FP8 auto-binding.

    `fp8_projection=True` keeps its historical meaning: the policy explicitly
    authorized a floating checkpoint to be converted to FP8 as a fallback.
    When it is false and the checkpoint is already FP8, use the same held FP8
    linear without changing the public provider id. In both cases the projected
    Q/K/V chunks handed to the carrier remain BF16.
    '''

    @property
    def installation_signature(self):
        return super().installation_signature + ('native_fp8_bf16_stream',)

    def try_project(
        self,
        module,
        x,
        rope_freqs,
        *,
        layer_index,
        transformer_options,
    ):
        actual = describe_linear(module.qkv_proj)
        if actual.fp8 and not (
            self.force_weights_bf16
            or self.fp8_projection
            or self.convrot_int8_projection
        ):
            delegate = _BASE_KITCHEN_PROJECTOR(
                chunk_rows=self.chunk_rows,
                fp8_projection=True,
                routing_summaries=self.routing_summaries,
                q_tile=self.q_tile,
                kv_tile=self.kv_tile,
                strided_qk_input=self.strided_qk_input,
                stream_output=self.stream_output,
            )
            return delegate.try_project(
                module,
                x,
                rope_freqs,
                layer_index=layer_index,
                transformer_options=transformer_options,
            )
        return super().try_project(
            module,
            x,
            rope_freqs,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )


# apply.py resolves these globals at execution time. Install the policy once so
# both Memory Optimization and Sparse Attention use identical priorities.
_base.resolve_qkv_provider = resolve_qkv_provider
_base.resolve_mlp_provider = resolve_mlp_provider
_base.ChunkedKitchenQKVProjector = PolicyChunkedKitchenQKVProjector


def apply_plan(model, plan):
    return _base.apply_plan(model, plan)
