'''Install the streamed-QKV policy used by every public optimization node.

The core apply module deliberately stays backend/mechanism focused. This layer
makes QKV streaming policy explicit without duplicating the optimizer: it swaps
in the policy resolver once, teaches the generic Kitchen projector to stream
native FP8 weights as BF16 Q/K/V chunks, and leaves plan application in the
owning apply module.
'''

from . import apply as _base
from .qkv.formats import describe_linear
from .qkv.policy import resolve_qkv_provider

_BASE_KITCHEN_PROJECTOR = _base.ChunkedKitchenQKVProjector


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
# both Memory Optimization and Sparse Attention use identical QKV priorities.
_base.resolve_qkv_provider = resolve_qkv_provider
_base.ChunkedKitchenQKVProjector = PolicyChunkedKitchenQKVProjector


def apply_plan(model, plan):
    return _base.apply_plan(model, plan)
