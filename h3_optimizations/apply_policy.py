'''Install the QKV policy and apply the small amount of plan-aware Auto logic.

The core apply module deliberately stays backend/mechanism focused. This layer
makes QKV streaming policy explicit without duplicating the optimizer: it swaps
in the policy resolver once, teaches the generic Kitchen projector to stream
native FP8 weights as BF16 Q/K/V chunks, and makes dense Auto yield when a
streamed producer cannot actually be established.
'''

from dataclasses import replace

from . import apply as _base
from .kitchen_qkv import producer_api_available
from .model import get_h3_blocks, is_minimax_h3
from .plan import ATTENTION_EXISTING, QKV_STREAMING_AUTO
from .qkv.formats import describe_linear, inspect_h3_linears
from .qkv.policy import is_dense_streamed_provider, resolve_qkv_provider

DENSE_KITCHEN_KIND = 'comfy_kitchen_int8'
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
        if actual.fp8 and not self.fp8_projection:
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


def _auto_streaming_plan(model, plan):
    memory = plan.memory
    if (
        memory is None
        or plan.sparse is not None
        or memory.qkv_streaming != QKV_STREAMING_AUTO
        or memory.attention == ATTENTION_EXISTING
        or not is_minimax_h3(model)
    ):
        return plan

    inventory = inspect_h3_linears(get_h3_blocks(model))
    environment = _base.RuntimeEnvironment.detect()
    resolved = resolve_qkv_provider(
        inventory,
        request=memory.fused_qkv,
        backend_kind=DENSE_KITCHEN_KIND,
        kitchen_producer_available=producer_api_available(
            device=getattr(environment, 'device_index', None)
        ),
        memory_optimize=True,
        fp8_available=_base._fp8_execution_available(environment),
    )
    if is_dense_streamed_provider(resolved.provider_id):
        return plan

    # Auto exists to obtain streaming, not to gratuitously change attention.
    # Forced remains untouched and therefore keeps its explicit override.
    return replace(
        plan,
        memory=replace(memory, attention=ATTENTION_EXISTING),
    )


def apply_plan(model, plan):
    return _base.apply_plan(model, _auto_streaming_plan(model, plan))
