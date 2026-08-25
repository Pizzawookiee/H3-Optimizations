'''Install the QKV policy and apply the small amount of plan-aware Auto logic.

The core apply module deliberately stays backend/mechanism focused. This layer
makes QKV streaming policy explicit without duplicating the optimizer: it swaps
in the policy resolver once, and makes dense Auto yield when the selected
checkpoint cannot actually establish a streamed Kitchen producer.
'''

from dataclasses import replace

from . import apply as _base
from .kitchen_qkv import producer_api_available
from .model import get_h3_blocks, is_minimax_h3
from .plan import ATTENTION_EXISTING, QKV_STREAMING_AUTO
from .qkv.formats import inspect_h3_linears
from .qkv.policy import is_dense_streamed_provider, resolve_qkv_provider

DENSE_KITCHEN_KIND = 'comfy_kitchen_int8'

# apply.py resolves this global at execution time. Install the policy once so
# both Memory Optimization and Sparse Attention use the same QKV priorities.
_base.resolve_qkv_provider = resolve_qkv_provider


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
