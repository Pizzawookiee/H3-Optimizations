"""Stage-granular dynamic-VRAM prefetch for streamed H3 execution.

Stock ComfyUI prefetches one MiniMax H3 transformer block as a unit. Streamed
attention deliberately gives qkv_proj, out_proj, fc1, and fc2 disjoint
lifetimes, so whole-block prefetch defeats the low-VRAM schedule. This module
keeps Comfy/AIMDO's own VBAR transfer machinery but changes the prefetch unit
from a block to one explicit linear stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import comfy.model_management
import comfy.model_prefetch
import comfy.ops

STAGE_PREFETCH_KEY = "h3_optimizations_stage_prefetch"
ATTENTION_MEMORY_MODE_KEY = "h3_optimizations_attention_memory_mode"
ATTENTION_MEMORY_STREAMED = "streamed"

_PREFETCH_OVERRIDE_ACTIVE_KEY = "h3_optimizations_prefetch_override_active"
_PREFETCH_PREVIOUS_PRESENT_KEY = "h3_optimizations_prefetch_previous_present"
_PREFETCH_PREVIOUS_VALUE_KEY = "h3_optimizations_prefetch_previous_value"

LOG_PREFIX = "[H3 Optimizations]"


@dataclass
class StagePrefetchTicket:
    module: object
    device: object
    offload_stream: object
    active: bool = True


def configure_stage_prefetch(transformer_options):
    """Replace stock whole-block H3 prefetch only for explicit streamed mode."""
    if transformer_options is None:
        return False

    enabled = (
        transformer_options.get(ATTENTION_MEMORY_MODE_KEY)
        == ATTENTION_MEMORY_STREAMED
    )
    transformer_options[STAGE_PREFETCH_KEY] = bool(enabled)

    override_active = bool(
        transformer_options.get(_PREFETCH_OVERRIDE_ACTIVE_KEY, False)
    )

    if enabled:
        # Record the upstream value only when taking ownership of this setting.
        # configure_stage_prefetch() can be called repeatedly during one
        # request, so do not overwrite the saved value with our own False.
        if not override_active:
            transformer_options[_PREFETCH_PREVIOUS_PRESENT_KEY] = (
                "prefetch_dynamic_vbars" in transformer_options
            )
            transformer_options[_PREFETCH_PREVIOUS_VALUE_KEY] = (
                transformer_options.get("prefetch_dynamic_vbars")
            )
            transformer_options[_PREFETCH_OVERRIDE_ACTIVE_KEY] = True

        transformer_options["prefetch_dynamic_vbars"] = False

    elif override_active:
        # Restore exactly what existed before streamed mode took ownership.
        previous_present = bool(
            transformer_options.pop(
                _PREFETCH_PREVIOUS_PRESENT_KEY,
                False,
            )
        )
        previous_value = transformer_options.pop(
            _PREFETCH_PREVIOUS_VALUE_KEY,
            None,
        )
        transformer_options.pop(_PREFETCH_OVERRIDE_ACTIVE_KEY, None)

        if previous_present:
            transformer_options["prefetch_dynamic_vbars"] = previous_value
        else:
            transformer_options.pop("prefetch_dynamic_vbars", None)

    return bool(enabled)

def stage_prefetch_enabled(transformer_options):
    return bool(
        transformer_options
        and transformer_options.get(STAGE_PREFETCH_KEY, False)
        and transformer_options.get(ATTENTION_MEMORY_MODE_KEY)
        == ATTENTION_MEMORY_STREAMED
    )


def begin_stage_prefetch(module, device, *, enabled=True):
    """Start an async VBAR fault for exactly one Comfy linear module."""
    if (
        not enabled
        or module is None
        or not hasattr(module, "_v")
        or comfy.model_management.NUM_STREAMS == 0
        or comfy.model_management.is_device_cpu(device)
        or not comfy.model_management.device_supports_non_blocking(device)
    ):
        return None

    # Never stack a second prefetch record onto a module. A live record means
    # another stage already owns its transfer lifetime.
    if getattr(module, "_prefetch", None) is not None:
        return None

    offload_stream, _fully_faulted = comfy.ops.cast_modules_with_vbar(
        [module],
        None,
        device,
        None,
        True,
        return_faulted=True,
    )
    return StagePrefetchTicket(
        module=module,
        device=device,
        offload_stream=offload_stream,
    )


def wait_stage_prefetch(ticket):
    """Wait for one staged transfer without discarding module._prefetch."""
    if ticket is None or not ticket.active:
        return
    comfy.model_management.sync_stream(
        ticket.device,
        ticket.offload_stream,
    )


def release_stage_prefetch(ticket):
    """Clean staged VBAR state only after the weight has been consumed."""
    if ticket is None or not ticket.active:
        return
    try:
        comfy.model_prefetch.cleanup_prefetched_modules(
            ticket.module,
            [ticket.module],
        )
    finally:
        ticket.active = False


def abandon_stage_prefetch(ticket):
    """Exception-safe cleanup for a stage that will no longer execute."""
    release_stage_prefetch(ticket)


_logged = False


def log_stage_prefetch_enabled():
    global _logged
    if not _logged:
        logging.info(
            "%s streamed memory mode: using stage-aware dynamic-VRAM prefetch",
            LOG_PREFIX,
        )
        _logged = True
