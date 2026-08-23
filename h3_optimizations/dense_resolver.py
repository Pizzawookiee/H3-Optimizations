'''Dense H3 attention selection through ComfyUI's public backend API.'''

from __future__ import annotations

from dataclasses import dataclass

from comfy.ldm.modules.attention import get_attention_function


ATTENTION_AUTO = 'auto'
ATTENTION_COMFY_KITCHEN_INT8 = 'comfy_kitchen_int8'
ATTENTION_EXISTING = 'existing'
OVERRIDE_MARKER = '_h3_optimizations_dense_backend'


@dataclass(frozen=True)
class DenseResolution:
    requested: str
    selected: str
    backend: object | None
    reason: str
    backend_kind: str


def _existing_resolution(requested, reason):
    return DenseResolution(
        requested,
        ATTENTION_EXISTING,
        None,
        reason,
        ATTENTION_EXISTING,
    )


def preserve_dense_attention(reason):
    return _existing_resolution(ATTENTION_EXISTING, reason)


def is_installed_dense_attention(transformer_options):
    options = transformer_options or {}
    override = options.get('optimized_attention_override')
    return getattr(override, OVERRIDE_MARKER, None) == ATTENTION_COMFY_KITCHEN_INT8


def resolve_dense_attention(model_patcher):
    options = (
        getattr(model_patcher, 'model_options', {})
        .get('transformer_options', {})
        or {}
    )
    backend = get_attention_function(ATTENTION_COMFY_KITCHEN_INT8, None)

    if 'optimized_attention_override' in options:
        if backend is None:
            return _existing_resolution(
                ATTENTION_AUTO,
                'preserved an explicit optimized-attention override; '
                'Comfy Kitchen INT8 is unavailable for the private H3 path',
            )
        return DenseResolution(
            ATTENTION_AUTO,
            ATTENTION_COMFY_KITCHEN_INT8,
            None,
            'preserved an explicit optimized-attention override; '
            'using Comfy Kitchen INT8 only for the private H3 memory path',
            ATTENTION_COMFY_KITCHEN_INT8,
        )

    if backend is None:
        return _existing_resolution(
            ATTENTION_AUTO,
            'Comfy Kitchen INT8 is unavailable; using normal Comfy selection',
        )
    return DenseResolution(
        ATTENTION_AUTO,
        ATTENTION_COMFY_KITCHEN_INT8,
        backend,
        'selected through ComfyUI public attention registry',
        ATTENTION_COMFY_KITCHEN_INT8,
    )


def install_dense_attention(model_patcher, resolution):
    if resolution.backend is None:
        return False
    model_patcher.set_model_optimized_attention(resolution.backend)
    override = model_patcher.model_options[
        'transformer_options'
    ]['optimized_attention_override']
    setattr(override, OVERRIDE_MARKER, ATTENTION_COMFY_KITCHEN_INT8)
    return True


def clear_installed_dense_attention(model_patcher):
    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    if not is_installed_dense_attention(options):
        return False
    del options['optimized_attention_override']
    return True
