'''Dense H3 attention selection through ComfyUI's public backend API.'''

from __future__ import annotations

from dataclasses import dataclass

import comfy.model_management
from comfy.ldm.modules.attention import get_attention_function


ATTENTION_AUTO = 'auto'
ATTENTION_COMFY_KITCHEN_INT8 = 'comfy_kitchen_int8'
ATTENTION_EXISTING = 'existing'
ATTENTION_SAGE = 'sage'
ATTENTION_SAGE_SM89 = 'dense_sage_sm89'
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


def sage_attention_selected(model_patcher):
    options = (
        getattr(model_patcher, 'model_options', {})
        .get('transformer_options', {})
        or {}
    )
    if 'optimized_attention_override' in options:
        backend = get_attention_function(ATTENTION_SAGE, None)
        return _override_wraps_backend(
            options.get('optimized_attention_override'),
            backend,
        )
    return bool(comfy.model_management.sage_attention_enabled())


def resolve_sage_fused_attention(model_patcher, environment):
    if not sage_attention_selected(model_patcher):
        return None
    if tuple(getattr(environment, 'capability', ()) or ()) != (8, 9):
        return DenseResolution(
            ATTENTION_SAGE,
            ATTENTION_SAGE,
            None,
            'native dense fused QKV requires SM89',
            ATTENTION_SAGE,
        )
    try:
        from .attention.sage_mem_eff import SM89SageMemoryEfficientBackend

        backend = SM89SageMemoryEfficientBackend()
    except Exception as exc:
        return DenseResolution(
            ATTENTION_SAGE,
            ATTENTION_SAGE,
            None,
            'native dense fused Sage preflight failed: %s: %s'
            % (type(exc).__name__, exc),
            ATTENTION_SAGE,
        )
    return DenseResolution(
        ATTENTION_SAGE,
        ATTENTION_SAGE,
        backend,
        'selected SageAttention with direct native-carrier QKV support',
        ATTENTION_SAGE_SM89,
    )


def is_installed_dense_attention(transformer_options):
    options = transformer_options or {}
    override = options.get('optimized_attention_override')
    return getattr(override, OVERRIDE_MARKER, None) == ATTENTION_COMFY_KITCHEN_INT8


def _override_wraps_backend(override, backend):
    '''Recognize Comfy's ModelPatcher wrapper around one registered backend.'''
    if override is None or backend is None:
        return False
    if override is backend:
        return True

    # ModelPatcher copies the backend's container function onto its wrapper.
    # This is the cheapest stable signal for current Comfy Kitchen attention.
    backend_container = getattr(backend, 'container_function', None)
    if (
        backend_container is not None
        and getattr(override, 'container_function', None) is backend_container
    ):
        return True

    # Current ModelPatcher.set_model_optimized_attention closes over the exact
    # registered backend callable. Keep this fallback so detection still works
    # if Kitchen ever stops exposing a container function.
    for cell in getattr(override, '__closure__', None) or ():
        try:
            wrapped = cell.cell_contents
        except ValueError:
            continue
        if wrapped is backend:
            return True
    return False


def is_comfy_kitchen_dense_attention(transformer_options):
    '''Whether the active dense override is our Kitchen path or Comfy's own.'''
    options = transformer_options or {}
    override = options.get('optimized_attention_override')
    if override is None:
        return False
    if is_installed_dense_attention(options):
        return True
    backend = get_attention_function(ATTENTION_COMFY_KITCHEN_INT8, None)
    return _override_wraps_backend(override, backend)


def resolve_current_dense_attention(model_patcher, environment):
    """Preserve Comfy's selected backend while recognizing external Kitchen."""
    options = (
        getattr(model_patcher, 'model_options', {})
        .get('transformer_options', {})
        or {}
    )
    if is_comfy_kitchen_dense_attention(options):
        return DenseResolution(
            ATTENTION_EXISTING,
            ATTENTION_COMFY_KITCHEN_INT8,
            None,
            'preserved the external Comfy Kitchen attention selection',
            ATTENTION_COMFY_KITCHEN_INT8,
        )
    sage = resolve_sage_fused_attention(model_patcher, environment)
    if sage is not None:
        return sage
    return _existing_resolution(
        ATTENTION_EXISTING,
        'preserved ComfyUI\'s current dense attention selection',
    )


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
        compatible_kitchen = is_comfy_kitchen_dense_attention(options)
        return DenseResolution(
            ATTENTION_AUTO,
            ATTENTION_COMFY_KITCHEN_INT8,
            None,
            (
                'upgraded an explicit Comfy Kitchen attention choice to the '
                'streamed private H3 Kitchen path'
                if compatible_kitchen
                else 'preserved an explicit optimized-attention override; '
                'using Comfy Kitchen INT8 only for the private H3 memory path'
            ),
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
    if (
        resolution.backend is None
        or resolution.backend_kind != ATTENTION_COMFY_KITCHEN_INT8
    ):
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
