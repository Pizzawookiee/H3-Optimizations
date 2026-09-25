'''Dense H3 attention selection through ComfyUI's public backend API.'''

from __future__ import annotations

from dataclasses import dataclass

import comfy.model_management
from comfy.ldm.modules.attention import get_attention_function

from .external_consumer import get_streamed_h3_qkv_consumer


ATTENTION_AUTO = 'auto'
ATTENTION_COMFY_KITCHEN_INT8 = 'comfy_kitchen_int8'
ATTENTION_EXISTING = 'existing'
ATTENTION_EXISTING_FULL_Q = 'existing_full_q'
ATTENTION_SAGE = 'sage'
ATTENTION_SAGE_PREFIX = 'dense_sage_sm'
ATTENTION_SAGE_SM89 = 'dense_sage_sm89'
OVERRIDE_MARKER = '_h3_optimizations_dense_backend'

# Recognized external dense attention implementations that sparse attention is
# allowed to displace. Each entry is a specific implementation we have
# identified as plain dense H3 attention, where swapping in sparse routing
# changes the kernel and not the semantics the user asked for. Overrides that
# are merely *compatible* with us -- including anything that only advertises
# the streamed-H3 QKV contract -- are deliberately not on this list.
KJ_SAGE_OVERRIDE_QUALNAME = (
    'make_sage_attention_override.<locals>.attention_override_sage'
)
KJ_SAGE_OVERRIDE_MODULE = 'model_optimization_nodes'


@dataclass(frozen=True)
class DenseResolution:
    requested: str
    selected: str
    backend: object | None
    reason: str
    backend_kind: str


def _existing_resolution(requested, reason, *, backend_kind=ATTENTION_EXISTING):
    return DenseResolution(
        requested,
        ATTENTION_EXISTING,
        None,
        reason,
        backend_kind,
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


def _prepared_sage_backend(capability):
    if capability == (8, 9):
        from .attention.sage_mem_eff import SM89SageMemoryEfficientBackend

        return SM89SageMemoryEfficientBackend()

    from .attention.sage_arch import (
        SageSM12xMemoryEfficientBackend,
        SageSM75MemoryEfficientBackend,
        SageSM80MemoryEfficientBackend,
        SageSM86MemoryEfficientBackend,
        SageSM90MemoryEfficientBackend,
    )

    backend_type = {
        (7, 5): SageSM75MemoryEfficientBackend,
        (8, 0): SageSM80MemoryEfficientBackend,
        (8, 6): SageSM86MemoryEfficientBackend,
        (8, 7): SageSM80MemoryEfficientBackend,
        (9, 0): SageSM90MemoryEfficientBackend,
        (10, 0): SageSM12xMemoryEfficientBackend,
        (12, 0): SageSM12xMemoryEfficientBackend,
        (12, 1): SageSM12xMemoryEfficientBackend,
    }.get(capability)
    return None if backend_type is None else backend_type()


def resolve_sage_fused_attention(model_patcher, environment):
    if not sage_attention_selected(model_patcher):
        return None
    capability = tuple(getattr(environment, 'capability', ()) or ())
    architecture = (
        'unknown'
        if len(capability) != 2
        else 'SM%d%d' % capability
    )
    try:
        backend = _prepared_sage_backend(capability)
    except Exception as exc:
        return DenseResolution(
            ATTENTION_SAGE,
            ATTENTION_SAGE,
            None,
            'native-carrier SageAttention preflight failed on %s: %s: %s'
            % (architecture, type(exc).__name__, exc),
            ATTENTION_SAGE,
        )
    if backend is None:
        return DenseResolution(
            ATTENTION_SAGE,
            ATTENTION_SAGE,
            None,
            'no native-carrier SageAttention2 adapter for %s' % architecture,
            ATTENTION_SAGE,
        )
    return DenseResolution(
        ATTENTION_SAGE,
        ATTENTION_SAGE,
        backend,
        'selected SageAttention2 %s with direct native-carrier QKV support'
        % architecture,
        '%s%d%d' % (ATTENTION_SAGE_PREFIX, capability[0], capability[1]),
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


def is_kj_sage_dense_attention(transformer_options):
    '''Whether the active override is KJNodes' SageAttention patch.

    Every KJNodes node that installs Sage builds its override through the same
    ``make_sage_attention_override`` factory, so one identity check covers all
    of them. Matching on the closure's own name plus its defining module keeps
    this narrow: it will not claim an unrelated pack, and if KJNodes ever
    restructures the factory the check simply stops matching and the override
    is preserved as before.
    '''
    options = transformer_options or {}
    override = options.get('optimized_attention_override')
    if override is None:
        return False
    if getattr(override, '__qualname__', '') != KJ_SAGE_OVERRIDE_QUALNAME:
        return False
    module = getattr(override, '__module__', '') or ''
    return module.rsplit('.', 1)[-1] == KJ_SAGE_OVERRIDE_MODULE


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


def is_replaceable_dense_attention(transformer_options):
    '''Whether sparse attention may take over from the active override.

    True only for dense implementations we recognize by identity: our own
    installed backend, Comfy Kitchen INT8, and KJNodes' Sage patch. Any other
    override is preserved, because we cannot tell whether it is dense at all.
    '''
    options = transformer_options or {}
    if options.get('optimized_attention_override') is None:
        return False
    return (
        is_installed_dense_attention(options)
        or is_comfy_kitchen_dense_attention(options)
        or is_kj_sage_dense_attention(options)
    )


def is_known_comfy_dense_attention(transformer_options):
    options = transformer_options or {}
    override = options.get('optimized_attention_override')
    if override is None:
        return True
    for name in ('pytorch', 'sub_quad', 'split', 'flash', 'xformers'):
        if _override_wraps_backend(
            override,
            get_attention_function(name, None),
        ):
            return True
    return False


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
    if get_streamed_h3_qkv_consumer(options) is not None:
        return _existing_resolution(
            ATTENTION_EXISTING,
            'preserved an external attention consumer with streamed-H3 QKV '
            'support',
        )
    sage = resolve_sage_fused_attention(model_patcher, environment)
    if sage is not None:
        return sage
    if not is_known_comfy_dense_attention(options):
        return _existing_resolution(
            ATTENTION_EXISTING,
            'preserved an unknown optimized-attention override with full-Q '
            'single-call semantics',
            backend_kind=ATTENTION_EXISTING_FULL_Q,
        )
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
