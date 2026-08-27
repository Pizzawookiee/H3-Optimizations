'''Reconcile one package-owned H3 attention forward across both nodes.'''

import logging

import comfy.quant_ops

from .model import get_h3_blocks, is_minimax_h3

OWNER_MARKER = '_h3_optimizations_attention'
SIGNATURE_MARKER = '_h3_optimizations_attention_signature'
ORIGINAL_MARKER = '_h3_optimizations_attention_original'
BLOCKS_ATTR = 'diffusion_model.blocks'
REQUIRED_ATTRS = (
    'qkv_proj',
    'q_norm',
    'k_norm',
    'out_proj',
    'heads',
    'head_dim',
)


class H3AttentionPatchError(RuntimeError):
    pass


def key_for(index):
    return '%s.%d.attn.forward' % (BLOCKS_ATTR, int(index))


def installation_signature(value):
    if value is None:
        return None
    signature = getattr(value, 'installation_signature', None)
    if callable(signature):
        signature = signature()
    if signature is not None:
        return signature
    if callable(value):
        function = getattr(value, '__func__', value)
        return (
            getattr(function, '__module__', type(function).__module__),
            getattr(function, '__qualname__', type(function).__qualname__),
            id(function),
        )
    return (
        type(value).__module__,
        type(value).__qualname__,
        getattr(value, 'name', None),
    )


def validate(model_patcher):
    if not is_minimax_h3(model_patcher):
        raise H3AttentionPatchError(
            'H3 optimized attention can only patch MiniMaxH3Model'
        )
    if not hasattr(comfy.quant_ops.ck, 'rms_rope_split_half_'):
        raise H3AttentionPatchError(
            'comfy-kitchen does not expose rms_rope_split_half_'
        )
    blocks = get_h3_blocks(model_patcher)
    if not blocks:
        raise H3AttentionPatchError('MiniMax H3 has no main blocks')
    modules = []
    for index, block in enumerate(blocks):
        attn = getattr(block, 'attn', None)
        if attn is None:
            raise H3AttentionPatchError(
                'H3 block %d has no attention module' % index
            )
        missing = [
            name for name in REQUIRED_ATTRS if not hasattr(attn, name)
        ]
        if missing:
            raise H3AttentionPatchError(
                'H3 block %d attention is missing %s'
                % (index, ', '.join(missing))
            )
        expected = int(attn.heads) * int(attn.head_dim) * 3
        actual = getattr(attn.qkv_proj, 'out_features', None)
        if actual is not None and int(actual) != expected:
            raise H3AttentionPatchError(
                'H3 block %d qkv_proj projects to %d, expected %d'
                % (index, actual, expected)
            )
        modules.append(attn)
    return tuple(modules)


def configure_backend(
    model_patcher,
    backend,
    projector=None,
    *,
    projector_fallback_to_original=False,
    backend_fallback_to_dense=False,
    force_out_proj_int8=False,
    force_rebuild=False,
):
    '''Install or replace the package-owned H3 attention transaction.'''

    if backend is None:
        raise TypeError('backend must not be None')
    from .attention_forward import make_forward

    modules = validate(model_patcher)
    existing = getattr(model_patcher, 'object_patches', {})
    desired = (
        installation_signature(backend),
        installation_signature(projector),
        bool(projector_fallback_to_original),
        bool(backend_fallback_to_dense),
        bool(force_out_proj_int8),
    )
    conflicts = []
    patched = 0
    for index, module in enumerate(modules):
        key = key_for(index)
        current = existing.get(key)
        if current is not None and not getattr(current, OWNER_MARKER, False):
            conflicts.append(key)
            continue
        if current is not None:
            if (
                getattr(current, SIGNATURE_MARKER, None) == desired
                and not force_rebuild
            ):
                continue
            original = getattr(current, ORIGINAL_MARKER, None)
            if original is None:
                raise H3AttentionPatchError(
                    'installed H3 attention patch for %s has no recoverable original'
                    % key
                )
        else:
            original = module.forward
        forward = make_forward(
            module,
            index,
            backend=backend,
            projector=projector,
            fallback_forward=(
                original
                if projector_fallback_to_original
                else None
            ),
            backend_fallback_to_dense=backend_fallback_to_dense,
            force_out_proj_int8=force_out_proj_int8,
        )
        setattr(forward, OWNER_MARKER, True)
        setattr(forward, SIGNATURE_MARKER, desired)
        setattr(forward, ORIGINAL_MARKER, original)
        model_patcher.add_object_patch(key, forward)
        patched += 1

    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    options['h3_optimizations_attention_backend'] = getattr(
        backend,
        'name',
        type(backend).__name__,
    )
    options['h3_optimizations_preserved_attention_patches'] = conflicts
    if conflicts:
        logging.debug(
            '[H3 Optimizations] preserved %d foreign attention forward '
            'patch(es); H3 attention is disabled for those blocks',
            len(conflicts),
        )
    logging.debug(
        '[H3 Optimizations] resolved %d attention forwards: backend=%s '
        'projector=%s',
        patched,
        getattr(backend, 'name', type(backend).__name__),
        getattr(projector, 'name', 'standard_qkv'),
    )
    return backend, patched


def clear_backend(model_patcher):
    '''Remove only package-owned attention replacements.'''

    existing = getattr(model_patcher, 'object_patches', {})
    removed = 0
    foreign = []
    for key, value in tuple(existing.items()):
        if getattr(value, OWNER_MARKER, False):
            existing.pop(key)
            removed += 1
        elif key.startswith(BLOCKS_ATTR + '.') and key.endswith('.attn.forward'):
            foreign.append(key)
    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    options.pop('h3_optimizations_attention_backend', None)
    options['h3_optimizations_preserved_attention_patches'] = sorted(foreign)
    return removed
