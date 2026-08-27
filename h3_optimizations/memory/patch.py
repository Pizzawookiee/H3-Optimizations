'''Reversible H3 DiT-block patches for bounded MLP execution.'''

import logging

from .config import ActivationMemoryConfig
from .forward import make_forward
from ..model import get_h3_blocks, is_minimax_h3

BLOCKS_ATTR = 'diffusion_model.blocks'
OWNER_MARKER = '_h3_optimizations_memory'
SIGNATURE_MARKER = '_h3_optimizations_memory_signature'
ORIGINAL_MARKER = '_h3_optimizations_memory_original'
REQUIRED_BLOCK_ATTRS = ('norm1', 'norm2', 'attn', 'mlp', 'adaln_proj')
REQUIRED_MLP_ATTRS = ('fc1', 'fc2')


class H3MemoryPatchError(RuntimeError):
    pass


def key_for(index):
    return '%s.%d.forward' % (BLOCKS_ATTR, int(index))


def validate(model_patcher):
    if not is_minimax_h3(model_patcher):
        raise H3MemoryPatchError(
            'H3 Memory Optimization can only patch MiniMaxH3Model'
        )
    blocks = get_h3_blocks(model_patcher)
    if not blocks:
        raise H3MemoryPatchError('MiniMax H3 has no main blocks')
    for index, block in enumerate(blocks):
        missing = [
            name for name in REQUIRED_BLOCK_ATTRS if not hasattr(block, name)
        ]
        if missing:
            raise H3MemoryPatchError(
                'H3 block %d is missing %s'
                % (index, ', '.join(missing))
            )
        missing = [
            name for name in REQUIRED_MLP_ATTRS
            if not hasattr(block.mlp, name)
        ]
        if missing:
            raise H3MemoryPatchError(
                'H3 block %d MLP is missing %s'
                % (index, ', '.join(missing))
            )
    return blocks


def install(model_patcher, config=None, *, force_rebuild=False):
    '''Patch every main H3 block; identical installation is idempotent.'''

    config = config or ActivationMemoryConfig()
    if not isinstance(config, ActivationMemoryConfig):
        raise TypeError('config must be ActivationMemoryConfig')
    blocks = validate(model_patcher)
    existing = getattr(model_patcher, 'object_patches', {})

    foreign = []
    patched = 0
    for index, block in enumerate(blocks):
        key = key_for(index)
        current = existing.get(key)
        if current is not None and not getattr(current, OWNER_MARKER, False):
            foreign.append(key)
            continue
        if current is not None:
            installed = getattr(current, SIGNATURE_MARKER, None)
            if installed == config.signature and not force_rebuild:
                continue
            original = getattr(current, ORIGINAL_MARKER, None)
            if original is None:
                raise H3MemoryPatchError(
                    'installed H3 memory patch for %s has no recoverable original'
                    % key
                )
        else:
            original = block.forward
        model_patcher.add_object_patch(
            key,
            make_forward(
                block,
                index,
                config,
                original_forward=original,
            ),
        )
        patched += 1
    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    options['h3_optimizations_preserved_block_patches'] = foreign
    if foreign:
        logging.debug(
            '[H3 Optimizations] preserved %d foreign block forward patch(es); '
            'bounded MLP execution is disabled for those blocks',
            len(foreign),
        )
    logging.debug(
        '[H3 Optimizations] patched %d MLP blocks: mode=%s chunk_rows=%d',
        patched,
        config.mode,
        config.chunk_rows,
    )
    return patched
