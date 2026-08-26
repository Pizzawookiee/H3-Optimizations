'''Reversible H3 DiT-block patches for bounded MLP execution.'''

import logging

from .config import ActivationMemoryConfig
from .forward import make_forward
from ..model import get_h3_blocks, is_minimax_h3

BLOCKS_ATTR = 'diffusion_model.blocks'
OWNER_MARKER = '_h3_optimizations_memory'
SIGNATURE_MARKER = '_h3_optimizations_memory_signature'
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


def install(model_patcher, config=None):
    '''Patch every main H3 block; identical installation is idempotent.'''

    config = config or ActivationMemoryConfig()
    if not isinstance(config, ActivationMemoryConfig):
        raise TypeError('config must be ActivationMemoryConfig')
    blocks = validate(model_patcher)
    existing = getattr(model_patcher, 'object_patches', {})

    foreign = [
        key_for(index)
        for index in range(len(blocks))
        if key_for(index) in existing
        and not getattr(existing[key_for(index)], OWNER_MARKER, False)
    ]
    if foreign:
        raise H3MemoryPatchError(
            'another patch already owns %s; remove one H3 memory patch'
            % foreign[0]
        )

    owned = [
        index
        for index in range(len(blocks))
        if getattr(existing.get(key_for(index)), OWNER_MARKER, False)
    ]
    if owned:
        if len(owned) != len(blocks):
            raise H3MemoryPatchError(
                'only %d of %d H3 blocks carry this memory patch'
                % (len(owned), len(blocks))
            )
        installed = {
            getattr(existing[key_for(index)], SIGNATURE_MARKER, None)
            for index in owned
        }
        if installed == {config.signature}:
            return 0
        raise H3MemoryPatchError(
            'H3 Memory Optimization is already configured for %s; requested '
            '%s. Remove the earlier node instead of relying on node order.'
            % (
                sorted(str(item) for item in installed),
                config.signature,
            )
        )

    for index, block in enumerate(blocks):
        model_patcher.add_object_patch(
            key_for(index),
            make_forward(
                block,
                index,
                config,
                original_forward=block.forward,
            ),
        )
    logging.debug(
        '[H3 Optimizations] patched %d MLP blocks: mode=%s chunk_rows=%d',
        len(blocks),
        config.mode,
        config.chunk_rows,
    )
    return len(blocks)
