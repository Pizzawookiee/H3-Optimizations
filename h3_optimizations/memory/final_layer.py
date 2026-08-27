'''Bounded MiniMax H3 FinalLayer execution.'''

import logging

import torch

from ..model import get_minimax_h3_model


FINAL_LAYER_KEY = 'diffusion_model.final_layer.forward'
OWNER_MARKER = '_h3_optimizations_final_layer'
SIGNATURE_MARKER = '_h3_optimizations_final_layer_signature'


class H3FinalLayerPatchError(RuntimeError):
    pass


def _selector(value, start, stop):
    return value if value.ndim == 1 else value[start:stop]


def chunked_final_layer(layer, x, t_emb, video_seg, audio_seg, chunk_rows):
    shift, scale = layer.adaln_proj(t_emb)

    def project(segment, output):
        first, last, row = segment
        selected_shift = shift[row]
        selected_scale = scale[row]
        pieces = []
        for start in range(first, last, int(chunk_rows)):
            stop = min(start + int(chunk_rows), last)
            local_start = start - first
            local_stop = stop - first
            value = (
                layer.norm(x[start:stop])
                * (1.0 + _selector(selected_scale, local_start, local_stop))
                + _selector(selected_shift, local_start, local_stop)
            ).to(torch.float32)
            pieces.append(output(value))
        if not pieces:
            value = (
                layer.norm(x[first:last])
                * (1.0 + _selector(selected_scale, 0, 0))
                + _selector(selected_shift, 0, 0)
            ).to(torch.float32)
            return output(value)
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    return project(video_seg, layer.video_out), project(audio_seg, layer.audio_out)


def _chunk_count(rows, chunk_rows):
    if rows <= 0:
        return 0
    return (int(rows) + int(chunk_rows) - 1) // int(chunk_rows)


def make_forward(layer, chunk_rows):
    signature = int(chunk_rows)
    # One line the first time the patched forward actually executes. The
    # install-time message only proves the patch was attached; it stays silent
    # when routing sends the forward somewhere else, which is exactly the case
    # that has to be visible. Reporting the chunk counts also separates real
    # chunking from a segment that fits in one chunk and is therefore bounded
    # in name only.
    announced = []

    def forward(x, t_emb, video_seg, audio_seg):
        if not announced:
            announced.append(True)
            video_rows = int(video_seg[1]) - int(video_seg[0])
            audio_rows = int(audio_seg[1]) - int(audio_seg[0])
            logging.info(
                '[H3 Optimizations] chunked FinalLayer ran: %d rows, '
                'video %d in %d chunk(s), audio %d in %d chunk(s), '
                'chunk_rows=%d',
                int(x.shape[0]),
                video_rows,
                _chunk_count(video_rows, signature),
                audio_rows,
                _chunk_count(audio_rows, signature),
                signature,
            )
        return chunked_final_layer(
            layer,
            x,
            t_emb,
            video_seg,
            audio_seg,
            signature,
        )

    setattr(forward, OWNER_MARKER, True)
    setattr(forward, SIGNATURE_MARKER, signature)
    return forward


def install(model_patcher, chunk_rows, *, force_rebuild=False):
    '''Patch FinalLayer once; identical installation is idempotent.'''

    chunk_rows = int(chunk_rows)
    if chunk_rows <= 0:
        raise ValueError('chunk_rows must be positive')
    model = get_minimax_h3_model(model_patcher)
    if model is None:
        raise H3FinalLayerPatchError(
            'H3 Memory Optimization can only patch MiniMaxH3Model'
        )
    layer = getattr(model, 'final_layer', None)
    if layer is None:
        raise H3FinalLayerPatchError('MiniMax H3 has no final layer')

    existing = getattr(model_patcher, 'object_patches', {}).get(FINAL_LAYER_KEY)
    if existing is not None:
        if not getattr(existing, OWNER_MARKER, False):
            options = model_patcher.model_options['transformer_options'] = (
                model_patcher.model_options.get('transformer_options', {}).copy()
            )
            options['h3_optimizations_preserved_final_layer_patch'] = True
            logging.debug(
                '[H3 Optimizations] preserved foreign %s; FinalLayer chunking '
                'is disabled',
                FINAL_LAYER_KEY,
            )
            return False
        installed = getattr(existing, SIGNATURE_MARKER, None)
        if installed == chunk_rows and not force_rebuild:
            return False

    model_patcher.add_object_patch(
        FINAL_LAYER_KEY,
        make_forward(layer, chunk_rows),
    )
    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    options['h3_optimizations_preserved_final_layer_patch'] = False
    logging.debug(
        '[H3 Optimizations] patched FinalLayer: chunk_rows=%d',
        chunk_rows,
    )
    return True
