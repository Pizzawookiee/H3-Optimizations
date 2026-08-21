'''H3 DiT block forward with bounded MLP activations.'''

import logging

import torch

import comfy.model_management

from .chunks import iter_mod_chunks, validate_mod_segments
from .linear import (
    ConvRotTwoSliceMLP,
    HeldMLP,
    UnsafeHeldWeights,
    bind_convrot_mlp,
    module_fc1,
    module_swiglu_fc2,
)

LOG_PREFIX = '[H3 Optimizations]'


def _scale_shift(h, shift, scale):
    return h.mul_(1.0 + scale.to(h.dtype)).add_(shift.to(h.dtype))


def _gate_add(x, other, gate):
    return x.addcmul_(other, gate.to(x.dtype))


def _open_generic_held(block, sample, config):
    if not config.prefer_held_weights:
        return None, None
    held = HeldMLP(block.mlp, sample)
    try:
        held.__enter__()
        return held, None
    except UnsafeHeldWeights as exc:
        return None, str(exc)
    except (RuntimeError, TypeError, ValueError) as exc:
        if config.strict:
            raise
        return None, '%s: %s' % (type(exc).__name__, exc)


def _open_mlp(block, sample, config):
    if not config.convrot_2slice:
        held, error = _open_generic_held(block, sample, config)
        return held, False, error

    held = ConvRotTwoSliceMLP(block.mlp, sample)
    try:
        held.__enter__()
        return held, True, None
    except (RuntimeError, TypeError, ValueError) as exc:
        if config.strict:
            raise
        held, generic_error = _open_generic_held(block, sample, config)
        detail = '%s: %s' % (type(exc).__name__, exc)
        if generic_error is not None:
            detail += '; held fallback unavailable: %s' % generic_error
        return held, False, detail


def make_forward(block, layer_index, config, original_forward=None):
    '''Build an unbound replacement for one MiniMax H3 DiT block.'''

    original_forward = original_forward or block.forward
    if config.convrot_2slice and isinstance(block.mlp, torch.nn.Module):
        bind_convrot_mlp(block.mlp)

    def forward(
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options={},
    ):
        if comfy.model_management.in_training:
            raise RuntimeError(
                'H3 Memory Optimization is inference-only; training requires '
                'the original block forward'
            )

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = block.adaln_proj(t_emb)
        segments = validate_mod_segments(
            mod_segments,
            x.shape[0],
            mod_rows=shift_msa.shape[0],
        )

        h = block.norm1(x)
        for start, stop, row in segments:
            _scale_shift(h[start:stop], shift_msa[row], scale_msa[row])
        attn_out = block.attn(
            h,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )
        for start, stop, row in segments:
            _gate_add(x[start:stop], attn_out[start:stop], gate_msa[row])
        del h, attn_out

        chunks = iter_mod_chunks(
            segments,
            x.shape[0],
            config.chunk_rows,
            alignment=config.alignment,
            mod_rows=shift_mlp.shape[0],
        )
        held, use_convrot, held_error = _open_mlp(block, x[:1], config)
        if held_error is not None:
            logging.warning(
                '%s block %d selected a format-compatible MLP fallback: %s',
                LOG_PREFIX,
                layer_index,
                held_error,
            )

        try:
            for chunk in chunks:
                h = block.norm2(x[chunk.start:chunk.stop])
                _scale_shift(
                    h,
                    shift_mlp[chunk.mod_row],
                    scale_mlp[chunk.mod_row],
                )
                expanded = None
                if use_convrot:
                    out, _path = held.fc1_fc2(h)
                elif held is not None:
                    expanded = held.fc1(h)
                    out, _path = held.fc2_swiglu(
                        expanded,
                        native=config.native_swiglu,
                    )
                else:
                    expanded = module_fc1(block.mlp, h)
                    out, _path = module_swiglu_fc2(
                        block.mlp,
                        expanded,
                        native=config.native_swiglu,
                    )
                _gate_add(
                    x[chunk.start:chunk.stop],
                    out,
                    gate_mlp[chunk.mod_row],
                )
                del h, expanded, out
        finally:
            if held is not None:
                held.__exit__(None, None, None)
        return x

    forward._h3_optimizations_memory = True
    forward._h3_optimizations_memory_signature = config.signature
    forward._h3_optimizations_memory_layer = int(layer_index)
    forward._h3_optimizations_memory_original = original_forward
    return forward
