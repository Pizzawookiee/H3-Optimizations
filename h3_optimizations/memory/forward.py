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
    swiglu_eager,
)
from ..qkv.fp8 import FP8BindingError, HeldFP8MLP

LOG_PREFIX = '[H3 Optimizations]'


def _mod_row(values, selector, dtype):
    if torch.is_tensor(selector) and selector.device != values.device:
        selector = selector.to(device=values.device)
    return values[selector].to(dtype)


def _scale_shift(h, shift, scale, selector):
    scale_rows = _mod_row(scale, selector, h.dtype)
    h.mul_(1.0 + scale_rows)
    del scale_rows
    shift_rows = _mod_row(shift, selector, h.dtype)
    h.add_(shift_rows)
    del shift_rows
    return h


def _gate_add(x, other, gate, selector):
    gate_rows = _mod_row(gate, selector, x.dtype)
    x.addcmul_(other, gate_rows)
    del gate_rows
    return x


def _open_generic_held(block, sample, config):
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


def _open_fp8(block, sample, config):
    allow_float_conversion = not hasattr(block.mlp.fc1.weight, '_layout_cls')
    held = HeldFP8MLP(
        block.mlp,
        sample,
        allow_float_conversion=allow_float_conversion,
    )
    try:
        held.__enter__()
        return held, None
    except (FP8BindingError, RuntimeError, TypeError, ValueError) as exc:
        if config.strict:
            raise
        held.release()
        return None, '%s: %s' % (type(exc).__name__, exc)


def _open_mlp(block, sample, config):
    if config.fp8:
        held, error = _open_fp8(block, sample, config)
        return held, 'fp8' if held is not None else 'module', error

    if not config.convrot_2slice:
        held, error = _open_generic_held(block, sample, config)
        return held, 'held' if held is not None else 'module', error

    held = ConvRotTwoSliceMLP(block.mlp, sample)
    try:
        held.__enter__()
        return held, 'convrot', None
    except (RuntimeError, TypeError, ValueError) as exc:
        if config.strict:
            raise
        held, generic_error = _open_generic_held(block, sample, config)
        detail = '%s: %s' % (type(exc).__name__, exc)
        if generic_error is not None:
            detail += '; held fallback unavailable: %s' % generic_error
        return held, 'held' if held is not None else 'module', detail


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
        chunks = tuple(
            iter_mod_chunks(
                segments,
                x.shape[0],
                config.chunk_rows,
                alignment=config.alignment,
                mod_rows=shift_mlp.shape[0],
            )
        )

        # ComfyUI may attach a per-token modulation-row selector to masked H3
        # video/audio segments. Apply those gathers in the same bounded slabs as
        # the MLP so a long masked sequence does not materialize full-sequence
        # [tokens, hidden] shift/scale/gate tensors.
        h = block.norm1(x)
        for chunk in chunks:
            _scale_shift(
                h[chunk.start:chunk.stop],
                shift_msa,
                scale_msa,
                chunk.mod_row,
            )
        attn_out = block.attn(
            h,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )
        for chunk in chunks:
            _gate_add(
                x[chunk.start:chunk.stop],
                attn_out[chunk.start:chunk.stop],
                gate_msa,
                chunk.mod_row,
            )
        del h, attn_out

        held, mlp_path, held_error = _open_mlp(block, x[:1], config)
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
                    shift_mlp,
                    scale_mlp,
                    chunk.mod_row,
                )
                expanded = None
                if mlp_path == 'convrot':
                    out, _path = held.fc1_fc2(h)
                elif mlp_path == 'fp8':
                    out, _path = held.fc1_fc2(h, swiglu_eager)
                elif mlp_path == 'held':
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
                    gate_mlp,
                    chunk.mod_row,
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
