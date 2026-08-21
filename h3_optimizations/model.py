"""Exact MiniMax H3 model identity and block discovery."""

from __future__ import annotations


def get_minimax_h3_model(model_patcher, model_type=None):
    """Return the diffusion model only when it is an actual MiniMax H3 model."""

    try:
        diffusion_model = model_patcher.get_model_object("diffusion_model")
    except Exception:
        return None

    if model_type is None:
        try:
            from comfy.ldm.minimax.model import MiniMaxH3Model
        except ImportError:
            return None
        model_type = MiniMaxH3Model

    return diffusion_model if isinstance(diffusion_model, model_type) else None


def is_minimax_h3(model_patcher, model_type=None):
    return get_minimax_h3_model(model_patcher, model_type=model_type) is not None


def get_h3_blocks(model_patcher, model_type=None):
    model = get_minimax_h3_model(model_patcher, model_type=model_type)
    if model is None:
        return ()
    try:
        blocks = tuple(model_patcher.get_model_object("diffusion_model.blocks"))
    except Exception:
        blocks = tuple(getattr(model, "blocks", ()))
    return blocks
