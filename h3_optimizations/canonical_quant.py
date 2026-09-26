"""One-time canonical ConvRot INT8 conversion for Force quant."""

from __future__ import annotations

import logging

import torch

import comfy.model_management
import comfy.quant_ops
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

from .model import get_h3_blocks
from .plan import (
    FUSED_QKV_FORCE_QUANT,
    MLP_MEMORY_FORCE_QUANT,
    QKV_STREAMING_OFF,
)
from .qkv.formats import describe_linear


CANONICAL_QUANT_KEY = 'h3_optimizations_canonical_quantization'
LOG_PREFIX = '[H3 Optimizations]'

_QKV_TARGETS = (
    ('attn.qkv_proj', lambda block: block.attn.qkv_proj),
    ('attn.out_proj', lambda block: block.attn.out_proj),
)
_MLP_TARGETS = (
    ('mlp.fc1', lambda block: block.mlp.fc1),
    ('mlp.fc2', lambda block: block.mlp.fc2),
)


def _requested_targets(plan):
    memory = plan.memory
    if memory is None:
        return ()
    targets = []
    if (
        memory.qkv_streaming != QKV_STREAMING_OFF
        and memory.fused_qkv == FUSED_QKV_FORCE_QUANT
    ):
        targets.extend(_QKV_TARGETS)
    if memory.mlp_memory == MLP_MEMORY_FORCE_QUANT:
        targets.extend(_MLP_TARGETS)
    return tuple(targets)


def _plain_float_targets(blocks, targets):
    pending = []
    for block_index, block in enumerate(blocks):
        for name, getter in targets:
            module = getter(block)
            if describe_linear(module).plain_float:
                pending.append((block_index, name, module))
    return pending


def _tensor_bytes(tensor):
    return int(tensor.numel()) * int(tensor.element_size())


def _quantized_bytes(weight):
    return sum(
        _tensor_bytes(tensor)
        for tensor in TensorWiseINT8Layout.get_plain_tensors(weight)
    )


def _conversion_device(model_patcher):
    if not bool(getattr(comfy.quant_ops, '_CK_AVAILABLE', False)):
        raise RuntimeError(
            'Force quant canonical conversion requires Comfy Kitchen.'
        )
    device = torch.device(model_patcher.load_device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError(
            'Force quant canonical conversion requires a CUDA or ROCm device.'
        )
    return device


def _deep_clone(model_patcher):
    try:
        return model_patcher.clone(force_deepcopy=True)
    except RuntimeError as exc:
        if getattr(model_patcher, 'cached_patcher_init', None) is None:
            raise RuntimeError(
                'Force quant canonical conversion requires a model loader with '
                'reload support. Use a core UNET loader or a native ConvRot INT8 '
                'checkpoint.'
            ) from exc
        raise


def _convert_weight(module, device, offload_device):
    source = module.weight
    source_bytes = _tensor_bytes(source)
    source_device = source.to(device=device)
    quantized = QuantizedTensor.from_float(
        source_device,
        'TensorWiseINT8Layout',
        scale='recalculate',
        is_weight=True,
        per_channel=True,
        convrot=True,
        convrot_groupsize=256,
    )
    if quantized.device != offload_device:
        quantized = quantized.to(device=offload_device)
    quantized_bytes = _quantized_bytes(quantized)
    module.register_parameter(
        'weight',
        torch.nn.Parameter(quantized, requires_grad=False),
    )
    return source_bytes, quantized_bytes


def clone_with_force_quant(model_patcher, plan):
    """Clone a model and permanently quantize requested floating H3 linears.

    A fresh model instance keeps the input and sibling graph branches untouched.
    Only one source weight is staged on the accelerator at a time.
    """
    targets = _requested_targets(plan)
    blocks = get_h3_blocks(model_patcher)
    if not targets or not _plain_float_targets(blocks, targets):
        return model_patcher.clone()

    device = _conversion_device(model_patcher)
    patched = _deep_clone(model_patcher)
    pending = _plain_float_targets(get_h3_blocks(patched), targets)
    converted = []
    original_bytes = 0
    quantized_bytes = 0
    offload_device = torch.device(patched.offload_device)
    for block_index, name, module in pending:
        before, after = _convert_weight(module, device, offload_device)
        original_bytes += before
        quantized_bytes += after
        converted.append('blocks.%d.%s.weight' % (block_index, name))

    patched.size = 0
    patched.model.model_loaded_weight_memory = 0
    report = {
        'format': 'ConvRot-256 INT8',
        'converted_weights': converted,
        'converted_count': len(converted),
        'original_bytes': original_bytes,
        'quantized_bytes': quantized_bytes,
        'released_bytes': original_bytes - quantized_bytes,
    }
    patched.model_options[CANONICAL_QUANT_KEY] = report
    comfy.model_management.soft_empty_cache()
    logging.info(
        '%s Force quant converted %d H3 weights to canonical ConvRot-256 INT8 '
        'and released %.2f GiB from the converted model branch.',
        LOG_PREFIX,
        len(converted),
        report['released_bytes'] / 2**30,
    )
    return patched
