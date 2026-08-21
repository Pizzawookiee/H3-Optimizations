'''Temporary compatibility boundary for the recognized upstream H3 V layout.'''

from __future__ import annotations

from dataclasses import dataclass
import inspect

import comfy.ldm.minimax.model as h3_model
import comfy.model_management
import comfy.quant_ops
from comfy.ldm.modules.attention import AttentionTensorContainer

from .patch import (
    H3AttentionPatchError,
    ORIGINAL_MARKER,
    OWNER_MARKER,
    SIGNATURE_MARKER,
    key_for,
    validate,
)


PROBE_REQUIRED = 'required'
PROBE_UNAVAILABLE = 'unavailable'
PROBE_UNNECESSARY = 'unnecessary'

KNOWN_BAD_FRAGMENTS = (
    'v = v.clone()',
    'v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))',
)


@dataclass(frozen=True)
class VLayoutProbe:
    state: str
    reason: str


@dataclass(frozen=True)
class VLayoutResolution:
    state: str
    reason: str
    patched_blocks: int = 0


def probe_v_layout(source_getter=inspect.getsource):
    try:
        source = source_getter(h3_model.Attention.forward)
    except (OSError, TypeError) as exc:
        return VLayoutProbe(
            PROBE_UNAVAILABLE,
            'upstream H3 source probe was unavailable: %s' % type(exc).__name__,
        )

    position = -1
    for fragment in KNOWN_BAD_FRAGMENTS:
        position = source.find(fragment, position + 1)
        if position < 0:
            return VLayoutProbe(
                PROBE_UNNECESSARY,
                'known-bad upstream H3 V signature was not present',
            )
    return VLayoutProbe(
        PROBE_REQUIRED,
        'known-bad upstream H3 V signature matched',
    )


def make_forward(module):
    def forward(x, rope_freqs=None, transformer_options={}):
        sequence = x.shape[0]
        inner = module.heads * module.head_dim
        q, k, v = module.qkv_proj(x).split(inner, dim=-1)
        v = v.view(sequence, module.heads, module.head_dim)
        if rope_freqs is not None:
            q = q.view(1, sequence, module.heads, module.head_dim)
            k = k.view(1, sequence, module.heads, module.head_dim)
            qw = comfy.model_management.cast_to(
                module.q_norm.weight,
                device=x.device,
            )
            kw = comfy.model_management.cast_to(
                module.k_norm.weight,
                device=x.device,
            )
            rot = rope_freqs.shape[-3] * 2
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q,
                    k,
                    rope_freqs,
                    qw,
                    kw,
                    epsilon=module.q_norm.eps,
                    rot_dim=rot,
                )
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q,
                    k,
                    rope_freqs,
                    qw,
                    kw,
                    epsilon=module.q_norm.eps,
                    rot_dim=rot,
                )
            q = q[0]
            k = k[0]
        else:
            q = module.q_norm(q.view(sequence, module.heads, module.head_dim))
            k = module.k_norm(k.view(sequence, module.heads, module.head_dim))
        v = v.clone()
        q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
        v = AttentionTensorContainer(
            v.transpose(0, 1).unsqueeze(0).contiguous()
        )
        out = h3_model.optimized_attention(
            q,
            k,
            v,
            module.heads,
            mask=None,
            skip_reshape=True,
            transformer_options=transformer_options,
        )
        return module.out_proj(out.squeeze(0))

    forward._h3_optimizations_v_layout_compat = True
    return forward


def not_applicable_v_layout(reason):
    return VLayoutResolution('not_applicable', reason)


def install_v_layout_compat(model_patcher):
    probe = probe_v_layout()
    if probe.state == PROBE_UNAVAILABLE:
        return VLayoutResolution(PROBE_UNAVAILABLE, probe.reason)
    if probe.state == PROBE_UNNECESSARY:
        return VLayoutResolution(PROBE_UNNECESSARY, probe.reason)

    modules = validate(model_patcher)
    existing = getattr(model_patcher, 'object_patches', {})
    desired = ('v_layout_compat', KNOWN_BAD_FRAGMENTS)
    owned = [
        index
        for index in range(len(modules))
        if getattr(existing.get(key_for(index)), OWNER_MARKER, False)
    ]
    if owned and len(owned) != len(modules):
        raise H3AttentionPatchError(
            'only %d of %d H3 attention blocks carry this patch'
            % (len(owned), len(modules))
        )

    if owned:
        installed = {
            getattr(existing[key_for(index)], SIGNATURE_MARKER, None)
            for index in owned
        }
        if installed == {desired}:
            return VLayoutResolution(
                'installed',
                'known-bad upstream H3 V signature matched',
            )
        originals = [
            getattr(existing[key_for(index)], ORIGINAL_MARKER, None)
            for index in range(len(modules))
        ]
        if any(original is None for original in originals):
            raise H3AttentionPatchError(
                'installed H3 attention patch has no recoverable original'
            )
    else:
        conflicts = [
            key_for(index)
            for index in range(len(modules))
            if key_for(index) in existing
        ]
        if conflicts:
            raise H3AttentionPatchError(
                'another patch already owns %s; remove one H3 attention patch'
                % conflicts[0]
            )
        originals = [module.forward for module in modules]

    for index, module in enumerate(modules):
        forward = make_forward(module)
        setattr(forward, OWNER_MARKER, True)
        setattr(forward, SIGNATURE_MARKER, desired)
        setattr(forward, ORIGINAL_MARKER, originals[index])
        model_patcher.add_object_patch(key_for(index), forward)

    return VLayoutResolution(
        'installed',
        'known-bad upstream H3 V signature matched',
        len(modules),
    )
