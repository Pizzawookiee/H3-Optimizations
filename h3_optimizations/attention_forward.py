'''H3 attention forward with negotiated projected-QKV fallback.'''

import comfy.ldm.minimax.model as h3_model
import comfy.model_management
import comfy.quant_ops

from .attention import AttentionBackendUnavailable


def project_qkv(module, x, rope_freqs):
    seq = x.shape[0]
    inner = module.heads * module.head_dim
    q, k, v = module.qkv_proj(x).split(inner, dim=-1)
    v = v.view(seq, module.heads, module.head_dim)

    if rope_freqs is not None:
        if comfy.model_management.in_training:
            raise RuntimeError('H3 optimized attention is inference-only')
        q = q.view(1, seq, module.heads, module.head_dim)
        k = k.view(1, seq, module.heads, module.head_dim)
        qw = comfy.model_management.cast_to(
            module.q_norm.weight,
            device=x.device,
        )
        kw = comfy.model_management.cast_to(
            module.k_norm.weight,
            device=x.device,
        )
        rot = rope_freqs.shape[-3] * 2
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
        q = module.q_norm(
            q.view(seq, module.heads, module.head_dim)
        )
        k = module.k_norm(
            k.view(seq, module.heads, module.head_dim)
        )
    return q, k, v


def to_hnd(q, k, v):
    return (
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
    )


def _legacy_attention(module, q, k, v, transformer_options, attention=None):
    attention_fn = (
        attention if attention is not None else h3_model.optimized_attention
    )
    return attention_fn(
        q,
        k,
        v,
        module.heads,
        mask=None,
        skip_reshape=True,
        transformer_options=transformer_options,
    )


def _project_or_none(
    projector,
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    transformer_options,
):
    callback = getattr(projector, 'try_project', None)
    if callback is None:
        callback = projector.project
    return callback(
        module,
        x,
        rope_freqs,
        layer_index=layer_index,
        transformer_options=transformer_options,
    )


def _finish_projected(module, backend, prepared):
    out_hnd = backend.execute(prepared)
    if out_hnd.ndim != 4:
        raise RuntimeError(
            '%s returned rank-%d output; expected HND rank 4'
            % (
                getattr(backend, 'name', type(backend).__name__),
                out_hnd.ndim,
            )
        )
    out = out_hnd.transpose(1, 2).reshape(
        out_hnd.shape[0],
        out_hnd.shape[2],
        module.heads * module.head_dim,
    )
    return module.out_proj(out.squeeze(0))


def make_forward(
    module,
    layer_index,
    backend=None,
    attention=None,
    projector=None,
    fallback_forward=None,
    backend_fallback_to_dense=False,
):
    if backend is not None and attention is not None:
        raise ValueError('pass either backend or attention, not both')
    if projector is not None and backend is None:
        raise ValueError('a fused QKV projector requires a consuming backend')
    bind_projector = getattr(projector, 'bind', None)
    if bind_projector is not None:
        bind_projector(module)

    def forward(x, rope_freqs=None, transformer_options=None):
        transformer_options = (
            transformer_options if transformer_options is not None else {}
        )
        if projector is not None:
            projected = _project_or_none(
                projector,
                module,
                x,
                rope_freqs,
                layer_index=layer_index,
                transformer_options=transformer_options,
            )
            if projected is not None:
                prepared = backend.prepare_projected(
                    projected,
                    layer_index=layer_index,
                    transformer_options=transformer_options,
                )
                del projected
                try:
                    return _finish_projected(module, backend, prepared)
                finally:
                    del prepared
            if fallback_forward is not None:
                return fallback_forward(
                    x,
                    rope_freqs=rope_freqs,
                    transformer_options=transformer_options,
                )

        q, k, v = project_qkv(module, x, rope_freqs)
        q, k, v = to_hnd(q, k, v)

        if backend is None:
            out = _legacy_attention(
                module,
                q,
                k,
                v,
                transformer_options,
                attention=attention,
            )
        else:
            fallback_inputs_available = True
            try:
                prepared = backend.prepare(
                    q,
                    k,
                    v,
                    layer_index=layer_index,
                    transformer_options=transformer_options,
                )
                retain_fallback_inputs = (
                    backend_fallback_to_dense
                    and backend.requires_fallback_inputs(prepared)
                )
                if not retain_fallback_inputs:
                    del q, k, v
                    fallback_inputs_available = False
                try:
                    out_hnd = backend.execute(prepared)
                finally:
                    del prepared
            except AttentionBackendUnavailable:
                if (
                    not backend_fallback_to_dense
                    or not fallback_inputs_available
                ):
                    raise
                v = v.contiguous()
                out_hnd = _legacy_attention(
                    module,
                    q,
                    k,
                    v,
                    transformer_options,
                )
            if out_hnd.ndim != 4:
                raise RuntimeError(
                    '%s returned rank-%d output; expected HND rank 4'
                    % (
                        getattr(backend, 'name', type(backend).__name__),
                        out_hnd.ndim,
                    )
                )
            out = out_hnd.transpose(1, 2).reshape(
                out_hnd.shape[0],
                out_hnd.shape[2],
                module.heads * module.head_dim,
            )

        return module.out_proj(out.squeeze(0))

    forward._h3_optimizations_attention = True
    forward._h3_optimizations_layer_index = int(layer_index)
    forward._h3_optimizations_backend = getattr(backend, 'name', None)
    forward._h3_optimizations_projector = getattr(projector, 'name', None)
    return forward
