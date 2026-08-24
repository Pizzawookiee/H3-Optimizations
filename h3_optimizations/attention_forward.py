'''H3 attention forward with negotiated projected-QKV fallback.'''

import comfy.ldm.minimax.model as h3_model
import comfy.model_management
import comfy.quant_ops

from . import diagnostics
from .attention import AttentionBackendUnavailable
from .ordering_probe import has_ordering_observer, observe_attention


DENSE_KITCHEN_PREQUANTIZED = 'comfy_kitchen_int8_prequantized'


def finish_qkv_projection(module, projected, rope_freqs):
    seq = projected.shape[0]
    inner = module.heads * module.head_dim
    q, k, v = projected.split(inner, dim=-1)
    v = v.view(seq, module.heads, module.head_dim)

    if rope_freqs is not None:
        if comfy.model_management.in_training:
            raise RuntimeError('H3 optimized attention is inference-only')
        q = q.view(1, seq, module.heads, module.head_dim)
        k = k.view(1, seq, module.heads, module.head_dim)
        qw = comfy.model_management.cast_to(
            module.q_norm.weight,
            device=projected.device,
        )
        kw = comfy.model_management.cast_to(
            module.k_norm.weight,
            device=projected.device,
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


def project_qkv(module, x, rope_freqs):
    with diagnostics.stage('qkv_linear'):
        projected = module.qkv_proj(x)
    with diagnostics.stage('qk_norm_rope'):
        return finish_qkv_projection(module, projected, rope_freqs)


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


def flatten_attention_output(module, out, source):
    """Reach [batch, sequence, heads * head_dim] from whatever the kernel wrote.

    An ``nhd`` kernel output already has sequence ahead of heads, so the
    flatten is a view. An ``hnd`` output needs a transpose whose last two
    dimensions cannot merge, so ``reshape`` copies a second full-sequence BF16
    tensor -- at the production shape that is the single largest avoidable
    allocation in the block.
    """
    if out.ndim != 4:
        raise RuntimeError(
            '%s returned rank-%d output; expected HND rank 4'
            % (source, out.ndim)
        )
    # One expression for both storage layouts. Over head-major storage the
    # dimensions cannot merge and `reshape` copies; over sequence-major
    # storage the transpose lands on contiguous memory and this is a view.
    return out.transpose(1, 2).reshape(
        out.shape[0], out.shape[2], module.heads * module.head_dim
    )


def _finish_projected(module, backend, prepared):
    # A streamed backend may own the full attention -> out_proj lifetime and
    # return the final hidden-size tensor directly. This is deliberately
    # opt-in so Kitchen, Triton, BF16 and legacy projected contracts stay put.
    execute_projected = getattr(backend, 'execute_projected', None)
    if execute_projected is not None:
        direct = execute_projected(module, prepared)
        if direct is not None:
            if direct.ndim != 2:
                raise RuntimeError(
                    '%s returned rank-%d direct projected output; expected rank 2'
                    % (
                        getattr(backend, 'name', type(backend).__name__),
                        direct.ndim,
                    )
                )
            return direct

    name = getattr(backend, 'name', type(backend).__name__)
    raw = backend.execute(prepared)
    out = flatten_attention_output(module, raw, name)
    # Only the flattened view is needed from here. Under `hnd` the reshape
    # already copied, so this returns a full-sequence buffer to the allocator
    # before the projection asks for one; under `nhd` `out` aliases `raw` and
    # this is a no-op.
    del raw
    if getattr(backend, 'release_carrier_before_out_proj', False):
        release = getattr(prepared, 'release', None)
        if release is None:
            raise RuntimeError(
                '%s asked to release its carrier but %s cannot release one'
                % (name, type(prepared).__name__)
            )
        release()
    with diagnostics.stage('attention_out'):
        return module.out_proj(out.squeeze(0))


def _finish_bf16_projected(
    module,
    backend,
    projected,
    *,
    layer_index,
    transformer_options,
):
    """Consume chunked/native BF16 QKV without running QKV projection again."""
    q, k, v = projected.q, projected.k, projected.v
    backend_name = getattr(backend, 'name', None)

    # The dense Kitchen backend object exists here only because that is the
    # package-owned projector slot used by Memory Optimization. Preserve
    # precision must not force Kitchen attention: feed the already-projected
    # BF16 tensors to whatever attention Comfy/upstream currently selected.
    if backend is None or backend_name == DENSE_KITCHEN_PREQUANTIZED:
        raw = _legacy_attention(
            module,
            q,
            k,
            v,
            transformer_options,
        )
        source = 'existing_attention_bf16'
    else:
        prepared = backend.prepare(
            q,
            k,
            v,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )
        try:
            raw = backend.execute(prepared)
        finally:
            del prepared
        source = backend_name or type(backend).__name__

    # The backend launch has consumed the BF16 inputs. Dropping the wrapper
    # here lets the stream-ordered allocator reclaim them before out_proj.
    del projected, q, k, v
    out = flatten_attention_output(module, raw, source)
    del raw
    with diagnostics.stage('attention_out'):
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
        with diagnostics.stage('attention_total'):
            return _forward(x, rope_freqs, transformer_options)

    def _forward(x, rope_freqs, transformer_options):
        transformer_options = (
            transformer_options if transformer_options is not None else {}
        )
        ordering_probe = has_ordering_observer(transformer_options)
        if projector is not None and not ordering_probe:
            projected = _project_or_none(
                projector,
                module,
                x,
                rope_freqs,
                layer_index=layer_index,
                transformer_options=transformer_options,
            )
            if projected is not None:
                from .qkv.bf16 import PreparedBF16QKV

                if isinstance(projected, PreparedBF16QKV):
                    return _finish_bf16_projected(
                        module,
                        backend,
                        projected,
                        layer_index=layer_index,
                        transformer_options=transformer_options,
                    )
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
        if ordering_probe:
            observe_attention(
                layer_index,
                transformer_options,
                q,
                k,
                v,
            )

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
            out = flatten_attention_output(
                module,
                out_hnd,
                getattr(backend, 'name', type(backend).__name__),
            )
            del out_hnd

        with diagnostics.stage('attention_out'):
            return module.out_proj(out.squeeze(0))

    forward._h3_optimizations_attention = True
    forward._h3_optimizations_layer_index = int(layer_index)
    forward._h3_optimizations_backend = getattr(backend, 'name', None)
    forward._h3_optimizations_projector = getattr(projector, 'name', None)
    return forward
