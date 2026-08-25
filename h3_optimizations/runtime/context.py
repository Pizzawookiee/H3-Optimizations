'''Per-request sampler-step and packed-layout context for H3 Sparse Attention.'''

from dataclasses import dataclass
import logging
import threading

import torch

import comfy.utils
import comfy.ops

from .layout import resolve_layout

RUNTIME_KEY = 'h3_optimizations_runtime'
RUNTIME_SESSION_KEY = 'h3_optimizations_runtime_session'
WRAPPER_KEY = 'h3_optimizations_runtime_context'
OUTER_WRAPPER_KEY = 'h3_optimizations_request_boundary'
LOG_PREFIX = '[H3 Optimizations]'


_FINAL_LAYER_STREAM = threading.local()
_ORIGINAL_FINAL_LAYER_FORWARD = None
_FINAL_LAYER_PATCHED = False


def _streamed_final_layer_rows(transformer_options):
    """Return streamed chunk rows, or 0 outside streamed attention mode."""
    try:
        from ..streamed_kitchen import (
            ATTENTION_MEMORY_STREAMED,
            requested_memory_mode,
            requested_query_chunk_rows,
        )
        if requested_memory_mode(transformer_options) != ATTENTION_MEMORY_STREAMED:
            return 0
        return int(requested_query_chunk_rows(transformer_options))
    except (ImportError, AttributeError, KeyError, TypeError, ValueError):
        return 0


def _slice_final_mod_row(row, *, segment_start, segment_stop, chunk_start, chunk_stop, total_rows):
    """Preserve MiniMax _mod_row semantics for scalar or per-token selectors."""
    if not torch.is_tensor(row) or row.ndim == 0:
        return row

    count = int(row.shape[0])
    segment_rows = int(segment_stop - segment_start)
    if count == segment_rows:
        local_start = int(chunk_start - segment_start)
        local_stop = int(chunk_stop - segment_start)
        return row[local_start:local_stop]
    if count == int(total_rows):
        return row[chunk_start:chunk_stop]
    raise RuntimeError(
        '%s chunked FinalLayer received selector length %d for segment %d:%d '
        'over %d total rows'
        % (
            LOG_PREFIX,
            count,
            int(segment_start),
            int(segment_stop),
            int(total_rows),
        )
    )


def _final_mod_row(vecs, row, dtype):
    return vecs[row].to(dtype)


def _chunked_final_layer_forward(self, x, t_emb, video_seg, audio_seg):
    """Chunked MiniMax FinalLayer with one Comfy weight live at a time.

    Phase 1 normalizes/modulates target rows back into x in BF16.
    Phase 2 projects video rows with only video_out held.
    Phase 3 projects audio rows with only audio_out held.

    This avoids simultaneous aliases into Comfy's reusable cast buffer while
    retaining bounded FP32 activation memory.
    """
    rows = int(getattr(_FINAL_LAYER_STREAM, 'chunk_rows', 0) or 0)
    if rows <= 0:
        return _ORIGINAL_FINAL_LAYER_FORWARD(
            self, x, t_emb, video_seg, audio_seg
        )

    shift, scale = self.adaln_proj(t_emb)

    def _segment_bounds(seg, label):
        a, b, row = seg
        a = int(a)
        b = int(b)
        if a < 0 or b < a or b > int(x.shape[0]):
            raise RuntimeError(
                '%s chunked FinalLayer invalid %s segment %d:%d for %d rows'
                % (LOG_PREFIX, label, a, b, int(x.shape[0]))
            )
        return a, b, row

    with comfy.ops.CastBiasWeightContext(
        self.norm if self.norm.weight is not None else None,
        dtype=x.dtype,
        device=x.device,
        bias_dtype=x.dtype,
        offloadable=True,
    ) as (norm_weight, _norm_bias):

        def normalize_segment(seg, label):
            a, b, row = _segment_bounds(seg, label)
            for start in range(a, b, rows):
                stop = min(start + rows, b)
                local_row = _slice_final_mod_row(
                    row,
                    segment_start=a,
                    segment_stop=b,
                    chunk_start=start,
                    chunk_stop=stop,
                    total_rows=int(x.shape[0]),
                )
                local = torch.nn.functional.rms_norm(
                    x[start:stop],
                    self.norm.normalized_shape,
                    norm_weight,
                    self.norm.eps,
                )
                local.mul_(
                    1.0
                    + _final_mod_row(
                        scale,
                        local_row,
                        local.dtype,
                    )
                )
                local.add_(
                    _final_mod_row(
                        shift,
                        local_row,
                        local.dtype,
                    )
                )
                x[start:stop].copy_(local)
                del local

        normalize_segment(video_seg, 'video')
        normalize_segment(audio_seg, 'audio')

    def project_segment(seg, module, label):
        a, b, _row = _segment_bounds(seg, label)

        with comfy.ops.CastBiasWeightContext(
            module,
            dtype=torch.float32,
            device=x.device,
            bias_dtype=torch.float32,
            offloadable=True,
        ) as (weight, bias):
            output = None
            for start in range(a, b, rows):
                stop = min(start + rows, b)

                hv = x[start:stop].to(torch.float32)
                local = torch.nn.functional.linear(hv, weight, bias)
                del hv

                if output is None:
                    output = local.new_empty(
                        b - a,
                        int(local.shape[-1]),
                    )
                if int(local.shape[0]) != int(stop - start):
                    raise RuntimeError(
                        '%s chunked FinalLayer %s head returned %d rows for %d:%d'
                        % (
                            LOG_PREFIX,
                            label,
                            int(local.shape[0]),
                            start,
                            stop,
                        )
                    )
                output[start - a:stop - a].copy_(local)
                del local

            if output is None:
                hv = x[a:b].to(torch.float32)
                try:
                    return torch.nn.functional.linear(hv, weight, bias)
                finally:
                    del hv
            return output

    video = project_segment(video_seg, self.video_out, 'video')
    audio = project_segment(audio_seg, self.audio_out, 'audio')
    return video, audio

def _install_chunked_final_layer_patch():
    global _ORIGINAL_FINAL_LAYER_FORWARD, _FINAL_LAYER_PATCHED
    if _FINAL_LAYER_PATCHED:
        return
    import comfy.ldm.minimax.model as h3_model

    current = h3_model.FinalLayer.forward
    if current is _chunked_final_layer_forward:
        _FINAL_LAYER_PATCHED = True
        return

    _ORIGINAL_FINAL_LAYER_FORWARD = current
    h3_model.FinalLayer.forward = _chunked_final_layer_forward
    _FINAL_LAYER_PATCHED = True


def _call_with_streamed_final_layer(executor, args, kwargs, transformer_options):
    rows = _streamed_final_layer_rows(transformer_options)
    previous = int(getattr(_FINAL_LAYER_STREAM, 'chunk_rows', 0) or 0)
    _FINAL_LAYER_STREAM.chunk_rows = rows
    try:
        return executor(*args, **kwargs)
    finally:
        _FINAL_LAYER_STREAM.chunk_rows = previous


@dataclass(frozen=True)
class RuntimeSnapshot:
    request_id: int
    step_index: int
    total_steps: int
    layout: object | None
    compute_dtype: object | None
    device: object | None
    error: str | None = None

    @property
    def valid_layout(self):
        return self.layout is not None and self.error is None


class H3RuntimeSession:
    '''Publish layout and callback-owned sampler-step metadata.'''

    def __init__(self, *, strict_layout=False):
        self.strict_layout = bool(strict_layout)
        self.request_id = -1
        self.last_snapshot = None
        self._active_request = None
        self._request_serial = 0
        self._step_index = -1
        self._total_steps = 0
        self._lock = threading.RLock()
        self._local = threading.local()

    @staticmethod
    def _schedule_length(transformer_options):
        schedule = transformer_options.get('sample_sigmas')
        if schedule is None or not torch.is_tensor(schedule):
            return 0
        return max(0, int(schedule.numel()) - 1)

    def begin_request(self, total_steps=0):
        depth = int(getattr(self._local, 'depth', 0))
        if depth:
            self._local.depth = depth + 1
            return self._local.token
        with self._lock:
            if self._active_request is not None:
                raise RuntimeError(
                    'concurrent sampler requests share one H3 runtime session'
                )
            self._request_serial += 1
            token = self._request_serial
            self._active_request = token
            self.request_id += 1
            self._step_index = 0
            self._total_steps = max(0, int(total_steps))
        self._local.depth = 1
        self._local.token = token
        return token

    def end_request(self, token):
        depth = int(getattr(self._local, 'depth', 0))
        if depth > 1:
            self._local.depth = depth - 1
            return
        self._local.depth = 0
        self._local.token = None
        with self._lock:
            if self._active_request == token:
                self._active_request = None
                self._step_index = -1
                self._total_steps = 0

    def complete_step(self, step_index, total_steps):
        with self._lock:
            if self._active_request is None:
                return
            total_steps = max(0, int(total_steps))
            self._total_steps = total_steps
            next_step = int(step_index) + 1
            if total_steps:
                next_step = min(next_step, total_steps - 1)
            self._step_index = max(0, next_step)

    def _step(self, transformer_options):
        if self._active_request is not None:
            return self._step_index, self._total_steps
        return -1, self._schedule_length(transformer_options)

    def observe(self, x, context, transformer_options, payload=None):
        step_index, total_steps = self._step(transformer_options)
        layout = None
        error = None
        try:
            layout = resolve_layout(x, context, payload or {})
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            error = '%s: %s' % (type(exc).__name__, exc)
            if self.strict_layout:
                raise RuntimeError(
                    '%s could not resolve the H3 packed layout: %s'
                    % (LOG_PREFIX, error)
                ) from exc

        device = None
        try:
            video = x[0] if isinstance(x, (list, tuple)) else x
            device = video.device
        except (AttributeError, IndexError, TypeError):
            pass

        snapshot = RuntimeSnapshot(
            request_id=self.request_id,
            step_index=step_index,
            total_steps=total_steps,
            layout=layout,
            compute_dtype=getattr(context, 'dtype', None),
            device=device,
            error=error,
        )
        self.last_snapshot = snapshot
        transformer_options[RUNTIME_KEY] = snapshot
        return snapshot


def get_runtime_snapshot(transformer_options):
    if not transformer_options:
        return None
    value = transformer_options.get(RUNTIME_KEY)
    return value if isinstance(value, RuntimeSnapshot) else None


def _transformer_options(args, kwargs, index):
    options = args[index] if len(args) > index else kwargs.get('transformer_options')
    if options is not None:
        return options, args, kwargs
    options = {}
    if len(args) > index:
        mutable = list(args)
        mutable[index] = options
        args = tuple(mutable)
    else:
        kwargs = dict(kwargs)
        kwargs['transformer_options'] = options
    return options, args, kwargs


def make_outer_wrapper(session):
    def wrapper(executor, *args, **kwargs):
        sigmas = args[3] if len(args) > 3 else kwargs.get('sigmas')
        total_steps = (
            max(0, int(sigmas.numel()) - 1)
            if torch.is_tensor(sigmas)
            else 0
        )
        callback = args[5] if len(args) > 5 else kwargs.get('callback')

        def step_callback(step, x0, x, callback_total_steps):
            result = None
            if callback is not None:
                result = callback(step, x0, x, callback_total_steps)
            session.complete_step(step, callback_total_steps)
            return result

        if len(args) > 5:
            mutable = list(args)
            mutable[5] = step_callback
            args = tuple(mutable)
        else:
            kwargs = dict(kwargs)
            kwargs['callback'] = step_callback

        token = session.begin_request(total_steps)
        try:
            return executor(*args, **kwargs)
        finally:
            session.end_request(token)

    return wrapper


def make_diffusion_wrapper(session):
    def wrapper(executor, *args, **kwargs):
        options, args, kwargs = _transformer_options(args, kwargs, 3)
        x = args[0] if args else kwargs.get('x')
        context = args[2] if len(args) > 2 else kwargs.get('context')
        session.observe(
            x,
            context,
            options,
            kwargs.get('minimax_payload') or {},
        )
        return _call_with_streamed_final_layer(
            executor,
            args,
            kwargs,
            options,
        )

    return wrapper


def make_apply_model_wrapper(session):
    '''Publish runtime state before entering a compiled diffusion model.'''

    def wrapper(executor, *args, **kwargs):
        options, args, kwargs = _transformer_options(args, kwargs, 5)
        x = args[0] if args else kwargs.get('x')
        latent_shapes = kwargs.get('latent_shapes')
        layout_x = (
            comfy.utils.unpack_latents(x, latent_shapes)
            if latent_shapes is not None and not isinstance(x, (list, tuple))
            else x
        )
        context = args[3] if len(args) > 3 else kwargs.get('c_crossattn')
        session.observe(
            layout_x,
            context,
            options,
            kwargs.get('minimax_payload') or {},
        )
        return _call_with_streamed_final_layer(
            executor,
            args,
            kwargs,
            options,
        )

    return wrapper


def install_runtime_wrapper(model_patcher, session=None):
    import comfy.patcher_extension

    _install_chunked_final_layer_patch()
    session = session or H3RuntimeSession()
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        OUTER_WRAPPER_KEY,
        make_outer_wrapper(session),
        model_patcher.model_options,
        is_model_options=True,
    )
    wrapper_type = (
        comfy.patcher_extension.WrappersMP.APPLY_MODEL
        if model_patcher.model_options.get('torch_compile_kwargs') is not None
        else comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    )
    wrapper = (
        make_apply_model_wrapper(session)
        if wrapper_type == comfy.patcher_extension.WrappersMP.APPLY_MODEL
        else make_diffusion_wrapper(session)
    )
    comfy.patcher_extension.add_wrapper_with_key(
        wrapper_type,
        WRAPPER_KEY,
        wrapper,
        model_patcher.model_options,
        is_model_options=True,
    )
    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    options[RUNTIME_SESSION_KEY] = session
    logging.info(
        '%s installed sampler-step and packed-layout runtime context',
        LOG_PREFIX,
    )
    return session
