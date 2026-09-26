'''In-place ``residual += gate * fc2(swiglu(expanded))`` for held INT8 MLPs.

The bounded native MLP ends every token slab with a TensorWise INT8 fc2 whose
output is only consumed by the block's gated residual ``x.addcmul_(out, gate)``.
Comfy Kitchen 0.2.34 added a CUTLASS INT8 kernel variant that computes
``D = resid + rscale * (acc * xs * ws + bias)`` on the same tuned GEMM configs
as the plain dequant kernel. Writing ``D`` over the residual rows removes the
hidden-width fc2 output and the separate ``addcmul_`` pass.

The result is bit-identical to ``int8_linear(..., input_act="swiglu")``
followed by ``addcmul_``: the kernel rounds the dequantized GEMM result (with
bias) to the output dtype exactly like the plain kernel and keeps only the
residual multiply-add in FP32. Aliasing ``D`` onto the residual is safe because
the residual is read by an AuxLoad visitor and written by an AuxStore visitor on
the thread that owns each element, and stream-K tiles finish each output once.

Kitchen's public ``int8_linear`` always allocates a fresh output, so this uses
   the Kitchen CUDA backend's internal quantizer and GEMM bindings, exactly as the
   public call would. Everything is feature-detected, self-tested once per device,
   dtype, and variant against the public path, and any mismatch, missing symbol, or
unexpected exception disables the fused route and falls back to ``addcmul_``.
It only runs where Kitchen's own dispatcher would pick its CUDA backend for the
equivalent call, and never during ``torch.compile`` tracing or CUDA graph
capture.
'''

import logging
import os
import threading

import torch

LOG_PREFIX = '[H3 Optimizations]'
DISABLE_ENV = 'H3_OPTIMIZATIONS_DISABLE_FUSED_RESIDUAL'
MIN_KITCHEN = '0.2.34'
VARIANT_CONVROT = 'convrot'
VARIANT_PLAIN = 'plain'

_REQUIRED_C = (
    'cutlass_int8_dequant_residual',
    'quantize_int8_rowwise_convrot64',
    'quantize_int8_rowwise',
)
_REQUIRED_BACKEND = (
    '_wrap_for_dlpack',
    '_input_act_code',
    '_act_weight_arg',
    '_apply_input_act',
    '_gemm_vector_arg',
    '_int8_weight_scale_arg',
    '_empty_cuda_tensor',
    '_aligned16',
    '_convrot_fused_shared_memory_fits',
    '_CONVROT_FUSED_MAX_K',
    'DTYPE_TO_CODE',
)

_LOCK = threading.Lock()
_VERIFIED = {}
_LOGGED = set()
_OVERRIDE = None


def _log_once(reason):
    reason = str(reason)
    if reason in _LOGGED:
        return
    _LOGGED.add(reason)
    logging.debug(
        '%s fused MLP residual kernel unavailable; using addcmul_. Detail: %s',
        LOG_PREFIX,
        reason,
    )


def set_enabled(enabled):
    '''Force the fused route on (subject to support), off, or back to Auto (None).

    Intended for benchmarks and tests. The environment variable
    ``H3_OPTIMIZATIONS_DISABLE_FUSED_RESIDUAL=1`` disables it process-wide.
    '''
    global _OVERRIDE
    _OVERRIDE = None if enabled is None else bool(enabled)


def _kitchen_backend():
    try:
        import comfy_kitchen
        import comfy_kitchen.backends.cuda as backend
    except Exception as exc:  # ImportError, or a failed native load
        return None, None, 'Comfy Kitchen CUDA backend is unavailable (%s)' % exc
    native = getattr(backend, '_C', None)
    if native is None:
        return None, None, 'Comfy Kitchen CUDA extension is not loaded'
    missing = [name for name in _REQUIRED_C if not hasattr(native, name)]
    missing += [name for name in _REQUIRED_BACKEND if not hasattr(backend, name)]
    if missing:
        return None, None, (
            'Comfy Kitchen lacks %s (the fused residual kernel needs %s or newer)'
            % (', '.join(missing), MIN_KITCHEN)
        )
    return comfy_kitchen, backend, None


def unavailable_reason(device=None):
    '''Cheap static check (no GPU work): None when the route can be attempted.'''
    if _OVERRIDE is False or os.environ.get(DISABLE_ENV, '').strip() not in ('', '0'):
        return 'disabled'
    if getattr(torch.version, 'hip', None):
        return 'requires the Comfy Kitchen CUDA backend (ROCm uses HIP kernels)'
    if not torch.cuda.is_available():
        return 'requires CUDA'
    _kitchen, backend, reason = _kitchen_backend()
    if reason is not None:
        return reason
    if getattr(backend, '_DISABLE_CUTLASS_INT8', False):
        return 'COMFY_KITCHEN_DISABLE_CUTLASS is set'
    if device is not None:
        device = torch.device(device)
        if device.type != 'cuda':
            return 'requires a CUDA tensor'
        if torch.cuda.get_device_capability(device) < (8, 0):
            return 'requires SM80 or newer (Turing uses different INT8 kernels)'
    return None


def _variant(convrot, convrot_groupsize):
    if convrot:
        return VARIANT_CONVROT if int(convrot_groupsize) == 256 else None
    return VARIANT_PLAIN


def _quantize(backend, expanded, variant):
    device = expanded.device
    wrap = backend._wrap_for_dlpack
    m = int(expanded.shape[0])
    k = int(expanded.shape[1]) // 2
    stream = torch.cuda.current_stream(device).cuda_stream
    x_q = torch.empty((m, k), dtype=torch.int8, device=device)
    x_s = torch.empty((m, 1), dtype=torch.float32, device=device)
    if variant == VARIANT_CONVROT:
        act = backend._act_weight_arg('swiglu', None, device, expanded.dtype)
        backend._C.quantize_int8_rowwise_convrot64(
            wrap(expanded),
            wrap(x_q),
            wrap(x_s),
            256,
            False,
            backend._input_act_code('swiglu'),
            0,
            wrap(act),
            0.0,
            stream,
        )
    else:
        activated = backend._apply_input_act(expanded, 'swiglu', None, 0.0)
        backend._C.quantize_int8_rowwise(
            wrap(activated),
            wrap(x_q),
            wrap(x_s),
            False,
            0,
            stream,
        )
    return x_q, x_s


def _gemm_residual_(backend, x_q, x_s, qdata, weight_scale, bias, residual, gate):
    device = residual.device
    dtype = residual.dtype
    wrap = backend._wrap_for_dlpack
    n = int(qdata.shape[0])
    ws = backend._int8_weight_scale_arg(weight_scale, device)
    ws = ws if ws.numel() == n else ws.expand(n).contiguous()
    bias_arg = (
        backend._gemm_vector_arg(bias, device, dtype)
        if bias is not None
        else backend._empty_cuda_tensor(device, dtype)
    )
    return bool(
        backend._C.cutlass_int8_dequant_residual(
            wrap(x_q),
            wrap(qdata),
            wrap(x_s),
            wrap(ws),
            wrap(bias_arg),
            wrap(backend._gemm_vector_arg(gate, device, dtype)),
            wrap(residual),
            wrap(residual),
            backend.DTYPE_TO_CODE[dtype],
            torch.cuda.current_stream(device).cuda_stream,
        )
    )


def _shape_ok(backend, expanded, qdata, residual, gate, variant):
    if expanded.ndim != 2 or residual.ndim != 2 or qdata.ndim != 2 or gate.ndim != 1:
        return False
    if expanded.dtype not in (torch.bfloat16, torch.float16):
        return False
    if residual.dtype != expanded.dtype or qdata.dtype != torch.int8:
        return False
    m = int(expanded.shape[0])
    k = int(expanded.shape[1]) // 2
    n = int(qdata.shape[0])
    # Kitchen serves one- and two-row calls with GEMV kernels, not CUTLASS.
    if m <= 2 or int(expanded.shape[1]) % 2 or k % 16:
        return False
    if tuple(qdata.shape) != (n, k) or tuple(residual.shape) != (m, n):
        return False
    if int(gate.shape[0]) != n:
        return False
    if not (expanded.is_contiguous() and residual.is_contiguous() and qdata.is_contiguous()):
        return False
    if not (expanded.is_cuda and residual.device == expanded.device == qdata.device):
        return False
    if not backend._aligned16(residual):
        return False
    if variant == VARIANT_CONVROT:
        if k % 256 or not 256 <= k <= int(backend._CONVROT_FUSED_MAX_K):
            return False
        if not backend._convrot_fused_shared_memory_fits(expanded, k, 256):
            return False
    return True


def _kitchen_routes_to_cuda(kitchen, expanded, qdata, weight_scale, bias, residual, gate, variant):
    '''The equivalent public call must dispatch to Kitchen's CUDA backend.'''
    kwargs = {
        'x': expanded,
        'weight': qdata,
        'weight_scale': weight_scale,
        'bias': bias,
        'out_dtype': expanded.dtype,
        'convrot': variant == VARIANT_CONVROT,
        'convrot_groupsize': 256,
        'input_act': 'swiglu',
        'input_act_weight': None,
        'input_act_eps': 0.0,
        'residual': residual,
        'residual_scale': gate,
    }
    try:
        return kitchen.registry.get_capable_backend('int8_linear', kwargs) == 'cuda'
    except Exception as exc:
        _log_once('Kitchen backend selection failed: %s' % exc)
        return False


def _self_test(kitchen, backend, device, dtype, variant):
    '''Compare the aliased kernel with the public path on a small problem.'''
    generator = torch.Generator(device=device).manual_seed(20260926)
    m, n, k = 48, 256, 256
    expanded = torch.randn((m, 2 * k), generator=generator, device=device, dtype=dtype)
    qdata = torch.randint(-127, 128, (n, k), generator=generator, device=device, dtype=torch.int8)
    weight_scale = torch.rand((n,), generator=generator, device=device) * 1e-3 + 1e-4
    bias = torch.randn((n,), generator=generator, device=device, dtype=dtype)
    gate = 0.05 * torch.randn((n,), generator=generator, device=device, dtype=dtype)
    residual = 4.0 * torch.randn((m, n), generator=generator, device=device, dtype=dtype)
    convrot = variant == VARIANT_CONVROT
    plain = kitchen.int8_linear(
        expanded, qdata, weight_scale, bias, dtype,
        convrot=convrot, convrot_groupsize=256, input_act='swiglu',
    )
    expected = residual.clone().addcmul_(plain, gate)
    actual = residual.clone()
    x_q, x_s = _quantize(backend, expanded, variant)
    if not _gemm_residual_(backend, x_q, x_s, qdata, weight_scale, bias, actual, gate):
        return False, 'the CUTLASS residual kernel declined the self-test shape'
    if not torch.equal(actual, expected):
        return False, 'aliased residual kernel is not bit-identical to int8_linear + addcmul_'
    return True, None


def _verified(kitchen, backend, device, dtype, variant):
    key = (int(device.index or 0), dtype, variant)
    with _LOCK:
        cached = _VERIFIED.get(key)
        if cached is not None:
            return cached
        if torch.cuda.is_current_stream_capturing():
            return False
        try:
            ok, reason = _self_test(kitchen, backend, device, dtype, variant)
        except Exception as exc:
            ok, reason = False, '%s: %s' % (type(exc).__name__, exc)
        if not ok:
            _log_once('%s %s self-test failed: %s' % (variant, dtype, reason))
        _VERIFIED[key] = ok
        return ok


def gated_swiglu_residual_(
    expanded,
    qdata,
    weight_scale,
    bias,
    residual,
    gate,
    *,
    convrot,
    convrot_groupsize=256,
):
    '''Try ``residual += gate * fc2(swiglu(expanded))`` in place.

    Returns True when the fused kernel wrote ``residual``. Returns False, with
    ``residual`` untouched, whenever the route is unsupported; the caller then
    runs the ordinary fc2 and ``addcmul_``.
    '''
    if torch.compiler.is_compiling():
        return False
    variant = _variant(convrot, convrot_groupsize)
    if variant is None:
        return False
    reason = unavailable_reason(expanded.device)
    if reason is not None:
        _log_once(reason)
        return False
    kitchen, backend, reason = _kitchen_backend()
    if reason is not None:
        return False
    if gate.dtype != residual.dtype:
        gate = gate.to(dtype=residual.dtype)
    if not _shape_ok(backend, expanded, qdata, residual, gate, variant):
        return False
    if bias is not None and (bias.ndim != 1 or int(bias.shape[0]) != int(qdata.shape[0])):
        return False
    if not _kitchen_routes_to_cuda(kitchen, expanded, qdata, weight_scale, bias, residual, gate, variant):
        return False
    if not _verified(kitchen, backend, expanded.device, expanded.dtype, variant):
        return False
    try:
        x_q, x_s = _quantize(backend, expanded, variant)
        # A declined launch writes nothing, so the caller's fallback is exact.
        return _gemm_residual_(backend, x_q, x_s, qdata, weight_scale, bias, residual, gate)
    except (RuntimeError, TypeError, ValueError) as exc:
        _log_once('%s: %s' % (type(exc).__name__, exc))
        return False


__all__ = [
    'DISABLE_ENV',
    'gated_swiglu_residual_',
    'set_enabled',
    'unavailable_reason',
]
