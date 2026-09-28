"""H3-owned BF16 ConvRot-256 activation quantization."""

from __future__ import annotations

import torch

from . import loader

_SYMBOL = 'h3_int8_quantize_bf16_rowwise_convrot256'
_RMS_ADALN_SYMBOL = 'h3_int8_quantize_bf16_rms_adaln_rowwise_convrot256'
_GROUP_SIZE = 256
RMS_ADALN_BF16_STAGED = 'bf16_staged'
RMS_ADALN_FP32_THROUGH = 'fp32_through'
_RMS_ADALN_MODES = {
    RMS_ADALN_BF16_STAGED: 0,
    RMS_ADALN_FP32_THROUGH: 1,
}


def int8_rowwise_convrot256_is_available():
    """Whether the loaded ABI-4 library includes the additive quantizer."""
    try:
        library = loader.load()
    except loader.NativeUnavailableError:
        return False
    return getattr(library, _SYMBOL, None) is not None


def rms_adaln_int8_rowwise_convrot256_is_available():
    """Whether the loaded ABI-4 library includes the experimental producer."""
    try:
        library = loader.load()
    except loader.NativeUnavailableError:
        return False
    return getattr(library, _RMS_ADALN_SYMBOL, None) is not None


def quantize_int8_rowwise_convrot256(input):
    """Rotate contiguous BF16 rows in 256-wide groups and quantize to INT8."""
    if not isinstance(input, torch.Tensor):
        raise TypeError('input must be a torch.Tensor')
    if input.ndim != 2:
        raise ValueError('input must be two-dimensional, got %s' % (tuple(input.shape),))
    if input.dtype != torch.bfloat16:
        raise TypeError('input must have dtype torch.bfloat16, got %r' % input.dtype)
    if not input.is_cuda:
        raise ValueError('input must be a CUDA tensor')
    if not input.is_contiguous():
        raise ValueError('input must be contiguous')
    rows, columns = input.shape
    if columns <= 0 or columns % _GROUP_SIZE:
        raise ValueError('input width must be a positive multiple of 256, got %d' % columns)

    library = loader.load()
    function = getattr(library, _SYMBOL, None)
    if function is None:
        raise loader.NativeUnavailableError(
            'the loaded ABI-4 native library has no ConvRot-256 quantizer; rebuild native/'
        )

    output = torch.empty_like(input, dtype=torch.int8)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=input.device)
    loader.check(
        function(
            input.data_ptr(), output.data_ptr(), scales.data_ptr(),
            rows, columns, torch.cuda.current_stream(input.device).cuda_stream,
        ),
        'quantize_bf16_rowwise_convrot256',
    )
    return output, scales


def quantize_rms_adaln_int8_rowwise_convrot256(
    input,
    norm_weight,
    shift,
    scale,
    epsilon,
    *,
    mode,
):
    """Fuse RMSNorm, one AdaLN row, ConvRot-256, and rowwise INT8 quantization."""
    if not isinstance(input, torch.Tensor):
        raise TypeError('input must be a torch.Tensor')
    if input.ndim != 2:
        raise ValueError('input must be two-dimensional, got %s' % (tuple(input.shape),))
    if input.dtype != torch.bfloat16:
        raise TypeError('input must have dtype torch.bfloat16, got %r' % input.dtype)
    if not input.is_cuda:
        raise ValueError('input must be a CUDA tensor')
    if not input.is_contiguous():
        raise ValueError('input must be contiguous')
    rows, columns = input.shape
    if columns <= 0 or columns % _GROUP_SIZE:
        raise ValueError('input width must be a positive multiple of 256, got %d' % columns)

    for name, tensor in (
        ('norm_weight', norm_weight),
        ('shift', shift),
        ('scale', scale),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('%s must be a torch.Tensor' % name)
        if tensor.ndim != 1 or tuple(tensor.shape) != (columns,):
            raise ValueError('%s must have shape (%d,), got %s' % (
                name, columns, tuple(tensor.shape)
            ))
        if tensor.device != input.device:
            raise ValueError('%s must be on the input device' % name)
        if not tensor.is_contiguous():
            raise ValueError('%s must be contiguous' % name)
    if norm_weight.dtype != torch.bfloat16:
        raise TypeError(
            'norm_weight must have dtype torch.bfloat16, got %r'
            % norm_weight.dtype
        )
    if shift.dtype != scale.dtype or shift.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError('shift and scale must have the same BF16 or FP32 dtype')
    try:
        mode_code = _RMS_ADALN_MODES[mode]
    except KeyError as exc:
        raise ValueError(
            'mode must be one of %s, got %r'
            % (tuple(_RMS_ADALN_MODES), mode)
        ) from exc
    epsilon = float(epsilon)
    if not epsilon > 0.0:
        raise ValueError('epsilon must be positive')

    library = loader.load()
    function = getattr(library, _RMS_ADALN_SYMBOL, None)
    if function is None:
        raise loader.NativeUnavailableError(
            'the loaded ABI-4 native library has no fused RMS+AdaLN ConvRot '
            'quantizer; rebuild native/'
        )

    output = torch.empty_like(input, dtype=torch.int8)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=input.device)
    modulation_dtype_code = 0 if shift.dtype == torch.bfloat16 else 1
    loader.check(
        function(
            input.data_ptr(), norm_weight.data_ptr(), shift.data_ptr(),
            scale.data_ptr(), output.data_ptr(), scales.data_ptr(), rows,
            columns, epsilon, modulation_dtype_code, mode_code,
            torch.cuda.current_stream(input.device).cuda_stream,
        ),
        'quantize_bf16_rms_adaln_rowwise_convrot256',
    )
    return output, scales
