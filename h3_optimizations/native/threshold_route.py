"""Native threshold routing for the streamed sparse Kitchen path.

A route row keeps video KV tile j when q . k_j exceeds the row's score mean plus
tau standard deviations. The mean and variance are exact statistics of the
row's scores over the video tiles, q . kbar and q^T Sigma q, so the native
kernel (native/src/h3_threshold_route.cu) never materializes a score matrix
and needs no top-k or sort. A KV budget maps to tau through the Gaussian
quantile, tau = PhiInv(1 - budget), which lands within about 0.7 percentage
points of the requested density on H3 video attention.
"""

from __future__ import annotations

from statistics import NormalDist

import torch

from . import loader

_SYMBOL = 'h3_int8_threshold_route'
_HEAD_DIM = 128


def threshold_route_is_available(device=None):
    """Whether the loaded library and device can run native threshold routing."""
    if not torch.cuda.is_available():
        return False
    try:
        if tuple(torch.cuda.get_device_capability(device)) < (8, 0):
            return False  # BF16 tensor-core MMA
        library = loader.load()
    except (loader.NativeUnavailableError, RuntimeError):
        return False
    return getattr(library, _SYMBOL, None) is not None


def budget_tau(budget):
    """Row threshold in score standard deviations that keeps ~budget of the tiles."""
    budget = min(max(float(budget), 1e-6), 1.0 - 1e-6)
    return NormalDist().inv_cdf(1.0 - budget)


def video_statistics(k_video):
    """Per-head mean (FP32) and population covariance (BF16) of video K summaries."""
    keys = k_video.float()
    count = keys.shape[-2]
    mean = keys.mean(dim=-2)
    # E[k k^T] - mean mean^T: one GEMM, no centred copy of the keys.
    covariance = torch.matmul(keys.transpose(-1, -2), keys).div_(count)
    covariance.sub_(mean.unsqueeze(-1) * mean.unsqueeze(-2))
    return mean.contiguous(), covariance.to(torch.bfloat16).contiguous()


def threshold_route(q_summary, k_summary, k_mean, k_cov, *, pure_video_kv_start, dense_rows, tau,
                    return_masks=False):
    """Delta route LUT ``[B, H, rows, kv_tiles]`` and valid counts for one Q chunk.

    ``dense_rows`` leading rows get the full dense route; the rest get the
    dense context prefix followed by the selected video tiles in ascending
    order.

    ``return_masks=True`` also returns flattened int64 selection masks, one
    64-bit word per head, route row and 64-video-tile stage. Dense rows have
    no initialized mask; consumers must skip them.
    """
    if q_summary.ndim != 4 or k_summary.ndim != 4:
        raise ValueError('Q and K summaries must be [B, H, tiles, 128]')
    if q_summary.shape[-1] != _HEAD_DIM or k_summary.shape[-1] != _HEAD_DIM:
        raise ValueError('threshold routing is built for head_dim 128')
    if q_summary.dtype != torch.bfloat16 or k_summary.dtype != torch.bfloat16:
        raise TypeError('threshold routing expects BF16 summaries')
    batch, heads, rows = (int(v) for v in q_summary.shape[:3])
    kv_tiles = int(k_summary.shape[-2])
    pvs = int(pure_video_kv_start)
    if tuple(k_summary.shape[:2]) != (batch, heads) or not 0 <= pvs < kv_tiles:
        raise ValueError('K summary does not match the Q chunk')
    if tuple(k_mean.shape) != (batch, heads, _HEAD_DIM) or tuple(k_cov.shape) != (
            batch, heads, _HEAD_DIM, _HEAD_DIM):
        raise ValueError('K statistics do not match the summaries')
    q = q_summary.contiguous()
    k = k_summary.contiguous()
    mean = k_mean.to(torch.float32).contiguous()
    cov = k_cov.to(torch.bfloat16).contiguous()
    device = q.device
    lut = torch.empty((batch, heads, rows, kv_tiles), dtype=torch.int32, device=device)
    valid = torch.empty((batch, heads, rows), dtype=torch.int32, device=device)
    # One 64-bit selection mask per row and 64-tile stage (select -> write).
    stages = (kv_tiles - pvs + 63) // 64
    masks = torch.empty((batch * heads * rows * stages,), dtype=torch.int64, device=device)
    if rows == 0:
        return (lut, valid, masks) if return_masks else (lut, valid)
    library = loader.load()
    function = getattr(library, _SYMBOL, None)
    if function is None:
        raise loader.NativeUnavailableError(
            'the loaded native library has no threshold router; rebuild native/')
    loader.check(
        function(
            q.data_ptr(),
            k.data_ptr() + pvs * _HEAD_DIM * k.element_size(),
            cov.data_ptr(),
            mean.data_ptr(),
            lut.data_ptr(),
            valid.data_ptr(),
            masks.data_ptr(),
            batch * heads,
            rows,
            rows * _HEAD_DIM,
            _HEAD_DIM,
            kv_tiles * _HEAD_DIM,
            kv_tiles - pvs,
            pvs,
            kv_tiles,
            max(0, min(int(dense_rows), rows)),
            kv_tiles,
            float(tau),
            torch.cuda.current_stream(device).cuda_stream,
        ),
        'threshold_route',
    )
    return (lut, valid, masks) if return_masks else (lut, valid)


__all__ = ['budget_tau', 'threshold_route', 'threshold_route_is_available', 'video_statistics']
