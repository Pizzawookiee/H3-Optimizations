'''Compare eager and fused RMSNorm + AdaLN + ConvRot INT8 producers.'''

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
from pathlib import Path


PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Measure the experimental BF16-staged and FP32-through fused '
            'RMSNorm + scalar AdaLN + ConvRot-256 quantizers.'
        )
    )
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument(
        '--library',
        type=Path,
        help='development native library containing the experimental symbol',
    )
    parser.add_argument('--block-index', type=int, default=24)
    parser.add_argument('--rows', type=int, default=4096)
    parser.add_argument('--hidden', type=int, default=5376)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.rows <= 0:
        parser.error('--rows must be positive')
    if args.hidden <= 0 or args.hidden % 256:
        parser.error('--hidden must be a positive multiple of 256')
    if args.epsilon <= 0:
        parser.error('--epsilon must be positive')
    if args.warmup < 0 or args.repeats <= 0:
        parser.error('--warmup must be non-negative and --repeats positive')
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error('--checkpoint does not exist: %s' % args.checkpoint)
    if args.library is not None and not args.library.is_file():
        parser.error('--library does not exist: %s' % args.library)
    return args


def load_development_library(loader, path):
    if path is None:
        return
    library = ctypes.CDLL(str(path.resolve()))
    library.h3_int8_abi_version.restype = ctypes.c_int
    library.h3_int8_abi_version.argtypes = []
    found = library.h3_int8_abi_version()
    if found != loader.ABI_VERSION:
        raise RuntimeError(
            '%s reports ABI %d, expected %d'
            % (path, found, loader.ABI_VERSION)
        )
    loader._bind(library)
    loader._library = library
    loader._load_error = None


def load_norm_weight(torch, checkpoint, block_index, hidden):
    if checkpoint is None:
        return torch.ones(hidden, dtype=torch.bfloat16), 'synthetic_ones'

    from safetensors import safe_open

    suffix = 'blocks.%d.norm1.weight' % block_index
    with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
        keys = [key for key in handle.keys() if key.endswith(suffix)]
        if len(keys) != 1:
            raise KeyError(
                'expected one checkpoint key ending in %r, found %s'
                % (suffix, keys)
            )
        weight = handle.get_tensor(keys[0])
    if tuple(weight.shape) != (hidden,):
        raise ValueError(
            '%s has shape %s, expected (%d,)'
            % (keys[0], tuple(weight.shape), hidden)
        )
    return weight.to(torch.bfloat16), keys[0]


def load_qkv_weight(torch, checkpoint, block_index, hidden):
    if checkpoint is None:
        return None

    from safetensors import safe_open

    prefix = next(
        (
            candidate
            for candidate in (
                'model.diffusion_model.blocks.%d.attn.qkv_proj.' % block_index,
                'diffusion_model.blocks.%d.attn.qkv_proj.' % block_index,
                'blocks.%d.attn.qkv_proj.' % block_index,
            )
            if candidate + 'weight' in _checkpoint_keys(checkpoint)
        ),
        None,
    )
    if prefix is None:
        raise KeyError('checkpoint has no layer-%d QKV weight' % block_index)
    with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
        weight = handle.get_tensor(prefix + 'weight')
        scale = handle.get_tensor(prefix + 'weight_scale')
    if weight.dtype != torch.int8 or weight.ndim != 2 or weight.shape[1] != hidden:
        raise ValueError('QKV weight is not a compatible ConvRot INT8 matrix')
    if scale.numel() != weight.shape[0]:
        raise ValueError('QKV weight scale does not match its output width')
    return weight, scale.reshape(-1), prefix + 'weight'


def _checkpoint_keys(checkpoint):
    from safetensors import safe_open

    with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
        return set(handle.keys())


def eager_producer(torch, quantize, x, norm_weight, shift, scale, epsilon):
    h = torch.nn.functional.rms_norm(
        x,
        (x.shape[1],),
        weight=norm_weight,
        eps=epsilon,
    )
    h.mul_(1.0 + scale.to(torch.bfloat16))
    h.add_(shift.to(torch.bfloat16))
    return quantize(h)


def carrier_error(torch, reference, candidate):
    reference_q, reference_scale = reference
    candidate_q, candidate_scale = candidate
    code_delta = candidate_q.to(torch.int16) - reference_q.to(torch.int16)
    reference_dequant = reference_q.float() * reference_scale
    candidate_dequant = candidate_q.float() * candidate_scale
    delta = candidate_dequant - reference_dequant
    denominator = reference_dequant.square().sum().sqrt().clamp_min(1e-12)
    return {
        'q_exact': bool(torch.equal(reference_q, candidate_q)),
        'q_same_percent': float((code_delta == 0).float().mean().item() * 100.0),
        'q_within_one_percent': float(
            (code_delta.abs() <= 1).float().mean().item() * 100.0
        ),
        'q_max_code_delta': int(code_delta.abs().max().item()),
        'scale_exact': bool(torch.equal(reference_scale, candidate_scale)),
        'scale_max_abs': float(
            (candidate_scale - reference_scale).abs().max().item()
        ),
        'dequant_relative_l2': float(delta.square().sum().sqrt().div(denominator).item()),
        'dequant_max_abs': float(delta.abs().max().item()),
    }


def tensor_error(torch, reference, candidate):
    delta = candidate.float() - reference.float()
    denominator = reference.float().square().sum().sqrt().clamp_min(1e-12)
    return {
        'exact': bool(torch.equal(reference, candidate)),
        'same_percent': float((reference == candidate).float().mean().item() * 100.0),
        'relative_l2': float(delta.square().sum().sqrt().div(denominator).item()),
        'max_abs': float(delta.abs().max().item()),
    }


def project_carrier(torch, carrier, weight, weight_scale):
    from comfy_kitchen.backends import cuda

    qdata, scale = carrier
    return cuda._int4_linear_via_int8_values(
        qdata,
        weight,
        scale,
        weight_scale,
        None,
        torch.bfloat16,
    )


def time_cuda(torch, function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del result, start, end
    return {
        'median_ms': statistics.median(samples),
        'mean_ms': statistics.fmean(samples),
        'min_ms': min(samples),
        'max_ms': max(samples),
        'samples_ms': samples,
    }


def main(argv=None):
    args = parse_args(argv)

    import torch

    from h3_optimizations.native import (
        RMS_ADALN_BF16_STAGED,
        RMS_ADALN_FP32_THROUGH,
        quantize_int8_rowwise_convrot256,
        quantize_rms_adaln_int8_rowwise_convrot256,
    )
    from h3_optimizations.native import loader

    load_development_library(loader, args.library)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    device = torch.device('cuda')
    generator = torch.Generator(device=device).manual_seed(args.seed)
    norm_weight, norm_source = load_norm_weight(
        torch, args.checkpoint, args.block_index, args.hidden
    )
    norm_weight = norm_weight.to(device=device, non_blocking=True).contiguous()
    x = torch.randn(
        (args.rows, args.hidden),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    shift = torch.randn(
        args.hidden, dtype=torch.float32, device=device, generator=generator
    ).mul_(0.1)
    scale = torch.randn(
        args.hidden, dtype=torch.float32, device=device, generator=generator
    ).mul_(0.1)

    def eager():
        return eager_producer(
            torch,
            quantize_int8_rowwise_convrot256,
            x,
            norm_weight,
            shift,
            scale,
            args.epsilon,
        )

    def fused(mode):
        return quantize_rms_adaln_int8_rowwise_convrot256(
            x,
            norm_weight,
            shift,
            scale,
            args.epsilon,
            mode=mode,
        )

    reference = eager()
    staged = fused(RMS_ADALN_BF16_STAGED)
    fp32 = fused(RMS_ADALN_FP32_THROUGH)
    torch.cuda.synchronize()

    projection = None
    qkv_source = None
    qkv = load_qkv_weight(
        torch, args.checkpoint, args.block_index, args.hidden
    )
    if qkv is not None:
        qkv_weight, qkv_scale, qkv_source = qkv
        qkv_weight = qkv_weight.to(device=device, non_blocking=True).contiguous()
        qkv_scale = qkv_scale.to(device=device, non_blocking=True).contiguous()
        reference_projection = project_carrier(
            torch, reference, qkv_weight, qkv_scale
        )
        staged_projection = project_carrier(torch, staged, qkv_weight, qkv_scale)
        projection = {
            RMS_ADALN_BF16_STAGED: tensor_error(
                torch, reference_projection, staged_projection
            )
        }
        del staged_projection
        fp32_projection = project_carrier(torch, fp32, qkv_weight, qkv_scale)
        projection[RMS_ADALN_FP32_THROUGH] = tensor_error(
            torch, reference_projection, fp32_projection
        )
        del reference_projection, fp32_projection, qkv_weight, qkv_scale
        torch.cuda.synchronize()

    result = {
        'checkpoint': None if args.checkpoint is None else str(args.checkpoint),
        'library': None if args.library is None else str(args.library),
        'block_index': args.block_index,
        'norm_source': norm_source,
        'qkv_source': qkv_source,
        'rows': args.rows,
        'hidden': args.hidden,
        'epsilon': args.epsilon,
        'seed': args.seed,
        'device': torch.cuda.get_device_name(device),
        'capability': list(torch.cuda.get_device_capability(device)),
        'correctness_vs_eager_bf16': {
            RMS_ADALN_BF16_STAGED: carrier_error(torch, reference, staged),
            RMS_ADALN_FP32_THROUGH: carrier_error(torch, reference, fp32),
        },
        'qkv_projection_vs_eager_bf16': projection,
        'timing': {
            'eager_bf16': time_cuda(torch, eager, args.warmup, args.repeats),
            RMS_ADALN_BF16_STAGED: time_cuda(
                torch,
                lambda: fused(RMS_ADALN_BF16_STAGED),
                args.warmup,
                args.repeats,
            ),
            RMS_ADALN_FP32_THROUGH: time_cuda(
                torch,
                lambda: fused(RMS_ADALN_FP32_THROUGH),
                args.warmup,
                args.repeats,
            ),
        },
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
