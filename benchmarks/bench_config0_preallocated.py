'''Isolate one-shot and chunked Kitchen CUTLASS config-0 QKV costs.'''

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path

from bench_chunked_kitchen_qkv import (
    DEFAULT_SEQUENCE,
    build_attention,
    chunk_ranges,
    forced_cutlass_projection,
    resolve_checkpoint,
    weight_contract,
)


CONFIG = 0
DEFAULT_CHUNK = 768


def summarize(samples):
    return {
        'median_ms': statistics.median(samples),
        'min_ms': min(samples),
        'max_ms': max(samples),
        'samples_ms': samples,
    }


def time_case(torch, fn, warmup, iterations, device):
    '''Time with a warm allocator; never empty the CUDA cache between samples.'''
    for _ in range(warmup):
        result = fn()
        torch.cuda.synchronize(device)
        del result

    cuda_samples = []
    wall_samples = []
    for _ in range(iterations):
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        result = fn()
        stop.record()
        stop.synchronize()
        wall_samples.append((time.perf_counter() - wall_start) * 1000.0)
        cuda_samples.append(float(start.elapsed_time(stop)))
        del result
    return {
        'cuda': summarize(cuda_samples),
        'wall': summarize(wall_samples),
    }


class Config0Experiment:
    def __init__(self, torch, module, x, chunk_size):
        from comfy_kitchen.backends import cuda

        self.torch = torch
        self.cuda = cuda
        self.extension = cuda._C
        self.runner = getattr(self.extension, 'cutlass_int8_dequant_config', None)
        if self.runner is None:
            raise RuntimeError('Kitchen extension has no forced-config binding')

        self.module = module
        self.x = x
        self.chunk_size = int(chunk_size)
        self.ranges = chunk_ranges(int(x.shape[0]), self.chunk_size)
        self.stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
        self.output_dtype_code = cuda.DTYPE_TO_CODE[torch.bfloat16]

        weight = module.qkv_proj.weight
        self.groupsize = int(weight._params.convrot_groupsize)
        self.weight_qdata = weight._qdata.contiguous()
        self.weight_scale = weight._params.scale.to(torch.float32).contiguous()
        rows = int(x.shape[0])
        hidden = int(x.shape[1])
        output_width = int(self.weight_qdata.shape[0])

        # Full-shape buffers make the one-shot path allocation-free.
        self.full_qdata = torch.empty(
            (rows, hidden), dtype=torch.int8, device=x.device
        )
        self.full_scale = torch.empty(
            (rows, 1), dtype=torch.float32, device=x.device
        )
        self.full_output = torch.empty(
            (rows, output_width), dtype=torch.bfloat16, device=x.device
        )

        # Chunk buffers are reused by every launch, matching the streaming path.
        self.chunk_qdata = torch.empty(
            (self.chunk_size, hidden), dtype=torch.int8, device=x.device
        )
        self.chunk_scale = torch.empty(
            (self.chunk_size, 1), dtype=torch.float32, device=x.device
        )
        self.chunk_output = torch.empty(
            (self.chunk_size, output_width), dtype=torch.bfloat16, device=x.device
        )

    def quantize_into(self, x, qdata, scale):
        self.extension.quantize_int8_rowwise_convrot64(
            self.cuda._wrap_for_dlpack(x),
            self.cuda._wrap_for_dlpack(qdata),
            self.cuda._wrap_for_dlpack(scale),
            self.groupsize,
            False,
            0,
            0,
            self.stream_ptr,
        )

    def gemm_into(self, qdata, scale, output):
        used = self.runner(
            self.cuda._wrap_for_dlpack(qdata),
            self.cuda._wrap_for_dlpack(self.weight_qdata),
            self.cuda._wrap_for_dlpack(scale),
            self.cuda._wrap_for_dlpack(self.weight_scale),
            self.cuda._wrap_for_dlpack(output),
            self.output_dtype_code,
            CONFIG,
            self.stream_ptr,
        )
        if not used:
            raise RuntimeError('Kitchen declined CUTLASS config 0')

    def full_quant(self):
        self.quantize_into(self.x, self.full_qdata, self.full_scale)
        return self.full_qdata

    def chunk_quant_all(self):
        for start, stop in self.ranges:
            self.quantize_into(
                self.x[start:stop],
                self.full_qdata[start:stop],
                self.full_scale[start:stop],
            )
        return self.full_qdata

    def full_gemm(self):
        self.gemm_into(self.full_qdata, self.full_scale, self.full_output)
        return self.full_output

    def chunk_gemm_one(self):
        rows = min(self.chunk_size, int(self.x.shape[0]))
        self.gemm_into(
            self.full_qdata[:rows],
            self.full_scale[:rows],
            self.chunk_output[:rows],
        )
        return self.chunk_output[:rows]

    def chunk_gemm_all(self):
        for start, stop in self.ranges:
            rows = stop - start
            self.gemm_into(
                self.full_qdata[start:stop],
                self.full_scale[start:stop],
                self.chunk_output[:rows],
            )
        return self.chunk_output[: self.ranges[-1][1] - self.ranges[-1][0]]

    def full_combined(self):
        self.quantize_into(self.x, self.full_qdata, self.full_scale)
        self.gemm_into(self.full_qdata, self.full_scale, self.full_output)
        return self.full_output

    def chunk_combined(self):
        for start, stop in self.ranges:
            rows = stop - start
            qdata = self.chunk_qdata[:rows]
            scale = self.chunk_scale[:rows]
            output = self.chunk_output[:rows]
            self.quantize_into(self.x[start:stop], qdata, scale)
            self.gemm_into(qdata, scale, output)
        return self.chunk_output[: self.ranges[-1][1] - self.ranges[-1][0]]


def auto_chunked_projection(module, x, chunk_size):
    result = None
    for start, stop in chunk_ranges(int(x.shape[0]), int(chunk_size)):
        result = module.qkv_proj(x[start:stop])
    return result


def run_profile(torch, experiment, profile_case, device):
    cases = {
        'full_gemm': (('profile_full_config0_gemm', experiment.full_gemm),),
        'chunk_gemm': (('profile_chunk768_config0_gemm', experiment.chunk_gemm_one),),
        'systems': (
            ('profile_full_config0_combined', experiment.full_combined),
            ('profile_chunked_config0_combined', experiment.chunk_combined),
            (
                'profile_cold_allocating_full_config0_combined',
                lambda: forced_cutlass_projection(
                    torch, experiment.module, experiment.x, CONFIG
                ),
            ),
        ),
    }[profile_case]

    # Warm extension dispatch, buffers, and kernels before profiler capture.
    experiment.full_quant()
    for _label, fn in cases:
        if 'cold_allocating' not in _label:
            fn()
    torch.cuda.synchronize(device)
    if profile_case == 'systems':
        # Leave no free cached blocks for the deliberately cold comparison.
        torch.cuda.empty_cache()

    torch.cuda.cudart().cudaProfilerStart()
    for label, fn in cases:
        torch.cuda.nvtx.range_push(label)
        fn()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    torch.cuda.cudart().cudaProfilerStop()
    return {'profile_case': profile_case, 'ranges': [label for label, _fn in cases]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=DEFAULT_SEQUENCE)
    parser.add_argument('--chunk', type=int, default=DEFAULT_CHUNK)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=7)
    parser.add_argument(
        '--profile-case',
        choices=('full_gemm', 'chunk_gemm', 'systems'),
    )
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if args.sequence <= 0 or args.chunk <= 0:
        parser.error('sequence and chunk must be positive')
    if args.chunk % 128:
        parser.error('chunk must be a multiple of 128')
    if args.warmup < 0 or args.iterations <= 0:
        parser.error('warmup/iterations are invalid')
    if not args.i_understand_this_uses_gpu:
        parser.error('pass --i-understand-this-uses-gpu after the idle preflight')
    return args


def main(argv=None):
    args = parse_args(argv)
    comfy_root = Path(__file__).resolve().parents[3]
    pack_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(pack_root))

    import torch

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    device = torch.device('cuda')
    checkpoint = resolve_checkpoint(args.checkpoint)
    module, hidden, prefix = build_attention(
        torch, checkpoint, args.block, args.epsilon, device
    )
    contract = weight_contract(module)
    if not (
        contract['quantized']
        and contract['layout'] == 'TensorWiseINT8Layout'
        and contract['convrot']
        and contract['convrot_groupsize'] == 256
    ):
        raise SystemExit('checkpoint QKV is not ConvRot-256 TensorWise INT8')

    generator = torch.Generator(device=device).manual_seed(1234)
    x = torch.randn(
        (args.sequence, hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    experiment = Config0Experiment(torch, module, x, args.chunk)

    if args.profile_case:
        result = run_profile(torch, experiment, args.profile_case, device)
    else:
        # Populate identical row-wise activations before isolated GEMM timing.
        experiment.full_quant()
        torch.cuda.synchronize(device)
        cases = {
            'preallocated_full_quant': experiment.full_quant,
            'preallocated_chunk_quant_all': experiment.chunk_quant_all,
            'preallocated_full_gemm': experiment.full_gemm,
            'preallocated_chunk_gemm_all': experiment.chunk_gemm_all,
            'preallocated_full_combined': experiment.full_combined,
            'preallocated_chunk_combined': experiment.chunk_combined,
            'warm_allocator_auto_full_projection': lambda: module.qkv_proj(x),
            'warm_allocator_auto_chunked_projection': lambda: auto_chunked_projection(
                module, x, args.chunk
            ),
        }
        result = {
            'cases': {
                name: time_case(
                    torch, fn, args.warmup, args.iterations, device
                )
                for name, fn in cases.items()
            }
        }

    result.update({
        'versions': {
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'comfy_kitchen': importlib.metadata.version('comfy-kitchen'),
        },
        'gpu': {
            'name': torch.cuda.get_device_name(device),
            'capability': list(torch.cuda.get_device_capability(device)),
        },
        'checkpoint': str(checkpoint),
        'checkpoint_prefix': prefix,
        'sequence': args.sequence,
        'chunk': args.chunk,
        'chunk_count': len(experiment.ranges),
        'final_chunk': experiment.ranges[-1][1] - experiment.ranges[-1][0],
        'config': CONFIG,
        'weight_contract': contract,
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
