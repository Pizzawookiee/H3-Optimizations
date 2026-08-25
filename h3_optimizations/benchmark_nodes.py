'''Benchmark-only H3 controls.

These nodes are intentionally absent from normal ComfyUI registration. They are
only exposed when H3_OPTIMIZATIONS_BENCHMARK_NODES=1 is present at server
startup. They exist to make repository benchmarks reproducible without turning
diagnostic controls into supported user-facing features.
'''

from __future__ import annotations

import torch

import comfy.ops
from comfy_api.latest import io
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

from .model import get_h3_blocks, is_minimax_h3


CONFIG = 0
PATCH_PREFIX = 'diffusion_model.blocks'


class BenchmarkConfig0Error(RuntimeError):
    pass


def _forced_config0_projection(linear, x):
    '''Execute one compatible ConvRot-256 INT8 linear through CUTLASS config 0.'''
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16:
        raise BenchmarkConfig0Error(
            'benchmark config 0 requires rank-2 CUDA BF16 activations'
        )

    from comfy_kitchen.backends import cuda

    runner = getattr(cuda._C, 'cutlass_int8_dequant_config', None)
    if runner is None:
        raise BenchmarkConfig0Error(
            'Comfy Kitchen does not expose cutlass_int8_dequant_config'
        )

    weight, bias, handle = comfy.ops.cast_bias_weight(
        linear,
        x,
        offloadable=True,
        compute_dtype=torch.bfloat16,
        want_requant=True,
    )
    try:
        if bias is not None:
            raise BenchmarkConfig0Error('benchmark config 0 does not support bias')
        if not isinstance(weight, QuantizedTensor):
            raise BenchmarkConfig0Error(
                'benchmark config 0 requires quantized ConvRot INT8 QKV weights'
            )
        if getattr(weight, '_layout_cls', None) != 'TensorWiseINT8Layout':
            raise BenchmarkConfig0Error(
                'benchmark config 0 requires TensorWiseINT8Layout QKV weights'
            )
        params = weight._params
        if (
            getattr(params, 'transposed', False)
            or not getattr(params, 'convrot', False)
            or int(getattr(params, 'convrot_groupsize', 0)) != 256
        ):
            raise BenchmarkConfig0Error(
                'benchmark config 0 requires non-transposed ConvRot-256 QKV weights'
            )

        weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
        weight_qdata = weight_qdata.contiguous()
        weight_scale = weight_scale.to(torch.float32).contiguous()
        x_qdata = torch.empty_like(x, dtype=torch.int8)
        x_scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
        output = torch.empty(
            (x.shape[0], weight_qdata.shape[0]),
            dtype=torch.bfloat16,
            device=x.device,
        )
        stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
        cuda._C.quantize_int8_rowwise_convrot64(
            cuda._wrap_for_dlpack(x),
            cuda._wrap_for_dlpack(x_qdata),
            cuda._wrap_for_dlpack(x_scale),
            256,
            False,
            0,
            0,
            stream_ptr,
        )
        used = runner(
            cuda._wrap_for_dlpack(x_qdata),
            cuda._wrap_for_dlpack(weight_qdata),
            cuda._wrap_for_dlpack(x_scale),
            cuda._wrap_for_dlpack(weight_scale),
            cuda._wrap_for_dlpack(output),
            cuda.DTYPE_TO_CODE[torch.bfloat16],
            CONFIG,
            stream_ptr,
        )
        if not used:
            raise BenchmarkConfig0Error('Comfy Kitchen declined CUTLASS config 0')
        return output
    finally:
        comfy.ops.uncast_bias_weight(linear, weight, bias, handle)


def _make_forward(linear):
    def forward(x):
        return _forced_config0_projection(linear, x)

    forward._h3_benchmark_forced_kitchen_config = CONFIG
    return forward


class H3BenchmarkForceQKVConfig0(io.ComfyNode):
    '''Force compatible H3 QKV projections through Kitchen CUTLASS config 0.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3BenchmarkForceQKVConfig0',
            display_name='H3 Benchmark Force QKV Config 0',
            category='H3-Optimizations/Benchmarks',
            description=(
                'Diagnostic benchmark control. Forces ConvRot-256 INT8 H3 QKV '
                'linears through Comfy Kitchen CUTLASS config 0. Not a public '
                'production node.'
            ),
            inputs=[io.Model.Input('model')],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model):
        if not is_minimax_h3(model):
            return io.NodeOutput(model)
        patched = model.clone()
        blocks = get_h3_blocks(patched)
        if not blocks:
            raise BenchmarkConfig0Error('MiniMax H3 has no main blocks')
        for index, block in enumerate(blocks):
            linear = block.attn.qkv_proj
            patched.add_object_patch(
                '%s.%d.attn.qkv_proj.forward' % (PATCH_PREFIX, index),
                _make_forward(linear),
            )
        return io.NodeOutput(patched)
