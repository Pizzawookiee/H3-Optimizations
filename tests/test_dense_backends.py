'''CPU fake-kernel tests for prepared dense Sage backends.'''

from pathlib import Path
import sys
import unittest

import torch

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))

from h3_optimizations.attention.sage_arch import (  # noqa: E402
    KernelBinding,
    SM12xAPI,
    SM80API,
    SM86API,
    SM90API,
    SageSM12xMemoryEfficientBackend,
    SageSM80MemoryEfficientBackend,
    SageSM86MemoryEfficientBackend,
    SageSM90MemoryEfficientBackend,
)
from h3_optimizations.attention.sage_mem_eff import (  # noqa: E402
    PreparedSM89,
    SageSM89API,
    SM89SageMemoryEfficientBackend,
)


def fused_hnd(sequence=65, heads=2):
    inner = heads * 128
    fused = torch.randn(
        sequence,
        inner * 3,
        dtype=torch.bfloat16,
    )
    q, k, v = fused.split(inner, dim=-1)
    return tuple(
        item.view(sequence, heads, 128)
        .transpose(0, 1)
        .unsqueeze(0)
        for item in (q, k, v)
    )


def fake_thread_quantizer(q, k, km=None, **_kwargs):
    if km is not None:
        raise AssertionError('K mean must remain disabled')
    batch, heads, sequence, _dim = q.shape
    q_scale = torch.ones(
        (batch, heads, max(1, (sequence + 63) // 64) * 32)
    )
    k_scale = torch.ones(
        (batch, heads, max(1, (sequence + 63) // 64) * 4)
    )
    return (
        torch.zeros(q.shape, dtype=torch.int8),
        q_scale,
        torch.zeros(k.shape, dtype=torch.int8),
        k_scale,
    )


class FakeKernel:
    def __init__(self, expected_granularity):
        self.expected_granularity = expected_granularity
        self.v_dtype = None

    def __call__(
        self,
        _q,
        _k,
        v,
        out,
        _q_scale,
        _k_scale,
        *args,
    ):
        self.v_dtype = v.dtype
        layout, causal, granularity, _scale, return_lse = args[-5:]
        if (
            layout != 1
            or causal != 0
            or granularity != self.expected_granularity
            or return_lse != 0
        ):
            raise AssertionError('unexpected dense Sage kernel ABI')
        out.zero_()


class FakeFP8:
    def __init__(self):
        self.scale_max = None
        self.sequence = None

    def __call__(self, v, tensor_layout, scale_max, smooth_v):
        if tensor_layout != 'HND' or smooth_v:
            raise AssertionError('unexpected FP8 V contract')
        self.scale_max = scale_max
        self.sequence = int(v.shape[2])
        return (
            torch.zeros(v.shape, dtype=torch.int8),
            torch.ones(
                (v.shape[0], v.shape[1], v.shape[3]),
                dtype=torch.float32,
            ),
            None,
        )


class DenseBackendTests(unittest.TestCase):
    def test_sm89_execute_does_not_requantize_prepared_v(self):
        kernel = FakeKernel(expected_granularity=3)

        def reject_fp8(_v, **_kwargs):
            raise AssertionError('prepared FP8 V must not be quantized again')

        backend = SM89SageMemoryEfficientBackend(
            api=SageSM89API(
                version='2.2.test',
                per_channel_fp8=reject_fp8,
                kernel=kernel,
                kernel_name='fake_sm89',
            ),
            quantizer=fake_thread_quantizer,
            allow_cpu_for_tests=True,
        )
        shape = (1, 2, 65, 128)
        prepared = PreparedSM89(
            q_int8=torch.zeros(shape, dtype=torch.int8),
            q_scale=torch.ones((1, 2, 64), dtype=torch.float32),
            k_int8=torch.zeros(shape, dtype=torch.int8),
            k_scale=torch.ones((1, 2, 8), dtype=torch.float32),
            v_fp8=torch.zeros(shape, dtype=torch.int8),
            v_scale=torch.ones((1, 2, 128), dtype=torch.float32),
            output_dtype=torch.bfloat16,
            layer_index=0,
            sequence=65,
            heads=2,
            head_dim=128,
            softmax_scale=128**-0.5,
            kernel=kernel,
            kernel_name='fake_sm89',
        )

        output = backend.execute(prepared)

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(kernel.v_dtype, torch.int8)

    def test_sm80_uses_independent_fp16_v(self):
        kernel = FakeKernel(expected_granularity=3)
        backend = SageSM80MemoryEfficientBackend(
            api=SM80API(
                version='2.2.test',
                kernel=KernelBinding(kernel, 'fake_sm80', 'test'),
            ),
            quantizer=fake_thread_quantizer,
            allow_cpu_for_tests=True,
        )
        q, k, v = fused_hnd()
        source_pointer = q.untyped_storage().data_ptr()
        prepared = backend.prepare(
            q,
            k,
            v,
            layer_index=0,
            transformer_options={},
        )
        for tensor in (
            prepared.q_int8,
            prepared.k_int8,
            prepared.v_source,
        ):
            self.assertNotEqual(
                tensor.untyped_storage().data_ptr(),
                source_pointer,
            )
        output = backend.execute(prepared)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(kernel.v_dtype, torch.float16)

    def test_sm86_preserves_hnd_and_output_dtype(self):
        seen = {}

        def quantizer(q, k, **kwargs):
            seen['sm_scale'] = kwargs['sm_scale']
            return fake_thread_quantizer(q, k)

        def attention(
            q,
            _k,
            v,
            _q_scale,
            _k_scale,
            **kwargs,
        ):
            seen['v_dtype'] = v.dtype
            seen['layout'] = kwargs['tensor_layout']
            return (
                torch.zeros(q.shape, dtype=kwargs['output_dtype']),
                torch.empty(0),
            )

        backend = SageSM86MemoryEfficientBackend(
            api=SM86API('2.2.test', quantizer, attention),
            allow_cpu_for_tests=True,
        )
        output = backend.execute(
            backend.prepare(
                *fused_hnd(),
                layer_index=1,
                transformer_options={},
            )
        )
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(seen['v_dtype'], torch.float16)
        self.assertEqual(seen['layout'], 'HND')

    def test_sm90_pads_v_and_uses_full_fp8_range(self):
        kernel = FakeKernel(expected_granularity=3)
        fp8 = FakeFP8()
        backend = SageSM90MemoryEfficientBackend(
            api=SM90API(
                '2.2.test',
                fp8,
                KernelBinding(kernel, 'fake_sm90', 'test'),
            ),
            quantizer=fake_thread_quantizer,
            allow_cpu_for_tests=True,
        )
        output = backend.execute(
            backend.prepare(
                *fused_hnd(sequence=65),
                layer_index=2,
                transformer_options={},
            )
        )
        self.assertEqual(output.shape, (1, 2, 65, 128))
        self.assertEqual(fp8.sequence, 128)
        self.assertEqual(fp8.scale_max, 448.0)

    def test_sm12x_uses_per_warp_q_and_bounded_fp8(self):
        kernel = FakeKernel(expected_granularity=2)
        fp8 = FakeFP8()
        seen = {}

        def per_warp(q, k, km=None, **kwargs):
            seen.update(kwargs)
            return fake_thread_quantizer(q, k, km)

        backend = SageSM12xMemoryEfficientBackend(
            api=SM12xAPI(
                '2.2.test',
                per_warp,
                fp8,
                KernelBinding(kernel, 'fake_sm12x', 'test'),
            ),
            allow_cpu_for_tests=True,
        )
        output = backend.execute(
            backend.prepare(
                *fused_hnd(),
                layer_index=3,
                transformer_options={},
            )
        )
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(seen['WARPQ'], 32)
        self.assertEqual(fp8.scale_max, 2.25)


if __name__ == '__main__':
    unittest.main()
