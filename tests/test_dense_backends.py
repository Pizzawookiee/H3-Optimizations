'''CPU fake-kernel tests for prepared dense Sage backends.'''

from pathlib import Path
import os
import sys
import unittest
from unittest import mock

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.attention.sage_arch import (  # noqa: E402
    KernelBinding,
    SM12xAPI,
    SM80API,
    SM86API,
    SM90API,
    SageSM12xMemoryEfficientBackend,
    SageSM75MemoryEfficientBackend,
    SageSM80MemoryEfficientBackend,
    SageSM86MemoryEfficientBackend,
    SageSM90MemoryEfficientBackend,
)
from h3_optimizations.attention.sage_mem_eff import (  # noqa: E402
    PreparedSM89,
    SageSM89API,
    SM89SageMemoryEfficientBackend,
    _load_api,
    _resolve_sm89_kernel,
)
from h3_optimizations.dense_fused_qkv import DenseFusedQKVProjector  # noqa: E402
import h3_optimizations.qkv.streamed as streamed_qkv  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


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
        self.shapes = []

    def __call__(
        self,
        q,
        k,
        v,
        out,
        _q_scale,
        _k_scale,
        *args,
    ):
        self.v_dtype = v.dtype
        self.shapes.append((q.shape[-2], k.shape[-2], v.shape[-2], out.shape[-2]))
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
    def test_every_sage_architecture_executes_rectangular_q_against_global_kv(self):
        kernel3 = FakeKernel(expected_granularity=3)
        kernel2 = FakeKernel(expected_granularity=2)
        fp8 = FakeFP8()

        def per_block(q, k, **_kwargs):
            return fake_thread_quantizer(q, k)

        def attention(q, k, v, _q_scale, _k_scale, **kwargs):
            return torch.zeros(q.shape, dtype=kwargs['output_dtype']), torch.empty(0)

        def per_warp(q, k, km=None, **_kwargs):
            return fake_thread_quantizer(q, k, km)

        backends = (
            SageSM75MemoryEfficientBackend(
                api=SM86API('2.2.test', per_block, attention),
                allow_cpu_for_tests=True,
            ),
            SageSM80MemoryEfficientBackend(
                api=SM80API('2.2.test', KernelBinding(kernel3, 'sm80', 'test')),
                quantizer=fake_thread_quantizer,
                allow_cpu_for_tests=True,
            ),
            SageSM86MemoryEfficientBackend(
                api=SM86API('2.2.test', per_block, attention),
                allow_cpu_for_tests=True,
            ),
            SM89SageMemoryEfficientBackend(
                api=SageSM89API(
                    '2.2.test',
                    fp8,
                    kernel3,
                    'sm89',
                ),
                quantizer=fake_thread_quantizer,
                allow_cpu_for_tests=True,
            ),
            SageSM90MemoryEfficientBackend(
                api=SM90API('2.2.test', fp8, KernelBinding(kernel3, 'sm90', 'test')),
                quantizer=fake_thread_quantizer,
                allow_cpu_for_tests=True,
            ),
            SageSM12xMemoryEfficientBackend(
                api=SM12xAPI(
                    '2.2.test',
                    per_warp,
                    fp8,
                    KernelBinding(kernel2, 'sm12x', 'test'),
                ),
                allow_cpu_for_tests=True,
            ),
        )
        q = torch.zeros((1, 2, 33, 128), dtype=torch.bfloat16)
        k = torch.zeros((1, 2, 65, 128), dtype=torch.bfloat16)
        v = torch.zeros_like(k)

        for backend in backends:
            with self.subTest(backend=backend.name):
                q_int8, q_scale = backend.quantize_projected_q(q)
                k_int8, k_scale = backend.quantize_projected_k(k)
                v_carrier, v_scale = backend.prepare_streamed_v(v)
                output = backend.execute_rectangular(
                    q_int8,
                    q_scale,
                    k_int8,
                    k_scale,
                    v_carrier,
                    v_scale,
                    output_dtype=torch.bfloat16,
                    softmax_scale=128**-0.5,
                    layer_index=4,
                )
                self.assertEqual(tuple(output.shape), (1, 2, 33, 128))
    def test_dense_sage_projector_assembles_chunked_kitchen_projection(self):
        projected_ranges = []
        quantized_lengths = []

        def project_chunk(_module, _x, _rope, start, stop):
            projected_ranges.append((start, stop))
            shape = (1, 2, stop - start, 128)
            value = len(projected_ranges)
            return (
                torch.full(shape, value, dtype=torch.bfloat16),
                torch.full(shape, value + 10, dtype=torch.bfloat16),
                torch.full(shape, value + 20, dtype=torch.bfloat16),
            )

        def quantizer(q, k, _km, **kwargs):
            self.assertEqual(kwargs['tensor_layout'], 'HND')
            length = int(q.shape[2])
            quantized_lengths.append(length)
            q_scales = ((length + 127) // 128) * 32
            k_scales = ((length + 63) // 64) * 4
            return (
                torch.full(q.shape, len(quantized_lengths), dtype=torch.int8),
                torch.full((1, 2, q_scales), len(quantized_lengths), dtype=torch.float32),
                torch.full(k.shape, len(quantized_lengths) + 10, dtype=torch.int8),
                torch.full((1, 2, k_scales), len(quantized_lengths) + 10, dtype=torch.float32),
            )

        projector = DenseFusedQKVProjector(
            chunk_rows=128,
            quantizer=quantizer,
            project_chunk=project_chunk,
            allow_cpu_for_tests=True,
        )
        prepared = projector.project(
            type('Attention', (), {'heads': 2, 'head_dim': 128})(),
            torch.empty((257, 4), dtype=torch.bfloat16),
            None,
            layer_index=7,
            transformer_options={},
        )

        self.assertEqual(projected_ranges, [(0, 128), (128, 256), (256, 257)])
        self.assertEqual(quantized_lengths, [128, 128, 1])
        self.assertEqual(tuple(prepared.q_int8.shape), (1, 2, 257, 128))
        self.assertEqual(tuple(prepared.q_scale.shape), (1, 2, 96))
        self.assertEqual(tuple(prepared.k_scale.shape), (1, 2, 20))
        self.assertTrue(torch.all(prepared.v[:, :, :128] == 21))
        self.assertTrue(torch.all(prepared.v[:, :, 128:256] == 22))
        self.assertTrue(torch.all(prepared.v[:, :, 256:] == 23))
        self.assertEqual(prepared.layer_index, 7)

    def test_dense_sage_projector_holds_source_binding_into_native_carrier(self):
        class HeldQKV:
            def __init__(self):
                self.calls = []
                self.released = False

            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _tb):
                self.released = True
                return False

            def project_hnd(self, _x, _rope, start, stop):
                self.calls.append((start, stop))
                shape = (1, 2, stop - start, 128)
                value = torch.zeros(shape, dtype=torch.bfloat16)
                return value, value, value

        def quantizer(q, k, _km, **_kwargs):
            length = int(q.shape[2])
            return (
                torch.zeros_like(q, dtype=torch.int8),
                torch.ones((1, 2, ((length + 127) // 128) * 32)),
                torch.zeros_like(k, dtype=torch.int8),
                torch.ones((1, 2, ((length + 63) // 64) * 4)),
            )

        held = HeldQKV()
        projector = DenseFusedQKVProjector(
            chunk_rows=128,
            quantizer=quantizer,
            projection_mode='force_bf16',
            allow_cpu_for_tests=True,
        )
        module = type('Attention', (), {'heads': 2, 'head_dim': 128})()
        source = torch.empty((257, 4), dtype=torch.bfloat16)
        with mock.patch.object(
            streamed_qkv,
            'create_held_qkv',
            return_value=held,
        ) as factory:
            prepared = projector.project(
                module,
                source,
                None,
                layer_index=7,
                transformer_options={},
            )

        factory.assert_called_once()
        self.assertIs(factory.call_args.args[0], module)
        self.assertIs(factory.call_args.args[1]._base, source)
        self.assertEqual(factory.call_args.args[2], 'force_bf16')
        self.assertEqual(held.calls, [(0, 128), (128, 256), (256, 257)])
        self.assertTrue(held.released)
        self.assertEqual(prepared.qk_format, 'sage_per_thread_int8')

    def test_architecture_backends_accept_their_direct_projected_carriers(self):
        kernel = FakeKernel(expected_granularity=3)
        fp8 = FakeFP8()

        def per_block(q, k, **_kwargs):
            return fake_thread_quantizer(q, k)

        def attention(*_args, **_kwargs):
            raise AssertionError('carrier preparation must not execute attention')

        def per_warp(q, k, km=None, **_kwargs):
            return fake_thread_quantizer(q, k, km)

        backends = (
            SageSM75MemoryEfficientBackend(
                api=SM86API('2.2.test', per_block, attention),
                allow_cpu_for_tests=True,
            ),
            SageSM80MemoryEfficientBackend(
                api=SM80API(
                    '2.2.test',
                    KernelBinding(kernel, 'fake_sm80', 'test'),
                ),
                quantizer=fake_thread_quantizer,
                allow_cpu_for_tests=True,
            ),
            SageSM86MemoryEfficientBackend(
                api=SM86API('2.2.test', per_block, attention),
                allow_cpu_for_tests=True,
            ),
            SageSM90MemoryEfficientBackend(
                api=SM90API(
                    '2.2.test',
                    fp8,
                    KernelBinding(kernel, 'fake_sm90', 'test'),
                ),
                quantizer=fake_thread_quantizer,
                allow_cpu_for_tests=True,
            ),
            SageSM12xMemoryEfficientBackend(
                api=SM12xAPI(
                    '2.2.test',
                    per_warp,
                    fp8,
                    KernelBinding(kernel, 'fake_sm12x', 'test'),
                ),
                allow_cpu_for_tests=True,
            ),
        )

        def project_chunk(_module, _x, _rope, start, stop):
            shape = (1, 2, stop - start, 128)
            value = torch.zeros(shape, dtype=torch.bfloat16)
            return value, value, value

        module = type('Attention', (), {'heads': 2, 'head_dim': 128})()
        source = torch.empty((257, 4), dtype=torch.bfloat16)
        for backend in backends:
            with self.subTest(backend=backend.name):
                projector = DenseFusedQKVProjector(
                    chunk_rows=128,
                    project_chunk=project_chunk,
                    carrier_backend=backend,
                    allow_cpu_for_tests=True,
                )
                projected = projector.project(
                    module,
                    source,
                    None,
                    layer_index=4,
                    transformer_options={},
                )
                prepared = backend.prepare_projected(
                    projected,
                    layer_index=4,
                    transformer_options={},
                )

                self.assertEqual(
                    projected.qk_format,
                    backend.projected_qkv_format,
                )
                self.assertIs(prepared.v_source, projected.v)
                self.assertEqual(prepared.sequence, 257)

    def test_sm89_resolves_extension_alias_used_by_public_dispatcher(self):
        kernel = FakeKernel(expected_granularity=3)
        extension = type('WheelExtension', (), {
            'qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf': kernel,
        })()
        namespace = {'wheel_chosen_alias': extension}
        exec('def dispatch():\n    return wheel_chosen_alias', namespace)
        core = type('SageCore', (), {
            'sageattn_qk_int8_pv_fp8_cuda': staticmethod(namespace['dispatch']),
        })()

        resolved, name, source, scale_max, accumulation = _resolve_sm89_kernel(core)

        self.assertIs(resolved, kernel)
        self.assertEqual(name, 'qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf')
        self.assertEqual(source, 'WheelExtension')
        self.assertEqual(scale_max, 2.25)
        self.assertEqual(accumulation, 'fp32+fp16')

    def test_sm89_accepts_any_version_with_compatible_capabilities(self):
        kernel = FakeKernel(expected_granularity=3)
        extension = type('WheelExtension', (), {
            'qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf': kernel,
        })()
        namespace = {'wheel_chosen_alias': extension}
        exec('def dispatch():\n    return wheel_chosen_alias', namespace)
        core = type(sys)('sageattention.core')
        core.SM89_ENABLED = True
        core.per_channel_fp8 = lambda *_args, **_kwargs: None
        core.sageattn_qk_int8_pv_fp8_cuda = namespace['dispatch']
        package = type(sys)('sageattention')
        package.__path__ = []
        package.core = core

        with mock.patch.dict(
            sys.modules,
            {'sageattention': package, 'sageattention.core': core},
        ), mock.patch(
            'h3_optimizations.attention.sage_mem_eff.importlib.metadata.version',
            return_value='9.7.test',
        ):
            api = _load_api()

        self.assertEqual(api.version, '9.7.test')
        self.assertIs(api.kernel, kernel)

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
