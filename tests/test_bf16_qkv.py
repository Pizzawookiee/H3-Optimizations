'''CPU contracts for the reusable chunked BF16 H3 QKV projector.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
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

from h3_optimizations.plan import (  # noqa: E402
    ATTENTION_EXISTING,
    FUSED_QKV_AUTO,
    FUSED_QKV_OFF,
    FUSED_QKV_PRESERVE_BF16,
    MemoryRequest,
)
from h3_optimizations.attention_forward import (  # noqa: E402
    _finish_projected,
    _legacy_attention,
    finish_qkv_projection,
    make_forward,
    to_hnd,
)
from h3_optimizations.kitchen_qkv import (  # noqa: E402
    ChunkedKitchenAttentionBackend,
)
import h3_optimizations.qkv.bf16 as bf16_module  # noqa: E402
from h3_optimizations.qkv.fp8 import HeldFP8QKV  # noqa: E402
from h3_optimizations.qkv.w4a8 import HeldW4A8QKV  # noqa: E402
from h3_optimizations.qkv.bf16 import (  # noqa: E402
    BF16QKVBindingError,
    CHUNK_ROWS,
    ChunkedBF16QKVProjector,
    PreparedBF16QKV,
    StreamedDenseBF16QKVProjector,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    QKV_BF16_CHUNKED,
    QKV_DENSE_KITCHEN_CHUNKED,
    QKV_STANDARD,
    QKV_STREAMED_BF16_KITCHEN,
    resolve_qkv_provider,
)
from h3_optimizations.qkv.projectors import (  # noqa: E402
    TritonSparseQKVProjector,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakeInventory:
    def __init__(self, dtype='torch.bfloat16'):
        item = SimpleNamespace(
            plain_float=True,
            logical_dtype=dtype,
            label='Tensor:%s' % dtype,
        )
        self.qkv = (item,)
        self.qkv_w4a8 = False
        self.qkv_fp8 = False
        self.qkv_plain_float = True
        self.qkv_convrot_int8_256 = False

    def homogeneous(self, name):
        return name == 'qkv'

    def labels(self, name):
        return tuple(item.label for item in getattr(self, name))


class ChunkedBF16QKVContracts(unittest.TestCase):
    @staticmethod
    def _module():
        torch.manual_seed(1234)
        return SimpleNamespace(
            heads=2,
            head_dim=4,
            qkv_proj=torch.nn.Linear(8, 24, bias=True, dtype=torch.bfloat16),
            q_norm=torch.nn.RMSNorm(4, eps=1e-6, dtype=torch.bfloat16),
            k_norm=torch.nn.RMSNorm(4, eps=1e-6, dtype=torch.bfloat16),
            out_proj=torch.nn.Linear(8, 8, bias=True, dtype=torch.bfloat16),
        )

    @staticmethod
    def _fake_held():
        class FakeHeld(bf16_module.HeldBF16QKV):
            def __enter__(self):
                self.weight = self.attention.qkv_proj.weight
                self.bias = self.attention.qkv_proj.bias
                return self

            def __exit__(self, exc_type, exc, tb):
                self.weight = self.bias = None
                return False

        return FakeHeld
    def test_existing_attention_is_requested_to_return_hnd(self):
        calls = []

        def attention(*args, **kwargs):
            calls.append((args, kwargs))
            return args[0]

        q = torch.empty((1, 2, 3, 4), dtype=torch.bfloat16)
        self.assertIs(
            _legacy_attention(
                SimpleNamespace(heads=2),
                q,
                q,
                q,
                {},
                attention=attention,
            ),
            q,
        )
        self.assertTrue(calls[0][1]['skip_reshape'])
        self.assertTrue(calls[0][1]['skip_output_reshape'])

    def test_default_chunk_rows_is_4096(self):
        projector = ChunkedBF16QKVProjector()
        self.assertEqual(CHUNK_ROWS, 4096)
        self.assertEqual(projector.chunk_rows, 4096)
        self.assertEqual(
            projector.installation_signature,
            ('chunked_bf16_qkv', 4096, False, False, False),
        )
        forced = ChunkedBF16QKVProjector(force_weights_bf16=True)
        self.assertEqual(
            forced.installation_signature,
            ('chunked_bf16_qkv', 4096, True, False, False),
        )
        quantized = ChunkedBF16QKVProjector(force_weights_fp8=True)
        self.assertEqual(
            quantized.installation_signature,
            ('chunked_bf16_qkv', 4096, False, True, False),
        )

        int8 = ChunkedBF16QKVProjector(force_weights_int8=True)
        self.assertEqual(
            int8.installation_signature,
            ('chunked_bf16_qkv', 4096, False, False, True),
        )

    def test_triton_force_int8_keeps_named_streamed_projector(self):
        projector = TritonSparseQKVProjector(force_weights_int8=True)
        self.assertFalse(projector.streamed_qkv)
        self.assertTrue(projector.streamed_q)
        self.assertTrue(projector._implementation.force_weights_int8)

    def test_direct_streamed_backend_uses_lazy_out_projection(self):
        expected = torch.empty((4, 8), dtype=torch.bfloat16)
        source = torch.empty((4, 8), dtype=torch.bfloat16)
        out_projection = SimpleNamespace(
            linear=mock.Mock(return_value=expected),
        )
        module = SimpleNamespace(
            out_proj=mock.Mock(side_effect=AssertionError('BF16 out_proj ran')),
        )

        class Backend:
            name = 'direct'

            @staticmethod
            def execute_projected(execution_module, _prepared):
                return execution_module.out_proj(source)

        self.assertIs(
            _finish_projected(
                module,
                Backend(),
                object(),
                out_projection,
            ),
            expected,
        )
        out_projection.linear.assert_called_once_with(source)
        module.out_proj.assert_not_called()

    def test_chunk_rows_must_be_positive(self):
        for value in (0, -1, -4096):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ChunkedBF16QKVProjector(value)

    def test_prepared_shape_metadata(self):
        q = torch.empty((1, 56, 17, 128), dtype=torch.bfloat16)
        prepared = PreparedBF16QKV(q=q, k=q.clone(), v=q.clone())
        self.assertEqual(prepared.sequence, 17)
        self.assertEqual(prepared.heads, 56)
        self.assertEqual(prepared.head_dim, 128)

    def test_cpu_activation_is_rejected_before_weight_acquisition(self):
        projector = ChunkedBF16QKVProjector()
        module = type('Attention', (), {})()
        module.qkv_proj = type('Linear', (), {})()
        x = torch.empty((8, 16), dtype=torch.bfloat16)
        with self.assertRaisesRegex(BF16QKVBindingError, 'CUDA BF16'):
            projector._validate(module, x, None)

    def test_try_project_is_non_strict_for_ineligible_calls(self):
        projector = ChunkedBF16QKVProjector()
        module = type('Attention', (), {})()
        module.qkv_proj = type('Linear', (), {})()
        x = torch.empty((8, 16), dtype=torch.bfloat16)
        self.assertIsNone(
            projector.try_project(
                module,
                x,
                None,
                layer_index=0,
                transformer_options={},
            )
        )
        forced = ChunkedBF16QKVProjector(force_weights_bf16=True)
        with self.assertRaisesRegex(BF16QKVBindingError, 'CUDA BF16'):
            forced.try_project(
                module,
                x,
                None,
                layer_index=0,
                transformer_options={},
            )

    def test_source_keeps_streaming_and_full_materialization_separate(self):
        text = (PACK / 'h3_optimizations' / 'qkv' / 'bf16.py').read_text(
            encoding='utf-8'
        )
        self.assertIn('def stream(', text)
        self.assertIn('def project(', text)
        self.assertIn('consume_chunk(start, end, q, k, v)', text)
        self.assertIn('for start in range(0, sequence, self.chunk_rows)', text)
        self.assertNotIn('torch.cat(', text)

    def test_dense_streaming_slices_kv_then_q_without_full_qkv_projection(self):
        module = self._module()
        x = torch.randn((7, 8), dtype=torch.bfloat16)
        projected = torch.nn.functional.linear(
            x,
            module.qkv_proj.weight,
            module.qkv_proj.bias,
        )
        expected = to_hnd(*finish_qkv_projection(module, projected, None))
        projector = StreamedDenseBF16QKVProjector(
            chunk_rows=3,
            allow_cpu_for_tests=True,
        )

        with mock.patch.object(
            bf16_module,
            'HeldBF16QKV',
            self._fake_held(),
        ), mock.patch.object(
            bf16_module.F,
            'linear',
            wraps=bf16_module.F.linear,
        ) as linear:
            prepared = projector.project(module, x, None)
            chunks = list(prepared.stream_q())

        self.assertEqual(
            [(start, end) for start, end, _q in chunks],
            [(0, 3), (3, 6), (6, 7)],
        )
        actual_q = torch.cat([q for _start, _end, q in chunks], dim=2)
        torch.testing.assert_close(actual_q, expected[0])
        torch.testing.assert_close(prepared.k, expected[1])
        torch.testing.assert_close(prepared.v, expected[2])
        self.assertEqual(
            [int(call.args[1].shape[0]) for call in linear.call_args_list],
            [16, 16, 16, 8, 8, 8],
        )
        prepared.release()

    def test_sliced_q_and_k_use_single_tensor_partial_split_half_rope(self):
        module = self._module()
        x = torch.randn((7, 8), dtype=torch.bfloat16)
        rope = torch.randn((1, 7, 1, 1, 2, 2), dtype=torch.float32)
        held = self._fake_held()(module, x[:1])
        held.__enter__()

        with mock.patch.object(
            bf16_module.comfy.quant_ops.ck,
            'apply_rope_split_half1_',
        ) as apply_rope:
            held.project_kv_hnd(x, rope, 2, 5)
            held.project_q_hnd(x, rope, 2, 5)

        self.assertEqual(apply_rope.call_count, 2)
        for call in apply_rope.call_args_list:
            self.assertEqual(tuple(call.args[0].shape), (1, 3, 2, 2))
            self.assertIs(call.args[1]._base, rope)
            self.assertEqual(tuple(call.args[1].shape), (1, 3, 1, 1, 2, 2))
        held.__exit__(None, None, None)

    def test_dense_streaming_partial_rope_matches_h3_projection(self):
        module = self._module()
        for layer in (
            module.qkv_proj,
            module.q_norm,
            module.k_norm,
            module.out_proj,
        ):
            layer.requires_grad_(False)
        x = torch.randn((7, 8), dtype=torch.bfloat16)
        rope = torch.randn((1, 7, 1, 1, 2, 2), dtype=torch.float32)
        projected = torch.nn.functional.linear(
            x,
            module.qkv_proj.weight,
            module.qkv_proj.bias,
        )
        expected = to_hnd(*finish_qkv_projection(module, projected, rope))
        projector = StreamedDenseBF16QKVProjector(
            chunk_rows=3,
            allow_cpu_for_tests=True,
        )

        with mock.patch.object(
            bf16_module,
            'HeldBF16QKV',
            self._fake_held(),
        ):
            prepared = projector.project(module, x, rope)
            actual_q = torch.cat(
                [q for _start, _end, q in prepared.stream_q()],
                dim=2,
            )

        torch.testing.assert_close(actual_q, expected[0])
        torch.testing.assert_close(prepared.k, expected[1])
        torch.testing.assert_close(prepared.v, expected[2])
        prepared.release()

    def test_dense_streaming_declines_when_single_tensor_rope_is_unavailable(self):
        module = self._module()
        x = torch.randn((7, 8), dtype=torch.bfloat16)
        rope = torch.randn((1, 7, 1, 3, 2, 2), dtype=torch.float32)
        projector = StreamedDenseBF16QKVProjector(
            chunk_rows=3,
            allow_cpu_for_tests=True,
        )

        with mock.patch.object(
            bf16_module.comfy.quant_ops.ck,
            'apply_rope_split_half1_',
            None,
        ):
            self.assertIsNone(
                projector.try_project(
                    module,
                    x,
                    rope,
                    layer_index=0,
                    transformer_options={},
                )
            )

    def test_dense_streaming_projects_each_attention_output_into_input(self):
        module = self._module()
        source = torch.randn((7, 8), dtype=torch.bfloat16)
        projector = StreamedDenseBF16QKVProjector(
            chunk_rows=3,
            allow_cpu_for_tests=True,
        )
        calls = []

        def attention(q, k, v, _heads, **_kwargs):
            calls.append((int(q.shape[2]), int(k.shape[2]), int(v.shape[2])))
            return q + k.mean(dim=2, keepdim=True) + v.mean(dim=2, keepdim=True)

        with mock.patch.object(
            bf16_module,
            'HeldBF16QKV',
            self._fake_held(),
        ):
            reference = projector.project(module, source.clone(), None)
            full_q = torch.cat(
                [q for _start, _end, q in reference.stream_q()],
                dim=2,
            )
            raw = attention(full_q, reference.k, reference.v, module.heads)
            expected = module.out_proj(
                raw.transpose(1, 2).reshape(1, 7, 8).squeeze(0)
            )
            reference.release()
            calls.clear()

            forward = make_forward(
                module,
                0,
                backend=ChunkedKitchenAttentionBackend(),
                projector=projector,
            )
            actual_buffer = source.clone()
            with mock.patch.object(
                bf16_module,
                'HeldBF16QKV',
                self._fake_held(),
            ), mock.patch(
                'h3_optimizations.attention_forward.h3_model.optimized_attention',
                side_effect=attention,
            ):
                actual = forward(actual_buffer, transformer_options={})

        self.assertIs(actual, actual_buffer)
        self.assertEqual(calls, [(3, 7, 7), (3, 7, 7), (1, 7, 7)])
        torch.testing.assert_close(actual, expected)

    def test_dense_streaming_uses_opted_in_external_consumer(self):
        module = self._module()
        source = torch.randn((7, 8), dtype=torch.bfloat16)
        projector = StreamedDenseBF16QKVProjector(
            chunk_rows=3,
            allow_cpu_for_tests=True,
        )
        calls = []

        def external_sage(*_args, **_kwargs):
            raise AssertionError('streamed ABI must bypass the ordinary override')

        def consume(q_chunk, global_k, global_v, q_start, q_total,
                    layer_index, transformer_options):
            calls.append((
                int(q_chunk.shape[2]),
                int(global_k.shape[2]),
                int(global_v.shape[2]),
                q_start,
                q_total,
                layer_index,
                transformer_options['consumer_token'],
            ))
            return q_chunk

        external_sage.supports_streamed_h3_qkv = True
        external_sage.consume = consume

        forward = make_forward(
            module,
            0,
            backend=ChunkedKitchenAttentionBackend(),
            projector=projector,
        )
        output = source.clone()
        with mock.patch.object(
            bf16_module,
            'HeldBF16QKV',
            self._fake_held(),
        ):
            actual = forward(
                output,
                transformer_options={
                    'optimized_attention_override': external_sage,
                    'consumer_token': 'kept',
                },
            )

        self.assertIs(actual, output)
        self.assertEqual(calls, [
            (3, 7, 7, 0, 7, 0, 'kept'),
            (3, 7, 7, 3, 7, 0, 'kept'),
            (1, 7, 7, 6, 7, 0, 'kept'),
        ])

    def test_full_q_projector_invokes_unknown_override_once(self):
        module = self._module()
        source = torch.randn((7, 8), dtype=torch.bfloat16)
        projector = ChunkedBF16QKVProjector(chunk_rows=3)
        calls = []

        def unknown_override(_original, q, k, v, _heads, **_kwargs):
            calls.append((int(q.shape[2]), int(k.shape[2]), int(v.shape[2])))
            return q

        forward = make_forward(
            module,
            0,
            backend=ChunkedKitchenAttentionBackend(),
            projector=projector,
        )
        with mock.patch.object(
            bf16_module,
            'HeldBF16QKV',
            self._fake_held(),
        ), mock.patch.object(
            projector,
            '_validate',
            return_value=SimpleNamespace(plain_float=True),
        ):
            forward(
                source.clone(),
                transformer_options={
                    'optimized_attention_override': unknown_override,
                },
            )

        self.assertEqual(calls, [(7, 7, 7)])

    def test_dense_native_quantized_source_never_retains_full_q(self):
        module = self._module()
        source = torch.randn((7, 8), dtype=torch.bfloat16)

        class FullProjectionOnlyHeld:
            def __init__(self):
                self.calls = []
                self.released = False

            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _tb):
                self.released = True
                return False

            def project_hnd(self, x, _rope, start, end):
                self.calls.append((start, end))
                rows = end - start
                values = x[start:end].view(rows, 2, 4).transpose(0, 1).unsqueeze(0)
                return values, values + 1, values + 2

        held = FullProjectionOnlyHeld()
        projector = StreamedDenseBF16QKVProjector(
            chunk_rows=3,
            allow_cpu_for_tests=True,
        )
        quantized_format = SimpleNamespace(
            plain_float=False,
            convrot_int8_256=False,
            w4a8=False,
            fp8=True,
            logical_dtype='float8_e4m3fn',
            label='FP8',
        )
        with mock.patch.object(
            projector,
            '_validate',
            return_value=quantized_format,
        ), mock.patch(
            'h3_optimizations.qkv.streamed.create_held_qkv',
            return_value=held,
        ) as factory:
            prepared = projector.project(module, source, None)
            q_chunks = list(prepared.stream_q())

        factory.assert_called_once()
        factory_args = factory.call_args.args
        self.assertIs(factory_args[0], module)
        self.assertIs(factory_args[1]._base, source)
        self.assertEqual(factory_args[2], 'native')
        self.assertFalse(hasattr(prepared, 'q'))
        self.assertEqual(
            [(start, end) for start, end, _q in q_chunks],
            [(0, 3), (3, 6), (6, 7)],
        )
        self.assertEqual(
            held.calls,
            [(0, 3), (3, 6), (6, 7), (0, 3), (3, 6), (6, 7)],
        )
        prepared.release()
        self.assertTrue(held.released)

    def test_dense_streaming_reacquires_reusable_cast_binding_per_q_slab(self):
        module = self._module()
        source = torch.randn((7, 8), dtype=torch.bfloat16)
        instances = []

        class ReusableCastHeld:
            def __init__(self):
                self.binding = SimpleNamespace(handle=object())
                self.released = False
                self.calls = []
                instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _tb):
                self.released = True
                self.binding.handle = None
                return False

            def project_hnd(self, x, _rope, start, end):
                self.calls.append((start, end))
                rows = end - start
                values = x[start:end].view(rows, 2, 4).transpose(0, 1).unsqueeze(0)
                return values, values + 1, values + 2

        projector = StreamedDenseBF16QKVProjector(
            chunk_rows=3,
            allow_cpu_for_tests=True,
        )
        quantized_format = SimpleNamespace(
            plain_float=False,
            convrot_int8_256=False,
            w4a8=True,
            fp8=False,
            logical_dtype='int4',
            label='W4A8',
        )
        with mock.patch.object(
            projector,
            '_validate',
            return_value=quantized_format,
        ), mock.patch(
            'h3_optimizations.qkv.streamed.create_held_qkv',
            side_effect=lambda *_args: ReusableCastHeld(),
        ):
            prepared = projector.project(module, source, None)
            self.assertTrue(instances[0].released)
            self.assertIsNone(prepared.held)
            for _start, _end, _q in prepared.stream_q():
                self.assertTrue(instances[-1].released)

        self.assertEqual(len(instances), 4)
        self.assertTrue(all(item.released for item in instances))
        prepared.release()

    def test_convrot_exposes_kv_only_projection(self):
        module = self._module()
        source = torch.randn((7, 8), dtype=torch.bfloat16)
        held = bf16_module.HeldConvRotINT8QKV(module, source[:1])
        calls = []

        class Binding:
            def linear_range(self, rows, start, end):
                calls.append((int(rows.shape[0]), start, end))
                return torch.randn(
                    (rows.shape[0], end - start),
                    dtype=torch.bfloat16,
                )

        held.binding = Binding()
        k, v = held.project_kv_hnd(source, None, 2, 5)

        self.assertEqual(calls, [(3, 8, 24)])
        self.assertEqual(tuple(k.shape), (1, 2, 3, 4))
        self.assertEqual(tuple(v.shape), (1, 2, 3, 4))

    def test_fp8_and_w4a8_expose_split_q_and_kv_projection(self):
        module = self._module()
        source = torch.randn((7, 8), dtype=torch.bfloat16)

        for held_type in (HeldFP8QKV, HeldW4A8QKV):
            with self.subTest(held_type=held_type.__name__):
                held = held_type.__new__(held_type)
                held.attention = module
                calls = []

                class Binding:
                    def linear_range(self, rows, start, end):
                        calls.append((int(rows.shape[0]), start, end))
                        return torch.randn(
                            (rows.shape[0], end - start),
                            dtype=torch.bfloat16,
                        )

                held.binding = Binding()
                q = held.project_q_hnd(source, None, 2, 5)
                k, v = held.project_kv_hnd(source, None, 2, 5)

                self.assertEqual(calls, [(3, 0, 8), (3, 8, 24)])
                self.assertEqual(tuple(q.shape), (1, 2, 3, 4))
                self.assertEqual(tuple(k.shape), (1, 2, 3, 4))
                self.assertEqual(tuple(v.shape), (1, 2, 3, 4))

    def test_preserve_precision_has_an_internal_bf16_qkv_request(self):
        request = MemoryRequest(
            attention=ATTENTION_EXISTING,
            fused_qkv=FUSED_QKV_OFF,
        )
        self.assertEqual(request.fused_qkv, FUSED_QKV_PRESERVE_BF16)
        explicit_off = MemoryRequest(fused_qkv=FUSED_QKV_OFF)
        self.assertEqual(explicit_off.fused_qkv, FUSED_QKV_OFF)

    def test_preserve_bf16_has_one_distinct_provider_for_all_consumers(self):
        backend_kinds = (
            'existing',
            'sparse_sage',
            'triton_sparse_bf16',
            'flex_attention_fp8',
        )
        for backend_kind in backend_kinds:
            with self.subTest(backend_kind=backend_kind):
                resolved = resolve_qkv_provider(
                    FakeInventory(),
                    request=FUSED_QKV_PRESERVE_BF16,
                    backend_kind=backend_kind,
                    triton_available=True,
                    memory_optimize=True,
                )
                self.assertEqual(resolved.provider_id, QKV_BF16_CHUNKED)
                self.assertFalse(resolved.fused)
                self.assertIn('held 4K', resolved.reason)

    def test_preserve_bf16_does_not_convert_fp16(self):
        resolved = resolve_qkv_provider(
            FakeInventory('torch.float16'),
            request=FUSED_QKV_PRESERVE_BF16,
            backend_kind='existing',
            memory_optimize=True,
        )
        self.assertEqual(resolved.provider_id, QKV_STANDARD)

    def test_preserve_precision_streams_convrot_bf16_into_sparse_kitchen(self):
        inventory = FakeInventory()
        inventory.qkv[0].plain_float = False
        inventory.qkv_plain_float = False
        inventory.qkv_convrot_int8_256 = True

        resolved = resolve_qkv_provider(
            inventory,
            request=FUSED_QKV_PRESERVE_BF16,
            backend_kind='sparse_kitchen_int8',
            kitchen_producer_available=True,
        )

        self.assertEqual(resolved.provider_id, QKV_STREAMED_BF16_KITCHEN)
        self.assertTrue(resolved.fused)
        self.assertIn('streams BF16 projection chunks', resolved.reason)

        unavailable = resolve_qkv_provider(
            inventory,
            request=FUSED_QKV_PRESERVE_BF16,
            backend_kind='sparse_kitchen_int8',
            kitchen_producer_available=False,
        )
        self.assertEqual(unavailable.provider_id, QKV_STANDARD)
        self.assertFalse(unavailable.fused)

    def test_attention_forward_consumes_bf16_without_reprojection(self):
        text = (
            PACK / 'h3_optimizations' / 'attention_forward.py'
        ).read_text(encoding='utf-8')
        self.assertIn('isinstance(projected, PreparedBF16QKV)', text)
        self.assertIn('backend.prepare(', text)
        self.assertIn('DENSE_KITCHEN_PREQUANTIZED', text)
        self.assertIn('_legacy_attention(', text)

    def test_apply_dispatches_bf16_to_dense_and_sparse_consumers(self):
        text = (PACK / 'h3_optimizations' / 'apply.py').read_text(
            encoding='utf-8'
        )
        self.assertIn('QKV_BF16_CHUNKED', text)
        self.assertIn('StreamedDenseBF16QKVProjector', text)
        self.assertGreaterEqual(
            text.count('_bounded_qkv_projector('),
            5,
        )
        self.assertIn('projector=attention.projector', text)

if __name__ == '__main__':
    unittest.main()
