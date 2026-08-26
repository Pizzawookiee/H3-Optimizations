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
    _legacy_attention,
    finish_qkv_projection,
    make_forward,
    to_hnd,
)
from h3_optimizations.kitchen_qkv import (  # noqa: E402
    ChunkedKitchenAttentionBackend,
)
import h3_optimizations.qkv.bf16 as bf16_module  # noqa: E402
from h3_optimizations.qkv.bf16 import (  # noqa: E402
    BF16QKVBindingError,
    CHUNK_ROWS,
    ChunkedBF16QKVProjector,
    FrostBF16QKVProjector,
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
            ('chunked_bf16_qkv', 4096, False, False),
        )
        forced = ChunkedBF16QKVProjector(force_weights_bf16=True)
        self.assertEqual(
            forced.installation_signature,
            ('chunked_bf16_qkv', 4096, True, False),
        )
        quantized = ChunkedBF16QKVProjector(force_weights_fp8=True)
        self.assertEqual(
            quantized.installation_signature,
            ('chunked_bf16_qkv', 4096, False, True),
        )

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

    def test_frost_projector_returns_hnd_views_over_sequence_major_storage(self):
        projector = FrostBF16QKVProjector(chunk_rows=3)
        module = SimpleNamespace(heads=2, head_dim=4)
        x = torch.empty((5, 8), dtype=torch.bfloat16)

        def stream(_module, _x, _rope, consume):
            for start, end in ((0, 3), (3, 5)):
                chunk = torch.full(
                    (1, 2, end - start, 4),
                    start + 1,
                    dtype=torch.bfloat16,
                )
                consume(start, end, chunk, chunk, chunk)

        with mock.patch.object(projector, '_validate'), mock.patch.object(
            projector, 'stream', side_effect=stream
        ):
            prepared = projector.project(module, x, None)

        expected_stride = (5 * 2 * 4, 4, 2 * 4, 1)
        for tensor in (prepared.q, prepared.k, prepared.v):
            self.assertEqual(tensor.stride(), expected_stride)
            self.assertFalse(tensor.is_contiguous())
            self.assertTrue(torch.all(tensor[:, :, :3] == 1))
            self.assertTrue(torch.all(tensor[:, :, 3:] == 4))

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
            'triton_sparse_int8',
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
            text.count('_bounded_qkv_projector(qkv)'),
            5,
        )
        self.assertIn('projector=attention.projector', text)

    def test_native_sol_architecture_reuses_the_kitchen_int8_provider(self):
        inventory = FakeInventory()
        inventory.qkv_convrot_int8_256 = True
        for request in (FUSED_QKV_AUTO, FUSED_QKV_PRESERVE_BF16):
            for backend_kind in (
                'native_int8_128x64',
                'native_int8_128x64_sol_residual_64x64',
                'native_int8_64x64',
                'native_int8_64x64_sol_residual_64x64',
                'native_int8_128x128_hard_control',
                'native_int8_128x128_sol_residual_64x64',
            ):
                with self.subTest(request=request, backend_kind=backend_kind):
                    resolved = resolve_qkv_provider(
                        inventory,
                        request=request,
                        backend_kind=backend_kind,
                        kitchen_producer_available=True,
                    )
                    self.assertEqual(
                        resolved.provider_id,
                        QKV_DENSE_KITCHEN_CHUNKED,
                    )
                    self.assertTrue(resolved.fused)
                    self.assertIn('ConvRot-256 INT8', resolved.reason)

        disabled = resolve_qkv_provider(
            inventory,
            request=FUSED_QKV_OFF,
            backend_kind='native_int8_128x128_sol_residual_64x64',
        )
        self.assertEqual(disabled.provider_id, QKV_STANDARD)


if __name__ == '__main__':
    unittest.main()
