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
from h3_optimizations.qkv.bf16 import (  # noqa: E402
    BF16QKVBindingError,
    CHUNK_ROWS,
    ChunkedBF16QKVProjector,
    PreparedBF16QKV,
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
    def test_default_chunk_rows_is_4096(self):
        projector = ChunkedBF16QKVProjector()
        self.assertEqual(CHUNK_ROWS, 4096)
        self.assertEqual(projector.chunk_rows, 4096)
        self.assertEqual(
            projector.installation_signature,
            ('chunked_bf16_qkv', 4096),
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

    def test_source_keeps_streaming_and_full_materialization_separate(self):
        text = (PACK / 'h3_optimizations' / 'qkv' / 'bf16.py').read_text(
            encoding='utf-8'
        )
        self.assertIn('def stream(', text)
        self.assertIn('def project(', text)
        self.assertIn('consume_chunk(start, end, q, k, v)', text)
        self.assertIn('for start in range(0, sequence, self.chunk_rows)', text)
        self.assertNotIn('torch.cat(', text)

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
            'comfy_kitchen_int8',
            'sparse_kitchen_int8',
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
        self.assertGreaterEqual(
            text.count('ChunkedBF16QKVProjector(chunk_rows=4096)'),
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
