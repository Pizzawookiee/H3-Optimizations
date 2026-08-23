'''CPU contracts for the reusable chunked BF16 H3 QKV projector.'''

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.plan import (  # noqa: E402
    ATTENTION_EXISTING,
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
    QKV_DENSE_KITCHEN_CHUNKED,
    QKV_SPARSE_CONVROT_INT8,
    QKV_TRITON_SPARSE_CHUNKED,
    QKV_STANDARD,
    resolve_qkv_provider,
)


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

    def test_preserve_bf16_routes_to_existing_attention_projector_slot(self):
        resolved = resolve_qkv_provider(
            FakeInventory(),
            request=FUSED_QKV_PRESERVE_BF16,
            backend_kind='existing',
            memory_optimize=True,
        )
        self.assertEqual(resolved.provider_id, QKV_DENSE_KITCHEN_CHUNKED)
        self.assertIn('existing attention backend', resolved.reason)

    def test_preserve_bf16_routes_to_all_production_sparse_slots(self):
        cases = (
            ('sparse_kitchen_int8', QKV_DENSE_KITCHEN_CHUNKED),
            ('sparse_sage', QKV_SPARSE_CONVROT_INT8),
            ('triton_sparse_int8', QKV_TRITON_SPARSE_CHUNKED),
        )
        for backend_kind, expected in cases:
            with self.subTest(backend_kind=backend_kind):
                resolved = resolve_qkv_provider(
                    FakeInventory(),
                    request=FUSED_QKV_PRESERVE_BF16,
                    backend_kind=backend_kind,
                    triton_available=True,
                    memory_optimize=True,
                )
                self.assertEqual(resolved.provider_id, expected)

    def test_preserve_bf16_does_not_convert_fp16(self):
        resolved = resolve_qkv_provider(
            FakeInventory('torch.float16'),
            request=FUSED_QKV_PRESERVE_BF16,
            backend_kind='existing',
            memory_optimize=True,
        )
        self.assertEqual(resolved.provider_id, QKV_STANDARD)

    def test_attention_forward_consumes_bf16_without_reprojection(self):
        text = (
            PACK / 'h3_optimizations' / 'attention_forward.py'
        ).read_text(encoding='utf-8')
        self.assertIn('isinstance(projected, PreparedBF16QKV)', text)
        self.assertIn('backend.prepare(', text)
        self.assertIn("DENSE_KITCHEN_PREQUANTIZED", text)
        self.assertIn('_legacy_attention(', text)

    def test_projector_slots_switch_to_bf16_before_backend_specific_packing(self):
        kitchen = (PACK / 'h3_optimizations' / 'kitchen_qkv.py').read_text(
            encoding='utf-8'
        )
        sparse = (
            PACK / 'h3_optimizations' / 'qkv' / 'projectors.py'
        ).read_text(encoding='utf-8')
        self.assertIn('native_bf16 and not self.fp8_projection', kitchen)
        self.assertIn('ChunkedBF16QKVProjector(self.chunk_rows)', kitchen)
        self.assertGreaterEqual(
            sparse.count('ChunkedBF16QKVProjector(self.chunk_rows)'),
            2,
        )


if __name__ == '__main__':
    unittest.main()
