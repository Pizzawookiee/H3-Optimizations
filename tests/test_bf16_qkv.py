'''CPU contracts for the reusable chunked BF16 H3 QKV projector.'''

from pathlib import Path
import sys
import unittest

import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.qkv.bf16 import (  # noqa: E402
    BF16QKVBindingError,
    CHUNK_ROWS,
    ChunkedBF16QKVProjector,
    PreparedBF16QKV,
)


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


if __name__ == '__main__':
    unittest.main()
