'''Readable QKV status contracts.'''

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.plan import STATUS_KEY  # noqa: E402
from h3_optimizations.status import (  # noqa: E402
    format_qkv_execution,
    format_sparse_status,
)


class QKVStatusReadabilityTests(unittest.TestCase):
    def setUp(self):
        self.status = {
            'attention': {'selected': 'sparse_kitchen_int8'},
            'sparse': {'video_budget': 0.05},
            'fused_qkv': {
                'provider': 'streamed_bf16_kitchen_qkv',
                'projector': 'chunked_kitchen_qkv',
                'chunk_rows': 4096,
                'output_streamed': True,
            },
            'weight_formats': {'qkv': ['Parameter:torch.bfloat16'] * 50},
            'mlp': {'provider': 'off'},
        }

    def test_streamed_bf16_kitchen_describes_the_execution(self):
        self.assertEqual(
            format_qkv_execution(self.status),
            (
                'BF16 weights -> 4096-row BF16 chunks -> Kitchen INT8 carrier; '
                'output streamed'
            ),
        )

    def test_sparse_preview_hides_the_internal_route_name(self):
        model = SimpleNamespace(
            model_options={
                'transformer_options': {STATUS_KEY: self.status},
            }
        )

        text = format_sparse_status(model)

        self.assertIn(
            'QKV: BF16 weights -> 4096-row BF16 chunks -> Kitchen INT8 carrier',
            text,
        )
        self.assertNotIn('streamed_bf16_kitchen_qkv', text)

    def test_standard_path_keeps_the_fallback_reason(self):
        status = {
            'fused_qkv': {
                'provider': 'standard_h3_qkv',
                'reason': 'Kitchen producer unavailable',
            },
            'weight_formats': {'qkv': ['Parameter:torch.bfloat16']},
        }

        self.assertEqual(
            format_qkv_execution(status),
            'BF16 weights -> standard QKV path (Kitchen producer unavailable)',
        )

    def test_every_public_route_has_a_readable_description(self):
        expected = {
            'chunked_bf16_qkv': 'full BF16 Q/K/V',
            'force_bf16_qkv': 'forced BF16 projection',
            'force_quant_qkv': 'forced FP8 projection',
            'convrot_int8_dense_sage': 'dense Sage carrier',
            'chunked_kitchen_qkv': 'Kitchen INT8 carrier',
            'streamed_bf16_kitchen_qkv': 'Kitchen INT8 carrier',
            'chunked_fp8_kitchen_qkv': 'FP8 projection',
            'convrot_int8_sparse_sage': 'Sparse Sage carrier',
            'chunked_fp8_sparse_sage': 'Sparse Sage carrier',
            'chunked_triton_bf16_sparse': 'Triton BF16 carrier',
        }
        for provider, phrase in expected.items():
            with self.subTest(provider=provider):
                status = {
                    'fused_qkv': {
                        'provider': provider,
                        'chunk_rows': 4096,
                    },
                    'weight_formats': {
                        'qkv': ['Parameter:torch.bfloat16'],
                    },
                }

                text = format_qkv_execution(status)

                self.assertIn(phrase, text)
                self.assertNotIn(provider, text)


if __name__ == '__main__':
    unittest.main()
