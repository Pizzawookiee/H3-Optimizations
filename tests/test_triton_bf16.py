'''CPU contracts for the production BF16 Triton sparse fallback.'''

import os
from pathlib import Path
import sys
import unittest
from unittest import mock


os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = next(parent for parent in PACK.parents if (parent / 'comfy').is_dir())
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
from h3_optimizations.attention.sparse import triton_bf16  # noqa: E402
from h3_optimizations.attention.sparse import triton_sparse  # noqa: E402
from h3_optimizations.plan import (  # noqa: E402
    SPARSE_BACKEND_TRITON,
    SPARSE_BACKEND_TRITON_LEGACY,
    SparseRequest,
)
from h3_optimizations.qkv.bf16 import ChunkedBF16QKVProjector  # noqa: E402
from h3_optimizations.qkv.projectors import TritonSparseQKVProjector  # noqa: E402
import h3_optimizations.qkv.projectors as projector_module  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class TritonBF16Tests(unittest.TestCase):
    def test_preflight_contract(self):
        spec = triton_sparse.preflight_triton_sparse(
            cuda_available=lambda: True,
            capability_getter=lambda: (8, 9),
            triton_available=True,
        )
        self.assertEqual(
            spec.signature,
            ('triton_bf16_qk_bf16pv_fp32', 64, 64, 128),
        )
        with self.assertRaisesRegex(triton_sparse.TritonSparseError, 'Triton'):
            triton_sparse.preflight_triton_sparse(
                cuda_available=lambda: True,
                capability_getter=lambda: (8, 9),
                triton_available=False,
            )

    def test_public_selector_constructs_bf16_backend(self):
        backend = triton_sparse.TritonSparseBackend()
        self.assertIsInstance(backend, triton_bf16.TritonBF16Backend)
        self.assertEqual(triton_sparse.TritonSparseBackend.name, 'triton_sparse_bf16')
        self.assertEqual(backend.as_status()['qkv_carrier'], 'bf16_hnd')

    def test_legacy_int8_label_resolves_to_bf16(self):
        request = SparseRequest(backend=SPARSE_BACKEND_TRITON_LEGACY)
        self.assertEqual(request.backend, SPARSE_BACKEND_TRITON)
        self.assertEqual(request.backend, 'BF16 Triton')

    def test_projector_produces_bf16_hnd_not_kitchen_carrier(self):
        projector = TritonSparseQKVProjector(chunk_rows=4096)
        self.assertEqual(projector.qk_format, 'bf16_hnd')
        self.assertEqual(projector.v_format, 'bf16_hnd')
        self.assertTrue(projector.streamed_qkv)
        self.assertIsInstance(projector._implementation, ChunkedBF16QKVProjector)

    def test_projector_exposes_the_bounded_chunk_stream_contract(self):
        projector = TritonSparseQKVProjector(chunk_rows=4096)
        projector._implementation.stream = mock.Mock()
        module = mock.Mock()
        consume = mock.Mock()
        fmt = mock.Mock(
            convrot_int8_256=False,
            w4a8=False,
            fp8=False,
            plain_float=True,
        )
        with mock.patch.object(projector_module, 'describe_linear', return_value=fmt):
            self.assertIsNone(projector.stream(module, 'x', 'rope', consume))
        projector._implementation.stream.assert_called_once_with(
            module, 'x', 'rope', consume
        )

    def test_full_bf16_carrier_is_filled_by_the_stream_contract(self):
        source = (
            PACK / 'h3_optimizations' / 'qkv' / 'bf16.py'
        ).read_text(encoding='utf-8')
        project = source.split('def project(self, module, x, rope_freqs):', 1)[1]
        project = project.split('\n    def try_project(', 1)[0]
        self.assertIn('self.stream(module, x, rope_freqs, consume)', project)
        self.assertNotIn('torch.cat(', project)

    def test_sparse_rows_are_compacted_to_absolute_indices(self):
        lut = torch.zeros((1, 1, 4, 4), dtype=torch.int32)
        lut[..., 0, 1:] = 1
        lut[..., 1:, 1] = 2
        valid = torch.tensor([[[4, 2, 2, 2]]], dtype=torch.int32)
        route, dense, sparse, selected = triton_bf16._compact_route(
            lut,
            valid,
            {
                'dense_q_tiles': 1,
                'sparse_q_tiles': 3,
                'pure_video_kv_tiles': 3,
                'retained_video_kv_tiles': 1,
            },
        )
        self.assertEqual((dense, sparse, selected), (1, 3, 2))
        self.assertEqual(route[0, 0].tolist(), [[0, 2], [0, 2], [0, 2]])

    def test_v_load_precedes_softmax_reduction(self):
        source = (
            PACK / 'h3_optimizations' / 'attention' / 'sparse' / 'triton_bf16.py'
        ).read_text(encoding='utf-8')
        kernel = source.split('def _bf16_sparse_kernel(', 1)[1]
        kernel = kernel.split('\n\ndef _launch(', 1)[0]
        self.assertLess(kernel.index('v = tl.load('), kernel.index('tile_max ='))
        self.assertIn('tl.dot(probability.to(v.dtype), v)', kernel)
        self.assertNotIn('int8', kernel.lower())

    def test_kernel_supports_strided_hnd_without_full_contiguous_copies(self):
        source = (
            PACK / 'h3_optimizations' / 'attention' / 'sparse' / 'triton_bf16.py'
        ).read_text(encoding='utf-8')
        self.assertIn('stride_qh: tl.constexpr', source)
        self.assertIn('stride_kn: tl.constexpr', source)
        self.assertIn('stride_vn: tl.constexpr', source)
        self.assertNotIn('prepared.q.contiguous()', source)
        self.assertNotIn('prepared.k.contiguous()', source)
        self.assertNotIn('prepared.v.contiguous()', source)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
