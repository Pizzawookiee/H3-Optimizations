'''CPU contracts for low-VRAM streamed Sparse Sage execution.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

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

from h3_optimizations.attention.sparse.router import SparseTileRouter  # noqa: E402
import h3_optimizations.attention.sparse.sparse_sage_streamed as streamed  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def sparse_spec():
    return SimpleNamespace(
        signature=('test-sparse-spec',),
        q_tile=128,
        kv_tile=64,
        qk_format='block_int8',
        q_scale_layout='per_q_tile_float32',
        k_scale_layout='per_kv_tile_float32',
        projected_v_format='floating_hnd',
        summary_format='tile_mean',
        v_format='fp16',
        accumulator='f16',
        kernel=lambda *_args: None,
    )


def cpu_packer(x, output, scales, summaries, *, row_start, block_size):
    for block_start in range(0, x.shape[-2], block_size):
        block_end = min(block_start + block_size, x.shape[-2])
        value = x[..., block_start:block_end, :].float()
        scale = value.abs().amax(dim=(-2, -1)) / 127.0 + 1e-7
        quantized = value / scale[..., None, None]
        quantized += torch.where(quantized >= 0, 0.5, -0.5)
        output[
            ...,
            row_start + block_start:row_start + block_end,
            :,
        ].copy_(quantized.to(torch.int8))
        destination_block = (row_start + block_start) // block_size
        scales[..., destination_block].copy_(scale)
        summaries[..., destination_block, :].copy_(
            value.mean(dim=-2).to(summaries.dtype)
        )


class StreamedSparseSageTests(unittest.TestCase):
    def test_streamed_lut_chunks_reconstruct_existing_route_exactly(self):
        router = SparseTileRouter(q_tile=2, kv_tile=2)
        layout = SimpleNamespace(
            seq_len=12,
            video_range=(4, 12),
            segments=((0, 4, 'text'), (4, 12, 'video')),
            video_shape=(1, 2, 4),
            audio_t=0,
        )
        generator = torch.Generator().manual_seed(1234)
        q_summary = torch.randn((1, 2, 6, 4), generator=generator)
        k_summary = torch.randn((1, 2, 6, 4), generator=generator)

        expected_lut, expected_valid, expected_meta = (
            router.build_lut_from_summaries(
                q_summary,
                k_summary,
                layout,
                0.5,
            )
        )
        route_plan, actual_meta = streamed._prepare_streamed_route_plan(
            router,
            q_summary,
            k_summary,
            layout,
            0.5,
        )
        pieces = list(
            streamed._iter_streamed_lut_chunks(
                router,
                route_plan,
                q_chunk_tiles=2,
                device=q_summary.device,
            )
        )
        actual_lut = torch.cat([piece[2] for piece in pieces], dim=2)
        actual_valid = torch.cat([piece[3] for piece in pieces], dim=2)

        self.assertTrue(torch.equal(actual_lut, expected_lut))
        self.assertTrue(torch.equal(actual_valid, expected_valid))
        self.assertEqual(actual_meta.as_dict(), expected_meta.as_dict())
        self.assertLess(route_plan.indices.numel(), expected_lut.numel())

    def test_projector_keeps_no_full_q_carrier(self):
        sequence = 300
        module = SimpleNamespace(heads=2, head_dim=128)
        x = torch.zeros((sequence, 256), dtype=torch.float32)
        rope = torch.arange(sequence).reshape(1, sequence, 1, 1, 1, 1)
        calls = []

        def project(_module, _x, rope_freqs, start, end):
            calls.append((start, end, tuple(rope_freqs[:, start:end].reshape(-1))))
            rows = torch.arange(start, end, dtype=torch.float32).view(1, 1, -1, 1)
            q = rows.expand(1, 2, -1, 128).clone()
            return q, q + 100, q + 200

        projected = streamed._assemble_streamed_sparse_qkv(
            module,
            x,
            rope,
            layer_index=3,
            spec=sparse_spec(),
            project_chunk_rows=128,
            query_chunk_rows=256,
            packer=cpu_packer,
            project_chunk=project,
        )

        self.assertEqual(
            [(start, end) for start, end, _rope in calls],
            [(0, 128), (128, 256), (256, 300)],
        )
        self.assertEqual(calls[-1][2], tuple(range(256, 300)))
        self.assertFalse(hasattr(projected, 'q_int8'))
        self.assertEqual(tuple(projected.k_int8.shape), (1, 2, sequence, 128))
        self.assertEqual(tuple(projected.k_scale.shape), (1, 2, 5))
        self.assertEqual(tuple(projected.q_summary.shape), (1, 2, 3, 128))
        self.assertEqual(float(projected.q_summary[0, 0, -1, 0]), 277.5)
        self.assertEqual(float(projected.k_summary[0, 0, -1, 0]), 377.5)
        self.assertEqual(float(projected.v[0, 0, -1, 0]), 499.0)
        self.assertEqual(projected.query_chunk_rows, 256)

    def test_projector_defaults_to_4096_query_rows(self):
        projector = streamed.StreamedSparseSageQKVProjector(sparse_spec())
        self.assertEqual(projector.chunk_rows, 4096)
        self.assertEqual(projector.query_chunk_rows, 4096)
        self.assertTrue(projector.streamed_q)


if __name__ == '__main__':
    unittest.main()
