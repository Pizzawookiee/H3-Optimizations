'''CPU contracts for chunked Sparse Sage QKV carrier assembly.'''

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

import h3_optimizations.attention.sparse.chunked_qkv as chunked_qkv  # noqa: E402

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


class ChunkedSparseQKVTests(unittest.TestCase):
    def test_projector_defaults_to_4096_rows(self):
        spec = sparse_spec()
        projector = chunked_qkv.ChunkedSparseQKVProjector(spec)
        result = object()

        with mock.patch.object(
            chunked_qkv,
            'run_chunked_sparse_qkv',
            return_value=result,
        ) as run:
            self.assertIs(
                projector.project(
                    object(),
                    object(),
                    object(),
                    layer_index=7,
                    transformer_options={},
                ),
                result,
            )

        self.assertEqual(projector.name, 'chunked_sparse_sage_qkv')
        self.assertEqual(projector.chunk_rows, 4096)
        self.assertEqual(projector.installation_signature[-2], 4096)
        self.assertEqual(run.call_args.kwargs['chunk_rows'], 4096)
        self.assertIs(run.call_args.kwargs['spec'], spec)

    def test_assembles_aligned_carriers_and_final_partial_tiles(self):
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

        with mock.patch.object(
            chunked_qkv,
            'project_chunk_hnd',
            side_effect=project,
        ):
            prepared = chunked_qkv._assemble_chunked_sparse_qkv(
                module,
                x,
                rope,
                layer_index=3,
                spec=sparse_spec(),
                chunk_rows=128,
                packer=cpu_packer,
            )

        self.assertEqual(
            [(start, end) for start, end, _rope in calls],
            [(0, 128), (128, 256), (256, 300)],
        )
        self.assertEqual(calls[-1][2], tuple(range(256, 300)))
        self.assertEqual(tuple(prepared.q_int8.shape), (1, 2, sequence, 128))
        self.assertEqual(tuple(prepared.q_scale.shape), (1, 2, 3))
        self.assertEqual(tuple(prepared.k_scale.shape), (1, 2, 5))
        self.assertEqual(float(prepared.q_summary[0, 0, -1, 0]), 277.5)
        self.assertEqual(float(prepared.k_summary[0, 0, -1, 0]), 377.5)
        self.assertEqual(float(prepared.v[0, 0, -1, 0]), 499.0)
        self.assertEqual(prepared.layer_index, 3)
        self.assertFalse(prepared.smooth_k)

    def test_rejects_chunk_rows_that_split_q_tiles(self):
        with self.assertRaisesRegex(
            chunked_qkv.FusedQKVError,
            'multiple of 128',
        ):
            chunked_qkv._validate_chunk_rows(192, 128, 64)


if __name__ == '__main__':
    unittest.main()
