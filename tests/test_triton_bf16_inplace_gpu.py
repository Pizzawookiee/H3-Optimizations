"""GPU parity gates for in-place streamed BF16 Triton Q -> O reuse."""

from pathlib import Path
import sys
import unittest

import torch

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from h3_optimizations.attention.sparse import triton_bf16  # noqa: E402


requires_triton_gpu = unittest.skipUnless(
    torch.cuda.is_available() and triton_bf16.TRITON_AVAILABLE,
    "requires CUDA and Triton",
)


@requires_triton_gpu
class TritonBF16InplaceGPUParityTests(unittest.TestCase):
    HEADS = 2
    SEQUENCE = 192

    def _qkv(self, seed=1234):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        shape = (1, self.HEADS, self.SEQUENCE, triton_bf16.HEAD_DIM)
        return tuple(
            torch.randn(
                shape,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            for _ in range(3)
        )

    def _reference(self, q, k, v, sparse_lut, dense, sparse, selected):
        prepared = triton_bf16.PreparedTritonBF16(
            q=q.clone(),
            k=k,
            v=v,
            sparse_lut=sparse_lut,
            dense_q_tiles=dense,
            sparse_q_tiles=sparse,
            sparse_selected=selected,
            layer_index=0,
            metadata={},
        )
        return triton_bf16._launch(prepared)

    def test_dense_partial_slab_matches_separate_output_bitwise(self):
        q, k, v = self._qkv()
        q_tiles = self.SEQUENCE // triton_bf16.Q_TILE
        sparse_lut = torch.empty(
            (1, self.HEADS, 0, 0),
            dtype=torch.int32,
            device="cuda",
        )
        reference = self._reference(
            q,
            k,
            v,
            sparse_lut,
            dense=q_tiles,
            sparse=0,
            selected=0,
        )

        start = triton_bf16.Q_TILE
        stop = start + triton_bf16.Q_TILE
        q_slab = q[..., start:stop, :].contiguous()
        pointer = q_slab.data_ptr()
        actual = triton_bf16._launch_streamed_chunk(
            q_slab,
            k,
            v,
            sparse_lut,
            dense_q_tiles=q_tiles,
            sparse_q_tiles=0,
            sparse_selected=0,
            sequence=self.SEQUENCE,
            q_row_start=start,
        )

        self.assertEqual(actual.data_ptr(), pointer)
        self.assertTrue(torch.equal(actual, reference[..., start:stop, :]))

    def test_dense_to_sparse_boundary_matches_separate_output_bitwise(self):
        q, k, v = self._qkv(seed=4321)
        dense_q_tiles = 1
        sparse_q_tiles = 2
        selected = 2
        sparse_lut = torch.tensor(
            [[[[0, 1], [1, 2]], [[0, 2], [0, 1]]]],
            dtype=torch.int32,
            device="cuda",
        )
        reference = self._reference(
            q,
            k,
            v,
            sparse_lut,
            dense=dense_q_tiles,
            sparse=sparse_q_tiles,
            selected=selected,
        )

        stop = triton_bf16.Q_TILE * 2
        q_slab = q[..., :stop, :].contiguous()
        pointer = q_slab.data_ptr()
        actual = triton_bf16._launch_streamed_chunk(
            q_slab,
            k,
            v,
            sparse_lut,
            dense_q_tiles=dense_q_tiles,
            sparse_q_tiles=sparse_q_tiles,
            sparse_selected=selected,
            sequence=self.SEQUENCE,
            q_row_start=0,
            sparse_lut_q_start=0,
        )

        self.assertEqual(actual.data_ptr(), pointer)
        self.assertTrue(torch.equal(actual, reference[..., :stop, :]))


if __name__ == "__main__":
    unittest.main()
