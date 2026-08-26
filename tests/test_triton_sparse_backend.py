'''CPU-only executor coverage for the optimized INT8 Triton sparse fallback.'''

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
from h3_optimizations.attention.sparse.triton_qkv import (  # noqa: E402
    pack_float_qkv,
)
from h3_optimizations.attention.sparse.triton_sparse_fast import (  # noqa: E402
    TritonSparseBackend,
    TritonSparseError,
    TritonSparseExecutor,
    TritonSparseSpec,
    preflight_triton_sparse,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


def dense_delta_lut(sequence, heads):
    q_tiles = (sequence + 63) // 64
    kv_tiles = (sequence + 63) // 64
    lut = torch.zeros((1, heads, q_tiles, kv_tiles), dtype=torch.int32)
    if kv_tiles > 1:
        lut[..., 1:] = 1
    valid = torch.full(
        (1, heads, q_tiles),
        kv_tiles,
        dtype=torch.int32,
    )
    return lut, valid


class TritonSparseBackendTests(unittest.TestCase):
    def test_preflight_accepts_sm89_and_rejects_missing_triton(self):
        spec = preflight_triton_sparse(
            cuda_available=lambda: True,
            capability_getter=lambda: (8, 9),
            triton_available=True,
        )
        self.assertEqual(spec.signature[0], 'triton_int8_qk_u8p_int8v')
        self.assertEqual((spec.q_tile, spec.kv_tile), (64, 64))
        self.assertEqual(spec.v_scale_group_size, 1)
        with self.assertRaisesRegex(TritonSparseError, 'requires Triton'):
            preflight_triton_sparse(
                cuda_available=lambda: True,
                capability_getter=lambda: (8, 9),
                triton_available=False,
            )

    def test_standard_prepare_owns_only_int8_qkv(self):
        sequence = 129
        heads = 2
        q = torch.randn(1, heads, sequence, 128, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        lut, valid = dense_delta_lut(sequence, heads)

        def fake_kernel(prepared):
            return torch.zeros(
                (1, prepared.heads, prepared.sequence, 128),
                dtype=prepared.output_dtype,
            )

        executor = TritonSparseExecutor(
            TritonSparseSpec(),
            allow_cpu_for_tests=True,
            kernel=fake_kernel,
        )
        prepared = executor.prepare(
            q,
            k,
            v,
            lut,
            valid,
            layer_index=4,
            metadata={},
        )

        self.assertEqual(prepared.q_int8.dtype, torch.int8)
        self.assertEqual(prepared.k_int8.dtype, torch.int8)
        self.assertEqual(prepared.v_int8.dtype, torch.int8)
        self.assertEqual(prepared.v_sum.dtype, torch.int32)
        self.assertEqual(prepared.dense_q_tiles, prepared.q_tiles)
        self.assertEqual(prepared.sparse_q_tiles, 0)
        self.assertEqual(prepared.kv_indices.numel(), 0)
        self.assertEqual(
            prepared.metadata['qkv_lifetime'],
            'independent_int8_carriers',
        )
        self.assertEqual(
            prepared.metadata['qkv_projection'],
            'standard_qkv_then_int8_pack',
        )
        source_ptrs = {
            q.untyped_storage().data_ptr(),
            k.untyped_storage().data_ptr(),
            v.untyped_storage().data_ptr(),
        }
        packed_ptrs = {
            prepared.q_int8.untyped_storage().data_ptr(),
            prepared.k_int8.untyped_storage().data_ptr(),
            prepared.v_int8.untyped_storage().data_ptr(),
        }
        self.assertTrue(source_ptrs.isdisjoint(packed_ptrs))
        output = executor.execute(prepared)
        self.assertEqual(tuple(output.shape), tuple(q.shape))
        self.assertEqual(output.dtype, q.dtype)

    def test_projected_prepare_does_not_repack(self):
        sequence = 128
        q = torch.randn(1, 1, sequence, 128, dtype=torch.float16)
        projected = pack_float_qkv(q, q, q, layer_index=3)
        lut, valid = dense_delta_lut(sequence, 1)
        executor = TritonSparseExecutor(
            TritonSparseSpec(),
            allow_cpu_for_tests=True,
            kernel=lambda prepared: torch.zeros(
                (1, prepared.heads, prepared.sequence, 128),
                dtype=prepared.output_dtype,
            ),
        )

        prepared = executor.prepare_projected(
            projected,
            lut,
            valid,
            layer_index=3,
            metadata={},
        )

        self.assertIs(prepared.q_int8, projected.q_int8)
        self.assertIs(prepared.k_int8, projected.k_int8)
        self.assertIs(prepared.v_int8, projected.v_int8)
        self.assertIs(prepared.v_sum, projected.v_sum)
        self.assertEqual(
            prepared.metadata['qkv_projection'],
            'chunked_convrot_int8',
        )

    def test_sparse_rows_are_compacted_to_absolute_indices(self):
        sequence = 256
        q = torch.zeros(1, 1, sequence, 128, dtype=torch.float16)
        projected = pack_float_qkv(q, q, q, layer_index=0)
        lut, valid = dense_delta_lut(sequence, 1)
        # Dense first Q tile. Every remaining Q tile attends context block 0
        # and selected video block 2: absolute [0, 2] -> delta [0, 2].
        lut[0, 0, 1:].zero_()
        lut[0, 0, 1:, 1] = 2
        valid[0, 0, 1:] = 2
        metadata = {
            'dense_q_tiles': 1,
            'sparse_q_tiles': 3,
            'pure_video_kv_tiles': 3,
            'retained_video_kv_tiles': 1,
        }
        executor = TritonSparseExecutor(
            TritonSparseSpec(),
            allow_cpu_for_tests=True,
            kernel=lambda prepared: prepared,
        )
        prepared = executor.prepare_projected(
            projected,
            lut,
            valid,
            layer_index=0,
            metadata=metadata,
        )
        self.assertEqual(prepared.dense_q_tiles, 1)
        self.assertEqual(prepared.sparse_q_tiles, 3)
        self.assertEqual(prepared.sparse_selected, 2)
        self.assertEqual(tuple(prepared.kv_indices.shape), (1, 1, 3, 2))
        self.assertEqual(prepared.kv_indices[0, 0, 0].tolist(), [0, 2])
        self.assertEqual(
            prepared.metadata['route_format'],
            'absolute_compact_int32',
        )

    def test_grouped_v_carrier_must_match_backend_spec(self):
        q = torch.zeros(1, 1, 64, 128, dtype=torch.float16)
        projected = pack_float_qkv(
            q, q, q, layer_index=0, v_scale_group_size=16
        )
        lut, valid = dense_delta_lut(64, 1)
        executor = TritonSparseExecutor(
            TritonSparseSpec(v_scale_group_size=1),
            allow_cpu_for_tests=True,
            kernel=lambda prepared: prepared,
        )
        with self.assertRaisesRegex(TritonSparseError, 'V scale group'):
            executor.prepare_projected(
                projected,
                lut,
                valid,
                layer_index=0,
                metadata={},
            )

    def test_wrong_lut_geometry_is_rejected(self):
        q = torch.randn(1, 1, 128, 128, dtype=torch.float16)
        executor = TritonSparseExecutor(
            TritonSparseSpec(),
            allow_cpu_for_tests=True,
            kernel=lambda prepared: prepared,
        )
        wrong_lut = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
        valid = torch.ones((1, 1, 1), dtype=torch.int32)
        with self.assertRaisesRegex(TritonSparseError, 'LUT shape'):
            executor.prepare(
                q,
                q,
                q,
                wrong_lut,
                valid,
                layer_index=0,
                metadata={},
            )

    def test_backend_name_is_separate_from_sparse_sage(self):
        self.assertEqual(TritonSparseBackend.name, 'triton_sparse_int8')


if __name__ == '__main__':
    unittest.main()
