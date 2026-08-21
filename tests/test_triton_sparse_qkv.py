'''CPU reference coverage for optimized INT8 Triton sparse QKV carriers.'''

import unittest

import torch

from h3_optimizations.attention.sparse.triton_qkv import (
    PreparedTritonSparseQKV,
    TritonSparseQKVError,
    pack_float_qkv,
    validate_prepared_triton_sparse_qkv,
)


class TritonSparseQKVTests(unittest.TestCase):
    def test_float_qkv_packs_independent_int8_carriers(self):
        sequence = 129
        heads = 2
        q = torch.randn(1, heads, sequence, 128, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        prepared = pack_float_qkv(q, k, v, layer_index=7)
        validate_prepared_triton_sparse_qkv(prepared)

        self.assertEqual(prepared.q_int8.dtype, torch.int8)
        self.assertEqual(prepared.k_int8.dtype, torch.int8)
        self.assertEqual(prepared.v_int8.dtype, torch.int8)
        self.assertEqual(tuple(prepared.q_scale.shape), (1, heads, 2))
        self.assertEqual(tuple(prepared.k_scale.shape), (1, heads, 3))
        self.assertEqual(tuple(prepared.v_scale.shape), (1, heads, 3, 128))
        self.assertEqual(tuple(prepared.v_sum.shape), (1, heads, 3, 128))
        self.assertEqual(prepared.v_sum.dtype, torch.int32)
        self.assertEqual(tuple(prepared.q_summary.shape), (1, heads, 2, 128))
        self.assertEqual(tuple(prepared.k_summary.shape), (1, heads, 3, 128))
        self.assertEqual(prepared.output_dtype, q.dtype)
        self.assertEqual(prepared.layer_index, 7)

        source_ptrs = {
            q.untyped_storage().data_ptr(),
            k.untyped_storage().data_ptr(),
            v.untyped_storage().data_ptr(),
        }
        carrier_ptrs = {
            prepared.q_int8.untyped_storage().data_ptr(),
            prepared.k_int8.untyped_storage().data_ptr(),
            prepared.v_int8.untyped_storage().data_ptr(),
        }
        self.assertTrue(source_ptrs.isdisjoint(carrier_ptrs))

    def test_q_and_k_use_scalar_block_scales(self):
        q = torch.zeros(1, 1, 128, 128, dtype=torch.float16)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        q[..., 0, 0] = 127.0
        k[..., 0, 1] = 63.5

        prepared = pack_float_qkv(q, k, v, layer_index=0)

        self.assertAlmostEqual(float(prepared.q_scale[0, 0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(prepared.k_scale[0, 0, 0]), 0.5, places=4)
        self.assertEqual(int(prepared.q_int8[0, 0, 0, 0]), 127)
        self.assertEqual(int(prepared.k_int8[0, 0, 0, 1]), 127)

    def test_v_scale_is_per_kv_tile_per_channel_by_default(self):
        v = torch.zeros(1, 1, 65, 128, dtype=torch.float16)
        q = torch.zeros_like(v)
        k = torch.zeros_like(v)
        v[0, 0, 0, 0] = 127.0
        v[0, 0, 64, 0] = 12.7
        v[0, 0, 0, 1] = 63.5

        prepared = pack_float_qkv(q, k, v, layer_index=0)

        self.assertEqual(tuple(prepared.v_scale.shape), (1, 1, 2, 128))
        self.assertAlmostEqual(float(prepared.v_scale[0, 0, 0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(prepared.v_scale[0, 0, 1, 0]), 0.1, places=4)
        self.assertAlmostEqual(float(prepared.v_scale[0, 0, 0, 1]), 0.5, places=4)
        self.assertEqual(int(prepared.v_int8[0, 0, 0, 0]), 127)
        self.assertEqual(int(prepared.v_int8[0, 0, 64, 0]), 127)
        self.assertEqual(
            int(prepared.v_sum[0, 0, 0, 0]),
            int(prepared.v_int8[0, 0, :64, 0].to(torch.int32).sum()),
        )

    def test_grouped_v_scale_reduces_scale_carrier(self):
        q = torch.zeros(1, 1, 64, 128, dtype=torch.float16)
        v = torch.zeros_like(q)
        v[..., 0, 0] = 127.0
        v[..., 0, 15] = 63.5
        prepared = pack_float_qkv(
            q,
            q,
            v,
            layer_index=0,
            v_scale_group_size=16,
        )
        self.assertEqual(prepared.v_scale_group_size, 16)
        self.assertEqual(tuple(prepared.v_scale.shape), (1, 1, 1, 8))
        self.assertAlmostEqual(float(prepared.v_scale[0, 0, 0, 0]), 1.0, places=4)
        self.assertEqual(tuple(prepared.v_sum.shape), (1, 1, 1, 128))

    def test_partial_tail_has_valid_summaries(self):
        q = torch.ones(1, 1, 129, 128, dtype=torch.bfloat16)
        prepared = pack_float_qkv(q, q, q, layer_index=0)

        self.assertTrue(torch.all(prepared.q_summary[0, 0, 0] == 1))
        self.assertTrue(torch.all(prepared.q_summary[0, 0, 1] == 1))
        self.assertTrue(torch.all(prepared.k_summary[0, 0, 2] == 1))

    def test_validation_rejects_wrong_v_scale_geometry(self):
        sequence = 64
        heads = 1
        carrier = torch.zeros((1, heads, sequence, 128), dtype=torch.int8)
        prepared = PreparedTritonSparseQKV(
            q_int8=carrier.clone(),
            q_scale=torch.ones((1, heads, 1), dtype=torch.float32),
            k_int8=carrier.clone(),
            k_scale=torch.ones((1, heads, 1), dtype=torch.float32),
            v_int8=carrier.clone(),
            v_scale=torch.ones((1, heads, 1, 1), dtype=torch.float32),
            v_sum=torch.zeros((1, heads, 1, 128), dtype=torch.int32),
            q_summary=torch.zeros((1, heads, 1, 128), dtype=torch.float16),
            k_summary=torch.zeros((1, heads, 1, 128), dtype=torch.float16),
            output_dtype=torch.float16,
            sequence=sequence,
            heads=heads,
            head_dim=128,
            layer_index=0,
        )
        with self.assertRaisesRegex(TritonSparseQKVError, 'v_scale shape'):
            validate_prepared_triton_sparse_qkv(prepared)


if __name__ == '__main__':
    unittest.main()
