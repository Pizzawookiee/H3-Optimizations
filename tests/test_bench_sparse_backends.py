'''CPU-only contracts for the three-way sparse backend benchmark.'''

import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / 'benchmarks'
    / 'bench_sparse_backends.py'
)
SPEC = importlib.util.spec_from_file_location('bench_sparse_backends', BENCHMARK)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)

try:
    import plaguekind_sla
except ImportError:
    plaguekind_sla = None


class SparseBackendBenchmarkTests(unittest.TestCase):
    def test_defaults_match_h3_and_current_sparse_budget(self):
        args = bench.parse_args(['--i-understand-this-uses-gpu'])
        self.assertEqual(args.sequence, 54006)
        self.assertEqual(args.heads, 56)
        self.assertEqual(args.video_budget, 0.3)
        self.assertEqual(args.parity_sequence, 1024)

    def test_gpu_acknowledgement_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args([])

    def test_backend_geometry_must_match(self):
        contracts = {
            'int8_triton': SimpleNamespace(q_tile=128, kv_tile=64),
            'fp8_flex': SimpleNamespace(q_tile=128, kv_tile=64),
            'sparse_sage': SimpleNamespace(q_tile=64, kv_tile=128),
        }
        with self.assertRaisesRegex(ValueError, 'geometries differ'):
            bench.validate_geometry(contracts)

    def test_delta_lut_conversion_is_absolute(self):
        lut = torch.tensor([[[[0, 2, 1], [1, 1, 2]]]], dtype=torch.int32)
        absolute = bench.absolute_lut(torch, lut)
        self.assertEqual(absolute.tolist(), [[[[0, 2, 3], [1, 2, 4]]]])

    def test_compact_fixed_topk_routes_are_expanded_for_comparison(self):
        payload = SimpleNamespace(
            valid_block_num=torch.tensor([[[3, 2]]], dtype=torch.int32),
            kv_indices=torch.tensor([[[[0, 2]]]], dtype=torch.int32),
            kv_tiles=3,
            dense_q_tiles=1,
            sparse_q_tiles=1,
            sparse_selected=2,
        )
        prepared = SimpleNamespace(sparse=payload)
        for name, value in (
            ('int8_triton', prepared),
            ('plaguekind_sla', payload),
        ):
            with self.subTest(name=name):
                valid, indices = bench.prepared_route(torch, name, value)
                self.assertEqual(valid.tolist(), [[[3, 2]]])
                self.assertEqual(indices.tolist(), [[[[0, 1, 2], [0, 2, 0]]]])

    def test_route_mismatch_stops_comparison(self):
        routes = {
            'int8_triton': {'valid_exact': True, 'indices_exact': True},
            'fp8_flex': {'valid_exact': True, 'indices_exact': False},
            'sparse_sage': {'valid_exact': True, 'indices_exact': True},
        }
        with self.assertRaisesRegex(RuntimeError, 'fp8_flex'):
            bench.require_matching_routes(routes)

    def test_prepared_tensor_bytes_counts_shared_storage_once(self):
        storage = torch.empty((16,), dtype=torch.float32)
        prepared = SimpleNamespace(
            kv_num_blocks=storage,
            kv_indices=storage.view(4, 4),
        )
        self.assertEqual(
            bench.prepared_tensor_bytes(torch, prepared),
            storage.untyped_storage().nbytes(),
        )

    def test_fp8_contract_is_derived_from_the_prepared_carrier(self):
        spec = SimpleNamespace(kernel_backend='TRITON')
        all_fp8 = bench.describe_fp8_carrier(
            SimpleNamespace(q_fp8=torch.empty(1), qk_scale=torch.empty(1)),
            spec,
        )
        mixed = bench.describe_fp8_carrier(
            SimpleNamespace(q=torch.empty(1), k_scale=torch.empty(1)),
            spec,
        )
        self.assertEqual(all_fp8['qkv'], 'FP8 Q/K/V')
        self.assertEqual(mixed['qkv'], 'floating Q, FP8 K/V')

    @unittest.skipIf(plaguekind_sla is None, 'Triton is unavailable')
    def test_plaguekind_adapter_splits_dense_and_sparse_fixed_topk_launches(self):
        from h3_optimizations.attention.sparse.config import HybridSparseConfig

        calls = []

        def kernel(q, k, v, lut, topk, q_tile, kv_tile, output):
            calls.append((tuple(q.shape), tuple(lut.shape), topk, q_tile, kv_tile))
            output.fill_(len(calls))

        config = HybridSparseConfig(video_budget=0.5)
        backend = plaguekind_sla.PlagueKindSLABackend(config, kernel=kernel)
        q = torch.zeros((1, 2, 384, 128), dtype=torch.bfloat16)
        options = bench.runtime_options(
            384,
            bench.make_layout(384, 256),
            q.dtype,
            q.device,
        )
        prepared = backend.prepare(
            q,
            q,
            q,
            layer_index=0,
            transformer_options=options,
        )
        output = backend.execute(prepared)

        self.assertEqual(tuple(output.shape), tuple(q.shape))
        self.assertEqual(prepared.dense_q_tiles, 2)
        self.assertEqual(prepared.sparse_q_tiles, 1)
        self.assertEqual(prepared.sparse_selected, 5)
        self.assertEqual(
            calls,
            [
                ((1, 256, 2, 128), (1, 2, 2, 6), 6, 128, 64),
                ((1, 128, 2, 128), (1, 2, 1, 5), 5, 128, 64),
            ],
        )
        self.assertTrue(torch.all(output[..., :256, :] == 1))
        self.assertTrue(torch.all(output[..., 256:, :] == 2))


if __name__ == '__main__':
    unittest.main()
