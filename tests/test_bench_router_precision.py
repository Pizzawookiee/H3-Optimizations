'''CPU contracts for the H3 router precision benchmark.'''

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

import torch


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / 'benchmarks'
    / 'bench_router_precision.py'
)
SPEC = importlib.util.spec_from_file_location('bench_router_precision', BENCHMARK)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class RouterPrecisionBenchmarkTests(unittest.TestCase):
    def test_defaults_match_h3_sparse_benchmark(self):
        args = bench.parse_args(['--i-understand-this-uses-gpu'])
        self.assertEqual(args.sequence, 54006)
        self.assertEqual(args.heads, 56)
        self.assertEqual(args.video_start, 256)
        self.assertEqual(args.video_budget, 0.3)

    def test_gpu_acknowledgement_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args([])

    def test_geometry_matches_128q_64kv_router(self):
        result = bench.geometry(54006, 256, 0.3)
        self.assertEqual(result['q_tiles'], 422)
        self.assertEqual(result['kv_tiles'], 844)
        self.assertEqual(result['pure_q_start'], 2)
        self.assertEqual(result['pure_kv_start'], 4)
        self.assertEqual(result['retained'], 252)

    def test_fp32_pool_preserves_values_bf16_store_rounds(self):
        values = torch.tensor(
            [[[[1.0], [1.0], [1.0], [1.0078125]]]],
            dtype=torch.bfloat16,
        )
        bf16 = bench.mean_pool(torch, values, 4)
        fp32 = bench.mean_pool(torch, values, 4, dtype=torch.float32)
        self.assertEqual(bf16.dtype, torch.bfloat16)
        self.assertEqual(fp32.dtype, torch.float32)
        self.assertNotEqual(float(bf16.item()), float(fp32.item()))

    def test_route_comparison_counts_set_substitutions(self):
        left = torch.tensor([[[[0, 2], [1, 2]]]])
        right = torch.tensor([[[[0, 1], [1, 2]]]])
        result = bench.compare_routes(torch, left, right, 3)
        self.assertEqual(result['changed_rows'], 1)
        self.assertEqual(result['total_rows'], 2)
        self.assertEqual(result['substituted_blocks'], 1)
        self.assertEqual(result['selected_blocks'], 4)
        self.assertEqual(result['max_substitutions_per_row'], 1)
        self.assertAlmostEqual(result['mean_jaccard'], 2 / 3)

    def test_all_precision_arms_return_the_same_route_shape(self):
        generator = torch.Generator().manual_seed(7)
        q = torch.randn(
            (1, 1, 256, 128),
            generator=generator,
            dtype=torch.bfloat16,
        )
        k = torch.randn(
            (1, 1, 256, 128),
            generator=generator,
            dtype=torch.bfloat16,
        )
        route_geometry = bench.geometry(256, 64, 0.5)
        for name in bench.ARM_ORDER:
            with self.subTest(name=name):
                selected, margin = bench.route_arm(
                    torch,
                    q,
                    k,
                    route_geometry,
                    name,
                    cutoff_margin=True,
                )
                self.assertEqual(tuple(selected.shape), (1, 1, 1, 2))
                self.assertEqual(tuple(margin.shape), (1, 1, 1))

    def test_theoretical_memory_includes_summaries(self):
        route_geometry = {
            'q_tiles': 897,
            'kv_tiles': 1794,
            'pure_q_tiles': 897,
            'pure_kv_tiles': 1794,
        }
        result = bench.theoretical_bytes(route_geometry, 56)
        self.assertEqual(result['bf16_score'], 90116208 * 2)
        self.assertEqual(
            result['fp32_increment_over_current'],
            (90116208 + 56 * (897 + 1794) * 128) * 2,
        )


if __name__ == '__main__':
    unittest.main()
