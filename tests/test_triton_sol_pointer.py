'''CPU/static contracts for the experimental Sol-shaped Triton kernel.'''

from contextlib import redirect_stderr
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = next(parent for parent in PACK.parents if (parent / 'comfy').is_dir())
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.attention.sparse import triton_sol_pointer as sol  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def load_benchmark():
    path = PACK / 'benchmarks' / 'bench_triton_sol_pointer.py'
    spec = importlib.util.spec_from_file_location('bench_triton_sol_pointer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TritonSolPointerTests(unittest.TestCase):
    def test_contract_is_one_full_width_64q_program(self):
        self.assertEqual(
            sol.kernel_contract(),
            {
                'q_tile': 64,
                'kv_tile': 64,
                'head_dim': 128,
                'program': 'one_64q_tile_x_one_head_x_full_d128',
                'route': 'dense_implicit_plus_sparse_absolute',
                'v_sum': 'precomputed_once_per_carrier_kv_tile',
                'autotune': ((4, 1), (8, 1), (4, 2), (8, 2)),
                'production_backend': False,
            },
        )

    def test_kernel_source_has_compact_route_and_no_inner_v_reduction(self):
        source = (PACK / 'h3_optimizations' / 'attention' / 'sparse'
                  / 'triton_sol_pointer.py').read_text(encoding='utf-8')
        kernel = source.split('def _sol_kitchen_kernel(', 1)[1]
        kernel = kernel.split('\n\ndef prepare_carrier(', 1)[0]
        self.assertIn('tl.program_id(1)', kernel)
        self.assertIn('tl.program_id(2)', kernel)
        self.assertIn('tl.range(0, N_SELECTED)', kernel)
        self.assertIn('V_SUM + (bh * KV_BLOCKS + key_block)', kernel)
        self.assertIn('tl.load(LUT + route_offset)', kernel)
        self.assertNotIn('tl.load(VALID', kernel)
        self.assertNotIn('tl.sum(v.to(tl.int32)', kernel)

    def test_production_selector_uses_bf16_on_every_supported_architecture(self):
        from h3_optimizations.attention.sparse import triton_sparse
        from h3_optimizations.attention.sparse.triton_bf16 import (
            TritonBF16Backend,
        )

        self.assertIsInstance(
            triton_sparse.TritonSparseBackend(), TritonBF16Backend
        )

    def test_sparse_delta_rows_are_compacted_to_absolute_indices(self):
        import torch

        lut = torch.zeros((1, 1, 4, 4), dtype=torch.int32)
        lut[..., 0, 1:] = 1
        lut[..., 1:, 1] = 2
        valid = torch.tensor([[[4, 2, 2, 2]]], dtype=torch.int32)
        route, dense, sparse, selected = sol._compact_route(
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
        self.assertEqual(tuple(route.shape), (1, 1, 3, 2))
        self.assertEqual(route[0, 0].tolist(), [[0, 2], [0, 2], [0, 2]])

    def test_benchmark_defaults_to_the_small_numerical_gate(self):
        bench = load_benchmark()
        args = bench.parse_args(['--i-understand-this-uses-gpu'])
        self.assertFalse(args.benchmark)
        self.assertFalse(args.benchmark_inexact)
        self.assertEqual(args.parity_sequence, 257)
        self.assertEqual(args.heads, 2)

    def test_benchmark_covers_every_plaguekind_geometry(self):
        bench = load_benchmark()
        self.assertEqual(
            bench.PLAGUEKIND_GEOMETRIES,
            {
                'plaguekind_64x64': (64, 64),
                'plaguekind_64x128': (64, 128),
                'plaguekind_128x64': (128, 64),
                'plaguekind_128x128': (128, 128),
            },
        )
        self.assertEqual(
            tuple(bench.PLAGUEKIND_GEOMETRIES), bench.ARM_ORDER[-4:]
        )

    def test_numerical_gate_has_kitchen_and_unquantized_references(self):
        source = (PACK / 'benchmarks' / 'bench_triton_sol_pointer.py').read_text(
            encoding='utf-8'
        )
        self.assertIn("'kitchen_64x64_full': outputs['kitchen_64x64_full']", source)
        self.assertIn("'dense_fp32_softmax_bf16_output':", source)

    def test_benchmark_mixed_route_has_dense_and_sparse_rows(self):
        import torch

        bench = load_benchmark()
        lut, valid, metadata = bench.mixed_delta_route(
            torch, 257, 2, torch.device('cpu')
        )
        self.assertEqual(tuple(lut.shape), (1, 2, 5, 5))
        self.assertEqual(valid[0, 0].tolist(), [5, 3, 3, 3, 3])
        self.assertEqual(lut[0, 0, 1, :3].tolist(), [0, 1, 3])
        self.assertEqual(metadata['dense_q_tiles'], 1)
        self.assertEqual(metadata['sparse_q_tiles'], 4)

    def test_inexact_timing_requires_benchmark_mode(self):
        bench = load_benchmark()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args([
                    '--i-understand-this-uses-gpu',
                    '--benchmark-inexact',
                ])

    def test_benchmark_requires_explicit_gpu_acknowledgement(self):
        bench = load_benchmark()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args([])


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
