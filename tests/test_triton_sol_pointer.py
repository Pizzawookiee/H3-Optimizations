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
                'route': 'full_absolute_compile_time',
                'v_sum': 'precomputed_once_per_carrier_kv_tile',
                'autotune': ((4, 1), (8, 1), (4, 2), (8, 2)),
                'production_backend': False,
            },
        )

    def test_kernel_source_has_no_route_or_inner_v_reduction(self):
        source = (PACK / 'h3_optimizations' / 'attention' / 'sparse'
                  / 'triton_sol_pointer.py').read_text(encoding='utf-8')
        kernel = source.split('def _sol_kitchen_full_kernel(', 1)[1]
        kernel = kernel.split('\n\ndef prepare_carrier(', 1)[0]
        self.assertIn('tl.program_id(1)', kernel)
        self.assertIn('tl.program_id(2)', kernel)
        self.assertIn('tl.range(0, KV_BLOCKS)', kernel)
        self.assertIn('V_SUM + (bh * KV_BLOCKS + key_block)', kernel)
        self.assertNotIn('tl.load(LUT', kernel)
        self.assertNotIn('tl.load(VALID', kernel)
        self.assertNotIn('tl.sum(v.to(tl.int32)', kernel)

    def test_experiment_is_not_wired_into_production_selector(self):
        public_surface = (PACK / 'h3_optimizations' / 'attention' / 'sparse'
                          / 'triton_sparse.py').read_text(encoding='utf-8')
        self.assertNotIn('triton_sol_pointer', public_surface)

    def test_benchmark_defaults_to_the_small_numerical_gate(self):
        bench = load_benchmark()
        args = bench.parse_args(['--i-understand-this-uses-gpu'])
        self.assertFalse(args.benchmark)
        self.assertFalse(args.benchmark_inexact)
        self.assertEqual(args.parity_sequence, 257)
        self.assertEqual(args.heads, 2)

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
