'''CPU contracts for the real H3 block benchmark harness.'''

import importlib.util
import contextlib
import io
import unittest
from pathlib import Path

import torch


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / 'benchmarks'
    / 'bench_h3_block.py'
)
SPEC = importlib.util.spec_from_file_location('bench_h3_block', BENCHMARK)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class H3BlockBenchmarkTests(unittest.TestCase):
    def test_tensor_error_reports_exact_identity(self):
        value = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)

        result = bench.tensor_error(torch, value, value.clone())

        self.assertTrue(result['exact'])
        self.assertEqual(result['max_abs'], 0.0)
        self.assertEqual(result['rmse'], 0.0)
        self.assertEqual(result['relative_rmse'], 0.0)

    def test_tensor_error_reports_relative_difference(self):
        reference = torch.tensor([1.0, -1.0])
        actual = torch.tensor([2.0, -1.0])

        result = bench.tensor_error(torch, reference, actual)

        self.assertFalse(result['exact'])
        self.assertEqual(result['max_abs'], 1.0)
        self.assertGreater(result['relative_rmse'], 0.0)

    def test_gpu_acknowledgement_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args(
                    [
                        '--checkpoint',
                        'model.safetensors',
                        '--kitchen-source',
                        'kitchen',
                    ]
                )


if __name__ == '__main__':
    unittest.main()
