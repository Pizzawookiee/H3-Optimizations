'''CPU contracts for the complete H3 forward experiment matrix.'''

import importlib.util
from pathlib import Path
import sys
import unittest


BENCHMARKS = Path(__file__).resolve().parents[1] / 'benchmarks'
sys.path.insert(0, str(BENCHMARKS))
SCRIPT = BENCHMARKS / 'bench_full_forward_experiments.py'
SPEC = importlib.util.spec_from_file_location(
    'bench_full_forward_experiments', SCRIPT
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class FullForwardExperimentsBenchmarkTests(unittest.TestCase):
    def test_matrix_uses_real_five_second_request(self):
        self.assertEqual(bench.benchmark.WORKLOADS, {'5s': 124})
        self.assertEqual(bench.benchmark.DEFAULT_PROMPT, '')

    def test_every_arm_uses_fixed_kitchen_and_aimdo_zero(self):
        for chain in bench.benchmark.ARMS.values():
            self.assertEqual(chain[0][0], 'H3MemoryOptimization')
            self.assertEqual(chain[1][0], 'H3SparseAttentionAdvanced')
            self.assertEqual(chain[1][1]['backend'], 'Kitchen INT8')
            self.assertEqual(chain[1][1]['video_budget'], 0.3)
            self.assertEqual(chain[1][1]['early_steps'], 0)
            self.assertEqual(chain[1][1]['late_steps'], 0)
            self.assertEqual(
                chain[-1],
                ('H3AIMDOResidencyLimiter', {'residency': '0 blocks'}),
            )

    def test_experiments_are_independent_and_have_combined_arm(self):
        expected = {
            'final_layer_chunking': 'final_layer_chunking',
            'streamed_kitchen_output': 'streamed_kitchen_output',
            'combined': 'combined',
        }
        self.assertEqual(set(bench.benchmark.ARMS), {'baseline', *expected})
        self.assertEqual(len(bench.benchmark.ARMS['baseline']), 3)
        for arm, variant in expected.items():
            experiments = [
                overrides
                for node, overrides in bench.benchmark.ARMS[arm]
                if node == 'H3FullForwardExperiment'
            ]
            self.assertEqual(
                experiments,
                [{'variant': variant, 'chunk_rows': 4096}],
            )


if __name__ == '__main__':
    unittest.main()
