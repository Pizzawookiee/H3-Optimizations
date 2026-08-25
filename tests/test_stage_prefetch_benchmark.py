'''CPU contracts for the stage-local prefetch residency experiment.'''

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest


BENCHMARK = Path(__file__).resolve().parents[1] / 'benchmarks' / 'bench_stage_prefetch.py'
SPEC = importlib.util.spec_from_file_location('bench_stage_prefetch', BENCHMARK)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class StagePrefetchBenchmarkTests(unittest.TestCase):
    def test_stage_footprints_match_h3_weight_groups(self):
        self.assertEqual(bench.STAGES, ('qkv', 'attention_out', 'mlp_expand', 'mlp_reduce'))
        self.assertEqual(bench.PAGE_FOOTPRINTS, (4, 2, 5, 3))
        self.assertEqual(max(bench.expected_stage_bytes().values()), 5 * bench.base.PAGE_SIZE)

    def test_arms_are_independent(self):
        self.assertEqual(bench.ARMS, ('whole_block', 'stage_local'))

    def test_result_requires_pin_cleanup(self):
        source = BENCHMARK.read_text(encoding='utf-8')
        self.assertIn("if final['pinned_pages']", source)
        self.assertIn('cleanup_prefetched_modules', source)

    def test_gpu_acknowledgement_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args([])


if __name__ == '__main__':
    unittest.main()
