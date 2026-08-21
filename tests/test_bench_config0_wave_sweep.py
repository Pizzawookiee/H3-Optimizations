'''CPU-only contracts for the config-0 wave-alignment benchmark.'''

import importlib.util
import sys
import unittest
from pathlib import Path


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / 'benchmarks'
    / 'bench_config0_wave_sweep.py'
)
sys.path.insert(0, str(BENCHMARK.parent))
SPEC = importlib.util.spec_from_file_location('bench_config0_wave_sweep', BENCHMARK)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class WaveGeometryTests(unittest.TestCase):
    def test_4096_is_exact_on_128_sms(self):
        geometry = bench.wave_geometry(4096, 21504, 128)
        self.assertEqual(geometry['ctas'], 2688)
        self.assertEqual(geometry['waves'], 21)
        self.assertTrue(geometry['exact_waves'])

    def test_2944_is_exact_on_46_sms(self):
        geometry = bench.wave_geometry(2944, 21504, 46)
        self.assertEqual(geometry['ctas'], 1932)
        self.assertEqual(geometry['waves'], 42)
        self.assertTrue(geometry['exact_waves'])

    def test_actual_sequence_final_chunk(self):
        geometry = bench.aggregate_geometry(54006, 4096, 21504, 46)
        self.assertEqual(geometry['chunks'], 14)
        self.assertEqual(geometry['final_rows'], 758)

    def test_parse_rows_rejects_unaligned(self):
        with self.assertRaisesRegex(ValueError, 'multiples of 128'):
            bench.parse_rows('4096,4100')

    def test_compact_samples_removes_only_sample_arrays(self):
        result = bench.compact_samples({
            'samples_ms': [1.0, 2.0],
            'median_ms': 1.5,
            'nested': {'samples_ms': [3.0], 'min_ms': 3.0},
        })
        self.assertEqual(result, {
            'median_ms': 1.5,
            'nested': {'min_ms': 3.0},
        })


if __name__ == '__main__':
    unittest.main()
