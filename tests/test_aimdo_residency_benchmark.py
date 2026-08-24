'''CPU-only contracts for the AIMDO residency behavior benchmark.'''

import importlib.util
from pathlib import Path
import sys
import unittest


TORCH_WAS_LOADED = 'torch' in sys.modules
AIMDO_WAS_LOADED = 'comfy_aimdo.control' in sys.modules
BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / 'benchmarks'
    / 'bench_aimdo_residency.py'
)
SPEC = importlib.util.spec_from_file_location('bench_aimdo_residency', BENCHMARK)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class AIMDOResidencyBenchmarkTests(unittest.TestCase):
    def test_import_does_not_load_torch_or_aimdo(self):
        self.assertEqual('torch' in sys.modules, TORCH_WAS_LOADED)
        self.assertEqual('comfy_aimdo.control' in sys.modules, AIMDO_WAS_LOADED)

    def test_levels_normalize_node_labels_and_remove_duplicates(self):
        self.assertEqual(
            benchmark.parse_levels('stock, 0 blocks, 1 block, 2, 4, 2 blocks'),
            ('stock', '0', '1', '2', '4'),
        )

    def test_block_equivalent_caps_match_limiter_arithmetic(self):
        self.assertEqual(
            {level: benchmark.cap_pages(level) for level in benchmark.LEVELS},
            {'stock': 10, '0': 0, '1': 4, '2': 7, '4': 10},
        )

    def test_routes_account_for_allocations_that_cross_the_watermark(self):
        self.assertEqual(
            {level: benchmark.expected_routes(level) for level in benchmark.LEVELS},
            {
                'stock': ('vbar', 'vbar', 'vbar', 'vbar'),
                '0': ('stream', 'stream', 'stream', 'stream'),
                '1': ('vbar', 'vbar', 'stream', 'stream'),
                '2': ('vbar', 'vbar', 'vbar', 'stream'),
                '4': ('vbar', 'vbar', 'vbar', 'vbar'),
            },
        )

    def test_gpu_confirmation_is_mandatory(self):
        with self.assertRaises(SystemExit):
            benchmark.parse_args([])

    def test_result_validation_accepts_fill_hit_stream_and_pin_lifecycle(self):
        events = []
        observations = []
        routes = benchmark.expected_routes('1')
        for pass_index in range(2):
            for module, (pages, route) in enumerate(
                zip(benchmark.PAGE_FOOTPRINTS, routes)
            ):
                outcome = (
                    'streamed' if route == 'stream' else
                    'resident_fill' if pass_index == 0 else
                    'resident_hit'
                )
                events.append(
                    {
                        'event': 'fault',
                        'pass': pass_index,
                        'module': module,
                        'outcome': outcome,
                    }
                )
                observations.append(
                    {
                        'pass': pass_index,
                        'module': module,
                        'max_abs_error': 0.0,
                        'after_cast': {
                            'pinned_pages': pages if route == 'vbar' else 0,
                        },
                        'after_release': {'pinned_pages': 0},
                    }
                )
        self.assertEqual(
            benchmark._validate_result('1', 4, events, observations),
            [],
        )

    def test_source_uses_comfy_cast_and_release_contract(self):
        source = BENCHMARK.read_text(encoding='utf-8')
        self.assertIn('ops.cast_bias_weight(', source)
        self.assertIn('ops.uncast_bias_weight(', source)
        self.assertIn('model_vbar.vbar_fault = fault', source)
        self.assertIn('model_management.STREAM_AIMDO_CAST_BUFFERS', source)
        self.assertIn('aimdo_control.get_total_vram_usage()', source)


if __name__ == '__main__':
    unittest.main()
