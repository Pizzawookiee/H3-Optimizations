'''CPU contracts for the Phase 1 producer attribution harness.'''

from pathlib import Path
import contextlib
import io
import sys
import unittest

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK / 'benchmarks'), str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import bench_qkv_attribution as bench  # noqa: E402
import h3_stage_recorder as recorder  # noqa: E402


def forward(**stages):
    return {
        name: {'gpu_ms': value, 'calls': 1, 'max_call_ms': value}
        for name, value in stages.items()
    }


class ScheduleTests(unittest.TestCase):
    def test_the_schedule_alternates_in_balanced_pairs(self):
        self.assertEqual(
            bench.balanced_schedule(12),
            list('ABBAABBAABBA'),
        )

    def test_both_arms_get_the_same_number_of_forwards(self):
        for count in (4, 8, 24, 40):
            schedule = bench.balanced_schedule(count)
            self.assertEqual(schedule.count('A'), schedule.count('B'))

    def test_neither_arm_runs_first_in_every_pair(self):
        schedule = bench.balanced_schedule(24)
        firsts = [schedule[index] for index in range(0, len(schedule), 2)]
        self.assertIn('A', firsts)
        self.assertIn('B', firsts)

    def test_odd_and_non_multiple_of_four_counts_are_rejected(self):
        for count in ('1', '2', '6', '0', '-4'):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bench.parse_args([
                        '--checkpoint', 'x',
                        '--forwards', count,
                        '--i-understand-this-uses-gpu',
                    ])

    def test_the_gpu_acknowledgement_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args(['--checkpoint', 'x'])

    def test_chunk_rows_must_be_tile_aligned(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args([
                    '--checkpoint', 'x',
                    '--qkv-chunk-rows', '4000',
                    '--i-understand-this-uses-gpu',
                ])


class DerivationTests(unittest.TestCase):
    '''The producer figure has to come out of the outer region, not a sum.'''

    def test_producer_is_attention_total_minus_the_regions_after_it(self):
        result = bench.derived(forward(
            attention_total=300.0,
            sparse_attention_kernel=100.0,
            attention_out=25.0,
            sparse_route=10.0,
            sparse_carrier_prepare=2.0,
        ))
        self.assertEqual(result['producer_with_route'], 175.0)
        self.assertEqual(result['producer_without_route'], 163.0)
        self.assertEqual(result['route'], 12.0)

    def test_unattributed_work_stays_inside_the_producer_figure(self):
        '''A child that nobody instrumented must not shrink the total.'''
        instrumented = bench.derived(forward(
            attention_total=300.0,
            sparse_attention_kernel=100.0,
            attention_out=25.0,
            sparse_route=10.0,
            sparse_carrier_prepare=2.0,
            qkv_linear=61.0,
        ))
        missing = bench.derived(forward(
            attention_total=300.0,
            sparse_attention_kernel=100.0,
            attention_out=25.0,
            sparse_route=10.0,
            sparse_carrier_prepare=2.0,
        ))
        self.assertEqual(
            instrumented['producer_without_route'],
            missing['producer_without_route'],
        )

    def test_a_missing_region_is_zero_rather_than_an_error(self):
        result = bench.derived(forward(attention_total=10.0))
        self.assertEqual(result['producer_with_route'], 10.0)
        self.assertEqual(result['sparse_attention_kernel'], 0.0)

    def test_the_producer_figure_never_goes_negative(self):
        result = bench.derived(forward(
            attention_total=10.0, sparse_attention_kernel=40.0
        ))
        self.assertEqual(result['producer_with_route'], 0.0)


class PairedDeltaTests(unittest.TestCase):
    def test_the_delta_pairs_forwards_in_order(self):
        rows_a = [{'v': 100.0}, {'v': 110.0}, {'v': 120.0}]
        rows_b = [{'v': 105.0}, {'v': 115.0}, {'v': 125.0}]
        delta = bench.paired_delta(rows_a, rows_b, 'v')
        self.assertEqual(delta['median_ms'], 5.0)
        self.assertEqual(delta['pairs'], 3)

    def test_a_shared_drift_cancels_in_the_pairing(self):
        '''Both arms slowing together must not read as a candidate cost.'''
        drift = [0.0, 5.0, 12.0, 30.0]
        rows_a = [{'v': 100.0 + value} for value in drift]
        rows_b = [{'v': 100.0 + value} for value in drift]
        self.assertEqual(bench.paired_delta(rows_a, rows_b, 'v')['median_ms'], 0.0)

    def test_unequal_arm_lengths_use_the_shorter_one(self):
        delta = bench.paired_delta(
            [{'v': 1.0}, {'v': 2.0}], [{'v': 3.0}], 'v'
        )
        self.assertEqual(delta['pairs'], 1)


class RecorderTests(unittest.TestCase):
    def test_regions_are_inert_until_a_forward_is_open(self):
        instance = recorder.StageRecorder(torch=None, device=None)
        with instance.stage('anything'):
            pass
        self.assertEqual(instance.forwards, [])

    def test_summarize_reports_median_and_spread(self):
        summary = recorder.summarize([
            forward(stage=10.0), forward(stage=20.0), forward(stage=30.0)
        ])
        self.assertEqual(summary['stage']['median_gpu_ms'], 20.0)
        self.assertEqual(summary['stage']['min_gpu_ms'], 10.0)
        self.assertEqual(summary['stage']['max_gpu_ms'], 30.0)
        self.assertEqual(summary['stage']['samples'], 3)

    def test_summarize_skips_stages_no_forward_recorded(self):
        self.assertEqual(recorder.summarize([forward(a=1.0)], names=('b',)), {})


class HarnessSourceTests(unittest.TestCase):
    def test_the_harness_drives_the_production_forward_and_backends(self):
        text = (
            PACK / 'benchmarks' / 'bench_qkv_attribution.py'
        ).read_text(encoding='utf-8')
        for required in (
            'from h3_optimizations.attention_forward import make_forward',
            'SparseKitchenBackend',
            'ChunkedKitchenQKVProjector',
            'from h3_optimizations.memory.forward import make_forward',
        ):
            self.assertIn(required, text)

    def test_the_harness_aborts_rather_than_measuring_another_route(self):
        text = (
            PACK / 'benchmarks' / 'bench_qkv_attribution.py'
        ).read_text(encoding='utf-8')
        for required in (
            'arm A must run with chunked QKV disabled',
            'arm B must run with chunked QKV enabled',
            'is not on native Kitchen sparse',
            'is not on the 128-wide KV tile',
            'requires ConvRot-256 TensorWise INT8 QKV',
            'the shipped Kitchen QKV producer is unavailable',
        ):
            self.assertIn(required, text)

    def test_the_harness_synchronizes_once_per_forward(self):
        text = (
            PACK / 'benchmarks' / 'bench_qkv_attribution.py'
        ).read_text(encoding='utf-8')
        measured = text.split('recorder.enabled = True', 1)[1]
        measured = measured.split('recorder.enabled = False', 1)[0]
        self.assertEqual(measured.count('torch.cuda.synchronize'), 1)


if __name__ == '__main__':
    unittest.main()
