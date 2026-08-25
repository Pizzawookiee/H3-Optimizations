'''CPU/static contracts for the benchmark-only native Q quantizer.'''

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest


PACK = Path(__file__).resolve().parents[1]
BENCHMARK = PACK / 'benchmarks' / 'bench_q_only_quantizer.py'
SPEC = importlib.util.spec_from_file_location('bench_q_only_quantizer', BENCHMARK)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class QOnlyQuantizerBenchmarkTests(unittest.TestCase):
    def test_default_cases_cover_both_rotations_and_ragged_q(self):
        cases = bench.parse_cases('129:256,333:4096')
        self.assertTrue(any(k_rows <= 256 for _q_rows, k_rows in cases))
        self.assertTrue(any(k_rows > 256 for _q_rows, k_rows in cases))
        self.assertTrue(all(q_rows % 128 for q_rows, _k_rows in cases))

    def test_gpu_acknowledgement_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args([])

    def test_sidecar_does_not_replace_the_shipped_library(self):
        build = (PACK / 'native' / 'build_q_only.ps1').read_text(encoding='utf-8')
        self.assertIn('h3_q_only.dll', build)
        self.assertNotIn('-o", (Join-Path $here "bin\\h3_int8_attention.dll', build)

    def test_launcher_selects_rotation_from_full_k(self):
        source = (PACK / 'native' / 'src' / 'sage_attention' / 'quant_qk_int8.cu').read_text(encoding='utf-8')
        body = source.split('void launch_quant_q_per_thread_int8', 1)[1].split('void launch_select_k_anchor', 1)[0]
        self.assertIn('full_Lk <= 256', body)
        self.assertIn('CHANNEL_TILES, 4, true', body)
        self.assertIn('CHANNEL_TILES, 128, true', body)


if __name__ == '__main__':
    unittest.main()
