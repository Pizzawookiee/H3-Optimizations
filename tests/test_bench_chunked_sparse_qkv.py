'''CPU-only contracts for the chunked Sparse Sage benchmark CLI.'''

import importlib.util
from pathlib import Path
import unittest


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / 'benchmarks'
    / 'bench_chunked_sparse_qkv.py'
)
SPEC = importlib.util.spec_from_file_location(
    'bench_chunked_sparse_qkv',
    BENCHMARK,
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class ChunkedSparseBenchmarkTests(unittest.TestCase):
    def test_layout_keeps_target_video_last(self):
        layout = bench.make_layout(54006, 256)
        self.assertEqual(layout.seq_len, 54006)
        self.assertEqual(layout.video_range, (256, 54006))
        self.assertEqual(layout.segments[-1], (256, 54006, 'video'))

    def test_rejects_unaligned_chunk_rows(self):
        with self.assertRaises(SystemExit):
            bench.parse_args(
                [
                    '--checkpoint',
                    'model.safetensors',
                    '--kitchen-source',
                    'kitchen',
                    '--chunk-rows',
                    '4097',
                    '--i-understand-this-uses-gpu',
                ]
            )


if __name__ == '__main__':
    unittest.main()
