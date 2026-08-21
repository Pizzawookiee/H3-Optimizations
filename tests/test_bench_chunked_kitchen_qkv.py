'''CPU-only contracts for the chunked Kitchen QKV benchmark CLI.'''

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / 'benchmarks'
    / 'bench_chunked_kitchen_qkv.py'
)
SPEC = importlib.util.spec_from_file_location('bench_chunked_kitchen_qkv', BENCHMARK)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class ChunkParsingTests(unittest.TestCase):
    def test_parse_chunk_sizes(self):
        self.assertEqual(bench.parse_chunk_sizes('4096, 8192,16384'), (4096, 8192, 16384))

    def test_rejects_unaligned_chunk(self):
        with self.assertRaisesRegex(ValueError, 'multiples of 128'):
            bench.parse_chunk_sizes('4097')

    def test_rejects_duplicate_chunk(self):
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            bench.parse_chunk_sizes('4096,4096')

    def test_chunk_ranges_keep_final_partial_tile(self):
        ranges = bench.chunk_ranges(54006, 8192)
        self.assertEqual(ranges[0], (0, 8192))
        self.assertEqual(ranges[-1], (49152, 54006))
        self.assertEqual(len(ranges), 7)

    def test_h3_mlp_shape_contract(self):
        self.assertEqual(
            bench.validate_mlp_shapes((28672, 5376), (5376, 14336), 5376),
            14336,
        )

    def test_h3_mlp_shape_contract_rejects_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'SwiGLU pair'):
            bench.validate_mlp_shapes(
                (28672, 5376),
                (5376, 14080),
                5376,
            )

    def test_two_slice_visible_peak_counts_live_tile_tensors(self):
        self.assertEqual(
            bench.two_slice_visible_peak_bytes(2048, 5376, 14336),
            2048 * (14336 + 3 * 5376) * 2,
        )

    def test_kitchen_carrier_bytes_counts_only_tensor_fields(self):
        class FakeTensor:
            def __init__(self, elements, element_size):
                self.elements = elements
                self.bytes = element_size

            def numel(self):
                return self.elements

            def element_size(self):
                return self.bytes

        carrier = SimpleNamespace(
            q=FakeTensor(10, 1),
            k=FakeTensor(20, 1),
            v=FakeTensor(30, 1),
            q_scale=FakeTensor(4, 4),
            k_scale=FakeTensor(5, 4),
            v_scale=FakeTensor(6, 4),
            attn_mask=None,
            metadata='ignored',
        )
        self.assertEqual(bench.kitchen_carrier_bytes(carrier), 120)


if __name__ == '__main__':
    unittest.main()
