'''CPU contracts for lazy pre-attention normalization rows.'''

import unittest

import torch

from h3_optimizations.normalized_rows import (
    NormalizedRows,
    NormalizedRowsUnsupported,
    attention_output_buffer,
)


def _apply_modulation(rows, shift, scale, selector):
    rows.mul_(1.0 + scale[selector].to(rows.dtype))
    rows.add_(shift[selector].to(rows.dtype))


class NormalizedRowsTests(unittest.TestCase):
    @staticmethod
    def _case():
        x = torch.arange(80, dtype=torch.float32).reshape(10, 8) / 20
        shift = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 100
        scale = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 200
        selector = torch.tensor([3, 0, 2, 1, 3], dtype=torch.long)
        segments = ((0, 2, 1), (2, 7, selector), (7, 10, 2))
        source = NormalizedRows(
            x,
            lambda rows: rows * 0.5,
            segments,
            shift,
            scale,
            _apply_modulation,
        )

        expected = x * 0.5
        _apply_modulation(expected[0:2], shift, scale, 1)
        _apply_modulation(expected[2:7], shift, scale, selector)
        _apply_modulation(expected[7:10], shift, scale, 2)
        return x, source, expected

    def test_contiguous_slices_and_materialization_match_full_math(self):
        _x, source, expected = self._case()
        self.assertTrue(torch.equal(source[1:9], expected[1:9]))
        self.assertTrue(torch.equal(source.materialize(), expected))

    def test_unsorted_row_gather_matches_full_math(self):
        _x, source, expected = self._case()
        index = torch.tensor([8, 2, 6, 0, 4], dtype=torch.long)
        self.assertTrue(
            torch.equal(
                source.index_select(0, index),
                expected.index_select(0, index),
            )
        )

    def test_attention_output_is_distinct_and_allocated_once(self):
        x, source, expected = self._case()
        output = attention_output_buffer(source)
        self.assertIs(output, attention_output_buffer(source))
        self.assertIsNot(output, x)
        output.copy_(expected + 5)
        self.assertTrue(torch.equal(output, expected + 5))
        self.assertTrue(torch.equal(source.materialize(), expected))
        self.assertIs(attention_output_buffer(x), x)

    def test_unsupported_tensor_operations_request_materialization(self):
        _x, source, _expected = self._case()
        with self.assertRaises(NormalizedRowsUnsupported):
            torch.mean(source)
        with self.assertRaises(NormalizedRowsUnsupported):
            source.reshape(5, 16)
        with self.assertRaises(NormalizedRowsUnsupported):
            source.index_select(1, torch.tensor([0], dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
