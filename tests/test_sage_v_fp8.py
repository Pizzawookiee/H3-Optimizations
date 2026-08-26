"""CPU contracts for Sage FP8 V preparation fallbacks."""

import os
from pathlib import Path
import sys
import unittest

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from h3_optimizations.attention.sage_v_fp8 import prepare_sage_v_fp8  # noqa: E402


class StockQuantizer:
    def __init__(self):
        self.calls = []

    def __call__(self, source, **kwargs):
        self.calls.append((tuple(source.shape), kwargs))
        carrier = source.new_empty(
            source.shape[0], source.shape[1], source.shape[3], source.shape[2]
        )
        scale = source.new_empty(
            source.shape[0], source.shape[1], source.shape[3], dtype=torch.float32
        )
        return carrier, scale, None


class SageVFP8Tests(unittest.TestCase):
    def test_stock_pad64_keeps_source_without_an_extra_copy(self):
        source = torch.empty((1, 2, 65, 128), dtype=torch.bfloat16)
        stock = StockQuantizer()

        prepare_sage_v_fp8(source, stock, scale_max=2.25, pad_to=64)

        self.assertEqual(stock.calls[0][0], (1, 2, 65, 128))

    def test_stock_pad128_matches_sm90_carrier_requirement(self):
        source = torch.empty((1, 2, 65, 128), dtype=torch.bfloat16)
        stock = StockQuantizer()

        prepare_sage_v_fp8(source, stock, scale_max=448.0, pad_to=128)

        shape, kwargs = stock.calls[0]
        self.assertEqual(shape, (1, 2, 128, 128))
        self.assertEqual(kwargs["tensor_layout"], "HND")
        self.assertEqual(kwargs["scale_max"], 448.0)
        self.assertFalse(kwargs["smooth_v"])


if __name__ == "__main__":
    unittest.main()
