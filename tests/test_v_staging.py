"""CPU contracts for two-pass V carrier staging."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.native.v_staging import (  # noqa: E402
    BACKEND_TORCH,
    TwoPassVCarrier,
    VStagingError,
)


class VStagingTests(unittest.TestCase):
    @staticmethod
    def _spec(sequence=23):
        return SimpleNamespace(
            k_input_shape=(1, 2, sequence, 8),
            kernel_head_dim=8,
            cta_k=16,
            device=torch.device('cpu'),
            input_dtype=torch.bfloat16,
        )

    def test_chunks_produce_global_scale_and_cover_sequence(self):
        source = torch.linspace(-3, 4, 1 * 2 * 23 * 8).reshape(1, 2, 23, 8)
        source = source.to(torch.bfloat16)
        staging = TwoPassVCarrier(self._spec(), backend=BACKEND_TORCH)
        for start, end in ((0, 7), (7, 16), (16, 23)):
            staging.update(source[..., start:end, :])
        scale = staging.finalize_scale()
        expected = torch.clamp(
            source.float().abs().amax(dim=-2) * (1.0 / 127.0), min=1e-12
        ).reshape(-1)
        self.assertTrue(torch.equal(scale, expected))
        for start, end in ((0, 7), (7, 16), (16, 23)):
            staging.quantize(source[..., start:end, :], start)
        carrier, actual_scale = staging.finish()
        self.assertIs(actual_scale, scale)
        self.assertEqual(tuple(carrier.shape), (16, 32))

    def test_finish_rejects_incomplete_coverage(self):
        source = torch.ones(1, 2, 23, 8, dtype=torch.bfloat16)
        staging = TwoPassVCarrier(self._spec(), backend=BACKEND_TORCH)
        staging.update(source)
        staging.finalize_scale()
        staging.quantize(source[..., :8, :], 0)
        with self.assertRaisesRegex(VStagingError, 'do not cover'):
            staging.finish()


if __name__ == '__main__':
    unittest.main()
