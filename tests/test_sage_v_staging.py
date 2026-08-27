"""CPU contracts for two-pass Sage FP8 V carrier staging."""

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

import torch  # noqa: E402

from h3_optimizations.attention.sage_v_staging import (  # noqa: E402
    BACKEND_TORCH,
    ROW_GROUP,
    SageVStagingError,
    TwoPassSageVCarrier,
    _permuted_rows,
)


SCALE_MAX = 2.25


def _one_pass_reference(v, scale_max, pad_to):
    """The obvious single-pass build of the same carrier, for comparison."""
    batch, heads, sequence, head_dim = v.shape
    padded = (sequence + pad_to - 1) // pad_to * pad_to
    amax = v.to(torch.float32).abs().amax(dim=-2)
    quantized = v.to(torch.float32) * (scale_max / amax.unsqueeze(-2))
    carrier = torch.zeros(
        (batch, heads, head_dim, padded), dtype=torch.float8_e4m3fn
    )
    destination = _permuted_rows(torch.arange(sequence, dtype=torch.int64))
    packed = quantized.permute(0, 1, 3, 2).contiguous().to(torch.float8_e4m3fn)
    carrier.view(torch.uint8).index_copy_(3, destination, packed.view(torch.uint8))
    return carrier, amax / scale_max


def _stage(v, chunk_rows, pad_to=64):
    batch, heads, sequence, head_dim = v.shape
    staging = TwoPassSageVCarrier(
        batch,
        heads,
        sequence,
        head_dim,
        scale_max=SCALE_MAX,
        device=v.device,
        dtype=v.dtype,
        pad_to=pad_to,
        backend=BACKEND_TORCH,
    )
    for start in range(0, sequence, chunk_rows):
        end = min(start + chunk_rows, sequence)
        staging.update(v[:, :, start:end, :])
    staging.finalize_scale()
    for start in range(0, sequence, chunk_rows):
        end = min(start + chunk_rows, sequence)
        staging.quantize(v[:, :, start:end, :], start)
    return staging.finish()


class SageVStagingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.v = torch.randn(1, 3, 256, 128, dtype=torch.bfloat16)

    def test_two_pass_matches_the_single_pass_carrier_bit_for_bit(self):
        expected_carrier, expected_scale = _one_pass_reference(
            self.v, SCALE_MAX, 64
        )

        carrier, scale = _stage(self.v, chunk_rows=64)

        self.assertEqual(carrier.shape, expected_carrier.shape)
        self.assertTrue(
            torch.equal(carrier.view(torch.uint8), expected_carrier.view(torch.uint8))
        )
        self.assertTrue(torch.equal(scale, expected_scale))

    def test_result_is_independent_of_chunk_size(self):
        reference, reference_scale = _stage(self.v, chunk_rows=256)

        for chunk_rows in (16, 32, 64, 128):
            carrier, scale = _stage(self.v, chunk_rows=chunk_rows)
            self.assertTrue(
                torch.equal(carrier.view(torch.uint8), reference.view(torch.uint8)),
                chunk_rows,
            )
            self.assertTrue(torch.equal(scale, reference_scale), chunk_rows)

    def test_padded_tail_is_zero_filled(self):
        v = torch.randn(1, 2, 192, 128, dtype=torch.bfloat16)

        carrier, _ = _stage(v, chunk_rows=64, pad_to=128)

        self.assertEqual(int(carrier.shape[-1]), 256)
        tail = carrier[..., 192:].view(torch.uint8)
        self.assertTrue(torch.equal(tail, torch.zeros_like(tail)))

    def test_unaligned_sequence_keeps_real_rows_and_zero_padding(self):
        # 200 is not a multiple of the 16-row permutation group, so the final
        # group holds real rows and padding interleaved.
        v = torch.randn(1, 2, 200, 128, dtype=torch.bfloat16)
        expected_carrier, _ = _one_pass_reference(v, SCALE_MAX, 64)

        carrier, _ = _stage(v, chunk_rows=200, pad_to=64)

        self.assertTrue(
            torch.equal(carrier.view(torch.uint8), expected_carrier.view(torch.uint8))
        )

    def test_scale_must_be_finalized_before_quantizing(self):
        staging = TwoPassSageVCarrier(
            1, 2, 64, 128,
            scale_max=SCALE_MAX,
            device=self.v.device,
            dtype=self.v.dtype,
            backend=BACKEND_TORCH,
        )

        with self.assertRaisesRegex(SageVStagingError, 'finalized'):
            staging.quantize(torch.randn(1, 2, 64, 128, dtype=torch.bfloat16), 0)

    def test_amax_updates_are_rejected_after_finalize(self):
        staging = TwoPassSageVCarrier(
            1, 2, 64, 128,
            scale_max=SCALE_MAX,
            device=self.v.device,
            dtype=self.v.dtype,
            backend=BACKEND_TORCH,
        )
        staging.finalize_scale()

        with self.assertRaisesRegex(SageVStagingError, 'already finalized'):
            staging.update(torch.randn(1, 2, 64, 128, dtype=torch.bfloat16))

    def test_unaligned_row_start_is_rejected(self):
        staging = TwoPassSageVCarrier(
            1, 2, 64, 128,
            scale_max=SCALE_MAX,
            device=self.v.device,
            dtype=self.v.dtype,
            backend=BACKEND_TORCH,
        )
        staging.finalize_scale()

        with self.assertRaisesRegex(SageVStagingError, 'not %d-aligned' % ROW_GROUP):
            staging.quantize(torch.randn(1, 2, 8, 128, dtype=torch.bfloat16), 8)

    def test_incomplete_coverage_is_rejected(self):
        staging = TwoPassSageVCarrier(
            1, 2, 128, 128,
            scale_max=SCALE_MAX,
            device=self.v.device,
            dtype=self.v.dtype,
            backend=BACKEND_TORCH,
        )
        staging.finalize_scale()
        staging.quantize(torch.randn(1, 2, 64, 128, dtype=torch.bfloat16), 0)

        with self.assertRaisesRegex(SageVStagingError, 'do not cover'):
            staging.finish()

    def test_chunk_dtype_mismatch_is_rejected(self):
        staging = TwoPassSageVCarrier(
            1, 2, 64, 128,
            scale_max=SCALE_MAX,
            device=self.v.device,
            dtype=torch.bfloat16,
            backend=BACKEND_TORCH,
        )

        with self.assertRaisesRegex(SageVStagingError, 'dtype'):
            staging.update(torch.randn(1, 2, 64, 128, dtype=torch.float16))

    def test_head_dim_other_than_128_is_rejected(self):
        with self.assertRaisesRegex(SageVStagingError, 'head_dim 128'):
            TwoPassSageVCarrier(
                1, 2, 64, 64,
                scale_max=SCALE_MAX,
                device=self.v.device,
                dtype=self.v.dtype,
                backend=BACKEND_TORCH,
            )


if __name__ == '__main__':
    unittest.main()
