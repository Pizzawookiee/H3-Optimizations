'''GPU parity checks for the Kitchen-carrier Triton 64x64 fallback.'''

import os
from pathlib import Path
import sys
import unittest

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]

import torch  # noqa: E402

from h3_optimizations.attention.sparse.triton_kitchen import (  # noqa: E402
    TRITON_AVAILABLE,
    _launch,
)
from h3_optimizations.native import carrier_selftest  # noqa: E402
from h3_optimizations.native import int8_attention as native  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def dense_delta_route(sequence, heads):
    q_tiles = (sequence + 63) // 64
    kv_tiles = (sequence + 63) // 64
    lut = torch.zeros(
        (1, heads, q_tiles, kv_tiles),
        dtype=torch.int32,
        device='cuda',
    )
    if kv_tiles > 1:
        lut[..., 1:] = 1
    valid = torch.full(
        (1, heads, q_tiles),
        kv_tiles,
        dtype=torch.int32,
        device='cuda',
    )
    return lut, valid


def sparse_delta_route(sequence, heads):
    q_tiles = (sequence + 63) // 64
    kv_tiles = (sequence + 63) // 64
    selected = min(3, kv_tiles)
    lut = torch.zeros(
        (1, heads, q_tiles, kv_tiles),
        dtype=torch.int32,
        device='cuda',
    )
    # Select absolute tiles [0, 2, 3] where available. Delta encoding is
    # [0, 2, 1]. For two tiles this naturally becomes [0, 1].
    if selected >= 2:
        lut[..., 1] = 2 if kv_tiles > 2 else 1
    if selected >= 3:
        lut[..., 2] = 1
    valid = torch.full(
        (1, heads, q_tiles),
        selected,
        dtype=torch.int32,
        device='cuda',
    )
    return lut, valid


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
@unittest.skipUnless(TRITON_AVAILABLE, 'Triton is required')
class TritonKitchenParityTests(unittest.TestCase):
    def carrier(self, sequence=257, heads=2):
        if not carrier_selftest.check('cuda'):
            self.skipTest('Kitchen carrier self-test failed')
        generator = torch.Generator(device='cuda').manual_seed(20260825)

        def sample():
            return torch.randn(
                1,
                heads,
                sequence,
                128,
                dtype=torch.bfloat16,
                device='cuda',
                generator=generator,
            )

        return native.prequantize_int8_attention(
            sample(), sample(), sample(), cta_k=64
        )

    def test_full_64x64_route_is_bit_identical_to_kitchen_dense(self):
        carrier = self.carrier()
        lut, valid = dense_delta_route(carrier.q.shape[2], carrier.q.shape[1])
        actual = _launch(carrier, lut, valid)
        expected = native.int8_attention_from_prequantized(carrier)
        torch.cuda.synchronize()
        max_abs = (actual.float() - expected.float()).abs().max().item()
        self.assertTrue(
            torch.equal(actual, expected),
            '64x64 Triton != Kitchen dense at full route; max_abs=%g' % max_abs,
        )

    def test_sparse_64x64_route_matches_native_kitchen_when_native_is_healthy(self):
        # This comparison is intentionally conditional. On architectures where
        # the native 64x64 traversal is what failed the global self-test (the
        # SM120 regression that motivated this fallback), it is not a valid
        # reference. The full-route/dense test above remains valid there.
        if not native.int8_attention_is_available('cuda'):
            self.skipTest('native sparse traversal failed its architecture self-test')
        carrier = self.carrier(sequence=257)
        lut, valid = sparse_delta_route(carrier.q.shape[2], carrier.q.shape[1])
        route = native.BlockSparseRoute(
            indices=lut,
            counts=valid,
            q_tile=64,
            kv_tile=64,
            encoding='delta',
        )
        actual = _launch(carrier, lut, valid)
        expected = native.block_sparse_int8_attention_from_prequantized(
            carrier, route
        )
        torch.cuda.synchronize()
        max_abs = (actual.float() - expected.float()).abs().max().item()
        self.assertTrue(
            torch.equal(actual, expected),
            '64x64 Triton != native Kitchen sparse; max_abs=%g' % max_abs,
        )


if __name__ == '__main__':
    unittest.main()
