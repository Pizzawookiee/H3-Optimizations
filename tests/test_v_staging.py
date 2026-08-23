'''Contracts for two-phase V carrier production.

The whole proposition is that the carrier comes out byte for byte identical to
the whole-V quantizer, so these are equality tests, not tolerance tests. The
CPU tests pin the seam itself -- that chunking changes nothing -- and the GPU
test pins the Torch reference against the real CUDA kernel.
'''

from pathlib import Path
import sys
import unittest

import torch

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from h3_optimizations.native import v_staging  # noqa: E402
from h3_optimizations.native.v_staging import (  # noqa: E402
    BACKEND_TORCH,
    TwoPassVCarrier,
    VStagingError,
    finalize_v_scale,
    new_v_amax,
)

CUDA = torch.cuda.is_available()


class Spec:
    '''The fields TwoPassVCarrier reads out of a producer spec.'''

    def __init__(self, batch, heads, sequence, head_dim, cta_k, device, dtype):
        self.k_input_shape = (batch, heads, sequence, head_dim)
        self.kernel_head_dim = head_dim
        self.cta_k = cta_k
        self.device = device
        self.input_dtype = dtype


def spec_for(sequence, *, batch=1, heads=2, head_dim=16, cta_k=128,
             device='cpu', dtype=torch.float32):
    return Spec(batch, heads, sequence, head_dim, cta_k, device, dtype)


def random_v(spec, seed=0):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn(
        spec.k_input_shape, generator=generator, dtype=torch.float32
    ).to(dtype=spec.input_dtype, device=spec.device)


def produce(spec, v, chunk_rows, backend=BACKEND_TORCH):
    carrier = TwoPassVCarrier(spec, backend=backend)
    sequence = spec.k_input_shape[2]
    for start in range(0, sequence, chunk_rows):
        carrier.update(v[:, :, start:min(start + chunk_rows, sequence), :])
    carrier.finalize_scale()
    for start in range(0, sequence, chunk_rows):
        stop = min(start + chunk_rows, sequence)
        carrier.quantize(v[:, :, start:stop, :], start)
    return carrier.finish()


class ScaleTests(unittest.TestCase):
    def test_the_scale_matches_kitchens_formula(self):
        amax = torch.tensor([[[254.0, 0.0]]], dtype=torch.float32)
        scale = finalize_v_scale(amax)
        self.assertAlmostEqual(float(scale[0]), 2.0, places=6)
        # The 1e-12 floor keeps an all-zero channel from dividing by zero.
        self.assertAlmostEqual(float(scale[1]), 1e-12, places=18)

    def test_a_chunked_maximum_equals_the_whole_sequence_maximum(self):
        '''Exactly equal, because max is associative and exact on floats.'''
        v = random_v(spec_for(512), seed=3)
        whole = v.to(torch.float32).abs().amax(dim=-2)
        for chunk_rows in (1, 7, 128, 511, 512):
            amax = new_v_amax(1, 2, 16, device='cpu')
            for start in range(0, 512, chunk_rows):
                piece = v[:, :, start:min(start + chunk_rows, 512), :]
                amax.copy_(
                    torch.maximum(
                        amax, piece.to(torch.float32).abs().amax(dim=-2)
                    )
                )
            self.assertTrue(
                torch.equal(amax, whole), 'chunk %d' % chunk_rows
            )

    def test_the_accumulator_must_be_float32(self):
        with self.assertRaises(VStagingError):
            finalize_v_scale(torch.zeros(1, 2, 16, dtype=torch.float16))


class SeamTests(unittest.TestCase):
    '''Chunking must not move a single byte.'''

    def test_every_chunk_size_produces_the_same_carrier(self):
        spec = spec_for(1024)
        v = random_v(spec, seed=11)
        reference, reference_scale = produce(spec, v, 1024)
        for chunk_rows in (16, 128, 256, 512, 768):
            packed, scale = produce(spec, v, chunk_rows)
            self.assertTrue(
                torch.equal(packed, reference),
                'V bytes differ at chunk %d' % chunk_rows,
            )
            self.assertTrue(
                torch.equal(scale, reference_scale),
                'V scales differ at chunk %d' % chunk_rows,
            )

    def test_a_ragged_final_chunk_is_handled(self):
        spec = spec_for(1000)
        v = random_v(spec, seed=5)
        reference, _ = produce(spec, v, 1000)
        packed, _ = produce(spec, v, 256)
        self.assertTrue(torch.equal(packed, reference))

    def test_padding_beyond_the_sequence_is_zero(self):
        spec = spec_for(1000, cta_k=128)
        packed, _ = produce(spec, random_v(spec, seed=7), 256)
        self.assertEqual(packed.shape[-1], 1024)
        # Destinations for rows >= sequence, under the same permutation.
        source = torch.arange(1000, 1024, dtype=torch.int64)
        inverse = v_staging._inverse_permutation_16(torch.device('cpu'))
        destination = (source & ~15) | inverse[source & 15]
        self.assertTrue(
            torch.equal(
                packed.index_select(1, destination),
                torch.zeros(
                    packed.shape[0], destination.numel(), dtype=torch.int8
                ),
            )
        )

    def test_real_rows_and_padding_never_share_a_destination(self):
        inverse = v_staging._inverse_permutation_16(torch.device('cpu'))
        source = torch.arange(0, 1024, dtype=torch.int64)
        destination = (source & ~15) | inverse[source & 15]
        self.assertEqual(len(set(destination.tolist())), 1024)


class OrderingTests(unittest.TestCase):
    def test_quantizing_before_the_scale_is_final_is_refused(self):
        spec = spec_for(256)
        carrier = TwoPassVCarrier(spec, backend=BACKEND_TORCH)
        with self.assertRaises(VStagingError):
            carrier.quantize(random_v(spec), 0)

    def test_updating_after_the_scale_is_final_is_refused(self):
        spec = spec_for(256)
        carrier = TwoPassVCarrier(spec, backend=BACKEND_TORCH)
        carrier.update(random_v(spec))
        carrier.finalize_scale()
        with self.assertRaises(VStagingError):
            carrier.update(random_v(spec))

    def test_a_gap_in_coverage_is_refused(self):
        spec = spec_for(512)
        v = random_v(spec)
        carrier = TwoPassVCarrier(spec, backend=BACKEND_TORCH)
        carrier.update(v)
        carrier.finalize_scale()
        carrier.quantize(v[:, :, :128, :], 0)
        carrier.quantize(v[:, :, 256:384, :], 256)
        with self.assertRaises(VStagingError):
            carrier.finish()

    def test_partial_coverage_is_refused(self):
        spec = spec_for(512)
        v = random_v(spec)
        carrier = TwoPassVCarrier(spec, backend=BACKEND_TORCH)
        carrier.update(v)
        carrier.finalize_scale()
        carrier.quantize(v[:, :, :256, :], 0)
        with self.assertRaises(VStagingError):
            carrier.finish()

    def test_a_chunk_outside_the_sequence_is_refused(self):
        spec = spec_for(512)
        v = random_v(spec)
        carrier = TwoPassVCarrier(spec, backend=BACKEND_TORCH)
        carrier.update(v)
        carrier.finalize_scale()
        with self.assertRaises(VStagingError):
            carrier.quantize(v[:, :, :128, :], 448)

    def test_a_mismatched_chunk_shape_is_refused(self):
        spec = spec_for(512)
        carrier = TwoPassVCarrier(spec, backend=BACKEND_TORCH)
        with self.assertRaises(VStagingError):
            carrier.update(torch.zeros(1, 3, 128, 16))
        with self.assertRaises(VStagingError):
            carrier.update(torch.zeros(1, 2, 128, 32))
        with self.assertRaises(VStagingError):
            carrier.update(torch.zeros(1, 2, 128, 16, dtype=torch.float16))

    def test_an_unknown_backend_is_refused_rather_than_guessed(self):
        with self.assertRaises(VStagingError):
            TwoPassVCarrier(spec_for(256), backend='fastest')

    def test_requesting_native_without_the_kernels_aborts_loudly(self):
        '''A benchmark must never silently measure the Torch reference.'''
        if v_staging.available_backend() == v_staging.BACKEND_NATIVE:
            self.skipTest('the native staging kernels are present')
        with self.assertRaises(VStagingError):
            TwoPassVCarrier(spec_for(256), backend=v_staging.BACKEND_NATIVE)


class RoundingTests(unittest.TestCase):
    def test_rounding_is_half_to_even_and_saturating(self):
        '''`cvt.rni.sat.s8.f32`: nearest-even, clamped to [-128, 127].'''
        spec = spec_for(16, heads=1, head_dim=16, cta_k=16)
        v = torch.zeros(1, 1, 16, 16, dtype=torch.float32)
        # Channel 0 peaks at 127 so its scale is exactly 1.0.
        v[0, 0, 0, 0] = 127.0
        v[0, 0, 1, 0] = 0.5   # -> 0, not 1
        v[0, 0, 2, 0] = 1.5   # -> 2
        v[0, 0, 3, 0] = 2.5   # -> 2, not 3
        v[0, 0, 4, 0] = -0.5  # -> 0
        packed, scale = produce(spec, v, 16)
        self.assertAlmostEqual(float(scale[0]), 1.0, places=6)
        inverse = v_staging._inverse_permutation_16(torch.device('cpu'))
        source = torch.arange(0, 5, dtype=torch.int64)
        destination = (source & ~15) | inverse[source & 15]
        values = packed[0].index_select(0, destination).tolist()
        self.assertEqual(values, [127, 0, 2, 2, 0])


@unittest.skipUnless(CUDA, 'requires CUDA')
class NativeParityTests(unittest.TestCase):
    '''The reference and the shipped whole-V quantizer must agree exactly.'''

    def _whole_v(self, spec, v):
        from h3_optimizations.native import producer as native_producer

        holder = type('Holder', (), {})()
        holder.spec = spec
        holder._finalized = False
        holder.v = None
        holder.v_scale = None
        native_producer.quantize_int8_attention_v(holder, v)
        return holder.v, holder.v_scale

    def _spec(self, sequence, heads=4, head_dim=128, dtype=torch.bfloat16):
        from h3_optimizations.native import producer as native_producer

        return native_producer.int8_attention_producer_spec(
            (1, heads, sequence, head_dim),
            (1, heads, sequence, head_dim),
            dtype=dtype,
            device=torch.device('cuda'),
        )

    def test_two_pass_matches_the_whole_v_quantizer_byte_for_byte(self):
        for sequence in (2048, 4096 + 128, 5000):
            spec = self._spec(sequence)
            v = random_v(spec, seed=sequence)
            reference, reference_scale = self._whole_v(spec, v)
            for chunk_rows in (128, 512, 1024, 4096):
                packed, scale = produce(
                    spec, v, chunk_rows, backend=v_staging.available_backend()
                )
                self.assertTrue(
                    torch.equal(scale, reference_scale),
                    'scales differ: sequence %d chunk %d'
                    % (sequence, chunk_rows),
                )
                self.assertTrue(
                    torch.equal(packed, reference),
                    'V bytes differ: sequence %d chunk %d'
                    % (sequence, chunk_rows),
                )

    def test_the_torch_reference_tracks_the_kernels_to_the_last_ulp(self):
        '''Equal scales, and INT8 differing only where rounding is a tie.

        Not byte equality, and it must not be asserted as such. The library is
        compiled with --use_fast_math, so the kernel's `1.f / sc` is an
        approximate reciprocal; an ulp there flips values sitting exactly on a
        rounding boundary. The shipped whole-V quantizer uses the same
        expression, so the native path is the one that matches the carrier --
        which the byte-parity test above is what actually proves.
        '''
        if v_staging.available_backend() != v_staging.BACKEND_NATIVE:
            self.skipTest('the native staging kernels are not built')
        spec = self._spec(4096)
        v = random_v(spec, seed=99)
        native, native_scale = produce(
            spec, v, 1024, backend=v_staging.BACKEND_NATIVE
        )
        torch_packed, torch_scale = produce(
            spec, v, 1024, backend=v_staging.BACKEND_TORCH
        )
        self.assertTrue(torch.equal(native_scale, torch_scale))
        delta = native.to(torch.int16) - torch_packed.to(torch.int16)
        self.assertLessEqual(int(delta.abs().max()), 1)
        differing = delta != 0
        self.assertLess(int(differing.sum()), delta.numel() // 1000)

        # Every disagreement has to be a tie, or something else is wrong.
        inverse = v_staging._inverse_permutation_16(v.device)
        source = torch.arange(v.shape[2], device=v.device)
        destination = (source & ~15) | inverse[source & 15]
        back = torch.empty_like(destination)
        back[destination] = source
        rows, columns = torch.nonzero(differing, as_tuple=True)
        head_dim = v.shape[3]
        head = (rows // head_dim) % v.shape[1]
        channel = rows % head_dim
        quotient = (
            v[0, head, back[columns], channel].float()
            / native_scale[rows].float()
        )
        residual = (quotient - torch.round(quotient)).abs()
        self.assertTrue(bool(((residual - 0.5).abs() < 1e-4).all()))

    def test_a_strided_chunk_view_is_accepted(self):
        '''The producer hands over transposed HND views, not copies.'''
        spec = self._spec(2048)
        contiguous = random_v(spec, seed=17)
        rows_major = (
            contiguous.squeeze(0).transpose(0, 1).contiguous()
        )  # [sequence, heads, dim]
        view = rows_major.transpose(0, 1).unsqueeze(0)
        self.assertFalse(view.is_contiguous())
        self.assertTrue(torch.equal(view, contiguous))
        reference, reference_scale = produce(
            spec, contiguous, 512, backend=v_staging.available_backend()
        )
        packed, scale = produce(
            spec, view, 512, backend=v_staging.available_backend()
        )
        self.assertTrue(torch.equal(scale, reference_scale))
        self.assertTrue(torch.equal(packed, reference))


if __name__ == '__main__':
    unittest.main()
