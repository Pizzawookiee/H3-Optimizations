'''The chunked Q/K carrier must not move a byte when input stays strided.

This is the property the shipped `strided_qk_input` default rests on. The
equivalent check in tests/test_native_producer.py is written for pytest, which
is not installed in the Comfy venv, so it never runs here -- and it is exactly
the test that would catch a regression in this default.

Deliberately in its own module: the other CPU test files pin
CUDA_VISIBLE_DEVICES=-1 at import, which would silently skip all of this.
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


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class StridedChunkCarrierParityTests(unittest.TestCase):

    def _produce(self, spec, q, k, chunk_rows, allow_strided):
        from h3_optimizations.native import producer as native

        samples = k.index_select(
            2,
            torch.tensor(
                spec.k_anchor_positions, dtype=torch.int64, device=k.device
            ),
        ).contiguous()
        anchor = native.select_int8_attention_k_anchor(spec, samples)
        handle = native.create_int8_attention_producer(spec, anchor)
        sequence = spec.q_input_shape[2]
        for start in range(0, sequence, chunk_rows):
            stop = min(start + chunk_rows, sequence)
            native.quantize_int8_attention_qk_chunk(
                handle,
                q[:, :, start:stop, :],
                k[:, :, start:stop, :],
                q_start=start,
                k_start=start,
                allow_strided_input=allow_strided,
            )
        return handle.q, handle.k, handle.q_scale, handle.k_scale

    def test_strided_and_contiguous_input_agree_byte_for_byte(self):
        from h3_optimizations.native import producer as native

        heads, head_dim = 4, 128
        for sequence in (2048, 4096 + 128):
            spec = native.int8_attention_producer_spec(
                (1, heads, sequence, head_dim),
                (1, heads, sequence, head_dim),
                dtype=torch.bfloat16,
                device=torch.device('cuda'),
            )
            generator = torch.Generator(device='cpu').manual_seed(sequence)
            # The shape the producer really hands over: a transposed view of a
            # [rows, heads, head_dim] projection, not a contiguous tensor.
            rows_major = torch.randn(
                (sequence, heads, head_dim), generator=generator
            ).to(torch.bfloat16).cuda()
            q = rows_major.transpose(0, 1).unsqueeze(0)
            k = (
                rows_major.flip(0).transpose(0, 1).unsqueeze(0).contiguous()
                .squeeze(0).transpose(0, 1).contiguous()
                .transpose(0, 1).unsqueeze(0)
            )
            self.assertFalse(q.is_contiguous())
            self.assertTrue(
                native._kernel_accepts_strided(q, head_dim),
                'the production view must qualify, or the default is a no-op',
            )
            for chunk_rows in (128, 512, 1024, 2048):
                copied = self._produce(spec, q, k, chunk_rows, False)
                strided = self._produce(spec, q, k, chunk_rows, True)
                for name, left, right in zip(
                    ('q_int8', 'k_int8', 'q_scale', 'k_scale'),
                    copied,
                    strided,
                ):
                    self.assertTrue(
                        torch.equal(left, right),
                        '%s differs at sequence %d chunk %d'
                        % (name, sequence, chunk_rows),
                    )


if __name__ == '__main__':
    unittest.main()
