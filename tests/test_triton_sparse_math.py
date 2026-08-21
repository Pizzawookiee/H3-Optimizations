'''Math contracts used by the optimized INT8 Triton sparse kernel.'''

import unittest

import torch


class TritonSparseMathTests(unittest.TestCase):
    def test_signed_offset_recovers_uint8_probability_dot(self):
        generator = torch.Generator().manual_seed(1234)
        p_u8 = torch.randint(
            0, 256, (16, 64), generator=generator, dtype=torch.int32
        )
        v_i8 = torch.randint(
            -128, 128, (64, 128), generator=generator, dtype=torch.int32
        )
        direct = p_u8 @ v_i8
        centered = (p_u8 - 128) @ v_i8
        corrected = centered + 128 * v_i8.sum(dim=0, keepdim=True)
        self.assertTrue(torch.equal(direct, corrected))

    def test_probability_codes_fit_signed_int8_after_offset(self):
        codes = torch.arange(256, dtype=torch.int16)
        signed = codes - 128
        self.assertEqual(int(signed.min()), -128)
        self.assertEqual(int(signed.max()), 127)
        self.assertTrue(torch.equal(signed.to(torch.int8).to(torch.int16), signed))


if __name__ == '__main__':
    unittest.main()
