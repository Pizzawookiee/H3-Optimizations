'''CPU contract for FP16 ConvRot two-slice MLP execution.'''

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import h3_optimizations.memory.linear as linear_module  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class FP16ConvRotTests(unittest.TestCase):
    def test_fp16_two_slice_matches_unsliced_fake_math(self):
        hidden = 256
        ffn = 512
        torch.manual_seed(75)
        fc1_q = torch.randint(-1, 2, (ffn * 2, hidden), dtype=torch.int8)
        fc2_q = torch.randint(-1, 2, (hidden, ffn), dtype=torch.int8)
        fc1_scale = torch.ones(ffn * 2)
        fc2_scale = torch.ones(hidden)

        mlp = type('MLP', (), {'fc1': object(), 'fc2': object()})()

        class FakeAcquired:
            def __init__(self):
                self.weight = object()
                self.bias = None
                self.released = False

            def release(self):
                self.released = True

        acquired = [FakeAcquired(), FakeAcquired()]

        def fake_convrot(x, qdata, _scale, input_act=None):
            if input_act == 'swiglu':
                gate, up = x.chunk(2, dim=-1)
                x = torch.nn.functional.silu(gate) * up
            return x @ qdata.to(x.dtype).t()

        x = torch.randn(3, hidden, dtype=torch.float16) * 0.01
        with patch.object(
            linear_module,
            'acquire_linear',
            side_effect=acquired,
        ), patch.object(
            linear_module,
            '_convrot_parts',
            side_effect=((fc1_q, fc1_scale), (fc2_q, fc2_scale)),
        ):
            with linear_module.ConvRotTwoSliceMLP(
                mlp,
                x[:1],
                fake_convrot,
            ) as session:
                actual, path = session.fc1_fc2(x)

        gate = x @ fc1_q[:ffn].to(torch.float16).t()
        up = x @ fc1_q[ffn:].to(torch.float16).t()
        expected = (
            torch.nn.functional.silu(gate) * up
        ) @ fc2_q.to(torch.float16).t()

        self.assertEqual(actual.dtype, torch.float16)
        self.assertEqual(actual.shape, (3, hidden))
        self.assertTrue(torch.allclose(actual, expected, atol=0.25, rtol=0.0))
        self.assertEqual(path, 'held_convrot_2slice')
        self.assertTrue(all(item.released for item in acquired))


if __name__ == '__main__':
    unittest.main()
