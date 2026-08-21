'''CPU contracts for bounded MLP execution and ConvRot two-slice math.'''

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

import comfy.ops  # noqa: E402
from comfy.ldm.minimax.model import DiTBlock  # noqa: E402

from h3_optimizations.memory import chunks  # noqa: E402
from h3_optimizations.memory.config import (  # noqa: E402
    MODE_CONVROT_2SLICE,
    MODE_NATIVE,
    ActivationMemoryConfig,
)
from h3_optimizations.memory.forward import make_forward  # noqa: E402
import h3_optimizations.memory.linear as linear_module  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class MemoryTests(unittest.TestCase):
    def test_config_contains_no_epilogue_mode(self):
        config = ActivationMemoryConfig()
        self.assertEqual(config.mode, MODE_NATIVE)
        self.assertEqual(config.chunk_rows, 2048)
        self.assertTrue(
            ActivationMemoryConfig(
                mode=MODE_CONVROT_2SLICE
            ).convrot_2slice
        )
        with self.assertRaises(ValueError):
            ActivationMemoryConfig(
                mode='mlp_chunked_convrot_epilogue'
            )

    def test_chunk_planner_preserves_modulation_boundaries(self):
        result = list(
            chunks.iter_mod_chunks(
                [(0, 5, 0), (5, 19, 1)],
                19,
                max_rows=8,
                alignment=4,
            )
        )
        self.assertEqual(
            [
                (chunk.start, chunk.stop, chunk.mod_row)
                for chunk in result
            ],
            [(0, 5, 0), (5, 13, 1), (13, 19, 1)],
        )
        with self.assertRaisesRegex(ValueError, 'gap'):
            chunks.validate_mod_segments(
                [(0, 4, 0), (5, 8, 1)],
                8,
            )
        with self.assertRaisesRegex(ValueError, 'overlap'):
            chunks.validate_mod_segments(
                [(0, 5, 0), (4, 8, 1)],
                8,
            )

    def test_acquired_weight_release_is_exactly_once(self):
        acquired = linear_module.AcquiredLinear(
            'module',
            'weight',
            'bias',
            'handle',
        )
        with patch.object(
            comfy.ops,
            'uncast_bias_weight',
        ) as release:
            acquired.release()
            acquired.release()
        release.assert_called_once_with(
            'module',
            'weight',
            'bias',
            'handle',
        )

    def test_convrot_two_slice_matches_unsliced_fake_math(self):
        class FakeQuantized:
            def __init__(self, qdata, scale):
                self.qdata = qdata
                self.scale = scale
                self._layout_cls = 'TensorWiseINT8Layout'
                self._params = type(
                    'Params',
                    (),
                    {
                        'transposed': False,
                        'convrot': True,
                        'convrot_groupsize': 256,
                    },
                )()

        class FakeLayout:
            @staticmethod
            def get_plain_tensors(weight):
                return weight.qdata, weight.scale

        class FakeLinear:
            def __init__(self, weight):
                self.weight = weight

        hidden = 256
        ffn = 512
        torch.manual_seed(27)
        fc1_q = torch.randint(
            -1,
            2,
            (ffn * 2, hidden),
            dtype=torch.int8,
        )
        fc2_q = torch.randint(
            -1,
            2,
            (hidden, ffn),
            dtype=torch.int8,
        )
        mlp = type(
            'MLP',
            (),
            {
                'fc1': FakeLinear(
                    FakeQuantized(fc1_q, torch.ones(ffn * 2))
                ),
                'fc2': FakeLinear(
                    FakeQuantized(fc2_q, torch.ones(hidden))
                ),
            },
        )()

        def fake_convrot(x, qdata, _scale, input_act=None):
            if input_act == 'swiglu':
                gate, up = x.chunk(2, dim=-1)
                x = torch.nn.functional.silu(gate) * up
            return x @ qdata.to(x.dtype).t()

        def fake_cast(module, _sample, **_kwargs):
            return module.weight, None, None

        x = torch.randn(3, hidden, dtype=torch.bfloat16) * 0.01
        with patch.object(
            linear_module,
            'QuantizedTensor',
            FakeQuantized,
        ), patch.object(
            linear_module,
            'TensorWiseINT8Layout',
            FakeLayout,
        ), patch.object(
            comfy.ops,
            'cast_bias_weight',
            side_effect=fake_cast,
        ), patch.object(
            comfy.ops,
            'uncast_bias_weight',
        ):
            with linear_module.ConvRotTwoSliceMLP(
                mlp,
                x[:1],
                fake_convrot,
            ) as session:
                actual, path = session.fc1_fc2(x)

        gate = x @ fc1_q[:ffn].to(torch.bfloat16).t()
        up = x @ fc1_q[ffn:].to(torch.bfloat16).t()
        expected = (
            torch.nn.functional.silu(gate) * up
        ) @ fc2_q.to(torch.bfloat16).t()
        self.assertEqual(actual.shape, (3, hidden))
        self.assertTrue(
            torch.allclose(actual, expected, atol=0.25, rtol=0.0)
        )
        self.assertEqual(path, 'held_convrot_2slice')
        self.assertIsNone(session.tiles)

    def test_generic_chunked_forward_matches_core(self):
        torch.manual_seed(2)
        block = DiTBlock(
            hidden=32,
            heads=2,
            head_dim=16,
            ffn=48,
            t_dim=24,
            eps=1e-6,
            qk_eps=1e-6,
            dtype=torch.float32,
            device='cpu',
            operations=comfy.ops.disable_weight_init,
        )
        for parameter in block.parameters():
            parameter.detach().copy_(
                torch.randn_like(parameter) * 0.03
            )
            parameter.requires_grad_(False)

        torch.manual_seed(3)
        x = torch.randn(19, 32) * 0.1
        t_emb = torch.randn(1, 24) * 0.1
        segments = [(0, 5, 0), (5, 13, 1), (13, 19, 2)]
        expected = block.forward(
            x.clone(),
            t_emb,
            segments,
            rope_freqs=None,
            transformer_options={},
        )
        actual = make_forward(
            block,
            0,
            ActivationMemoryConfig(
                mode=MODE_NATIVE,
                chunk_rows=256,
                alignment=256,
            ),
        )(
            x.clone(),
            t_emb,
            segments,
            rope_freqs=None,
            transformer_options={},
        )
        self.assertTrue(
            torch.allclose(actual, expected, rtol=1e-5, atol=2e-6)
        )
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == '__main__':
    unittest.main()
