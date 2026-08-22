'''CPU output-exact contract for the MLP sharing observation seam.'''

import os
from pathlib import Path
import sys
import unittest

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

from h3_optimizations.memory.config import (  # noqa: E402
    MODE_NATIVE,
    ActivationMemoryConfig,
)
from h3_optimizations.memory.forward import make_forward  # noqa: E402
from h3_optimizations.memory.observer import OBSERVER_KEY  # noqa: E402
from h3_optimizations.memory.sharing import SHARING_KEY  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class RecordingObserver:
    def __init__(self):
        self.chunks = 0
        self.block_ends = 0

    def observe_exact_mlp(self, _layer, _options, **payload):
        self.chunks += 1
        h = payload['h']
        extra = payload['evaluate_mlp']((h[:1] + h[-1:]).mul(0.5))
        if tuple(extra.shape) != (1, h.shape[-1]):
            raise AssertionError('diagnostic MLP evaluator returned the wrong shape')

    def end_mlp_block(self, _layer, _options):
        self.block_ends += 1


class ExactSharing:
    def __init__(self):
        self.chunks = 0
        self.block_ends = 0

    def evaluate_chunk(self, _layer, _options, **payload):
        self.chunks += 1
        return payload['evaluate_mlp'](payload['h'])

    def end_mlp_block(self, _layer, _options):
        self.block_ends += 1


class MLPSharingForwardTests(unittest.TestCase):
    @staticmethod
    def _block():
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
            parameter.detach().copy_(torch.randn_like(parameter) * 0.03)
            parameter.requires_grad_(False)
        return block

    def test_observer_evaluates_counterfactual_without_changing_output(self):
        torch.manual_seed(90)
        block = self._block()
        x = torch.randn(19, 32) * 0.1
        t_emb = torch.randn(1, 24) * 0.1
        segments = [(0, 5, 0), (5, 13, 1), (13, 19, 2)]
        config = ActivationMemoryConfig(
            mode=MODE_NATIVE,
            chunk_rows=256,
            alignment=256,
        )
        expected = make_forward(block, 0, config)(
            x.clone(),
            t_emb,
            segments,
            rope_freqs=None,
            transformer_options={},
        )
        observer = RecordingObserver()
        actual = make_forward(block, 0, config)(
            x.clone(),
            t_emb,
            segments,
            rope_freqs=None,
            transformer_options={OBSERVER_KEY: observer},
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(observer.chunks, 3)
        self.assertEqual(observer.block_ends, 1)

    def test_executable_sharing_seam_owns_only_mlp_evaluation(self):
        torch.manual_seed(91)
        block = self._block()
        x = torch.randn(19, 32) * 0.1
        t_emb = torch.randn(1, 24) * 0.1
        segments = [(0, 5, 0), (5, 13, 1), (13, 19, 2)]
        config = ActivationMemoryConfig(
            mode=MODE_NATIVE,
            chunk_rows=256,
            alignment=256,
        )
        expected = make_forward(block, 0, config)(
            x.clone(),
            t_emb,
            segments,
            rope_freqs=None,
            transformer_options={},
        )
        sharing = ExactSharing()
        actual = make_forward(block, 0, config)(
            x.clone(),
            t_emb,
            segments,
            rope_freqs=None,
            transformer_options={SHARING_KEY: sharing},
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(sharing.chunks, 3)
        self.assertEqual(sharing.block_ends, 1)


if __name__ == '__main__':
    unittest.main()
