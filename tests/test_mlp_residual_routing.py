'''CPU contracts for in-place SwiGLU, short-sequence fused routing, and
runtime ConvRot INT8 fc2 dispatch.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
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

from h3_optimizations.memory.config import (  # noqa: E402
    MODE_CONVROT_INT8_RUNTIME,
    MODE_NATIVE,
    ActivationMemoryConfig,
)
from h3_optimizations.memory.forward import make_forward  # noqa: E402
import h3_optimizations.memory.forward as forward_module  # noqa: E402
import h3_optimizations.memory.linear as linear_module  # noqa: E402
from h3_optimizations.qkv.int8 import HeldConvRotINT8MLP  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def _make_block():
    block = DiTBlock(
        32, 2, 16, 48, 24, 1e-5, 1e-5,
        apply_silu=False, adaln_dtype=torch.float32,
        dtype=torch.float32, operations=comfy.ops.disable_weight_init,
    )
    for parameter in block.parameters():
        parameter.detach().copy_(torch.randn_like(parameter) * 0.03)
        parameter.requires_grad_(False)
    return block


class SwigluInplaceTests(unittest.TestCase):
    def test_inplace_matches_eager_bit_for_bit_and_reuses_storage(self):
        torch.manual_seed(1)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.randn(7, 24, dtype=dtype)
            expected = linear_module.swiglu_eager(x.clone())
            work = x.clone()
            actual = linear_module.swiglu_inplace_(work)
            self.assertTrue(torch.equal(actual, expected), dtype)
            self.assertEqual(actual.data_ptr(), work.data_ptr())
            self.assertEqual(tuple(actual.shape), (7, 12))

    def test_held_float_fc2_uses_the_inplace_activation(self):
        torch.manual_seed(2)
        weight = torch.randn(5, 12)
        held = linear_module.HeldMLP.__new__(linear_module.HeldMLP)
        held.fc2_weight = SimpleNamespace(weight=weight, bias=None)
        expanded = torch.randn(4, 24)
        expected = torch.nn.functional.linear(linear_module.swiglu_eager(expanded.clone()), weight)
        out, path = held.fc2_swiglu(expanded.clone(), native=True)
        self.assertEqual(path, 'held_bf16_swiglu')
        self.assertTrue(torch.equal(out, expected))


class ShortSequenceRoutingTests(unittest.TestCase):
    def test_only_runtime_int8_moves_short_sequences_to_the_held_path(self):
        # Native checkpoints keep one upstream call: per-segment slabs re-read the
        # MLP weights per segment and measured slower than the fused residual saves.
        self.assertFalse(forward_module._short_sequence_uses_held(ActivationMemoryConfig(mode=MODE_NATIVE)))
        self.assertTrue(forward_module._short_sequence_uses_held(
            ActivationMemoryConfig(mode=MODE_CONVROT_INT8_RUNTIME)
        ))

    def test_short_sequence_held_path_uses_one_slab_per_segment(self):
        torch.manual_seed(3)
        block = _make_block()
        x = torch.randn(19, 32) * 0.1
        t_emb = torch.randn(1, 24) * 0.1
        segments = [(0, 5, 0), (5, 13, 1), (13, 19, 2)]
        expected = block.forward(x.clone(), t_emb, segments, rope_freqs=None, transformer_options={})
        seen = []
        real = linear_module.HeldMLP.fc2_swiglu_residual_

        def record(held, expanded, residual, gate):
            seen.append(int(residual.shape[0]))
            return False

        config = ActivationMemoryConfig(mode=MODE_NATIVE, chunk_rows=4096)
        with patch.object(forward_module, '_short_sequence_uses_held', return_value=True), \
                patch.object(linear_module.HeldMLP, 'fc2_swiglu_residual_', record):
            actual = make_forward(block, 0, config)(
                x.clone(), t_emb, segments, rope_freqs=None, transformer_options={},
            )
        self.assertIs(linear_module.HeldMLP.fc2_swiglu_residual_, real)
        # One slab per modulation segment, since the kernel takes one gate each.
        self.assertEqual(seen, [5, 8, 6])
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=2e-6))

    def test_short_native_sequence_keeps_upstream_mlp(self):
        torch.manual_seed(4)
        block = _make_block()
        x = torch.randn(19, 32) * 0.1
        t_emb = torch.randn(1, 24) * 0.1
        segments = [(0, 5, 0), (5, 13, 1), (13, 19, 2)]
        with patch.object(forward_module, '_open_mlp',
                          side_effect=AssertionError('held path must stay unused')):
            make_forward(block, 0, ActivationMemoryConfig(mode=MODE_NATIVE, chunk_rows=4096))(
                x.clone(), t_emb, segments, rope_freqs=None, transformer_options={},
            )


class RuntimeInt8Fc2Tests(unittest.TestCase):
    def _held(self, override=None):
        held = HeldConvRotINT8MLP.__new__(HeldConvRotINT8MLP)
        module = SimpleNamespace()
        if override is not None:
            module._h3_benchmark_convrot_linear = override
        held.fc2_binding = SimpleNamespace(module=module, weight=object(), bias=None, linear=lambda a: a * 2)
        return held

    def test_benchmark_override_keeps_eager_swiglu_and_declines_fusion(self):
        held = self._held(override=lambda *args: None)
        expanded = torch.randn(3, 8)
        out, path = held.fc2_swiglu(expanded.clone())
        self.assertEqual(path, 'held_convrot_int8')
        self.assertTrue(torch.equal(out, linear_module.swiglu_eager(expanded) * 2))
        self.assertFalse(held.fc2_swiglu_residual_(expanded, torch.zeros(3, 4), torch.zeros(4)))

    def test_runtime_int8_fallback_still_uses_fused_swiglu_dispatch(self):
        expanded = torch.randn(3, 8)
        seen = []
        held = SimpleNamespace(
            fc1=lambda h: expanded,
            fc1_fc2=lambda *_args: self.fail('separate SwiGLU path was used'),
            fc2_swiglu=lambda value: (seen.append(value) or value[..., :4], 'fused'),
        )
        out, work, path = forward_module._run_mlp(
            None,
            torch.randn(3, 4),
            held,
            'int8',
            ActivationMemoryConfig(mode=MODE_CONVROT_INT8_RUNTIME),
        )
        self.assertEqual(seen, [expanded])
        self.assertIs(work, expanded)
        self.assertEqual(path, 'fused')
        self.assertTrue(torch.equal(out, expanded[..., :4]))


if __name__ == '__main__':
    unittest.main()
