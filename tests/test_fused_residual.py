'''CPU contracts for the fused INT8 gated-residual route.

These prove routing and fallback, not the CUDA kernel: the kernel's
bit-identity is checked on the GPU by the route's own one-time self-test.
'''

import os
from pathlib import Path
import sys
import types
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

from h3_optimizations.memory import fused_residual  # noqa: E402
from h3_optimizations.memory.config import MODE_NATIVE, ActivationMemoryConfig  # noqa: E402
from h3_optimizations.memory.forward import make_forward  # noqa: E402
import h3_optimizations.memory.forward as forward_module  # noqa: E402
import h3_optimizations.memory.linear as linear_module  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def _make_block():
    operations = comfy.ops.disable_weight_init
    block = DiTBlock(
        32, 2, 16, 48, 24, 1e-5, 1e-5,
        apply_silu=False, adaln_dtype=torch.float32,
        dtype=torch.float32, operations=operations,
    )
    for parameter in block.parameters():
        parameter.detach().copy_(torch.randn_like(parameter) * 0.03)
        parameter.requires_grad_(False)
    return block


class FusedResidualRouteTests(unittest.TestCase):
    def tearDown(self):
        fused_residual.set_enabled(None)
        fused_residual._VERIFIED.clear()

    def test_cpu_tensors_never_take_the_fused_route(self):
        expanded = torch.zeros(8, 512, dtype=torch.bfloat16)
        qdata = torch.zeros(256, 256, dtype=torch.int8)
        residual = torch.ones(8, 256, dtype=torch.bfloat16)
        before = residual.clone()
        used = fused_residual.gated_swiglu_residual_(
            expanded, qdata, torch.ones(256), None, residual,
            torch.ones(256, dtype=torch.bfloat16), convrot=True,
        )
        self.assertFalse(used)
        self.assertTrue(torch.equal(residual, before))

    def test_disable_switches(self):
        fused_residual.set_enabled(False)
        self.assertEqual(fused_residual.unavailable_reason(), 'disabled')
        fused_residual.set_enabled(None)
        with patch.dict(os.environ, {fused_residual.DISABLE_ENV: '1'}):
            self.assertEqual(fused_residual.unavailable_reason(), 'disabled')

    def test_old_kitchen_is_reported_with_the_minimum_version(self):
        # Kitchen before 0.2.34: a loaded CUDA extension without the kernel.
        kitchen = types.ModuleType('comfy_kitchen')
        backends = types.ModuleType('comfy_kitchen.backends')
        cuda = types.ModuleType('comfy_kitchen.backends.cuda')
        cuda._C = SimpleNamespace()
        kitchen.backends = backends
        backends.cuda = cuda
        with patch.dict(sys.modules, {
            'comfy_kitchen': kitchen,
            'comfy_kitchen.backends': backends,
            'comfy_kitchen.backends.cuda': cuda,
        }):
            _kitchen, backend, reason = fused_residual._kitchen_backend()
        self.assertIsNone(backend)
        self.assertIn('cutlass_int8_dequant_residual', reason)
        self.assertIn(fused_residual.MIN_KITCHEN, reason)

    def test_only_groupsize_256_convrot_is_a_fused_variant(self):
        self.assertEqual(fused_residual._variant(True, 256), fused_residual.VARIANT_CONVROT)
        self.assertIsNone(fused_residual._variant(True, 128))
        self.assertEqual(fused_residual._variant(False, 256), fused_residual.VARIANT_PLAIN)

    def test_verification_is_cached_per_execution_dtype(self):
        device = torch.device('cuda', 0)
        with (
            patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False),
            patch.object(fused_residual, '_self_test', return_value=(True, None)) as self_test,
        ):
            self.assertTrue(fused_residual._verified(None, None, device, torch.bfloat16, 'plain'))
            self.assertTrue(fused_residual._verified(None, None, device, torch.bfloat16, 'plain'))
            self.assertTrue(fused_residual._verified(None, None, device, torch.float16, 'plain'))

        self.assertEqual(self_test.call_count, 2)
        self.assertEqual(self_test.call_args_list[0].args[3], torch.bfloat16)
        self.assertEqual(self_test.call_args_list[1].args[3], torch.float16)

    def test_status_reports_applicability(self):
        from h3_optimizations import apply
        from h3_optimizations.qkv.providers import (
            MLP_CONVROT_INT8_CHUNKED,
            MLP_FP8_CHUNKED,
        )

        environment = SimpleNamespace(device_index=None)
        self.assertEqual(
            apply._fused_residual_status(SimpleNamespace(provider_id=MLP_FP8_CHUNKED), environment),
            'not applicable',
        )
        # CUDA is hidden in this suite, so the eligible provider reports why not.
        status = apply._fused_residual_status(
            SimpleNamespace(provider_id=MLP_CONVROT_INT8_CHUNKED), environment
        )
        self.assertTrue(status.startswith('unavailable: '), status)

    def test_non_int8_held_weights_decline(self):
        held = linear_module.HeldMLP.__new__(linear_module.HeldMLP)
        held.fc2_weight = SimpleNamespace(weight=torch.zeros(4, 4), bias=None)
        with patch.object(fused_residual, 'gated_swiglu_residual_') as fused:
            self.assertFalse(held.fc2_swiglu_residual_(torch.zeros(3, 8), torch.zeros(3, 4), torch.zeros(4)))
        fused.assert_not_called()


class FusedResidualForwardTests(unittest.TestCase):
    def _run(self, block, x, t_emb, segments, fused_result):
        calls = []
        real_residual = linear_module.HeldMLP.fc2_swiglu_residual_

        def fake_residual(held, expanded, residual, gate):
            calls.append((tuple(residual.shape), tuple(gate.shape)))
            if not fused_result:
                return False
            # Stand-in for the kernel: exactly what the fallback computes.
            out, _path = held.fc2_swiglu(expanded, native=True)
            residual.addcmul_(out, gate)
            return True

        with patch.object(linear_module.HeldMLP, 'fc2_swiglu_residual_', fake_residual):
            output = make_forward(
                block, 0, ActivationMemoryConfig(mode=MODE_NATIVE, chunk_rows=8, alignment=4),
            )(x.clone(), t_emb, segments, rope_freqs=None, transformer_options={})
        self.assertIs(linear_module.HeldMLP.fc2_swiglu_residual_, real_residual)
        return output, calls

    def test_scalar_segments_offer_every_slab_to_the_fused_route(self):
        torch.manual_seed(5)
        block = _make_block()
        x = torch.randn(19, 32) * 0.1
        t_emb = torch.randn(1, 24) * 0.1
        segments = [(0, 5, 0), (5, 13, 1), (13, 19, 2)]
        expected = block.forward(x.clone(), t_emb, segments, rope_freqs=None, transformer_options={})
        for fused_result in (True, False):
            actual, calls = self._run(block, x, t_emb, segments, fused_result)
            self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=2e-6))
            self.assertTrue(calls)
            for residual_shape, gate_shape in calls:
                self.assertEqual(gate_shape, (32,))
                self.assertEqual(residual_shape[1], 32)

    def test_per_token_selectors_keep_the_addcmul_path(self):
        torch.manual_seed(6)
        block = _make_block()
        x = torch.randn(12, 32) * 0.1
        t_emb = torch.randn(2, 24) * 0.1
        selector = torch.tensor([0, 1] * 6, dtype=torch.long)
        segments = [(0, 12, selector)]
        expected = block.forward(x.clone(), t_emb, segments, rope_freqs=None, transformer_options={})
        actual, calls = self._run(block, x, t_emb, segments, True)
        self.assertEqual(calls, [])
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=2e-6))


if __name__ == '__main__':
    unittest.main()
