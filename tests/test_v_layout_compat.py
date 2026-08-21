'''CPU-only contracts for the temporary upstream H3 V-layout shim.'''

import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
import comfy.ldm.minimax.model as h3_model  # noqa: E402
from h3_optimizations.v_layout_compat import (  # noqa: E402
    PROBE_REQUIRED,
    PROBE_UNAVAILABLE,
    PROBE_UNNECESSARY,
    VLayoutProbe,
    install_v_layout_compat,
    make_forward,
    probe_v_layout,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


KNOWN_BAD_SOURCE = '''
def forward(self, x):
    v = v.clone()
    q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
'''


class FakePatcher:
    def __init__(self):
        self.model_options = {'transformer_options': {}}
        self.object_patches = {}

    def add_object_patch(self, name, value):
        self.object_patches[name] = value


def reference_attention(q, k, v, _heads, **_kwargs):
    q = q.take()
    k = k.take()
    v = v.take()
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    out = torch.matmul(torch.softmax(scores, dim=-1), v)
    return out.transpose(1, 2).reshape(
        out.shape[0],
        out.shape[2],
        out.shape[1] * out.shape[3],
    )


class VLayoutProbeTests(unittest.TestCase):
    def test_exact_known_bad_signature_requires_the_shim(self):
        result = probe_v_layout(lambda _target: KNOWN_BAD_SOURCE)
        self.assertEqual(result.state, PROBE_REQUIRED)

    def test_changed_signature_leaves_upstream_untouched(self):
        changed = KNOWN_BAD_SOURCE.replace('v = v.clone()', 'v = v.contiguous()')
        result = probe_v_layout(lambda _target: changed)
        self.assertEqual(result.state, PROBE_UNNECESSARY)

    def test_unavailable_source_fails_open(self):
        def unavailable(_target):
            raise OSError('source is unavailable')

        result = probe_v_layout(unavailable)
        self.assertEqual(result.state, PROBE_UNAVAILABLE)


class VLayoutShimTests(unittest.TestCase):
    def test_changed_or_unavailable_source_does_not_patch(self):
        for state in (PROBE_UNNECESSARY, PROBE_UNAVAILABLE):
            with self.subTest(state=state):
                patcher = FakePatcher()
                probe = VLayoutProbe(state, 'synthetic fail-open result')
                with patch(
                    'h3_optimizations.v_layout_compat.probe_v_layout',
                    return_value=probe,
                ), patch(
                    'h3_optimizations.v_layout_compat.validate',
                    side_effect=AssertionError('fail-open must not inspect blocks'),
                ):
                    result = install_v_layout_compat(patcher)
                self.assertEqual(result.state, state)
                self.assertEqual(patcher.object_patches, {})

    def test_shim_v_is_independent_contiguous_and_numerically_identical(self):
        torch.manual_seed(7)
        module = h3_model.Attention(
            hidden=8,
            heads=2,
            head_dim=4,
            eps=1e-6,
            operations=torch.nn,
        )
        x = torch.randn(7, 8)
        seen = []

        def capture(q, k, v, heads, **kwargs):
            q_tensor = q.peek()
            k_tensor = k.peek()
            v_tensor = v.peek()
            seen.append(
                {
                    'contiguous': v_tensor.is_contiguous(),
                    'v_storage': v_tensor.untyped_storage().data_ptr(),
                    'q_storage': q_tensor.untyped_storage().data_ptr(),
                    'k_storage': k_tensor.untyped_storage().data_ptr(),
                }
            )
            return reference_attention(q, k, v, heads, **kwargs)

        with patch.object(h3_model, 'optimized_attention', capture):
            expected = module.forward(x)
            actual = make_forward(module)(x)

        self.assertFalse(seen[0]['contiguous'])
        self.assertTrue(seen[1]['contiguous'])
        self.assertNotEqual(seen[1]['v_storage'], seen[1]['q_storage'])
        self.assertNotEqual(seen[1]['v_storage'], seen[1]['k_storage'])
        torch.testing.assert_close(actual, expected)

    def test_installer_patches_only_dense_main_blocks_once(self):
        patcher = FakePatcher()
        modules = (SimpleNamespace(forward=lambda: None),) * 2
        probe = VLayoutProbe(PROBE_REQUIRED, 'known-bad source signature matched')

        with patch(
            'h3_optimizations.v_layout_compat.validate',
            return_value=modules,
        ), patch(
            'h3_optimizations.v_layout_compat.probe_v_layout',
            return_value=probe,
        ):
            result = install_v_layout_compat(patcher)

        self.assertEqual(result.state, 'installed')
        self.assertEqual(result.patched_blocks, 2)
        self.assertEqual(
            sorted(patcher.object_patches),
            [
                'diffusion_model.blocks.0.attn.forward',
                'diffusion_model.blocks.1.attn.forward',
            ],
        )


if __name__ == '__main__':
    unittest.main()
