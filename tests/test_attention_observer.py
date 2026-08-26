'''Compatibility tests for external post-RoPE attention observers.'''

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations import attention_forward  # noqa: E402
from h3_optimizations.ordering_probe import ORDERING_OBSERVER_KEY  # noqa: E402


class AttentionObserverTests(unittest.TestCase):
    def test_observer_bypasses_projected_qkv_and_receives_hnd_tensors(self):
        calls = []

        class Observer:
            @staticmethod
            def observe_attention(layer_index, options, q, k, v):
                calls.append((layer_index, options, q, k, v))

        class Projector:
            name = 'must_be_bypassed'

            @staticmethod
            def try_project(*_args, **_kwargs):
                raise AssertionError('observer must bypass projected QKV')

        class Backend:
            name = 'test_backend'

            @staticmethod
            def prepare(q, k, v, **_kwargs):
                return v

            @staticmethod
            def execute(prepared):
                return prepared

            @staticmethod
            def requires_fallback_inputs(_prepared):
                return False

        module = mock.Mock()
        module.heads = 1
        module.head_dim = 2
        module.out_proj.side_effect = lambda value: value
        q = torch.arange(12, dtype=torch.float32).reshape(6, 1, 2)
        k = q + 100
        v = q + 200
        options = {ORDERING_OBSERVER_KEY: Observer()}
        forward = attention_forward.make_forward(
            module,
            7,
            backend=Backend(),
            projector=Projector(),
        )

        with mock.patch.object(
            attention_forward,
            'project_qkv',
            return_value=(q, k, v),
        ):
            output = forward(torch.empty(6, 2), transformer_options=options)

        self.assertEqual(tuple(output.shape), (6, 2))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 7)
        self.assertIs(calls[0][1], options)
        self.assertEqual(tuple(calls[0][2].shape), (1, 1, 6, 2))
        self.assertTrue(torch.equal(calls[0][4][0, 0], v[:, 0]))


if __name__ == '__main__':
    unittest.main()
