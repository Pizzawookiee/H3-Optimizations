'''CPU contracts for the post-RoPE H3 attention ordering experiment.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

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

from h3_optimizations import attention_forward  # noqa: E402
from h3_optimizations import ordering_probe  # noqa: E402
from h3_optimizations.attention_ordering import (  # noqa: E402
    ORDERINGS,
    analyze_orderings,
    apply_permutation,
    inverse_permutation,
    packed_permutation,
    restore_permutation,
    video_permutation,
)
from h3_optimizations.ordering_probe import (  # noqa: E402
    AttentionOrderingConfig,
    AttentionOrderingSession,
    ORDERING_OBSERVER_KEY,
)
from h3_optimizations.runtime.context import RUNTIME_KEY, RuntimeSnapshot  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def layout(context=3, shape=(2, 2, 4)):
    video_tokens = shape[0] * shape[1] * shape[2]
    return SimpleNamespace(
        seq_len=context + video_tokens,
        video_range=(context, context + video_tokens),
        video_shape=shape,
    )


class AttentionOrderingTests(unittest.TestCase):
    def test_all_orderings_are_exact_permutations(self):
        shape = (3, 2, 5)
        expected = list(range(30))
        for name in ORDERINGS:
            with self.subTest(name=name):
                permutation = video_permutation(shape, name)
                self.assertEqual(sorted(permutation), expected)
                inverse = inverse_permutation(permutation)
                self.assertEqual(
                    tuple(permutation[inverse[index]] for index in expected),
                    tuple(expected),
                )

    def test_explicit_controls_have_expected_order(self):
        shape = (2, 2, 2)
        self.assertEqual(video_permutation(shape, 'native'), tuple(range(8)))
        self.assertEqual(video_permutation(shape, 'row_major'), tuple(range(8)))
        self.assertEqual(
            video_permutation(shape, 'time_major'),
            (0, 4, 1, 5, 2, 6, 3, 7),
        )

    def test_hilbert_cube_moves_only_to_adjacent_cells(self):
        shape = (2, 2, 2)
        permutation = video_permutation(shape, 'hilbert')
        coordinates = [
            (index // 4, (index % 4) // 2, index % 2)
            for index in permutation
        ]
        for left, right in zip(coordinates, coordinates[1:]):
            self.assertEqual(sum(abs(a - b) for a, b in zip(left, right)), 1)

    def test_packed_permutation_keeps_context_fixed_and_restores(self):
        resolved = layout()
        permutation = packed_permutation(resolved, 'hilbert')
        self.assertEqual(permutation[:3], (0, 1, 2))
        source = torch.arange(resolved.seq_len * 2).reshape(1, resolved.seq_len, 2)
        ordered = apply_permutation(source, permutation)
        restored = restore_permutation(ordered, permutation)
        self.assertTrue(torch.equal(restored, source))

    def test_fixed_density_analysis_uses_identical_queries(self):
        resolved = layout(context=3, shape=(2, 2, 4))
        generator = torch.Generator().manual_seed(73)
        q = torch.randn((1, 2, resolved.seq_len, 8), generator=generator)
        k = torch.randn((1, 2, resolved.seq_len, 8), generator=generator)
        v = torch.randn((1, 2, resolved.seq_len, 8), generator=generator)
        result = analyze_orderings(
            q,
            k,
            v,
            resolved,
            budgets=(0.5, 1.0),
            query_samples=5,
            q_tile=4,
            kv_tile=2,
            head_chunk=1,
        )
        self.assertTrue(result['post_rope'])
        self.assertTrue(result['visual_tokens_only'])
        self.assertEqual(result['query_samples'], 5)
        self.assertEqual(set(result['orderings']), set(ORDERINGS))
        self.assertEqual(result['orderings']['native'], result['orderings']['row_major'])
        for name in ORDERINGS:
            rows = result['orderings'][name]['budgets']
            self.assertEqual(rows[0]['retained_video_kv_tiles'], 4)
            self.assertEqual(rows[0]['pure_video_kv_tiles'], 8)
            self.assertLess(rows[1]['relative_l1_error'], 1.0e-6)
            self.assertLess(rows[1]['relative_l2_error'], 1.0e-6)
            self.assertGreater(rows[0]['retained_dense_attention_mass']['mean'], 0.0)

    def test_probe_config_normalizes_percent_budgets(self):
        config = AttentionOrderingConfig(
            layers='0,24,49',
            steps='0,7',
            budgets='20,30,50',
            query_samples=32,
        )
        self.assertEqual(config.layers, (0, 24, 49))
        self.assertEqual(config.steps, (0, 7))
        self.assertEqual(config.budgets, (0.2, 0.3, 0.5))

    def test_observer_bypasses_projected_qkv_and_sees_hnd_tensors(self):
        calls = []

        class Observer:
            @staticmethod
            def observe_attention(layer_index, options, q, k, v):
                calls.append((layer_index, options, q.clone(), k.clone(), v.clone()))

        class Projector:
            name = 'must_be_bypassed'

            @staticmethod
            def try_project(*_args, **_kwargs):
                raise AssertionError('projected QKV must be bypassed while probing')

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

        module = SimpleNamespace(
            heads=1,
            head_dim=2,
            out_proj=lambda value: value,
        )
        q = torch.arange(12, dtype=torch.float32).reshape(6, 1, 2)
        k = q + 100
        v = q + 200
        forward = attention_forward.make_forward(
            module,
            7,
            backend=Backend(),
            projector=Projector(),
        )
        options = {ORDERING_OBSERVER_KEY: Observer()}
        with mock.patch.object(
            attention_forward,
            'project_qkv',
            return_value=(q, k, v),
        ):
            output = forward(torch.empty(6, 2), transformer_options=options)
        self.assertEqual(tuple(output.shape), (6, 2))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 7)
        self.assertEqual(tuple(calls[0][2].shape), (1, 1, 6, 2))
        self.assertTrue(torch.equal(calls[0][4][0, 0], v[:, 0]))

    def test_session_measures_each_selected_layer_step_branch_once(self):
        resolved = layout(context=3, shape=(2, 2, 4))
        config = AttentionOrderingConfig(
            layers=(7,),
            steps=(2,),
            budgets=(0.5,),
            query_samples=4,
        )
        session = AttentionOrderingSession(config)
        session.begin_request()
        options = {
            RUNTIME_KEY: RuntimeSnapshot(
                request_id=0,
                step_index=2,
                total_steps=20,
                layout=resolved,
                compute_dtype=torch.float32,
                device=torch.device('cpu'),
            ),
            'cond_or_uncond': [0],
        }
        q = torch.randn(1, 1, resolved.seq_len, 4)
        measured = {
            'post_rope': True,
            'visual_tokens_only': True,
            'video_shape': list(resolved.video_shape),
            'query_samples': 4,
            'orderings': {},
        }
        with mock.patch.object(
            ordering_probe,
            'analyze_orderings',
            return_value=measured,
        ) as analyze:
            session.observe_attention(7, options, q, q, q)
            session.observe_attention(7, options, q, q, q)
            session.observe_attention(8, options, q, q, q)
        self.assertEqual(analyze.call_count, 1)
        self.assertEqual(len(session.records), 1)
        self.assertEqual(session.records[0]['layer_index'], 7)
        self.assertEqual(session.records[0]['step_index'], 2)
        self.assertEqual(session.records[0]['branch'], 0)


if __name__ == '__main__':
    unittest.main()
