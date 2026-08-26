'''CPU contracts for low-VRAM streamed-query FROST BF16 execution.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
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

from h3_optimizations.attention.sparse.frost_bf16 import (  # noqa: E402
    FrostBF16Spec,
)
from h3_optimizations.attention.sparse.frost_bf16_streamed import (  # noqa: E402
    PreparedStreamedFrostBF16,
    _assemble_streamed_frost_qkv,
    execute_streamed_frost_bf16,
    prepare_streamed_frost_bf16,
)
from h3_optimizations.attention.sparse.router import SparseTileRouter  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakeHeld:
    def __init__(self, factory, heads=2, head_dim=128):
        self.factory = factory
        self.heads = heads
        self.head_dim = head_dim
        self.full_calls = []
        self.q_calls = []
        self.active = False

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *_args):
        self.active = False
        return False

    def _q(self, x, start, end):
        rows = end - start
        return (
            x[start:end]
            .reshape(rows, self.heads, self.head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
            .contiguous()
        )

    def project_hnd(self, x, _rope, start, end):
        self.full_calls.append((start, end))
        q = self._q(x, start, end)
        return q, q + 100, q + 200

    def project_q_hnd(self, x, _rope, start, end):
        self.q_calls.append((start, end))
        return self._q(x, start, end)


class FakeHeldFactory:
    def __init__(self):
        self.bindings = []

    def __call__(self, _module, _sample, _projection_mode):
        held = FakeHeld(self)
        self.bindings.append(held)
        return held


def packed_layout(sequence=130, video_start=64):
    return SimpleNamespace(
        seq_len=sequence,
        video_range=(video_start, sequence),
        segments=((0, video_start, 'context'), (video_start, sequence, 'video')),
        video_shape=(1, 1, 1),
        audio_t=0,
    )


class StreamedFrostBF16Tests(unittest.TestCase):
    def setUp(self):
        self.sequence = 130
        self.heads = 2
        self.hidden = self.heads * 128
        self.x = torch.arange(
            self.sequence * self.hidden,
            dtype=torch.bfloat16,
        ).reshape(self.sequence, self.hidden)
        self.module = SimpleNamespace(
            heads=self.heads,
            head_dim=128,
            out_proj=lambda value: value,
        )
        self.spec = FrostBF16Spec(heads=self.heads)

    def project(self, factory):
        return _assemble_streamed_frost_qkv(
            self.module,
            self.x,
            None,
            spec=self.spec,
            layer_index=3,
            chunk_rows=64,
            held_factory=factory,
        )

    def test_projection_retains_sequence_major_kv_and_summaries_but_no_q(self):
        factory = FakeHeldFactory()
        projected = self.project(factory)

        self.assertEqual(factory.bindings[0].full_calls, [(0, 64), (64, 128), (128, 130)])
        self.assertFalse(factory.bindings[0].active)
        self.assertFalse(hasattr(projected, 'q'))
        self.assertEqual(tuple(projected.k.shape), (1, 2, 130, 128))
        self.assertEqual(tuple(projected.v.shape), (1, 2, 130, 128))
        expected_stride = (130 * 2 * 128, 128, 2 * 128, 1)
        self.assertEqual(projected.k.stride(), expected_stride)
        self.assertEqual(projected.v.stride(), expected_stride)
        self.assertEqual(tuple(projected.q_summary.shape), (1, 2, 3, 128))
        self.assertEqual(tuple(projected.k_summary.shape), (1, 2, 3, 128))
        projected.release()

    def test_prepare_builds_absolute_route_from_summaries(self):
        factory = FakeHeldFactory()
        projected = self.project(factory)
        backend = SimpleNamespace(
            name='frost_bf16_sm89',
            config=SimpleNamespace(
                video_budget=0.5,
                denser_early_late_steps=False,
                early_steps=None,
                early_kv=None,
                late_steps=None,
                late_kv=None,
                layer_video_budgets=None,
            ),
            router=SparseTileRouter(q_tile=64, kv_tile=64),
            _snapshot=lambda _options, _sequence: SimpleNamespace(
                step_index=0,
                total_steps=1,
                layout=packed_layout(),
            ),
        )

        prepared = prepare_streamed_frost_bf16(
            backend,
            projected,
            layer_index=3,
            transformer_options={},
        )

        self.assertIsNone(projected.q_summary)
        self.assertIsNone(projected.k_summary)
        self.assertEqual(tuple(prepared.route.shape), (1, 2, 3, 3))
        self.assertEqual(tuple(prepared.counts.shape), (1, 2, 3))
        self.assertEqual(
            prepared.metadata['qkv_lifetime'],
            'streamed_q_global_sequence_major_bf16_kv',
        )
        prepared.release()

    def test_execute_uses_bounded_q_against_global_kv_and_releases_before_launch(self):
        factory = FakeHeldFactory()
        original = self.x.clone()
        projected = self.project(factory)
        route = torch.arange(3, dtype=torch.int32).view(1, 1, 1, 3).expand(
            1, self.heads, 3, 3
        ).contiguous()
        counts = torch.full((1, self.heads, 3), 3, dtype=torch.int32)
        prepared = PreparedStreamedFrostBF16(
            projected=projected,
            route=route,
            counts=counts,
            metadata={},
        )
        launches = []

        class Executor:
            def prepare(inner_self, q, k, v, route_chunk, counts_chunk, **_kwargs):
                self.assertTrue(all(not binding.active for binding in factory.bindings))
                launches.append(
                    (
                        int(q.shape[-2]),
                        int(k.shape[-2]),
                        tuple(route_chunk.shape),
                        tuple(counts_chunk.shape),
                        q.stride(),
                    )
                )
                self.assertEqual(k.shape, v.shape)
                return SimpleNamespace(q=q)

            @staticmethod
            def execute(chunk):
                return chunk.q

        backend = SimpleNamespace(
            spec=self.spec,
            executor=Executor(),
        )
        actual = execute_streamed_frost_bf16(
            self.module,
            backend,
            prepared,
        )

        self.assertIs(actual, self.x)
        torch.testing.assert_close(actual, original)
        self.assertEqual([item[:2] for item in launches], [(64, 130), (64, 130), (2, 130)])
        self.assertEqual(
            [item[2:4] for item in launches],
            [((1, 2, 1, 3), (1, 2, 1))] * 3,
        )
        self.assertEqual(
            [binding.q_calls for binding in factory.bindings[1:]],
            [[(0, 64)], [(64, 128)], [(128, 130)]],
        )
        self.assertTrue(all(not binding.active for binding in factory.bindings))
        self.assertIsNone(projected.k)
        self.assertIsNone(projected.v)


if __name__ == '__main__':
    unittest.main()
