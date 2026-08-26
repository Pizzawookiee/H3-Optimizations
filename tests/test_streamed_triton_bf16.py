"""CPU contracts for low-VRAM streamed-query BF16 Triton execution."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.attention.sparse import triton_bf16  # noqa: E402
from h3_optimizations.attention.sparse.router import SparseTileRouter  # noqa: E402
from h3_optimizations.attention.sparse.triton_bf16_streamed import (  # noqa: E402
    PreparedStreamedTritonBF16,
    _assemble_streamed_triton_qkv,
    execute_streamed_triton_bf16,
    prepare_streamed_triton_bf16,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakeHeld:
    def __init__(self, heads=2, head_dim=128):
        self.heads = heads
        self.head_dim = head_dim
        self.full_calls = []
        self.q_calls = []
        self.released = False

    def project_hnd(self, x, _rope, start, end):
        self.full_calls.append((start, end))
        rows = end - start
        base = x[start:end].reshape(rows, self.heads, self.head_dim)
        q = base.transpose(0, 1).unsqueeze(0).contiguous()
        return q, q + 100, q + 200

    def project_q_hnd(self, x, _rope, start, end):
        self.q_calls.append((start, end))
        rows = end - start
        return (
            x[start:end]
            .reshape(rows, self.heads, self.head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
            .contiguous()
        )

    def __exit__(self, *_args):
        self.released = True
        return False


class OutProjectionProxy:
    def __init__(self, module, out_proj):
        self._module = module
        self.out_proj = out_proj

    def __getattr__(self, name):
        return getattr(self._module, name)


def packed_layout(sequence=130, video_start=64):
    return SimpleNamespace(
        seq_len=sequence,
        video_range=(video_start, sequence),
        segments=((0, video_start, "context"), (video_start, sequence, "video")),
        video_shape=(1, 1, 1),
        audio_t=0,
    )


class StreamedTritonBF16Tests(unittest.TestCase):
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

    def test_projection_retains_kv_and_summaries_but_no_full_q(self):
        held = FakeHeld(self.heads)
        projected = _assemble_streamed_triton_qkv(
            self.module,
            self.x,
            None,
            held,
            layer_index=3,
            chunk_rows=64,
        )

        self.assertEqual(held.full_calls, [(0, 64), (64, 128), (128, 130)])
        self.assertFalse(hasattr(projected, "q"))
        self.assertEqual(tuple(projected.k.shape), (1, 2, 130, 128))
        self.assertEqual(tuple(projected.v.shape), (1, 2, 130, 128))
        self.assertEqual(tuple(projected.q_summary.shape), (1, 2, 3, 128))
        self.assertEqual(tuple(projected.k_summary.shape), (1, 2, 3, 128))
        self.assertFalse(held.released)
        projected.release()
        self.assertTrue(held.released)

    def test_prepare_keeps_only_the_compact_absolute_route(self):
        held = FakeHeld(self.heads)
        projected = _assemble_streamed_triton_qkv(
            self.module,
            self.x,
            None,
            held,
            layer_index=3,
            chunk_rows=64,
        )
        backend = SimpleNamespace(
            name="triton_sparse_bf16",
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

        prepared = prepare_streamed_triton_bf16(
            backend,
            projected,
            layer_index=3,
            transformer_options={},
        )

        self.assertIsNone(projected.q_summary)
        self.assertIsNone(projected.k_summary)
        self.assertEqual(prepared.sparse_lut.ndim, 4)
        self.assertEqual(
            prepared.metadata["qkv_lifetime"],
            "streamed_q_global_bf16_kv",
        )
        prepared.release()

    def test_execute_projects_q_and_output_in_bounded_slabs(self):
        held = FakeHeld(self.heads)
        original = self.x.clone()
        projected = _assemble_streamed_triton_qkv(
            self.module,
            self.x,
            None,
            held,
            layer_index=3,
            chunk_rows=64,
        )
        prepared = PreparedStreamedTritonBF16(
            projected=projected,
            sparse_lut=torch.empty((1, self.heads, 0, 0), dtype=torch.int32),
            dense_q_tiles=3,
            sparse_q_tiles=0,
            sparse_selected=0,
            metadata={},
        )
        launches = []

        def launch(q, _k, _v, _route, **kwargs):
            launches.append((kwargs["q_row_start"], int(q.shape[-2])))
            return q.clone()

        backend = SimpleNamespace()
        with mock.patch.object(
            triton_bf16,
            "_launch_streamed_chunk",
            side_effect=launch,
        ):
            actual = execute_streamed_triton_bf16(
                self.module,
                backend,
                prepared,
            )

        self.assertIs(actual, self.x)
        torch.testing.assert_close(actual, original)
        self.assertEqual(held.q_calls, [(0, 64), (64, 128), (128, 130)])
        self.assertEqual(launches, [(0, 64), (64, 64), (128, 2)])
        self.assertTrue(held.released)
        self.assertIsNone(projected.k)
        self.assertIsNone(projected.v)

    def test_execute_accepts_the_runtime_int8_output_projection_proxy(self):
        held = FakeHeld(self.heads)
        projected = _assemble_streamed_triton_qkv(
            self.module,
            self.x,
            None,
            held,
            layer_index=3,
            chunk_rows=64,
        )
        prepared = PreparedStreamedTritonBF16(
            projected=projected,
            sparse_lut=torch.empty((1, self.heads, 0, 0), dtype=torch.int32),
            dense_q_tiles=3,
            sparse_q_tiles=0,
            sparse_selected=0,
            metadata={},
        )
        out_proj = mock.Mock(side_effect=lambda value: value)
        proxy = OutProjectionProxy(self.module, out_proj)
        with mock.patch.object(
            triton_bf16,
            "_launch_streamed_chunk",
            side_effect=lambda q, *_args, **_kwargs: q.clone(),
        ):
            actual = execute_streamed_triton_bf16(
                proxy,
                SimpleNamespace(),
                prepared,
            )

        self.assertIs(actual, self.x)
        self.assertGreater(out_proj.call_count, 0)


if __name__ == "__main__":
    unittest.main()
