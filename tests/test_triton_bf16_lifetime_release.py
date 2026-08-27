"""CPU lifetime contracts for streamed BF16 Triton execution."""

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
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.attention.sparse import triton_bf16  # noqa: E402
from h3_optimizations.attention.sparse import triton_bf16_streamed  # noqa: E402
from h3_optimizations.attention.sparse.triton_bf16_streamed import (  # noqa: E402
    PreparedStreamedTritonBF16,
    execute_streamed_triton_bf16,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class _RoutePlan:
    def __init__(self):
        self.release_calls = 0

    def release(self):
        self.release_calls += 1


class _Projected:
    def __init__(self, module, sequence=128, heads=2, head_dim=128):
        self.module = module
        self.sequence = sequence
        self.heads = heads
        self.head_dim = head_dim
        self.chunk_rows = 64
        self.x = torch.zeros(sequence, heads * head_dim, dtype=torch.bfloat16)
        self.k = torch.zeros(1, heads, sequence, head_dim, dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)
        self.weight_released = False
        self.released = False

    def project_q(self, start, end):
        rows = end - start
        return torch.zeros(
            1,
            self.heads,
            rows,
            self.head_dim,
            dtype=torch.bfloat16,
        )

    def release_weight(self):
        self.weight_released = True

    def release(self):
        self.released = True
        self.k = None
        self.v = None


class TritonBF16LifetimeReleaseTests(unittest.TestCase):
    def test_final_route_and_kv_die_before_final_out_proj(self):
        out_proj_route_states = []
        prepared_box = {}

        def out_proj(value):
            prepared = prepared_box["prepared"]
            out_proj_route_states.append(prepared.route_plan is None)
            return value

        module = SimpleNamespace(heads=2, head_dim=128, out_proj=out_proj)
        projected = _Projected(module)
        route_plan = _RoutePlan()
        prepared = PreparedStreamedTritonBF16(
            projected=projected,
            route_plan=route_plan,
            dense_q_tiles=2,
            sparse_q_tiles=0,
            sparse_selected=0,
            metadata={},
        )
        prepared_box["prepared"] = prepared

        empty_route = torch.empty((1, 2, 0, 0), dtype=torch.int32)
        with mock.patch.object(
            triton_bf16_streamed,
            "build_compact_absolute_route_chunk",
            return_value=empty_route,
        ), mock.patch.object(
            triton_bf16,
            "_launch_streamed_chunk",
            side_effect=lambda q, *_args, **_kwargs: q,
        ):
            actual = execute_streamed_triton_bf16(
                module,
                SimpleNamespace(router=object()),
                prepared,
            )

        self.assertIs(actual, projected.x)
        self.assertTrue(projected.weight_released)
        self.assertIsNone(projected.k)
        self.assertIsNone(projected.v)
        self.assertEqual(route_plan.release_calls, 1)
        self.assertEqual(out_proj_route_states, [False, True])
        self.assertTrue(projected.released)


if __name__ == "__main__":
    unittest.main()
