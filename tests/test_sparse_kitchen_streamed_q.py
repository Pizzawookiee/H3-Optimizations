"""Fail-closed contracts for the disabled Kitchen Q-only carrier path."""

import os
from pathlib import Path
import sys
import unittest

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

from h3_optimizations.attention.sparse import kitchen_sparse  # noqa: E402
from h3_optimizations.kitchen_qkv import ChunkedKitchenQKVProjector  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class SparseKitchenQGateTests(unittest.TestCase):
    def test_sparse_projector_retains_the_full_q_carrier(self):
        projector = ChunkedKitchenQKVProjector(
            routing_summaries=True,
            q_tile=64,
            kv_tile=64,
            stream_output=True,
        )
        self.assertFalse(projector.streamed_q)

    def test_q_only_carrier_cannot_be_enabled(self):
        with self.assertRaisesRegex(ValueError, "global K quantization transform"):
            ChunkedKitchenQKVProjector(stream_output=True, streamed_q=True)

    def test_sparse_backend_uses_the_compatibility_implementation(self):
        self.assertEqual(kitchen_sparse.SparseKitchenBackend.__module__, kitchen_sparse.__name__)


if __name__ == "__main__":
    unittest.main()
