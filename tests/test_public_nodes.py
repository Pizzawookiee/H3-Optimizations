'''CPU-only tests for the production ComfyUI node registry.'''

import asyncio
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.nodes import (  # noqa: E402
    H3MemoryOptimization,
    H3SparseAttention,
    H3SparseAttentionAdvanced,
    _memory_request,
)
from h3_optimizations.plan import (  # noqa: E402
    ATTENTION_EXISTING,
    FUSED_QKV_OFF,
    MLP_MEMORY_OFF,
    MLP_MEMORY_PRESERVE,
)
from h3_optimizations.public_nodes import H3OptimizationsExtension  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class PublicNodeTests(unittest.TestCase):
    def test_public_registry_contains_only_production_nodes(self):
        nodes = asyncio.run(H3OptimizationsExtension().get_node_list())
        self.assertEqual(
            nodes,
            [
                H3MemoryOptimization,
                H3SparseAttention,
                H3SparseAttentionAdvanced,
            ],
        )
        self.assertFalse(any('MLPSharing' in node.__name__ for node in nodes))

    def test_preserve_precision_overrides_only_quantizing_auto_paths(self):
        request = _memory_request(
            fused_qkv='auto',
            mlp_memory='auto',
            preserve_precision=True,
        )
        self.assertEqual(request.attention, ATTENTION_EXISTING)
        self.assertEqual(request.fused_qkv, FUSED_QKV_OFF)
        self.assertEqual(request.mlp_memory, MLP_MEMORY_PRESERVE)

        mlp_off = _memory_request(
            fused_qkv='auto',
            mlp_memory=MLP_MEMORY_OFF,
            preserve_precision=True,
        )
        self.assertEqual(mlp_off.mlp_memory, MLP_MEMORY_OFF)


if __name__ == '__main__':
    unittest.main()
