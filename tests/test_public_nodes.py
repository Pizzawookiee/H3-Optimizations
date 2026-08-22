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


if __name__ == '__main__':
    unittest.main()
