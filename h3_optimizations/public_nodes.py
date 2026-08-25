'''Public ComfyUI node registration for H3 Optimizations.'''

import os

from comfy_api.latest import ComfyExtension

# Import first so every public node, including Sparse Attention used without the
# Memory node, resolves QKV through the same streamed-BF16 priority policy.
from . import apply_policy as _apply_policy  # noqa: F401
from .aimdo_limiter import H3AIMDOResidencyLimiter
from .memory_migration_node import H3MemoryOptimization
from .nodes import (
    H3SparseAttention,
    H3SparseAttentionAdvanced,
)


BENCHMARK_NODES_ENV = 'H3_OPTIMIZATIONS_BENCHMARK_NODES'


class H3OptimizationsExtension(ComfyExtension):
    '''Register production nodes, plus explicit opt-in benchmark controls.'''

    async def get_node_list(self):
        nodes = [
            H3MemoryOptimization,
            H3AIMDOResidencyLimiter,
            H3SparseAttention,
            H3SparseAttentionAdvanced,
        ]
        if os.environ.get(BENCHMARK_NODES_ENV) == '1':
            from .benchmark_nodes import H3BenchmarkForceQKVConfig0

            nodes.append(H3BenchmarkForceQKVConfig0)
        return nodes
