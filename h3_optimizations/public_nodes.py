'''Public ComfyUI node registration for H3 Optimizations.'''

import os

from comfy_api.latest import ComfyExtension

from .aimdo_limiter import H3AIMDOResidencyLimiter
from .nodes import (
    H3MemoryOptimization,
    H3SparseAttention,
    H3SparseAttentionAdvanced,
)


class H3OptimizationsExtension(ComfyExtension):
    '''Register only production-ready H3 optimization nodes.'''

    async def get_node_list(self):
        nodes = [
            H3MemoryOptimization,
            H3AIMDOResidencyLimiter,
            H3SparseAttention,
            H3SparseAttentionAdvanced,
        ]
        if os.environ.get('H3_ENABLE_BENCHMARK_NODES') == '1':
            from .benchmark_nodes import (
                H3FullForwardDigest,
                H3FullForwardExperiment,
            )
            nodes.extend((H3FullForwardExperiment, H3FullForwardDigest))
        return nodes
