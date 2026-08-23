'''Public ComfyUI node registration for H3 Optimizations.'''

from comfy_api.latest import ComfyExtension

from .nodes import (
    H3MemoryOptimization,
    H3SparseAttention,
    H3SparseAttentionAdvanced,
)


class H3OptimizationsExtension(ComfyExtension):
    '''Register only production-ready H3 optimization nodes.'''

    async def get_node_list(self):
        return [
            H3MemoryOptimization,
            H3SparseAttention,
            H3SparseAttentionAdvanced,
        ]
