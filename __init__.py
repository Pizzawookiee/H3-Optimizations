'''ComfyUI entry point for H3 Optimizations.'''

from .h3_optimizations.nodes import H3OptimizationsExtension


async def comfy_entrypoint() -> H3OptimizationsExtension:
    return H3OptimizationsExtension()
