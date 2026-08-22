'''ComfyUI entry point for H3 Optimizations.'''

import sys

try:
    import h3_optimizations as _h3_optimizations
except ModuleNotFoundError:
    from . import h3_optimizations as _h3_optimizations

    sys.modules['h3_optimizations'] = _h3_optimizations

from h3_optimizations.public_nodes import H3OptimizationsExtension


async def comfy_entrypoint() -> H3OptimizationsExtension:
    return H3OptimizationsExtension()
