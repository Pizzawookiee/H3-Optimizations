"""SM75 prepared-QKV Sage backend."""

from .sm86 import SageSM86MemoryEfficientBackend


class SageSM75MemoryEfficientBackend(SageSM86MemoryEfficientBackend):
    """Sage's Triton per-block INT8 Q/K and FP16-V path."""

    name = "sage_mem_eff_sm75"
    capabilities = frozenset({(7, 5)})
