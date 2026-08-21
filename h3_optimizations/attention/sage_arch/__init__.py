"""Prepared Sage backends matching upstream architecture dispatch."""

from .common import (
    KernelBinding,
    PreparedArchitecture,
)
from .sm80 import (
    SM80API,
    SageSM80MemoryEfficientBackend,
)
from .sm86 import (
    SM86API,
    SageSM86MemoryEfficientBackend,
)
from .sm90 import (
    SM90API,
    SageSM90MemoryEfficientBackend,
)
from .sm12x import (
    SM12xAPI,
    SageSM12xMemoryEfficientBackend,
)

__all__ = [
    "KernelBinding",
    "PreparedArchitecture",
    "SM80API",
    "SM86API",
    "SM90API",
    "SM12xAPI",
    "SageSM80MemoryEfficientBackend",
    "SageSM86MemoryEfficientBackend",
    "SageSM90MemoryEfficientBackend",
    "SageSM12xMemoryEfficientBackend",
]
