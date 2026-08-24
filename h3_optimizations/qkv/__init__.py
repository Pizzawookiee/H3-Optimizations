"""Format-neutral QKV inspection, provider selection, and projectors."""

from .bf16 import (
    BF16QKVBindingError,
    ChunkedBF16QKVProjector,
    HeldBF16QKV,
    PreparedBF16QKV,
)
from .formats import (
    H3LinearInventory,
    LinearWeightFormat,
    describe_linear,
    inspect_h3_linears,
)
from .providers import (
    MLPProviderResolution,
    QKVProviderResolution,
    resolve_mlp_provider,
    resolve_qkv_provider,
)
from .projectors import (
    SparseFusedQKVProjector,
)

__all__ = [
    "BF16QKVBindingError",
    "ChunkedBF16QKVProjector",
    "HeldBF16QKV",
    "PreparedBF16QKV",
    "H3LinearInventory",
    "LinearWeightFormat",
    "describe_linear",
    "inspect_h3_linears",
    "MLPProviderResolution",
    "QKVProviderResolution",
    "resolve_mlp_provider",
    "resolve_qkv_provider",
    "SparseFusedQKVProjector",
]
