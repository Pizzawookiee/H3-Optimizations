"""Format-guarded fused-QKV projector for Sparse Sage."""

from __future__ import annotations

from .formats import (
    describe_linear,
    is_fused_weight_format_error,
)


def _unsupported(required, message):
    if required:
        raise RuntimeError(
            "required fused QKV became unavailable at runtime: %s"
            % message
        )
    return None


class SparseFusedQKVProjector:
    """Guard the sparse ConvRot projector and fall back for auto requests."""

    name = "h3_fused_qkv_sparse_sage"
    qk_format = "sparge_block_int8"

    def __init__(self, required=False, tensor_core=None):
        from ..attention.sparse.fused_qkv import (
            FusedQKVProjector as Implementation,
        )

        self.required = bool(required)
        self._implementation = Implementation(
            tensor_core=tensor_core
        )

    @property
    def installation_signature(self):
        return (
            self.name,
            self.qk_format,
            bool(self.required),
            self._implementation.installation_signature,
        )

    def bind(self, module):
        callback = getattr(
            self._implementation, "bind", None
        )
        return None if callback is None else callback(module)

    def try_project(
        self,
        module,
        x,
        rope_freqs,
        *,
        layer_index,
        transformer_options,
    ):
        actual = describe_linear(module.qkv_proj)
        if not actual.convrot_int8_256:
            return _unsupported(
                self.required,
                "QKV format is %s" % actual.label,
            )
        try:
            return self._implementation.project(
                module,
                x,
                rope_freqs,
                layer_index=layer_index,
                transformer_options=transformer_options,
            )
        except Exception as exc:
            if is_fused_weight_format_error(exc):
                return _unsupported(self.required, str(exc))
            raise
