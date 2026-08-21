"""Format-guarded chunked QKV projectors for sparse attention backends."""

from __future__ import annotations

from .formats import (
    describe_linear,
    is_fused_weight_format_error,
)


def _unsupported(required, message):
    if required:
        raise RuntimeError(
            "required sparse QKV optimization became unavailable at runtime: %s"
            % message
        )
    return None


class SparseFusedQKVProjector:
    """Guard chunked Sparse Sage QKV and fall back for auto requests."""

    name = "chunked_sparse_sage_qkv"
    qk_format = "sparge_block_int8"

    def __init__(self, spec, required=False, chunk_rows=4096):
        from ..attention.sparse.chunked_qkv import (
            ChunkedSparseQKVProjector as Implementation,
        )

        self.required = bool(required)
        self.chunk_rows = int(chunk_rows)
        self._implementation = Implementation(
            spec,
            chunk_rows=self.chunk_rows,
        )

    @property
    def installation_signature(self):
        return (
            self.name,
            self.qk_format,
            bool(self.required),
            self._implementation.installation_signature,
        )

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


class TritonSparseQKVProjector:
    """Guard chunked INT8 Triton sparse QKV and fall back for auto requests."""

    name = "chunked_triton_sparse_qkv"
    qk_format = "block_int8"
    v_format = "per_kv_tile_channel_int8"

    def __init__(self, required=False, chunk_rows=4096):
        from ..attention.sparse.triton_qkv import (
            ChunkedTritonSparseQKVProjector as Implementation,
        )

        self.required = bool(required)
        self.chunk_rows = int(chunk_rows)
        self._implementation = Implementation(chunk_rows=self.chunk_rows)

    @property
    def installation_signature(self):
        return (
            self.name,
            self.qk_format,
            self.v_format,
            bool(self.required),
            self._implementation.installation_signature,
        )

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
