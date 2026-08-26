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


def _bf16_streamable(actual):
    """Whether normal Comfy execution can project this checkpoint to BF16 chunks."""
    return bool(
        actual.convrot_int8_256
        or actual.w4a8
        or actual.fp8
        or actual.plain_float
    )


class SparseFusedQKVProjector:
    """Guard streamed Sparse Sage QKV and fall back for auto requests.

    Checkpoint weight precision is independent of the streaming contract: every
    supported source projects one bounded BF16 Q/K/V slab at a time. Sparse
    Sage then packs the carrier it needs from those BF16 slabs without ever
    materializing full-sequence BF16 Q.
    """

    name = "chunked_sparse_sage_qkv"
    qk_format = "streamed_q_sparge_block_int8"
    streamed_q = True

    def __init__(
        self,
        spec,
        required=False,
        chunk_rows=4096,
        query_chunk_rows=4096,
    ):
        from ..attention.sparse.sparse_sage_streamed import (
            StreamedSparseSageQKVProjector as Implementation,
        )

        self.required = bool(required)
        self.chunk_rows = int(chunk_rows)
        self.query_chunk_rows = int(query_chunk_rows)
        self._implementation = Implementation(
            spec,
            project_chunk_rows=self.chunk_rows,
            query_chunk_rows=self.query_chunk_rows,
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
        if not _bf16_streamable(actual):
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
    """Stream bounded projection chunks into the final BF16 HND carrier."""

    name = "chunked_triton_sparse_qkv"
    qk_format = "bf16_hnd"
    streamed_qkv = True

    def __init__(
        self,
        required=False,
        chunk_rows=4096,
        v_scale_group_size=None,
    ):
        from .bf16 import ChunkedBF16QKVProjector

        self.required = bool(required)
        self.chunk_rows = int(chunk_rows)
        if v_scale_group_size is not None:
            raise ValueError(
                'BF16 Triton does not use an INT8 V scale group'
            )
        self._implementation = ChunkedBF16QKVProjector(
            chunk_rows=self.chunk_rows
        )

    @property
    def v_format(self):
        return "bf16_hnd"

    @property
    def installation_signature(self):
        return (
            self.name,
            self.qk_format,
            self.v_format,
            bool(self.required),
            self._implementation.installation_signature,
        )

    def stream(self, module, x, rope_freqs, consume_chunk):
        actual = describe_linear(module.qkv_proj)
        if not _bf16_streamable(actual):
            return _unsupported(
                self.required,
                "QKV format is %s" % actual.label,
            )
        try:
            self._implementation.stream(
                module,
                x,
                rope_freqs,
                consume_chunk,
            )
            return True
        except Exception as exc:
            if is_fused_weight_format_error(exc):
                return _unsupported(self.required, str(exc))
            raise

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
        if not _bf16_streamable(actual):
            return _unsupported(
                self.required,
                "QKV format is %s" % actual.label,
            )
        try:
            projected = self._implementation.try_project(
                module,
                x,
                rope_freqs,
                layer_index=layer_index,
                transformer_options=transformer_options,
            )
            if projected is None:
                return _unsupported(
                    self.required,
                    "bounded BF16 QKV projection is unavailable at runtime",
                )
            return projected
        except Exception as exc:
            if is_fused_weight_format_error(exc):
                return _unsupported(self.required, str(exc))
            raise
