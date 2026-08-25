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
    """Produce the exact Kitchen carrier consumed by the Triton fallback.

    The old fallback had its own coarse block-INT8 carrier. 64x64 numerical
    parity requires that projection/quantization stop being backend-specific,
    so this compatibility wrapper keeps the existing provider ID and public
    projector name while delegating to the Kitchen producer.
    """

    name = "chunked_triton_sparse_qkv"
    qk_format = "kitchen_per_thread_int8"

    def __init__(
        self,
        required=False,
        chunk_rows=4096,
        v_scale_group_size=None,
    ):
        from ..attention.sparse.triton_qkv import normalize_v_scale_group_size
        from ..kitchen_qkv import ChunkedKitchenQKVProjector

        self.required = bool(required)
        self.chunk_rows = int(chunk_rows)
        requested_group = normalize_v_scale_group_size(v_scale_group_size)
        if requested_group != 1:
            raise ValueError(
                'Kitchen-parity Triton uses Kitchen per-channel V scaling; '
                'H3_TRITON_V_SCALE_GROUP must be 1'
            )
        self.v_scale_group_size = 1
        self._implementation = ChunkedKitchenQKVProjector(
            chunk_rows=self.chunk_rows,
            routing_summaries=True,
            q_tile=64,
            kv_tile=64,
            strided_qk_input=True,
        )

    @property
    def v_format(self):
        return "kitchen_per_channel_permuted_int8"

    @property
    def installation_signature(self):
        return (
            self.name,
            self.qk_format,
            self.v_format,
            self.v_scale_group_size,
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
                    "Kitchen INT8 producer is unavailable at runtime",
                )
            return projected
        except Exception as exc:
            if is_fused_weight_format_error(exc):
                return _unsupported(self.required, str(exc))
            raise
