"""Format-guarded chunked QKV projectors for sparse attention backends."""

from __future__ import annotations

from .formats import (
    describe_linear,
    is_fused_weight_format_error,
)
from .streamed import PROJECTION_NATIVE


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
        projection_mode="native",
        v_mode=None,
    ):
        from ..attention.sparse.sparse_sage_streamed import (
            StreamedSparseSageQKVProjector as Implementation,
        )
        from ..plan import V_MEMORY_RETAIN

        self.required = bool(required)
        self.chunk_rows = int(chunk_rows)
        self.query_chunk_rows = int(query_chunk_rows)
        self.projection_mode = projection_mode
        self.requested_v_mode = V_MEMORY_RETAIN if v_mode is None else v_mode
        self._implementation = Implementation(
            spec,
            project_chunk_rows=self.chunk_rows,
            query_chunk_rows=self.query_chunk_rows,
            projection_mode=self.projection_mode,
            v_mode=self.requested_v_mode,
        )
        # Mirror the effective mode the implementation settled on, not the
        # request: status reads v_mode off whichever projector is installed.
        self.v_mode = self._implementation.v_mode

    @property
    def installation_signature(self):
        return (
            self.name,
            self.qk_format,
            bool(self.required),
            self.projection_mode,
            self.v_mode,
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
    """Provide either full BF16 carriers or low-VRAM streamed Q to Triton."""

    name = "chunked_triton_sparse_qkv"
    qk_format = "bf16_hnd"
    streamed_qkv = False

    def __init__(
        self,
        required=False,
        chunk_rows=4096,
        v_scale_group_size=None,
        force_weights_int8=False,
        stream_native_bf16=False,
        projection_mode=None,
    ):
        from .bf16 import ChunkedBF16QKVProjector

        self.required = bool(required)
        self.chunk_rows = int(chunk_rows)
        self._legacy_force_weights_int8 = bool(force_weights_int8)
        self._legacy_stream_native_bf16 = bool(stream_native_bf16)
        self.force_weights_int8 = bool(force_weights_int8)
        self.stream_native_bf16 = bool(stream_native_bf16)
        if self.force_weights_int8 and self.stream_native_bf16:
            raise ValueError(
                "Triton QKV cannot force INT8 and preserve native BF16 weights"
            )
        if projection_mode is not None and (
            self.force_weights_int8 or self.stream_native_bf16
        ):
            raise ValueError(
                "projection_mode cannot be combined with legacy Triton stream flags"
            )
        if self.force_weights_int8:
            projection_mode = "force_int8"
        elif self.stream_native_bf16:
            projection_mode = "native"
        self.projection_mode = projection_mode
        self.force_weights_int8 = (
            self.force_weights_int8 or self.projection_mode == "force_int8"
        )
        self.stream_native_bf16 = (
            self.stream_native_bf16 or self.projection_mode == "native"
        )
        self.streamed_q = self.projection_mode is not None
        if v_scale_group_size is not None:
            raise ValueError(
                'BF16 Triton does not use an INT8 V scale group'
            )
        self._implementation = ChunkedBF16QKVProjector(
            chunk_rows=self.chunk_rows,
            force_weights_int8=self.force_weights_int8,
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
            self.projection_mode,
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
            return self._implementation.stream(
                module,
                x,
                rope_freqs,
                consume_chunk,
            )
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
        if self._legacy_force_weights_int8:
            if not actual.plain_float:
                return _unsupported(
                    self.required,
                    "runtime ConvRot INT8 streaming requires floating QKV weights",
                )
            from ..attention.sparse.triton_bf16_streamed import (
                run_streamed_triton_qkv,
            )

            return run_streamed_triton_qkv(
                module,
                x,
                rope_freqs,
                layer_index=layer_index,
                chunk_rows=self.chunk_rows,
            )
        if self._legacy_stream_native_bf16:
            dtype = str(getattr(actual, "logical_dtype", "")).lower()
            if not actual.plain_float or not (
                "bfloat16" in dtype or "bf16" in dtype
            ):
                return _unsupported(
                    self.required,
                    "native BF16 query streaming requires BF16 QKV weights",
                )
            from ..attention.sparse.triton_bf16_streamed import (
                run_streamed_bf16_triton_qkv,
            )

            return run_streamed_bf16_triton_qkv(
                module,
                x,
                rope_freqs,
                layer_index=layer_index,
                chunk_rows=self.chunk_rows,
            )
        if self.projection_mode is not None:
            from ..attention.sparse.triton_bf16_streamed import (
                run_streamed_source_triton_qkv,
            )

            return run_streamed_source_triton_qkv(
                module,
                x,
                rope_freqs,
                layer_index=layer_index,
                chunk_rows=self.chunk_rows,
                projection_mode=self.projection_mode,
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


class FrostStreamedQKVProjector:
    '''Stream bounded BF16 Q against global sequence-major BF16 K/V.'''

    name = 'streamed_frost_bf16_qkv'
    qk_format = 'bf16_sequence_major'
    streamed_q = True
    streamed_qkv = False

    def __init__(
        self,
        spec,
        *,
        required=False,
        chunk_rows=4096,
        projection_mode=PROJECTION_NATIVE,
    ):
        self.spec = spec
        self.required = bool(required)
        self.chunk_rows = int(chunk_rows)
        self.projection_mode = projection_mode

    @property
    def installation_signature(self):
        return (
            self.name,
            self.qk_format,
            bool(self.required),
            self.chunk_rows,
            self.projection_mode,
            self.spec.signature,
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
        del transformer_options
        actual = describe_linear(module.qkv_proj)
        if not _bf16_streamable(actual):
            return _unsupported(
                self.required,
                'QKV format is %s' % actual.label,
            )
        from ..attention.sparse.frost_bf16_streamed import (
            _assemble_streamed_frost_qkv,
        )

        try:
            return _assemble_streamed_frost_qkv(
                module,
                x,
                rope_freqs,
                spec=self.spec,
                layer_index=layer_index,
                chunk_rows=self.chunk_rows,
                projection_mode=self.projection_mode,
            )
        except Exception as error:
            if is_fused_weight_format_error(error):
                return _unsupported(self.required, str(error))
            raise
