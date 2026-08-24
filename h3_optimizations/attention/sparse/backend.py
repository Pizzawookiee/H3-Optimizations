'''Prepared fixed-density Sparse Sage backend for MiniMax H3.'''

from dataclasses import dataclass

from ...runtime.context import get_runtime_snapshot
from .config import (
    HybridSparseConfig,
    MODE_SAGE128_FUSED_QKV,
    resolve_video_budget,
)
from .chunked_qkv import ChunkedSparseQKVProjector
from .fused_qkv import sparse_fused_qkv_contract_mismatch
from .router import SparseRouterError, SparseTileRouter
from .sparse_sage import (
    SparseSageError,
    SparseSageExecutor,
    load_sparse_sage_spec,
)
from ...mlp_sharing.route import router_kwargs as _route_kwargs


@dataclass
class PreparedHybrid:
    sparse: object


class HybridSparseBackend:
    name = 'sparse_sage'
    requires_registered_sage = True
    requires_runtime_context = True
    approximate = True

    def __init__(
        self,
        config=None,
        *,
        kernel_spec=None,
        router=None,
        allow_cpu_for_tests=False,
        qk_quantizer=None,
        v_preparer=None,
        low_level_selector=None,
        projector=None,
    ):
        self.config = config or HybridSparseConfig()
        if not isinstance(self.config, HybridSparseConfig):
            raise TypeError('config must be HybridSparseConfig')
        kernel_spec = (
            kernel_spec
            if kernel_spec is not None
            else load_sparse_sage_spec()
        )
        self.projector = projector
        if (
            self.config.mode == MODE_SAGE128_FUSED_QKV
            and self.projector is None
        ):
            self.projector = ChunkedSparseQKVProjector(kernel_spec)
        self.executor = SparseSageExecutor(
            kernel_spec,
            allow_cpu_for_tests=allow_cpu_for_tests,
            qk_quantizer=qk_quantizer,
            v_preparer=v_preparer,
            low_level_selector=low_level_selector,
        )
        self.router = (
            router
            if router is not None
            else SparseTileRouter(self.config, spec=self.executor.spec)
        )
        if (self.router.q_tile, self.router.kv_tile) != (
            self.executor.spec.q_tile,
            self.executor.spec.kv_tile,
        ):
            raise SparseSageError(
                'router geometry %dQ x %dKV does not match %s ABI %dQ x %dKV'
                % (
                    self.router.q_tile,
                    self.router.kv_tile,
                    self.executor.spec.architecture,
                    self.executor.spec.q_tile,
                    self.executor.spec.kv_tile,
                )
            )
        mismatch = sparse_fused_qkv_contract_mismatch(self.executor.spec)
        if self.config.mode == MODE_SAGE128_FUSED_QKV and mismatch is not None:
            raise SparseSageError(
                'fused QKV does not match the Sparse Sage carrier contract: %s'
                % mismatch
            )

    @staticmethod
    def _callable_signature(value):
        if value is None:
            return None
        function = getattr(value, '__func__', value)
        return (
            getattr(function, '__module__', type(function).__module__),
            getattr(function, '__qualname__', type(function).__qualname__),
            id(function),
        )

    @property
    def installation_signature(self):
        projector = self.projector
        spec = self.executor.spec
        return (
            self.name,
            self.config.signature,
            (type(self.router).__module__, type(self.router).__qualname__),
            spec.signature,
            self._callable_signature(self.executor.qk_quantizer),
            self._callable_signature(self.executor.v_preparer),
            self._callable_signature(self.executor.low_level_selector),
            None
            if projector is None
            else (
                type(projector).__module__,
                type(projector).__qualname__,
                getattr(projector, 'name', None),
                getattr(projector, 'installation_signature', None),
            ),
        )

    @staticmethod
    def _snapshot(transformer_options, sequence):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None:
            raise SparseSageError(
                'Sparse Attention requires an H3 runtime snapshot'
            )
        if not snapshot.valid_layout:
            raise SparseSageError(
                'Sparse Attention requires a valid packed layout: %s'
                % (snapshot.error or 'layout unavailable')
            )
        if int(snapshot.layout.seq_len) != int(sequence):
            raise SparseSageError(
                'runtime layout sequence %d does not match attention sequence %d'
                % (snapshot.layout.seq_len, sequence)
            )
        return snapshot

    @staticmethod
    def _metadata(mask_metadata, layer_index, heads):
        metadata = mask_metadata.as_dict()
        metadata.update(
            {
                'layer': int(layer_index),
                'sparse_sage_heads': int(heads),
                'total_q_video_tiles': (
                    int(mask_metadata.pure_video_q_tiles) * int(heads)
                ),
            }
        )
        return metadata

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        snapshot = self._snapshot(transformer_options, q.shape[-2])
        video_budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
            layer_index,
        )
        try:
            lut, valid_block_num, mask_metadata = self.router.build_lut(
                q,
                k,
                snapshot.layout,
                video_budget,
                **_route_kwargs(transformer_options, layer_index),
            )
        except SparseRouterError as exc:
            raise SparseSageError('sparse routing failed: %s' % exc) from exc
        sparse = self.executor.prepare(
            q,
            k,
            v,
            lut,
            valid_block_num,
            layer_index=layer_index,
            metadata=self._metadata(
                mask_metadata,
                layer_index,
                q.shape[1],
            ),
        )
        return PreparedHybrid(sparse=sparse)

    def prepare_projected(
        self,
        projected,
        *,
        layer_index,
        transformer_options,
    ):
        # The ConvRot Sparse Sage projector uses a separate lifetime contract:
        # it keeps global K/V, routing summaries, and the original input but no
        # full Q carrier.  Detect it here without changing provider resolution.
        from .sparse_sage_streamed import (
            StreamedSparseSageQKV,
            prepare_streamed_sparse_sage,
        )

        if isinstance(projected, StreamedSparseSageQKV):
            return PreparedHybrid(
                sparse=prepare_streamed_sparse_sage(
                    self,
                    projected,
                    layer_index=layer_index,
                    transformer_options=transformer_options,
                )
            )

        snapshot = self._snapshot(
            transformer_options,
            projected.sequence,
        )
        video_budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
            layer_index,
        )
        try:
            lut, valid_block_num, mask_metadata = (
                self.router.build_lut_from_summaries(
                    projected.q_summary,
                    projected.k_summary,
                    snapshot.layout,
                    video_budget,
                    **_route_kwargs(transformer_options, layer_index),
                )
            )
        except SparseRouterError as exc:
            raise SparseSageError('sparse routing failed: %s' % exc) from exc
        sparse = self.executor.prepare_projected(
            projected,
            lut,
            valid_block_num,
            metadata=self._metadata(
                mask_metadata,
                layer_index,
                projected.heads,
            ),
        )
        return PreparedHybrid(sparse=sparse)

    def execute_projected(self, module, prepared):
        '''Return a final hidden-size tensor when the projected path owns out_proj.'''
        from .sparse_sage_streamed import (
            PreparedStreamedSparseSage,
            execute_streamed_sparse_sage,
        )

        if (
            isinstance(prepared, PreparedHybrid)
            and isinstance(prepared.sparse, PreparedStreamedSparseSage)
        ):
            return execute_streamed_sparse_sage(
                module,
                self,
                prepared.sparse,
            )
        return None

    def execute(self, prepared):
        from .sparse_sage_streamed import PreparedStreamedSparseSage

        if isinstance(prepared.sparse, PreparedStreamedSparseSage):
            raise SparseSageError(
                'streamed Sparse Sage must execute through execute_projected'
            )
        return self.executor.execute(prepared.sparse)

    def as_status(self):
        return {
            'mode': self.config.mode,
            'video_budget': float(self.config.video_budget),
            'denser_early_late_steps': bool(
                self.config.denser_early_late_steps
            ),
            'density_mode': self.config.density_mode,
            'sparge_attention': self.executor.spec.version,
            'sparse_architecture': self.executor.spec.architecture,
            'sparse_q_tile': self.executor.spec.q_tile,
            'sparse_kv_tile': self.executor.spec.kv_tile,
            'sparse_v_format': self.executor.spec.v_format,
            'sparse_v_quant_bound': self.executor.spec.v_quant_bound,
            'sparse_extension_layout': self.executor.spec.extension_layout,
            'approximate': True,
            'fused_qkv': self.projector is not None,
            'qkv_projector': getattr(self.projector, 'name', None),
            'qkv_chunk_rows': getattr(self.projector, 'chunk_rows', None),
            'smooth_k': False if self.projector is not None else True,
            'streamed_q': bool(getattr(self.projector, 'streamed_q', False)),
            'query_chunk_rows': getattr(self.projector, 'query_chunk_rows', None),
        }
