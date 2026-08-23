'''Block-sparse INT8 attention on Comfy Kitchen, without Sparge.

Same fixed-density route as the Sparse Sage backend, executed by Kitchen's
pure-INT8 attention kernel over a routed subset of the KV tiles. Nothing here
imports spas_sage_attn, which is the point: this is the path Sparge is meant
to be retired in favour of.

Two things differ from the Sparse Sage backend and both are deliberate.

The KV tile is 128, not 64. 128x64 was Sparge's SM89 ABI rather than an H3
decision, and on the production H3 shape the wider tile measured ~21% faster
against Sparse Sage at every density from 100% down to 10%, with no loss in
selection quality. The router already takes its tile size as a parameter.

Q/K/V are quantized by Kitchen rather than by the Triton per-tile quantizer.
Kitchen's carrier is four transforms -- K-anchor detection, anchor subtraction,
a randomized Walsh-Hadamard rotation shared by Q and K, then per-thread abs-max
scaling -- and that is what makes it more accurate than Sparge's FP8 V path.
The chunked Kitchen QKV producer emits this carrier directly and retains only
tile-mean Q/K summaries for routing. The Sparge fused projector remains
incompatible because it emits a different per-tile carrier.
'''

from dataclasses import dataclass

from ...runtime.context import get_runtime_snapshot
from .config import HybridSparseConfig, MODE_SAGE128_FUSED_QKV, resolve_video_budget
from .router import SparseRouterError, SparseTileRouter
from ...mlp_sharing.route import router_kwargs as _route_kwargs
from ...kitchen_qkv import PreparedChunkedKitchenQKV


class SparseKitchenError(RuntimeError):
    pass


Q_TILE = 128
# Measured on SM89: the 128-wide KV tile beats the 64-wide one by 17-19% in
# the sparse kernel and leaves selection quality unchanged at matched density.
KV_TILE = 128
HEAD_DIM = 128


def _kitchen():
    """The vendored kernels, or an installed comfy-kitchen that carries them.

    Vendored first. ComfyUI pins comfy-kitchen==0.2.31, which has no
    block-sparse attention, so relying on the installed package means this
    backend is unavailable for everyone -- the same way the chunked QKV
    producer was integrated here for months without running once.
    """
    from ...native import int8_attention as vendored

    if vendored.int8_attention_is_available():
        return vendored

    reason = vendored.loader.unavailable_reason() or 'unknown'
    try:
        import comfy_kitchen
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SparseKitchenError(
            'Kitchen sparse attention needs the vendored library or a '
            'comfy-kitchen carrying the sparse kernel. Vendored: %s' % reason
        ) from exc
    if not hasattr(comfy_kitchen, 'block_sparse_int8_attention_from_prequantized'):
        raise SparseKitchenError(
            'the installed comfy-kitchen has no block-sparse INT8 attention, '
            'and the vendored library is unavailable: %s' % reason
        )
    return comfy_kitchen


def preflight_sparse_kitchen(*, cuda_available, capability_getter, kitchen=None):
    '''Resolve the Kitchen sparse kernel, or say exactly why it is unavailable.'''
    if not cuda_available():
        raise SparseKitchenError('Kitchen sparse attention requires CUDA')
    module = _kitchen() if kitchen is None else kitchen
    capability = capability_getter()
    if capability is None:
        raise SparseKitchenError('Kitchen sparse GPU capability is unavailable')
    capability = tuple(int(value) for value in capability)
    if len(capability) != 2 or capability < (8, 0):
        raise SparseKitchenError(
            'Kitchen sparse attention requires NVIDIA compute capability 8.0 '
            'or newer; found %d.%d' % capability
        )
    if not module.int8_attention_is_available():
        raise SparseKitchenError(
            'the comfy-kitchen CUDA extension is not available on this device'
        )
    return module


def snapshot_for(transformer_options, sequence):
    '''The runtime layout this attention call belongs to.

    Duplicated from the Sparse Sage backend rather than shared, because that
    module imports spas_sage_attn at module scope and this path must not.
    Both collapse into one when Sparge is removed.
    '''
    snapshot = get_runtime_snapshot(transformer_options)
    if snapshot is None:
        raise SparseKitchenError(
            'Kitchen sparse attention requires an H3 runtime snapshot'
        )
    if not snapshot.valid_layout:
        raise SparseKitchenError(
            'Kitchen sparse attention requires a valid packed layout: %s'
            % (snapshot.error or 'layout unavailable')
        )
    if int(snapshot.layout.seq_len) != int(sequence):
        raise SparseKitchenError(
            'runtime layout sequence %d does not match attention sequence %d'
            % (snapshot.layout.seq_len, sequence)
        )
    return snapshot


def route_metadata(mask_metadata, layer_index, heads):
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


@dataclass
class PreparedSparseKitchen:
    quantized: object
    route: object
    original_head_dim: int
    layer_index: int
    metadata: dict


class SparseKitchenExecutor:
    '''Quantize with Kitchen, then attend over the routed KV tiles.'''

    def __init__(self, kitchen, *, kv_tile=KV_TILE, allow_cpu_for_tests=False):
        self.kitchen = kitchen
        self.kv_tile = int(kv_tile)
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)

    def prepare(self, q, k, v, lut, valid_block_num, *, layer_index, metadata):
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise SparseKitchenError('Kitchen sparse attention expects HND rank-4 Q/K/V')
        if q.shape[-1] != HEAD_DIM:
            raise SparseKitchenError(
                'Kitchen sparse attention is built for head_dim %d, got %d'
                % (HEAD_DIM, q.shape[-1])
            )
        if not self.allow_cpu_for_tests and not q.is_cuda:
            raise SparseKitchenError('Kitchen sparse attention requires CUDA tensors')

        quantized = self.kitchen.prequantize_int8_attention(
            q, k, v, cta_k=self.kv_tile
        )
        # The router emits Sparge's delta encoding. Declaring it rather than
        # converting keeps this a zero-conversion path: the route knows how to
        # reach whatever encoding the compiled kernel walks.
        return self.prepare_projected(
            quantized,
            lut,
            valid_block_num,
            layer_index=layer_index,
            metadata=metadata,
        )

    def prepare_projected(
        self,
        quantized,
        lut,
        valid_block_num,
        *,
        layer_index,
        metadata,
    ):
        if quantized.q.ndim != 4 or quantized.k.ndim != 4:
            raise SparseKitchenError(
                'Kitchen sparse attention received invalid Q/K carriers'
            )
        if quantized.q.shape != quantized.k.shape:
            raise SparseKitchenError(
                'Kitchen sparse attention requires equal Q/K carrier shapes'
            )
        if int(quantized.original_head_dim) != HEAD_DIM:
            raise SparseKitchenError(
                'Kitchen sparse attention is built for head_dim %d, got %d'
                % (HEAD_DIM, quantized.original_head_dim)
            )
        if int(quantized.cta_k) != self.kv_tile:
            raise SparseKitchenError(
                'Kitchen QKV carrier uses KV tile %d, expected %d'
                % (quantized.cta_k, self.kv_tile)
            )
        route = self.kitchen.BlockSparseRoute(
            indices=lut,
            counts=valid_block_num,
            q_tile=Q_TILE,
            kv_tile=self.kv_tile,
            encoding='delta',
        )
        return PreparedSparseKitchen(
            quantized=quantized,
            route=route,
            original_head_dim=int(quantized.original_head_dim),
            layer_index=int(layer_index),
            metadata=metadata,
        )

    def execute(self, prepared):
        return self.kitchen.block_sparse_int8_attention_from_prequantized(
            prepared.quantized, prepared.route
        )


class SparseKitchenBackend:
    name = 'sparse_kitchen_int8'
    requires_runtime_context = True
    approximate = True

    def __init__(
        self,
        config=None,
        *,
        kitchen=None,
        router=None,
        executor=None,
        projector=None,
        allow_cpu_for_tests=False,
    ):
        self.config = config or HybridSparseConfig()
        if not isinstance(self.config, HybridSparseConfig):
            raise TypeError('config must be HybridSparseConfig')
        if self.config.mode == MODE_SAGE128_FUSED_QKV and projector is None:
            raise SparseKitchenError(
                'Kitchen sparse attention cannot consume the fused QKV '
                'mode without the chunked Kitchen producer'
            )
        module = _kitchen() if kitchen is None else kitchen
        self.executor = executor or SparseKitchenExecutor(
            module,
            allow_cpu_for_tests=allow_cpu_for_tests,
        )
        self.router = router or SparseTileRouter(
            self.config,
            q_tile=Q_TILE,
            kv_tile=self.executor.kv_tile,
        )
        self.projector = projector
        if (self.router.q_tile, self.router.kv_tile) != (
            Q_TILE,
            self.executor.kv_tile,
        ):
            raise SparseKitchenError(
                'router geometry %dQ x %dKV does not match the kernel %dQ x %dKV'
                % (
                    self.router.q_tile,
                    self.router.kv_tile,
                    Q_TILE,
                    self.executor.kv_tile,
                )
            )

    @property
    def installation_signature(self):
        return (
            self.name,
            self.config.signature,
            (type(self.router).__module__, type(self.router).__qualname__),
            int(self.executor.kv_tile),
            str(getattr(self.executor.kitchen, '__version__', 'unknown')),
            (
                None
                if self.projector is None
                else self.projector.installation_signature
            ),
        )

    def _route(self, q, k, *, layer_index, transformer_options):
        snapshot = snapshot_for(transformer_options, q.shape[-2])
        video_budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
            layer_index,
        )
        try:
            return self.router.build_lut(
                q,
                k,
                snapshot.layout,
                video_budget,
                **_route_kwargs(transformer_options, layer_index),
            )
        except SparseRouterError as exc:
            raise SparseKitchenError('sparse routing failed: %s' % exc) from exc

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        lut, valid_block_num, mask_metadata = self._route(
            q,
            k,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )
        return self.executor.prepare(
            q,
            k,
            v,
            lut,
            valid_block_num,
            layer_index=layer_index,
            metadata=route_metadata(mask_metadata, layer_index, q.shape[1]),
        )

    def prepare_projected(
        self,
        projected,
        *,
        layer_index,
        transformer_options,
    ):
        if self.projector is None:
            raise SparseKitchenError(
                'Kitchen sparse attention has no chunked QKV producer'
            )
        if not isinstance(projected, PreparedChunkedKitchenQKV):
            raise SparseKitchenError(
                'Kitchen sparse attention received an invalid projected carrier'
            )
        if projected.q_summary is None or projected.k_summary is None:
            raise SparseKitchenError(
                'Kitchen sparse QKV carrier has no routing summaries'
            )
        sequence = int(projected.carrier.q.shape[-2])
        snapshot = snapshot_for(transformer_options, sequence)
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
            raise SparseKitchenError('sparse routing failed: %s' % exc) from exc
        return self.executor.prepare_projected(
            projected.carrier,
            lut,
            valid_block_num,
            layer_index=layer_index,
            metadata=route_metadata(
                mask_metadata,
                layer_index,
                projected.q_summary.shape[1],
            ),
        )

    def execute(self, prepared):
        return self.executor.execute(prepared)

    def as_status(self):
        return {
            'mode': self.config.mode,
            'video_budget': float(self.config.video_budget),
            'denser_early_late_steps': bool(self.config.denser_early_late_steps),
            'density_mode': self.config.density_mode,
            'sparse_architecture': 'comfy_kitchen_int8',
            'sparse_q_tile': Q_TILE,
            'sparse_kv_tile': int(self.executor.kv_tile),
            'sparse_v_format': 'int8',
            'route_encoding': 'delta',
            'approximate': True,
            'fused_qkv': self.projector is not None,
            'qkv_projector': (
                None if self.projector is None else self.projector.name
            ),
            'smooth_k': False,
        }
