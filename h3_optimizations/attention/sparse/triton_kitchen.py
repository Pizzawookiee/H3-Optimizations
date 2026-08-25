'''Triton 64x64 sparse attention over the exact Kitchen INT8 carrier.

This is deliberately a *consumer* port, not another quantizer. Q/K/V are packed by
the same Kitchen-compatible producer used by the native kernel: K-anchor detection
and subtraction, randomized Hadamard/ConvRot Q/K, per-thread Q/K scales, and the
per-channel permuted V carrier. The Triton kernel only replaces sparse traversal and
attention math.

The probability path mirrors Kitchen's pure-INT8 kernel as well: scores stay INT32,
per-thread dequant scales are applied in FP32, the online-softmax maximum is shifted
by S_U8_OFFSET, exp2 probabilities are rounded-to-nearest-even into logical UINT8,
and UINT8 P x INT8 V accumulates in INT32 before FP32 normalization. V scale is
applied once after normalization, matching the native kernel.
'''

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...kitchen_qkv import PreparedChunkedKitchenQKV
from ...mlp_sharing.route import router_kwargs as _route_kwargs
from ...runtime.context import get_runtime_snapshot
from .config import HybridSparseConfig, resolve_video_budget
from .router import SparseRouterError, SparseTileRouter

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    triton = None
    tl = None
    TRITON_AVAILABLE = False


Q_TILE = 64
KV_TILE = 64
HEAD_DIM = 128
S_U8_OFFSET = 7.9943534
LOG2E = 1.4426950408889634


class TritonKitchenError(RuntimeError):
    pass


@dataclass(frozen=True)
class TritonKitchenSpec:
    q_tile: int = Q_TILE
    kv_tile: int = KV_TILE
    head_dim: int = HEAD_DIM
    implementation: str = 'triton_kitchen_carrier_u8p_int8v_64x64'

    @property
    def signature(self):
        return (
            self.implementation,
            int(self.q_tile),
            int(self.kv_tile),
            int(self.head_dim),
        )


def preflight_triton_kitchen(
    *, cuda_available, capability_getter, triton_available=None
):
    if not cuda_available():
        raise TritonKitchenError('Kitchen-parity Triton sparse attention requires CUDA')
    available = TRITON_AVAILABLE if triton_available is None else bool(triton_available)
    if not available:
        raise TritonKitchenError('Kitchen-parity Triton sparse attention requires Triton')
    capability = capability_getter()
    if capability is None:
        raise TritonKitchenError('GPU capability is unavailable')
    capability = tuple(int(value) for value in capability)
    if len(capability) != 2 or capability[0] < 8:
        raise TritonKitchenError(
            'Kitchen-parity Triton sparse attention requires compute capability '
            '>= 8.0; got %s' % (capability,)
        )
    return TritonKitchenSpec()


def _validate_carrier(carrier):
    required = (
        'q', 'k', 'v', 'q_scale', 'k_scale', 'v_scale',
        'original_head_dim', 'input_dtype', 'attention_scale', 'cta_k',
    )
    missing = [name for name in required if not hasattr(carrier, name)]
    if missing:
        raise TritonKitchenError(
            'Kitchen carrier is missing %s' % ', '.join(missing)
        )
    if carrier.q.ndim != 4 or carrier.k.ndim != 4:
        raise TritonKitchenError('Kitchen Q/K carrier must be rank-4 HND')
    if carrier.q.shape != carrier.k.shape:
        raise TritonKitchenError('Kitchen Q/K carrier shapes differ')
    if carrier.q.dtype != torch.int8 or carrier.k.dtype != torch.int8:
        raise TritonKitchenError('Kitchen Q/K carrier must be INT8')
    if carrier.v.dtype != torch.int8:
        raise TritonKitchenError('Kitchen V carrier must be INT8')
    if int(carrier.original_head_dim) != HEAD_DIM:
        raise TritonKitchenError(
            'Kitchen-parity Triton is fixed to head_dim %d, got %d'
            % (HEAD_DIM, carrier.original_head_dim)
        )
    if int(carrier.cta_k) != KV_TILE:
        raise TritonKitchenError(
            'Kitchen-parity Triton requires a 64-wide V/K carrier, got %d'
            % carrier.cta_k
        )
    if carrier.input_dtype not in (torch.float16, torch.bfloat16):
        raise TritonKitchenError('Kitchen-parity Triton output must be FP16 or BF16')
    batch, heads, sequence, head_dim = carrier.q.shape
    if batch != 1 or head_dim != HEAD_DIM or sequence <= 0:
        raise TritonKitchenError('Kitchen Q/K carrier shape is invalid')
    padded_sequence = ((sequence + KV_TILE - 1) // KV_TILE) * KV_TILE
    if tuple(carrier.v.shape) != (batch * heads * head_dim, padded_sequence):
        raise TritonKitchenError(
            'Kitchen V carrier shape %s does not match (%d, %d)'
            % (tuple(carrier.v.shape), batch * heads * head_dim, padded_sequence)
        )
    q_scales = ((sequence + 127) // 128) * 32
    k_scales = ((sequence + KV_TILE - 1) // KV_TILE) * 4
    if tuple(carrier.q_scale.shape) != (batch, heads, q_scales):
        raise TritonKitchenError('Kitchen Q scale layout is incompatible')
    if tuple(carrier.k_scale.shape) != (batch, heads, k_scales):
        raise TritonKitchenError('Kitchen K scale layout is incompatible')
    if tuple(carrier.v_scale.shape) != (batch * heads * head_dim,):
        raise TritonKitchenError('Kitchen V scale layout is incompatible')
    return carrier


def _snapshot(transformer_options, sequence):
    snapshot = get_runtime_snapshot(transformer_options)
    if snapshot is None:
        raise TritonKitchenError(
            'Kitchen-parity Triton sparse attention requires an H3 runtime snapshot'
        )
    if not snapshot.valid_layout:
        raise TritonKitchenError(
            'Kitchen-parity Triton requires a valid packed layout: %s'
            % (snapshot.error or 'layout unavailable')
        )
    if int(snapshot.layout.seq_len) != int(sequence):
        raise TritonKitchenError(
            'runtime layout sequence %d does not match carrier sequence %d'
            % (snapshot.layout.seq_len, sequence)
        )
    return snapshot


def _route_metadata(mask_metadata, layer_index, heads):
    metadata = mask_metadata.as_dict()
    metadata.update(
        {
            'layer': int(layer_index),
            'triton_kitchen_heads': int(heads),
            'total_q_video_tiles': (
                int(mask_metadata.pure_video_q_tiles) * int(heads)
            ),
        }
    )
    return metadata


if TRITON_AVAILABLE:
    @triton.jit
    def _rni_s32(x):
        return tl.inline_asm_elementwise(
            asm='cvt.rni.s32.f32 $0, $1;',
            constraints='=r,f',
            args=[x],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _v_perm16(position):
        low = position & 15
        perm = (
            (low & 1)
            | (((low >> 3) & 1) << 1)
            | (((low >> 1) & 1) << 2)
            | (((low >> 2) & 1) << 3)
        )
        return (position & ~15) | perm

    _CONFIGS = [
        triton.Config({'BLOCK_M': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=2),
    ]

    def _autotune(configs, key):
        try:
            return triton.autotune(configs=configs, key=key, cache_results=True)
        except TypeError:  # older Triton
            return triton.autotune(configs=configs, key=key)

    @_autotune(_CONFIGS, key=['KV_BLOCKS', 'OUTPUT_BF16'])
    @triton.jit
    def _kitchen_sparse_kernel(
        Q,
        K,
        V,
        Q_SCALE,
        K_SCALE,
        V_SCALE,
        LUT,
        VALID,
        O,
        sequence,
        heads,
        padded_sequence,
        q_tiles,
        KV_BLOCKS: tl.constexpr,
        softmax_scale: tl.constexpr,
        OUTPUT_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        work = tl.program_id(0).to(tl.int64)
        bh = tl.program_id(1).to(tl.int64)
        subblocks = Q_TILE // BLOCK_M
        q_block = work // subblocks
        q_sub = work - q_block * subblocks
        if q_block >= q_tiles:
            return

        q_rows = q_block * Q_TILE + q_sub * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, HEAD_DIM)
        kv_rows = tl.arange(0, KV_TILE)
        q_mask = q_rows < sequence
        hnd_base = bh * sequence * HEAD_DIM

        q = tl.load(
            Q + hnd_base + q_rows[:, None] * HEAD_DIM + dims[None, :],
            mask=q_mask[:, None],
            other=0,
        ).to(tl.int8)
        q_scale_count = ((sequence + 127) // 128) * 32
        q_scale_index = (q_rows // 32) * 8 + (q_rows & 7)
        q_scale = tl.load(
            Q_SCALE + bh * q_scale_count + q_scale_index,
            mask=q_mask,
            other=1.0,
        ).to(tl.float32)

        m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
        d_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

        route_row = bh * q_tiles + q_block
        lut_base = LUT + route_row * KV_BLOCKS
        valid = tl.load(VALID + route_row).to(tl.int32)
        key_block = tl.zeros((), dtype=tl.int32)
        k_scale_count = KV_BLOCKS * 4

        for slot in tl.range(0, valid):
            key_block += tl.load(lut_base + slot).to(tl.int32)
            k_positions = key_block * KV_TILE + kv_rows
            k_valid = k_positions < sequence
            k = tl.load(
                K + hnd_base + k_positions[None, :] * HEAD_DIM + dims[:, None],
                mask=k_valid[None, :],
                other=0,
            ).to(tl.int8)
            score_i32 = tl.dot(q, k, out_dtype=tl.int32)

            k_scale_index = key_block * 4 + ((kv_rows & 7) >> 1)
            k_scale = tl.load(
                K_SCALE + bh * k_scale_count + k_scale_index
            ).to(tl.float32)
            logits = score_i32.to(tl.float32) * (
                q_scale[:, None]
                * k_scale[None, :]
                * softmax_scale
                * LOG2E
            )
            logits = tl.where(k_valid[None, :], logits, -float('inf'))

            tile_max = tl.max(logits, axis=1)
            tile_m = tile_max - S_U8_OFFSET
            new_m = tl.maximum(m_i, tile_m)
            o_scale = tl.math.exp2(m_i - new_m)
            tile_scale = tl.math.exp2(tile_m - new_m)

            probability = tl.math.exp2(logits - tile_m[:, None])
            probability = tl.where(k_valid[None, :], probability, 0.0)
            p_code = _rni_s32(probability)
            p_code = tl.minimum(tl.maximum(p_code, 0), 255)
            p_signed = (p_code - 128).to(tl.int8)

            v_positions = _v_perm16(k_positions)
            v = tl.load(
                V
                + (bh * HEAD_DIM + dims[None, :]) * padded_sequence
                + v_positions[:, None],
            ).to(tl.int8)
            v_sum = tl.sum(v.to(tl.int32), axis=0)
            pv_i32 = tl.dot(p_signed, v, out_dtype=tl.int32)
            pv_i32 += 128 * v_sum[None, :]

            acc = acc * o_scale[:, None]
            acc += pv_i32.to(tl.float32) * tile_scale[:, None]
            d_i = d_i * o_scale
            d_i += tl.sum(p_code, axis=1).to(tl.float32) * tile_scale
            m_i = new_m

        v_scale = tl.load(
            V_SCALE + bh * HEAD_DIM + dims
        ).to(tl.float32)
        output = (acc / d_i[:, None]) * v_scale[None, :]
        tl.store(
            O + hnd_base + q_rows[:, None] * HEAD_DIM + dims[None, :],
            output.to(O.type.element_ty),
            mask=q_mask[:, None],
        )


def _launch(carrier, lut, valid):
    sequence = int(carrier.q.shape[2])
    heads = int(carrier.q.shape[1])
    q_tiles = (sequence + Q_TILE - 1) // Q_TILE
    kv_blocks = (sequence + KV_TILE - 1) // KV_TILE
    padded = ((sequence + KV_TILE - 1) // KV_TILE) * KV_TILE
    output = torch.empty(
        (1, heads, sequence, HEAD_DIM),
        dtype=carrier.input_dtype,
        device=carrier.q.device,
    )

    def grid(meta):
        return (q_tiles * (Q_TILE // meta['BLOCK_M']), heads)

    _kitchen_sparse_kernel[grid](
        carrier.q,
        carrier.k,
        carrier.v,
        carrier.q_scale,
        carrier.k_scale,
        carrier.v_scale,
        lut,
        valid,
        output,
        sequence,
        heads,
        padded,
        q_tiles,
        KV_BLOCKS=kv_blocks,
        softmax_scale=float(carrier.attention_scale),
        OUTPUT_BF16=carrier.input_dtype == torch.bfloat16,
    )
    return output


@dataclass
class PreparedTritonKitchen:
    carrier: object
    lut: torch.Tensor
    valid: torch.Tensor
    layer_index: int
    metadata: dict


class TritonKitchenBackend:
    name = 'triton_sparse_int8'
    requires_runtime_context = True
    approximate = True

    def __init__(self, config=None, *, router=None, projector=None, spec=None):
        self.config = config or HybridSparseConfig()
        if not isinstance(self.config, HybridSparseConfig):
            raise TypeError('config must be HybridSparseConfig')
        self.spec = spec or TritonKitchenSpec()
        self.projector = projector
        self.router = router or SparseTileRouter(
            self.config, q_tile=Q_TILE, kv_tile=KV_TILE
        )
        if (self.router.q_tile, self.router.kv_tile) != (Q_TILE, KV_TILE):
            raise TritonKitchenError('router geometry must be 64Q x 64KV')

    @property
    def installation_signature(self):
        return (
            self.name,
            self.config.signature,
            self.spec.signature,
            None if self.projector is None else self.projector.installation_signature,
        )

    def _route(self, q, k, *, layer_index, transformer_options):
        snapshot = _snapshot(transformer_options, q.shape[-2])
        budget = resolve_video_budget(
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
                budget,
                **_route_kwargs(transformer_options, layer_index),
            )
        except SparseRouterError as exc:
            raise TritonKitchenError('sparse routing failed: %s' % exc) from exc

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        if not TRITON_AVAILABLE:
            raise TritonKitchenError('Kitchen-parity Triton requires Triton')
        if not q.is_cuda:
            raise TritonKitchenError('Kitchen-parity Triton requires CUDA tensors')
        from ...native import carrier_selftest
        from ...native import int8_attention as native

        if not carrier_selftest.check(q.device):
            raise TritonKitchenError('Kitchen INT8 carrier self-test failed')
        lut, valid, mask_metadata = self._route(
            q, k, layer_index=layer_index, transformer_options=transformer_options
        )
        carrier = native.prequantize_int8_attention(q, k, v, cta_k=KV_TILE)
        return PreparedTritonKitchen(
            carrier=_validate_carrier(carrier),
            lut=lut,
            valid=valid,
            layer_index=int(layer_index),
            metadata=_route_metadata(mask_metadata, layer_index, q.shape[1]),
        )

    def prepare_projected(
        self, projected, *, layer_index, transformer_options
    ):
        if not isinstance(projected, PreparedChunkedKitchenQKV):
            raise TritonKitchenError(
                'Kitchen-parity Triton requires a chunked Kitchen QKV carrier'
            )
        carrier = _validate_carrier(projected.carrier)
        sequence = int(carrier.q.shape[2])
        snapshot = _snapshot(transformer_options, sequence)
        budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
            layer_index,
        )
        if projected.q_summary is None or projected.k_summary is None:
            raise TritonKitchenError('projected Kitchen carrier has no routing summaries')
        try:
            lut, valid, mask_metadata = self.router.build_lut_from_summaries(
                projected.q_summary,
                projected.k_summary,
                snapshot.layout,
                budget,
                **_route_kwargs(transformer_options, layer_index),
            )
        except SparseRouterError as exc:
            raise TritonKitchenError('sparse routing failed: %s' % exc) from exc
        return PreparedTritonKitchen(
            carrier=carrier,
            lut=lut,
            valid=valid,
            layer_index=int(layer_index),
            metadata=_route_metadata(
                mask_metadata, layer_index, carrier.q.shape[1]
            ),
        )

    def execute(self, prepared):
        if not isinstance(prepared, PreparedTritonKitchen):
            raise TritonKitchenError('invalid Kitchen-parity Triton payload')
        try:
            return _launch(prepared.carrier, prepared.lut, prepared.valid)
        except Exception as exc:
            raise TritonKitchenError(
                'Kitchen-parity Triton kernel failed: layer=%d sequence=%d heads=%d'
                % (
                    prepared.layer_index,
                    prepared.carrier.q.shape[2],
                    prepared.carrier.q.shape[1],
                )
            ) from exc

    def as_status(self):
        return {
            'mode': self.name,
            'video_budget': float(self.config.video_budget),
            'denser_early_late_steps': bool(self.config.denser_early_late_steps),
            'density_mode': self.config.density_mode,
            'sparse_q_tile': Q_TILE,
            'sparse_kv_tile': KV_TILE,
            'qkv_carrier': 'kitchen_int8_exact',
            'qk_quantization': 'kitchen_anchor_convrot_per_thread',
            'v_quantization': 'kitchen_per_channel_permuted_int8',
            'probability_value_path': 'kitchen_u8_x_int8_int32',
            'route_format': 'delta',
            'chunked_qkv': self.projector is not None,
            'approximate': True,
        }
