'''Optimized INT8 Triton fallback for fixed-density H3 sparse attention.'''

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...runtime.context import get_runtime_snapshot
from .config import HybridSparseConfig, resolve_video_budget
from .router import KV_TILE, Q_TILE, SparseRouterError, SparseTileRouter
from .triton_qkv import (
    TritonSparseQKVError,
    normalize_v_scale_group_size,
    pack_float_qkv,
    validate_prepared_triton_sparse_qkv,
)

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


class TritonSparseError(RuntimeError):
    pass


@dataclass(frozen=True)
class TritonSparseSpec:
    q_tile: int = Q_TILE
    kv_tile: int = KV_TILE
    head_dim: int = 128
    v_scale_group_size: int = 1
    implementation: str = 'triton_int8_qk_u8p_int8v'

    def __post_init__(self):
        object.__setattr__(
            self,
            'v_scale_group_size',
            normalize_v_scale_group_size(self.v_scale_group_size),
        )

    @property
    def signature(self):
        return (
            self.implementation,
            int(self.q_tile),
            int(self.kv_tile),
            int(self.head_dim),
            int(self.v_scale_group_size),
        )

    def validate_lut(self, lut, valid, *, batch, heads, sequence):
        expected = (
            int(batch),
            int(heads),
            (int(sequence) + self.q_tile - 1) // self.q_tile,
            (int(sequence) + self.kv_tile - 1) // self.kv_tile,
        )
        if tuple(lut.shape) != expected:
            raise TritonSparseError(
                'Triton sparse LUT shape %s does not match %s'
                % (tuple(lut.shape), expected)
            )
        if tuple(valid.shape) != expected[:-1]:
            raise TritonSparseError(
                'Triton sparse valid-count shape %s does not match %s'
                % (tuple(valid.shape), expected[:-1])
            )
        if lut.dtype != torch.int32 or valid.dtype != torch.int32:
            raise TritonSparseError(
                'Triton sparse LUT and valid counts must be int32'
            )
        if not lut.is_contiguous() or not valid.is_contiguous():
            raise TritonSparseError(
                'Triton sparse LUT and valid counts must be contiguous'
            )
        if lut.device != valid.device:
            raise TritonSparseError(
                'Triton sparse LUT and valid counts devices differ'
            )


def preflight_triton_sparse(
    *,
    cuda_available,
    capability_getter,
    triton_available=None,
    v_scale_group_size=None,
):
    if not cuda_available():
        raise TritonSparseError('INT8 Triton sparse attention requires CUDA')
    available = TRITON_AVAILABLE if triton_available is None else bool(triton_available)
    if not available:
        raise TritonSparseError('INT8 Triton sparse attention requires Triton')
    capability = capability_getter()
    if capability is None:
        raise TritonSparseError('INT8 Triton sparse GPU capability is unavailable')
    capability = tuple(int(value) for value in capability)
    if len(capability) != 2 or capability[0] < 8:
        raise TritonSparseError(
            'INT8 Triton sparse attention requires NVIDIA compute capability '
            '>= 8.0; got %s' % (capability,)
        )
    try:
        group = normalize_v_scale_group_size(v_scale_group_size)
    except TritonSparseQKVError as exc:
        raise TritonSparseError(str(exc)) from exc
    return TritonSparseSpec(v_scale_group_size=group)


def _route_plan(metadata, *, q_tiles, kv_tiles):
    sparse_q_tiles = int(metadata.get('sparse_q_tiles', 0))
    dense_q_tiles = int(metadata.get('dense_q_tiles', q_tiles - sparse_q_tiles))
    if dense_q_tiles < 0 or sparse_q_tiles < 0 or dense_q_tiles + sparse_q_tiles != q_tiles:
        raise TritonSparseError(
            'INT8 Triton sparse route metadata has invalid dense/sparse Q counts'
        )
    if not sparse_q_tiles:
        return dense_q_tiles, sparse_q_tiles, 0
    try:
        pure_video_kv_tiles = int(metadata['pure_video_kv_tiles'])
        retained_video_kv_tiles = int(metadata['retained_video_kv_tiles'])
    except (KeyError, TypeError, ValueError) as exc:
        raise TritonSparseError(
            'INT8 Triton sparse route metadata is missing fixed-density KV counts'
        ) from exc
    context = kv_tiles - pure_video_kv_tiles
    selected = context + retained_video_kv_tiles
    if not 0 < selected <= kv_tiles:
        raise TritonSparseError('INT8 Triton sparse fixed KV count is invalid')
    return dense_q_tiles, sparse_q_tiles, selected


def _compact_absolute_indices(lut, *, dense_q_tiles, sparse_q_tiles, selected):
    if not sparse_q_tiles:
        return torch.empty(
            (lut.shape[0], lut.shape[1], 0, 0),
            dtype=torch.int32,
            device=lut.device,
        )
    sparse_delta = lut[
        ...,
        dense_q_tiles:dense_q_tiles + sparse_q_tiles,
        :selected,
    ]
    return torch.cumsum(sparse_delta, dim=-1, dtype=torch.int32).contiguous()


if TRITON_AVAILABLE:
    _AUTOTUNE_CONFIGS = [
        triton.Config({'BLOCK_M': 16, 'PIPE_STAGES': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'PIPE_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'PIPE_STAGES': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'PIPE_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'PIPE_STAGES': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'PIPE_STAGES': 3}, num_warps=8, num_stages=3),
    ]

    def _autotune(configs, key):
        try:
            return triton.autotune(
                configs=configs,
                key=key,
                cache_results=True,
            )
        except TypeError:  # older Triton without persistent autotune cache
            return triton.autotune(configs=configs, key=key)

    @triton.jit
    def _probability_int8_pv(
        probabilities,
        v_int8,
        v_sum,
        v_scale,
        BLOCK_M: tl.constexpr,
        D: tl.constexpr,
    ):
        # tl.dot only accepts signed INT8, not UINT8. Encode logical UINT8
        # probabilities as p_s8 = p_u8 - 128, then correct the dot product by
        # adding 128 * sum(V). V_SUM is precomputed during streaming packing.
        p_max = tl.max(probabilities, axis=1)
        p_scale = tl.maximum(p_max / 255.0, 1.0e-8)
        p_code = probabilities / p_scale[:, None] + 0.5
        p_code = tl.minimum(tl.maximum(p_code, 0.0), 255.0)
        p_signed = (p_code - 128.0).to(tl.int8)
        pv_i32 = tl.dot(p_signed, v_int8, out_dtype=tl.int32)
        pv_i32 += 128 * v_sum[None, :]
        return pv_i32.to(tl.float32) * p_scale[:, None] * v_scale[None, :]

    @_autotune(
        _AUTOTUNE_CONFIGS,
        key=['KV_BLOCKS', 'OUTPUT_BF16', 'V_SCALE_GROUP'],
    )
    @triton.jit
    def _int8_dense_q_kernel(
        Q,
        K,
        V,
        Q_SCALE,
        K_SCALE,
        V_SCALE,
        V_SUM,
        O,
        sequence,
        heads,
        DENSE_Q_TILES,
        KV_BLOCKS: tl.constexpr,
        softmax_scale: tl.constexpr,
        Q_TILE_: tl.constexpr,
        KV_TILE_: tl.constexpr,
        D: tl.constexpr,
        V_SCALE_GROUP: tl.constexpr,
        OUTPUT_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
    ):
        work = tl.program_id(0).to(tl.int64)
        bh = tl.program_id(1).to(tl.int64)
        subblocks = Q_TILE_ // BLOCK_M
        q_block = work // subblocks
        q_sub = work - q_block * subblocks
        if q_block >= DENSE_Q_TILES:
            return
        batch = bh // heads
        head = bh - batch * heads
        q_rows = (
            q_block * Q_TILE_
            + q_sub * BLOCK_M
            + tl.arange(0, BLOCK_M).to(tl.int64)
        )
        kv_rows = tl.arange(0, KV_TILE_).to(tl.int64)
        dims = tl.arange(0, D).to(tl.int64)
        q_mask = q_rows < sequence
        hnd_base = (batch * heads + head) * sequence * D
        q_tiles = (sequence + Q_TILE_ - 1) // Q_TILE_
        q = tl.load(
            Q + hnd_base + q_rows[:, None] * D + dims[None, :],
            mask=q_mask[:, None],
            other=0,
        ).to(tl.int8)
        q_scale = tl.load(Q_SCALE + bh * q_tiles + q_block).to(tl.float32)

        m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        v_groups = D // V_SCALE_GROUP

        for key_block in tl.range(0, KV_BLOCKS, num_stages=PIPE_STAGES):
            k_start = key_block * KV_TILE_
            k_positions = k_start + kv_rows
            k_mask = k_positions < sequence
            k = tl.load(
                K + hnd_base + k_positions[None, :] * D + dims[:, None],
                mask=k_mask[None, :],
                other=0,
            ).to(tl.int8)
            score_i32 = tl.dot(q, k, out_dtype=tl.int32)
            k_scale = tl.load(K_SCALE + bh * KV_BLOCKS + key_block).to(tl.float32)
            logits = score_i32.to(tl.float32) * (
                q_scale * k_scale * softmax_scale * 1.4426950408889634
            )
            logits = tl.where(k_mask[None, :], logits, -float('inf'))
            local_m = tl.max(logits, axis=1)
            new_m = tl.maximum(m_i, local_m)
            probabilities = tl.math.exp2(logits - new_m[:, None])
            alpha = tl.math.exp2(m_i - new_m)

            v_int8 = tl.load(
                V + hnd_base + k_positions[:, None] * D + dims[None, :],
                mask=k_mask[:, None],
                other=0,
            ).to(tl.int8)
            v_sum = tl.load(
                V_SUM + (bh * KV_BLOCKS + key_block) * D + dims
            ).to(tl.int32)
            v_group = dims // V_SCALE_GROUP
            v_scale = tl.load(
                V_SCALE + (bh * KV_BLOCKS + key_block) * v_groups + v_group
            ).to(tl.float32)

            acc = acc * alpha[:, None]
            acc += _probability_int8_pv(
                probabilities,
                v_int8,
                v_sum,
                v_scale,
                BLOCK_M=BLOCK_M,
                D=D,
            )
            l_i = l_i * alpha + tl.sum(probabilities, axis=1)
            m_i = new_m

        output = acc / l_i[:, None]
        tl.store(
            O + hnd_base + q_rows[:, None] * D + dims[None, :],
            output.to(O.type.element_ty),
            mask=q_mask[:, None],
        )

    @_autotune(
        _AUTOTUNE_CONFIGS,
        key=['N_SELECTED', 'KV_BLOCKS', 'OUTPUT_BF16', 'V_SCALE_GROUP'],
    )
    @triton.jit
    def _int8_sparse_fixed_kernel(
        Q,
        K,
        V,
        Q_SCALE,
        K_SCALE,
        V_SCALE,
        V_SUM,
        KV_INDICES,
        O,
        sequence,
        heads,
        SPARSE_Q_START,
        SPARSE_Q_TILES,
        N_SELECTED: tl.constexpr,
        KV_BLOCKS: tl.constexpr,
        softmax_scale: tl.constexpr,
        Q_TILE_: tl.constexpr,
        KV_TILE_: tl.constexpr,
        D: tl.constexpr,
        V_SCALE_GROUP: tl.constexpr,
        OUTPUT_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
    ):
        work = tl.program_id(0).to(tl.int64)
        bh = tl.program_id(1).to(tl.int64)
        subblocks = Q_TILE_ // BLOCK_M
        sparse_local = work // subblocks
        q_sub = work - sparse_local * subblocks
        if sparse_local >= SPARSE_Q_TILES:
            return
        q_block = SPARSE_Q_START + sparse_local
        batch = bh // heads
        head = bh - batch * heads
        q_rows = (
            q_block * Q_TILE_
            + q_sub * BLOCK_M
            + tl.arange(0, BLOCK_M).to(tl.int64)
        )
        kv_rows = tl.arange(0, KV_TILE_).to(tl.int64)
        dims = tl.arange(0, D).to(tl.int64)
        q_mask = q_rows < sequence
        hnd_base = (batch * heads + head) * sequence * D
        q_tiles = (sequence + Q_TILE_ - 1) // Q_TILE_
        q = tl.load(
            Q + hnd_base + q_rows[:, None] * D + dims[None, :],
            mask=q_mask[:, None],
            other=0,
        ).to(tl.int8)
        q_scale = tl.load(Q_SCALE + bh * q_tiles + q_block).to(tl.float32)

        m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        v_groups = D // V_SCALE_GROUP
        route_base = (bh * SPARSE_Q_TILES + sparse_local) * N_SELECTED

        for slot in tl.range(0, N_SELECTED, num_stages=PIPE_STAGES):
            key_block = tl.load(KV_INDICES + route_base + slot).to(tl.int32)
            k_start = key_block * KV_TILE_
            k_positions = k_start + kv_rows
            k_mask = k_positions < sequence
            k = tl.load(
                K + hnd_base + k_positions[None, :] * D + dims[:, None],
                mask=k_mask[None, :],
                other=0,
            ).to(tl.int8)
            score_i32 = tl.dot(q, k, out_dtype=tl.int32)
            k_scale = tl.load(K_SCALE + bh * KV_BLOCKS + key_block).to(tl.float32)
            logits = score_i32.to(tl.float32) * (
                q_scale * k_scale * softmax_scale * 1.4426950408889634
            )
            logits = tl.where(k_mask[None, :], logits, -float('inf'))
            local_m = tl.max(logits, axis=1)
            new_m = tl.maximum(m_i, local_m)
            probabilities = tl.math.exp2(logits - new_m[:, None])
            alpha = tl.math.exp2(m_i - new_m)

            v_int8 = tl.load(
                V + hnd_base + k_positions[:, None] * D + dims[None, :],
                mask=k_mask[:, None],
                other=0,
            ).to(tl.int8)
            v_sum = tl.load(
                V_SUM + (bh * KV_BLOCKS + key_block) * D + dims
            ).to(tl.int32)
            v_group = dims // V_SCALE_GROUP
            v_scale = tl.load(
                V_SCALE + (bh * KV_BLOCKS + key_block) * v_groups + v_group
            ).to(tl.float32)

            acc = acc * alpha[:, None]
            acc += _probability_int8_pv(
                probabilities,
                v_int8,
                v_sum,
                v_scale,
                BLOCK_M=BLOCK_M,
                D=D,
            )
            l_i = l_i * alpha + tl.sum(probabilities, axis=1)
            m_i = new_m

        output = acc / l_i[:, None]
        tl.store(
            O + hnd_base + q_rows[:, None] * D + dims[None, :],
            output.to(O.type.element_ty),
            mask=q_mask[:, None],
        )


def _launch_int8_sparse(prepared, spec, output):
    q_tile = int(spec.q_tile)
    kv_blocks = int(prepared.kv_tiles)
    bh = prepared.q_int8.shape[0] * prepared.heads
    if prepared.dense_q_tiles:
        def dense_grid(meta):
            return (
                prepared.dense_q_tiles * (q_tile // meta['BLOCK_M']),
                bh,
            )

        _int8_dense_q_kernel[dense_grid](
            prepared.q_int8,
            prepared.k_int8,
            prepared.v_int8,
            prepared.q_scale,
            prepared.k_scale,
            prepared.v_scale,
            prepared.v_sum,
            output,
            prepared.sequence,
            prepared.heads,
            prepared.dense_q_tiles,
            KV_BLOCKS=kv_blocks,
            softmax_scale=spec.head_dim ** -0.5,
            Q_TILE_=spec.q_tile,
            KV_TILE_=spec.kv_tile,
            D=spec.head_dim,
            V_SCALE_GROUP=spec.v_scale_group_size,
            OUTPUT_BF16=prepared.output_dtype == torch.bfloat16,
        )
    if prepared.sparse_q_tiles:
        def sparse_grid(meta):
            return (
                prepared.sparse_q_tiles * (q_tile // meta['BLOCK_M']),
                bh,
            )

        _int8_sparse_fixed_kernel[sparse_grid](
            prepared.q_int8,
            prepared.k_int8,
            prepared.v_int8,
            prepared.q_scale,
            prepared.k_scale,
            prepared.v_scale,
            prepared.v_sum,
            prepared.kv_indices,
            output,
            prepared.sequence,
            prepared.heads,
            prepared.dense_q_tiles,
            prepared.sparse_q_tiles,
            N_SELECTED=prepared.sparse_selected,
            KV_BLOCKS=kv_blocks,
            softmax_scale=spec.head_dim ** -0.5,
            Q_TILE_=spec.q_tile,
            KV_TILE_=spec.kv_tile,
            D=spec.head_dim,
            V_SCALE_GROUP=spec.v_scale_group_size,
            OUTPUT_BF16=prepared.output_dtype == torch.bfloat16,
        )
    return output


@dataclass
class PreparedTritonSparse:
    q_int8: torch.Tensor
    q_scale: torch.Tensor
    k_int8: torch.Tensor
    k_scale: torch.Tensor
    v_int8: torch.Tensor
    v_scale: torch.Tensor
    v_sum: torch.Tensor
    kv_indices: torch.Tensor
    valid_block_num: torch.Tensor
    output_dtype: torch.dtype
    layer_index: int
    sequence: int
    heads: int
    q_tiles: int
    kv_tiles: int
    dense_q_tiles: int
    sparse_q_tiles: int
    sparse_selected: int
    metadata: dict


class TritonSparseExecutor:
    def __init__(
        self,
        spec,
        *,
        allow_cpu_for_tests=False,
        kernel=None,
        float_packer=None,
    ):
        if not isinstance(spec, TritonSparseSpec):
            raise TypeError('TritonSparseExecutor requires TritonSparseSpec')
        self.spec = spec
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)
        self.kernel = kernel
        self.float_packer = float_packer or pack_float_qkv

    def _validate_float(self, q, k, v, lut, valid):
        if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
            raise TritonSparseError(
                'INT8 Triton sparse requires equal HND rank-4 Q/K/V shapes'
            )
        batch, heads, sequence, head_dim = q.shape
        if batch != 1 or head_dim != self.spec.head_dim:
            raise TritonSparseError(
                'INT8 Triton sparse requires batch 1 and head_dim %d'
                % self.spec.head_dim
            )
        if (
            q.dtype not in (torch.float16, torch.bfloat16)
            or q.dtype != k.dtype
            or q.dtype != v.dtype
        ):
            raise TritonSparseError(
                'INT8 Triton sparse Q/K/V require matching fp16 or bf16 dtypes'
            )
        if q.device != k.device or q.device != v.device:
            raise TritonSparseError('INT8 Triton sparse Q/K/V devices differ')
        if any(tensor.stride(-1) != 1 for tensor in (q, k, v)):
            raise TritonSparseError(
                'INT8 Triton sparse Q/K/V last dimension must be contiguous'
            )
        self.spec.validate_lut(
            lut,
            valid,
            batch=batch,
            heads=heads,
            sequence=sequence,
        )
        if lut.device != q.device:
            raise TritonSparseError('INT8 Triton sparse LUT device differs')
        if not self.allow_cpu_for_tests and not q.is_cuda:
            raise TritonSparseError('INT8 Triton sparse attention requires CUDA')

    def _prepare_projected(self, projected, lut, valid_block_num, metadata):
        try:
            validate_prepared_triton_sparse_qkv(projected)
        except TritonSparseQKVError as exc:
            raise TritonSparseError(str(exc)) from exc
        if int(projected.v_scale_group_size) != int(self.spec.v_scale_group_size):
            raise TritonSparseError(
                'projected Triton V scale group %d does not match backend group %d'
                % (projected.v_scale_group_size, self.spec.v_scale_group_size)
            )
        self.spec.validate_lut(
            lut,
            valid_block_num,
            batch=1,
            heads=projected.heads,
            sequence=projected.sequence,
        )
        if lut.device != projected.q_int8.device:
            raise TritonSparseError('INT8 Triton sparse LUT device differs')
        if not self.allow_cpu_for_tests and not projected.q_int8.is_cuda:
            raise TritonSparseError('INT8 Triton sparse attention requires CUDA')

        q_tiles = (projected.sequence + self.spec.q_tile - 1) // self.spec.q_tile
        kv_tiles = (projected.sequence + self.spec.kv_tile - 1) // self.spec.kv_tile
        dense_q_tiles, sparse_q_tiles, selected = _route_plan(
            metadata,
            q_tiles=q_tiles,
            kv_tiles=kv_tiles,
        )
        kv_indices = _compact_absolute_indices(
            lut,
            dense_q_tiles=dense_q_tiles,
            sparse_q_tiles=sparse_q_tiles,
            selected=selected,
        )
        details = dict(metadata)
        details.update(
            {
                'sparse_backend': 'triton_int8',
                'sparse_kernel': self.spec.implementation,
                'qkv_lifetime': 'independent_int8_carriers',
                'v_format': 'per_kv_tile_group%d_int8'
                % self.spec.v_scale_group_size,
                'route_format': 'absolute_compact_int32',
                'probability_value_path': 'u8_logical_x_int8_tensorcore',
                'dense_q_kernel': bool(dense_q_tiles),
                'fixed_sparse_kv_blocks': int(selected),
            }
        )
        return PreparedTritonSparse(
            q_int8=projected.q_int8,
            q_scale=projected.q_scale,
            k_int8=projected.k_int8,
            k_scale=projected.k_scale,
            v_int8=projected.v_int8,
            v_scale=projected.v_scale,
            v_sum=projected.v_sum,
            kv_indices=kv_indices,
            valid_block_num=valid_block_num,
            output_dtype=projected.output_dtype,
            layer_index=int(projected.layer_index),
            sequence=int(projected.sequence),
            heads=int(projected.heads),
            q_tiles=q_tiles,
            kv_tiles=kv_tiles,
            dense_q_tiles=dense_q_tiles,
            sparse_q_tiles=sparse_q_tiles,
            sparse_selected=selected,
            metadata=details,
        )

    def prepare(
        self,
        q,
        k,
        v,
        lut,
        valid_block_num,
        *,
        layer_index,
        metadata,
    ):
        self._validate_float(q, k, v, lut, valid_block_num)
        try:
            projected = self.float_packer(
                q,
                k,
                v,
                layer_index=layer_index,
                v_scale_group_size=self.spec.v_scale_group_size,
            )
        except TritonSparseQKVError as exc:
            raise TritonSparseError(str(exc)) from exc
        prepared = self._prepare_projected(
            projected,
            lut,
            valid_block_num,
            metadata,
        )
        prepared.metadata['qkv_projection'] = 'standard_qkv_then_int8_pack'
        return prepared

    def prepare_projected(
        self,
        projected,
        lut,
        valid_block_num,
        *,
        layer_index,
        metadata,
    ):
        if int(projected.layer_index) != int(layer_index):
            raise TritonSparseError(
                'projected Triton QKV layer %d does not match attention layer %d'
                % (projected.layer_index, layer_index)
            )
        prepared = self._prepare_projected(
            projected,
            lut,
            valid_block_num,
            metadata,
        )
        prepared.metadata['qkv_projection'] = 'chunked_convrot_int8'
        return prepared

    def execute(self, prepared):
        if self.allow_cpu_for_tests and self.kernel is not None:
            return self.kernel(prepared)
        if not TRITON_AVAILABLE:
            raise TritonSparseError('INT8 Triton sparse attention requires Triton')
        if not prepared.q_int8.is_cuda:
            raise TritonSparseError('INT8 Triton sparse attention requires CUDA')
        output = torch.empty(
            (1, prepared.heads, prepared.sequence, self.spec.head_dim),
            dtype=prepared.output_dtype,
            device=prepared.q_int8.device,
        )
        try:
            return _launch_int8_sparse(prepared, self.spec, output)
        except Exception as exc:
            raise TritonSparseError(
                'INT8 Triton sparse kernel failed: layer=%d sequence=%d '
                'heads=%d dtype=%s group=%d'
                % (
                    prepared.layer_index,
                    prepared.sequence,
                    prepared.heads,
                    prepared.output_dtype,
                    self.spec.v_scale_group_size,
                )
            ) from exc


@dataclass
class PreparedTritonHybrid:
    sparse: PreparedTritonSparse


class TritonSparseBackend:
    name = 'triton_sparse_int8'
    requires_runtime_context = True
    approximate = True

    def __init__(
        self,
        config=None,
        *,
        spec=None,
        router=None,
        projector=None,
        allow_cpu_for_tests=False,
        executor=None,
    ):
        self.config = config or HybridSparseConfig()
        if not isinstance(self.config, HybridSparseConfig):
            raise TypeError('config must be HybridSparseConfig')
        self.spec = spec or TritonSparseSpec()
        self.projector = projector
        self.router = router or SparseTileRouter(
            self.config,
            q_tile=self.spec.q_tile,
            kv_tile=self.spec.kv_tile,
        )
        if (self.router.q_tile, self.router.kv_tile) != (
            self.spec.q_tile,
            self.spec.kv_tile,
        ):
            raise TritonSparseError(
                'router geometry does not match INT8 Triton sparse geometry'
            )
        self.executor = executor or TritonSparseExecutor(
            self.spec,
            allow_cpu_for_tests=allow_cpu_for_tests,
        )

    @property
    def installation_signature(self):
        return (
            self.name,
            self.config.signature,
            self.spec.signature,
            None
            if self.projector is None
            else getattr(self.projector, 'installation_signature', None),
        )

    @staticmethod
    def _snapshot(transformer_options, sequence):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None:
            raise TritonSparseError(
                'INT8 Triton sparse attention requires an H3 runtime snapshot'
            )
        if not snapshot.valid_layout:
            raise TritonSparseError(
                'INT8 Triton sparse attention requires a valid packed layout: %s'
                % (snapshot.error or 'layout unavailable')
            )
        if int(snapshot.layout.seq_len) != int(sequence):
            raise TritonSparseError(
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
                'triton_sparse_heads': int(heads),
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
        )
        try:
            lut, valid_block_num, mask_metadata = self.router.build_lut(
                q,
                k,
                snapshot.layout,
                video_budget,
            )
        except SparseRouterError as exc:
            raise TritonSparseError('sparse routing failed: %s' % exc) from exc
        return PreparedTritonHybrid(
            sparse=self.executor.prepare(
                q,
                k,
                v,
                lut,
                valid_block_num,
                layer_index=layer_index,
                metadata=self._metadata(mask_metadata, layer_index, q.shape[1]),
            )
        )

    def prepare_projected(
        self,
        projected,
        *,
        layer_index,
        transformer_options,
    ):
        try:
            validate_prepared_triton_sparse_qkv(projected)
        except TritonSparseQKVError as exc:
            raise TritonSparseError(str(exc)) from exc
        snapshot = self._snapshot(transformer_options, projected.sequence)
        video_budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
        )
        try:
            lut, valid_block_num, mask_metadata = self.router.build_lut_from_summaries(
                projected.q_summary,
                projected.k_summary,
                snapshot.layout,
                video_budget,
            )
        except SparseRouterError as exc:
            raise TritonSparseError('sparse routing failed: %s' % exc) from exc
        return PreparedTritonHybrid(
            sparse=self.executor.prepare_projected(
                projected,
                lut,
                valid_block_num,
                layer_index=layer_index,
                metadata=self._metadata(
                    mask_metadata,
                    layer_index,
                    projected.heads,
                ),
            )
        )

    def execute(self, prepared):
        return self.executor.execute(prepared.sparse)

    def as_status(self):
        return {
            'mode': self.name,
            'video_budget': float(self.config.video_budget),
            'denser_early_late_steps': bool(self.config.denser_early_late_steps),
            'density_mode': self.config.density_mode,
            'sparse_q_tile': self.spec.q_tile,
            'sparse_kv_tile': self.spec.kv_tile,
            'qkv_dtype': 'int8',
            'probability_value_path': 'logical_u8_x_int8_tensorcore',
            'route_format': 'absolute_compact_int32',
            'fixed_density_specialization': True,
            'dense_q_kernel': True,
            'autotune_block_m': [16, 32, 64],
            'autotune_warps': [4, 8],
            'v_scale_group_size': self.spec.v_scale_group_size,
            'v_scale_layout': 'per_kv_tile_group_float32',
            'chunked_qkv': self.projector is not None,
            'approximate': True,
        }
