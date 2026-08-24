# SPDX-License-Identifier: Apache-2.0
# The residual preprocessing and pointer-kernel structure are adapted from
# Saganaki22/ComfyUI-sol-attn at SOURCE_COMMIT. The Apache-2.0 license text is
# included at native/LICENSE.

'''Native sparse attention with an optional 64x64 Sol residual.'''

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch

from ... import diagnostics
from ...kitchen_qkv import PreparedChunkedKitchenQKV
from .config import HybridSparseConfig
from .kitchen_sparse import (
    HEAD_DIM,
    KV_TILE as EXACT_KV_TILE,
    OUTPUT_NHD,
    Q_TILE as EXACT_Q_TILE,
    SparseKitchenBackend,
    SparseKitchenError,
)

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


RESIDUAL_TILE = 64
CARRIER_Q_TILE = 128
ROUTE_GROUP = 32
SOURCE_COMMIT = '930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf'
SOURCE_REPOSITORY = 'https://github.com/Saganaki22/ComfyUI-sol-attn'
LOG_PREFIX = '[H3 Optimizations]'


class SolResidualError(RuntimeError):
    pass


@dataclass(frozen=True)
class SolResidualSpec:
    exact_q_tile: int = EXACT_Q_TILE
    exact_kv_tile: int = EXACT_KV_TILE
    residual_q_tile: int = RESIDUAL_TILE
    residual_kv_tile: int = RESIDUAL_TILE
    head_dim: int = HEAD_DIM

    def __post_init__(self):
        if self.residual_q_tile != self.residual_kv_tile:
            raise SolResidualError('Sol residual attention requires square residual tiles')
        if (
            self.exact_q_tile % self.residual_q_tile
            or self.exact_kv_tile % self.residual_kv_tile
        ):
            raise SolResidualError(
                'Sol residual tiles must divide the exact sparse geometry'
            )

    @property
    def implementation(self):
        return 'native_int8_%dx%d_plus_sol_residual_%dx%d' % (
            self.exact_q_tile,
            self.exact_kv_tile,
            self.residual_q_tile,
            self.residual_kv_tile,
        )

    @property
    def signature(self):
        return (
            self.implementation,
            int(self.exact_q_tile),
            int(self.exact_kv_tile),
            int(self.residual_q_tile),
            int(self.residual_kv_tile),
            int(self.head_dim),
            SOURCE_COMMIT,
        )


def preflight_sol_residual(
    *,
    cuda_available,
    capability_getter,
    kitchen,
    exact_q_tile=EXACT_Q_TILE,
    exact_kv_tile=EXACT_KV_TILE,
    triton_available=None,
):
    if not cuda_available():
        raise SolResidualError('Sol residual attention requires NVIDIA CUDA')
    available = TRITON_AVAILABLE if triton_available is None else bool(triton_available)
    if not available:
        raise SolResidualError('Sol residual attention requires Triton')
    capability = capability_getter()
    if capability is None:
        raise SolResidualError('Sol residual GPU capability is unavailable')
    capability = tuple(int(value) for value in capability)
    if capability not in ((8, 6), (8, 9), (12, 0)):
        raise SolResidualError(
            'Sol residual attention supports SM86, SM89, and SM120; got SM%s'
            % ''.join(str(value) for value in capability)
        )
    if not hasattr(
        kitchen,
        'block_sparse_int8_attention_with_lse_from_prequantized',
    ):
        raise SolResidualError(
            'Sol residual attention needs the native sparse kernel with LSE output'
        )
    spec = SolResidualSpec(
        exact_q_tile=int(exact_q_tile),
        exact_kv_tile=int(exact_kv_tile),
    )
    return spec


def pack_exact_route(route, *, q_tiles, kv_tiles):
    '''Pack selected exact parent tiles into int64 bitset words.'''
    absolute = route.to_absolute()
    indices = absolute.indices
    counts = absolute.counts
    if indices.ndim != 4:
        raise SolResidualError('exact route indices must be rank-4')
    expected = indices.shape[:2] + (int(q_tiles),)
    if tuple(indices.shape[:3]) != expected:
        raise SolResidualError(
            'exact route shape %s does not match %s'
            % (tuple(indices.shape), expected + (indices.shape[-1],))
        )
    if tuple(counts.shape) != expected:
        raise SolResidualError(
            'exact route counts shape %s does not match %s'
            % (tuple(counts.shape), expected)
        )
    if indices.dtype != torch.int32 or counts.dtype != torch.int32:
        raise SolResidualError('exact route indices and counts must be int32')

    positions = torch.arange(indices.shape[-1], device=indices.device)
    live = positions < counts.to(torch.int64).unsqueeze(-1)
    words = (int(kv_tiles) + ROUTE_GROUP - 1) // ROUTE_GROUP
    word_indices = torch.div(indices, ROUTE_GROUP, rounding_mode='floor').long()
    bits = torch.bitwise_left_shift(
        torch.ones_like(indices, dtype=torch.int64),
        torch.remainder(indices, ROUTE_GROUP).long(),
    )
    word_indices = torch.where(live, word_indices, torch.zeros_like(word_indices))
    bits = torch.where(live, bits, torch.zeros_like(bits))
    packed = torch.zeros(
        indices.shape[:3] + (words,),
        dtype=torch.int64,
        device=indices.device,
    )
    packed.scatter_add_(-1, word_indices, bits)
    return packed.contiguous()


if TRITON_AVAILABLE:
    @triton.jit
    def _reduce_kv_kernel(
        k_ptr, v_ptr, k_scale_ptr, v_scale_ptr, k_mean_ptr, v_sum_ptr,
        sequence,
        head_dim: tl.constexpr,
        blocks: tl.constexpr,
        exact_kv_tile: tl.constexpr,
        exact_kv_tiles: tl.constexpr,
        padded_sequence: tl.constexpr,
        block_size: tl.constexpr,
    ):
        block = tl.program_id(0)
        batch_head = tl.program_id(1)
        rows = block * block_size + tl.arange(0, block_size)
        dims = tl.arange(0, head_dim)
        valid = rows < sequence
        k = tl.load(
            k_ptr
            + (batch_head * sequence + rows[:, None].to(tl.int64)) * head_dim
            + dims[None, :],
            mask=valid[:, None],
            other=0,
        ).to(tl.float32)
        k_scale = tl.load(
            k_scale_ptr
            + batch_head * exact_kv_tiles * 4
            + (rows // exact_kv_tile) * 4
            + (rows % 8) // 2,
            mask=valid,
            other=0.0,
        )
        k = k * k_scale[:, None]
        v = tl.load(
            v_ptr
            + (batch_head * head_dim + dims[None, :]) * padded_sequence
            + rows[:, None].to(tl.int64),
            mask=valid[:, None],
            other=0,
        ).to(tl.float32)
        v_scale = tl.load(
            v_scale_ptr + batch_head * head_dim + dims
        )
        v = v * v_scale[None, :]
        length = tl.minimum(block_size, sequence - block * block_size)
        output = (batch_head * blocks + block) * head_dim
        tl.store(k_mean_ptr + output + dims, tl.sum(k, axis=0) / length)
        tl.store(v_sum_ptr + output + dims, tl.sum(v, axis=0))


    @triton.jit
    def _sol_residual_kernel(
        q_ptr, q_scale_ptr, k_mean_ptr, v_sum_ptr,
        exact_output_ptr, exact_lse_ptr, route_ptr, output_ptr,
        sequence, scale_log2,
        sq_b, sq_h, sq_t,
        so_b, so_h, so_t,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        exact_q_tile: tl.constexpr,
        exact_kv_tile: tl.constexpr,
        exact_q_tiles: tl.constexpr,
        exact_kv_tiles: tl.constexpr,
        residual_kv_tiles: tl.constexpr,
        q_scales_per_head: tl.constexpr,
        route_words: tl.constexpr,
        block_size: tl.constexpr,
        group_size: tl.constexpr,
        route_group: tl.constexpr,
    ):
        q_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // heads
        head = batch_head - batch * heads
        token_offsets = tl.max_contiguous(tl.arange(0, block_size), block_size)
        group_offsets = tl.max_contiguous(tl.arange(0, group_size), group_size)
        dims = tl.arange(0, head_dim)
        q_rows = q_block * block_size + token_offsets
        q_valid = q_rows < sequence
        q = tl.load(
            q_ptr
            + batch * sq_b
            + head * sq_h
            + q_rows[:, None].to(tl.int64) * sq_t
            + dims[None, :],
            mask=q_valid[:, None],
            other=0,
        ).to(tl.float32)
        q_scale = tl.load(
            q_scale_ptr
            + batch_head * q_scales_per_head
            + (q_rows // 32) * 8
            + q_rows % 8,
            mask=q_valid,
            other=0.0,
        )
        q = (q * q_scale[:, None]).to(tl.bfloat16)
        exact_output = tl.load(
            exact_output_ptr
            + batch * so_b
            + head * so_h
            + q_rows[:, None].to(tl.int64) * so_t
            + dims[None, :],
            mask=q_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        exact_lse = tl.load(
            exact_lse_ptr + (batch_head * sequence + q_rows),
            mask=q_valid,
            other=0.0,
        )

        numerator = exact_output
        denominator = tl.full((block_size,), 1.0, tl.float32)
        row_max = exact_lse
        tail_length = sequence - (residual_kv_tiles - 1) * block_size
        parent_q = q_block // (exact_q_tile // block_size)

        for group_start in range(0, residual_kv_tiles, group_size):
            block_indices = group_start + group_offsets
            valid_blocks = block_indices < residual_kv_tiles
            parent_kv = block_indices // (exact_kv_tile // block_size)
            route_word = tl.load(
                route_ptr
                + (batch_head * exact_q_tiles + parent_q) * route_words
                + parent_kv // route_group,
                mask=valid_blocks & (parent_kv < exact_kv_tiles),
                other=0,
            ).to(tl.int64)
            selected_parent = (
                (route_word >> (parent_kv % route_group).to(tl.int64)) & 1
            ) != 0
            approximate = valid_blocks & ~selected_parent
            k_mean = tl.load(
                k_mean_ptr
                + (batch_head * residual_kv_tiles + block_indices[:, None])
                * head_dim
                + dims[None, :],
                mask=valid_blocks[:, None],
                other=0.0,
            )
            scores = tl.dot(q, k_mean.T).to(tl.float32) * scale_log2
            scores = tl.where(approximate[None, :], scores, -float('inf'))
            new_max = tl.maximum(row_max, tl.max(scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            probability = tl.where(
                approximate[None, :],
                tl.math.exp2(scores - new_max[:, None]),
                0.0,
            )
            v_sum = tl.load(
                v_sum_ptr
                + (batch_head * residual_kv_tiles + block_indices[:, None])
                * head_dim
                + dims[None, :],
                mask=valid_blocks[:, None],
                other=0.0,
            )
            numerator = numerator * alpha[:, None] + tl.dot(
                probability.to(v_sum.dtype), v_sum
            )
            lengths = tl.where(
                block_indices == residual_kv_tiles - 1,
                tail_length,
                block_size,
            ).to(tl.float32)
            denominator = denominator * alpha + tl.sum(
                probability * lengths[None, :], axis=1
            )
            row_max = new_max

        tl.store(
            output_ptr
            + ((batch * sequence + q_rows[:, None]) * heads + head) * head_dim
            + dims[None, :],
            (numerator / denominator[:, None]).to(tl.bfloat16),
            mask=q_valid[:, None],
        )


def _summarize_kv(carrier):
    if not TRITON_AVAILABLE:
        raise SolResidualError('Sol residual attention requires Triton')
    sequence = int(carrier.k.shape[-2])
    exact_kv_tile = int(carrier.cta_k)
    blocks = (sequence + RESIDUAL_TILE - 1) // RESIDUAL_TILE
    k_mean = torch.empty(
        carrier.k.shape[0], carrier.k.shape[1], blocks, HEAD_DIM,
        dtype=torch.bfloat16,
        device=carrier.k.device,
    )
    v_sum = torch.empty_like(k_mean)
    with diagnostics.stage('sol_residual_preprocess'):
        _reduce_kv_kernel[(blocks, carrier.k.shape[0] * carrier.k.shape[1])](
            carrier.k,
            carrier.v,
            carrier.k_scale,
            carrier.v_scale,
            k_mean,
            v_sum,
            sequence,
            head_dim=HEAD_DIM,
            blocks=blocks,
            exact_kv_tile=exact_kv_tile,
            exact_kv_tiles=(sequence + exact_kv_tile - 1) // exact_kv_tile,
            padded_sequence=carrier.v.shape[-1],
            block_size=RESIDUAL_TILE,
            num_warps=8,
            num_stages=1,
        )
    return k_mean, v_sum


def _summarize_kv_cpu(carrier):
    '''Reference carrier decode used only by CPU contract tests.'''
    batch, heads, sequence, head_dim = carrier.k.shape
    exact_kv_tile = int(carrier.cta_k)
    exact_kv_tiles = (sequence + exact_kv_tile - 1) // exact_kv_tile
    rows = torch.arange(sequence, device=carrier.k.device)
    k_scale_index = (rows // exact_kv_tile) * 4 + (rows % 8) // 2
    k_scale = carrier.k_scale.reshape(
        batch, heads, exact_kv_tiles * 4
    ).index_select(-1, k_scale_index)
    k = carrier.k.float() * k_scale.unsqueeze(-1)
    v = carrier.v.reshape(
        batch, heads, head_dim, carrier.v.shape[-1]
    )[..., :sequence].transpose(-1, -2).float()
    v = v * carrier.v_scale.reshape(batch, heads, head_dim).unsqueeze(-2)
    k_mean = []
    v_sum = []
    for start in range(0, sequence, RESIDUAL_TILE):
        k_mean.append(k[..., start:start + RESIDUAL_TILE, :].mean(-2, keepdim=True))
        v_sum.append(v[..., start:start + RESIDUAL_TILE, :].sum(-2, keepdim=True))
    return (
        torch.cat(k_mean, dim=-2).to(torch.bfloat16),
        torch.cat(v_sum, dim=-2).to(torch.bfloat16),
    )


@dataclass
class PreparedSolResidual:
    exact: object
    q: torch.Tensor | None
    q_scale: torch.Tensor | None
    k_mean: torch.Tensor | None
    v_sum: torch.Tensor | None
    exact_route: torch.Tensor | None
    batch: int
    sequence: int
    heads: int
    exact_q_tile: int
    exact_kv_tile: int
    residual_q_tile: int
    residual_kv_tile: int
    exact_q_tiles: int
    exact_kv_tiles: int
    residual_q_tiles: int
    residual_kv_tiles: int
    scale_log2: float
    layer_index: int
    metadata: dict

    def release_exact(self):
        release = getattr(self.exact, 'release', None)
        if release is not None:
            release()
        self.exact = None


def launch_sol_residual(prepared, exact_output, exact_lse):
    if not TRITON_AVAILABLE:
        raise SolResidualError('Sol residual attention requires Triton')
    if prepared.q.device.type != 'cuda':
        raise SolResidualError('Sol residual attention requires CUDA tensors')
    expected_output = (
        prepared.batch,
        prepared.heads,
        prepared.sequence,
        HEAD_DIM,
    )
    if tuple(exact_output.shape) != expected_output:
        raise SolResidualError('native exact attention returned an invalid shape')
    if exact_output.dtype != torch.bfloat16 or exact_output.device != prepared.q.device:
        raise SolResidualError('native exact attention returned an invalid carrier')
    if tuple(exact_lse.shape) != expected_output[:3]:
        raise SolResidualError('native exact attention returned an invalid LSE shape')
    if exact_lse.dtype != torch.float32 or exact_lse.device != prepared.q.device:
        raise SolResidualError('native exact attention returned an invalid LSE carrier')
    storage = torch.empty(
        prepared.q.shape[0],
        prepared.sequence,
        prepared.heads,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=prepared.q.device,
    )
    with diagnostics.stage('sol_residual_kernel'):
        _sol_residual_kernel[
            (prepared.residual_q_tiles, prepared.q.shape[0] * prepared.heads)
        ](
            prepared.q,
            prepared.q_scale,
            prepared.k_mean,
            prepared.v_sum,
            exact_output,
            exact_lse,
            prepared.exact_route,
            storage,
            prepared.sequence,
            prepared.scale_log2,
            prepared.q.stride(0),
            prepared.q.stride(1),
            prepared.q.stride(2),
            exact_output.stride(0),
            exact_output.stride(1),
            exact_output.stride(2),
            heads=prepared.heads,
            head_dim=HEAD_DIM,
            exact_q_tile=prepared.exact_q_tile,
            exact_kv_tile=prepared.exact_kv_tile,
            exact_q_tiles=prepared.exact_q_tiles,
            exact_kv_tiles=prepared.exact_kv_tiles,
            residual_kv_tiles=prepared.residual_kv_tiles,
            q_scales_per_head=prepared.q_scale.shape[-1],
            route_words=prepared.exact_route.shape[-1],
            block_size=prepared.residual_q_tile,
            group_size=ROUTE_GROUP,
            route_group=ROUTE_GROUP,
            num_warps=8,
            num_stages=1,
        )
    return storage.permute(0, 2, 1, 3)


class SolResidualBackend:
    requires_runtime_context = True
    output_layout = OUTPUT_NHD

    def __init__(
        self,
        config=None,
        *,
        approximate_rejected,
        kitchen=None,
        exact_backend=None,
        projector=None,
        spec=None,
        launcher=None,
        allow_cpu_for_tests=False,
    ):
        self.config = config or HybridSparseConfig()
        if not isinstance(self.config, HybridSparseConfig):
            raise TypeError('config must be HybridSparseConfig')
        self.approximate_rejected = bool(approximate_rejected)
        self.approximate = self.approximate_rejected
        self.spec = spec or SolResidualSpec()
        self.name = (
            'native_int8_%dx%d_sol_residual_%dx%d' % (
                self.spec.exact_q_tile,
                self.spec.exact_kv_tile,
                self.spec.residual_q_tile,
                self.spec.residual_kv_tile,
            )
            if self.approximate_rejected
            else 'native_int8_128x128_hard_control'
        )
        self.projector = projector
        self.exact_backend = exact_backend or SparseKitchenBackend(
            self.config,
            kitchen=kitchen,
            projector=projector,
            q_tile=self.spec.exact_q_tile,
            kv_tile=self.spec.exact_kv_tile,
            allow_cpu_for_tests=allow_cpu_for_tests,
            output_layout=OUTPUT_NHD,
        )
        if (
            self.exact_backend.router.q_tile != self.spec.exact_q_tile
            or self.exact_backend.router.kv_tile != self.spec.exact_kv_tile
        ):
            raise SolResidualError(
                'exact sparse geometry must be %dQ x %dKV'
                % (self.spec.exact_q_tile, self.spec.exact_kv_tile)
            )
        self.launcher = launcher or launch_sol_residual
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)
        self._logged_execution = False

    @property
    def installation_signature(self):
        return (
            self.name,
            self.config.signature,
            self.spec.signature,
            self.exact_backend.installation_signature,
        )

    def _validate_carrier(self, carrier):
        if carrier.q.ndim != 4 or carrier.q.shape != carrier.k.shape:
            raise SolResidualError('Sol residual needs equal HND INT8 Q/K carriers')
        if carrier.q.shape[-1] != self.spec.head_dim:
            raise SolResidualError('Sol residual attention requires head dimension 128')
        if carrier.q.dtype != torch.int8 or carrier.k.dtype != torch.int8:
            raise SolResidualError('Sol residual attention requires INT8 Q/K carriers')
        if carrier.v.dtype != torch.int8 or carrier.v.ndim != 2:
            raise SolResidualError('Sol residual attention requires the INT8 V carrier')
        if int(carrier.cta_k) != self.spec.exact_kv_tile:
            raise SolResidualError(
                'Sol residual carrier KV tile does not match exact attention'
            )
        if carrier.input_dtype != torch.bfloat16:
            raise SolResidualError('Sol residual attention requires BF16 compute output')
        carrier_tensors = (
            carrier.q,
            carrier.k,
            carrier.v,
            carrier.q_scale,
            carrier.k_scale,
            carrier.v_scale,
        )
        if any(not value.is_contiguous() for value in carrier_tensors):
            raise SolResidualError('Sol residual INT8 carriers must be contiguous')
        batch, heads, sequence, head_dim = carrier.q.shape
        exact_tiles = (
            sequence + self.spec.exact_kv_tile - 1
        ) // self.spec.exact_kv_tile
        if tuple(carrier.q_scale.shape) != (
            batch,
            heads,
            ((sequence + CARRIER_Q_TILE - 1) // CARRIER_Q_TILE) * 32,
        ):
            raise SolResidualError('Sol residual Q scale layout is incompatible')
        if tuple(carrier.k_scale.shape) != (batch, heads, exact_tiles * 4):
            raise SolResidualError('Sol residual K scale layout is incompatible')
        if carrier.v.shape[0] != batch * heads * head_dim or carrier.v.shape[1] < sequence:
            raise SolResidualError('Sol residual V carrier layout is incompatible')
        if carrier.v_scale.numel() != batch * heads * head_dim:
            raise SolResidualError('Sol residual V scale layout is incompatible')

    def _finish_prepare(self, exact, carrier, *, layer_index):
        self._validate_carrier(carrier)
        if (
            int(exact.route.q_tile) != self.spec.exact_q_tile
            or int(exact.route.kv_tile) != self.spec.exact_kv_tile
        ):
            raise SolResidualError(
                'exact route geometry does not match Sol residual attention'
            )
        batch, heads, sequence, _head_dim = carrier.q.shape
        exact_q_tiles = (
            sequence + self.spec.exact_q_tile - 1
        ) // self.spec.exact_q_tile
        exact_kv_tiles = (
            sequence + self.spec.exact_kv_tile - 1
        ) // self.spec.exact_kv_tile
        residual_q_tiles = (
            sequence + self.spec.residual_q_tile - 1
        ) // self.spec.residual_q_tile
        residual_kv_tiles = (
            sequence + self.spec.residual_kv_tile - 1
        ) // self.spec.residual_kv_tile
        metadata = dict(exact.metadata)
        metadata.update(
            {
                'sparse_backend': self.name,
                'qkv_carrier': 'kitchen_int8',
                'exact_attention': 'native_int8_%dx%d' % (
                    self.spec.exact_q_tile,
                    self.spec.exact_kv_tile,
                ),
                'rejected_blocks': (
                    'sol_int8_k_mean_v_sum_64x64'
                    if self.approximate_rejected
                    else 'dropped'
                ),
                'source_commit': SOURCE_COMMIT,
                'source_repository': SOURCE_REPOSITORY,
            }
        )

        if not self.approximate_rejected:
            return PreparedSolResidual(
                exact=exact,
                q=None,
                q_scale=None,
                k_mean=None,
                v_sum=None,
                exact_route=None,
                batch=int(batch),
                sequence=int(sequence),
                heads=int(heads),
                exact_q_tile=int(self.spec.exact_q_tile),
                exact_kv_tile=int(self.spec.exact_kv_tile),
                residual_q_tile=int(self.spec.residual_q_tile),
                residual_kv_tile=int(self.spec.residual_kv_tile),
                exact_q_tiles=exact_q_tiles,
                exact_kv_tiles=exact_kv_tiles,
                residual_q_tiles=residual_q_tiles,
                residual_kv_tiles=residual_kv_tiles,
                scale_log2=float(carrier.attention_scale) * 1.4426950408889634,
                layer_index=int(layer_index),
                metadata=metadata,
            )

        exact_route = pack_exact_route(
            exact.route,
            q_tiles=exact_q_tiles,
            kv_tiles=exact_kv_tiles,
        )
        if carrier.q.device.type == 'cuda':
            k_mean, v_sum = _summarize_kv(carrier)
        elif self.allow_cpu_for_tests:
            k_mean, v_sum = _summarize_kv_cpu(carrier)
        else:  # pragma: no cover - guarded by the exact backend
            raise SolResidualError('Sol residual attention requires CUDA tensors')
        return PreparedSolResidual(
            exact=exact,
            q=carrier.q,
            q_scale=carrier.q_scale,
            k_mean=k_mean,
            v_sum=v_sum,
            exact_route=exact_route,
            batch=int(batch),
            sequence=int(sequence),
            heads=int(heads),
            exact_q_tile=int(self.spec.exact_q_tile),
            exact_kv_tile=int(self.spec.exact_kv_tile),
            residual_q_tile=int(self.spec.residual_q_tile),
            residual_kv_tile=int(self.spec.residual_kv_tile),
            exact_q_tiles=exact_q_tiles,
            exact_kv_tiles=exact_kv_tiles,
            residual_q_tiles=residual_q_tiles,
            residual_kv_tiles=residual_kv_tiles,
            scale_log2=float(carrier.attention_scale) * 1.4426950408889634,
            layer_index=int(layer_index),
            metadata=metadata,
        )

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
            raise SolResidualError('Sol residual attention expects equal HND rank-4 Q/K/V')
        if q.shape[-1] != self.spec.head_dim:
            raise SolResidualError('Sol residual attention requires head dimension 128')
        if any(value.dtype != torch.bfloat16 for value in (q, k, v)):
            raise SolResidualError('Sol residual attention requires BF16 Q/K/V')
        if k.device != q.device or v.device != q.device:
            raise SolResidualError('Sol residual Q/K/V devices differ')
        if not self.allow_cpu_for_tests and q.device.type != 'cuda':
            raise SolResidualError('Sol residual attention requires CUDA tensors')

        try:
            exact = self.exact_backend.prepare(
                q,
                k,
                v,
                layer_index=layer_index,
                transformer_options=transformer_options,
            )
        except SparseKitchenError as exc:
            raise SolResidualError('native exact attention preparation failed: %s' % exc) from exc
        return self._finish_prepare(
            exact,
            exact.quantized,
            layer_index=layer_index,
        )

    def prepare_projected(
        self,
        projected,
        *,
        layer_index,
        transformer_options,
    ):
        if not isinstance(projected, PreparedChunkedKitchenQKV):
            raise SolResidualError('Sol residual received an invalid Kitchen QKV carrier')
        try:
            exact = self.exact_backend.prepare_projected(
                projected,
                layer_index=layer_index,
                transformer_options=transformer_options,
            )
        except SparseKitchenError as exc:
            raise SolResidualError('native exact attention preparation failed: %s' % exc) from exc
        return self._finish_prepare(
            exact,
            projected.carrier,
            layer_index=layer_index,
        )

    def execute(self, prepared):
        if not self.approximate_rejected:
            try:
                output = self.exact_backend.execute(prepared.exact)
            finally:
                prepared.release_exact()
        else:
            try:
                exact_output, exact_lse = self.exact_backend.execute_with_lse(
                    prepared.exact
                )
            except SparseKitchenError as exc:
                raise SolResidualError(
                    'native exact attention could not expose softmax state: %s' % exc
                ) from exc
            finally:
                prepared.release_exact()
            output = self.launcher(prepared, exact_output, exact_lse)

        expected = (
            prepared.batch,
            prepared.heads,
            prepared.sequence,
            HEAD_DIM,
        )
        if tuple(output.shape) != expected:
            raise SolResidualError(
                'attention returned shape %s, expected %s'
                % (tuple(output.shape), expected)
            )
        if output.dtype != torch.bfloat16:
            raise SolResidualError('attention returned the wrong dtype')
        if not self._logged_execution:
            logging.info(
                '%s executed %s: layer=%d exact=%dx%d rejected=%s',
                LOG_PREFIX,
                self.name,
                prepared.layer_index,
                prepared.exact_q_tile,
                prepared.exact_kv_tile,
                prepared.metadata['rejected_blocks'],
            )
            self._logged_execution = True
        return output

    def as_status(self):
        return {
            'mode': self.name,
            'video_budget': float(self.config.video_budget),
            'density_mode': self.config.density_mode,
            'kernel': self.name,
            'exact_attention': 'native_int8_%dx%d' % (
                self.spec.exact_q_tile,
                self.spec.exact_kv_tile,
            ),
            'qkv_carrier': 'kitchen_int8',
            'exact_q_tile': int(self.spec.exact_q_tile),
            'exact_kv_tile': int(self.spec.exact_kv_tile),
            'residual_q_tile': (
                int(self.spec.residual_q_tile)
                if self.approximate_rejected else None
            ),
            'residual_kv_tile': (
                int(self.spec.residual_kv_tile)
                if self.approximate_rejected else None
            ),
            'rejected_blocks': (
                'sol_int8_k_mean_v_sum_64x64'
                if self.approximate_rejected else 'dropped'
            ),
            'fused_qkv': self.projector is not None,
            'source_commit': SOURCE_COMMIT,
            'source_repository': SOURCE_REPOSITORY,
        }
