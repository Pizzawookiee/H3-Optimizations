'''Chunked H3 QKV production through Comfy Kitchen's public carrier API.'''

from __future__ import annotations

from dataclasses import dataclass

import torch

import comfy.model_management
import comfy.quant_ops

from .attention_forward import project_qkv
from .qkv.chunked import project_chunk_hnd
from .qkv.formats import describe_linear
from .qkv.fp8 import FP8BindingError, HeldFP8QKV
from .qkv.w4a8 import HeldW4A8QKV, W4A8BindingError


CHUNK_ROWS = 4096
PRODUCER_ABI_VERSION = 1
PRODUCER_ABI = 'external_v%d' % PRODUCER_ABI_VERSION
PRODUCER_API = (
    'INT8_ATTENTION_PRODUCER_ABI_VERSION',
    'Int8AttentionProducerUnavailableError',
    'int8_attention_producer_is_available',
    'int8_attention_producer_spec',
    'select_int8_attention_k_anchor',
    'create_int8_attention_producer',
    'quantize_int8_attention_qk_chunk',
    'quantize_int8_attention_v',
    'finalize_int8_attention_producer',
    'int8_attention_from_prequantized',
)
H3_ATTENTION_BACKEND_KEY = 'h3_optimizations_attention_backend'
DENSE_KITCHEN_BACKEND = 'comfy_kitchen_int8_prequantized'


class FusedQKVError(RuntimeError):
    pass


def resolve_kitchen(device=None):
    """The module providing the producer API, vendored first.

    The vendored library ships with this pack. comfy_kitchen is whatever pip
    installed, and ComfyUI pins 0.2.31, which has no producer API at all --
    which is why this whole path was integrated and then never ran.
    """
    from .native import int8_attention as _  # noqa: F401 - import check

    from . import native

    if _supports_producer(native, device):
        return native
    kitchen = comfy.quant_ops.ck
    return kitchen if _supports_producer(kitchen, device) else None


def _supports_producer(module, device):
    if module is None:
        return False
    if not all(hasattr(module, name) for name in PRODUCER_API):
        return False
    if module.INT8_ATTENTION_PRODUCER_ABI_VERSION != PRODUCER_ABI_VERSION:
        return False
    try:
        return bool(module.int8_attention_producer_is_available(device))
    except Exception:  # noqa: BLE001 - an unavailable backend is not an error
        return False


def producer_api_available(kitchen=None, device=None):
    if kitchen is not None:
        return _supports_producer(kitchen, device)
    return resolve_kitchen(device) is not None


@dataclass(frozen=True)
class PreparedChunkedKitchenQKV:
    carrier: object
    q_summary: torch.Tensor | None = None
    k_summary: torch.Tensor | None = None


def _rope_rows(rope_freqs, rows):
    if rope_freqs is None:
        return None
    return rope_freqs.index_select(1, rows)


def _project_anchor_samples(module, x, rope_freqs, positions, projector=None):
    rows = torch.tensor(positions, dtype=torch.int64, device=x.device)
    if projector is not None:
        _q, k, _v = projector.project_rows(x, rope_freqs, rows)
        return k
    sample_x = x.index_select(0, rows)
    sample_rope = _rope_rows(rope_freqs, rows)
    _q, k, _v = project_qkv(module, sample_x, sample_rope)
    return k.transpose(0, 1).unsqueeze(0)


def _tile_mean(x, tile):
    full = x.shape[-2] // tile
    remainder = x.shape[-2] % tile
    pieces = []
    if full:
        pieces.append(
            x[..., :full * tile, :]
            .reshape(*x.shape[:-2], full, tile, x.shape[-1])
            .mean(dim=-2)
        )
    if remainder:
        pieces.append(x[..., full * tile:, :].mean(dim=-2, keepdim=True))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)


def run_chunked_kitchen_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    transformer_options,
    spec,
    chunk_rows=CHUNK_ROWS,
    fp8_projection=False,
    routing_summaries=False,
):
    del layer_index, transformer_options
    kitchen = resolve_kitchen(x.device)
    if kitchen is None:
        raise FusedQKVError('no INT8 attention producer is available')
    held = None
    try:
        fmt = describe_linear(module.qkv_proj)
        if fp8_projection:
            held = HeldFP8QKV(
                module,
                x[:1],
                allow_float_conversion=fmt.plain_float,
            )
            held.__enter__()
        elif fmt.w4a8:
            held = HeldW4A8QKV(module, x[:1])
            held.__enter__()

        samples = _project_anchor_samples(
            module,
            x,
            rope_freqs,
            spec.k_anchor_positions,
            projector=held,
        )
        anchor = kitchen.select_int8_attention_k_anchor(spec, samples)
        del samples
        producer = kitchen.create_int8_attention_producer(spec, anchor)
        del anchor

        sequence = int(x.shape[0])
        retained_v = None
        q_summaries = []
        k_summaries = []
        for start in range(0, sequence, int(chunk_rows)):
            end = min(start + int(chunk_rows), sequence)
            q, k, v = project_chunk_hnd(
                module,
                x,
                rope_freqs,
                start,
                end,
                projector=held,
            )
            if retained_v is None:
                retained_v = v.new_empty(
                    (1, int(module.heads), sequence, int(module.head_dim))
                )
            if routing_summaries:
                q_summaries.append(_tile_mean(q, int(spec.q_tile)))
                k_summaries.append(_tile_mean(k, int(spec.k_tile)))
            kitchen.quantize_int8_attention_qk_chunk(
                producer,
                q,
                k,
                q_start=start,
                k_start=start,
            )
            retained_v[:, :, start:end, :].copy_(v)
            del q, k, v

        kitchen.quantize_int8_attention_v(producer, retained_v)
        del retained_v
        return PreparedChunkedKitchenQKV(
            kitchen.finalize_int8_attention_producer(producer),
            q_summary=(
                torch.cat(q_summaries, dim=-2) if q_summaries else None
            ),
            k_summary=(
                torch.cat(k_summaries, dim=-2) if k_summaries else None
            ),
        )
    finally:
        if held is not None:
            held.__exit__(None, None, None)


class ChunkedKitchenQKVProjector:
    name = 'chunked_kitchen_qkv'

    def __init__(
        self,
        chunk_rows=CHUNK_ROWS,
        fp8_projection=False,
        routing_summaries=False,
    ):
        self.chunk_rows = int(chunk_rows)
        self.fp8_projection = bool(fp8_projection)
        self.routing_summaries = bool(routing_summaries)

    @property
    def installation_signature(self):
        return (
            self.name,
            self.chunk_rows,
            self.fp8_projection,
            self.routing_summaries,
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
        kitchen = resolve_kitchen(x.device)
        if kitchen is None:
            return None
        fmt = describe_linear(module.qkv_proj)
        format_ok = (
            fmt.fp8 or fmt.plain_float
            if self.fp8_projection
            else (fmt.convrot_int8_256 or fmt.w4a8)
        )
        owns_dense_h3 = (
            (transformer_options or {}).get(H3_ATTENTION_BACKEND_KEY)
            == DENSE_KITCHEN_BACKEND
        )
        if (
            (not self.routing_summaries and not owns_dense_h3)
            or comfy.model_management.in_training
            or x.ndim != 2
            or not x.is_cuda
            or not format_ok
            or not producer_api_available(kitchen, x.device)
        ):
            return None

        shape = (1, int(module.heads), int(x.shape[0]), int(module.head_dim))
        try:
            spec = kitchen.int8_attention_producer_spec(
                shape,
                shape,
                dtype=x.dtype,
                device=x.device,
            )
        except kitchen.Int8AttentionProducerUnavailableError:
            return None
        if (
            getattr(spec, 'abi_version', None) != PRODUCER_ABI_VERSION
            or self.chunk_rows % int(spec.sequence_alignment)
        ):
            return None
        try:
            return run_chunked_kitchen_qkv(
                module,
                x,
                rope_freqs,
                layer_index=layer_index,
                transformer_options=transformer_options,
                spec=spec,
                chunk_rows=self.chunk_rows,
                fp8_projection=self.fp8_projection,
                routing_summaries=self.routing_summaries,
            )
        except (
            FP8BindingError,
            W4A8BindingError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            if not self.fp8_projection and not fmt.w4a8:
                raise
            return None


class ChunkedKitchenAttentionBackend:
    name = DENSE_KITCHEN_BACKEND

    @property
    def installation_signature(self):
        return (self.name,)

    def prepare(self, *_args, **_kwargs):
        raise RuntimeError('chunked Kitchen attention requires its QKV producer')

    def prepare_projected(
        self,
        projected,
        *,
        layer_index,
        transformer_options,
    ):
        del layer_index, transformer_options
        if not isinstance(projected, PreparedChunkedKitchenQKV):
            raise TypeError('chunked Kitchen attention received an invalid carrier')
        return projected

    def execute(self, prepared):
        if not isinstance(prepared, PreparedChunkedKitchenQKV):
            raise TypeError('chunked Kitchen attention received an invalid carrier')
        # The carrier has to be consumed by the module that produced it:
        # the two implementations agree byte for byte, but only one of them is
        # guaranteed to be present.
        kitchen = resolve_kitchen()
        if kitchen is None:
            raise FusedQKVError('no INT8 attention producer is available')
        return kitchen.int8_attention_from_prequantized(prepared.carrier)
