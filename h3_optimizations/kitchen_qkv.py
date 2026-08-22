'''Chunked H3 QKV production through Comfy Kitchen's public carrier API.'''

from __future__ import annotations

from dataclasses import dataclass

import torch

import comfy.model_management
import comfy.quant_ops

from .attention_forward import project_qkv
from .dense_resolver import is_installed_dense_attention
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


def producer_api_available(kitchen=None, device=None):
    kitchen = comfy.quant_ops.ck if kitchen is None else kitchen
    return (
        all(hasattr(kitchen, name) for name in PRODUCER_API)
        and kitchen.INT8_ATTENTION_PRODUCER_ABI_VERSION == PRODUCER_ABI_VERSION
        and bool(kitchen.int8_attention_producer_is_available(device))
    )


@dataclass(frozen=True)
class PreparedChunkedKitchenQKV:
    carrier: object


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
):
    del layer_index, transformer_options
    kitchen = comfy.quant_ops.ck
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
            kitchen.finalize_int8_attention_producer(producer)
        )
    finally:
        if held is not None:
            held.__exit__(None, None, None)


class ChunkedKitchenQKVProjector:
    name = 'chunked_kitchen_qkv'

    def __init__(self, chunk_rows=CHUNK_ROWS, fp8_projection=False):
        self.chunk_rows = int(chunk_rows)
        self.fp8_projection = bool(fp8_projection)

    @property
    def installation_signature(self):
        return (self.name, self.chunk_rows, self.fp8_projection)

    def try_project(
        self,
        module,
        x,
        rope_freqs,
        *,
        layer_index,
        transformer_options,
    ):
        kitchen = comfy.quant_ops.ck
        fmt = describe_linear(module.qkv_proj)
        format_ok = (
            fmt.fp8 or fmt.plain_float
            if self.fp8_projection
            else (fmt.convrot_int8_256 or fmt.w4a8)
        )
        if (
            not is_installed_dense_attention(transformer_options)
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
    name = 'comfy_kitchen_int8_prequantized'

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
        return comfy.quant_ops.ck.int8_attention_from_prequantized(
            prepared.carrier
        )
