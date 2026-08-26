"""SM86 prepared-QKV Sage backend."""

from dataclasses import dataclass

import torch

from .. import stats
from ..sage_mem_eff import EfficientSageError
from .common import (
    ArchitectureBackend,
    guard_signed_offsets,
    independent_contiguous,
    load_core,
)


@dataclass(frozen=True)
class SM86API:
    version: str
    quantizer: object
    attention: object


class SageSM86MemoryEfficientBackend(ArchitectureBackend):
    """Sage's Triton per-block INT8 Q/K and FP16-V path."""

    name = "sage_mem_eff_sm86"
    capabilities = frozenset({(8, 6)})
    projected_qkv_format = "sage_per_block_128_64"
    projected_q_tile = 128
    projected_k_tile = 64

    def __init__(
        self,
        api=None,
        allow_cpu_for_tests=False,
    ):
        super().__init__(
            allow_cpu_for_tests=allow_cpu_for_tests
        )
        if api is None:
            version, core = load_core()
            quantizer = getattr(
                core,
                "per_block_int8_triton",
                None,
            )
            attention = getattr(core, "attn_false", None)
            if (
                not callable(quantizer)
                or not callable(attention)
            ):
                raise EfficientSageError(
                    "SageAttention 2.2.x lacks the SM86 "
                    "Triton per-block path"
                )
            api = SM86API(
                version,
                quantizer,
                attention,
            )
        self.api = api

    def quantize_projected_qk(self, q, k):
        return self.api.quantizer(
            q,
            k,
            km=None,
            BLKQ=128,
            BLKK=64,
            sm_scale=q.shape[-1] ** -0.5,
            tensor_layout="HND",
        )

    def prepare(
        self,
        q,
        k,
        v,
        *,
        layer_index,
        transformer_options,
    ):
        _, heads, sequence, head_dim = self.validate(
            q,
            k,
            v,
        )
        q_source = guard_signed_offsets(q)
        k_source = guard_signed_offsets(k)
        q_int8, q_scale, k_int8, k_scale = self.quantize_projected_qk(
            q_source,
            k_source,
        )
        v_source = independent_contiguous(v)
        del q_source, k_source
        self.log_once(
            self.api.version,
            "HND, per-block INT8 Q/K, deferred FP16 V, "
            "Triton attention",
        )
        return self.prepared(
            q,
            q_int8,
            q_scale,
            k_int8,
            k_scale,
            v_source,
            layer_index=layer_index,
            heads=heads,
            sequence=sequence,
            head_dim=head_dim,
        )

    def execute(self, prepared):
        v_source = prepared.v_source
        v_fp16 = (
            v_source
            if v_source.dtype == torch.float16
            else v_source.to(torch.float16)
        )
        prepared.v_source = None
        del v_source
        try:
            output, _ = self.api.attention(
                prepared.q_int8,
                prepared.k_int8,
                v_fp16,
                prepared.q_scale,
                prepared.k_scale,
                tensor_layout="HND",
                attn_mask=None,
                output_dtype=prepared.output_dtype,
                return_lse=False,
            )
        except Exception as exc:
            self.kernel_error(
                prepared,
                "triton_attn_qk_int8_per_block",
                exc,
            )
        stats.increment("executed")
        return output
