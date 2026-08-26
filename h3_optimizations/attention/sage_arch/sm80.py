"""SM80 prepared-QKV Sage backend."""

from dataclasses import dataclass

import torch

from .. import stats
from ..sage_mem_eff import EfficientSageError
from ..triton_i64 import per_thread_int8_i64
from .common import (
    ArchitectureBackend,
    KernelBinding,
    independent_contiguous,
    load_core,
    resolve_kernel,
)

KERNEL_NAMES = ("qk_int8_sv_f16_accum_f32_attn",)


@dataclass(frozen=True)
class SM80API:
    version: str
    kernel: KernelBinding


class SageSM80MemoryEfficientBackend(ArchitectureBackend):
    """Per-thread INT8 Q/K and FP16 V with FP32 accumulation."""

    name = "sage_mem_eff_sm80"
    capabilities = frozenset({(8, 0), (8, 7)})
    projected_qkv_format = "sage_per_thread_128_64"
    projected_q_tile = 128
    projected_k_tile = 64
    requires_h3_triton = True

    def __init__(
        self,
        api=None,
        quantizer=None,
        allow_cpu_for_tests=False,
    ):
        super().__init__(
            allow_cpu_for_tests=allow_cpu_for_tests
        )
        if api is None:
            version, core = load_core()
            if not getattr(core, "SM80_ENABLED", False):
                raise EfficientSageError(
                    "SageAttention's SM80 CUDA extension "
                    "is unavailable"
                )
            kernel = resolve_kernel(
                core,
                "sm80",
                KERNEL_NAMES,
                ("sageattn_qk_int8_pv_fp16_cuda",),
            )
            api = SM80API(version, kernel)
        self.api = api
        self.quantizer = quantizer or per_thread_int8_i64

    def quantize_projected_qk(self, q, k):
        return self.quantizer(
            q,
            k,
            None,
            BLKQ=128,
            WARPQ=32,
            BLKK=64,
            WARPK=64,
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
        q_int8, q_scale, k_int8, k_scale = self.quantize_projected_qk(q, k)
        v_source = independent_contiguous(v)
        self.log_once(
            self.api.version,
            "HND, per-thread INT8 Q/K, deferred FP16 V, "
            "kernel=%s via %s"
            % (
                self.api.kernel.name,
                self.api.kernel.source,
            ),
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
        v_carrier, v_scale = self.prepare_streamed_v(prepared.v_source)
        prepared.v_source = None
        return self.execute_rectangular(
            prepared.q_int8,
            prepared.q_scale,
            prepared.k_int8,
            prepared.k_scale,
            v_carrier,
            v_scale,
            output_dtype=prepared.output_dtype,
            softmax_scale=prepared.softmax_scale,
            layer_index=prepared.layer_index,
            prepared=prepared,
        )

    def prepare_streamed_v(self, v_source):
        v_fp16 = (
            v_source
            if v_source.dtype == torch.float16
            else v_source.to(torch.float16)
        )
        return v_fp16, None

    def execute_rectangular(
        self,
        q_int8,
        q_scale,
        k_int8,
        k_scale,
        v_carrier,
        v_scale,
        *,
        output_dtype,
        softmax_scale,
        layer_index,
        prepared=None,
    ):
        del v_scale
        output = torch.empty(
            q_int8.shape,
            dtype=output_dtype,
            device=q_int8.device,
        )
        try:
            self.api.kernel.fn(
                q_int8,
                k_int8,
                v_carrier,
                output,
                q_scale,
                k_scale,
                1,
                0,
                3,
                softmax_scale,
                0,
            )
        except Exception as exc:
            if prepared is not None:
                self.kernel_error(prepared, self.api.kernel.name, exc)
            raise EfficientSageError(
                "%s rectangular kernel failed at layer %d" % (self.name, layer_index)
            ) from exc
        stats.increment("executed")
        return output
