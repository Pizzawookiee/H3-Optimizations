"""SM90 prepared-QKV Sage backend."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

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

KERNEL_NAMES = (
    "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
)


@dataclass(frozen=True)
class SM90API:
    version: str
    per_channel_fp8: object
    kernel: KernelBinding


class SageSM90MemoryEfficientBackend(ArchitectureBackend):
    """Per-thread INT8 Q/K and FP8 V with FP32+FP32."""

    name = "sage_mem_eff_sm90"
    capabilities = frozenset({(9, 0)})
    projected_qkv_format = "sage_per_thread_64_128"
    projected_q_tile = 64
    projected_k_tile = 128
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
            if not getattr(core, "SM90_ENABLED", False):
                raise EfficientSageError(
                    "SageAttention's SM90 CUDA extension "
                    "is unavailable"
                )
            per_channel_fp8 = getattr(
                core,
                "per_channel_fp8",
                None,
            )
            if not callable(per_channel_fp8):
                raise EfficientSageError(
                    "SageAttention 2.2.x lacks per_channel_fp8"
                )
            kernel = resolve_kernel(
                core,
                "sm90",
                KERNEL_NAMES,
                ("sageattn_qk_int8_pv_fp8_cuda_sm90",),
            )
            api = SM90API(
                version,
                per_channel_fp8,
                kernel,
            )
        self.api = api
        self.quantizer = quantizer or per_thread_int8_i64

    def quantize_projected_qk(self, q, k):
        return self.quantizer(
            q,
            k,
            None,
            BLKQ=64,
            WARPQ=16,
            BLKK=128,
            WARPK=128,
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
            "HND, per-thread INT8 Q/K, deferred FP8 V, "
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
        pad_rows = (-int(v_source.shape[-2])) % 128
        if pad_rows:
            v_source = F.pad(
                v_source,
                (0, 0, 0, pad_rows),
            )
        v_fp8, v_scale, _ = self.api.per_channel_fp8(
            v_source,
            tensor_layout="HND",
            scale_max=448.0,
            smooth_v=False,
        )
        return v_fp8, v_scale

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
                v_scale,
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
