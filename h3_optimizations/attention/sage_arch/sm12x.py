"""SM120/121 prepared-QKV Sage backend."""

from dataclasses import dataclass

import torch

from .. import stats
from ..sage_mem_eff import EfficientSageError
from ..sage_v_fp8 import TRITON_AVAILABLE, prepare_sage_v_fp8
from .common import (
    ArchitectureBackend,
    KernelBinding,
    guard_signed_offsets,
    independent_contiguous,
    load_core,
    resolve_kernel,
)

KERNEL_NAMES = (
    "qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf",
)


@dataclass(frozen=True)
class SM12xAPI:
    version: str
    per_warp_int8: object
    per_channel_fp8: object
    kernel: KernelBinding


class SageSM12xMemoryEfficientBackend(ArchitectureBackend):
    """SM100/120/121 per-warp Q/K and SM89-family FP8 kernel."""

    name = "sage_mem_eff_sm12x"
    capabilities = frozenset({(10, 0), (12, 0), (12, 1)})
    projected_qkv_format = "sage_per_warp_128_64"
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
            per_warp = getattr(
                core,
                "per_warp_int8_cuda",
                None,
            )
            per_channel_fp8 = getattr(
                core,
                "per_channel_fp8",
                None,
            )
            if (
                not callable(per_warp)
                or not callable(per_channel_fp8)
            ):
                raise EfficientSageError(
                    "SageAttention 2.2.x lacks Blackwell's "
                    "per-warp/FP8 path"
                )
            kernel = resolve_kernel(
                core,
                "sm89",
                KERNEL_NAMES,
                ("sageattn_qk_int8_pv_fp8_cuda",),
            )
            api = SM12xAPI(
                version,
                per_warp,
                per_channel_fp8,
                kernel,
            )
        self.api = api

    def quantize_projected_qk(self, q, k):
        return self.api.per_warp_int8(
            q,
            k,
            None,
            tensor_layout="HND",
            BLKQ=128,
            WARPQ=32,
            BLKK=64,
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
            "HND, per-warp INT8 Q/K, deferred FP8 V, "
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
        return prepare_sage_v_fp8(
            v_source,
            self.api.per_channel_fp8,
            scale_max=2.25,
        )

    def v_staging_parameters(self):
        return (2.25, 64) if TRITON_AVAILABLE else None

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
                2,
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
