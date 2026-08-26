"""Resolver-to-projector matrix for streamed Sparse Sage and BF16 Triton."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import h3_optimizations.apply as apply_module  # noqa: E402
from h3_optimizations.qkv.policy import (  # noqa: E402
    resolve_qkv_provider as resolve_policy_qkv_provider,
)
from h3_optimizations.plan import (  # noqa: E402
    FUSED_QKV_AUTO,
    FUSED_QKV_FORCE_BF16,
    FUSED_QKV_FORCE_QUANT,
    FUSED_QKV_PRESERVE_BF16,
    H3OptimizationPlan,
    MemoryRequest,
    SparseRequest,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


def sparse_spec():
    return SimpleNamespace(
        signature=("test-sparse",),
        q_tile=128,
        kv_tile=64,
        qk_format="block_int8",
        q_scale_layout="per_q_tile_float32",
        k_scale_layout="per_kv_tile_float32",
        projected_v_format="floating_hnd",
        summary_format="tile_mean",
        v_format="fp16",
        accumulator="f16",
        kernel=lambda *_args: None,
    )


def inventory(source):
    flags = {
        "bf16": (False, False, False, True),
        "convrot": (True, False, False, False),
        "w4a8": (False, True, False, False),
        "fp8": (False, False, True, False),
    }[source]
    convrot, w4a8, fp8, plain = flags
    item = SimpleNamespace(
        convrot_int8_256=convrot,
        w4a8=w4a8,
        fp8=fp8,
        plain_float=plain,
        logical_dtype="torch.bfloat16" if plain else source,
    )
    return SimpleNamespace(
        qkv=(item,),
        qkv_convrot_int8_256=convrot,
        qkv_w4a8=w4a8,
        qkv_fp8=fp8,
        qkv_plain_float=plain,
        homogeneous=lambda name: name == "qkv",
        labels=lambda _name: (source,),
    )


def plan(request):
    return H3OptimizationPlan(
        memory=MemoryRequest(fused_qkv=request),
        sparse=SparseRequest(),
    )


class StreamedConsumerMatrixTests(unittest.TestCase):
    def test_supported_sources_and_precision_modes_select_streamed_q(self):
        requests = (
            FUSED_QKV_AUTO,
            FUSED_QKV_PRESERVE_BF16,
            FUSED_QKV_FORCE_BF16,
            FUSED_QKV_FORCE_QUANT,
        )
        sources = ("bf16", "convrot", "w4a8", "fp8")
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
            device_index=None,
        )
        spec = sparse_spec()

        def backend(config, **kwargs):
            return SimpleNamespace(config=config, **kwargs)

        patches = (
            mock.patch.object(apply_module, "SPARSE_TRITON_AVAILABLE", True),
            mock.patch.object(apply_module, "preflight_sparse_sage", return_value=spec),
            mock.patch.object(apply_module, "preflight_triton_sparse", return_value=SimpleNamespace(signature=("test-triton",))),
            mock.patch.object(apply_module, "HybridSparseBackend", side_effect=backend),
            mock.patch.object(apply_module, "TritonSparseBackend", side_effect=backend),
            mock.patch.object(apply_module, "_fp8_execution_available", return_value=True),
            mock.patch.object(
                apply_module,
                "resolve_qkv_provider",
                resolve_policy_qkv_provider,
            ),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            for source in sources:
                for request in requests:
                    for consumer in ("sage", "triton"):
                        with self.subTest(
                            source=source,
                            request=request,
                            consumer=consumer,
                        ):
                            if consumer == "sage":
                                attention, _qkv = apply_module._resolve_sparse(
                                    plan(request),
                                    environment,
                                    inventory(source),
                                )
                            else:
                                attention, _qkv = apply_module._resolve_triton_sparse(
                                    plan(request),
                                    environment,
                                    inventory(source),
                                    None,
                                )

                            self.assertIsNotNone(attention.projector)
                            self.assertTrue(attention.projector.streamed_q)
                            expected_mode = "native"
                            if request == FUSED_QKV_FORCE_BF16:
                                expected_mode = "force_bf16"
                            elif request == FUSED_QKV_FORCE_QUANT and source == "bf16":
                                expected_mode = (
                                    "force_fp8" if consumer == "sage" else "force_int8"
                                )
                            self.assertEqual(
                                attention.projector.projection_mode,
                                expected_mode,
                            )


if __name__ == "__main__":
    unittest.main()
