'''CPU policy contracts for ROCm + ConvRot-256 with public Auto defaults.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import h3_optimizations.apply_policy as apply_policy  # noqa: E402
apply_module = apply_policy._base
from h3_optimizations.attention.sparse import TritonSparseSpec  # noqa: E402
from h3_optimizations.memory_migration_node import (  # noqa: E402
    PRECISION_MODE_AUTO,
    QKV_STREAMING_MODE_AUTO,
    _memory_request_for_modes,
)
from h3_optimizations.plan import (  # noqa: E402
    ATTENTION_EXISTING,
    FUSED_QKV_AUTO,
    H3OptimizationPlan,
    MLP_MEMORY_AUTO,
    QKV_STREAMING_AUTO,
    SparseRequest,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    MLP_CONVROT_INT8_TWO_SLICE,
    QKV_BF16_CHUNKED,
    QKV_STANDARD,
    QKV_TRITON_SPARSE_CHUNKED,
    QKVProviderResolution,
)
from h3_optimizations.qkv.streamed import PROJECTION_NATIVE  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def rocm_environment():
    return SimpleNamespace(
        cuda_available=False,
        capability=None,
        device_index=0,
        device_name='fake AMD',
        backend='rocm',
        architecture='rocm',
    )


def convrot_inventory():
    return SimpleNamespace(
        qkv=(object(),),
        fc1=(object(),),
        fc2=(object(),),
        qkv_convrot_int8_256=True,
        qkv_w4a8=False,
        qkv_fp8=False,
        qkv_plain_float=False,
        mlp_convrot_int8_256=True,
        mlp_w4a8=False,
        mlp_fp8=False,
        mlp_plain_float=False,
        homogeneous=lambda name: name in ('qkv', 'fc1', 'fc2'),
        labels=lambda _name: ('TensorWiseINT8Layout+convrot256',),
    )


def auto_memory_request():
    return _memory_request_for_modes(
        fused_qkv=FUSED_QKV_AUTO,
        mlp_memory=MLP_MEMORY_AUTO,
        chunk_rows=4096,
        precision_mode=PRECISION_MODE_AUTO,
        qkv_streaming_mode=QKV_STREAMING_MODE_AUTO,
    )


class RocmConvRotAutoTests(unittest.TestCase):
    def test_public_defaults_map_to_preserve_existing_attention_and_auto_memory(self):
        memory = auto_memory_request()
        sparse = SparseRequest()

        self.assertEqual(memory.attention, ATTENTION_EXISTING)
        self.assertEqual(memory.fused_qkv, FUSED_QKV_AUTO)
        self.assertEqual(memory.mlp_memory, MLP_MEMORY_AUTO)
        self.assertEqual(memory.qkv_streaming, QKV_STREAMING_AUTO)
        self.assertFalse(memory.mlp_strict)
        self.assertEqual(sparse.backend, 'auto')

    def test_rocm_auto_never_enables_fp8_memory_conversion(self):
        self.assertFalse(
            apply_module._fp8_execution_available(rocm_environment())
        )

    def test_convrot_auto_keeps_native_qkv_and_mlp_formats(self):
        environment = rocm_environment()
        inventory = convrot_inventory()
        with mock.patch.object(
            apply_policy.RuntimeEnvironment,
            'detect',
            return_value=environment,
        ):
            qkv = apply_policy.resolve_qkv_provider(
                inventory,
                request=FUSED_QKV_AUTO,
                backend_kind=apply_module.ATTENTION_TRITON_SPARSE,
                triton_available=True,
                memory_optimize=True,
                fp8_available=False,
            )
            mlp = apply_policy.resolve_mlp_provider(
                inventory,
                request=MLP_MEMORY_AUTO,
                fp8_available=False,
            )

        self.assertEqual(qkv.provider_id, QKV_TRITON_SPARSE_CHUNKED)
        self.assertTrue(qkv.fused)
        self.assertIn('checkpoint-native ConvRot-256 INT8', qkv.reason)
        self.assertEqual(mlp.provider_id, MLP_CONVROT_INT8_TWO_SLICE)
        self.assertEqual(mlp.activation_mode, 'mlp_chunked_convrot_2slice')

    def test_memory_only_auto_streams_convrot_without_requantizing(self):
        environment = rocm_environment()
        inventory = convrot_inventory()
        plan = H3OptimizationPlan(memory=auto_memory_request())
        model = SimpleNamespace(model_options={})
        dense = SimpleNamespace(
            requested='existing',
            selected='existing',
            backend=None,
            reason='synthetic existing attention',
            backend_kind='existing',
        )

        with mock.patch.object(
            apply_policy.RuntimeEnvironment,
            'detect',
            return_value=environment,
        ), mock.patch.object(
            apply_module,
            'resolve_current_dense_attention',
            return_value=dense,
        ), mock.patch.object(
            apply_module,
            'producer_api_available',
            return_value=False,
        ):
            attention, qkv = apply_module._resolve_dense(
                plan,
                model,
                inventory,
                environment,
            )

        self.assertEqual(qkv.provider_id, QKV_BF16_CHUNKED)
        self.assertIn('checkpoint-native ConvRot-256 INT8', qkv.reason)
        self.assertEqual(attention.projector.name, 'streamed_dense_bf16_qkv')
        self.assertEqual(attention.projector.projection_mode, PROJECTION_NATIVE)
        self.assertTrue(attention.projector.streamed_q)

    def test_rocm_convrot_triton_uses_native_streamed_projection(self):
        environment = rocm_environment()
        inventory = convrot_inventory()
        plan = H3OptimizationPlan(
            memory=auto_memory_request(),
            sparse=SparseRequest(),
        )

        with mock.patch.object(
            apply_policy.RuntimeEnvironment,
            'detect',
            return_value=environment,
        ), mock.patch.object(
            apply_module,
            'preflight_triton_sparse',
            return_value=TritonSparseSpec(),
        ):
            attention, qkv = apply_module._resolve_triton_sparse(
                plan,
                environment,
                inventory,
                'CUDA-only sparse backends unavailable on ROCm',
            )

        self.assertEqual(attention.selected, apply_module.ATTENTION_TRITON_SPARSE)
        self.assertEqual(qkv.provider_id, QKV_TRITON_SPARSE_CHUNKED)
        self.assertEqual(attention.projector.projection_mode, PROJECTION_NATIVE)
        self.assertTrue(attention.projector.streamed_q)
        self.assertFalse(attention.projector.force_weights_int8)
        self.assertIs(attention.backend.projector, attention.projector)

    def test_rocm_auto_reaches_triton_before_flex_for_convrot(self):
        environment = rocm_environment()
        inventory = convrot_inventory()
        plan = H3OptimizationPlan(
            memory=auto_memory_request(),
            sparse=SparseRequest(),
        )
        dense_attention = apply_module.ResolvedAttention(
            requested='existing',
            selected='existing',
            backend=None,
            reason='dense fallback',
            backend_kind='existing',
        )
        dense_qkv = QKVProviderResolution(QKV_STANDARD, False, 'dense fallback')

        with mock.patch.object(
            apply_policy.RuntimeEnvironment,
            'detect',
            return_value=environment,
        ), mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=(dense_attention, dense_qkv),
        ), mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            side_effect=apply_module.SparseKitchenError('CUDA-only Kitchen'),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=apply_module.SparseSageError('CUDA-only Sparse Sage'),
        ), mock.patch.object(
            apply_module,
            'preflight_triton_sparse',
            return_value=TritonSparseSpec(),
        ), mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
        ) as flex:
            attention, qkv = apply_module._resolve_attention(
                plan,
                object(),
                inventory,
                environment,
            )

        self.assertEqual(attention.selected, apply_module.ATTENTION_TRITON_SPARSE)
        self.assertEqual(qkv.provider_id, QKV_TRITON_SPARSE_CHUNKED)
        flex.assert_not_called()


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
