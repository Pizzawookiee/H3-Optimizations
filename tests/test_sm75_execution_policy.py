'''CPU contracts for Turing-safe automatic execution policy.'''

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

import h3_optimizations.apply_policy as policy  # noqa: E402
from h3_optimizations.attention.sparse.kitchen_sparse import (  # noqa: E402
    SparseKitchenError,
    preflight_sparse_kitchen,
)
from h3_optimizations.plan import (  # noqa: E402
    FUSED_QKV_AUTO,
    FUSED_QKV_FORCE_BF16,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_PRESERVE,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    MLP_FLOAT_CHUNKED,
    MLP_RUNTIME_CONVROT_INT8_CHUNKED,
    QKV_FORCE_BF16_STREAMED_KITCHEN,
    QKV_FORCE_CONVROT_INT8_KITCHEN,
    QKV_STANDARD,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


def qkv_inventory(*, plain_float=False, convrot=False):
    return SimpleNamespace(
        qkv=(object(),),
        qkv_plain_float=bool(plain_float),
        qkv_convrot_int8_256=bool(convrot),
        qkv_w4a8=False,
        qkv_fp8=False,
        homogeneous=lambda name: name == 'qkv',
        labels=lambda _name: ('synthetic',),
    )


def mlp_inventory(*, plain_float=False, convrot=False):
    return SimpleNamespace(
        fc1=(object(),),
        fc2=(object(),),
        mlp_plain_float=bool(plain_float),
        mlp_convrot_int8_256=bool(convrot),
        mlp_w4a8=False,
        mlp_fp8=False,
        homogeneous=lambda name: name in ('fc1', 'fc2'),
        labels=lambda _name: ('synthetic',),
    )


class SM75ExecutionPolicyTests(unittest.TestCase):
    def test_kitchen_sparse_preflight_accepts_sm75_and_rejects_sm70(self):
        kitchen = SimpleNamespace(
            SPARSE_GEOMETRIES=((64, 64),),
            int8_attention_is_available=lambda: True,
        )
        selected = preflight_sparse_kitchen(
            cuda_available=lambda: True,
            capability_getter=lambda: (7, 5),
            kitchen=kitchen,
            q_tile=64,
            kv_tile=64,
        )
        self.assertIs(selected, kitchen)

        with self.assertRaisesRegex(SparseKitchenError, '7.5'):
            preflight_sparse_kitchen(
                cuda_available=lambda: True,
                capability_getter=lambda: (7, 0),
                kitchen=kitchen,
                q_tile=64,
                kv_tile=64,
            )

    def test_sm75_auto_plain_float_uses_direct_kitchen_convrot(self):
        with mock.patch.object(policy, '_current_capability', return_value=(7, 5)):
            resolved = policy.resolve_qkv_provider(
                qkv_inventory(plain_float=True),
                request=FUSED_QKV_AUTO,
                backend_kind='sparse_kitchen_int8',
                kitchen_producer_available=True,
                memory_optimize=True,
                fp8_available=False,
            )
        self.assertEqual(resolved.provider_id, QKV_FORCE_CONVROT_INT8_KITCHEN)
        self.assertTrue(resolved.fused)

    def test_sm75_auto_plain_float_keeps_generic_dense_qkv_upstream(self):
        with mock.patch.object(policy, '_current_capability', return_value=(7, 5)):
            resolved = policy.resolve_qkv_provider(
                qkv_inventory(plain_float=True),
                request=FUSED_QKV_AUTO,
                backend_kind='existing',
                memory_optimize=True,
                fp8_available=False,
            )
        self.assertEqual(resolved.provider_id, QKV_STANDARD)
        self.assertIn('upstream FP16', resolved.reason)

    def test_sm75_explicit_bf16_is_not_silently_rewritten(self):
        with mock.patch.object(policy, '_current_capability', return_value=(7, 5)):
            resolved = policy.resolve_qkv_provider(
                qkv_inventory(plain_float=True),
                request=FUSED_QKV_FORCE_BF16,
                backend_kind='sparse_kitchen_int8',
                kitchen_producer_available=True,
                memory_optimize=True,
                fp8_available=False,
            )
        self.assertEqual(resolved.provider_id, QKV_FORCE_BF16_STREAMED_KITCHEN)

    def test_sm75_auto_unknown_quantized_qkv_stays_upstream(self):
        unknown = qkv_inventory()
        with mock.patch.object(policy, '_current_capability', return_value=(7, 5)):
            resolved = policy.resolve_qkv_provider(
                unknown,
                request=FUSED_QKV_AUTO,
                backend_kind='sparse_kitchen_int8',
                kitchen_producer_available=True,
                memory_optimize=True,
                fp8_available=False,
            )
        self.assertEqual(resolved.provider_id, QKV_STANDARD)
        self.assertIn('dequantize safely', resolved.reason)

    def test_sm80_auto_policy_is_unchanged(self):
        with mock.patch.object(policy, '_current_capability', return_value=(8, 0)):
            resolved = policy.resolve_qkv_provider(
                qkv_inventory(plain_float=True),
                request=FUSED_QKV_AUTO,
                backend_kind='sparse_kitchen_int8',
                kitchen_producer_available=True,
                memory_optimize=True,
                fp8_available=False,
            )
        self.assertNotEqual(resolved.provider_id, QKV_FORCE_CONVROT_INT8_KITCHEN)

    def test_sm75_auto_plain_float_mlp_uses_runtime_convrot(self):
        with mock.patch.object(policy, '_current_capability', return_value=(7, 5)):
            resolved = policy.resolve_mlp_provider(
                mlp_inventory(plain_float=True),
                request=MLP_MEMORY_AUTO,
                fp8_available=False,
            )
        self.assertEqual(
            resolved.provider_id,
            MLP_RUNTIME_CONVROT_INT8_CHUNKED,
        )

    def test_sm75_preserve_mlp_is_not_rewritten(self):
        with mock.patch.object(policy, '_current_capability', return_value=(7, 5)):
            resolved = policy.resolve_mlp_provider(
                mlp_inventory(plain_float=True),
                request=MLP_MEMORY_PRESERVE,
                fp8_available=False,
            )
        self.assertEqual(resolved.provider_id, MLP_FLOAT_CHUNKED)


if __name__ == '__main__':
    unittest.main()
