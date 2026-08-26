'''FROST consumer coverage for every declared streamed checkpoint format.'''

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
from h3_optimizations.attention.sparse.frost_bf16 import (  # noqa: E402
    FrostBF16Spec,
)

from h3_optimizations.plan import (  # noqa: E402
    FUSED_QKV_AUTO,
    FUSED_QKV_FORCE_BF16,
    FUSED_QKV_FORCE_QUANT,
    FUSED_QKV_PRESERVE_BF16,
    H3OptimizationPlan,
    MemoryRequest,
    SPARSE_BACKEND_FROST,
    SparseRequest,
)
from h3_optimizations.qkv.formats import inspect_h3_linears  # noqa: E402
from h3_optimizations.qkv.policy import resolve_qkv_provider  # noqa: E402
from h3_optimizations.qkv.providers import (  # noqa: E402
    QKV_FORCE_BF16_CHUNKED,
    QKV_FORCE_CONVROT_INT8_FROST,
    QKV_FROST_STREAMED,
)
from h3_optimizations.qkv.streamed import (  # noqa: E402
    PROJECTION_FORCE_BF16,
    PROJECTION_FORCE_INT8,
    PROJECTION_NATIVE,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakeWeight:
    def __init__(
        self,
        *,
        layout=None,
        dtype='bfloat16',
        storage_dtype=None,
        convrot=False,
        group=0,
    ):
        self._layout_cls = layout
        self._params = SimpleNamespace(
            convrot=convrot,
            convrot_groupsize=group,
            transposed=False,
        )
        self.dtype = dtype
        self.storage_dtype = storage_dtype if storage_dtype is not None else dtype
        self.shape = (16, 16)


def inventory(weight):
    linear = lambda: SimpleNamespace(weight=weight, bias=None)
    block = SimpleNamespace(
        attn=SimpleNamespace(qkv_proj=linear(), out_proj=linear()),
        mlp=SimpleNamespace(fc1=linear(), fc2=linear()),
    )
    return inspect_h3_linears([block])


class FrostStreamingMatrixTests(unittest.TestCase):
    def setUp(self):
        self.formats = {
            'BF16': FakeWeight(dtype='bfloat16'),
            'ConvRot': FakeWeight(
                layout='TensorWiseINT8Layout',
                storage_dtype='int8',
                convrot=True,
                group=256,
            ),
            'W4A8': FakeWeight(
                layout='AsymW4A8Int8Layout',
                storage_dtype='int8',
            ),
            'FP8': FakeWeight(
                layout='TensorCoreFP8E4M3Layout',
                storage_dtype='float8_e4m3fn',
            ),
        }
        self.policies = {
            'Auto': FUSED_QKV_AUTO,
            'BF16': FUSED_QKV_FORCE_BF16,
            'Preserve native': FUSED_QKV_PRESERVE_BF16,
            'Force quant': FUSED_QKV_FORCE_QUANT,
        }

    def test_all_sixteen_rows_select_a_genuine_frost_stream_consumer(self):
        for format_name, weight in self.formats.items():
            for policy_name, request in self.policies.items():
                with self.subTest(format=format_name, policy=policy_name):
                    resolved = resolve_qkv_provider(
                        inventory(weight),
                        request=request,
                        backend_kind='frost_bf16_sm89',
                        fp8_available=True,
                    )
                    expected = QKV_FROST_STREAMED
                    if policy_name == 'BF16':
                        expected = QKV_FORCE_BF16_CHUNKED
                    elif policy_name == 'Force quant' and format_name == 'BF16':
                        expected = QKV_FORCE_CONVROT_INT8_FROST
                    self.assertEqual(resolved.provider_id, expected)
                    self.assertTrue(resolved.fused)
                    self.assertIn('FROST', resolved.reason)

    def test_all_sixteen_rows_install_the_streamed_frost_projector(self):
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
            device_index=0,
        )
        spec = FrostBF16Spec()
        with mock.patch.object(
            apply_module,
            'preflight_frost_bf16',
            return_value=spec,
        ), mock.patch.object(
            apply_module,
            '_fp8_execution_available',
            return_value=True,
        ):
            for format_name, weight in self.formats.items():
                for policy_name, request in self.policies.items():
                    with self.subTest(format=format_name, policy=policy_name):
                        plan = H3OptimizationPlan(
                            memory=MemoryRequest(fused_qkv=request),
                            sparse=SparseRequest(backend=SPARSE_BACKEND_FROST),
                        )
                        attention, resolved = apply_module._resolve_frost_bf16(
                            plan,
                            environment,
                            inventory(weight),
                        )

                        expected_mode = PROJECTION_NATIVE
                        if policy_name == 'BF16':
                            expected_mode = PROJECTION_FORCE_BF16
                        elif policy_name == 'Force quant' and format_name == 'BF16':
                            expected_mode = PROJECTION_FORCE_INT8
                        self.assertTrue(resolved.fused)
                        self.assertIsNotNone(attention.projector)
                        self.assertEqual(
                            attention.requested,
                            apply_module.ATTENTION_FROST_BF16,
                        )
                        self.assertEqual(
                            attention.selected,
                            apply_module.ATTENTION_FROST_BF16,
                        )
                        self.assertTrue(attention.projector.streamed_q)
                        self.assertFalse(attention.projector.streamed_qkv)
                        self.assertEqual(
                            attention.projector.projection_mode,
                            expected_mode,
                        )


if __name__ == '__main__':
    unittest.main()
