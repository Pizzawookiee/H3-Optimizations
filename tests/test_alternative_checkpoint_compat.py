'''Alternative checkpoint compatibility policy for H3 optimization providers.'''

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.qkv.formats import inspect_h3_linears  # noqa: E402
from h3_optimizations.qkv.providers import (  # noqa: E402
    MLP_FP8_CHUNKED,
    MLP_PRESERVE_UPSTREAM,
    MLP_W4A8_CHUNKED,
    QKV_DENSE_W4A8_CHUNKED,
    QKV_SPARSE_FP8_CHUNKED,
    QKV_SPARSE_W4A8_CHUNKED,
    QKV_STANDARD,
    QKV_TRITON_W4A8_CHUNKED,
    resolve_mlp_provider,
    resolve_qkv_provider,
)


class FakeWeight:
    def __init__(self, layout=None, dtype='bfloat16', storage_dtype=None):
        self._layout_cls = layout
        self._params = SimpleNamespace(
            convrot=False,
            convrot_groupsize=0,
            transposed=False,
        )
        self.dtype = dtype
        self.storage_dtype = storage_dtype if storage_dtype is not None else dtype
        self.shape = (16, 16)


def linear(weight):
    return SimpleNamespace(weight=weight, bias=None)


def block(weight):
    return SimpleNamespace(
        attn=SimpleNamespace(qkv_proj=linear(weight)),
        mlp=SimpleNamespace(fc1=linear(weight), fc2=linear(weight)),
    )


def sparse_spec():
    return SimpleNamespace(
        q_tile=128,
        kv_tile=64,
        qk_format='block_int8',
        q_scale_layout='per_q_tile_float32',
        k_scale_layout='per_kv_tile_float32',
        projected_v_format='floating_hnd',
        summary_format='tile_mean',
        v_format='fp16',
        accumulator='f16',
        kernel=lambda *_args: None,
        fused_v_ops=None,
    )


class AlternativeCheckpointCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.bf16 = FakeWeight()
        self.fp8 = FakeWeight(
            layout='TensorCoreFP8E4M3Layout',
            storage_dtype='float8_e4m3fn',
        )
        self.w4a8 = FakeWeight(
            layout='AsymW4A8Int8Layout',
            storage_dtype='int8',
        )
        self.nvfp4 = FakeWeight(
            layout='TensorCoreNVFP4Layout',
            storage_dtype='uint8',
        )

    def test_sparse_only_never_requires_memory_normalization(self):
        for weight in (self.bf16, self.fp8, self.nvfp4):
            inventory = inspect_h3_linears([block(weight)])
            resolved = resolve_qkv_provider(
                inventory,
                request='auto',
                backend_kind='sparse_sage',
                triton_available=True,
                sparse_spec=sparse_spec(),
                memory_optimize=False,
                fp8_available=True,
            )
            self.assertEqual(resolved.provider_id, QKV_STANDARD)

    def test_memory_plus_sparse_uses_fp8_chunked_qkv_for_float_and_fp8(self):
        for weight in (self.bf16, self.fp8):
            inventory = inspect_h3_linears([block(weight)])
            resolved = resolve_qkv_provider(
                inventory,
                request='auto',
                backend_kind='sparse_sage',
                triton_available=True,
                sparse_spec=sparse_spec(),
                memory_optimize=True,
                fp8_available=True,
            )
            self.assertEqual(resolved.provider_id, QKV_SPARSE_FP8_CHUNKED)
            self.assertEqual(
                resolve_mlp_provider(
                    inventory,
                    request='auto',
                    fp8_available=True,
                ).provider_id,
                MLP_FP8_CHUNKED,
            )

    def test_w4a8_uses_native_chunked_providers(self):
        inventory = inspect_h3_linears([block(self.w4a8)])
        self.assertTrue(inventory.qkv[0].w4a8)
        self.assertTrue(inventory.qkv_w4a8)
        self.assertTrue(inventory.mlp_w4a8)
        self.assertFalse(inventory.qkv[0].other_quantized)

        kitchen_sparse = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='sparse_kitchen_int8',
            kitchen_producer_available=True,
            memory_optimize=False,
            fp8_available=False,
        )
        self.assertEqual(
            kitchen_sparse.provider_id,
            QKV_DENSE_W4A8_CHUNKED,
        )
        self.assertTrue(kitchen_sparse.fused)

        kitchen_sparse_without_producer = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='sparse_kitchen_int8',
            kitchen_producer_available=False,
            memory_optimize=False,
            fp8_available=False,
        )
        self.assertEqual(
            kitchen_sparse_without_producer.provider_id,
            QKV_STANDARD,
        )

        sparse = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='sparse_sage',
            triton_available=True,
            sparse_spec=sparse_spec(),
            memory_optimize=False,
            fp8_available=False,
        )
        self.assertEqual(sparse.provider_id, QKV_SPARSE_W4A8_CHUNKED)

        triton = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='triton_sparse_int8',
            triton_available=True,
            memory_optimize=False,
            fp8_available=False,
        )
        self.assertEqual(triton.provider_id, QKV_TRITON_W4A8_CHUNKED)

        dense_without_memory = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=True,
            memory_optimize=False,
            fp8_available=False,
        )
        self.assertEqual(dense_without_memory.provider_id, QKV_STANDARD)

        dense_with_memory = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=True,
            memory_optimize=True,
            fp8_available=False,
        )
        self.assertEqual(dense_with_memory.provider_id, QKV_DENSE_W4A8_CHUNKED)

        mlp = resolve_mlp_provider(
            inventory,
            request='auto',
            fp8_available=False,
        )
        self.assertEqual(mlp.provider_id, MLP_W4A8_CHUNKED)
        self.assertEqual(mlp.activation_mode, 'mlp_chunked_native')

    def test_raw_torch_fp8_uses_fp8_memory_providers(self):
        dtypes = [
            getattr(torch, 'float8_e4m3fn', None),
            getattr(torch, 'float8_e5m2', None),
        ]
        dtypes = [dtype for dtype in dtypes if dtype is not None]
        if not dtypes:
            self.skipTest('this PyTorch build has no float8 dtypes')

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                raw_fp8 = FakeWeight(dtype=dtype)
                inventory = inspect_h3_linears([block(raw_fp8)])
                self.assertTrue(inventory.qkv[0].raw_fp8)
                self.assertTrue(inventory.qkv[0].fp8)
                self.assertFalse(inventory.qkv[0].plain_float)

                sparse_only = resolve_qkv_provider(
                    inventory,
                    request='auto',
                    backend_kind='sparse_sage',
                    triton_available=True,
                    sparse_spec=sparse_spec(),
                    memory_optimize=False,
                    fp8_available=True,
                )
                self.assertEqual(sparse_only.provider_id, QKV_STANDARD)

                memory_sparse = resolve_qkv_provider(
                    inventory,
                    request='auto',
                    backend_kind='sparse_sage',
                    triton_available=True,
                    sparse_spec=sparse_spec(),
                    memory_optimize=True,
                    fp8_available=True,
                )
                self.assertEqual(
                    memory_sparse.provider_id,
                    QKV_SPARSE_FP8_CHUNKED,
                )
                self.assertEqual(
                    resolve_mlp_provider(
                        inventory,
                        request='auto',
                        fp8_available=True,
                    ).provider_id,
                    MLP_FP8_CHUNKED,
                )

    def test_nvfp4_memory_preserves_upstream(self):
        inventory = inspect_h3_linears([block(self.nvfp4)])
        qkv = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='sparse_sage',
            triton_available=True,
            sparse_spec=sparse_spec(),
            memory_optimize=True,
            fp8_available=True,
        )
        self.assertEqual(qkv.provider_id, QKV_STANDARD)
        self.assertEqual(
            resolve_mlp_provider(
                inventory,
                request='auto',
                fp8_available=True,
            ).provider_id,
            MLP_PRESERVE_UPSTREAM,
        )


if __name__ == '__main__':
    unittest.main()
