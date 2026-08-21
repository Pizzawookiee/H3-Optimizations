'''Alternative checkpoint compatibility policy for H3 optimization providers.'''

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.qkv.formats import inspect_h3_linears  # noqa: E402
from h3_optimizations.qkv.providers import (  # noqa: E402
    MLP_FP8_CHUNKED,
    MLP_PRESERVE_UPSTREAM,
    QKV_SPARSE_FP8_CHUNKED,
    QKV_STANDARD,
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
