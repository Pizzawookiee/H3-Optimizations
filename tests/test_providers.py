'''Pure weight-format and provider-resolution contracts.'''

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.qkv.formats import (  # noqa: E402
    describe_linear,
    inspect_h3_linears,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    MLP_CONVROT_INT8_TWO_SLICE,
    MLP_FLOAT_CHUNKED,
    MLP_FP8_CHUNKED,
    MLP_PRESERVE_UPSTREAM,
    MLP_W4A8_CHUNKED,
    QKV_DENSE_FP8_CHUNKED,
    QKV_DENSE_KITCHEN_CHUNKED,
    QKV_DENSE_W4A8_CHUNKED,
    QKV_SPARSE_CONVROT_INT8,
    QKV_STANDARD,
    QKV_TRITON_SPARSE_CHUNKED,
    resolve_mlp_provider,
    resolve_qkv_provider,
)
from h3_optimizations.plan import (  # noqa: E402
    MLP_MEMORY_LEGACY_BF16,
    MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
)


class FakeWeight:
    def __init__(
        self,
        *,
        layout=None,
        convrot=False,
        group=0,
        transposed=False,
        dtype='bf16',
        storage_dtype=None,
        shape=(10, 10),
    ):
        self._layout_cls = layout
        self._params = SimpleNamespace(
            convrot=convrot,
            convrot_groupsize=group,
            transposed=transposed,
        )
        self.dtype = dtype
        self.storage_dtype = storage_dtype if storage_dtype is not None else dtype
        self.shape = shape


def linear(weight, bias=None):
    return SimpleNamespace(weight=weight, bias=bias)


def block(weight):
    return SimpleNamespace(
        attn=SimpleNamespace(qkv_proj=linear(weight)),
        mlp=SimpleNamespace(
            fc1=linear(weight),
            fc2=linear(weight),
        ),
    )


def sparse_spec(*, capability=(12, 0), q_tile=128, kv_tile=64):
    fused_v = SimpleNamespace(
        transpose_pad_permute_cuda=lambda *_args: None,
        scale_fuse_quant_cuda=lambda *_args: None,
    )
    return SimpleNamespace(
        capability=capability,
        q_tile=q_tile,
        kv_tile=kv_tile,
        qk_format='block_int8',
        q_scale_layout='per_q_tile_float32',
        k_scale_layout='per_kv_tile_float32',
        projected_v_format='floating_hnd',
        summary_format='tile_mean',
        v_format='fp8',
        accumulator='f16',
        kernel=lambda *_args: None,
        fused_v_ops=fused_v,
    )


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.convrot = FakeWeight(
            layout='TensorWiseINT8Layout',
            convrot=True,
            group=256,
            dtype='bfloat16',
            storage_dtype='int8',
        )
        self.plain = FakeWeight(dtype='bfloat16')
        self.fp8 = FakeWeight(
            layout='TensorCoreFP8E4M3Layout',
            dtype='bfloat16',
            storage_dtype='float8_e4m3fn',
        )
        self.nvfp4 = FakeWeight(
            layout='TensorCoreNVFP4Layout',
            dtype='bfloat16',
            storage_dtype='uint8',
        )

    def test_format_inspection_is_conservative(self):
        self.assertTrue(describe_linear(linear(self.convrot)).convrot_int8_256)
        self.assertTrue(describe_linear(linear(self.fp8)).fp8)
        self.assertTrue(describe_linear(linear(self.nvfp4)).nvfp4)
        self.assertTrue(describe_linear(linear(self.plain)).plain_float)
        self.assertEqual(describe_linear(linear(self.nvfp4)).storage_dtype, 'uint8')
        biased = describe_linear(linear(self.convrot, bias=object()))
        self.assertFalse(biased.convrot_int8_256)

    def test_compatible_convrot_selects_specialized_providers(self):
        inventory = inspect_h3_linears([block(self.convrot), block(self.convrot)])
        dense = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=True,
        )
        self.assertEqual(dense.provider_id, QKV_DENSE_KITCHEN_CHUNKED)

        kitchen_sparse = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='sparse_kitchen_int8',
            kitchen_producer_available=True,
        )
        self.assertEqual(
            kitchen_sparse.provider_id,
            QKV_DENSE_KITCHEN_CHUNKED,
        )
        self.assertTrue(kitchen_sparse.fused)

        sparse = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='sparse_sage',
            triton_available=True,
            sparse_spec=sparse_spec(),
        )
        self.assertEqual(sparse.provider_id, QKV_SPARSE_CONVROT_INT8)
        self.assertTrue(sparse.fused)

        triton_sparse = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='triton_sparse_int8',
            triton_available=True,
        )
        self.assertEqual(triton_sparse.provider_id, QKV_TRITON_SPARSE_CHUNKED)

        mlp = resolve_mlp_provider(inventory, request='auto')
        self.assertEqual(mlp.provider_id, MLP_CONVROT_INT8_TWO_SLICE)

    def test_sparse_alone_preserves_alternative_checkpoint_projection(self):
        for weight in (self.plain, self.fp8, self.nvfp4):
            inventory = inspect_h3_linears([block(weight)])
            qkv = resolve_qkv_provider(
                inventory,
                request='auto',
                backend_kind='sparse_sage',
                triton_available=True,
                sparse_spec=sparse_spec(),
                memory_optimize=False,
                fp8_available=True,
            )
            self.assertEqual(qkv.provider_id, QKV_STANDARD)

    def test_memory_fp8_and_bf16_use_fp8_providers(self):
        for weight in (self.plain, self.fp8):
            inventory = inspect_h3_linears([block(weight)])
            qkv = resolve_qkv_provider(
                inventory,
                request='auto',
                backend_kind='comfy_kitchen_int8',
                kitchen_producer_available=True,
                memory_optimize=True,
                fp8_available=True,
            )
            self.assertEqual(qkv.provider_id, QKV_DENSE_FP8_CHUNKED)
            mlp = resolve_mlp_provider(
                inventory,
                request='auto',
                fp8_available=True,
            )
            self.assertEqual(mlp.provider_id, MLP_FP8_CHUNKED)

    def test_bf16_uses_float_chunking_without_fp8_hardware(self):
        inventory = inspect_h3_linears([block(self.plain)])
        mlp = resolve_mlp_provider(
            inventory,
            request='auto',
            fp8_available=False,
        )
        self.assertEqual(mlp.provider_id, MLP_FLOAT_CHUNKED)

    def test_nvfp4_memory_preserves_upstream(self):
        inventory = inspect_h3_linears([block(self.nvfp4)])
        qkv = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=True,
            memory_optimize=True,
            fp8_available=True,
        )
        self.assertEqual(qkv.provider_id, QKV_STANDARD)
        mlp = resolve_mlp_provider(
            inventory,
            request='auto',
            fp8_available=True,
        )
        self.assertEqual(mlp.provider_id, MLP_PRESERVE_UPSTREAM)

    def test_dense_uses_kitchen_capability_instead_of_gpu_model(self):
        inventory = inspect_h3_linears([block(self.convrot)])
        qkv = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=False,
        )
        self.assertEqual(qkv.provider_id, QKV_STANDARD)

    def test_dense_w4a8_qkv_uses_bounded_kitchen_projection(self):
        w4a8 = FakeWeight(layout='AsymW4A8Int8Layout', dtype='bfloat16', storage_dtype='int8')
        inventory = inspect_h3_linears([block(w4a8)])
        qkv = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=True,
            memory_optimize=True,
            fp8_available=True,
        )
        self.assertEqual(qkv.provider_id, QKV_DENSE_W4A8_CHUNKED)
        self.assertEqual(
            resolve_mlp_provider(inventory, request='auto', fp8_available=True).provider_id,
            MLP_W4A8_CHUNKED,
        )

    def test_sparse_fusion_declines_mismatched_geometry(self):
        inventory = inspect_h3_linears([block(self.convrot)])
        qkv = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='sparse_sage',
            triton_available=True,
            sparse_spec=sparse_spec(capability=(9, 0), q_tile=64, kv_tile=128),
        )
        self.assertEqual(qkv.provider_id, QKV_STANDARD)

    def test_required_fused_qkv_fails_instead_of_using_fp8_conversion(self):
        inventory = inspect_h3_linears([block(self.plain)])
        with self.assertRaisesRegex(RuntimeError, 'required fused QKV'):
            resolve_qkv_provider(
                inventory,
                request='required',
                backend_kind='comfy_kitchen_int8',
                kitchen_producer_available=True,
                memory_optimize=True,
                fp8_available=True,
            )

    def test_legacy_mlp_requests_use_supported_production_modes(self):
        convrot = inspect_h3_linears([block(self.convrot)])
        required = resolve_mlp_provider(
            convrot,
            request=MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
        )
        self.assertEqual(required.provider_id, MLP_CONVROT_INT8_TWO_SLICE)
        plain = inspect_h3_linears([block(self.plain)])
        bf16 = resolve_mlp_provider(
            plain,
            request=MLP_MEMORY_LEGACY_BF16,
        )
        self.assertEqual(bf16.provider_id, MLP_FLOAT_CHUNKED)
        self.assertEqual(bf16.activation_mode, 'mlp_chunked_bf16')


if __name__ == '__main__':
    unittest.main()
