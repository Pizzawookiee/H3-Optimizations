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
    MLP_GENERIC_CHUNKED,
    QKV_DENSE_KITCHEN_CHUNKED,
    QKV_SPARSE_CONVROT_INT8,
    QKV_STANDARD,
    resolve_mlp_provider,
    resolve_qkv_provider,
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
        shape=(10, 10),
    ):
        self._layout_cls = layout
        self._params = SimpleNamespace(
            convrot=convrot,
            convrot_groupsize=group,
            transposed=transposed,
        )
        self.dtype = dtype
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
            dtype='int8',
        )
        self.plain = FakeWeight(dtype='bfloat16')

    def test_format_inspection_is_conservative(self):
        self.assertTrue(
            describe_linear(linear(self.convrot)).convrot_int8_256
        )
        self.assertFalse(
            describe_linear(linear(self.plain)).convrot_int8_256
        )
        biased = describe_linear(linear(self.convrot, bias=object()))
        self.assertFalse(biased.convrot_int8_256)

    def test_compatible_convrot_selects_specialized_providers(self):
        inventory = inspect_h3_linears(
            [block(self.convrot), block(self.convrot)]
        )
        dense = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=True,
        )
        self.assertEqual(dense.provider_id, QKV_DENSE_KITCHEN_CHUNKED)
        self.assertFalse(dense.fused)

        sparse = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='sparse_sage',
            triton_available=True,
            sparse_spec=sparse_spec(),
        )
        self.assertEqual(sparse.provider_id, QKV_SPARSE_CONVROT_INT8)
        self.assertTrue(sparse.fused)

        mlp = resolve_mlp_provider(inventory, request='auto')
        self.assertEqual(mlp.provider_id, MLP_CONVROT_INT8_TWO_SLICE)
        self.assertEqual(
            mlp.activation_mode,
            'mlp_chunked_convrot_2slice',
        )

    def test_auto_falls_back_without_a_complete_contract(self):
        inventory = inspect_h3_linears(
            [block(self.plain), block(self.plain)]
        )
        qkv = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=True,
        )
        self.assertEqual(qkv.provider_id, QKV_STANDARD)
        self.assertFalse(qkv.fused)
        mlp = resolve_mlp_provider(inventory, request='auto')
        self.assertEqual(mlp.provider_id, MLP_GENERIC_CHUNKED)
        self.assertEqual(mlp.activation_mode, 'mlp_chunked_native')

    def test_dense_uses_kitchen_capability_instead_of_gpu_model(self):
        inventory = inspect_h3_linears([block(self.convrot)])
        qkv = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=False,
        )
        self.assertEqual(qkv.provider_id, QKV_STANDARD)

    def test_dense_preserves_an_explicit_attention_backend(self):
        inventory = inspect_h3_linears([block(self.convrot)])
        qkv = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='existing',
            kitchen_producer_available=True,
        )
        self.assertEqual(qkv.provider_id, QKV_STANDARD)

    def test_dense_w4a8_qkv_stays_standard(self):
        w4a8 = FakeWeight(
            layout='AsymW4A8Int8Layout',
            dtype='int8',
        )
        inventory = inspect_h3_linears([block(w4a8)])
        qkv = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=True,
        )
        self.assertEqual(qkv.provider_id, QKV_STANDARD)

    def test_sparse_fusion_accepts_matching_non_sm89_contract(self):
        inventory = inspect_h3_linears([block(self.convrot)])
        qkv = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='sparse_sage',
            triton_available=True,
            sparse_spec=sparse_spec(capability=(12, 0)),
        )
        self.assertEqual(qkv.provider_id, QKV_SPARSE_CONVROT_INT8)

    def test_sparse_fusion_declines_mismatched_geometry(self):
        inventory = inspect_h3_linears([block(self.convrot)])
        qkv = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='sparse_sage',
            triton_available=True,
            sparse_spec=sparse_spec(
                capability=(9, 0),
                q_tile=64,
                kv_tile=128,
            ),
        )
        self.assertEqual(qkv.provider_id, QKV_STANDARD)


if __name__ == '__main__':
    unittest.main()
