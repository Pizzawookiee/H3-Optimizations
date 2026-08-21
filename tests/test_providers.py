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
    QKV_DENSE_CONVROT_INT8,
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
            backend_kind='dense_sage_sm89',
            capability=(8, 9),
            triton_available=True,
        )
        self.assertEqual(dense.provider_id, QKV_DENSE_CONVROT_INT8)
        self.assertTrue(dense.fused)

        sparse = resolve_qkv_provider(
            inventory,
            request='auto',
            backend_kind='sparse_sage',
            capability=(8, 9),
            triton_available=True,
            sparse_spec=SimpleNamespace(
                capability=(8, 9),
                q_tile=128,
                kv_tile=64,
            ),
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
            backend_kind='dense_sage_sm89',
            capability=(8, 9),
            triton_available=True,
        )
        self.assertEqual(qkv.provider_id, QKV_STANDARD)
        self.assertFalse(qkv.fused)
        mlp = resolve_mlp_provider(inventory, request='auto')
        self.assertEqual(mlp.provider_id, MLP_GENERIC_CHUNKED)
        self.assertEqual(mlp.activation_mode, 'mlp_chunked_native')

    def test_architecture_and_triton_gates_are_mandatory(self):
        inventory = inspect_h3_linears([block(self.convrot)])
        for capability, triton_available in (
            ((8, 6), True),
            ((8, 9), False),
        ):
            qkv = resolve_qkv_provider(
                inventory,
                request='auto',
                backend_kind='dense_sage_sm89',
                capability=capability,
                triton_available=triton_available,
            )
            self.assertEqual(qkv.provider_id, QKV_STANDARD)


if __name__ == '__main__':
    unittest.main()
