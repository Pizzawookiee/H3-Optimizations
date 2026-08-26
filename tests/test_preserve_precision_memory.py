'''CPU contracts for the preserve-precision H3 memory path.'''

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.plan import MLP_MEMORY_PRESERVE  # noqa: E402
from h3_optimizations.qkv.formats import inspect_h3_linears  # noqa: E402
from h3_optimizations.qkv.providers import (  # noqa: E402
    MLP_CONVROT_INT8_TWO_SLICE,
    MLP_FLOAT_CHUNKED,
    MLP_FP8_CHUNKED,
    MLP_PRESERVE_UPSTREAM,
    MLP_W4A8_CHUNKED,
    resolve_mlp_provider,
)


class FakeWeight:
    def __init__(
        self,
        layout=None,
        dtype='bfloat16',
        storage_dtype=None,
        *,
        convrot=False,
        convrot_groupsize=0,
    ):
        self._layout_cls = layout
        self._params = SimpleNamespace(
            convrot=convrot,
            convrot_groupsize=convrot_groupsize,
            transposed=False,
        )
        self.dtype = dtype
        self.storage_dtype = storage_dtype if storage_dtype is not None else dtype
        self.shape = (16, 16)


def linear(weight):
    return SimpleNamespace(weight=weight, bias=None)


def block(weight):
    return SimpleNamespace(
        attn=SimpleNamespace(
            qkv_proj=linear(weight),
            out_proj=linear(weight),
        ),
        mlp=SimpleNamespace(fc1=linear(weight), fc2=linear(weight)),
    )


class PreservePrecisionMemoryTests(unittest.TestCase):
    def resolve(self, weight, fp8_available=True):
        return resolve_mlp_provider(
            inspect_h3_linears([block(weight)]),
            request=MLP_MEMORY_PRESERVE,
            fp8_available=fp8_available,
        )

    def test_bf16_stays_float_chunked_even_when_fp8_is_available(self):
        resolved = self.resolve(FakeWeight())
        self.assertEqual(resolved.provider_id, MLP_FLOAT_CHUNKED)
        self.assertEqual(resolved.activation_mode, 'mlp_chunked_native')
        self.assertIn('keeps its checkpoint precision', resolved.reason)

    def test_checkpoint_native_fp8_remains_fp8(self):
        resolved = self.resolve(FakeWeight(
            layout='TensorCoreFP8E4M3Layout',
            storage_dtype='float8_e4m3fn',
        ))
        self.assertEqual(resolved.provider_id, MLP_FP8_CHUNKED)
        self.assertIn('checkpoint-native FP8', resolved.reason)

    def test_checkpoint_native_fp8_preserves_upstream_without_fp8_compute(self):
        resolved = self.resolve(
            FakeWeight(
                layout='TensorCoreFP8E4M3Layout',
                storage_dtype='float8_e4m3fn',
            ),
            fp8_available=False,
        )
        self.assertEqual(resolved.provider_id, MLP_PRESERVE_UPSTREAM)

    def test_checkpoint_native_w4a8_remains_native(self):
        resolved = self.resolve(FakeWeight(
            layout='AsymW4A8Int8Layout',
            storage_dtype='int8',
        ))
        self.assertEqual(resolved.provider_id, MLP_W4A8_CHUNKED)
        self.assertIn('without requantization', resolved.reason)

    def test_checkpoint_native_convrot_remains_native(self):
        resolved = self.resolve(FakeWeight(
            layout='TensorWiseINT8Layout',
            storage_dtype='int8',
            convrot=True,
            convrot_groupsize=256,
        ))
        self.assertEqual(resolved.provider_id, MLP_CONVROT_INT8_TWO_SLICE)
        self.assertIn('without requantization', resolved.reason)

    def test_unsupported_quantized_format_preserves_upstream(self):
        resolved = self.resolve(FakeWeight(
            layout='TensorCoreNVFP4Layout',
            storage_dtype='uint8',
        ))
        self.assertEqual(resolved.provider_id, MLP_PRESERVE_UPSTREAM)
        self.assertIn('preserving upstream Comfy execution', resolved.reason)


if __name__ == '__main__':
    unittest.main()
