'''CPU contracts for the preserve-precision H3 memory path.'''

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.plan import (  # noqa: E402
    ATTENTION_EXISTING,
    FUSED_QKV_OFF,
    MLP_MEMORY_PRESERVE,
    MemoryRequest,
)
from h3_optimizations.qkv.formats import inspect_h3_linears  # noqa: E402
from h3_optimizations.qkv.providers import (  # noqa: E402
    MLP_FLOAT_CHUNKED,
    MLP_FP8_CHUNKED,
    MLP_PRESERVE_UPSTREAM,
    MLP_W4A8_CHUNKED,
    resolve_mlp_provider,
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


class PreservePrecisionMemoryTests(unittest.TestCase):
    def test_request_preserves_dense_attention_and_qkv(self):
        request = MemoryRequest(
            attention=ATTENTION_EXISTING,
            fused_qkv=FUSED_QKV_OFF,
            mlp_memory=MLP_MEMORY_PRESERVE,
        )
        self.assertEqual(request.attention, ATTENTION_EXISTING)
        self.assertEqual(request.fused_qkv, FUSED_QKV_OFF)
        self.assertEqual(request.mlp_memory, MLP_MEMORY_PRESERVE)

    def test_bf16_stays_float_chunked_even_when_fp8_is_available(self):
        inventory = inspect_h3_linears([block(FakeWeight())])
        resolved = resolve_mlp_provider(
            inventory,
            request=MLP_MEMORY_PRESERVE,
            fp8_available=True,
        )
        self.assertEqual(resolved.provider_id, MLP_FLOAT_CHUNKED)
        self.assertEqual(resolved.activation_mode, 'mlp_chunked_native')
        self.assertIn('keeps its checkpoint precision', resolved.reason)

    def test_checkpoint_native_fp8_remains_fp8(self):
        inventory = inspect_h3_linears([
            block(FakeWeight(
                layout='TensorCoreFP8E4M3Layout',
                storage_dtype='float8_e4m3fn',
            ))
        ])
        resolved = resolve_mlp_provider(
            inventory,
            request=MLP_MEMORY_PRESERVE,
            fp8_available=True,
        )
        self.assertEqual(resolved.provider_id, MLP_FP8_CHUNKED)
        self.assertIn('checkpoint-native FP8', resolved.reason)

    def test_checkpoint_native_w4a8_remains_native(self):
        inventory = inspect_h3_linears([
            block(FakeWeight(
                layout='AsymW4A8Int8Layout',
                storage_dtype='int8',
            ))
        ])
        resolved = resolve_mlp_provider(
            inventory,
            request=MLP_MEMORY_PRESERVE,
            fp8_available=True,
        )
        self.assertEqual(resolved.provider_id, MLP_W4A8_CHUNKED)
        self.assertIn('without requantization', resolved.reason)

    def test_unsupported_quantized_format_preserves_upstream(self):
        inventory = inspect_h3_linears([
            block(FakeWeight(
                layout='TensorCoreNVFP4Layout',
                storage_dtype='uint8',
            ))
        ])
        resolved = resolve_mlp_provider(
            inventory,
            request=MLP_MEMORY_PRESERVE,
            fp8_available=True,
        )
        self.assertEqual(resolved.provider_id, MLP_PRESERVE_UPSTREAM)
        self.assertIn('preserving upstream Comfy execution', resolved.reason)


if __name__ == '__main__':
    unittest.main()
