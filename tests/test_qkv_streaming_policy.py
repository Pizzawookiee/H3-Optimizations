'''Matrix contracts for streamed-BF16 QKV policy.'''

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.plan import (  # noqa: E402
    FUSED_QKV_AUTO,
    FUSED_QKV_FORCE_BF16,
    FUSED_QKV_FORCE_QUANT,
    FUSED_QKV_OFF,
    FUSED_QKV_PRESERVE_BF16,
    FUSED_QKV_REQUIRED,
)
from h3_optimizations.qkv.formats import inspect_h3_linears  # noqa: E402
from h3_optimizations.qkv.policy import resolve_qkv_provider  # noqa: E402
from h3_optimizations.qkv.providers import (  # noqa: E402
    QKV_BF16_CHUNKED,
    QKV_DENSE_FP8_CHUNKED,
    QKV_DENSE_KITCHEN_CHUNKED,
    QKV_FORCE_BF16_CHUNKED,
    QKV_FORCE_QUANT_CHUNKED,
    QKV_SPARSE_CONVROT_INT8,
    QKV_STANDARD,
    QKV_TRITON_SPARSE_CHUNKED,
)


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


def linear(weight):
    return SimpleNamespace(weight=weight, bias=None)


def block(weight):
    return SimpleNamespace(
        attn=SimpleNamespace(qkv_proj=linear(weight)),
        mlp=SimpleNamespace(fc1=linear(weight), fc2=linear(weight)),
    )


def inventory(weight):
    return inspect_h3_linears([block(weight)])


def sparse_spec():
    fused_v = SimpleNamespace(
        transpose_pad_permute_cuda=lambda *_args: None,
        scale_fuse_quant_cuda=lambda *_args: None,
    )
    return SimpleNamespace(
        capability=(12, 0),
        q_tile=128,
        kv_tile=64,
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


class QKVStreamingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.bf16 = FakeWeight(dtype='bfloat16')
        self.convrot = FakeWeight(
            layout='TensorWiseINT8Layout',
            storage_dtype='int8',
            convrot=True,
            group=256,
        )
        self.w4a8 = FakeWeight(
            layout='AsymW4A8Int8Layout',
            storage_dtype='int8',
        )
        self.fp8 = FakeWeight(
            layout='TensorCoreFP8E4M3Layout',
            storage_dtype='float8_e4m3fn',
        )
        self.nvfp4 = FakeWeight(
            layout='TensorCoreNVFP4Layout',
            storage_dtype='uint8',
        )

    def test_dense_kitchen_streaming_beats_float_to_fp8_conversion(self):
        resolved = resolve_qkv_provider(
            inventory(self.bf16),
            request=FUSED_QKV_AUTO,
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=True,
            memory_optimize=True,
            fp8_available=True,
        )
        self.assertEqual(resolved.provider_id, QKV_DENSE_KITCHEN_CHUNKED)
        self.assertNotEqual(resolved.provider_id, QKV_DENSE_FP8_CHUNKED)
        self.assertIn('BF16 Q/K/V chunks', resolved.reason)

    def test_lower_precision_checkpoints_still_use_bf16_stream_contract(self):
        for weight in (self.convrot, self.w4a8, self.fp8):
            with self.subTest(layout=weight._layout_cls):
                resolved = resolve_qkv_provider(
                    inventory(weight),
                    request=FUSED_QKV_PRESERVE_BF16,
                    backend_kind='comfy_kitchen_int8',
                    kitchen_producer_available=True,
                    memory_optimize=True,
                    fp8_available=True,
                )
                self.assertEqual(resolved.provider_id, QKV_DENSE_KITCHEN_CHUNKED)
                self.assertIn('BF16 Q/K/V chunks', resolved.reason)

    def test_sparse_sage_streams_bf16_chunks_for_all_supported_checkpoints(self):
        for weight in (self.bf16, self.convrot, self.w4a8, self.fp8):
            with self.subTest(layout=weight._layout_cls):
                resolved = resolve_qkv_provider(
                    inventory(weight),
                    request=FUSED_QKV_PRESERVE_BF16,
                    backend_kind='sparse_sage',
                    triton_available=True,
                    sparse_spec=sparse_spec(),
                    fp8_available=True,
                )
                self.assertEqual(resolved.provider_id, QKV_SPARSE_CONVROT_INT8)
                self.assertIn('BF16 Q/K/V chunks', resolved.reason)

    def test_triton_streams_bf16_chunks_for_all_supported_checkpoints(self):
        for weight in (self.bf16, self.convrot, self.w4a8, self.fp8):
            with self.subTest(layout=weight._layout_cls):
                resolved = resolve_qkv_provider(
                    inventory(weight),
                    request=FUSED_QKV_PRESERVE_BF16,
                    backend_kind='triton_sparse_int8',
                    triton_available=True,
                    fp8_available=True,
                )
                self.assertEqual(resolved.provider_id, QKV_TRITON_SPARSE_CHUNKED)
                self.assertIn('BF16 Q/K/V chunks', resolved.reason)

    def test_generic_attention_uses_bounded_projection_for_all_streamable_formats(self):
        for weight in (self.bf16, self.convrot, self.w4a8, self.fp8):
            with self.subTest(layout=weight._layout_cls):
                resolved = resolve_qkv_provider(
                    inventory(weight),
                    request=FUSED_QKV_PRESERVE_BF16,
                    backend_kind='existing',
                    memory_optimize=True,
                    fp8_available=True,
                )
                self.assertEqual(resolved.provider_id, QKV_BF16_CHUNKED)
                self.assertFalse(resolved.fused)
                self.assertIn('bounded token chunks', resolved.reason)
                self.assertIn('complete BF16 Q/K/V', resolved.reason)

    def test_allow_fp8_still_uses_bounded_native_projection_before_conversion(self):
        for weight in (self.bf16, self.convrot, self.w4a8, self.fp8):
            with self.subTest(layout=weight._layout_cls):
                resolved = resolve_qkv_provider(
                    inventory(weight),
                    request=FUSED_QKV_AUTO,
                    backend_kind='existing',
                    memory_optimize=True,
                    fp8_available=True,
                )
                self.assertEqual(resolved.provider_id, QKV_BF16_CHUNKED)
                self.assertNotEqual(resolved.provider_id, QKV_DENSE_FP8_CHUNKED)

    def test_bf16_mode_forces_bf16_projection_for_supported_formats(self):
        for weight in (self.bf16, self.convrot, self.w4a8, self.fp8):
            with self.subTest(layout=weight._layout_cls):
                resolved = resolve_qkv_provider(
                    inventory(weight),
                    request=FUSED_QKV_FORCE_BF16,
                    backend_kind='comfy_kitchen_int8',
                    kitchen_producer_available=True,
                    fp8_available=True,
                )
                self.assertEqual(resolved.provider_id, QKV_FORCE_BF16_CHUNKED)

    def test_force_quant_converts_plain_qkv_before_native_streaming(self):
        resolved = resolve_qkv_provider(
            inventory(self.bf16),
            request=FUSED_QKV_FORCE_QUANT,
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=True,
            memory_optimize=True,
            fp8_available=True,
        )
        self.assertEqual(resolved.provider_id, QKV_FORCE_QUANT_CHUNKED)

    def test_force_quant_fails_when_plain_qkv_cannot_run_fp8(self):
        with self.assertRaisesRegex(RuntimeError, 'accelerated FP8'):
            resolve_qkv_provider(
                inventory(self.bf16),
                request=FUSED_QKV_FORCE_QUANT,
                backend_kind='comfy_kitchen_int8',
                kitchen_producer_available=True,
                fp8_available=False,
            )

    def test_required_does_not_accept_nonfused_bounded_fallback(self):
        with self.assertRaisesRegex(RuntimeError, 'required fused QKV'):
            resolve_qkv_provider(
                inventory(self.convrot),
                request=FUSED_QKV_REQUIRED,
                backend_kind='existing',
            )

    def test_streaming_off_is_absolute(self):
        resolved = resolve_qkv_provider(
            inventory(self.bf16),
            request=FUSED_QKV_OFF,
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=True,
            fp8_available=True,
        )
        self.assertEqual(resolved.provider_id, QKV_STANDARD)

    def test_unsupported_nvfp4_does_not_get_requantized(self):
        resolved = resolve_qkv_provider(
            inventory(self.nvfp4),
            request=FUSED_QKV_AUTO,
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=True,
            memory_optimize=True,
            fp8_available=True,
        )
        self.assertEqual(resolved.provider_id, QKV_STANDARD)


if __name__ == '__main__':
    unittest.main()
