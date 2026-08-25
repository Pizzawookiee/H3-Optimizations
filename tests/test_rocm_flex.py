'''CPU contracts for the ROCm BF16/FP16 FlexAttention fallback.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

from h3_optimizations.attention.sparse.config import HybridSparseConfig  # noqa: E402
from h3_optimizations.attention.sparse.fp8_flex import (  # noqa: E402
    FLEX_BACKEND_TRITON,
    FP8FlexBackend,
    FP8FlexSpec,
    preflight_fp8_flex,
)
from torch.nn.attention.flex_attention import BlockMask  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakeMetadata:
    def as_dict(self):
        return {'requested_video_budget': 0.5}


class FakeRouter:
    q_tile = 64
    kv_tile = 64

    def build_lut(self, q, _k, _layout, _video_budget, *, sink=None):
        del sink
        q_tiles = (q.shape[-2] + self.q_tile - 1) // self.q_tile
        kv_tiles = (q.shape[-2] + self.kv_tile - 1) // self.kv_tile
        delta = torch.ones(kv_tiles, dtype=torch.int32)
        delta[0] = 0
        lut = delta.view(1, 1, 1, -1).expand(
            q.shape[0], q.shape[1], q_tiles, -1
        ).clone()
        valid = torch.full(
            (q.shape[0], q.shape[1], q_tiles),
            kv_tiles,
            dtype=torch.int32,
        )
        return lut, valid, FakeMetadata()


class RocmFlexTests(unittest.TestCase):
    @staticmethod
    def _spec(attention):
        return FP8FlexSpec(
            version='test-rocm-flex',
            attention=attention,
            block_mask_type=BlockMask,
            kernel_backend=FLEX_BACKEND_TRITON,
            quantize_fp8=False,
        )

    def test_rocm_preflight_forces_triton_without_fp8_probe(self):
        calls = []

        def load(backend, quantize_fp8):
            calls.append((backend, quantize_fp8))
            return self._spec(lambda *_args, **_kwargs: None)

        fp8_probe = mock.Mock(side_effect=AssertionError('must not probe NVIDIA FP8'))
        spec = preflight_fp8_flex(
            cuda_available=lambda: False,
            capability_getter=lambda: None,
            fp8_supported=fp8_probe,
            dynamo_supported=lambda: True,
            rocm_available=lambda: True,
            loader=load,
        )

        self.assertFalse(spec.quantize_fp8)
        self.assertEqual(calls, [(FLEX_BACKEND_TRITON, False)])
        fp8_probe.assert_not_called()

    def test_rocm_backend_preserves_bf16_qkv_and_output(self):
        calls = []

        def attention(q, k, v, **kwargs):
            calls.append((q, k, v, kwargs))
            return torch.ones_like(q)

        backend = FP8FlexBackend(
            HybridSparseConfig(video_budget=0.5),
            spec=self._spec(attention),
            router=FakeRouter(),
            allow_cpu_for_tests=True,
        )
        q = torch.ones((1, 2, 128, 128), dtype=torch.bfloat16)
        k = torch.full_like(q, 2)
        v = torch.full_like(q, 3)
        snapshot = SimpleNamespace(
            valid_layout=True,
            error=None,
            layout=SimpleNamespace(seq_len=128),
            step_index=2,
            total_steps=20,
        )

        with mock.patch.object(backend, '_snapshot', return_value=snapshot):
            prepared = backend.prepare(
                q, k, v, layer_index=4, transformer_options={}
            )

        self.assertIs(prepared.q_fp8, q)
        self.assertIs(prepared.k_fp8, k)
        self.assertIs(prepared.v_fp8, v)
        self.assertEqual(prepared.q_fp8.dtype, torch.bfloat16)
        self.assertEqual(
            prepared.metadata['qkv_projection'],
            'standard_qkv_bf16_or_fp16',
        )

        output = backend.execute(prepared)

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(tuple(output.shape), tuple(q.shape))
        self.assertEqual(backend.name, 'flex_attention_rocm_bf16')
        self.assertEqual(len(calls), 1)
        call_q, call_k, call_v, kwargs = calls[0]
        self.assertIs(call_q, q)
        self.assertIs(call_k, k)
        self.assertIs(call_v, v)
        self.assertIsNone(kwargs['score_mod'])
        self.assertEqual(kwargs['kernel_options']['BACKEND'], FLEX_BACKEND_TRITON)
        self.assertEqual(kwargs['kernel_options']['BLOCK_M'], 64)
        self.assertEqual(kwargs['kernel_options']['BLOCK_N'], 64)

        status = backend.as_status()
        self.assertEqual(status['qkv_dtype'], 'native_fp16_or_bf16')
        self.assertTrue(status['approximate'])


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
