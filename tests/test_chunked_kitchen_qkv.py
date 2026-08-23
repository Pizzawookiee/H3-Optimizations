'''CPU-only contracts for the H3 chunked Kitchen producer integration.'''

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

import h3_optimizations.kitchen_qkv as kitchen_qkv  # noqa: E402
import h3_optimizations.qkv.chunked as chunked_qkv  # noqa: E402
from h3_optimizations.attention_forward import make_forward  # noqa: E402
from h3_optimizations.dense_resolver import (  # noqa: E402
    ATTENTION_COMFY_KITCHEN_INT8,
    OVERRIDE_MARKER,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakeKitchen:
    INT8_ATTENTION_PRODUCER_ABI_VERSION = 1

    class Int8AttentionProducerUnavailableError(RuntimeError):
        pass

    def __init__(self):
        self.anchor_samples = None
        self.qk_chunks = []
        self.v = None

    @staticmethod
    def int8_attention_producer_is_available(_device=None):
        return True

    def int8_attention_producer_spec(self, *_args, **_kwargs):
        return SimpleNamespace(
            abi_version=1,
            k_anchor_positions=tuple(range(9)),
            sequence_alignment=4,
        )

    def select_int8_attention_k_anchor(self, _spec, samples):
        self.anchor_samples = samples.clone()
        return object()

    def create_int8_attention_producer(self, _spec, _anchor):
        return object()

    def quantize_int8_attention_qk_chunk(
        self,
        _producer,
        q,
        k,
        *,
        q_start,
        k_start,
    ):
        self.qk_chunks.append(
            (q_start, k_start, q.clone(), k.clone())
        )

    def quantize_int8_attention_v(self, _producer, v):
        self.v = v.clone()

    def finalize_int8_attention_producer(self, _producer):
        return 'carrier'

    def int8_attention_from_prequantized(self, carrier):
        return carrier


class ChunkedKitchenQKVTests(unittest.TestCase):
    @staticmethod
    def _kitchen_options():
        def override(*_args, **_kwargs):
            return None

        setattr(override, OVERRIDE_MARKER, ATTENTION_COMFY_KITCHEN_INT8)
        return {'optimized_attention_override': override}

    def test_anchor_prepass_rope_slices_chunks_and_retains_one_v(self):
        fake = FakeKitchen()
        module = SimpleNamespace(heads=2, head_dim=4)
        x = torch.arange(10, dtype=torch.float32).unsqueeze(1).expand(10, 6)
        rope = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1, 1, 1, 1)
        calls = []

        def project(_module, values, rope_rows):
            rows = values[:, 0].to(torch.int64)
            calls.append(
                (
                    tuple(int(value) for value in rows),
                    tuple(int(value) for value in rope_rows.reshape(-1)),
                )
            )
            base = rows.to(torch.float32).view(-1, 1, 1)
            q = base.expand(-1, 2, 4).clone()
            k = q + 100
            v = q + 200
            return q, k, v

        spec = fake.int8_attention_producer_spec()
        # Steer the resolver, not comfy_kitchen: the producer now comes from
        # the vendored library first, so patching the installed package no
        # longer decides which module is used.
        with mock.patch.object(
            kitchen_qkv,
            'resolve_kitchen',
            return_value=fake,
        ), mock.patch.object(
            kitchen_qkv,
            'project_qkv',
            side_effect=project,
        ), mock.patch.object(
            chunked_qkv,
            'project_qkv',
            side_effect=project,
        ):
            prepared = kitchen_qkv.run_chunked_kitchen_qkv(
                module,
                x,
                rope,
                layer_index=0,
                transformer_options={},
                spec=spec,
                chunk_rows=4,
            )

        self.assertEqual(prepared.carrier, 'carrier')
        self.assertEqual(calls[0], (tuple(range(9)), tuple(range(9))))
        self.assertEqual(
            [entry[:2] for entry in fake.qk_chunks],
            [(0, 0), (4, 4), (8, 8)],
        )
        self.assertEqual(
            [tuple(entry[2].shape) for entry in fake.qk_chunks],
            [(1, 2, 4, 4), (1, 2, 4, 4), (1, 2, 2, 4)],
        )
        self.assertEqual(tuple(fake.v.shape), (1, 2, 10, 4))
        self.assertTrue(
            torch.equal(
                fake.v[0, 0, :, 0],
                torch.arange(10, dtype=torch.float32) + 200,
            )
        )
        self.assertTrue(
            torch.equal(
                fake.anchor_samples[0, 0, :, 0],
                torch.arange(9, dtype=torch.float32) + 100,
            )
        )

    def test_public_api_probe_requires_the_complete_contract(self):
        fake = FakeKitchen()
        self.assertTrue(kitchen_qkv.producer_api_available(fake))
        incomplete = SimpleNamespace(
            **{
                name: object()
                for name in kitchen_qkv.PRODUCER_API
                if name != 'quantize_int8_attention_v'
            }
        )
        self.assertFalse(kitchen_qkv.producer_api_available(incomplete))
        fake.INT8_ATTENTION_PRODUCER_ABI_VERSION = 2
        self.assertFalse(kitchen_qkv.producer_api_available(fake))

    def test_backend_consumes_only_finalized_wrapper(self):
        fake = FakeKitchen()
        backend = kitchen_qkv.ChunkedKitchenAttentionBackend()
        prepared = kitchen_qkv.PreparedChunkedKitchenQKV('carrier')
        # Steer the resolver: the producer comes from the vendored library
        # first, so patching the installed package no longer decides this.
        with mock.patch.object(kitchen_qkv, 'resolve_kitchen', return_value=fake):
            self.assertEqual(backend.execute(prepared), 'carrier')
        with self.assertRaises(TypeError):
            backend.execute(object())

    def test_runtime_capability_decline_returns_to_upstream_forward(self):
        fake = FakeKitchen()

        def unavailable(*_args, **_kwargs):
            raise fake.Int8AttentionProducerUnavailableError('synthetic decline')

        fake.int8_attention_producer_spec = unavailable
        module = SimpleNamespace(qkv_proj=object(), heads=2, head_dim=4)
        x = SimpleNamespace(
            ndim=2,
            is_cuda=True,
            shape=(8, 8),
            dtype=torch.bfloat16,
            device=torch.device('cuda:0'),
        )
        projector = kitchen_qkv.ChunkedKitchenQKVProjector()
        # Steer the resolver, not comfy_kitchen: the producer now comes from
        # the vendored library first, so patching the installed package no
        # longer decides which module is used.
        with mock.patch.object(
            kitchen_qkv,
            'resolve_kitchen',
            return_value=fake,
        ), mock.patch.object(
            kitchen_qkv.comfy.model_management,
            'in_training',
            False,
        ), mock.patch.object(
            kitchen_qkv,
            'describe_linear',
            return_value=SimpleNamespace(convrot_int8_256=True),
        ):
            self.assertIsNone(
                projector.try_project(
                    module,
                    x,
                    None,
                    layer_index=0,
                    transformer_options=self._kitchen_options(),
                )
            )

    def test_runtime_projector_declines_after_an_explicit_override(self):
        fake = FakeKitchen()
        module = SimpleNamespace(qkv_proj=object(), heads=2, head_dim=4)
        x = SimpleNamespace(
            ndim=2,
            is_cuda=True,
            shape=(8, 8),
            dtype=torch.bfloat16,
            device=torch.device('cuda:0'),
        )
        projector = kitchen_qkv.ChunkedKitchenQKVProjector()
        # Steer the resolver, not comfy_kitchen: the producer now comes from
        # the vendored library first, so patching the installed package no
        # longer decides which module is used.
        with mock.patch.object(
            kitchen_qkv,
            'resolve_kitchen',
            return_value=fake,
        ), mock.patch.object(
            kitchen_qkv,
            'producer_api_available',
            side_effect=AssertionError('explicit override must win before probing'),
        ):
            self.assertIsNone(
                projector.try_project(
                    module,
                    x,
                    None,
                    layer_index=0,
                    transformer_options={
                        'optimized_attention_override': lambda *_args: None,
                    },
                )
            )

    def test_forward_uses_original_when_runtime_projector_declines(self):
        calls = []

        class Projector:
            name = 'declining_projector'

            @staticmethod
            def try_project(*_args, **_kwargs):
                return None

        class Backend:
            name = 'must_not_run'

            @staticmethod
            def prepare_projected(*_args, **_kwargs):
                raise AssertionError('backend must not consume a declined projection')

        module = SimpleNamespace(heads=2, head_dim=4)

        def original(x, rope_freqs=None, transformer_options=None):
            calls.append((x, rope_freqs, transformer_options))
            return 'upstream'

        forward = make_forward(
            module,
            3,
            backend=Backend(),
            projector=Projector(),
            fallback_forward=original,
        )
        x = object()
        rope = object()
        options = {'marker': True}
        self.assertEqual(forward(x, rope, options), 'upstream')
        self.assertEqual(calls, [(x, rope, options)])


if __name__ == '__main__':
    unittest.main()
