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
import h3_optimizations.apply_policy as apply_policy  # noqa: E402
import h3_optimizations.qkv.chunked as chunked_qkv  # noqa: E402
from h3_optimizations.attention_forward import make_forward  # noqa: E402
from h3_optimizations.attention.sparse.config import (  # noqa: E402
    HybridSparseConfig,
    MODE_SAGE128_FUSED_QKV,
)
from h3_optimizations.attention.sparse.kitchen_sparse import (  # noqa: E402
    SparseKitchenBackend,
)
from h3_optimizations.dense_resolver import (  # noqa: E402
    ATTENTION_COMFY_KITCHEN_INT8,
    OVERRIDE_MARKER,
)
from h3_optimizations.runtime.context import (  # noqa: E402
    RUNTIME_KEY,
    RuntimeSnapshot,
)
from h3_optimizations.native import producer as native_producer  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakeKitchen:
    INT8_ATTENTION_PRODUCER_ABI_VERSION = 1

    class Int8AttentionProducerUnavailableError(RuntimeError):
        pass

    def __init__(self):
        self.anchor_samples = None
        self.qk_chunks = []
        self.v = None
        self.spec_requests = []

    @staticmethod
    def int8_attention_producer_is_available(_device=None):
        return True

    def int8_attention_producer_spec(self, *_args, **_kwargs):
        self.spec_requests.append(dict(_kwargs))
        return SimpleNamespace(
            abi_version=1,
            k_anchor_positions=tuple(range(9)),
            sequence_alignment=4,
            q_tile=4,
            k_tile=4,
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

    def quantize_int8_attention_q_chunk(
        self,
        _producer,
        q,
        *,
        q_start,
        **_kwargs,
    ):
        self.qk_chunks.append((q_start, None, q.clone(), None))

    def quantize_int8_attention_k_chunk(
        self,
        _producer,
        k,
        *,
        k_start,
        **_kwargs,
    ):
        self.qk_chunks.append((None, k_start, None, k.clone()))

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
        module = SimpleNamespace(qkv_proj=object(), heads=2, head_dim=4)
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
            'describe_linear',
            return_value=SimpleNamespace(plain_float=False, w4a8=False),
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
        self.assertIsNone(prepared.q_summary)
        self.assertIsNone(prepared.k_summary)

    def test_streamed_kitchen_projection_keeps_global_kv_but_no_global_q(self):
        class Held:
            def __init__(self):
                self.kv_calls = []
                self.released = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.released = True
                return False

            def project_rows(self, x, _rope, rows):
                values = x.index_select(0, rows)[:, :256].reshape(-1, 2, 128)
                hnd = values.transpose(0, 1).unsqueeze(0)
                return hnd, hnd + 10, hnd + 20

            def project_kv_hnd(self, x, _rope, start, stop):
                self.kv_calls.append((start, stop))
                values = x[start:stop, :256].reshape(-1, 2, 128)
                hnd = values.transpose(0, 1).unsqueeze(0)
                return hnd + 10, hnd + 20

        class Kitchen:
            def __init__(self):
                self.k_starts = []
                self.v = None

            @staticmethod
            def select_int8_attention_k_anchor(_spec, _samples):
                return object()

            @staticmethod
            def create_int8_attention_producer(_spec, _anchor):
                return object()

            def quantize_int8_attention_k_chunk(
                self, _producer, _k, *, k_start, **_kwargs
            ):
                self.k_starts.append(k_start)

            def quantize_int8_attention_v(self, _producer, v):
                self.v = v.clone()

            @staticmethod
            def finalize_int8_attention_producer(_producer):
                return SimpleNamespace(q=torch.empty(1, 2, 1, 128, dtype=torch.int8))

        held = Held()
        kitchen = Kitchen()
        spec = SimpleNamespace(k_anchor_positions=(0, 0, 1, 1, 2, 2, 3, 3, 4))
        x = (torch.arange(5 * 256).reshape(5, 256) % 100).to(torch.float16)
        with mock.patch.object(
            kitchen_qkv,
            'resolve_kitchen',
            return_value=kitchen,
        ), mock.patch.object(
            kitchen_qkv,
            'create_held_qkv',
            return_value=held,
        ):
            projected = kitchen_qkv.run_streamed_kitchen_qkv(
                SimpleNamespace(heads=2, head_dim=128),
                x,
                None,
                layer_index=0,
                transformer_options={},
                spec=spec,
                chunk_rows=2,
                projection_mode='native',
            )

        self.assertEqual(held.kv_calls, [(0, 2), (2, 4), (4, 5)])
        self.assertTrue(held.released)
        self.assertEqual(kitchen.k_starts, [0, 2, 4])
        self.assertEqual(tuple(kitchen.v.shape), (1, 2, 5, 128))
        self.assertEqual(projected.carrier.q.shape[-2], 1)
        self.assertIs(projected.output_buffer, x)

    def test_native_split_packers_keep_only_a_one_row_companion(self):
        with mock.patch.object(
            native_producer,
            'int8_attention_producer_is_available',
            return_value=True,
        ):
            k_spec = native_producer.int8_attention_producer_spec(
                (1, 2, 1, 128),
                (1, 2, 256, 128),
                dtype=torch.bfloat16,
                device=torch.device('cpu'),
                cta_k=64,
            )
            q_spec = native_producer.int8_attention_producer_spec(
                (1, 2, 256, 128),
                (1, 2, 1, 128),
                dtype=torch.bfloat16,
                device=torch.device('cpu'),
                cta_k=64,
            )
        anchor = native_producer.Int8AttentionKAnchor(
            values=torch.zeros(1, 2, 128, dtype=torch.bfloat16),
            indices=torch.zeros(1, 2, dtype=torch.int32),
        )
        k_producer = native_producer.create_int8_attention_producer(k_spec, anchor)
        q_producer = native_producer.create_int8_attention_producer(q_spec, anchor)
        calls = []
        with mock.patch.object(
            native_producer.loader,
            'load',
            return_value=object(),
        ), mock.patch.object(
            native_producer,
            '_quantize_qk_chunk',
            side_effect=lambda _lib, _producer, _spec, q, k, q_start, k_start: calls.append(
                (tuple(q.shape), tuple(k.shape), q_start, k_start)
            ),
        ):
            for start in (0, 128):
                native_producer.quantize_int8_attention_k_chunk(
                    k_producer,
                    torch.zeros(1, 2, 128, 128, dtype=torch.bfloat16),
                    k_start=start,
                )
                native_producer.quantize_int8_attention_q_chunk(
                    q_producer,
                    torch.zeros(1, 2, 128, 128, dtype=torch.bfloat16),
                    q_start=start,
                )

        self.assertEqual(k_producer.q.shape[-2], 1)
        self.assertEqual(q_producer.k.shape[-2], 1)
        self.assertEqual(k_producer._q_ranges, [(0, 1)])
        self.assertEqual(k_producer._k_ranges, [(0, 128), (128, 256)])
        self.assertEqual(q_producer._q_ranges, [(0, 128), (128, 256)])
        self.assertEqual(q_producer._k_ranges, [(0, 1)])
        self.assertEqual(
            calls,
            [
                ((1, 2, 1, 128), (1, 2, 128, 128), 0, 0),
                ((1, 2, 128, 128), (1, 2, 1, 128), 0, 0),
                ((1, 2, 1, 128), (1, 2, 128, 128), 0, 128),
                ((1, 2, 128, 128), (1, 2, 1, 128), 128, 0),
            ],
        )

    def test_policy_fp8_auto_binding_does_not_override_forced_bf16(self):
        projector = apply_policy.PolicyChunkedKitchenQKVProjector(
            force_weights_bf16=True,
        )
        expected = object()
        with mock.patch.object(
            apply_policy,
            'describe_linear',
            return_value=SimpleNamespace(fp8=True),
        ), mock.patch.object(
            apply_policy._BASE_KITCHEN_PROJECTOR,
            'try_project',
            autospec=True,
            return_value=expected,
        ) as base_try:
            actual = projector.try_project(
                SimpleNamespace(qkv_proj=object()),
                object(),
                None,
                layer_index=0,
                transformer_options={},
            )

        self.assertIs(actual, expected)
        self.assertIs(base_try.call_args.args[0], projector)

    def test_sparse_producer_retains_only_tile_summaries(self):
        fake = FakeKitchen()
        module = SimpleNamespace(qkv_proj=object(), heads=2, head_dim=4)
        x = torch.arange(10, dtype=torch.float32).unsqueeze(1).expand(10, 6)

        def project(_module, values, _rope_rows):
            rows = values[:, 0].to(torch.int64)
            base = rows.to(torch.float32).view(-1, 1, 1)
            q = base.expand(-1, 2, 4).clone()
            return q, q + 100, q + 200

        with mock.patch.object(
            kitchen_qkv,
            'resolve_kitchen',
            return_value=fake,
        ), mock.patch.object(
            kitchen_qkv,
            'describe_linear',
            return_value=SimpleNamespace(plain_float=False, w4a8=False),
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
                None,
                layer_index=0,
                transformer_options={},
                spec=fake.int8_attention_producer_spec(),
                chunk_rows=4,
                routing_summaries=True,
            )

        self.assertEqual(tuple(prepared.q_summary.shape), (1, 2, 3, 4))
        self.assertEqual(tuple(prepared.k_summary.shape), (1, 2, 3, 4))
        self.assertTrue(
            torch.equal(
                prepared.q_summary[0, 0, :, 0],
                torch.tensor([1.5, 5.5, 8.5]),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.k_summary[0, 0, :, 0],
                torch.tensor([101.5, 105.5, 108.5]),
            )
        )
        self.assertTrue(
            torch.equal(
                fake.anchor_samples[0, 0, :, 0],
                torch.arange(9, dtype=torch.float32) + 100,
            )
        )

    def test_native_bf16_run_uses_held_bf16_binding(self):
        fake = FakeKitchen()
        module = SimpleNamespace(qkv_proj=object(), heads=2, head_dim=4)
        x = torch.arange(10, dtype=torch.bfloat16).unsqueeze(1).expand(10, 6)
        held = SimpleNamespace(entered=False, exited=False)

        def project_rows(_x, _rope, rows):
            base = rows.to(torch.bfloat16).view(1, 1, -1, 1)
            q = base.expand(1, 2, -1, 4).clone()
            return q, q + 100, q + 200

        def project_hnd(_x, _rope, start, end):
            rows = torch.arange(start, end, dtype=torch.bfloat16).view(1, 1, -1, 1)
            q = rows.expand(1, 2, -1, 4).clone()
            return q, q + 100, q + 200

        held.project_rows = project_rows
        held.project_hnd = project_hnd
        held.__enter__ = lambda: setattr(held, 'entered', True) or held
        held.__exit__ = lambda *_args: setattr(held, 'exited', True) or False

        class Binding:
            def __init__(self, _module, _sample):
                pass

            def __enter__(self):
                return held.__enter__()

            def __exit__(self, *args):
                return held.__exit__(*args)

            def project_rows(self, *args):
                return held.project_rows(*args)

            def project_hnd(self, *args):
                return held.project_hnd(*args)

        with mock.patch.object(
            kitchen_qkv,
            'resolve_kitchen',
            return_value=fake,
        ), mock.patch.object(
            kitchen_qkv,
            'describe_linear',
            return_value=SimpleNamespace(
                plain_float=True,
                logical_dtype='torch.bfloat16',
                w4a8=False,
            ),
        ), mock.patch.object(
            kitchen_qkv,
            'HeldBF16QKV',
            Binding,
        ):
            prepared = kitchen_qkv.run_chunked_kitchen_qkv(
                module,
                x,
                None,
                layer_index=0,
                transformer_options={},
                spec=fake.int8_attention_producer_spec(),
                chunk_rows=4,
            )

        self.assertTrue(held.entered)
        self.assertTrue(held.exited)
        self.assertEqual(prepared.carrier, 'carrier')
        self.assertEqual(
            [entry[:2] for entry in fake.qk_chunks],
            [(0, 0), (4, 4), (8, 8)],
        )
        self.assertEqual(fake.v.dtype, torch.bfloat16)

    def test_public_api_probe_requires_the_complete_contract(self):
        fake = FakeKitchen()
        self.assertTrue(kitchen_qkv.producer_api_available(fake))
        legacy = SimpleNamespace(
            **{
                name: getattr(fake, name)
                for name in kitchen_qkv.PRODUCER_API
            }
        )
        self.assertTrue(kitchen_qkv.producer_api_available(legacy))
        self.assertFalse(
            kitchen_qkv._supports_streamed_producer(legacy, None)
        )
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

    def test_sparse_backend_consumes_projected_carrier_and_summaries(self):
        class Route:
            def __init__(self, **kwargs):
                vars(self).update(kwargs)

        class SparseKitchen:
            BlockSparseRoute = Route
            __version__ = 'test'

        class Layout:
            seq_len = 256
            video_range = (128, 256)
            segments = ((0, 128, 'text'), (128, 256, 'video'))
            video_shape = (1, 8, 16)
            audio_t = 0

        projector = kitchen_qkv.ChunkedKitchenQKVProjector(
            routing_summaries=True
        )
        backend = SparseKitchenBackend(
            HybridSparseConfig(mode=MODE_SAGE128_FUSED_QKV),
            kitchen=SparseKitchen(),
            projector=projector,
            allow_cpu_for_tests=True,
        )
        carrier = SimpleNamespace(
            q=torch.empty((1, 2, 256, 128), dtype=torch.int8),
            k=torch.empty((1, 2, 256, 128), dtype=torch.int8),
            original_head_dim=128,
            cta_k=128,
        )
        projected = kitchen_qkv.PreparedChunkedKitchenQKV(
            carrier,
            q_summary=torch.ones((1, 2, 2, 128)),
            k_summary=torch.ones((1, 2, 2, 128)),
        )
        options = {
            RUNTIME_KEY: RuntimeSnapshot(
                request_id=1,
                step_index=0,
                total_steps=4,
                layout=Layout(),
                compute_dtype=torch.bfloat16,
                device=torch.device('cpu'),
            )
        }

        prepared = backend.prepare_projected(
            projected,
            layer_index=3,
            transformer_options=options,
        )

        self.assertIs(prepared.quantized, carrier)
        self.assertEqual(prepared.route.q_tile, 128)
        self.assertEqual(prepared.route.kv_tile, 128)
        self.assertEqual(prepared.layer_index, 3)
        self.assertTrue(backend.as_status()['fused_qkv'])

    def test_sparse_projector_does_not_require_a_dense_override(self):
        fake = FakeKitchen()
        module = SimpleNamespace(qkv_proj=object(), heads=2, head_dim=4)
        x = SimpleNamespace(
            ndim=2,
            is_cuda=True,
            shape=(8, 8),
            dtype=torch.bfloat16,
            device=torch.device('cuda:0'),
        )
        projector = kitchen_qkv.ChunkedKitchenQKVProjector(
            routing_summaries=True,
            q_tile=64,
            kv_tile=64,
        )
        expected = object()
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
        ), mock.patch.object(
            kitchen_qkv,
            'run_chunked_kitchen_qkv',
            return_value=expected,
        ) as run:
            actual = projector.try_project(
                module,
                x,
                None,
                layer_index=0,
                transformer_options={},
            )

        self.assertIs(actual, expected)
        self.assertEqual(fake.spec_requests[-1]['cta_k'], 64)
        self.assertEqual(run.call_args.kwargs['routing_q_tile'], 64)
        self.assertEqual(run.call_args.kwargs['routing_kv_tile'], 64)

    def test_sparse_projector_accepts_native_bf16_for_kitchen_carrier(self):
        fake = FakeKitchen()
        module = SimpleNamespace(qkv_proj=object(), heads=2, head_dim=4)
        x = SimpleNamespace(
            ndim=2,
            is_cuda=True,
            shape=(8, 8),
            dtype=torch.bfloat16,
            device=torch.device('cuda:0'),
        )
        projector = kitchen_qkv.ChunkedKitchenQKVProjector(
            routing_summaries=True,
            q_tile=64,
            kv_tile=64,
        )
        expected = object()
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
            return_value=SimpleNamespace(
                fp8=False,
                plain_float=True,
                logical_dtype='torch.bfloat16',
                convrot_int8_256=False,
                w4a8=False,
            ),
        ), mock.patch.object(
            kitchen_qkv,
            'run_chunked_kitchen_qkv',
            return_value=expected,
        ) as run:
            actual = projector.try_project(
                module,
                x,
                None,
                layer_index=0,
                transformer_options={},
            )

        self.assertIs(actual, expected)
        self.assertFalse(run.call_args.kwargs['force_weights_bf16'])
        self.assertFalse(run.call_args.kwargs['fp8_projection'])
        self.assertEqual(fake.spec_requests[-1]['cta_k'], 64)

    def test_forced_bf16_projector_streams_quantized_source_into_kitchen(self):
        fake = FakeKitchen()
        module = SimpleNamespace(qkv_proj=object(), heads=2, head_dim=4)
        x = SimpleNamespace(
            ndim=2,
            is_cuda=True,
            shape=(8, 8),
            dtype=torch.bfloat16,
            device=torch.device('cuda:0'),
        )
        projector = kitchen_qkv.ChunkedKitchenQKVProjector(
            force_weights_bf16=True,
            routing_summaries=True,
            q_tile=64,
            kv_tile=64,
        )
        expected = object()
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
            return_value=SimpleNamespace(
                fp8=False,
                plain_float=False,
                convrot_int8_256=True,
                w4a8=False,
            ),
        ), mock.patch.object(
            kitchen_qkv,
            'run_chunked_kitchen_qkv',
            return_value=expected,
        ) as run:
            actual = projector.try_project(
                module,
                x,
                None,
                layer_index=0,
                transformer_options={},
            )

        self.assertIs(actual, expected)
        self.assertTrue(run.call_args.kwargs['force_weights_bf16'])
        self.assertFalse(run.call_args.kwargs['fp8_projection'])
        self.assertFalse(run.call_args.kwargs['convrot_int8_projection'])

    def test_forced_bf16_run_materializes_binding_once_for_all_chunks(self):
        fake = FakeKitchen()
        module = SimpleNamespace(qkv_proj=object(), heads=2, head_dim=4)
        x = torch.arange(10, dtype=torch.bfloat16).unsqueeze(1).expand(10, 6)
        binding = SimpleNamespace(entered=False, exited=False)

        def project_rows(_x, _rope, rows):
            base = rows.to(torch.bfloat16).view(1, 1, -1, 1)
            q = base.expand(1, 2, -1, 4).clone()
            return q, q + 100, q + 200

        def project_hnd(_x, _rope, start, end):
            rows = torch.arange(start, end, dtype=torch.bfloat16).view(1, 1, -1, 1)
            q = rows.expand(1, 2, -1, 4).clone()
            return q, q + 100, q + 200

        binding.project_rows = project_rows
        binding.project_hnd = project_hnd

        class Binding:
            def __init__(self, _module, _sample, *, allow_quantized_source=False):
                self.allow_quantized_source = allow_quantized_source

            def __enter__(self):
                binding.entered = True
                binding.allow_quantized_source = self.allow_quantized_source
                return self

            def __exit__(self, *_args):
                binding.exited = True
                return False

            def project_rows(self, *args):
                return binding.project_rows(*args)

            def project_hnd(self, *args):
                return binding.project_hnd(*args)

        with mock.patch.object(
            kitchen_qkv,
            'resolve_kitchen',
            return_value=fake,
        ), mock.patch.object(
            kitchen_qkv,
            'describe_linear',
            return_value=SimpleNamespace(
                plain_float=False,
                convrot_int8_256=True,
                w4a8=False,
                fp8=False,
            ),
        ), mock.patch.object(
            kitchen_qkv,
            'HeldBF16QKV',
            Binding,
        ):
            prepared = kitchen_qkv.run_chunked_kitchen_qkv(
                module,
                x,
                None,
                layer_index=0,
                transformer_options={},
                spec=fake.int8_attention_producer_spec(),
                chunk_rows=4,
                force_weights_bf16=True,
            )

        self.assertTrue(binding.entered)
        self.assertTrue(binding.exited)
        self.assertTrue(binding.allow_quantized_source)
        self.assertEqual(prepared.carrier, 'carrier')
        self.assertEqual(
            [entry[:2] for entry in fake.qk_chunks],
            [(0, 0), (4, 4), (8, 8)],
        )

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
    def test_streamed_projector_keeps_the_disposable_input_with_the_carrier(self):
        fake = FakeKitchen()
        module = SimpleNamespace(qkv_proj=object(), heads=2, head_dim=4)
        x = SimpleNamespace(
            ndim=2,
            is_cuda=True,
            shape=(8, 8),
            dtype=torch.bfloat16,
            device=torch.device('cuda:0'),
        )
        projector = kitchen_qkv.ChunkedKitchenQKVProjector(
            routing_summaries=True,
            stream_output=True,
        )
        projected = kitchen_qkv.PreparedChunkedKitchenQKV(object())
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
        ), mock.patch.object(
            kitchen_qkv,
            'run_chunked_kitchen_qkv',
            return_value=projected,
        ):
            actual = projector.try_project(
                module,
                x,
                None,
                layer_index=0,
                transformer_options={},
            )

        self.assertIs(actual.output_buffer, x)
        self.assertIsNone(projected.output_buffer)


if __name__ == '__main__':
    unittest.main()

