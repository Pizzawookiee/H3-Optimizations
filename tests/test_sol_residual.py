'''CPU contracts for native exact attention plus the INT8 Sol residual.'''

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

import h3_optimizations.apply as apply_module  # noqa: E402
from h3_optimizations.attention.sparse.config import HybridSparseConfig  # noqa: E402
from h3_optimizations.attention.sparse.sol_residual import (  # noqa: E402
    EXACT_KV_TILE,
    EXACT_Q_TILE,
    RESIDUAL_TILE,
    SolResidualBackend,
    SolResidualError,
    SolResidualSpec,
    _summarize_kv_cpu,
    pack_exact_route,
    preflight_sol_residual,
)
from h3_optimizations.kitchen_qkv import (  # noqa: E402
    ChunkedKitchenQKVProjector,
    PreparedChunkedKitchenQKV,
)
from h3_optimizations.native.int8_attention import BlockSparseRoute  # noqa: E402
from h3_optimizations.plan import (  # noqa: E402
    H3OptimizationPlan,
    MemoryRequest,
    SPARSE_BACKEND_NATIVE_128X64,
    SPARSE_BACKEND_NATIVE_64X64,
    SPARSE_BACKEND_NATIVE_HARD,
    SPARSE_BACKEND_SOL,
    SPARSE_BACKEND_SOL_128X64,
    SPARSE_BACKEND_SOL_64X64,
    SparseRequest,
)
from h3_optimizations.qkv.providers import QKV_DENSE_KITCHEN_CHUNKED  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def make_route(
    sequence,
    *,
    selected=(0,),
    q_tile=EXACT_Q_TILE,
    kv_tile=EXACT_KV_TILE,
):
    q_tiles = (sequence + q_tile - 1) // q_tile
    kv_tiles = (sequence + kv_tile - 1) // kv_tile
    indices = torch.zeros(1, 1, q_tiles, kv_tiles, dtype=torch.int32)
    indices[..., :len(selected)] = torch.tensor(selected, dtype=torch.int32)
    counts = torch.full((1, 1, q_tiles), len(selected), dtype=torch.int32)
    return BlockSparseRoute(
        indices=indices,
        counts=counts,
        q_tile=q_tile,
        kv_tile=kv_tile,
        encoding='absolute',
    )


def make_carrier(sequence=256, *, cta_k=EXACT_KV_TILE):
    batch = heads = 1
    head_dim = 128
    padded = ((sequence + 127) // 128) * 128
    exact_tiles = (sequence + cta_k - 1) // cta_k
    return SimpleNamespace(
        q=torch.ones(batch, heads, sequence, head_dim, dtype=torch.int8),
        k=torch.ones(batch, heads, sequence, head_dim, dtype=torch.int8),
        v=torch.ones(batch * heads * head_dim, padded, dtype=torch.int8),
        q_scale=torch.ones(batch, heads, ((sequence + 127) // 128) * 32),
        k_scale=torch.full((batch, heads, exact_tiles * 4), 2.0),
        v_scale=torch.ones(batch * heads * head_dim),
        original_head_dim=head_dim,
        input_dtype=torch.bfloat16,
        attention_scale=head_dim ** -0.5,
        cta_k=cta_k,
    )


class FakePreparedExact:
    def __init__(self, carrier, route):
        self.quantized = carrier
        self.route = route
        self.metadata = {'requested_video_budget': 0.5}
        self.released = False

    def release(self):
        self.released = True
        self.quantized = None
        self.route = None


class FakeExactBackend:
    installation_signature = ('fake_exact',)

    def __init__(self, route):
        self.router = SimpleNamespace(q_tile=route.q_tile, kv_tile=route.kv_tile)
        self.route = route
        self.calls = []

    def prepare_projected(self, projected, **_kwargs):
        self.calls.append('prepare_projected')
        return FakePreparedExact(projected.carrier, self.route)

    def execute(self, prepared):
        self.calls.append('execute')
        q = prepared.quantized.q
        return torch.zeros(q.shape, dtype=torch.bfloat16)

    def execute_with_lse(self, prepared):
        self.calls.append('execute_with_lse')
        q = prepared.quantized.q
        output = torch.zeros(q.shape, dtype=torch.bfloat16)
        lse = torch.zeros(q.shape[:3], dtype=torch.float32)
        return output, lse


def unpack_route(packed, kv_tiles):
    decoded = torch.empty((*packed.shape[:-1], kv_tiles), dtype=torch.bool)
    for block in range(kv_tiles):
        decoded[..., block] = (
            (packed[..., block // 32] >> (block % 32)) & 1
        ).bool()
    return decoded


class RouteBitsetTests(unittest.TestCase):
    def test_delta_route_is_packed_as_selected_128kv_parents(self):
        absolute = make_route(512, selected=(0, 2, 3))
        route = absolute.to_delta()
        packed = pack_exact_route(route, q_tiles=4, kv_tiles=4)
        decoded = unpack_route(packed, 4)
        expected = torch.tensor([True, False, True, True])
        self.assertTrue(torch.equal(decoded[0, 0, 0], expected))
        self.assertTrue(torch.equal(decoded[0, 0, 3], expected))


class CarrierSummaryTests(unittest.TestCase):
    def test_int8_k_and_v_are_decoded_into_64_row_summaries(self):
        carrier = make_carrier(130)
        k_mean, v_sum = _summarize_kv_cpu(carrier)
        self.assertEqual(tuple(k_mean.shape), (1, 1, 3, 128))
        self.assertEqual(tuple(v_sum.shape), (1, 1, 3, 128))
        self.assertTrue(torch.equal(k_mean.float(), torch.full_like(k_mean.float(), 2.0)))
        self.assertEqual(v_sum[0, 0, :, 0].float().tolist(), [64.0, 64.0, 2.0])

    def test_k64_carrier_uses_each_native_k_scale_tile(self):
        carrier = make_carrier(130, cta_k=64)
        carrier.k_scale.copy_(torch.tensor(
            [1.0] * 4 + [2.0] * 4 + [3.0] * 4
        ).reshape_as(carrier.k_scale))
        k_mean, _v_sum = _summarize_kv_cpu(carrier)
        self.assertEqual(k_mean[0, 0, :, 0].float().tolist(), [1.0, 2.0, 3.0])


class SolResidualBackendTests(unittest.TestCase):
    def test_both_arms_share_native_exact_and_only_sol_launches_residual(self):
        carrier = make_carrier()
        projected = PreparedChunkedKitchenQKV(
            carrier,
            q_summary=torch.zeros(1, 1, 2, 128, dtype=torch.bfloat16),
            k_summary=torch.zeros(1, 1, 2, 128, dtype=torch.bfloat16),
        )
        route = make_route(256, selected=(0,))
        residual_routes = []

        def launcher(prepared, exact_output, exact_lse):
            self.assertEqual(prepared.q.dtype, torch.int8)
            self.assertEqual(prepared.q_scale.dtype, torch.float32)
            self.assertEqual(exact_lse.dtype, torch.float32)
            residual_routes.append(prepared.exact_route.clone())
            return exact_output

        hard_exact = FakeExactBackend(route)
        hard = SolResidualBackend(
            HybridSparseConfig(video_budget=0.5),
            approximate_rejected=False,
            exact_backend=hard_exact,
            allow_cpu_for_tests=True,
            launcher=launcher,
        )
        hard_prepared = hard.prepare_projected(
            projected,
            layer_index=3,
            transformer_options={},
        )
        hard_output = hard.execute(hard_prepared)

        sol_exact = FakeExactBackend(route)
        sol = SolResidualBackend(
            HybridSparseConfig(video_budget=0.5),
            approximate_rejected=True,
            exact_backend=sol_exact,
            allow_cpu_for_tests=True,
            launcher=launcher,
        )
        sol_prepared = sol.prepare_projected(
            projected,
            layer_index=3,
            transformer_options={},
        )
        sol_output = sol.execute(sol_prepared)

        self.assertEqual(tuple(hard_output.shape), tuple(carrier.q.shape))
        self.assertEqual(tuple(sol_output.shape), tuple(carrier.q.shape))
        self.assertEqual(hard_exact.calls, ['prepare_projected', 'execute'])
        self.assertEqual(sol_exact.calls, ['prepare_projected', 'execute_with_lse'])
        self.assertEqual(len(residual_routes), 1)
        self.assertTrue(torch.equal(residual_routes[0], pack_exact_route(route, q_tiles=2, kv_tiles=2)))
        self.assertEqual(hard_prepared.metadata['rejected_blocks'], 'dropped')
        self.assertEqual(sol_prepared.metadata['rejected_blocks'], 'sol_int8_k_mean_v_sum_64x64')

    def test_sol_residual_matches_128x64_and_64x64_exact_geometry(self):
        for q_tile, kv_tile in ((128, 64), (64, 64)):
            with self.subTest(geometry=(q_tile, kv_tile)):
                carrier = make_carrier(cta_k=kv_tile)
                projected = PreparedChunkedKitchenQKV(
                    carrier,
                    q_summary=torch.zeros(
                        1, 1, 256 // q_tile, 128, dtype=torch.bfloat16
                    ),
                    k_summary=torch.zeros(
                        1, 1, 256 // kv_tile, 128, dtype=torch.bfloat16
                    ),
                )
                route = make_route(
                    256,
                    selected=(0,),
                    q_tile=q_tile,
                    kv_tile=kv_tile,
                )
                exact = FakeExactBackend(route)
                backend = SolResidualBackend(
                    HybridSparseConfig(video_budget=0.5),
                    approximate_rejected=True,
                    exact_backend=exact,
                    spec=SolResidualSpec(
                        exact_q_tile=q_tile,
                        exact_kv_tile=kv_tile,
                    ),
                    allow_cpu_for_tests=True,
                    launcher=lambda prepared, output, _lse: output,
                )
                prepared = backend.prepare_projected(
                    projected,
                    layer_index=3,
                    transformer_options={},
                )
                output = backend.execute(prepared)

                self.assertEqual(tuple(output.shape), tuple(carrier.q.shape))
                self.assertEqual(prepared.exact_q_tile, q_tile)
                self.assertEqual(prepared.exact_kv_tile, kv_tile)
                self.assertEqual(
                    prepared.metadata['exact_attention'],
                    'native_int8_%dx%d' % (q_tile, kv_tile),
                )
                self.assertEqual(backend.as_status()['exact_q_tile'], q_tile)
                self.assertEqual(backend.as_status()['exact_kv_tile'], kv_tile)

    def test_preflight_requires_native_lse_and_supported_gpu(self):
        kitchen = SimpleNamespace(
            block_sparse_int8_attention_with_lse_from_prequantized=object()
        )
        spec = preflight_sol_residual(
            cuda_available=lambda: True,
            capability_getter=lambda: (8, 9),
            kitchen=kitchen,
            triton_available=True,
        )
        self.assertEqual(spec.exact_q_tile, 128)
        self.assertEqual(spec.residual_q_tile, 64)
        with self.assertRaisesRegex(SolResidualError, 'LSE output'):
            preflight_sol_residual(
                cuda_available=lambda: True,
                capability_getter=lambda: (8, 9),
                kitchen=SimpleNamespace(),
                triton_available=True,
            )

    def test_preflight_accepts_native_k64_geometries(self):
        kitchen = SimpleNamespace(
            block_sparse_int8_attention_with_lse_from_prequantized=object()
        )
        for q_tile in (128, 64):
            spec = preflight_sol_residual(
                cuda_available=lambda: True,
                capability_getter=lambda: (8, 9),
                kitchen=kitchen,
                exact_q_tile=q_tile,
                exact_kv_tile=64,
                triton_available=True,
            )
            self.assertEqual(spec.exact_q_tile, q_tile)
            self.assertEqual(spec.exact_kv_tile, 64)


class SolResidualSelectionTests(unittest.TestCase):
    def test_all_sol_routes_use_sparse_execution(self):
        self.assertIn(
            apply_module.ATTENTION_SOL_RESIDUAL,
            apply_module.SPARSE_EXECUTION_BACKENDS,
        )
        self.assertIn(
            apply_module.ATTENTION_SOL_128X64,
            apply_module.SPARSE_EXECUTION_BACKENDS,
        )
        self.assertIn(
            apply_module.ATTENTION_SOL_64X64,
            apply_module.SPARSE_EXECUTION_BACKENDS,
        )

    def test_exact_native_geometry_arms_are_explicit(self):
        target = object()
        for request, q_tile, kv_tile, selected in (
            (
                SPARSE_BACKEND_NATIVE_128X64,
                128,
                64,
                apply_module.ATTENTION_NATIVE_128X64,
            ),
            (
                SPARSE_BACKEND_NATIVE_64X64,
                64,
                64,
                apply_module.ATTENTION_NATIVE_64X64,
            ),
        ):
            with self.subTest(request=request), mock.patch.object(
                apply_module,
                '_resolve_native_geometry',
                return_value=target,
            ) as resolve, mock.patch.object(apply_module, '_resolve_dense') as dense:
                plan = H3OptimizationPlan(sparse=SparseRequest(backend=request))
                actual = apply_module._resolve_attention(
                    plan,
                    object(),
                    object(),
                    object(),
                )
            self.assertIs(actual, target)
            resolve.assert_called_once_with(
                plan,
                mock.ANY,
                mock.ANY,
                q_tile=q_tile,
                kv_tile=kv_tile,
                selected=selected,
            )
            dense.assert_not_called()

    def test_exact_native_geometry_uses_matching_router_carrier_and_projector(self):
        item = SimpleNamespace(label='TensorWiseINT8:torch.bfloat16')
        inventory = SimpleNamespace(
            qkv=(item,),
            qkv_convrot_int8_256=True,
            homogeneous=lambda name: name == 'qkv',
            labels=lambda name: tuple(
                value.label for value in getattr(inventory, name)
            ),
        )
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
            device_index=0,
        )
        kitchen = SimpleNamespace(__version__='test')
        for q_tile, kv_tile, selected in (
            (128, 64, apply_module.ATTENTION_NATIVE_128X64),
            (64, 64, apply_module.ATTENTION_NATIVE_64X64),
        ):
            with self.subTest(geometry=(q_tile, kv_tile)), mock.patch.object(
                apply_module,
                'preflight_sparse_kitchen',
                return_value=kitchen,
            ) as preflight, mock.patch.object(
                apply_module,
                'producer_api_available',
                return_value=True,
            ):
                attention, qkv = apply_module._resolve_native_geometry(
                    H3OptimizationPlan(
                        memory=MemoryRequest(),
                        sparse=SparseRequest(),
                    ),
                    environment,
                    inventory,
                    q_tile=q_tile,
                    kv_tile=kv_tile,
                    selected=selected,
                )

            self.assertEqual(qkv.provider_id, QKV_DENSE_KITCHEN_CHUNKED)
            self.assertEqual(attention.backend.router.q_tile, q_tile)
            self.assertEqual(attention.backend.router.kv_tile, kv_tile)
            self.assertEqual(attention.projector.q_tile, q_tile)
            self.assertEqual(attention.projector.kv_tile, kv_tile)
            preflight.assert_called_once_with(
                cuda_available=mock.ANY,
                capability_getter=mock.ANY,
                q_tile=q_tile,
                kv_tile=kv_tile,
            )

    def test_quality_arms_are_explicit_and_do_not_enter_auto(self):
        target = object()
        for request, approximate in (
            (SPARSE_BACKEND_NATIVE_HARD, False),
            (SPARSE_BACKEND_SOL, True),
        ):
            with self.subTest(request=request), mock.patch.object(
                apply_module,
                '_resolve_sol_experiment',
                return_value=target,
            ) as resolve, mock.patch.object(apply_module, '_resolve_dense') as dense:
                plan = H3OptimizationPlan(sparse=SparseRequest(backend=request))
                actual = apply_module._resolve_attention(
                    plan,
                    object(),
                    object(),
                    object(),
                )
            self.assertIs(actual, target)
            resolve.assert_called_once_with(
                plan,
                mock.ANY,
                mock.ANY,
                approximate_rejected=approximate,
            )
            dense.assert_not_called()

    def test_k64_sol_arms_propagate_exact_geometry(self):
        target = object()
        for request, q_tile, selected in (
            (
                SPARSE_BACKEND_SOL_128X64,
                128,
                apply_module.ATTENTION_SOL_128X64,
            ),
            (
                SPARSE_BACKEND_SOL_64X64,
                64,
                apply_module.ATTENTION_SOL_64X64,
            ),
        ):
            with self.subTest(request=request), mock.patch.object(
                apply_module,
                '_resolve_sol_experiment',
                return_value=target,
            ) as resolve, mock.patch.object(apply_module, '_resolve_dense') as dense:
                plan = H3OptimizationPlan(sparse=SparseRequest(backend=request))
                actual = apply_module._resolve_attention(
                    plan,
                    object(),
                    object(),
                    object(),
                )
            self.assertIs(actual, target)
            resolve.assert_called_once_with(
                plan,
                mock.ANY,
                mock.ANY,
                approximate_rejected=True,
                q_tile=q_tile,
                kv_tile=64,
                selected=selected,
            )
            dense.assert_not_called()

    def test_convrot_checkpoint_uses_the_normal_chunked_int8_carrier(self):
        item = SimpleNamespace(label='TensorWiseINT8:torch.bfloat16')
        inventory = SimpleNamespace(
            qkv=(item,),
            qkv_convrot_int8_256=True,
            homogeneous=lambda name: name == 'qkv',
            labels=lambda name: tuple(
                value.label for value in getattr(inventory, name)
            ),
        )
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
            device_index=0,
        )
        kitchen = SimpleNamespace(
            block_sparse_int8_attention_with_lse_from_prequantized=object()
        )
        for request in (SPARSE_BACKEND_NATIVE_HARD, SPARSE_BACKEND_SOL):
            with self.subTest(request=request), mock.patch.object(
                apply_module,
                'preflight_sparse_kitchen',
                return_value=kitchen,
            ), mock.patch.object(
                apply_module,
                'preflight_sol_residual',
                return_value=SolResidualSpec(),
            ) as residual_preflight, mock.patch.object(
                apply_module,
                'producer_api_available',
                return_value=True,
            ):
                attention, qkv = apply_module._resolve_sol_experiment(
                    H3OptimizationPlan(
                        memory=MemoryRequest(),
                        sparse=SparseRequest(backend=request),
                    ),
                    environment,
                    inventory,
                    approximate_rejected=request == SPARSE_BACKEND_SOL,
                )

            self.assertEqual(qkv.provider_id, QKV_DENSE_KITCHEN_CHUNKED)
            self.assertIsInstance(attention.projector, ChunkedKitchenQKVProjector)
            self.assertTrue(attention.projector.routing_summaries)
            if request == SPARSE_BACKEND_NATIVE_HARD:
                residual_preflight.assert_not_called()
            else:
                residual_preflight.assert_called_once_with(
                    cuda_available=mock.ANY,
                    capability_getter=mock.ANY,
                    kitchen=kitchen,
                    exact_q_tile=128,
                    exact_kv_tile=128,
                )


if __name__ == '__main__':
    unittest.main()
