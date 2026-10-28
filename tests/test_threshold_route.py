'''CPU contracts for native threshold routing.'''

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

from h3_optimizations.attention.sparse import kitchen_streamed_q as ksq  # noqa: E402
from h3_optimizations.attention.sparse.router import SparseTileRouter  # noqa: E402
from h3_optimizations.native import threshold_route  # noqa: E402
from h3_optimizations.native import int8_attention, selftest  # noqa: E402
from h3_optimizations.attention.sparse.kitchen_sparse import preflight_sparse_kitchen, SparseKitchenError  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class ThresholdRouteTests(unittest.TestCase):
    def test_budget_maps_to_the_gaussian_quantile(self):
        self.assertAlmostEqual(threshold_route.budget_tau(0.5), 0.0, places=6)
        self.assertAlmostEqual(threshold_route.budget_tau(0.15), 1.0364, places=3)
        self.assertAlmostEqual(threshold_route.budget_tau(0.30), 0.5244, places=3)
        self.assertGreater(threshold_route.budget_tau(0.05), threshold_route.budget_tau(0.10))
        # clamped instead of infinite at the ends
        self.assertTrue(abs(threshold_route.budget_tau(0.0)) < 10)
        self.assertTrue(abs(threshold_route.budget_tau(1.0)) < 10)

    def test_statistics_are_the_population_mean_and_covariance(self):
        torch.manual_seed(0)
        keys = (torch.randn(1, 2, 300, 128) + 3.0).to(torch.bfloat16)
        mean, covariance = threshold_route.video_statistics(keys)
        reference = keys.float()
        centred = reference - reference.mean(dim=-2, keepdim=True)
        expected = centred.transpose(-1, -2) @ centred / reference.shape[-2]
        self.assertEqual(mean.dtype, torch.float32)
        self.assertEqual(covariance.dtype, torch.bfloat16)
        self.assertTrue(torch.allclose(mean, reference.mean(dim=-2), atol=1e-5))
        self.assertTrue(torch.allclose(covariance.float(), expected, atol=2e-2, rtol=2e-2))
        # a row's score variance over the tiles is q^T Sigma q
        q = torch.randn(128)
        scores = reference[0, 0] @ q
        self.assertAlmostEqual(
            float(q @ expected[0, 0] @ q), float(scores.var(unbiased=False)), delta=1e-2 * float(scores.var()))

    def test_unavailable_without_cuda(self):
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            self.assertFalse(threshold_route.threshold_route_is_available())

    def test_plan_falls_back_to_top_k_when_the_native_router_is_unavailable(self):
        layout = SimpleNamespace(seq_len=8192, video_range=(1024, 8192),
                                 segments=((0, 1024, 'text'), (1024, 8192, 'video')),
                                 video_shape=(7, 32, 32), audio_t=0)
        router = SparseTileRouter(None, q_tile=64, kv_tile=64)
        geometry = router.geometry(layout)
        summary = torch.zeros(1, 2, geometry.kv_tiles, 128, dtype=torch.bfloat16)
        with mock.patch.object(threshold_route, 'threshold_route_is_available', return_value=False):
            plan, _ = ksq._prepare_route_plan(router, summary, layout, 0.15,
                                              mode=ksq.ROUTE_MODE_THRESHOLD)
        self.assertEqual(plan.mode, ksq.ROUTE_MODE_TOPK)
        self.assertIsNone(plan.k_cov)
        with mock.patch.object(threshold_route, 'threshold_route_is_available', return_value=True):
            plan, _ = ksq._prepare_route_plan(router, summary, layout, 0.15,
                                              mode=ksq.ROUTE_MODE_THRESHOLD)
        self.assertEqual(plan.mode, ksq.ROUTE_MODE_THRESHOLD)
        self.assertEqual(tuple(plan.k_cov.shape), (1, 2, 128, 128))
        self.assertAlmostEqual(
            plan.tau, threshold_route.budget_tau(plan.retained / geometry.pure_video_kv_tiles))
        plan.release()
        self.assertIsNone(plan.k_cov)

    def test_default_mode_is_threshold(self):
        self.assertEqual(ksq.DEFAULT_ROUTE_MODE, ksq.ROUTE_MODE_THRESHOLD)
        self.assertEqual(ksq.ROUTE_MODE_KEY, 'h3_sparse_route_mode')

    def test_packed32_failure_is_rejected_before_backend_installation(self):
        module = SimpleNamespace(
            int8_attention_is_available=lambda: True,
            SPARSE_GEOMETRIES=((64, 32), (64, 64)),
            sparse_geometry_is_available=lambda q, k, device: (q, k) == (64, 64),
        )
        with self.assertRaisesRegex(SparseKitchenError, '64Q x 32KV failed'):
            preflight_sparse_kitchen(cuda_available=lambda: True,
                capability_getter=lambda: (8, 9), kitchen=module, q_tile=64, kv_tile=32)
        self.assertIs(preflight_sparse_kitchen(cuda_available=lambda: True,
            capability_getter=lambda: (8, 9), kitchen=module, q_tile=64, kv_tile=64), module)

    def test_native_geometry_probe_preserves_only_the_carrier_compatible_fallback(self):
        with mock.patch.object(int8_attention, 'int8_attention_is_available', return_value=True), \
             mock.patch.object(selftest, 'sparse_geometry_check',
                               side_effect=lambda q, k, device: (q, k) == (128, 64)):
            self.assertFalse(int8_attention.sparse_geometry_is_available(64, 32))
            self.assertTrue(int8_attention.sparse_geometry_is_available(64, 64))


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
