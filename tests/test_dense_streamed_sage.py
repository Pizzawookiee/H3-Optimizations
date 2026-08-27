"""CPU contracts for all-architecture dense Sage Q/output streaming."""

import os
from pathlib import Path
import sys
import unittest

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.dense_streamed_sage import (  # noqa: E402
    StreamedDenseSageBackend,
    StreamedDenseSageQKVProjector,
)
from h3_optimizations.attention_forward import _AttentionOutProjectionProxy  # noqa: E402
from h3_optimizations.attention.sage_v_staging import _permuted_rows  # noqa: E402
from h3_optimizations.plan import V_MEMORY_TWO_PASS  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakeHeld:
    def __init__(self, factory):
        self.factory = factory
        self.active = False
        self.kv_calls = []
        self.q_calls = []

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *_args):
        self.active = False
        return False

    @staticmethod
    def _hnd(x, start, stop):
        rows = stop - start
        return x[start:stop].reshape(rows, 2, 128).transpose(0, 1).unsqueeze(0)

    def project_kv_hnd(self, x, _rope, start, stop):
        self.kv_calls.append((start, stop))
        base = self._hnd(x, start, stop)
        return base + 10, base + 20

    def project_q_hnd(self, x, _rope, start, stop):
        self.q_calls.append((start, stop))
        return self._hnd(x, start, stop)


class FakeHeldFactory:
    def __init__(self):
        self.bindings = []

    def __call__(self, *_args):
        held = FakeHeld(self)
        self.bindings.append(held)
        return held


class FakeSage:
    name = "sage_mem_eff_test"
    projected_q_tile = 2
    projected_k_tile = 2
    requires_registered_sage = True
    requires_runtime_context = False
    runtime_listeners = ()

    def __init__(self, factory):
        self.factory = factory
        self.launches = []

    @staticmethod
    def _scale(value):
        return torch.ones(
            value.shape[0],
            value.shape[1],
            (value.shape[-2] + 1) // 2,
            dtype=torch.float32,
        )

    def quantize_projected_k(self, k):
        return k.to(torch.int8), self._scale(k)

    def quantize_projected_q(self, q):
        return q.to(torch.int8), self._scale(q)

    @staticmethod
    def prepare_streamed_v(v):
        return v.clone(), None

    def execute_rectangular(
        self,
        q,
        _q_scale,
        k,
        _k_scale,
        v,
        _v_scale,
        **_kwargs,
    ):
        self.assert_no_active_binding()
        self.launches.append((q.shape[-2], k.shape[-2], v.shape[-2]))
        return q.to(torch.float32)

    def assert_no_active_binding(self):
        if any(binding.active for binding in self.factory.bindings):
            raise AssertionError("a source weight binding escaped into Sage execution")


class Module:
    heads = 2
    head_dim = 128
    out_proj = staticmethod(lambda value: value)


class DenseStreamedSageTests(unittest.TestCase):
    def test_global_kv_and_bounded_q_output_use_the_architecture_adapter(self):
        factory = FakeHeldFactory()
        sage = FakeSage(factory)
        projector = StreamedDenseSageQKVProjector(
            sage,
            chunk_rows=2,
            projection_mode="native",
            held_factory=factory,
            allow_cpu_for_tests=True,
        )
        backend = StreamedDenseSageBackend(sage)
        module = Module()
        x = (torch.arange(5 * 256).reshape(5, 256) % 100).to(torch.float16)
        expected = x.clone()

        projected = projector.project(
            module,
            x,
            None,
            layer_index=7,
            transformer_options={},
        )

        self.assertFalse(hasattr(projected, "q_int8"))
        self.assertEqual(tuple(projected.k_int8.shape), (1, 2, 5, 128))
        self.assertEqual(factory.bindings[0].kv_calls, [(0, 2), (2, 4), (4, 5)])
        self.assertFalse(factory.bindings[0].active)

        prepared = backend.prepare_projected(
            projected,
            layer_index=7,
            transformer_options={},
        )
        proxy = _AttentionOutProjectionProxy(module, lambda value: value)
        actual = backend.execute_projected(proxy, prepared)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(sage.launches, [(2, 5, 5), (2, 5, 5), (1, 5, 5)])
        self.assertEqual(
            [binding.q_calls for binding in factory.bindings[1:]],
            [[(0, 2)], [(2, 4)], [(4, 5)]],
        )
        self.assertTrue(all(not binding.active for binding in factory.bindings))
        self.assertIsNone(prepared.projected.k_int8)
        self.assertIsNone(prepared.v_carrier)


class StagingHeld(FakeHeld):
    """FakeHeld plus the V-only projection two-pass staging needs."""

    def __init__(self, factory):
        super().__init__(factory)
        self.v_calls = []

    def project_v_hnd(self, x, _rope, start, stop):
        self.v_calls.append((start, stop))
        return self._hnd(x, start, stop) + 20


class StagingHeldFactory:
    def __init__(self):
        self.bindings = []

    def __call__(self, *_args):
        held = StagingHeld(self)
        self.bindings.append(held)
        return held


class StagingSage(FakeSage):
    """FakeSage that advertises two-pass V support."""

    def __init__(self, factory):
        super().__init__(factory)
        self.prepare_streamed_v_calls = 0

    def v_staging_parameters(self):
        return (2.25, 64)

    def prepare_streamed_v(self, v):
        self.prepare_streamed_v_calls += 1
        return v.clone(), None


class DenseSageTwoPassVTests(unittest.TestCase):
    SEQUENCE = 32
    CHUNK_ROWS = 16

    def _run(self):
        factory = StagingHeldFactory()
        sage = StagingSage(factory)
        projector = StreamedDenseSageQKVProjector(
            sage,
            chunk_rows=self.CHUNK_ROWS,
            projection_mode="native",
            v_mode=V_MEMORY_TWO_PASS,
            held_factory=factory,
            allow_cpu_for_tests=True,
        )
        module = Module()
        x = (
            torch.arange(self.SEQUENCE * 256).reshape(self.SEQUENCE, 256) % 100
        ).to(torch.float16)
        projected = projector.project(
            module, x, None, layer_index=0, transformer_options={}
        )
        return factory, sage, projector, x, projected

    def test_two_pass_never_materializes_a_full_bf16_v(self):
        factory, _sage, projector, _x, projected = self._run()

        self.assertEqual(projector.v_mode, V_MEMORY_TWO_PASS)
        self.assertIsNone(projected.v)
        self.assertEqual(
            tuple(projected.staged_v_carrier.shape), (1, 2, 128, 64)
        )
        self.assertEqual(projected.staged_v_carrier.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(projected.staged_v_scale.shape), (1, 2, 128))
        # V is reprojected once per chunk in the second pass, and only V.
        self.assertEqual(factory.bindings[0].v_calls, [(0, 16), (16, 32)])
        self.assertEqual(factory.bindings[0].kv_calls, [(0, 16), (16, 32)])

    def test_staged_carrier_matches_a_single_pass_build(self):
        _factory, _sage, _projector, x, projected = self._run()

        full_v = torch.cat(
            [
                FakeHeld._hnd(x, 0, 16) + 20,
                FakeHeld._hnd(x, 16, 32) + 20,
            ],
            dim=-2,
        )
        amax = full_v.to(torch.float32).abs().amax(dim=-2)
        quantized = full_v.to(torch.float32) * (2.25 / amax.unsqueeze(-2))
        expected = torch.zeros((1, 2, 128, 64), dtype=torch.float8_e4m3fn)
        destination = _permuted_rows(torch.arange(32, dtype=torch.int64))
        packed = (
            quantized.permute(0, 1, 3, 2).contiguous().to(torch.float8_e4m3fn)
        )
        expected.view(torch.uint8).index_copy_(
            3, destination, packed.view(torch.uint8)
        )

        self.assertTrue(
            torch.equal(
                projected.staged_v_carrier.view(torch.uint8),
                expected.view(torch.uint8),
            )
        )
        self.assertTrue(torch.equal(projected.staged_v_scale, amax / 2.25))

    def test_prepare_adopts_the_staged_carrier_without_requantizing(self):
        _factory, sage, _projector, _x, projected = self._run()
        backend = StreamedDenseSageBackend(sage)
        carrier = projected.staged_v_carrier

        prepared = backend.prepare_projected(
            projected, layer_index=0, transformer_options={}
        )

        self.assertIs(prepared.v_carrier, carrier)
        self.assertEqual(sage.prepare_streamed_v_calls, 0)
        self.assertIsNone(prepared.projected.staged_v_carrier)

    def test_retain_mode_still_builds_a_full_v(self):
        factory = StagingHeldFactory()
        sage = StagingSage(factory)
        projector = StreamedDenseSageQKVProjector(
            sage,
            chunk_rows=self.CHUNK_ROWS,
            projection_mode="native",
            held_factory=factory,
            allow_cpu_for_tests=True,
        )
        x = (
            torch.arange(self.SEQUENCE * 256).reshape(self.SEQUENCE, 256) % 100
        ).to(torch.float16)

        projected = projector.project(
            Module(), x, None, layer_index=0, transformer_options={}
        )

        self.assertEqual(tuple(projected.v.shape), (1, 2, self.SEQUENCE, 128))
        self.assertIsNone(projected.staged_v_carrier)
        self.assertEqual(factory.bindings[0].v_calls, [])


if __name__ == "__main__":
    unittest.main()
