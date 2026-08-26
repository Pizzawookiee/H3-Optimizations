'''CPU contracts for the packaged NVIDIA FROST-derived BF16 backend.'''

import math
import ctypes
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(PACK.parents[1]))

from h3_optimizations.attention.sparse.frost_bf16 import (
    FrostBF16Error,
    FrostBF16Executor,
    preflight_frost_bf16,
)
from h3_optimizations.attention.sparse import frost_loader
from h3_optimizations.attention.sparse.frost_route import (
    build_full_absolute_route,
    build_full_absolute_route_from_summaries,
)
from h3_optimizations.attention.sparse.router import SparseTileRouter

def layout(sequence=512, video_start=192):
    return SimpleNamespace(
        seq_len=sequence,
        video_range=(video_start, sequence),
        segments=((0, video_start, 'context'), (video_start, sequence, 'video')),
        video_shape=(1, 1, 1),
        audio_t=0,
    )


class FrostBF16Tests(unittest.TestCase):
    def test_packaged_cubin_and_symbol(self):
        frost = PACK / 'native' / 'frost'
        cubin = frost / 'h3_frost_bf16_sm89.cubin'
        symbol = (frost / 'h3_frost_bf16_sm89.symbol').read_text(
            encoding='ascii'
        ).strip()

        self.assertGreater(cubin.stat().st_size, 20_000)
        image = cubin.read_bytes()
        self.assertEqual(image[:4], b'\x7fELF')
        self.assertEqual(
            hashlib.sha256(image).hexdigest(),
            '4be095512e6a117634a3c0dddac9cedeab2722543ea2f37537b6602dd5002cd3',
        )
        self.assertTrue(symbol.startswith('kernel_cutlass__sdpa_kernel_'))
        self.assertIn('ptrbf16', symbol)

    def test_preflight_is_explicitly_sm89(self):
        spec = preflight_frost_bf16(
            cuda_available=lambda: True,
            capability_getter=lambda: (8, 9),
            driver_probe=lambda: True,
        )
        self.assertEqual(spec.signature, ('frost_bf16_sm89', 1, 64, 64, 56, 128))

        with self.assertRaisesRegex(FrostBF16Error, 'compiled for SM89'):
            preflight_frost_bf16(
                cuda_available=lambda: True,
                capability_getter=lambda: (8, 6),
                driver_probe=lambda: True,
            )

    def test_direct_absolute_route_matches_delta_route(self):
        torch.manual_seed(29)
        packed = layout()
        router = SparseTileRouter(q_tile=64, kv_tile=64)
        q = torch.randn(1, 2, packed.seq_len, 16)
        k = torch.randn_like(q)

        delta, expected_counts, expected_metadata = router.build_lut(
            q, k, packed, 0.3
        )
        route, counts, metadata = build_full_absolute_route(
            router, q, k, packed, 0.3
        )
        expected = torch.cumsum(delta, dim=-1, dtype=torch.int32)

        self.assertEqual(metadata, expected_metadata)
        self.assertTrue(torch.equal(counts, expected_counts))
        for row in range(route.shape[-2]):
            valid = int(counts[0, 0, row])
            self.assertTrue(
                torch.equal(
                    route[:, :, row, :valid],
                    expected[:, :, row, :valid],
                )
            )

    def test_full_budget_route_is_dense(self):
        packed = layout()
        router = SparseTileRouter(q_tile=64, kv_tile=64)
        q = torch.zeros(1, 1, packed.seq_len, 8)
        route, counts, metadata = build_full_absolute_route(
            router, q, q, packed, 1.0
        )

        expected = torch.arange(route.shape[-1], dtype=torch.int32)
        self.assertTrue(torch.equal(route[0, 0, 0], expected))
        self.assertTrue(torch.all(counts == route.shape[-1]))
        self.assertEqual(metadata.sparse_q_tiles, 0)

    def test_summary_route_matches_full_qk_route(self):
        torch.manual_seed(31)
        packed = layout()
        router = SparseTileRouter(q_tile=64, kv_tile=64)
        q = torch.randn(1, 2, packed.seq_len, 16)
        k = torch.randn_like(q)

        expected = build_full_absolute_route(router, q, k, packed, 0.3)
        actual = build_full_absolute_route_from_summaries(
            router,
            router._mean_pool(q, router.q_tile),
            router._mean_pool(k, router.kv_tile),
            packed,
            0.3,
        )

        self.assertEqual(actual[2], expected[2])
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_executor_preserves_strided_hnd_inputs_and_writes_nhd(self):
        sequence = 129
        storages = [
            torch.empty(1, sequence, 56, 128, dtype=torch.bfloat16)
            for _ in range(3)
        ]
        q, k, v = (tensor.permute(0, 2, 1, 3) for tensor in storages)
        route = torch.zeros(1, 56, 3, 3, dtype=torch.int32)
        counts = torch.ones(1, 56, 3, dtype=torch.int32)
        launches = []

        def launcher(*args, **kwargs):
            launches.append((args, kwargs))
            args[3].fill_(2)

        executor = FrostBF16Executor(
            launcher=launcher,
            stream_getter=lambda _device: 77,
            allow_cpu_for_tests=True,
        )
        prepared = executor.prepare(
            q, k, v, route, counts, layer_index=4, metadata={'density': 0.3}
        )
        output = executor.execute(prepared)

        self.assertEqual(tuple(output.shape), (1, 56, sequence, 128))
        self.assertEqual(tuple(prepared.output_storage.shape), (1, sequence, 56, 128))
        self.assertEqual(output.stride(), (sequence * 56 * 128, 128, 56 * 128, 1))
        self.assertTrue(torch.all(output == 2))
        self.assertEqual(launches[0][1]['stream'], 77)
        self.assertAlmostEqual(
            launches[0][1]['scale_log2'],
            math.log2(math.e) / math.sqrt(128),
        )

    def test_executor_rejects_non_bf16(self):
        executor = FrostBF16Executor(allow_cpu_for_tests=True)
        q = torch.empty(1, 56, 128, 128)
        route = torch.zeros(1, 56, 1, 2, dtype=torch.int32)
        counts = torch.ones(1, 56, 1, dtype=torch.int32)
        with self.assertRaisesRegex(FrostBF16Error, 'requires BF16'):
            executor.prepare(
                q, q, q, route, counts, layer_index=0, metadata={}
            )

    def test_executor_reports_wrong_head_count(self):
        executor = FrostBF16Executor(allow_cpu_for_tests=True)
        storage = torch.empty(1, 65, 42, 128, dtype=torch.bfloat16)
        q = storage.permute(0, 2, 1, 3)
        route = torch.zeros(1, 42, 2, 2, dtype=torch.int32)
        counts = torch.ones(1, 42, 2, dtype=torch.int32)
        with self.assertRaisesRegex(
            FrostBF16Error,
            r'requires \[1,56,S,128\].*got \(1, 42, 65, 128\)',
        ):
            executor.prepare(
                q, q, q, route, counts, layer_index=0, metadata={}
            )

    def test_executor_accepts_bounded_q_with_global_kv(self):
        q_sequence = 65
        kv_sequence = 129
        q = torch.empty(
            1, q_sequence, 56, 128, dtype=torch.bfloat16
        ).permute(0, 2, 1, 3)
        k, v = (
            torch.empty(
                1, kv_sequence, 56, 128, dtype=torch.bfloat16
            ).permute(0, 2, 1, 3)
            for _ in range(2)
        )
        route = torch.zeros(1, 56, 2, 3, dtype=torch.int32)
        counts = torch.ones(1, 56, 2, dtype=torch.int32)
        executor = FrostBF16Executor(
            launcher=lambda *_args, **_kwargs: None,
            allow_cpu_for_tests=True,
        )

        prepared = executor.prepare(
            q, k, v, route, counts, layer_index=0, metadata={}
        )

        self.assertEqual(tuple(prepared.output.shape), (1, 56, q_sequence, 128))
        self.assertEqual(int(prepared.k.shape[-2]), kv_sequence)

    def test_executor_rejects_contiguous_hnd_storage(self):
        executor = FrostBF16Executor(allow_cpu_for_tests=True)
        q = torch.empty(1, 56, 129, 128, dtype=torch.bfloat16)
        route = torch.zeros(1, 56, 3, 3, dtype=torch.int32)
        counts = torch.ones(1, 56, 3, dtype=torch.int32)
        with self.assertRaisesRegex(FrostBF16Error, 'sequence-major storage'):
            executor.prepare(
                q, q, q, route, counts, layer_index=0, metadata={}
            )

    def test_driver_launch_uses_the_emitted_64x64_abi(self):
        decoded = {}

        class Driver:
            def cuLaunchKernel(self, *args):
                decoded['launch'] = args[:10]
                params = args[9]
                types = [ctypes.c_uint64] * 13
                types += [ctypes.c_uint32] * 3
                types += [ctypes.c_float]
                types += [ctypes.c_uint32] * 4
                types += [ctypes.c_float]
                decoded['params'] = [
                    ctypes.cast(params[index], ctypes.POINTER(kind)).contents.value
                    for index, kind in enumerate(types)
                ]
                return 0

        sequence_q = 65
        sequence_kv = 129
        q = torch.empty(
            1, sequence_q, 56, 128, dtype=torch.bfloat16
        ).permute(0, 2, 1, 3)
        k, v = (
            torch.empty(
                1, sequence_kv, 56, 128, dtype=torch.bfloat16
            ).permute(0, 2, 1, 3)
            for _ in range(2)
        )
        output = torch.empty(1, sequence_q, 56, 128, dtype=torch.bfloat16)
        route = torch.zeros(1, 56, 2, 3, dtype=torch.int32)
        counts = torch.ones(1, 56, 2, dtype=torch.int32)
        driver = Driver()

        with mock.patch.object(
            frost_loader, 'load_driver', return_value=driver
        ), mock.patch.object(
            frost_loader,
            '_load_function',
            return_value=(ctypes.c_void_p(1), ctypes.c_void_p(2)),
        ):
            frost_loader.launch(
                q, k, v, output, route, counts, scale_log2=0.125, stream=99
            )

        launch = decoded['launch']
        self.assertEqual(launch[1:7], (2, 56, 1, frost_loader.THREADS, 1, 1))
        self.assertEqual(launch[7], 64 * 1024)
        self.assertEqual(launch[8].value, 99)
        params = decoded['params']
        self.assertEqual(params[6:13], [0] * 7)
        self.assertEqual(params[13:16], [2, 3, 3])
        self.assertAlmostEqual(params[16], 0.125)
        self.assertEqual(params[17:21], [sequence_q, sequence_kv, 128, 0])
        self.assertAlmostEqual(params[21], math.sqrt(128), places=6)


if __name__ == '__main__':
    unittest.main()
