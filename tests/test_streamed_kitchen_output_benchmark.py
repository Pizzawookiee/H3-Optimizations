'''CPU contracts for the isolated streamed Kitchen output experiment.'''

from pathlib import Path
import sys
import unittest

import torch


PACK = Path(__file__).resolve().parents[1]
BENCHMARKS = PACK / 'benchmarks'
COMFY_ROOT = PACK.parents[1]
for root in (str(PACK), str(BENCHMARKS), str(COMFY_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

from h3_optimizations.native.int8_attention import (  # noqa: E402
    BlockSparseRoute,
    PrequantizedInt8Attention,
)
from streamed_kitchen_output import (  # noqa: E402
    PreparedStreamedOutput,
    StreamedOutputBackend,
)
from h3_optimizations.attention.sparse.kitchen_sparse import (  # noqa: E402
    PreparedSparseKitchen,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class _Kitchen:
    def __init__(self):
        self.calls = []

    def block_sparse_int8_attention_from_prequantized(self, carrier, route, output_layout):
        self.calls.append((carrier.q.shape[-2], route.indices.shape[-2], output_layout))
        return carrier.q.to(torch.float32)


class _Backend:
    requires_runtime_context = True
    approximate = True
    installation_signature = ('fake',)

    def __init__(self, kitchen, q_tile=128):
        self.executor = type('E', (), {'q_tile': q_tile, 'kitchen': kitchen})()


class _Module:
    heads = 2
    head_dim = 128
    out_proj = staticmethod(lambda value: value)


class StreamedKitchenOutputBenchmarkTests(unittest.TestCase):
    def test_ragged_query_chunks_preserve_output_and_route_slices(self):
        q = torch.arange(2 * 257 * 128).to(torch.int8).reshape(1, 2, 257, 128)
        carrier = PrequantizedInt8Attention(
            q=q,
            k=torch.empty_like(q),
            v=torch.empty(256, 384, dtype=torch.int8),
            q_scale=torch.empty(1, 2, 96),
            k_scale=torch.empty(1),
            v_scale=torch.empty(1),
            original_head_dim=128,
            input_dtype=torch.float32,
            attention_scale=1.0,
            cta_k=128,
        )
        route = BlockSparseRoute(
            indices=torch.zeros(1, 2, 3, 1, dtype=torch.int32),
            counts=torch.ones(1, 2, 3, dtype=torch.int32),
            q_tile=128,
            kv_tile=128,
            encoding='delta',
        )
        base = PreparedSparseKitchen(carrier, route, 128, 0, {})
        wrapped = PreparedStreamedOutput(base, torch.empty(257, 256))
        kitchen = _Kitchen()
        backend = StreamedOutputBackend(_Backend(kitchen), 256)

        output = backend.execute_projected(_Module(), wrapped)

        expected = q.transpose(1, 2).reshape(1, 257, 256).squeeze(0).float()
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(kitchen.calls, [(256, 2, 'nhd'), (1, 1, 'nhd')])
        self.assertIsNone(base.quantized)
        self.assertIsNone(base.route)

    def test_query_chunks_must_follow_native_tile_geometry(self):
        with self.assertRaises(ValueError):
            StreamedOutputBackend(_Backend(_Kitchen()), 192)

    def test_64q_route_slices_twice_as_many_route_rows_as_carrier_scales(self):
        q = torch.arange(2 * 256 * 128).to(torch.int8).reshape(1, 2, 256, 128)
        carrier = PrequantizedInt8Attention(
            q=q,
            k=torch.empty_like(q),
            v=torch.empty(256, 384, dtype=torch.int8),
            q_scale=torch.empty(1, 2, 64),
            k_scale=torch.empty(1),
            v_scale=torch.empty(1),
            original_head_dim=128,
            input_dtype=torch.float32,
            attention_scale=1.0,
            cta_k=64,
        )
        route = BlockSparseRoute(
            indices=torch.zeros(1, 2, 4, 1, dtype=torch.int32),
            counts=torch.ones(1, 2, 4, dtype=torch.int32),
            q_tile=64,
            kv_tile=64,
            encoding='delta',
        )
        base = PreparedSparseKitchen(carrier, route, 64, 0, {})
        wrapped = PreparedStreamedOutput(base, torch.empty(256, 256))
        kitchen = _Kitchen()
        backend = StreamedOutputBackend(_Backend(kitchen, q_tile=64), 128)

        output = backend.execute_projected(_Module(), wrapped)

        expected = q.transpose(1, 2).reshape(1, 256, 256).squeeze(0).float()
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(kitchen.calls, [(128, 2, 'nhd'), (128, 2, 'nhd')])


if __name__ == '__main__':
    unittest.main()
