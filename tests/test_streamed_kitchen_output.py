'''Production contracts for query-sliced Kitchen output projection.'''

import os
from pathlib import Path
import sys
import unittest

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.attention.sparse.kitchen_sparse import (
    PreparedSparseKitchen,
    SparseKitchenBackend,
)
from h3_optimizations.kitchen_qkv import ChunkedKitchenQKVProjector
from h3_optimizations.native.int8_attention import (
    BlockSparseRoute,
    PrequantizedInt8Attention,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class _Kitchen:
    __version__ = 'test'

    def __init__(self):
        self.calls = []

    def block_sparse_int8_attention_from_prequantized(
        self, carrier, route, output_layout
    ):
        self.calls.append(
            (carrier.q.shape[-2], route.indices.shape[-2], output_layout)
        )
        return carrier.q.to(torch.float32)


class _Executor:
    output_layout = 'nhd'

    def __init__(self, kitchen, q_tile=64, kv_tile=64):
        self.kitchen = kitchen
        self.q_tile = q_tile
        self.kv_tile = kv_tile


class _Router:
    score_chunk_tiles = None

    def __init__(self, q_tile=64, kv_tile=64):
        self.q_tile = q_tile
        self.kv_tile = kv_tile


class _Module:
    heads = 2
    head_dim = 128
    out_proj = staticmethod(lambda value: value)


class StreamedKitchenOutputTests(unittest.TestCase):
    def _backend(self, kitchen, query_chunk_rows=128):
        projector = ChunkedKitchenQKVProjector(stream_output=True)
        return SparseKitchenBackend(
            kitchen=kitchen,
            executor=_Executor(kitchen),
            router=_Router(),
            projector=projector,
            stream_output=True,
            query_chunk_rows=query_chunk_rows,
        )

    def test_64q_routes_and_128q_scales_are_sliced_independently(self):
        q = torch.arange(2 * 257 * 128).to(torch.int8).reshape(1, 2, 257, 128)
        carrier = PrequantizedInt8Attention(
            q=q,
            k=torch.empty_like(q),
            v=torch.empty(2 * 128, 257, dtype=torch.int8),
            q_scale=torch.empty(1, 2, 96),
            k_scale=torch.empty(1),
            v_scale=torch.empty(1),
            original_head_dim=128,
            input_dtype=torch.float32,
            attention_scale=1.0,
            cta_k=64,
        )
        route = BlockSparseRoute(
            indices=torch.zeros(1, 2, 5, 1, dtype=torch.int32),
            counts=torch.ones(1, 2, 5, dtype=torch.int32),
            q_tile=64,
            kv_tile=64,
            encoding='delta',
        )
        output = torch.empty(257, 256)
        prepared = PreparedSparseKitchen(
            carrier, route, 128, 0, {}, output_buffer=output
        )
        kitchen = _Kitchen()

        actual = self._backend(kitchen).execute_projected(_Module(), prepared)

        expected = q.transpose(1, 2).reshape(1, 257, 256).squeeze(0).float()
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            kitchen.calls,
            [(128, 2, 'nhd'), (128, 2, 'nhd'), (1, 1, 'nhd')],
        )
        self.assertIsNone(prepared.quantized)
        self.assertIsNone(prepared.route)
        self.assertIsNone(prepared.output_buffer)

    def test_streaming_requires_a_capturing_projector(self):
        kitchen = _Kitchen()
        with self.assertRaisesRegex(RuntimeError, 'output-capturing projector'):
            SparseKitchenBackend(
                kitchen=kitchen,
                executor=_Executor(kitchen),
                router=_Router(),
                projector=ChunkedKitchenQKVProjector(),
                stream_output=True,
            )

    def test_low_level_defaults_remain_opt_in(self):
        self.assertFalse(ChunkedKitchenQKVProjector().stream_output)
        kitchen = _Kitchen()
        backend = SparseKitchenBackend(
            kitchen=kitchen,
            executor=_Executor(kitchen),
            router=_Router(),
        )
        self.assertFalse(backend.stream_output)


if __name__ == '__main__':
    unittest.main()
