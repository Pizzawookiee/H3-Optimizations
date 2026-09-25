"""CPU contracts for trained VSA attention (FastH3).

The references here restate FastVideo's VSA-H3 semantics independently of the
production code: tiles built by explicit loops, the block mask built the way
``video_sparse_attn_h3._build_block_mask`` does, and dense masked attention
over the padded tile buffer.
"""

import math
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
from torch import nn  # noqa: E402

from h3_optimizations.vsa.attention import vsa_attention  # noqa: E402
from h3_optimizations.vsa.node import (  # noqa: E402
    BACKEND_AUTO_OPTION,
    BACKEND_INT8_OPTION,
    MEMORY_LOWER_VRAM,
    VSAState,
    parse_block_list,
)
from h3_optimizations.vsa.streamed import _slab_route  # noqa: E402
from h3_optimizations.vsa.tiling import (  # noqa: E402
    TILE,
    VSAGeometryError,
    build_geometry,
    compute_topk,
    sparsity_from_keep_percent,
)

HEAD_DIM = 128


def reference_tiles(segments, grid):
    """Explicit-loop restatement: [(live packed rows, ...)] per tile."""
    tiles = []
    for start, stop, kind in segments[:-1]:
        for first in range(start, stop, TILE):
            tiles.append(tuple(range(first, min(first + TILE, stop))))
    video_start = segments[-1][0]
    t, h, w = grid
    for t0 in range(0, t, 4):
        for h0 in range(0, h, 4):
            for w0 in range(0, w, 4):
                rows = []
                for ti in range(t0, min(t0 + 4, t)):
                    for hi in range(h0, min(h0 + 4, h)):
                        for wi in range(w0, min(w0 + 4, w)):
                            rows.append(video_start + (ti * h + hi) * w + wi)
                tiles.append(tuple(rows))
    return tiles


class TinyAttention(nn.Module):
    def __init__(self, hidden=64, heads=2):
        super().__init__()
        self.heads = heads
        self.head_dim = HEAD_DIM
        inner = heads * HEAD_DIM
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.q_norm = nn.RMSNorm(HEAD_DIM, eps=1e-5)
        self.k_norm = nn.RMSNorm(HEAD_DIM, eps=1e-5)
        self.to_gate_compress = nn.Linear(hidden, inner, bias=False)
        self.out_proj = nn.Linear(inner, hidden, bias=False)


def reference_vsa(attn, x, segments, grid, sparsity, dense=False):
    heads, dim = attn.heads, attn.head_dim
    tiles = reference_tiles(segments, grid)
    n_prefix = sum(math.ceil((b - a) / TILE) for a, b, _ in segments[:-1])
    n_video = len(tiles) - n_prefix
    qkv = attn.qkv_proj(x)
    q, k, v = (part.reshape(-1, heads, dim) for part in qkv.split(heads * dim, dim=-1))
    q, k = attn.q_norm(q), attn.k_norm(k)

    padded = len(tiles) * TILE
    slot_of = {}
    bufs = [torch.zeros(heads, padded, dim) for _ in range(3)]
    live = torch.zeros(padded, dtype=torch.bool)
    sizes = torch.tensor([len(rows) for rows in tiles], dtype=torch.float32)
    for index, rows in enumerate(tiles):
        for offset, row in enumerate(rows):
            slot = index * TILE + offset
            slot_of[row] = slot
            live[slot] = True
            for buf, src in zip(bufs, (q, k, v)):
                buf[:, slot] = src[row]
    qb, kb, vb = bufs

    def pool(buf):
        return buf.view(heads, len(tiles), TILE, dim).sum(2) / sizes.view(1, -1, 1)

    scores = pool(qb) @ pool(kb).transpose(-1, -2) / math.sqrt(dim)
    # video_sparse_attn_h3._build_block_mask, exempt mode
    keep = compute_topk(0.0 if dense else sparsity, n_video)
    if keep == n_video:
        mask = torch.ones_like(scores, dtype=torch.bool)
    else:
        mask = torch.zeros_like(scores, dtype=torch.bool)
        idx = scores[..., n_prefix:].topk(keep, dim=-1).indices + n_prefix
        mask.scatter_(-1, idx, True)
        mask[..., :n_prefix] = True
    mask[:, :n_prefix, :] = True

    row_mask = mask.repeat_interleave(TILE, 1).repeat_interleave(TILE, 2) & live[None, None, :]
    logits = (qb @ kb.transpose(-1, -2) / math.sqrt(dim)).masked_fill(~row_mask, float("-inf"))
    fine = torch.softmax(logits, -1) @ vb
    coarse = torch.softmax(scores, -1) @ pool(vb)
    gate = attn.to_gate_compress(x).reshape(-1, heads, dim)
    out = torch.empty(x.shape[0], heads, dim)
    for row, slot in slot_of.items():
        out[row] = fine[:, slot] + coarse[:, slot // TILE] * gate[row]
    return attn.out_proj(out.reshape(x.shape[0], heads * dim))


SEGMENTS_GRID = (5, 6, 7)


def segments_for(grid, text=5, audio=70):
    video = math.prod(grid)
    return [
        (0, text, "text"),
        (text, text + audio, "audio"),
        (text + audio, text + audio + video, "video"),
    ]


class VSAGeometryTest(unittest.TestCase):
    def test_matches_explicit_loop_tiles(self):
        for grid in ((5, 6, 7), (8, 4, 4), (1, 3, 9), (72, 24, 43)):
            segments = segments_for(grid, text=29, audio=810)
            geometry = build_geometry(segments, grid)
            expected = reference_tiles(segments, grid)
            self.assertEqual(geometry.num_tiles, len(expected))
            self.assertEqual(geometry.num_prefix_tiles, 1 + math.ceil(810 / TILE))
            src = geometry.src.view(-1, TILE)
            for index, rows in enumerate(expected):
                live = src[index][src[index] >= 0].tolist()
                self.assertEqual(sorted(live), sorted(rows))
                # live rows first, pads after
                self.assertTrue(bool((src[index][: len(rows)] >= 0).all()))
                self.assertEqual(int(geometry.block_len[index]), len(rows))
            self.assertTrue(torch.equal(geometry.src[geometry.inv], torch.arange(geometry.seq_len)))

    def test_prefix_tiles_never_mix_segments(self):
        segments = segments_for((4, 4, 4), text=70, audio=70)
        geometry = build_geometry(segments, (4, 4, 4))
        src = geometry.src.view(-1, TILE)
        for index in range(geometry.num_prefix_tiles):
            rows = src[index][src[index] >= 0]
            kinds = {("text" if int(r) < 70 else "audio") for r in rows}
            self.assertEqual(len(kinds), 1)

    def test_rejects_non_final_video_and_grid_mismatch(self):
        with self.assertRaises(VSAGeometryError):
            build_geometry([(0, 64, "video"), (64, 70, "audio")], (1, 8, 8))
        with self.assertRaises(VSAGeometryError):
            build_geometry(segments_for((2, 2, 2)), (2, 2, 3))

    def test_topk_rounds_like_fastvideo(self):
        self.assertEqual(sparsity_from_keep_percent(20), 0.8)
        self.assertEqual(sparsity_from_keep_percent(10), 0.9)
        self.assertEqual(compute_topk(0.8, 1188), math.ceil((1 - 0.8) * 1188))
        self.assertEqual(compute_topk(0.9, 8), 1)
        self.assertEqual(compute_topk(0.0, 8), 8)
        self.assertEqual(compute_topk(0.999, 3), 1)


class VSAAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.attn = TinyAttention().eval()
        self.segments = segments_for(SEGMENTS_GRID)
        self.geometry = build_geometry(self.segments, SEGMENTS_GRID)
        self.x = torch.randn(self.geometry.seq_len, 64)

    def _compare(self, sparsity, dense):
        with torch.no_grad():
            actual = vsa_attention(
                self.attn, self.x, None, self.geometry, sparsity, dense=dense, eager=True,
            )
            expected = reference_vsa(
                self.attn, self.x, self.segments, SEGMENTS_GRID, sparsity, dense=dense,
            )
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_sparse_matches_fastvideo_semantics(self):
        self._compare(0.8, dense=False)
        self._compare(0.9, dense=False)

    def test_dense_keeps_every_tile_and_the_coarse_branch(self):
        self._compare(0.8, dense=True)

    def test_sparse_differs_from_dense(self):
        with torch.no_grad():
            sparse = vsa_attention(self.attn, self.x, None, self.geometry, 0.9, eager=True)
            dense = vsa_attention(self.attn, self.x, None, self.geometry, 0.9, dense=True, eager=True)
        self.assertGreater(float((sparse - dense).abs().max()), 1e-4)

    def test_requires_gate(self):
        self.attn.to_gate_compress = None
        with self.assertRaises(RuntimeError):
            vsa_attention(self.attn, self.x, None, self.geometry, 0.8, eager=True)


class VSANodeHelpersTest(unittest.TestCase):
    def test_parse_block_list(self):
        self.assertEqual(parse_block_list("0, 1, 47-49"), {0, 1, 47, 48, 49})
        self.assertEqual(parse_block_list(""), set())

    def test_auto_backend_uses_int8_only_after_parity(self):
        state = VSAState(0.8, 0, set(), False, backend=BACKEND_AUTO_OPTION)
        with mock.patch(
            "h3_optimizations.vsa.node.int8_parity",
            return_value=(True, "rel_l2 0.01"),
        ):
            self.assertEqual(state.backend(torch.device("cuda", 0)), "int8")

        state = VSAState(0.8, 0, set(), False, backend=BACKEND_AUTO_OPTION)
        with (
            mock.patch(
                "h3_optimizations.vsa.node.int8_parity",
                return_value=(False, "broken kernel"),
            ),
            mock.patch("h3_optimizations.vsa.node.logging.warning") as warning,
        ):
            self.assertEqual(state.backend(torch.device("cuda", 0)), "bf16")
            self.assertEqual(state.backend(torch.device("cuda", 0)), "bf16")
            warning.assert_called_once()

    def test_forced_int8_fails_closed_and_lower_vram_is_explicit(self):
        state = VSAState(
            0.8, 0, set(), False,
            backend=BACKEND_INT8_OPTION,
            memory_mode=MEMORY_LOWER_VRAM,
        )
        self.assertTrue(state.lower_vram)
        with mock.patch(
            "h3_optimizations.vsa.node.int8_parity",
            return_value=(False, "missing tile traversal"),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing tile traversal"):
                state.backend(torch.device("cuda", 0))


class VSAStreamedRouteTest(unittest.TestCase):
    def setUp(self):
        self.geometry = build_geometry(segments_for((8, 4, 4), text=29, audio=70), (8, 4, 4))
        self.heads = 2
        self.tiles = self.geometry.num_tiles

    def _absolute(self, indices, counts):
        absolute = indices.cumsum(dim=-1)
        live = torch.arange(indices.shape[-1]).view(1, 1, 1, -1) < counts.unsqueeze(-1)
        return torch.where(live, absolute, -1)

    def test_prefix_queries_are_dense_and_video_queries_keep_prefix_plus_topk(self):
        scores = torch.arange(
            self.heads * self.tiles * self.tiles,
            dtype=torch.float32,
        ).reshape(self.heads, self.tiles, self.tiles)
        indices, counts = _slab_route(scores, self.geometry, 0.5, 0, False)
        absolute = self._absolute(indices, counts)[0]
        prefix = self.geometry.num_prefix_tiles
        keep = compute_topk(0.5, self.geometry.num_video_tiles)

        self.assertTrue(torch.equal(counts[0, :, :prefix], torch.full((self.heads, prefix), self.tiles)))
        self.assertTrue(torch.equal(
            absolute[:, :prefix],
            torch.arange(self.tiles).view(1, 1, -1).expand(self.heads, prefix, -1),
        ))
        self.assertTrue(torch.equal(counts[0, :, prefix:], torch.full(
            (self.heads, self.geometry.num_video_tiles), prefix + keep,
        )))
        self.assertTrue(bool((absolute[:, prefix:, :prefix] == torch.arange(prefix)).all()))
        self.assertTrue(bool((absolute[:, prefix:, prefix:prefix + keep] >= prefix).all()))

    def test_dense_route_walks_every_tile(self):
        scores = torch.zeros(self.heads, self.tiles, self.tiles)
        indices, counts = _slab_route(scores, self.geometry, 0.8, 0, True)
        absolute = self._absolute(indices, counts)[0]
        self.assertTrue(torch.equal(
            absolute,
            torch.arange(self.tiles).view(1, 1, -1).expand(self.heads, self.tiles, -1),
        ))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
