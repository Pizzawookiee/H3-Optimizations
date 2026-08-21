'''CPU tests for fixed-density Sparse Sage routing.'''

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))

from h3_optimizations.attention.sparse.router import SparseTileRouter  # noqa: E402
from h3_optimizations.attention.sparse.config import (  # noqa: E402
    HybridSparseConfig,
    resolve_video_budget,
)


def layout(sequence=384, video_start=128):
    return SimpleNamespace(
        seq_len=sequence,
        video_range=(video_start, sequence),
        segments=[
            (0, video_start - 32, 'text'),
            (video_start - 32, video_start, 'audio'),
            (video_start, sequence, 'video'),
        ],
        video_shape=(1, 1, sequence - video_start),
        audio_t=16,
    )


def routed_inputs():
    q = torch.zeros((1, 2, 384, 2), dtype=torch.float32)
    k = torch.zeros_like(q)
    q[0, 0, 128:, 0] = 1
    q[0, 1, 128:, 1] = 1
    for index, value in enumerate(((4, 0), (3, 0), (2, 0), (1, 0)), start=2):
        k[0, 0, index * 64:(index + 1) * 64] = torch.tensor(value)
    for index, value in enumerate(((0, 1), (0, 2), (0, 3), (0, 4)), start=2):
        k[0, 1, index * 64:(index + 1) * 64] = torch.tensor(value)
    return q, k


def decode(lut, valid):
    mask = torch.zeros(lut.shape, dtype=torch.bool)
    for index in range(valid.shape[-1]):
        count = int(valid[..., index].max().item())
        if count:
            delta = lut[..., index, :count]
            mask[..., index, :] = torch.nn.functional.one_hot(
                torch.cumsum(delta, dim=-1).long(),
                num_classes=lut.shape[-1],
            ).any(dim=-2)
    return mask


class RouterTests(unittest.TestCase):
    def test_optional_early_late_budget_is_bounded(self):
        config = HybridSparseConfig(
            video_budget=0.5,
            denser_early_late_steps=True,
        )
        self.assertEqual(resolve_video_budget(config, 0, 10), 0.8)
        self.assertEqual(resolve_video_budget(config, 1, 10), 0.8)
        self.assertEqual(resolve_video_budget(config, 2, 10), 0.5)
        self.assertEqual(resolve_video_budget(config, 7, 10), 0.5)
        self.assertEqual(resolve_video_budget(config, 8, 10), 0.8)
        self.assertEqual(resolve_video_budget(config, 9, 10), 0.8)
        self.assertEqual(resolve_video_budget(config, -1, 10), 0.5)
        capped = HybridSparseConfig(
            video_budget=0.85,
            denser_early_late_steps=True,
        )
        self.assertEqual(resolve_video_budget(capped, 0, 10), 1.0)

    def test_per_head_top_k_and_dense_context(self):
        q, k = routed_inputs()
        lut, valid, metadata = SparseTileRouter().build_lut(
            q,
            k,
            layout(),
            0.5,
        )
        mask = decode(lut, valid)
        self.assertEqual(mask.shape, (1, 2, 3, 6))
        self.assertTrue(mask[..., :2].all())
        self.assertTrue(mask[:, :, 0].all())
        self.assertEqual(
            set(torch.where(mask[0, 0, 1, 2:])[0].tolist()),
            {0, 1},
        )
        self.assertEqual(
            set(torch.where(mask[0, 1, 1, 2:])[0].tolist()),
            {2, 3},
        )
        self.assertEqual(metadata.retained_video_kv_tiles, 2)
        self.assertEqual(metadata.actual_video_tile_density, 0.5)

    def test_mixed_boundaries_and_partial_tiles_stay_safe(self):
        q = torch.randn((1, 1, 350, 4))
        lut, valid, metadata = SparseTileRouter().build_lut(
            q,
            q,
            layout(sequence=350, video_start=96),
            0.5,
        )
        mask = decode(lut, valid)
        self.assertTrue(mask[:, :, 0].all())
        self.assertTrue(mask[..., 1].all())
        self.assertEqual((metadata.q_tiles, metadata.kv_tiles), (3, 6))
        self.assertEqual(metadata.pure_video_q_tiles, 2)
        self.assertEqual(metadata.pure_video_kv_tiles, 4)

    def test_full_budget_skips_similarity_scoring(self):
        class NoPoolingRouter(SparseTileRouter):
            @staticmethod
            def _mean_pool(_x, _block):
                raise AssertionError('full budget must not pool Q or K')

        q = torch.randn((1, 2, 384, 8))
        lut, valid, metadata = NoPoolingRouter().build_lut(
            q,
            q,
            layout(),
            1.0,
        )
        self.assertTrue(decode(lut, valid).all())
        self.assertEqual(metadata.full_mask_density, 1.0)
        self.assertEqual(metadata.sparse_q_tiles, 0)


if __name__ == '__main__':
    unittest.main()
