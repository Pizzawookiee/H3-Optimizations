'''FastVideo VSA-H3 tile geometry for MiniMax H3 packed sequences.

A VSA-trained checkpoint learned its sparsity pattern and coarse branch over a
fixed tiling, so this must reproduce FastVideo's
``video_sparse_attn_h3._h3_tile_geometry`` rather than the pack's own cube
orders:

- every non-video segment before the target video is tiled on its own into
  64-row tiles, with a short final tile, so no tile mixes segments;
- the target video is tiled into 4x4x4 (t, h, w) cubes in (t-tile, h-tile,
  w-tile) order; edge cubes are padded, keep their live rows first, and report
  their live count in ``block_len``.

Pad slots never hold a packed row. Consumers zero them, exclude them from
tile means, and mask them as keys.
'''

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

TILE = 64
CUBE = (4, 4, 4)


class VSAGeometryError(ValueError):
    pass


@dataclass(frozen=True)
class VSAGeometry:
    seq_len: int
    num_prefix_tiles: int
    num_video_tiles: int
    # padded slot -> packed row, -1 for a pad slot
    src: torch.Tensor
    # packed row -> padded slot
    inv: torch.Tensor
    # live rows per tile, int32
    block_len: torch.Tensor

    @property
    def num_tiles(self):
        return self.num_prefix_tiles + self.num_video_tiles

    @property
    def padded_len(self):
        return self.num_tiles * TILE


def _prefix_tiles(start, stop, device):
    count = stop - start
    tiles = math.ceil(count / TILE)
    slots = torch.full((tiles * TILE,), -1, dtype=torch.int64, device=device)
    slots[:count] = torch.arange(start, stop, dtype=torch.int64, device=device)
    return slots.view(tiles, TILE)


def _video_tiles(start, grid, device):
    t, h, w = grid
    ct, ch, cw = CUBE
    pt, ph, pw = (math.ceil(size / cube) * cube for size, cube in zip(grid, CUBE))
    padded = torch.full((pt, ph, pw), -1, dtype=torch.int64, device=device)
    padded[:t, :h, :w] = torch.arange(
        start, start + t * h * w, dtype=torch.int64, device=device,
    ).view(t, h, w)
    cubes = (
        padded.view(pt // ct, ct, ph // ch, ch, pw // cw, cw)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(-1, TILE)
    )
    # live rows first within each tile; the stable sort keeps their order
    order = torch.argsort((cubes < 0).to(torch.int8), dim=1, stable=True)
    return torch.gather(cubes, 1, order)


def build_geometry(segments, video_grid, device='cpu'):
    '''Tile a packed H3 sequence.

    ``segments`` is the layout's contiguous ``(start, stop, kind)`` list and
    ``video_grid`` the target video's (t, h, w) token grid.
    '''
    segments = [(int(a), int(b), str(kind)) for a, b, kind in segments]
    if not segments:
        raise VSAGeometryError('VSA needs a packed H3 layout')
    position = 0
    for start, stop, _kind in segments:
        if start != position or stop < start:
            raise VSAGeometryError('H3 layout segments are not contiguous')
        position = stop
    video_start, video_stop, video_kind = segments[-1]
    if video_kind != 'video':
        raise VSAGeometryError(
            'VSA needs the target video as the last packed segment, got %r' % video_kind
        )
    grid = tuple(int(value) for value in video_grid)
    if len(grid) != 3 or math.prod(grid) != video_stop - video_start:
        raise VSAGeometryError(
            'video segment of %d rows does not match the token grid %r'
            % (video_stop - video_start, grid)
        )

    tiles = [
        _prefix_tiles(start, stop, device)
        for start, stop, _kind in segments[:-1]
        if stop > start
    ]
    num_prefix_tiles = sum(int(block.shape[0]) for block in tiles)
    video = _video_tiles(video_start, grid, device)
    tiles.append(video)
    tiles = torch.cat(tiles)

    src = tiles.reshape(-1)
    live = src >= 0
    seq_len = video_stop
    inv = torch.empty(seq_len, dtype=torch.int64, device=device)
    inv[src[live]] = torch.nonzero(live).flatten()
    block_len = (tiles >= 0).sum(dim=1).to(torch.int32)
    if int(block_len.min()) < 1 or int(block_len.sum()) != seq_len:
        raise VSAGeometryError('VSA tile geometry lost or duplicated packed rows')
    return VSAGeometry(
        seq_len=seq_len,
        num_prefix_tiles=num_prefix_tiles,
        num_video_tiles=int(video.shape[0]),
        src=src,
        inv=inv,
        block_len=block_len,
    )


def compute_topk(sparsity, num_blocks):
    '''FastVideo's kept-tile count: ceil((1 - sparsity) * n), clamped to [1, n].'''
    return max(1, min(math.ceil((1 - sparsity) * num_blocks), num_blocks))


def sparsity_from_keep_percent(keep_percent):
    '''The trained configs state sparsity (0.8, 0.9); keep the same float so
    compute_topk rounds exactly as FastVideo does for those values.'''
    return round(1.0 - float(keep_percent) / 100.0, 6)
