'''Block-sparse BF16 attention over padded VSA tiles.

This is the pack's BF16 Triton sparse loop with one addition: every KV tile
carries a live length, and keys past it are masked out of the softmax. VSA
edge cubes are padded, and a zero pad key would otherwise still score 0 and
take softmax mass. BF16 matches the precision FastH3 was trained against.

Q, K, V and O are ``[heads, padded_len, 128]`` with ``padded_len`` a multiple
of 64. Query tiles below ``dense_q_tiles`` attend every KV tile; the rest
attend the tiles listed in ``lut`` (``[heads, sparse_q_tiles, selected]``).
'''

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    triton = None
    tl = None
    TRITON_AVAILABLE = False

TILE = 64
HEAD_DIM = 128


if TRITON_AVAILABLE:
    @triton.jit
    def _vsa_sparse_kernel(
        Q,
        K,
        V,
        LUT,
        BLOCK_LEN,
        O,
        Q_BLOCK_START,
        Q_BLOCK_COUNT,
        stride_h: tl.constexpr,
        stride_n: tl.constexpr,
        N_SELECTED: tl.constexpr,
        USE_ROUTE: tl.constexpr,
        softmax_scale: tl.constexpr,
        TILE_: tl.constexpr,
        D: tl.constexpr,
    ):
        local_q_block = tl.program_id(0)
        q_block = Q_BLOCK_START + local_q_block
        head = tl.program_id(1)

        q_rows = q_block * TILE_ + tl.arange(0, TILE_)
        kv_rows = tl.arange(0, TILE_)
        dims = tl.arange(0, D)
        q = tl.load(Q + head * stride_h + q_rows[:, None] * stride_n + dims[None, :])
        row_max = tl.full((TILE_,), -float('inf'), dtype=tl.float32)
        row_sum = tl.zeros((TILE_,), dtype=tl.float32)
        output = tl.zeros((TILE_, D), dtype=tl.float32)

        for route_position in tl.range(0, N_SELECTED):
            if USE_ROUTE:
                route_offset = (
                    (head * Q_BLOCK_COUNT + local_q_block) * N_SELECTED
                    + route_position
                )
                key_block = tl.load(LUT + route_offset)
            else:
                key_block = route_position
            live = tl.load(BLOCK_LEN + key_block)
            k_mask = kv_rows < live
            k_rows = key_block * TILE_ + kv_rows
            k = tl.load(
                K + head * stride_h + k_rows[None, :] * stride_n + dims[:, None],
                mask=k_mask[None, :],
                other=0.0,
            )
            logits = tl.dot(q, k) * (softmax_scale * 1.4426950408889634)
            logits = tl.where(k_mask[None, :], logits, -float('inf'))
            v = tl.load(
                V + head * stride_h + k_rows[:, None] * stride_n + dims[None, :],
                mask=k_mask[:, None],
                other=0.0,
            )

            tile_max = tl.max(logits, axis=1)
            new_row_max = tl.maximum(row_max, tile_max)
            probability = tl.math.exp2(logits - new_row_max[:, None])
            tile_sum = tl.sum(probability, axis=1)
            old_scale = tl.math.exp2(row_max - new_row_max)

            output = output * old_scale[:, None]
            output += tl.dot(probability.to(v.dtype), v)
            row_sum = row_sum * old_scale + tile_sum
            row_max = new_row_max

        tl.store(
            O + head * stride_h + q_rows[:, None] * stride_n + dims[None, :],
            (output / row_sum[:, None]).to(O.type.element_ty),
        )


def _check(q, k, v, block_len):
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError('VSA attention needs matching Q/K/V shapes')
    heads, padded, dim = q.shape
    if dim != HEAD_DIM or padded % TILE:
        raise ValueError('VSA attention needs [heads, 64*tiles, 128] tensors')
    if block_len.numel() != padded // TILE:
        raise ValueError('VSA block_len does not match the tile count')
    for tensor in (q, k, v):
        if tensor.stride(-1) != 1 or tensor.stride() != q.stride():
            raise ValueError('VSA attention needs identically strided, row-contiguous Q/K/V')
    return heads, padded // TILE


def sparse_attention(q, k, v, block_len, lut, dense_q_tiles):
    '''Triton path. ``lut`` is None when every query tile is dense.'''
    if not TRITON_AVAILABLE:
        raise RuntimeError('H3 VSA attention requires Triton')
    heads, tiles = _check(q, k, v, block_len)
    block_len = block_len.to(device=q.device, dtype=torch.int32).contiguous()
    output = torch.empty_like(q)
    dummy = block_len if lut is None else lut

    def launch(start, count, selected, use_route, route):
        if not count:
            return
        _vsa_sparse_kernel[(count, heads)](
            q, k, v, route, block_len, output,
            start, count,
            stride_h=q.stride(0),
            stride_n=q.stride(1),
            N_SELECTED=selected,
            USE_ROUTE=use_route,
            softmax_scale=HEAD_DIM ** -0.5,
            TILE_=TILE,
            D=HEAD_DIM,
            num_warps=4,
            num_stages=1,
        )

    if lut is None:
        launch(0, tiles, tiles, False, dummy)
        return output
    sparse_q_tiles = tiles - int(dense_q_tiles)
    if tuple(lut.shape[:2]) != (heads, sparse_q_tiles) or lut.dtype != torch.int32:
        raise ValueError('VSA route has the wrong shape or dtype')
    launch(0, int(dense_q_tiles), tiles, False, dummy)
    launch(int(dense_q_tiles), sparse_q_tiles, int(lut.shape[-1]), True, lut.contiguous())
    return output


def sparse_attention_eager(q, k, v, block_len, lut, dense_q_tiles):
    '''fp32 reference with the same contract: exact masked softmax per row.'''
    heads, tiles = _check(q, k, v, block_len)
    block_len = block_len.to(q.device)
    allowed = torch.ones(heads, tiles, tiles, dtype=torch.bool, device=q.device)
    if lut is not None:
        allowed[:, int(dense_q_tiles):] = False
        allowed[:, int(dense_q_tiles):].scatter_(-1, lut.long(), True)
    key_live = (
        torch.arange(TILE, device=q.device)[None, :] < block_len[:, None]
    ).reshape(-1)
    allowed_rows = allowed.repeat_interleave(TILE, dim=1).repeat_interleave(TILE, dim=2)
    allowed_rows &= key_live[None, None, :]
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * HEAD_DIM ** -0.5
    scores = scores.masked_fill(~allowed_rows, float('-inf'))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)
