'''VSA attention for one MiniMax H3 block, following FastVideo's VSA-H3.

Per call, for a VSA-trained checkpoint (one with ``to_gate_compress``):

1. Q/K/V are projected in bounded chunks of packed rows, RMS-normed and
   RoPE'd, and scattered into zeroed padded tile buffers. The full unpadded
   QKV projection is never materialized.
2. fp32 means of post-RoPE Q, K and V over each tile's live rows give the
   tile scores ``q_mean . k_mean / sqrt(d)``.
3. Prefix query tiles (text, audio) are dense. Video query tiles attend every
   prefix tile plus their top-k video tiles, k = ceil((1 - sparsity) * n).
   There is no tail and no forced diagonal.
4. The coarse branch ``softmax(scores) @ v_mean`` over all tiles is added to
   every row, scaled per token by ``to_gate_compress`` of the attention input.

Output rows are assembled and projected in chunks. Dense steps or layers keep
every tile but still add the coarse branch, as FastVideo does.
'''

from __future__ import annotations

import torch

import comfy.model_management
import comfy.quant_ops

from . import kernel
from .tiling import TILE, compute_topk

PRODUCER_ROWS = 4096
BACKEND_BF16 = 'bf16'
BACKEND_INT8 = 'int8'


def _project_into_tiles(attn, x, rope_freqs, geometry, buffers):
    '''Fill the padded Q/K/V buffers ([heads, padded, dim]) from packed rows.'''
    heads, head_dim = attn.heads, attn.head_dim
    inner = heads * head_dim
    q_buf, k_buf, v_buf = buffers
    rot = None
    if rope_freqs is not None:
        rot = rope_freqs.shape[-3] * 2
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
    for start in range(0, x.shape[0], PRODUCER_ROWS):
        stop = min(start + PRODUCER_ROWS, x.shape[0])
        rows = stop - start
        qkv = attn.qkv_proj(x[start:stop])
        q, k, v = qkv.split(inner, dim=-1)
        if rot is not None:
            q = q.view(1, rows, heads, head_dim)
            k = k.view(1, rows, heads, head_dim)
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs[:, start:stop], qw, kw,
                epsilon=attn.q_norm.eps, rot_dim=rot,
            )
            q, k = q[0], k[0]
        else:
            q = attn.q_norm(q.reshape(rows, heads, head_dim))
            k = attn.k_norm(k.reshape(rows, heads, head_dim))
        slots = geometry.inv[start:stop]
        for buffer, values in ((q_buf, q), (k_buf, k), (v_buf, v)):
            buffer.index_copy_(1, slots, values.reshape(rows, heads, head_dim).transpose(0, 1).to(buffer.dtype))
        del qkv, q, k, v


def _tile_means(buffer, block_len):
    heads, padded, dim = buffer.shape
    sums = buffer.view(heads, padded // TILE, TILE, dim).sum(dim=2, dtype=torch.float32)
    return sums / block_len.to(torch.float32).view(1, -1, 1)


def _route(scores, geometry, sparsity):
    '''None when every video tile is kept; else [heads, video_tiles, selected].'''
    prefix = geometry.num_prefix_tiles
    keep = compute_topk(sparsity, geometry.num_video_tiles)
    if keep >= geometry.num_video_tiles:
        return None
    video = scores[:, prefix:, prefix:].topk(keep, dim=-1).indices + prefix
    video = video.sort(dim=-1).values
    heads, rows = video.shape[:2]
    prefix_cols = torch.arange(prefix, device=scores.device).expand(heads, rows, prefix)
    return torch.cat((prefix_cols, video), dim=-1).to(torch.int32).contiguous()


def _kernel_route(route, geometry, heads, device):
    """Route for the INT8 kernel, built directly in its delta encoding.

    Prefix query tiles walk every tile; video query tiles walk the prefix plus
    their selected video tiles. Building the delta form in int32 avoids the
    int64 conversions of a full [heads, tiles, tiles] absolute table.
    """
    from ..native import int8_attention as native

    tiles, prefix = geometry.num_tiles, geometry.num_prefix_tiles
    delta = native.loader.route_encoding() == 'delta'
    walk_all = torch.ones(tiles, dtype=torch.int32, device=device)
    if delta:
        walk_all[0] = 0
    else:
        walk_all = torch.arange(tiles, dtype=torch.int32, device=device)
    indices = torch.zeros((1, heads, tiles, tiles), dtype=torch.int32, device=device)
    counts = torch.full((1, heads, tiles), tiles, dtype=torch.int32, device=device)
    if route is None:
        indices[0] = walk_all
    else:
        indices[0, :, :prefix] = walk_all
        selected = route.shape[-1]
        if delta:
            steps = torch.diff(route, dim=-1, prepend=torch.zeros_like(route[..., :1]))
            indices[0, :, prefix:, :selected] = steps
        else:
            indices[0, :, prefix:, :selected] = route
        counts[0, :, prefix:] = selected
    return native.BlockSparseRoute(
        indices=indices, counts=counts, q_tile=TILE, kv_tile=TILE,
        encoding='delta' if delta else 'absolute',
    )


def _int8_fine(buffers, block_len, route, geometry):
    """The exact block-sparse pass on the vendored Kitchen INT8 kernel.

    ``buffers`` is a list of the padded BF16 Q/K/V; it is emptied once they are
    quantized so the BF16 copies do not live through the kernel. Pad keys are
    masked by ``block_len``.
    """
    from ..native import int8_attention as native

    heads, device = buffers[0].shape[0], buffers[0].device
    quantized = native.prequantize_int8_attention(
        *(buffer.unsqueeze(0) for buffer in buffers), cta_k=TILE,
    )
    buffers.clear()
    kernel_route = _kernel_route(route, geometry, heads, device)
    return native.block_sparse_int8_attention_tiles_from_prequantized(
        quantized, kernel_route, block_len,
    )[0]


_STREAMED_FALLBACK_LOGGED = set()


def _log_streamed_fallback(error):
    import logging
    key = str(error)
    if key not in _STREAMED_FALLBACK_LOGGED:
        _STREAMED_FALLBACK_LOGGED.add(key)
        logging.warning('[H3 VSA] streamed INT8 unavailable, using buffered INT8: %s', error)


def vsa_attention(attn, x, rope_freqs, geometry, sparsity, dense=False, eager=False,
                  backend=BACKEND_BF16, lower_vram=False):
    '''VSA self-attention for packed rows ``x`` [seq, hidden] -> [seq, hidden].'''
    heads, head_dim = attn.heads, attn.head_dim
    if attn.to_gate_compress is None:
        raise RuntimeError('VSA attention needs the checkpoint to_gate_compress layers')
    if x.shape[0] != geometry.seq_len:
        raise RuntimeError(
            'VSA geometry covers %d rows, got %d' % (geometry.seq_len, x.shape[0])
        )
    if backend == BACKEND_INT8 and not eager:
        from .streamed import StreamedVSAUnavailable, vsa_attention_streamed_int8
        try:
            return vsa_attention_streamed_int8(
                attn, x, rope_freqs, geometry, sparsity, dense=dense, lower_vram=lower_vram,
            )
        except StreamedVSAUnavailable as error:
            _log_streamed_fallback(error)
    shape = (heads, geometry.padded_len, head_dim)
    buffers = tuple(torch.zeros(shape, dtype=x.dtype, device=x.device) for _ in range(3))
    _project_into_tiles(attn, x, rope_freqs, geometry, buffers)
    q_buf, k_buf, v_buf = buffers
    block_len = geometry.block_len.to(x.device)

    scores = torch.matmul(
        _tile_means(q_buf, block_len),
        _tile_means(k_buf, block_len).transpose(-1, -2),
    ) * (head_dim ** -0.5)
    route = None if dense else _route(scores, geometry, sparsity)
    coarse = torch.matmul(torch.softmax(scores, dim=-1), _tile_means(v_buf, block_len)).to(x.dtype)
    del scores
    if backend == BACKEND_INT8 and not eager:
        tiles = [q_buf, k_buf, v_buf]
        del buffers, q_buf, k_buf, v_buf
        fine = _int8_fine(tiles, block_len, route, geometry)
    else:
        attend = kernel.sparse_attention_eager if eager else kernel.sparse_attention
        fine = attend(q_buf, k_buf, v_buf, block_len, route, geometry.num_prefix_tiles)
        del buffers, q_buf, k_buf, v_buf
    del route

    output = None
    for start in range(0, x.shape[0], PRODUCER_ROWS):
        stop = min(start + PRODUCER_ROWS, x.shape[0])
        rows = stop - start
        slots = geometry.inv[start:stop]
        chunk = fine.index_select(1, slots).transpose(0, 1)
        gate = attn.to_gate_compress(x[start:stop]).view(rows, heads, head_dim)
        chunk = chunk + coarse.index_select(1, slots // TILE).transpose(0, 1) * gate
        projected = attn.out_proj(chunk.reshape(rows, heads * head_dim))
        if output is None:
            output = torch.empty(
                (x.shape[0], projected.shape[-1]), dtype=projected.dtype, device=x.device,
            )
        output[start:stop] = projected
        del chunk, gate, projected
    return output
