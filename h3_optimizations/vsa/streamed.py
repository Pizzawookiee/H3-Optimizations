'''Streamed INT8 VSA: the pack's Kitchen producer lifetime, in VSA tile order.

Same contract as the streamed Sparse Kitchen path: retain only the global INT8
K/V carrier plus per-tile routing summaries, then project, route, attend and
project out one query slab at a time. The full BF16 Q/K/V never exists.

VSA differences:

- every chunk is a range of *padded* tile slots; its rows are gathered through
  ``geometry.src``, projected with the held weights, and pad slots are zeroed
  afterwards (pads gather a real row, so no all-zero activation reaches a
  quantizing linear);
- K anchor samples are taken at the producer's padded positions, pads reading
  as zero, so the anchor matches the carrier it centres;
- routing is FastVideo's: fp32 post-RoPE tile means, ``ceil`` top-k video
  tiles, prefix keys always, prefix queries dense;
- the coarse branch ``softmax(scores) @ v_mean`` times ``to_gate_compress``
  is added per slab before ``out_proj``.

``lower_vram`` mirrors H3 Memory Optimization's Lower VRAM mode: V is staged in
two passes (absmax, then re-projection into INT8) instead of retained in BF16,
and each slab's ``out_proj`` rows are written into the attention input, which
the H3 block discards after attention. Slabs are disjoint and the K/V pass has
finished by then, so no row is read after it is overwritten.
'''

from __future__ import annotations

import torch

from .. import diagnostics
from .tiling import TILE, compute_topk

SLAB_ROWS = 4096


class StreamedVSAUnavailable(RuntimeError):
    pass


def _gather(x, rope_freqs, slots, geometry):
    src = geometry.src[slots]
    live = src >= 0
    rows = src.clamp_min(0)
    xg = x.index_select(0, rows)
    rope = None if rope_freqs is None else rope_freqs.index_select(1, rows)
    return xg, rope, live


def _zero_pads(tensor_hnd, live):
    # [1, H, m, D]; pads gathered a real row, so masking cannot meet a NaN
    return tensor_hnd.masked_fill(~live.view(1, 1, -1, 1), 0)


def _tile_sums(tensor_hnd):
    batch, heads, rows, dim = tensor_hnd.shape
    return tensor_hnd.reshape(batch, heads, rows // TILE, TILE, dim).sum(
        dim=3, dtype=torch.float32,
    )[0]


def _slab_route(scores, geometry, sparsity, tile_start, dense):
    '''Delta route for one slab of query tiles: [1, H, tiles, width].'''
    heads, slab_tiles, tiles = scores.shape
    prefix = geometry.num_prefix_tiles
    device = scores.device
    keep = compute_topk(sparsity, geometry.num_video_tiles)
    sparse = not dense and keep < geometry.num_video_tiles
    dense_rows = max(0, min(prefix - tile_start, slab_tiles)) if sparse else slab_tiles
    width = tiles if dense_rows else prefix + keep
    walk_all = torch.ones(tiles, dtype=torch.int32, device=device)
    walk_all[0] = 0
    indices = torch.zeros((1, heads, slab_tiles, width), dtype=torch.int32, device=device)
    counts = torch.full((1, heads, slab_tiles), width, dtype=torch.int32, device=device)
    if dense_rows:
        indices[0, :, :dense_rows, :tiles] = walk_all
        counts[0, :, :dense_rows] = tiles
    if dense_rows < slab_tiles:
        video = scores[:, dense_rows:, prefix:].topk(keep, dim=-1).indices + prefix
        absolute = torch.cat((
            torch.arange(prefix, device=device).expand(heads, slab_tiles - dense_rows, prefix),
            video.sort(dim=-1).values,
        ), dim=-1).to(torch.int32)
        steps = torch.diff(absolute, dim=-1, prepend=torch.zeros_like(absolute[..., :1]))
        indices[0, :, dense_rows:, :prefix + keep] = steps
        counts[0, :, dense_rows:] = prefix + keep
    return indices, counts


def vsa_attention_streamed_int8(attn, x, rope_freqs, geometry, sparsity, dense=False,
                                lower_vram=False):
    '''Streamed INT8 VSA for packed rows ``x`` -> [seq, hidden].

    Raises StreamedVSAUnavailable when the producer or held projection cannot
    serve this module, so the caller can fall back.
    '''
    from ..kitchen_qkv import _qk_chunk_kwargs, _quantize_q_chunk, resolve_kitchen
    from ..native import int8_attention as native
    from ..qkv.streamed import (
        PROJECTION_NATIVE,
        StreamedQKVBindingError,
        create_held_qkv,
        project_kv_hnd,
        project_q_hnd,
        project_v_hnd,
    )

    kitchen = resolve_kitchen(x.device)
    if kitchen is None or not hasattr(kitchen, 'quantize_int8_attention_k_chunk'):
        raise StreamedVSAUnavailable('no INT8 attention producer on this device')
    if not native.sparse_tiles_attention_is_available():
        raise StreamedVSAUnavailable('the loaded INT8 library has no per-tile KV traversal')

    heads, head_dim = int(attn.heads), int(attn.head_dim)
    padded, tiles = geometry.padded_len, geometry.num_tiles
    device = x.device
    block_len = geometry.block_len.to(device)
    lengths = block_len.to(torch.float32).view(1, -1, 1)
    try:
        spec = kitchen.int8_attention_producer_spec(
            (1, heads, 1, head_dim), (1, heads, padded, head_dim),
            dtype=x.dtype, device=device, cta_k=TILE,
        )
        held = create_held_qkv(attn, x[:1], PROJECTION_NATIVE)
    except (kitchen.Int8AttentionProducerUnavailableError, StreamedQKVBindingError) as error:
        raise StreamedVSAUnavailable(str(error)) from error
    if SLAB_ROWS % int(spec.sequence_alignment):
        raise StreamedVSAUnavailable('slab rows do not match the producer alignment')

    held.__enter__()
    try:
        # K anchor sampled at the carrier's own padded positions
        with diagnostics.stage('anchor_projection'):
            positions = torch.tensor(spec.k_anchor_positions, dtype=torch.int64, device=device)
            xs, rope_s, live_s = _gather(x, rope_freqs, positions, geometry)
            _k, k_samples, _v = held.project_rows(xs, rope_s, torch.arange(xs.shape[0], device=device))
            k_samples = _zero_pads(k_samples, live_s)
            del _k, _v, xs, rope_s
        anchor = kitchen.select_int8_attention_k_anchor(spec, k_samples)
        del k_samples
        producer = kitchen.create_int8_attention_producer(spec, anchor)
        del anchor
        chunk_kwargs = _qk_chunk_kwargs(kitchen, True)

        k_sums = torch.empty((heads, tiles, head_dim), dtype=torch.float32, device=device)
        v_sums = torch.empty_like(k_sums)
        retained_v = None
        staging = None
        if lower_vram:
            from ..native.v_staging import TwoPassVCarrier

            staging = TwoPassVCarrier(spec)
        for start in range(0, padded, SLAB_ROWS):
            stop = min(start + SLAB_ROWS, padded)
            slots = torch.arange(start, stop, device=device)
            xg, rope, live = _gather(x, rope_freqs, slots, geometry)
            k, v = project_kv_hnd(held, xg, rope, 0, stop - start)
            del xg, rope
            k, v = _zero_pads(k, live), _zero_pads(v, live)
            kitchen.quantize_int8_attention_k_chunk(producer, k, k_start=start, **chunk_kwargs)
            first, last = start // TILE, stop // TILE
            k_sums[:, first:last] = _tile_sums(k)
            v_sums[:, first:last] = _tile_sums(v)
            if staging is not None:
                with diagnostics.stage('v_amax_update'):
                    staging.update(v)
            else:
                if retained_v is None:
                    retained_v = v.new_empty((1, heads, padded, head_dim))
                retained_v[..., start:stop, :].copy_(v)
            del k, v, live
        if staging is None:
            kitchen.quantize_int8_attention_v(producer, retained_v)
            del retained_v
        else:
            staging.finalize_scale()
            for start in range(0, padded, SLAB_ROWS):
                stop = min(start + SLAB_ROWS, padded)
                slots = torch.arange(start, stop, device=device)
                xg, rope, live = _gather(x, rope_freqs, slots, geometry)
                with diagnostics.stage('v_reprojection'):
                    v = _zero_pads(project_v_hnd(held, xg, rope, 0, stop - start), live)
                del xg, rope, live
                with diagnostics.stage('v_carrier_pack'):
                    staging.quantize(v, start)
                del v
            producer.v, producer.v_scale = staging.finish()
            del staging
        carrier = kitchen.finalize_int8_attention_producer(producer)
        del producer
        k_means = k_sums / lengths
        v_means = v_sums / lengths
        del k_sums, v_sums

        output = None
        if lower_vram and x.is_contiguous() and x.shape[-1] == int(attn.out_proj.weight.shape[0]):
            output = x
        scale = head_dim ** -0.5
        for start in range(0, padded, SLAB_ROWS):
            stop = min(start + SLAB_ROWS, padded)
            rows = stop - start
            slots = torch.arange(start, stop, device=device)
            xg, rope, live = _gather(x, rope_freqs, slots, geometry)
            q = _zero_pads(project_q_hnd(held, xg, rope, 0, rows), live)
            del rope
            first = start // TILE
            slab_tiles = rows // TILE
            with diagnostics.stage('sparse_route'):
                q_means = _tile_sums(q) / lengths[:, first:first + slab_tiles]
                scores = torch.matmul(q_means, k_means.transpose(-1, -2)) * scale
                del q_means
                indices, counts = _slab_route(scores, geometry, sparsity, first, dense)
            chunk_carrier = _quantize_q_chunk(kitchen, carrier, q)
            del q
            route = native.BlockSparseRoute(
                indices=indices, counts=counts, q_tile=TILE, kv_tile=TILE, encoding='delta',
            )
            fine = native.block_sparse_int8_attention_tiles_from_prequantized(
                chunk_carrier, route, block_len,
            )[0]
            del chunk_carrier, route, indices, counts
            coarse = torch.matmul(torch.softmax(scores, dim=-1), v_means).to(fine.dtype)
            del scores
            live_rows = torch.nonzero(live).flatten()
            with diagnostics.stage('attention_out'):
                chunk = fine.index_select(1, live_rows).transpose(0, 1)
                gate = attn.to_gate_compress(xg.index_select(0, live_rows)).view(-1, heads, head_dim)
                chunk = chunk + coarse.index_select(1, live_rows // TILE).transpose(0, 1) * gate
                projected = attn.out_proj(chunk.reshape(-1, heads * head_dim))
                if output is None:
                    output = torch.empty(
                        (x.shape[0], projected.shape[-1]), dtype=projected.dtype, device=device,
                    )
                output.index_copy_(
                    0, geometry.src[start:stop][live_rows], projected.to(output.dtype),
                )
            del fine, coarse, chunk, gate, projected, xg, live, live_rows
        return output
    finally:
        held.__exit__(None, None, None)
