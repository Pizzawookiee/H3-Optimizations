'''Attention-derived target-video importance published to MLP experiments.

Sparse Sage already ranks pure-video KV tiles per head and per query tile while
building its route. The packed delta LUT discards that ranking, so this module
keeps it alive for one block forward and records the matching gated
attention-residual energy over the same tiles.

Two signals are collected per pure-video KV tile:

``hit_rate``
    Fraction of ``(head, query tile)`` draws that retained the tile as a KV
    source. It answers "did other target-video tokens want to read here".
``attention_update_energy``
    Mean ``|gate_msa * attention_output|^2`` over the tile's rows. It answers
    "did attention change this region". This is a property of the tile's own
    query rows, not of how often other rows read it, so it is deliberately not
    named to match ``hit_rate``.

Nothing here changes attention or MLP output. Recording is opt-in: with no
recorder in ``transformer_options`` the router and the block forward take their
normal paths and pay one dictionary lookup.
'''

from __future__ import annotations

import torch

ROUTE_KEY = 'h3_optimizations_attention_route'
DEFAULT_KV_TILE = 64
SIGNALS = ('hit_rate', 'attention_update_energy')


class AttentionRouteError(RuntimeError):
    pass


class BlockRoute:
    '''Attention importance over the pure-video KV tiles of one block.'''

    def __init__(self, layer, kv_tile):
        self.layer = int(layer)
        self.kv_tile = int(kv_tile)
        self.video_kv_start = None
        self.video_kv_tiles = None
        self.draws = 0
        self.dense = False
        self.counts = None
        # Absolute per-KV-tile means over the whole packed sequence; the
        # accessor below slices them to the video-relative indexing that
        # hit_rate and every consumer use.
        self.tile_energy = None

    @property
    def has_selection(self):
        return self.video_kv_tiles is not None

    def hit_rate(self):
        '''Per-tile retention rate in [0, 1], or None without a selection.'''
        if not self.has_selection:
            return None
        if self.dense or not self.draws:
            device = None if self.counts is None else self.counts.device
            return torch.ones(
                self.video_kv_tiles,
                dtype=torch.float32,
                device=device,
            )
        return self.counts.float() / float(self.draws)

    def attention_update_energy(self):
        '''Per-tile gated attention-residual energy, video-relative.'''
        if self.tile_energy is None or not self.has_selection:
            return None
        start = int(self.video_kv_start)
        stop = start + int(self.video_kv_tiles)
        if int(self.tile_energy.shape[0]) < stop:
            raise AttentionRouteError(
                'recorded %d KV tiles but the route needs %d'
                % (int(self.tile_energy.shape[0]), stop)
            )
        return self.tile_energy[start:stop]

    def signal(self, name):
        if name == 'hit_rate':
            return self.hit_rate()
        if name == 'attention_update_energy':
            return self.attention_update_energy()
        raise AttentionRouteError('unknown attention route signal %r' % name)

    def as_dict(self):
        return {
            'layer': self.layer,
            'kv_tile': self.kv_tile,
            'video_kv_start': self.video_kv_start,
            'video_kv_tiles': self.video_kv_tiles,
            'draws': int(self.draws),
            'dense': bool(self.dense),
            'has_counts': self.counts is not None,
            'has_attention_update_energy': self.tile_energy is not None,
        }


def _tile_means(values, kv_tile):
    '''Mean of ``values`` over consecutive ``kv_tile`` row groups.'''
    rows = int(values.shape[0])
    tiles = (rows + kv_tile - 1) // kv_tile
    padded = torch.zeros(
        tiles * kv_tile,
        dtype=torch.float32,
        device=values.device,
    )
    padded[:rows] = values.float()
    sums = padded.view(tiles, kv_tile).sum(dim=1)
    occupancy = torch.full(
        (tiles,),
        float(kv_tile),
        dtype=torch.float32,
        device=values.device,
    )
    remainder = rows % kv_tile
    if remainder:
        occupancy[-1] = float(remainder)
    return sums / occupancy


class AttentionRouteRecorder:
    '''Collect attention importance for the blocks of one forward pass.'''

    def __init__(self, kv_tile=DEFAULT_KV_TILE):
        kv_tile = int(kv_tile)
        if kv_tile <= 0:
            raise AttentionRouteError('kv_tile must be positive')
        self.kv_tile = kv_tile
        self._routes = {}

    def _route(self, layer_index):
        layer = int(layer_index)
        route = self._routes.get(layer)
        if route is None:
            route = BlockRoute(layer, self.kv_tile)
            self._routes[layer] = route
        return route

    def route(self, layer_index):
        return self._routes.get(int(layer_index))

    def clear(self, layer_index):
        self._routes.pop(int(layer_index), None)

    def clear_all(self):
        self._routes.clear()

    def sink(self, layer_index):
        '''Return the per-layer callable the sparse router reports through.'''

        def record(indices, geometry, kv_tile):
            self.record_selection(layer_index, indices, geometry, kv_tile)

        return record

    def record_selection(self, layer_index, indices, geometry, kv_tile):
        '''Record one route; ``indices`` is None for a fully dense route.'''
        if int(kv_tile) != self.kv_tile:
            raise AttentionRouteError(
                'attention route recorder is bound to %d-row KV tiles but the '
                'router reported %d' % (self.kv_tile, int(kv_tile))
            )
        route = self._route(layer_index)
        route.video_kv_start = int(geometry.pure_video_kv_start)
        route.video_kv_tiles = int(geometry.pure_video_kv_tiles)
        if indices is None:
            route.dense = True
            route.draws = 0
            route.counts = None
            return route

        if indices.ndim != 4:
            raise AttentionRouteError(
                'route selection must be rank 4 [batch, heads, q_tiles, kept]'
            )
        flat = indices.reshape(-1).to(torch.int64)
        counts = torch.zeros(
            route.video_kv_tiles,
            dtype=torch.int32,
            device=indices.device,
        )
        counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
        route.dense = False
        route.draws = int(indices.shape[0] * indices.shape[1] * indices.shape[2])
        route.counts = counts
        return route

    def record_attention_energy(
        self,
        layer_index,
        attn_out,
        gate,
        segments,
        chunk_rows,
    ):
        '''Record mean gated attention-residual energy per KV tile.'''
        # Imported here: h3_optimizations.memory imports the block forward,
        # which imports this module to look the recorder up.
        from ..memory.chunks import iter_modulation_chunks

        rows = int(attn_out.shape[0])
        energy = torch.empty(
            rows,
            dtype=torch.float32,
            device=attn_out.device,
        )
        for start, stop, selector in iter_modulation_chunks(segments, chunk_rows):
            if torch.is_tensor(selector) and selector.device != gate.device:
                selector = selector.to(device=gate.device)
            gate_rows = gate[selector].to(attn_out.dtype)
            gated = attn_out[start:stop] * gate_rows
            energy[start:stop] = gated.float().pow(2).sum(dim=-1)
            del gate_rows, gated
        route = self._route(layer_index)
        route.tile_energy = _tile_means(energy, self.kv_tile)
        return route


def get_route_recorder(transformer_options):
    if not transformer_options:
        return None
    value = transformer_options.get(ROUTE_KEY)
    return value if isinstance(value, AttentionRouteRecorder) else None


def route_sink(transformer_options, layer_index):
    '''Per-layer router sink, or None when no experiment is recording.'''
    recorder = get_route_recorder(transformer_options)
    if recorder is None:
        return None
    return recorder.sink(layer_index)


def router_kwargs(transformer_options, layer_index):
    '''Preserve the existing router call contract when recording is disabled.'''
    sink = route_sink(transformer_options, layer_index)
    return {} if sink is None else {'sink': sink}


def row_scores(route, chunk_start, chunk_stop, *, signal='hit_rate'):
    '''Per-row attention importance for one MLP chunk.

    Rows that no pure-video KV tile covers - every non-video segment and the
    mixed boundary tile that Sparse Sage keeps dense - come back as NaN so a
    caller keeps them exact instead of scoring them.
    '''
    if route is None or not route.has_selection:
        return None
    values = route.signal(signal)
    if values is None:
        return None
    chunk_start = int(chunk_start)
    chunk_stop = int(chunk_stop)
    if chunk_stop < chunk_start:
        raise AttentionRouteError('chunk_stop precedes chunk_start')
    rows = torch.arange(
        chunk_start,
        chunk_stop,
        dtype=torch.int64,
        device=values.device,
    )
    relative = torch.div(
        rows,
        route.kv_tile,
        rounding_mode='floor',
    ) - int(route.video_kv_start)
    inside = (relative >= 0) & (relative < int(route.video_kv_tiles))
    scores = torch.full(
        (chunk_stop - chunk_start,),
        float('nan'),
        dtype=torch.float32,
        device=values.device,
    )
    if bool(inside.any()):
        scores[inside] = values.index_select(0, relative[inside]).float()
    return scores
