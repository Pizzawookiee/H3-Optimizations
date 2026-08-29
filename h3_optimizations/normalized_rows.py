'''Lazily normalized attention input rows.

The QKV producers only ever read their input as ``x[start:end]`` row slices,
plus ``shape``/``dtype``/``device``/``new_empty``. That lets the pre-attention
``norm1`` + modulation run per requested slice instead of materializing a full
``[sequence, hidden]`` tensor that stays resident for the whole block.

The lazy row source is private to the package-owned attention/QKV transaction.
The DiT block still calls attention with an ordinary Tensor; the optimized
attention forward extracts this row source from its private transformer option
and hands it only to a projector that explicitly owns the projected-QKV path.
Foreign or standard attention forwards therefore never have to emulate a new
Tensor protocol.

The trade is recomputation: a route that walks the sequence more than once
(streamed Kitchen projects K/V and Q in separate passes, and two-pass V adds a
third) normalizes those rows once per pass. RMSNorm at H3 shapes is bandwidth
bound, so this is close to even on time and removes one full-sequence BF16
tensor from the block's peak.

Consumers that need a real tensor can call ``materialize()``. Streamed
attention consumers call ``attention_output_buffer()`` when they are ready to
overwrite the disposable normalized input. For lazy rows that allocates a
distinct output tensor at that late point, because the wrapped tensor is still
the block residual.
'''

from __future__ import annotations

import torch


NORM1_SOURCE_KEY = 'h3_optimizations_norm1_source'


class NormalizedRowsUnsupported(AttributeError):
    pass


class NormalizedRows:
    '''A read-only ``[sequence, hidden]`` row source with fused norm + modulation.

    Only the surface the QKV producers actually use is implemented. Anything
    else is deliberately absent so an unsupported projector fails loudly
    instead of silently reading unnormalized rows.
    '''

    __slots__ = (
        '_x',
        '_norm',
        '_segments',
        '_shift',
        '_scale',
        '_apply',
        '_output',
        'shape',
        'dtype',
        'device',
        'ndim',
        'is_cuda',
        'requires_grad',
    )

    def __init__(self, x, norm, segments, shift, scale, apply_modulation):
        self._x = x
        self._norm = norm
        self._segments = tuple(segments)
        self._shift = shift
        self._scale = scale
        self._apply = apply_modulation
        self._output = None
        self.shape = x.shape
        self.dtype = x.dtype
        self.device = x.device
        self.ndim = x.ndim
        self.is_cuda = x.is_cuda
        self.requires_grad = False

    def new_empty(self, *args, **kwargs):
        return self._x.new_empty(*args, **kwargs)

    def size(self, dim=None):
        return self._x.shape if dim is None else self._x.shape[dim]

    def dim(self):
        return self._x.dim()

    def numel(self):
        return self._x.numel()

    def element_size(self):
        return self._x.element_size()

    def stride(self, dim=None):
        return self._x.stride() if dim is None else self._x.stride(dim)

    def is_contiguous(self, *args, **kwargs):
        return self._x.is_contiguous(*args, **kwargs)

    def index_select(self, dim, index):
        '''Gathered rows, normalized and modulated. Anchor/sample selection.'''
        if dim != 0:
            raise NormalizedRowsUnsupported(
                'NormalizedRows supports row (dim 0) selection only'
            )
        rows = self._norm(self._x.index_select(0, index))
        self._apply(rows, self._shift, self._scale, self._row_selectors(index))
        return rows

    def __len__(self):
        return int(self._x.shape[0])

    def __getitem__(self, item):
        start, stop = self._bounds(item)
        rows = self._norm(self._x[start:stop])
        self._modulate(rows, start, stop)
        return rows

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types, args, kwargs
        raise NormalizedRowsUnsupported(
            'NormalizedRows does not support torch.%s' % func.__name__
        )

    def __getattr__(self, name):
        raise NormalizedRowsUnsupported(
            'NormalizedRows does not expose tensor attribute %r' % name
        )

    def _bounds(self, item):
        sequence = int(self._x.shape[0])
        if isinstance(item, slice):
            if item.step not in (None, 1):
                raise NormalizedRowsUnsupported(
                    'NormalizedRows supports contiguous row slices only'
                )
            start, stop, _ = item.indices(sequence)
            if stop < start:
                stop = start
            return start, stop
        raise NormalizedRowsUnsupported(
            'NormalizedRows supports row slices only, got %r' % type(item).__name__
        )

    def _modulate(self, rows, start, stop):
        for seg_start, seg_stop, selector in self._segments:
            lo = seg_start if seg_start > start else start
            hi = seg_stop if seg_stop < stop else stop
            if lo >= hi:
                continue
            if torch.is_tensor(selector):
                sub = selector[lo - seg_start:hi - seg_start]
            else:
                sub = selector
            self._apply(rows[lo - start:hi - start], self._shift, self._scale, sub)

    def _row_selectors(self, index):
        out = torch.zeros(
            int(index.shape[0]), dtype=torch.long, device=index.device
        )
        for seg_start, seg_stop, selector in self._segments:
            mask = (index >= seg_start) & (index < seg_stop)
            if torch.is_tensor(selector):
                local = index[mask] - seg_start
                if selector.device != index.device:
                    selector = selector.to(device=index.device)
                out[mask] = selector[local]
            else:
                out[mask] = int(selector)
        return out

    def materialize(self):
        '''Full ``[sequence, hidden]`` tensor, for consumers that need one.'''
        return self[0:int(self._x.shape[0])]

    def output_buffer(self):
        '''Allocate the full attention output only after input projection.'''
        if self._output is None:
            self._output = self._x.new_empty(self.shape)
        return self._output


def attention_output_buffer(x):
    '''Return the disposable tensor that streamed attention may overwrite.'''
    if isinstance(x, NormalizedRows):
        return x.output_buffer()
    return x
