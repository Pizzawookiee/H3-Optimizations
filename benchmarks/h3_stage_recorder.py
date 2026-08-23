'''CUDA event recorder for h3_optimizations.diagnostics stage regions.

Events are recorded on the active stream and never queried until the caller
closes a forward, so nothing here synchronizes inside the measured region.
Regions nest, and nesting is preserved: every region records its own pair, so
``attention_total`` remains the authoritative figure and the children explain
where its time went rather than defining it.
'''

from __future__ import annotations

import statistics


class StageRecorder:
    '''Record one CUDA event pair per entered region.'''

    def __init__(self, torch, device):
        self.torch = torch
        self.device = device
        self.enabled = False
        self._open = None
        self._forwards = []

    # -- the diagnostics recorder protocol ---------------------------------

    def stage(self, name):
        if not self.enabled or self._open is None:
            return _NOOP
        return _Region(self, str(name))

    # -- forward bookkeeping ------------------------------------------------

    def begin_forward(self):
        self._open = {}

    def end_forward(self):
        '''Resolve the forward's events. The caller must have synchronized.'''
        resolved = {}
        for name, pairs in self._open.items():
            samples = [
                float(start.elapsed_time(stop)) for start, stop in pairs
            ]
            resolved[name] = {
                'gpu_ms': sum(samples),
                'calls': len(samples),
                'max_call_ms': max(samples),
            }
        self._open = None
        self._forwards.append(resolved)
        return resolved

    def discard_forward(self):
        self._open = None

    @property
    def forwards(self):
        return list(self._forwards)

    def reset(self):
        self._forwards = []

    # -- internals ----------------------------------------------------------

    def _record(self):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _close(self, name, start):
        self._open.setdefault(name, []).append((start, self._record()))


class _Region:
    __slots__ = ('_recorder', '_name', '_start')

    def __init__(self, recorder, name):
        self._recorder = recorder
        self._name = name
        self._start = None

    def __enter__(self):
        self._start = self._recorder._record()
        return None

    def __exit__(self, exc_type, exc, traceback):
        if self._start is not None:
            self._recorder._close(self._name, self._start)
            self._start = None
        return False


class _Noop:
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


_NOOP = _Noop()


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(forwards, names=None):
    '''Median and spread per stage across forwards.'''
    forwards = list(forwards)
    if names is None:
        names = sorted({name for forward in forwards for name in forward})
    summary = {}
    for name in names:
        samples = [
            forward[name]['gpu_ms'] for forward in forwards if name in forward
        ]
        if not samples:
            continue
        calls = [
            forward[name]['calls'] for forward in forwards if name in forward
        ]
        median = statistics.median(samples)
        p10 = percentile(samples, 0.10)
        p90 = percentile(samples, 0.90)
        summary[name] = {
            'median_gpu_ms': median,
            'p10_gpu_ms': p10,
            'p90_gpu_ms': p90,
            'min_gpu_ms': min(samples),
            'max_gpu_ms': max(samples),
            'iqr_gpu_ms': percentile(samples, 0.75) - percentile(samples, 0.25),
            'spread_percent_of_median': (
                None if not median else (p90 - p10) * 100.0 / median
            ),
            'samples': len(samples),
            'median_calls': statistics.median(calls),
        }
    return summary
