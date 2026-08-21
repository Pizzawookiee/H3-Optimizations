"""Process-local diagnostics for the efficient H3 Sage backends."""

import threading

_LOCK = threading.Lock()
_DEFAULTS = {
    "configured": 0,
    "prepared": 0,
    "executed": 0,
    "tokens": 0,
    "max_sequence": 0,
    "v_guard_copies": 0,
    "qk_guard_copies": 0,
    "kernel_errors": 0,
    "compatibility_errors": 0,
}
_STATS = dict(_DEFAULTS)


def reset_stats():
    with _LOCK:
        _STATS.clear()
        _STATS.update(_DEFAULTS)


def increment(name, value=1):
    with _LOCK:
        _STATS[name] = _STATS.get(name, 0) + value


def observe_sequence(sequence):
    with _LOCK:
        _STATS["prepared"] += 1
        _STATS["tokens"] += int(sequence)
        _STATS["max_sequence"] = max(
            _STATS["max_sequence"],
            int(sequence),
        )


def get_stats():
    with _LOCK:
        return dict(_STATS)
