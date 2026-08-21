'''Architecture-aware dense Sage selection for H3 Memory Optimization.'''

from __future__ import annotations

from dataclasses import dataclass


ATTENTION_AUTO = 'auto'
ATTENTION_EXISTING = 'existing'


@dataclass(frozen=True)
class DenseResolution:
    requested: str
    selected: str
    backend: object | None
    reason: str
    backend_kind: str


_BACKENDS = {
    (8, 0): ('SageSM80MemoryEfficientBackend', 'dense_sage_sm80'),
    (8, 6): ('SageSM86MemoryEfficientBackend', 'dense_sage_sm86'),
    (8, 9): ('SM89SageMemoryEfficientBackend', 'dense_sage_sm89'),
    (9, 0): ('SageSM90MemoryEfficientBackend', 'dense_sage_sm90'),
    (12, 0): ('SageSM12xMemoryEfficientBackend', 'dense_sage_sm12x'),
    (12, 1): ('SageSM12xMemoryEfficientBackend', 'dense_sage_sm12x'),
}


def _backend_class(name):
    from . import attention

    return getattr(attention, name)


def _existing_resolution(requested, reason):
    return DenseResolution(
        requested,
        ATTENTION_EXISTING,
        None,
        reason,
        'existing',
    )


def preserve_dense_attention(reason):
    return _existing_resolution(ATTENTION_EXISTING, reason)


def resolve_dense_attention(environment):
    if not environment.cuda_available or environment.capability is None:
        return _existing_resolution(ATTENTION_AUTO, environment.device_name)

    entry = _BACKENDS.get(tuple(environment.capability))
    if entry is None:
        return _existing_resolution(
            ATTENTION_AUTO,
            'no prepared dense Sage backend supports %s'
            % environment.architecture,
        )
    class_name, backend_kind = entry
    try:
        backend = _backend_class(class_name)()
    except Exception as exc:
        return _existing_resolution(
            ATTENTION_AUTO,
            '%s preflight failed: %s: %s'
            % (class_name, type(exc).__name__, exc),
        )
    return DenseResolution(
        ATTENTION_AUTO,
        backend_kind,
        backend,
        '%s detected' % environment.architecture.upper(),
        backend_kind,
    )
