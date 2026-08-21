"""Dense prepared-Sage selection without the monolithic memory optimizer."""

from __future__ import annotations

from dataclasses import dataclass

from .plan import ATTENTION_AUTO, ATTENTION_EXISTING


@dataclass(frozen=True)
class DenseResolution:
    requested: str
    selected: str
    backend: object | None
    reason: str
    backend_kind: str


_BACKENDS = {
    (8, 0): ("SageSM80MemoryEfficientBackend", "dense_sage_sm80"),
    (8, 6): ("SageSM86MemoryEfficientBackend", "dense_sage_sm86"),
    (8, 9): ("SM89SageMemoryEfficientBackend", "dense_sage_sm89"),
    (9, 0): ("SageSM90MemoryEfficientBackend", "dense_sage_sm90"),
    (12, 0): ("SageSM12xMemoryEfficientBackend", "dense_sage_sm12x"),
    (12, 1): ("SageSM12xMemoryEfficientBackend", "dense_sage_sm12x"),
}


def _backend_class(name):
    from . import attention

    return getattr(attention, name)


def resolve_dense_attention(requested, environment):
    if requested == ATTENTION_EXISTING:
        return DenseResolution(
            requested,
            ATTENTION_EXISTING,
            None,
            "existing attention requested",
            "existing",
        )
    if requested != ATTENTION_AUTO:
        raise ValueError("unknown dense attention request %r" % requested)
    if not environment.cuda_available or environment.capability is None:
        return DenseResolution(
            requested,
            ATTENTION_EXISTING,
            None,
            environment.device_name,
            "existing",
        )

    entry = _BACKENDS.get(tuple(environment.capability))
    if entry is None:
        return DenseResolution(
            requested,
            ATTENTION_EXISTING,
            None,
            "no prepared dense Sage backend supports %s"
            % environment.architecture,
            "existing",
        )
    class_name, backend_kind = entry
    try:
        backend = _backend_class(class_name)()
    except Exception as exc:
        return DenseResolution(
            requested,
            ATTENTION_EXISTING,
            None,
            "%s preflight failed: %s: %s"
            % (class_name, type(exc).__name__, exc),
            "existing",
        )
    return DenseResolution(
        requested,
        backend_kind,
        backend,
        "%s detected" % environment.architecture.upper(),
        backend_kind,
    )
