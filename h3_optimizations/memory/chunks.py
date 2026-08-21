"""Segment-aware token slab planning for MiniMax H3."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenChunk:
    start: int
    stop: int
    mod_row: int

    @property
    def rows(self):
        return self.stop - self.start


def validate_mod_segments(segments, seq_len, mod_rows=None):
    """Return normalized ``(start, stop, mod_row)`` tuples.

    H3's modulation segments must cover the packed sequence contiguously. A
    chunk is never allowed to cross a segment boundary because shift, scale and
    gate are constant only within one segment.
    """
    seq_len = int(seq_len)
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    normalized = []
    expected = 0
    for index, segment in enumerate(segments):
        if len(segment) != 3:
            raise ValueError("segment %d must contain (start, stop, mod_row)" % index)
        start, stop, row = (int(v) for v in segment)
        if start != expected:
            relation = "gap" if start > expected else "overlap"
            raise ValueError(
                "segment %d starts at %d, expected %d (%s)"
                % (index, start, expected, relation)
            )
        if stop <= start:
            raise ValueError(
                "segment %d has non-positive span [%d, %d)" % (index, start, stop)
            )
        if stop > seq_len:
            raise ValueError(
                "segment %d stops at %d past sequence length %d"
                % (index, stop, seq_len)
            )
        if row < 0 or (mod_rows is not None and row >= int(mod_rows)):
            raise ValueError(
                "segment %d modulation row %d is outside [0, %s)"
                % (index, row, "?" if mod_rows is None else int(mod_rows))
            )
        normalized.append((start, stop, row))
        expected = stop

    if expected != seq_len:
        raise ValueError(
            "segments cover [0, %d), expected [0, %d)" % (expected, seq_len)
        )
    if seq_len and not normalized:
        raise ValueError("non-empty sequence requires at least one modulation segment")
    return tuple(normalized)


def iter_mod_chunks(segments, seq_len, max_rows, alignment=1, mod_rows=None):
    """Yield the largest aligned chunks that stay inside modulation segments."""
    max_rows = int(max_rows)
    alignment = int(alignment)
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    if max_rows < alignment:
        raise ValueError("max_rows must be at least alignment")

    normalized = validate_mod_segments(segments, seq_len, mod_rows=mod_rows)
    for segment_start, segment_stop, row in normalized:
        start = segment_start
        while start < segment_stop:
            remaining = segment_stop - start
            size = min(remaining, max_rows)
            if size < remaining and alignment > 1:
                size = (size // alignment) * alignment
                if size <= 0:
                    raise ValueError(
                        "alignment %d leaves no rows inside max_rows %d"
                        % (alignment, max_rows)
                    )
            stop = start + size
            yield TokenChunk(start, stop, row)
            start = stop
