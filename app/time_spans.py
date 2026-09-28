"""The keep spans a file's stored time ranges describe, in seconds.

Shared by the Time ranges tab (what it draws), the episode Working screen
(where the OCR is) and ProjectController.episode_kept_duration (the length
the run will read): all three must agree with what `ocr_call_for` hands OCR,
so they read the stored text with one rule. Pure Python: no Qt, no core.
"""
from __future__ import annotations


def parse_clock(text: str | None) -> float | None:
    """A stored "M:SS" / "H:MM:SS" as seconds, or None for None and for text
    that is not a time (a hand-edited `.ocr.json` can hold anything; a range
    the timeline cannot place is one it must not silently move)."""
    if text is None:
        return None
    try:
        parts = [int(part) for part in str(text).split(":")]
    except ValueError:
        return None
    if not 2 <= len(parts) <= 3 or any(part < 0 for part in parts):
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return float(seconds)


def read_ranges(entry, duration: float) -> tuple[list[tuple[float, float]], list[str]]:
    """(the keep spans the timeline can place, the stored ranges it cannot).

    The spans are in seconds, clamped to [0, duration], ordered and merged --
    overlapping stored ranges would otherwise put the drawn boundaries out of
    order, and a grip between two of them would snap backwards.

    A range whose times cannot be parsed, or that ends before it starts, or
    that falls entirely outside the file, is **not** placed: it is returned
    as the text it is stored as, for the view to name. Reading such a range
    as "0:00 → the end" would be the one thing this view must never do --
    show something other than what the run will use, since `ocr_call_for`
    still hands OCR exactly what is stored.

    Empty and empty means the whole file is kept: `time_ranges` None (never
    set) or an empty MANUAL list ("use whole file"). With no duration yet
    nothing can be placed and nothing is blamed -- see NO_DURATION_TEXT."""
    ranges = getattr(entry, "time_ranges", None)
    if entry is None or ranges is None or not ranges.ranges or duration <= 0:
        return [], []
    keeps, unreadable = [], []
    for stored in ranges.ranges:
        span = _stored_span(stored, duration)
        if span is None:
            unreadable.append(f"{stored.start or '0:00'} → {stored.end or 'end'}")
        else:
            keeps.append(span)
    return _merged(keeps), unreadable


def _stored_span(stored, duration: float) -> tuple[float, float] | None:
    """One stored range as (start_sec, end_sec) inside the file, or None when
    the timeline cannot place it (see `read_ranges`). None times are the open
    start and the open end, which are placeable; unparseable text is not."""
    start = 0.0 if stored.start is None else parse_clock(stored.start)
    end = duration if stored.end is None else parse_clock(stored.end)
    if start is None or end is None or end <= start:
        return None
    start, end = max(0.0, min(duration, start)), max(0.0, min(duration, end))
    return (start, end) if end > start else None


def _merged(keeps) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(keeps):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
