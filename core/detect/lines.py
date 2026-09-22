"""Random subtitle lines of one file, for the Brightness tab's gallery.

sample_lines() draws random times inside the file's keep ranges (weighted
towards speech), grabs each strip exactly as the OCR pass sees it
(ocr_view.grab_ocr_strips_at) and keeps the strips text detection finds a
subtitle on. No OCR, no threshold: the view masks the strips itself. See
docs/superpowers/specs/2026-09-18-brightness-gallery-design.md section 1.
No Qt imports.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from core.detect import brightness, ocr_view
from core.detect.flags import is_cancelled as _is_cancelled

logger = logging.getLogger(__name__)

LINE_COUNT = 6
ROUND_SIZE = 12
MAX_ROUNDS = 4
MIN_GAP_SEC = 2.0
MIN_SPEECH_SEC = 10.0

# A speech candidate sits in the middle half of its speech piece: the middle
# of speech is where a subtitle is on screen (see vad.py).
SPEECH_MIDDLE = (0.25, 0.75)
# Draws of one candidate that may land within MIN_GAP_SEC of a taken time
# before the round is closed as full. Enough that a round only closes early
# when its spans are all but covered.
MAX_DRAW_TRIES = 64


@dataclass(frozen=True)
class LineSample:
    """One strip text detection found a subtitle on.

    time: the time the strip was grabbed at (ocr_view.grab_ocr_strips_at).
    boxes: the detector's polygons as (x, y, w, h) in strip pixels
        (brightness._poly_box).
    lines: distinct text rows among the boxes (brightness._count_lines).
    """
    time: float
    boxes: tuple[tuple[int, int, int, int], ...]
    lines: int


@dataclass(frozen=True)
class LinesResult:
    """samples: at most `count` lines, ascending time. tried: strips grabbed
    and run through text detection. seed: the seed the draw used.
    cancelled: the call stopped at cancel_check (no samples are trusted)."""
    samples: tuple[LineSample, ...]
    tried: int
    seed: int
    cancelled: bool = False

    def to_evidence(self) -> dict:
        return {
            "seed": int(self.seed),
            "tried": int(self.tried),
            "samples": [{"time": float(s.time), "boxes": [list(box) for box in s.boxes], "lines": int(s.lines)}
                        for s in self.samples],
        }


def _speech_pieces(speech: Iterable | None, spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The speech segments ([start, end] seconds, evidence["audio"]["speech"])
    intersected with the keep spans, ascending. Speech in a skipped intro or
    outro is not a line the OCR pass will read."""
    pieces = []
    for start, end in speech or ():
        for s, e in spans:
            lo, hi = max(float(start), s), min(float(end), e)
            if hi > lo:
                pieces.append((lo, hi))
    return sorted(pieces)


def _locate(spans: list[tuple[float, float]], pos: float) -> tuple[float, float, float]:
    """(start, end, offset) of the span `pos` seconds falls in along the
    concatenated spans. A pos at or past the total lands at the last span's
    end."""
    for i, (s, e) in enumerate(spans):
        if pos <= e - s or i == len(spans) - 1:
            return s, e, min(pos, e - s)
        pos -= e - s
    raise ValueError("no spans")


def _draw(rng: random.Random, spans: list[tuple[float, float]], pieces: list[tuple[float, float]]) -> float:
    """One candidate time. With speech `pieces`: a piece drawn with
    probability proportional to its length, then a time uniformly within its
    middle half. Without: uniform over the keep `spans`. Rounded to the
    millisecond, so the time stored as evidence is the exact time grabbed
    (grab_ocr_strips_at maps it to round(t * fps); a view that re-fetches the
    stored time gets the same frame)."""
    if pieces:
        s, e, _ = _locate(pieces, rng.uniform(0.0, sum(e - s for s, e in pieces)))
        t = s + (e - s) * rng.uniform(*SPEECH_MIDDLE)
    else:
        s, _, offset = _locate(spans, rng.uniform(0.0, sum(e - s for s, e in spans)))
        t = s + offset
    return round(t, 3)


def _clear_of(t: float, taken: list[float]) -> bool:
    return all(abs(t - u) >= MIN_GAP_SEC for u in taken)


def _draw_round(rng: random.Random, spans, pieces, taken: list[float]) -> list[float]:
    """Up to ROUND_SIZE candidate times, in draw order, each at least
    MIN_GAP_SEC from every `taken` time (excluded times and lines already
    held) and from the round's other candidates. A candidate that cannot be
    placed in MAX_DRAW_TRIES draws closes the round: the spans are full, and
    the round is simply smaller (possibly empty)."""
    taken = list(taken)
    times: list[float] = []
    while len(times) < ROUND_SIZE:
        for _ in range(MAX_DRAW_TRIES):
            t = _draw(rng, spans, pieces)
            if _clear_of(t, taken):
                break
        else:
            break
        times.append(t)
        taken.append(t)
    return times


def _line_sample(time: float, strip: np.ndarray, polys) -> LineSample | None:
    """The LineSample of one detected strip, or None when it holds no line:
    no polygons, or glyphs too few to measure (brightness._glyph_level; the
    tiles.py rule -- a strip too small to split is usually a speck the
    detector boxed, not a subtitle)."""
    if not polys or brightness._glyph_level(strip, polys) is None:
        return None
    boxes = tuple(box for box in (brightness._poly_box(p) for p in polys) if box is not None)
    if not boxes:
        return None
    return LineSample(time=float(time), boxes=boxes, lines=brightness._count_lines(boxes))


def sample_lines(video_path: str, crop_box, time_ranges, speech: Iterable | None, det_engine, *,
                 count: int = LINE_COUNT, seed: int, exclude: Iterable[float] = (),
                 cancel_check: Callable[[], bool] | None = None) -> LinesResult:
    """Up to `count` random subtitle lines of one file.

    Candidate times come from random.Random(seed): inside the keep ranges
    (ocr_view.keep_spans), from the middle of speech when the speech inside
    them totals at least MIN_SPEECH_SEC, else uniformly; never within
    MIN_GAP_SEC of an `exclude` time (the lines a shuffle replaces, the
    detector's own tiles), of a line already held or of another candidate
    of the same round. Rounds of ROUND_SIZE candidates, at most MAX_ROUNDS:
    each round's strips are grabbed exactly as the OCR pass sees them and
    run through text detection in one batch, unmasked; a strip is a line
    when detection boxes it and its glyphs are measurable (_line_sample).
    Lines are kept in draw order until `count` are held, so a line never
    depends on where a later candidate fell.

    `det_engine` is a detection engine the caller holds an exclusive lease
    on for the whole call (videocr.engine_registry.lease_detection_engine).
    `cancel_check` is polled before each round; a cancelled call returns
    no samples, cancelled=True. A file that cannot be opened, has no
    duration or whose keep ranges select nothing returns no samples, tried
    0. Deterministic for the same seed, inputs and decoded frames.
    """
    try:
        duration = ocr_view.video_duration(video_path)
    except ocr_view.FETCH_ERRORS as exc:
        logger.warning("%s: cannot open (%s: %s)", video_path, type(exc).__name__, exc)
        return LinesResult(samples=(), tried=0, seed=seed)
    spans = [(s, e) for s, e in ocr_view.keep_spans(duration, time_ranges) if e > s] if duration > 0 else []
    if not spans:
        return LinesResult(samples=(), tried=0, seed=seed)
    pieces = _speech_pieces(speech, spans)
    if sum(e - s for s, e in pieces) < MIN_SPEECH_SEC:
        pieces = []

    rng = random.Random(seed)
    excluded = [float(t) for t in exclude]
    held: list[LineSample] = []
    tried = 0
    for _ in range(MAX_ROUNDS):
        if len(held) >= count:
            break
        if _is_cancelled(cancel_check):
            return LinesResult(samples=(), tried=tried, seed=seed, cancelled=True)
        times = _draw_round(rng, spans, pieces, excluded + [line.time for line in held])
        if not times:
            break           # no room left for a candidate; later rounds would find none either
        grabbed = ocr_view.grab_ocr_strips_at(video_path, crop_box, times)
        polys_per_strip = brightness._detect_text_polys(det_engine, [strip for _, strip in grabbed])
        tried += len(grabbed)
        for (t, strip), polys in zip(grabbed, polys_per_strip):
            if len(held) >= count:
                break
            sample = _line_sample(t, strip, polys)
            if sample is not None:
                held.append(sample)
    return LinesResult(samples=tuple(sorted(held, key=lambda line: line.time)), tried=tried, seed=seed)
