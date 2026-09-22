"""The Time ranges review view (plan 3C Task 4, ui-spec §3.6, ruling B5;
`tabs-hifi.html` figure 3 is canonical).

`RangesTab` is a `StageTab`: the editable `Timeline` -- a 52 px track with
the waveform, the keep and skip spans and their amber grips, fused to a
16 px speech lane -- plus one warning row per speech span that falls inside
a span the run would skip, and an inspector panel listing the keep ranges.
Every px here is a mockup px: the whole of the geometry block below goes
through `tokens.px`, so at UI_SCALE 1.25 that track is 65 px on screen.

Two modes, one widget (ruling B5). The Time ranges tab hosts the editable
one; the Crop and Brightness tabs mount the same widget read-only and
compact under their stage, ticked with the times the tab is showing -- the
crop samples unless the tab says otherwise (`set_marks`; the Brightness tab
ticks its gallery tiles). Compact mode has no grips and never writes: it
reports where it was clicked (`seek_requested`) and lets the tab decide what
that means -- the Crop tab jumps to the nearest sample, the Brightness tab
highlights the nearest tile. Nothing here knows about either.

What it draws
    The keep spans are the file's own `entry.time_ranges` (whole file when
    it is None **or** when its range list is empty -- a manual "use whole
    file" is stored as an empty MANUAL list). The skip spans are the
    complement: what the OCR run will really leave out, whatever the
    detector matched. A skip span borrows the kind and the agreement of the
    detected block it overlaps most, because that is the answer to "why is
    this being skipped".

    Evidence is a disposable cache, so every key is read with `.get`: with
    no `evidence["audio"]` there is no waveform and no speech lane, and with
    no `evidence["ranges"]` no block names -- the keep and skip spans, and
    every edit, still work from the stored ranges and the duration.

    A stored range the timeline cannot place (unparseable times, or an end
    before its start) is never redrawn as something else: `read_ranges`
    leaves it out and hands back the text it is stored as, which the page
    names in a warn line, because `ocr_kwargs.ocr_call_for` still hands OCR
    exactly what is stored and the screen must not say otherwise. An edit
    writes the ranges that are drawn, so it is also what drops such a range
    -- deliberately, after the line has said what is wrong with it.

Editing (edit mode only)
    A grip drag previews live and commits once on release, through
    `controller.set_time_ranges` (MANUAL), snapped to whole seconds and
    clamped between its neighbouring boundaries. A range ending at the
    duration is stored with an open end (None) and one starting at 0 with an
    open start, so the stored value says "to the end of the file" rather
    than pinning a duration the metadata may still refine.

    A right-click on the track (`context_menu`) deletes the keep range under
    it, or adds a range in the skipped block under it (`with_range_at`),
    through the same commit. A drag never removes a range: a boundary stops
    MIN_SPAN short of its neighbour.

Ruling C3: no hint offer for time ranges -- an intro's length is the
folder's business (the detector already shares fingerprints across it), not
something one episode hints to the others.

Views import no `core` module (tests/ui/test_main_window.py), so
`core.detect.audio_profile.speech_in_skips` is reached through
`app.masking`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMenu, QSizePolicy, QVBoxLayout, QWidget

from app.masking import speech_in_skips
from app.state_text import format_duration
from app.theme import tokens
from app.views.inspector_sections import Section, note_label, small_button
from app.widgets.base import Button, KvRow, repolish

# --- copy ------------------------------------------------------------------

NOTE_TEXT = ("Detected blocks come from audio fingerprints shared across the folder; the speech "
             "lane is computed per episode, so a post-credits scene can't be silently dropped.")
WARNING_TEXT = "⚠ Speech at {start}–{end} falls inside the {kind} block"
UNREADABLE_TEXT = "⚠ A stored time range could not be read ({values}) — fix it or use whole file"
NO_DURATION_TEXT = "Duration not scanned yet — the timeline is drawn once the file has been read."
EXTEND_TEXT = "extend keep →"
ADD_TEXT = "+ add range"
WHOLE_FILE_TEXT = "use whole file"
REMOVE_TEXT = "✕"
DELETE_RANGE_TEXT = "Delete range"       # the track's right-click menu, on a keep span ...
ADD_HERE_TEXT = "Add range here"         # ... and on a skip span
EDIT_NOTE = "Drag an edge to move it. Right-click a range to delete it, or a skipped block to add one there."
KEEP_KEY = "Keep"
WHOLE_FILE_VALUE = "whole file"
LANE_LABEL = "speech"
SKIP_LABEL = "SKIP"                   # a skip span no detected block explains ...
SKIP_KIND = "skipped"                 # ... and how its warning sentence names it

# --- geometry (the literal CSS of tabs-hifi.html figure 3) -----------------
#
# Every length here is a mockup pixel put through `tokens.px`, so the track,
# its grips, its block labels and the speech lane grow with the rest of the
# window (app/theme/tokens.py's UI_SCALE). The comments name the mockup's own
# value, which is what `tokens.px(n)` is called with -- at UI_SCALE 1.0 every
# constant below is exactly the number in its comment.
#
# Nothing under "editing" is a length: those are seconds, and the time <-> x
# maths in `x_for` / `time_at` is a fraction of the widget's own width. The
# scale changes what is DRAWN and how big a hit target is, never what a drag
# stores.

TRACK_HEIGHT = tokens.px(52)          # .track
LANE_HEIGHT = tokens.px(16)           # .lane, fused under it
TIMELINE_HEIGHT = TRACK_HEIGHT + LANE_HEIGHT
# The stage hands this page about 760 px at 1440x900 for ~180 px of content.
# The track is the thing being edited -- blocks to read, boundaries to drag,
# a waveform to aim at -- so it takes what it can use of that before the
# page centres what is still over. Three times the mockup's own height is
# where a wider-than-tall strip stops reading as one; the lane under it is a
# 10 px speech bar and never grows. Compact mode (ruling B5) stays fixed:
# it is a strip under another tab's stage, not the subject of the page.
TRACK_MAX_HEIGHT = 3 * TRACK_HEIGHT
TIMELINE_MAX_HEIGHT = TRACK_MAX_HEIGHT + LANE_HEIGHT
PAGE_MARGIN = tokens.px(12)           # the page's own padding, all four sides
ROWS_GAP = tokens.px(10)              # between the timeline and the warning rows
ROW_SPACING = tokens.px(5)            # between two warning rows, and two panel rows
GRIP_WIDTH = tokens.px(5)             # .grip
GRIP_ALPHA = 0.85
GRIP_GRAB = tokens.px(7)              # how far from a grip a press still takes it
POSITION_WIDTH = tokens.px(1)         # compact mode's "you clicked here" hairline
WAVE_ALPHA = 0.45                     # .wave
WAVE_AMPLITUDE = tokens.px(16)        # the mockup's polyline swings 26 ± 16, at its loudest
WAVE_STEP = tokens.px(8)              # ... every 8 units of its 600-wide viewBox
SPEECH_TOP = tokens.px(3)             # .speech: top:3px; height:10px
SPEECH_HEIGHT = tokens.px(10)
SPEECH_ALPHA = 0.55
WARN_ALPHA = 0.9
KEEP_FILL_ALPHA = 0.10                # .blk.keep  rgba(111,212,138,.10)
KEEP_EDGE_ALPHA = 0.55
SKIP_FILL_ALPHA = 0.07                # .blk.skip  the fainter half of the stripe gradient
SKIP_STRIPE_ALPHA = 0.16              # ... and the stronger one
SKIP_EDGE_ALPHA = 0.6
STRIPE_WIDTH = tokens.px(6)           # repeating-linear-gradient(45deg, ... 0 6px, ... 6px 12px)
STRIPE_PERIOD = tokens.px(12)
LABEL_PADDING_X = tokens.px(5)        # .blk padding: 3px 5px
LABEL_PADDING_Y = tokens.px(3)
LABEL_GAP = tokens.px(6)              # between a keep's label and its inline length
LABEL_MIN_WIDTH = tokens.px(26)       # a block narrower than this is left unlabelled
SUB_LINE_GAP = tokens.px(2)           # the sub-line's margin-top
LANE_LABEL_INSET = tokens.px(4)       # "speech", off the lane's left edge
MARK_ALPHA = 0.75                     # a crop sample's tick, compact mode ...
MARK_HEIGHT = tokens.px(6)            # ... 6 px along the bottom, as workbench-hifi's `.marks s`
MARK_WIDTH = tokens.px(1)
WARN_ROW_PADDING_X = tokens.px(8)     # the warn-tinted .kv around the sentence
WARN_ROW_PADDING_Y = tokens.px(5)
WARN_ROW_GAP = tokens.px(8)           # ... between it and "extend keep →"
BUTTON_GAP = tokens.px(6)             # "+ add range" / "use whole file"

# --- editing ---------------------------------------------------------------

SNAP_SECONDS = 1                      # every boundary lands on a whole second
MIN_SPAN = 1.0                        # and no two boundaries land on the same one
ADD_RANGE_SECONDS = 60.0              # "+ add range" adds a minute


def _alpha(colour: str, alpha: float) -> QColor:
    value = QColor(colour)
    value.setAlphaF(alpha)
    return value


def _font(size: float) -> QFont:
    font = QFont()
    font.setFamilies(tokens.FONT_STACK)
    font.setPointSizeF(size * 0.75)          # px -> pt, as the other views do
    return font


# --------------------------------------------------------------------------
# Times
# --------------------------------------------------------------------------

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


def complement(keeps, duration: float) -> list[tuple[float, float]]:
    """The spans [0, duration] leaves out of `keeps` -- what a run skips."""
    gaps, cursor = [], 0.0
    for start, end in sorted(keeps):
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration > cursor:
        gaps.append((cursor, duration))
    return gaps


def store_ranges(keeps, duration: float) -> list[tuple[str | None, str | None]] | None:
    """`keeps` as `controller.set_time_ranges` stores them: "M:SS" /
    "H:MM:SS" strings, an open start for a range beginning at 0, an open end
    for one reaching the duration, and None for the whole file."""
    ranges = [(None if start <= 0 else format_duration(start),
               None if _reaches_end(end, duration) else format_duration(end))
              for start, end in sorted(keeps)]
    return None if not ranges or ranges == [(None, None)] else ranges


def _reaches_end(end: float, duration: float) -> bool:
    """Whether a keep range runs to the end of the file.

    A duration is frames / fps and is stored fractional, while every boundary
    the user can drag is a whole second, so `end >= duration` alone is
    unreachable on most files: a range that reads as ending at the same clock
    as the file does end there, as far as anything the user or the OCR run
    can see, and is stored as the open end."""
    return end >= duration > 0 or (duration > 0 and format_duration(end) == format_duration(duration))


def with_extended_keep(keeps, span) -> list[tuple[float, float]]:
    """`keeps` with the range nearest `span` grown outward -- to whole
    seconds -- to cover it. The result is merged, so an extension that runs
    into the next range becomes one range."""
    start, end = float(span[0]), float(span[1])
    best: tuple[float, int, tuple[float, float]] | None = None
    for index, (keep_start, keep_end) in enumerate(keeps):
        if keep_end <= start:
            candidate = (start - keep_end, index, (keep_start, math.ceil(end)))
        elif keep_start >= end:
            candidate = (keep_start - end, index, (math.floor(start), keep_end))
        else:
            continue                        # it already overlaps: nothing to extend
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return sorted([*keeps, (math.floor(start), math.ceil(end))])
    _distance, index, grown = best
    return _merged([*keeps[:index], grown, *keeps[index + 1:]])


def with_added_range(keeps, duration: float) -> list[tuple[float, float]]:
    """`keeps` plus a one-minute range in the middle of the largest skipped
    gap -- or the first minute of the file when the whole file is kept, as
    there is no gap to put it in. Centred rather than flush against the gap's
    edge so the new range reads as its own block, with two grips of its own.

    The new range stays inside its gap: rounding its start down to a whole
    second could otherwise put it inside the keep before it, and a gap
    narrower than MIN_SPAN cannot hold a range at all -- `keeps` comes back
    untouched rather than growing a zero-length one."""
    gaps = [gap for gap in (complement(keeps, duration) if keeps else [(0.0, duration)])
            if gap[1] - gap[0] >= MIN_SPAN]
    if not gaps:
        return list(keeps)
    gap_start, gap_end = max(gaps, key=lambda gap: gap[1] - gap[0])
    if not keeps:
        gap_end = min(gap_end, gap_start + ADD_RANGE_SECONDS)     # "at the start"
    length = min(ADD_RANGE_SECONDS, gap_end - gap_start)
    start = max(math.floor(gap_start + (gap_end - gap_start - length) / 2), math.ceil(gap_start))
    end = min(start + length, gap_end)
    if end - start < MIN_SPAN:
        return list(keeps)
    return sorted([*keeps, (float(start), float(end))])


def with_range_at(keeps, time: float, duration: float) -> list[tuple[float, float]]:
    """`keeps` plus a keep of up to ADD_RANGE_SECONDS centred on `time`, the
    track's "Add range here". It is trimmed to the skipped gap `time` falls
    in and kept MIN_SPAN clear of the keeps either side -- touching one would
    merge into it, leaving no block and no grips of its own -- but reaches
    the file's start or end when the gap does, so it is stored open there.
    `keeps` comes back untouched when `time` is not in a gap, or the gap
    cannot hold MIN_SPAN."""
    gap = next(((start, end) for start, end in complement(keeps, duration) if start <= time <= end), None)
    if gap is None:
        return list(keeps)
    gap_start, gap_end = gap
    low = math.ceil(gap_start) + MIN_SPAN if gap_start > 0 else 0.0
    high = math.floor(gap_end) - MIN_SPAN if gap_end < duration else duration
    first = round(float(time) - ADD_RANGE_SECONDS / 2)
    start, end = max(low, first), min(high, first + ADD_RANGE_SECONDS)
    if end - start < MIN_SPAN:
        return list(keeps)
    return sorted([*keeps, (float(start), float(end))])


def _merged(keeps) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(keeps):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def entry_of(controller, name: str | None):
    """The selected file's entry, or None when nothing is selected or the
    folder no longer holds it (the queue picks another file next)."""
    if name is None or name not in controller.names():
        return None
    return controller.entry(name)


def header_text(name: str, duration: float, others) -> str:
    """"ep01.mkv · 27:08 · other episodes 23:38 – 27:08" (ui-spec §3.3): the
    stage head's dim line for this tab, spaced around the dash as
    `tabs-hifi` renders it. Parts nothing is known about are left out rather
    than shown as "—"."""
    parts = [name]
    if duration > 0:
        parts.append(format_duration(duration))
    known = sorted(float(value) for value in others if value and float(value) > 0)
    if known:
        low, high = format_duration(known[0]), format_duration(known[-1])
        parts.append(f"other episodes {low}" if low == high else f"other episodes {low} – {high}")
    return " · ".join(parts)


# --------------------------------------------------------------------------
# The timeline
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Span:
    """One block on the track: a keep range or one of the gaps between
    them. `label` is what the block is titled with ("KEEP · 2:33–23:05",
    "OUTRO · 23:05–27:08") and `detail` its second line -- the OCR length
    for a keep, how many episodes agreed for a matched skip."""

    start: float
    end: float
    keep: bool
    label: str
    detail: str
    kind: str = ""                   # a skip span's block kind ("intro", "outro", "repeat", ...)


@dataclass(frozen=True)
class SpeechWarning:
    """Speech the run would skip: the overlap itself, and the block it falls
    in."""

    start: float
    end: float
    kind: str

    @property
    def text(self) -> str:
        return WARNING_TEXT.format(start=format_duration(self.start),
                                   end=format_duration(self.end), kind=self.kind)


class Timeline(QWidget):
    """The 52 px track and its 16 px speech lane -- TRACK_HEIGHT and
    LANE_HEIGHT at the current UI scale -- editable (`mode="edit"`) or
    read-only and compact (`mode="compact"`, ruling B5).

    It draws the file and, in edit mode, edits it; it decides nothing about
    the tab it is mounted in. A compact click leaves as `seek_requested`,
    and `position()` remembers it (the marker drawn at it)."""

    seek_requested = pyqtSignal(float)        # compact click: "the user pointed at t"
    committed = pyqtSignal()                  # an edit reached the controller

    def __init__(self, controller, mode: str = "edit", parent: QWidget | None = None):
        super().__init__(parent)
        if mode not in ("edit", "compact"):
            raise ValueError(f"mode is 'edit' or 'compact', not {mode!r}")
        self.mode = mode
        self._controller = controller
        self._file: str | None = None
        self._duration = 0.0
        self._keeps: list[tuple[float, float]] = []      # stored; [] = the whole file
        self._unreadable: list[str] = []                 # ... and the stored ranges it could not place
        self._bounds: list[float] = []                   # the drawn boundaries, live during a drag
        self._blocks: list[dict] = []
        self._envelope: list[float] = []
        self._speech: list[tuple[float, float]] = []
        self._marks: list[float] = []
        self._mark_override: list[float] | None = None   # set_marks; None: the crop samples
        self._drag: int | None = None
        self._position: float | None = None
        self._warnings: list[SpeechWarning] = []
        self._warnings_key: tuple | None = None
        self.setObjectName("Timeline")
        if self.editable:
            self.setMinimumHeight(TIMELINE_HEIGHT)
            self.setMaximumHeight(TIMELINE_MAX_HEIGHT)
            self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        else:
            self.setFixedHeight(TIMELINE_HEIGHT)
        self.setMouseTracking(self.editable)
        if not self.editable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    # --- state ------------------------------------------------------------

    @property
    def editable(self) -> bool:
        return self.mode == "edit"

    def set_file(self, name: str | None) -> None:
        self._file = name
        self._drag = None
        self._position = None
        self.refresh()

    def refresh(self) -> None:
        entry = self._entry()
        evidence = (entry.evidence if entry is not None else {}) or {}
        ranges = evidence.get("ranges") or {}
        audio = evidence.get("audio") or {}
        crop = evidence.get("crop") or {}
        self._duration = self._read_duration(entry, ranges)
        self._keeps, self._unreadable = read_ranges(entry, self._duration)
        if self._drag is None:                      # a drag owns the boundaries until it ends
            self._bounds = [value for span in self._drawn_keeps() for value in span]
        self._blocks = [block for block in (ranges.get("blocks") or []) if _block_span(block)]
        self._envelope = [float(value) for value in (audio.get("envelope") or [])]
        self._speech = [(float(start), float(end)) for start, end in (audio.get("speech") or [])]
        self._marks = _sample_times(crop) if self._mark_override is None else list(self._mark_override)
        self.update()

    def set_marks(self, times) -> None:
        """Tick `times` (seconds) instead of the crop samples; None goes back
        to the crop samples. Kept across refreshes and files: the tab that
        set it keeps it current."""
        self._mark_override = None if times is None else sorted({float(time) for time in times})
        if self._mark_override is None:
            entry = self._entry()
            evidence = (entry.evidence if entry is not None else {}) or {}
            self._marks = _sample_times(evidence.get("crop") or {})
        else:
            self._marks = list(self._mark_override)
        self.update()

    def duration(self) -> float:
        return self._duration

    def keeps(self) -> list[tuple[float, float]]:
        """The stored keep spans; empty means the whole file is kept --
        unless `unreadable()` is not, in which case something is stored that
        could not be placed."""
        return list(self._keeps)

    def unreadable(self) -> list[str]:
        """The stored ranges the timeline could not place, as they are
        stored ("10:00 → 2:00"). See `read_ranges`."""
        return list(self._unreadable)

    def envelope(self) -> list[float]:
        return list(self._envelope)

    def speech_segments(self) -> list[tuple[float, float]]:
        return list(self._speech)

    def sample_marks(self) -> list[float]:
        """The times ticked on the track, in order (compact mode draws them)."""
        return list(self._marks)

    def lane_label(self) -> str:
        """"speech" -- but not in compact mode, which carries no label but a
        block's kind (ruling B5)."""
        return LANE_LABEL if self.editable else ""

    def position(self) -> float | None:
        """Where the timeline was last clicked, in seconds, or None."""
        return self._position

    def grips(self) -> list[float]:
        """The boundary times a grip sits on -- none in compact mode, which
        is strictly read-only (ruling B5)."""
        return list(self._bounds) if self.editable else []

    def spans(self) -> list[Span]:
        """The track's blocks in time order: the keep ranges and the gaps
        between them, each with the copy it is labelled with."""
        if self._duration <= 0:
            return []
        keeps = self._drag_keeps()
        spans = [Span(start, end, True, self._keep_label(start, end),
                      f"{format_duration(end - start)} of OCR")
                 for start, end in keeps]
        for start, end in complement(keeps, self._duration):
            block = self._block_for(start, end)
            kind = str(block.get("kind", "")) if block else ""
            spans.append(Span(start, end, False,
                              f"{(kind or SKIP_LABEL).upper()} · {_span_times(start, end)}",
                              _block_detail(block), kind))
        return sorted(spans, key=lambda span: span.start)

    def warnings(self) -> list[SpeechWarning]:
        """One per speech span the run would skip (`speech_in_skips`), each
        clipped to the skipped span it falls in.

        Cached on the state it was computed from: `paintEvent` asks for it on
        every repaint (the lane colours those spans) and the tab asks again
        for its warning rows, and the answer only changes when the speech or
        the skipped spans do."""
        skips = tuple(complement(self._drag_keeps(), self._duration))
        key = (tuple(self._speech), skips)
        if self._warnings_key != key:
            self._warnings_key = key
            self._warnings = ([] if not skips or not self._speech else
                              [SpeechWarning(start, end, self._kind_at(start, end))
                               for start, end in speech_in_skips(self._speech, skips)])
        return list(self._warnings)

    def warn_spans(self) -> list[tuple[float, float]]:
        return [(warning.start, warning.end) for warning in self.warnings()]

    # --- geometry ---------------------------------------------------------

    def x_for(self, time: float) -> float:
        """Seconds -> pixels across the full width. Out-of-range clamps, so a
        drag past either edge lands on the edge."""
        if self._duration <= 0:
            return 0.0
        return max(0.0, min(1.0, float(time) / self._duration)) * self.width()

    def time_at(self, x: float) -> float:
        if self._duration <= 0 or self.width() <= 0:
            return 0.0
        return max(0.0, min(1.0, float(x) / self.width())) * self._duration

    # --- editing ----------------------------------------------------------

    def _drawn_keeps(self) -> list[tuple[float, float]]:
        """What the track shows: the stored ranges, or the whole file as one
        keep span when nothing is stored."""
        return self._keeps or ([(0.0, self._duration)] if self._duration > 0 else [])

    def _drag_keeps(self) -> list[tuple[float, float]]:
        """The keep spans as drawn right now -- the live ones during a drag."""
        bounds = self._bounds
        return [(bounds[index], bounds[index + 1]) for index in range(0, len(bounds) - 1, 2)]

    def _grip_at(self, x: float) -> int | None:
        near = [(abs(self.x_for(time) - x), index) for index, time in enumerate(self._bounds)]
        if not near:
            return None
        distance, index = min(near)
        return index if distance <= GRIP_GRAB else None

    def _clamped(self, index: int, seconds: float) -> float:
        """A boundary snapped to a whole second and kept inside its
        neighbours and [0, duration], never closer than MIN_SPAN to either --
        a zero-length range would be a range the user cannot see or grab.

        The file's own end is a landing point of its own: a duration is
        frames / fps and lands between whole seconds, so snapping alone could
        never reach it and "to the end of the file" would be undraggable --
        the last boundary would stop a fraction of a second short, leaving a
        skip sliver too narrow to grab and an OCR run that ends early."""
        last = index + 1 >= len(self._bounds)
        low = self._bounds[index - 1] + MIN_SPAN if index > 0 else 0.0
        high = self._duration if last else self._bounds[index + 1] - MIN_SPAN
        snapped = round(float(seconds) / SNAP_SECONDS) * SNAP_SECONDS
        value = max(low, min(high, snapped))
        if last and self._duration - value < SNAP_SECONDS:
            value = self._duration
        return float(value)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        x = event.position().x()
        if not self.editable:
            self._position = self.time_at(x)
            self.update()
            self.seek_requested.emit(self._position)
            return
        index = self._grip_at(x)
        if index is None:
            return
        self._drag = index          # taking the grip moves nothing: see mouseReleaseEvent

    def mouseMoveEvent(self, event) -> None:
        if not self.editable:
            return
        x = event.position().x()
        if self._drag is None:
            over = self._grip_at(x) is not None
            self.setCursor(Qt.CursorShape.SizeHorCursor if over else Qt.CursorShape.ArrowCursor)
            return
        self._bounds[self._drag] = self._clamped(self._drag, self.time_at(x))
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        """A gesture that moved a boundary commits; one that did not commits
        nothing at all. Writing the same times back would still be a MANUAL
        write -- it would freeze a detected value against every later
        detection, which is not what a stray click on a grip asks for."""
        if self._drag is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag = None
        keeps = self._drag_keeps()
        if keeps != self._drawn_keeps():
            self.commit(keeps)

    def context_menu(self, x: float) -> QMenu | None:
        """The right-click menu at `x`, built fresh so it reads what is
        stored now; the caller shows it (or triggers its actions).

        On a keep span, "Delete range" -- what that range's ✕ in the panel
        does, so the last one leaves the whole file. On a skip span, "Add
        range here" (`with_range_at`), disabled when the gap is too narrow
        to hold one. None where there is nothing to offer: compact mode
        (strictly read-only, ruling B5), mid-drag, no duration yet, or the
        whole file kept with nothing stored to delete and no skip to add to."""
        if not self.editable or self._drag is not None or self._duration <= 0 or not self._keeps:
            return None
        time = self.time_at(x)
        keeps = list(self._keeps)
        menu = QMenu(self)
        index = next((i for i, (start, end) in enumerate(keeps) if start <= time <= end), None)
        if index is not None:
            action = menu.addAction(DELETE_RANGE_TEXT)
            action.triggered.connect(lambda _checked=False: self.commit([*keeps[:index], *keeps[index + 1:]]))
        else:
            added = with_range_at(keeps, time, self._duration)
            action = menu.addAction(ADD_HERE_TEXT)
            action.setEnabled(added != keeps)
            action.triggered.connect(lambda _checked=False: self.commit(added))
        return menu

    def contextMenuEvent(self, event) -> None:
        menu = self.context_menu(event.pos().x())
        if menu is None:
            super().contextMenuEvent(event)
            return
        menu.exec(event.globalPos())
        menu.deleteLater()

    def commit(self, keeps) -> None:
        """Store `keeps` as this file's time ranges (MANUAL) and redraw from
        what was stored."""
        if self._file is not None:
            self._controller.set_time_ranges(self._file, store_ranges(keeps, self._duration))
        self.refresh()
        self.committed.emit()

    # --- reading the model ------------------------------------------------

    def _entry(self):
        return entry_of(self._controller, self._file)

    @staticmethod
    def _read_duration(entry, ranges: dict) -> float:
        """The file's own duration, else the one the ranges analysis
        measured (the metadata job may not have run yet)."""
        duration = float(entry.media.duration) if entry is not None else 0.0
        if duration > 0:
            return duration
        try:
            return max(0.0, float(ranges.get("duration") or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _keep_label(self, start: float, end: float) -> str:
        return f"KEEP · {_span_times(start, end)}"

    def _block_for(self, start: float, end: float) -> dict | None:
        """The detected block a skip span overlaps most -- the one that
        explains why it is skipped. None when no block reaches into it."""
        best, best_overlap = None, 0.0
        for block in self._blocks:
            block_start, block_end = _block_span(block)
            overlap = min(end, block_end) - max(start, block_start)
            if overlap > best_overlap:
                best, best_overlap = block, overlap
        return best

    def _kind_at(self, start: float, end: float) -> str:
        block = self._block_for(start, end)
        return str(block.get("kind") or SKIP_KIND) if block else SKIP_KIND

    # --- painting ---------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = float(self.width())
        # The lane keeps its 16 px; whatever the widget has over the mockup's
        # own height goes to the track (see TRACK_MAX_HEIGHT).
        track_height = max(float(TRACK_HEIGHT), self.height() - LANE_HEIGHT)
        track = QRectF(0.5, 0.5, max(0.0, width - 1), track_height - 1)
        lane = QRectF(0.5, track_height + 0.5, max(0.0, width - 1), LANE_HEIGHT - 1)
        self._paint_track(painter, track)
        self._paint_lane(painter, lane)
        painter.end()

    def _paint_track(self, painter: QPainter, rect: QRectF) -> None:
        path = _rounded(rect, top=True, bottom=False)
        painter.save()
        painter.fillPath(path, QColor(tokens.PANEL))
        painter.setClipPath(path)
        self._paint_wave(painter, rect)
        for span in self.spans():
            self._paint_span(painter, rect, span)
        if not self.editable:
            for time in self._marks:
                self._paint_mark(painter, rect, time)
        if self.editable:
            for time in self._bounds:
                x = self.x_for(time)
                painter.fillRect(QRectF(x - GRIP_WIDTH / 2, rect.top(), GRIP_WIDTH, rect.height()),
                                 _alpha(tokens.ACC, GRIP_ALPHA))
        elif self._position is not None:
            x = self.x_for(self._position)
            painter.fillRect(QRectF(x - POSITION_WIDTH / 2, rect.top(),
                                    POSITION_WIDTH, rect.height()),
                             _alpha(tokens.ACC, GRIP_ALPHA))
        painter.restore()
        painter.setPen(QPen(QColor(tokens.LINE), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _paint_wave(self, painter: QPainter, rect: QRectF) -> None:
        """The envelope as the mockup's polyline: a zigzag about the middle
        whose swing is that bin's RMS.

        Drawn at about one point every WAVE_STEP px (the mockup's spacing),
        each the loudest bin it stands for -- 600 bins across a 900 px track
        would otherwise be a solid mesh rather than a waveform, and the peaks
        are what the eye is looking for."""
        values = _resampled(self._envelope, max(2, int(rect.width() / WAVE_STEP)))
        if len(values) < 2:
            return
        middle = rect.center().y()
        step = rect.width() / (len(values) - 1)
        # The mockup's swing, kept in proportion when the track is taller.
        amplitude = WAVE_AMPLITUDE * rect.height() / (TRACK_HEIGHT - 1)
        points = [QPointF(rect.left() + index * step,
                          middle + (1 if index % 2 else -1) * value * amplitude)
                  for index, value in enumerate(values)]
        painter.save()
        painter.setOpacity(WAVE_ALPHA)
        painter.setPen(QPen(QColor(tokens.WAVEFORM), 1))
        painter.drawPolyline(*points)
        painter.restore()

    def _paint_span(self, painter: QPainter, rect: QRectF, span: Span) -> None:
        left, right = self.x_for(span.start), self.x_for(span.end)
        block = QRectF(left, rect.top(), max(0.0, right - left), rect.height())
        colour = tokens.OK if span.keep else tokens.BAD
        painter.save()
        painter.setClipRect(block, Qt.ClipOperation.IntersectClip)
        painter.fillRect(block, _alpha(colour, KEEP_FILL_ALPHA if span.keep else SKIP_FILL_ALPHA))
        if not span.keep:
            self._paint_stripes(painter, block)
        edge = _alpha(colour, KEEP_EDGE_ALPHA if span.keep else SKIP_EDGE_ALPHA)
        painter.setPen(QPen(edge, 1))
        painter.drawLine(QPointF(block.left() + 0.5, block.top()),
                         QPointF(block.left() + 0.5, block.bottom()))
        painter.drawLine(QPointF(block.right() - 0.5, block.top()),
                         QPointF(block.right() - 0.5, block.bottom()))
        self._paint_label(painter, block, span, colour)
        painter.restore()

    @staticmethod
    def _paint_stripes(painter: QPainter, block: QRectF) -> None:
        """`repeating-linear-gradient(45deg, …16% 0 6px, …7% 6px 12px)`: the
        fill is the fainter half, these are the stronger 6 px bands."""
        painter.save()
        painter.setPen(QPen(_alpha(tokens.BAD, SKIP_STRIPE_ALPHA), STRIPE_WIDTH))
        height = block.height()
        spacing = STRIPE_PERIOD * math.sqrt(2)         # perpendicular 12 px at 45 degrees
        x = block.left() - height
        while x < block.right():
            painter.drawLine(QPointF(x, block.bottom()), QPointF(x + height, block.top()))
            x += spacing
        painter.restore()

    def _paint_label(self, painter: QPainter, block: QRectF, span: Span, colour: str) -> None:
        """The block's title, and its second line. Compact mode shows the
        kind alone (ruling B5: "no labels except kind abbreviations")."""
        text = span.label if self.editable else ("" if span.keep else span.label.split(" · ")[0])
        if not text:
            return
        painter.setFont(_font(tokens.FONT_SIZE_XS))
        metrics = painter.fontMetrics()
        inner = block.adjusted(LABEL_PADDING_X, LABEL_PADDING_Y, -LABEL_PADDING_X, 0)
        if inner.width() < LABEL_MIN_WIDTH:
            return                                     # too narrow to read: leave it clean
        line = QRectF(inner.left(), inner.top(), inner.width(), metrics.height())
        flags = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        painter.setPen(QColor(colour))
        painter.drawText(line, flags, _elided(metrics, text, line.width()))
        if not self.editable or not span.detail:
            return
        painter.setPen(QColor(tokens.DIM2))
        if span.keep:                                  # the keep's length sits on the same line
            offset = metrics.horizontalAdvance(text) + LABEL_GAP
            detail = line.adjusted(offset, 0, 0, 0)
            painter.drawText(detail, flags, _elided(metrics, span.detail, detail.width()))
            return
        sub_line = line.translated(0, metrics.height() + SUB_LINE_GAP)
        painter.drawText(sub_line, flags, _elided(metrics, span.detail, sub_line.width()))

    def _paint_mark(self, painter: QPainter, rect: QRectF, time: float) -> None:
        """A sample's tick: 6 px along the bottom of the track, so it reads
        as a mark rather than as a grip it cannot be."""
        painter.fillRect(QRectF(self.x_for(time) - MARK_WIDTH / 2, rect.bottom() - MARK_HEIGHT,
                                MARK_WIDTH, MARK_HEIGHT),
                         _alpha(tokens.ACC, MARK_ALPHA))

    def _paint_lane(self, painter: QPainter, rect: QRectF) -> None:
        path = _rounded(rect, top=False, bottom=True)
        painter.save()
        painter.fillPath(path, QColor(tokens.PANEL))
        painter.setClipPath(path)
        for start, end in self._speech:
            self._paint_speech(painter, rect, start, end, _alpha(tokens.BLUE, SPEECH_ALPHA))
        for start, end in self.warn_spans():
            self._paint_speech(painter, rect, start, end, _alpha(tokens.WARN, WARN_ALPHA))
        if self.lane_label():
            painter.setFont(_font(tokens.FONT_SIZE_XS))
            painter.setPen(QColor(tokens.DIM2))
            painter.drawText(rect.adjusted(LANE_LABEL_INSET, 0, 0, 0),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             self.lane_label())
        painter.restore()
        painter.setPen(QPen(QColor(tokens.LINE), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _paint_speech(self, painter: QPainter, rect: QRectF,
                      start: float, end: float, colour: QColor) -> None:
        left, right = self.x_for(start), self.x_for(end)
        segment = QRectF(left, rect.top() + SPEECH_TOP - 0.5, max(1.0, right - left), SPEECH_HEIGHT)
        path = QPainterPath()
        path.addRoundedRect(segment, tokens.RADIUS_THUMB_BOX, tokens.RADIUS_THUMB_BOX)
        painter.fillPath(path, colour)


def _rounded(rect: QRectF, *, top: bool, bottom: bool) -> QPainterPath:
    """`rect` with `tokens.RADIUS_SEG` corners on the ends that ask for them
    -- the track rounds its top, the lane fused under it its bottom."""
    radius = float(tokens.RADIUS_SEG)
    path = QPainterPath()
    path.moveTo(rect.left(), rect.top() + (radius if top else 0))
    if top:
        path.arcTo(QRectF(rect.left(), rect.top(), 2 * radius, 2 * radius), 180, -90)
        path.lineTo(rect.right() - radius, rect.top())
        path.arcTo(QRectF(rect.right() - 2 * radius, rect.top(), 2 * radius, 2 * radius), 90, -90)
    else:
        path.lineTo(rect.left(), rect.top())
        path.lineTo(rect.right(), rect.top())
    if bottom:
        path.lineTo(rect.right(), rect.bottom() - radius)
        path.arcTo(QRectF(rect.right() - 2 * radius, rect.bottom() - 2 * radius,
                          2 * radius, 2 * radius), 0, -90)
        path.lineTo(rect.left() + radius, rect.bottom())
        path.arcTo(QRectF(rect.left(), rect.bottom() - 2 * radius, 2 * radius, 2 * radius),
                   270, -90)
    else:
        path.lineTo(rect.right(), rect.bottom())
        path.lineTo(rect.left(), rect.bottom())
    path.closeSubpath()
    return path


def _elided(metrics, text: str, width: float) -> str:
    """`text` cut to `width` with an ellipsis rather than sliced mid-glyph by
    the block's clip."""
    return metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(width))


def _resampled(values, count: int) -> list[float]:
    """`values` in `count` buckets, each the loudest value it holds."""
    if not values or count <= 0:
        return []
    if len(values) <= count:
        return list(values)
    step = len(values) / count
    return [max(values[int(index * step):max(int((index + 1) * step), int(index * step) + 1)])
            for index in range(count)]


def _span_times(start: float, end: float) -> str:
    return f"{format_duration(start)}–{format_duration(end)}"


def _sample_times(crop: dict) -> list[float]:
    """The crop detector's probe times, in order and without repeats -- the
    ticks compact mode marks. The cache is disposable and may come back
    partial, so anything that is not a time is dropped."""
    times = set()
    for sample in crop.get("samples") or []:
        try:
            times.add(float(sample["time"]))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return sorted(times)


def _block_span(block) -> tuple[float, float] | None:
    try:
        start, end = float(block["start_sec"]), float(block["end_sec"])
    except (KeyError, TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def _block_detail(block: dict | None) -> str:
    """"matches 4 episodes · 98%": how many OTHER episodes carried the same
    segment (`matched_files` counts this one too) and how well it matched."""
    if not block:
        return ""
    try:
        others = int(block["matched_files"]) - 1
        score = round(float(block["score"]) * 100)
    except (KeyError, TypeError, ValueError):
        return ""
    return f"matches {others} episodes · {score}%"


# --------------------------------------------------------------------------
# The warning rows
# --------------------------------------------------------------------------

class WarningRow(QWidget):
    """A warn-tinted `.kv`: the sentence, and "extend keep →"."""

    extend_requested = pyqtSignal()

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("KvRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("tone", "warn")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(WARN_ROW_PADDING_X, WARN_ROW_PADDING_Y,
                                  WARN_ROW_PADDING_X, WARN_ROW_PADDING_Y)
        layout.setSpacing(WARN_ROW_GAP)
        self._label = QLabel(text)
        self._label.setProperty("kvRole", "value")
        self._label.setProperty("tone", "warn")
        self.button = small_button(EXTEND_TEXT)
        self.button.clicked.connect(self.extend_requested)
        layout.addWidget(self._label)
        layout.addStretch(1)
        layout.addWidget(self.button)

    def text(self) -> str:
        return self._label.text()


# --------------------------------------------------------------------------
# The inspector panel
# --------------------------------------------------------------------------

class KeepRow(KvRow):
    """"Keep" / "2:33 → 23:05", with a "✕" that drops the range."""

    remove_requested = pyqtSignal()

    def __init__(self, value: str, removable: bool = True, parent: QWidget | None = None):
        super().__init__(KEEP_KEY, value, parent=parent)
        self.remove_button: Button | None = None
        if removable:
            self.remove_button = small_button(REMOVE_TEXT, "ghost")
            self.remove_button.clicked.connect(self.remove_requested)
            self.layout().addWidget(self.remove_button)


class RangesInspectorPanel(Section):
    """The Time ranges tab's slice of the inspector (ruling B4): one row per
    keep range, "+ add range" and "use whole file".

    No hint offer (ruling C3) and no scope header -- the inspector's own
    header already says "◆ this episode only"."""

    add_requested = pyqtSignal()
    whole_file_requested = pyqtSignal()
    remove_requested = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.body.setSpacing(ROW_SPACING)
        self._rows: list[KeepRow] = []
        self._buttons = QWidget()
        buttons = QHBoxLayout(self._buttons)
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(BUTTON_GAP)
        self.add_button = small_button(ADD_TEXT)
        self.add_button.clicked.connect(self.add_requested)
        self.whole_file_button = small_button(WHOLE_FILE_TEXT, "ghost")
        self.whole_file_button.clicked.connect(self.whole_file_requested)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.whole_file_button)
        buttons.addStretch(1)
        self.body.addWidget(self._buttons)
        self.edit_note = note_label(EDIT_NOTE)       # the track's gestures say nothing on their own
        self.body.addWidget(self.edit_note)
        self.body.addStretch(1)

    def set_ranges(self, keeps, duration: float, *, unreadable: bool = False) -> None:
        """One row per keep range; the whole file reads as one row with
        nothing to remove.

        `unreadable`: the file stores a range the timeline could not place
        (`read_ranges`), so "whole file" would be a lie -- there is something
        stored, and the page's warn line names it."""
        values = [f"{format_duration(start)} → {format_duration(end)}" for start, end in keeps]
        whole_file = duration > 0 and not unreadable
        for row in self._rows:
            self.body.removeWidget(row)
            row.setParent(None)              # a row left parented would paint over the buttons
            row.deleteLater()
        self._rows = []
        for index, value in enumerate(values or ([WHOLE_FILE_VALUE] if whole_file else [])):
            row = KeepRow(value, removable=bool(values))
            row.remove_requested.connect(lambda i=index: self.remove_requested.emit(i))
            self.body.insertWidget(index, row)
            self._rows.append(row)
        self._buttons.setEnabled(duration > 0)

    def keep_rows(self) -> list[tuple[str, str]]:
        return [(KEEP_KEY, row.value()) for row in self._rows]

    def remove_buttons(self) -> list[Button]:
        return [row.remove_button for row in self._rows if row.remove_button is not None]


# --------------------------------------------------------------------------
# The tab
# --------------------------------------------------------------------------

class RangesTab:
    """`StageTab` for "Time ranges": the editable timeline, the speech
    warnings and their panel."""

    title = "Time ranges"

    def __init__(self, controller):
        self._controller = controller
        self._file: str | None = None
        self._warnings: list[SpeechWarning] = []
        self._rows: list[WarningRow] = []

        self._page = QWidget()
        self._page.setObjectName("RangesPage")
        self._page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        column = QVBoxLayout(self._page)
        column.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
        column.setSpacing(0)
        # The track grows into a tall stage first (TRACK_MAX_HEIGHT); these
        # two stretches split whatever is still over, so the page sits in the
        # middle rather than clinging to the top over a field of black.
        column.addStretch(1)
        self.timeline = Timeline(controller, mode="edit")
        self.timeline.committed.connect(self.refresh)
        # A stretch of its own, or the two margins would take every spare
        # pixel and the track would never leave its minimum: QBoxLayout
        # gives a stretch-0 item its size hint whenever anything else has a
        # stretch factor, Expanding policy or not. Its maximum still caps it,
        # and what it cannot take goes back to the margins.
        column.addWidget(self.timeline, 1)
        column.addSpacing(ROWS_GAP)
        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(ROW_SPACING)
        column.addWidget(self._rows_host)
        # Why the timeline is not showing what the file stores: a range it
        # could not place (warn), or a duration nothing has measured yet.
        self.status = note_label("")
        column.addWidget(self.status)
        self.note = note_label(NOTE_TEXT)
        column.addWidget(self.note)
        column.addStretch(1)

        # Ruling B3: the dim header line lives in the stage head, not on the page.
        self._toolbar = QWidget()
        self._toolbar.setObjectName("RangesToolbar")
        bar = QHBoxLayout(self._toolbar)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(0)
        self._header = QLabel("")
        self._header.setObjectName("Note")
        bar.addWidget(self._header)

        self.panel = RangesInspectorPanel()
        self.panel.add_requested.connect(self.add_range)
        self.panel.whole_file_requested.connect(self.use_whole_file)
        self.panel.remove_requested.connect(self.remove_range)

    # --- StageTab ---------------------------------------------------------

    def page(self) -> QWidget:
        return self._page

    def inspector_panel(self) -> QWidget:
        return self.panel

    def toolbar(self) -> QWidget:
        """"{name} · {duration} · other episodes {min} – {max}" (ui-spec §3.3),
        which the Stage mounts in the stage head."""
        return self._toolbar

    def set_file(self, name: str | None) -> None:
        self._file = name
        self.timeline.set_file(name)
        self.refresh()

    def refresh(self) -> None:
        entry = self._entry()
        self.timeline.refresh()
        duration = self.timeline.duration()
        unreadable = self.timeline.unreadable()
        self._page.setEnabled(entry is not None)
        self._header.setText("" if entry is None else header_text(
            self._file, duration, self._other_durations()))
        self._sync_warnings()
        self._sync_status(entry, duration, unreadable)
        self.panel.set_ranges(self.timeline.keeps(), duration, unreadable=bool(unreadable))

    # --- reading ----------------------------------------------------------

    def current_file(self) -> str | None:
        return self._file

    def header_text(self) -> str:
        return self._header.text()

    def warning_rows(self) -> list[WarningRow]:
        return list(self._rows)

    def warning_texts(self) -> list[str]:
        return [row.text() for row in self._rows]

    def extend_buttons(self) -> list[Button]:
        return [row.button for row in self._rows]

    def status_text(self) -> str:
        """The line under the warnings when the timeline is not showing what
        the file stores, or "" when it is."""
        return self.status.text() if not self.status.isHidden() else ""

    def status_tone(self) -> str:
        return self.status.property("tone") or ""

    # --- commands ---------------------------------------------------------

    def extend_keep(self, index: int) -> None:
        """Grow the keep range next to the `index`-th warning until it covers
        the speech (ui-spec §3.6's "extend keep →")."""
        if 0 <= index < len(self._warnings):
            warning = self._warnings[index]
            self.timeline.commit(with_extended_keep(self.timeline.keeps() or self._whole_file(),
                                                    (warning.start, warning.end)))

    def add_range(self) -> None:
        self.timeline.commit(with_added_range(self.timeline.keeps(), self.timeline.duration()))

    def remove_range(self, index: int) -> None:
        keeps = self.timeline.keeps()
        if 0 <= index < len(keeps):
            self.timeline.commit([*keeps[:index], *keeps[index + 1:]])

    def use_whole_file(self) -> None:
        if self._file is not None:
            self._controller.set_time_ranges(self._file, None)
        self.refresh()

    # --- internals --------------------------------------------------------

    def _entry(self):
        return entry_of(self._controller, self._file)

    def _whole_file(self) -> list[tuple[float, float]]:
        return [(0.0, self.timeline.duration())]

    def _other_durations(self) -> list[float]:
        return [self._controller.entry(name).media.duration
                for name in self._controller.names() if name != self._file]

    def _sync_warnings(self) -> None:
        self._warnings = self.timeline.warnings()
        for row in self._rows:
            self._rows_layout.removeWidget(row)
            row.setParent(None)              # removeWidget alone leaves it parented and painting
            row.deleteLater()
        self._rows = []
        for index, warning in enumerate(self._warnings):
            row = WarningRow(warning.text)
            row.extend_requested.connect(lambda i=index: self.extend_keep(i))
            self._rows_layout.addWidget(row)
            self._rows.append(row)
        self._rows_host.setVisible(bool(self._rows))

    def _sync_status(self, entry, duration: float, unreadable: list[str]) -> None:
        """Say why the timeline is not showing what the file stores -- an
        unreadable stored range, or a duration nothing has measured yet
        (a v1-migrated file whose metadata job has not landed reads as "no
        ranges" otherwise). Nothing to say: the line goes."""
        if entry is None:
            text, tone = "", ""
        elif unreadable:
            text, tone = UNREADABLE_TEXT.format(values=", ".join(unreadable)), "warn"
        elif duration <= 0:
            text, tone = NO_DURATION_TEXT, ""
        else:
            text, tone = "", ""
        self.status.setText(text)
        self.status.setProperty("tone", tone)
        repolish(self.status)
        self.status.setVisible(bool(text))
