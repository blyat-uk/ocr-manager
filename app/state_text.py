"""Pure, Qt-free badge and caption text for the review queue and inspector
(ruling B10; caption rules from Task 1's brief). No PyQt6 import belongs
anywhere in this module -- see tests/ui/test_state_text.py's subprocess
import check -- so the window's Qt layer can format review-queue text
without pulling Qt into, say, a headless job runner that also wants it.

Reused from plan 3A rather than re-derived (core/detect/, core/jobs/apply.py
are Qt-free too, see ruling C8):
- `core.detect.crop.INFORMATIONAL_FLAGS` / `core.detect.brightness.
  INFORMATIONAL_FLAGS` and `core.detect.flags.only_informational()`: whether
  a detector's `flagged` string blocks review (evidence stores `flagged`,
  not `auto_applicable` -- see those modules' `CropResult`/`BrightnessResult`
  docstrings).
- `core.jobs.apply.brightness_is_stale()`: whether a stored brightness value
  was measured against a crop the file no longer has (apply.py's own
  "Stale brightness" rule).

`core.jobs.apply._required()` only ever asks for "crop" and "brightness"
(labels-only folders need neither, and time ranges are never required) --
FLAGGED can never actually happen over ranges in the current app. B10's
badge table nonetheless lists "check time ranges" as a badge text, so
`badge_for()` still recognises a blocking `entry.flags["ranges"]` (which
nothing in `core/jobs/apply.py` writes today) for it, rather than silently
dropping a table row plan 3A hasn't wired up yet.
"""
from __future__ import annotations

from collections.abc import Iterable

from core.detect import brightness as _brightness
from core.detect import crop as _crop
from core.detect import ocr_view as _ocr_view
from core.detect.flags import only_informational
from core.jobs.apply import brightness_is_stale
from core.jobs.detect_jobs import proof_window as _proof_window
from core.project.model import FileEntry, ReviewState, Source

# --------------------------------------------------------------------------
# badge_for()
# --------------------------------------------------------------------------

# PENDING badge text by which detector is running, checked in this order --
# core.jobs.autopilot.PRIORITY's own ordering (crop=3, brightness=2,
# ranges=0): the badge names whichever of a file's running detectors autopilot
# would service first.
_PENDING_ORDER = ("crop", "brightness", "ranges")
_PENDING_TEXT = {
    "crop": "finding subtitles…",
    "brightness": "measuring brightness…",
    "ranges": "matching intro/outro…",
}

_FIELD_LABEL = {"crop": "crop", "brightness": "brightness", "ranges": "time ranges"}
_INFORMATIONAL_FLAGS = {
    "crop": _crop.INFORMATIONAL_FLAGS,
    "brightness": _brightness.INFORMATIONAL_FLAGS,
    "ranges": frozenset(),  # core/jobs/apply.py never writes entry.flags["ranges"] today
}


def _flag_blocks(flag: str | None, informational) -> bool:
    return bool(flag) and not only_informational(flag, informational)


def _is_detected_value(value) -> bool:
    """Mirrors core.jobs.apply._is_detected(): only a DETECTED/HINT value
    counts as "currently detected" for blocking-flag purposes."""
    return value is not None and value.source in (Source.DETECTED, Source.HINT)


def _flag_blocks_detected_value(entry: FileEntry, name: str, value) -> bool:
    """Mirrors core.jobs.apply's `_is_detected(value) and _blocking(entry,
    name)`, the ONLY test apply.py uses to decide a stored flag counts
    against a field. `entry.flags[name]` is never cleared when a flagged
    DETECTED/HINT value is accepted as MANUAL (mark_reviewed keeps the flag
    string -- see its own docstring: "its evidence and flag stay stored"),
    so a leftover blocking flag on a MANUAL/IMPORTED value must NOT count
    here, or a badge would keep warning about a value the user already
    accepted."""
    return _is_detected_value(value) and _flag_blocks(entry.flags.get(name), _INFORMATIONAL_FLAGS[name])


def _blocking_fields(entry: FileEntry) -> list[str]:
    """Which of crop/brightness/time-ranges a FLAGGED badge should name, in
    B10's order -- mirroring core.jobs.apply.compute_review_state()'s two
    independent reasons a required field can need attention:
    - missing (`_counts_as_missing`): crop is None; brightness is None or
      stale (`brightness_is_stale` itself only ever calls a DETECTED/HINT
      brightness stale, so this needs no extra source check);
    - a blocking flag on a value apply.py still considers "detected"
      (`_flag_blocks_detected_value`, `_is_detected(value) and
      _blocking(entry, name)`) -- NOT simply "entry.flags[name] blocks",
      since that string outlives the value becoming MANUAL/IMPORTED.
    ranges: no "missing" concept (no keep ranges just means the whole
    file), so only a blocking flag on a still-DETECTED/HINT time_ranges
    value counts -- core/jobs/apply.py writes neither today (ranges is
    never required and never flagged), see the module docstring."""
    fields = []
    if entry.crop is None or _flag_blocks_detected_value(entry, "crop", entry.crop):
        fields.append("crop")
    if (entry.brightness is None or brightness_is_stale(entry)
            or _flag_blocks_detected_value(entry, "brightness", entry.brightness)):
        fields.append("brightness")
    if _flag_blocks_detected_value(entry, "ranges", entry.time_ranges):
        fields.append("ranges")
    return fields


def _flagged_text(entry: FileEntry) -> str:
    fields = _blocking_fields(entry)
    if not fields:
        # Defensive only: compute_review_state() (core/jobs/apply.py) never
        # stores FLAGGED without one of the above being true.
        return "flagged"
    return "check " + " + ".join(_FIELD_LABEL[name] for name in fields)


def _pending_text(running_detectors: set[str]) -> str:
    for name in _PENDING_ORDER:
        if name in running_detectors:
            return _PENDING_TEXT[name]
    return "waiting"


def badge_for(entry: FileEntry, *, running_detectors: set[str], done: bool,
              run_state: str | None) -> tuple[str, str]:
    """(text, tone) for the review-queue badge (ruling B10). `tone` is one
    of `app.widgets.base.Badge`'s tones: "default" | "warn" | "good" | "bad".

    `run_state` (None outside a run) takes over the badge entirely once a
    run exists for this file, per B10's "during a run" row: "queued" and
    "running" carry no special colour here ("default" -- the blue running
    dot is a separate Chip/Dot, not this text badge, per B10's "(blue dot)"
    annotation), "done" is good, "failed" is bad.

    Outside a run, in order: skipped (an explicit user choice) always shows
    first and stays "default" (B10: "default, row dimmed" -- the dimming is
    the caller's row style, not this badge's tone); then the non-empty
    `chi/<stem>.ass` `done` state; then the file's ReviewState ladder.
    PENDING's text names whichever required detector `running_detectors`
    has running (see _PENDING_ORDER), else "waiting".
    """
    if run_state is not None:
        if run_state == "failed":
            return "failed", "bad"
        if run_state == "done":
            return "done", "good"
        if run_state == "queued":
            return "queued", "default"
        return "running", "default"

    if entry.skipped:
        return "skipped", "default"
    if done:
        return "done", "good"
    if entry.review == ReviewState.REVIEWED:
        return "reviewed", "good"
    if entry.review == ReviewState.FLAGGED:
        return _flagged_text(entry), "warn"
    if entry.review == ReviewState.PROPOSED:
        return "ready", "default"
    return _pending_text(running_detectors), "default"


# --------------------------------------------------------------------------
# Captions (crop_caption / brightness_caption / ranges_caption)
# --------------------------------------------------------------------------

_NOT_DETECTED = "not detected yet"  # evidence=None, source neither MANUAL nor
                                     # IMPORTED: not covered by the brief's
                                     # caption rules (a field with no value
                                     # and no detection run yet) -- a neutral
                                     # placeholder rather than a guessed reason.


def _manual_or_imported(source: Source | None) -> tuple[str, float, str] | None:
    """MANUAL always reads "set by you" -- even over a stale/older evidence
    dict, since a DETECTED/HINT value accepted through Mark reviewed becomes
    MANUAL and must read that way (plan 3A Task 7) -- checked before
    `evidence` is even looked at. IMPORTED with no evidence reads "imported
    from the previous version". Returns None when neither applies, so the
    caller falls through to its evidence-based rule."""
    if source == Source.MANUAL:
        return "set by you", 1.0, "ok"
    if source == Source.IMPORTED:
        return "imported from the previous version", 1.0, "ok"
    return None


def crop_caption(evidence: dict | None, source: Source | None, *,
                  blocking: bool = False) -> tuple[str, float, str]:
    """(caption, bar fraction, tone) for the Detected section's Crop row.

    `blocking`: the caller's own read of whether `entry.flags["crop"]` holds
    a blocking flag that is not in `evidence["flagged"]` -- e.g.
    "differs-from-hint?" (core/jobs/apply.py's FLAG_DIFFERS_FROM_HINT),
    composed onto `entry.flags` but never written into the stored evidence
    dict itself. True forces tone "warn" on top of whatever the evidence
    alone would say. Ignored for the MANUAL/IMPORTED/no-evidence branches,
    which are provenance text, not a confidence reading."""
    if source == Source.MANUAL:
        return _manual_or_imported(source)
    if evidence is None:
        return _manual_or_imported(source) or (_NOT_DETECTED, 0.0, "default")
    agreed = int(evidence.get("agreed") or 0)
    probes_used = int(evidence.get("probes_used") or 0)
    bar = (agreed / probes_used) if probes_used else 0.0
    ok = only_informational(evidence.get("flagged"), _crop.INFORMATIONAL_FLAGS) and not blocking
    return f"{agreed} of {probes_used} samples agree", bar, ("ok" if ok else "warn")


def brightness_caption(evidence: dict | None, source: Source | None, *,
                        blocking: bool = False) -> tuple[str, float, str]:
    """(caption, bar fraction, tone) for the Detected section's Brightness
    row.

    An evidence dict with an empty `curve` is a cheap-path result (the
    cheap path -- `detect_brightness(..., folder_plateau=...)` -- never
    populates `curve`; see core/detect/brightness.py's module docstring and
    `detect_brightness`'s cheap-path branch), whose `plateau` is the
    FOLDER's plateau, not one measured on this file: its caption reads
    "folder safe range {lo}-{hi}" and never takes the "narrow " prefix (that
    prefix is reserved for a genuinely per-file measured plateau) -- but
    still turns the tone "warn" when narrow, same as the per-file form.

    `blocking`: as `crop_caption`'s, e.g. entry.flags["brightness"] holding
    "differs-from-hint?" from a hint re-detection. Ignored for the MANUAL/
    IMPORTED/no-evidence branches."""
    if source == Source.MANUAL:
        return _manual_or_imported(source)
    if evidence is None:
        return _manual_or_imported(source) or (_NOT_DETECTED, 0.0, "default")
    plateau = evidence.get("plateau")
    if plateau is None:
        return "not verified", 0.0, "warn"
    lo, hi = plateau
    narrow = (hi - lo) < 20
    flagged_blocked = not only_informational(evidence.get("flagged"), _brightness.INFORMATIONAL_FLAGS)
    cheap_path = not evidence.get("curve")
    if cheap_path:
        caption = f"folder safe range {lo}–{hi}"
    else:
        caption = f"safe range {lo}–{hi}"
        if narrow:
            caption = "narrow " + caption
    bar = min(1.0, (hi - lo) / 60)
    tone = "warn" if (narrow or flagged_blocked or blocking) else "ok"
    return caption, bar, tone


def ranges_caption(evidence: dict | None, source: Source | None, *,
                    blocking: bool = False) -> tuple[str, float, str]:
    """(caption, bar fraction, tone) for the Detected section's OCR window
    row. `n` is the highest matched_files among the file's intro/outro
    blocks, minus 1 (matched_files counts this file itself -- core/detect/
    ranges/pipeline.py's Block docstring).

    `blocking`: as `crop_caption`'s -- entry.flags["ranges"] holding a
    blocking flag the caller found (core/jobs/apply.py does not write one
    today; see the module docstring). Forces tone "warn" whether or not
    blocks were found. Ignored for the MANUAL/IMPORTED/no-evidence
    branches."""
    if source == Source.MANUAL:
        return _manual_or_imported(source)
    if evidence is None:
        return _manual_or_imported(source) or (_NOT_DETECTED, 0.0, "default")
    blocks = [b for b in (evidence.get("blocks") or []) if b.get("kind") in ("intro", "outro")]
    if not blocks:
        return "no repeating intro/outro found", 0.0, ("warn" if blocking else "default")
    n = max(int(b["matched_files"]) for b in blocks) - 1
    bar = max(float(b["score"]) for b in blocks)
    return f"intro+outro matched in {n} episodes", bar, ("warn" if blocking else "ok")


# --------------------------------------------------------------------------
# format_duration()
# --------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """"27:08" (< 1h) / "1:02:03" (>= 1h). Negative input clamps to 0."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# --------------------------------------------------------------------------
# Value texts (plan 3B Task 3: the inspector's Detected rows and header, the
# placeholder tabs)
# --------------------------------------------------------------------------

_NO_VALUE = "—"
_RANGES_SHOWN = 2          # more keep ranges than this read "first, second +N"


def crop_text(crop) -> str:
    """"288, 784 · 1344 × 55" (x, y · width × height), or "—"."""
    if crop is None:
        return _NO_VALUE
    return f"{crop.x}, {crop.y} · {crop.width} × {crop.height}"


def brightness_text(brightness) -> str:
    return _NO_VALUE if brightness is None else str(brightness.value)


def _time_text(value: str | None, missing: str) -> str:
    """A stored "MM:SS" / "H:MM:SS" as format_duration writes it ("02:33"
    reads "2:33"); text that is not a time is shown as stored."""
    if value is None:
        return missing
    try:
        parts = [int(part) for part in value.split(":")]
    except ValueError:
        return value
    if not 2 <= len(parts) <= 3 or any(part < 0 for part in parts):
        return value
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return format_duration(seconds)


def ranges_text(time_ranges) -> str:
    """"2:33 → 23:05" per keep range; "whole file" for None or an empty list
    (a manual "use whole file" is stored as an empty MANUAL list)."""
    if time_ranges is None or not time_ranges.ranges:
        return "whole file"
    spans = [f"{_time_text(r.start, '0:00')} → {_time_text(r.end, 'end')}" for r in time_ranges.ranges]
    if len(spans) > _RANGES_SHOWN:
        return ", ".join(spans[:_RANGES_SHOWN]) + f" +{len(spans) - _RANGES_SHOWN}"
    return ", ".join(spans)


def media_text(media) -> str:
    """"1920×888 · 27:08 · 25 fps"; unknown (zero) parts are left out."""
    parts = []
    if media.width > 0 and media.height > 0:
        parts.append(f"{media.width}×{media.height}")
    if media.duration > 0:
        parts.append(format_duration(media.duration))
    if media.fps > 0:
        parts.append(f"{media.fps:g} fps")
    return " · ".join(parts)


def clock(seconds: float) -> str:
    """"MM:SS" (minutes are not wrapped into hours): a proof line's time."""
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def proof_window_clock(sample_time: float | None, duration: float) -> str | None:
    """"09:38–10:08": the proof window ruling C4 picks for a file, as the
    inspector's "running on ...…" line reads it, or None while the duration
    is unknown (`run_proof` refuses such a file).

    Uses `core.jobs.detect_jobs.proof_window`, the same function ProofOcrJob
    resolves its window with, so the line names the window OCR really runs
    on. `clock()` truncates to whole seconds exactly as the job does."""
    try:
        start, end = _proof_window(sample_time, duration)
    except ValueError:
        return None
    return f"{clock(start)}–{clock(end)}"


# --------------------------------------------------------------------------
# The series median brightness (ui-spec §3.7's Detected note)
# --------------------------------------------------------------------------

SERIES_MEDIAN_MIN_FILES = 3         # fewer measured files than this: no median worth quoting
SERIES_MEDIAN_NOTE = "Series median brightness is {median} — this episode keeps its own."


def series_median_brightness(entries: Iterable[FileEntry]) -> int | None:
    """The median of the brightness values the folder's files actually have,
    or None while fewer than SERIES_MEDIAN_MIN_FILES have one.

    The lower middle value is taken for an even count, so the answer is
    always a value some episode really uses rather than an average of two.
    Every source counts (detected, hint, manual, imported): the note is about
    what the series is graded like, not about who chose the numbers."""
    values = sorted(int(entry.brightness.value) for entry in entries if entry.brightness is not None)
    if len(values) < SERIES_MEDIAN_MIN_FILES:
        return None
    return values[(len(values) - 1) // 2]


def series_median_note(median: int | None) -> str:
    """ui-spec §3.7's second sentence of the Detected note, or "" when there
    is no median to quote. The caller only asks for it when the file has a
    brightness of its own -- "this episode keeps its own" says nothing about
    a file with no value."""
    return "" if median is None else SERIES_MEDIAN_NOTE.format(median=median)


_FIELD_VALUE = {"crop": "crop", "brightness": "brightness", "ranges": "time_ranges"}


def field_blocking(entry: FileEntry, field: str) -> bool:
    """Whether `entry.flags[field]` holds a blocking flag on a DETECTED/HINT
    value -- the `blocking=` argument of the captions above. `field`: "crop" |
    "brightness" | "ranges"."""
    return _flag_blocks_detected_value(entry, field, getattr(entry, _FIELD_VALUE[field]))


# --------------------------------------------------------------------------
# Marking reviewed
# --------------------------------------------------------------------------

REVIEW_WAIT_TOOLTIP = "waiting for detections to finish"


def is_reviewed(entry: FileEntry) -> bool:
    return entry.review == ReviewState.REVIEWED


def is_pending(entry: FileEntry) -> bool:
    """Detections for a required value are still outstanding."""
    return entry.review == ReviewState.PENDING


def is_manual(value) -> bool:
    """A crop, brightness or time ranges value the user set (or pasted)."""
    return value is not None and value.source == Source.MANUAL


def can_mark_reviewed(entry: FileEntry) -> bool:
    """"Mark reviewed" (the button, Space, the queue menu) is offered unless
    the file is still PENDING: its detections have not finished, so there is
    nothing settled to accept yet."""
    return not is_pending(entry)


# --------------------------------------------------------------------------
# Pure detector helpers the review views need (plan 3C)
# --------------------------------------------------------------------------
# `app/views/*.py` may import no core module at all (tests/ui/
# test_main_window.py's test_views_import_no_core_modules), so the two
# Qt-free, pure detector helpers the crop view needs are reached through
# this module -- already the app's one Qt-free bridge into core/detect.
# Both call through the module objects (`_crop.` / `_ocr_view.`) rather
# than a bound name, so a test may monkeypatch `core.detect.crop.
# aggregate_box` / `core.detect.ocr_view.mask` and see the call.

def _rectangle(box) -> list[tuple[int, int]]:
    """(x, y, width, height) as the four corner points of its rectangle --
    the polygon shape `aggregate_box` reads."""
    x, y, width, height = (int(value) for value in box)
    return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]


def aggregate_crop_box(boxes_per_sample, frame_size, settings: dict | None = None,
                       sample_times=None) -> tuple[int, int, int, int] | None:
    """`core.detect.crop.aggregate_box` over one list of (x, y, w, h) boxes
    per sampled frame ("fit to all samples").

    `settings` mirrors the folder's detector settings and must carry the
    `bottom_half_cutoff` the detection itself used (`evidence["crop"]
    ["cutoff_frac"]`, 0.0 after a full-frame retry) -- judging the samples
    with another band would keep or drop different text than the box being
    replaced. None when there is nothing to build a box from."""
    polygons = [[_rectangle(box) for box in boxes] for boxes in boxes_per_sample]
    box = _crop.aggregate_box(polygons, tuple(int(value) for value in frame_size), settings=settings,
                              sample_times=None if sample_times is None else [float(t) for t in sample_times])
    return None if box is None else tuple(int(value) for value in box)


def mask_region(region, threshold: int):
    """`core.detect.ocr_view.mask`: the OCR pass's brightness filter over a
    BGR region -- keep the pixels whose every channel is >= `threshold`."""
    return _ocr_view.mask(region, int(threshold))
