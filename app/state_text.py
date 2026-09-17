"""Pure, Qt-free badge and caption text for the review queue and inspector
(ruling B10; caption rules from Task 1's brief). No PyQt6 import belongs
anywhere in this module -- see tests/app/test_state_text.py's subprocess
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

from core.detect import brightness as _brightness
from core.detect import crop as _crop
from core.detect.flags import only_informational
from core.jobs.apply import brightness_is_stale
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


def _blocking_fields(entry: FileEntry) -> list[str]:
    """Which of crop/brightness/time-ranges a FLAGGED badge should name, in
    B10's order. crop/brightness: missing (None, or a stale brightness --
    see brightness_is_stale) or carrying a blocking flag, mirroring
    core.jobs.apply's own `_counts_as_missing`/`_blocking`. ranges: no
    "missing" concept (no keep ranges just means the whole file), so only a
    blocking `entry.flags["ranges"]` counts."""
    fields = []
    if entry.crop is None or _flag_blocks(entry.flags.get("crop"), _INFORMATIONAL_FLAGS["crop"]):
        fields.append("crop")
    if (entry.brightness is None or brightness_is_stale(entry)
            or _flag_blocks(entry.flags.get("brightness"), _INFORMATIONAL_FLAGS["brightness"])):
        fields.append("brightness")
    if _flag_blocks(entry.flags.get("ranges"), _INFORMATIONAL_FLAGS["ranges"]):
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


def crop_caption(evidence: dict | None, source: Source | None) -> tuple[str, float, str]:
    """(caption, bar fraction, tone) for the Detected section's Crop row."""
    if source == Source.MANUAL:
        return _manual_or_imported(source)
    if evidence is None:
        return _manual_or_imported(source) or (_NOT_DETECTED, 0.0, "default")
    agreed = int(evidence.get("agreed") or 0)
    probes_used = int(evidence.get("probes_used") or 0)
    bar = (agreed / probes_used) if probes_used else 0.0
    ok = only_informational(evidence.get("flagged"), _crop.INFORMATIONAL_FLAGS)
    return f"{agreed} of {probes_used} samples agree", bar, ("ok" if ok else "warn")


def brightness_caption(evidence: dict | None, source: Source | None) -> tuple[str, float, str]:
    """(caption, bar fraction, tone) for the Detected section's Brightness
    row. A narrow plateau (< 20 wide) prefixes "narrow " and forces warn
    even when nothing is flagged (ui-spec: "narrow safe range (203-216)")."""
    if source == Source.MANUAL:
        return _manual_or_imported(source)
    if evidence is None:
        return _manual_or_imported(source) or (_NOT_DETECTED, 0.0, "default")
    plateau = evidence.get("plateau")
    if plateau is None:
        return "not verified", 0.0, "warn"
    lo, hi = plateau
    narrow = (hi - lo) < 20
    blocked = not only_informational(evidence.get("flagged"), _brightness.INFORMATIONAL_FLAGS)
    caption = f"safe range {lo}–{hi}"
    if narrow:
        caption = "narrow " + caption
    bar = min(1.0, (hi - lo) / 60)
    return caption, bar, ("warn" if (narrow or blocked) else "ok")


def ranges_caption(evidence: dict | None, source: Source | None) -> tuple[str, float, str]:
    """(caption, bar fraction, tone) for the Detected section's OCR window
    row. `n` is the highest matched_files among the file's intro/outro
    blocks, minus 1 (matched_files counts this file itself -- core/detect/
    ranges/pipeline.py's Block docstring)."""
    if source == Source.MANUAL:
        return _manual_or_imported(source)
    if evidence is None:
        return _manual_or_imported(source) or (_NOT_DETECTED, 0.0, "default")
    blocks = [b for b in (evidence.get("blocks") or []) if b.get("kind") in ("intro", "outro")]
    if not blocks:
        return "no repeating intro/outro found", 0.0, "default"
    n = max(int(b["matched_files"]) for b in blocks) - 1
    bar = max(float(b["score"]) for b in blocks)
    return f"intro+outro matched in {n} episodes", bar, "ok"


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
