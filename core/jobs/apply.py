"""Applying job results and user edits to a Project, and each file's review state.

Pure functions over the Qt-free model (core.project.model). They are called
only on the model owner's thread (ruling C8) and do no I/O. Job results come
from core.jobs.detect_jobs. A None result is a cancelled job (that module's
cancellation convention), and every apply_* function ignores it.

Detection never overrides the user (rulings C1/C2)
    A detection result writes a value only into an empty field, or over a
    value whose own source is DETECTED or HINT, and only when the result is
    auto_applicable. MANUAL and IMPORTED values are never written by
    detection. The detector's evidence and flags are stored either way, so
    the inspector can show "detected X, yours Y".

Review
    compute_review_state() is the only place a file's state is derived.
    REVIEWED is the one stored decision: it is set by the user
    (mark_reviewed, set_manual_*, paste_settings), and it is cleared by an
    apply only when a detected or hinted value the file was reviewed with
    changes. A field that was empty had no reviewed value, so filling it
    leaves REVIEWED alone, and so do the user's own values, which detection
    never changes.
    Functions here do not know which jobs are still pending. When they
    clear REVIEWED, they store PENDING as a placeholder. The model owner
    follows applies and edits with recompute_all(), which derives every
    non-reviewed file's real state.
"""
from __future__ import annotations

from core.detect import brightness as _brightness
from core.detect import crop as _crop
from core.detect.flags import compose_flag, only_informational
from core.jobs.detect_jobs import (
    AudioProfileResult,
    BrightnessJobResult,
    CropJobResult,
    MetadataResult,
    RangesJobResult,
    detector_cancelled,
)
from core.project.model import (
    Brightness,
    Crop,
    FileEntry,
    FolderSettings,
    Media,
    Project,
    ReviewState,
    Source,
    TimeRange,
    TimeRanges,
)

FLAG_DIFFERS_FROM_HINT = "differs-from-hint?"   # blocking: a verified plateau that excludes the user's hint value

# Sources detection may overwrite (ruling C2).
DETECTION_SOURCES = frozenset({Source.DETECTED, Source.HINT})

# The detectors a file needs before it can run, when dialogue is extracted.
# A labels-only folder needs none of them.
DIALOGUE_DETECTORS = ("crop", "brightness")

# Per detector: flags that describe how the result was reached, not doubt about it.
INFORMATIONAL_FLAGS = {
    "crop": _crop.INFORMATIONAL_FLAGS,
    "brightness": _brightness.INFORMATIONAL_FLAGS,
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _value_key(value):
    """What the OCR pass sees of a value, without its source."""
    if value is None:
        return None
    if isinstance(value, Crop):
        return (value.x, value.y, value.width, value.height)
    if isinstance(value, Brightness):
        return value.value
    return tuple((r.start, r.end) for r in value.ranges)


def _detection_may_write(value) -> bool:
    return value is None or value.source in DETECTION_SOURCES


def _write_detected(entry: FileEntry, field: str, new) -> None:
    """Write a detected value into a field _detection_may_write() allowed.
    A REVIEWED file loses its review only when the detected or hinted value
    it was reviewed with changes."""
    old = getattr(entry, field)
    setattr(entry, field, new)
    if (entry.review == ReviewState.REVIEWED and old is not None
            and old.source in DETECTION_SOURCES and _value_key(old) != _value_key(new)):
        entry.review = ReviewState.PENDING


def _crop_tuple(box) -> tuple[int, int, int, int]:
    x, y, width, height = box
    return int(x), int(y), int(width), int(height)


# --------------------------------------------------------------------------
# Job results
# --------------------------------------------------------------------------

def apply_metadata(project: Project, r: MetadataResult | None) -> None:
    if r is None or r.file not in project.files:
        return
    project.files[r.file].media = Media(width=int(r.width), height=int(r.height),
                                        duration=float(r.duration), fps=float(r.fps))


def apply_crop(project: Project, r: CropJobResult | None) -> None:
    """Store the crop evidence and flags; write the box when allowed.

    sample_time becomes the first frame a subtitle was seen on (hit_pts[0],
    never sample_pts[0], whose first probe may have found nothing) when the
    file has none yet, or when this apply wrote the crop.
    """
    if r is None or detector_cancelled(r.result.flagged):
        return
    entry = project.files.get(r.file)
    if entry is None:
        return
    result = r.result
    entry.evidence["crop"] = result.to_evidence()
    entry.flags["crop"] = result.flagged or ""

    written = False
    if _detection_may_write(entry.crop) and result.box is not None and result.auto_applicable:
        source = Source.HINT if r.hint is not None else Source.DETECTED
        _write_detected(entry, "crop", Crop(*_crop_tuple(result.box), source))
        written = True

    if result.hit_pts and (entry.sample_time is None or written):
        entry.sample_time = float(result.hit_pts[0])


def apply_brightness(project: Project, r: BrightnessJobResult | None) -> None:
    """Store the brightness evidence (with its tiles) and flags; write the
    value when allowed. A hint re-detection whose plateau does not contain
    the hint value gains "differs-from-hint?" on the file's flags (ruling C3).
    Its value is still written when auto-applicable, because it is the
    detector's verified value, and the flag sends the file to review."""
    if r is None or detector_cancelled(r.result.flagged):
        return
    entry = project.files.get(r.file)
    if entry is None:
        return
    result = r.result
    entry.evidence["brightness"] = {**result.to_evidence(),
                                    "tiles": {kind: float(t) for kind, t in r.tiles.items()}}
    flag = result.flagged or ""
    if r.hint_value is not None:
        plateau = result.plateau
        if plateau is None or not plateau[0] <= r.hint_value <= plateau[1]:
            flag = compose_flag(flag, FLAG_DIFFERS_FROM_HINT)
    entry.flags["brightness"] = flag

    if _detection_may_write(entry.brightness) and result.auto_applicable:
        _write_detected(entry, "brightness", Brightness(int(result.value), Source.DETECTED))


def apply_ranges(project: Project, r: RangesJobResult | None) -> None:
    """For every file the analysis covered (every key of `durations`):
    store its matched blocks and duration as evidence, and write its keep
    ranges when allowed. A file with no keep ranges (analyse_detailed leaves
    it out of `keep`) gets None: the whole file. Files the analysis did not
    cover are left alone."""
    if r is None:
        return
    analysis = r.analysis
    for name, duration in analysis.durations.items():
        entry = project.files.get(name)
        if entry is None:
            continue
        entry.evidence["ranges"] = {
            "blocks": [{
                "start_sec": float(block.start_sec),
                "end_sec": float(block.end_sec),
                "kind": str(block.kind),
                "matched_files": int(block.matched_files),
                "score": float(block.score),
            } for block in analysis.blocks.get(name, [])],
            "duration": float(duration),
        }
        if _detection_may_write(entry.time_ranges):
            keep = analysis.keep.get(name) or []
            detected = (TimeRanges([TimeRange(start, end) for start, end in keep], Source.DETECTED)
                        if keep else None)
            _write_detected(entry, "time_ranges", detected)


def apply_audio_profile(project: Project, r: AudioProfileResult | None) -> None:
    """Store the envelope and speech spans under evidence["audio"]."""
    if r is None or r.file not in project.files:
        return
    profile = r.profile
    project.files[r.file].evidence["audio"] = {
        "envelope": [float(v) for v in profile.envelope],
        "speech": [[float(start), float(end)] for start, end in profile.speech],
        "duration": float(profile.duration),
    }


# --------------------------------------------------------------------------
# User edits
# --------------------------------------------------------------------------

def set_manual_crop(project: Project, file: str, box: tuple[int, int, int, int]) -> None:
    entry = project.files[file]
    entry.crop = Crop(*_crop_tuple(box), Source.MANUAL)
    entry.review = ReviewState.REVIEWED


def set_manual_brightness(project: Project, file: str, value: int) -> None:
    entry = project.files[file]
    entry.brightness = Brightness(int(value), Source.MANUAL)
    entry.review = ReviewState.REVIEWED


def set_manual_time_ranges(project: Project, file: str,
                           ranges: list[tuple[str | None, str | None]] | None) -> None:
    """None (or []) is the whole file. It is stored as TimeRanges([], MANUAL),
    not as None, so that detection cannot replace the user's choice.
    ocr_call_for maps both to [] (the whole file)."""
    entry = project.files[file]
    entry.time_ranges = TimeRanges([TimeRange(start, end) for start, end in (ranges or [])], Source.MANUAL)
    entry.review = ReviewState.REVIEWED


def mark_reviewed(project: Project, file: str, reviewed: bool = True) -> None:
    """Mark a file reviewed, or clear the mark (PENDING until recompute_all)."""
    entry = project.files[file]
    if reviewed:
        entry.review = ReviewState.REVIEWED
    elif entry.review == ReviewState.REVIEWED:
        entry.review = ReviewState.PENDING


def set_skipped(project: Project, file: str, skipped: bool) -> None:
    project.files[file].skipped = bool(skipped)


def copy_settings(project: Project, source: str) -> dict:
    """{"crop": (x, y, w, h) | None, "brightness": int | None,
    "time_ranges": [(start, end), ...] | None} of `source`, as plain,
    JSON-able data detached from the model. None means the file has no
    value to paste. An explicit whole-file choice is [], and pastes as one.
    """
    entry = project.files[source]
    return {
        "crop": None if entry.crop is None else _value_key(entry.crop),
        "brightness": None if entry.brightness is None else entry.brightness.value,
        "time_ranges": None if entry.time_ranges is None else [(r.start, r.end) for r in entry.time_ranges.ranges],
    }


def paste_settings(project: Project, target: str, clip: dict) -> None:
    """Apply every value the clip has (key present and not None) to `target`
    as MANUAL, and mark the target REVIEWED. A clip with nothing in it
    changes nothing."""
    if target not in project.files:
        raise KeyError(target)
    if clip.get("crop") is not None:
        set_manual_crop(project, target, clip["crop"])
    if clip.get("brightness") is not None:
        set_manual_brightness(project, target, clip["brightness"])
    if clip.get("time_ranges") is not None:
        set_manual_time_ranges(project, target, clip["time_ranges"])


# --------------------------------------------------------------------------
# Review state
# --------------------------------------------------------------------------

def compute_review_state(entry: FileEntry, folder: FolderSettings, *,
                         detections_pending: set[str], ranges_pending: bool) -> ReviewState:
    """REVIEWED if the user reviewed it. Otherwise PENDING while a required
    detector (crop and brightness when dialogue is extracted, none for
    labels-only) or the folder's ranges analysis is pending. Otherwise
    FLAGGED when a required value is missing, or when a required detector's
    stored flags include a reason that is not informational for that
    detector ("differs-from-hint?" and unknown reasons block). Otherwise
    PROPOSED."""
    if entry.review == ReviewState.REVIEWED:
        return ReviewState.REVIEWED
    required = DIALOGUE_DETECTORS if folder.dialogue_enabled else ()
    if ranges_pending or any(name in detections_pending for name in required):
        return ReviewState.PENDING
    for name in required:
        if getattr(entry, name) is None:
            return ReviewState.FLAGGED
        if not only_informational(entry.flags.get(name) or None, INFORMATIONAL_FLAGS[name]):
            return ReviewState.FLAGGED
    return ReviewState.PROPOSED


def recompute_all(project: Project, *, pending: dict[str, set[str]], ranges_pending: bool) -> None:
    """Derive every file's review state. `pending` maps file name to the
    kinds of its queued or running jobs."""
    for name, entry in project.files.items():
        entry.review = compute_review_state(entry, project.folder,
                                            detections_pending=pending.get(name, set()),
                                            ranges_pending=ranges_pending)
