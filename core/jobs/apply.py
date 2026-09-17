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
    the inspector can show "detected X, yours Y". Hint-driven results (a
    crop consensus seeded from an edit, a brightness re-detect checked
    against an edited value) are written with source HINT.

Stale brightness
    A brightness result is dropped entirely (no value, evidence or flag)
    unless the file's crop still has the (x, y, width, height) the result was
    measured with (BrightnessJobResult.crop_box). An applied result records
    that box as evidence["brightness"]["crop_box"] (the latest result's
    crop). When it also writes the value, the box goes into
    evidence["brightness"]["value_crop_box"] (the crop the stored value was
    measured on), which later unapplied results carry over unchanged. A
    detected or hinted brightness whose value_crop_box is no longer the
    file's crop (the crop was re-detected or edited since) counts as missing
    (brightness_is_stale). MANUAL and IMPORTED brightness is never stale.

Hints
    "differs-from-hint?" (blocking) is added to the file's flags when a hint
    re-detection disagrees with its hint: a brightness plateau without the
    hint value, or a crop box inconsistent with the hint under the crop
    detector's own consensus rule (core.detect.crop.consistent_with_consensus).

Review
    compute_review_state() is the only place a file's state is derived.
    REVIEWED is the one stored decision, and it never hides a required value
    that is missing or stale. It is never stored beside a blocking flag on a
    required detected or hinted value either, by construction:
    - mark_reviewed(True) is explicit acceptance: every required detected or
      hinted value carrying a blocking flag becomes MANUAL (detection never
      overwrites it again; its evidence and flag stay stored), except a stale
      brightness. It then stores REVIEWED, or, when a flagged detected or
      hinted value is left (a stale one), the computed state.
    - set_manual_* and paste_settings write MANUAL values and store REVIEWED
      only when no required field still holds a detected or hinted value with
      a blocking flag; otherwise they store the computed state (the file stays
      FLAGGED until the user accepts or edits that field).
    - apply_folder_change un-reviews files whose newly required values are
      missing, stale or flagged when the folder starts extracting dialogue.
    - An apply clears REVIEWED when it changes a value (a detected or hinted
      value the file was reviewed with, a stale brightness, or an empty field
      detection fills: the user never saw that value) or stores a blocking
      flag for a required field holding no value or a detected/hinted one.
    The user's own values are never changed by detection, so they never
    clear it.
    Flags describe detection results. A blocking flag counts against a file
    only while the field holds a detected or hinted value (or none): a
    detection that was not applied over the user's value is kept as evidence
    and does not flag the file. The exception is SOURCE_INDEPENDENT_FLAGS --
    reasons about the stored value rather than about a detection (a crop cut
    to fit the frame, a brightness pasted for a crop the file cannot hold).
    Those count whoever set the value, until the user answers them by marking
    the file reviewed or writing the field again (see _counts_flagged).

Crops the frame can hold
    A stored crop is always a box the file's frame can hold, because the OCR
    pass clamps differently (videocr.video.infer_crop_region narrows the box
    or drops it and reads the bottom third instead), so a box that does not
    fit would make the run read a region the stored value does not name.
    set_manual_crop and paste_settings clamp with
    core.project.model.clamp_crop_box; apply_metadata re-checks the stored
    box when the frame size first becomes known and cuts it then. A crop that
    had to be cut is not the value the caller gave, so the file is flagged
    FLAG_CROP_CLAMPED and un-reviewed.
    Functions here do not know which jobs are still pending. When they
    clear REVIEWED, they store PENDING as a placeholder. The model owner
    follows applies and edits with recompute_all(), which derives every
    non-reviewed file's real state.
"""
from __future__ import annotations

from dataclasses import replace

from core.detect import brightness as _brightness
from core.detect import crop as _crop
from core.detect.flags import compose_flag, only_informational, remove_flag
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
    clamp_crop_box,
    frame_size_known,
)

FLAG_DIFFERS_FROM_HINT = "differs-from-hint?"   # blocking: a hint re-detection that disagrees with its hint

# Blocking reasons about the STORED value rather than about a detection, so
# they count whoever set it -- see _counts_flagged.
FLAG_CROP_CLAMPED = "crop-cut-to-fit"           # the box given did not fit the frame and was cut
FLAG_BRIGHTNESS_OTHER_CROP = "brightness-other-crop"   # pasted from a file whose crop this one cannot hold
SOURCE_INDEPENDENT_FLAGS = {
    "crop": frozenset({FLAG_CROP_CLAMPED}),
    "brightness": frozenset({FLAG_BRIGHTNESS_OTHER_CROP}),
}

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


def _required(folder: FolderSettings) -> tuple[str, ...]:
    return DIALOGUE_DETECTORS if folder.dialogue_enabled else ()


def _blocking(entry: FileEntry, name: str) -> bool:
    return not only_informational(entry.flags.get(name) or None, INFORMATIONAL_FLAGS[name])


def _is_detected(value) -> bool:
    return value is not None and value.source in DETECTION_SOURCES


def brightness_is_stale(entry: FileEntry) -> bool:
    """True when the file's brightness is detected or hinted and the crop its
    value was measured on (evidence["brightness"]["value_crop_box"]) is not
    the file's crop. A brightness without that record (its value was not
    written by apply_brightness) is not judged stale; the latest result's
    crop_box says nothing about the stored value and is not consulted."""
    if not _is_detected(entry.brightness):
        return False
    measured = (entry.evidence.get("brightness") or {}).get("value_crop_box")
    if measured is None:
        return False
    return entry.crop is None or _value_key(entry.crop) != _crop_tuple(measured)


def counts_as_missing(entry: FileEntry, name: str) -> bool:
    """Whether the required field `name` ("crop" | "brightness") has no value
    the file can be run with: None, or a brightness measured on a crop the
    file no longer has. The one rule for "a required value is missing",
    shared with the window (app.state_text.can_mark_reviewed) so the button
    and the store cannot disagree about it: REVIEWED is never stored while
    this is true (see compute_review_state)."""
    return getattr(entry, name) is None or (name == "brightness" and brightness_is_stale(entry))


_counts_as_missing = counts_as_missing        # the old private spelling, still used below


def _source_independent(entry: FileEntry, name: str) -> set[str]:
    """The reasons in `entry.flags[name]` that describe the stored value
    itself, so they count whatever its source is."""
    reasons = set((entry.flags.get(name) or "").split("+"))
    return reasons & SOURCE_INDEPENDENT_FLAGS.get(name, frozenset())


def _counts_flagged(entry: FileEntry, name: str) -> bool:
    """Whether `entry.flags[name]` counts against the file.

    A detector's flag is its opinion of its OWN result, so it counts only
    while the field still holds that detected or hinted value (a value the
    user has accepted or replaced is not in doubt). A source-independent
    reason is a fact about the stored value -- a crop that had to be cut to
    fit the frame, a brightness pasted for a crop this file cannot hold --
    and counts on a MANUAL or IMPORTED value too, until the user answers it
    (mark_reviewed) or writes the field again.
    """
    if _source_independent(entry, name):
        return True
    return _is_detected(getattr(entry, name)) and _blocking(entry, name)


def _set_own_flag(entry: FileEntry, name: str, reason: str, on: bool) -> None:
    """Add or drop one source-independent reason, leaving the detector's own
    reasons in the string untouched."""
    existing = entry.flags.get(name) or ""
    flag = compose_flag(existing, reason) if on else remove_flag(existing, reason)
    if flag or name in entry.flags:
        entry.flags[name] = flag


def _fitted_crop(entry: FileEntry, box) -> tuple[tuple[int, int, int, int], bool]:
    """(`box` as the file's frame can hold it, whether that changed it). An
    unknown frame size cannot change it: see clamp_crop_box."""
    box = _crop_tuple(box)
    fitted = clamp_crop_box(box, (entry.media.width, entry.media.height))
    return fitted, fitted != box


def _write_detected(entry: FileEntry, field: str, new, *, old_stale: bool = False) -> None:
    """Write a detected value into a field _detection_may_write() allowed.
    A REVIEWED file loses its review when the OCR-visible value changes (a
    detected or hinted value replaced by a different one, or an empty field
    filled), or when the value replaced was stale (`old_stale`)."""
    old = getattr(entry, field)
    setattr(entry, field, new)
    if (entry.review == ReviewState.REVIEWED and (old is None or old.source in DETECTION_SOURCES)
            and (old_stale or _value_key(old) != _value_key(new))):
        entry.review = ReviewState.PENDING


def _unreview_on_blocking_flag(project: Project, entry: FileEntry, name: str) -> None:
    """S4/S5: a blocking flag just stored for a required field that holds no
    value or a detected/hinted one clears REVIEWED."""
    value = getattr(entry, name)
    if (entry.review == ReviewState.REVIEWED and name in _required(project.folder)
            and (value is None or value.source in DETECTION_SOURCES) and _blocking(entry, name)):
        entry.review = ReviewState.PENDING


def _unreview(entry: FileEntry) -> None:
    """Drop the stored REVIEWED mark (PENDING is a placeholder the model
    owner's recompute_all replaces with the real state)."""
    if entry.review == ReviewState.REVIEWED:
        entry.review = ReviewState.PENDING


def _crop_tuple(box) -> tuple[int, int, int, int]:
    x, y, width, height = box
    return int(x), int(y), int(width), int(height)


# --------------------------------------------------------------------------
# Job results
# --------------------------------------------------------------------------

def apply_metadata(project: Project, r: MetadataResult | None) -> list[str]:
    """Store the file's frame size, duration and fps, and re-check its stored
    crop against the size now that it is known.

    Returns the files whose crop box this changed (at most one), so the model
    owner can re-measure what was measured on the old box, exactly as it does
    after an edit. A crop is stored before the size is known -- by a paste, a
    v1 import, or an edit made while the metadata job was still queued -- and
    a box the frame cannot hold is not the region the run would read
    (clamp_crop_box). It is cut to fit here rather than left to be narrowed
    or dropped silently at OCR time, and the file is flagged FLAG_CROP_CLAMPED
    and un-reviewed: the stored value is now honest, but it is not the value
    the user drew, so they are sent back to look at it.
    """
    if r is None or r.file not in project.files:
        return []
    entry = project.files[r.file]
    entry.media = Media(width=int(r.width), height=int(r.height),
                        duration=float(r.duration), fps=float(r.fps))
    if entry.crop is None or not frame_size_known(entry.media):
        return []
    fitted, cut = _fitted_crop(entry, (entry.crop.x, entry.crop.y, entry.crop.width, entry.crop.height))
    _set_own_flag(entry, "crop", FLAG_CROP_CLAMPED, cut)
    if not cut:
        return []
    entry.crop = Crop(*fitted, entry.crop.source)
    _unreview(entry)
    return [r.file]


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
    flag = result.flagged or ""
    if r.hint is not None and result.box is not None and not _crop_agrees_with_hint(result, r.hint):
        flag = compose_flag(flag, FLAG_DIFFERS_FROM_HINT)
    # A detector's flags replace the previous ones, but a source-independent
    # reason is not the detector's to withdraw: a MANUAL box that had to be
    # cut to fit is still cut, whatever this detection found.
    kept = _source_independent(entry, "crop")
    entry.flags["crop"] = flag

    written = False
    if _detection_may_write(entry.crop) and result.box is not None and result.auto_applicable:
        source = Source.HINT if r.hint is not None else Source.DETECTED
        _write_detected(entry, "crop", Crop(*_crop_tuple(result.box), source))
        written = True
    if not written:                         # the box those reasons describe is still the stored one
        for reason in kept:
            _set_own_flag(entry, "crop", reason, True)
    _unreview_on_blocking_flag(project, entry, "crop")

    if result.hit_pts and (entry.sample_time is None or written):
        entry.sample_time = float(result.hit_pts[0])


def _crop_agrees_with_hint(result, hint: tuple[float, float]) -> bool:
    """The crop detector's own consensus rule, with the hint as the whole
    consensus (as CropJob seeds it). A result without a usable frame size
    cannot be checked and does not agree."""
    if result.frame_size is None or not result.frame_size[1] or result.frame_size[1] <= 0:
        return False
    return _crop.consistent_with_consensus(result.box, result.frame_size,
                                           [hint] * _crop.CONSENSUS_MIN_ENTRIES)


def apply_brightness(project: Project, r: BrightnessJobResult | None) -> None:
    """Store the brightness evidence (with its tiles) and flags; write the
    value when allowed, with source HINT for a hint re-detection. A hint
    re-detection whose plateau does not contain the hint value gains
    "differs-from-hint?" on the file's flags (ruling C3). Its value is still
    written when auto-applicable, because it is the detector's verified
    value, and the flag sends the file to review.

    Stale results are dropped entirely: when the file has no crop, or its
    crop is not the box the result was measured with (r.crop_box), nothing
    about the file changes. An applied result records r.crop_box in its
    evidence, which brightness_is_stale() reads."""
    if r is None or detector_cancelled(r.result.flagged):
        return
    entry = project.files.get(r.file)
    if entry is None:
        return
    if entry.crop is None or r.crop_box is None or _value_key(entry.crop) != _crop_tuple(r.crop_box):
        return
    result = r.result
    old_stale = brightness_is_stale(entry)
    previous = entry.evidence.get("brightness") or {}
    evidence = {**result.to_evidence(),
                "tiles": {kind: float(t) for kind, t in r.tiles.items()},
                "crop_box": list(_crop_tuple(r.crop_box))}
    if previous.get("value_crop_box") is not None:          # the stored value's crop, until a value is written
        evidence["value_crop_box"] = list(previous["value_crop_box"])
    entry.evidence["brightness"] = evidence
    flag = result.flagged or ""
    if r.hint_value is not None:
        plateau = result.plateau
        if plateau is None or not plateau[0] <= r.hint_value <= plateau[1]:
            flag = compose_flag(flag, FLAG_DIFFERS_FROM_HINT)
    kept = _source_independent(entry, "brightness")     # not this detection's to withdraw (see apply_crop)
    entry.flags["brightness"] = flag

    written = False
    if _detection_may_write(entry.brightness) and result.auto_applicable:
        source = Source.HINT if r.hint_value is not None else Source.DETECTED
        _write_detected(entry, "brightness", Brightness(int(result.value), source), old_stale=old_stale)
        evidence["value_crop_box"] = list(_crop_tuple(r.crop_box))
        written = True
    if not written:                                    # the value those reasons describe is still the stored one
        for reason in kept:
            _set_own_flag(entry, "brightness", reason, True)
    _unreview_on_blocking_flag(project, entry, "brightness")


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

def _flagged_fields(folder: FolderSettings, entry: FileEntry) -> list[str]:
    """Required fields whose stored flags count against the file
    (_counts_flagged): a detected or hinted value with a blocking flag (stale
    or not), or a value of any source carrying a source-independent reason."""
    return [name for name in _required(folder) if _counts_flagged(entry, name)]


def _store_computed_state(folder: FolderSettings, entry: FileEntry) -> None:
    """Replace the stored state with the derived one, never REVIEWED (no
    pending information here: recompute_all refines PENDING)."""
    entry.review = ReviewState.PENDING
    entry.review = compute_review_state(entry, folder, detections_pending=set(), ranges_pending=False)


def _review_unless_flagged(folder: FolderSettings, entry: FileEntry) -> None:
    """Store REVIEWED when no required field holds a flagged detected or
    hinted value; otherwise store the computed state."""
    if _flagged_fields(folder, entry):
        _store_computed_state(folder, entry)
    else:
        entry.review = ReviewState.REVIEWED


def _review_after_edit(project: Project, entry: FileEntry) -> None:
    _review_unless_flagged(project.folder, entry)


def set_manual_crop(project: Project, file: str, box: tuple[int, int, int, int]) -> None:
    """MANUAL crop, cut to a box the file's frame can hold (clamp_crop_box);
    REVIEWED unless another required field is still flagged.

    The box a view commits is already inside the frame it drew on, so the
    clamp is normally a no-op. It is not one for a box drawn against a
    guessed frame size, or pasted from a bigger file: the file is then
    flagged FLAG_CROP_CLAMPED, because what is stored is no longer what the
    caller asked for. A frame size that is not known yet cannot cut anything;
    apply_metadata re-checks the value when it arrives."""
    entry = project.files[file]
    fitted, cut = _fitted_crop(entry, box)
    entry.crop = Crop(*fitted, Source.MANUAL)
    _set_own_flag(entry, "crop", FLAG_CROP_CLAMPED, cut)
    _review_after_edit(project, entry)


def set_manual_brightness(project: Project, file: str, value: int) -> None:
    """MANUAL brightness; REVIEWED unless another required field is still flagged."""
    entry = project.files[file]
    entry.brightness = Brightness(int(value), Source.MANUAL)
    _review_after_edit(project, entry)


def set_manual_time_ranges(project: Project, file: str,
                           ranges: list[tuple[str | None, str | None]] | None) -> None:
    """None (or []) is the whole file. It is stored as TimeRanges([], MANUAL),
    not as None, so that detection cannot replace the user's choice.
    ocr_call_for maps both to [] (the whole file). REVIEWED unless a required
    field is still flagged."""
    entry = project.files[file]
    entry.time_ranges = TimeRanges([TimeRange(start, end) for start, end in (ranges or [])], Source.MANUAL)
    _review_after_edit(project, entry)


def mark_reviewed(project: Project, file: str, reviewed: bool = True) -> None:
    """Mark a file reviewed, or clear the mark (PENDING until recompute_all).

    Marking is explicit acceptance: every required detected or hinted value
    carrying a blocking flag becomes MANUAL first, so detection never
    overwrites the value the user accepted; its evidence and flag strings
    stay stored. A stale brightness (measured on another crop) is not
    accepted: it counts as missing, and the file stays FLAGGED or PENDING
    until it is measured again or edited. REVIEWED is stored only when no
    required field is left holding a flagged detected or hinted value (a
    stale flagged brightness is); otherwise the computed state is stored, so
    restoring the crop later cannot turn the unaccepted value into a
    reviewed one. Clearing the mark reverts nothing."""
    entry = project.files[file]
    if not reviewed:
        if entry.review == ReviewState.REVIEWED:
            entry.review = ReviewState.PENDING
        return
    for name in _flagged_fields(project.folder, entry):
        if not _counts_as_missing(entry, name):
            value = getattr(entry, name)
            setattr(entry, name, replace(value, source=Source.MANUAL))
            # A reason about the stored value itself (a crop cut to fit, a
            # brightness pasted for another crop) is exactly what the user is
            # answering here, so it is dropped rather than kept as evidence.
            for reason in _source_independent(entry, name):
                _set_own_flag(entry, name, reason, False)
    _review_unless_flagged(project.folder, entry)


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
    as MANUAL, then store REVIEWED, or the computed state when a required
    field it did not replace still holds a flagged detected or hinted value.
    A clip with nothing in it changes nothing.

    The crop is cut to a box `target`'s frame can hold, as set_manual_crop
    does: the clip may come from a file of another resolution. When it had to
    be cut, the file is flagged FLAG_CROP_CLAMPED, and a brightness pasted
    with it FLAG_BRIGHTNESS_OTHER_CROP -- it was measured inside a region
    this file does not have, so it is stored (the user asked for it) but not
    claimed as reviewed until they say so."""
    entry = project.files[target]
    pasted = False
    cut = False
    if clip.get("crop") is not None:
        fitted, cut = _fitted_crop(entry, clip["crop"])
        entry.crop = Crop(*fitted, Source.MANUAL)
        _set_own_flag(entry, "crop", FLAG_CROP_CLAMPED, cut)
        pasted = True
    if clip.get("brightness") is not None:
        entry.brightness = Brightness(int(clip["brightness"]), Source.MANUAL)
        _set_own_flag(entry, "brightness", FLAG_BRIGHTNESS_OTHER_CROP, cut)
        pasted = True
    if clip.get("time_ranges") is not None:
        entry.time_ranges = TimeRanges([TimeRange(start, end) for start, end in clip["time_ranges"]],
                                       Source.MANUAL)
        pasted = True
    if pasted:
        _review_after_edit(project, entry)


def apply_folder_change(project: Project, old: FolderSettings, new: FolderSettings) -> None:
    """After the folder's settings change from `old` to `new`: when fields
    become required (the folder starts extracting dialogue), every REVIEWED
    file whose newly required value is missing, stale, or a detected/hinted
    value with a blocking flag stores the computed state (under `new`)
    instead: it was reviewed while those values did not matter. Nothing is
    accepted or changed otherwise, and a change that adds no required field
    (including dialogue -> labels-only) changes no file."""
    newly_required = [name for name in _required(new) if name not in _required(old)]
    if not newly_required:
        return
    for entry in project.files.values():
        if entry.review != ReviewState.REVIEWED:
            continue
        if any(_counts_as_missing(entry, name) or _counts_flagged(entry, name) for name in newly_required):
            _store_computed_state(new, entry)


# --------------------------------------------------------------------------
# Review state
# --------------------------------------------------------------------------

def compute_review_state(entry: FileEntry, folder: FolderSettings, *,
                         detections_pending: set[str], ranges_pending: bool) -> ReviewState:
    """The file's review state. Required fields are crop and brightness when
    dialogue is extracted, none for labels-only. In order:

    1. Each required field, crop first: a missing value (None, or a stale
       brightness, see brightness_is_stale) makes the file PENDING if that
       field's detector is pending, else FLAGGED. REVIEWED never hides it.
    2. REVIEWED if the user reviewed the file. (The API never stores REVIEWED
       beside a blocking flag on a required detected or hinted value: see the
       module docstring.)
    3. PENDING while a required detector is pending for a field holding a
       detected or hinted value, or while the folder's ranges are pending
       and the file's time_ranges may still be written by them (None,
       DETECTED or HINT; MANUAL and IMPORTED ranges never wait).
    4. FLAGGED when a required field's stored flags count against it
       (_counts_flagged): a value that is detected or hinted and whose flags
       include a reason that is not informational for that detector
       ("differs-from-hint?" and unknown reasons block), or a value of any
       source carrying one of SOURCE_INDEPENDENT_FLAGS. Other flags beside a
       MANUAL or IMPORTED value do not count.
    5. PROPOSED.
    """
    required = _required(folder)
    for name in required:
        if _counts_as_missing(entry, name):
            return ReviewState.PENDING if name in detections_pending else ReviewState.FLAGGED
    if entry.review == ReviewState.REVIEWED:
        return ReviewState.REVIEWED
    blocked = any(_counts_flagged(entry, name) for name in required)
    if (ranges_pending and _detection_may_write(entry.time_ranges)) or any(
            name in detections_pending and _is_detected(getattr(entry, name)) for name in required):
        return ReviewState.PENDING
    if blocked:
        return ReviewState.FLAGGED
    return ReviewState.PROPOSED


def recompute_all(project: Project, *, pending: dict[str, set[str]], ranges_pending: bool) -> None:
    """Derive every file's review state. `pending` maps file name to the
    kinds of its queued or running jobs."""
    for name, entry in project.files.items():
        entry.review = compute_review_state(entry, project.folder,
                                            detections_pending=pending.get(name, set()),
                                            ranges_pending=ranges_pending)
