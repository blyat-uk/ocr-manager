"""A doubted brightness the OCR engine answered (core.jobs.apply.apply_confirm).

`core.detect.brightness` measures a threshold and, when it doubts the
reading, flags it: the file lands in FLAGGED and the queue badges it "check
brightness". `core.detect.confirm` retires most of those doubts without the
user -- it masks ONE strip the file is known to hold a subtitle on and asks
the OCR engine to read it, at the stored value first and then at lower rungs.

The rules pinned here are apply's half of that: a passing ladder writes the
rung that passed (keeping the value's own source) and clears the field's
flag, so the file simply becomes PROPOSED and badges "ready" like any file
the detectors got right; a failing ladder changes nothing but the record; and
the result is dropped whole whenever the world moved between the job being
submitted and its result arriving -- another crop, another stored value, a
value the user has since set by hand.

The record itself is written either way, because it is both what the
Brightness tab can quote and the memo that stops the folder re-probing the
same failed ladder on every open.
"""
from __future__ import annotations

import pytest

from core.detect.brightness import FLAG_DIM_TEXT, FLAG_NARROW_PLATEAU
from core.detect.confirm import ConfirmResult, Rung
from core.jobs.apply import (
    FLAG_BRIGHTNESS_OTHER_CROP,
    apply_confirm,
    compute_review_state,
    counts_flagged,
    recompute_all,
)
from core.jobs.detect_jobs import ConfirmJobResult
from core.project.model import (
    Brightness,
    Crop,
    FileEntry,
    FolderSettings,
    Media,
    Project,
    ReviewState,
    Source,
)

BOX = (288, 969, 1344, 61)
OTHER_BOX = (300, 950, 1320, 70)
HD = Media(1920, 1080, 1200.0, 23.976)
CONF = 95                 # the folder's conf_threshold, the rungs' pass mark
STORED = 185              # the value the detector measured and doubted
PROBE_TIME = 178.304      # the strip the ladder was walked on


def _entry(*, value=STORED, source=Source.DETECTED, crop_box=BOX, flag=FLAG_NARROW_PLATEAU,
           value_crop_box=BOX, review=ReviewState.FLAGGED) -> FileEntry:
    """A file the brightness detector measured and doubted: a DETECTED value
    with a blocking flag, on a crop it was measured on. `value=None` is a
    file with no brightness, `crop_box=None` one with no crop, `flag=None`
    one whose doubt is already gone, and `value_crop_box=None` a value with
    no record of the crop it was measured on."""
    evidence: dict = {"brightness": {"value": STORED, "flagged": flag}}
    if value_crop_box is not None:
        evidence["brightness"]["value_crop_box"] = list(value_crop_box)
    return FileEntry("ep.mkv", media=HD,
                     crop=None if crop_box is None else Crop(*crop_box, Source.MANUAL),
                     brightness=None if value is None else Brightness(value, source),
                     flags={} if flag is None else {"brightness": flag},
                     evidence=evidence, review=review)


def _project(entry: FileEntry) -> Project:
    return Project(path="/proj", folder=FolderSettings(conf_threshold=CONF),
                   files={entry.name: entry})


def _rung(threshold: int, passed: bool) -> Rung:
    return Rung(threshold=threshold, gated=True, text="你好世界" if passed else "",
                confidence=0.99 if passed else 0.0, passed=passed)


def _passed(value: int = STORED) -> ConfirmResult:
    """A ladder that read the strip at `value` -- the rungs above it, if any,
    having failed, as confirm_brightness returns them."""
    rungs = [_rung(t, t == value) for t in range(STORED, value - 1, -10)]
    return ConfirmResult(value=value, probe_time=PROBE_TIME, rungs=tuple(rungs))


def _failed() -> ConfirmResult:
    """A ladder that read the strip at no rung at all."""
    return ConfirmResult(value=None, probe_time=PROBE_TIME,
                         rungs=tuple(_rung(t, False) for t in range(STORED, 139, -10)))


def _confirm(entry: FileEntry, result: ConfirmResult, *, crop_box=BOX,
             start_value: int = STORED, conf_threshold: int = CONF) -> Project:
    project = _project(entry)
    apply_confirm(project, ConfirmJobResult(entry.name, result, crop_box, start_value, conf_threshold))
    return project


def _settle(project: Project) -> Project:
    recompute_all(project, pending={}, ranges_pending=False)
    return project


def _confirm_record(entry: FileEntry):
    return (entry.evidence.get("brightness") or {}).get("confirm")


# --------------------------------------------------------------------------
# A ladder that passed
# --------------------------------------------------------------------------

def test_a_ladder_that_passes_at_the_stored_value_clears_the_flag_and_leaves_the_value():
    """The common case: the value the detector doubted is read at the
    folder's own confidence, so there is nothing to change but the doubt."""
    entry = _entry()
    _confirm(entry, _passed(STORED))
    assert entry.brightness == Brightness(STORED, Source.DETECTED)
    assert entry.flags["brightness"] == ""


@pytest.mark.parametrize("source", [Source.DETECTED, Source.HINT])
def test_a_ladder_that_passes_at_a_lower_rung_writes_that_value_and_keeps_its_source(source):
    """The rung that read the line becomes the file's value. The source is
    the detector's, not this stage's: a value checked against a user's edit
    is still a HINT afterwards, so a later re-detection treats it the same
    way it would have before the probe."""
    entry = _entry(source=source)
    _confirm(entry, _passed(165))
    assert entry.brightness == Brightness(165, source)
    assert entry.flags["brightness"] == ""


def test_a_value_the_ladder_wrote_records_the_crop_it_belongs_to():
    """brightness_is_stale reads value_crop_box, so a value written here must
    name its crop exactly as apply_brightness's does -- otherwise the next
    crop edit could not tell that this value was measured on the old one."""
    entry = _entry(value_crop_box=None)
    _confirm(entry, _passed(165))
    assert entry.evidence["brightness"]["value_crop_box"] == list(BOX)


def test_a_passing_ladder_stops_the_flag_counting_against_the_file():
    """counts_flagged is the one test for "this field's flag counts", and it
    is what compute_review_state derives FLAGGED from."""
    entry = _entry()
    assert counts_flagged(entry, "brightness") is True
    _confirm(entry, _passed(STORED))
    assert counts_flagged(entry, "brightness") is False


def test_a_confirmed_file_is_proposed_like_any_file_the_detectors_got_right():
    """The whole point of the stage. No marker, no flag of its own: the file
    that was FLAGGED ("check brightness") now badges "ready"."""
    entry = _entry()
    folder = FolderSettings(conf_threshold=CONF)
    assert compute_review_state(entry, folder, detections_pending=set(),
                                ranges_pending=False) == ReviewState.FLAGGED
    project = _settle(_confirm(entry, _passed(165)))
    assert project.files["ep.mkv"].review == ReviewState.PROPOSED


def test_a_reason_about_the_stored_value_survives_a_passing_ladder():
    """SOURCE_INDEPENDENT_FLAGS reasons describe the stored VALUE rather than
    a detection -- this one says the value was pasted for a crop the file
    cannot hold -- so no engine reading retires them; only the user does.
    The detector's own doubts around it go, and the order of what is left is
    the order it was composed in."""
    entry = _entry(flag=f"{FLAG_DIM_TEXT}+{FLAG_BRIGHTNESS_OTHER_CROP}+{FLAG_NARROW_PLATEAU}")
    project = _settle(_confirm(entry, _passed(STORED)))
    assert entry.flags["brightness"] == FLAG_BRIGHTNESS_OTHER_CROP
    assert counts_flagged(entry, "brightness") is True
    assert project.files["ep.mkv"].review == ReviewState.FLAGGED


# --------------------------------------------------------------------------
# A ladder that passed nowhere
# --------------------------------------------------------------------------

def test_a_ladder_that_passes_nowhere_changes_neither_value_nor_flag():
    """140 was not shown to be better than the detector's pick, so nothing
    was learned about the value: the file goes on saying "check brightness"
    until the user looks at it."""
    entry = _entry()
    project = _settle(_confirm(entry, _failed()))
    assert entry.brightness == Brightness(STORED, Source.DETECTED)
    assert entry.flags["brightness"] == FLAG_NARROW_PLATEAU
    assert project.files["ep.mkv"].review == ReviewState.FLAGGED


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------

@pytest.mark.parametrize("result,value", [(_passed(165), 165), (_failed(), None)])
def test_the_record_is_stored_whether_the_ladder_passed_or_failed(result, value):
    """It is two things at once: what the Brightness tab can quote, and the
    memo that stops the folder re-probing the same ladder on every open. A
    failed ladder needs the memo most of all."""
    entry = _entry()
    _confirm(entry, result)
    record = _confirm_record(entry)
    assert record is not None
    assert record["value"] == value


def test_the_record_carries_everything_the_next_open_compares():
    """core.detect.confirm.matches asks "is this the record of exactly this
    probe?" of the starting value, the confidence threshold, the crop box and
    the strip, so all four have to be in it."""
    entry = _entry()
    _confirm(entry, _passed(165))
    record = _confirm_record(entry)
    assert record["start_value"] == STORED
    assert record["conf_threshold"] == CONF
    assert record["crop_box"] == list(BOX)
    assert record["probe_time"] == PROBE_TIME


def test_evidence_that_is_missing_entirely_does_not_stop_the_record_being_written():
    """Evidence lives in the disposable .ocr-cache/ and can be gone while the
    value and its flags (in .ocr.json) are perfectly good."""
    entry = _entry()
    entry.evidence.clear()
    _confirm(entry, _passed(STORED))
    assert _confirm_record(entry) is not None
    assert entry.flags["brightness"] == ""


# --------------------------------------------------------------------------
# Results there is no longer a question for
# --------------------------------------------------------------------------

def test_no_result_changes_nothing():
    entry = _entry()
    project = _project(entry)
    apply_confirm(project, None)
    assert entry.brightness == Brightness(STORED, Source.DETECTED)
    assert entry.flags["brightness"] == FLAG_NARROW_PLATEAU
    assert _confirm_record(entry) is None


def test_a_cancelled_ladder_changes_nothing():
    """Cancellation is cooperative: what a cancelled ladder saw is a partial
    walk, and a partial walk is not an answer."""
    entry = _entry()
    _confirm(entry, ConfirmResult(value=165, probe_time=PROBE_TIME,
                                  rungs=(_rung(165, True),), cancelled=True))
    assert entry.brightness == Brightness(STORED, Source.DETECTED)
    assert entry.flags["brightness"] == FLAG_NARROW_PLATEAU
    assert _confirm_record(entry) is None


def test_a_result_for_a_file_that_is_gone_changes_nothing():
    entry = _entry()
    project = _project(entry)
    apply_confirm(project, ConfirmJobResult("other.mkv", _passed(165), BOX, STORED, CONF))
    assert entry.brightness == Brightness(STORED, Source.DETECTED)
    assert _confirm_record(entry) is None


def test_a_result_measured_on_a_crop_the_file_no_longer_has_changes_nothing():
    """The strip was cut from a region the file does not use any more, so
    what the engine read there says nothing about this file."""
    entry = _entry(crop_box=OTHER_BOX, value_crop_box=OTHER_BOX)
    _confirm(entry, _passed(165), crop_box=BOX)
    assert entry.brightness == Brightness(STORED, Source.DETECTED)
    assert entry.flags["brightness"] == FLAG_NARROW_PLATEAU
    assert _confirm_record(entry) is None


def test_a_result_for_a_file_with_no_crop_changes_nothing():
    entry = _entry(crop_box=None, value_crop_box=None)
    _confirm(entry, _passed(165))
    assert entry.brightness == Brightness(STORED, Source.DETECTED)
    assert _confirm_record(entry) is None


@pytest.mark.parametrize("source", [Source.MANUAL, Source.IMPORTED])
def test_a_brightness_the_user_owns_is_not_this_stage_to_confirm(source):
    """Rulings C1/C2: a MANUAL or IMPORTED value is the user's. Detection may
    not overwrite it, and confirming it would be claiming the same authority
    -- clearing the flag is a change to what the window says about a value
    nobody asked us about."""
    entry = _entry(source=source)
    _confirm(entry, _passed(165))
    assert entry.brightness == Brightness(STORED, source)
    assert entry.flags["brightness"] == FLAG_NARROW_PLATEAU
    assert _confirm_record(entry) is None


def test_a_file_with_no_brightness_at_all_changes_nothing():
    entry = _entry(value=None)
    _confirm(entry, _passed(165))
    assert entry.brightness is None
    assert _confirm_record(entry) is None


def test_a_stored_value_that_moved_since_the_job_was_submitted_changes_nothing():
    """The ladder started at 185 and answered a question about 185. The file
    stores 200 now -- a re-detection landed while the probe was queued -- and
    nothing the engine read is about that value."""
    entry = _entry(value=200)
    _confirm(entry, _passed(165), start_value=STORED)
    assert entry.brightness == Brightness(200, Source.DETECTED)
    assert entry.flags["brightness"] == FLAG_NARROW_PLATEAU
    assert _confirm_record(entry) is None


def test_a_stale_brightness_is_re_measured_not_confirmed():
    """The strip came from today's crop, but the value judged on it was
    measured on yesterday's (value_crop_box). A stale value counts as
    missing, and missing is not something a reading retires."""
    entry = _entry(value_crop_box=OTHER_BOX)
    _confirm(entry, _passed(STORED))
    assert entry.brightness == Brightness(STORED, Source.DETECTED)
    assert entry.evidence["brightness"]["value_crop_box"] == list(OTHER_BOX)
    assert entry.flags["brightness"] == FLAG_NARROW_PLATEAU
    assert _confirm_record(entry) is None
