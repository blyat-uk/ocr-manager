"""Detection stores what it measured, even when it doubts it.

A detector that produced a value it will not vouch for (a crop box built from
one subtitle frame, a brightness whose plateau is narrow) used to have its
value withheld: `entry.crop` stayed None while the evidence held the box, the
views drew that box as a proposal, and the file sat FLAGGED with nothing the
user could accept -- "Mark reviewed" is refused while a required value is
missing, so the one gesture that means "yes, that box" was not offered.

The rule pinned here: a measured value is stored with its flags, the file is
FLAGGED for it ("check crop + brightness"), and the user accepts it with Mark
reviewed, which turns it MANUAL. Nothing auto-applies that did not before --
a FLAGGED file is never startable (ruling B6) -- and a result that measured
nothing (no box, no crop to measure on, no text) still writes no value.
"""
from __future__ import annotations

import pytest

from core.detect.brightness import (
    DEFAULT_BRIGHTNESS,
    FLAG_COLOURED_TEXT,
    FLAG_DIM_TEXT,
    FLAG_ESCALATE,
    FLAG_NARROW_PLATEAU,
    FLAG_NEEDS_CROP,
    FLAG_NO_PLATEAU,
    FLAG_NO_TEXT,
    FLAG_RANGES_EMPTY,
    FLAG_THIN_EVIDENCE,
    BrightnessResult,
)
from core.detect.crop import (
    FLAG_CEILING_EXCEEDED,
    FLAG_LOW_AGREEMENT,
    FLAG_MULTIPLE_POSITIONS,
    FLAG_OUTLIER_DISCARDED,
    FLAG_STATIC_CONTENT,
    FLAG_TOP_POSITIONED,
    FLAG_UNKNOWN_REJECTION,
    FLAG_WATERMARK_UNCERTAIN,
    CropResult,
)
from core.jobs.apply import (
    apply_brightness,
    apply_crop,
    mark_reviewed,
    recompute_all,
)
from core.jobs.detect_jobs import BrightnessJobResult, CropJobResult
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

BOX = (576, 1876, 2688, 167)
UHD = Media(3840, 2160, 300.16, 25.0)


def _project(entry: FileEntry) -> Project:
    return Project(path="/proj", folder=FolderSettings(), files={entry.name: entry})


def _settle(project: Project) -> Project:
    recompute_all(project, pending={}, ranges_pending=False)
    return project


def _crop_result(*, box=BOX, flagged=None, agreed=1, probes=10) -> CropResult:
    return CropResult(box=box, envelope=box, agreed=agreed, probes_used=probes,
                      flagged=flagged, frame_size=(UHD.width, UHD.height),
                      hit_pts=[178.304], sample_pts=[178.304])


def _brightness_result(*, value=185, plateau=(180, 190), flagged=None) -> BrightnessResult:
    return BrightnessResult(value=value, plateau=plateau, seed=value, gate_floor=None,
                            flagged=flagged, curve=[(185, 0.9)])


def _apply_crop(entry: FileEntry, result: CropResult) -> Project:
    project = _project(entry)
    apply_crop(project, CropJobResult(entry.name, result, None))
    return _settle(project)


def _apply_brightness(entry: FileEntry, result: BrightnessResult) -> Project:
    project = _project(entry)
    apply_brightness(project, BrightnessJobResult(entry.name, result, {}, None, BOX))
    return _settle(project)


# --------------------------------------------------------------------------
# Crop: a box it measured is stored, whatever it thinks of it
# --------------------------------------------------------------------------

@pytest.mark.parametrize("flag", [FLAG_LOW_AGREEMENT, FLAG_TOP_POSITIONED, FLAG_MULTIPLE_POSITIONS,
                                  FLAG_WATERMARK_UNCERTAIN, FLAG_OUTLIER_DISCARDED])
def test_a_flagged_crop_box_is_stored_as_detected(flag):
    """The box the views already drew from the evidence is the stored value."""
    entry = FileEntry("ep.mkv", media=UHD)
    project = _apply_crop(entry, _crop_result(flagged=flag))
    assert entry.crop == Crop(*BOX, Source.DETECTED)
    assert entry.flags["crop"] == flag
    assert project.files["ep.mkv"].review == ReviewState.FLAGGED


@pytest.mark.parametrize("flag", [FLAG_UNKNOWN_REJECTION, FLAG_STATIC_CONTENT, FLAG_CEILING_EXCEEDED])
def test_a_crop_result_with_no_box_still_stores_nothing(flag):
    """The detector's three no-box outcomes measured nothing: a confirmed
    watermark, a union too tall for a subtitle, and the safety net. There is
    nothing to store and nothing for the user to accept."""
    entry = FileEntry("ep.mkv", media=UHD)
    _apply_crop(entry, _crop_result(box=None, flagged=flag))
    assert entry.crop is None


def test_a_clean_crop_result_still_proposes_the_file():
    entry = FileEntry("ep.mkv", media=UHD, brightness=Brightness(185, Source.MANUAL))
    project = _apply_crop(entry, _crop_result(agreed=8))
    assert entry.crop == Crop(*BOX, Source.DETECTED)
    assert project.files["ep.mkv"].review == ReviewState.PROPOSED


def test_a_flagged_crop_never_overwrites_a_value_the_user_set():
    mine = Crop(0, 100, 1920, 80, Source.MANUAL)
    entry = FileEntry("ep.mkv", media=UHD, crop=mine)
    _apply_crop(entry, _crop_result(flagged=FLAG_LOW_AGREEMENT))
    assert entry.crop == mine


def test_a_doubted_box_is_stored_but_the_file_is_not_startable():
    """Storing the box does not let a run use it: FLAGGED is not a ready state."""
    from app.controller import READY_STATES

    entry = FileEntry("ep.mkv", media=UHD, brightness=Brightness(185, Source.MANUAL))
    project = _apply_crop(entry, _crop_result(flagged=FLAG_WATERMARK_UNCERTAIN))
    assert entry.crop == Crop(*BOX, Source.DETECTED)
    assert project.files["ep.mkv"].review not in READY_STATES


# --------------------------------------------------------------------------
# Brightness: the same, for a value it actually measured
# --------------------------------------------------------------------------

@pytest.mark.parametrize("flag", [FLAG_NARROW_PLATEAU, FLAG_DIM_TEXT, FLAG_THIN_EVIDENCE,
                                  FLAG_NO_PLATEAU, FLAG_COLOURED_TEXT])
def test_a_flagged_brightness_value_is_stored_as_detected(flag):
    entry = FileEntry("ep.mkv", media=UHD, crop=Crop(*BOX, Source.MANUAL))
    project = _apply_brightness(entry, _brightness_result(flagged=flag))
    assert entry.brightness == Brightness(185, Source.DETECTED)
    assert entry.flags["brightness"] == flag
    assert project.files["ep.mkv"].review == ReviewState.FLAGGED
    assert entry.evidence["brightness"]["value_crop_box"] == list(BOX)


@pytest.mark.parametrize("flag", [FLAG_NEEDS_CROP, FLAG_RANGES_EMPTY, FLAG_NO_TEXT, FLAG_ESCALATE])
def test_a_brightness_that_measured_nothing_stores_no_value(flag):
    """needs-crop / ranges-empty? / no-text carry DEFAULT_BRIGHTNESS, not a
    reading; an "escalate" seed is a cheap attempt a full run is about to
    replace. None of them is something to show the user as measured."""
    entry = FileEntry("ep.mkv", media=UHD, crop=Crop(*BOX, Source.MANUAL))
    _apply_brightness(entry, _brightness_result(value=DEFAULT_BRIGHTNESS, plateau=None, flagged=flag))
    assert entry.brightness is None


def test_a_flagged_brightness_never_overwrites_a_value_the_user_set():
    mine = Brightness(210, Source.MANUAL)
    entry = FileEntry("ep.mkv", media=UHD, crop=Crop(*BOX, Source.MANUAL), brightness=mine)
    _apply_brightness(entry, _brightness_result(flagged=FLAG_NARROW_PLATEAU))
    assert entry.brightness == mine


# --------------------------------------------------------------------------
# The point of all of it: the user can accept what they are looking at
# --------------------------------------------------------------------------

def test_the_user_can_mark_a_doubted_crop_and_brightness_reviewed():
    """The bug this file exists for: a low-agreement crop left the file with
    no crop and no brightness, so "Mark reviewed" was refused ("set a crop
    and brightness first") and the correct box on screen could not be
    accepted."""
    from app.state_text import can_mark_reviewed, missing_required_values

    entry = FileEntry("ep.mkv", media=UHD)
    project = _project(entry)
    apply_crop(project, CropJobResult("ep.mkv", _crop_result(flagged=FLAG_LOW_AGREEMENT), None))
    apply_brightness(project, BrightnessJobResult(
        "ep.mkv", _brightness_result(flagged=FLAG_NARROW_PLATEAU), {}, None, BOX))
    _settle(project)

    assert entry.review == ReviewState.FLAGGED
    assert missing_required_values(entry) == []
    assert can_mark_reviewed(entry) is True

    mark_reviewed(project, "ep.mkv", True)
    assert entry.review == ReviewState.REVIEWED
    assert entry.crop == Crop(*BOX, Source.MANUAL)
    assert entry.brightness == Brightness(185, Source.MANUAL)


def test_accepting_does_not_erase_why_the_file_was_flagged():
    """The reasons stay stored beside the accepted value: the evidence views
    still say what the detector thought."""
    entry = FileEntry("ep.mkv", media=UHD, brightness=Brightness(185, Source.MANUAL))
    project = _apply_crop(entry, _crop_result(flagged=FLAG_LOW_AGREEMENT))
    mark_reviewed(project, "ep.mkv", True)
    assert entry.flags["crop"] == FLAG_LOW_AGREEMENT
    assert entry.evidence["crop"]["flagged"] == FLAG_LOW_AGREEMENT
