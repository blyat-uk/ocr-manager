"""Detector, proof and metadata jobs (core.jobs.detect_jobs) and the pure
functions that apply their results to a Project (core.jobs.apply).

Two things are pinned here.

1. The apply rules (rulings C1/C2): detection fills a value only when the
   field is empty or was itself detected/hinted, and only from an
   auto-applicable result; the user's manual and imported values are never
   overwritten; evidence and flags are always stored; a reviewed file is
   un-reviewed only when a detected/hinted value it was reviewed with
   changes; and compute_review_state's branches, including labels-only.

2. The jobs: each calls its detector (or the OCR API) with exactly the
   documented arguments, holds its engine leases for the whole call, passes
   cancellation through, returns its documented result type, and returns
   None when cancelled (the module's cancellation convention). Detectors
   and engines are fakes here; the one slow test runs the real ones on a
   reference episode.
"""
import inspect
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.detect import audio_profile as audio_profile_mod
from core.detect import brightness as brightness_mod
from core.detect import crop as crop_mod
from core.detect import ocr_view, vad
from core.detect.audio_profile import AudioProfile
from core.detect.brightness import BrightnessResult, StripSample
from core.detect.crop import CropResult, CropSample
from core.detect.ranges import pipeline as ranges_pipeline
from core.detect.ranges.config import MatchConfig, RangesConfig
from core.detect.ranges.pipeline import Block, ProgressEvent, RangesAnalysis
from core.detect.ranges.pipeline import FileEntry as RangesFile
from core.detect.tiles import choose_tiles
from core.jobs import JobContext, JobRunner, Lane
from core.jobs.apply import (
    FLAG_DIFFERS_FROM_HINT,
    apply_audio_profile,
    apply_brightness,
    apply_crop,
    apply_folder_change,
    apply_metadata,
    apply_ranges,
    brightness_is_stale,
    compute_review_state,
    copy_settings,
    mark_reviewed,
    paste_settings,
    recompute_all,
    set_manual_brightness,
    set_manual_crop,
    set_manual_time_ranges,
    set_skipped,
)
from core.jobs.detect_jobs import (
    THUMB_HEIGHT,
    AudioProfileJob,
    AudioProfileResult,
    BrightnessJob,
    BrightnessJobResult,
    CropJob,
    CropJobResult,
    MetadataJob,
    MetadataResult,
    ProofOcrJob,
    ProofResult,
    RangesJob,
    RangesJobResult,
    ThumbnailJob,
    ThumbnailResult,
    format_mss,
    proof_lines,
    proof_window,
)
from core.project import (
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
    from_json,
    load_project,
    ocr_call_for,
    save_project,
    to_json,
)
from videocr import api, engine_registry
from videocr import utils as vc_utils

pytestmark = pytest.mark.timeout(120)

WAIT = 10.0

PROJECT_DIR = "/proj"
NAMES = ("a.mp4", "b.mp4", "c.mp4")

OLD_BOX = (300, 800, 1300, 60)
NEW_BOX = (288, 786, 1344, 53)
OTHER_BRIGHTNESS = 230
NEW_BRIGHTNESS = 209


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def _project(folder: FolderSettings | None = None, names=NAMES) -> Project:
    return Project(path=PROJECT_DIR, folder=folder or FolderSettings(),
                   files={name: FileEntry(name=name) for name in names})


def _crop_result(box=NEW_BOX, flagged=None, sample_pts=(5.0, 12.0, 30.0), hit_pts=(12.0, 30.0)) -> CropResult:
    """A CropResult whose first probe (5.0) found nothing, as after a fallback
    round: sample_pts[0] is a no-text frame, hit_pts[0] a subtitle frame."""
    samples = [
        CropSample(time=t, boxes=((box[0], box[1], box[2], box[3]),) if (box and t in hit_pts) else (),
                   kept=bool(box) and t in hit_pts, lines=1 if (box and t in hit_pts) else 0)
        for t in sample_pts
    ]
    return CropResult(box=box, sample_pts=list(sample_pts), envelope=box, agreed=len(hit_pts),
                      probes_used=len(sample_pts), flagged=flagged, hit_pts=list(hit_pts),
                      frame_size=(1920, 1080), samples=samples)


STRIPS = [
    StripSample(time=100.0, is_text=True, glyph_level=235, background_level=40.0, stroke_px=3.1,
                lines=1, boxes=((10, 5, 200, 30),), gate_at_value=None),
    StripSample(time=200.0, is_text=True, glyph_level=240, background_level=120.0, stroke_px=2.4,
                lines=2, boxes=((10, 5, 200, 30), (12, 40, 180, 30)), gate_at_value=None),
    StripSample(time=300.0, is_text=False, glyph_level=None, background_level=150.0, stroke_px=None,
                lines=0, boxes=(), gate_at_value=True),
]


def _brightness_result(value=NEW_BRIGHTNESS, plateau=(190, 235), flagged=None, strips=STRIPS) -> BrightnessResult:
    return BrightnessResult(value=value, plateau=plateau, seed=229, gate_floor=150, flagged=flagged,
                            curve=[(185, 0.9), (190, 0.99), (235, 0.99)],
                            strips=list(strips), clutter_curve=[(185, 0.2), (190, 0.0)])


def _crop_job_result(file="a.mp4", hint=None, **kwargs) -> CropJobResult:
    return CropJobResult(file=file, result=_crop_result(**kwargs), hint=hint)


def _brightness_job_result(file="a.mp4", hint_value=None, crop_box=NEW_BOX, **kwargs) -> BrightnessJobResult:
    """A brightness result measured with `crop_box` (the crop the entry under
    test must still have for the result to apply)."""
    result = _brightness_result(**kwargs)
    return BrightnessJobResult(file=file, result=result, tiles=choose_tiles(result.strips, result.value),
                               hint_value=hint_value, crop_box=crop_box)


def _complete(entry: FileEntry) -> None:
    """Give an entry both required values from detection, so only the field
    under test decides its review state. The crop is the one brightness
    results are measured with by default."""
    entry.crop = Crop(*NEW_BOX, Source.DETECTED)
    entry.brightness = Brightness(OTHER_BRIGHTNESS, Source.DETECTED)


def _state(project: Project, name: str) -> ReviewState:
    recompute_all(project, pending={}, ranges_pending=False)
    return project.files[name].review


# --------------------------------------------------------------------------
# apply_crop
# --------------------------------------------------------------------------

@pytest.mark.parametrize("prior_source, overwritten", [
    (None, True),
    (Source.DETECTED, True),
    (Source.HINT, True),
    (Source.MANUAL, False),
    (Source.IMPORTED, False),
])
def test_apply_crop_writes_only_over_an_empty_detected_or_hinted_crop(prior_source, overwritten):
    project = _project()
    entry = project.files["a.mp4"]
    if prior_source is not None:
        entry.crop = Crop(*OLD_BOX, prior_source)
    r = _crop_job_result()

    apply_crop(project, r)

    if overwritten:
        assert entry.crop == Crop(*NEW_BOX, Source.DETECTED)
    else:
        assert entry.crop == Crop(*OLD_BOX, prior_source)
    # Evidence and flags are stored whether or not the value was written.
    assert entry.evidence["crop"] == r.result.to_evidence()
    assert entry.flags["crop"] == ""


@pytest.mark.parametrize("flagged", ["no-speech", "speech-probes-exhausted", "no-speech+speech-probes-exhausted"])
def test_apply_crop_writes_a_box_whose_flags_are_all_informational(flagged):
    project = _project()
    apply_crop(project, _crop_job_result(flagged=flagged))
    entry = project.files["a.mp4"]
    assert entry.crop == Crop(*NEW_BOX, Source.DETECTED)
    assert entry.flags["crop"] == flagged


@pytest.mark.parametrize("box, flagged", [
    (NEW_BOX, "low-agreement"),
    (NEW_BOX, "multiple-positions?"),
    (NEW_BOX, "top-positioned?"),
    (NEW_BOX, "no-speech+outlier-discarded?"),
    (None, "static-content"),
    (None, "ceiling-exceeded"),
])
@pytest.mark.parametrize("prior_source", [None, Source.DETECTED])
def test_apply_crop_never_writes_a_result_that_is_not_auto_applicable(box, flagged, prior_source):
    project = _project()
    entry = project.files["a.mp4"]
    _complete(entry)
    entry.crop = None if prior_source is None else Crop(*OLD_BOX, prior_source)
    before = entry.crop
    r = _crop_job_result(box=box, flagged=flagged, hit_pts=(12.0, 30.0) if box else ())
    assert not r.result.auto_applicable

    apply_crop(project, r)

    assert entry.crop == before
    assert entry.flags["crop"] == flagged
    assert entry.evidence["crop"] == r.result.to_evidence()
    assert _state(project, "a.mp4") == ReviewState.FLAGGED


@pytest.mark.parametrize("prior_source, expected", [
    (None, Crop(*NEW_BOX, Source.HINT)),
    (Source.DETECTED, Crop(*NEW_BOX, Source.HINT)),
    (Source.HINT, Crop(*NEW_BOX, Source.HINT)),
    (Source.MANUAL, Crop(*OLD_BOX, Source.MANUAL)),
    (Source.IMPORTED, Crop(*OLD_BOX, Source.IMPORTED)),
])
def test_a_hinted_crop_result_writes_with_source_hint(prior_source, expected):
    project = _project()
    entry = project.files["a.mp4"]
    if prior_source is not None:
        entry.crop = Crop(*OLD_BOX, prior_source)
    apply_crop(project, _crop_job_result(hint=(0.73, 0.05)))
    assert entry.crop == expected


def test_a_detected_crop_over_a_hinted_one_takes_source_detected():
    project = _project()
    entry = project.files["a.mp4"]
    entry.crop = Crop(*NEW_BOX, Source.HINT)
    apply_crop(project, _crop_job_result())
    assert entry.crop == Crop(*NEW_BOX, Source.DETECTED)


# --- sample_time -------------------------------------------------------------

def test_sample_time_is_the_first_hit_not_the_first_probe():
    project = _project()
    apply_crop(project, _crop_job_result(sample_pts=(5.0, 12.0, 30.0), hit_pts=(12.0, 30.0)))
    assert project.files["a.mp4"].sample_time == 12.0


def test_sample_time_is_set_from_hits_even_when_the_box_is_not_applied():
    project = _project()
    entry = project.files["a.mp4"]
    entry.crop = Crop(*OLD_BOX, Source.MANUAL)
    apply_crop(project, _crop_job_result(flagged="low-agreement", hit_pts=(12.0, 30.0)))
    assert entry.sample_time == 12.0


def test_an_existing_sample_time_survives_a_result_that_did_not_write_the_crop():
    project = _project()
    entry = project.files["a.mp4"]
    entry.crop = Crop(*OLD_BOX, Source.IMPORTED)
    entry.sample_time = 169.75          # migrated subtitle_position
    apply_crop(project, _crop_job_result())
    assert entry.crop.source == Source.IMPORTED
    assert entry.sample_time == 169.75


@pytest.mark.parametrize("prior_source", [None, Source.DETECTED, Source.HINT])
def test_a_crop_written_by_this_apply_replaces_the_sample_time(prior_source):
    project = _project()
    entry = project.files["a.mp4"]
    if prior_source is not None:
        entry.crop = Crop(*OLD_BOX, prior_source)
    entry.sample_time = 169.75
    apply_crop(project, _crop_job_result(hit_pts=(12.0, 30.0)))
    assert entry.sample_time == 12.0


def test_no_hits_leave_the_sample_time_alone():
    project = _project()
    entry = project.files["a.mp4"]
    apply_crop(project, _crop_job_result(box=None, flagged="static-content", hit_pts=()))
    assert entry.sample_time is None
    entry.sample_time = 50.0
    apply_crop(project, _crop_job_result(box=None, flagged="static-content", hit_pts=()))
    assert entry.sample_time == 50.0


# --- review reset ----------------------------------------------------------------

@pytest.mark.parametrize("prior_source", [Source.DETECTED, Source.HINT])
def test_a_changed_detected_value_un_reviews_the_file(prior_source):
    project = _project()
    entry = project.files["a.mp4"]
    _complete(entry)
    entry.crop = Crop(*OLD_BOX, prior_source)
    entry.review = ReviewState.REVIEWED

    apply_crop(project, _crop_job_result())

    assert entry.crop == Crop(*NEW_BOX, Source.DETECTED)
    assert entry.review != ReviewState.REVIEWED
    assert _state(project, "a.mp4") == ReviewState.PROPOSED


def test_the_same_detected_value_again_keeps_the_file_reviewed():
    project = _project()
    entry = project.files["a.mp4"]
    _complete(entry)
    entry.crop = Crop(*NEW_BOX, Source.DETECTED)
    entry.review = ReviewState.REVIEWED
    apply_crop(project, _crop_job_result())
    assert entry.review == ReviewState.REVIEWED


def _fill_crop(project):
    apply_crop(project, _crop_job_result())
    return project.files["a.mp4"].crop


def _fill_brightness(project):
    apply_brightness(project, _brightness_job_result())
    return project.files["a.mp4"].brightness


def _fill_ranges(project):
    apply_ranges(project, _ranges_result({"a.mp4": [(None, "01:30")]}))
    return project.files["a.mp4"].time_ranges


@pytest.mark.parametrize("field, fill", [
    ("crop", _fill_crop), ("brightness", _fill_brightness), ("time_ranges", _fill_ranges)])
def test_detection_filling_an_empty_field_un_reviews_the_file(field, fill):
    """The user never saw a value that detection put into an empty field."""
    project = _project()
    entry = project.files["a.mp4"]
    entry.crop = Crop(*NEW_BOX, Source.MANUAL)
    entry.brightness = Brightness(200, Source.MANUAL)
    entry.time_ranges = TimeRanges([TimeRange("01:00", None)], Source.MANUAL)
    setattr(entry, field, None)
    entry.review = ReviewState.REVIEWED

    assert fill(project) is not None

    assert entry.review != ReviewState.REVIEWED
    assert _state(project, "a.mp4") == ReviewState.PROPOSED


def test_whole_file_detected_into_an_empty_ranges_field_changes_nothing_and_keeps_review():
    project = _project()
    entry = project.files["a.mp4"]
    entry.review = ReviewState.REVIEWED
    apply_ranges(project, _ranges_result({}))
    assert entry.time_ranges is None
    assert entry.review == ReviewState.REVIEWED


@pytest.mark.parametrize("source", [Source.MANUAL, Source.IMPORTED])
def test_a_reviewed_file_with_the_users_values_stays_reviewed_after_every_detection(source):
    project = _project()
    entry = project.files["a.mp4"]
    entry.crop = Crop(*OLD_BOX, source)
    entry.brightness = Brightness(OTHER_BRIGHTNESS, source)
    entry.time_ranges = TimeRanges([TimeRange("01:00", "20:00")], source)
    entry.review = ReviewState.REVIEWED

    apply_crop(project, _crop_job_result())
    apply_crop(project, _crop_job_result(flagged="low-agreement"))
    apply_brightness(project, _brightness_job_result(crop_box=OLD_BOX))
    assert "brightness" in entry.evidence            # applied (not dropped as stale), value untouched
    apply_brightness(project, _brightness_job_result(value=180, plateau=(170, 200), hint_value=250,
                                                     crop_box=OLD_BOX))
    assert entry.flags["brightness"] == FLAG_DIFFERS_FROM_HINT
    apply_ranges(project, RangesJobResult(RangesAnalysis(
        keep={"a.mp4": [(None, "01:30"), ("03:00", None)]}, blocks={}, durations={"a.mp4": 1418.0})))
    apply_ranges(project, RangesJobResult(RangesAnalysis(keep={}, blocks={}, durations={"a.mp4": 1418.0})))

    assert entry.crop == Crop(*OLD_BOX, source)
    assert entry.brightness == Brightness(OTHER_BRIGHTNESS, source)
    assert entry.time_ranges == TimeRanges([TimeRange("01:00", "20:00")], source)
    assert entry.review == ReviewState.REVIEWED
    assert _state(project, "a.mp4") == ReviewState.REVIEWED


# --------------------------------------------------------------------------
# apply_brightness
# --------------------------------------------------------------------------

@pytest.mark.parametrize("prior_source, overwritten", [
    (None, True),
    (Source.DETECTED, True),
    (Source.HINT, True),
    (Source.MANUAL, False),
    (Source.IMPORTED, False),
])
def test_apply_brightness_writes_only_over_an_empty_detected_or_hinted_value(prior_source, overwritten):
    project = _project()
    entry = project.files["a.mp4"]
    entry.crop = Crop(*NEW_BOX, Source.DETECTED)
    if prior_source is not None:
        entry.brightness = Brightness(OTHER_BRIGHTNESS, prior_source)
    r = _brightness_job_result()

    apply_brightness(project, r)

    if overwritten:
        assert entry.brightness == Brightness(NEW_BRIGHTNESS, Source.DETECTED)
    else:
        assert entry.brightness == Brightness(OTHER_BRIGHTNESS, prior_source)
    expected_evidence = {**r.result.to_evidence(), "tiles": r.tiles, "crop_box": list(NEW_BOX)}
    if overwritten:                                     # the box the stored value was measured on
        expected_evidence["value_crop_box"] = list(NEW_BOX)
    assert entry.evidence["brightness"] == expected_evidence
    assert entry.evidence["brightness"]["tiles"]      # the fixture strips give tiles
    assert entry.flags["brightness"] == ""


def test_brightness_with_only_the_informational_flag_is_written():
    project = _project()
    project.files["a.mp4"].crop = Crop(*NEW_BOX, Source.DETECTED)
    apply_brightness(project, _brightness_job_result(flagged="no-clean-threshold"))
    entry = project.files["a.mp4"]
    assert entry.brightness == Brightness(NEW_BRIGHTNESS, Source.DETECTED)
    assert entry.flags["brightness"] == "no-clean-threshold"


@pytest.mark.parametrize("flagged", ["escalate", "no-plateau?", "narrow-plateau?", "dim-text?",
                                     "no-clean-threshold+thin-evidence?", "needs-crop"])
def test_brightness_that_is_not_auto_applicable_is_stored_as_evidence_only(flagged):
    project = _project()
    entry = project.files["a.mp4"]
    _complete(entry)
    r = _brightness_job_result(value=215, plateau=None, flagged=flagged)
    assert not r.result.auto_applicable

    apply_brightness(project, r)

    assert entry.brightness == Brightness(OTHER_BRIGHTNESS, Source.DETECTED)
    assert entry.flags["brightness"] == flagged
    assert entry.evidence["brightness"]["value"] == 215
    assert _state(project, "a.mp4") == ReviewState.FLAGGED


def test_an_escalating_cheap_result_does_not_fill_an_empty_brightness():
    project = _project()
    project.files["a.mp4"].crop = Crop(*NEW_BOX, Source.DETECTED)
    apply_brightness(project, _brightness_job_result(value=240, plateau=None, flagged="escalate"))
    entry = project.files["a.mp4"]
    assert entry.brightness is None
    assert entry.flags["brightness"] == "escalate"


@pytest.mark.parametrize("hint, plateau, flagged, expected", [
    (None, (180, 230), None, ""),
    (200, (180, 230), None, ""),
    (180, (180, 230), None, ""),
    (230, (180, 230), None, ""),
    (231, (180, 230), None, FLAG_DIFFERS_FROM_HINT),
    (240, (180, 230), None, FLAG_DIFFERS_FROM_HINT),
    (170, (180, 230), "no-clean-threshold", "no-clean-threshold+" + FLAG_DIFFERS_FROM_HINT),
    (200, None, "no-plateau?", "no-plateau?+" + FLAG_DIFFERS_FROM_HINT),
    (None, None, "no-plateau?", "no-plateau?"),
])
def test_differs_from_hint_is_added_when_the_plateau_does_not_contain_the_hint(hint, plateau, flagged, expected):
    project = _project()
    entry = project.files["a.mp4"]
    _complete(entry)
    r = _brightness_job_result(value=210, plateau=plateau, flagged=flagged, hint_value=hint)

    apply_brightness(project, r)

    assert entry.flags["brightness"] == expected
    # The flag lives on the file, not in the detector's own evidence.
    assert entry.evidence["brightness"]["flagged"] == flagged
    blocking = expected not in ("", "no-clean-threshold")
    assert _state(project, "a.mp4") == (ReviewState.FLAGGED if blocking else ReviewState.PROPOSED)


def test_a_verified_value_outside_the_hint_is_written_and_flagged():
    project = _project()
    entry = project.files["a.mp4"]
    entry.crop = Crop(*NEW_BOX, Source.DETECTED)
    apply_brightness(project, _brightness_job_result(value=210, plateau=(180, 230), hint_value=245))
    assert entry.brightness == Brightness(210, Source.HINT)
    assert entry.flags["brightness"] == FLAG_DIFFERS_FROM_HINT
    assert _state(project, "a.mp4") == ReviewState.FLAGGED


@pytest.mark.parametrize("hint_value, source", [(None, Source.DETECTED), (200, Source.HINT)])
@pytest.mark.parametrize("prior_source", [None, Source.DETECTED, Source.HINT])
def test_a_hint_driven_brightness_is_written_with_source_hint(hint_value, source, prior_source):
    project = _project()
    entry = project.files["a.mp4"]
    entry.crop = Crop(*NEW_BOX, Source.MANUAL)
    if prior_source is not None:
        entry.brightness = Brightness(OTHER_BRIGHTNESS, prior_source)
    apply_brightness(project, _brightness_job_result(hint_value=hint_value))
    assert entry.brightness == Brightness(NEW_BRIGHTNESS, source)


@pytest.mark.parametrize("source", [Source.MANUAL, Source.IMPORTED])
def test_a_hint_driven_brightness_never_overwrites_the_users_value(source):
    project = _project()
    entry = project.files["a.mp4"]
    entry.crop = Crop(*NEW_BOX, Source.MANUAL)
    entry.brightness = Brightness(OTHER_BRIGHTNESS, source)
    apply_brightness(project, _brightness_job_result(hint_value=200))
    assert entry.brightness == Brightness(OTHER_BRIGHTNESS, source)


# --- stale results (ruling 1) ---------------------------------------------------------

@pytest.mark.parametrize("current_crop", [
    None,                                               # crop cleared (or never set) meanwhile
    Crop(*OLD_BOX, Source.MANUAL),                      # the user edited the crop while the job ran
    Crop(288, 786, 1344, 54, Source.DETECTED),          # re-detected one row taller
])
@pytest.mark.parametrize("hint_value", [None, 250])
def test_a_brightness_result_measured_on_another_crop_is_dropped_entirely(current_crop, hint_value):
    project = _project()
    entry = project.files["a.mp4"]
    entry.crop = current_crop
    entry.brightness = Brightness(OTHER_BRIGHTNESS, Source.DETECTED)
    entry.flags = {"brightness": "escalate"}
    entry.evidence = {"brightness": {"value": OTHER_BRIGHTNESS}}
    entry.review = ReviewState.REVIEWED
    before = to_json(project)

    apply_brightness(project, _brightness_job_result(crop_box=NEW_BOX, hint_value=hint_value))

    assert to_json(project) == before


@pytest.mark.parametrize("current_crop", [None, Crop(*NEW_BOX, Source.DETECTED)])
def test_a_brightness_result_measured_without_a_crop_is_dropped(current_crop):
    project = _project()
    project.files["a.mp4"].crop = current_crop
    before = to_json(project)
    apply_brightness(project, _brightness_job_result(value=230, plateau=None, flagged="needs-crop", strips=[],
                                                     crop_box=None))
    assert to_json(project) == before


@pytest.mark.parametrize("source", [Source.DETECTED, Source.HINT, Source.MANUAL, Source.IMPORTED])
def test_a_brightness_result_applies_when_the_crop_coordinates_still_match(source):
    project = _project()
    entry = project.files["a.mp4"]
    entry.crop = Crop(*NEW_BOX, source)
    apply_brightness(project, _brightness_job_result(crop_box=NEW_BOX))
    assert entry.brightness == Brightness(NEW_BRIGHTNESS, Source.DETECTED)
    assert "brightness" in entry.evidence


def test_a_changed_detected_brightness_un_reviews_the_file():
    project = _project()
    entry = project.files["a.mp4"]
    _complete(entry)
    entry.review = ReviewState.REVIEWED
    apply_brightness(project, _brightness_job_result())
    assert entry.brightness == Brightness(NEW_BRIGHTNESS, Source.DETECTED)
    assert entry.review != ReviewState.REVIEWED


# --------------------------------------------------------------------------
# apply_ranges
# --------------------------------------------------------------------------

BLOCKS_A = [Block(0.0, 90.0, "intro", 3, 0.9), Block(1300.0, 1418.0, "outro", 2, 0.7)]


def _ranges_result(keep, blocks=None, durations=None) -> RangesJobResult:
    durations = durations if durations is not None else {name: 1418.0 for name in NAMES}
    return RangesJobResult(RangesAnalysis(keep=keep, blocks=blocks or {}, durations=durations))


@pytest.mark.parametrize("prior", [None, TimeRanges([TimeRange(None, "05:00")], Source.DETECTED)])
def test_apply_ranges_writes_detected_keep_ranges(prior):
    project = _project()
    entry = project.files["a.mp4"]
    entry.time_ranges = prior
    apply_ranges(project, _ranges_result({"a.mp4": [(None, "01:30"), ("03:00", None)]},
                                         blocks={"a.mp4": BLOCKS_A}))
    assert entry.time_ranges == TimeRanges([TimeRange(None, "01:30"), TimeRange("03:00", None)], Source.DETECTED)
    assert entry.evidence["ranges"] == {
        "blocks": [
            {"start_sec": 0.0, "end_sec": 90.0, "kind": "intro", "matched_files": 3, "score": 0.9},
            {"start_sec": 1300.0, "end_sec": 1418.0, "kind": "outro", "matched_files": 2, "score": 0.7},
        ],
        "duration": 1418.0,
    }


def test_a_file_the_analysis_found_no_keep_ranges_for_is_whole_file():
    """analyse_detailed omits such a file from keep and blocks; it was still
    analysed (it has a duration), so a detected range it had is cleared."""
    project = _project()
    b = project.files["b.mp4"]
    b.time_ranges = TimeRanges([TimeRange(None, "05:00")], Source.DETECTED)

    apply_ranges(project, _ranges_result({"a.mp4": [(None, "01:30")]}, blocks={"a.mp4": BLOCKS_A}))

    assert b.time_ranges is None
    assert b.evidence["ranges"] == {"blocks": [], "duration": 1418.0}


def test_an_empty_keep_list_is_whole_file():
    project = _project()
    entry = project.files["a.mp4"]
    entry.time_ranges = TimeRanges([TimeRange(None, "05:00")], Source.DETECTED)
    apply_ranges(project, _ranges_result({"a.mp4": []}))
    assert entry.time_ranges is None


@pytest.mark.parametrize("source", [Source.MANUAL, Source.IMPORTED])
def test_apply_ranges_never_touches_the_users_ranges_but_stores_evidence(source):
    project = _project()
    entry = project.files["a.mp4"]
    mine = TimeRanges([TimeRange("02:33", "21:20")], source)
    entry.time_ranges = mine
    apply_ranges(project, _ranges_result({"a.mp4": [(None, "01:30")]}, blocks={"a.mp4": BLOCKS_A}))
    assert entry.time_ranges == mine
    assert len(entry.evidence["ranges"]["blocks"]) == 2


def test_apply_ranges_ignores_files_it_did_not_analyse_and_files_not_in_the_project():
    project = _project()
    c = project.files["c.mp4"]
    c.time_ranges = TimeRanges([TimeRange(None, "05:00")], Source.DETECTED)
    apply_ranges(project, _ranges_result({"a.mp4": [(None, "01:30")], "gone.mp4": [(None, "01:00")]},
                                         durations={"a.mp4": 1418.0, "b.mp4": 1400.0, "gone.mp4": 1300.0}))
    assert c.time_ranges == TimeRanges([TimeRange(None, "05:00")], Source.DETECTED)
    assert "ranges" not in c.evidence
    assert "gone.mp4" not in project.files


def test_changed_detected_ranges_un_review_the_file():
    project = _project()
    entry = project.files["a.mp4"]
    _complete(entry)
    entry.time_ranges = TimeRanges([TimeRange(None, "05:00")], Source.DETECTED)
    entry.review = ReviewState.REVIEWED
    apply_ranges(project, _ranges_result({}))
    assert entry.time_ranges is None
    assert entry.review != ReviewState.REVIEWED


# --------------------------------------------------------------------------
# apply_metadata / apply_audio_profile
# --------------------------------------------------------------------------

def test_apply_metadata_sets_media_and_ignores_unknown_files():
    project = _project()
    apply_metadata(project, MetadataResult("a.mp4", 1920, 1080, 1418.5, 23.976))
    apply_metadata(project, MetadataResult("gone.mp4", 1, 1, 1.0, 1.0))
    assert project.files["a.mp4"].media == Media(1920, 1080, 1418.5, 23.976)
    assert "gone.mp4" not in project.files


def test_apply_audio_profile_stores_envelope_speech_and_duration_as_evidence():
    project = _project()
    profile = AudioProfile(duration=1418.0, envelope=[0.0, 0.5, 1.0], speech=[(1.5, 3.25), (10.0, 12.0)])
    apply_audio_profile(project, AudioProfileResult("a.mp4", profile))
    entry = project.files["a.mp4"]
    assert entry.evidence["audio"] == {"envelope": [0.0, 0.5, 1.0], "speech": [[1.5, 3.25], [10.0, 12.0]],
                                       "duration": 1418.0}
    assert entry.crop is None and entry.brightness is None and entry.time_ranges is None


# --------------------------------------------------------------------------
# Cancelled results
# --------------------------------------------------------------------------

@pytest.mark.parametrize("apply", [apply_metadata, apply_crop, apply_brightness, apply_ranges, apply_audio_profile])
def test_every_apply_ignores_a_cancelled_job_result(apply):
    project = _project()
    before = to_json(project)
    apply(project, None)
    assert to_json(project) == before


@pytest.mark.parametrize("flagged", ["cancelled", "no-speech+cancelled"])
def test_apply_crop_ignores_a_result_the_detector_marked_cancelled(flagged):
    project = _project()
    before = to_json(project)
    apply_crop(project, _crop_job_result(flagged=flagged))
    assert to_json(project) == before


def test_apply_brightness_ignores_a_result_the_detector_marked_cancelled():
    project = _project()
    project.files["a.mp4"].crop = Crop(*NEW_BOX, Source.DETECTED)    # measured on the current crop: not stale
    before = to_json(project)
    apply_brightness(project, _brightness_job_result(value=230, plateau=None, flagged="cancelled", strips=[]))
    assert to_json(project) == before


# --------------------------------------------------------------------------
# Evidence round-trips through the store
# --------------------------------------------------------------------------

def test_applied_evidence_is_json_serialisable_and_round_trips(tmp_path):
    project = _project()
    project.path = str(tmp_path)
    apply_metadata(project, MetadataResult("a.mp4", 1920, 1080, 1418.0, 23.976))
    apply_crop(project, _crop_job_result())
    apply_brightness(project, _brightness_job_result(hint_value=250))
    apply_ranges(project, _ranges_result({"a.mp4": [(None, "01:30")]}, blocks={
        "a.mp4": [Block(np.float64(0.0), np.float64(90.0), "intro", 3, np.float32(0.9))]}))
    apply_audio_profile(project, AudioProfileResult("a.mp4", AudioProfile(1418.0, [0.1], [(1.0, 2.0)])))

    text = json.dumps(to_json(project))
    restored = from_json(json.loads(text), str(tmp_path))
    assert to_json(restored) == json.loads(text)
    save_project(project)


# --------------------------------------------------------------------------
# compute_review_state
# --------------------------------------------------------------------------

def _ready_entry() -> FileEntry:
    entry = FileEntry(name="a.mp4")
    entry.crop = Crop(*NEW_BOX, Source.DETECTED)
    entry.brightness = Brightness(NEW_BRIGHTNESS, Source.DETECTED)
    entry.flags = {"crop": "", "brightness": ""}
    return entry


DIALOGUE = FolderSettings(dialogue_enabled=True, labels_enabled=True)
DIALOGUE_ONLY = FolderSettings(dialogue_enabled=True, labels_enabled=False)
LABELS_ONLY = FolderSettings(dialogue_enabled=False, labels_enabled=True)


def _review(entry, folder=DIALOGUE, pending=(), ranges_pending=False):
    return compute_review_state(entry, folder, detections_pending=set(pending), ranges_pending=ranges_pending)


@pytest.mark.parametrize("missing", ["crop", "brightness"])
def test_a_missing_required_value_overrides_reviewed(missing):
    entry = _ready_entry()
    entry.review = ReviewState.REVIEWED
    setattr(entry, missing, None)
    other = "brightness" if missing == "crop" else "crop"
    assert _review(entry) == ReviewState.FLAGGED
    assert _review(entry, pending={missing}) == ReviewState.PENDING
    assert _review(entry, pending={other}) == ReviewState.FLAGGED
    assert _review(entry, ranges_pending=True) == ReviewState.FLAGGED


def test_missing_values_are_judged_crop_first():
    entry = FileEntry(name="a.mp4", review=ReviewState.REVIEWED)
    assert _review(entry, pending={"crop"}) == ReviewState.PENDING           # brightness waits for the crop
    assert _review(entry, pending={"brightness"}) == ReviewState.FLAGGED     # the crop itself is the problem
    assert _review(entry, pending={"crop", "brightness"}) == ReviewState.PENDING


def test_a_reviewed_labels_only_file_needs_no_values():
    entry = FileEntry(name="a.mp4", review=ReviewState.REVIEWED, flags={"crop": "low-agreement"})
    assert _review(entry, LABELS_ONLY) == ReviewState.REVIEWED


def test_reviewed_wins_over_pending_detection_and_pending_ranges():
    entry = _ready_entry()
    entry.review = ReviewState.REVIEWED
    assert _review(entry, pending={"crop", "brightness"}, ranges_pending=True) == ReviewState.REVIEWED


@pytest.mark.parametrize("detector", ["crop", "brightness"])
@pytest.mark.parametrize("source, expected", [
    (Source.DETECTED, ReviewState.PENDING),
    (Source.HINT, ReviewState.PENDING),
    (Source.MANUAL, ReviewState.PROPOSED),
    (Source.IMPORTED, ReviewState.PROPOSED),
])
def test_a_pending_detector_holds_only_a_detected_or_hinted_value(detector, source, expected):
    entry = _ready_entry()
    getattr(entry, detector).source = source
    assert _review(entry, pending={detector}) == expected


@pytest.mark.parametrize("detector, flag", [("crop", "low-agreement"), ("brightness", "escalate"),
                                            ("brightness", FLAG_DIFFERS_FROM_HINT)])
@pytest.mark.parametrize("source", [Source.DETECTED, Source.HINT])
def test_step_2_reviewed_comes_before_pending_and_blocking_flags(detector, flag, source):
    """As ruled. The API never stores REVIEWED beside a blocking flag on a
    detected value (mark_reviewed accepts it as MANUAL; the setters and
    applies do not set or keep REVIEWED then), so this state is built by hand."""
    entry = _ready_entry()
    entry.review = ReviewState.REVIEWED
    getattr(entry, detector).source = source
    entry.flags[detector] = flag
    assert _review(entry) == ReviewState.REVIEWED
    assert _review(entry, pending={detector}) == ReviewState.REVIEWED
    entry.review = ReviewState.FLAGGED
    assert _review(entry) == ReviewState.FLAGGED
    assert _review(entry, pending={detector}) == ReviewState.PENDING


# --- explicit acceptance: mark_reviewed ---------------------------------------------------

def _flagged_brightness_project(source=Source.DETECTED, flag="escalate", folder=None):
    """a.mp4: a clean detected crop and a brightness from detection that carries
    a blocking flag (the value was written by an earlier clean result)."""
    project = _project(folder=folder)
    apply_crop(project, _crop_job_result())
    apply_brightness(project, _brightness_job_result(hint_value=200 if source == Source.HINT else None))
    apply_brightness(project, _brightness_job_result(value=240, plateau=None, flagged=flag,
                                                     hint_value=200 if source == Source.HINT else None))
    entry = project.files["a.mp4"]
    assert entry.brightness == Brightness(NEW_BRIGHTNESS, source)
    return project, entry


@pytest.mark.parametrize("source", [Source.DETECTED, Source.HINT])
def test_mark_reviewed_accepts_a_flagged_detected_value_as_manual(source):
    project, entry = _flagged_brightness_project(source)
    assert _state(project, "a.mp4") == ReviewState.FLAGGED
    flags, evidence = dict(entry.flags), json.loads(json.dumps(entry.evidence))

    mark_reviewed(project, "a.mp4")

    assert entry.brightness == Brightness(NEW_BRIGHTNESS, Source.MANUAL)
    assert entry.crop == Crop(*NEW_BOX, Source.DETECTED)                 # not flagged: not converted
    assert entry.review == ReviewState.REVIEWED
    assert entry.flags == flags and entry.evidence == evidence            # still shown by the inspector
    assert _state(project, "a.mp4") == ReviewState.REVIEWED

    apply_brightness(project, _brightness_job_result(value=180, plateau=(170, 200)))   # later detection
    assert entry.brightness == Brightness(NEW_BRIGHTNESS, Source.MANUAL)
    assert _state(project, "a.mp4") == ReviewState.REVIEWED


def test_mark_reviewed_accepts_a_flagged_detected_crop_too():
    project = _project()
    entry = project.files["a.mp4"]
    _complete(entry)
    apply_crop(project, _crop_job_result(box=OLD_BOX, flagged="low-agreement"))
    mark_reviewed(project, "a.mp4")
    assert entry.crop == Crop(*NEW_BOX, Source.MANUAL)
    assert entry.brightness == Brightness(OTHER_BRIGHTNESS, Source.DETECTED)
    assert entry.flags["crop"] == "low-agreement"
    assert _state(project, "a.mp4") == ReviewState.REVIEWED


@pytest.mark.parametrize("flag", ["", "no-clean-threshold"])
def test_mark_reviewed_leaves_unflagged_detected_values_detected(flag):
    project = _project()
    apply_crop(project, _crop_job_result(flagged="no-speech"))
    apply_brightness(project, _brightness_job_result(flagged=flag or None))
    mark_reviewed(project, "a.mp4")
    entry = project.files["a.mp4"]
    assert (entry.crop.source, entry.brightness.source) == (Source.DETECTED, Source.DETECTED)
    assert _state(project, "a.mp4") == ReviewState.REVIEWED


def test_mark_reviewed_converts_nothing_labels_only_does_not_need():
    project, entry = _flagged_brightness_project(folder=FolderSettings(dialogue_enabled=False, labels_enabled=True))
    mark_reviewed(project, "a.mp4")
    assert entry.brightness.source == Source.DETECTED
    assert _state(project, "a.mp4") == ReviewState.REVIEWED


def test_mark_reviewed_does_not_accept_a_stale_brightness():
    project, entry = _flagged_brightness_project()
    apply_crop(project, _crop_job_result(box=OLD_BOX))                     # brightness measured on NEW_BOX
    mark_reviewed(project, "a.mp4")
    assert entry.brightness.source == Source.DETECTED
    assert entry.review == ReviewState.FLAGGED                             # stored: the computed state
    assert _state(project, "a.mp4") == ReviewState.FLAGGED


@pytest.mark.parametrize("unstale", ["set_manual_crop", "paste"])
@pytest.mark.parametrize("recompute_between", [False, True])
def test_restoring_the_crop_after_marking_does_not_review_a_flagged_brightness(unstale, recompute_between):
    """repro_c: the brightness was stale (not accepted) when the file was marked;
    putting the crop back un-stales it, and its blocking flag must still count."""
    project, entry = _flagged_brightness_project()
    apply_crop(project, _crop_job_result(box=OLD_BOX))
    recompute_all(project, pending={}, ranges_pending=False)
    mark_reviewed(project, "a.mp4")
    assert entry.review != ReviewState.REVIEWED
    if recompute_between:
        recompute_all(project, pending={}, ranges_pending=False)

    if unstale == "set_manual_crop":
        set_manual_crop(project, "a.mp4", NEW_BOX)
    else:
        paste_settings(project, "a.mp4", {"crop": NEW_BOX, "brightness": None, "time_ranges": None})

    assert not brightness_is_stale(entry)
    assert entry.brightness == Brightness(NEW_BRIGHTNESS, Source.DETECTED)
    assert entry.review == ReviewState.FLAGGED
    assert _state(project, "a.mp4") == ReviewState.FLAGGED
    mark_reviewed(project, "a.mp4")                                        # now it can be accepted
    assert entry.brightness.source == Source.MANUAL
    assert _state(project, "a.mp4") == ReviewState.REVIEWED


def test_unmarking_does_not_revert_accepted_values():
    project, entry = _flagged_brightness_project()
    mark_reviewed(project, "a.mp4")
    mark_reviewed(project, "a.mp4", False)
    assert entry.brightness.source == Source.MANUAL
    assert entry.review != ReviewState.REVIEWED
    assert _state(project, "a.mp4") == ReviewState.PROPOSED


def test_unmarking_converts_nothing():
    project, entry = _flagged_brightness_project()
    mark_reviewed(project, "a.mp4", False)
    assert entry.brightness.source == Source.DETECTED
    assert _state(project, "a.mp4") == ReviewState.FLAGGED


# --- setters: REVIEWED only when no other required field is flagged --------------------------

@pytest.mark.parametrize("edit", [                 # crop edits keep the box: the brightness must not go stale
    lambda project: set_manual_crop(project, "a.mp4", NEW_BOX),
    lambda project: set_manual_time_ranges(project, "a.mp4", [("01:00", None)]),
    lambda project: paste_settings(project, "a.mp4", {"crop": NEW_BOX, "time_ranges": None}),
])
@pytest.mark.parametrize("stored", [ReviewState.FLAGGED, ReviewState.PROPOSED, ReviewState.REVIEWED])
def test_an_edit_stores_the_computed_state_while_another_required_field_is_flagged(edit, stored):
    project, entry = _flagged_brightness_project()
    entry.review = stored               # REVIEWED here is built by hand: the API never stores it beside the flag

    edit(project)

    assert entry.review == ReviewState.FLAGGED                             # the computed state, never REVIEWED
    assert entry.brightness.source == Source.DETECTED
    assert _state(project, "a.mp4") == ReviewState.FLAGGED
    mark_reviewed(project, "a.mp4")                                        # the user accepts the brightness
    assert _state(project, "a.mp4") == ReviewState.REVIEWED


@pytest.mark.parametrize("stored", [ReviewState.PROPOSED, ReviewState.REVIEWED])
def test_set_manual_brightness_stores_the_computed_state_while_the_crop_is_flagged(stored):
    project = _project()
    entry = project.files["a.mp4"]
    _complete(entry)
    apply_crop(project, _crop_job_result(box=OLD_BOX, flagged="low-agreement"))
    entry.review = stored
    set_manual_brightness(project, "a.mp4", 205)
    assert entry.review == ReviewState.FLAGGED
    assert _state(project, "a.mp4") == ReviewState.FLAGGED


def test_mark_reviewed_stores_reviewed_when_a_stale_brightness_carries_no_blocking_flag():
    project = _project()
    entry = project.files["a.mp4"]
    apply_crop(project, _crop_job_result())
    apply_brightness(project, _brightness_job_result())
    apply_crop(project, _crop_job_result(box=OLD_BOX))
    mark_reviewed(project, "a.mp4")
    assert entry.review == ReviewState.REVIEWED                            # nothing flagged is left
    assert _state(project, "a.mp4") == ReviewState.FLAGGED                 # step 1: stale still counts


# --- folder mode switch ---------------------------------------------------------------------------

LABELS_ONLY_FOLDER = FolderSettings(dialogue_enabled=False, labels_enabled=True)
DIALOGUE_FOLDER = FolderSettings(dialogue_enabled=True, labels_enabled=True)


def _switch(project, new):
    old = project.folder
    project.folder = new
    apply_folder_change(project, old, new)


def test_repro_folder_switching_back_to_dialogue_does_not_keep_a_flagged_file_reviewed():
    project, entry = _flagged_brightness_project()
    _switch(project, FolderSettings(dialogue_enabled=False, labels_enabled=True))
    mark_reviewed(project, "a.mp4")
    recompute_all(project, pending={}, ranges_pending=False)
    assert entry.brightness.source == Source.DETECTED                      # not required: not accepted
    assert entry.review == ReviewState.REVIEWED

    _switch(project, FolderSettings(dialogue_enabled=True, labels_enabled=True))

    assert entry.review == ReviewState.FLAGGED
    assert _state(project, "a.mp4") == ReviewState.FLAGGED


def _labels_only_reviewed(setup):
    project = _project(folder=LABELS_ONLY_FOLDER)
    entry = project.files["a.mp4"]
    setup(project, entry)
    entry.review = ReviewState.REVIEWED
    return project, entry


def _missing_crop(project, entry):
    entry.brightness = Brightness(NEW_BRIGHTNESS, Source.MANUAL)


def _stale_brightness(project, entry):
    entry.crop = Crop(*OLD_BOX, Source.MANUAL)
    entry.brightness = Brightness(NEW_BRIGHTNESS, Source.DETECTED)
    entry.evidence["brightness"] = {"crop_box": list(NEW_BOX), "value_crop_box": list(NEW_BOX)}


def _flagged_detected_crop(project, entry):
    entry.crop = Crop(*NEW_BOX, Source.DETECTED)
    entry.brightness = Brightness(NEW_BRIGHTNESS, Source.MANUAL)
    entry.flags["crop"] = "low-agreement"


@pytest.mark.parametrize("setup", [_missing_crop, _stale_brightness, _flagged_detected_crop])
def test_switching_to_dialogue_un_reviews_a_file_whose_new_required_values_are_not_ready(setup):
    project, entry = _labels_only_reviewed(setup)
    values = (entry.crop, entry.brightness)
    _switch(project, DIALOGUE_FOLDER)
    assert entry.review == ReviewState.FLAGGED
    assert (entry.crop, entry.brightness) == values                         # nothing accepted or changed


@pytest.mark.parametrize("sources", [(Source.DETECTED, Source.DETECTED), (Source.MANUAL, Source.IMPORTED)])
def test_switching_to_dialogue_keeps_a_reviewed_file_whose_values_are_ready(sources):
    def ready(project, entry):
        entry.crop = Crop(*NEW_BOX, sources[0])
        entry.brightness = Brightness(NEW_BRIGHTNESS, sources[1])
        entry.flags = {"crop": "no-speech", "brightness": "escalate" if sources[1] != Source.DETECTED else ""}
    project, entry = _labels_only_reviewed(ready)
    _switch(project, DIALOGUE_FOLDER)
    assert entry.review == ReviewState.REVIEWED


def test_switching_to_dialogue_leaves_files_that_are_not_reviewed_alone():
    project = _project(folder=LABELS_ONLY_FOLDER)
    for entry, stored in zip(project.files.values(), (ReviewState.PROPOSED, ReviewState.PENDING, ReviewState.FLAGGED)):
        entry.review = stored
    before = to_json(project)
    _switch(project, DIALOGUE_FOLDER)
    assert to_json(project)["files"] == before["files"]


@pytest.mark.parametrize("old, new", [
    (DIALOGUE_FOLDER, LABELS_ONLY_FOLDER),                                           # nothing newly required
    (DIALOGUE_FOLDER, FolderSettings(dialogue_enabled=True, labels_enabled=False)),  # same required set
    (LABELS_ONLY_FOLDER, FolderSettings(dialogue_enabled=False, labels_enabled=True, ocr_lang="en")),
])
def test_a_folder_change_that_does_not_add_required_fields_changes_no_file(old, new):
    project = _project(folder=old)
    a, b, c = project.files.values()
    _missing_crop(project, a)
    a.review = ReviewState.REVIEWED
    _flagged_detected_crop(project, b)
    b.review = ReviewState.FLAGGED
    c.review = ReviewState.PROPOSED
    before = to_json(project)["files"]
    project.folder = new
    apply_folder_change(project, old, new)
    assert to_json(project)["files"] == before


@pytest.mark.parametrize("edit", [
    lambda project: set_manual_brightness(project, "a.mp4", 205),          # the flagged field itself
    lambda project: paste_settings(project, "a.mp4", {"crop": NEW_BOX, "brightness": 205}),
])
def test_an_edit_that_replaces_the_flagged_value_reviews_the_file(edit):
    project, entry = _flagged_brightness_project()
    edit(project)
    assert entry.brightness == Brightness(205, Source.MANUAL)
    assert entry.review == ReviewState.REVIEWED
    assert _state(project, "a.mp4") == ReviewState.REVIEWED


@pytest.mark.parametrize("flag", ["no-clean-threshold", "escalate"])
def test_an_edit_reviews_the_file_when_the_other_flag_is_informational_or_beside_a_user_value(flag):
    project = _project()
    entry = project.files["a.mp4"]
    apply_crop(project, _crop_job_result())
    entry.brightness = Brightness(OTHER_BRIGHTNESS, Source.IMPORTED if flag == "escalate" else Source.DETECTED)
    entry.flags["brightness"] = flag
    set_manual_crop(project, "a.mp4", OLD_BOX)
    assert entry.review == ReviewState.REVIEWED


def test_an_edit_on_a_labels_only_file_reviews_it_whatever_the_flags():
    project, entry = _flagged_brightness_project(folder=FolderSettings(dialogue_enabled=False, labels_enabled=True))
    set_manual_time_ranges(project, "a.mp4", None)
    assert entry.review == ReviewState.REVIEWED


@pytest.mark.parametrize("source", [Source.MANUAL, Source.IMPORTED])
def test_a_reviewed_file_with_a_flag_beside_the_users_value_stays_reviewed(source):
    entry = _ready_entry()
    entry.review = ReviewState.REVIEWED
    entry.crop.source = source
    entry.brightness.source = source
    entry.flags = {"crop": "low-agreement", "brightness": FLAG_DIFFERS_FROM_HINT}
    assert _review(entry) == ReviewState.REVIEWED


# --- S7: brightness measured on another crop counts as missing ---------------------------

@pytest.mark.parametrize("measured_on, expected", [(NEW_BOX, ReviewState.PROPOSED), (OLD_BOX, ReviewState.FLAGGED)])
@pytest.mark.parametrize("source", [Source.DETECTED, Source.HINT])
def test_a_detected_brightness_measured_on_another_crop_counts_as_missing(measured_on, expected, source):
    entry = _ready_entry()
    entry.brightness.source = source
    entry.evidence = {"brightness": {"value": NEW_BRIGHTNESS, "crop_box": list(NEW_BOX),
                                     "value_crop_box": list(measured_on)}}
    assert _review(entry) == expected
    entry.review = ReviewState.REVIEWED
    stale = expected == ReviewState.FLAGGED
    assert _review(entry) == (ReviewState.FLAGGED if stale else ReviewState.REVIEWED)
    assert _review(entry, pending={"brightness"}) == (ReviewState.PENDING if stale else ReviewState.REVIEWED)


@pytest.mark.parametrize("source", [Source.MANUAL, Source.IMPORTED])
def test_the_users_brightness_is_never_stale(source):
    entry = _ready_entry()
    entry.brightness.source = source
    entry.evidence = {"brightness": {"crop_box": list(OLD_BOX), "value_crop_box": list(OLD_BOX)}}
    assert _review(entry) == ReviewState.PROPOSED


@pytest.mark.parametrize("evidence", [{}, {"brightness": {"value": NEW_BRIGHTNESS}},
                                      {"brightness": {"crop_box": list(OLD_BOX)}}])   # the latest result's box only
def test_a_detected_brightness_without_a_recorded_value_crop_is_not_judged_stale(evidence):
    entry = _ready_entry()
    entry.evidence = evidence
    assert _review(entry) == ReviewState.PROPOSED


@pytest.mark.parametrize("hint_value", [None, 200])
def test_s7_a_re_detected_crop_makes_the_brightness_stale_until_it_is_measured_again(hint_value):
    project = _project()
    entry = project.files["a.mp4"]
    apply_crop(project, _crop_job_result())
    apply_brightness(project, _brightness_job_result(hint_value=hint_value))
    assert _state(project, "a.mp4") == ReviewState.PROPOSED

    apply_crop(project, _crop_job_result(box=OLD_BOX))           # the crop moves; brightness is from NEW_BOX
    assert entry.brightness.source == (Source.DETECTED if hint_value is None else Source.HINT)
    assert _state(project, "a.mp4") == ReviewState.FLAGGED
    recompute_all(project, pending={"a.mp4": {"brightness"}}, ranges_pending=False)
    assert entry.review == ReviewState.PENDING
    mark_reviewed(project, "a.mp4")
    assert _state(project, "a.mp4") == ReviewState.FLAGGED

    apply_brightness(project, _brightness_job_result(crop_box=OLD_BOX))
    assert _state(project, "a.mp4") == ReviewState.PROPOSED


def test_an_unapplied_brightness_result_does_not_hide_that_the_value_is_stale():
    """repro_s6s7: re-measuring on the new crop fails; the stored value is still
    the one measured on the old crop."""
    project = _project()
    entry = project.files["a.mp4"]
    apply_crop(project, _crop_job_result())
    apply_brightness(project, _brightness_job_result())                       # 209 measured on NEW_BOX
    apply_crop(project, _crop_job_result(box=OLD_BOX))
    assert brightness_is_stale(entry)

    apply_brightness(project, _brightness_job_result(value=230, plateau=None, flagged="no-plateau?",
                                                     crop_box=OLD_BOX))       # not applied
    assert entry.brightness == Brightness(NEW_BRIGHTNESS, Source.DETECTED)
    assert entry.evidence["brightness"]["crop_box"] == list(OLD_BOX)          # the latest result
    assert entry.evidence["brightness"]["value_crop_box"] == list(NEW_BOX)    # the stored value
    assert brightness_is_stale(entry)
    assert _state(project, "a.mp4") == ReviewState.FLAGGED

    mark_reviewed(project, "a.mp4")
    assert entry.brightness.source == Source.DETECTED                          # not accepted
    assert _state(project, "a.mp4") == ReviewState.FLAGGED


def test_a_written_brightness_records_its_crop_and_an_unapplied_one_keeps_it():
    project = _project()
    entry = project.files["a.mp4"]
    apply_crop(project, _crop_job_result())
    apply_brightness(project, _brightness_job_result(value=240, plateau=None, flagged="escalate"))
    assert "value_crop_box" not in entry.evidence["brightness"]               # no value was written
    apply_brightness(project, _brightness_job_result())
    assert entry.evidence["brightness"]["value_crop_box"] == list(NEW_BOX)
    apply_brightness(project, _brightness_job_result(value=240, plateau=None, flagged="escalate"))
    assert entry.evidence["brightness"]["value_crop_box"] == list(NEW_BOX)
    assert entry.evidence["brightness"]["flagged"] == "escalate"


def test_a_fresh_brightness_over_a_stale_one_un_reviews_even_with_the_same_value():
    project = _project()
    entry = project.files["a.mp4"]
    apply_crop(project, _crop_job_result())
    apply_brightness(project, _brightness_job_result())
    apply_crop(project, _crop_job_result(box=OLD_BOX))
    entry.review = ReviewState.REVIEWED                         # marked before any recompute
    apply_brightness(project, _brightness_job_result(crop_box=OLD_BOX))
    assert entry.brightness == Brightness(NEW_BRIGHTNESS, Source.DETECTED)
    assert entry.review != ReviewState.REVIEWED
    assert _state(project, "a.mp4") == ReviewState.PROPOSED


def test_a_manual_crop_edit_makes_a_detected_brightness_stale():
    project = _project()
    apply_crop(project, _crop_job_result())
    apply_brightness(project, _brightness_job_result())
    set_manual_crop(project, "a.mp4", OLD_BOX)
    assert _state(project, "a.mp4") == ReviewState.FLAGGED
    set_manual_crop(project, "a.mp4", NEW_BOX)
    assert _state(project, "a.mp4") == ReviewState.REVIEWED


# --- S1-S3: REVIEWED never hides a missing value ------------------------------------------

@pytest.mark.parametrize("flagged", ["no-plateau?", "thin-evidence?"])
def test_s1_manual_crop_then_an_unapplied_brightness_is_not_reviewed(flagged):
    project = _project()
    entry = project.files["a.mp4"]
    set_manual_crop(project, "a.mp4", NEW_BOX)
    recompute_all(project, pending={"a.mp4": {"brightness"}}, ranges_pending=False)
    assert entry.review == ReviewState.PENDING

    apply_brightness(project, _brightness_job_result(value=230, plateau=None, flagged=flagged))
    assert entry.brightness is None
    assert _state(project, "a.mp4") == ReviewState.FLAGGED

    apply_brightness(project, _brightness_job_result())         # the value arrives: confirm again
    assert _state(project, "a.mp4") == ReviewState.PROPOSED


def test_s2_a_crop_without_a_box_then_a_manual_brightness_is_not_reviewed():
    project = _project()
    apply_crop(project, _crop_job_result(box=None, flagged="static-content", hit_pts=()))
    set_manual_brightness(project, "a.mp4", 210)
    assert project.files["a.mp4"].review == ReviewState.REVIEWED
    assert _state(project, "a.mp4") == ReviewState.FLAGGED
    set_manual_crop(project, "a.mp4", NEW_BOX)
    assert _state(project, "a.mp4") == ReviewState.REVIEWED


def test_s3_a_pasted_crop_only_clip_is_not_reviewed_while_brightness_is_missing():
    project = _project()
    set_manual_crop(project, "b.mp4", NEW_BOX)
    paste_settings(project, "a.mp4", copy_settings(project, "b.mp4"))
    assert _state(project, "a.mp4") == ReviewState.FLAGGED
    recompute_all(project, pending={"a.mp4": {"brightness"}}, ranges_pending=False)
    assert project.files["a.mp4"].review == ReviewState.PENDING
    apply_brightness(project, _brightness_job_result())
    assert _state(project, "a.mp4") == ReviewState.PROPOSED


def test_mark_reviewed_on_an_empty_file_is_not_reviewed():
    project = _project()
    mark_reviewed(project, "a.mp4")
    assert _state(project, "a.mp4") == ReviewState.FLAGGED


# --- S4/S5: an apply storing a blocking flag beside a detected value un-reviews ----------

def test_s4_a_hint_result_with_the_same_value_but_differs_from_hint_un_reviews():
    project = _project()
    entry = project.files["a.mp4"]
    _complete(entry)
    entry.review = ReviewState.REVIEWED
    apply_brightness(project, _brightness_job_result(value=OTHER_BRIGHTNESS, plateau=(190, 235), hint_value=250))
    assert entry.brightness == Brightness(OTHER_BRIGHTNESS, Source.HINT)
    assert entry.flags["brightness"] == FLAG_DIFFERS_FROM_HINT
    assert entry.review != ReviewState.REVIEWED
    assert _state(project, "a.mp4") == ReviewState.FLAGGED


def test_s5_a_flagged_re_detection_beside_a_detected_crop_un_reviews():
    project = _project()
    entry = project.files["a.mp4"]
    _complete(entry)
    entry.review = ReviewState.REVIEWED
    apply_crop(project, _crop_job_result(box=OLD_BOX, flagged="low-agreement"))
    assert entry.crop == Crop(*NEW_BOX, Source.DETECTED)
    assert entry.review != ReviewState.REVIEWED
    assert _state(project, "a.mp4") == ReviewState.FLAGGED


def test_a_blocking_flag_stored_beside_an_empty_field_un_reviews():
    project = _project()
    entry = project.files["a.mp4"]
    entry.crop = Crop(*NEW_BOX, Source.MANUAL)
    entry.review = ReviewState.REVIEWED
    apply_brightness(project, _brightness_job_result(value=230, plateau=None, flagged="no-plateau?"))
    assert entry.review != ReviewState.REVIEWED


def test_informational_flags_beside_detected_values_keep_review():
    project = _project()
    entry = project.files["a.mp4"]
    _complete(entry)
    entry.brightness = Brightness(NEW_BRIGHTNESS, Source.DETECTED)
    entry.review = ReviewState.REVIEWED
    apply_crop(project, _crop_job_result(flagged="no-speech"))
    apply_brightness(project, _brightness_job_result(flagged="no-clean-threshold"))
    assert entry.review == ReviewState.REVIEWED
    assert _state(project, "a.mp4") == ReviewState.REVIEWED


def test_a_blocking_flag_on_a_field_labels_only_does_not_need_keeps_review():
    project = _project(folder=FolderSettings(dialogue_enabled=False, labels_enabled=True))
    entry = project.files["a.mp4"]
    _complete(entry)
    entry.review = ReviewState.REVIEWED
    apply_crop(project, _crop_job_result(box=OLD_BOX, flagged="low-agreement"))
    apply_brightness(project, _brightness_job_result(value=180, plateau=None, flagged="escalate"))
    assert entry.review == ReviewState.REVIEWED
    assert _state(project, "a.mp4") == ReviewState.REVIEWED


# --- C3: a hinted crop that disagrees with its hint is flagged -----------------------------

FRAME = (1920, 1080)


def test_consistent_with_consensus_is_the_detectors_own_rule_on_a_box():
    boxes = [NEW_BOX, OLD_BOX, (0, 900, 1920, 100), (288, 500, 1344, 53), (288, 786, 1344, 150), (288, 786, 1344, 20)]
    series = [[], [(0.73, 0.05)], [(0.73, 0.05)] * 3, [(0.5, 0.1), (0.52, 0.1), (0.9, 0.2)]]
    for box in boxes:
        for consensus in series:
            assert crop_mod.consistent_with_consensus(box, FRAME, consensus) == \
                crop_mod._consistent_with_consensus(box[1] / 1080, box[3] / 1080, consensus)
    hint = [(786 / 1080, 53 / 1080)]
    assert crop_mod.consistent_with_consensus(NEW_BOX, FRAME, hint)
    assert not crop_mod.consistent_with_consensus((288, 500, 1344, 53), FRAME, hint)      # far above the hint
    assert not crop_mod.consistent_with_consensus((288, 786, 1344, 120), FRAME, hint)     # over 1.5x taller
    assert not crop_mod.consistent_with_consensus((288, 786, 1344, 30), FRAME, hint)      # under 1/1.5 the height


def test_consistent_with_consensus_needs_a_frame_height():
    with pytest.raises(ValueError):
        crop_mod.consistent_with_consensus(NEW_BOX, (1920, 0), [(0.73, 0.05)])


HINT_AT_BOX = (786 / 1080, 53 / 1080)


@pytest.mark.parametrize("hint, box, flagged, expected_flag", [
    (HINT_AT_BOX, NEW_BOX, None, ""),
    ((0.74, 0.06), NEW_BOX, "no-speech", "no-speech"),
    ((0.5, 0.05), NEW_BOX, None, FLAG_DIFFERS_FROM_HINT),                   # hint far above the box
    ((0.73, 0.12), NEW_BOX, None, FLAG_DIFFERS_FROM_HINT),                  # hint two lines tall
    ((0.5, 0.05), NEW_BOX, "no-speech", "no-speech+" + FLAG_DIFFERS_FROM_HINT),
    ((0.5, 0.05), NEW_BOX, "low-agreement", "low-agreement+" + FLAG_DIFFERS_FROM_HINT),
    ((0.5, 0.05), None, "static-content", "static-content"),                # no box to compare
    (None, (288, 500, 1344, 53), None, ""),                                  # not a hint result
])
def test_a_hinted_crop_that_disagrees_with_its_hint_is_flagged(hint, box, flagged, expected_flag):
    project = _project()
    entry = project.files["a.mp4"]
    _complete(entry)
    entry.crop = None
    r = _crop_job_result(hint=hint, box=box, flagged=flagged, hit_pts=(12.0, 30.0) if box else ())

    apply_crop(project, r)

    assert entry.flags["crop"] == expected_flag
    assert entry.evidence["crop"]["flagged"] == flagged
    if r.result.auto_applicable:
        assert entry.crop == Crop(*box, Source.HINT if hint else Source.DETECTED)
    blocking = expected_flag not in ("", "no-speech")
    assert _state(project, "a.mp4") == (ReviewState.FLAGGED if blocking else ReviewState.PROPOSED)


def test_a_hinted_crop_without_a_frame_size_cannot_be_checked_and_is_flagged():
    project = _project()
    result = _crop_result()
    result.frame_size = None
    apply_crop(project, CropJobResult("a.mp4", result, HINT_AT_BOX))
    assert project.files["a.mp4"].flags["crop"] == FLAG_DIFFERS_FROM_HINT


@pytest.mark.parametrize("folder", [DIALOGUE, DIALOGUE_ONLY])
@pytest.mark.parametrize("pending", [{"crop"}, {"brightness"}, {"crop", "brightness", "thumbnail"}])
def test_a_pending_required_detector_makes_the_file_pending(folder, pending):
    assert _review(_ready_entry(), folder, pending=pending) == ReviewState.PENDING


@pytest.mark.parametrize("folder", [DIALOGUE, LABELS_ONLY])
@pytest.mark.parametrize("ranges_source", [None, Source.DETECTED, Source.HINT])
def test_pending_ranges_make_a_file_whose_ranges_detection_may_write_pending(folder, ranges_source):
    entry = _ready_entry()
    if ranges_source is not None:
        entry.time_ranges = TimeRanges([TimeRange("1:30", "23:00")], ranges_source)
    assert _review(entry, folder, ranges_pending=True) == ReviewState.PENDING


@pytest.mark.parametrize("folder", [DIALOGUE, LABELS_ONLY])
@pytest.mark.parametrize("ranges", [[], [TimeRange("1:30", "23:00")]])
@pytest.mark.parametrize("ranges_source", [Source.MANUAL, Source.IMPORTED])
def test_pending_ranges_do_not_hold_a_file_whose_ranges_are_the_users(folder, ranges, ranges_source):
    """F5: a ranges analysis can never write MANUAL or IMPORTED ranges, so it
    does not make such a file wait; the rest of the state is derived as usual."""
    entry = _ready_entry()
    entry.time_ranges = TimeRanges(list(ranges), ranges_source)
    assert _review(entry, folder, ranges_pending=True) == ReviewState.PROPOSED
    entry.flags["crop"] = "low-agreement"
    expected = ReviewState.FLAGGED if folder.dialogue_enabled else ReviewState.PROPOSED
    assert _review(entry, folder, ranges_pending=True) == expected
    assert _review(entry, folder, pending={"crop"}, ranges_pending=True) == (
        ReviewState.PENDING if folder.dialogue_enabled else ReviewState.PROPOSED)


def test_pending_jobs_that_are_not_required_detectors_do_not_hold_a_file():
    assert _review(_ready_entry(), pending={"metadata", "thumbnail", "audio_profile", "proof"}) == ReviewState.PROPOSED


@pytest.mark.parametrize("missing", ["crop", "brightness"])
def test_a_missing_required_value_flags_the_file(missing):
    entry = _ready_entry()
    setattr(entry, missing, None)
    assert _review(entry) == ReviewState.FLAGGED


@pytest.mark.parametrize("detector, flag, expected", [
    ("crop", "no-speech", ReviewState.PROPOSED),
    ("crop", "speech-probes-exhausted", ReviewState.PROPOSED),
    ("crop", "no-speech+speech-probes-exhausted", ReviewState.PROPOSED),
    ("crop", "low-agreement", ReviewState.FLAGGED),
    ("crop", "no-speech+multiple-positions?", ReviewState.FLAGGED),
    ("crop", "no-clean-threshold", ReviewState.FLAGGED),          # brightness's informational flag, not crop's
    ("crop", "something-new", ReviewState.FLAGGED),
    ("brightness", "no-clean-threshold", ReviewState.PROPOSED),
    ("brightness", "no-speech", ReviewState.FLAGGED),             # crop's informational flag, not brightness's
    ("brightness", "escalate", ReviewState.FLAGGED),
    ("brightness", FLAG_DIFFERS_FROM_HINT, ReviewState.FLAGGED),
    ("brightness", "no-clean-threshold+" + FLAG_DIFFERS_FROM_HINT, ReviewState.FLAGGED),
])
def test_stored_flags_block_unless_informational_for_their_own_detector(detector, flag, expected):
    entry = _ready_entry()
    entry.flags[detector] = flag
    assert _review(entry) == expected


def test_a_file_with_both_values_and_clean_flags_is_proposed():
    assert _review(_ready_entry()) == ReviewState.PROPOSED
    entry = _ready_entry()
    entry.flags = {}                                    # never detected (e.g. imported values)
    assert _review(entry) == ReviewState.PROPOSED


@pytest.mark.parametrize("detector, flag", [("crop", "low-agreement"), ("brightness", "escalate"),
                                            ("brightness", FLAG_DIFFERS_FROM_HINT)])
@pytest.mark.parametrize("source, expected", [
    (Source.DETECTED, ReviewState.FLAGGED),
    (Source.HINT, ReviewState.FLAGGED),
    (Source.MANUAL, ReviewState.PROPOSED),
    (Source.IMPORTED, ReviewState.PROPOSED),
])
def test_a_blocking_flag_counts_only_against_a_detected_or_hinted_value(detector, flag, source, expected):
    entry = _ready_entry()
    getattr(entry, detector).source = source
    entry.flags[detector] = flag
    assert _review(entry) == expected


@pytest.mark.parametrize("detector", ["crop", "brightness"])
def test_a_blocking_flag_on_a_missing_value_still_flags(detector):
    entry = _ready_entry()
    setattr(entry, detector, None)
    entry.flags[detector] = "low-agreement" if detector == "crop" else "escalate"
    assert _review(entry) == ReviewState.FLAGGED


@pytest.mark.parametrize("source", [Source.MANUAL, Source.IMPORTED])
def test_a_rejected_detection_over_the_users_values_is_stored_but_does_not_flag(source):
    project = _project()
    entry = project.files["a.mp4"]
    entry.crop = Crop(*OLD_BOX, source)
    entry.brightness = Brightness(OTHER_BRIGHTNESS, source)

    apply_crop(project, _crop_job_result(flagged="low-agreement"))
    apply_brightness(project, _brightness_job_result(value=180, plateau=(170, 200), flagged="dim-text?",
                                                     hint_value=250, crop_box=OLD_BOX))

    assert entry.crop == Crop(*OLD_BOX, source)
    assert entry.brightness == Brightness(OTHER_BRIGHTNESS, source)
    assert entry.flags == {"crop": "low-agreement", "brightness": "dim-text?+" + FLAG_DIFFERS_FROM_HINT}
    assert entry.evidence["crop"]["flagged"] == "low-agreement"
    assert entry.evidence["brightness"]["flagged"] == "dim-text?"
    assert _state(project, "a.mp4") == ReviewState.PROPOSED


def test_labels_only_requires_neither_crop_nor_brightness():
    entry = FileEntry(name="a.mp4", flags={"crop": "low-agreement", "brightness": "escalate"})
    assert _review(entry, LABELS_ONLY) == ReviewState.PROPOSED
    assert _review(entry, LABELS_ONLY, pending={"crop", "brightness"}) == ReviewState.PROPOSED


def test_recompute_all_uses_each_files_own_pending_set():
    project = _project()
    for entry in project.files.values():
        _complete(entry)
    project.files["b.mp4"].crop = None
    project.files["c.mp4"].review = ReviewState.REVIEWED

    recompute_all(project, pending={"a.mp4": {"brightness"}, "c.mp4": {"crop"}}, ranges_pending=False)
    assert [e.review for e in project.files.values()] == [
        ReviewState.PENDING, ReviewState.FLAGGED, ReviewState.REVIEWED]

    recompute_all(project, pending={}, ranges_pending=False)
    assert project.files["a.mp4"].review == ReviewState.PROPOSED

    recompute_all(project, pending={}, ranges_pending=True)
    assert [e.review for e in project.files.values()] == [      # b's missing crop is not waiting on ranges
        ReviewState.PENDING, ReviewState.FLAGGED, ReviewState.REVIEWED]


# --------------------------------------------------------------------------
# User edits: manual values, review mark, skip, copy/paste
# --------------------------------------------------------------------------

def test_set_manual_crop_is_manual_and_reviewed():
    project = _project()
    set_manual_crop(project, "a.mp4", (10, 20, 30, 40))
    entry = project.files["a.mp4"]
    assert entry.crop == Crop(10, 20, 30, 40, Source.MANUAL)
    assert entry.review == ReviewState.REVIEWED


def test_set_manual_brightness_is_manual_and_reviewed():
    project = _project()
    set_manual_brightness(project, "a.mp4", 199)
    entry = project.files["a.mp4"]
    assert entry.brightness == Brightness(199, Source.MANUAL)
    assert entry.review == ReviewState.REVIEWED


def test_set_manual_time_ranges_is_manual_and_reviewed():
    project = _project()
    set_manual_time_ranges(project, "a.mp4", [("02:33", "21:20"), ("22:00", None)])
    entry = project.files["a.mp4"]
    assert entry.time_ranges == TimeRanges([TimeRange("02:33", "21:20"), TimeRange("22:00", None)], Source.MANUAL)
    assert entry.review == ReviewState.REVIEWED


def test_manual_whole_file_ocrs_the_whole_file_and_survives_detection():
    project = _project()
    entry = project.files["a.mp4"]
    entry.time_ranges = TimeRanges([TimeRange(None, "05:00")], Source.DETECTED)
    set_manual_time_ranges(project, "a.mp4", None)
    assert entry.review == ReviewState.REVIEWED
    assert ocr_call_for(entry, project.folder, PROJECT_DIR).time_ranges == []

    apply_ranges(project, _ranges_result({"a.mp4": [(None, "01:30")]}))
    assert ocr_call_for(entry, project.folder, PROJECT_DIR).time_ranges == []
    assert entry.time_ranges.source == Source.MANUAL


def test_an_empty_manual_range_list_is_the_whole_file_for_ocr(tmp_path):
    entry = FileEntry(name="a.mp4", time_ranges=TimeRanges([], Source.MANUAL))
    folder = FolderSettings()
    assert ocr_call_for(entry, folder, PROJECT_DIR).time_ranges == []
    assert ocr_call_for(entry, folder, PROJECT_DIR) == ocr_call_for(FileEntry(name="a.mp4"), folder, PROJECT_DIR)

    project = Project(path=str(tmp_path), folder=folder, files={"a.mp4": entry})
    restored = from_json(json.loads(json.dumps(to_json(project))), str(tmp_path)).files["a.mp4"]
    assert restored.time_ranges == TimeRanges([], Source.MANUAL)
    assert ocr_call_for(restored, folder, PROJECT_DIR).time_ranges == []


def test_manual_values_are_never_overwritten_by_detection():
    project = _project()
    set_manual_crop(project, "a.mp4", OLD_BOX)
    set_manual_brightness(project, "a.mp4", OTHER_BRIGHTNESS)
    apply_crop(project, _crop_job_result())
    apply_brightness(project, _brightness_job_result(crop_box=OLD_BOX))
    entry = project.files["a.mp4"]
    assert "crop" in entry.evidence and "brightness" in entry.evidence
    assert entry.crop == Crop(*OLD_BOX, Source.MANUAL)
    assert entry.brightness == Brightness(OTHER_BRIGHTNESS, Source.MANUAL)
    assert entry.review == ReviewState.REVIEWED


def test_mark_reviewed_and_unreviewed():
    project = _project()
    entry = project.files["a.mp4"]
    _complete(entry)
    mark_reviewed(project, "a.mp4")
    assert entry.review == ReviewState.REVIEWED
    assert _state(project, "a.mp4") == ReviewState.REVIEWED
    mark_reviewed(project, "a.mp4", False)
    assert entry.review != ReviewState.REVIEWED
    assert _state(project, "a.mp4") == ReviewState.PROPOSED


def test_set_skipped_toggles_skip_and_leaves_review_alone():
    project = _project()
    entry = project.files["a.mp4"]
    entry.review = ReviewState.REVIEWED
    set_skipped(project, "a.mp4", True)
    assert entry.skipped is True
    assert entry.review == ReviewState.REVIEWED
    set_skipped(project, "a.mp4", False)
    assert entry.skipped is False


def test_copy_then_paste_makes_the_target_ocr_exactly_like_the_source():
    project = _project()
    source = project.files["a.mp4"]
    source.crop = Crop(*NEW_BOX, Source.DETECTED)
    source.brightness = Brightness(NEW_BRIGHTNESS, Source.IMPORTED)
    source.time_ranges = TimeRanges([TimeRange("02:33", "21:20")], Source.DETECTED)
    target = project.files["b.mp4"]
    target.crop = Crop(*OLD_BOX, Source.MANUAL)
    target.brightness = Brightness(OTHER_BRIGHTNESS, Source.DETECTED)

    clip = copy_settings(project, "a.mp4")
    assert clip == {"crop": NEW_BOX, "brightness": NEW_BRIGHTNESS, "time_ranges": [("02:33", "21:20")]}
    paste_settings(project, "b.mp4", clip)

    assert target.crop == Crop(*NEW_BOX, Source.MANUAL)
    assert target.brightness == Brightness(NEW_BRIGHTNESS, Source.MANUAL)
    assert target.time_ranges == TimeRanges([TimeRange("02:33", "21:20")], Source.MANUAL)
    assert target.review == ReviewState.REVIEWED
    source_call = ocr_call_for(source, project.folder, PROJECT_DIR)
    target_call = ocr_call_for(target, project.folder, PROJECT_DIR)
    assert {**target_call.kwargs, "video_path": None} == {**source_call.kwargs, "video_path": None}
    assert target_call.time_ranges == source_call.time_ranges


def test_a_clip_is_a_snapshot_of_the_source():
    project = _project()
    set_manual_time_ranges(project, "a.mp4", [("02:33", "21:20")])
    clip = copy_settings(project, "a.mp4")
    project.files["a.mp4"].time_ranges.ranges.append(TimeRange("22:00", None))
    json.dumps(clip)
    paste_settings(project, "b.mp4", clip)
    assert project.files["b.mp4"].time_ranges == TimeRanges([TimeRange("02:33", "21:20")], Source.MANUAL)


def test_paste_applies_only_the_values_the_clip_has():
    project = _project()
    target = project.files["b.mp4"]
    target.crop = Crop(*OLD_BOX, Source.DETECTED)
    target.time_ranges = TimeRanges([TimeRange(None, "05:00")], Source.DETECTED)

    clip = copy_settings(project, "a.mp4")          # a.mp4 has nothing yet
    assert clip == {"crop": None, "brightness": None, "time_ranges": None}
    paste_settings(project, "b.mp4", clip)
    assert target.crop == Crop(*OLD_BOX, Source.DETECTED)
    assert target.time_ranges == TimeRanges([TimeRange(None, "05:00")], Source.DETECTED)
    assert target.review != ReviewState.REVIEWED

    paste_settings(project, "b.mp4", {"brightness": 190})
    assert target.brightness == Brightness(190, Source.MANUAL)
    assert target.crop == Crop(*OLD_BOX, Source.DETECTED)
    assert target.review == ReviewState.REVIEWED


def test_a_manual_whole_file_choice_pastes_as_whole_file():
    project = _project()
    set_manual_time_ranges(project, "a.mp4", None)
    clip = copy_settings(project, "a.mp4")
    assert clip["time_ranges"] == []
    target = project.files["b.mp4"]
    target.time_ranges = TimeRanges([TimeRange(None, "05:00")], Source.DETECTED)
    paste_settings(project, "b.mp4", clip)
    assert target.time_ranges == TimeRanges([], Source.MANUAL)
    assert ocr_call_for(target, project.folder, PROJECT_DIR).time_ranges == []


# --------------------------------------------------------------------------
# Jobs: identity
# --------------------------------------------------------------------------

def _entry_for_proof(**kwargs) -> FileEntry:
    values = {
        "name": "a.mp4",
        "crop": Crop(*NEW_BOX, Source.DETECTED),
        "brightness": Brightness(NEW_BRIGHTNESS, Source.DETECTED),
        "time_ranges": TimeRanges([TimeRange("02:33", "21:20"), TimeRange("22:00", None)], Source.DETECTED),
        "media": Media(1920, 1080, 1418.0, 23.976),
        "sample_time": 407.4,
    }
    values.update(kwargs)
    return FileEntry(**values)


@pytest.mark.parametrize("make, key, kind, lane, file, priority", [
    (lambda f: MetadataJob(PROJECT_DIR, "a.mp4"), "metadata:a.mp4", "metadata", Lane.CPU, "a.mp4", 0),
    (lambda f: ThumbnailJob(PROJECT_DIR, "a.mp4", 12.0), "thumbnail:a.mp4", "thumbnail", Lane.CPU, "a.mp4", 0),
    (lambda f: CropJob(PROJECT_DIR, "a.mp4", 1418.0, [], f), "crop:a.mp4", "crop", Lane.GPU, "a.mp4", 0),
    (lambda f: BrightnessJob(PROJECT_DIR, "a.mp4", NEW_BOX, None, f), "brightness:a.mp4", "brightness",
     Lane.GPU, "a.mp4", 0),
    (lambda f: RangesJob(PROJECT_DIR, list(NAMES), f), "ranges:*", "ranges", Lane.CPU, None, 0),
    (lambda f: AudioProfileJob(PROJECT_DIR, "a.mp4", 1418.0), "audio_profile:a.mp4", "audio_profile",
     Lane.CPU, "a.mp4", 0),
    (lambda f: ProofOcrJob(PROJECT_DIR, _entry_for_proof(), f), "proof:a.mp4", "proof", Lane.GPU, "a.mp4", 10),
])
def test_job_identity(make, key, kind, lane, file, priority):
    job = make(FolderSettings())
    assert (job.key, job.kind, job.lane, job.file, job.priority) == (key, kind, lane, file, priority)


# --------------------------------------------------------------------------
# Jobs: fakes
# --------------------------------------------------------------------------

@pytest.fixture
def fake_engines(monkeypatch):
    """Patch the registry's builders: every build is a recorded fake engine."""
    built = {"det": [], "ocr": []}

    def build_detection(det_model_dir, use_gpu):
        engine = SimpleNamespace(kind="det", args=(det_model_dir, use_gpu))
        built["det"].append(engine)
        return engine

    def build_ocr(lang, det_model_dir, rec_model_dir, use_gpu):
        engine = SimpleNamespace(kind="ocr", args=(lang, det_model_dir, rec_model_dir, use_gpu))
        built["ocr"].append(engine)
        return engine

    monkeypatch.setattr(engine_registry, "_build_detection_engine", build_detection)
    monkeypatch.setattr(engine_registry, "_build_ocr_engine", build_ocr)
    return built


def _idle(pool: dict) -> list:
    return [engine for engines in pool.values() for engine in engines]


class Recording:
    """Stand-in for a detector: binds each call to the real function's
    signature, so positional and keyword calls compare the same."""

    def __init__(self, real, returns=None, during=None):
        self.signature = inspect.signature(real)
        self.calls: list[dict] = []
        self.returns = returns
        self.during = during

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        call = dict(bound.arguments)
        self.calls.append(call)
        if self.during is not None:
            outcome = self.during(call)
            if outcome is not None:
                return outcome
        return self.returns


def _ctx(job, events=None) -> JobContext:
    return JobContext(job.key, job.kind, job.file, None if events is None else events.append)


CROP_FOLDER = {"crop_width_fraction": 0.8, "crop_vertical_padding": 0.01, "crop_min_height_fraction": 0.06,
               "bottom_half_cutoff": 0.5, "use_gpu": False}


# --- CropJob -----------------------------------------------------------------------

def test_crop_job_calls_detect_crop_with_the_documented_arguments(monkeypatch, fake_engines):
    result = _crop_result()

    def during(call):
        # The lease is held for the whole detector call.
        assert call["det_engine"] not in _idle(engine_registry._idle_detection_engines)

    fake = Recording(crop_mod.detect_crop, returns=result, during=during)
    monkeypatch.setattr(crop_mod, "detect_crop", fake)
    folder = FolderSettings(**CROP_FOLDER)
    consensus = [(0.72, 0.05), (0.73, 0.049)]
    job = CropJob(PROJECT_DIR, "a.mp4", 1418.0, consensus, folder)
    # Inputs are captured at construction.
    consensus.append((0.1, 0.9))
    folder.crop_width_fraction = 0.1
    ctx = _ctx(job)

    out = job.run(ctx)

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["video_path"] == "/proj/a.mp4"
    assert call["duration_sec"] == 1418.0
    assert call["consensus"] == [(0.72, 0.05), (0.73, 0.049)]
    assert call["settings"] == {"crop_width_fraction": 0.8, "crop_vertical_padding": 0.01,
                                "crop_min_height_fraction": 0.06, "bottom_half_cutoff": 0.5}
    assert [e.args for e in fake_engines["det"]] == [(None, False)]
    assert call["det_engine"] is fake_engines["det"][0]
    assert fake_engines["ocr"] == []
    assert _idle(engine_registry._idle_detection_engines) == [fake_engines["det"][0]]
    cancel = call["cancel_check"]
    assert cancel() is False
    ctx.cancel_event.set()
    assert cancel() is True
    assert out == CropJobResult(file="a.mp4", result=result, hint=None)
    assert isinstance(out, CropJobResult)


def test_a_crop_hint_seeds_the_consensus_and_rides_on_the_result(monkeypatch, fake_engines):
    result = _crop_result()
    fake = Recording(crop_mod.detect_crop, returns=result)
    monkeypatch.setattr(crop_mod, "detect_crop", fake)
    job = CropJob(PROJECT_DIR, "a.mp4", 1418.0, [(0.5, 0.2)], FolderSettings(), hint=(0.73, 0.05))

    out = job.run(_ctx(job))

    assert fake.calls[0]["consensus"] == [(0.73, 0.05)] * crop_mod.CONSENSUS_MIN_ENTRIES
    assert out.hint == (0.73, 0.05)
    project = _project()
    apply_crop(project, out)
    assert project.files["a.mp4"].crop.source == Source.HINT


@pytest.mark.parametrize("flagged", ["cancelled", "no-speech+cancelled"])
def test_a_crop_job_returns_none_when_the_detector_was_cancelled(monkeypatch, fake_engines, flagged):
    def during(call):
        call["cancel_check"]()
        return CropResult(box=NEW_BOX, flagged=flagged, frame_size=(1920, 1080))

    monkeypatch.setattr(crop_mod, "detect_crop", Recording(crop_mod.detect_crop, during=during))
    job = CropJob(PROJECT_DIR, "a.mp4", 1418.0, [], FolderSettings())
    ctx = _ctx(job)
    ctx.cancel_event.set()
    assert job.run(ctx) is None


def test_a_crop_job_returns_none_when_audio_extraction_was_cancelled(monkeypatch, fake_engines):
    def during(call):
        raise vad.AudioExtractionCancelled()

    monkeypatch.setattr(crop_mod, "detect_crop", Recording(crop_mod.detect_crop, during=during))
    job = CropJob(PROJECT_DIR, "a.mp4", 1418.0, [], FolderSettings())
    ctx = _ctx(job)
    ctx.cancel_event.set()
    assert job.run(ctx) is None


def test_a_failing_detector_fails_the_job_and_returns_the_lease(monkeypatch, fake_engines):
    def during(call):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(crop_mod, "detect_crop", Recording(crop_mod.detect_crop, during=during))
    job = CropJob(PROJECT_DIR, "a.mp4", 1418.0, [], FolderSettings())
    with pytest.raises(RuntimeError):
        job.run(_ctx(job))
    assert _idle(engine_registry._idle_detection_engines) == fake_engines["det"]


# --- BrightnessJob -----------------------------------------------------------------

def test_brightness_job_calls_detect_brightness_with_the_documented_arguments(monkeypatch, fake_engines):
    result = _brightness_result()

    def during(call):
        assert call["det_engine"] not in _idle(engine_registry._idle_detection_engines)
        assert call["ocr_engine"] not in _idle(engine_registry._idle_ocr_engines)

    fake = Recording(brightness_mod.detect_brightness, returns=result, during=during)
    monkeypatch.setattr(brightness_mod, "detect_brightness", fake)
    folder = FolderSettings(ocr_lang="chinese_cht", use_gpu=False)
    ranges = [("02:33", "21:20"), ("22:00", None)]
    job = BrightnessJob(PROJECT_DIR, "a.mp4", NEW_BOX, ranges, folder, folder_plateau=(190, 235))
    ranges.append(("23:00", None))
    folder.ocr_lang = "en"
    ctx = _ctx(job)

    out = job.run(ctx)

    call = fake.calls[0]
    assert call["video_path"] == "/proj/a.mp4"
    assert tuple(call["crop_box"]) == NEW_BOX
    assert [tuple(pair) for pair in call["time_ranges"]] == [("02:33", "21:20"), ("22:00", None)]
    assert call["folder_plateau"] == (190, 235)
    assert [e.args for e in fake_engines["det"]] == [(None, False)]
    assert [e.args for e in fake_engines["ocr"]] == [("chinese_cht", None, None, False)]
    assert call["det_engine"] is fake_engines["det"][0]
    assert call["ocr_engine"] is fake_engines["ocr"][0]
    cancel = call["cancel_check"]
    assert cancel() is False
    ctx.cancel_event.set()
    assert cancel() is True
    assert isinstance(out, BrightnessJobResult)
    assert out.file == "a.mp4"
    assert out.result is result
    assert out.tiles == choose_tiles(result.strips, result.value)
    assert out.tiles
    assert out.hint_value is None
    assert out.crop_box == NEW_BOX
    assert _idle(engine_registry._idle_detection_engines) == fake_engines["det"]
    assert _idle(engine_registry._idle_ocr_engines) == fake_engines["ocr"]


def test_a_brightness_hint_runs_full_detection_and_rides_on_the_result(monkeypatch, fake_engines):
    fake = Recording(brightness_mod.detect_brightness, returns=_brightness_result())
    monkeypatch.setattr(brightness_mod, "detect_brightness", fake)
    job = BrightnessJob(PROJECT_DIR, "a.mp4", NEW_BOX, None, FolderSettings(), hint_value=250)
    out = job.run(_ctx(job))
    assert fake.calls[0]["folder_plateau"] is None
    assert fake.calls[0]["time_ranges"] is None
    assert out.hint_value == 250


@pytest.mark.parametrize("crop_box", [NEW_BOX, None])
def test_a_brightness_job_result_carries_the_crop_box_it_measured_with(monkeypatch, fake_engines, crop_box):
    monkeypatch.setattr(brightness_mod, "detect_brightness",
                        Recording(brightness_mod.detect_brightness, returns=_brightness_result()))
    box = None if crop_box is None else list(crop_box)
    job = BrightnessJob(PROJECT_DIR, "a.mp4", box, None, FolderSettings())
    if box is not None:
        box[0] = 0                                     # captured at construction
    assert job.run(_ctx(job)).crop_box == crop_box


def test_a_crop_edited_while_brightness_ran_drops_the_brightness_result(monkeypatch, fake_engines):
    monkeypatch.setattr(brightness_mod, "detect_brightness",
                        Recording(brightness_mod.detect_brightness, returns=_brightness_result()))
    project = _project()
    entry = project.files["a.mp4"]
    entry.crop = Crop(*NEW_BOX, Source.DETECTED)
    job = BrightnessJob(PROJECT_DIR, "a.mp4", (entry.crop.x, entry.crop.y, entry.crop.width, entry.crop.height),
                        None, project.folder)
    out = job.run(_ctx(job))

    set_manual_crop(project, "a.mp4", OLD_BOX)         # the user edits the crop while the job runs
    before = to_json(project)
    apply_brightness(project, out)

    assert to_json(project) == before
    assert entry.brightness is None
    assert "brightness" not in entry.evidence and "brightness" not in entry.flags


def test_a_brightness_hint_with_a_folder_plateau_is_refused():
    with pytest.raises(ValueError):
        BrightnessJob(PROJECT_DIR, "a.mp4", NEW_BOX, None, FolderSettings(), folder_plateau=(190, 235),
                      hint_value=250)


def test_a_brightness_job_returns_none_when_the_detector_was_cancelled(monkeypatch, fake_engines):
    def during(call):
        call["cancel_check"]()
        return BrightnessResult(230, None, 230, None, "cancelled", [])

    monkeypatch.setattr(brightness_mod, "detect_brightness",
                        Recording(brightness_mod.detect_brightness, during=during))
    job = BrightnessJob(PROJECT_DIR, "a.mp4", NEW_BOX, None, FolderSettings())
    ctx = _ctx(job)
    ctx.cancel_event.set()
    assert job.run(ctx) is None


# --- RangesJob -----------------------------------------------------------------------

def test_ranges_job_calls_analyse_detailed_with_the_documented_arguments(monkeypatch):
    analysis = RangesAnalysis(keep={"a.mp4": [(None, "01:30")]}, blocks={}, durations={"a.mp4": 1418.0})

    def during(call):
        progress = call["progress"]
        progress(ProgressEvent("phase", "Fingerprinting"))
        progress(ProgressEvent("file", "a.mp4", 1, 4))
        progress(ProgressEvent("log", "found segment"))

    fake = Recording(ranges_pipeline.analyse_detailed, returns=analysis, during=during)
    monkeypatch.setattr(ranges_pipeline, "analyse_detailed", fake)
    folder = FolderSettings(min_segment_length=45.0, merge_repeating_silences=True)
    names = ["a.mp4", "b.mkv"]
    job = RangesJob(PROJECT_DIR, names, folder)
    names.append("c.mp4")
    folder.min_segment_length = 10.0
    events = []
    ctx = _ctx(job, events)

    out = job.run(ctx)

    call = fake.calls[0]
    assert list(call["files"]) == [RangesFile("a.mp4", "/proj/a.mp4"), RangesFile("b.mkv", "/proj/b.mkv")]
    assert call["cfg"] == RangesConfig(match=MatchConfig(min_length_sec=45.0), merge_repeating_silences=True)
    assert call["cache_dir"] == ranges_pipeline.default_cache_dir(PROJECT_DIR)
    assert call["workers"] == ranges_pipeline.DEFAULT_WORKERS
    cancel = call["cancel"]
    assert cancel() is False
    ctx.cancel_event.set()
    assert cancel() is True
    assert out == RangesJobResult(analysis)
    progress = [(e.progress, e.message) for e in events if e.type == "progress"]
    assert (0.25, "a.mp4") in progress
    assert any(message == "Fingerprinting" for _, message in progress)
    assert [e.message for e in events if e.type == "log"] == ["found segment"]


def test_a_ranges_job_returns_none_when_cancelled(monkeypatch):
    def during(call):
        call["cancel"]()
        raise ranges_pipeline.AnalysisCancelled()

    monkeypatch.setattr(ranges_pipeline, "analyse_detailed", Recording(ranges_pipeline.analyse_detailed, during=during))
    job = RangesJob(PROJECT_DIR, ["a.mp4"], FolderSettings())
    ctx = _ctx(job)
    ctx.cancel_event.set()
    assert job.run(ctx) is None


# --- AudioProfileJob -----------------------------------------------------------------

def test_audio_profile_job_calls_audio_profile_with_the_documented_arguments(monkeypatch):
    profile = AudioProfile(1418.0, [0.0, 1.0], [(1.0, 2.0)])
    fake = Recording(audio_profile_mod.audio_profile, returns=profile)
    monkeypatch.setattr(audio_profile_mod, "audio_profile", fake)
    job = AudioProfileJob(PROJECT_DIR, "a.mp4", 1418.0)
    ctx = _ctx(job)

    out = job.run(ctx)

    call = fake.calls[0]
    assert call["video_path"] == "/proj/a.mp4"
    assert call["duration_sec"] == 1418.0
    cancel = call["cancel_check"]
    assert cancel() is False
    ctx.cancel_event.set()
    assert cancel() is True
    assert out == AudioProfileResult("a.mp4", profile)


def test_an_audio_profile_job_returns_none_when_cancelled(monkeypatch):
    def during(call):
        call["cancel_check"]()
        raise vad.AudioExtractionCancelled()

    monkeypatch.setattr(audio_profile_mod, "audio_profile", Recording(audio_profile_mod.audio_profile, during=during))
    job = AudioProfileJob(PROJECT_DIR, "a.mp4", 1418.0)
    ctx = _ctx(job)
    ctx.cancel_event.set()
    assert job.run(ctx) is None


# --- MetadataJob / ThumbnailJob ----------------------------------------------------------

def test_metadata_job_reads_the_file_as_the_ocr_pass_counts_it(synthetic_video):
    job = MetadataJob(str(synthetic_video.parent), synthetic_video.name)
    out = job.run(_ctx(job))
    assert out == MetadataResult(synthetic_video.name, 320, 240, pytest.approx(0.4), pytest.approx(25.0))
    assert isinstance(out, MetadataResult)


def test_metadata_job_takes_duration_and_fps_from_video_timing(synthetic_video, monkeypatch):
    seen = []

    def video_timing(path):
        seen.append(path)
        return 1418.5, 23.976

    monkeypatch.setattr(ocr_view, "video_timing", video_timing)
    job = MetadataJob(str(synthetic_video.parent), synthetic_video.name)
    out = job.run(_ctx(job))
    assert seen == [str(synthetic_video)]
    assert (out.width, out.height, out.duration, out.fps) == (320, 240, 1418.5, 23.976)


def test_thumbnail_job_grabs_one_full_frame_at_thumbnail_height(monkeypatch):
    image = np.zeros((THUMB_HEIGHT, 128, 3), dtype=np.uint8)
    fake = Recording(crop_mod.grab_frames, returns=[image])
    monkeypatch.setattr(crop_mod, "grab_frames", fake)
    job = ThumbnailJob(PROJECT_DIR, "a.mp4", 407.4)

    out = job.run(_ctx(job))

    assert fake.calls == [{"video_path": "/proj/a.mp4", "times": [407.4], "band_frac": 1.0,
                           "target_height": THUMB_HEIGHT}]
    assert isinstance(out, ThumbnailResult)
    assert (out.file, out.time) == ("a.mp4", 407.4)
    assert out.image is image


def test_a_thumbnail_that_could_not_be_grabbed_has_no_image(monkeypatch):
    monkeypatch.setattr(crop_mod, "grab_frames", Recording(crop_mod.grab_frames, returns=[]))
    job = ThumbnailJob(PROJECT_DIR, "a.mp4", 99999.0)
    out = job.run(_ctx(job))
    assert out == ThumbnailResult("a.mp4", 99999.0, None)


def test_thumbnail_job_on_a_real_file(synthetic_video):
    job = ThumbnailJob(str(synthetic_video.parent), synthetic_video.name, 0.2)
    out = job.run(_ctx(job))
    assert out.image.shape[0] == THUMB_HEIGHT
    assert out.image.shape[2] == 3


@pytest.mark.parametrize("make", [
    lambda: MetadataJob(PROJECT_DIR, "a.mp4"),
    lambda: ThumbnailJob(PROJECT_DIR, "a.mp4", 1.0),
])
def test_quick_jobs_cancelled_before_they_start_return_none(make, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("no work after cancel")

    monkeypatch.setattr(ocr_view, "video_timing", refuse)
    monkeypatch.setattr(crop_mod, "grab_frames", refuse)
    job = make()
    ctx = _ctx(job)
    ctx.cancel_event.set()
    assert job.run(ctx) is None


# --- ProofOcrJob ------------------------------------------------------------------------

@pytest.mark.parametrize("sample_time, duration, window, strings", [
    (407.4, 1418.0, (407.4, 437.4), ("6:47", "7:17")),
    (None, 1418.0, (567.2, 597.2), ("9:27", "9:57")),
    (1400.0, 1418.0, (1388.0, 1418.0), ("23:08", "23:38")),
    (2000.0, 1418.0, (1388.0, 1418.0), ("23:08", "23:38")),
    (-3.0, 1418.0, (0.0, 30.0), ("0:00", "0:30")),
    (None, 20.0, (0.0, 20.0), ("0:00", "0:20")),
    (5.0, 20.0, (0.0, 20.0), ("0:00", "0:20")),
    (3700.0, 7200.0, (3700.0, 3730.0), ("61:40", "62:10")),
])
def test_proof_window(sample_time, duration, window, strings):
    start, end = proof_window(sample_time, duration)
    assert (start, end) == (pytest.approx(window[0]), pytest.approx(window[1]))
    assert (format_mss(start), format_mss(end)) == strings


def _dialogue_ass(*lines) -> str:
    """ASS text as the dialogue pass writes it (videocr.utils' own formatters)."""
    return vc_utils.format_ass_header(1920, 1080) + "".join(
        vc_utils.format_ass_dialogue(vc_utils.get_ass_timestamp_from_seconds(start),
                                     vc_utils.get_ass_timestamp_from_seconds(end), text)
        for start, end, text in lines)


@pytest.mark.parametrize("duration", [0.0, -1.0])
def test_a_proof_needs_the_files_duration(duration):
    with pytest.raises(ValueError):
        ProofOcrJob(PROJECT_DIR, _entry_for_proof(media=Media(1920, 1080, duration, 23.976)), FolderSettings())


def test_proof_job_runs_the_files_exact_ocr_call_on_one_window(monkeypatch):
    entry = _entry_for_proof()
    folder = FolderSettings(label_mask_crops=[(1, 2, 3, 4)], use_gpu=False, conf_threshold=90)
    expected_kwargs = ocr_call_for(entry, folder, PROJECT_DIR).kwargs

    fake = Recording(api.get_subtitles, returns=_dialogue_ass(
        (408.0, 410.5, "第一句"), (411.0, 413.25, "第二句\n第二行, 带逗号")))
    monkeypatch.setattr(api, "get_subtitles", fake)
    job = ProofOcrJob(PROJECT_DIR, entry, folder)
    # Captured at construction: later edits do not change the proof.
    entry.brightness = Brightness(100, Source.MANUAL)
    folder.conf_threshold = 10
    ctx = _ctx(job)

    out = job.run(ctx)

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert [tuple(r) for r in call["time_ranges"]] == [("6:47", "7:17")]
    assert call["cancel_event"] is ctx.cancel_event
    defaults = {name: p.default for name, p in fake.signature.parameters.items()}
    passed = {name: value for name, value in call.items()
              if name not in ("time_ranges", "cancel_event")
              and (name in expected_kwargs or value != defaults[name])}
    assert passed == expected_kwargs
    assert isinstance(out, ProofResult)
    assert out.file == "a.mp4"
    assert out.window == (407.0, 437.0)                         # exactly what OCR ran on: "6:47"-"7:17"
    assert out.lines == [(408.0, 410.5, "第一句"), (411.0, 413.25, "第二句\n第二行, 带逗号")]
    assert out.seconds >= 0.0


def test_a_labels_only_proof_lists_the_label_lines(monkeypatch):
    labels = [SimpleNamespace(start_pts=409.0, end_pts=411.5, text="张三", pos_x=100, pos_y=200),
              SimpleNamespace(start_pts=420.25, end_pts=422.0, text="李四\n护法", pos_x=1500, pos_y=90)]
    fake = Recording(api.get_subtitles, returns=vc_utils.format_labels_only_ass(labels, 1920, 1080))
    monkeypatch.setattr(api, "get_subtitles", fake)
    folder = FolderSettings(dialogue_enabled=False, labels_enabled=True)
    job = ProofOcrJob(PROJECT_DIR, _entry_for_proof(), folder)

    out = job.run(_ctx(job))

    assert fake.calls[0]["only_labels"] is True
    assert out.lines == [(409.0, 411.5, "张三"), (420.25, 422.0, "李四\n护法")]


def test_a_proof_lists_dialogue_and_label_lines_as_the_run_writes_them(monkeypatch):
    dialogue = _dialogue_ass((408.0, 410.5, "第一句"), (415.0, 417.0, "第三句"))
    labels = [SimpleNamespace(start_pts=412.0, end_pts=413.0, text="张三", pos_x=100, pos_y=200)]
    monkeypatch.setattr(api, "get_subtitles",
                        Recording(api.get_subtitles, returns=vc_utils.merge_ass_output(dialogue, labels, 1920, 1080)))
    job = ProofOcrJob(PROJECT_DIR, _entry_for_proof(), FolderSettings())
    out = job.run(_ctx(job))
    assert out.lines == [(408.0, 410.5, "第一句"), (412.0, 413.0, "张三"), (415.0, 417.0, "第三句")]


def test_proof_text_loses_override_tags_and_keeps_line_breaks(monkeypatch):
    ass = (vc_utils.format_ass_header(1920, 1080, include_label_style=True)
           + "Dialogue: 0,0:06:48.00,0:06:50.50,Label,,0,0,0,,{\\pos(10,20)}{\\an8}上{\\i1}行\\N下行\n")
    monkeypatch.setattr(api, "get_subtitles", Recording(api.get_subtitles, returns=ass))
    job = ProofOcrJob(PROJECT_DIR, _entry_for_proof(), FolderSettings())
    assert job.run(_ctx(job)).lines == [(408.0, 410.5, "上行\n下行")]


def test_proof_lines_split_only_on_ass_line_ends():
    ass = _dialogue_ass((408.0, 410.5, "甲\u2028乙\x1c丙")).replace("\n", "\r\n")
    assert proof_lines(ass) == [(408.0, 410.5, "甲\u2028乙\x1c丙")]


def test_a_proof_shows_the_lines_after_the_runs_qa_pass(monkeypatch):
    raw = _dialogue_ass((408.0, 410.5, "第一句"), (410.6, 412.0, "第一句"), (413.0, 414.0, "-"), (415.0, 417.0, "第三句"))
    monkeypatch.setattr(api, "get_subtitles", Recording(api.get_subtitles, returns=raw))
    job = ProofOcrJob(PROJECT_DIR, _entry_for_proof(), FolderSettings())
    out = job.run(_ctx(job))
    assert proof_lines(raw) != out.lines
    assert out.lines == [(408.0, 412.0, "第一句"), (415.0, 417.0, "第三句")]


def test_the_proof_qa_pass_runs_with_the_run_defaults_on_a_temporary_copy(monkeypatch):
    from core import ass_qafix
    raw = _dialogue_ass((408.0, 410.5, "第一句"))
    seen = []
    real = ass_qafix.process_file

    def process_file(*args, **kwargs):
        path = args[0]
        with open(path, encoding="utf-8") as handle:
            seen.append((args, kwargs, os.path.dirname(path), handle.read()))
        return real(*args, **kwargs)

    monkeypatch.setattr(ass_qafix, "process_file", process_file)
    monkeypatch.setattr(api, "get_subtitles", Recording(api.get_subtitles, returns=raw))
    job = ProofOcrJob(PROJECT_DIR, _entry_for_proof(), FolderSettings())
    out = job.run(_ctx(job))

    assert len(seen) == 1
    args, kwargs, directory, content = seen[0]
    assert len(args) == 1 and kwargs == {}                    # process_file's defaults, as the run calls it
    assert content == raw
    assert not directory.startswith(PROJECT_DIR)
    assert not os.path.exists(directory)                      # the temporary copy is gone
    assert out.lines == [(408.0, 410.5, "第一句")]


def test_a_proof_window_at_the_files_end_is_whole_seconds(monkeypatch):
    fake = Recording(api.get_subtitles, returns="")
    monkeypatch.setattr(api, "get_subtitles", fake)
    job = ProofOcrJob(PROJECT_DIR, _entry_for_proof(sample_time=1400.0, media=Media(1920, 1080, 1418.08, 25.0)),
                      FolderSettings())
    out = job.run(_ctx(job))
    assert [tuple(r) for r in fake.calls[0]["time_ranges"]] == [("23:08", "23:38")]
    assert out.window == (1388.0, 1418.0)


def test_a_proof_does_not_collect_lines_through_the_callback(monkeypatch):
    def during(call):
        if call["subtitle_callback"] is not None:
            call["subtitle_callback"](408.0, 410.5, "callback only")
        return ""

    monkeypatch.setattr(api, "get_subtitles", Recording(api.get_subtitles, during=during))
    job = ProofOcrJob(PROJECT_DIR, _entry_for_proof(), FolderSettings())
    assert job.run(_ctx(job)).lines == []


@pytest.mark.parametrize("ass", ["", vc_utils.format_ass_header(1920, 1080, include_label_style=True)])
def test_a_proof_with_no_lines_still_returns_a_result(monkeypatch, ass):
    monkeypatch.setattr(api, "get_subtitles", Recording(api.get_subtitles, returns=ass))
    job = ProofOcrJob(PROJECT_DIR, _entry_for_proof(sample_time=None), FolderSettings())
    out = job.run(_ctx(job))
    assert out.lines == []
    assert out.window == (567.0, 597.0)


def test_a_cancelled_proof_returns_none(monkeypatch):
    job = ProofOcrJob(PROJECT_DIR, _entry_for_proof(), FolderSettings())
    ctx = _ctx(job)

    def during(call):
        ctx.cancel_event.set()
        return _dialogue_ass((408.0, 410.5, "第一句"))

    monkeypatch.setattr(api, "get_subtitles", Recording(api.get_subtitles, during=during))
    assert job.run(ctx) is None


# --------------------------------------------------------------------------
# Jobs through the runner
# --------------------------------------------------------------------------

class Events:
    def __init__(self):
        self._cond = threading.Condition()
        self.events = []

    def __call__(self, event):
        with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    def terminal(self, key, timeout=WAIT):
        def found():
            return [e for e in self.events if e.key == key and e.type in ("finished", "failed", "cancelled")]
        with self._cond:
            assert self._cond.wait_for(found, timeout), f"no terminal event for {key}"
            return found()[0]


def test_a_crop_job_through_the_runner_finishes_then_cancels_to_none(monkeypatch, fake_engines):
    started = threading.Event()

    def during(call):
        if call["duration_sec"] == 1.0:
            return None
        started.set()
        deadline = time.monotonic() + WAIT
        while not call["cancel_check"]() and time.monotonic() < deadline:
            time.sleep(0.01)
        return CropResult(box=NEW_BOX, flagged="cancelled", frame_size=(1920, 1080))

    monkeypatch.setattr(crop_mod, "detect_crop", Recording(crop_mod.detect_crop, returns=_crop_result(),
                                                           during=during))
    events = Events()
    runner = JobRunner(events)
    try:
        runner.submit(CropJob(PROJECT_DIR, "a.mp4", 1.0, [], FolderSettings()))
        finished = events.terminal("crop:a.mp4")
        assert finished.type == "finished"
        assert isinstance(finished.result, CropJobResult)

        runner.submit(CropJob(PROJECT_DIR, "b.mp4", 1418.0, [], FolderSettings()))
        assert started.wait(WAIT)
        runner.cancel("crop:b.mp4")
        cancelled = events.terminal("crop:b.mp4")
        assert cancelled.type == "cancelled"
        assert cancelled.result is None
    finally:
        assert runner.shutdown(WAIT)

    project = _project()
    apply_crop(project, finished.result)
    apply_crop(project, cancelled.result)
    assert project.files["a.mp4"].crop == Crop(*NEW_BOX, Source.DETECTED)
    assert project.files["b.mp4"].crop is None
    assert "crop" not in project.files["b.mp4"].evidence


def test_job_modules_import_no_qt():
    code = ("import sys, core.jobs.detect_jobs, core.jobs.apply; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('PyQt6', 'PyQt5', 'PySide6')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False,
                          cwd=Path(__file__).resolve().parent.parent)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------------------------
# Real detectors on a reference episode
# --------------------------------------------------------------------------

REFERENCE_VIDEO = Path("/mnt/FAST/work/Slay the Gods S2/ZS2_-_11_[1080p]TXHBR.mp4")
REFERENCE_KEEP = [("02:33", "21:20")]      # the project's own keep range, as migration imports it


@pytest.mark.slow
@pytest.mark.needs_media
@pytest.mark.timeout(900)
def test_real_jobs_on_a_reference_episode_leave_a_proposed_entry(tmp_path):
    if not REFERENCE_VIDEO.exists():
        pytest.skip(f"{REFERENCE_VIDEO} not present")
    (tmp_path / REFERENCE_VIDEO.name).symlink_to(REFERENCE_VIDEO)
    project = load_project(str(tmp_path))
    name = REFERENCE_VIDEO.name
    entry = project.files[name]
    entry.time_ranges = TimeRanges([TimeRange(s, e) for s, e in REFERENCE_KEEP], Source.IMPORTED)

    events = Events()
    runner = JobRunner(events)
    timings = {}
    try:
        def run(job, apply):
            started = time.perf_counter()
            runner.submit(job)
            event = _wait_long(events, job.key)
            timings[job.kind] = time.perf_counter() - started
            assert event.type == "finished", event.error
            apply(project, event.result)
            return event.result

        meta = run(MetadataJob(str(tmp_path), name), apply_metadata)
        crop = run(CropJob(str(tmp_path), name, entry.media.duration, [], project.folder), apply_crop)
        assert entry.crop is not None
        bright = run(BrightnessJob(str(tmp_path), name, (entry.crop.x, entry.crop.y, entry.crop.width,
                                                         entry.crop.height),
                                   [(r.start, r.end) for r in entry.time_ranges.ranges], project.folder),
                     apply_brightness)
        proof = run(ProofOcrJob(str(tmp_path), entry, project.folder), lambda project, result: None)
    finally:
        assert runner.shutdown(60)

    print(f"metadata={meta} ({timings['metadata']:.1f}s)")
    print(f"crop box={crop.result.box} flagged={crop.result.flagged} hit_pts={crop.result.hit_pts[:3]} "
          f"probes={crop.result.probes_used} ({timings['crop']:.1f}s)")
    print(f"brightness value={bright.result.value} plateau={bright.result.plateau} "
          f"flagged={bright.result.flagged} tiles={bright.tiles} ({timings['brightness']:.1f}s)")
    print(f"proof window={proof.window} lines={len(proof.lines)} seconds={proof.seconds:.1f} "
          f"first={proof.lines[:3]} ({timings['proof']:.1f}s)")

    assert meta.width > 0 and meta.height > 0
    assert meta.duration > 1000 and meta.fps > 20
    assert crop.result.auto_applicable
    assert bright.result.auto_applicable
    assert entry.crop == Crop(*crop.result.box, Source.DETECTED)
    assert entry.brightness == Brightness(bright.result.value, Source.DETECTED)
    assert entry.sample_time == crop.result.hit_pts[0]
    assert isinstance(proof, ProofResult)
    assert proof.window == (float(int(entry.sample_time)), float(int(entry.sample_time + 30.0)))
    assert proof.lines
    assert all(proof.window[0] - 1.0 <= start < end <= proof.window[1] + 5.0 and text
               for start, end, text in proof.lines)
    recompute_all(project, pending={}, ranges_pending=False)
    assert entry.review == ReviewState.PROPOSED
    save_project(project)
    assert load_project(str(tmp_path)).files[name].review == ReviewState.PROPOSED


def _wait_long(events: Events, key: str) -> object:
    return events.terminal(key, timeout=800.0)
