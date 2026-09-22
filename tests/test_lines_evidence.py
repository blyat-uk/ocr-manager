"""The gallery lines' evidence (core.jobs.apply.apply_lines) and why lines
never count as a detection.

Lines are view evidence for the Brightness tab: random subtitle lines of the
file, grabbed on its crop (docs/superpowers/specs/2026-09-18-brightness-
gallery-design.md, section 2). Pinned here:

- apply_lines keeps a result only while the file still has the crop the
  lines were grabbed on, stores it as evidence["lines"] with that crop, and
  touches nothing else: no value, source, flag, review state or sample time;
- a "lines" kind pending or running never changes a file's review state
  (compute_review_state) or its queue badge (app.state_text.badge_for).
"""
from __future__ import annotations

import copy
import json

import pytest

from app.state_text import badge_for
from core.detect.crop import FLAG_LOW_AGREEMENT
from core.detect.lines import LineSample, LinesResult
from core.jobs.apply import apply_lines, compute_review_state, recompute_all
from core.jobs.detect_jobs import LinesJobResult
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
    load_project,
    save_project,
    to_json,
)

BOX = (288, 786, 1344, 53)
OTHER_BOX = (300, 800, 1300, 60)
NAME = "a.mp4"


def _result(box=BOX, seed=7, samples=None, tried=24, cancelled=False, file=NAME) -> LinesJobResult:
    if samples is None:
        samples = (LineSample(512.3, ((10, 5, 200, 30),), 1),
                   LineSample(900.0, ((10, 5, 200, 30), (12, 40, 180, 30)), 2))
    return LinesJobResult(file, LinesResult(tuple(samples), tried, seed, cancelled), box)


def _project(tmp_path=None, crop=BOX, **entry) -> Project:
    values = {"crop": None if crop is None else Crop(*crop, Source.DETECTED),
              "brightness": Brightness(209, Source.DETECTED),
              "media": Media(1920, 1080, 1418.0, 23.976), "sample_time": 407.4}
    values.update(entry)
    files = {NAME: FileEntry(NAME, **values), "b.mp4": FileEntry("b.mp4")}
    return Project(path="/proj" if tmp_path is None else str(tmp_path), folder=FolderSettings(), files=files)


# --------------------------------------------------------------------------
# apply_lines: what is kept
# --------------------------------------------------------------------------

def test_lines_on_the_files_crop_are_stored_with_that_crop():
    project = _project()
    apply_lines(project, _result())
    assert project.files[NAME].evidence["lines"] == {
        "crop_box": list(BOX),
        "seed": 7,
        "tried": 24,
        "samples": [{"time": 512.3, "boxes": [[10, 5, 200, 30]], "lines": 1},
                    {"time": 900.0, "boxes": [[10, 5, 200, 30], [12, 40, 180, 30]], "lines": 2}],
    }


def test_the_crop_box_is_stored_as_plain_ints():
    project = _project()
    apply_lines(project, _result(box=tuple(float(v) for v in BOX)))
    stored = project.files[NAME].evidence["lines"]["crop_box"]
    assert stored == list(BOX) and all(type(v) is int for v in stored)


def test_a_draw_that_found_no_lines_is_stored_too():
    """"no subtitle lines found in {tried} frames" is something to show."""
    project = _project()
    apply_lines(project, _result(samples=(), tried=48))
    assert project.files[NAME].evidence["lines"] == {"crop_box": list(BOX), "seed": 7, "tried": 48, "samples": []}


def test_a_new_draw_replaces_the_old_lines():
    project = _project()
    apply_lines(project, _result(seed=7))
    apply_lines(project, _result(seed=8, samples=(LineSample(30.0, ((1, 2, 3, 4),), 1),)))
    lines = project.files[NAME].evidence["lines"]
    assert lines["seed"] == 8 and [s["time"] for s in lines["samples"]] == [30.0]


@pytest.mark.parametrize("source", list(Source))
def test_lines_follow_the_crop_whatever_its_source(source):
    project = _project()
    project.files[NAME].crop = Crop(*BOX, source)
    apply_lines(project, _result())
    assert project.files[NAME].evidence["lines"]["crop_box"] == list(BOX)


# --------------------------------------------------------------------------
# apply_lines: what is dropped
# --------------------------------------------------------------------------

def _dropped(project: Project, result) -> None:
    before = copy.deepcopy(to_json(project))
    evidence = copy.deepcopy({name: entry.evidence for name, entry in project.files.items()})
    apply_lines(project, result)
    assert to_json(project) == before
    assert {name: entry.evidence for name, entry in project.files.items()} == evidence


def test_a_cancelled_job_changes_nothing():
    _dropped(_project(), None)


def test_a_result_the_sampler_marked_cancelled_changes_nothing():
    _dropped(_project(), _result(samples=(), cancelled=True))


def test_lines_of_a_file_that_is_gone_change_nothing():
    _dropped(_project(), _result(file="gone.mp4"))


def test_lines_of_a_file_without_a_crop_are_dropped():
    _dropped(_project(crop=None), _result())


@pytest.mark.parametrize("current", [OTHER_BOX, (BOX[0], BOX[1], BOX[2], BOX[3] + 1)])
def test_lines_grabbed_on_another_crop_are_dropped_and_the_old_lines_stay(current):
    project = _project(crop=current)
    old = {"crop_box": list(current), "seed": 1, "tried": 12, "samples": []}
    project.files[NAME].evidence["lines"] = copy.deepcopy(old)
    _dropped(project, _result(box=BOX))
    assert project.files[NAME].evidence["lines"] == old


def test_lines_without_a_crop_box_are_dropped():
    _dropped(_project(), _result(box=None))


# --------------------------------------------------------------------------
# apply_lines touches nothing but evidence["lines"]
# --------------------------------------------------------------------------

@pytest.mark.parametrize("review", list(ReviewState))
@pytest.mark.parametrize("flags", [{}, {"crop": FLAG_LOW_AGREEMENT, "brightness": "escalate"}])
@pytest.mark.parametrize("sources", [(Source.DETECTED, Source.HINT), (Source.MANUAL, Source.IMPORTED)])
def test_applying_lines_changes_no_value_flag_review_or_sample_time(review, flags, sources):
    crop_source, brightness_source = sources
    project = _project(crop=None, review=review, flags=dict(flags),
                       time_ranges=TimeRanges([TimeRange("01:30", "20:00")], Source.DETECTED))
    entry = project.files[NAME]
    entry.crop = Crop(*BOX, crop_source)
    entry.brightness = Brightness(209, brightness_source)
    entry.evidence["brightness"] = {"crop_box": list(BOX), "value_crop_box": list(BOX), "tiles": {"dark": 12.0}}
    entry.evidence["audio"] = {"speech": [[1.0, 2.0]]}
    before = copy.deepcopy(to_json(project, include_evidence=False))
    other_evidence = copy.deepcopy(entry.evidence)

    apply_lines(project, _result())

    # values, sources, flags, review, sample_time, media: all of .ocr.json
    assert to_json(project, include_evidence=False) == before
    assert {k: v for k, v in entry.evidence.items() if k != "lines"} == other_evidence
    assert set(entry.evidence) == set(other_evidence) | {"lines"}


def test_lines_evidence_round_trips_through_the_evidence_cache(tmp_path):
    project = _project(tmp_path)
    for name in project.files:
        (tmp_path / name).write_bytes(b"placeholder video")
    apply_lines(project, _result())
    json.dumps(project.files[NAME].evidence)            # JSON-able as stored
    save_project(project)
    restored = load_project(str(tmp_path))
    assert restored.files[NAME].evidence["lines"] == project.files[NAME].evidence["lines"]


# --------------------------------------------------------------------------
# "lines" is never a detection the review state or the badge waits for
# --------------------------------------------------------------------------

DIALOGUE = FolderSettings()
LABELS_ONLY = FolderSettings(dialogue_enabled=False, labels_enabled=True)


def _entries() -> list[FileEntry]:
    """One entry for every rung of compute_review_state."""
    def entry(**values):
        base = {"media": Media(1920, 1080, 1418.0, 23.976), "crop": Crop(*BOX, Source.DETECTED),
                "brightness": Brightness(209, Source.DETECTED)}
        base.update(values)
        return FileEntry(NAME, **base)

    return [
        entry(),                                                            # proposed
        entry(crop=None),                                                   # crop missing
        entry(brightness=None),                                             # brightness missing
        entry(flags={"crop": FLAG_LOW_AGREEMENT}),                          # flagged
        entry(review=ReviewState.REVIEWED, crop=Crop(*BOX, Source.MANUAL),
              brightness=Brightness(209, Source.MANUAL)),                   # reviewed
        entry(evidence={"brightness": {"value_crop_box": list(OTHER_BOX)}}),   # stale brightness
        entry(time_ranges=None),                                            # ranges still to come
        entry(skipped=True),
    ]


@pytest.mark.parametrize("folder", [DIALOGUE, LABELS_ONLY])
@pytest.mark.parametrize("index", range(len(_entries())))
@pytest.mark.parametrize("others", [set(), {"crop"}, {"brightness"}, {"thumbnail", "audio_profile"}])
@pytest.mark.parametrize("ranges_pending", [False, True])
def test_a_pending_lines_job_never_changes_the_review_state(folder, index, others, ranges_pending):
    entry = _entries()[index]
    without = compute_review_state(entry, folder, detections_pending=set(others), ranges_pending=ranges_pending)
    with_lines = compute_review_state(entry, folder, detections_pending=set(others) | {"lines"},
                                      ranges_pending=ranges_pending)
    assert with_lines == without


def test_recompute_all_ignores_lines_in_the_pending_map():
    project = _project()
    project.files["b.mp4"].crop = Crop(*BOX, Source.MANUAL)
    recompute_all(project, pending={}, ranges_pending=False)
    states = {name: entry.review for name, entry in project.files.items()}
    recompute_all(project, pending={name: {"lines"} for name in project.files}, ranges_pending=False)
    assert {name: entry.review for name, entry in project.files.items()} == states


@pytest.mark.parametrize("review", list(ReviewState))
@pytest.mark.parametrize("others", [set(), {"crop"}, {"brightness"}, {"ranges"}, {"audio_profile"}])
@pytest.mark.parametrize("skipped, done", [(False, False), (True, False), (False, True)])
def test_a_running_lines_job_never_changes_the_queue_badge(review, others, skipped, done):
    entry = FileEntry(NAME, crop=Crop(*BOX, Source.DETECTED), brightness=None, review=review,
                      skipped=skipped, flags={"crop": FLAG_LOW_AGREEMENT})
    without = badge_for(entry, running_detectors=set(others), done=done, run_state=None)
    with_lines = badge_for(entry, running_detectors=set(others) | {"lines"}, done=done, run_state=None)
    assert with_lines == without
