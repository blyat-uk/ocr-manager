"""Bulk edits: one brightness or one crop box for every file of the folder
(core.jobs.apply.apply_brightness_to_all / apply_crop_to_all).

Pinned here:
- the targets are the file the user is on plus every file not skipped;
- each target gets the value as MANUAL; nothing else about it changes (the
  other field, time ranges, evidence);
- the file the user is on is reviewed exactly as a single edit reviews it;
  another file keeps a REVIEWED mark and is never MARKED reviewed by a bulk
  edit it was not looking at;
- a crop is cut to fit each file's own frame, and a file whose box was cut
  is flagged and not left reviewed.
"""
from __future__ import annotations

from core.jobs.apply import (
    FLAG_CROP_CLAMPED,
    apply_brightness_to_all,
    apply_crop_to_all,
    brightness_is_stale,
    bulk_targets,
    recompute_all,
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
)

BOX = (288, 969, 1344, 61)
NEW_BOX = (300, 950, 1320, 70)
HD = Media(1920, 1080, 1200.0, 23.976)
SMALL = Media(1280, 720, 1200.0, 23.976)


IMPORTED_BRIGHTNESS = Brightness(200, Source.IMPORTED)
IMPORTED_CROP = Crop(*BOX, Source.IMPORTED)


def _entry(name: str, *, media=HD, review=ReviewState.PROPOSED, skipped=False,
           brightness=IMPORTED_BRIGHTNESS, crop=IMPORTED_CROP) -> FileEntry:
    return FileEntry(name, crop=crop, brightness=brightness, media=media, review=review, skipped=skipped,
                     time_ranges=TimeRanges([TimeRange("02:16", "18:04")], Source.IMPORTED))


def _project(*entries: FileEntry) -> Project:
    return Project(path="/proj", folder=FolderSettings(), files={e.name: e for e in entries})


def _settle(project: Project) -> None:
    recompute_all(project, pending={}, ranges_pending=False)


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------

def test_the_targets_are_every_file_not_skipped_in_name_order():
    project = _project(_entry("c.mkv"), _entry("a.mkv"), _entry("b.mkv", skipped=True))
    assert bulk_targets(project, "c.mkv") == ["a.mkv", "c.mkv"]


def test_the_file_the_user_is_on_is_a_target_even_when_skipped():
    project = _project(_entry("a.mkv", skipped=True), _entry("b.mkv"))
    assert bulk_targets(project, "a.mkv") == ["a.mkv", "b.mkv"]


# --------------------------------------------------------------------------
# Brightness
# --------------------------------------------------------------------------

def test_every_target_gets_the_brightness_as_manual():
    project = _project(_entry("a.mkv"), _entry("b.mkv"), _entry("c.mkv", skipped=True))
    assert apply_brightness_to_all(project, "a.mkv", 185) == ["a.mkv", "b.mkv"]
    for name in ("a.mkv", "b.mkv"):
        assert project.files[name].brightness == Brightness(185, Source.MANUAL)
    assert project.files["c.mkv"].brightness == Brightness(200, Source.IMPORTED)


def test_a_brightness_bulk_edit_changes_nothing_else():
    project = _project(_entry("a.mkv"), _entry("b.mkv"))
    project.files["b.mkv"].evidence["lines"] = {"crop_box": list(BOX), "samples": []}
    apply_brightness_to_all(project, "a.mkv", 185)
    other = project.files["b.mkv"]
    assert other.crop == Crop(*BOX, Source.IMPORTED)
    assert other.time_ranges == TimeRanges([TimeRange("02:16", "18:04")], Source.IMPORTED)
    assert other.evidence == {"lines": {"crop_box": list(BOX), "samples": []}}


def test_the_file_the_user_is_on_is_reviewed_as_after_keep():
    project = _project(_entry("a.mkv"), _entry("b.mkv"))
    apply_brightness_to_all(project, "a.mkv", 185)
    _settle(project)
    assert project.files["a.mkv"].review == ReviewState.REVIEWED


def test_another_file_keeps_its_review_mark_but_is_never_marked_reviewed():
    project = _project(_entry("a.mkv"), _entry("b.mkv", review=ReviewState.REVIEWED), _entry("c.mkv"))
    apply_brightness_to_all(project, "a.mkv", 185)
    _settle(project)
    assert project.files["b.mkv"].review == ReviewState.REVIEWED
    assert project.files["c.mkv"].review == ReviewState.PROPOSED


# --------------------------------------------------------------------------
# Crop
# --------------------------------------------------------------------------

def test_every_target_gets_the_crop_as_manual():
    project = _project(_entry("a.mkv"), _entry("b.mkv"), _entry("c.mkv", skipped=True))
    assert apply_crop_to_all(project, "a.mkv", NEW_BOX) == ["a.mkv", "b.mkv"]
    for name in ("a.mkv", "b.mkv"):
        assert project.files[name].crop == Crop(*NEW_BOX, Source.MANUAL)
    assert project.files["c.mkv"].crop == Crop(*BOX, Source.IMPORTED)


def test_a_crop_bulk_edit_changes_nothing_else():
    project = _project(_entry("a.mkv"), _entry("b.mkv"))
    apply_crop_to_all(project, "a.mkv", NEW_BOX)
    other = project.files["b.mkv"]
    assert other.brightness == Brightness(200, Source.IMPORTED)
    assert other.time_ranges == TimeRanges([TimeRange("02:16", "18:04")], Source.IMPORTED)


def test_the_crop_is_cut_to_each_files_own_frame_and_the_cut_file_is_flagged():
    project = _project(_entry("a.mkv"), _entry("small.mkv", media=SMALL, review=ReviewState.REVIEWED))
    apply_crop_to_all(project, "a.mkv", NEW_BOX)
    _settle(project)
    small = project.files["small.mkv"]
    assert small.crop.x + small.crop.width <= 1280 and small.crop.y + small.crop.height <= 720
    assert small.crop.source == Source.MANUAL
    assert FLAG_CROP_CLAMPED in small.flags.get("crop", "")
    assert small.review == ReviewState.FLAGGED
    assert FLAG_CROP_CLAMPED not in project.files["a.mkv"].flags.get("crop", "")


def test_a_file_whose_box_fits_is_not_flagged_and_keeps_its_review_mark():
    project = _project(_entry("a.mkv"), _entry("b.mkv", review=ReviewState.REVIEWED), _entry("c.mkv"))
    apply_crop_to_all(project, "a.mkv", NEW_BOX)
    _settle(project)
    assert project.files["a.mkv"].review == ReviewState.REVIEWED
    assert project.files["b.mkv"].review == ReviewState.REVIEWED
    assert project.files["c.mkv"].review == ReviewState.PROPOSED
    assert not project.files["b.mkv"].flags.get("crop")


def test_a_detected_brightness_measured_on_the_old_crop_goes_stale():
    measured = _entry("b.mkv", brightness=Brightness(211, Source.DETECTED))
    measured.evidence["brightness"] = {"value": 211, "value_crop_box": list(BOX)}
    project = _project(_entry("a.mkv"), measured)
    apply_crop_to_all(project, "a.mkv", NEW_BOX)
    assert brightness_is_stale(project.files["b.mkv"]) is True
