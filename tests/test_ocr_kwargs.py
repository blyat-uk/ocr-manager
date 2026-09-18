"""Tests for core/project/ocr_kwargs.py -- the exact OCR call per file.

Task 2 fidelity bridge: `ocr_call_for` must reproduce, for the same
resolved settings, exactly the kwargs and time ranges that
`core/ocr_worker.py`'s `OCRWorker._build_ocr_kwargs()` /
`_get_time_ranges()` derived from the old `core.config.Config`/
`FileConfig`. Reduced OCR fidelity is the one unforgivable failure here,
so every case is pinned twice:

1. Against a literal expected dict/list, so this test still protects the
   new code after `core/ocr_worker.py` and `core/config.py` are deleted
   (plan 3B).
2. (Skip-guarded) against the old `OCRWorker` unbound methods themselves,
   fed an equivalent `Config`/`FileConfig` pair via a `SimpleNamespace`
   stand-in. Those modules were deleted in plan 3B Task 6, so this half now
   always skips; it is kept as the shape of the comparison, runnable again
   against a branch that still has the old code.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.project.migrate import migrate_v1
from core.project.model import (
    Brightness,
    Crop,
    FileEntry,
    FolderSettings,
    Source,
    TimeRange,
    TimeRanges,
)
from core.project.ocr_kwargs import DEFAULT_BRIGHTNESS, OcrCall, ocr_call_for

try:
    from core.config import Config, FileConfig
    from core.ocr_worker import OCRWorker
except ImportError:
    Config = FileConfig = OCRWorker = None

skip_if_no_old_code = pytest.mark.skipif(
    OCRWorker is None,
    reason="core.ocr_worker / core.config not importable (removed in plan 3B)",
)

PROJECT_DIR = "/tmp/proj"


def _old_kwargs_and_ranges(entry: FileEntry, folder: FolderSettings, project_dir: str):
    """Build an equivalent old Config/FileConfig pair and run the old
    unbound OCRWorker methods against a SimpleNamespace stand-in, per
    task-2-brief.md Step 1.
    """
    gc = Config(
        crop_x=0, crop_y=0, crop_width=0, crop_height=0,
        brightness=DEFAULT_BRIGHTNESS,
        time_ranges=[],
        labels_enabled=folder.labels_enabled,
        labels_only=folder.labels_only,
        label_min_duration=folder.label_min_duration,
        label_max_duration=folder.label_max_duration,
        label_conf_threshold=folder.label_conf_threshold,
        label_conf_threshold_min=folder.label_conf_threshold_min,
        label_mask_crops=list(folder.label_mask_crops),
        ocr_lang=folder.ocr_lang,
        conf_threshold=folder.conf_threshold,
        sim_threshold=folder.sim_threshold,
        similar_image=folder.similar_image,
        frames_to_skip=folder.frames_to_skip,
        use_gpu=folder.use_gpu,
    )

    fc = FileConfig(filename=entry.name)
    if entry.brightness is not None:
        fc.brightness = entry.brightness.value
    if entry.crop is not None:
        fc.crop_x = entry.crop.x
        fc.crop_y = entry.crop.y
        fc.crop_width = entry.crop.width
        fc.crop_height = entry.crop.height
    if entry.time_ranges is not None:
        fc.time_ranges = [(r.start, r.end) for r in entry.time_ranges.ranges]

    stub = SimpleNamespace(
        file_config=fc,
        config=gc,
        video_path=Path(project_dir) / entry.name,
    )
    old_kwargs = OCRWorker._build_ocr_kwargs(stub)
    old_ranges = OCRWorker._get_time_ranges(stub)
    return old_kwargs, old_ranges


# --- fixtures shared between the literal-pin and parity sections -----------


def _entry_crop_and_brightness() -> FileEntry:
    return FileEntry(
        name="crop_and_brightness.mkv",
        crop=Crop(x=10, y=20, width=300, height=50, source=Source.DETECTED),
        brightness=Brightness(value=200, source=Source.DETECTED),
    )


def _entry_no_crop_no_brightness() -> FileEntry:
    return FileEntry(name="no_crop.mkv")


def _entry_zero_width_crop() -> FileEntry:
    return FileEntry(
        name="zero_width_crop.mkv",
        crop=Crop(x=1, y=2, width=0, height=50, source=Source.MANUAL),
    )


def _entry_multi_range() -> FileEntry:
    return FileEntry(
        name="multi_range.mkv",
        time_ranges=TimeRanges(
            ranges=[
                TimeRange(start="01:00", end="02:00"),
                TimeRange(start="05:00", end=None),
                TimeRange(start=None, end="09:00"),
            ],
            source=Source.MANUAL,
        ),
    )


def _entry_open_ended_range() -> FileEntry:
    return FileEntry(
        name="open_ended.mkv",
        time_ranges=TimeRanges(ranges=[TimeRange(start="10:00", end=None)], source=Source.MANUAL),
    )


def _entry_no_ranges() -> FileEntry:
    return FileEntry(name="no_ranges.mkv")


# --- literal-pin tests (always run, survive OCRWorker's deletion) ----------


def test_default_brightness_constant_is_230():
    assert DEFAULT_BRIGHTNESS == 230


def test_crop_and_brightness_produce_expected_kwargs():
    entry = _entry_crop_and_brightness()
    # labels_enabled=False: isolates crop/brightness from the label kwargs,
    # which get their own tests below. FolderSettings()'s own default is
    # now True (matches the old app -- see test_labels_disabled_... for the
    # labels-off shape pinned on its own).
    folder = FolderSettings(labels_enabled=False)

    call = ocr_call_for(entry, folder, PROJECT_DIR)

    assert call == OcrCall(
        kwargs={
            "video_path": os.path.join(PROJECT_DIR, "crop_and_brightness.mkv"),
            "lang": "ch",
            "conf_threshold": 95,
            "sim_threshold": 82,
            "brightness_threshold": 200,
            "similar_image_threshold": 0.3,
            "frames_to_skip": 0,
            "use_gpu": True,
            "crop_x": 10,
            "crop_y": 20,
            "crop_width": 300,
            "crop_height": 50,
            "detect_labels": False,
        },
        time_ranges=[],
    )


def test_no_crop_falls_back_to_default_brightness():
    entry = _entry_no_crop_no_brightness()
    folder = FolderSettings(labels_enabled=False)  # isolates brightness fallback from label kwargs

    call = ocr_call_for(entry, folder, PROJECT_DIR)

    assert call == OcrCall(
        kwargs={
            "video_path": os.path.join(PROJECT_DIR, "no_crop.mkv"),
            "lang": "ch",
            "conf_threshold": 95,
            "sim_threshold": 82,
            "brightness_threshold": DEFAULT_BRIGHTNESS,
            "similar_image_threshold": 0.3,
            "frames_to_skip": 0,
            "use_gpu": True,
            "detect_labels": False,
        },
        time_ranges=[],
    )
    assert "crop_x" not in call.kwargs
    assert "crop_y" not in call.kwargs
    assert "crop_width" not in call.kwargs
    assert "crop_height" not in call.kwargs


def test_crop_with_zero_width_is_excluded_from_kwargs():
    entry = _entry_zero_width_crop()
    folder = FolderSettings()

    call = ocr_call_for(entry, folder, PROJECT_DIR)

    assert "crop_x" not in call.kwargs
    assert "crop_width" not in call.kwargs
    assert "crop_height" not in call.kwargs


def test_labels_on_dialogue_also_on_produces_label_kwargs_without_only_labels():
    entry = _entry_no_crop_no_brightness()
    folder = FolderSettings(dialogue_enabled=True, labels_enabled=True)
    assert folder.labels_only is False

    call = ocr_call_for(entry, folder, PROJECT_DIR)

    assert call.kwargs["detect_labels"] is True
    assert "only_labels" not in call.kwargs
    assert call.kwargs["label_min_duration"] == 0.5
    assert call.kwargs["label_max_duration"] == 5.0
    assert call.kwargs["label_conf_threshold"] == 95
    assert call.kwargs["label_conf_threshold_min"] == 80
    assert "label_mask_crops" not in call.kwargs


def test_labels_only_sets_only_labels_true():
    entry = _entry_no_crop_no_brightness()
    folder = FolderSettings(dialogue_enabled=False, labels_enabled=True)
    assert folder.labels_only is True

    call = ocr_call_for(entry, folder, PROJECT_DIR)

    assert call.kwargs["detect_labels"] is True
    assert call.kwargs["only_labels"] is True


def test_labels_disabled_sets_detect_labels_false_and_no_other_label_kwargs():
    entry = _entry_no_crop_no_brightness()
    folder = FolderSettings(labels_enabled=False)

    call = ocr_call_for(entry, folder, PROJECT_DIR)

    assert call.kwargs["detect_labels"] is False
    for key in (
        "only_labels", "label_min_duration", "label_max_duration",
        "label_conf_threshold", "label_conf_threshold_min", "label_mask_crops",
    ):
        assert key not in call.kwargs


def test_label_mask_crops_included_when_non_empty():
    entry = _entry_no_crop_no_brightness()
    folder = FolderSettings(labels_enabled=True, label_mask_crops=[(1, 2, 3, 4), (5, 6, 7, 8)])

    call = ocr_call_for(entry, folder, PROJECT_DIR)

    assert call.kwargs["label_mask_crops"] == [(1, 2, 3, 4), (5, 6, 7, 8)]


def test_label_mask_crops_excluded_when_empty_even_with_labels_on():
    entry = _entry_no_crop_no_brightness()
    folder = FolderSettings(labels_enabled=True, label_mask_crops=[])

    call = ocr_call_for(entry, folder, PROJECT_DIR)

    assert "label_mask_crops" not in call.kwargs


def test_multi_range_maps_none_start_to_0_00_and_none_end_to_empty_string():
    entry = _entry_multi_range()
    folder = FolderSettings()

    call = ocr_call_for(entry, folder, PROJECT_DIR)

    assert call.time_ranges == [
        ("01:00", "02:00"),
        ("05:00", ""),
        ("0:00", "09:00"),
    ]


def test_one_open_ended_range():
    entry = _entry_open_ended_range()
    folder = FolderSettings()

    call = ocr_call_for(entry, folder, PROJECT_DIR)

    assert call.time_ranges == [("10:00", "")]


def test_no_ranges_yields_empty_list():
    entry = _entry_no_ranges()
    folder = FolderSettings()

    call = ocr_call_for(entry, folder, PROJECT_DIR)

    assert call.time_ranges == []


# --- composing regression: migrate_v1() feeding straight into ocr_call_for -


def test_migrated_all_empty_time_ranges_entry_composes_to_empty_ocr_call_time_ranges():
    """A v1 file entry whose only `time_ranges` list entries are all empty
    (start AND end both blank) migrates via `migrate_v1()` to
    `time_ranges=None` (whole file -- the Task 1 follow-up drop-empty-
    ranges fix in core/project/migrate.py), and `ocr_call_for()` then maps
    that `None` the same way it maps a file with no `time_ranges` key at
    all: to `[]`.

    This composes to the exact same OCR request the old app made for this
    shape: the run job only ever calls `save_subtitles_to_file` (its
    single-session, <=1-range call) when `len(time_ranges) <= 1`, passing
    `time_start='0:00', time_end=''` for the implicit whole-file case --
    i.e. an empty `[]` list here and the old code's one-element
    `[('0:00', '')]` list both resolve to that identical single whole-file
    call; `[]` is just how a fully-resolved FileEntry represents "no
    ranges" instead of carrying an explicit placeholder range around.
    """
    data = {
        "files": {
            "vid.mkv": {
                "time_ranges": [
                    {"start": "", "end": ""},
                    {"start": None, "end": None},
                ],
            },
        },
    }
    project = migrate_v1(data, PROJECT_DIR, ["vid.mkv"])
    entry = project.files["vid.mkv"]
    assert entry.time_ranges is None  # pins the Task 1 follow-up this composes with

    call = ocr_call_for(entry, project.folder, PROJECT_DIR)

    assert call.time_ranges == []


def test_video_path_joins_project_dir_and_entry_name():
    entry = FileEntry(name="weird name (1).mkv")
    folder = FolderSettings()

    call = ocr_call_for(entry, folder, "/mnt/FAST/work/proj")

    assert call.kwargs["video_path"] == os.path.join("/mnt/FAST/work/proj", "weird name (1).mkv")


# --- parity vs the old OCRWorker (skipped once core/ocr_worker.py is gone) -


@skip_if_no_old_code
@pytest.mark.parametrize(
    ("entry_factory", "folder"),
    [
        (_entry_crop_and_brightness, FolderSettings()),
        (_entry_no_crop_no_brightness, FolderSettings()),
        (_entry_zero_width_crop, FolderSettings()),
        (_entry_no_crop_no_brightness, FolderSettings(dialogue_enabled=True, labels_enabled=True)),
        (_entry_no_crop_no_brightness, FolderSettings(dialogue_enabled=False, labels_enabled=True)),
        (
            _entry_no_crop_no_brightness,
            FolderSettings(labels_enabled=True, label_mask_crops=[(1, 2, 3, 4), (5, 6, 7, 8)]),
        ),
        (_entry_multi_range, FolderSettings()),
        (_entry_open_ended_range, FolderSettings()),
        (_entry_no_ranges, FolderSettings()),
    ],
    ids=[
        "crop_and_brightness",
        "no_crop",
        "zero_width_crop",
        "labels_on",
        "labels_only",
        "mask_crops",
        "multi_range",
        "open_ended_range",
        "no_ranges",
    ],
)
def test_matches_old_ocr_worker(entry_factory, folder):
    entry = entry_factory()

    new_call = ocr_call_for(entry, folder, PROJECT_DIR)
    old_kwargs, old_ranges = _old_kwargs_and_ranges(entry, folder, PROJECT_DIR)

    assert new_call.kwargs == old_kwargs
    assert new_call.time_ranges == old_ranges


# --------------------------------------------------------------------------
# The stored crop is the region videocr really slices
# --------------------------------------------------------------------------
# A crop the frame cannot hold is narrowed by videocr/video.py's own clamp
# (or dropped entirely, at which point the OCR pass falls back to "only use
# bottom third of the frame"), so the run would not read the band the user
# reviewed. Every writer of a crop clamps with `clamp_crop_box`, and this
# pins that the clamped value survives videocr's clamp untouched.

@pytest.mark.parametrize(
    "frame_size, box",
    [
        ((1280, 720), (288, 784, 1344, 55)),      # a 1080p crop pasted onto a 720p file
        ((1920, 1080), (288, 786, 1344, 53)),     # already fits: unchanged
        ((640, 360), (0, 0, 1920, 1080)),         # larger than the frame in both directions
        ((1920, 1080), (-40, -10, 4000, 4000)),   # negative origin
        ((1920, 1080), (1919, 1079, 2, 2)),       # smaller than the minimum side
        ((16, 12), (5, 5, 400, 400)),             # a frame smaller than the minimum side
        ((3840, 2160), (100, 2100, 3000, 400)),   # runs past the bottom edge
    ],
    ids=["paste_1080p_onto_720p", "fits", "larger_both_ways", "negative_origin",
         "below_minimum", "tiny_frame", "past_bottom"],
)
def test_a_stored_crop_is_the_region_videocr_slices(frame_size, box):
    from core.project.model import clamp_crop_box
    from videocr.video import infer_crop_region

    stored = clamp_crop_box(box, frame_size)
    width, height = frame_size
    region = infer_crop_region(width, height, *stored)
    assert region is not None, "videocr would drop this crop and OCR the bottom third instead"
    x_start, y_start, x_end, y_end = region
    assert (x_start, y_start, x_end - x_start, y_end - y_start) == stored


def test_videocr_narrows_or_drops_a_crop_the_frame_cannot_hold():
    """The behaviour clamp_crop_box exists to keep out of the model."""
    from videocr.video import infer_crop_region

    # The verified paste-across-resolutions box: videocr keeps x and cuts the
    # width, and the height clamps to zero, so the crop is dropped entirely.
    assert infer_crop_region(1280, 720, 288, 784, 1344, 55) is None
    # Same box one row higher: kept, but 352 px narrower than the value says.
    assert infer_crop_region(1280, 720, 288, 600, 1344, 55) == (288, 600, 1280, 655)
