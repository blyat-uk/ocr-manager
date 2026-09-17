"""Tests for v1 -> v2 project migration (core/project/migrate.py).

Two real, read-only v1 `.ocr.json` fixtures back these tests:
tests/fixtures/ocr_json_v1/{slay,dragon}.json, copied verbatim from
/mnt/FAST/work (never modified; see task-1-brief.md Step 1).
"""
import json
from pathlib import Path

import pytest

from core.project.model import FolderSettings, ReviewState, Source, TimeRange
from core.project.migrate import migrate_v1
from core.project.store import to_json

FIXTURES = Path(__file__).parent / "fixtures" / "ocr_json_v1"

SLAY_NAMES = [
    "ZS2_-_11_[1080p]TXHBR.mp4",
    "ZS2_-_12_[1080p]TXHBR.mp4",
    "ZS2_-_13_[1080p]TXHBR.mp4",
    "ZS2_-_14_[1080p]TXHBR.mp4",
    "ZS2_-_15_[1080p]TXHBR.mp4",
]

DRAGON_NAMES = [
    "LPJT_-_037_[4K]YK10B.mp4",
    "LPJT_-_038_[4K]YK10B.mp4",
    "LPJT_-_039_[4K]YK10B.mp4",
    "LPJT_-_040_[4K]YK10B.mp4",
]

SLAY_DURATIONS = {
    "ZS2_-_11_[1080p]TXHBR.mp4": 1418.0,
    "ZS2_-_12_[1080p]TXHBR.mp4": 1628.0,
    "ZS2_-_13_[1080p]TXHBR.mp4": 1478.0,
    "ZS2_-_14_[1080p]TXHBR.mp4": 1504.0,
    "ZS2_-_15_[1080p]TXHBR.mp4": 1516.0,
}


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# --- Slay the Gods S2 fixture -----------------------------------------


def test_migrate_slay_every_file_has_imported_crop_and_brightness():
    data = _load_fixture("slay.json")
    project = migrate_v1(data, "/tmp/slay", SLAY_NAMES)
    v1_files = data["files"]

    for name in SLAY_NAMES:
        entry = project.files[name]
        v1_crop = v1_files[name]["crop"]
        assert entry.crop.x == v1_crop["x"]
        assert entry.crop.y == v1_crop["y"]
        assert entry.crop.width == v1_crop["width"]
        assert entry.crop.height == v1_crop["height"]
        assert entry.crop.source == Source.IMPORTED

        assert entry.brightness.value == 209
        assert entry.brightness.source == Source.IMPORTED


def test_migrate_slay_zs2_15_has_two_imported_ranges():
    data = _load_fixture("slay.json")
    project = migrate_v1(data, "/tmp/slay", SLAY_NAMES)
    entry = project.files["ZS2_-_15_[1080p]TXHBR.mp4"]

    assert entry.time_ranges.source == Source.IMPORTED
    ranges = [(r.start, r.end) for r in entry.time_ranges.ranges]
    assert ranges == [("02:33", "22:33"), ("23:40", "24:00")]


def test_migrate_slay_media_and_durations():
    data = _load_fixture("slay.json")
    project = migrate_v1(data, "/tmp/slay", SLAY_NAMES)

    for name in SLAY_NAMES:
        entry = project.files[name]
        assert entry.media.width == 1920
        assert entry.media.height == 888
        assert entry.media.duration == SLAY_DURATIONS[name]


def test_migrate_slay_zs2_15_sample_time_from_subtitle_position():
    data = _load_fixture("slay.json")
    project = migrate_v1(data, "/tmp/slay", SLAY_NAMES)
    entry = project.files["ZS2_-_15_[1080p]TXHBR.mp4"]

    assert entry.sample_time == pytest.approx(4039 / 10000 * 1516)


def test_migrate_slay_review_state_is_reviewed_for_every_file():
    data = _load_fixture("slay.json")
    project = migrate_v1(data, "/tmp/slay", SLAY_NAMES)

    for name in SLAY_NAMES:
        assert project.files[name].review == ReviewState.REVIEWED


def test_migrate_slay_folder_settings():
    data = _load_fixture("slay.json")
    project = migrate_v1(data, "/tmp/slay", SLAY_NAMES)
    folder = project.folder

    assert folder.conf_threshold == 95
    assert folder.sim_threshold == 82
    assert folder.similar_image == 0.3
    assert folder.labels_enabled is False
    assert folder.crop_vertical_padding == 0.003   # "0" -> new default (ruling A5)
    assert folder.bottom_half_cutoff == 0.55        # "0.50" -> new default (ruling A5)
    assert folder.min_segment_length == 30.0
    assert folder.merge_repeating_silences is False
    assert project.migrated_from_v1 is True


# --- Dragon fixture -----------------------------------------------------


def test_migrate_dragon_labels_enabled_true():
    data = _load_fixture("dragon.json")
    project = migrate_v1(data, "/tmp/dragon", DRAGON_NAMES)
    assert project.folder.labels_enabled is True


def test_migrate_dragon_global_brightness_applied_to_every_file():
    data = _load_fixture("dragon.json")
    for entry in data["files"].values():
        assert "brightness" not in entry  # sanity: no per-file brightness in this fixture

    project = migrate_v1(data, "/tmp/dragon", DRAGON_NAMES)
    for name in DRAGON_NAMES:
        entry = project.files[name]
        assert entry.brightness.value == 230
        assert entry.brightness.source == Source.IMPORTED


def test_migrate_dragon_detection_batch_size_not_present_anywhere_in_to_json():
    data = _load_fixture("dragon.json")
    assert "detection_batch_size" in data["automation"]  # sanity: fixture really has it

    project = migrate_v1(data, "/tmp/dragon", DRAGON_NAMES)
    payload = to_json(project)
    assert "detection_batch_size" not in payload["folder"]
    assert "detection_batch_size" not in json.dumps(payload)


def test_migrate_dragon_automation_values_not_exactly_old_defaults_carry_over_as_float():
    """Dragon's automation strings are "0.0"/"0.5", not the literal
    "0"/"0.50" the old app actually wrote as defaults (see slay.json), so
    ruling A5's exact-string default mapping does not fire here -- they
    carry over unchanged as floats. This is a real-fixture case beyond
    what the brief's example list spells out.
    """
    data = _load_fixture("dragon.json")
    project = migrate_v1(data, "/tmp/dragon", DRAGON_NAMES)
    folder = project.folder

    assert folder.crop_width_fraction == 0.7
    assert folder.crop_vertical_padding == 0.0
    assert folder.crop_min_height_fraction == 0.05
    assert folder.bottom_half_cutoff == 0.5


# --- Synthetic edge cases -------------------------------------------------


def test_migrate_globals_fill_gaps_for_file_without_own_values():
    data = {
        "version": 1,
        "global": {
            "crop": {"x": 10, "y": 20, "width": 100, "height": 30},
            "time_range": {"start": "01:00", "end": ""},
        },
        "files": {},
    }
    project = migrate_v1(data, "/tmp/proj", ["only.mkv"])
    entry = project.files["only.mkv"]

    assert entry.crop.x == 10
    assert entry.crop.y == 20
    assert entry.crop.width == 100
    assert entry.crop.height == 30
    assert entry.crop.source == Source.IMPORTED

    assert entry.time_ranges.ranges == [TimeRange(start="01:00", end=None)]
    assert entry.time_ranges.source == Source.IMPORTED


def test_migrate_legacy_time_start_end_becomes_one_range():
    data = {
        "version": 1,
        "files": {
            "vid.mkv": {"time_start": "05:00", "time_end": "10:00"},
        },
    }
    project = migrate_v1(data, "/tmp/proj", ["vid.mkv"])
    entry = project.files["vid.mkv"]

    assert entry.time_ranges.ranges == [TimeRange(start="05:00", end="10:00")]
    assert entry.time_ranges.source == Source.IMPORTED


def test_migrate_empty_time_ranges_list_falls_back_to_global():
    data = {
        "global": {"time_range": {"start": "00:10", "end": "00:20"}},
        "files": {"vid.mkv": {"time_ranges": []}},
    }
    project = migrate_v1(data, "/tmp/proj", ["vid.mkv"])
    entry = project.files["vid.mkv"]

    assert entry.time_ranges.ranges == [TimeRange(start="00:10", end="00:20")]


def test_migrate_global_time_range_both_empty_yields_no_ranges():
    data = {
        "global": {"time_range": {"start": "", "end": ""}},
        "files": {"vid.mkv": {}},
    }
    project = migrate_v1(data, "/tmp/proj", ["vid.mkv"])
    assert project.files["vid.mkv"].time_ranges is None


def test_migrate_drops_vanished_files_and_adds_new_globals_only():
    data = {
        "version": 1,
        "global": {},
        "files": {
            "gone.mkv": {
                "crop": {"x": 1, "y": 2, "width": 3, "height": 4},
                "brightness": 100,
            },
            "stays.mkv": {
                "crop": {"x": 5, "y": 6, "width": 7, "height": 8},
                "brightness": 100,
            },
        },
    }
    project = migrate_v1(data, "/tmp/proj", ["stays.mkv", "brand_new.mkv"])

    assert set(project.files.keys()) == {"stays.mkv", "brand_new.mkv"}
    assert project.files["stays.mkv"].review == ReviewState.REVIEWED

    new_entry = project.files["brand_new.mkv"]
    assert new_entry.crop is None
    assert new_entry.brightness is None
    assert new_entry.review == ReviewState.PENDING


def test_migrate_automation_preserves_nondefault_values():
    data = {
        "version": 1,
        "automation": {
            "crop_vertical_padding": "0.01",
            "bottom_half_cutoff": "0.60",
        },
        "files": {},
    }
    project = migrate_v1(data, "/tmp/proj", [])
    assert project.folder.crop_vertical_padding == 0.01
    assert project.folder.bottom_half_cutoff == 0.60


def test_migrate_review_reviewed_when_labels_only_and_brightness_set_without_crop():
    data = {
        "version": 1,
        "global": {"labels_enabled": True, "dialogue_enabled": False},
        "files": {
            "vid.mkv": {"brightness": 200},
        },
    }
    project = migrate_v1(data, "/tmp/proj", ["vid.mkv"])
    assert project.folder.labels_only is True

    entry = project.files["vid.mkv"]
    assert entry.crop is None
    assert entry.brightness is not None
    assert entry.review == ReviewState.REVIEWED


def test_migrate_review_pending_when_crop_set_but_no_brightness():
    data = {
        "version": 1,
        "files": {
            "vid.mkv": {"crop": {"x": 1, "y": 2, "width": 3, "height": 4}},
        },
    }
    project = migrate_v1(data, "/tmp/proj", ["vid.mkv"])
    entry = project.files["vid.mkv"]

    assert entry.crop is not None
    assert entry.brightness is None
    assert entry.review == ReviewState.PENDING


def test_migrate_v1_missing_files_key_entirely_uses_globals_for_every_video():
    """files is omitted entirely by the v1 writer when no file has custom
    config or cached metadata (current-app-inventory.md SS2) -- a real,
    common v1 shape not spelled out by name in the brief's example list.
    """
    data = {"version": 1, "global": {"brightness": 180}}
    project = migrate_v1(data, "/tmp/proj", ["a.mkv", "b.mkv"])

    assert set(project.files.keys()) == {"a.mkv", "b.mkv"}
    for name in project.files:
        assert project.files[name].brightness.value == 180
        assert project.files[name].crop is None


def test_migrate_v1_missing_optional_sections_uses_folder_defaults():
    data = {"version": 1}
    project = migrate_v1(data, "/tmp/proj", [])
    assert project.folder == FolderSettings()


def test_migrate_files_insertion_order_is_sorted():
    data = {"version": 1, "files": {}}
    project = migrate_v1(data, "/tmp/proj", ["c.mkv", "a.mkv", "b.mkv"])
    assert list(project.files.keys()) == ["a.mkv", "b.mkv", "c.mkv"]
