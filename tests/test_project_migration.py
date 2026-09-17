"""Tests for v1 -> v2 project migration (core/project/migrate.py).

Two real, read-only v1 `.ocr.json` fixtures back these tests:
tests/fixtures/ocr_json_v1/{slay,dragon}.json, copied verbatim from
/mnt/FAST/work (never modified; see task-1-brief.md Step 1).
"""
import json
import logging
from pathlib import Path

import pytest

from core.project.model import FolderSettings, ReviewState, Source, TimeRange, TimeRanges
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


def test_migrate_dragon_automation_values_map_numerically_despite_different_literal():
    """Dragon's automation strings are "0.0"/"0.5", not the literal
    "0"/"0.50" slay.json uses -- but ruling A5 (amended) compares
    numerically, not as an exact string, because "0.0"/"0.5" are the same
    old-app default re-serialised, just with a different literal. So they
    DO map to the new measured defaults, same as slay.json. This is a
    real-fixture case beyond what the brief's original example list
    spelled out (the amendment exists specifically because of it).
    """
    data = _load_fixture("dragon.json")
    project = migrate_v1(data, "/tmp/dragon", DRAGON_NAMES)
    folder = project.folder

    assert folder.crop_width_fraction == 0.7           # no default-mapping rule; carries over
    assert folder.crop_vertical_padding == 0.003        # "0.0" -> numerically 0 -> new default
    assert folder.crop_min_height_fraction == 0.05      # no default-mapping rule; carries over
    assert folder.bottom_half_cutoff == 0.55             # "0.5" -> numerically 0.50 -> new default


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


def test_migrate_per_file_range_entry_both_empty_is_dropped():
    """A per-file `time_ranges` list entry whose start AND end are both
    empty must not become a TimeRange(None, None) -- dropped instead, same
    as the global.time_range both-empty case above.
    """
    data = {
        "files": {
            "vid.mkv": {
                "time_ranges": [
                    {"start": "01:00", "end": "02:00"},
                    {"start": "", "end": ""},
                ],
            },
        },
    }
    project = migrate_v1(data, "/tmp/proj", ["vid.mkv"])
    entry = project.files["vid.mkv"]

    assert entry.time_ranges.ranges == [TimeRange(start="01:00", end="02:00")]
    assert entry.time_ranges.source == Source.IMPORTED


def test_migrate_per_file_range_list_all_entries_empty_yields_no_ranges():
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
    project = migrate_v1(data, "/tmp/proj", ["vid.mkv"])
    assert project.files["vid.mkv"].time_ranges is None


def test_migrate_legacy_time_start_end_both_empty_yields_no_ranges():
    data = {
        "files": {
            "vid.mkv": {"time_start": "", "time_end": ""},
        },
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 0.003),      # old literal default
        ("0.0", 0.003),    # same default, re-serialised (Dragon's actual shape)
        ("0.01", 0.01),    # non-default, preserved
    ],
)
def test_migrate_crop_vertical_padding_numeric_mapping(raw, expected):
    data = {"automation": {"crop_vertical_padding": raw}, "files": {}}
    project = migrate_v1(data, "/tmp/proj", [])
    assert project.folder.crop_vertical_padding == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.5", 0.55),     # old default, re-serialised (Dragon's actual shape)
        ("0.50", 0.55),    # old literal default
        ("0.60", 0.60),    # non-default, preserved
    ],
)
def test_migrate_bottom_half_cutoff_numeric_mapping(raw, expected):
    data = {"automation": {"bottom_half_cutoff": raw}, "files": {}}
    project = migrate_v1(data, "/tmp/proj", [])
    assert project.folder.bottom_half_cutoff == expected


def test_migrate_automation_unparseable_value_falls_back_to_default_and_logs_warning(caplog):
    data = {
        "automation": {
            "crop_vertical_padding": "not-a-number",
            "bottom_half_cutoff": "also-not-a-number",
        },
        "files": {},
    }
    defaults = FolderSettings()

    with caplog.at_level(logging.WARNING):
        project = migrate_v1(data, "/tmp/proj", [])

    assert project.folder.crop_vertical_padding == defaults.crop_vertical_padding
    assert project.folder.bottom_half_cutoff == defaults.bottom_half_cutoff
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("crop_vertical_padding" in m for m in warnings)
    assert any("bottom_half_cutoff" in m for m in warnings)


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


def test_migrate_resolution_and_duration_only_entry_fills_in_from_globals_and_is_reviewed():
    """A v1 file entry that only has cached `resolution`/`duration` (no own
    crop/brightness) must still pick up the global crop/brightness as
    IMPORTED, keep its own media, and land REVIEWED -- same rule as any
    other file whose crop/brightness come entirely from globals.
    """
    data = {
        "version": 1,
        "global": {
            "crop": {"x": 10, "y": 20, "width": 100, "height": 30},
            "brightness": 215,
        },
        "files": {
            "vid.mkv": {
                "resolution": {"width": 1920, "height": 1080},
                "duration": 600.0,
            },
        },
    }
    project = migrate_v1(data, "/tmp/proj", ["vid.mkv"])
    entry = project.files["vid.mkv"]

    assert entry.crop.x == 10
    assert entry.crop.y == 20
    assert entry.crop.width == 100
    assert entry.crop.height == 30
    assert entry.crop.source == Source.IMPORTED

    assert entry.brightness.value == 215
    assert entry.brightness.source == Source.IMPORTED

    assert entry.media.width == 1920
    assert entry.media.height == 1080
    assert entry.media.duration == 600.0

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


# --- F1: a migrated reviewed file keeps whole-file OCR ----------------------


def _slay_without_ranges() -> dict:
    """The Slay fixture as a folder whose user never set time ranges: no
    per-file ranges and no global time_range."""
    data = _load_fixture("slay.json")
    data["global"].pop("time_range", None)
    for entry in data["files"].values():
        entry.pop("time_ranges", None)
    return data


def _ranges_result(names):
    from core.detect.ranges.pipeline import Block, RangesAnalysis
    from core.jobs.detect_jobs import RangesJobResult

    return RangesJobResult(RangesAnalysis(
        keep={name: [("1:30", "23:00")] for name in names},
        blocks={name: [Block(0.0, 90.0, "intro", len(names), 0.9)] for name in names},
        durations={name: 1500.0 for name in names},
    ))


def test_migrate_reviewed_file_without_ranges_gets_imported_whole_file():
    project = migrate_v1(_slay_without_ranges(), "/tmp/slay", SLAY_NAMES)

    for name in SLAY_NAMES:
        entry = project.files[name]
        assert entry.review == ReviewState.REVIEWED
        assert entry.time_ranges == TimeRanges([], Source.IMPORTED)


def test_migrate_reviewed_whole_file_survives_a_ranges_result():
    from core.jobs.apply import apply_ranges, recompute_all
    from core.project.ocr_kwargs import ocr_call_for

    project = migrate_v1(_slay_without_ranges(), "/tmp/slay", SLAY_NAMES)

    apply_ranges(project, _ranges_result(SLAY_NAMES))
    recompute_all(project, pending={}, ranges_pending=False)

    for name in SLAY_NAMES:
        entry = project.files[name]
        assert entry.review == ReviewState.REVIEWED
        assert entry.time_ranges == TimeRanges([], Source.IMPORTED)
        assert ocr_call_for(entry, project.folder, "/tmp/slay").time_ranges == []
        assert entry.evidence["ranges"]["blocks"]      # the analysis is still kept as evidence


def test_migrate_not_reviewed_file_without_ranges_stays_open_to_detection():
    from core.jobs.apply import apply_ranges

    data = {
        "global": {"brightness": 220, "dialogue_enabled": True, "labels_enabled": False},
        "files": {
            "a.mp4": {"crop": {"x": 1, "y": 800, "width": 1000, "height": 60}},   # reviewed
            "b.mp4": {},                                                           # no crop: pending
        },
    }
    project = migrate_v1(data, "/tmp/proj", ["a.mp4", "b.mp4"])
    assert project.files["a.mp4"].review == ReviewState.REVIEWED
    assert project.files["a.mp4"].time_ranges == TimeRanges([], Source.IMPORTED)
    assert project.files["b.mp4"].review == ReviewState.PENDING
    assert project.files["b.mp4"].time_ranges is None

    apply_ranges(project, _ranges_result(["a.mp4", "b.mp4"]))

    assert project.files["a.mp4"].time_ranges == TimeRanges([], Source.IMPORTED)
    assert project.files["b.mp4"].time_ranges == TimeRanges([TimeRange("1:30", "23:00")], Source.DETECTED)


def test_migrate_reviewed_file_keeps_its_own_imported_ranges():
    """Only a MISSING range becomes the whole file; imported ranges stay."""
    project = migrate_v1(_load_fixture("slay.json"), "/tmp/slay", SLAY_NAMES)
    entry = project.files["ZS2_-_11_[1080p]TXHBR.mp4"]
    assert entry.time_ranges == TimeRanges([TimeRange("02:33", "21:20")], Source.IMPORTED)


def test_migrate_reviewed_whole_file_round_trips_through_the_store(tmp_path):
    from core.project.store import load_project, save_project

    (tmp_path / ".ocr.json").write_text(json.dumps(_slay_without_ranges()), encoding="utf-8")
    for name in SLAY_NAMES:
        (tmp_path / name).write_bytes(b"")

    save_project(load_project(str(tmp_path)))
    reloaded = load_project(str(tmp_path))

    for name in SLAY_NAMES:
        assert reloaded.files[name].time_ranges == TimeRanges([], Source.IMPORTED)
        assert reloaded.files[name].review == ReviewState.REVIEWED
