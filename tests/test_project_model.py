"""Tests for the Qt-free project model, JSON store and reconciliation
(core/project/model.py, core/project/store.py).
"""
import json
import logging
from pathlib import Path

import pytest

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
from core.project.store import (
    from_json,
    list_video_files,
    load_project,
    reconcile_files,
    save_project,
    to_json,
)

FIXTURES_V1 = Path(__file__).parent / "fixtures" / "ocr_json_v1"

SLAY_NAMES = [
    "ZS2_-_11_[1080p]TXHBR.mp4",
    "ZS2_-_12_[1080p]TXHBR.mp4",
    "ZS2_-_13_[1080p]TXHBR.mp4",
    "ZS2_-_14_[1080p]TXHBR.mp4",
    "ZS2_-_15_[1080p]TXHBR.mp4",
]


def _touch(path: Path) -> None:
    path.write_bytes(b"")


# --- FolderSettings.labels_only -----------------------------------------


def test_labels_only_true_when_labels_enabled_and_dialogue_disabled():
    fs = FolderSettings(dialogue_enabled=False, labels_enabled=True)
    assert fs.labels_only is True


def test_labels_only_false_when_dialogue_also_enabled():
    fs = FolderSettings(dialogue_enabled=True, labels_enabled=True)
    assert fs.labels_only is False


def test_labels_only_false_when_labels_disabled():
    fs = FolderSettings(dialogue_enabled=False, labels_enabled=False)
    assert fs.labels_only is False


def test_folder_settings_default_labels_enabled_is_true():
    """Matches the old app's default (core.config.Config.labels_enabled =
    True, old UI checkbox default checked) so a fresh, never-saved project
    keeps label detection on like before.
    """
    assert FolderSettings().labels_enabled is True


# --- list_video_files -----------------------------------------------------


def test_list_video_files_sorted_and_filtered_by_extension(tmp_path):
    _touch(tmp_path / "b.mkv")
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "c.txt")
    _touch(tmp_path / "z.mkv")

    assert list_video_files(str(tmp_path)) == ["a.mp4", "b.mkv", "z.mkv"]


def test_list_video_files_empty_directory(tmp_path):
    assert list_video_files(str(tmp_path)) == []


# --- load_project: fresh / v2 ---------------------------------------------


def test_load_project_fresh_when_no_config_file(tmp_path):
    _touch(tmp_path / "one.mkv")
    _touch(tmp_path / "two.mkv")

    project = load_project(str(tmp_path))

    assert project.migrated_from_v1 is False
    assert list(project.files.keys()) == ["one.mkv", "two.mkv"]
    assert all(isinstance(e, FileEntry) for e in project.files.values())
    assert project.files["one.mkv"].review == ReviewState.PENDING
    assert project.folder == FolderSettings()


def test_load_project_fresh_when_no_config_file_has_labels_enabled(tmp_path):
    """A fresh project (no .ocr.json at all) must keep label detection on,
    matching the old app's default -- only a migrated v1 project's own
    saved value should ever turn it off.
    """
    _touch(tmp_path / "one.mkv")

    project = load_project(str(tmp_path))

    assert project.folder.labels_enabled is True


def test_load_project_reads_v2_round_trip(tmp_path):
    _touch(tmp_path / "vid.mkv")
    project = Project(
        path=str(tmp_path),
        folder=FolderSettings(),
        files={"vid.mkv": FileEntry(name="vid.mkv", review=ReviewState.FLAGGED)},
    )
    save_project(project)

    reloaded = load_project(str(tmp_path))

    assert reloaded.files["vid.mkv"].review == ReviewState.FLAGGED
    assert reloaded.migrated_from_v1 is False


def test_load_project_migrates_v1_and_reconciles_with_disk(tmp_path):
    v1_bytes = (FIXTURES_V1 / "slay.json").read_bytes()
    (tmp_path / ".ocr.json").write_bytes(v1_bytes)
    for name in SLAY_NAMES[:-1]:  # drop the last video -> should be reconciled away
        _touch(tmp_path / name)
    _touch(tmp_path / "brand_new_file.mp4")

    project = load_project(str(tmp_path))

    assert project.migrated_from_v1 is True
    assert set(project.files.keys()) == set(SLAY_NAMES[:-1]) | {"brand_new_file.mp4"}
    assert project.files["brand_new_file.mp4"].crop is None
    assert project.files["brand_new_file.mp4"].review == ReviewState.PENDING


# --- reconcile_files --------------------------------------------------------


def test_reconcile_files_returns_added_and_removed(tmp_path):
    project = Project(
        path=str(tmp_path),
        folder=FolderSettings(),
        files={
            "old.mkv": FileEntry(name="old.mkv"),
            "keep.mkv": FileEntry(name="keep.mkv"),
        },
    )

    added, removed = reconcile_files(project, ["keep.mkv", "new.mkv"])

    assert added == ["new.mkv"]
    assert removed == ["old.mkv"]
    assert list(project.files.keys()) == ["keep.mkv", "new.mkv"]


def test_reconcile_files_orders_entries_by_sorted_name(tmp_path):
    project = Project(path=str(tmp_path), folder=FolderSettings(), files={})
    reconcile_files(project, ["c.mkv", "a.mkv", "b.mkv"])
    assert list(project.files.keys()) == ["a.mkv", "b.mkv", "c.mkv"]


def test_reconcile_files_no_changes_returns_empty_lists(tmp_path):
    project = Project(
        path=str(tmp_path),
        folder=FolderSettings(),
        files={"a.mkv": FileEntry(name="a.mkv")},
    )
    added, removed = reconcile_files(project, ["a.mkv"])
    assert added == []
    assert removed == []


# --- save_project: atomicity, v1 backup -------------------------------------


def test_save_project_leaves_no_tmp_file_and_writes_config(tmp_path):
    _touch(tmp_path / "vid.mkv")
    project = Project(
        path=str(tmp_path),
        folder=FolderSettings(),
        files={"vid.mkv": FileEntry(name="vid.mkv")},
    )
    save_project(project)

    assert not (tmp_path / ".ocr.json.tmp").exists()
    assert (tmp_path / ".ocr.json").exists()


def test_save_project_writes_non_ascii_verbatim_not_escaped(tmp_path):
    _touch(tmp_path / "vid.mkv")
    project = Project(
        path=str(tmp_path),
        folder=FolderSettings(),
        files={"vid.mkv": FileEntry(name="vid.mkv", flags={"crop": "中文"})},
    )
    save_project(project)

    raw = (tmp_path / ".ocr.json").read_text(encoding="utf-8")
    assert "中文" in raw
    assert "\\u" not in raw


def test_save_project_backs_up_v1_once_and_second_save_does_not_overwrite(tmp_path):
    v1_bytes = (FIXTURES_V1 / "slay.json").read_bytes()
    (tmp_path / ".ocr.json").write_bytes(v1_bytes)
    for name in SLAY_NAMES:
        _touch(tmp_path / name)

    project = load_project(str(tmp_path))
    assert project.migrated_from_v1 is True

    save_project(project)
    backup = tmp_path / ".ocr.json.v1.bak"
    assert backup.read_bytes() == v1_bytes
    assert project.migrated_from_v1 is False

    config_after_first_save = (tmp_path / ".ocr.json").read_text(encoding="utf-8")
    assert json.loads(config_after_first_save)["version"] == 2

    # Even if migrated_from_v1 were True again, an existing backup must
    # never be clobbered.
    project.migrated_from_v1 = True
    save_project(project)
    assert backup.read_bytes() == v1_bytes


def test_save_project_fresh_project_does_not_write_v1_backup(tmp_path):
    _touch(tmp_path / "vid.mkv")
    project = Project(
        path=str(tmp_path),
        folder=FolderSettings(),
        files={"vid.mkv": FileEntry(name="vid.mkv")},
    )
    save_project(project)
    assert not (tmp_path / ".ocr.json.v1.bak").exists()


# --- corrupt JSON -------------------------------------------------------


def test_load_project_corrupt_json_is_renamed_and_fresh_project_returned(tmp_path, caplog):
    (tmp_path / ".ocr.json").write_text("{not valid json", encoding="utf-8")
    _touch(tmp_path / "vid.mkv")

    with caplog.at_level(logging.WARNING):
        project = load_project(str(tmp_path))

    assert project.migrated_from_v1 is False
    assert list(project.files.keys()) == ["vid.mkv"]
    assert not (tmp_path / ".ocr.json").exists()

    corrupt_files = list(tmp_path.glob(".ocr.json.corrupt-*"))
    assert len(corrupt_files) == 1
    assert any(record.levelno == logging.WARNING for record in caplog.records)


# --- to_json / from_json round trip ----------------------------------------


def _fully_populated_project(tmp_path) -> Project:
    folder = FolderSettings(
        dialogue_enabled=False,
        labels_enabled=True,
        ocr_lang="ch",
        conf_threshold=90,
        sim_threshold=80,
        similar_image=0.25,
        frames_to_skip=2,
        use_gpu=False,
        label_min_duration=0.4,
        label_max_duration=4.5,
        label_conf_threshold=90,
        label_conf_threshold_min=70,
        label_mask_crops=[(1, 2, 3, 4), (5, 6, 7, 8)],
        ocr_parallel=6,
        autopilot_enabled=False,
        brightness_full_detect_files=5,
        min_segment_length=45.0,
        merge_repeating_silences=True,
        crop_width_fraction=0.6,
        crop_vertical_padding=0.01,
        crop_min_height_fraction=0.1,
        bottom_half_cutoff=0.6,
    )
    files = {
        "a.mkv": FileEntry(
            name="a.mkv",
            crop=Crop(x=1, y=2, width=3, height=4, source=Source.MANUAL),
            brightness=Brightness(value=210, source=Source.DETECTED),
            time_ranges=TimeRanges(
                ranges=[
                    TimeRange(start="01:00", end=None),
                    TimeRange(start=None, end="02:00"),
                ],
                source=Source.HINT,
            ),
            media=Media(width=1920, height=1080, duration=120.5, fps=23.976),
            review=ReviewState.FLAGGED,
            skipped=True,
            sample_time=42.5,
            flags={"crop": "low-conf+edge"},
            evidence={"crop": {"score": 0.9}, "brightness": {"plateau": [200, 220]}},
        ),
        "b.mkv": FileEntry(name="b.mkv"),
    }
    return Project(path=str(tmp_path), folder=folder, files=files, migrated_from_v1=False)


def test_round_trip_from_json_to_json_every_field(tmp_path):
    project = _fully_populated_project(tmp_path)
    round_tripped = from_json(to_json(project), str(tmp_path))
    assert round_tripped == project


def test_to_json_output_is_json_serializable_with_string_enum_values(tmp_path):
    project = _fully_populated_project(tmp_path)
    payload = to_json(project)

    dumped = json.dumps(payload)
    reloaded = json.loads(dumped)

    assert reloaded["version"] == 2
    assert reloaded["files"]["a.mkv"]["review"] == "flagged"
    assert reloaded["files"]["a.mkv"]["crop"]["source"] == "manual"
    assert reloaded["files"]["a.mkv"]["brightness"]["source"] == "detected"
    assert reloaded["files"]["a.mkv"]["time_ranges"]["source"] == "hint"
    assert isinstance(reloaded["folder"]["similar_image"], float)
    assert reloaded["files"]["b.mkv"]["crop"] is None
    assert reloaded["files"]["b.mkv"]["review"] == "pending"


def test_to_json_folder_includes_label_mask_crops_as_lists(tmp_path):
    project = _fully_populated_project(tmp_path)
    payload = to_json(project)
    assert payload["folder"]["label_mask_crops"] == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_to_json_evidence_is_deep_copied_mutating_nested_value_does_not_mutate_model(tmp_path):
    """entry.evidence is dict[str, dict] -- a shallow dict(...) copy only
    protects the top-level dict, not the nested per-key dicts, so a caller
    mutating a NESTED value in the returned payload (e.g. changing a score
    inside evidence["crop"]) must not reach back into the model.
    """
    project = _fully_populated_project(tmp_path)
    payload = to_json(project)

    # Nested mutation -- what a shallow dict(...) copy fails to protect.
    payload["files"]["a.mkv"]["evidence"]["crop"]["score"] = 999
    # Top-level mutations too, for good measure.
    payload["files"]["a.mkv"]["evidence"]["new_key"] = {"whatever": True}
    del payload["files"]["a.mkv"]["evidence"]["brightness"]

    assert project.files["a.mkv"].evidence == {
        "crop": {"score": 0.9},
        "brightness": {"plateau": [200, 220]},
    }


def test_from_json_deep_copies_evidence_so_caller_held_input_cant_alias_model(tmp_path):
    """from_json must not alias the model's evidence with the caller's
    input dict either -- mutating the dict the caller passed in (including
    a nested value) after the call must not reach the model.
    """
    evidence_in = {"crop": {"score": 0.5}}
    data = {
        "version": 2,
        "folder": {},
        "files": {"a.mkv": {"evidence": evidence_in}},
    }

    project = from_json(data, str(tmp_path))

    evidence_in["crop"]["score"] = 999
    evidence_in["new_key"] = "leak"

    assert project.files["a.mkv"].evidence == {"crop": {"score": 0.5}}


# --- F6: store robustness ---------------------------------------------------


def _write_config(tmp_path: Path, data) -> bytes:
    raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    (tmp_path / ".ocr.json").write_bytes(raw)
    return raw


def _assert_corrupt_path_taken(tmp_path: Path, raw: bytes, project: Project) -> None:
    assert not (tmp_path / ".ocr.json").exists()
    (corrupt,) = list(tmp_path.glob(".ocr.json.corrupt-*"))
    assert corrupt.read_bytes() == raw
    assert project.migrated_from_v1 is False
    assert project.folder == FolderSettings()
    assert list(project.files) == ["vid.mkv"]
    assert project.files["vid.mkv"] == FileEntry(name="vid.mkv")


@pytest.mark.parametrize("data", [[1, 2], "a string", None, 42])
def test_load_project_json_that_is_not_an_object_takes_the_corrupt_path(tmp_path, caplog, data):
    raw = _write_config(tmp_path, data)
    _touch(tmp_path / "vid.mkv")
    with caplog.at_level(logging.WARNING):
        project = load_project(str(tmp_path))
    _assert_corrupt_path_taken(tmp_path, raw, project)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


@pytest.mark.parametrize("entry", [
    {"review": "bogus"},
    {"crop": {"x": 1, "y": 2, "width": 3, "height": 4, "source": "bogus"}},
    {"brightness": {"value": 200, "source": "guessed"}},
    {"time_ranges": {"ranges": [], "source": 7}},
    {"crop": {"y": 2, "width": 3, "height": 4, "source": "manual"}},       # no x
    {"brightness": {"source": "manual"}},                                  # no value
    {"time_ranges": {"ranges": [{"start": "01:00", "end": None}]}},       # no source
    "not an entry",
    None,
])
def test_load_project_v2_with_a_bad_enum_or_missing_key_takes_the_corrupt_path(tmp_path, entry):
    raw = _write_config(tmp_path, {"version": 2, "folder": {}, "files": {"vid.mkv": entry}})
    _touch(tmp_path / "vid.mkv")
    project = load_project(str(tmp_path))
    _assert_corrupt_path_taken(tmp_path, raw, project)


@pytest.mark.parametrize("data", [
    {"version": 2, "folder": [], "files": {}},
    {"version": 2, "folder": {}, "files": []},
    {"version": 2, "folder": {"label_mask_crops": 5}, "files": {}},
])
def test_load_project_v2_with_malformed_sections_takes_the_corrupt_path(tmp_path, data):
    raw = _write_config(tmp_path, data)
    _touch(tmp_path / "vid.mkv")
    project = load_project(str(tmp_path))
    _assert_corrupt_path_taken(tmp_path, raw, project)


@pytest.mark.parametrize("version", [3, 17, "2", 2.0, 1.5, True, [2], {"major": 2}, 0, -1])
def test_load_project_unsupported_version_raises_and_leaves_the_file_untouched(tmp_path, version):
    from core.project.store import UnsupportedProjectVersion

    raw = _write_config(tmp_path, {"version": version, "folder": {}, "files": {}})
    _touch(tmp_path / "vid.mkv")

    with pytest.raises(UnsupportedProjectVersion) as info:
        load_project(str(tmp_path))

    assert info.value.version == version
    assert (tmp_path / ".ocr.json").read_bytes() == raw
    assert sorted(p.name for p in tmp_path.iterdir()) == [".ocr.json", "vid.mkv"]


def test_save_project_never_overwrites_an_unsupported_version(tmp_path):
    from core.project.store import UnsupportedProjectVersion

    raw = _write_config(tmp_path, {"version": 3, "folder": {}, "files": {"vid.mkv": {"something": "new"}}})
    _touch(tmp_path / "vid.mkv")
    project = Project(path=str(tmp_path), folder=FolderSettings(),
                      files={"vid.mkv": FileEntry(name="vid.mkv", evidence={"crop": {"box": [1, 2, 3, 4]}})})

    with pytest.raises(UnsupportedProjectVersion):
        save_project(project)

    assert (tmp_path / ".ocr.json").read_bytes() == raw
    assert sorted(p.name for p in tmp_path.iterdir()) == [".ocr.json", "vid.mkv"]


def test_unsupported_project_version_is_exported():
    import core.project as package
    from core.project.store import UnsupportedProjectVersion

    assert package.UnsupportedProjectVersion is UnsupportedProjectVersion
    assert issubclass(UnsupportedProjectVersion, Exception)


def test_save_project_fsyncs_every_file_before_replacing_it(tmp_path, monkeypatch):
    import os

    import core.project.store as store

    calls = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        calls.append(("fsync", os.readlink(f"/proc/self/fd/{fd}")))
        return real_fsync(fd)

    def replace(src, dst):
        calls.append(("replace", str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(store.os, "fsync", fsync)
    monkeypatch.setattr(store.os, "replace", replace)
    _touch(tmp_path / "vid.mkv")
    project = Project(path=str(tmp_path), folder=FolderSettings(),
                      files={"vid.mkv": FileEntry(name="vid.mkv", evidence={"crop": {"box": [1, 2, 3, 4]}})})

    save_project(project)

    replaces = [call for call in calls if call[0] == "replace"]
    assert {Path(dst).name for _, _, dst in replaces} >= {".ocr.json"}
    for index, call in enumerate(calls):
        if call[0] == "replace":
            assert calls[index - 1] == ("fsync", call[1])      # that temp file was synced just before
    assert not list(tmp_path.rglob("*.tmp"))
