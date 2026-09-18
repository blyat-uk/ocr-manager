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
    assert len(replaces) == 2                                 # the config and one evidence file
    for index, call in enumerate(calls):
        if call[0] == "replace":
            assert calls[index - 1] == ("fsync", call[1])      # that temp file was synced just before
    assert not list(tmp_path.rglob("*.tmp"))


# --- F3: evidence lives in .ocr-cache/evidence, not in .ocr.json -------------


def _crop_evidence(i: int = 0) -> dict:
    return {"box": [288, 780 + i, 1344, 61], "envelope": [300, 786, 1300, 50], "agreed": 8, "probes_used": 20,
            "flagged": None, "hit_pts": [500.0 + k for k in range(8)], "frame_size": [1920, 888],
            "cutoff_frac": 0.55,
            "samples": [{"time": 500.123 + k, "boxes": [[400 + k, 786, 900, 30], [420, 820, 850, 30]],
                         "kept": k % 2 == 0, "lines": 2} for k in range(20)]}


def _brightness_evidence(i: int = 0) -> dict:
    return {"value": 209 - i, "plateau": [190, 229], "seed": 228, "gate_floor": 180, "flagged": "no-speech",
            "curve": [[t, 0.97] for t in range(203, 254, 5)], "clutter_curve": [[t, 0.1] for t in range(203, 254, 5)],
            "strips": [{"time": 100.5 + k, "is_text": k % 3 != 0, "glyph_level": 240, "background_level": 40.123,
                        "stroke_px": 3.4567, "lines": 1, "boxes": [[10, 5, 600, 40], [620, 5, 300, 40]],
                        "gate_at_value": None} for k in range(48)],
            "tiles": {"dark": 101.5, "bright": 110.5, "thin": 120.5, "two_line": 130.5, "leaking": 103.5},
            "crop_box": [288, 780, 1344, 61], "value_crop_box": [288, 780, 1344, 61]}


def _audio_evidence() -> dict:
    return {"envelope": [round((k * 37 % 101) / 100, 4) for k in range(600)],
            "speech": [[k * 7.25, k * 7.25 + 3.5] for k in range(200)], "duration": 1418.0}


def _ranges_evidence() -> dict:
    return {"blocks": [{"start_sec": 0.0, "end_sec": 90.0, "kind": "intro", "matched_files": 5, "score": 0.9},
                       {"start_sec": 1330.5, "end_sec": 1418.0, "kind": "outro", "matched_files": 4, "score": 0.8}],
            "duration": 1418.0}


def _full_evidence(i: int = 0) -> dict:
    return {"crop": _crop_evidence(i), "brightness": _brightness_evidence(i), "audio": _audio_evidence(),
            "ranges": _ranges_evidence()}


def _evidence_project(tmp_path: Path, names, evidence=_full_evidence) -> Project:
    files = {}
    for i, name in enumerate(names):
        _touch(tmp_path / name)
        files[name] = FileEntry(
            name=name,
            crop=Crop(288, 780 + i, 1344, 61, Source.DETECTED),
            brightness=Brightness(209 - i, Source.DETECTED),
            time_ranges=TimeRanges([TimeRange("1:30", "23:00")], Source.DETECTED),
            media=Media(1920, 888, 1418.0, 23.976),
            review=ReviewState.PROPOSED,
            sample_time=500.0,
            flags={"crop": "", "brightness": "no-speech"},
            evidence=evidence(i) if evidence else {},
        )
    return Project(path=str(tmp_path), folder=FolderSettings(), files=files)


def _evidence_dir(tmp_path: Path) -> Path:
    return tmp_path / ".ocr-cache" / "evidence"


class _WriteSpy:
    def __init__(self, monkeypatch):
        import core.project.store as store

        self.paths: list[Path] = []
        real = store._atomic_write_text

        def spy(path, text):
            self.paths.append(Path(path))
            return real(path, text)

        monkeypatch.setattr(store, "_atomic_write_text", spy)

    def take(self) -> list[str]:
        names = sorted(path.name for path in self.paths)
        self.paths.clear()
        return names


NAMES_CJK = ["ep01 第一集.mkv", "ep02.mp4"]


def test_every_evidence_kind_round_trips_through_the_evidence_cache(tmp_path):
    project = _evidence_project(tmp_path, NAMES_CJK)
    project.files["ep02.mp4"].evidence["crop"]["flagged"] = "中文+low-agreement"
    expected = {name: json.loads(json.dumps(entry.evidence)) for name, entry in project.files.items()}

    save_project(project)
    reloaded = load_project(str(tmp_path))

    assert {name: entry.evidence for name, entry in reloaded.files.items()} == expected
    assert reloaded == project                                   # every other field too


def test_ocr_json_holds_no_evidence_and_its_size_does_not_depend_on_it(tmp_path):
    small, large = tmp_path / "small", tmp_path / "large"
    small.mkdir()
    large.mkdir()
    # Only the staleness record (value_crop_box) is kept in .ocr.json: the same in both.
    save_project(_evidence_project(small, NAMES_CJK, evidence=lambda i: {
        "crop": {"box": [1, 2, 3, 4]}, "brightness": {"value_crop_box": [288, 780, 1344, 61]}}))
    save_project(_evidence_project(large, NAMES_CJK))

    config = json.loads((large / ".ocr.json").read_text(encoding="utf-8"))
    assert all("evidence" not in entry for entry in config["files"].values())
    assert (small / ".ocr.json").read_bytes() == (large / ".ocr.json").read_bytes()


def test_evidence_files_are_compact_utf8_json_named_by_the_hash_of_the_file_name(tmp_path):
    import hashlib

    from core.project.store import evidence_path

    project = _evidence_project(tmp_path, NAMES_CJK)
    save_project(project)

    for name, entry in project.files.items():
        path = evidence_path(str(tmp_path), name)
        assert path == _evidence_dir(tmp_path) / (hashlib.sha256(name.encode("utf-8")).hexdigest() + ".json")
        text = path.read_text(encoding="utf-8")
        assert text == json.dumps({"version": 1, "name": name, "evidence": entry.evidence},
                                  separators=(",", ":"), ensure_ascii=False)
    assert "第一集" in evidence_path(str(tmp_path), NAMES_CJK[0]).read_text(encoding="utf-8")
    assert sorted(p.name for p in _evidence_dir(tmp_path).iterdir()) == sorted(
        evidence_path(str(tmp_path), name).name for name in NAMES_CJK)


def test_unchanged_evidence_is_not_rewritten(tmp_path, monkeypatch):
    from core.project.store import evidence_path

    spy = _WriteSpy(monkeypatch)
    names = ["a.mkv", "b.mkv", "c.mkv"]
    file_of = {name: evidence_path(str(tmp_path), name).name for name in names}
    project = _evidence_project(tmp_path, names)

    save_project(project)
    assert spy.take() == sorted([".ocr.json"] + list(file_of.values()))

    save_project(project)
    assert spy.take() == [".ocr.json"]

    reloaded = load_project(str(tmp_path))                       # digests come from what was loaded
    save_project(reloaded)
    assert spy.take() == [".ocr.json"]

    reloaded.files["b.mkv"].evidence["crop"]["samples"][3]["kept"] = True     # mutated in place
    save_project(reloaded)
    assert spy.take() == sorted([".ocr.json", file_of["b.mkv"]])

    evidence_path(str(tmp_path), "c.mkv").unlink()                # the cache was cleaned behind our back
    save_project(reloaded)
    assert spy.take() == sorted([".ocr.json", file_of["c.mkv"]])


def test_a_file_without_evidence_has_no_evidence_file(tmp_path):
    from core.project.store import evidence_path

    project = _evidence_project(tmp_path, ["a.mkv", "b.mkv"])
    project.files["b.mkv"].evidence = {}
    save_project(project)
    assert evidence_path(str(tmp_path), "a.mkv").exists()
    assert not evidence_path(str(tmp_path), "b.mkv").exists()

    project.files["a.mkv"].evidence.clear()
    save_project(project)
    assert not evidence_path(str(tmp_path), "a.mkv").exists()
    assert load_project(str(tmp_path)).files["a.mkv"].evidence == {}


def test_no_evidence_no_cache_directory(tmp_path):
    save_project(_evidence_project(tmp_path, ["a.mkv"], evidence=None))
    assert not (tmp_path / ".ocr-cache").exists()


def test_a_removed_files_evidence_is_deleted_and_other_cache_files_are_kept(tmp_path):
    from core.project.store import evidence_path

    project = _evidence_project(tmp_path, ["a.mkv", "b.mkv"])
    save_project(project)
    other = _evidence_dir(tmp_path) / "notes.json"
    other.write_text("{}", encoding="utf-8")
    fingerprints = tmp_path / ".ocr-cache" / "fingerprints"
    fingerprints.mkdir()
    (fingerprints / "x.npz").write_bytes(b"x")

    (tmp_path / "b.mkv").unlink()
    reloaded = load_project(str(tmp_path))
    assert list(reloaded.files) == ["a.mkv"]
    save_project(reloaded)

    assert evidence_path(str(tmp_path), "a.mkv").exists()
    assert not evidence_path(str(tmp_path), "b.mkv").exists()
    assert other.exists() and (fingerprints / "x.npz").exists()


@pytest.mark.parametrize("content", [
    b"{not json",
    b"[1, 2]",
    b"\xff\xfe",
    json.dumps({"version": 1, "name": "someone-else.mkv", "evidence": {"crop": {}}}).encode(),
    json.dumps({"version": 1, "name": "a.mkv", "evidence": [1]}).encode(),
    json.dumps({"version": 1, "name": "a.mkv"}).encode(),
])
def test_a_corrupt_evidence_file_gives_empty_evidence_and_a_warning(tmp_path, caplog, content):
    from core.project.store import evidence_path

    project = _evidence_project(tmp_path, ["a.mkv", "b.mkv"])
    save_project(project)
    evidence_path(str(tmp_path), "a.mkv").write_bytes(content)

    with caplog.at_level(logging.WARNING):
        reloaded = load_project(str(tmp_path))

    # Empty but for the staleness record, which .ocr.json keeps (see the follow-up tests).
    record = project.files["a.mkv"].evidence["brightness"]["value_crop_box"]
    assert reloaded.files["a.mkv"].evidence == {"brightness": {"value_crop_box": record}}
    assert reloaded.files["b.mkv"].evidence == project.files["b.mkv"].evidence
    assert reloaded.files["a.mkv"].crop == project.files["a.mkv"].crop          # values are not a cache
    assert any(r.levelno == logging.WARNING and "a.mkv" in r.getMessage() for r in caplog.records)

    save_project(reloaded)                                         # the corrupt cache file is replaced
    assert json.loads(evidence_path(str(tmp_path), "a.mkv").read_text(encoding="utf-8"))["evidence"] == \
        {"brightness": {"value_crop_box": record}}
    reloaded.files["a.mkv"].evidence = {}
    save_project(reloaded)                                         # and goes away with the evidence
    assert not evidence_path(str(tmp_path), "a.mkv").exists()


def test_a_missing_evidence_file_warns_only_when_evidence_was_stored(tmp_path, caplog):
    from core.project.store import evidence_path

    project = _evidence_project(tmp_path, ["a.mkv", "b.mkv"])
    project.files["b.mkv"].flags = {}
    project.files["b.mkv"].evidence = {"audio": _audio_evidence()}
    save_project(project)
    evidence_path(str(tmp_path), "a.mkv").unlink()
    evidence_path(str(tmp_path), "b.mkv").unlink()
    _touch(tmp_path / "fresh.mkv")

    with caplog.at_level(logging.WARNING):
        reloaded = load_project(str(tmp_path))

    record = project.files["a.mkv"].evidence["brightness"]["value_crop_box"]
    assert reloaded.files["a.mkv"].evidence == {"brightness": {"value_crop_box": record}}   # kept by .ocr.json
    assert reloaded.files["b.mkv"].evidence == {} and reloaded.files["fresh.mkv"].evidence == {}
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "a.mkv" in warnings[0] and "b.mkv" not in warnings[0] and "fresh.mkv" not in warnings[0]


def test_a_v2_file_with_inline_evidence_loads_and_the_next_save_moves_it_out(tmp_path, monkeypatch):
    from core.project.store import evidence_path

    project = _evidence_project(tmp_path, NAMES_CJK)
    (tmp_path / ".ocr.json").write_text(json.dumps(to_json(project), ensure_ascii=False, indent=2),
                                        encoding="utf-8")        # as this branch wrote v2 before

    loaded = load_project(str(tmp_path))
    assert loaded == project
    assert {n: e.evidence for n, e in loaded.files.items()} == {n: e.evidence for n, e in project.files.items()}

    spy = _WriteSpy(monkeypatch)
    save_project(loaded)
    assert spy.take() == sorted([".ocr.json"] + [evidence_path(str(tmp_path), n).name for n in NAMES_CJK])
    config = json.loads((tmp_path / ".ocr.json").read_text(encoding="utf-8"))
    assert all("evidence" not in entry for entry in config["files"].values())
    again = load_project(str(tmp_path))
    assert {n: e.evidence for n, e in again.files.items()} == {n: e.evidence for n, e in project.files.items()}


def test_inline_evidence_wins_over_an_evidence_file_left_by_an_interrupted_save(tmp_path):
    from core.project.store import evidence_path

    project = _evidence_project(tmp_path, ["a.mkv"])
    newer = _full_evidence(5)
    save_project(Project(path=str(tmp_path), folder=FolderSettings(),
                         files={"a.mkv": FileEntry(name="a.mkv", evidence=newer)}))
    (tmp_path / ".ocr.json").write_text(json.dumps(to_json(project)), encoding="utf-8")   # the old inline file

    loaded = load_project(str(tmp_path))
    assert loaded.files["a.mkv"].evidence == project.files["a.mkv"].evidence
    save_project(loaded)
    stored = json.loads(evidence_path(str(tmp_path), "a.mkv").read_text(encoding="utf-8"))
    assert stored["evidence"] == project.files["a.mkv"].evidence


def test_evidence_digests_belong_to_each_project_not_the_module(tmp_path):
    from core.project.store import evidence_path

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    save_project(_evidence_project(first, ["a.mkv"]))
    save_project(_evidence_project(second, ["a.mkv"]))              # same name, same evidence
    assert evidence_path(str(second), "a.mkv").exists()
    assert Project(path="x", folder=FolderSettings(), files={}).evidence_digests == {}


def test_a_failed_evidence_write_is_retried_on_the_next_save(tmp_path, monkeypatch):
    import core.project.store as store
    from core.project.store import evidence_path

    project = _evidence_project(tmp_path, ["a.mkv"])
    real = store._atomic_write_text

    def failing(path, text):
        if Path(path).parent == _evidence_dir(tmp_path):
            raise OSError("disk full")
        return real(path, text)

    monkeypatch.setattr(store, "_atomic_write_text", failing)
    save_project(project)                                           # a cache failure never aborts the save
    assert json.loads((tmp_path / ".ocr.json").read_text(encoding="utf-8"))["files"]["a.mkv"]["review"] == "proposed"
    assert not evidence_path(str(tmp_path), "a.mkv").exists()
    monkeypatch.setattr(store, "_atomic_write_text", real)
    save_project(project)
    assert json.loads(evidence_path(str(tmp_path), "a.mkv").read_text(encoding="utf-8"))["evidence"] == \
        project.files["a.mkv"].evidence


def test_to_json_can_leave_evidence_out():
    project = _evidence_project(Path("/nonexistent"), [], evidence=None)
    project.files["a.mkv"] = FileEntry(name="a.mkv", evidence={"crop": {"box": [1, 2, 3, 4]}})
    assert to_json(project)["files"]["a.mkv"]["evidence"] == {"crop": {"box": [1, 2, 3, 4]}}
    assert "evidence" not in to_json(project, include_evidence=False)["files"]["a.mkv"]


def test_the_evidence_cache_shares_the_ranges_cache_directory():
    from core.detect.ranges.pipeline import CACHE_DIRNAME as RANGES_CACHE_DIRNAME
    from core.project.store import CACHE_DIRNAME

    assert CACHE_DIRNAME == RANGES_CACHE_DIRNAME == ".ocr-cache"


def test_saving_forty_files_with_realistic_evidence_is_fast(tmp_path):
    import time as _time

    project = _evidence_project(tmp_path, [f"ep{i:03d}.mp4" for i in range(40)])
    started = _time.perf_counter()
    save_project(project)
    first = _time.perf_counter() - started
    timings = []
    for _ in range(5):
        started = _time.perf_counter()
        save_project(project)
        timings.append(_time.perf_counter() - started)
    unchanged = sorted(timings)[len(timings) // 2]
    config_size = (tmp_path / ".ocr.json").stat().st_size
    evidence_size = sum(p.stat().st_size for p in _evidence_dir(tmp_path).iterdir())
    print(f"40 files: first save {first * 1000:.1f} ms, unchanged save median {unchanged * 1000:.1f} ms, "
          f".ocr.json {config_size / 1e3:.1f} kB, evidence {evidence_size / 1e6:.2f} MB")
    assert config_size < 100_000
    assert unchanged < 1.0                                          # generous: GUI-thread budget, shared machine


# --- Follow-up: brightness staleness never depends on the evidence cache -------


OLD_CROP = [288, 780, 1344, 60]
NEW_CROP = (288, 800, 1344, 60)


def _stale_project(directory: Path) -> Project:
    """The user moved the crop after brightness was measured on the old one: stale, FLAGGED."""
    files = {}
    for name in ("a.mkv", "b.mkv"):
        _touch(directory / name)
        files[name] = FileEntry(
            name=name, crop=Crop(*NEW_CROP, Source.MANUAL), brightness=Brightness(209, Source.DETECTED),
            time_ranges=TimeRanges([], Source.MANUAL), media=Media(1920, 1080, 1500.0, 23.976),
            review=ReviewState.PROPOSED, flags={"crop": "", "brightness": ""},
            evidence={"brightness": {"crop_box": list(OLD_CROP), "value_crop_box": list(OLD_CROP), "strips": []}})
    return Project(str(directory), FolderSettings(), files)


def _state_after_reload(directory: Path):
    from core.jobs.apply import brightness_is_stale, recompute_all

    project = load_project(str(directory))
    recompute_all(project, pending={}, ranges_pending=False)
    entry = project.files["a.mkv"]
    return project, entry, brightness_is_stale(entry)


def test_the_brightness_staleness_record_is_saved_in_ocr_json(tmp_path):
    project = _stale_project(tmp_path)
    project.files["b.mkv"].evidence = {"crop": {"box": [1, 2, 3, 4]}}         # never had a value measured
    save_project(project)
    config = json.loads((tmp_path / ".ocr.json").read_text(encoding="utf-8"))
    assert config["files"]["a.mkv"]["brightness_crop_box"] == OLD_CROP
    assert config["files"]["b.mkv"]["brightness_crop_box"] is None
    assert "evidence" not in config["files"]["a.mkv"]
    assert to_json(project)["files"]["a.mkv"]["brightness_crop_box"] == OLD_CROP     # with evidence as well


def test_a_stale_brightness_stays_stale_when_the_evidence_cache_is_deleted(tmp_path):
    import shutil

    from core.jobs.apply import recompute_all

    project = _stale_project(tmp_path)
    recompute_all(project, pending={}, ranges_pending=False)
    assert project.files["a.mkv"].review == ReviewState.FLAGGED
    save_project(project)
    shutil.rmtree(tmp_path / ".ocr-cache")

    reloaded, entry, stale = _state_after_reload(tmp_path)

    assert stale is True
    assert entry.review == ReviewState.FLAGGED
    assert entry.evidence == {"brightness": {"value_crop_box": OLD_CROP}}


def test_a_stale_brightness_stays_stale_beside_newer_evidence(tmp_path):
    """.ocr.json says the value was measured on the old crop; the evidence cache
    was written by a later save whose .ocr.json never landed."""
    older, newer = tmp_path / "older", tmp_path / "newer"
    older.mkdir()
    newer.mkdir()
    save_project(_stale_project(older))
    remeasured = _stale_project(newer)
    for entry in remeasured.files.values():
        entry.brightness = Brightness(230, Source.DETECTED)
        entry.evidence["brightness"] = {"crop_box": list(NEW_CROP), "value_crop_box": list(NEW_CROP), "strips": []}
    save_project(remeasured)
    for cache in (newer / ".ocr-cache" / "evidence").iterdir():
        (older / ".ocr-cache" / "evidence" / cache.name).write_bytes(cache.read_bytes())

    reloaded, entry, stale = _state_after_reload(older)

    assert entry.brightness == Brightness(209, Source.DETECTED)
    assert entry.evidence["brightness"]["value_crop_box"] == OLD_CROP       # .ocr.json wins over the cache
    assert entry.evidence["brightness"]["crop_box"] == list(NEW_CROP)       # the rest of the evidence is the cache's
    assert stale is True
    assert entry.review == ReviewState.FLAGGED


def test_a_missing_staleness_record_in_ocr_json_overrides_the_cache(tmp_path):
    project = _stale_project(tmp_path)
    for entry in project.files.values():
        entry.brightness = Brightness(209, Source.MANUAL)
        entry.evidence["brightness"].pop("value_crop_box")
    save_project(project)
    other = tmp_path / "other"
    other.mkdir()
    save_project(_stale_project(other))                                        # a cache with a record
    for cache in (other / ".ocr-cache" / "evidence").iterdir():
        (tmp_path / ".ocr-cache" / "evidence" / cache.name).write_bytes(cache.read_bytes())

    reloaded = load_project(str(tmp_path))

    assert reloaded.files["a.mkv"].evidence["brightness"]["crop_box"] == OLD_CROP     # the cache was read
    assert "value_crop_box" not in reloaded.files["a.mkv"].evidence["brightness"]


def test_the_staleness_record_round_trips_and_an_unchanged_reload_rewrites_nothing(tmp_path, monkeypatch):
    project = _stale_project(tmp_path)
    save_project(project)
    reloaded = load_project(str(tmp_path))
    assert reloaded == project
    assert {n: e.evidence for n, e in reloaded.files.items()} == {n: e.evidence for n, e in project.files.items()}
    assert from_json(to_json(project), str(tmp_path)) == project
    assert from_json(to_json(project), str(tmp_path)).files["a.mkv"].evidence == project.files["a.mkv"].evidence

    spy = _WriteSpy(monkeypatch)
    save_project(reloaded)
    assert spy.take() == [".ocr.json"]


def test_an_old_v2_file_without_the_record_falls_back_to_the_evidence(tmp_path):
    from core.jobs.apply import recompute_all

    project = _stale_project(tmp_path)
    save_project(project)
    config = json.loads((tmp_path / ".ocr.json").read_text(encoding="utf-8"))
    for entry in config["files"].values():
        del entry["brightness_crop_box"]
    (tmp_path / ".ocr.json").write_text(json.dumps(config), encoding="utf-8")

    _, entry, stale = _state_after_reload(tmp_path)
    assert entry.evidence["brightness"]["value_crop_box"] == OLD_CROP and stale is True

    inline = tmp_path / "inline"                                               # evidence inline, no record
    inline.mkdir()
    old = _stale_project(inline)
    data = to_json(old)
    for entry in data["files"].values():
        del entry["brightness_crop_box"]
    (inline / ".ocr.json").write_text(json.dumps(data), encoding="utf-8")
    loaded = load_project(str(inline))
    recompute_all(loaded, pending={}, ranges_pending=False)
    assert loaded.files["a.mkv"].evidence == old.files["a.mkv"].evidence
    assert loaded.files["a.mkv"].review == ReviewState.FLAGGED


@pytest.mark.parametrize("record", [[1, 2, 3], "288,780,1344,60", [1, 2, 3, "x"], {"x": 1}, 5, [1, 2, 3, True]])
def test_a_malformed_staleness_record_takes_the_corrupt_path(tmp_path, record):
    raw = _write_config(tmp_path, {"version": 2, "folder": {}, "files": {
        "vid.mkv": {"brightness": {"value": 200, "source": "detected"}, "brightness_crop_box": record}}})
    _touch(tmp_path / "vid.mkv")
    project = load_project(str(tmp_path))
    _assert_corrupt_path_taken(tmp_path, raw, project)


# --- Follow-up: file names that are not valid UTF-8 ------------------------------


def test_a_file_name_that_is_not_utf8_loads_and_saves(tmp_path, monkeypatch):
    import hashlib
    import os

    from core.project.store import evidence_path

    name = os.fsdecode(b"bad\xffname.mkv")                                     # a surrogate-escaped name
    (tmp_path / name).write_bytes(b"")
    _touch(tmp_path / "good.mkv")

    project = load_project(str(tmp_path))
    assert sorted(project.files) == sorted([name, "good.mkv"])
    project.files[name].evidence = {"crop": {"box": [1, 2, 3, 4], "note": "中文"}}
    project.files[name].flags = {"crop": ""}
    save_project(project)

    (tmp_path / ".ocr.json").read_bytes().decode("utf-8")                     # still valid UTF-8
    assert evidence_path(str(tmp_path), name).name == hashlib.sha256(os.fsencode(name)).hexdigest() + ".json"
    evidence_path(str(tmp_path), name).read_bytes().decode("utf-8")
    reloaded = load_project(str(tmp_path))
    assert reloaded == project
    assert reloaded.files[name].evidence == project.files[name].evidence
    spy = _WriteSpy(monkeypatch)
    save_project(reloaded)
    assert spy.take() == [".ocr.json"]


# --- Follow-up: a cache failure never aborts saving the user's values ------------


def test_evidence_that_cannot_be_written_does_not_stop_the_values_being_saved(tmp_path, caplog):
    from core.project.store import evidence_path

    project = _evidence_project(tmp_path, ["a.mkv"])
    (tmp_path / ".ocr-cache").mkdir()
    (tmp_path / ".ocr-cache" / "evidence").write_text("not a directory", encoding="utf-8")
    project.files["a.mkv"].review = ReviewState.REVIEWED
    project.files["a.mkv"].brightness = Brightness(199, Source.MANUAL)

    with caplog.at_level(logging.WARNING):
        save_project(project)

    config = json.loads((tmp_path / ".ocr.json").read_text(encoding="utf-8"))
    assert config["files"]["a.mkv"]["review"] == "reviewed"
    assert config["files"]["a.mkv"]["brightness"] == {"value": 199, "source": "manual"}
    assert any(r.levelno == logging.WARNING and "evidence" in r.getMessage() for r in caplog.records)

    (tmp_path / ".ocr-cache" / "evidence").unlink()
    save_project(project)                                                    # retried
    assert evidence_path(str(tmp_path), "a.mkv").exists()


def test_an_unwritable_evidence_directory_does_not_stop_the_values_being_saved(tmp_path, caplog, monkeypatch):
    import os

    from core.project.store import evidence_path

    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    project = _evidence_project(tmp_path, ["a.mkv", "b.mkv"])
    save_project(project)
    directory = tmp_path / ".ocr-cache" / "evidence"
    project.files["a.mkv"].evidence["crop"]["agreed"] = 99                    # changed: must be written
    del project.files["b.mkv"]                                               # removed: its file must go
    project.files["a.mkv"].review = ReviewState.FLAGGED
    directory.chmod(0o555)
    try:
        with caplog.at_level(logging.WARNING):
            save_project(project)
        config = json.loads((tmp_path / ".ocr.json").read_text(encoding="utf-8"))
        assert list(config["files"]) == ["a.mkv"] and config["files"]["a.mkv"]["review"] == "flagged"
        assert sum(r.levelno == logging.WARNING for r in caplog.records) >= 2      # the write and the delete
    finally:
        directory.chmod(0o755)
    spy = _WriteSpy(monkeypatch)
    save_project(project)
    assert spy.take() == sorted([".ocr.json", evidence_path(str(tmp_path), "a.mkv").name])
    assert not evidence_path(str(tmp_path), "b.mkv").exists()


def test_ocr_json_is_written_before_the_evidence(tmp_path, monkeypatch):
    spy = _WriteSpy(monkeypatch)
    save_project(_evidence_project(tmp_path, ["a.mkv", "b.mkv"]))
    assert [path.name for path in spy.paths][0] == ".ocr.json"
    assert len(spy.paths) == 3


# --------------------------------------------------------------------------
# clamp_crop_box
# --------------------------------------------------------------------------

def test_clamp_crop_box_leaves_a_box_the_frame_holds_alone():
    from core.project.model import clamp_crop_box

    assert clamp_crop_box((288, 786, 1344, 53), (1920, 1080)) == (288, 786, 1344, 53)
    assert clamp_crop_box((0, 0, 1920, 1080), (1920, 1080)) == (0, 0, 1920, 1080)


def test_clamp_crop_box_keeps_the_size_and_moves_the_origin():
    """Unlike videocr's own clamp, which keeps the origin and cuts the size:
    a subtitle band keeps its width and slides inside the frame."""
    from core.project.model import clamp_crop_box

    assert clamp_crop_box((288, 784, 1344, 55), (1280, 720)) == (0, 665, 1280, 55)


def test_clamp_crop_box_without_a_frame_size_changes_nothing():
    """The metadata job has not run: there is nothing to clamp against, and
    apply_metadata re-checks the value when the size arrives."""
    from core.project.model import clamp_crop_box, frame_size_known

    assert clamp_crop_box((288, 784, 1344, 55), (0, 0)) == (288, 784, 1344, 55)
    assert not frame_size_known(Media())
    assert frame_size_known(Media(1920, 1080, 10.0, 25.0))
