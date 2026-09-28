"""core.project.episode_cache: the cache root's bookkeeping (index.json).

Pinned here: the folder index an episode save updates, the same-folder crop
seed read from it, the remembered run speed, and pruning (age first, then
size, oldest first, never the key being opened). A corrupt index reads as
empty and never raises.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core.project import episode_cache
from core.project.episode_cache import (
    index_path,
    last_speed,
    prune,
    record_episode,
    record_speed,
    sibling_seed,
)
from core.project.layout import episode_layout
from core.project.model import Crop, Media, Source
from core.project.store import load_project, save_project

DAY = 24 * 3600.0
NOW = 1_800_000_000.0


def _video(path: Path, data: bytes | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(os.urandom(2048) if data is None else data)
    return path


def _index(root: Path) -> dict:
    return json.loads(index_path(str(root)).read_text(encoding="utf-8"))


def _save_episode(root: Path, video: Path, crop: Crop | None = None, height: int = 1080,
                  flags: dict | None = None) -> str:
    """Save `video` as an episode with `crop` and return its key."""
    layout = episode_layout(str(video), str(root))
    project = load_project(layout.video_dir, layout)
    entry = project.files[video.name]
    entry.crop = crop
    entry.media = Media(width=1920, height=height, duration=1400.0, fps=23.976)
    entry.flags = dict(flags or {})
    save_project(project)
    return layout.key


# --------------------------------------------------------------------------
# The index
# --------------------------------------------------------------------------

def test_the_index_is_index_json_under_the_root(tmp_path):
    assert index_path(str(tmp_path)) == tmp_path / "index.json"


def test_recording_an_episode_files_it_under_its_folder(tmp_path):
    root = tmp_path / "root"
    video = _video(tmp_path / "Show" / "EP05.mkv")
    record_episode(str(root), str(video), "a" * 32, now=NOW)
    assert _index(root)["folders"] == {
        str(tmp_path / "Show"): [{"key": "a" * 32, "name": "EP05.mkv", "saved_at": NOW}],
    }


def test_recording_again_updates_the_record_in_place(tmp_path):
    root = tmp_path / "root"
    show = tmp_path / "Show"
    record_episode(str(root), str(show / "EP05.mkv"), "a" * 32, now=NOW)
    record_episode(str(root), str(show / "EP06.mkv"), "b" * 32, now=NOW + 1)
    record_episode(str(root), str(show / "EP05 renamed.mkv"), "a" * 32, now=NOW + 2)
    records = _index(root)["folders"][str(show)]
    assert sorted((r["key"], r["name"], r["saved_at"]) for r in records) == [
        ("a" * 32, "EP05 renamed.mkv", NOW + 2),
        ("b" * 32, "EP06.mkv", NOW + 1),
    ]


def test_an_episode_that_moved_leaves_its_old_folder(tmp_path):
    root = tmp_path / "root"
    record_episode(str(root), str(tmp_path / "Old" / "EP05.mkv"), "a" * 32, now=NOW)
    record_episode(str(root), str(tmp_path / "New" / "EP05.mkv"), "a" * 32, now=NOW + 1)
    assert list(_index(root)["folders"]) == [str(tmp_path / "New")]


def test_the_index_is_written_atomically(tmp_path, monkeypatch):
    root = tmp_path / "root"
    record_episode(str(root), str(tmp_path / "EP01.mkv"), "a" * 32, now=NOW)
    before = index_path(str(root)).read_bytes()

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(episode_cache.os, "replace", fail)
    record_episode(str(root), str(tmp_path / "EP02.mkv"), "b" * 32, now=NOW)   # logged, not raised
    assert index_path(str(root)).read_bytes() == before
    assert [p.name for p in root.iterdir()] == ["index.json"]


@pytest.mark.parametrize("content", [b"{not json", b"[]", b'{"folders": [], "speed": "fast"}',
                                     b'{"folders": {"/x": "nope"}}', b"\xff\xfe"])
def test_a_corrupt_index_reads_as_empty_and_never_raises(tmp_path, content):
    root = tmp_path / "root"
    root.mkdir()
    index_path(str(root)).write_bytes(content)
    video = _video(tmp_path / "Show" / "EP06.mkv")
    assert last_speed(str(root)) is None
    assert sibling_seed(str(root), str(video), "b" * 32) is None
    record_episode(str(root), str(video), "b" * 32, now=NOW)
    assert _index(root)["folders"][str(tmp_path / "Show")][0]["key"] == "b" * 32


def test_nothing_recorded_means_nothing_known(tmp_path):
    root = tmp_path / "missing-root"
    assert last_speed(str(root)) is None
    assert sibling_seed(str(root), str(tmp_path / "EP01.mkv"), "a" * 32) is None
    assert prune(str(root), "a" * 32, now=NOW) == []
    assert not root.exists()


# --------------------------------------------------------------------------
# The same-folder seed
# --------------------------------------------------------------------------

def test_the_seed_is_the_newest_siblings_crop_as_fractions(tmp_path):
    root, show = tmp_path / "root", tmp_path / "Show"
    ep04 = _video(show / "EP04.mkv")
    ep05 = _video(show / "EP05.mkv")
    ep06 = _video(show / "EP06.mkv")
    k4 = _save_episode(root, ep04, Crop(0, 900, 1920, 60, Source.MANUAL))
    k5 = _save_episode(root, ep05, Crop(0, 972, 1920, 54, Source.DETECTED), height=1080)
    record_episode(str(root), str(ep04), k4, now=NOW)
    record_episode(str(root), str(ep05), k5, now=NOW + 10)
    k6 = episode_layout(str(ep06), str(root)).key
    assert sibling_seed(str(root), str(ep06), k6) == ("EP05.mkv", (972 / 1080, 54 / 1080))


@pytest.mark.parametrize("source", [Source.MANUAL, Source.IMPORTED, Source.DETECTED])
def test_a_trusted_crop_seeds(tmp_path, source):
    root, show = tmp_path / "root", tmp_path / "Show"
    ep05, ep06 = _video(show / "EP05.mkv"), _video(show / "EP06.mkv")
    record_episode(str(root), str(ep05), _save_episode(root, ep05, Crop(0, 900, 1920, 60, source), height=1000),
                   now=NOW)
    assert sibling_seed(str(root), str(ep06), "f" * 32) == ("EP05.mkv", (0.9, 0.06))


@pytest.mark.parametrize("crop, height, flags", [
    (None, 1080, {}),                                              # no crop
    (Crop(0, 900, 1920, 60, Source.DETECTED), 1080, {"crop": "check-crop?"}),   # a doubted detection
    (Crop(0, 900, 1920, 60, Source.HINT), 1080, {}),               # a hint is not the pool's
    (Crop(0, 900, 1920, 60, Source.MANUAL), 0, {}),                # no media height
])
def test_an_unusable_newer_sibling_is_passed_over_for_an_older_usable_one(tmp_path, crop, height, flags):
    root, show = tmp_path / "root", tmp_path / "Show"
    ep04, ep05, ep06 = (_video(show / f"EP0{n}.mkv") for n in (4, 5, 6))
    record_episode(str(root), str(ep04), _save_episode(root, ep04, Crop(0, 972, 1920, 54, Source.MANUAL)), now=NOW)
    record_episode(str(root), str(ep05), _save_episode(root, ep05, crop, height=height, flags=flags), now=NOW + 10)
    assert sibling_seed(str(root), str(ep06), "f" * 32) == ("EP04.mkv", (972 / 1080, 54 / 1080))


def test_the_seed_never_comes_from_the_episode_itself_or_another_folder(tmp_path):
    root = tmp_path / "root"
    other = _video(tmp_path / "Other" / "EP05.mkv")
    ep06 = _video(tmp_path / "Show" / "EP06.mkv")
    record_episode(str(root), str(other), _save_episode(root, other, Crop(0, 900, 1920, 60, Source.MANUAL)), now=NOW)
    k6 = _save_episode(root, ep06, Crop(0, 900, 1920, 60, Source.MANUAL))
    record_episode(str(root), str(ep06), k6, now=NOW)
    assert sibling_seed(str(root), str(ep06), k6) is None


def test_a_sibling_whose_settings_are_gone_or_corrupt_does_not_seed(tmp_path):
    root, show = tmp_path / "root", tmp_path / "Show"
    ep04, ep05, ep06 = (_video(show / f"EP0{n}.mkv") for n in (4, 5, 6))
    k4 = _save_episode(root, ep04, Crop(0, 900, 1920, 60, Source.MANUAL))
    k5 = _save_episode(root, ep05, Crop(0, 900, 1920, 60, Source.MANUAL))
    record_episode(str(root), str(ep04), k4, now=NOW)
    record_episode(str(root), str(ep05), k5, now=NOW + 1)
    (root / "videos" / k5 / "settings.json").write_text("{broken", encoding="utf-8")
    (root / "videos" / k4 / "settings.json").unlink()
    assert sibling_seed(str(root), str(ep06), "f" * 32) is None


# --------------------------------------------------------------------------
# Speed
# --------------------------------------------------------------------------

def test_the_last_speed_is_video_seconds_per_wall_second(tmp_path):
    root = tmp_path / "root"
    record_speed(str(root), 1400.0, 350.0, now=NOW)
    assert last_speed(str(root)) == pytest.approx(4.0)
    assert _index(root)["speed"] == {"video_per_wall": 4.0, "measured_at": NOW}
    record_speed(str(root), 1200.0, 600.0, now=NOW + 1)
    assert last_speed(str(root)) == pytest.approx(2.0)


@pytest.mark.parametrize("video, wall", [(1400.0, 4.9), (1400.0, 0.0), (0.0, 100.0), (-1.0, 100.0),
                                          (float("nan"), 100.0), (1400.0, float("inf"))])
def test_a_run_too_short_or_meaningless_to_measure_is_not_remembered(tmp_path, video, wall):
    root = tmp_path / "root"
    record_speed(str(root), 1400.0, 350.0, now=NOW)
    record_speed(str(root), video, wall, now=NOW + 1)
    assert last_speed(str(root)) == pytest.approx(4.0)


def test_speed_and_folders_live_side_by_side(tmp_path):
    root = tmp_path / "root"
    record_speed(str(root), 1400.0, 350.0, now=NOW)
    record_episode(str(root), str(tmp_path / "EP01.mkv"), "a" * 32, now=NOW)
    assert last_speed(str(root)) == pytest.approx(4.0)
    record_speed(str(root), 1400.0, 700.0, now=NOW)
    assert list(_index(root)["folders"]) == [str(tmp_path)]


# --------------------------------------------------------------------------
# Pruning
# --------------------------------------------------------------------------

def _entry(root: Path, key: str, size: int, age_days: float) -> Path:
    directory = root / "videos" / key
    (directory / "view" / "x").mkdir(parents=True, exist_ok=True)
    blob = directory / "view" / "x" / "blob.webp"
    blob.write_bytes(b"\0" * size)
    settings = directory / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    stamp = NOW - age_days * DAY
    for path in (blob, settings, blob.parent, blob.parent.parent, directory):
        os.utime(path, (stamp, stamp))
    return directory


def _keys(root: Path) -> list[str]:
    return sorted(p.name for p in (root / "videos").iterdir())


def test_entries_untouched_for_more_than_90_days_are_removed(tmp_path):
    root = tmp_path / "root"
    a, b, c = "a" * 32, "b" * 32, "c" * 32
    _entry(root, a, 10, 91)
    _entry(root, b, 10, 89)
    _entry(root, c, 10, 400)
    assert sorted(prune(str(root), "f" * 32, now=NOW)) == [a, c]
    assert _keys(root) == [b]


def test_a_recently_touched_file_keeps_an_old_entry(tmp_path):
    root = tmp_path / "root"
    directory = _entry(root, "a" * 32, 10, 200)
    os.utime(directory / "view" / "x" / "blob.webp", (NOW - DAY, NOW - DAY))
    assert prune(str(root), "f" * 32, now=NOW) == []


def test_the_key_being_opened_is_never_removed(tmp_path):
    root = tmp_path / "root"
    _entry(root, "a" * 32, 5000, 400)
    assert prune(str(root), "a" * 32, now=NOW, max_bytes=100) == []
    assert _keys(root) == ["a" * 32]


def test_over_the_size_limit_the_oldest_go_first_until_it_fits(tmp_path):
    root = tmp_path / "root"
    a, b, c, d = "a" * 32, "b" * 32, "c" * 32, "d" * 32
    _entry(root, a, 1000, 30)
    _entry(root, b, 1000, 10)
    _entry(root, c, 1000, 20)
    _entry(root, d, 1000, 40)                        # the one being opened, oldest of all
    # Four entries of ~1002 bytes; 2100 fits two of them.
    assert prune(str(root), d, now=NOW, max_bytes=2100) == [a, c]
    assert _keys(root) == [b, d]


def test_pruning_drops_the_removed_keys_from_the_index(tmp_path):
    root = tmp_path / "root"
    a, b = "a" * 32, "b" * 32
    _entry(root, a, 10, 200)
    _entry(root, b, 10, 1)
    record_episode(str(root), str(tmp_path / "Show" / "EP01.mkv"), a, now=NOW)
    record_episode(str(root), str(tmp_path / "Show" / "EP02.mkv"), b, now=NOW)
    record_speed(str(root), 1400.0, 350.0, now=NOW)
    assert prune(str(root), "f" * 32, now=NOW) == [a]
    assert [r["key"] for r in _index(root)["folders"][str(tmp_path / "Show")]] == [b]
    assert last_speed(str(root)) == pytest.approx(4.0)


def test_pruning_leaves_what_is_not_an_entry_alone(tmp_path):
    root = tmp_path / "root"
    (root / "videos" / "not-a-key").mkdir(parents=True)
    (root / "videos" / ("e" * 32 + ".json")).write_text("{}")
    stamp = NOW - 1000 * DAY
    os.utime(root / "videos" / "not-a-key", (stamp, stamp))
    assert prune(str(root), "f" * 32, now=NOW, max_bytes=0) == []
    assert _keys(root) == ["e" * 32 + ".json", "not-a-key"]


def test_an_entry_that_will_not_go_is_skipped(tmp_path, monkeypatch):
    root = tmp_path / "root"
    a, b = "a" * 32, "b" * 32
    _entry(root, a, 10, 200)
    _entry(root, b, 10, 300)

    real_rmtree = episode_cache.shutil.rmtree

    def rmtree(path, *args, **kwargs):
        if Path(path).name == b:
            raise OSError("busy")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(episode_cache.shutil, "rmtree", rmtree)
    assert prune(str(root), "f" * 32, now=NOW) == [a]
    assert _keys(root) == [b]
