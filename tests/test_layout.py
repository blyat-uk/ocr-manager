"""core.project.layout: every path a project reads or writes.

Pinned here:

- The folder layout is today's paths, as literals: `.ocr.json`,
  `.ocr-cache/`, the v1 backup -- and the evidence and view caches under its
  cache_dir land where they always did.
- The episode layout keeps everything but the output out of the video's
  directory: settings and caches under `<root>/videos/<key>/`.
- The output, folder and episode alike: `<tag>/<stem>.<tag>.ass` with
  output_subfolder on (the default), `<stem>.<tag>.ass` next to the video
  with it off; the tag is the OCR language's (core.project.languages). A
  run creates only `<tag>/`, or nothing. layout_of follows the project's
  current settings.
- video_key is content-based (a rename keeps it, a changed head or tail
  byte does not), also for files shorter than two chunks.
- episode_target routes a path the way the spec's table does.
- The store loads and saves an episode without writing into the video's
  directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from core.jobs.run import output_name
from core.jobs.view_cache import view_dir
from core.project import layout as layout_module
from core.project.layout import (
    KEY_CHUNK,
    ProjectLayout,
    _default_cache_root,
    cache_root,
    episode_layout,
    episode_target,
    folder_layout,
    layout_of,
    video_key,
)
from core.project.model import Crop, FolderSettings, Project, Source
from core.project.store import evidence_path, load_project, save_project


def _write(path: Path, data: bytes = b"video") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _listing(directory: Path) -> list[str]:
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*"))


# --------------------------------------------------------------------------
# The folder layout is today's paths
# --------------------------------------------------------------------------

def test_the_folder_layout_is_todays_paths_literally():
    d = os.path.join(os.sep, "work", "Show S01")
    layout = folder_layout(d)
    assert layout == ProjectLayout(
        kind="folder",
        video_dir=d,
        config_path=d + os.sep + ".ocr.json",
        cache_dir=d + os.sep + ".ocr-cache",
    )
    assert not layout.is_episode
    assert layout.only_file is None and layout.key is None
    assert layout.output_tag == "zh" and layout.output_subfolder
    assert layout.output_path("EP01.mkv") == d + os.sep + "zh" + os.sep + "EP01.zh.ass"
    assert layout.output_path("a.b.mkv") == d + os.sep + "zh" + os.sep + "a.b.zh.ass"
    assert layout.output_path("第一集.mp4") == d + os.sep + "zh" + os.sep + "第一集.zh.ass"
    assert layout.output_label("EP01.mkv") == "zh/EP01.zh.ass"
    assert layout.output_dirs() == (d + os.sep + "zh",)
    assert layout.v1_backup_path == d + os.sep + ".ocr.json.v1.bak"


def test_the_folder_layout_joins_onto_the_directory_exactly_as_given(tmp_path):
    for d in (str(tmp_path), str(tmp_path) + os.sep, "relative/dir"):
        layout = folder_layout(d)
        assert layout.video_dir == d
        assert layout.config_path == os.path.join(d, ".ocr.json")
        assert layout.cache_dir == os.path.join(d, ".ocr-cache")
        assert layout.output_path("x.mkv") == os.path.join(d, "zh", output_name("x.mkv"))


def test_the_folder_layouts_caches_are_where_they_always_were(tmp_path):
    layout = folder_layout(str(tmp_path))
    digest = hashlib.sha256(os.fsencode("EP01.mkv")).hexdigest()
    assert evidence_path(layout.cache_dir, "EP01.mkv") == tmp_path / ".ocr-cache" / "evidence" / f"{digest}.json"
    assert view_dir(layout.cache_dir, "EP01.mkv") == tmp_path / ".ocr-cache" / "view" / digest


def test_the_folder_layouts_videos_are_the_store_rule(tmp_path):
    for name in ("b.mkv", "a.MP4", "notes.txt", "c.avi", "d.Mkv"):
        _write(tmp_path / name)
    assert folder_layout(str(tmp_path)).video_names() == ["a.MP4", "b.mkv", "d.Mkv"]


def test_layout_of_a_project_built_without_one_is_its_folder_layout(tmp_path):
    project = Project(path=str(tmp_path), folder=FolderSettings(), files={})
    assert project.layout is None
    assert layout_of(project) == folder_layout(str(tmp_path))
    episode = episode_layout(str(_write(tmp_path / "EP01.mkv")), str(tmp_path / "root"))
    project.layout = episode
    assert layout_of(project) == episode                       # default settings: the default output naming


def test_the_layout_is_not_part_of_a_projects_equality(tmp_path):
    a = Project(path=str(tmp_path), folder=FolderSettings(), files={})
    b = Project(path=str(tmp_path), folder=FolderSettings(), files={}, layout=folder_layout(str(tmp_path)))
    assert a == b


# --------------------------------------------------------------------------
# The episode layout
# --------------------------------------------------------------------------

def test_the_episode_layout_keeps_state_under_the_root_and_output_next_to_the_video(tmp_path):
    video = _write(tmp_path / "Show" / "EP06.mkv")
    root = tmp_path / "cache"
    layout = episode_layout(str(video), str(root))
    key = video_key(str(video))
    assert layout.is_episode and layout.kind == "episode"
    assert layout.key == key
    assert layout.only_file == "EP06.mkv"
    assert layout.video_dir == str(tmp_path / "Show")
    assert layout.cache_dir == str(root / "videos" / key)
    assert layout.config_path == str(root / "videos" / key / "settings.json")
    assert layout.output_path("EP06.mkv") == str(tmp_path / "Show" / "zh" / "EP06.zh.ass")
    assert layout.output_path("a.b.mp4") == str(tmp_path / "Show" / "zh" / "a.b.zh.ass")
    assert layout.output_label("EP06.mkv") == "zh/EP06.zh.ass"
    assert layout.output_dirs() == (str(tmp_path / "Show" / "zh"),)
    assert layout.v1_backup_path is None


def test_the_episode_layout_makes_a_relative_video_path_absolute(tmp_path, monkeypatch):
    _write(tmp_path / "EP01.mkv")
    monkeypatch.chdir(tmp_path)
    layout = episode_layout("EP01.mkv", str(tmp_path / "root"))
    assert layout.video_dir == str(tmp_path)
    assert layout.only_file == "EP01.mkv"


def test_the_episode_holds_only_its_own_video_while_it_is_there(tmp_path):
    video = _write(tmp_path / "EP01.mkv")
    _write(tmp_path / "EP02.mkv")
    layout = episode_layout(str(video), str(tmp_path / "root"))
    assert layout.video_names() == ["EP01.mkv"]
    video.unlink()
    assert layout.video_names() == []


def test_an_episode_layout_takes_a_known_key_without_hashing(tmp_path):
    layout = episode_layout(str(tmp_path / "gone.mkv"), str(tmp_path), key="0" * 32)
    assert layout.key == "0" * 32
    assert layout.cache_dir == str(tmp_path / "videos" / ("0" * 32))


def test_an_episode_layout_uses_the_cache_root_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_MANAGER_CACHE_DIR", str(tmp_path / "env-root"))
    video = _write(tmp_path / "EP01.mkv")
    assert episode_layout(str(video)).cache_dir.startswith(str(tmp_path / "env-root" / "videos"))


# --------------------------------------------------------------------------
# The output: language tag x subfolder, folder and episode alike
# --------------------------------------------------------------------------

def _layouts(tmp_path) -> dict[str, ProjectLayout]:
    show = tmp_path / "Show"
    return {"folder": folder_layout(str(show)),
            "episode": episode_layout(str(_write(show / "EP01.mkv")), str(tmp_path / "root"))}


@pytest.mark.parametrize("kind", ["folder", "episode"])
@pytest.mark.parametrize("ocr_lang, tag", [("ch", "zh"), ("japan", "ja"), ("rs_latin", "sr-Latn")])
def test_the_output_is_in_the_language_subfolder_by_default(tmp_path, kind, ocr_lang, tag):
    base = _layouts(tmp_path)[kind]
    project = Project(path=str(tmp_path / "Show"), folder=FolderSettings(ocr_lang=ocr_lang), files={}, layout=base)
    layout = layout_of(project)
    show = tmp_path / "Show"
    assert (layout.output_tag, layout.output_subfolder) == (tag, True)
    assert layout.output_path("EP01.mkv") == str(show / tag / f"EP01.{tag}.ass")
    assert layout.output_path("a.b.mp4") == str(show / tag / f"a.b.{tag}.ass")
    assert layout.output_label("EP01.mkv") == f"{tag}/EP01.{tag}.ass"
    assert layout.output_dirs() == (str(show / tag),)
    # Only the output naming differs from the layout the project was loaded with.
    assert (layout.kind, layout.config_path, layout.cache_dir) == (base.kind, base.config_path, base.cache_dir)


@pytest.mark.parametrize("kind", ["folder", "episode"])
@pytest.mark.parametrize("ocr_lang, tag", [("ch", "zh"), ("japan", "ja"), ("rs_latin", "sr-Latn")])
def test_without_the_subfolder_the_output_is_next_to_the_video(tmp_path, kind, ocr_lang, tag):
    folder = FolderSettings(ocr_lang=ocr_lang, output_subfolder=False)
    project = Project(path=str(tmp_path / "Show"), folder=folder, files={}, layout=_layouts(tmp_path)[kind])
    layout = layout_of(project)
    show = tmp_path / "Show"
    assert layout.output_path("EP01.mkv") == str(show / f"EP01.{tag}.ass")
    assert layout.output_path("a.b.mp4") == str(show / f"a.b.{tag}.ass")
    assert layout.output_label("EP01.mkv") == f"EP01.{tag}.ass"
    assert layout.output_dirs() == ()


def test_output_name_is_the_stem_and_the_tag():
    assert output_name("EP01.mkv") == "EP01.zh.ass"
    assert output_name("a.b.MP4", "sr-Latn") == "a.b.sr-Latn.ass"


def test_layout_of_follows_a_settings_change_on_the_same_project(tmp_path):
    project = Project(path=str(tmp_path), folder=FolderSettings(), files={})
    assert layout_of(project).output_path("EP01.mkv") == str(tmp_path / "zh" / "EP01.zh.ass")
    project.folder.ocr_lang = "japan"
    assert layout_of(project).output_path("EP01.mkv") == str(tmp_path / "ja" / "EP01.ja.ass")
    project.folder.output_subfolder = False
    snapshot = layout_of(project)
    assert snapshot.output_path("EP01.mkv") == str(tmp_path / "EP01.ja.ass")
    project.folder.ocr_lang = "korean"                           # a layout already taken does not move
    assert snapshot.output_path("EP01.mkv") == str(tmp_path / "EP01.ja.ass")
    assert layout_of(project).output_path("EP01.mkv") == str(tmp_path / "EP01.ko.ass")


# --------------------------------------------------------------------------
# The cache root
# --------------------------------------------------------------------------

def test_the_cache_root_is_the_environment_override_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("OCR_MANAGER_CACHE_DIR", str(tmp_path))
    assert cache_root() == str(tmp_path)


def test_the_cache_root_falls_back_to_the_platform_default(monkeypatch):
    monkeypatch.delenv("OCR_MANAGER_CACHE_DIR", raising=False)
    assert cache_root() == _default_cache_root(layout_module.sys.platform, os.environ, os.path.expanduser("~"))


@pytest.mark.parametrize("platform, environ, expected", [
    ("linux", {"XDG_CACHE_HOME": "/xdg"}, os.path.join("/xdg", "ocr-manager")),
    ("linux", {}, os.path.join("/home/u", ".cache", "ocr-manager")),
    ("linux", {"XDG_CACHE_HOME": ""}, os.path.join("/home/u", ".cache", "ocr-manager")),
    ("linux", {"XDG_CACHE_HOME": "relative"}, os.path.join("/home/u", ".cache", "ocr-manager")),
    ("freebsd13", {}, os.path.join("/home/u", ".cache", "ocr-manager")),
    ("win32", {"LOCALAPPDATA": "C:\\Users\\u\\AppData\\Local"},
     os.path.join("C:\\Users\\u\\AppData\\Local", "ocr-manager", "cache")),
    ("win32", {}, os.path.join("/home/u", "AppData", "Local", "ocr-manager", "cache")),
    ("darwin", {"XDG_CACHE_HOME": "/xdg"}, os.path.join("/home/u", "Library", "Caches", "ocr-manager")),
])
def test_the_platform_default_cache_root(platform, environ, expected):
    assert _default_cache_root(platform, environ, "/home/u") == expected


# --------------------------------------------------------------------------
# video_key
# --------------------------------------------------------------------------

def _expected_key(data: bytes) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(len(data).to_bytes(8, "little"))
    digest.update(data[:KEY_CHUNK])
    digest.update(data[max(0, len(data) - KEY_CHUNK):])
    return digest.hexdigest()


@pytest.mark.parametrize("size", [0, 1, 1000, KEY_CHUNK, KEY_CHUNK + 1, 2 * KEY_CHUNK - 1, 2 * KEY_CHUNK + 4096])
def test_the_key_is_blake2b_over_the_size_the_head_and_the_tail(tmp_path, size):
    data = bytes((i * 7 + 3) % 251 for i in range(size))
    key = video_key(str(_write(tmp_path / "v.mkv", data)))
    assert key == _expected_key(data)
    assert len(key) == 32 and all(c in "0123456789abcdef" for c in key)


@pytest.mark.parametrize("size", [10, KEY_CHUNK + 10, 3 * KEY_CHUNK])
def test_the_same_content_under_another_name_or_folder_has_the_same_key(tmp_path, size):
    data = os.urandom(size)
    a = _write(tmp_path / "a" / "EP01.mkv", data)
    b = _write(tmp_path / "elsewhere" / "renamed.mp4", data)
    assert video_key(str(a)) == video_key(str(b))


@pytest.mark.parametrize("size, offset", [
    (10, 5),                                   # a small file: any byte
    (KEY_CHUNK + 10, KEY_CHUNK // 2),          # under two chunks: the head and tail overlap
    (3 * KEY_CHUNK, 0),                        # the first byte
    (3 * KEY_CHUNK, KEY_CHUNK - 1),            # the head's last byte
    (3 * KEY_CHUNK, 2 * KEY_CHUNK),            # the tail's first byte
    (3 * KEY_CHUNK, 3 * KEY_CHUNK - 1),        # the last byte
])
def test_a_changed_head_or_tail_byte_changes_the_key(tmp_path, size, offset):
    data = bytearray(os.urandom(size))
    before = video_key(str(_write(tmp_path / "a.mkv", bytes(data))))
    data[offset] ^= 0xFF
    assert video_key(str(_write(tmp_path / "a.mkv", bytes(data)))) != before


def test_a_different_size_changes_the_key(tmp_path):
    data = os.urandom(3 * KEY_CHUNK)
    before = video_key(str(_write(tmp_path / "a.mkv", data)))
    # The same head and tail with one byte more in the middle.
    longer = data[:KEY_CHUNK + 5] + b"x" + data[KEY_CHUNK + 5:]
    assert video_key(str(_write(tmp_path / "a.mkv", longer))) != before


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def test_a_video_file_opens_as_an_episode(tmp_path):
    video = _write(tmp_path / "Show" / "EP01.mkv")
    _write(tmp_path / "Show" / "EP02.mkv")
    assert episode_target(str(video)) == str(video)
    upper = _write(tmp_path / "Show" / "EP03.MP4")
    assert episode_target(str(upper)) == str(upper)


def test_a_relative_video_path_routes_to_its_absolute_path(tmp_path, monkeypatch):
    _write(tmp_path / "EP01.mkv")
    monkeypatch.chdir(tmp_path)
    assert episode_target("EP01.mkv") == str(tmp_path / "EP01.mkv")


def test_a_folder_with_exactly_one_video_opens_that_video(tmp_path):
    _write(tmp_path / "EP01.mkv")
    _write(tmp_path / "EP01.zh.ass", b"out")
    _write(tmp_path / "notes.txt")
    assert episode_target(str(tmp_path)) == str(tmp_path / "EP01.mkv")


@pytest.mark.parametrize("videos", [[], ["EP01.mkv", "EP02.mp4"], ["a.mkv", "b.mkv", "c.mkv"]])
def test_a_folder_of_zero_or_several_videos_opens_as_a_folder(tmp_path, videos):
    for name in videos:
        _write(tmp_path / name)
    _write(tmp_path / "notes.txt")
    assert episode_target(str(tmp_path)) is None


def test_anything_else_is_not_an_episode(tmp_path):
    assert episode_target(str(_write(tmp_path / "notes.txt"))) is None
    assert episode_target(str(tmp_path / "missing.mkv")) is None
    assert episode_target(str(tmp_path / "missing")) is None
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "fake.mkv").mkdir()           # a directory with a video's name
    assert episode_target(str(tmp_path / "dir")) is None
    assert episode_target(str(tmp_path / "dir" / "fake.mkv")) is None


# --------------------------------------------------------------------------
# The store with an episode layout
# --------------------------------------------------------------------------

def _episode(tmp_path) -> tuple[Path, Path, ProjectLayout]:
    show = tmp_path / "Show"
    video = _write(show / "EP06.mkv", os.urandom(4096))
    _write(show / "EP05.mkv", os.urandom(4096))
    _write(show / "EP05.zh.ass", b"old output")
    return show, video, episode_layout(str(video), str(tmp_path / "root"))


def test_an_episode_round_trips_without_writing_into_the_video_directory(tmp_path):
    show, video, layout = _episode(tmp_path)
    before = _listing(show)

    project = load_project(str(show), layout)
    assert project.layout is layout
    assert project.path == str(show)
    assert list(project.files) == ["EP06.mkv"]                   # not its sibling
    entry = project.files["EP06.mkv"]
    entry.crop = Crop(288, 786, 1344, 53, Source.MANUAL)
    entry.evidence["crop"] = {"hit_pts": [1.0, 2.0]}
    save_project(project)

    assert _listing(show) == before
    config = Path(layout.config_path)
    assert config.is_file()
    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["version"] == 2 and list(data["files"]) == ["EP06.mkv"]
    assert "evidence" not in data["files"]["EP06.mkv"]
    assert evidence_path(layout.cache_dir, "EP06.mkv").is_file()
    assert not (tmp_path / "Show" / ".ocr-cache").exists()
    assert not (tmp_path / "root" / "videos" / layout.key / ".ocr-cache").exists()

    again = load_project(str(show), layout)
    assert again.files["EP06.mkv"].crop == Crop(288, 786, 1344, 53, Source.MANUAL)
    assert again.files["EP06.mkv"].evidence == {"crop": {"hit_pts": [1.0, 2.0]}}
    assert _listing(show) == before


def test_an_episode_without_settings_is_a_fresh_project_and_loading_writes_nothing(tmp_path):
    show, _, layout = _episode(tmp_path)
    project = load_project(str(show), layout)
    assert project.folder == FolderSettings()
    assert list(project.files) == ["EP06.mkv"]
    assert not Path(layout.cache_dir).exists()


def test_a_corrupt_episode_config_is_moved_aside_next_to_itself(tmp_path):
    show, _, layout = _episode(tmp_path)
    before = _listing(show)
    _write(Path(layout.config_path), b"{not json")
    project = load_project(str(show), layout)
    assert list(project.files) == ["EP06.mkv"]
    assert not Path(layout.config_path).exists()
    aside = [p.name for p in Path(layout.cache_dir).iterdir()]
    assert len(aside) == 1 and aside[0].startswith("settings.json.corrupt-")
    assert _listing(show) == before


def test_an_episode_config_is_never_migrated_from_v1(tmp_path):
    show, _, layout = _episode(tmp_path)
    v1 = {"file_settings": {"EP06.mkv": {"crop_box": [0, 900, 1920, 60]}}}
    _write(Path(layout.config_path), json.dumps(v1).encode())
    project = load_project(str(show), layout)
    assert not project.migrated_from_v1
    assert project.files["EP06.mkv"].crop is None
    save_project(project)
    assert not any(p.name.endswith(".v1.bak") for p in tmp_path.rglob("*"))


def test_an_episode_of_an_unsupported_version_is_refused_and_left_alone(tmp_path):
    from core.project.store import UnsupportedProjectVersion

    show, _, layout = _episode(tmp_path)
    _write(Path(layout.config_path), b'{"version": 3}')
    with pytest.raises(UnsupportedProjectVersion):
        load_project(str(show), layout)
    assert Path(layout.config_path).read_bytes() == b'{"version": 3}'


def test_an_episode_whose_video_is_gone_loads_with_no_files(tmp_path):
    show, video, layout = _episode(tmp_path)
    project = load_project(str(show), layout)
    save_project(project)
    video.unlink()
    assert load_project(str(show), layout).files == {}


def test_a_folder_load_carries_its_folder_layout(tmp_path):
    _write(tmp_path / "a.mkv")
    project = load_project(str(tmp_path))
    assert project.layout == folder_layout(str(tmp_path))
    save_project(project)
    assert (tmp_path / ".ocr.json").is_file()


def test_saving_a_folder_whose_directory_is_gone_does_not_recreate_it(tmp_path):
    folder = tmp_path / "gone"
    _write(folder / "a.mkv")
    project = load_project(str(folder))
    shutil.rmtree(folder)
    with pytest.raises(OSError):
        save_project(project)
    assert not folder.exists()
