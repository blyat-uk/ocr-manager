"""Where a project's files live: one Qt-free answer for every path decision.

A project is opened one of two ways, and everything that reads or writes a
file on its behalf -- the store, the view cache, the detection jobs, the run
-- asks a ProjectLayout rather than joining paths onto `Project.path` itself:

    folder   a folder of episodes, exactly as the app has always kept it:
             `<dir>/.ocr.json` and `<dir>/.ocr-cache/`. folder_layout
             reproduces those paths byte for byte -- tests/test_layout.py
             pins them as literals -- so a folder opened after this module
             existed reads and writes the same files it did before.

    episode  one video, OCR'd on its own. Nothing but the output is written
             next to it: its settings and caches live under the per-user
             cache root (cache_root), in `<root>/videos/<key>/`, keyed by the
             video's content (video_key) so they follow the file through a
             rename or a move.

Both write their output the same way, named with the OCR language's tag
(core.project.languages.output_tag, "zh" for "ch"): `<video dir>/<tag>/
<stem>.<tag>.ass` with FolderSettings.output_subfolder on (the default),
`<video dir>/<stem>.<tag>.ass` with it off. Those two settings live on the
project, not the layout it was loaded with, so layout_of fills them in each
time it is asked. An output anywhere else -- the old `chi/<stem>.ass` --
is not this project's output: it neither counts as done nor is replaced.

The episode's `settings.json` is the same v2 document as a folder's
`.ocr.json`, with one file entry, and its `evidence/` and `view/` have the
cache directory's layout, so the store and the view cache need to know only
`config_path` and `cache_dir`.

This module sits below core.project.store (which re-exports the video-file
rule from here), so nothing in it may import the store.
"""
from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from core.project.languages import DEFAULT_TAG, output_tag

if TYPE_CHECKING:
    from core.project.model import Project

VIDEO_EXTENSIONS = (".mkv", ".mp4")
CONFIG_FILENAME = ".ocr.json"
CACHE_DIRNAME = ".ocr-cache"      # a folder's cache directory (same as core.detect.ranges.pipeline's)
V1_BACKUP_SUFFIX = ".v1.bak"

EPISODE_CONFIG_FILENAME = "settings.json"
EPISODES_DIRNAME = "videos"       # <root>/videos/<key>/
CACHE_ENV = "OCR_MANAGER_CACHE_DIR"
APP_DIRNAME = "ocr-manager"

FOLDER, EPISODE = "folder", "episode"

KEY_CHUNK = 1024 * 1024           # video_key hashes this much of the head and of the tail
KEY_DIGEST_SIZE = 16              # bytes: 32 hex characters


def list_video_files(project_dir: str) -> list[str]:
    """Sorted video file names in `project_dir` -- the v1 app's rule
    (core/pipeline.py get_video_files: names ending in a VIDEO_EXTENSIONS
    entry), except that the extension's case does not matter on any OS:
    "EP01.MKV" is a video on Linux too, as it already was on Windows.
    """
    names = [
        f.name
        for f in Path(project_dir).iterdir()
        if is_video_name(f.name)
    ]
    return sorted(names)


def is_video_name(name: str) -> bool:
    """`name` ends in a VIDEO_EXTENSIONS entry, in any case."""
    return name.lower().endswith(VIDEO_EXTENSIONS)


def output_name(video_name: str, tag: str = DEFAULT_TAG) -> str:
    """The output file of video `video_name`: "<stem>.<tag>.ass", the stem as
    the v1 OCRWorker took it (pathlib's, so "a.b.mkv" gives "a.b.zh.ass")."""
    return f"{Path(video_name).stem}.{tag}.ass"


@dataclass(frozen=True)
class ProjectLayout:
    """Every path of one open project (see the module docstring).

    Strings, joined with os.path.join onto what was passed in and never
    normalised further, so a folder layout's paths are the very strings the
    code before it built.
    """
    kind: str                     # FOLDER | EPISODE
    video_dir: str                # the directory holding the video(s); == Project.path
    config_path: str              # folder: <dir>/.ocr.json   episode: <root>/videos/<key>/settings.json
    cache_dir: str                # folder: <dir>/.ocr-cache  episode: <root>/videos/<key>
    only_file: str | None = None  # episode: the video's file name
    key: str | None = None        # episode: video_key of the video
    output_tag: str = DEFAULT_TAG           # names the output (see the module docstring)
    output_subfolder: bool = True           # output in <video_dir>/<tag>/, else next to the video

    @property
    def is_episode(self) -> bool:
        return self.kind == EPISODE

    def video_names(self) -> list[str]:
        """The videos the project holds: a folder's by list_video_files, an
        episode's own file while it is there (and nothing once it is gone,
        which the store reconciles like a deleted file in a folder)."""
        if not self.is_episode:
            return list_video_files(self.video_dir)
        return [self.only_file] if os.path.isfile(os.path.join(self.video_dir, self.only_file)) else []

    def output_path(self, name: str) -> str:
        """The finished subtitle file of video `name`. A run writes
        `<this>.partial` first and replaces this only when that is ready."""
        return os.path.join(self.video_dir, *self.output_label(name).split("/"))

    def output_label(self, name: str) -> str:
        """output_path relative to the video directory, "/"-separated on
        every OS, for messages: "zh/EP01.zh.ass", or "EP01.zh.ass" without
        the subfolder."""
        file_name = output_name(name, self.output_tag)
        return f"{self.output_tag}/{file_name}" if self.output_subfolder else file_name

    def output_dirs(self) -> tuple[str, ...]:
        """The directories a run creates before its first file: the output
        subfolder, or none when the output goes next to the video."""
        if not self.output_subfolder:
            return ()
        return (os.path.join(self.video_dir, self.output_tag),)

    @property
    def v1_backup_path(self) -> str | None:
        """Where the first save after a v1 migration copies the v1 file. Only
        a folder can have been a v1 project."""
        if self.is_episode:
            return None
        return self.config_path + V1_BACKUP_SUFFIX


def folder_layout(directory: str) -> ProjectLayout:
    """The layout of a folder of episodes, `directory` as given."""
    return ProjectLayout(
        kind=FOLDER,
        video_dir=directory,
        config_path=os.path.join(directory, CONFIG_FILENAME),
        cache_dir=os.path.join(directory, CACHE_DIRNAME),
    )


def episode_layout(video_path: str, root: str | None = None, *, key: str | None = None) -> ProjectLayout:
    """The layout of the single video `video_path`, its state under `root`
    (cache_root() by default). Hashes the file for its key unless `key` is
    given; OSError when it cannot be read."""
    video_path = os.path.abspath(video_path)
    key = video_key(video_path) if key is None else key
    cache_dir = os.path.join(cache_root() if root is None else root, EPISODES_DIRNAME, key)
    return ProjectLayout(
        kind=EPISODE,
        video_dir=os.path.dirname(video_path),
        config_path=os.path.join(cache_dir, EPISODE_CONFIG_FILENAME),
        cache_dir=cache_dir,
        only_file=os.path.basename(video_path),
        key=key,
    )


def layout_of(project: Project) -> ProjectLayout:
    """The project's layout: the one it was loaded with, or -- for a Project
    built without one -- the folder layout of its path, with the output
    naming of its current settings (ocr_lang, output_subfolder)."""
    base = project.layout or folder_layout(project.path)
    return replace(base, output_tag=output_tag(project.folder.ocr_lang),
                   output_subfolder=bool(project.folder.output_subfolder))


def cache_root() -> str:
    """Where episode state lives: $OCR_MANAGER_CACHE_DIR when it is set,
    otherwise the platform's per-user cache directory (_default_cache_root)."""
    override = os.environ.get(CACHE_ENV)
    if override:
        return override
    return _default_cache_root(sys.platform, os.environ, os.path.expanduser("~"))


def _default_cache_root(platform: str, environ: Mapping[str, str], home: str) -> str:
    """Linux and the rest: $XDG_CACHE_HOME/ocr-manager (an absolute value
    only, as the XDG spec says), else ~/.cache/ocr-manager. Windows:
    %LOCALAPPDATA%\\ocr-manager\\cache. macOS: ~/Library/Caches/ocr-manager."""
    if platform.startswith("win"):
        base = environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        return os.path.join(base, APP_DIRNAME, "cache")
    if platform == "darwin":
        return os.path.join(home, "Library", "Caches", APP_DIRNAME)
    xdg = environ.get("XDG_CACHE_HOME")
    if xdg and os.path.isabs(xdg):
        return os.path.join(xdg, APP_DIRNAME)
    return os.path.join(home, ".cache", APP_DIRNAME)


def video_key(path: str) -> str:
    """The content key of a video: blake2b (16-byte digest, hex) over its
    size as 8 bytes little-endian, its first KEY_CHUNK bytes and its last
    KEY_CHUNK bytes (overlapping in a file shorter than two chunks).

    Content, not name: the key survives a rename or a move, and reading two
    MiB keeps it cheap on a multi-GB episode. A change in the middle of a
    large file keeps the key -- the view cache still notices, by size and
    mtime, that its pixels belong to another file."""
    digest = hashlib.blake2b(digest_size=KEY_DIGEST_SIZE)
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        digest.update(size.to_bytes(8, "little"))
        digest.update(handle.read(KEY_CHUNK))
        handle.seek(max(0, size - KEY_CHUNK))
        digest.update(handle.read(KEY_CHUNK))
    return digest.hexdigest()


def episode_target(path: str) -> str | None:
    """The video an open of `path` is about, as an absolute path: `path`
    itself when it is a video file, the one video of a directory holding
    exactly one; None for anything else (a folder of 0 or several videos
    opens as a folder, and anything else fails the way a folder open does)."""
    try:
        if os.path.isfile(path):
            return os.path.abspath(path) if is_video_name(os.path.basename(path)) else None
        if os.path.isdir(path):
            names = list_video_files(path)
            if len(names) == 1 and os.path.isfile(os.path.join(path, names[0])):
                return os.path.abspath(os.path.join(path, names[0]))
    except OSError:
        pass
    return None
