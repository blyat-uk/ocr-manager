"""The entry points of the episode view (Stream D): what a drop, the pickers,
the empty state and the command line accept before anything is opened.

`app/views/open_folder.py` decides only *which path* the user meant; what
opens for it (episode view or workbench) is `MainWindow.open_path`'s call.
"""
from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path

import pytest
from PyQt6.QtCore import QMimeData, QSettings, QUrl
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QWidget

from app.views import open_folder as open_folder_module
from app.views.open_folder import (
    EpisodePicker,
    FolderPicker,
    OpenFolderView,
    folder_from_mime,
    last_dir,
    last_path,
    path_from_mime,
    remember_path,
)

WAIT_MS = 5000


def wait_for(predicate, timeout_ms: int = WAIT_MS) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        QTest.qWait(10)
    return True


@pytest.fixture(autouse=True)
def settings_dir(tmp_path):
    path = tmp_path / "qsettings"
    path.mkdir()
    for fmt in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, str(path))
    return path


@pytest.fixture
def parent(qapp):
    widget = QWidget()
    yield widget
    widget.deleteLater()


def mime_of(*paths) -> QMimeData:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


# --------------------------------------------------------------------------
# Drops
# --------------------------------------------------------------------------

def test_path_from_mime_takes_a_folder_or_one_video(tmp_path, qapp):
    video = tmp_path / "ep01.mkv"
    video.write_bytes(b"")
    upper = tmp_path / "EP02.MP4"                          # store.list_video_files: case does not matter
    upper.write_bytes(b"")
    text = tmp_path / "notes.txt"
    text.write_text("x")
    assert path_from_mime(mime_of(tmp_path)) == str(tmp_path)
    assert path_from_mime(mime_of(video)) == str(video)
    assert path_from_mime(mime_of(upper)) == str(upper)
    assert path_from_mime(mime_of(text)) is None           # not a video
    assert path_from_mime(mime_of(tmp_path / "gone.mkv")) is None   # a video name that is not a file
    assert path_from_mime(mime_of(video, upper)) is None   # one thing at a time
    remote = QMimeData()
    remote.setUrls([QUrl("https://example.com/ep01.mkv")])
    assert path_from_mime(remote) is None
    assert path_from_mime(QMimeData()) is None
    assert path_from_mime(None) is None


def test_folder_from_mime_still_takes_folders_only(tmp_path, qapp):
    """The window's drop handler calls folder_from_mime and then open_folder
    until it is wired to open_path: a video must not reach open_folder."""
    video = tmp_path / "ep01.mkv"
    video.write_bytes(b"")
    assert folder_from_mime(mime_of(tmp_path)) == str(tmp_path)
    assert folder_from_mime(mime_of(video)) is None
    assert folder_from_mime(None) is None


# --------------------------------------------------------------------------
# The remembered path
# --------------------------------------------------------------------------

def test_last_dir_is_the_folder_of_a_remembered_episode(tmp_path, qapp):
    assert last_path() == str(Path.home()) and last_dir() == str(Path.home())
    video = tmp_path / "ep01.mkv"
    video.write_bytes(b"")
    remember_path(str(video))
    assert last_path() == str(video)                      # unchanged: the episode itself
    assert last_dir() == str(tmp_path)
    remember_path(str(tmp_path))
    assert last_dir() == str(tmp_path)


# --------------------------------------------------------------------------
# Pickers
# --------------------------------------------------------------------------

def fake_kdialog(tmp_path, monkeypatch, answer: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "kdialog-args"
    script = bin_dir / "kdialog"
    script.write_text(f'#!/bin/sh\nfor a in "$@"; do echo "$a"; done > "{record}"\necho "{answer}"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return record


@pytest.mark.skipif(sys.platform == "win32", reason="a shell-script kdialog")
def test_episode_picker_runs_kdialog_for_videos(tmp_path, monkeypatch, parent):
    video = tmp_path / "ep01.mkv"
    video.write_bytes(b"")
    record = fake_kdialog(tmp_path, monkeypatch, str(video))
    picker = EpisodePicker(parent)
    chosen = []
    picker.chosen.connect(chosen.append)
    picker.pick(str(tmp_path))
    assert wait_for(lambda: chosen)
    assert chosen == [str(video)]
    assert record.read_text().splitlines() == ["--getopenfilename", str(tmp_path), "*.mkv *.mp4|Videos"]


def test_episode_picker_uses_the_qt_dialog_without_kdialog(tmp_path, monkeypatch, parent):
    video = tmp_path / "ep01.mkv"
    asked = []

    def dialog(parent_widget, caption, start, filter_text):
        asked.append((start, filter_text))
        return str(video), filter_text

    monkeypatch.setattr(open_folder_module.shutil, "which", lambda tool: None)
    monkeypatch.setattr(open_folder_module.QFileDialog, "getOpenFileName", dialog)
    picker = EpisodePicker(parent)
    chosen = []
    picker.chosen.connect(chosen.append)
    picker.pick(str(tmp_path))
    assert chosen == [str(video)]
    assert asked == [(str(tmp_path), "Videos (*.mkv *.mp4)")]


def test_episode_picker_cancelled_chooses_nothing(tmp_path, monkeypatch, parent):
    monkeypatch.setattr(open_folder_module.shutil, "which", lambda tool: None)
    monkeypatch.setattr(open_folder_module.QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    picker = EpisodePicker(parent)
    chosen = []
    picker.chosen.connect(chosen.append)
    picker.pick(str(tmp_path))
    assert chosen == []


def test_episode_picker_falls_back_when_kdialog_fails_to_start(tmp_path, monkeypatch, parent):
    video = tmp_path / "ep01.mkv"
    monkeypatch.setattr(open_folder_module.shutil, "which", lambda tool: str(tmp_path / "no-such-kdialog"))
    monkeypatch.setattr(open_folder_module.QFileDialog, "getOpenFileName", lambda *args: (str(video), ""))
    picker = EpisodePicker(parent)
    chosen = []
    picker.chosen.connect(chosen.append)
    picker.pick(str(tmp_path))
    assert wait_for(lambda: chosen)
    assert chosen == [str(video)]


def test_folder_picker_is_unchanged(tmp_path, monkeypatch, parent):
    monkeypatch.setattr(open_folder_module.shutil, "which", lambda tool: None)
    monkeypatch.setattr(open_folder_module.QFileDialog, "getExistingDirectory",
                        lambda parent_widget, caption, start: str(tmp_path))
    picker = FolderPicker(parent)
    chosen = []
    picker.chosen.connect(chosen.append)
    picker.pick(str(Path.home()))
    assert chosen == [str(tmp_path)]


# --------------------------------------------------------------------------
# The empty state
# --------------------------------------------------------------------------

def test_empty_state_offers_an_episode_first_and_a_folder_second(qapp):
    view = OpenFolderView()
    assert view.title_label.text() == "Open an episode or a folder of episodes"
    assert view.hint_label.text() == "or drop one anywhere on this window"
    assert view.episode_button.text() == "Open episode…"
    assert view.episode_button.property("variant") == "primary"
    assert view.choose_button.text() == "Open folder…"
    assert view.choose_button.property("variant") != "primary"
    episode, folder = [], []
    view.choose_episode_requested.connect(lambda: episode.append(1))
    view.choose_requested.connect(lambda: folder.append(1))
    view.episode_button.click()
    assert (episode, folder) == ([1], [])
    view.choose_button.click()
    assert (episode, folder) == ([1], [1])
    view.deleteLater()


# --------------------------------------------------------------------------
# python -m app PATH
# --------------------------------------------------------------------------

class _FakeWindow(QWidget):
    opened: list = []

    def __init__(self, tabs_factory=None):
        super().__init__()

    def report_unexpected_error(self, text):
        pass


class _FolderOnlyWindow(_FakeWindow):
    def open_folder(self, path):
        self.opened.append(("open_folder", path))


class _PathWindow(_FolderOnlyWindow):
    def open_path(self, path):
        self.opened.append(("open_path", path))


@pytest.mark.parametrize("window_class, method", [(_PathWindow, "open_path"), (_FolderOnlyWindow, "open_folder")])
def test_the_command_line_path_goes_to_open_path_when_the_window_has_it(qapp, monkeypatch, tmp_path,
                                                                        window_class, method):
    import app.__main__ as entry
    import app.main_window

    video = tmp_path / "ep01.mkv"
    video.write_bytes(b"")
    window_class.opened = []
    monkeypatch.setattr(entry, "apply_theme", lambda app: None)
    monkeypatch.setattr(app.main_window, "MainWindow", window_class)
    assert entry.main([str(video), "--quit-after", "0.1"]) == 0
    assert window_class.opened == [(method, str(video))]


def test_the_positional_argument_is_an_episode_or_a_folder():
    import app.__main__ as entry

    args, _ = entry._parse(["/some/ep01.mkv"])
    assert args.path == "/some/ep01.mkv"
    assert entry._parse([])[0].path is None
