"""Before anything is open: the empty state ("Open an episode or a folder of
episodes"), the two pickers, drag-and-drop payloads and the remembered last
path.

This module only decides *which path* the user meant -- a folder, or one
video file. What opens for it (the episode view or the workbench) is the
window's `open_path` routing.

Both pickers use kdialog when it is installed, else Qt's dialog: the folder
picker is today's (`kdialog --getexistingdirectory <start>`), the episode
picker asks for one video (`kdialog --getopenfilename <start> "*.mkv
*.mp4|Videos"`). kdialog runs through an asynchronous QProcess so the window
keeps painting and draining job events while it is open. The remembered path
(QSettings "project/last_path") is whatever was opened last, a folder or an
episode's video; `last_dir()` turns it into the start directory, defaulting
to the user's home directory.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QSettings, Qt, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.theme import tokens
from app.widgets.base import Button

SETTINGS_ORGANIZATION = "OCRManager"
SETTINGS_APPLICATION = "OCRTool"
LAST_PATH_KEY = "project/last_path"
PICKER_CAPTION = "Select Project Directory"
EPISODE_PICKER_CAPTION = "Select an Episode"
ERROR_WIDTH = 560                # mockup px; scaled through tokens.px() where it is used

# The video extensions, lower case. The source of truth is
# core/project/store.py VIDEO_EXTENSIONS (list_video_files' rule, case-blind);
# app/views may not import core (test_views_import_no_core_modules), so the
# tuple is repeated here and must follow it.
VIDEO_EXTENSIONS = (".mkv", ".mp4")
KDIALOG_VIDEO_FILTER = " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS) + "|Videos"
QT_VIDEO_FILTER = "Videos (" + " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS) + ")"


def app_settings() -> QSettings:
    """The settings today's window uses (geometry, last folder)."""
    return QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)


def last_path() -> str:
    """The folder or episode video opened last (home when there is none)."""
    value = app_settings().value(LAST_PATH_KEY)
    return value if isinstance(value, str) and value else str(Path.home())


def last_dir() -> str:
    """Where a picker starts: the last path, or its folder when it was an
    episode's video -- kdialog and Qt both want a directory to open in."""
    path = last_path()
    return os.path.dirname(path) if os.path.isfile(path) else path


def remember_path(path: str) -> None:
    app_settings().setValue(LAST_PATH_KEY, path)


def is_video_file(path: str) -> bool:
    """An existing file whose name ends in a video extension, in any case
    (store.list_video_files' rule)."""
    return path.lower().endswith(VIDEO_EXTENSIONS) and os.path.isfile(path)


def path_from_mime(mime) -> str | None:
    """What a drag carries: exactly one local URL that is a folder or a video
    file. Anything else (several URLs, a remote URL, another file) is None,
    so the window never accepts a drop it cannot open."""
    if mime is None or not mime.hasUrls():
        return None
    urls = mime.urls()
    if len(urls) != 1 or not urls[0].isLocalFile():
        return None
    path = urls[0].toLocalFile()
    if not path:
        return None
    return path if os.path.isdir(path) or is_video_file(path) else None


def folder_from_mime(mime) -> str | None:
    """`path_from_mime` narrowed to folders -- the workbench's drop rule, kept
    for callers that hand the path to `open_folder`, which takes no video."""
    path = path_from_mime(mime)
    return path if path is not None and os.path.isdir(path) else None


class _Picker(QObject):
    """Asks for a path with kdialog when it is installed, else Qt's dialog;
    `chosen(path)` when the user picks one, nothing when they cancel.
    Subclasses give kdialog's arguments and the Qt dialog."""

    chosen = pyqtSignal(str)

    def __init__(self, parent_widget: QWidget):
        super().__init__(parent_widget)
        self._parent_widget = parent_widget
        self._process: QProcess | None = None

    def pick(self, start_dir: str) -> None:
        kdialog = shutil.which("kdialog")
        if kdialog:
            self._pick_with_kdialog(kdialog, start_dir)
        else:
            self._pick_with_qt(start_dir)

    def _kdialog_args(self, start_dir: str) -> list[str]:
        raise NotImplementedError

    def _ask_qt(self, start_dir: str) -> str:
        raise NotImplementedError

    def _pick_with_qt(self, start_dir: str) -> None:
        path = self._ask_qt(start_dir)
        if path:
            self.chosen.emit(path)

    def _pick_with_kdialog(self, program: str, start_dir: str) -> None:
        if self._process is not None:                  # one picker at a time
            return
        process = QProcess(self)
        process.finished.connect(lambda code, status: self._on_kdialog_finished(process, code, status))
        process.errorOccurred.connect(lambda error: self._on_kdialog_error(process, error, start_dir))
        self._process = process
        process.start(program, self._kdialog_args(start_dir))

    def _on_kdialog_error(self, process: QProcess, error, start_dir: str) -> None:
        """kdialog never started (no "finished" follows): forget it and ask
        with Qt's dialog instead."""
        if error != QProcess.ProcessError.FailedToStart or self._process is not process:
            return
        self._process = None
        process.deleteLater()
        self._pick_with_qt(start_dir)

    def _on_kdialog_finished(self, process: QProcess, code: int, status) -> None:
        self._process = None
        output = bytes(process.readAllStandardOutput()).decode("utf-8", "replace").strip()
        process.deleteLater()
        if status == QProcess.ExitStatus.NormalExit and code == 0 and output:
            self.chosen.emit(output)


class FolderPicker(_Picker):
    """Asks for a folder; `chosen(path)` when the user picks one."""

    def _kdialog_args(self, start_dir: str) -> list[str]:
        return ["--getexistingdirectory", start_dir]

    def _ask_qt(self, start_dir: str) -> str:
        return QFileDialog.getExistingDirectory(self._parent_widget, PICKER_CAPTION, start_dir)


class EpisodePicker(_Picker):
    """Asks for one video file (an episode); `chosen(path)` when the user
    picks one. Both dialogs filter on the video extensions."""

    def _kdialog_args(self, start_dir: str) -> list[str]:
        return ["--getopenfilename", start_dir, KDIALOG_VIDEO_FILTER]

    def _ask_qt(self, start_dir: str) -> str:
        path, _ = QFileDialog.getOpenFileName(self._parent_widget, EPISODE_PICKER_CAPTION, start_dir,
                                              QT_VIDEO_FILTER)
        return path


class OpenFolderView(QWidget):
    """The centre of the window while nothing is open. The episode is the
    primary action (most sessions are one episode); a folder of episodes is
    the second button. `choose_episode_requested` asks for the episode
    picker, `choose_requested` for the folder picker."""

    choose_requested = pyqtSignal()
    choose_episode_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("OpenFolder")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setSpacing(tokens.px(10))
        layout.addStretch(1)
        self.title_label = QLabel("Open an episode or a folder of episodes")
        self.title_label.setObjectName("OpenTitle")
        self.hint_label = QLabel("or drop one anywhere on this window")
        self.hint_label.setObjectName("Note")
        self.episode_button = Button("Open episode…", "primary")
        self.episode_button.clicked.connect(self.choose_episode_requested)
        self.choose_button = Button("Open folder…")
        self.choose_button.clicked.connect(self.choose_requested)
        self.error_label = QLabel()
        self.error_label.setObjectName("OpenError")
        self.error_label.setWordWrap(True)
        self.error_label.setFixedWidth(tokens.px(ERROR_WIDTH))
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.hide()
        for widget in (self.title_label, self.hint_label):
            layout.addWidget(widget, 0, Qt.AlignmentFlag.AlignHCenter)
        buttons = QHBoxLayout()
        buttons.setSpacing(tokens.px(8))
        buttons.addStretch(1)
        buttons.addWidget(self.episode_button)
        buttons.addWidget(self.choose_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        error_row = QHBoxLayout()                # no alignment flag: the label wraps to its full height
        error_row.addStretch(1)
        error_row.addWidget(self.error_label)
        error_row.addStretch(1)
        layout.addLayout(error_row)
        layout.addStretch(1)

    def show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()

    def clear_error(self) -> None:
        self.error_label.clear()
        self.error_label.hide()
