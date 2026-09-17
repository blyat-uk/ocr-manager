"""Before a folder is open: the empty state ("Open a folder of episodes"),
the folder picker, drag-and-drop payloads and the remembered last folder.

The picker is today's (current-app inventory): `kdialog
--getexistingdirectory <start>` when kdialog is installed, else Qt's
directory dialog. kdialog runs through an asynchronous QProcess so the
window keeps painting and draining job events while it is open. The start
directory is the last folder opened (QSettings "project/last_path"),
defaulting to the user's home directory.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QSettings, Qt, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QLabel, QVBoxLayout, QWidget

from app.widgets.base import Button

SETTINGS_ORGANIZATION = "OCRManager"
SETTINGS_APPLICATION = "OCRTool"
LAST_PATH_KEY = "project/last_path"
PICKER_CAPTION = "Select Project Directory"
ERROR_WIDTH = 560


def app_settings() -> QSettings:
    """The settings today's window uses (geometry, last folder)."""
    return QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)


def last_path() -> str:
    value = app_settings().value(LAST_PATH_KEY)
    return value if isinstance(value, str) and value else str(Path.home())


def remember_path(path: str) -> None:
    app_settings().setValue(LAST_PATH_KEY, path)


def folder_from_mime(mime) -> str | None:
    """The directory a drag carries: exactly one local URL that is a folder."""
    if mime is None or not mime.hasUrls():
        return None
    urls = mime.urls()
    if len(urls) != 1 or not urls[0].isLocalFile():
        return None
    path = urls[0].toLocalFile()
    return path if path and os.path.isdir(path) else None


class FolderPicker(QObject):
    """Asks for a folder; `chosen(path)` when the user picks one."""

    chosen = pyqtSignal(str)

    def __init__(self, parent_widget: QWidget):
        super().__init__(parent_widget)
        self._parent_widget = parent_widget
        self._process: QProcess | None = None

    def pick(self, start_dir: str) -> None:
        kdialog = shutil.which("kdialog")
        if kdialog:
            self._pick_with_kdialog(kdialog, start_dir)
            return
        folder = QFileDialog.getExistingDirectory(self._parent_widget, PICKER_CAPTION, start_dir)
        if folder:
            self.chosen.emit(folder)

    def _pick_with_kdialog(self, program: str, start_dir: str) -> None:
        if self._process is not None:                  # one picker at a time
            return
        process = QProcess(self)
        process.finished.connect(lambda code, status: self._on_kdialog_finished(process, code, status))
        self._process = process
        process.start(program, ["--getexistingdirectory", start_dir])

    def _on_kdialog_finished(self, process: QProcess, code: int, status) -> None:
        self._process = None
        output = bytes(process.readAllStandardOutput()).decode("utf-8", "replace").strip()
        process.deleteLater()
        if status == QProcess.ExitStatus.NormalExit and code == 0 and output:
            self.chosen.emit(output)


class OpenFolderView(QWidget):
    """The centre of the window while no folder is open."""

    choose_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("OpenFolder")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addStretch(1)
        self.title_label = QLabel("Open a folder of episodes")
        self.title_label.setObjectName("OpenTitle")
        self.hint_label = QLabel("or drop a folder anywhere on this window")
        self.hint_label.setObjectName("Note")
        self.choose_button = Button("Choose folder…", "primary")
        self.choose_button.clicked.connect(self.choose_requested)
        self.error_label = QLabel()
        self.error_label.setObjectName("OpenError")
        self.error_label.setWordWrap(True)
        self.error_label.setFixedWidth(ERROR_WIDTH)
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.hide()
        for widget in (self.title_label, self.hint_label, self.choose_button, self.error_label):
            layout.addWidget(widget, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(1)

    def show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()

    def clear_error(self) -> None:
        self.error_label.clear()
        self.error_label.hide()
