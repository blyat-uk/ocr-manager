"""Before anything is open: the empty state ("Open an episode or a folder of
episodes"), the two pickers, drag-and-drop payloads, the remembered last
path and the recent files & folders.

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

The recent files & folders (QSettings "project/recents", the rules in
`app/recents.py`) are listed under the two buttons: what was opened, newest
first, each row opening it again through the window's `open_path`. Removing
one, or clearing them all, only edits the list.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QSettings, QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app import recents
from app.recents import EPISODE, FOLDER, KINDS, RecentEntry
from app.theme import tokens
from app.widgets.base import Badge, Button, ElidedLabel, SectionHeader

SETTINGS_ORGANIZATION = "OCRManager"
SETTINGS_APPLICATION = "OCRTool"
LAST_PATH_KEY = "project/last_path"
RECENTS_KEY = "project/recents"
PICKER_CAPTION = "Select Project Directory"
EPISODE_PICKER_CAPTION = "Select an Episode"
ERROR_WIDTH = 560                # mockup px; scaled through tokens.px() where it is used
RECENTS_TITLE = "Recent files & folders"
RECENTS_WIDTH = 600              # mockup px, like ERROR_WIDTH
RECENTS_GAP = 18                 # mockup px between the buttons and the list
CLEAR_RECENTS_TITLE = "Clear recent files & folders"
CLEAR_RECENTS_TEXT = "Clear the list of recent files and folders?\n\nNothing on disk is touched."
REMOVE_RECENT_TIP = "Take this off the list (Delete). Nothing on disk is touched."

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


def _stored_recents() -> list[RecentEntry]:
    value = app_settings().value(RECENTS_KEY)
    return recents.parse(value) if isinstance(value, str) else []


def _store_recents(entries: list[RecentEntry]) -> None:
    app_settings().setValue(RECENTS_KEY, recents.dump(entries))


def recent_entries() -> list[RecentEntry]:
    """The recent files & folders, newest first: those still there. One that
    is not (a drive that is not mounted) stays stored and comes back."""
    return [entry for entry in _stored_recents() if entry.exists()]


def record_recent(path: str, *, episode: bool, count: int | None = None) -> None:
    """`path` was opened: an episode's video, or a folder holding `count`
    episodes."""
    entry = RecentEntry(path, EPISODE if episode else FOLDER, time.time(), None if episode else count)
    _store_recents(recents.recorded(_stored_recents(), entry))


def forget_recent(path: str) -> None:
    _store_recents(recents.without(_stored_recents(), path))


def clear_recents() -> None:
    _store_recents([])


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


def _badge_width() -> int:
    """The kind badge's width: the widest kind's, so every row's name starts
    at the same x."""
    widths = []
    for kind in KINDS:
        probe = Badge(kind)
        probe.ensurePolished()                  # the stylesheet's font and padding, not the defaults
        widths.append(probe.sizeHint().width())
        probe.deleteLater()
    return max(widths)


class RecentRow(QWidget):
    """One recent file or folder: what it is, its name and where it is, how
    many episodes and when it was opened. A left click, Enter or Space asks
    to open it; Delete, or the "Remove" that shows while the pointer is over
    the row, asks to take it off the list."""

    open_requested = pyqtSignal(str)
    remove_requested = pyqtSignal(str)

    def __init__(self, entry: RecentEntry, now: float, badge_width: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.entry = entry
        self._hovered = False
        self.setObjectName("RecentRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(entry.path)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(tokens.px(10), tokens.px(6), tokens.px(8), tokens.px(6))
        layout.setSpacing(tokens.px(10))
        self.kind_badge = Badge(entry.kind)
        self.kind_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.kind_badge.setFixedWidth(badge_width)
        layout.addWidget(self.kind_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        names = QVBoxLayout()
        names.setContentsMargins(0, 0, 0, 0)
        names.setSpacing(tokens.px(1))
        self.title_label = ElidedLabel(recents.title(entry), tip_when_elided=True)
        self.title_label.setObjectName("RecentName")
        self.location_label = ElidedLabel(recents.location(entry), Qt.TextElideMode.ElideMiddle,
                                          tip_when_elided=True)
        self.location_label.setObjectName("RecentPath")
        names.addWidget(self.title_label)
        names.addWidget(self.location_label)
        layout.addLayout(names, 1)
        self.detail_label = QLabel(recents.detail(entry, now))
        self.detail_label.setObjectName("RecentDetail")
        layout.addWidget(self.detail_label, 0, Qt.AlignmentFlag.AlignVCenter)
        self.remove_button = Button("Remove", "ghost", small=True)
        self.remove_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)      # the row's Delete is the key for it
        self.remove_button.setToolTip(REMOVE_RECENT_TIP)
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self.entry.path))
        policy = self.remove_button.sizePolicy()
        policy.setRetainSizeWhenHidden(True)                          # showing it never shifts the row
        self.remove_button.setSizePolicy(policy)
        self.remove_button.hide()
        layout.addWidget(self.remove_button, 0, Qt.AlignmentFlag.AlignVCenter)

    def _sync_remove(self) -> None:
        self.remove_button.setVisible(self._hovered or self.hasFocus())

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self._hovered = True
        self._sync_remove()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._hovered = False
        self._sync_remove()

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        self._sync_remove()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self._sync_remove()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            event.accept()
            self.open_requested.emit(self.entry.path)
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            event.accept()
            self.open_requested.emit(self.entry.path)
            return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            event.accept()
            self.remove_requested.emit(self.entry.path)
            return
        super().keyPressEvent(event)


class _RecentList(QScrollArea):
    """The rows, as tall as all of them when the window has the room, and
    scrolling when it is too short for them."""

    def __init__(self, content: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("RecentList")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.setWidget(content)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def sizeHint(self) -> QSize:
        return self.widget().sizeHint()

    def minimumSizeHint(self) -> QSize:
        rows = self.widget().layout()
        first = rows.itemAt(0) if rows.count() else None
        height = first.sizeHint().height() if first is not None else 0
        return QSize(0, height)


class OpenFolderView(QWidget):
    """The centre of the window while nothing is open. The episode is the
    primary action (most sessions are one episode); a folder of episodes is
    the second button. `choose_episode_requested` asks for the episode
    picker, `choose_requested` for the folder picker, and
    `open_recent_requested(path)` for one of the recent files & folders
    listed under them."""

    choose_requested = pyqtSignal()
    choose_episode_requested = pyqtSignal()
    open_recent_requested = pyqtSignal(str)

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

        self.recent_section = QWidget()
        self.recent_section.setFixedWidth(tokens.px(RECENTS_WIDTH))
        section = QVBoxLayout(self.recent_section)
        section.setContentsMargins(0, tokens.px(RECENTS_GAP), 0, 0)
        section.setSpacing(tokens.px(6))
        self.clear_recents_button = Button("Clear all", "ghost", small=True)
        self.clear_recents_button.clicked.connect(self._clear_recents)
        self.recent_header = SectionHeader(RECENTS_TITLE, trailing=self.clear_recents_button)
        section.addWidget(self.recent_header)
        rows = QWidget()
        rows.setObjectName("RecentRows")
        self._rows_layout = QVBoxLayout(rows)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(tokens.px(4))
        self._recent_list = _RecentList(rows)
        section.addWidget(self._recent_list)
        recent_row = QHBoxLayout()
        recent_row.addStretch(1)
        recent_row.addWidget(self.recent_section)
        recent_row.addStretch(1)
        layout.addLayout(recent_row)
        layout.addStretch(1)
        self._badge_width = _badge_width()
        self._rows: list[RecentRow] = []
        self.refresh_recents()

    # --- recent files & folders ---------------------------------------------------------

    def recent_rows(self) -> list[RecentRow]:
        return list(self._rows)

    def refresh_recents(self) -> None:
        """List the recent files & folders again (what is there now, and
        "2 h ago" from now). The section hides when there are none."""
        for row in self._rows:
            self._rows_layout.removeWidget(row)
            row.hide()
            row.deleteLater()            # the row may be the one whose signal got us here
        now = time.time()
        self._rows = [RecentRow(entry, now, self._badge_width) for entry in recent_entries()]
        for row in self._rows:
            row.open_requested.connect(self.open_recent_requested)
            row.remove_requested.connect(self._remove_recent)
            self._rows_layout.addWidget(row)
        self.recent_section.setVisible(bool(self._rows))
        self._recent_list.updateGeometry()

    def _remove_recent(self, path: str) -> None:
        """Off the list; the keyboard stays on the list, on the row that
        took its place."""
        index = next((i for i, row in enumerate(self._rows) if row.entry.path == path), 0)
        had_focus = any(row.hasFocus() for row in self._rows)
        forget_recent(path)
        self.refresh_recents()
        if had_focus and self._rows:
            self._rows[min(index, len(self._rows) - 1)].setFocus(Qt.FocusReason.OtherFocusReason)

    def _clear_recents(self) -> None:
        answers = QMessageBox.StandardButton
        if QMessageBox.question(self, CLEAR_RECENTS_TITLE, CLEAR_RECENTS_TEXT, answers.Yes | answers.No,
                                answers.No) != answers.Yes:
            return
        clear_recents()
        self.refresh_recents()

    def show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()

    def clear_error(self) -> None:
        self.error_label.clear()
        self.error_label.hide()
