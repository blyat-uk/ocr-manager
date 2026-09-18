"""The top bar (`.topbar`, ui-spec §3.1): project name and path, the state
chips, and Folder settings / Logs / Start.

- Chips follow `controller.counts()`, which agrees with the row badges
  (ruling B10). The "detecting" dot is blue while any detection job runs and
  grey when detections are only queued or held (ruling B7).
- "▶ Start N ready files" counts `controller.startable_files()` (ruling B6).
  With no ready file but done ones to re-run it reads "▶ Re-run N done
  files" (the overwrite question follows); with neither it stays
  "▶ Start 0 ready files", disabled, as it is when both extraction toggles
  are off. A start the controller refuses is shown beside it
  (`show_start_error`).
- The "Review · Run" switch shows while a run exists, until another folder
  opens (ruling B12); the window switches the centre and right area.
- During a run (plan 3B Task 5, B12/B13) the chips, Folder settings and
  Start give way to the run status in place of the path (`run_status_text`,
  refreshed every second) and "⏸ pause" / "▶ resume" and "■ stop"; both are
  disabled once a stop was asked for. "⤓ Logs" stays: the logs are
  read-only and reachable at any time.

Badges change on job "started" events, which emit only `activity_changed`,
so everything here refreshes on that signal too. Signals only mark the bar
dirty; it refreshes once per event-loop turn, so a burst of `file_changed`
costs one `counts()`.
"""
from __future__ import annotations

import os
import time

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget

from app.run_snapshot import run_status_text
from app.theme import tokens
from app.views.deferred import Deferred
from app.widgets.base import Button, Chip, ElidedLabel, SegmentedControl

APP_NAME = "OCR Manager"
PAUSE_TEXT, RESUME_TEXT, STOP_TEXT = "⏸ pause", "▶ resume", "■ stop"
STATUS_REFRESH_MS = 1000


def start_text(count: int) -> str:
    return f"▶ Start {count} ready {'file' if count == 1 else 'files'}"


def rerun_text(count: int) -> str:
    """The Start button when only files that already have output can run."""
    return f"▶ Re-run {count} done {'file' if count == 1 else 'files'}"


class _ProjectBlock(QWidget):
    """Project name + "· path"; clicking it opens another folder."""

    clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(tokens.px(5))
        self.name_label = ElidedLabel(APP_NAME)
        self.name_label.setObjectName("ProjectName")
        self.name_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.path_label = ElidedLabel(mode=Qt.TextElideMode.ElideMiddle)
        self.path_label.setObjectName("ProjectPath")
        self.path_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.name_label, 0, Qt.AlignmentFlag.AlignBaseline)
        layout.addWidget(self.path_label, 1, Qt.AlignmentFlag.AlignBaseline)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Open another folder (Ctrl+O)")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class TopBar(QWidget):
    folder_settings_requested = pyqtSignal()
    logs_requested = pyqtSignal()
    start_requested = pyqtSignal()
    open_requested = pyqtSignal()
    mode_changed = pyqtSignal(int)          # 0 Review, 1 Run

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__(parent)
        self._controller = controller
        self.setObjectName("TopBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(tokens.px(14), tokens.px(9), tokens.px(14), tokens.px(9))
        layout.setSpacing(tokens.px(14))

        self._project = _ProjectBlock()
        self._project.clicked.connect(self.open_requested)
        self.project_label = self._project.name_label
        self.path_label = self._project.path_label
        layout.addWidget(self._project)

        self._chips = QWidget()
        self._chips.setObjectName("ChipBar")
        chips = QHBoxLayout(self._chips)
        chips.setContentsMargins(tokens.px(6), 0, 0, 0)
        chips.setSpacing(tokens.px(6))
        self.reviewed_chip = Chip("ok", 0, "reviewed")
        self.needs_chip = Chip("warn", 0, "needs you")
        self.detecting_chip = Chip("idle", 0, "detecting")
        for chip in (self.reviewed_chip, self.needs_chip, self.detecting_chip):
            chips.addWidget(chip)
        layout.addWidget(self._chips)

        self.run_switch = SegmentedControl(["Review", "Run"])
        self.run_switch.current_changed.connect(self.mode_changed)
        self.run_switch.hide()
        layout.addWidget(self.run_switch)
        layout.addStretch(1)

        self.settings_button = Button("⚙ Folder settings", "ghost")
        self.settings_button.clicked.connect(self.folder_settings_requested)
        self.logs_button = Button("⤓ Logs", "ghost")
        self.logs_button.clicked.connect(self.logs_requested)
        self.start_error_label = ElidedLabel()
        self.start_error_label.setObjectName("StartError")
        self.start_error_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.start_error_label.hide()
        self.start_button = Button(start_text(0), "primary")
        self.start_button.clicked.connect(self.start_requested)
        self.pause_button = Button(PAUSE_TEXT, "ghost")
        self.pause_button.clicked.connect(self._toggle_run_pause)
        self.stop_button = Button(STOP_TEXT, "ghost")
        self.stop_button.clicked.connect(lambda _checked=False: self._controller.stop_run())
        layout.addWidget(self.settings_button)
        layout.addWidget(self.logs_button)
        layout.addWidget(self.start_error_label)
        for button in (self.settings_button, self.logs_button, self.start_button, self.pause_button,
                       self.stop_button):
            button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        for button in (self.start_button, self.pause_button, self.stop_button):
            layout.addWidget(button)
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(STATUS_REFRESH_MS)
        self._status_timer.timeout.connect(self.refresh)
        controller.project_opened.connect(self.clear_start_error)
        controller.project_closed.connect(self.clear_start_error)

        self._refresh_later = Deferred(self.refresh, self)
        for signal in (controller.project_opened, controller.project_closed, controller.files_changed,
                       controller.file_changed, controller.activity_changed, controller.folder_changed,
                       controller.run_changed):
            signal.connect(self._refresh_later.schedule)
        self.refresh()

    def set_mode(self, mode: int) -> None:
        """Show `mode` (0 Review, 1 Run) on the switch without emitting."""
        self.run_switch.set_current(mode)

    def show_start_error(self, text: str) -> None:
        self.start_error_label.set_full_text(text)
        self.start_error_label.show()

    def clear_start_error(self, *_args) -> None:
        self.start_error_label.set_full_text("")
        self.start_error_label.hide()

    def _toggle_run_pause(self) -> None:
        snapshot = self._controller.run_snapshot()
        if snapshot is None or snapshot.finished:
            return
        if snapshot.paused:
            self._controller.resume_run()
        else:
            self._controller.pause_run()

    def refresh(self, *_args) -> None:
        self._refresh_later.cancel()
        controller = self._controller
        project = controller.project
        is_open = project is not None
        snapshot = controller.run_snapshot() if is_open else None
        run_active = snapshot is not None and not snapshot.finished
        for widget in (self._chips, self.settings_button, self.start_button):
            widget.setVisible(is_open and not run_active)
        self.logs_button.setVisible(is_open)                 # logs stay reachable during a run
        for widget in (self.pause_button, self.stop_button):
            widget.setVisible(run_active)
        if run_active:
            self.start_error_label.hide()
        self.run_switch.setVisible(snapshot is not None)
        if run_active and not self._status_timer.isActive():
            self._status_timer.start()
        elif not run_active:
            self._status_timer.stop()
        if not is_open:
            self.project_label.set_full_text(APP_NAME)
            self.path_label.set_full_text("")
            return
        self.project_label.set_full_text(os.path.basename(project.path) or project.path)
        if run_active:
            self.path_label.set_full_text(run_status_text(snapshot, time.monotonic()))
            self.pause_button.setText(RESUME_TEXT if snapshot.paused else PAUSE_TEXT)
            for button in (self.pause_button, self.stop_button):
                button.setEnabled(not snapshot.stopping)
            return
        self.path_label.set_full_text(f"· {project.path}")

        counts = controller.counts()
        self.reviewed_chip.set_count(counts["reviewed"])
        self.needs_chip.set_count(counts["needs_you"])
        self.detecting_chip.set_count(counts["detecting"])
        activity = controller.activity()
        running = any(controller.is_detection_kind(kind) for kind, _file in activity.running)
        self.detecting_chip.set_tone("run" if running else "idle")

        # One call: startable_files() is this list without the done files (its include_done rule).
        startable = controller.startable_files(include_done=True)         # done files can be re-run
        ready = sum(1 for name in startable if not controller.is_done(name))
        folder = project.folder
        self.start_button.setText(start_text(ready) if ready or not startable else rerun_text(len(startable)))
        self.start_button.setEnabled(bool(startable) and (folder.dialogue_enabled or folder.labels_enabled))
