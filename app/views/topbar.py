"""The top bar (`.topbar`, ui-spec §3.1): project name and path, the state
chips, and Folder settings / Logs / Start.

- Chips follow `controller.counts()`, which agrees with the row badges
  (ruling B10). The "detecting" dot is blue while any detection job runs and
  grey when detections are only queued or held (ruling B7).
- "▶ Start N ready files" counts `controller.startable_files()` (ruling B6)
  and is disabled when N is 0 or both extraction toggles are off.
- The "Review · Run" switch shows only while a run exists (plan 3B Task 5
  wires what it switches).

Badges change on job "started" events, which emit only `activity_changed`,
so everything here refreshes on that signal too. Signals only mark the bar
dirty; it refreshes once per event-loop turn, so a burst of `file_changed`
costs one `counts()`.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget

from app.views.deferred import Deferred
from app.widgets.base import Button, Chip, ElidedLabel, SegmentedControl

APP_NAME = "OCR Manager"


def start_text(count: int) -> str:
    return f"▶ Start {count} ready {'file' if count == 1 else 'files'}"


class _ProjectBlock(QWidget):
    """Project name + "· path"; clicking it opens another folder."""

    clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
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
        layout.setContentsMargins(14, 9, 14, 9)
        layout.setSpacing(14)

        self._project = _ProjectBlock()
        self._project.clicked.connect(self.open_requested)
        self.project_label = self._project.name_label
        self.path_label = self._project.path_label
        layout.addWidget(self._project)

        self._chips = QWidget()
        self._chips.setObjectName("ChipBar")
        chips = QHBoxLayout(self._chips)
        chips.setContentsMargins(6, 0, 0, 0)
        chips.setSpacing(6)
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
        self.start_button = Button(start_text(0), "primary")
        self.start_button.clicked.connect(self.start_requested)
        for button in (self.settings_button, self.logs_button, self.start_button):
            button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            layout.addWidget(button)

        self._refresh_later = Deferred(self.refresh, self)
        for signal in (controller.project_opened, controller.project_closed, controller.files_changed,
                       controller.file_changed, controller.activity_changed, controller.folder_changed,
                       controller.run_changed):
            signal.connect(self._refresh_later.schedule)
        self.refresh()

    def refresh(self, *_args) -> None:
        self._refresh_later.cancel()
        controller = self._controller
        project = controller.project
        is_open = project is not None
        for widget in (self._chips, self.settings_button, self.logs_button, self.start_button):
            widget.setVisible(is_open)
        self.run_switch.setVisible(is_open and controller.run_snapshot() is not None)
        if not is_open:
            self.project_label.set_full_text(APP_NAME)
            self.path_label.set_full_text("")
            return
        self.project_label.set_full_text(os.path.basename(project.path) or project.path)
        self.path_label.set_full_text(f"· {project.path}")

        counts = controller.counts()
        self.reviewed_chip.set_count(counts["reviewed"])
        self.needs_chip.set_count(counts["needs_you"])
        self.detecting_chip.set_count(counts["detecting"])
        activity = controller.activity()
        running = any(controller.is_detection_kind(kind) for kind, _file in activity.running)
        self.detecting_chip.set_tone("run" if running else "idle")

        ready = len(controller.startable_files())
        folder = project.folder
        self.start_button.setText(start_text(ready))
        self.start_button.setEnabled(ready > 0 and (folder.dialogue_enabled or folder.labels_enabled))
