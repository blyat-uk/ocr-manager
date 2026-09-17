"""The activity strip (`.activity`, ui-spec §3.8): what the background is
doing, instead of modal progress dialogs.

- The current job: a blue dot, "{file} · {message}", a mini progress bar and
  its percent; "idle" with a grey dot when nothing runs. Detection jobs report
  no progress fraction, so until a progress event arrives the bar is
  indeterminate (animated) and no percent is shown.
- Trailing dim text: the latest finished jobs, "· {kind} done {n} s ago", up
  to two, one per kind.
- "pause auto-pilot" / "resume auto-pilot".
"""
from __future__ import annotations

import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from app.widgets.base import Button, Dot, MiniProgress

RECENT_SHOWN = 2
AGO_REFRESH_MS = 1000
# What finished, as a noun: "audio analysis done 11 s ago".
DONE_LABELS = {
    "metadata": "metadata",
    "thumbnail": "thumbnail",
    "crop": "subtitle search",
    "brightness": "brightness",
    "ranges": "intro/outro matching",
    "audio_profile": "audio analysis",
    "proof": "test OCR",
}


def ago_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds} s ago" if seconds < 60 else f"{seconds // 60} min ago"


def recent_text(recent, now: float) -> str:
    """"· audio analysis done 11 s ago · metadata done 12 s ago": the newest
    finished job of each kind, newest first."""
    parts, kinds = [], set()
    for kind, _file, finished_at in recent:
        if kind in kinds:
            continue
        kinds.add(kind)
        parts.append(f"· {DONE_LABELS.get(kind, kind)} done {ago_text(now - finished_at)}")
        if len(parts) == RECENT_SHOWN:
            break
    return " ".join(parts)


class ActivityStrip(QWidget):
    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__(parent)
        self._controller = controller
        self.setObjectName("ActivityStrip")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 6, 14, 6)
        layout.setSpacing(10)
        self.dot = Dot("idle")
        self.text_label = QLabel("idle")
        self.progress = MiniProgress()
        self.percent_label = QLabel()
        self.recent_label = QLabel()
        self.recent_label.setObjectName("ActivityRecent")
        self.pause_button = Button("pause auto-pilot", "ghost", small=True)
        self.pause_button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.pause_button.clicked.connect(self._toggle_pause)
        layout.addWidget(self.dot)
        layout.addWidget(self.text_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.percent_label)
        layout.addWidget(self.recent_label)
        layout.addStretch(1)
        layout.addWidget(self.pause_button)

        self._ago_timer = QTimer(self)
        self._ago_timer.setInterval(AGO_REFRESH_MS)
        self._ago_timer.timeout.connect(self.refresh)
        for signal in (controller.activity_changed, controller.project_opened, controller.project_closed):
            signal.connect(self.refresh)
        self.refresh()

    def refresh(self, *_args) -> None:
        controller = self._controller
        snapshot = controller.activity()
        current = snapshot.current
        if current is None:
            self.dot.set_tone("idle")
            self.text_label.setText("idle")
            self.progress.hide()
            self.percent_label.hide()
        else:
            kind, file, fraction, message = current
            self.dot.set_tone("run")
            self.text_label.setText(f"{file} · {message}" if file else message)
            self.progress.show()
            if fraction is None:
                self.progress.set_indeterminate(True)
                self.percent_label.hide()
            else:
                self.progress.set_value(fraction)
                self.percent_label.setText(f"{round(fraction * 100)}%")
                self.percent_label.show()
        text = recent_text(snapshot.recent, time.monotonic())
        self.recent_label.setText(text)
        self.recent_label.setVisible(bool(text))
        if text:
            if not self._ago_timer.isActive():
                self._ago_timer.start()
        else:
            self._ago_timer.stop()
        self.pause_button.setVisible(controller.project is not None)
        self.pause_button.setText("resume auto-pilot" if snapshot.paused else "pause auto-pilot")

    def _toggle_pause(self) -> None:
        if self._controller.activity().paused:
            self._controller.resume_autopilot()
        else:
            self._controller.pause_autopilot()
        self.refresh()
