"""The Done screen (episode-view.html, screen 5): the episode's subtitles
are written, and here is what they say.

"Subtitles unburned": N lines in T, saved as the output's label next to the
video (`controller.output_label`: "zh/EP06.zh.ass", or "EP06.zh.ass" with
the subfolder off), then Show in folder · Open the .ass · Open another episode…, and
"Review the settings used ▸" back to the tabs. The Script panel is loaded
from the written file (`controller.output_lines`), not from what the run
reported along the way, so the user proofreads the exact output: QA-fixed,
deduplicated, as a player will show it. Clicking a line shows its frame in
the preview under the summary.

A run that failed or was stopped shows the same screen with what happened
and "Try again". Nothing was deleted: an earlier output is still there, and
the screen says so.

Opening a folder or the .ass goes through `open_url`, QDesktopServices by
default, injectable so tests open nothing.
"""
from __future__ import annotations

import os
from collections.abc import Callable

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from app.episode_feed import Line, duration_words, format_ts, mid_time
from app.imaging import bgr_to_qimage
from app.run_snapshot import CANCELLED, DONE, FAILED
from app.theme import tokens
from app.views.deferred import Deferred
from app.views.episode.common import (
    BODY_SIZE,
    HEADLINE_SIZE,
    SMALL_SIZE,
    text_label,
)
from app.views.episode.script import ScriptPanel
from app.views.episode.slideshow import FramePreview
from app.views.episode.working import SCRIPT_MIN_WIDTH, SCRIPT_STRETCH, STAGE_STRETCH
from app.widgets.base import Button

TITLE_DONE = "Subtitles unburned"
TITLE_FAILED = "OCR failed"
TITLE_STOPPED = "OCR stopped"
SUMMARY = "{lines} in {elapsed} · saved as {output} next to the video"
SUMMARY_NO_TIME = "{lines} · saved as {output} next to the video"
STOPPED_TEXT = "The run was stopped before it finished."
KEPT_TEXT = "The earlier {output} is unchanged."
SHOW_IN_FOLDER = "Show in folder"
OPEN_ASS = "Open the .ass"
OPEN_ANOTHER = "Open another episode…"
TRY_AGAIN = "Try again"
REVIEW_LINK = "Review the settings used ▸"

OUTCOME_DONE, OUTCOME_FAILED, OUTCOME_STOPPED = "done", "failed", "stopped"
PREVIEW_RESERVED_HEIGHT = 330      # mockup px of summary and buttons above the preview


def lines_words(count: int) -> str:
    return f"{count} line" if count == 1 else f"{count} lines"


class DoneView(QWidget):
    review_requested = pyqtSignal()
    open_another_requested = pyqtSignal()
    retry_requested = pyqtSignal()

    def __init__(self, controller, parent: QWidget | None = None, *,
                 open_url: Callable[[QUrl], object] | None = None, animated: bool = True):
        super().__init__(parent)
        self._controller = controller
        self._open_url = open_url or QDesktopServices.openUrl
        self._name: str | None = None
        self._outcome = OUTCOME_DONE
        self._preview_line: Line | None = None
        self._loaded_from: tuple | None = None
        self.setObjectName("EpisodeDone")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        summary = QWidget()
        column = QVBoxLayout(summary)
        column.setContentsMargins(tokens.px(24), tokens.px(24), tokens.px(24), tokens.px(24))
        column.setSpacing(tokens.px(8))
        column.addStretch(1)
        self.icon = text_label("✓", color=tokens.OK, size=34, center=True)
        self.title = text_label(TITLE_DONE, size=HEADLINE_SIZE, weight=600, center=True, wrap=True)
        self.summary = text_label(color=tokens.DIM, size=BODY_SIZE, center=True, wrap=True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.detail = text_label(color=tokens.DIM2, size=SMALL_SIZE, center=True, wrap=True)
        for label in (self.icon, self.title, self.summary, self.detail):
            column.addWidget(label)

        buttons = QHBoxLayout()
        buttons.setSpacing(tokens.px(6))
        buttons.addStretch(1)
        self.retry_button = Button(TRY_AGAIN, "primary")
        self.folder_button = Button(SHOW_IN_FOLDER)
        self.ass_button = Button(OPEN_ASS)
        self.another_button = Button(OPEN_ANOTHER, "primary")
        self.retry_button.clicked.connect(self.retry_requested)
        self.folder_button.clicked.connect(self.show_in_folder)
        self.ass_button.clicked.connect(self.open_output)
        self.another_button.clicked.connect(self.open_another_requested)
        for button in (self.retry_button, self.folder_button, self.ass_button, self.another_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        column.addSpacing(tokens.px(4))
        column.addLayout(buttons)
        self.review_link = Button(REVIEW_LINK, "ghost", small=True)
        self.review_link.setStyleSheet(f"QPushButton {{ border: none; color: {tokens.DIM2}; background: transparent; }}"
                                       f"QPushButton:hover {{ color: {tokens.TXT}; }}")
        self.review_link.clicked.connect(self.review_requested)
        column.addWidget(self.review_link, 0, Qt.AlignmentFlag.AlignHCenter)

        column.addSpacing(tokens.px(12))
        self.preview = FramePreview(width=420)
        self.preview_caption = text_label(color=tokens.DIM, size=SMALL_SIZE, center=True, wrap=True)
        column.addWidget(self.preview, 0, Qt.AlignmentFlag.AlignHCenter)
        column.addWidget(self.preview_caption)
        self.preview.setVisible(False)
        self.preview_caption.setVisible(False)
        column.addStretch(1)

        self.script = ScriptPanel(animated=animated, live=False)
        self.script.setMinimumWidth(tokens.px(SCRIPT_MIN_WIDTH))
        self.script.line_clicked.connect(self._on_line_clicked)
        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(summary, STAGE_STRETCH)
        body.addWidget(self.script, SCRIPT_STRETCH)

        self._refresh_later = Deferred(self.refresh, self)
        controller.run_changed.connect(self._refresh_later.schedule)
        controller.file_changed.connect(self._on_file_changed)
        controller.frame_ready.connect(self._on_frame_ready)

    # --- reading (window, tests) --------------------------------------------------------

    def file(self) -> str | None:
        return self._name

    def outcome(self) -> str:
        return self._outcome

    def preview_line(self) -> Line | None:
        return self._preview_line

    # --- refreshing ---------------------------------------------------------------------

    def set_file(self, name: str | None) -> None:
        if name != self._name:
            self._name = name
            self._loaded_from = None
            self._set_preview(None)
        self.refresh()

    def reload(self) -> None:
        """Read the output file again (the window calls it when a run ends)."""
        self._loaded_from = None
        self.refresh()

    def refresh(self, *_args) -> None:
        self._refresh_later.cancel()
        name = self._name
        if name is None:
            self.script.set_lines([])
            return
        snapshot = self._controller.run_snapshot()
        row = None if snapshot is None else snapshot.row(name)
        if row is not None and row.state == FAILED:
            self._outcome = OUTCOME_FAILED
        elif row is not None and row.state == CANCELLED:
            self._outcome = OUTCOME_STOPPED
        else:
            self._outcome = OUTCOME_DONE

        output = self._controller.output_label(name)
        key = (name, snapshot.finished_at if snapshot is not None else None, self._controller.is_done(name))
        if key != self._loaded_from:
            self._loaded_from = key
            self.script.set_lines(self._controller.output_lines(name))
        count = self.script.count()
        ok = self._outcome == OUTCOME_DONE
        self.icon.setText("✓" if ok else "!" if self._outcome == OUTCOME_FAILED else "■")
        colour = tokens.OK if ok else tokens.BAD if self._outcome == OUTCOME_FAILED else tokens.DIM
        self.icon.setStyleSheet(f"color: {colour}; background: transparent; font-size: {round(tokens.pt(34))}px;")
        self.title.setText({OUTCOME_DONE: TITLE_DONE, OUTCOME_FAILED: TITLE_FAILED,
                            OUTCOME_STOPPED: TITLE_STOPPED}[self._outcome])
        if ok:
            lines = row.lines if row is not None and row.state == DONE and row.lines else count
            elapsed = None
            if row is not None and row.started_at is not None and row.finished_at is not None:
                elapsed = row.finished_at - row.started_at
            bold = f"<b style='color:{tokens.TXT}'>{_escape(output)}</b>"
            template = SUMMARY if elapsed is not None else SUMMARY_NO_TIME
            self.summary.setText(template.format(lines=lines_words(lines), elapsed=duration_words(elapsed or 0),
                                                 output=bold))
            self.detail.setText("")
        else:
            error = (row.error if row is not None else "") or (snapshot.error if snapshot is not None else "")
            self.summary.setText(_escape(error) if error else STOPPED_TEXT)
            self.detail.setText(KEPT_TEXT.format(output=output) if self._controller.is_done(name) else "")
        self.detail.setVisible(bool(self.detail.text()))
        has_output = self._controller.is_done(name)
        self.retry_button.setVisible(not ok)
        self.another_button.set_variant("primary" if ok else "default")
        self.ass_button.setEnabled(has_output)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        primary = self.retry_button if not self.retry_button.isHidden() else self.another_button
        primary.setFocus(Qt.FocusReason.OtherFocusReason)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        room = self.height() - tokens.px(PREVIEW_RESERVED_HEIGHT)
        self.preview.fit(min(self.width() // 3, int(room * FramePreview.ASPECT)))

    def _on_file_changed(self, name: str) -> None:
        if name == self._name:
            self._refresh_later.schedule()

    # --- actions --------------------------------------------------------------------------

    def show_in_folder(self) -> None:
        path = self._controller.output_path(self._name) if self._name else ""
        if path:
            directory = os.path.dirname(path)
            if not os.path.isdir(directory):        # an output subfolder no run has made yet: the video's folder
                directory = os.path.dirname(directory)
            self._open_url(QUrl.fromLocalFile(directory))

    def open_output(self) -> None:
        path = self._controller.output_path(self._name) if self._name else ""
        if path and os.path.exists(path):
            self._open_url(QUrl.fromLocalFile(path))

    # --- the preview ----------------------------------------------------------------------

    def _on_line_clicked(self, start: float, end: float, text: str) -> None:
        self._set_preview(Line(start, end, text))

    def _set_preview(self, line: Line | None) -> None:
        self._preview_line = line
        self.preview.setVisible(line is not None)
        self.preview_caption.setVisible(line is not None)
        self.preview.set_image(None)
        if line is None or self._name is None:
            return
        self.preview_caption.setText(f"{format_ts(line.start)}   {line.text.replace(chr(92) + 'N', ' / ')}")
        entry = None
        try:
            entry = self._controller.entry(self._name)
        except KeyError:
            pass
        if entry is not None:
            crop = entry.crop
            box = None if crop is None else (crop.x, crop.y, crop.width, crop.height)
            self.preview.set_box(box, (entry.media.width, entry.media.height), "ok")
        frame = self._controller.frame(self._name, mid_time(line))
        if frame is not None:
            self.preview.set_image(bgr_to_qimage(frame))
        else:
            self._controller.request_frames(self._name, [mid_time(line)])

    def _on_frame_ready(self, name: str, time_value: float) -> None:
        line = self._preview_line
        if name != self._name or line is None or abs(mid_time(line) - time_value) > 1e-3:
            return
        frame = self._controller.frame(name, mid_time(line))
        if frame is not None:
            self.preview.set_image(bgr_to_qimage(frame))


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
