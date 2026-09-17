"""The Run view (`tabs-hifi.html` figure 4; ui-spec §1.4, §3.10; rulings
B12, B13): the window's run mode, shown in place of the stage and inspector
while the review queue stays.

- Table "File / Phase / Progress / Result" (grid 1.7fr .9fr 1.2fr .7fr), one
  row per file of the run from `controller.run_snapshot()`: "queued" (dim),
  a running dot + the phase ("dialogue" / "labels"), "done" (ok), "failed"
  (bad, the error on hover) or "cancelled" (dim); a progress bar, blue while
  running and full when done; "{n} lines".
- Footer: "GPU {p}%" from `nvidia-smi`, run through an async QProcess every
  2 s only while this view is visible and a run is active, hidden when the
  query fails; and one hint about the files still waiting (`raise_offer`):
    - "{k} workers idle" -- text only, since the run already has every
      worker its parallel allows and there is nothing to raise to -- when it
      leaves worker slots unused while files wait (`idle_workers`, B13);
    - otherwise "{q} files queued ·" with a ghost "raise to {m}" button,
      m = parallel + 2 capped at MAX_PARALLEL, while files wait and the
      folder is below that cap. This is the case that actually comes up,
      since the run job fills a free worker at once (B13 as amended). The
      button calls `update_folder(ocr_parallel=m)`, which reaches the
      running job (RunJob.set_parallel).
- Right panel (300 px) "LIVE · {file}": the recognised lines of the followed
  file ("MM:SS text"), by default the most recently started file; "follow ▾"
  picks another. Then the note that reviewing goes on meanwhile.

Decode fps and OCR img/s are omitted (B13: they need instrumentation inside
videocr). Views import no core module.
"""
from __future__ import annotations

from PyQt6.QtCore import QObject, QPoint, QProcess, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.run_snapshot import CANCELLED, DONE, FAILED, QUEUED, RUNNING, RunFileRow, idle_workers
from app.theme import tokens
from app.views.deferred import Deferred
from app.widgets.base import Button, Dot, ElidedLabel, repolish

GPU_PROGRAM = "nvidia-smi"
GPU_ARGUMENTS = ["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
GPU_POLL_MS = 2000
GPU_SHUTDOWN_WAIT_MS = 500

HEADERS = ("File", "Phase", "Progress", "Result")
COLUMN_STRETCH = (17, 9, 12, 7)             # .rrow grid-template-columns: 1.7fr .9fr 1.2fr .7fr
ROW_GAP = 10                                # .rrow gap
PBAR_HEIGHT = 5                             # .pbar height
PBAR_RADIUS = 3                             # .pbar border-radius
LIVE_WIDTH = 300                            # the live panel's width
HEADER_LETTER_SPACING_EM = 0.06             # the table header's letter-spacing
FEED_LIMIT = 300                            # lines kept in the live feed

LIVE_TITLE = "LIVE"
NEWEST_FILE = "Newest file"
LIVE_NOTE = ("You can keep reviewing other episodes while this runs — nothing is blocked, and edits apply to "
             "files that haven't started yet.")
RAISE_TOOLTIP = ("Run up to {m} files at once. Files already running carry on; the new limit applies to files "
                 "that have not started yet.")
MAX_PARALLEL = 8                            # the folder settings sheet's upper bound for "parallel files"
PARALLEL_STEP = 2                           # how much "raise to" offers above the folder's parallel
IDLE_HINT = "{count} {noun} idle"
QUEUED_HINT = "{count} {noun} queued ·"
RAISE_TEXT = "raise to {m}"

_PROGRESS_COLOURS = {"run": tokens.BLUE, "done": tokens.ACC, "bad": tokens.BAD, "dim": tokens.DIM2}


def phase_text(phase: str) -> str:
    """videocr's phase name as the table shows it: "Extracting dialogue" →
    "dialogue"; "starting" before the first progress report."""
    phase = (phase or "").strip().lower()
    if not phase:
        return "starting"
    return phase.removeprefix("extracting ").strip() or phase


def feed_time(seconds: float) -> str:
    """"MM:SS" of a subtitle's start (minutes keep counting past 59)."""
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def lines_text(count: int) -> str:
    return f"{count} line" if count == 1 else f"{count} lines"


def raise_offer(snapshot, parallel: int) -> tuple[str, int | None] | None:
    """The footer's hint about the files still waiting: (text, the parallel
    to offer), the parallel None when the line is text only because there is
    nothing to raise to. None at all when there is nothing to say (see this
    module's docstring)."""
    if snapshot is None or snapshot.finished or snapshot.paused or snapshot.stopping:
        return None
    idle = idle_workers(snapshot, parallel)
    if idle > 0:
        return IDLE_HINT.format(count=idle, noun="worker" if idle == 1 else "workers"), None
    queued = snapshot.count(QUEUED)
    if queued > 0 and parallel < MAX_PARALLEL:
        return (QUEUED_HINT.format(count=queued, noun="file" if queued == 1 else "files"),
                min(MAX_PARALLEL, parallel + PARALLEL_STEP))
    return None


def parse_gpu_utilisation(output: str) -> int | None:
    """The busiest GPU's utilisation (0-100) from `nvidia-smi
    --query-gpu=utilization.gpu --format=csv,noheader,nounits` (one line per
    GPU); None when there is no number to read."""
    values = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(round(float(line)))
        except ValueError:
            return None
    if not values:
        return None
    return max(0, min(100, max(values)))


# --------------------------------------------------------------------------
# GPU utilisation
# --------------------------------------------------------------------------

class GpuMeter(QObject):
    """Polls `GPU_PROGRAM GPU_ARGUMENTS` every `GPU_POLL_MS` while started.
    Each query is an asynchronous QProcess, one at a time, so the GUI thread
    never waits for it. `value()` is None until a query succeeds and after
    one fails; a program that cannot be started is not tried again."""

    changed = pyqtSignal(object)            # int percent | None

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._process: QProcess | None = None
        self._value: int | None = None
        self._queries = 0
        self._unavailable = False

    def start(self) -> None:
        if self._timer.isActive() or self._unavailable:
            return
        self._timer.start(GPU_POLL_MS)
        self._poll()

    def stop(self) -> None:
        self._timer.stop()

    def is_polling(self) -> bool:
        return self._timer.isActive()

    def busy(self) -> bool:
        return self._process is not None

    def value(self) -> int | None:
        return self._value

    def queries(self) -> int:
        return self._queries

    def shutdown(self) -> None:
        """Stop polling; a query still running gets a moment to end, so no
        process is left behind when the window goes."""
        self.stop()
        process = self._process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.waitForFinished(GPU_SHUTDOWN_WAIT_MS)

    def _poll(self) -> None:
        if self._process is not None:
            return                          # the previous query has not answered yet
        process = QProcess(self)
        process.finished.connect(lambda code, status, p=process: self._on_finished(p, code, status))
        process.errorOccurred.connect(lambda error, p=process: self._on_error(p, error))
        self._process = process
        self._queries += 1
        process.start(GPU_PROGRAM, list(GPU_ARGUMENTS))

    def _on_finished(self, process: QProcess, code: int, status: QProcess.ExitStatus) -> None:
        if process is not self._process:
            return
        value = None
        if status == QProcess.ExitStatus.NormalExit and code == 0:
            value = parse_gpu_utilisation(bytes(process.readAllStandardOutput()).decode("utf-8", "replace"))
        self._release(process)
        self._set(value)

    def _on_error(self, process: QProcess, error: QProcess.ProcessError) -> None:
        if process is not self._process or error != QProcess.ProcessError.FailedToStart:
            return                          # any other error is followed by finished()
        self._unavailable = True
        self._timer.stop()
        self._release(process)
        self._set(None)

    def _release(self, process: QProcess) -> None:
        self._process = None
        process.deleteLater()

    def _set(self, value: int | None) -> None:
        if value == self._value:
            return                          # every poll otherwise repaints the footer for nothing
        self._value = value
        self.changed.emit(value)


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------

class ProgressTrack(QWidget):
    """`.pbar` / `.pbar u`: a 5 px rounded track filled to `fraction`; tone
    "run" (blue), "done" (accent), "bad" or "dim"."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(PBAR_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._fraction = 0.0
        self._tone = "dim"

    def set_value(self, fraction: float, tone: str) -> None:
        fraction = max(0.0, min(1.0, float(fraction)))
        if (fraction, tone) != (self._fraction, self._tone):
            self._fraction, self._tone = fraction, tone
            self.update()

    def fraction(self) -> float:
        return self._fraction

    def tone(self) -> str:
        return self._tone

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        path = QPainterPath()
        radius = min(float(PBAR_RADIUS), rect.height() / 2)
        path.addRoundedRect(rect, radius, radius)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(tokens.BADGE_BG))
        painter.drawPath(path)
        if self._fraction > 0:
            painter.setClipPath(path)
            painter.setBrush(QColor(_PROGRESS_COLOURS.get(self._tone, tokens.DIM2)))
            painter.drawRect(QRectF(0, 0, rect.width() * self._fraction, rect.height()))
        painter.end()


def _fill_grid(row: QWidget, cells: list[QWidget]) -> QWidget:
    """Lay `cells` out in `row` as the mockup's `.rrow` grid: widths purely
    by COLUMN_STRETCH (the cells' own hints are ignored)."""
    row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(10, 7, 10, 7)
    layout.setSpacing(ROW_GAP)
    for cell, stretch in zip(cells, COLUMN_STRETCH, strict=True):
        cell.setSizePolicy(QSizePolicy.Policy.Ignored, cell.sizePolicy().verticalPolicy())
        layout.addWidget(cell, stretch)
    return row


class RunRow(QWidget):
    """One file of the run (`.rrow`)."""

    def __init__(self, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name
        self._shown: RunFileRow | None = None
        self.setObjectName("RunRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("last", False)
        self.name_label = ElidedLabel(name)
        self.name_label.setObjectName("RunFile")

        phase = QWidget()
        phase_layout = QHBoxLayout(phase)
        phase_layout.setContentsMargins(0, 0, 0, 0)
        phase_layout.setSpacing(6)
        self.phase_dot = Dot("run")
        self.phase_label = ElidedLabel()
        self.phase_label.setObjectName("RunPhase")
        phase_layout.addWidget(self.phase_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        phase_layout.addWidget(self.phase_label, 1)

        bar = QWidget()
        bar_layout = QVBoxLayout(bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        self.progress = ProgressTrack()
        bar_layout.addWidget(self.progress, 0, Qt.AlignmentFlag.AlignVCenter)

        self.result_label = ElidedLabel()
        self.result_label.setObjectName("RunResult")
        _fill_grid(self, [self.name_label, phase, bar, self.result_label])
        self.update_from(RunFileRow(name))

    def set_last(self, last: bool) -> None:
        """The last row of the table draws no bottom border (ui-spec §3.10)."""
        if self.property("last") != last:
            self.setProperty("last", last)
            repolish(self)

    def phase_tone(self) -> str:
        return self.phase_label.property("tone")

    def result_tone(self) -> str:
        return self.result_label.property("tone")

    def update_from(self, row: RunFileRow) -> None:
        if row == self._shown:
            return
        self._shown = row
        state = row.state
        result, result_tone, tooltip = "", "", ""
        if state == RUNNING:
            text, tone, fill = phase_text(row.phase), "run", (row.progress, "run")
            result = lines_text(row.lines) if row.lines else ""
        elif state == DONE:
            text, tone, fill = "done", "ok", (1.0, "done")
            result, result_tone = lines_text(row.lines), "ok"
        elif state == FAILED:
            text, tone, fill, tooltip = "failed", "bad", (row.progress, "bad"), row.error
        elif state == CANCELLED:
            text, tone, fill = "cancelled", "dim", (row.progress, "dim")
        else:
            text, tone, fill = QUEUED, "dim", (0.0, "dim")
        self.phase_dot.setVisible(state == RUNNING)
        self.phase_label.set_full_text(text)
        if tooltip:
            self.phase_label.setToolTip(tooltip)             # a failed file's error, on hover
        self.progress.set_value(*fill)
        self.result_label.set_full_text(result)
        for label, value in ((self.phase_label, tone), (self.result_label, result_tone)):
            if label.property("tone") != value:
                label.setProperty("tone", value)
                repolish(label)


class ListScrollArea(QScrollArea):
    """A vertical, frameless scroll area over a top-aligned list of widgets
    (the file rows, the live feed); it takes the space its column has."""

    def __init__(self, object_name: str, content_name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.content = QWidget()
        self.content.setObjectName(content_name)
        self.rows = QVBoxLayout(self.content)
        self.rows.setContentsMargins(0, 0, 0, 0)
        self.rows.setSpacing(0)
        self.rows.addStretch(1)                  # rows stay at the top
        self.setWidget(self.content)

    def add_row(self, widget: QWidget) -> None:
        self.rows.insertWidget(self.rows.count() - 1, widget)

    def row_count(self) -> int:
        return self.rows.count() - 1

    def row_at(self, index: int) -> QWidget:
        return self.rows.itemAt(index).widget()

    def take_first_row(self) -> QWidget:
        return self.rows.takeAt(0).widget()

    def remove_row(self, widget: QWidget) -> None:
        self.rows.removeWidget(widget)


class FeedLine(QWidget):
    """`.feed`: "MM:SS" and the recognised text."""

    def __init__(self, start: float, text: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("FeedLine")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(8)
        self.time_label = QLabel(feed_time(start))
        self.time_label.setObjectName("FeedTime")
        self.text_label = QLabel(text.replace("\\N", "\n"))
        self.text_label.setObjectName("FeedText")
        self.text_label.setWordWrap(True)
        layout.addWidget(self.time_label, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.text_label, 1)

    def text(self) -> str:
        return f"{self.time_label.text()} {self.text_label.text()}"


# --------------------------------------------------------------------------
# The view
# --------------------------------------------------------------------------

class RunView(QWidget):
    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__(parent)
        self._controller = controller
        self._rows: dict[str, RunRow] = {}
        self._pinned: str | None = None          # the file chosen with "follow ▾"; None follows the newest
        self._followed: str | None = None
        self._raise_to: int | None = None        # what "raise to" offers; None when there is no button
        self.setObjectName("RunView")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_table(), 1)
        body.addWidget(self._build_live_panel())

        self.gpu_meter = GpuMeter(self)
        self.gpu_meter.changed.connect(lambda _value: self._refresh_footer())
        self._refresh_later = Deferred(self.refresh, self)
        for signal in (controller.run_changed, controller.folder_changed):
            signal.connect(self._refresh_later.schedule)
        controller.project_opened.connect(self._reset)
        controller.project_closed.connect(self._reset)
        controller.run_subtitle.connect(self._on_subtitle)
        self.refresh()

    def _build_table(self) -> QWidget:
        table = QWidget()
        table.setObjectName("RunTable")
        column = QVBoxLayout(table)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        header_cells = []
        for title in HEADERS:
            label = QLabel(title.upper())
            label.setObjectName("RunHeaderCell")
            font = label.font()
            font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 100 + HEADER_LETTER_SPACING_EM * 100)
            label.setFont(font)
            header_cells.append(label)
        self._header_cells = header_cells
        header = QWidget()
        header.setObjectName("RunHeader")
        column.addWidget(_fill_grid(header, header_cells))

        self._rows_scroll = ListScrollArea("RunScroll", "RunRows")
        column.addWidget(self._rows_scroll, 1)

        self.footer = QWidget()
        self.footer.setObjectName("RunFooter")
        self.footer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        footer = QHBoxLayout(self.footer)
        footer.setContentsMargins(10, 8, 10, 8)
        footer.setSpacing(6)
        self.gpu_label = QLabel()
        self.gpu_separator = QLabel("·")
        self.hint_label = QLabel()
        self.raise_button = Button(RAISE_TEXT.format(m=0), "ghost", small=True)
        self.raise_button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.raise_button.clicked.connect(self._raise_parallel)
        for widget in (self.gpu_label, self.gpu_separator, self.hint_label, self.raise_button):
            footer.addWidget(widget)
        footer.addStretch(1)
        column.addWidget(self.footer)
        return table

    def _build_live_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("LivePanel")
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setFixedWidth(LIVE_WIDTH)
        self.live_panel = panel
        column = QVBoxLayout(panel)
        column.setContentsMargins(11, 10, 10, 10)          # 10 px padding + the 1 px left border
        column.setSpacing(0)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 7)
        head.setSpacing(6)
        self.live_title = ElidedLabel(LIVE_TITLE)
        self.live_title.setObjectName("LiveTitle")
        font = self.live_title.font()
        font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 100 + tokens.LETTER_SPACING_SCOPE_EM * 100)
        self.live_title.setFont(font)
        self.follow_button = Button("follow ▾", "ghost", small=True)
        self.follow_button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.follow_button.clicked.connect(self._show_follow_menu)
        head.addWidget(self.live_title, 1)
        head.addWidget(self.follow_button)
        column.addLayout(head)

        self._feed_scroll = ListScrollArea("FeedScroll", "FeedLines")
        self._feed_scroll.verticalScrollBar().rangeChanged.connect(self._follow_bottom)
        self._stick_to_bottom = True
        self._feed_scroll.verticalScrollBar().valueChanged.connect(self._on_feed_scrolled)
        column.addWidget(self._feed_scroll, 1)

        self.note_label = QLabel(LIVE_NOTE)
        self.note_label.setObjectName("Note")
        self.note_label.setWordWrap(True)
        self.note_label.setContentsMargins(0, 8, 0, 0)
        column.addWidget(self.note_label)
        return panel

    # --- reading ------------------------------------------------------------------------

    def row(self, name: str) -> RunRow:
        return self._rows[name]

    def names(self) -> list[str]:
        return list(self._rows)

    def header_texts(self) -> list[str]:
        return [label.text() for label in self._header_cells]

    def followed(self) -> str | None:
        return self._followed

    def feed_lines(self) -> list[str]:
        return [line.text() for line in self._feed_lines()]

    def hint_text(self) -> str:
        """The footer's hint as one line ("3 files queued · raise to 6", or
        "2 workers idle" when there is no button); "" when there is none."""
        if self.hint_label.isHidden():
            return ""
        if self.raise_button.isHidden():
            return self.hint_label.text()
        return f"{self.hint_label.text()} {self.raise_button.text()}"

    # --- refreshing ---------------------------------------------------------------------

    def refresh(self, *_args) -> None:
        self._refresh_later.cancel()
        snapshot = self._controller.run_snapshot()
        names = [] if snapshot is None else [row.name for row in snapshot.files]
        if names != list(self._rows):
            self._rebuild_rows(names)
        if snapshot is not None:
            for row in snapshot.files:
                self._rows[row.name].update_from(row)
        self._sync_follow(snapshot)
        self._sync_gpu(snapshot)
        self._refresh_footer(snapshot)

    def _rebuild_rows(self, names: list[str]) -> None:
        for row in self._rows.values():
            self._rows_scroll.remove_row(row)
            row.deleteLater()
        self._rows = {}
        for index, name in enumerate(names):
            row = RunRow(name, self._rows_scroll.content)
            row.set_last(index == len(names) - 1)        # ui-spec §3.10: the last row has no bottom border
            self._rows[name] = row
            self._rows_scroll.add_row(row)

    def _reset(self, *_args) -> None:
        self._pinned = None
        self._followed = None
        self._clear_feed()
        self.refresh()

    def _run_active(self, snapshot=None) -> bool:
        snapshot = self._controller.run_snapshot() if snapshot is None else snapshot
        return snapshot is not None and not snapshot.finished

    def _refresh_footer(self, snapshot=None) -> None:
        controller = self._controller
        snapshot = controller.run_snapshot() if snapshot is None else snapshot
        active = self._run_active(snapshot)
        gpu = self.gpu_meter.value() if active and self.isVisible() else None
        self.gpu_label.setVisible(gpu is not None)
        if gpu is not None:
            self.gpu_label.setText(f"GPU {gpu}%")
        project = controller.project
        offer = raise_offer(snapshot, project.folder.ocr_parallel) if project is not None else None
        hint, self._raise_to = offer if offer is not None else ("", None)
        self.hint_label.setVisible(offer is not None)
        self.raise_button.setVisible(self._raise_to is not None)
        if offer is not None:
            self.hint_label.setText(hint)
        if self._raise_to is not None:
            self.raise_button.setText(RAISE_TEXT.format(m=self._raise_to))
            self.raise_button.setToolTip(RAISE_TOOLTIP.format(m=self._raise_to))
        self.gpu_separator.setVisible(gpu is not None and offer is not None)
        self.footer.setVisible(gpu is not None or offer is not None)

    def _raise_parallel(self) -> None:
        if self._raise_to is not None and self._controller.project is not None:
            self._controller.update_folder(ocr_parallel=self._raise_to)

    # --- GPU ------------------------------------------------------------------------------

    def _sync_gpu(self, snapshot=None) -> None:
        if self.isVisible() and self._run_active(snapshot):
            self.gpu_meter.start()
        else:
            self.gpu_meter.stop()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._sync_gpu()
        self._refresh_footer()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._sync_gpu()

    def shutdown(self) -> None:
        self.gpu_meter.shutdown()

    # --- the live feed --------------------------------------------------------------------

    def follow(self, name: str | None) -> None:
        """Follow `name`; None follows the most recently started file."""
        self._pinned = name
        self._sync_follow(self._controller.run_snapshot())

    def follow_menu(self) -> QMenu:
        """"Newest file", then each file of the run that has started."""
        menu = QMenu(self)
        newest = menu.addAction(NEWEST_FILE)
        newest.setCheckable(True)
        newest.setChecked(self._pinned is None)
        newest.triggered.connect(lambda _checked=False: self.follow(None))
        snapshot = self._controller.run_snapshot()
        started = [] if snapshot is None else [row.name for row in snapshot.files if row.state != QUEUED]
        if started:
            menu.addSeparator()
        for name in started:
            action = menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(self._pinned == name)
            action.triggered.connect(lambda _checked=False, n=name: self.follow(n))
        return menu

    def _show_follow_menu(self) -> None:
        menu = self.follow_menu()
        menu.exec(self.follow_button.mapToGlobal(QPoint(0, self.follow_button.height())))
        menu.deleteLater()

    def _sync_follow(self, snapshot) -> None:
        target = None
        if snapshot is not None:
            names = {row.name for row in snapshot.files}
            if self._pinned is not None and self._pinned not in names:
                self._pinned = None
            if self._pinned is not None:
                target = self._pinned
            else:
                started = [(row.started_at, index, row.name) for index, row in enumerate(snapshot.files)
                           if row.started_at is not None]
                target = max(started)[2] if started else None
        else:
            self._pinned = None
        self.live_title.set_full_text(f"{LIVE_TITLE} · {target}" if target else LIVE_TITLE)
        if target == self._followed:
            return
        self._followed = target
        self._clear_feed()
        if target is not None:
            self._stick_to_bottom = True
            for start, _end, text in self._controller.run_subtitles(target)[-FEED_LIMIT:]:
                self._add_line(start, text)

    def _on_subtitle(self, name: str, start: float, _end: float, text: str) -> None:
        if name == self._followed:
            self._add_line(start, text)

    def _feed_lines(self) -> list[FeedLine]:
        return [self._feed_scroll.row_at(index) for index in range(self._feed_scroll.row_count())]

    def _add_line(self, start: float, text: str) -> None:
        self._feed_scroll.add_row(FeedLine(start, text, self._feed_scroll.content))
        while self._feed_scroll.row_count() > FEED_LIMIT:
            self._feed_scroll.take_first_row().deleteLater()

    def _clear_feed(self) -> None:
        while self._feed_scroll.row_count():
            self._feed_scroll.take_first_row().deleteLater()

    def _on_feed_scrolled(self, value: int) -> None:
        bar = self._feed_scroll.verticalScrollBar()
        self._stick_to_bottom = value >= bar.maximum()

    def _follow_bottom(self, _minimum: int, maximum: int) -> None:
        if self._stick_to_bottom:
            self._feed_scroll.verticalScrollBar().setValue(maximum)
