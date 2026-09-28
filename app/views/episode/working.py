"""The Working screen (episode-view-v2.html, screen 4): the run of one
episode, shown as it happens.

Left, the screenshot (`SlideShow`) with the HUD over its foot: what the run
is doing, the ETA, a progress bar with a green tick where each found line
sits, "position of duration" and the speed ("4.1× real time"). Right, the
Script panel.

The drip. `run_subtitle` arrives in the batches videocr emits. The lines of
one event-loop turn are gathered by a zero-delay timer into one batch for
`app.episode_feed.LineFeed`, and every line of it has its screenshot asked
for in one `request_frames` call, at the line's middle (frames already held
count as ready at once: `request_frames` fetches nothing for them, so no
`frame_ready` would follow). A POLL_MS timer asks the feed what is due and
shows it: the line joins the script, and the slideshow fades to its frame.
While the user reads an older line (scrolled up, or clicked one), the
slideshow holds that line's frame and the script stops following; the
"following" pill brings both back. When the run finishes, whatever is still
queued is flushed into the script at once.

What is honest and what is paced. Only the script and the slideshow are
paced; the line count, the progress bar and its ticks show every line the
run has reported, the moment it reports it.

Position and ETA. Position is `progress × kept duration` (the file minus its
skipped ranges; with keep ranges the ticks are placed along the kept time,
in order). The ETA is `EtaEstimator` over the file's `run_file_progress`,
blended with the speed remembered from the last run. videocr counts
dialogue, then labels, each from 0 to 100 %; the second phase restarts the
bar and the estimate, and the HUD says which phase it is reading.

The clock is injectable, and `tick()` is public, so tests drive the drip
without real time passing.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from PyQt6.QtCore import QRectF, Qt, QTimer, QVariantAnimation
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget

from app.episode_feed import (
    EtaEstimator,
    Line,
    LineFeed,
    eta_words,
    kept_position,
    kept_total,
    speed_words,
)
from app.imaging import bgr_to_qimage
from app.run_snapshot import CANCELLED, DONE, FAILED, QUEUED
from app.state_text import format_duration
from app.theme import tokens
from app.time_spans import read_ranges
from app.views.deferred import Deferred
from app.views.episode.common import SMALL_SIZE, TITLE_SIZE, font
from app.views.episode.script import ScriptPanel
from app.views.episode.slideshow import SlideShow

POLL_MS = 100
SHIMMER_MS = 2200
HUD_HEIGHT = 120                # mockup px (scaled): tall enough to shade the burned-in line under it
SCRIPT_MIN_WIDTH = 300
STAGE_STRETCH, SCRIPT_STRETCH = 17, 10     # .work flex 1.7 / .script flex 1

TITLE_STARTING = "Starting…"
TITLE_DIALOGUE = "Reading subtitles…"
TITLE_LABELS = "Reading labels…"
TITLE_PAUSED = "Paused"
TITLE_STOPPING = "Stopping…"
TITLE_FINISHED = "Finished"
POSITION = "{position} of {duration}"


def hud_title(snapshot, row) -> str:
    if snapshot is None or row is None or row.state == QUEUED:
        return TITLE_STARTING
    if row.state in (DONE, FAILED, CANCELLED) or snapshot.finished:
        return TITLE_FINISHED
    if snapshot.stopping:
        return TITLE_STOPPING
    if snapshot.paused:
        return TITLE_PAUSED
    if "label" in (row.phase or "").lower():
        return TITLE_LABELS
    if not row.phase:
        return TITLE_STARTING
    return TITLE_DIALOGUE


class Hud(QWidget):
    """The painted overlay at the foot of the screenshot."""

    def __init__(self, parent: QWidget | None = None, *, animated: bool = True):
        super().__init__(parent)
        self.setObjectName("EpisodeHud")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.title = TITLE_STARTING
        self.eta = ""
        self.position = ""
        self.speed = ""
        self.progress = 0.0
        self.ticks: list[float] = []
        self._shimmer = 0.0
        self._animated = animated
        self._shimmer_animation = QVariantAnimation(self)
        self._shimmer_animation.setStartValue(0.0)
        self._shimmer_animation.setEndValue(1.0)
        self._shimmer_animation.setDuration(SHIMMER_MS)
        self._shimmer_animation.setLoopCount(-1)
        self._shimmer_animation.valueChanged.connect(self._set_shimmer)

    def set_state(self, *, title: str, eta: str, position: str, speed: str, progress: float,
                  ticks: list[float], shimmer: bool) -> None:
        self.title, self.eta, self.position, self.speed = title, eta, position, speed
        self.progress = max(0.0, min(1.0, progress))
        self.ticks = ticks
        running = shimmer and self._animated and self.isVisible()
        if running and self._shimmer_animation.state() != QVariantAnimation.State.Running:
            self._shimmer_animation.start()
        elif not running:
            self._shimmer_animation.stop()
            self._shimmer = 0.0
        self.update()

    def _set_shimmer(self, value) -> None:
        self._shimmer = float(value)
        self.update()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._shimmer_animation.stop()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        shade = QLinearGradient(0, rect.top(), 0, rect.bottom())    # linear-gradient(0deg, rgba(7,9,12,.95) 45%, transparent)
        shade.setColorAt(0.0, QColor(7, 9, 12, 0))
        shade.setColorAt(0.45, QColor(7, 9, 12, 242))
        shade.setColorAt(1.0, QColor(7, 9, 12, 242))
        painter.fillRect(rect, shade)

        pad_x, pad_bottom = tokens.px(14), tokens.px(10)
        inner = rect.adjusted(pad_x, 0, -pad_x, -pad_bottom)
        small = font(SMALL_SIZE)
        painter.setFont(small)
        small_height = painter.fontMetrics().height()
        bottom_row = QRectF(inner.left(), inner.bottom() - small_height, inner.width(), small_height)
        painter.setPen(QColor(tokens.DIM2))
        painter.drawText(bottom_row, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), self.position)
        painter.drawText(bottom_row, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), self.speed)

        track_height = tokens.px(5)
        track = QRectF(inner.left(), bottom_row.top() - tokens.px(6) - track_height, inner.width(), track_height)
        self._paint_track(painter, track)

        title_font = font(TITLE_SIZE, 600)
        painter.setFont(title_font)
        title_height = painter.fontMetrics().height()
        top_row = QRectF(inner.left(), track.top() - tokens.px(6) - title_height, inner.width(), title_height)
        painter.setPen(QColor(tokens.TXT))
        painter.drawText(top_row, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), self.title)
        if self.eta:
            painter.setFont(small)
            left_word = " left"
            left_width = painter.fontMetrics().horizontalAdvance(left_word)
            painter.setPen(QColor(tokens.DIM2))
            painter.drawText(top_row, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), left_word)
            painter.setFont(font(TITLE_SIZE, 600))
            painter.setPen(QColor(tokens.ACC))
            painter.drawText(top_row.adjusted(0, 0, -left_width, 0),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), self.eta)
        painter.end()

    def _paint_track(self, painter: QPainter, track: QRectF) -> None:
        radius = track.height() / 2
        path = QPainterPath()
        path.addRoundedRect(track, radius, radius)
        painter.fillPath(path, QColor(255, 255, 255, 31))                 # rgba(255,255,255,.12)
        painter.save()
        painter.setClipPath(path)
        fill_width = track.width() * self.progress
        if fill_width > 0:
            fill = QLinearGradient(track.left(), 0, track.left() + fill_width, 0)
            fill.setColorAt(0.0, QColor(tokens.ACC))
            fill.setColorAt(1.0, QColor("#ffd784"))
            painter.fillRect(QRectF(track.left(), track.top(), fill_width, track.height()), fill)
            if self._shimmer > 0:
                band = track.width() * 0.18
                left = track.left() - band + (fill_width + band) * self._shimmer
                glow = QLinearGradient(left, 0, left + band, 0)
                glow.setColorAt(0.0, QColor(255, 255, 255, 0))
                glow.setColorAt(0.5, QColor(255, 255, 255, 115))
                glow.setColorAt(1.0, QColor(255, 255, 255, 0))
                painter.fillRect(QRectF(left, track.top(), band, track.height()), glow)
        tick = QColor(tokens.OK)
        tick.setAlphaF(0.85)
        tick_width = max(1.0, 1.5 * tokens.UI_SCALE)
        for fraction in self.ticks:
            x = track.left() + track.width() * fraction
            painter.fillRect(QRectF(x - tick_width / 2, track.top(), tick_width, track.height()), tick)
        painter.restore()


class _Stage(QWidget):
    """The slideshow with the HUD laid over its foot."""

    def __init__(self, slideshow: SlideShow, hud: Hud, parent: QWidget | None = None):
        super().__init__(parent)
        self.slideshow, self.hud = slideshow, hud
        slideshow.setParent(self)
        hud.setParent(self)
        hud.raise_()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.slideshow.setGeometry(self.rect())
        height = tokens.px(HUD_HEIGHT)
        self.hud.setGeometry(0, max(0, self.height() - height), self.width(), height)


class WorkingView(QWidget):
    def __init__(self, controller, parent: QWidget | None = None, *, clock: Callable[[], float] = time.monotonic,
                 poll_ms: int = POLL_MS, animated: bool = True):
        super().__init__(parent)
        self._controller = controller
        self._clock = clock
        self._name: str | None = None
        self.feed = LineFeed()
        self._eta = EtaEstimator()
        self._received: list[Line] = []
        self._incoming: list[Line] = []
        self._shown: Line | None = None
        self._held: Line | None = None              # the line the user clicked, while not following
        self._flushed = False
        self.setObjectName("EpisodeWorking")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.slideshow = SlideShow(animated=animated)
        self.hud = Hud(animated=animated)
        self.stage = _Stage(self.slideshow, self.hud)
        self.script = ScriptPanel(animated=animated)
        self.script.setMinimumWidth(tokens.px(SCRIPT_MIN_WIDTH))
        self.script.line_clicked.connect(self._on_line_clicked)
        self.script.following_changed.connect(self._on_following_changed)
        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self.stage, STAGE_STRETCH)
        body.addWidget(self.script, SCRIPT_STRETCH)

        self._batcher = Deferred(self._take_batch, self)
        self._run_later = Deferred(self._on_run_changed, self)
        self._poll = QTimer(self)
        self._poll.setInterval(poll_ms)
        self._poll.timeout.connect(self.tick)
        controller.run_subtitle.connect(self._on_subtitle)
        controller.frame_ready.connect(self._on_frame_ready)
        controller.run_changed.connect(self._run_later.schedule)

    # --- reading (window, tests) --------------------------------------------------------

    def file(self) -> str | None:
        return self._name

    def shown_line(self) -> Line | None:
        """The line the slideshow is showing."""
        return self._shown

    def received(self) -> list[Line]:
        return list(self._received)

    def hud_texts(self) -> dict[str, str]:
        return {"title": self.hud.title, "eta": self.hud.eta, "position": self.hud.position, "speed": self.hud.speed}

    # --- the file and the run -----------------------------------------------------------

    def set_file(self, name: str | None) -> None:
        if name == self._name:
            return
        self._name = name
        self.reset()

    def reset(self) -> None:
        """Clear everything for a new run of the file (the window calls it
        on Start). Lines the current run already reported -- the view came
        up after they did -- go straight into the script."""
        self._batcher.cancel()
        self.feed.reset()
        self._incoming = []
        self._received = []
        self._shown = self._held = None
        self._flushed = False
        self.slideshow.clear()
        self.script.clear()
        name = self._name
        kept = self._kept_duration()
        self._eta = EtaEstimator(self._controller.episode_speed_hint() if name else None, kept)
        if name is not None:
            for start, end, text in self._controller.run_subtitles(name):
                line = Line(float(start), float(end), str(text))
                self._received.append(line)
                self.script.append(line)
        self._refresh_hud()
        self._sync_poll()

    def _kept_duration(self) -> float:
        if self._name is None:
            return 0.0
        try:
            return float(self._controller.episode_kept_duration(self._name) or 0.0)
        except KeyError:
            return 0.0

    def _spans(self) -> list[tuple[float, float]]:
        """The kept spans in video seconds: the keep ranges, or the whole file."""
        try:
            entry = self._controller.entry(self._name)
        except (KeyError, AttributeError, TypeError):
            return []
        duration = entry.media.duration
        keeps, _unreadable = read_ranges(entry, duration)
        return keeps or ([(0.0, duration)] if duration > 0 else [])

    # --- the drip -----------------------------------------------------------------------

    def _on_subtitle(self, name: str, start: float, end: float, text: str) -> None:
        if name != self._name:
            return
        line = Line(float(start), float(end), str(text))
        self._received.append(line)
        self._incoming.append(line)
        self._batcher.schedule()
        self.script.set_total(len(self._received))

    def _take_batch(self) -> None:
        batch, self._incoming = self._incoming, []
        if not batch or self._name is None:
            return
        self.feed.add_batch(batch, self._clock())
        times = [line.mid for line in batch]
        self._controller.request_frames(self._name, times)
        for time_value in times:
            if self._controller.frame(self._name, time_value) is not None:
                self.feed.frame_ready(time_value)
        self._refresh_hud()
        self._sync_poll()

    def _on_frame_ready(self, name: str, time_value: float) -> None:
        if name != self._name:
            return
        self.feed.frame_ready(time_value)
        if self._held is not None and abs(self._held.mid - time_value) < 1e-3:
            self._show(self._held, hold=0.0)

    def tick(self) -> None:
        """Release what is due (the poll timer calls this every POLL_MS)."""
        if self._batcher.pending():
            self._take_batch()
        for line in self.feed.due(self._clock()):
            self._release(line)
        self._refresh_hud(eta_only=True)
        self._sync_poll()

    def flush(self) -> None:
        """The run ended: everything still queued joins the script now."""
        if self._batcher.pending():
            self._batcher.cancel()
            batch, self._incoming = self._incoming, []
            self.feed.add_batch(batch, self._clock())
        lines = self.feed.flush()
        for line in lines:
            self.script.append(line)
        if lines and self._held is None:
            self._show(lines[-1], hold=0.0)
        self._flushed = True
        self._sync_poll()

    def _release(self, line: Line) -> None:
        self.script.append(line)
        if self._held is None:
            self._show(line, hold=self.feed.interval)

    def _show(self, line: Line, hold: float) -> None:
        frame = self._controller.frame(self._name, line.mid) if self._name else None
        image = bgr_to_qimage(frame) if frame is not None else None
        self.slideshow.show_line(image, line.text, format_duration(line.start), hold)
        self._shown = line

    def _on_line_clicked(self, start: float, end: float, text: str) -> None:
        line = Line(start, end, text)
        self._held = line
        if self._name is None:
            return
        if self._controller.frame(self._name, line.mid) is not None:
            self._show(line, hold=0.0)
        else:
            self._controller.request_frames(self._name, [line.mid])
            self.slideshow.show_line(None, line.text, format_duration(line.start), 0.0)
            self._shown = line

    def _on_following_changed(self, following: bool) -> None:
        if following:
            self._held = None

    def _sync_poll(self) -> None:
        busy = self._name is not None and (self.feed.pending > 0 or self._batcher.pending() or self._run_active())
        if busy and not self._poll.isActive():
            self._poll.start()
        elif not busy:
            self._poll.stop()

    # --- the run's state ------------------------------------------------------------------

    def _run_active(self) -> bool:
        snapshot = self._controller.run_snapshot()
        return snapshot is not None and not snapshot.finished

    def _row(self):
        snapshot = self._controller.run_snapshot()
        row = None if snapshot is None or self._name is None else snapshot.row(self._name)
        return snapshot, row

    def _on_run_changed(self) -> None:
        snapshot, row = self._row()
        if row is not None and row.started_at is not None and row.state not in (DONE, FAILED, CANCELLED):
            self._eta.update(row.progress, self._clock())
        finished = snapshot is not None and (snapshot.finished or (row is not None and row.state in
                                                                     (DONE, FAILED, CANCELLED)))
        if finished and not self._flushed:
            self.flush()
        self._refresh_hud()
        self._sync_poll()

    def _refresh_hud(self, eta_only: bool = False) -> None:
        snapshot, row = self._row()
        now = self._clock()
        kept = self._kept_duration()
        active = row is not None and snapshot is not None and not snapshot.finished and row.state not in (
            DONE, FAILED, CANCELLED)
        left = self._eta.seconds_left(now) if active and not (snapshot.paused or snapshot.stopping) else None
        eta = eta_words(left) if left is not None else ""
        if eta_only and eta == self.hud.eta:
            return
        progress = 0.0 if row is None else (1.0 if row.state == DONE else row.progress)
        spans = self._spans()
        total = kept_total(spans)
        ticks = [kept_position(line.start, spans) / total for line in self._received] if total > 0 else []
        speed = self._eta.speed(kept) if active else None
        self.hud.set_state(
            title=hud_title(snapshot, row),
            eta=eta,
            position=POSITION.format(position=format_duration(progress * kept), duration=format_duration(kept))
            if kept > 0 else "",
            speed=speed_words(speed) if speed else "",
            progress=progress,
            ticks=ticks,
            shimmer=active and not snapshot.paused,
        )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_hud()
