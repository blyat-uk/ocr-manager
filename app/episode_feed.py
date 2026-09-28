"""The Working screen's clockwork, and the episode screens' words for time.

Pure Python: no Qt and no `core` import, so every rule here is tested with a
fake clock (tests/test_episode_feed.py). Every `now` is a caller's
monotonic seconds; nothing in this module reads a clock itself.

The drip (`LineFeed`). A run reports its subtitles in the batches videocr
emits them in -- thirty-odd lines at once, then nothing for a while. Shown
as they arrive, the script would lurch and the screenshot would flick
through thirty frames in one go. So each batch is queued and released one
line at a time, at a pace that drains the queue just as the next batch is
expected: the mean gap between batches over the lines queued, clamped to
MIN_INTERVAL..MAX_INTERVAL a line. A line waits for its screenshot (the
view fetches every line of a batch at `mid_time(line)`, in one request);
a frame that has not arrived FRAME_TIMEOUT after the line became due lets
the line go anyway, and the view keeps its current picture. Order is never
changed. When the run ends the view `flush()`es: the script must end up
holding everything the run found, however far behind the pacing was.

The ETA (`EtaEstimator`). The rate is the progress gained over a sliding
WINDOW of progress reports, so a slow start or a stall long ago stops
counting. Until BLEND_UNTIL of the file is done that measurement is too
thin to trust alone, and it is blended with the speed remembered from the
last run on this computer (video seconds per wall second), weighted by how
far into the blend span the run is. A report that goes backwards is a new
phase (videocr counts dialogue, then labels, each from 0 to 100 %): the
window starts over.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from typing import NamedTuple


class Line(NamedTuple):
    """One subtitle as the run reported it: the raw float times, in seconds."""
    start: float
    end: float
    text: str

    @property
    def mid(self) -> float:
        return mid_time(self)


def mid_time(line) -> float:
    """The middle of a line's span: the time its screenshot is taken at
    (the start can be the very frame the subtitle fades in on)."""
    return (float(line[0]) + float(line[1])) / 2


def _frame_key(time: float) -> float:
    """Frame times are matched to the millisecond: the time comes back from
    the controller after a trip through a job and a Qt signal."""
    return round(float(time), 3)


# --------------------------------------------------------------------------
# Words
# --------------------------------------------------------------------------

def format_ts(seconds: float) -> str:
    """"H:MM:SS.mmm", the script's time stamp."""
    total_ms = max(0, int(round(float(seconds) * 1000)))
    total_s, ms = divmod(total_ms, 1000)
    hours, rest = divmod(total_s, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}.{ms:03d}"


def duration_words(seconds: float) -> str:
    """"45 s", "3 min 10 s", "3 min", "1 h 5 min", "2 h" (seconds rounded up,
    so a countdown never reads 0 s while something is left)."""
    total = max(0, math.ceil(float(seconds) - 1e-9))
    if total < 60:
        return f"{total} s"
    if total < 3600:
        minutes, secs = divmod(total, 60)
        return f"{minutes} min {secs} s" if secs else f"{minutes} min"
    hours, minutes = divmod(total // 60, 60)
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def eta_words(seconds: float) -> str:
    """"≈ 3 min 10 s" (the HUD adds "left")."""
    return f"≈ {duration_words(seconds)}"


def speed_words(speed: float) -> str:
    """"4.1× real time"."""
    return f"{speed:.1f}× real time"


def estimate_words(seconds: float) -> str:
    """The Preparing screen's "about 6 min" (minutes rounded up), or "under a
    minute"."""
    if seconds < 60:
        return "under a minute"
    return f"about {math.ceil(seconds / 60)} min"


# --------------------------------------------------------------------------
# Kept time
# --------------------------------------------------------------------------

def kept_total(spans: Iterable[tuple[float, float]]) -> float:
    """The seconds the keep spans add up to."""
    return sum(max(0.0, end - start) for start, end in spans)


def kept_position(time: float, spans: Iterable[tuple[float, float]]) -> float:
    """A video time as a position along the kept time: the kept seconds
    before it. A time in a skipped gap sits at the end of the span before
    it. `spans` are ordered and do not overlap (ranges_view.read_ranges)."""
    return sum(min(max(0.0, time - start), max(0.0, end - start)) for start, end in spans)


# --------------------------------------------------------------------------
# LineFeed
# --------------------------------------------------------------------------

class LineFeed:
    MIN_INTERVAL = 0.25        # s a line, at the fastest
    MAX_INTERVAL = 3.0         # s a line, at the slowest
    FRAME_TIMEOUT = 1.5        # s a due line waits for its screenshot
    FIRST_INTERVAL = 1.0       # s a line before a second batch has shown the gap between batches
    GAPS_KEPT = 8              # the mean gap is over this many most recent gaps

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._queue: deque[Line] = deque()
        self._ready: set[float] = set()
        self._batch_times: deque[float] = deque(maxlen=self.GAPS_KEPT + 1)
        self._interval = self.FIRST_INTERVAL
        self._next_due: float | None = None

    @property
    def pending(self) -> int:
        """Lines queued and not released yet."""
        return len(self._queue)

    @property
    def interval(self) -> float:
        """The current pace, in seconds a line."""
        return self._interval

    def mean_gap(self) -> float | None:
        times = list(self._batch_times)
        if len(times) < 2:
            return None
        return (times[-1] - times[0]) / (len(times) - 1)

    def add_batch(self, lines: Iterable, now: float) -> None:
        batch = [Line(float(start), float(end), str(text)) for start, end, text in lines]
        if not batch:
            return
        self._batch_times.append(float(now))
        if not self._queue:
            # The head of a fresh queue is due now (or when the pace allows),
            # never in the past: its frame timeout counts from here.
            self._next_due = float(now) if self._next_due is None else max(self._next_due, float(now))
        self._queue.extend(batch)
        gap = self.mean_gap()
        pace = self.FIRST_INTERVAL if gap is None else gap / len(self._queue)
        self._interval = min(self.MAX_INTERVAL, max(self.MIN_INTERVAL, pace))

    def frame_ready(self, time: float) -> None:
        """The screenshot at `time` (a line's `mid_time`) can be shown."""
        self._ready.add(_frame_key(time))

    def due(self, now: float) -> list[Line]:
        """The lines to show now, in order: at most one a call while the pace
        holds, none while the head waits for its frame."""
        released: list[Line] = []
        while self._queue:
            if now < self._next_due:
                break
            head = self._queue[0]
            if _frame_key(head.mid) not in self._ready and now < self._next_due + self.FRAME_TIMEOUT:
                break
            released.append(self._queue.popleft())
            self._next_due = now + self._interval
        return released

    def flush(self) -> list[Line]:
        """Everything still queued, in order; the queue is left empty."""
        released = list(self._queue)
        self._queue.clear()
        self._next_due = None
        return released


# --------------------------------------------------------------------------
# EtaEstimator
# --------------------------------------------------------------------------

class EtaEstimator:
    WINDOW = 30.0              # s of progress reports the rate is measured over
    BLEND_UNTIL = 0.05         # the remembered speed counts until this much is done

    def __init__(self, prior_speed: float | None = None, kept_duration: float = 0.0) -> None:
        self._prior_speed = prior_speed if prior_speed and prior_speed > 0 else None
        self._kept = float(kept_duration or 0.0)
        self._samples: deque[tuple[float, float]] = deque()
        self._progress = 0.0
        self._last: float | None = None

    @property
    def progress(self) -> float:
        return self._progress

    def update(self, progress: float, now: float) -> None:
        progress = min(1.0, max(0.0, float(progress)))
        if self._samples and progress < self._samples[-1][1] - 1e-9:
            self._samples.clear()                      # a new phase counts from 0 again
        self._samples.append((float(now), progress))
        while len(self._samples) > 2 and self._samples[1][0] <= now - self.WINDOW:
            self._samples.popleft()                    # keep one anchor at the window's start
        self._progress = progress
        self._last = float(now)

    def _measured(self) -> float | None:
        if len(self._samples) < 2:
            return None
        (t0, p0), (t1, p1) = self._samples[0], self._samples[-1]
        if t1 <= t0 or p1 <= p0:
            return None
        return (p1 - p0) / (t1 - t0)

    def _prior(self) -> float | None:
        if self._prior_speed is None or self._kept <= 0:
            return None
        return self._prior_speed / self._kept

    def rate(self) -> float | None:
        """Progress (0..1) a wall second, or None while nothing says."""
        measured, prior = self._measured(), self._prior()
        if measured is None:
            return prior
        if prior is None:
            return measured
        weight = min(1.0, self._progress / self.BLEND_UNTIL)
        return weight * measured + (1 - weight) * prior

    def seconds_left(self, now: float) -> float | None:
        rate = self.rate()
        if not rate:
            return None
        since = 0.0 if self._last is None else max(0.0, now - self._last)
        return max(0.0, (1.0 - self._progress) / rate - since)

    def speed(self, kept_duration: float) -> float | None:
        """Video seconds a wall second: the HUD's "4.1× real time"."""
        rate = self.rate()
        if not rate or kept_duration <= 0:
            return None
        return rate * kept_duration
