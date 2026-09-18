"""The OCR run as the window sees it: per-file rows, the top bar's run
status (elapsed, ETA), the Run view's idle-worker count, and the end-of-run
desktop notification text.

Qt-free and pure: RunTracker is fed the run job's events (core/jobs/run.py's
module docstring lists them) by the controller, on the GUI thread, and
answers with frozen RunSnapshots. Per-file outcomes come from
"run_file_finished" events and the RunSummary, never from the run job's
terminal event type (a stop that arrives after the last file still ends the
job "cancelled").

Times are time.monotonic() seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from app.state_text import format_duration
from core.jobs.run import ERROR_CANCELLED, RunSummary

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
ACTIVE_STATES = frozenset({QUEUED, RUNNING})

NOTIFY_COMPLETE = "OCR Complete"
NOTIFY_FAILED = "OCR Failed"
NOTIFY_FAILED_BODY = "Pipeline encountered an error"
NOTIFY_NO_TIMING_BODY = "All phases completed"


@dataclass(frozen=True)
class RunFileRow:
    name: str
    state: str = QUEUED           # queued | running | done | failed | cancelled
    phase: str = ""               # videocr's phase name, e.g. "Extracting dialogue"
    progress: float = 0.0         # 0..1
    lines: int = 0                # subtitles seen so far; the final file's Dialogue lines once done
    error: str = ""               # failed: the first line of the error
    started_at: float | None = None
    finished_at: float | None = None


@dataclass(frozen=True)
class RunSnapshot:
    files: tuple[RunFileRow, ...]     # the run's files, in the order they were given
    started_at: float
    parallel: int
    paused: bool = False
    stopping: bool = False            # the user asked to stop
    finished: bool = False
    finished_at: float | None = None
    summary: RunSummary | None = None  # None when the run job failed or was cancelled before it ran
    error: str = ""                    # the run job itself failed: its first error line

    def row(self, name: str) -> RunFileRow | None:
        return next((row for row in self.files if row.name == name), None)

    def count(self, *states: str) -> int:
        return sum(1 for row in self.files if row.state in states)

    @property
    def active_names(self) -> list[str]:
        """Files not finished yet (queued or running), while the run is on."""
        if self.finished:
            return []
        return [row.name for row in self.files if row.state in ACTIVE_STATES]


class RunTracker:
    def __init__(self, names: list[str], parallel: int, now: float) -> None:
        self._rows: dict[str, RunFileRow] = {name: RunFileRow(name) for name in names}
        self._subtitles: dict[str, list[tuple[float, float, str]]] = {name: [] for name in names}
        self._started_at = now
        self._parallel = int(parallel)
        self._paused = False
        self._stopping = False
        self._finished = False
        self._finished_at: float | None = None
        self._summary: RunSummary | None = None
        self._error = ""

    # --- events -------------------------------------------------------------------

    def file_started(self, name: str, now: float) -> None:
        self._update(name, state=RUNNING, started_at=now)

    def file_progress(self, name: str, fraction: float | None, phase: str) -> None:
        row = self._rows.get(name)
        if row is None:
            return
        self._update(name, progress=row.progress if fraction is None else float(fraction),
                     phase=phase or row.phase)

    def subtitle(self, name: str, start: float, end: float, text: str) -> None:
        row = self._rows.get(name)
        if row is None:
            return
        self._subtitles[name].append((start, end, text))
        self._update(name, lines=row.lines + 1)

    def file_finished(self, name: str, result: dict, now: float) -> None:
        row = self._rows.get(name)
        if row is None:
            return
        if result.get("ok"):
            self._update(name, state=DONE, progress=1.0, lines=int(result.get("lines", 0)), finished_at=now)
        elif result.get("error") == ERROR_CANCELLED:
            self._update(name, state=CANCELLED, finished_at=now)
        else:
            self._update(name, state=FAILED, error=str(result.get("error", "")), finished_at=now)

    def finish(self, summary: RunSummary | None, error: str, now: float) -> None:
        """The run job ended. Rows the file events did not settle take the
        summary's outcome; without a summary they are failed (the run job
        failed) or cancelled."""
        self._finished = True
        self._finished_at = now
        self._summary = summary
        self._error = error
        outcomes: dict[str, tuple[str, str]] = {}
        if summary is not None:
            outcomes.update({name: (DONE, "") for name in summary.succeeded})
            outcomes.update({name: (FAILED, message) for name, message in summary.failed.items()})
            outcomes.update({name: (CANCELLED, "") for name in summary.cancelled})
        for name, row in self._rows.items():
            if row.state not in ACTIVE_STATES:
                continue
            state, message = outcomes.get(name, (FAILED, error) if error else (CANCELLED, ""))
            self._update(name, state=state, error=message, finished_at=now,
                         progress=1.0 if state == DONE else row.progress)

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)

    def set_parallel(self, parallel: int) -> None:
        self._parallel = int(parallel)

    def set_stopping(self) -> None:
        self._stopping = True

    # --- reading --------------------------------------------------------------------

    def snapshot(self) -> RunSnapshot:
        return RunSnapshot(
            files=tuple(self._rows.values()),
            started_at=self._started_at,
            parallel=self._parallel,
            paused=self._paused,
            stopping=self._stopping,
            finished=self._finished,
            finished_at=self._finished_at,
            summary=self._summary,
            error=self._error,
        )

    def subtitles(self, name: str) -> list[tuple[float, float, str]]:
        return list(self._subtitles.get(name, ()))

    def _update(self, name: str, **changes) -> None:
        row = self._rows.get(name)
        if row is not None:
            self._rows[name] = replace(row, **changes)


# --------------------------------------------------------------------------
# Run status (the top bar during a run, ruling B13) and idle workers
# --------------------------------------------------------------------------

def elapsed_seconds(snapshot: RunSnapshot, now: float) -> float:
    end = snapshot.finished_at if snapshot.finished and snapshot.finished_at is not None else now
    return max(0.0, end - snapshot.started_at)


def eta_seconds(snapshot: RunSnapshot) -> float | None:
    """Today's OCRManager._calculate_eta: the average time of the files that
    finished (done or failed; a stopped file is no measure) x the files still
    queued or running / the files running now (at least 1, at most the
    remaining). None until a file finished, or when nothing remains."""
    durations = [row.finished_at - row.started_at for row in snapshot.files
                 if row.state in (DONE, FAILED) and row.started_at is not None and row.finished_at is not None]
    remaining = snapshot.count(QUEUED, RUNNING)
    if not durations or remaining == 0:
        return None
    average = sum(durations) / len(durations)
    active = min(snapshot.count(RUNNING), remaining) or 1
    return average * remaining / active


def eta_text(seconds: float) -> str:
    """"~20 s left", "~9 min left", "~1 h 5 min left" (rounded up)."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"~{max(1, math.ceil(seconds))} s left"
    minutes = math.ceil(seconds / 60)
    if minutes < 60:
        return f"~{minutes} min left"
    hours, minutes = divmod(minutes, 60)
    return f"~{hours} h {minutes} min left" if minutes else f"~{hours} h left"


def run_status_text(snapshot: RunSnapshot, now: float) -> str:
    """"· running · 2 of 5 done · 14:22 elapsed · ~9 min left" ("paused" /
    "stopping" instead of "running"; no ETA before a file finished or while
    stopping)."""
    word = "stopping" if snapshot.stopping else "paused" if snapshot.paused else "running"
    parts = [word, f"{snapshot.count(DONE)} of {len(snapshot.files)} done",
             f"{format_duration(elapsed_seconds(snapshot, now))} elapsed"]
    eta = None if snapshot.stopping else eta_seconds(snapshot)
    if eta is not None:
        parts.append(eta_text(eta))
    return "· " + " · ".join(parts)


def idle_workers(snapshot: RunSnapshot, parallel: int) -> int:
    """Worker slots `parallel` leaves unused while files wait (ruling B13's
    "N workers idle"): parallel - running files, while the run is going (not
    paused, stopping or finished), has started a file, and still has queued
    files; otherwise 0."""
    if snapshot.finished or snapshot.paused or snapshot.stopping or snapshot.count(QUEUED) == 0:
        return 0
    if not any(row.started_at is not None for row in snapshot.files):
        return 0                                # the run job has not started its files yet
    return max(0, int(parallel) - snapshot.count(RUNNING))


# --------------------------------------------------------------------------
# The desktop notification (today's main.py on_pipeline_finished texts)
# --------------------------------------------------------------------------

def notify_duration(seconds: float) -> str:
    """Today's main.py _format_duration: "42s", "3m 5s", "1h 2m"."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def notification_for(snapshot: RunSnapshot) -> tuple[str, str, str]:
    """(title, body, urgency) for a run that ended without a user stop.

    Any failed file, or a failed run job: "OCR Failed" (critical). Otherwise
    "OCR Complete" with "Finished in X | Avg: Y/file", the average over the
    files that finished (as today's TimingStats counted them)."""
    if snapshot.error or snapshot.count(FAILED) or (snapshot.summary is not None and snapshot.summary.failed):
        return NOTIFY_FAILED, NOTIFY_FAILED_BODY, "critical"
    if snapshot.summary is not None:
        total = snapshot.summary.seconds
    else:
        total = (snapshot.finished_at or snapshot.started_at) - snapshot.started_at
    durations = [row.finished_at - row.started_at for row in snapshot.files
                 if row.started_at is not None and row.finished_at is not None]
    average = sum(durations) / len(durations) if durations else 0.0
    if total > 0:
        return NOTIFY_COMPLETE, f"Finished in {notify_duration(total)} | Avg: {notify_duration(average)}/file", "normal"
    return NOTIFY_COMPLETE, NOTIFY_NO_TIMING_BODY, "normal"
