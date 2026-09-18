"""What the background is doing: the activity strip's and the chips' model.

Qt-free and pure: ActivityTracker is fed the JobEvents the controller drains
(on the GUI thread) and answers with frozen ActivitySnapshots. It never
touches a runner. The OCR run (kind "run") is not activity: the run view and
the top bar report it (app/run_snapshot.py).

Times are time.monotonic() seconds.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

RUN_KIND = "run"
TERMINAL_EVENTS = frozenset({"finished", "failed", "cancelled"})
RECENT_LIMIT = 5

# Human wording per job kind, for views that name a job ("audio analysis done 11 s ago").
KIND_LABELS = {
    "metadata": "metadata",
    "thumbnail": "thumbnail",
    "crop": "finding subtitle frames",
    "brightness": "measuring brightness",
    "ranges": "matching intro/outro",
    "audio_profile": "audio analysis",
    "proof": "test OCR",
}


@dataclass(frozen=True)
class ActivitySnapshot:
    # The running job the strip shows: (kind, file, fraction, message). The
    # longest-running job wins. fraction is None until the job reports
    # progress (detectors never do: show an indeterminate bar, no percent).
    current: tuple[str, str | None, float | None, str] | None
    # Finished jobs, newest first: (kind, file, finished_at).
    recent: list[tuple[str, str | None, float]]
    paused: bool                  # the user paused auto-pilot
    running: tuple[tuple[str, str | None], ...] = field(default_factory=tuple)   # (kind, file), oldest first


@dataclass
class _Job:
    kind: str
    file: str | None
    started_at: float | None = None
    fraction: float | None = None
    message: str = ""


class ActivityTracker:
    def __init__(self) -> None:
        self._jobs: dict[int, _Job] = {}          # job_id -> queued or running job
        self._recent: deque[tuple[str, str | None, float]] = deque(maxlen=RECENT_LIMIT)

    def on_event(self, event, now: float) -> bool:
        """Account for one drained event; True when the snapshot changed."""
        if event.kind == RUN_KIND:
            return False
        if event.type == "queued":
            self._jobs[event.job_id] = _Job(event.kind, event.file)
        elif event.type == "started":
            job = self._jobs.setdefault(event.job_id, _Job(event.kind, event.file))
            job.started_at = now
        elif event.type == "progress":
            job = self._jobs.get(event.job_id)
            if job is None:
                return False
            job.fraction = event.progress
            job.message = event.message or job.message
        elif event.type in TERMINAL_EVENTS:
            job = self._jobs.pop(event.job_id, None)
            if event.type == "finished":
                self._recent.appendleft((event.kind, event.file, now))
            elif job is None:
                return False
        else:
            return False
        return True

    def running_kinds(self, file: str) -> set[str]:
        """Kinds of the file's running jobs; a running folder-wide ranges
        analysis counts for every file."""
        return {job.kind for job in self._running()
                if job.file == file or (job.file is None and job.kind == "ranges")}

    def snapshot(self, *, paused: bool) -> ActivitySnapshot:
        running = self._running()
        current = None
        if running:
            job = running[0]
            current = (job.kind, job.file, job.fraction, job.message or KIND_LABELS.get(job.kind, job.kind))
        return ActivitySnapshot(
            current=current,
            recent=list(self._recent),
            paused=paused,
            running=tuple((job.kind, job.file) for job in running),
        )

    def clear(self) -> None:
        self._jobs.clear()
        self._recent.clear()

    def _running(self) -> list[_Job]:
        running = [job for job in self._jobs.values() if job.started_at is not None]
        running.sort(key=lambda job: job.started_at)
        return running
