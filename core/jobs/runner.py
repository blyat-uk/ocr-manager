"""Qt-free background job runner with GPU, CPU and RUN lanes.

Every background activity of the Stage 3 window is a Job submitted here: metadata
scans and audio analysis (CPU lane), crop/brightness detection and the proof OCR
(GPU lane), and the OCR run (RUN lane). Nothing here imports Qt; the window drains
the events on its own thread through one adapter (ruling C8).

Lanes
    GPU  one worker thread: at most one job at a time (detectors share engines).
    CPU  `cpu_workers` worker threads.
    RUN  one worker thread: the run job, which manages its own file workers.
    Within a lane, higher `priority` starts first; equal priorities start in
    submission order.

Events (JobEvent), per job, always in this order:
    queued -> started -> (progress | log | custom)* -> finished | failed | cancelled
    or, for a job cancelled or replaced before it started:  queued -> cancelled
    "queued", and the "cancelled" of a job removed from a queue, are delivered on
    the caller's thread before submit/cancel/cancel_where/shutdown returns. All
    other events are delivered on worker threads (or on whichever thread the job
    emits from), never two at a time for the same job. `on_event` must not block
    and must marshal to its own thread; its exceptions are logged and swallowed.
    Events a job emits after its terminal event are dropped.

Cancellation is cooperative only: nothing is ever killed.
    A job sees a request through ctx.cancelled(), the ctx.cancel_check()
    callable (for core/detect APIs), or ctx.cancel_event.is_set()/.wait()
    (for videocr). When run() returns, the job reports "cancelled" if it saw a
    request, else "finished" with its result: a job that completed its work
    without looking at the request (or before it arrived) delivers that result.
    An exception from run() is always "failed".

Pause holds queued jobs; running jobs are never affected (ruling C6).
"""
from __future__ import annotations

import heapq
import itertools
import logging
import math
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

logger = logging.getLogger(__name__)


class Lane(str, Enum):
    GPU = "gpu"      # one worker thread: detector jobs, proof OCR
    CPU = "cpu"      # N worker threads: metadata, thumbnails, audio analysis/profile
    RUN = "run"      # one worker thread: the OCR run job (manages its own file workers)


@dataclass(frozen=True)
class JobEvent:
    type: str        # "queued" | "started" | "progress" | "log" | "finished" | "failed" | "cancelled"
    key: str
    kind: str
    file: str | None
    progress: float | None = None      # 0..1 for "progress"
    message: str = ""                  # progress text, log line, or failure summary
    result: object = None              # "finished" only
    error: str = ""                    # "failed": formatted traceback


# Event types only the runner emits; JobContext.emit refuses them.
LIFECYCLE_EVENTS = frozenset({"queued", "started", "finished", "failed", "cancelled"})
# JobEvent fields a custom event may set; key and kind always come from the job.
_EMIT_FIELDS = frozenset({"file", "progress", "message", "result", "error"})


def _deliver(listener: Callable[[JobEvent], None] | None, event: JobEvent) -> None:
    if listener is None:
        return
    try:
        listener(event)
    except Exception:
        logger.exception("job event listener raised on %r for %s", event.type, event.key)


class _CancelEvent(threading.Event):
    """A threading.Event that remembers whether a reader found it set.

    Lets the runner tell a job that stopped for a cancel request from one that
    finished without ever looking, whichever way the job reads the request.
    """

    def __init__(self) -> None:
        super().__init__()
        self._seen_set = False

    def is_set(self) -> bool:
        flag = super().is_set()
        if flag:
            self._seen_set = True
        return flag

    def wait(self, timeout: float | None = None) -> bool:
        flag = super().wait(timeout)
        if flag:
            self._seen_set = True
        return flag


class JobContext:
    """What a running job gets: its cancel request and its event channel.

    Usable on its own (e.g. to call a job's run() in a test): events go to
    `on_event`, or nowhere when it is None.
    """

    def __init__(self, key: str = "", kind: str = "", file: str | None = None,
                 on_event: Callable[[JobEvent], None] | None = None) -> None:
        self.key = key
        self.kind = kind
        self.file = file
        self.cancel_event: threading.Event = _CancelEvent()
        self._on_event = on_event
        # Serialises this job's events (helper threads may emit concurrently)
        # and closes the channel at the terminal event. Re-entrant so a
        # listener may emit for the same job.
        self._emit_lock = threading.RLock()
        self._closed = False

    def cancelled(self) -> bool:
        """True once cancellation has been requested (and marks it seen)."""
        return self.cancel_event.is_set()

    def cancel_check(self) -> Callable[[], bool]:
        """Zero-argument callable for core/detect's `cancel_check` parameters."""
        return self.cancelled

    def progress(self, fraction: float, message: str = "") -> None:
        value = float(fraction)
        value = 0.0 if math.isnan(value) else min(1.0, max(0.0, value))
        self._send(JobEvent("progress", self.key, self.kind, self.file,
                            progress=value, message=message))

    def log(self, line: str) -> None:
        self._send(JobEvent("log", self.key, self.kind, self.file, message=line))

    def emit(self, event_type: str, **fields) -> None:
        """A custom event, e.g. the run job's per-file events.

        `fields` may set file, progress, message, result and error; key and
        kind are the job's, and file defaults to the job's.
        """
        if event_type in LIFECYCLE_EVENTS:
            raise ValueError(f"{event_type!r} is a lifecycle event only the runner emits")
        unknown = set(fields) - _EMIT_FIELDS
        if unknown:
            raise TypeError(f"emit() got unsupported JobEvent fields: {sorted(unknown)}")
        fields.setdefault("file", self.file)
        self._send(JobEvent(event_type, self.key, self.kind, **fields))

    def _send(self, event: JobEvent, *, final: bool = False) -> None:
        with self._emit_lock:
            if self._closed:
                logger.debug("dropped %r for %s: job already ended", event.type, event.key)
                return
            if final:
                self._closed = True
            _deliver(self._on_event, event)

    def _saw_cancel(self) -> bool:
        event = self.cancel_event
        return bool(getattr(event, "_seen_set", False))


class Job(Protocol):
    key: str          # unique identity, e.g. "crop:ZS2_-_12.mp4"
    kind: str         # "metadata" | "thumbnail" | "crop" | "brightness" | "ranges" | "audio_profile" | "proof" | "run"
    lane: Lane
    file: str | None
    priority: int     # higher runs first; FIFO within a priority

    def run(self, ctx: JobContext) -> object: ...


@dataclass(eq=False)
class _Entry:
    job: Job
    lane: Lane
    priority: int
    seq: int
    ctx: JobContext

    def event(self, event_type: str, **fields) -> JobEvent:
        return JobEvent(event_type, self.job.key, self.job.kind, self.job.file, **fields)


def _first_line(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else type(exc).__name__


class JobRunner:
    """Runs jobs on daemon worker threads, one pool per lane.

    All state is guarded by one re-entrant lock; each lane's workers wait on
    their own Condition over that lock. Each lane's queue is a heap keyed by
    (-priority, seq).
    """

    def __init__(self, on_event: Callable[[JobEvent], None], cpu_workers: int = 2) -> None:
        if cpu_workers < 1:
            raise ValueError(f"cpu_workers must be at least 1, got {cpu_workers}")
        self._on_event = on_event
        self._lock = threading.RLock()
        self._conds = {lane: threading.Condition(self._lock) for lane in Lane}
        self._heaps: dict[Lane, list[tuple[int, int, _Entry]]] = {lane: [] for lane in Lane}
        self._queued_by_key: dict[str, _Entry] = {}
        self._running: list[_Entry] = []
        self._hold_all = {lane: False for lane in Lane}
        self._hold_only: dict[Lane, list[Callable[[Job], bool]]] = {lane: [] for lane in Lane}
        self._seq = itertools.count()
        self._closed = False
        self._threads: list[threading.Thread] = []
        for lane, count in ((Lane.GPU, 1), (Lane.CPU, cpu_workers), (Lane.RUN, 1)):
            for index in range(count):
                self._threads.append(threading.Thread(
                    target=self._work, args=(lane,),
                    name=f"jobs-{lane.value}-{index}", daemon=True))
        for thread in self._threads:
            thread.start()

    # --- submitting and cancelling -------------------------------------------

    def submit(self, job: Job) -> None:
        """Queue `job`. A queued job with the same key is replaced in place."""
        lane = Lane(job.lane)
        priority = int(job.priority)
        with self._lock:
            if self._closed:
                raise RuntimeError("JobRunner has been shut down")
            old = self._queued_by_key.get(job.key)
            if old is not None:
                self._unqueue(old)
                seq = old.seq
            else:
                seq = next(self._seq)
            entry = _Entry(job, lane, priority, seq,
                           JobContext(job.key, job.kind, job.file, self._on_event))
            heapq.heappush(self._heaps[lane], (-priority, seq, entry))
            self._queued_by_key[job.key] = entry
            # Delivered under the lock, so no worker can start the new job
            # (and emit "started") before its "queued" is out.
            if old is not None:
                old.ctx._send(old.event("cancelled"), final=True)
            entry.ctx._send(entry.event("queued"))
            self._conds[lane].notify_all()

    def cancel(self, key: str) -> None:
        """Drop the queued job with `key` and request cancel of running ones."""
        self.cancel_where(lambda job: job.key == key)

    def cancel_where(self, predicate: Callable[[Job], bool]) -> None:
        """`cancel` every queued and running job for which `predicate` is true.

        `predicate` runs under the runner's lock and must be quick. It sees
        every job before any is cancelled, so if it raises, nothing is.
        """
        with self._lock:
            queued = [entry for entry in self._queued_entries() if predicate(entry.job)]
            running = [entry for entry in self._running if predicate(entry.job)]
            for entry in queued:
                self._unqueue(entry)
                entry.ctx._send(entry.event("cancelled"), final=True)
            for entry in running:
                entry.ctx.cancel_event.set()

    # --- pause ------------------------------------------------------------------

    def pause(self, lane: Lane, *, only: Callable[[Job], bool] | None = None) -> None:
        """Hold queued jobs on `lane`: all of them, or just those matching `only`.

        Holds accumulate: several `only` predicates hold what any of them
        matches, and a pause without `only` holds everything. `only` runs on
        worker threads under the runner's lock whenever a worker looks for a
        job, so it must be quick; if it raises, the job is held (and logged).
        """
        lane = Lane(lane)
        with self._lock:
            if only is None:
                self._hold_all[lane] = True
            else:
                self._hold_only[lane].append(only)

    def resume(self, lane: Lane) -> None:
        """Clear every hold on `lane`."""
        lane = Lane(lane)
        with self._lock:
            self._hold_all[lane] = False
            self._hold_only[lane].clear()
            self._conds[lane].notify_all()

    def is_paused(self, lane: Lane) -> bool:
        """True while any hold (full or `only=`) is in place on `lane`."""
        lane = Lane(lane)
        with self._lock:
            return self._hold_all[lane] or bool(self._hold_only[lane])

    # --- inspection -----------------------------------------------------------

    def queued(self) -> list[Job]:
        """Queued jobs, lane by lane (GPU, CPU, RUN), each in start order."""
        with self._lock:
            return [entry.job for entry in self._queued_entries()]

    def running(self) -> list[Job]:
        """Jobs whose run() has been or is about to be called, oldest first."""
        with self._lock:
            return [entry.job for entry in self._running]

    # --- shutdown -------------------------------------------------------------

    def shutdown(self, timeout: float = 10.0) -> bool:
        """Cancel everything cooperatively and join the workers.

        Requests cancel of every running job, empties every queue ("cancelled"
        for each queued job) and waits up to `timeout` seconds in total for the
        worker threads. Returns True only if all of them exited. Threads are
        never killed; a job that ignores the request keeps running on its
        daemon thread. After shutdown, submit raises RuntimeError.
        """
        with self._lock:
            self._closed = True
            for entry in self._running:
                entry.ctx.cancel_event.set()
            for entry in self._queued_entries():
                self._unqueue(entry)
                entry.ctx._send(entry.event("cancelled"), final=True)
            for cond in self._conds.values():
                cond.notify_all()
        deadline = time.monotonic() + max(0.0, timeout)
        current = threading.current_thread()
        for thread in self._threads:
            if thread is not current:
                thread.join(max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in self._threads)

    # --- internals (callers hold self._lock where noted) ------------------------

    def _queued_entries(self) -> list[_Entry]:
        """Lock held."""
        return [entry for lane in Lane for _, _, entry in sorted(self._heaps[lane])]

    def _unqueue(self, entry: _Entry) -> None:
        """Lock held. Remove a queued entry from its heap and the key index."""
        heap = self._heaps[entry.lane]
        for index, (_, _, candidate) in enumerate(heap):
            if candidate is entry:
                del heap[index]
                heapq.heapify(heap)
                break
        if self._queued_by_key.get(entry.job.key) is entry:
            del self._queued_by_key[entry.job.key]

    def _is_held(self, lane: Lane, job: Job) -> bool:
        """Lock held."""
        for predicate in self._hold_only[lane]:
            try:
                if predicate(job):
                    return True
            except Exception:
                logger.exception("pause predicate raised for %s; holding it", job.key)
                return True
        return False

    def _take(self, lane: Lane) -> _Entry | None:
        """Lock held. The next startable entry on `lane`, now running."""
        heap = self._heaps[lane]
        if not heap or self._hold_all[lane]:
            return None
        if not self._hold_only[lane]:
            entry = heap[0][2]
        else:
            entry = next((candidate for _, _, candidate in sorted(heap)
                          if not self._is_held(lane, candidate.job)), None)
            if entry is None:
                return None
        self._unqueue(entry)
        self._running.append(entry)
        return entry

    def _work(self, lane: Lane) -> None:
        cond = self._conds[lane]
        while True:
            with cond:
                entry = self._take(lane)
                while entry is None:
                    if self._closed:
                        return
                    cond.wait()
                    entry = self._take(lane)
            try:
                self._execute(entry)
            except BaseException:  # never let a runner bug take the lane down
                logger.exception("job runner error while running %s", entry.job.key)
                with self._lock:
                    if entry in self._running:
                        self._running.remove(entry)

    def _execute(self, entry: _Entry) -> None:
        ctx = entry.ctx
        ctx._send(entry.event("started"))
        try:
            result = entry.job.run(ctx)
        except BaseException as exc:  # noqa: BLE001 - a job's failure must not end the worker
            terminal = entry.event("failed", message=_first_line(exc),
                                   error=traceback.format_exc())
        else:
            if ctx._saw_cancel():
                terminal = entry.event("cancelled")
            else:
                terminal = entry.event("finished", result=result)
        # Leave running() before the terminal event goes out, so a listener
        # reacting to it sees the job gone; this worker starts nothing else
        # until the event has been delivered.
        with self._lock:
            self._running.remove(entry)
        ctx._send(terminal, final=True)
