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
    submission order. A job never starts while a job with the same key is still
    running (on any lane): it waits until that job's terminal event has been
    delivered, and jobs behind it that can start do.

Events (JobEvent), per job instance, always in this order:
    queued -> started -> (progress | log | custom)* -> finished | failed | cancelled
    or, for a job cancelled or replaced before it started:  queued -> cancelled
    Each submit gets a new `job_id`, carried by every event of that instance, so
    two instances with the same key can be told apart. Events a job emits after
    its terminal event are dropped.

The listener (`on_event`) contract
    - "queued", and the "cancelled" of a job removed from a queue, are delivered
      on the thread calling submit/cancel/cancel_where/shutdown, WITH THE RUNNER'S
      LOCK HELD, before that call returns. The lock is re-entrant, so the listener
      may call the runner from that same thread. Every other event is delivered
      on a worker thread (or whichever thread the job emits from) without the
      runner's lock, never two at a time for the same job.
    - While a listener runs with the lock held, no worker can start or finish
      a job: a slow "queued" listener stalls every lane.
    - Never call the runner while holding a lock the listener also takes. Thread
      A in submit() holds the runner's lock and waits in the listener for your
      lock, while thread B holds your lock and waits for the runner's: deadlock.
    - Never call shutdown(), join a thread or wait for another thread from the
      listener. A shutdown() from a "queued"/"cancelled" delivery holds the lock
      the workers need in order to exit, so it stalls until its timeout.
    - The intended consumer, the Qt adapter, only enqueues the event and returns.
    - Exceptions (Exception) from the listener are logged and swallowed. If a
      listener raises a BaseException on a worker, the job still gets a terminal
      "failed" event (unless its terminal event was the one being delivered).

Cancellation is cooperative only: nothing is ever killed.
    A job sees a request through ctx.cancelled(), the ctx.cancel_check()
    callable (for core/detect APIs), or ctx.cancel_event.is_set()/.wait()
    (for videocr). When run() returns, the job reports "cancelled" if it saw a
    request, else "finished"; either way `result` is what run() returned. A job
    that completed its work without looking at the request (or before it
    arrived) therefore reports "finished". An exception from run() is always
    "failed".

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
from dataclasses import KW_ONLY, dataclass
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
    _: KW_ONLY                         # everything below is keyword-only
    job_id: int = 0                    # one per submit, on every event of that instance (0: context outside a runner)
    progress: float | None = None      # 0..1 for "progress"
    message: str = ""                  # progress text, log line, or failure summary
    result: object = None              # "finished", and "cancelled" when run() returned: run()'s return value
    error: str = ""                    # "failed": formatted traceback


# Event types only the runner emits; JobContext.emit refuses them.
LIFECYCLE_EVENTS = frozenset({"queued", "started", "finished", "failed", "cancelled"})
# JobEvent fields a custom event may set; key, kind and job_id always come from the job.
_EMIT_FIELDS = frozenset({"file", "progress", "message", "result", "error"})


def _deliver(listener: Callable[[JobEvent], None] | None, event: JobEvent) -> None:
    if listener is None:
        return
    try:
        listener(event)
    except Exception:
        logger.exception("job event listener raised on %r for %s", event.type, event.key)


def _first_line(exc: BaseException) -> str:
    try:
        lines = str(exc).strip().splitlines()
    except Exception:  # noqa: BLE001 - an exception whose __str__ itself raises
        lines = []
    return lines[0] if lines else type(exc).__name__


def _format_traceback(exc: BaseException) -> str:
    try:
        return "".join(traceback.format_exception(exc))
    except Exception:  # noqa: BLE001 - never let reporting a failure fail
        return f"{type(exc).__name__}: <traceback unavailable>"


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

    Only the job itself should read its context's cancel state. Any read that
    finds the request set (cancelled(), the cancel_check() callable,
    cancel_event.is_set() or .wait()) counts as the job having seen it and
    turns a normal return into "cancelled".

    Usable on its own (e.g. to call a job's run() in a test): events go to
    `on_event`, or nowhere when it is None.
    """

    def __init__(self, key: str = "", kind: str = "", file: str | None = None,
                 on_event: Callable[[JobEvent], None] | None = None, *,
                 job_id: int = 0) -> None:
        self._key = key
        self._kind = kind
        self._file = file
        self._job_id = job_id
        self.cancel_event: threading.Event = _CancelEvent()
        self._on_event = on_event
        # Serialises this job's events (helper threads may emit concurrently)
        # and closes the channel at the terminal event. Re-entrant so a
        # listener may emit for the same job.
        self._emit_lock = threading.RLock()
        self._closed = False

    # Identity is fixed at submit: read-only, so a job cannot re-label its
    # events or detach itself from the runner's bookkeeping.
    @property
    def key(self) -> str:
        return self._key

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def file(self) -> str | None:
        return self._file

    @property
    def job_id(self) -> int:
        return self._job_id

    def cancelled(self) -> bool:
        """True once cancellation has been requested (and marks it seen)."""
        return self.cancel_event.is_set()

    def cancel_check(self) -> Callable[[], bool]:
        """Zero-argument callable for core/detect's `cancel_check` parameters."""
        return self.cancelled

    def progress(self, fraction: float, message: str = "") -> None:
        value = float(fraction)
        value = 0.0 if math.isnan(value) else min(1.0, max(0.0, value))
        self._send(self._event("progress", progress=value, message=message))

    def log(self, line: str) -> None:
        self._send(self._event("log", message=line))

    def emit(self, event_type: str, **fields) -> None:
        """A custom event, e.g. the run job's per-file events.

        `fields` may set file, progress, message, result and error; key, kind
        and job_id are the job's, and file defaults to the job's.
        """
        if event_type in LIFECYCLE_EVENTS:
            raise ValueError(f"{event_type!r} is a lifecycle event only the runner emits")
        unknown = set(fields) - _EMIT_FIELDS
        if unknown:
            raise TypeError(f"emit() got unsupported JobEvent fields: {sorted(unknown)}")
        self._send(self._event(event_type, **fields))

    def _event(self, event_type: str, **fields) -> JobEvent:
        fields.setdefault("file", self.file)
        return JobEvent(event_type, self.key, self.kind, job_id=self.job_id, **fields)

    def _send(self, event: JobEvent, *, final: bool = False) -> None:
        with self._emit_lock:
            if self._closed:
                logger.debug("dropped %r for %s: job already ended", event.type, event.key)
                return
            if final:
                self._closed = True
            _deliver(self._on_event, event)

    def _saw_cancel(self) -> bool:
        return bool(getattr(self.cancel_event, "_seen_set", False))


class Job(Protocol):
    key: str          # unique identity, e.g. "crop:ZS2_-_12.mp4"
    kind: str         # "metadata" | "thumbnail" | "crop" | "brightness" | "ranges" | "audio_profile" | "proof" | "run"
    lane: Lane
    file: str | None
    priority: int     # higher runs first; FIFO within a priority

    def run(self, ctx: JobContext) -> object: ...


@dataclass(eq=False)
class _Entry:
    """One submitted job instance. Its identity is captured here at submit
    (and read-only on `ctx`); the runner never reads the job's attributes again."""
    job: Job
    lane: Lane
    priority: int
    seq: int
    key: str
    kind: str
    file: str | None
    ctx: JobContext


class JobRunner:
    """Runs jobs on daemon worker threads, one pool per lane.

    All state is guarded by one re-entrant lock; each lane's workers wait on
    their own Condition over that lock. Each lane's queue is a heap keyed by
    (-priority, seq).

    `on_event` receives every JobEvent; read the listener contract in this
    module's docstring. In short: "queued" and queue-removal "cancelled" arrive
    with the runner's lock held; never call the runner while holding a lock the
    listener also takes; never call shutdown/join/wait from the listener; a
    slow listener stalls the lanes; just enqueue the event and return.
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
        # Keys of started jobs whose terminal event has not been delivered yet.
        self._active_keys: set[str] = set()
        self._hold_all = {lane: False for lane in Lane}
        self._hold_only: dict[Lane, list[Callable[[Job], bool]]] = {lane: [] for lane in Lane}
        self._seq = itertools.count()
        self._job_ids = itertools.count(1)
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
        """Queue `job`, with a new job_id.

        A queued job with the same key is replaced in place: it gets
        "cancelled", then this job gets "queued" at its position. A running
        job with the same key is not touched; this job waits until it ends.
        `key`, `kind`, `file`, `lane` and `priority` are read once, here.
        """
        key, kind, file = job.key, job.kind, job.file
        lane = Lane(job.lane)
        priority = int(job.priority)
        with self._lock:
            if self._closed:
                raise RuntimeError("JobRunner has been shut down")
            seq = None
            # A loop, because a listener may queue another job with this key
            # from inside the replaced job's "cancelled".
            while (old := self._queued_by_key.get(key)) is not None:
                self._unqueue(old)
                if seq is None:
                    seq = old.seq
                old.ctx._send(old.ctx._event("cancelled"), final=True)
            if self._closed:  # a listener shut the runner down meanwhile
                raise RuntimeError("JobRunner has been shut down")
            if seq is None:
                seq = next(self._seq)
            ctx = JobContext(key, kind, file, self._on_event, job_id=next(self._job_ids))
            entry = _Entry(job, lane, priority, seq, key, kind, file, ctx)
            heapq.heappush(self._heaps[lane], (-priority, seq, entry))
            self._queued_by_key[key] = entry
            # Delivered under the lock, so no worker can start this job (and
            # emit "started") before its "queued" is out.
            ctx._send(ctx._event("queued"))
            self._conds[lane].notify_all()

    def cancel(self, key: str) -> None:
        """Drop the queued job with `key` and request cancel of running ones."""
        self._cancel_matching(lambda entry: entry.key == key)

    def cancel_where(self, predicate: Callable[[Job], bool]) -> None:
        """`cancel` every queued and running job for which `predicate` is true.

        `predicate` runs under the runner's lock and must be quick. It sees
        every job before any is cancelled, so if it raises, nothing is.
        """
        self._cancel_matching(lambda entry: predicate(entry.job))

    # --- pause ------------------------------------------------------------------

    def pause(self, lane: Lane, *, only: Callable[[Job], bool] | None = None) -> None:
        """Hold queued jobs on `lane`: all of them, or just those matching `only`.

        Holds accumulate: several `only` predicates hold what any of them
        matches, and a pause without `only` holds everything.

        `only` is evaluated on worker threads, under the runner's lock, each
        time the lane is woken (a submit, resume, or a job ending), not
        continuously: its answer for a job should not change while the hold is
        in place. It must be quick; if it raises, the job is held (and logged).
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
        """Queued jobs, lane by lane (GPU, CPU, RUN), each in priority order."""
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
        never killed: a job that ignores the request keeps running on its
        daemon thread, and its events (including its terminal event) still
        reach `on_event` after shutdown has returned. After shutdown, submit
        raises RuntimeError.

        Called from a job (on its worker thread), that thread cannot join
        itself and is left out of the result. Never call it from `on_event`
        (see the listener contract).
        """
        with self._lock:
            self._closed = True
            for entry in self._running:
                entry.ctx.cancel_event.set()
            for entry in self._queued_entries():
                if self._unqueue(entry):
                    entry.ctx._send(entry.ctx._event("cancelled"), final=True)
            for cond in self._conds.values():
                cond.notify_all()
        deadline = time.monotonic() + max(0.0, timeout)
        current = threading.current_thread()
        others = [thread for thread in self._threads if thread is not current]
        for thread in others:
            thread.join(max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in others)

    # --- internals (callers hold self._lock where noted) ------------------------

    def _cancel_matching(self, matches: Callable[[_Entry], bool]) -> None:
        with self._lock:
            queued = [entry for entry in self._queued_entries() if matches(entry)]
            running = [entry for entry in self._running if matches(entry)]
            for entry in queued:
                # A listener may already have removed it from inside an
                # earlier "cancelled".
                if self._unqueue(entry):
                    entry.ctx._send(entry.ctx._event("cancelled"), final=True)
            for entry in running:
                entry.ctx.cancel_event.set()

    def _queued_entries(self) -> list[_Entry]:
        """Lock held."""
        return [entry for lane in Lane for _, _, entry in sorted(self._heaps[lane])]

    def _unqueue(self, entry: _Entry) -> bool:
        """Lock held. Remove a queued entry; False if it was not queued."""
        if self._queued_by_key.get(entry.key) is not entry:
            return False
        del self._queued_by_key[entry.key]
        heap = self._heaps[entry.lane]
        for index, (_, _, candidate) in enumerate(heap):
            if candidate is entry:
                del heap[index]
                heapq.heapify(heap)
                break
        return True

    def _is_held(self, lane: Lane, job: Job) -> bool:
        """Lock held."""
        for predicate in self._hold_only[lane]:
            try:
                if predicate(job):
                    return True
            except Exception:
                logger.exception("pause predicate raised; holding the job")
                return True
        return False

    def _startable(self, lane: Lane, entry: _Entry) -> bool:
        """Lock held."""
        return entry.key not in self._active_keys and not self._is_held(lane, entry.job)

    def _take(self, lane: Lane) -> _Entry | None:
        """Lock held. The next startable entry on `lane`, now running."""
        heap = self._heaps[lane]
        if not heap or self._hold_all[lane]:
            return None
        entry = heap[0][2]
        if not self._startable(lane, entry):
            entry = next((candidate for _, _, candidate in sorted(heap)[1:]
                          if self._startable(lane, candidate)), None)
            if entry is None:
                return None
        self._unqueue(entry)
        self._running.append(entry)
        self._active_keys.add(entry.key)
        return entry

    def _leave_running(self, entry: _Entry) -> None:
        with self._lock:
            if entry in self._running:
                self._running.remove(entry)

    def _release_key(self, entry: _Entry) -> None:
        """After the terminal event: a queued job with this key may start now."""
        with self._lock:
            self._active_keys.discard(entry.key)
            waiting = self._queued_by_key.get(entry.key)
            if waiting is not None:
                self._conds[waiting.lane].notify_all()

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
            except BaseException as exc:  # noqa: BLE001 - e.g. a listener raising BaseException
                self._fail_after_runner_error(entry, exc)
            finally:
                self._release_key(entry)

    def _execute(self, entry: _Entry) -> None:
        ctx = entry.ctx
        ctx._send(ctx._event("started"))
        try:
            result = entry.job.run(ctx)
        except BaseException as exc:  # noqa: BLE001 - a job's failure must not end the worker
            terminal = ctx._event("failed", message=_first_line(exc),
                                  error=_format_traceback(exc))
        else:
            outcome = "cancelled" if ctx._saw_cancel() else "finished"
            terminal = ctx._event(outcome, result=result)
        # Leave running() before the terminal event goes out, so a listener
        # reacting to it sees the job gone; this worker starts nothing else,
        # and no job with this key starts, until the event has been delivered.
        self._leave_running(entry)
        ctx._send(terminal, final=True)

    def _fail_after_runner_error(self, entry: _Entry, exc: BaseException) -> None:
        """The job's own path raised (a listener BaseException, a runner bug):
        make sure it still ends with a terminal event. Dropped if it already has one."""
        try:
            logger.error("job runner error around %s", entry.key, exc_info=exc)
            self._leave_running(entry)
            entry.ctx._send(entry.ctx._event("failed", message=_first_line(exc),
                                             error=_format_traceback(exc)), final=True)
        except BaseException:  # the worker must survive
            logger.exception("could not report the failure of %s", entry.key)
