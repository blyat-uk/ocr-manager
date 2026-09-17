"""The OCR run job (RUN lane): real OCR over a snapshot of files.

Replaces the old window's Pipeline/OCRManager/OCRWorker flow for the Stage 3
app, which submits one RunJob on the runner's RUN lane. Nothing here imports
Qt (ruling C8): file workers are plain threads and report through ctx.emit.

Snapshot
    RunJob deep-copies its RunFiles when it is constructed. Edits the user
    makes to the project during the run do not change a file's call.
    Construction raises ValueError when two files would write the same
    output (chi/<stem>.ass, case-sensitive, like the filesystem), e.g. a.mkv
    and a.mp4, or the same name twice; nothing is created on disk.

Output, per file (ruling C5)
    OCR writes chi/<stem>.ass.partial. core.ass_qafix.process_file runs on
    the .partial with its default arguments, then os.replace() moves it onto
    chi/<stem>.ass. A stopped or failed file deletes only its .partial. An
    existing chi/<stem>.ass is never deleted or modified except by that
    replace. A .partial left over from an interrupted run is deleted before
    the file starts, so it can never be promoted.

Call shape (fidelity: today's OCRWorker._run_ocr, one-to-one)
    No range or one range: videocr.api.save_subtitles_to_file(**kwargs,
    file_path=.partial, time_start=, time_end=, callbacks, cancel_event).
    Several ranges: videocr.api.get_subtitles(**kwargs, time_ranges=,
    callbacks, cancel_event); the text is written (UTF-8) only if there is
    some and the file was not stopped. kwargs are OcrCall.kwargs unchanged.

Engines (ruling A1)
    videocr.api leases the engines for each call itself. This module builds
    no engine and never calls suppress_output() or the create_* builders.

File workers, pause and stop (ruling C6)
    `parallel` plain threads take files from the snapshot in order, so at
    most `parallel` files run at once. pause() starts no new file; in-flight
    files continue. Stop is ctx.cancel_event: every in-flight file has its
    own threading.Event, set when the run is stopped, and no file starts
    after that. Nothing is killed; run() joins every file worker without a
    timeout, so no thread is abandoned while it holds an engine lease.

    The runner gives no callback on cancel, and a thread blocked on
    ctx.cancel_event.wait() could not be woken at the end of the run without
    setting the event (which would report the run cancelled). So run()'s own
    thread watches it: while file workers are alive it re-checks
    ctx.cancel_event every STOP_POLL_SECONDS, and a worker ending wakes it at
    once. A paused worker waits on the same condition, so a paused run that is
    stopped returns within about STOP_POLL_SECONDS.

    A run that never read the request reports "finished"; one that read it
    reports "cancelled", even if the request came too late to stop any file.
    Either way the result is the RunSummary, which says what happened to
    each file.

Outcomes
    ok        the final file was replaced; lines = Dialogue lines in it.
    failed    an exception (its first line; the traceback goes to ctx.log),
              or "no subtitles produced" when OCR wrote nothing.
    cancelled the file was stopped mid-way ("cancelled"), or never started
              because the run was stopped (no events for such a file).

Events (JobEvent.type; key and kind are "run")
    "run_file_started"   file=name
    "run_file_progress"  file=name, progress=0..1, message=phase name
                         ("Extracting dialogue" / "Extracting labels");
                         emitted only when the phase or its percent changes
                         (videocr reports every frame)
    "run_subtitle"       file=name, result=(start, end, text)
    "run_file_finished"  file=name, result={"ok": bool, "lines": int, "error": str}
    "log"                a failed file's traceback
    Per file, always: started, then progress/subtitle events, then finished.
"""
from __future__ import annotations

import copy
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from core import ass_qafix as _qafix
from core.jobs.runner import JobContext, Lane, _first_line, _format_traceback
from core.project.ocr_kwargs import OcrCall
from videocr import api as _api

OUTPUT_DIRS = ("chi", "eng", "translate")
PARTIAL_SUFFIX = ".partial"
STOP_POLL_SECONDS = 0.1

ERROR_CANCELLED = "cancelled"
ERROR_NO_SUBTITLES = "no subtitles produced"

_OK, _FAILED, _CANCELLED = "ok", "failed", "cancelled"


@dataclass(frozen=True)
class RunFile:
    """One file of a run: immutable snapshot taken when the run starts."""
    name: str
    call: OcrCall     # from ocr_call_for()


@dataclass(frozen=True)
class RunSummary:
    succeeded: list[str]          # snapshot order
    failed: dict[str, str]        # name -> error, snapshot order
    cancelled: list[str]          # stopped mid-way or never started; snapshot order
    seconds: float                # wall time of run()


class RunJob:
    kind = "run"
    lane = Lane.RUN
    priority = 0
    file = None
    key = "run"

    def __init__(self, project_dir: str, files: list[RunFile], parallel: int):
        parallel = int(parallel)
        if parallel < 1:
            raise ValueError(f"parallel must be at least 1, got {parallel}")
        files = list(files)
        _refuse_shared_outputs(files)
        self.project_dir = project_dir
        self.files: tuple[RunFile, ...] = tuple(copy.deepcopy(files))
        self.parallel = parallel
        # Guards everything below. File workers wait on it while paused, and
        # run() waits on it for them; never held while OCR runs or an event
        # is emitted.
        self._cond = threading.Condition()
        self._paused = False
        self._pending: deque[tuple[int, RunFile]] = deque()
        self._in_flight: set[threading.Event] = set()
        self._stopping = False
        self._workers_left = 0
        self._outcomes: dict[int, tuple[str, str]] = {}

    def pause(self) -> None:
        """Start no new files; in-flight files continue."""
        with self._cond:
            self._paused = True

    def resume(self) -> None:
        with self._cond:
            self._paused = False
            self._cond.notify_all()

    def run(self, ctx: JobContext) -> RunSummary:
        started = time.perf_counter()
        for sub in OUTPUT_DIRS:
            os.makedirs(os.path.join(self.project_dir, sub), exist_ok=True)

        workers = [threading.Thread(target=self._work, args=(ctx,), name=f"run-file-{index}", daemon=True)
                   for index in range(min(self.parallel, len(self.files)))]
        with self._cond:
            self._pending = deque(enumerate(self.files))
            self._in_flight = set()
            self._stopping = False
            self._outcomes = {}
            self._workers_left = len(workers)

        running: list[threading.Thread] = []
        try:
            for worker in workers:
                worker.start()
                running.append(worker)
        except BaseException:
            # Stop what did start, cooperatively, before reporting the failure.
            with self._cond:
                self._workers_left -= len(workers) - len(running)
                self._stop_locked()
            self._wait_for_workers(ctx, running)
            raise
        self._wait_for_workers(ctx, running)

        succeeded: list[str] = []
        failed: dict[str, str] = {}
        cancelled: list[str] = []
        with self._cond:
            outcomes = dict(self._outcomes)
        for index, run_file in enumerate(self.files):
            status, error = outcomes.get(index, (_CANCELLED, ERROR_CANCELLED))
            if status == _OK:
                succeeded.append(run_file.name)
            elif status == _FAILED:
                failed[run_file.name] = error
            else:
                cancelled.append(run_file.name)
        return RunSummary(succeeded, failed, cancelled, time.perf_counter() - started)

    # --- file workers -----------------------------------------------------------

    def _wait_for_workers(self, ctx: JobContext, workers: list[threading.Thread]) -> None:
        with self._cond:
            while self._workers_left > 0:
                if not self._stopping and ctx.cancel_event.is_set():
                    self._stop_locked()
                self._cond.wait(STOP_POLL_SECONDS)
        for worker in workers:
            worker.join()

    def _stop_locked(self) -> None:
        """self._cond held. Start no more files and stop every in-flight one."""
        self._stopping = True
        for cancel_event in self._in_flight:
            cancel_event.set()
        self._cond.notify_all()

    def _work(self, ctx: JobContext) -> None:
        try:
            while (taken := self._take(ctx)) is not None:
                index, run_file, cancel_event = taken
                outcome = None
                try:
                    outcome = self._run_file(ctx, run_file, cancel_event)
                finally:
                    with self._cond:
                        self._in_flight.discard(cancel_event)
                        if outcome is not None:
                            self._outcomes[index] = outcome
        finally:
            with self._cond:
                self._workers_left -= 1
                self._cond.notify_all()

    def _take(self, ctx: JobContext) -> tuple[int, RunFile, threading.Event] | None:
        """The next file with its own cancel event, or None when there is none to start."""
        with self._cond:
            while True:
                if not self._pending or self._stopping:
                    return None
                if ctx.cancel_event.is_set():
                    self._stop_locked()
                    return None
                if not self._paused:
                    break
                self._cond.wait()
            index, run_file = self._pending.popleft()
            cancel_event = threading.Event()
            self._in_flight.add(cancel_event)
            return index, run_file, cancel_event

    # --- one file -------------------------------------------------------------

    def _run_file(self, ctx: JobContext, run_file: RunFile,
                  cancel_event: threading.Event) -> tuple[str, str]:
        name = run_file.name
        final = os.path.join(self.project_dir, "chi", _output_name(name))
        partial = final + PARTIAL_SUFFIX
        ctx.emit("run_file_started", file=name)
        lines = 0
        try:
            _remove(partial)     # a leftover of an interrupted run is not this run's output
            self._ocr(ctx, run_file, partial, cancel_event)
            if cancel_event.is_set():
                _remove(partial)
                status, error = _CANCELLED, ERROR_CANCELLED
            elif os.path.exists(partial):
                _qafix.process_file(partial)
                os.replace(partial, final)
                lines = _count_dialogue_lines(final)
                status, error = _OK, ""
            else:
                status, error = _FAILED, ERROR_NO_SUBTITLES
        except Exception as exc:  # noqa: BLE001 - one file's failure must not stop the others
            status, error = _FAILED, _first_line(exc)
            try:
                _remove(partial)
            except Exception as cleanup:  # noqa: BLE001
                ctx.log(f"{name}: could not delete {partial}: {_first_line(cleanup)}")
            ctx.log(f"{name}: failed\n{_format_traceback(exc)}")
        ctx.emit("run_file_finished", file=name,
                 result={"ok": status == _OK, "lines": lines, "error": error})
        return status, error

    def _ocr(self, ctx: JobContext, run_file: RunFile, partial: str,
             cancel_event: threading.Event) -> None:
        name = run_file.name
        kwargs = copy.deepcopy(run_file.call.kwargs)
        time_ranges = list(run_file.call.time_ranges)
        progress_callback = _progress_reporter(ctx, name)

        def subtitle_callback(start, end, text):
            ctx.emit("run_subtitle", file=name, result=(start, end, text))

        if len(time_ranges) <= 1:
            _api.save_subtitles_to_file(
                **kwargs,
                file_path=partial,
                time_start=time_ranges[0][0] if time_ranges else "0:00",
                time_end=time_ranges[0][1] if time_ranges else "",
                progress_callback=progress_callback,
                subtitle_callback=subtitle_callback,
                cancel_event=cancel_event,
            )
        else:
            text = _api.get_subtitles(
                **kwargs,
                time_ranges=time_ranges,
                progress_callback=progress_callback,
                subtitle_callback=subtitle_callback,
                cancel_event=cancel_event,
            )
            if text and not cancel_event.is_set():
                with open(partial, "w", encoding="utf-8") as f:
                    f.write(text)


def _output_name(name: str) -> str:
    """The file a video's run writes into chi/: <stem>.ass, stem as today's OCRWorker took it."""
    return Path(name).stem + ".ass"


def _refuse_shared_outputs(files: list[RunFile]) -> None:
    """ValueError naming every group of files that would write the same chi/ output."""
    by_output: dict[str, list[str]] = {}
    for run_file in files:
        by_output.setdefault(_output_name(run_file.name), []).append(run_file.name)
    collisions = []
    for output, names in sorted(by_output.items()):
        if len(names) > 1:
            names = sorted(names)
            collisions.append(f"{', '.join(names[:-1])} and {names[-1]} both write chi/{output}")
    if collisions:
        raise ValueError("; ".join(collisions))


def _progress_reporter(ctx: JobContext, name: str):
    """videocr's progress_callback(phase_name, percent) for one file.

    videocr reports every frame, from more than one thread; this emits only
    when the phase or the percent changes, one report at a time.
    """
    lock = threading.Lock()
    last: list[tuple[str, int] | None] = [None]

    def report(phase_name, percent):
        with lock:
            if last[0] == (phase_name, percent):
                return
            last[0] = (phase_name, percent)
            fraction = min(1.0, max(0.0, percent / 100.0))
            ctx.emit("run_file_progress", file=name, progress=fraction, message=phase_name)

    return report


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _count_dialogue_lines(path: str) -> int:
    with open(path, encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.startswith("Dialogue:"))
