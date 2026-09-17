"""ProjectController: the one object the window's views talk to (plan 3B).

It owns the open folder's Project, the JobRunner and the AutoPilot. It is the
only code that mutates the model, always on the GUI thread (ruling C8).
Views read through it and call its commands; they never touch core/.

Events
    The runner's listener only puts each JobEvent on a queue.SimpleQueue and
    returns: it never calls the runner and never blocks. A 30 ms QTimer drains
    the queue on the GUI thread (drain_events). For each terminal event of an
    auto-pilot kind (metadata, thumbnail, crop, brightness, ranges,
    audio_profile), in this order (AutoPilot's owner protocol):
      1. current = autopilot.is_current(event), before anything else;
      2. when current, the result is applied by type with core.jobs.apply
         (a thumbnail becomes an in-memory QImage copy); a superseded result
         is dropped;
      3. autopilot.on_job_event(event), after the apply, so the scheduler sees
         the applied values;
      4. recompute_all(project, pending=..., ranges_pending=...);
      5. signals, and a debounced save.
    Proof, run and view (frames/strips) events have their own branches.
    Events of any other kind are logged to "Pipeline". A failed detection is
    logged under "Detections", never retried: re-detect is the user's retry.
    An apply that raises is logged to "Pipeline" and steps 3-4 still run, so
    the key never stays outstanding. The controller never submits an
    auto-pilot kind itself: AutoPilot counts every submission per key.

Frames and strips
    request_frames / request_strips submit their own CPU-lane jobs (not
    through AutoPilot: they are nobody's pipeline step) for the times that
    are neither cached nor already on their way, so a view may call them on
    every repaint. Each job's terminal event caches what came back, releases
    the times it held and emits frame_ready / strips_ready. A time that could
    not be read is remembered as unavailable (a finished job marks what is
    missing from its result, a failed one everything it was asked for), so it
    is attempted once per session and not once per repaint; a CANCELLED job
    marks nothing, since nothing was learned. The two sources are never mixed
    -- whole frames for the crop views, OCR-exact strips (keyed by the crop
    box they were grabbed with) for anything showing OCR pixels. The cache,
    markers included, is dropped per file when the file disappears, its
    strips when its crop box changes, and all of it when the folder closes.

    These jobs are deliberately kept out of the ActivityTracker: a view
    repainting must not put "frames" in the activity strip or push the
    detectors out of its five-deep history.

Sessions
    One runner lives as long as the controller. Closing a folder cancels its
    jobs cooperatively. The runner delivers "queued" before submit() returns,
    and close drains the queue, so every event a closed folder's jobs send
    later has a job_id no newer than the last one seen then; those events are
    dropped.

Edits
    set_manual_* / mark_reviewed / set_skipped / paste_settings, then
    on_crop_changed when the crop box changed, then recompute_all (apply
    stores placeholder states only a recompute corrects), file_changed and a
    debounced save.

Auto-pilot holds
    JobRunner.resume() clears every hold on a lane, so two flags decide:
    the user's "pause auto-pilot" and "a run is in progress". AutoPilot is
    paused while either is set and resumed only when both are clear.

Run
    start_run snapshots each file's OCR call (ocr_call_for), holds auto-pilot
    and submits a RunJob; nothing is deleted (ruling C5). Per-file outcomes
    come from run_file_finished events and the RunSummary. Changing
    ocr_parallel during the run reaches the run job (RunJob.set_parallel):
    files not yet started follow the new limit, in-flight files carry on.
    When the run ends:
    auto-pilot holds are re-derived, done states re-read, the folder is
    reconciled (the watcher ignores changes while a run writes chi/) and,
    unless the user stopped it, notify-send reports it as today.

Saving
    save_project, debounced (500 ms); close_folder and shutdown save at once.
    A project that failed to load is never installed, so never saved.

Views import no core module: UnsupportedProjectVersion (raised by
open_folder) and DETECTION_KINDS are re-exported here.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import queue
import subprocess
import time
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QImage

from app.activity import TERMINAL_EVENTS, ActivitySnapshot, ActivityTracker
from app.folder_watch import OUTPUT_DIR, FolderWatch
from app.imaging import FrameCache, bgr_to_qimage
from app.logbook import DETECTIONS_LOG, PIPELINE_LOG, LogBook
from app.run_snapshot import DONE, FAILED, RunSnapshot, RunTracker, notification_for
from app.state_text import badge_for
from core.jobs import apply as rules
from core.jobs.autopilot import AUTOPILOT_KINDS, DETECTION_KINDS, AutoPilot
from core.jobs.detect_jobs import (
    AudioProfileResult,
    BrightnessJobResult,
    CropJobResult,
    MetadataResult,
    ProofOcrJob,
    ProofResult,
    RangesJobResult,
    ThumbnailResult,
)
from core.jobs.run import RunFile, RunJob, RunSummary, output_name
from core.jobs.runner import JobEvent, JobRunner
from core.jobs.view_jobs import FrameJob, FramesResult, StripJob, StripsResult
from core.project import store
from core.project.model import FileEntry, FolderSettings, Project, ReviewState
from core.project.ocr_kwargs import ocr_call_for
from core.project.store import UnsupportedProjectVersion

if TYPE_CHECKING:
    import numpy as np                      # only for the frame/strip annotations

logger = logging.getLogger(__name__)

BOTH_OFF_MESSAGE = "At least one of dialogue or labels must be on."
PARALLEL_MESSAGE = "Parallel files must be at least 1."
READY_STATES = frozenset({ReviewState.PROPOSED, ReviewState.REVIEWED})
BADGE_SKIPPED, BADGE_DONE = "skipped", "done"          # app.state_text.badge_for's texts for those rows
# The chip a row counts under, once its badge is neither "skipped" nor "done":
# "reviewed", a "check ..." badge (FLAGGED) or a pending badge. PROPOSED ("ready") has no chip.
_BADGE_BUCKETS = {ReviewState.REVIEWED: "reviewed", ReviewState.FLAGGED: "needs_you",
                  ReviewState.PENDING: "detecting"}
# Jobs a view asks for directly (core/jobs/view_jobs.py): pixels for the crop,
# brightness and time-range views. Not auto-pilot kinds -- the controller
# submits them itself -- and not activity: a view repainting must not push the
# detectors out of the activity strip.
VIEW_KINDS = frozenset({"frames", "strips"})
MAX_EVENTS_PER_DRAIN = 2000
MAX_DRAINS_AT_CLOSE = 50                 # close/shutdown apply what arrived, without chasing a busy runner forever
NOTIFY_TIMEOUT_SECONDS = 10
_FOLDER_FIELDS = frozenset(field.name for field in dataclasses.fields(FolderSettings))


def default_runner_factory(on_event: Callable[[JobEvent], None]) -> JobRunner:
    return JobRunner(on_event, cpu_workers=2)


def _box(crop) -> tuple[int, int, int, int] | None:
    return None if crop is None else (crop.x, crop.y, crop.width, crop.height)


def _frame_key(name: str, time: float) -> tuple:
    """FrameCache key of a whole frame."""
    return (name, "frame", float(time))


def _strip_key(name: str, crop_box, time: float) -> tuple:
    """FrameCache key of an OCR-exact strip: the crop box is part of it, so
    strips of another box are other entries."""
    return (name, "strip", tuple(int(value) for value in crop_box), float(time))


def _same_folder(first: str, second: str) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return os.path.realpath(first) == os.path.realpath(second)


def _non_empty_file(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


class ProjectController(QObject):
    # Job kinds that are detections (crop, brightness, ranges, audio_profile): what "pause auto-pilot"
    # holds and what the top bar's "detecting" dot means (ruling B7). Metadata and thumbnails are not.
    DETECTION_KINDS = DETECTION_KINDS

    project_opened = pyqtSignal(str)            # folder path
    project_closed = pyqtSignal()
    files_changed = pyqtSignal()                # entries added/removed/reordered
    file_changed = pyqtSignal(str)              # one entry's values/state/evidence/done changed
    folder_changed = pyqtSignal()               # FolderSettings changed
    activity_changed = pyqtSignal()             # running/queued jobs or auto-pilot holds changed
    thumbnail_ready = pyqtSignal(str)
    frame_ready = pyqtSignal(str, float)        # file, time: frame(file, time) now has pixels
    strips_ready = pyqtSignal(str)              # file: strips of the requested crop box arrived
    proof_started = pyqtSignal(str)
    proof_finished = pyqtSignal(str)            # result in proof_result(file)
    run_changed = pyqtSignal()                  # run started/progress/finished/paused
    run_subtitle = pyqtSignal(str, float, float, str)
    log_appended = pyqtSignal(str, str)         # key ("Pipeline" or filename), text
    logs_cleared = pyqtSignal()                 # logs restarted (a run start keeps "Detections"): re-read log_keys()
    save_failed = pyqtSignal(str)               # message; the values stay in memory

    def __init__(self, runner_factory: Callable[[Callable[[JobEvent], None]], JobRunner] = default_runner_factory,
                 parent: QObject | None = None, *, save_debounce_ms: int = 500, drain_interval_ms: int = 30,
                 watch_debounce_ms: int = 300):
        super().__init__(parent)
        self._events: queue.SimpleQueue[JobEvent] = queue.SimpleQueue()
        self._runner = runner_factory(self._events.put)       # the listener only enqueues
        self._project: Project | None = None
        self._autopilot: AutoPilot | None = None
        self._fence: int | None = None       # events with job_id <= fence belong to a closed folder
        self._last_job_id = 0
        self._shut_down = False
        self._draining = False
        self._activity = ActivityTracker()
        self._logs = LogBook()
        self._reset_session()
        self._reset_emits()

        self._drain_timer = QTimer(self)
        self._drain_timer.setInterval(drain_interval_ms)
        self._drain_timer.timeout.connect(self.drain_events)
        self._drain_timer.start()
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(save_debounce_ms)
        self._save_timer.timeout.connect(self._save_now)
        self._folder_watch = FolderWatch(watch_debounce_ms, self)
        self._folder_watch.settled.connect(self._on_folder_settled)

    def _reset_session(self) -> None:
        self._thumbnails: dict[str, QImage] = {}
        self._frames = FrameCache()
        self._frame_inflight: set[tuple] = set()        # cache keys a frames/strips job is fetching
        self._view_jobs: dict[str, list[tuple]] = {}    # job key -> the cache keys it was submitted for
        self._proof_results: dict[str, ProofResult] = {}
        self._proof_outstanding: dict[str, int] = {}
        self._proof_stale: dict[str, int] = {}          # proofs of removed entries, still to end: never reported
        self._clipboard: dict | None = None
        self._done: set[str] = set()
        self._user_paused = False
        self._run: RunTracker | None = None
        self._run_job: RunJob | None = None
        self._run_active = False
        self._run_stop_requested = False
        self._save_blocked = False
        self._activity.clear()

    def _reset_emits(self) -> None:
        self._emit_files = False
        self._emit_folder = False
        self._emit_activity = False
        self._emit_run = False
        self._emit_changed: set[str] = set()
        self._emit_thumbnails: list[str] = []
        self._emit_frames: list[tuple[str, float]] = []
        self._emit_strips: list[str] = []
        self._emit_proofs: list[str] = []

    # --- lifecycle ------------------------------------------------------------------

    def open_folder(self, path: str) -> None:
        """Open `path`. It is loaded first: when its `.ocr.json` has a version
        this app cannot read, UnsupportedProjectVersion is raised and the open
        folder stays open, untouched (not even saved). Only after a successful
        load is the open folder closed (and saved) and the new one installed;
        reopening the open folder reads it again after that save, so no edit
        is lost."""
        self._check_alive()
        path = os.path.abspath(os.fspath(path))
        if not os.path.isdir(path):
            raise NotADirectoryError(path)
        project = store.load_project(path)
        reopening = self._project is not None and _same_folder(self._project.path, path)
        self.close_folder()
        if reopening:
            project = store.load_project(path)
        self._project = project
        self._autopilot = AutoPilot(self._runner, self._current_project)
        self._refresh_done(notify=False)
        self._folder_watch.watch(path)
        self._emit_files = self._emit_folder = self._emit_activity = True
        self._autopilot.on_open()
        self._recompute()
        if project.migrated_from_v1:
            self._schedule_save()
        self.project_opened.emit(path)
        self._flush()

    def close_folder(self) -> None:
        """Save, cancel the folder's jobs cooperatively and forget it."""
        if self._project is None:
            return
        self._drain_all()
        self._save_now()
        self._runner.cancel_where(lambda job: True)
        self._autopilot.resume()            # lift this folder's holds from the lanes
        self._discard_events()
        self._fence = self._last_job_id
        self._save_timer.stop()
        self._folder_watch.stop()
        self._project = None
        self._autopilot = None
        self._reset_session()
        self._reset_emits()
        self._clear_logs()
        self.project_closed.emit()

    def shutdown(self, timeout: float = 10.0) -> bool:
        """Save synchronously, then shut the runner down cooperatively; True
        when every worker thread exited within `timeout`."""
        if self._shut_down:
            return True
        if self._project is not None:
            self._drain_all()
            self._save_now()
        self._shut_down = True
        self._drain_timer.stop()
        self._save_timer.stop()
        self._folder_watch.stop()
        return bool(self._runner.shutdown(timeout))

    # --- reading ----------------------------------------------------------------------

    @property
    def project(self) -> Project | None:
        return self._project

    def entry(self, name: str) -> FileEntry:
        return self._require()[0].files[name]

    def names(self) -> list[str]:
        return [] if self._project is None else list(self._project.files)

    def is_done(self, name: str) -> bool:
        """chi/<output_name(name)> exists and is not empty (the run's stem rule)."""
        return name in self._done

    @staticmethod
    def is_detection_kind(kind: str) -> bool:
        return kind in DETECTION_KINDS

    def running_detectors(self, name: str) -> set[str]:
        return self._activity.running_kinds(name) & AUTOPILOT_KINDS

    def pending_detectors(self) -> dict[str, set[str]]:
        """file -> the detection kinds still to come for it: queued, running,
        or held by auto-pilot until something else finishes (a brightness
        measurement waits for the folder's ranges analysis, so it has no job
        yet). `AutoPilot.pending()`, the same map the review states are
        recomputed from; a fresh dict, empty with no folder open."""
        return {} if self._project is None else self._autopilot.pending()

    def thumbnail(self, name: str) -> QImage | None:
        return self._thumbnails.get(name)

    def proof_result(self, name: str) -> ProofResult | None:
        return self._proof_results.get(name)

    def proof_pending(self, name: str) -> bool:
        return self._proof_outstanding.get(name, 0) > 0

    def activity(self) -> ActivitySnapshot:
        return self._activity.snapshot(paused=self._user_paused, held=self._user_paused or self._run_active)

    def counts(self) -> dict[str, int]:
        """Chip counts that agree with the row badges (ruling B10):
        "reviewed" (badge "reviewed"), "needs_you" (a "check ..." badge),
        "detecting" (a pending badge: "finding subtitles…", "waiting", ...).
        A row badged "skipped" or "done" counts in none of those three.
        "ready": PROPOSED or REVIEWED, not skipped, not done."""
        counts = {"reviewed": 0, "needs_you": 0, "detecting": 0, "ready": 0}
        if self._project is None:
            return counts
        for name, entry in self._project.files.items():
            done = name in self._done
            # run_state=None: a run's "running"/"failed" badge is transient, so a
            # file in a run stays in the bucket of its review state.
            text, _tone = badge_for(entry, running_detectors=self.running_detectors(name), done=done,
                                    run_state=None)
            if text not in (BADGE_SKIPPED, BADGE_DONE):
                bucket = _BADGE_BUCKETS.get(entry.review)     # badge_for's remaining rungs are the review state
                if bucket is not None:
                    counts[bucket] += 1
            if entry.review in READY_STATES and not entry.skipped and not done:
                counts["ready"] += 1
        return counts

    def run_snapshot(self) -> RunSnapshot | None:
        return None if self._run is None else self._run.snapshot()

    def run_subtitles(self, name: str) -> list[tuple[float, float, str]]:
        """Every subtitle the current (or last) run reported for `name`."""
        return [] if self._run is None else self._run.subtitles(name)

    def log_keys(self) -> list[str]:
        return self._logs.keys()

    def log_text(self, key: str) -> str:
        return self._logs.text(key)

    def append_log(self, key: str, text: str) -> None:
        """Append `text` to the log `key` ("Pipeline", "Detections" or a file
        name), as job events do: the window records unexpected errors here."""
        self._log(key, text)

    def can_paste(self) -> bool:
        return self._clipboard is not None and any(value is not None for value in self._clipboard.values())

    # --- frames and strips for the review views ---------------------------------------

    def request_frames(self, name: str, times: list[float]) -> None:
        """Fetch `name`'s whole frames at `times` for the crop views.

        Only the times that are worth fetching are: the ones not cached, not
        already on their way and not already found unreadable. A view may
        therefore call this on every repaint. frame_ready(name, time) follows
        for each frame that arrives; a time that could not be read stays None
        (the view draws its placeholder) and is not asked for again until the
        file or the folder is reopened.

        A view asks from paintEvent, so with no folder open (or after
        shutdown) this does nothing rather than raise. An unknown file, while
        a folder is open, is a bug: KeyError.
        """
        project = self._view_project(name)
        if project is None:
            return
        wanted, keys = self._missing_times(times, lambda time_value: _frame_key(name, time_value))
        if wanted:
            self._submit_view_job(FrameJob(project.path, name, wanted), keys)

    def frame(self, name: str, time: float) -> np.ndarray | None:
        """The cached whole frame at `time` (a BGR numpy array), or None --
        for a time not fetched yet, one still on its way, and one that could
        not be read."""
        return self._frames.get(_frame_key(name, time))

    def request_strips(self, name: str, crop_box: tuple[int, int, int, int], times: list[float]) -> None:
        """Fetch `name`'s OCR-exact crop strips at `times`, for `crop_box`.

        Same rules as request_frames; strips_ready(name) follows once the
        strips of a request have arrived. Strips are cached per crop box, so
        editing the crop never shows strips measured on the old one -- and a
        time that could not be read for one box is asked for again for the
        next, since the box is part of the key. With no folder open (or after
        shutdown) this does nothing, as request_frames does.
        """
        project = self._view_project(name)
        if project is None:
            return
        box = tuple(int(value) for value in crop_box)
        wanted, keys = self._missing_times(times, lambda time_value: _strip_key(name, box, time_value))
        if wanted:
            self._submit_view_job(StripJob(project.path, name, box, wanted), keys)

    def strip(self, name: str, crop_box: tuple[int, int, int, int], time: float) -> np.ndarray | None:
        """The cached OCR-exact strip at `time` for `crop_box`, or None (same
        three cases as frame())."""
        return self._frames.get(_strip_key(name, crop_box, time))

    def _missing_times(self, times: list[float],
                       key_of: Callable[[float], tuple]) -> tuple[list[float], list[tuple]]:
        """The times of `times` that are worth fetching -- not cached, not
        already being fetched and not known to be unreadable -- with their
        cache keys, marked as being fetched now. A repeated time counts
        once."""
        wanted, keys = [], []
        for value in times:
            time_value = float(value)
            key = key_of(time_value)
            if self._frames.knows(key) or key in self._frame_inflight:
                continue
            self._frame_inflight.add(key)
            wanted.append(time_value)
            keys.append(key)
        return wanted, keys

    def _view_project(self, name: str) -> Project | None:
        """The open project, or None when there is nothing to fetch from: no
        folder open, or the controller shut down. A view repaints on its own
        schedule -- possibly between close_folder() and hearing about it --
        and a repaint must never raise."""
        if self._shut_down or self._project is None:
            return None
        if name not in self._project.files:
            raise KeyError(name)
        return self._project

    def _submit_view_job(self, job, keys: list[tuple]) -> None:
        try:
            self._runner.submit(job)
        except BaseException:
            self._frame_inflight.difference_update(keys)
            raise
        # Extend rather than replace: should two requests ever share a key
        # (the same file and times), one terminal event releases both sets.
        self._view_jobs.setdefault(job.key, []).extend(keys)

    # --- edits ------------------------------------------------------------------------

    def set_crop(self, name: str, box: tuple[int, int, int, int]) -> None:
        project, _ = self._require()
        before = _box(project.files[name].crop)
        rules.set_manual_crop(project, name, box)
        if _box(project.files[name].crop) != before:
            self._crop_box_changed(name)
        self._after_edit(name)

    def _crop_box_changed(self, name: str) -> None:
        """The file's crop box is a new one: re-detect what depends on it and
        drop the strips grabbed with the old box (they are keyed by it, so
        they would otherwise sit in the cache until they aged out).

        A strips job already fetching the old box is left to finish; its
        result lands under that box's keys, which nothing asks for again, and
        the LRU ages it out."""
        self._autopilot.on_crop_changed(name)
        self._frames.clear_file(name, "strip")

    def set_brightness(self, name: str, value: int) -> None:
        project, _ = self._require()
        rules.set_manual_brightness(project, name, value)
        self._after_edit(name)

    def set_time_ranges(self, name: str, ranges: list[tuple[str | None, str | None]] | None) -> None:
        project, _ = self._require()
        rules.set_manual_time_ranges(project, name, ranges)
        self._after_edit(name)

    def mark_reviewed(self, name: str, reviewed: bool = True) -> None:
        project, _ = self._require()
        rules.mark_reviewed(project, name, reviewed)
        self._after_edit(name)

    def set_skipped(self, name: str, skipped: bool) -> None:
        project, _ = self._require()
        rules.set_skipped(project, name, skipped)
        self._after_edit(name)

    def copy_settings(self, name: str) -> None:
        project, _ = self._require()
        self._clipboard = rules.copy_settings(project, name)

    def paste_settings(self, name: str) -> bool:
        """False when the clipboard holds nothing to paste."""
        project, _ = self._require()
        before = _box(project.files[name].crop)
        if not self.can_paste():
            return False
        rules.paste_settings(project, name, self._clipboard)
        if _box(project.files[name].crop) != before:
            self._crop_box_changed(name)
        self._after_edit(name)
        return True

    def update_folder(self, **changes) -> None:
        """Replace FolderSettings fields. TypeError for an unknown field;
        ValueError when dialogue and labels would both be off, or when
        ocr_parallel is below 1. A new ocr_parallel during a run also goes to
        the run job."""
        project, autopilot = self._require()
        unknown = sorted(set(changes) - _FOLDER_FIELDS)
        if unknown:
            raise TypeError(f"unknown folder settings: {', '.join(unknown)}")
        old = dataclasses.replace(project.folder)
        if not changes.get("dialogue_enabled", old.dialogue_enabled) and \
                not changes.get("labels_enabled", old.labels_enabled):
            raise ValueError(BOTH_OFF_MESSAGE)
        if "ocr_parallel" in changes and int(changes["ocr_parallel"]) < 1:
            raise ValueError(PARALLEL_MESSAGE)
        if all(getattr(old, key) == value for key, value in changes.items()):
            return
        before = self._states()
        for key, value in changes.items():
            setattr(project.folder, key, value)
        new = project.folder
        if new.ocr_parallel != old.ocr_parallel:
            self._set_run_parallel(new.ocr_parallel)
        rules.apply_folder_change(project, old, new)
        autopilot.on_folder_changed(old, new)
        self._emit_folder = True
        self._recompute(before)
        self._schedule_save()
        self._flush()

    def set_label_masks(self, masks: list[tuple[int, int, int, int]]) -> None:
        crops = [tuple(int(value) for value in mask) for mask in masks]
        if any(len(crop) != 4 for crop in crops):
            raise ValueError("a label mask is (x, y, width, height)")
        self.update_folder(label_mask_crops=crops)

    # --- detections ---------------------------------------------------------------------

    def redetect(self, name: str) -> None:
        project, autopilot = self._require()
        if name not in project.files:
            raise KeyError(name)
        autopilot.redetect(name)
        self._recompute()
        self._flush()

    def redetect_others_with_hint(self, name: str, what: str) -> None:
        """what: "crop" or "brightness" (ruling C3)."""
        project, autopilot = self._require()
        if name not in project.files:
            raise KeyError(name)
        if what == "crop":
            autopilot.redetect_others_with_crop_hint(name)
        elif what == "brightness":
            autopilot.redetect_others_with_brightness_hint(name)
        else:
            raise ValueError(f"hint re-detects are for 'crop' or 'brightness', not {what!r}")
        self._recompute()
        self._flush()

    def hint_targets(self, name: str, what: str) -> list[str]:
        return self._require()[1].hint_targets(name, what)

    def run_proof(self, name: str) -> None:
        """Real OCR of a 30 s window (ruling C4). ValueError while the file's
        duration is unknown."""
        self._check_alive()
        project, _ = self._require()
        self._runner.submit(ProofOcrJob(project.path, project.files[name], project.folder))
        self._proof_outstanding[name] = self._proof_outstanding.get(name, 0) + 1
        self._proof_results.pop(name, None)
        self.proof_started.emit(name)

    def pause_autopilot(self) -> None:
        self._set_user_pause(True)

    def resume_autopilot(self) -> None:
        self._set_user_pause(False)

    def _set_user_pause(self, paused: bool) -> None:
        if self._project is None:
            return
        self._user_paused = paused
        self._sync_autopilot_hold()
        self._flush()

    def _sync_autopilot_hold(self) -> None:
        if self._user_paused or self._run_active:
            self._autopilot.pause()
        else:
            self._autopilot.resume()
        self._emit_activity = True

    # --- run ------------------------------------------------------------------------------

    def startable_files(self, include_flagged: bool = False, *, include_done: bool = False) -> list[str]:
        """Not skipped, not in the running run, PROPOSED or REVIEWED (FLAGGED
        too with include_flagged), and not done unless include_done."""
        if self._project is None:
            return []
        states = READY_STATES | ({ReviewState.FLAGGED} if include_flagged else set())
        in_run = set(self._run.snapshot().active_names) if self._run_active else set()
        return [name for name, entry in self._project.files.items()
                if entry.review in states and not entry.skipped and name not in in_run
                and (include_done or name not in self._done)]

    def files_needing_overwrite(self, names: list[str]) -> list[str]:
        return [name for name in names if name in self._done]

    def start_run(self, names: list[str]) -> None:
        """Snapshot each file's OCR call, hold auto-pilot and submit a RunJob.
        ValueError when two files write the same chi/ output (RunJob's
        message), when no file is given or when both extraction toggles are
        off; RuntimeError while a run is in progress."""
        self._check_alive()
        project, _ = self._require()
        if self._run_active:
            raise RuntimeError("a run is already in progress")
        folder = project.folder
        if not folder.dialogue_enabled and not folder.labels_enabled:
            raise ValueError(BOTH_OFF_MESSAGE)
        names = list(names)
        if not names:
            raise ValueError("no files to run")
        files = [RunFile(name, ocr_call_for(project.files[name], folder, project.path)) for name in names]
        job = RunJob(project.path, files, folder.ocr_parallel)
        self._run_active = True
        self._sync_autopilot_hold()
        try:
            self._runner.submit(job)
        except BaseException:
            self._run_active = False
            self._sync_autopilot_hold()
            self._flush()
            raise
        self._run_job = job
        self._run_stop_requested = False
        self._run = RunTracker(names, folder.ocr_parallel, time.monotonic())
        self._clear_logs(keep=(DETECTIONS_LOG,))
        self._emit_run = True
        self._flush()

    def pause_run(self) -> None:
        self._set_run_paused(True)

    def resume_run(self) -> None:
        self._set_run_paused(False)

    def _set_run_paused(self, paused: bool) -> None:
        if not self._run_active:
            return
        if paused:
            self._run_job.pause()
        else:
            self._run_job.resume()
        self._run.set_paused(paused)
        self._emit_run = True
        self._flush()

    def _set_run_parallel(self, parallel: int) -> None:
        """The running run job takes `parallel` files at once from now on.
        A job that cannot add workers keeps running with those it has; the
        error is logged and the folder setting stays."""
        if not self._run_active:
            return
        try:
            self._run_job.set_parallel(parallel)
        except Exception as exc:                        # e.g. no thread could be started
            logger.warning("could not change the run's parallel files: %s", exc)
            self._log(PIPELINE_LOG, f"Could not run {parallel} files at once: {exc}")
            return
        self._run.set_parallel(parallel)
        self._emit_run = True

    def stop_run(self) -> None:
        """Cooperative: every in-flight file's cancel is set (ruling C6)."""
        if not self._run_active:
            return
        self._run_stop_requested = True
        self._run.set_stopping()
        self._runner.cancel("run")
        self._emit_run = True
        self._flush()

    # --- the event drain ------------------------------------------------------------------

    def drain_events(self) -> None:
        """Handle the events the runner queued (the drain timer's slot)."""
        if self._draining:
            return
        self._draining = True
        try:
            for _ in range(MAX_EVENTS_PER_DRAIN):
                try:
                    event = self._events.get_nowait()
                except queue.Empty:
                    break
                self._last_job_id = max(self._last_job_id, event.job_id)
                if self._project is None or (self._fence is not None and event.job_id <= self._fence):
                    continue
                try:
                    self._handle_event(event)
                except Exception:           # one bad event must not stop the drain
                    logger.exception("could not handle %r for %s", event.type, event.key)
                    self._log(PIPELINE_LOG, f"Internal error while handling {event.type} of {event.key}")
        finally:
            self._draining = False
        self._flush()

    def _drain_all(self) -> None:
        for _ in range(MAX_DRAINS_AT_CLOSE):
            if self._draining or self._events.empty():
                return
            self.drain_events()

    def _discard_events(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            self._last_job_id = max(self._last_job_id, event.job_id)

    def _handle_event(self, event: JobEvent) -> None:
        if event.kind in VIEW_KINDS:                    # a view's own fetches are not activity
            self._on_view_event(event)
            return
        if self._activity.on_event(event, time.monotonic()):
            self._emit_activity = True
        if event.kind == "run":
            self._on_run_event(event)
        elif event.kind == "proof":
            self._on_proof_event(event)
        elif event.kind in AUTOPILOT_KINDS:
            self._on_detection_event(event)
        else:
            self._log(PIPELINE_LOG, f"Unexpected {event.type!r} event from job {event.key} (kind {event.kind!r})")

    def _on_detection_event(self, event: JobEvent) -> None:
        if event.type == "log":
            self._log(event.file or PIPELINE_LOG, event.message)
            return
        if event.type not in TERMINAL_EVENTS:
            return
        autopilot = self._autopilot
        current = autopilot.is_current(event)           # 1. before anything consumes the submission
        if current:
            try:
                self._apply_result(event.result)        # 2. a superseded result is dropped
            except Exception:                           # the key must still be released below
                logger.exception("could not apply the result of %s", event.key)
                self._log(PIPELINE_LOG, f"Could not apply the result of {event.key}:\n{traceback.format_exc()}")
        autopilot.on_job_event(event)                   # 3. after the apply
        self._recompute()                               # 4.
        if event.type == "failed":
            self._log(DETECTIONS_LOG, f"{event.file or 'folder'}: {event.kind} failed: {event.message}\n{event.error}")

    def _apply_result(self, result) -> None:
        project = self._project
        if result is None:
            return
        if isinstance(result, ThumbnailResult):
            image = bgr_to_qimage(result.image)
            if image is not None and result.file in project.files:     # None: keep the placeholder
                self._thumbnails[result.file] = image
                self._emit_thumbnails.append(result.file)
            return
        if isinstance(result, MetadataResult):
            rules.apply_metadata(project, result)
            touched = [result.file]
        elif isinstance(result, CropJobResult):
            rules.apply_crop(project, result)
            touched = [result.file]
        elif isinstance(result, BrightnessJobResult):
            rules.apply_brightness(project, result)
            touched = [result.file]
        elif isinstance(result, RangesJobResult):
            rules.apply_ranges(project, result)
            touched = list(result.analysis.durations)
        elif isinstance(result, AudioProfileResult):
            rules.apply_audio_profile(project, result)
            touched = [result.file]
        else:
            self._log(PIPELINE_LOG, f"Unexpected job result {type(result).__name__}")
            return
        self._emit_changed.update(name for name in touched if name in project.files)
        self._schedule_save()

    def _on_view_event(self, event: JobEvent) -> None:
        """A frames/strips job ended: cache what came back, release the times
        it was fetching, and remember the ones that could not be read.

        A finished job marks the times missing from its result, a failed one
        every time it was asked for: each is attempted once per session, not
        once per repaint. A CANCELLED job marks nothing -- nothing was
        learned about those times -- so the next request fetches them again.
        """
        if event.type not in TERMINAL_EVENTS:
            return
        requested = self._view_jobs.pop(event.key, [])
        for key in requested:
            self._frame_inflight.discard(key)
        if event.type == "failed":
            self._log(PIPELINE_LOG, f"Could not load {event.kind} for {event.file}: {event.message}")
        result, files = event.result, self._project.files
        arrived = set()
        if isinstance(result, FramesResult) and result.file in files:
            for time_value, image in result.frames.items():
                key = _frame_key(result.file, time_value)
                self._frames.put(key, image)
                arrived.add(key)
                self._emit_frames.append((result.file, float(time_value)))
        elif isinstance(result, StripsResult) and result.file in files and result.strips:
            for time_value, strip in result.strips.items():
                key = _strip_key(result.file, result.crop_box, time_value)
                self._frames.put(key, strip)
                arrived.add(key)
            self._emit_strips.append(result.file)
        if event.type != "cancelled" and event.file in files:
            # What is missing comes from the RESULT, never from what is still
            # in the cache: the last put of a batch may have evicted the first.
            for key in requested:
                if key not in arrived:
                    self._frames.mark_unavailable(key)

    def _on_proof_event(self, event: JobEvent) -> None:
        name = event.file
        if event.type == "log":
            self._log(name or PIPELINE_LOG, event.message)
            return
        if event.type not in TERMINAL_EVENTS:
            return
        if event.type == "failed":
            self._log(DETECTIONS_LOG, f"{name}: proof failed: {event.message}\n{event.error}")
        stale = self._proof_stale.get(name, 0)
        if stale:                                       # the proof of a removed entry (it ends before any newer one)
            if stale > 1:
                self._proof_stale[name] = stale - 1
            else:
                del self._proof_stale[name]
            return
        count = self._proof_outstanding.get(name, 0)
        if count <= 0:
            return
        if count > 1:                                   # a newer proof of this file is outstanding
            self._proof_outstanding[name] = count - 1
            return
        del self._proof_outstanding[name]
        if isinstance(event.result, ProofResult) and name in self._project.files:
            self._proof_results[name] = event.result
        self._emit_proofs.append(name)

    def _on_run_event(self, event: JobEvent) -> None:
        run = self._run
        if run is None or not self._run_active:
            return
        now = time.monotonic()
        kind = event.type
        if kind == "run_file_started":
            run.file_started(event.file, now)
        elif kind == "run_file_progress":
            run.file_progress(event.file, event.progress, event.message)
        elif kind == "run_subtitle":
            start, end, text = event.result
            start, end, text = float(start), float(end), str(text)
            run.subtitle(event.file, start, end, text)
            self.run_subtitle.emit(event.file, start, end, text)
        elif kind == "run_file_log":
            self._log(event.file, event.message)
        elif kind == "run_file_finished":
            run.file_finished(event.file, event.result if isinstance(event.result, dict) else {}, now)
            self._refresh_done([event.file])
        elif kind == "log":
            self._log(PIPELINE_LOG, event.message)
        elif kind in TERMINAL_EVENTS:
            self._finish_run(event, now)
        else:
            return
        self._emit_run = True

    def _finish_run(self, event: JobEvent, now: float) -> None:
        run = self._run
        if event.type == "failed":
            self._log(PIPELINE_LOG, f"Run failed: {event.message}\n{event.error}")
        run.finish(event.result if isinstance(event.result, RunSummary) else None,
                   event.message if event.type == "failed" else "", now)
        stopped = self._run_stop_requested
        self._run_active = False
        self._run_job = None
        self._run_stop_requested = False
        self._sync_autopilot_hold()
        snapshot = run.snapshot()
        succeeded, failed, total = snapshot.count(DONE), snapshot.count(FAILED), len(snapshot.files)
        verb = "stopped" if stopped else "completed"
        self._log(PIPELINE_LOG, f"\nOCR {verb}: {succeeded}/{total} files successful\n")
        if failed:
            self._log(PIPELINE_LOG, f"Warning: {failed} file(s) failed OCR\n")
        self._refresh_done()
        self._reconcile_folder()
        if not stopped:
            self._notify(*notification_for(snapshot))

    def _notify(self, title: str, body: str, urgency: str) -> None:
        try:
            subprocess.run(["notify-send", "-a", "OCR Manager", "-u", urgency, title, body],
                           check=False, timeout=NOTIFY_TIMEOUT_SECONDS)
        except FileNotFoundError:
            pass                                        # notify-send not installed
        except subprocess.TimeoutExpired:
            logger.warning("notify-send did not return within %s s", NOTIFY_TIMEOUT_SECONDS)

    # --- model bookkeeping ----------------------------------------------------------------

    def _current_project(self) -> Project:
        return self._project

    def _require(self) -> tuple[Project, AutoPilot]:
        if self._project is None or self._autopilot is None:
            raise RuntimeError("no folder is open")
        return self._project, self._autopilot

    def _check_alive(self) -> None:
        if self._shut_down:
            raise RuntimeError("the controller has been shut down")

    def _after_edit(self, name: str) -> None:
        self._emit_changed.add(name)
        self._recompute()
        self._schedule_save()
        self._flush()

    def _states(self) -> dict[str, ReviewState]:
        return {name: entry.review for name, entry in self._project.files.items()}

    def _recompute(self, before: dict[str, ReviewState] | None = None) -> None:
        """recompute_all; files whose state differs from `before` (default:
        the states now) get file_changed and a save."""
        project, autopilot = self._project, self._autopilot
        before = self._states() if before is None else before
        rules.recompute_all(project, pending=autopilot.pending(), ranges_pending=autopilot.ranges_pending())
        changed = [name for name, entry in project.files.items() if before.get(name) != entry.review]
        if changed:
            self._emit_changed.update(changed)
            self._schedule_save()

    def _refresh_done(self, names: list[str] | None = None, *, notify: bool = True) -> None:
        project = self._project
        chi = os.path.join(project.path, OUTPUT_DIR)
        for name in (list(project.files) if names is None else names):
            done = name in project.files and _non_empty_file(os.path.join(chi, output_name(name)))
            if done != (name in self._done):
                (self._done.add if done else self._done.discard)(name)
                if notify:
                    self._emit_changed.add(name)
        self._done &= set(project.files)

    def _flush(self) -> None:
        """Emit the signals collected while handling events or a command."""
        files, folder, activity, run = self._emit_files, self._emit_folder, self._emit_activity, self._emit_run
        changed, thumbnails, proofs = self._emit_changed, self._emit_thumbnails, self._emit_proofs
        frames, strips = self._emit_frames, self._emit_strips
        self._reset_emits()
        project = self._project
        if files:
            self.files_changed.emit()
        if folder:
            self.folder_changed.emit()
        if project is not None:
            for name in [name for name in project.files if name in changed]:
                self.file_changed.emit(name)
            for name in dict.fromkeys(thumbnails):
                self.thumbnail_ready.emit(name)
            for name, time_value in dict.fromkeys(frames):
                if name in project.files:
                    self.frame_ready.emit(name, time_value)
            for name in dict.fromkeys(strips):
                if name in project.files:
                    self.strips_ready.emit(name)
            for name in dict.fromkeys(proofs):
                if name in project.files:
                    self.proof_finished.emit(name)
        if activity:
            self.activity_changed.emit()
        if run:
            self.run_changed.emit()

    # --- logs -------------------------------------------------------------------------------

    def _log(self, key: str, text: str) -> None:
        text = self._logs.append(key, text)
        if text:
            self.log_appended.emit(key, text)

    def _clear_logs(self, keep: tuple[str, ...] = ()) -> None:
        self._logs.clear(keep)
        self.logs_cleared.emit()

    # --- saving -----------------------------------------------------------------------------

    def _schedule_save(self) -> None:
        if self._project is not None and not self._shut_down and not self._save_blocked:
            self._save_timer.start()

    def _save_now(self) -> None:
        self._save_timer.stop()
        project = self._project
        if project is None or self._save_blocked:
            return
        try:
            store.save_project(project)
        except UnsupportedProjectVersion as exc:
            self._save_blocked = True           # someone put a newer project file there: never overwrite it
            self._save_failure(f"Not saving: {exc}")
        except (OSError, TypeError, ValueError) as exc:        # TypeError/ValueError: a value JSON cannot hold
            self._save_failure(f"Could not save {os.path.join(project.path, store.CONFIG_FILENAME)}: {exc}")

    def _save_failure(self, message: str) -> None:
        logger.warning("%s", message)
        self._log(PIPELINE_LOG, message)
        self.save_failed.emit(message)

    # --- folder watcher ---------------------------------------------------------------------

    def _on_folder_settled(self) -> None:
        if self._project is None or self._run_active:     # a run's own writes; the run's end catches up
            return
        self._reconcile_folder()
        self._flush()

    def _reconcile_folder(self) -> None:
        project, autopilot = self._project, self._autopilot
        if not os.path.isdir(project.path):
            logger.warning("%s is not reachable: its file list is left as it was", project.path)
            return
        self._folder_watch.rewatch()
        names = store.list_video_files(project.path)
        gone = frozenset(project.files) - set(names)
        if gone:
            self._runner.cancel_where(lambda job: job.file in gone)
        added, removed = store.reconcile_files(project, names)
        for name in removed:
            self._thumbnails.pop(name, None)
            self._frames.clear_file(name)
            self._proof_results.pop(name, None)
            outstanding = self._proof_outstanding.pop(name, 0)
            if outstanding:
                self._proof_stale[name] = self._proof_stale.get(name, 0) + outstanding
        if removed:
            autopilot.on_files_removed(removed)
        if added:
            autopilot.on_files_added(added)
        self._refresh_done()
        if added or removed:
            self._emit_files = True
            self._schedule_save()
        self._recompute()
