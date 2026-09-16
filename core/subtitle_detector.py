"""Background subtitle detection worker for finding frames with hardcoded subtitles.

This class is a thin Qt adapter over the pure, Qt-free detector in
core/detect/crop.py: it owns threading, progress reporting and
cancellation, and nothing else. The speech-guided probing, cross-file
consensus math and crop aggregation all live in core/detect/crop.py's
detect_crop() -- see that module for the algorithm itself.
"""
import logging
from contextlib import ExitStack

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.detect.crop import CropResult, detect_crop

logger = logging.getLogger(__name__)


class SubtitleDetectionWorker(QObject):
    """Worker that scans video files to find frames containing subtitles.

    Runs core.detect.crop.detect_crop() against each file in turn, on a
    background QThread. Resolved files' crop shapes accumulate into a
    running `consensus` list of (y_frac, h_frac) that later files'
    detect_crop() calls use to validate their own result against the
    series (see core/detect/crop.py's consensus handling).

    Signals:
        file_detected(str, int, int, int, int, int): filename, slider_position, crop_x, crop_y, crop_w, crop_h
        progress(int, int): (files_resolved, total_files)
        finished(): all files processed
        error(str): error message
    """

    file_detected = pyqtSignal(str, int, int, int, int, int)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, video_files: list[tuple[str, str, float]],
                 automation_settings: dict | None = None):
        """Initialize worker.

        Args:
            video_files: list of (filename, full_path, duration_seconds) tuples
            automation_settings: optional dict with auto-crop overrides, passed
                through unchanged as detect_crop()'s `settings` argument.
        """
        super().__init__()
        self._video_files = video_files
        self._auto_settings = automation_settings
        self._cancel_requested = False
        self._thread: QThread | None = None

    def start(self):
        """Start the worker in a new thread."""
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.start()

    def cancel(self):
        """Request cancellation. `detect_crop()` accepts this as a
        `cancel_check` callable and polls it between probe batches (see
        core/detect/crop.py's `_run_round()`), so cancellation takes
        effect within roughly one batch of the file currently being
        processed, not only once that file finishes."""
        self._cancel_requested = True

    def _cancel_check(self) -> bool:
        return self._cancel_requested

    def cleanup(self, timeout_ms: int | None = None) -> bool:
        """Stop the thread and clean up.

        `timeout_ms=None` (the default) waits indefinitely for the thread
        to finish, matching every existing caller (main.py's included --
        unedited). Callers that want a bound (tests, notably: an unbounded
        `wait()` here would defeat an event-loop timeout guard placed
        around `start()`) can pass one; this returns True if the thread
        had actually finished within it, False if the wait timed out. On
        a timeout the QThread object is deliberately NOT dropped, so it
        can't be garbage-collected while its OS thread may still be
        running.
        """
        if self._thread is None:
            return True
        self._thread.quit()
        finished = self._thread.wait() if timeout_ms is None else self._thread.wait(timeout_ms)
        if finished:
            self._thread = None
        return finished

    def _run(self):
        """Run detect_crop() over each file in turn, maintaining cross-file consensus.

        The detection engine is leased from videocr.engine_registry for the
        whole run and returned before `finished` is emitted. OCR workers run
        as threads in this same process, and an engine must never serve two
        threads at once, so this worker only ever uses the instance it holds
        a lease on. (No suppress_output() here: the registry's builder
        already silences construction, and that redirection is process-wide,
        so it must only ever happen under the registry's construction lock.)
        """
        with ExitStack() as lease:
            try:
                from videocr import engine_registry

                det_engine = lease.enter_context(engine_registry.lease_detection_engine(None, True))
            except Exception as e:
                logger.exception("Failed to create subtitle detection engine")
                self.error.emit(str(e))
                return
            self._detect_files(det_engine)

        self.finished.emit()

    def _detect_files(self, det_engine):
        total = len(self._video_files)
        resolved_count = 0
        consensus: list[tuple[float, float]] = []

        self.progress.emit(resolved_count, total)

        for filename, full_path, duration in self._video_files:
            if self._cancel_requested:
                break

            try:
                result: CropResult = detect_crop(
                    full_path, duration, det_engine,
                    consensus=consensus, settings=self._auto_settings,
                    cancel_check=self._cancel_check,
                )
            except Exception:
                logger.exception("Subtitle detection failed for %s", filename)
                resolved_count += 1
                self.progress.emit(resolved_count, total)
                continue

            if self._cancel_requested:
                # Cancelled mid-file: detect_crop() returned early (see
                # cancel_check above) with whatever partial evidence it
                # had gathered, which is not a trustworthy result -- treat
                # this file as unresolved rather than emit it.
                break

            if result.box is None:
                # No way for the current UI to show a *reason* for a
                # missing crop -- emitting a zero box would look like a
                # real (empty) result rather than "nothing found". Skip
                # the file and log the flag so it's at least visible.
                # Stage 3 will surface flags properly.
                logger.info(
                    "%s: no crop detected (flagged=%s, agreed=%d, probes_used=%d)",
                    filename, result.flagged, result.agreed, result.probes_used,
                )
                resolved_count += 1
                self.progress.emit(resolved_count, total)
                continue

            crop_x, crop_y, crop_w, crop_h = result.box

            if not result.hit_pts:
                # box is not None should imply at least one contributing
                # hit (see core/detect/crop.py's detect_crop() docstring),
                # but this is the one place a wrong position could reach
                # the UI silently -- refuse to guess rather than fall back
                # to sample_pts[0], which after a fallback round is
                # guaranteed to be a frame with no detected text.
                logger.warning(
                    "%s: box resolved but hit_pts is empty -- cannot derive a slider "
                    "position, skipping", filename,
                )
                resolved_count += 1
                self.progress.emit(resolved_count, total)
                continue

            slider_pos = 5000
            if duration > 0:
                slider_pos = int((result.hit_pts[0] / duration) * 10000)
                slider_pos = max(0, min(10000, slider_pos))

            self.file_detected.emit(filename, slider_pos, crop_x, crop_y, crop_w, crop_h)

            # A resolved, boxed file may still be absent from the
            # consensus pool: either it was flagged (an uncertain/
            # ambiguous result -- multiple-positions?, outlier-discarded?,
            # static-content?, low-agreement, etc. -- is exactly what
            # consensus must not learn from), or its frame dimensions
            # weren't available to convert the box into (y_frac, h_frac).
            if result.flagged is None and result.frame_size is not None:
                _orig_w, orig_h = result.frame_size
                if orig_h > 0:
                    consensus.append((crop_y / orig_h, crop_h / orig_h))

            resolved_count += 1
            self.progress.emit(resolved_count, total)
