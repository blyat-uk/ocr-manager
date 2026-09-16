"""Audio-fingerprint time-range detection worker.

This class is a thin Qt adapter over the pure, Qt-free pipeline in
core/detect/ranges/pipeline.py: it owns threading, progress forwarding and
cancellation, and nothing else. Fingerprinting, segment discovery and
keep-range computation all live in core.detect.ranges.pipeline.analyse()
-- see that module for the algorithm itself.
"""
import logging
import os

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.detect.ranges.config import MatchConfig, RangesConfig
from core.detect.ranges.pipeline import (
    DEFAULT_WORKERS,
    AnalysisCancelled,
    FileEntry,
    ProgressEvent,
    analyse,
    default_cache_dir,
)

logger = logging.getLogger(__name__)

# Default minimum repeating-segment length (seconds). Kept importable here
# (main.py's UI default references this name) even though the value now
# also lives as core.detect.ranges.pipeline.DEFAULT_MIN_SEGMENT_SEC.
DEFAULT_MIN_SEGMENT_SEC = 30.0


class AudioAnalysisWorker(QObject):
    """Worker that fingerprints video files and discovers repeating segments.

    Runs core.detect.ranges.pipeline.analyse() on a background QThread,
    forwarding its progress events through this worker's own signals and
    translating cancellation into the existing finished({}) contract.

    Signals:
        phase_changed(str): Current phase name
        file_progress(str, int, int): filename, current (1-based), total
        analysis_progress(str): Status messages from segment discovery
        error(str): Error message
        finished(dict): {filename: [(start_mmss | None, end_mmss | None), ...]}
    """

    phase_changed = pyqtSignal(str)
    file_progress = pyqtSignal(str, int, int)
    analysis_progress = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, project_path: str, video_files: list[str],
                 min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC,
                 merge_repeating_silences: bool = False):
        super().__init__()
        self._project_path = project_path
        self._video_files = video_files
        self._min_segment_sec = min_segment_sec
        self._merge_repeating_silences = merge_repeating_silences
        self._cancel_requested = False
        self._thread: QThread | None = None

    def start(self):
        """Start the worker in a new thread."""
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.start()

    def cancel(self):
        """Request cancellation. Polled by analyse() via _cancel_check
        between fingerprinting files and between phases, so cancellation
        takes effect well before the whole analysis finishes on its own."""
        self._cancel_requested = True

    def _cancel_check(self) -> bool:
        return self._cancel_requested

    def cleanup(self, timeout_ms: int | None = None) -> bool:
        """Stop the thread and clean up.

        `timeout_ms=None` (the default) waits indefinitely, matching every
        existing caller (main.py's included -- unedited). Callers that want
        a bound (tests, notably) can pass one; returns True if the thread
        had actually finished within it, False on a timeout. On a timeout
        the QThread object is deliberately NOT dropped, so it can't be
        garbage-collected while its OS thread may still be running.
        """
        if self._thread is None:
            return True
        self._thread.quit()
        finished = self._thread.wait() if timeout_ms is None else self._thread.wait(timeout_ms)
        if finished:
            self._thread = None
        return finished

    def _on_progress(self, event: ProgressEvent) -> None:
        if event.kind == "phase":
            self.phase_changed.emit(event.message)
        elif event.kind == "file":
            self.file_progress.emit(event.message, event.current, event.total)
        elif event.kind == "log":
            self.analysis_progress.emit(event.message)

    def _run(self):
        """Run analyse() over the project's video files."""
        try:
            files = [
                FileEntry(name=name, path=os.path.join(self._project_path, name))
                for name in self._video_files
            ]
            cfg = RangesConfig(
                match=MatchConfig(min_length_sec=self._min_segment_sec),
                merge_repeating_silences=self._merge_repeating_silences,
            )
            results = analyse(
                files, cfg, self._on_progress,
                cache_dir=default_cache_dir(self._project_path),
                workers=DEFAULT_WORKERS,
                cancel=self._cancel_check,
            )
        except AnalysisCancelled:
            self.finished.emit({})
            return
        except Exception as e:
            logger.exception("Audio analysis failed")
            self.error.emit(str(e))
            return

        self.finished.emit(results)
