"""Parallel OCR orchestration with queue-based concurrency."""

from dataclasses import dataclass, field
from pathlib import Path
from time import time

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from core.config import Config, FileConfig, FileConfigStore
from core.ocr_worker import OCRWorker, FileStatus


@dataclass
class TimingStats:
    """Track timing statistics for OCR processing."""

    start_time: float = 0.0
    file_start_times: dict[str, float] = field(default_factory=dict)
    file_durations: dict[str, float] = field(default_factory=dict)

    def start_session(self):
        """Start timing the overall session."""
        self.start_time = time()
        self.file_start_times.clear()
        self.file_durations.clear()

    def start_file(self, filename: str):
        """Record start time for a file."""
        self.file_start_times[filename] = time()

    def finish_file(self, filename: str) -> float:
        """Record completion time for a file and return duration."""
        if filename in self.file_start_times:
            duration = time() - self.file_start_times[filename]
            self.file_durations[filename] = duration
            del self.file_start_times[filename]
            return duration
        return 0.0

    def get_elapsed(self) -> float:
        """Get elapsed time since session start."""
        if self.start_time == 0.0:
            return 0.0
        return time() - self.start_time

    def get_average_duration(self) -> float:
        """Get average duration of completed files."""
        if not self.file_durations:
            return 0.0
        return sum(self.file_durations.values()) / len(self.file_durations)

    def get_completed_count(self) -> int:
        """Get number of completed files."""
        return len(self.file_durations)


class OCRManager(QObject):
    """Manages parallel OCR workers with queue-based concurrency."""

    file_status_changed = pyqtSignal(str, object)   # filename, FileStatus
    file_progress_updated = pyqtSignal(str, int)    # filename, percent
    timing_updated = pyqtSignal(float, float, float)  # elapsed, eta, avg_per_file
    overall_progress = pyqtSignal(int, int)           # completed_files, total_files
    all_completed = pyqtSignal(int, int, float, float)  # successful, total, total_time, avg_time

    def __init__(self, config: Config, file_config_store: FileConfigStore = None, parent=None):
        super().__init__(parent)
        self.config = config
        self.file_config_store = file_config_store
        self.max_workers = config.ocr_parallel

        self._queue: list[Path] = []          # Pending files
        self._active_workers: dict[str, OCRWorker] = {}  # filename -> OCRWorker
        self._output_dir: Path | None = None

        # Tracking
        self._total = 0
        self._successful = 0
        self._failed = 0

        # Timing
        self._timing = TimingStats()
        self._timing_timer = QTimer(self)
        self._timing_timer.timeout.connect(self._emit_timing_update)

    def set_files(self, video_files: list[Path], output_dir: Path):
        """Set files to process and output directory."""
        self._queue = list(video_files)
        self._output_dir = output_dir
        self._total = len(video_files)
        self._successful = 0
        self._failed = 0

        # Emit QUEUED status for all files
        for video_path in video_files:
            self.file_status_changed.emit(video_path.name, FileStatus.QUEUED)

    def start(self):
        """Start processing the queue."""
        self._timing.start_session()
        self._timing_timer.start(1000)  # Update every second
        self._start_next_workers()

    def _start_next_workers(self):
        """Start workers up to max_workers limit."""
        while self._queue and len(self._active_workers) < self.max_workers:
            video_path = self._queue.pop(0)
            self._start_worker(video_path)

    def _start_worker(self, video_path: Path):
        """Create and start a worker for a single video file."""
        # Get per-file config if available
        file_config = None
        if self.file_config_store:
            file_config = self.file_config_store.get(video_path.name)

        worker = OCRWorker(video_path, self._output_dir, self.config, file_config, self)

        # Connect signals
        worker.status_changed.connect(self._on_worker_status_changed)
        worker.progress_updated.connect(self._on_worker_progress)
        worker.finished.connect(self._on_worker_finished)

        self._active_workers[video_path.name] = worker
        self._timing.start_file(video_path.name)
        worker.start()

    def _on_worker_status_changed(self, filename: str, status: FileStatus):
        """Forward status change signal."""
        self.file_status_changed.emit(filename, status)

    def _on_worker_progress(self, filename: str, percent: int):
        """Forward progress update signal."""
        self.file_progress_updated.emit(filename, percent)

    def _on_worker_finished(self, filename: str, success: bool):
        """Handle worker completion."""
        # Record timing before cleanup
        self._timing.finish_file(filename)

        # Remove from active workers
        if filename in self._active_workers:
            worker = self._active_workers.pop(filename)
            worker.deleteLater()

        # Track results
        if success:
            self._successful += 1
        else:
            self._failed += 1

        # Emit overall progress
        completed = self._successful + self._failed
        self.overall_progress.emit(completed, self._total)

        # Start next worker if queue has items
        if self._queue:
            self._start_next_workers()
        elif not self._active_workers:
            # All done - stop timer and emit completion
            self._timing_timer.stop()
            total_time = self._timing.get_elapsed()
            avg_time = self._timing.get_average_duration()
            self.all_completed.emit(self._successful, self._total, total_time, avg_time)

    def stop(self):
        """Stop all active workers and clear queue."""
        self._timing_timer.stop()
        self._queue.clear()

        # Copy to list to avoid modification during iteration
        # (worker.stop() may trigger finished signal synchronously)
        for worker in list(self._active_workers.values()):
            worker.stop()

        self._active_workers.clear()

    def is_running(self) -> bool:
        """Check if any workers are active."""
        return bool(self._active_workers) or bool(self._queue)

    def _calculate_eta(self) -> float:
        """Calculate estimated time remaining based on completed files."""
        completed = self._timing.get_completed_count()
        remaining = self._total - self._successful - self._failed

        if completed == 0 or remaining == 0:
            return 0.0

        avg_duration = self._timing.get_average_duration()
        active_workers = min(len(self._active_workers), remaining) or 1

        # Account for parallelism
        return (remaining * avg_duration) / active_workers

    def _emit_timing_update(self):
        """Emit current timing statistics."""
        elapsed = self._timing.get_elapsed()
        eta = self._calculate_eta()
        avg = self._timing.get_average_duration()
        self.timing_updated.emit(elapsed, eta, avg)
