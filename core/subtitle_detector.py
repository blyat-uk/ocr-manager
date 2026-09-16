"""Background subtitle detection worker for finding frames with hardcoded subtitles.

This class is a thin Qt adapter over the pure, Qt-free detector in
core/detect/crop.py: it owns threading, progress reporting and
cancellation, and nothing else. The speech-guided probing, cross-file
consensus math and crop aggregation all live in core/detect/crop.py's
detect_crop() -- see that module for the algorithm itself.
"""
import logging

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.detect.crop import CropResult, _probe_dimensions, detect_crop

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
        """Request cancellation. Honoured between files -- detect_crop()
        runs a single file's probing/consensus check as one synchronous
        unit and offers no way to interrupt it mid-call."""
        self._cancel_requested = True

    def cleanup(self):
        """Stop the thread and clean up."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None

    def _run(self):
        """Run detect_crop() over each file in turn, maintaining cross-file consensus."""
        total = len(self._video_files)
        resolved_count = 0
        consensus: list[tuple[float, float]] = []

        try:
            from videocr.utils import create_detection_engine, suppress_output

            with suppress_output():
                det_engine = create_detection_engine(None, True)
        except Exception as e:
            logger.exception("Failed to create subtitle detection engine")
            self.error.emit(str(e))
            return

        self.progress.emit(resolved_count, total)

        for filename, full_path, duration in self._video_files:
            if self._cancel_requested:
                break

            try:
                result: CropResult = detect_crop(
                    full_path, duration, det_engine,
                    consensus=consensus, settings=self._auto_settings,
                )
            except Exception:
                logger.exception("Subtitle detection failed for %s", filename)
                resolved_count += 1
                self.progress.emit(resolved_count, total)
                continue

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

            slider_pos = 5000
            if result.sample_pts and duration > 0:
                slider_pos = int((result.sample_pts[0] / duration) * 10000)
                slider_pos = max(0, min(10000, slider_pos))

            self.file_detected.emit(filename, slider_pos, crop_x, crop_y, crop_w, crop_h)

            try:
                _orig_w, orig_h = _probe_dimensions(full_path)
                consensus.append((crop_y / orig_h, crop_h / orig_h))
            except Exception:
                logger.warning(
                    "%s: could not probe dimensions for consensus tracking", filename,
                )

            resolved_count += 1
            self.progress.emit(resolved_count, total)

        self.finished.emit()
