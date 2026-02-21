"""Single file OCR worker using QThread and direct videocr API calls."""

import shutil
import traceback
from enum import Enum
from pathlib import Path
from threading import Event

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.config import Config, FileConfig


class FileStatus(Enum):
    """Status of an OCR file operation."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DONE = "done"  # Pre-existing (already processed in previous run)


class OCRWorker(QObject):
    """Runs videocr OCR directly via QThread with cooperative cancellation."""

    status_changed = pyqtSignal(str, object)  # filename, FileStatus
    status_text_changed = pyqtSignal(str, str)  # filename, status_text (e.g., "Extracting dialogue")
    progress_updated = pyqtSignal(str, int)   # filename, percent 0-100
    finished = pyqtSignal(str, bool)          # filename, success
    raw_output = pyqtSignal(str, str)         # filename, raw_text
    subtitle_detected = pyqtSignal(str, float, float, str)  # filename, start, end, text

    def __init__(self, video_path: Path, output_dir: Path, config: Config,
                 file_config: FileConfig = None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.output_dir = output_dir
        self.config = config
        self.file_config = file_config  # Per-file config overrides
        self.filename = video_path.name
        self._cancel_event = Event()
        self._thread = None
        self._last_percent = -1
        self._current_phase = ""  # Track current progress phase name

    def _build_ocr_kwargs(self) -> dict:
        """Build kwargs dict for videocr API from Config/FileConfig.

        Uses per-file config when available, falling back to global config.
        """
        fc = self.file_config
        gc = self.config

        # Brightness: use file-specific if set, otherwise global
        brightness = fc.brightness if (fc and fc.brightness is not None) else gc.brightness

        # Crop: use file-specific if set, otherwise global
        if fc and fc.has_custom_crop():
            crop_x, crop_y, crop_w, crop_h = fc.crop_x, fc.crop_y, fc.crop_width, fc.crop_height
        else:
            crop_x, crop_y, crop_w, crop_h = gc.crop_x, gc.crop_y, gc.crop_width, gc.crop_height

        # Time range: use file-specific if set, otherwise global
        time_start = fc.time_start if (fc and fc.time_start) else gc.time_start
        time_end = fc.time_end if (fc and fc.time_end) else gc.time_end

        kwargs = {
            'video_path': str(self.video_path),
            'file_path': str(self.video_path.parent / (self.video_path.stem + ".ass")),
            'lang': gc.ocr_lang,
            'conf_threshold': gc.conf_threshold,
            'sim_threshold': gc.sim_threshold,
            'brightness_threshold': brightness,
            'similar_image_threshold': gc.similar_image,
            'frames_to_skip': gc.frames_to_skip,
            'use_gpu': gc.use_gpu,
            'time_start': time_start or '0:00',
            'time_end': time_end or '',
        }

        # Crop region
        if crop_w > 0 and crop_h > 0:
            kwargs['crop_x'] = crop_x
            kwargs['crop_y'] = crop_y
            kwargs['crop_width'] = crop_w
            kwargs['crop_height'] = crop_h

        # Label detection
        if not gc.labels_enabled:
            kwargs['detect_labels'] = False
        else:
            kwargs['detect_labels'] = True
            if gc.labels_only:
                kwargs['only_labels'] = True
            kwargs['label_min_duration'] = gc.label_min_duration
            kwargs['label_max_duration'] = gc.label_max_duration
            kwargs['label_conf_threshold'] = gc.label_conf_threshold
            kwargs['label_conf_threshold_min'] = gc.label_conf_threshold_min
            if gc.label_mask_crops:
                kwargs['label_mask_crops'] = [
                    (mask[0], mask[1], mask[2], mask[3])
                    for mask in gc.label_mask_crops
                ]

        return kwargs

    def start(self):
        """Start the OCR in a background QThread."""
        self.status_changed.emit(self.filename, FileStatus.PROCESSING)

        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run_ocr)
        self._thread.start()

    def _on_progress(self, phase_name: str, percent: int):
        """Progress callback invoked from videocr (runs in worker thread).

        Emits signals with dedup logic matching the old tqdm parser.
        """
        # Cap progress at 99% during processing (100% only on successful completion)
        display_percent = min(percent, 99)

        # Check if we've moved to a new phase
        if phase_name != self._current_phase:
            self._current_phase = phase_name
            self._last_percent = -1  # Reset progress for new phase
            self.status_text_changed.emit(self.filename, phase_name)

        # Only emit if percentage changed (avoid spam)
        if display_percent != self._last_percent:
            self._last_percent = display_percent
            self.progress_updated.emit(self.filename, display_percent)

    def _on_subtitle(self, start: float, end: float, text: str):
        """Subtitle callback invoked from videocr (runs in worker thread)."""
        self.subtitle_detected.emit(self.filename, start, end, text)

    def _run_ocr(self):
        """Execute videocr directly (runs in QThread)."""
        success = False
        try:
            # Lazy import to avoid loading PaddleOCR at app startup
            from videocr.api import save_subtitles_to_file

            kwargs = self._build_ocr_kwargs()

            self.raw_output.emit(self.filename, f"Starting OCR: {self.filename}\n")

            save_subtitles_to_file(
                **kwargs,
                progress_callback=self._on_progress,
                subtitle_callback=self._on_subtitle,
                cancel_event=self._cancel_event,
            )

            # Check if cancelled
            if self._cancel_event.is_set():
                self.raw_output.emit(self.filename, "OCR cancelled.\n")
                # Clean up partial output file
                ass_source = self.video_path.parent / (self.video_path.stem + ".ass")
                if ass_source.exists():
                    ass_source.unlink(missing_ok=True)
                self.status_changed.emit(self.filename, FileStatus.FAILED)
                self.finished.emit(self.filename, False)
                return

            # Move .ass file from video directory to output directory
            ass_source = self.video_path.parent / (self.video_path.stem + ".ass")
            ass_dest = self.output_dir / (self.video_path.stem + ".ass")

            if ass_source.exists():
                # Run QA on the .ass file before moving
                from core.ass_qafix import process_file
                self.status_text_changed.emit(self.filename, "QA fixing")
                stats = process_file(str(ass_source))
                self.raw_output.emit(
                    self.filename,
                    f"QA: {stats.dialogue_lines} dialogues, "
                    f"{stats.fixed_lines} fixed, "
                    f"{stats.duplicates_removed} deduped\n",
                )
                shutil.move(str(ass_source), str(ass_dest))

            success = True
            self.raw_output.emit(self.filename, "OCR completed successfully.\n")
            self.status_changed.emit(self.filename, FileStatus.COMPLETED)
            self.progress_updated.emit(self.filename, 100)

        except Exception as e:
            error_msg = f"OCR failed: {e}\n{traceback.format_exc()}"
            self.raw_output.emit(self.filename, error_msg)
            self.status_changed.emit(self.filename, FileStatus.FAILED)

        self.finished.emit(self.filename, success)

    def stop(self):
        """Request cooperative cancellation."""
        self._cancel_event.set()

    def cleanup(self):
        """Release references to allow garbage collection."""
        self.video_path = None
        self.output_dir = None
        self.config = None
        self.file_config = None
        self._cancel_event = None
        if self._thread is not None:
            try:
                self._thread.started.disconnect(self._run_ocr)
            except TypeError:
                pass
            self._thread = None
