"""Single file OCR worker with QProcess and tqdm progress parsing."""

import re
import shutil
import subprocess
from enum import Enum
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, QProcess

from core.config import Config, FileConfig


class FileStatus(Enum):
    """Status of an OCR file operation."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DONE = "done"  # Pre-existing (already processed in previous run)


class OCRWorker(QObject):
    """Wraps QProcess for single video file OCR with progress parsing."""

    status_changed = pyqtSignal(str, object)  # filename, FileStatus
    status_text_changed = pyqtSignal(str, str)  # filename, status_text (e.g., "Extracting dialogue")
    progress_updated = pyqtSignal(str, int)   # filename, percent 0-100
    finished = pyqtSignal(str, bool)          # filename, success

    # tqdm progress pattern: "Extracting dialogue:  45%|..." captures title and percentage
    TQDM_PATTERN = re.compile(r'([^:\r\n]+):\s*(\d+)%\|')

    def __init__(self, video_path: Path, output_dir: Path, config: Config,
                 file_config: FileConfig = None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.output_dir = output_dir
        self.config = config
        self.file_config = file_config  # Per-file config overrides
        self.filename = video_path.name
        self.process = None
        self._last_percent = -1
        self._current_phase = ""  # Track current progress bar title

    def build_command(self) -> list[str]:
        """Build videocr.py command line arguments.

        Uses per-file config when available, falling back to global config.
        """
        # Output .ass file next to the video file (will be moved to output_dir after completion)
        output_path = self.video_path.parent / (self.video_path.stem + ".ass")

        # Determine effective settings (per-file overrides global)
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

        cmd = [
            gc.videocr_python,
            gc.videocr_script,
            str(self.video_path),
            "-o", str(output_path),
            "-l", gc.ocr_lang,
            "-c", str(gc.conf_threshold),
            "-s", str(gc.sim_threshold),
            "-b", str(brightness),
            "--similar-image", str(gc.similar_image),
            "--skip", str(gc.frames_to_skip),
        ]

        # Crop region
        if crop_w > 0 and crop_h > 0:
            crop = f"{crop_x},{crop_y},{crop_w},{crop_h}"
            cmd.extend(["--crop", crop])

        # GPU option
        if not gc.use_gpu:
            cmd.append("--no-gpu")

        # Time range
        if time_start:
            cmd.extend(["-ts", time_start])
        if time_end:
            cmd.extend(["-te", time_end])

        # Label detection
        if not gc.labels_enabled:
            cmd.append("--no-labels")
        else:
            if gc.labels_only:
                cmd.append("--only-labels")
            if gc.label_min_duration != 1.0:
                cmd.extend(["--label-min-duration", str(gc.label_min_duration)])
            if gc.label_max_duration != 8.0:
                cmd.extend(["--label-max-duration", str(gc.label_max_duration)])
            if gc.label_conf_threshold != 95:
                cmd.extend(["--label-conf-threshold", str(gc.label_conf_threshold)])

        return cmd

    def start(self):
        """Start the OCR process."""
        self.status_changed.emit(self.filename, FileStatus.PROCESSING)

        cmd = self.build_command()

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_output)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)

        # Set working directory to project path
        self.process.setWorkingDirectory(str(self.video_path.parent))

        self.process.start(cmd[0], cmd[1:])

    def _on_output(self):
        """Handle merged stdout/stderr output."""
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._parse_progress(data)

    def _on_stderr(self):
        """Handle stderr output (tqdm writes here)."""
        data = self.process.readAllStandardError().data().decode("utf-8", errors="replace")
        self._parse_progress(data)

    def _parse_progress(self, data: str):
        """Parse tqdm progress from output data, extracting title and percentage."""
        match = self.TQDM_PATTERN.search(data)
        if match:
            title = match.group(1).strip()
            percent = int(match.group(2))

            # Check if we've moved to a new phase (new progress bar title)
            if title != self._current_phase:
                self._current_phase = title
                self._last_percent = -1  # Reset progress for new phase
                self.status_text_changed.emit(self.filename, title)

            # Cap progress at 99% during processing (100% only on successful completion)
            display_percent = min(percent, 99)

            # Only emit if percentage changed (avoid spam)
            if display_percent != self._last_percent:
                self._last_percent = display_percent
                self.progress_updated.emit(self.filename, display_percent)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        """Handle process completion."""
        success = exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit

        if success:
            # Move .ass file from video directory to output directory
            ass_source = self.video_path.parent / (self.video_path.stem + ".ass")
            ass_dest = self.output_dir / (self.video_path.stem + ".ass")

            if ass_source.exists():
                shutil.move(str(ass_source), str(ass_dest))

            self.status_changed.emit(self.filename, FileStatus.COMPLETED)
            self.progress_updated.emit(self.filename, 100)
        else:
            self.status_changed.emit(self.filename, FileStatus.FAILED)

        self.finished.emit(self.filename, success)

    def stop(self):
        """Stop the OCR process and all child processes."""
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            pid = self.process.processId()

            # Kill child processes first using pkill
            try:
                subprocess.run(["pkill", "-TERM", "-P", str(pid)],
                               capture_output=True, timeout=2)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            # Terminate main process
            self.process.terminate()
            self.process.waitForFinished(3000)

            # Force kill if still running
            if self.process.state() == QProcess.ProcessState.Running:
                try:
                    subprocess.run(["pkill", "-KILL", "-P", str(pid)],
                                   capture_output=True, timeout=2)
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
                self.process.kill()
