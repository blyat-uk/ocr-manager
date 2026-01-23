"""Single file OCR worker with QProcess and tqdm progress parsing."""

import re
import shutil
import subprocess
from enum import Enum
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, QProcess

from core.config import Config


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
    progress_updated = pyqtSignal(str, int)   # filename, percent 0-100
    finished = pyqtSignal(str, bool)          # filename, success

    # tqdm progress pattern: "45%|..." or "Processing frames:  45%|..."
    TQDM_PATTERN = re.compile(r'(\d+)%\|')

    def __init__(self, video_path: Path, output_dir: Path, config: Config, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.output_dir = output_dir
        self.config = config
        self.filename = video_path.name
        self.process = None
        self._last_percent = -1

    def build_command(self) -> list[str]:
        """Build videocr.py command line arguments."""
        # Output .ass file next to the video file (will be moved to output_dir after completion)
        output_path = self.video_path.parent / (self.video_path.stem + ".ass")

        cmd = [
            self.config.videocr_python,
            self.config.videocr_script,
            str(self.video_path),
            "-o", str(output_path),
            "-l", self.config.ocr_lang,
            "-c", str(self.config.conf_threshold),
            "-s", str(self.config.sim_threshold),
            "-b", str(self.config.brightness),
            "--similar-image", str(self.config.similar_image),
            "--skip", str(self.config.frames_to_skip),
        ]

        # Crop region
        if self.config.crop_width > 0 and self.config.crop_height > 0:
            crop = f"{self.config.crop_x},{self.config.crop_y},{self.config.crop_width},{self.config.crop_height}"
            cmd.extend(["--crop", crop])

        # GPU option
        if not self.config.use_gpu:
            cmd.append("--no-gpu")

        # Time range
        if self.config.time_start:
            cmd.extend(["-ts", self.config.time_start])
        if self.config.time_end:
            cmd.extend(["-te", self.config.time_end])

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
        """Parse tqdm progress from output data."""
        match = self.TQDM_PATTERN.search(data)
        if match:
            percent = int(match.group(1))
            # Only emit if percentage changed (avoid spam)
            if percent != self._last_percent:
                self._last_percent = percent
                self.progress_updated.emit(self.filename, percent)

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
