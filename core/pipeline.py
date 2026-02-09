"""Pipeline execution and workflow management."""

import shutil
from itertools import chain
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal, QProcess

from core.config import Config, FileConfigStore
from core.ocr_manager import OCRManager
from core.ocr_worker import FileStatus

VIDEO_EXTENSIONS = ("*.mkv", "*.mp4")


def get_video_files(directory: Path, sort_by_name: bool = False) -> list[Path]:
    """Get all video files (MKV and MP4) from a directory."""
    files = list(chain.from_iterable(directory.glob(ext) for ext in VIDEO_EXTENSIONS))
    if sort_by_name:
        files.sort(key=lambda f: f.name)
    return files


def detect_file_statuses(project_path: Path) -> dict[str, FileStatus]:
    """Detect status of each video file based on existing ASS files.

    Returns:
        Dict mapping filename to FileStatus.DONE or FileStatus.QUEUED
    """
    chi_dir = project_path / "chi"
    completed_stems = set()

    if chi_dir.exists():
        for ass_file in chi_dir.glob("*.ass"):
            # Only count non-empty ASS files as completed
            if ass_file.stat().st_size > 0:
                completed_stems.add(ass_file.stem)

    statuses = {}
    for video_file in get_video_files(project_path):
        if video_file.stem in completed_stems:
            statuses[video_file.name] = FileStatus.DONE
        else:
            statuses[video_file.name] = FileStatus.QUEUED

    return statuses


class Pipeline(QObject):
    """Manages OCR workflow execution."""

    # Signals
    command_started = pyqtSignal(str)
    output_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    phase_started = pyqtSignal(int, str)  # phase_index, phase_name
    phase_completed = pyqtSignal(str)
    pipeline_finished = pyqtSignal(bool)
    pipeline_stopped = pyqtSignal()  # Emitted when user manually stops

    # Progress table signals (Phase 2)
    ocr_file_status = pyqtSignal(str, object)   # filename, FileStatus
    ocr_file_status_text = pyqtSignal(str, str)  # filename, status_text (e.g., "Extracting dialogue")
    ocr_file_progress = pyqtSignal(str, int)    # filename, percent
    ocr_files_detected = pyqtSignal(list)       # list of filenames for table init

    # Timing signals (Phase 2)
    ocr_timing_updated = pyqtSignal(float, float, float)  # elapsed, eta, avg
    ocr_overall_progress = pyqtSignal(int, int)           # completed, total
    ocr_log_output = pyqtSignal(str, str)                 # filename, raw_text

    PHASES = ["Create Directory", "OCR Extraction", "Quality Assurance"]

    def __init__(self, config: Config, file_config_store: FileConfigStore = None):
        super().__init__()
        self.config = config
        self.file_config_store = file_config_store
        self.process = None
        self.ocr_manager = None
        self.current_phase = 0
        self.subphase = 0

        # OCR timing data (stored for completion dialog)
        self._ocr_total_time = 0.0
        self._ocr_avg_time = 0.0

    def get_video_stems(self) -> set[str]:
        """Get stems of all video files in project root."""
        project_path = Path(self.config.project_path)
        return {f.stem for f in get_video_files(project_path)}

    def get_completed_stems(self) -> set[str]:
        """Get stems that have completed OCR (have .ass files in chi/)."""
        project_path = Path(self.config.project_path)
        chi_dir = project_path / "chi"
        return {f.stem for f in chi_dir.glob("*.ass")} if chi_dir.exists() else set()

    def detect_resume_phase(self) -> int:
        """Detect which phase to resume from based on per-file completion.

        Returns:
            0 - Start from beginning (no chi/ or incomplete OCR)
            2 - Resume at QA (all videos have corresponding .ass in chi/)
        """
        all_videos = self.get_video_stems()
        if not all_videos:
            return 0  # No video files to process

        ocr_done = self.get_completed_stems()

        # If any video is missing its .ass file, start from beginning
        if all_videos - ocr_done:
            return 0  # Start from Phase 0 (Create Directory)

        # All videos have .ass files, resume at QA (phase 2)
        return 2

    def start(self):
        """Start pipeline execution, resuming from detected phase."""
        # Always ensure directories exist (even when resuming)
        self.ensure_directories()

        detected_phase = self.detect_resume_phase()

        self.current_phase = detected_phase
        self.subphase = 0

        if detected_phase > 0:
            phase_name = self.PHASES[detected_phase]
            self.output_received.emit(f"Resuming from phase {detected_phase + 1}: {phase_name}...\n")

        self.run_next_phase()

    def ensure_directories(self):
        """Ensure output directories exist."""
        project_path = Path(self.config.project_path)
        for subdir in ("chi", "eng", "translate"):
            (project_path / subdir).mkdir(exist_ok=True)

    def run_next_phase(self):
        """Execute next phase in sequence."""
        if self.current_phase >= len(self.PHASES):
            self.pipeline_finished.emit(True)
            return

        # Emit phase_started signal for status visualization
        phase_name = self.PHASES[self.current_phase]
        self.phase_started.emit(self.current_phase, phase_name)

        phase_method = getattr(self, f"phase_{self.current_phase + 1}")
        phase_method()

    def phase_1(self):
        """Phase 1: Create Directory - Create output folders."""
        self.ensure_directories()
        self.output_received.emit("Created chi/, eng/, translate/ directories\n")

        self.phase_completed.emit("Create Directory")
        self.current_phase += 1
        self.run_next_phase()

    def phase_2(self):
        """Phase 2: OCR Extraction - uses embedded OCRManager."""
        project_path = Path(self.config.project_path)
        chi_dir = project_path / "chi"

        if self.subphase == 0:
            # Detect which files need processing
            statuses = detect_file_statuses(project_path)
            pending_videos = [
                f for f in get_video_files(project_path, sort_by_name=True)
                if statuses.get(f.name) == FileStatus.QUEUED
            ]

            # Log skipped files
            skipped_count = sum(1 for s in statuses.values() if s == FileStatus.DONE)
            if skipped_count > 0:
                self.output_received.emit(f"Skipping {skipped_count} already-processed file(s)\n")

            if not pending_videos:
                # No files to process
                self.output_received.emit("No video files to process\n")
                self.phase_completed.emit("OCR Extraction")
                self.current_phase += 1
                self.subphase = 0
                self.run_next_phase()
                return

            # Create and configure OCRManager with file config store
            self.ocr_manager = OCRManager(self.config, self.file_config_store, self)
            self.ocr_manager.file_status_changed.connect(self.ocr_file_status.emit)
            self.ocr_manager.file_status_text_changed.connect(self.ocr_file_status_text.emit)
            self.ocr_manager.file_progress_updated.connect(self.ocr_file_progress.emit)
            self.ocr_manager.file_log_output.connect(self.ocr_log_output.emit)
            self.ocr_manager.timing_updated.connect(self.ocr_timing_updated.emit)
            self.ocr_manager.overall_progress.connect(self.ocr_overall_progress.emit)
            self.ocr_manager.all_completed.connect(self._on_ocr_completed)

            # Set files and start processing
            self.ocr_manager.set_files(pending_videos, chi_dir)
            self.ocr_manager.start()

        elif self.subphase == 1:
            self.phase_completed.emit("OCR Extraction")
            self.current_phase += 1
            self.subphase = 0
            self.run_next_phase()

    def _on_ocr_completed(self, successful: int, total: int, total_time: float, avg_time: float):
        """Handle OCR completion from OCRManager."""
        self.output_received.emit(f"\nOCR completed: {successful}/{total} files successful\n")

        if successful < total:
            failed = total - successful
            self.output_received.emit(f"Warning: {failed} file(s) failed OCR\n")

        # Store timing for completion dialog
        self._ocr_total_time = total_time
        self._ocr_avg_time = avg_time

        # Clean up OCR manager
        self.ocr_manager = None

        # Continue to next phase
        self.subphase += 1
        self.phase_2()

    def phase_3(self):
        """Phase 3: Quality Assurance - Run ass-qafix twice."""
        chi_dir = Path(self.config.project_path) / "chi"

        if self.subphase == 0 and not shutil.which('ass-qafix'):
            self.phase_completed.emit("Quality Assurance (skipped - ass-qafix not found)")
            self.current_phase += 1
            self.subphase = 0
            self.run_next_phase()
            return

        if self.subphase == 0:
            # Run ass-qafix first pass
            cmd = ["ass-qafix", "--inplace"]
            self.run_command(cmd, cwd=str(chi_dir))
        elif self.subphase == 1:
            # Run ass-qafix second pass (sometimes first pass doesn't fix all issues)
            cmd = ["ass-qafix", "--inplace"]
            self.run_command(cmd, cwd=str(chi_dir))
        elif self.subphase == 2:
            self.phase_completed.emit("Quality Assurance")
            self.current_phase += 1
            self.subphase = 0
            self.run_next_phase()

    def run_command(self, cmd: list, cwd: str = None):
        """Execute command and emit output signals."""
        self.command_started.emit(" ".join(cmd))

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.on_stdout)
        self.process.readyReadStandardError.connect(self.on_stderr)
        self.process.finished.connect(self.on_process_finished)

        if cwd:
            self.process.setWorkingDirectory(cwd)

        self.process.start(cmd[0], cmd[1:])

    def on_stdout(self):
        """Handle stdout output."""
        data = self.process.readAllStandardOutput().data().decode("utf-8")
        self.output_received.emit(data)

    def on_stderr(self):
        """Handle stderr output."""
        data = self.process.readAllStandardError().data().decode("utf-8")
        self.output_received.emit(data)

    def on_process_finished(self, exit_code, exit_status):
        """Handle process completion."""
        if exit_code != 0:
            self.error_occurred.emit(f"Command failed with exit code {exit_code}")
            self.pipeline_finished.emit(False)
            return

        # Advance subphase or phase
        self.subphase += 1

        # Check if we need to continue with subphases
        phase_method = getattr(self, f"phase_{self.current_phase + 1}")
        phase_method()

    def stop(self):
        """Stop pipeline execution."""
        # Stop OCR manager if running
        if self.ocr_manager:
            self.ocr_manager.stop()
            self.ocr_manager = None

        # Stop QProcess if running
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            self.process.terminate()
            self.process.waitForFinished(3000)
            if self.process.state() == QProcess.ProcessState.Running:
                self.process.kill()

        # Notify UI that pipeline was stopped
        self.pipeline_stopped.emit()

    def is_running(self) -> bool:
        """Check if pipeline is running."""
        if self.ocr_manager and self.ocr_manager.is_running():
            return True
        return self.process is not None and self.process.state() == QProcess.ProcessState.Running

    def get_ocr_timing(self) -> tuple[float, float]:
        """Get OCR timing data (total_time, avg_time)."""
        return self._ocr_total_time, self._ocr_avg_time
