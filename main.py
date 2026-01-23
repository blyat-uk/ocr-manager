#!/usr/bin/env python3
"""Main application entry point."""
import subprocess
import sys
import shutil
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLineEdit, QPushButton,
                              QSpinBox, QLabel, QMessageBox,
                              QSlider, QSizePolicy, QFileDialog, QProgressBar)
from PyQt6.QtCore import Qt, QFileSystemWatcher
from PyQt6.QtGui import QKeySequence, QShortcut

from core.config import Config, validate_config
from core.pipeline import Pipeline, get_video_files, detect_file_statuses
from core.video_utils import get_video_duration
from theme import apply_theme
from widgets.crop_selector import CropSelectorDialog
from widgets.brightness_tester import BrightnessTesterDialog
from widgets.phase_indicator import PhaseIndicator
from widgets.time_range_slider import TimeRangeSlider
from widgets.progress_table import ProgressTableWidget
from widgets.videocr_settings_dialog import VideoCRSettingsDialog


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Tool")
        self.resize(1000, 700)

        # State
        self.project_path = None
        self.pipeline = None
        self.is_running = False
        self.last_selected_episode = 0
        self.last_timeline_position = 5000  # Default 50% (on 0-10000 scale)
        self.folder_watcher = None
        self.current_video_files = set()  # Track current video files for change detection

        # VideoCR settings (temporary, reset on app restart)
        self.videocr_settings = {
            'ocr_lang': 'ch',
            'conf_threshold': '95',
            'sim_threshold': '82',
            'similar_image': '0.3',
        }

        # UI components
        self.folder_btn = None
        self.folder_path_label = None
        self.crop_input = None
        self.crop_select_btn = None
        self.brightness_spin = None
        self.brightness_test_btn = None
        self.time_range_slider = None
        self.parallel_slider = None
        self.parallel_label = None
        self.progress_table = None
        self.start_button = None
        self.phase_indicator = None
        self.overall_progress = None
        self.timing_label = None

        self.init_ui()
        self.check_dependencies()

    def update_window_title(self):
        """Update window title with project name and running status."""
        base = "OCR Tool"
        if self.project_path:
            project_name = Path(self.project_path).name
            base = f"OCR Tool - {project_name}"

        if self.is_running:
            self.setWindowTitle(f"[Running] {base}")
        else:
            self.setWindowTitle(base)

    def init_ui(self):
        """Setup all UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Folder selection row
        folder_layout = QHBoxLayout()
        self.folder_btn = QPushButton("Select Folder")
        self.folder_btn.setObjectName("secondary")
        self.folder_btn.clicked.connect(self.on_folder_select_clicked)
        folder_layout.addWidget(self.folder_btn)
        self.folder_path_label = QLabel("No folder selected")
        self.folder_path_label.setObjectName("muted")
        folder_layout.addWidget(self.folder_path_label, 1)
        main_layout.addLayout(folder_layout)

        # Configuration section
        config_widget = self.create_config_section()
        main_layout.addWidget(config_widget)

        # Pipeline section (includes phase indicator and progress table) - stretches to fill
        pipeline_widget = self.create_pipeline_section()
        main_layout.addWidget(pipeline_widget, 1)

        # Bottom bar: progress/timing on left, start button on right
        button_layout = QHBoxLayout()

        # Progress bar (hidden until processing starts)
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(0)
        self.overall_progress.setTextVisible(True)
        self.overall_progress.setFormat("%v/%m files")
        self.overall_progress.setMinimumWidth(200)
        self.overall_progress.setMaximumWidth(300)
        self.overall_progress.setVisible(False)
        button_layout.addWidget(self.overall_progress)

        # Timing label (hidden until processing starts)
        self.timing_label = QLabel("")
        self.timing_label.setObjectName("muted")
        self.timing_label.setVisible(False)
        button_layout.addWidget(self.timing_label)

        button_layout.addStretch()

        self.start_button = QPushButton("Start Processing")
        self.start_button.setObjectName("primary-action")
        self.start_button.clicked.connect(self.on_start_processing_clicked)
        button_layout.addWidget(self.start_button)

        main_layout.addLayout(button_layout)

        # Keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

    def create_config_section(self) -> QWidget:
        """Create configuration section with OCR parameters."""
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Fixed label width for alignment
        label_width = 90
        button_width = 70

        # Crop region row
        crop_layout = QHBoxLayout()
        crop_label = QLabel("Crop Region:")
        crop_label.setMinimumWidth(label_width)
        crop_layout.addWidget(crop_label)
        self.crop_input = QLineEdit()
        self.crop_input.setPlaceholderText("x, y, width, height")
        self.crop_input.setMaximumWidth(180)
        crop_layout.addWidget(self.crop_input)
        self.crop_select_btn = QPushButton("Select")
        self.crop_select_btn.setObjectName("secondary")
        self.crop_select_btn.setMinimumWidth(button_width)
        self.crop_select_btn.clicked.connect(self.on_crop_select_clicked)
        crop_layout.addWidget(self.crop_select_btn)
        crop_layout.addStretch()
        layout.addLayout(crop_layout)

        # Brightness row (aligned with crop row)
        brightness_layout = QHBoxLayout()
        brightness_label = QLabel("Brightness:")
        brightness_label.setMinimumWidth(label_width)
        brightness_layout.addWidget(brightness_label)
        self.brightness_spin = QSpinBox()
        self.brightness_spin.setRange(0, 255)
        self.brightness_spin.setValue(230)
        self.brightness_spin.setMinimumWidth(180)
        self.brightness_spin.setMaximumWidth(180)
        brightness_layout.addWidget(self.brightness_spin)
        self.brightness_test_btn = QPushButton("Test")
        self.brightness_test_btn.setObjectName("secondary")
        self.brightness_test_btn.setMinimumWidth(button_width)
        self.brightness_test_btn.clicked.connect(self.on_brightness_test_clicked)
        brightness_layout.addWidget(self.brightness_test_btn)
        brightness_layout.addStretch()
        layout.addLayout(brightness_layout)

        # Time range slider
        self.time_range_slider = TimeRangeSlider()
        layout.addWidget(self.time_range_slider)

        # Parallel workers slider row
        parallel_layout = QHBoxLayout()
        parallel_label = QLabel("Parallel:")
        parallel_label.setMinimumWidth(label_width)
        parallel_layout.addWidget(parallel_label)

        self.parallel_slider = QSlider(Qt.Orientation.Horizontal)
        self.parallel_slider.setRange(1, 8)
        self.parallel_slider.setValue(4)
        self.parallel_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.parallel_slider.setTickInterval(1)
        self.parallel_slider.valueChanged.connect(self._on_parallel_changed)
        parallel_layout.addWidget(self.parallel_slider)

        self.parallel_label = QLabel("4 workers")
        self.parallel_label.setMinimumWidth(70)
        parallel_layout.addWidget(self.parallel_label)

        layout.addLayout(parallel_layout)

        return widget

    def _on_parallel_changed(self, value: int):
        """Update parallel workers label."""
        self.parallel_label.setText(f"{value} workers")

    def create_pipeline_section(self) -> QWidget:
        """Create pipeline status section with phase indicator and progress table."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Pipeline status section
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Pipeline Status:"))
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # Phase indicator (index 1 = "OCR Extraction" is clickable for settings)
        self.phase_indicator = PhaseIndicator(Pipeline.PHASES, clickable_indices=[1])
        self.phase_indicator.badge_clicked.connect(self.on_phase_badge_clicked)
        layout.addWidget(self.phase_indicator)

        # Progress table
        self.progress_table = ProgressTableWidget()
        layout.addWidget(self.progress_table)

        return widget

    def on_folder_select_clicked(self):
        """Open folder browser dialog."""
        start_dir = self.project_path or "/mnt/FAST/work/"
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Project Directory",
            start_dir
        )
        if folder:
            self.set_project_directory(folder)

    def set_project_directory(self, directory: str):
        """Set project directory and initialize time range slider."""
        self.project_path = directory
        self.folder_path_label.setText(directory)
        self.folder_path_label.setObjectName("")  # Remove muted style
        self.folder_path_label.style().unpolish(self.folder_path_label)
        self.folder_path_label.style().polish(self.folder_path_label)
        self.update_window_title()

        # Set up folder watcher
        if self.folder_watcher:
            self.folder_watcher.removePaths(self.folder_watcher.directories())
        else:
            self.folder_watcher = QFileSystemWatcher(self)
            self.folder_watcher.directoryChanged.connect(self.on_folder_changed)
        self.folder_watcher.addPath(directory)

        # Scan and display video files
        self.refresh_file_list()

    def refresh_file_list(self):
        """Scan folder for video files and update the progress table."""
        if not self.project_path:
            return

        # Find video files (non-recursive) sorted by filename
        project_path = Path(self.project_path)
        video_files = get_video_files(project_path, sort_by_name=True)
        new_video_set = {f.name for f in video_files}

        # Only update if files changed
        if new_video_set != self.current_video_files:
            self.current_video_files = new_video_set

            # Populate progress table with sorted filenames and detected statuses
            if video_files:
                statuses = detect_file_statuses(project_path)
                self.progress_table.set_files([f.name for f in video_files], statuses)
            else:
                self.progress_table.clear()

        # Update time range slider with longest video
        if video_files:
            longest_file = None
            longest_duration = 0
            for video in video_files:
                try:
                    duration = get_video_duration(str(video))
                    if duration > longest_duration:
                        longest_duration = duration
                        longest_file = video
                except Exception:
                    continue

            if longest_file and longest_duration > 0:
                self.time_range_slider.set_duration(
                    longest_duration,
                    longest_file.name
                )

    def on_folder_changed(self, path: str):
        """Handle folder content changes."""
        # Don't refresh while pipeline is running to avoid disrupting progress
        if not self.is_running:
            self.refresh_file_list()

    def on_crop_select_clicked(self):
        """Open crop selector dialog."""
        if not self.project_path:
            QMessageBox.warning(self, "Error", "Please select a project directory first")
            return

        # Find video files
        video_files = get_video_files(Path(self.project_path))
        if not video_files:
            QMessageBox.warning(self, "Error", "No video files found in project directory")
            return

        # Get existing crop coordinates if available
        existing_crop = None
        crop_text = self.crop_input.text()
        if crop_text:
            try:
                parts = [int(x.strip()) for x in crop_text.split(',')]
                if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
                    existing_crop = tuple(parts)
            except ValueError:
                pass

        dialog = CropSelectorDialog([str(f) for f in video_files], existing_crop, self.last_timeline_position, self)
        dialog.crop_selected.connect(self.on_crop_selected)
        if dialog.exec():
            self.last_selected_episode = dialog.get_selected_episode()
            self.last_timeline_position = dialog.get_timeline_position()

    def on_crop_selected(self, x: int, y: int, width: int, height: int):
        """Handle crop selection."""
        self.crop_input.setText(f"{x}, {y}, {width}, {height}")

    def on_brightness_test_clicked(self):
        """Open brightness tester dialog."""
        if not self.project_path:
            QMessageBox.warning(self, "Error", "Please select a project directory first")
            return

        # Find video files
        video_files = get_video_files(Path(self.project_path))
        if not video_files:
            QMessageBox.warning(self, "Error", "No video files found in project directory")
            return

        # Parse crop coordinates if available
        crop_region = None
        crop_text = self.crop_input.text()
        if crop_text:
            try:
                parts = [int(x.strip()) for x in crop_text.split(',')]
                if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
                    crop_region = tuple(parts)  # (x, y, width, height)
            except ValueError:
                pass

        dialog = BrightnessTesterDialog(
            [str(f) for f in video_files],
            self.last_selected_episode,
            self.last_timeline_position,
            self.brightness_spin.value(),
            crop_region,
            self
        )
        dialog.brightness_selected.connect(self.on_brightness_selected)
        dialog.exec()

    def on_brightness_selected(self, brightness: int):
        """Handle brightness selection."""
        self.brightness_spin.setValue(brightness)

    def on_phase_badge_clicked(self, index: int):
        """Handle phase badge click."""
        if index == 1:  # OCR Extraction
            self.open_videocr_settings()

    def open_videocr_settings(self):
        """Open the videocr settings dialog."""
        dialog = VideoCRSettingsDialog(self.videocr_settings, self)
        dialog.settings_changed.connect(self.on_videocr_settings_changed)
        dialog.exec()

    def on_videocr_settings_changed(self, settings: dict):
        """Handle videocr settings changes."""
        self.videocr_settings.update(settings)

    def _format_duration(self, seconds: float) -> str:
        """Format duration as human-readable string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _send_notification(self, title: str, message: str, urgency: str = "normal"):
        """Send a desktop notification."""
        try:
            subprocess.run(
                ["notify-send", "-a", "OCR Manager", "-u", urgency, title, message],
                check=False
            )
        except FileNotFoundError:
            pass  # notify-send not available

    def on_timing_updated(self, elapsed: float, eta: float, avg_per_file: float):
        """Handle timing update from pipeline."""
        elapsed_str = self._format_duration(elapsed)
        eta_str = self._format_duration(eta) if eta > 0 else "calculating..."
        self.timing_label.setText(f"Elapsed: {elapsed_str}  •  Remaining: {eta_str}")

    def on_overall_progress(self, completed: int, total: int):
        """Handle overall progress update from pipeline."""
        self.overall_progress.setMaximum(total)
        self.overall_progress.setValue(completed)

    def on_start_processing_clicked(self):
        """Start or stop pipeline processing."""
        if self.is_running:
            # Stop pipeline
            if self.pipeline:
                self.pipeline.stop()
            return

        # Validate project directory
        if not self.project_path:
            QMessageBox.warning(self, "Error", "Please select a project directory first")
            return

        # Build config from UI
        config = Config(project_path=self.project_path)
        config.brightness = self.brightness_spin.value()
        config.time_start, config.time_end = self.time_range_slider.get_time_strings()
        config.ocr_parallel = self.parallel_slider.value()

        # Apply videocr settings
        config.ocr_lang = self.videocr_settings['ocr_lang']
        config.conf_threshold = int(self.videocr_settings['conf_threshold'])
        config.sim_threshold = int(self.videocr_settings['sim_threshold'])
        config.similar_image = float(self.videocr_settings['similar_image'])

        # Parse crop values
        crop_text = self.crop_input.text()
        if crop_text:
            try:
                parts = [int(x.strip()) for x in crop_text.split(',')]
                if len(parts) == 4:
                    config.crop_x, config.crop_y, config.crop_width, config.crop_height = parts
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid crop values")
                return

        # Validate config
        valid, error_msg = validate_config(config)
        if not valid:
            QMessageBox.warning(self, "Configuration Error", error_msg)
            return

        # Create and start pipeline
        self.pipeline = Pipeline(config)
        self.pipeline.error_occurred.connect(self.on_pipeline_error)
        self.pipeline.phase_started.connect(self.on_phase_started)
        self.pipeline.pipeline_finished.connect(self.on_pipeline_finished)

        # Progress table signals (table is pre-populated on folder load)
        self.pipeline.ocr_file_status.connect(self.progress_table.update_status)
        self.pipeline.ocr_file_progress.connect(self.progress_table.update_progress)

        # Timing signals
        self.pipeline.ocr_timing_updated.connect(self.on_timing_updated)
        self.pipeline.ocr_overall_progress.connect(self.on_overall_progress)

        # Stopped signal
        self.pipeline.pipeline_stopped.connect(self.on_pipeline_stopped)

        # Update UI
        self.is_running = True
        self.start_button.setText("Stop")
        self.start_button.setObjectName("danger-action")
        # Force style refresh
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self.update_window_title()
        self.disable_ui()

        # Show progress widgets and reset them
        self.overall_progress.setValue(0)
        self.overall_progress.setVisible(True)
        self.timing_label.setText("Elapsed: 0s  •  Remaining: calculating...")
        self.timing_label.setVisible(True)

        # Reset phase indicator (table keeps its pre-loaded DONE/QUEUED statuses)
        self.phase_indicator.reset()

        # Start pipeline
        self.pipeline.start()

    def on_phase_started(self, phase_index: int, phase_name: str):
        """Handle phase start - update phase indicator."""
        self.phase_indicator.set_active_phase(phase_index)

    def on_pipeline_error(self, error_msg: str):
        """Handle pipeline errors."""
        # Mark current phase as error
        if self.pipeline:
            self.phase_indicator.mark_error(self.pipeline.current_phase)

    def on_pipeline_finished(self, success: bool):
        """Handle pipeline completion."""
        self.is_running = False
        self.start_button.setText("Start Processing")
        self.start_button.setObjectName("primary-action")
        # Force style refresh
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self.update_window_title()
        self.enable_ui()

        # Hide progress widgets
        self.overall_progress.setVisible(False)
        self.timing_label.setVisible(False)

        # Refresh file list to update statuses from filesystem
        self.current_video_files = set()  # Force refresh
        self.refresh_file_list()

        if success:
            self.phase_indicator.mark_complete()
            # Build completion message with timing info
            total_time, avg_time = self.pipeline.get_ocr_timing()
            if total_time > 0:
                total_str = self._format_duration(total_time)
                avg_str = self._format_duration(avg_time)
                completed = self.overall_progress.value()
                msg = f"Total: {total_str} | Files: {completed} | Avg: {avg_str}/file"
            else:
                msg = "All phases completed"
            self._send_notification("OCR Complete", msg)
        else:
            self._send_notification("OCR Failed", "Pipeline encountered an error", "critical")

    def on_pipeline_stopped(self):
        """Handle user-initiated pipeline stop."""
        self.is_running = False
        self.start_button.setText("Start Processing")
        self.start_button.setObjectName("primary-action")
        # Force style refresh
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self.update_window_title()
        self.enable_ui()

        # Hide progress widgets
        self.overall_progress.setVisible(False)
        self.timing_label.setVisible(False)

        # Refresh file list to update statuses from filesystem
        self.current_video_files = set()  # Force refresh
        self.refresh_file_list()

    def disable_ui(self):
        """Disable UI during processing."""
        self.folder_btn.setEnabled(False)
        self.crop_input.setEnabled(False)
        self.crop_select_btn.setEnabled(False)
        self.brightness_spin.setEnabled(False)
        self.brightness_test_btn.setEnabled(False)
        self.time_range_slider.setEnabled(False)
        self.parallel_slider.setEnabled(False)

    def enable_ui(self):
        """Re-enable UI after processing."""
        self.folder_btn.setEnabled(True)
        self.crop_input.setEnabled(True)
        self.crop_select_btn.setEnabled(True)
        self.brightness_spin.setEnabled(True)
        self.brightness_test_btn.setEnabled(True)
        self.time_range_slider.setEnabled(True)
        self.parallel_slider.setEnabled(True)

    def check_dependencies(self):
        """Check if required CLI tools are available."""
        tools = ['ass-qafix', 'ffmpeg']

        missing = []
        for tool in tools:
            if not shutil.which(tool):
                missing.append(tool)

        if missing:
            QMessageBox.critical(
                self, "Missing Dependencies",
                f"Required tools not found in PATH:\n{', '.join(missing)}\n\n"
                f"Please install them before using this application."
            )

    def closeEvent(self, event):
        """Handle window close."""
        if self.is_running:
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Pipeline is running. Stop and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                if self.pipeline:
                    self.pipeline.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Apply dark theme
    apply_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
