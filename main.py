#!/usr/bin/env python3
"""Main application entry point."""
import subprocess
import sys
import shutil
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLineEdit, QPushButton, QCheckBox,
                              QSpinBox, QLabel, QMessageBox, QToolButton,
                              QSlider, QSizePolicy, QFileDialog, QProgressBar)
from PyQt6.QtCore import Qt, QSize, QFileSystemWatcher, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut, QIcon

from core.config import Config, validate_config, FileConfig, FileConfigStore, ProjectConfigManager
from core.config_saver import AsyncConfigSaver
from core.log_store import LogStore
from core.pipeline import Pipeline, get_video_files, detect_file_statuses
from core.video_utils import VideoMetadataScanner
from resources import get_icon_path
from theme import apply_theme
from widgets.crop_selector import CropSelectorDialog
from widgets.brightness_tester import BrightnessTesterDialog
from widgets.phase_indicator import PhaseIndicator
from widgets.time_range_slider import TimeRangeSlider
from widgets.file_table import FileTableWidget
from widgets.videocr_settings_dialog import VideoCRSettingsDialog
from widgets.label_settings_dialog import LabelSettingsDialog
from widgets.logs_dialog import LogsDialog


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

        # Per-file configuration store
        self.file_config_store = FileConfigStore()
        self.clipboard_config: FileConfig | None = None  # For copy/paste

        # VideoCR settings (temporary, reset on app restart)
        self.videocr_settings = {
            'ocr_lang': 'ch',
            'conf_threshold': '95',
            'sim_threshold': '82',
            'similar_image': '0.3',
        }

        # Label detection settings
        self.label_settings = {
            'labels_only': False,
            'label_min_duration': '1.0',
            'label_max_duration': '8.0',
            'label_conf_threshold': '95',
        }

        # Config persistence
        self._save_timer: QTimer | None = None
        self._pending_time_range: tuple[str, str] | None = None
        self._pending_file_configs: dict | None = None
        self._async_saver: AsyncConfigSaver | None = None

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
        self.file_table = None
        self.labels_checkbox = None
        self.labels_settings_btn = None
        self.start_button = None
        self.phase_indicator = None
        self.overall_progress = None
        self.timing_label = None
        self.logs_btn = None
        self.loading_label = None

        # Log store and dialog
        self.log_store = LogStore(self)
        self._logs_dialog: LogsDialog | None = None

        # Background scanner
        self._metadata_scanner: VideoMetadataScanner | None = None

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

        # Logs button (hidden until pipeline runs)
        self.logs_btn = QPushButton("Logs")
        self.logs_btn.setObjectName("secondary")
        self.logs_btn.setVisible(False)
        self.logs_btn.clicked.connect(self._on_logs_clicked)
        button_layout.addWidget(self.logs_btn)

        button_layout.addStretch()

        self.start_button = QPushButton("Start Processing")
        self.start_button.setObjectName("primary-action")
        self.start_button.clicked.connect(self.on_start_processing_clicked)
        button_layout.addWidget(self.start_button)

        main_layout.addLayout(button_layout)

        # Keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

        # Connect signals for auto-save
        self.crop_input.textChanged.connect(self._schedule_save)
        self.brightness_spin.valueChanged.connect(self._on_brightness_changed)
        self.time_range_slider.range_committed.connect(self._on_time_range_changed)
        self.parallel_slider.valueChanged.connect(self._schedule_save)
        self.labels_checkbox.toggled.connect(self._schedule_save)

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

        # Labels row
        labels_layout = QHBoxLayout()
        labels_label = QLabel("Labels:")
        labels_label.setMinimumWidth(label_width)
        labels_layout.addWidget(labels_label)
        self.labels_checkbox = QCheckBox("Enable label detection")
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.toggled.connect(self._on_labels_toggled)
        labels_layout.addWidget(self.labels_checkbox)
        self.labels_settings_btn = QToolButton()
        self.labels_settings_btn.setIcon(QIcon(str(get_icon_path("settings"))))
        self.labels_settings_btn.setIconSize(QSize(18, 18))
        self.labels_settings_btn.clicked.connect(self._open_label_settings)
        labels_layout.addWidget(self.labels_settings_btn)
        labels_layout.addStretch()
        layout.addLayout(labels_layout)

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

    def _schedule_save(self):
        """Schedule a debounced save (300ms after last change)."""
        if not self.project_path:
            return
        if self._save_timer is None:
            self._save_timer = QTimer(self)
            self._save_timer.setSingleShot(True)
            self._save_timer.timeout.connect(self._save_project_config)
        self._save_timer.start(300)

    def _save_project_config(self):
        """Save current configuration to .ocr.json asynchronously."""
        if not self.project_path:
            return

        # Build global settings from UI (fast, on main thread)
        global_settings = {
            'brightness': self.brightness_spin.value(),
            'ocr_parallel': self.parallel_slider.value(),
            'labels_enabled': self.labels_checkbox.isChecked(),
        }

        # Crop region
        crop_text = self.crop_input.text()
        if crop_text:
            try:
                parts = [int(x.strip()) for x in crop_text.split(',')]
                if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
                    global_settings['crop'] = {
                        'x': parts[0], 'y': parts[1],
                        'width': parts[2], 'height': parts[3]
                    }
            except ValueError:
                pass

        # Time range
        time_start, time_end = self.time_range_slider.get_time_strings()
        if time_start or time_end:
            global_settings['time_range'] = {'start': time_start, 'end': time_end}

        # VideoCR settings
        videocr_settings = dict(self.videocr_settings)

        # Label settings
        labels_settings = dict(self.label_settings)

        # Build save data using ProjectConfigManager's format
        config_manager = ProjectConfigManager(Path(self.project_path))
        save_data = config_manager.build_save_data(
            global_settings, videocr_settings, self.file_config_store, labels_settings
        )

        # Save asynchronously to prevent GUI blocking
        if self._async_saver is None:
            self._async_saver = AsyncConfigSaver(self)
        self._async_saver.save(str(config_manager.config_path), save_data)

    def _load_project_config(self):
        """Load project configuration from .ocr.json if it exists."""
        if not self.project_path:
            return

        config_manager = ProjectConfigManager(Path(self.project_path))
        if not config_manager.exists():
            return

        global_settings, videocr_settings, file_configs, labels_settings = config_manager.load()

        # Apply global settings to UI (block signals to avoid triggering saves)
        if 'brightness' in global_settings:
            self.brightness_spin.blockSignals(True)
            self.brightness_spin.setValue(global_settings['brightness'])
            self.brightness_spin.blockSignals(False)

        if 'ocr_parallel' in global_settings:
            self.parallel_slider.blockSignals(True)
            self.parallel_slider.setValue(global_settings['ocr_parallel'])
            self.parallel_slider.blockSignals(False)
            self._on_parallel_changed(global_settings['ocr_parallel'])

        if 'crop' in global_settings:
            crop = global_settings['crop']
            self.crop_input.blockSignals(True)
            self.crop_input.setText(f"{crop['x']}, {crop['y']}, {crop['width']}, {crop['height']}")
            self.crop_input.blockSignals(False)

        # Store pending time range - will be applied after duration is set in refresh_file_list
        if 'time_range' in global_settings:
            tr = global_settings['time_range']
            self._pending_time_range = (tr.get('start', ''), tr.get('end', ''))

        # Apply videocr settings
        if videocr_settings:
            self.videocr_settings.update(videocr_settings)

        # Apply label settings
        if labels_settings:
            self.label_settings.update(labels_settings)

        if 'labels_enabled' in global_settings:
            self.labels_checkbox.blockSignals(True)
            self.labels_checkbox.setChecked(global_settings['labels_enabled'])
            self.labels_checkbox.blockSignals(False)
            self.labels_settings_btn.setEnabled(global_settings['labels_enabled'])
            # Update crop enabled state
            if global_settings['labels_enabled'] and self.label_settings.get('labels_only', False):
                self.crop_input.setEnabled(False)
                self.crop_select_btn.setEnabled(False)

        # Store pending file configs - will be applied after files are scanned
        if file_configs:
            self._pending_file_configs = file_configs

    def _apply_pending_file_configs(self):
        """Apply pending per-file configurations after files are scanned."""
        if not self._pending_file_configs:
            return

        for filename, cfg in self._pending_file_configs.items():
            config = self.file_config_store.get(filename)
            if not config:
                continue  # File no longer exists

            if 'crop' in cfg:
                crop = cfg['crop']
                config.set_crop(crop['x'], crop['y'], crop['width'], crop['height'])

            if 'brightness' in cfg:
                config.brightness = cfg['brightness']

            if 'time_start' in cfg:
                config.time_start = cfg['time_start']

            if 'time_end' in cfg:
                config.time_end = cfg['time_end']

            # Update table indicator
            self.file_table.update_config_indicator(filename)

        self._pending_file_configs = None

    def create_pipeline_section(self) -> QWidget:
        """Create pipeline status section with phase indicator and file table."""
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

        # Loading indicator (shown during metadata scan)
        self.loading_label = QLabel("Scanning video files...")
        self.loading_label.setObjectName("muted")
        self.loading_label.setVisible(False)
        layout.addWidget(self.loading_label)

        # File table
        self.file_table = FileTableWidget()
        self.file_table.set_file_store(self.file_config_store)
        self.file_table.selection_changed.connect(self.on_file_selection_changed)
        self.file_table.config_action_requested.connect(self.on_config_action_requested)
        layout.addWidget(self.file_table)

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

        # Hide previous run stats and logs button
        self.timing_label.setVisible(False)
        self.logs_btn.setVisible(False)

        # Set up folder watcher
        if self.folder_watcher:
            self.folder_watcher.removePaths(self.folder_watcher.directories())
        else:
            self.folder_watcher = QFileSystemWatcher(self)
            self.folder_watcher.directoryChanged.connect(self.on_folder_changed)
        self.folder_watcher.addPath(directory)

        # Load saved project configuration
        self._load_project_config()

        # Scan and display video files
        self.refresh_file_list()

    def refresh_file_list(self):
        """Scan folder for video files and update the file table."""
        if not self.project_path:
            return

        # Stop any existing scanner
        if self._metadata_scanner is not None:
            self._metadata_scanner.stop()
            self._metadata_scanner.wait()
            self._metadata_scanner = None

        # Find video files (non-recursive) sorted by filename
        project_path = Path(self.project_path)
        video_files = get_video_files(project_path, sort_by_name=True)
        new_video_set = {f.name for f in video_files}

        # Only update if files changed
        if new_video_set != self.current_video_files:
            self.current_video_files = new_video_set

            # Clear old configs for removed files
            for old_filename in self.file_config_store.get_all_filenames():
                if old_filename not in new_video_set:
                    self.file_config_store.remove(old_filename)

            # Populate file table with sorted filenames and detected statuses
            if video_files:
                statuses = detect_file_statuses(project_path)
                self.file_table.set_files([f.name for f in video_files], statuses)

                # Show loading indicator and start background scan
                self.loading_label.setVisible(True)
                self._start_metadata_scan(video_files)
            else:
                self.file_table.clear()
                self.file_config_store.clear()
                self.loading_label.setVisible(False)
        else:
            # Files unchanged, but still need to apply pending configs
            self._apply_pending_file_configs()

    def _start_metadata_scan(self, video_files: list[Path]):
        """Start background scan of video metadata."""
        self._metadata_scanner = VideoMetadataScanner(video_files, self)
        self._metadata_scanner.file_scanned.connect(self._on_file_metadata_scanned)
        self._metadata_scanner.scan_complete.connect(self._on_metadata_scan_complete)
        self._metadata_scanner.start()

    def _on_file_metadata_scanned(self, filename: str, width: int, height: int, duration: int):
        """Handle metadata scanned for a single file."""
        config = self.file_config_store.get_or_create(filename)
        config.resolution_width = width
        config.resolution_height = height
        config.duration_seconds = duration
        self.file_table.update_resolution(filename, config.get_resolution_label())

    def _on_metadata_scan_complete(self, longest_filename: str, longest_duration: int):
        """Handle completion of metadata scan."""
        self.loading_label.setVisible(False)
        self._metadata_scanner = None

        # Update time range slider with longest video
        if longest_filename and longest_duration > 0:
            self.time_range_slider.set_duration(longest_duration, longest_filename)

            # Apply pending time range after duration is set
            if self._pending_time_range:
                start_str, end_str = self._pending_time_range
                self._pending_time_range = None
                # Block signals to avoid triggering save
                self.time_range_slider.blockSignals(True)
                self.time_range_slider.set_time_range(start_str, end_str)
                self.time_range_slider.blockSignals(False)

        # Apply pending per-file configs after files are scanned
        self._apply_pending_file_configs()

    def on_folder_changed(self, path: str):
        """Handle folder content changes."""
        # Don't refresh while pipeline is running to avoid disrupting progress
        if not self.is_running:
            self.refresh_file_list()

    def on_file_selection_changed(self, filenames: list[str]):
        """Handle file selection change - update inputs to reflect selected files' config."""
        if not filenames:
            # No selection - keep current values as global defaults
            return

        # Check if first selected file has custom config
        first_file = filenames[0]
        config = self.file_config_store.get(first_file)

        if config and config.has_custom_crop():
            crop = config.get_crop_tuple()
            self.crop_input.setText(f"{crop[0]}, {crop[1]}, {crop[2]}, {crop[3]}")
        else:
            self.crop_input.clear()

        self.brightness_spin.blockSignals(True)
        if config and config.has_custom_brightness():
            self.brightness_spin.setValue(config.brightness)
        else:
            self.brightness_spin.setValue(230)  # Default
        self.brightness_spin.blockSignals(False)

        if config and config.has_custom_time_range():
            self.time_range_slider.set_time_range(
                config.time_start or "",
                config.time_end or ""
            )
        else:
            # Reset to full duration
            self.time_range_slider.set_time_range("", "")

    def _get_target_files(self) -> list[str]:
        """Get target files for config changes: selected files or all if none selected."""
        selected = self.file_table.get_selected_filenames()
        if selected:
            return selected
        return self.file_table.get_all_filenames()

    def _apply_crop_to_files(self, x: int, y: int, w: int, h: int, filenames: list[str]):
        """Apply crop settings to specified files."""
        for filename in filenames:
            config = self.file_config_store.get_or_create(filename)
            config.set_crop(x, y, w, h)
            self.file_table.update_config_indicator(filename)
        self._schedule_save()

    def _apply_brightness_to_files(self, brightness: int, filenames: list[str]):
        """Apply brightness setting to specified files."""
        for filename in filenames:
            config = self.file_config_store.get_or_create(filename)
            config.brightness = brightness
            self.file_table.update_config_indicator(filename)
        self._schedule_save()

    def _apply_time_range_to_files(self, start_str: str, end_str: str, filenames: list[str]):
        """Apply time range setting to specified files."""
        for filename in filenames:
            config = self.file_config_store.get_or_create(filename)
            config.time_start = start_str if start_str else None
            config.time_end = end_str if end_str else None
            self.file_table.update_config_indicator(filename)
        self._schedule_save()

    def _on_brightness_changed(self, value: int):
        """Handle brightness spinbox change - apply to selected files or all if none selected."""
        target_files = self._get_target_files()
        self._apply_brightness_to_files(value, target_files)

    def _on_time_range_changed(self, start: int, end: int):
        """Handle time range slider change - apply to selected files or all if none selected."""
        start_str, end_str = self.time_range_slider.get_time_strings()
        target_files = self._get_target_files()
        self._apply_time_range_to_files(start_str, end_str, target_files)

    def on_config_action_requested(self, action: str, filename: str):
        """Handle config action from file table context menu (copy/paste)."""
        selected = self.file_table.get_selected_filenames()

        if action == "copy" and filename:
            # Copy settings from the specified file
            config = self.file_config_store.get(filename)
            if config:
                self.clipboard_config = FileConfig(
                    filename="clipboard",
                    crop_x=config.crop_x,
                    crop_y=config.crop_y,
                    crop_width=config.crop_width,
                    crop_height=config.crop_height,
                    brightness=config.brightness,
                    time_start=config.time_start,
                    time_end=config.time_end
                )

        elif action == "paste" and self.clipboard_config:
            # Paste settings to all selected files
            for f in selected:
                self.file_config_store.copy_settings_to_files(self.clipboard_config, [f])
                self.file_table.update_config_indicator(f)
            self._schedule_save()

    def on_crop_select_clicked(self):
        """Open crop selector dialog."""
        if not self.project_path:
            QMessageBox.warning(self, "Error", "Please select a project directory first")
            return

        # Use selected files, or all files if none selected
        project_path = Path(self.project_path)
        selected = self.file_table.get_selected_filenames()
        if selected:
            video_files = [project_path / f for f in selected]
        else:
            video_files = get_video_files(project_path)

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

        dialog = CropSelectorDialog([str(f) for f in video_files], existing_crop, self.last_timeline_position, parent=self)
        dialog.crop_selected.connect(self.on_crop_selected)
        if dialog.exec():
            self.last_selected_episode = dialog.get_selected_episode()
            self.last_timeline_position = dialog.get_timeline_position()

    def on_crop_selected(self, x: int, y: int, width: int, height: int):
        """Handle crop selection - apply to selected files or all if none selected."""
        self.crop_input.setText(f"{x}, {y}, {width}, {height}")
        # Apply to target files
        target_files = self._get_target_files()
        self._apply_crop_to_files(x, y, width, height, target_files)

    def on_brightness_test_clicked(self):
        """Open brightness tester dialog."""
        if not self.project_path:
            QMessageBox.warning(self, "Error", "Please select a project directory first")
            return

        # Use selected files, or all files if none selected
        project_path = Path(self.project_path)
        selected = self.file_table.get_selected_filenames()
        if selected:
            video_files = [project_path / f for f in selected]
        else:
            video_files = get_video_files(project_path)

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
            parent=self
        )
        dialog.brightness_selected.connect(self.on_brightness_selected)
        dialog.exec()

    def on_brightness_selected(self, brightness: int):
        """Handle brightness selection - apply to selected files or all if none selected."""
        self.brightness_spin.setValue(brightness)
        # Apply to target files
        target_files = self._get_target_files()
        self._apply_brightness_to_files(brightness, target_files)

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
        self._schedule_save()

    def _on_labels_toggled(self, checked: bool):
        """Handle labels checkbox toggle."""
        self.labels_settings_btn.setEnabled(checked)
        # If labels unchecked, re-enable crop controls
        # If labels checked and labels_only, dim crop controls
        if not checked:
            self.crop_input.setEnabled(True)
            self.crop_select_btn.setEnabled(True)
        elif self.label_settings.get('labels_only', False):
            self.crop_input.setEnabled(False)
            self.crop_select_btn.setEnabled(False)

    def _open_label_settings(self):
        """Open label settings dialog."""
        dialog = LabelSettingsDialog(self.label_settings, self)
        dialog.settings_changed.connect(self._on_label_settings_changed)
        dialog.exec()

    def _on_label_settings_changed(self, settings: dict):
        """Handle label settings changes."""
        self.label_settings.update(settings)
        # Update crop enabled state based on labels_only
        if self.label_settings.get('labels_only', False):
            self.crop_input.setEnabled(False)
            self.crop_select_btn.setEnabled(False)
        else:
            self.crop_input.setEnabled(True)
            self.crop_select_btn.setEnabled(True)
        self._schedule_save()

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

        # Apply label settings
        config.labels_enabled = self.labels_checkbox.isChecked()
        config.labels_only = self.label_settings.get('labels_only', False)
        config.label_min_duration = float(self.label_settings.get('label_min_duration', '1.0'))
        config.label_max_duration = float(self.label_settings.get('label_max_duration', '8.0'))
        config.label_conf_threshold = int(self.label_settings.get('label_conf_threshold', '95'))

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

        # Create and start pipeline with file config store
        self.pipeline = Pipeline(config, self.file_config_store)
        self.pipeline.error_occurred.connect(self.on_pipeline_error)
        self.pipeline.phase_started.connect(self.on_phase_started)
        self.pipeline.pipeline_finished.connect(self.on_pipeline_finished)

        # File table signals (table is pre-populated on folder load)
        self.pipeline.ocr_file_status.connect(self.file_table.update_status)
        self.pipeline.ocr_file_status_text.connect(self.file_table.update_status_text)
        self.pipeline.ocr_file_progress.connect(self.file_table.update_progress)

        # Timing signals
        self.pipeline.ocr_timing_updated.connect(self.on_timing_updated)
        self.pipeline.ocr_overall_progress.connect(self.on_overall_progress)

        # Stopped signal
        self.pipeline.pipeline_stopped.connect(self.on_pipeline_stopped)

        # Log signals
        self.pipeline.ocr_log_output.connect(self._on_ocr_log_output)
        self.pipeline.output_received.connect(self._on_pipeline_log_output)

        # Clear previous logs and show button
        self.log_store.clear()
        self.logs_btn.setVisible(True)

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

        # Hide progress bar but show stats in timing label
        self.overall_progress.setVisible(False)

        # Refresh file list to update statuses from filesystem
        self.current_video_files = set()  # Force refresh
        self.refresh_file_list()

        if success:
            self.phase_indicator.mark_complete()
            # Show completion stats in timing label
            total_time, avg_time = self.pipeline.get_ocr_timing()
            if total_time > 0:
                total_str = self._format_duration(total_time)
                avg_str = self._format_duration(avg_time)
                self.timing_label.setText(f"Finished in {total_str}  •  Average {avg_str}/file")
                self.timing_label.setVisible(True)
                msg = f"Finished in {total_str} | Avg: {avg_str}/file"
            else:
                self.timing_label.setVisible(False)
                msg = "All phases completed"
            self._send_notification("OCR Complete", msg)
        else:
            self.timing_label.setVisible(False)
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

    def _on_ocr_log_output(self, filename: str, text: str):
        """Accumulate per-file OCR output in the log store."""
        self.log_store.append(filename, text)

    def _on_pipeline_log_output(self, text: str):
        """Accumulate Phase 3 (QA) output in the log store."""
        self.log_store.append(LogStore.QA_KEY, text)

    def _on_logs_clicked(self):
        """Open or raise the logs dialog."""
        if self._logs_dialog is None or not self._logs_dialog.isVisible():
            self._logs_dialog = LogsDialog(self.log_store, self)
        self._logs_dialog.show()
        self._logs_dialog.raise_()
        self._logs_dialog.activateWindow()

    def disable_ui(self):
        """Disable UI during processing."""
        self.folder_btn.setEnabled(False)
        self.crop_input.setEnabled(False)
        self.crop_select_btn.setEnabled(False)
        self.brightness_spin.setEnabled(False)
        self.brightness_test_btn.setEnabled(False)
        self.time_range_slider.setEnabled(False)
        self.parallel_slider.setEnabled(False)
        self.labels_checkbox.setEnabled(False)
        self.labels_settings_btn.setEnabled(False)
        # Note: file_table is not disabled to allow scrolling during processing

    def enable_ui(self):
        """Re-enable UI after processing."""
        self.folder_btn.setEnabled(True)
        self.brightness_spin.setEnabled(True)
        self.brightness_test_btn.setEnabled(True)
        self.time_range_slider.setEnabled(True)
        self.parallel_slider.setEnabled(True)
        self.labels_checkbox.setEnabled(True)
        self.labels_settings_btn.setEnabled(self.labels_checkbox.isChecked())
        # Respect labels_only for crop controls
        labels_only = self.labels_checkbox.isChecked() and self.label_settings.get('labels_only', False)
        self.crop_input.setEnabled(not labels_only)
        self.crop_select_btn.setEnabled(not labels_only)

    def check_dependencies(self):
        """Check if required CLI tools are available."""
        tools = ['ffmpeg']

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
                if self._metadata_scanner:
                    self._metadata_scanner.stop()
                    self._metadata_scanner.wait()
                if self._async_saver:
                    self._async_saver.stop()
                event.accept()
            else:
                event.ignore()
        else:
            # Stop metadata scanner if running
            if self._metadata_scanner:
                self._metadata_scanner.stop()
                self._metadata_scanner.wait()
            # Stop async saver thread
            if self._async_saver:
                self._async_saver.stop()
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
