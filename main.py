#!/usr/bin/env python3
"""Main application entry point."""
import subprocess
import sys
import shutil
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QGridLayout, QLineEdit, QPushButton,
                              QCheckBox, QSpinBox, QLabel, QMessageBox, QGroupBox,
                              QSlider, QSizePolicy, QFileDialog, QProgressBar,
                              QProgressDialog)
from PyQt6.QtCore import Qt, QFileSystemWatcher, QTimer, QSize, QSettings
from PyQt6.QtGui import QKeySequence, QShortcut
import qtawesome as qta

from core.config import Config, validate_config, FileConfig, FileConfigStore, ProjectConfigManager
from core.config_saver import AsyncConfigSaver
from core.log_store import LogStore
from core.pipeline import Pipeline, get_video_files, detect_file_statuses
from core.ocr_worker import FileStatus
from core.video_utils import VideoMetadataScanner
from theme import apply_theme
from widgets.crop_selector import CropSelectorDialog
from widgets.brightness_tester import BrightnessTesterDialog
from widgets.time_range_slider import TimeRangeSlider
from widgets.time_range_chips import TimeRangeChipsWidget
from widgets.file_table import FileTableWidget
from widgets.settings_dialog import SettingsDialog
from widgets.file_details_dialog import FileDetailsDialog, FileDetailsData
from widgets.logs_dialog import LogsDialog
from widgets.subtitle_preview_dialog import SubtitlePreviewDialog
from core.audio_analysis import AudioAnalysisWorker
from core.subtitle_detector import SubtitleDetectionWorker


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

        # Label mask crops (list of (x, y, w, h) tuples)
        self.label_mask_crops: list[tuple] = []

        # Label detection settings
        self.label_settings = {
            'label_min_duration': '0.5',
            'label_max_duration': '5.0',
            'label_conf_threshold': '95',
            'label_conf_threshold_min': '80',
        }

        # Autodetect settings
        self.autodetect_settings = {
            'min_segment_length': '30',
            'merge_repeating_silences': 'false',
        }

        # Automation settings (auto-crop parameters)
        self.automation_settings = {
            'crop_width_fraction': '0.70',
            'crop_vertical_padding': '0',
            'crop_min_height_fraction': '0.05',
            'bottom_half_cutoff': '0.50',
        }

        # Config persistence
        self._save_timer: QTimer | None = None
        self._pending_time_range: tuple[str, str] | None = None
        self._pending_file_configs: dict | None = None
        self._async_saver: AsyncConfigSaver | None = None

        # UI components
        self.folder_btn = None
        self.folder_path_label = None
        self.settings_btn = None
        self.settings_summary_label = None
        self.crop_input = None
        self.crop_select_btn = None
        self.brightness_spin = None
        self.brightness_test_btn = None
        self.time_range_slider = None
        self.time_range_chips = None
        self._active_range_index: int = -1
        self.parallel_slider = None
        self.parallel_label = None
        self.file_table = None
        self.dialogue_checkbox = None
        self.labels_checkbox = None
        self.processing_warning = None
        self.start_button = None
        self.overall_progress = None
        self.timing_label = None
        self.logs_btn = None
        self.loading_label = None
        self.auto_time_range_btn = None
        self.detect_subtitle_btn = None
        self._audio_analysis_worker = None
        self._audio_progress_dialog = None
        self._subtitle_detection_worker = None
        self._subtitle_progress_dialog = None

        # Log store and dialog
        self.log_store = LogStore(self)
        self._logs_dialog: LogsDialog | None = None
        self._subtitle_preview: SubtitlePreviewDialog | None = None

        # Background scanner
        self._metadata_scanner: VideoMetadataScanner | None = None

        self.init_ui()
        self._restore_geometry()
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
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(12, 8, 12, 8)

        # Folder selection row: Open Folder + path + summary + settings cog
        folder_layout = QHBoxLayout()
        self.folder_btn = QPushButton("Open Folder")
        self.folder_btn.clicked.connect(self.on_folder_select_clicked)
        folder_layout.addWidget(self.folder_btn)
        self.folder_path_label = QLabel("No folder selected")
        folder_layout.addWidget(self.folder_path_label, 1)

        self.settings_summary_label = QLabel("")
        self.settings_summary_label.setStyleSheet("color: rgba(255,255,255,0.45); font-size: 12px;")
        folder_layout.addWidget(self.settings_summary_label)

        self.settings_btn = QPushButton()
        self.settings_btn.setIcon(qta.icon("mdi.cog", color='white'))
        self.settings_btn.setIconSize(QSize(22, 22))
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.setFixedSize(QSize(30, 30))
        self.settings_btn.clicked.connect(self._open_settings)
        folder_layout.addWidget(self.settings_btn)

        main_layout.addLayout(folder_layout)

        # Configuration section — 3 QGroupBox cards
        config_widget = self.create_config_section()
        main_layout.addWidget(config_widget)

        # Pipeline section (includes phase indicator and progress table) - stretches to fill
        pipeline_widget = self.create_pipeline_section()
        main_layout.addWidget(pipeline_widget, 1)

        # Bottom bar: logs icon, progress, start button
        button_layout = QHBoxLayout()

        self.logs_btn = QPushButton()
        self.logs_btn.setIcon(qta.icon("mdi.text-box-outline", color='white'))
        self.logs_btn.setIconSize(QSize(22, 22))
        self.logs_btn.setToolTip("View Logs")
        self.logs_btn.setFixedSize(QSize(30, 30))
        self.logs_btn.setVisible(False)
        self.logs_btn.clicked.connect(self._on_logs_clicked)
        button_layout.addWidget(self.logs_btn)

        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(0)
        self.overall_progress.setTextVisible(True)
        self.overall_progress.setFormat("%v/%m files")
        self.overall_progress.setMinimumWidth(200)
        self.overall_progress.setMaximumWidth(300)
        self.overall_progress.setVisible(False)
        button_layout.addWidget(self.overall_progress)

        button_layout.addStretch()

        self.timing_label = QLabel("Elapsed: --")
        button_layout.addWidget(self.timing_label)

        self.start_button = QPushButton("Start Processing")
        self.start_button.setCheckable(True)
        self.start_button.setChecked(True)
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
        self.dialogue_checkbox.toggled.connect(self._schedule_save)
        self.labels_checkbox.toggled.connect(self._schedule_save)

        # Initial summary
        self._update_settings_summary()

    def _make_icon_btn(self, icon_name: str, tooltip: str, size: int = 28) -> QPushButton:
        """Create an icon-only button using qtawesome."""
        btn = QPushButton()
        btn.setIcon(qta.icon(icon_name, color='white'))
        btn.setIconSize(QSize(size, size))
        btn.setToolTip(tooltip)
        btn.setFixedSize(QSize(size + 8, size + 8))
        return btn

    def create_config_section(self) -> QWidget:
        """Create configuration section with 3 QGroupBox cards."""
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Card 1: Crop & Brightness (grid for vertical alignment)
        crop_group = QGroupBox("Crop && Brightness")
        crop_grid = QGridLayout(crop_group)
        crop_grid.setContentsMargins(8, 4, 8, 4)
        crop_grid.setVerticalSpacing(4)
        crop_grid.setHorizontalSpacing(6)

        crop_grid.addWidget(QLabel("Crop"), 0, 0, Qt.AlignmentFlag.AlignRight)
        self.crop_input = QLineEdit()
        self.crop_input.setPlaceholderText("x, y, w, h")
        self.crop_input.setFixedWidth(250)
        crop_grid.addWidget(self.crop_input, 0, 1)
        self.crop_select_btn = self._make_icon_btn("mdi.crop", "Pick Crop")
        self.crop_select_btn.clicked.connect(self.on_crop_select_clicked)
        crop_grid.addWidget(self.crop_select_btn, 0, 2)

        crop_grid.addWidget(QLabel("Brightness"), 1, 0, Qt.AlignmentFlag.AlignRight)
        self.brightness_spin = QSpinBox()
        self.brightness_spin.setRange(0, 255)
        self.brightness_spin.setValue(230)
        self.brightness_spin.setFixedWidth(250)
        crop_grid.addWidget(self.brightness_spin, 1, 1)
        self.brightness_test_btn = self._make_icon_btn("mdi.brightness-6", "Preview Brightness")
        self.brightness_test_btn.clicked.connect(self.on_brightness_test_clicked)
        crop_grid.addWidget(self.brightness_test_btn, 1, 2)

        self.detect_subtitle_btn = self._make_icon_btn("mdi.text-search", "Detect subtitle frames")
        self.detect_subtitle_btn.clicked.connect(self._on_detect_subtitle_clicked)
        crop_grid.addWidget(self.detect_subtitle_btn, 0, 3)

        crop_grid.setColumnStretch(4, 1)
        layout.addWidget(crop_group)

        # Card 2: Time Range
        time_group = QGroupBox("Time Range")
        time_layout = QVBoxLayout(time_group)
        time_layout.setContentsMargins(8, 4, 8, 4)
        time_layout.setSpacing(4)
        # Row 1: slider + auto-detect
        slider_row = QHBoxLayout()
        self.time_range_slider = TimeRangeSlider()
        slider_row.addWidget(self.time_range_slider, 1)
        self.auto_time_range_btn = self._make_icon_btn("mdi.auto-fix", "Autodetect time ranges")
        self.auto_time_range_btn.clicked.connect(self._on_auto_time_range_clicked)
        slider_row.addWidget(self.auto_time_range_btn)
        time_layout.addLayout(slider_row)
        # Row 2: range chips
        self.time_range_chips = TimeRangeChipsWidget()
        self.time_range_chips.range_selected.connect(self._on_range_chip_selected)
        self.time_range_chips.range_removed.connect(self._on_range_chip_removed)
        self.time_range_chips.add_clicked.connect(self._on_add_range_clicked)
        time_layout.addWidget(self.time_range_chips)
        layout.addWidget(time_group)

        # Card 3: Processing
        proc_group = QGroupBox("Processing")
        proc_layout = QVBoxLayout(proc_group)
        proc_layout.setContentsMargins(8, 4, 8, 4)
        proc_layout.setSpacing(4)

        self.dialogue_checkbox = QCheckBox("Dialogue")
        self.dialogue_checkbox.setChecked(True)
        self.dialogue_checkbox.toggled.connect(self._on_processing_checkboxes_changed)
        proc_layout.addWidget(self.dialogue_checkbox)

        self.labels_checkbox = QCheckBox("Labels")
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.toggled.connect(self._on_processing_checkboxes_changed)
        proc_layout.addWidget(self.labels_checkbox)

        self.processing_warning = QLabel("At least one of Dialogue or Labels must be enabled")
        self.processing_warning.setStyleSheet("color: #f38ba8; font-size: 12px;")
        self.processing_warning.setVisible(False)
        proc_layout.addWidget(self.processing_warning)

        parallel_row = QHBoxLayout()
        parallel_row.addWidget(QLabel("Parallel"))
        self.parallel_slider = QSlider(Qt.Orientation.Horizontal)
        self.parallel_slider.setRange(1, 8)
        self.parallel_slider.setValue(4)
        self.parallel_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.parallel_slider.setTickInterval(1)
        self.parallel_slider.valueChanged.connect(self._on_parallel_changed)
        parallel_row.addWidget(self.parallel_slider, 1)
        self.parallel_label = QLabel("4")
        parallel_row.addWidget(self.parallel_label)
        proc_layout.addLayout(parallel_row)

        layout.addWidget(proc_group)

        return widget

    def _on_parallel_changed(self, value: int):
        """Update parallel workers label."""
        self.parallel_label.setText(str(value))

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
            'dialogue_enabled': self.dialogue_checkbox.isChecked(),
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

        # Time range (global: single range from slider, used as default for files without custom ranges)
        time_start, time_end = self.time_range_slider.get_time_strings()
        if time_start or time_end:
            global_settings['time_range'] = {'start': time_start, 'end': time_end}

        # VideoCR settings
        videocr_settings = dict(self.videocr_settings)

        # Label settings
        labels_settings = dict(self.label_settings)
        if self.label_mask_crops:
            labels_settings['mask_crops'] = [list(m) for m in self.label_mask_crops]

        # Autodetect settings
        autodetect_settings = dict(self.autodetect_settings)

        # Build save data using ProjectConfigManager's format
        config_manager = ProjectConfigManager(Path(self.project_path))
        save_data = config_manager.build_save_data(
            global_settings, videocr_settings, self.file_config_store, labels_settings
        )
        if autodetect_settings:
            save_data['autodetect'] = autodetect_settings

        # Automation settings (auto-crop parameters)
        save_data['automation'] = dict(self.automation_settings)

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

        # Load autodetect settings (not covered by ProjectConfigManager.load())
        autodetect_settings = config_manager.load_section('autodetect')
        if autodetect_settings:
            self.autodetect_settings.update(autodetect_settings)

        # Load automation settings
        automation_settings = config_manager.load_section('automation')
        if automation_settings:
            self.automation_settings.update(automation_settings)

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
            # Extract mask_crops before updating label_settings dict
            if 'mask_crops' in labels_settings:
                self.label_mask_crops = [tuple(m) for m in labels_settings['mask_crops']]
            else:
                self.label_mask_crops = []
            self.label_settings.update({k: v for k, v in labels_settings.items() if k != 'mask_crops'})

        if 'dialogue_enabled' in global_settings:
            self.dialogue_checkbox.blockSignals(True)
            self.dialogue_checkbox.setChecked(global_settings['dialogue_enabled'])
            self.dialogue_checkbox.blockSignals(False)

        if 'labels_enabled' in global_settings:
            self.labels_checkbox.blockSignals(True)
            self.labels_checkbox.setChecked(global_settings['labels_enabled'])
            self.labels_checkbox.blockSignals(False)

        # Sync UI state from loaded checkboxes
        self._on_processing_checkboxes_changed()

        # Store pending file configs - will be applied after files are scanned
        if file_configs:
            self._pending_file_configs = file_configs

        # Update summary with loaded settings
        self._update_settings_summary()

    def _apply_pending_file_configs(self):
        """Apply pending per-file configurations (custom settings and cached metadata)."""
        if not self._pending_file_configs:
            return

        for filename, cfg in self._pending_file_configs.items():
            config = self.file_config_store.get_or_create(filename)

            if 'crop' in cfg:
                crop = cfg['crop']
                config.set_crop(crop['x'], crop['y'], crop['width'], crop['height'])

            if 'brightness' in cfg:
                config.brightness = cfg['brightness']

            # Time ranges: new format or backward compat from old time_start/time_end
            if 'time_ranges' in cfg:
                config.time_ranges = [
                    (r.get('start') or None, r.get('end') or None)
                    for r in cfg['time_ranges']
                ]
            elif 'time_start' in cfg or 'time_end' in cfg:
                start = cfg.get('time_start')
                end = cfg.get('time_end')
                if start or end:
                    config.time_ranges = [(start or None, end or None)]

            if 'resolution' in cfg:
                res = cfg['resolution']
                config.resolution_width = res.get('width', 0)
                config.resolution_height = res.get('height', 0)
                self.file_table.update_resolution(filename, config.get_resolution_label())

            if 'duration' in cfg:
                config.duration_seconds = cfg['duration']

            if 'subtitle_position' in cfg:
                config.subtitle_position = cfg['subtitle_position']

            # Update table indicator
            self.file_table.update_config_indicator(filename)

        self._pending_file_configs = None

    def create_pipeline_section(self) -> QWidget:
        """Create pipeline status section with phase indicator and file table."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Status row: loading label
        status_row = QHBoxLayout()
        self.loading_label = QLabel("Scanning...")
        self.loading_label.setVisible(False)
        status_row.addWidget(self.loading_label)
        status_row.addStretch()
        layout.addLayout(status_row)

        # File table
        self.file_table = FileTableWidget()
        self.file_table.set_file_store(self.file_config_store)
        self.file_table.selection_changed.connect(self.on_file_selection_changed)
        self.file_table.config_action_requested.connect(self.on_config_action_requested)
        self.file_table.file_double_clicked.connect(self._on_file_double_clicked)
        self.file_table.file_details_requested.connect(self._on_file_details_requested)
        layout.addWidget(self.file_table)

        return widget

    def on_folder_select_clicked(self):
        """Open folder browser dialog using native system picker."""
        start_dir = self.project_path or "/mnt/FAST/work/"
        if shutil.which("kdialog"):
            result = subprocess.run(
                ["kdialog", "--getexistingdirectory", start_dir],
                capture_output=True, text=True,
            )
            folder = result.stdout.strip() if result.returncode == 0 else ""
        else:
            folder = QFileDialog.getExistingDirectory(
                self, "Select Project Directory", start_dir,
            )
        if folder:
            self.set_project_directory(folder)

    def set_project_directory(self, directory: str):
        """Set project directory and initialize time range slider."""
        # Stop subtitle detection if running
        if self._subtitle_detection_worker:
            self._subtitle_detection_worker.cancel()
            self._subtitle_detection_worker.cleanup()
            self._subtitle_detection_worker = None

        self.project_path = directory
        self.folder_path_label.setText(directory)
        self.update_window_title()

        # Reset previous run stats and hide logs/progress
        self.timing_label.setText("Elapsed: --")
        self.logs_btn.setVisible(False)
        self.overall_progress.setVisible(False)

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

                # Apply pending configs early so cached metadata is available
                self._apply_pending_file_configs()

                # Only scan files that don't already have cached resolution
                files_to_scan = [
                    f for f in video_files
                    if not self._has_cached_metadata(f.name)
                ]

                if files_to_scan:
                    self.loading_label.setVisible(True)
                    self._start_metadata_scan(files_to_scan)
                else:
                    # All files have cached metadata, finalize immediately
                    self._finalize_metadata()
            else:
                self.file_table.clear()
                self.file_config_store.clear()
                self.loading_label.setVisible(False)
        else:
            # Files unchanged, but still need to apply pending configs
            self._apply_pending_file_configs()

    def _has_cached_metadata(self, filename: str) -> bool:
        """Check if a file already has cached resolution metadata."""
        config = self.file_config_store.get(filename)
        return config is not None and config.resolution_height > 0 and config.duration_seconds > 0

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
        """Handle completion of metadata scan (for newly scanned files)."""
        self.loading_label.setVisible(False)
        self._metadata_scanner = None
        self._finalize_metadata()

    def _get_longest_duration(self, filenames: list[str] | None = None) -> tuple[int, str]:
        """Get the longest duration among specified files (or all files if None)."""
        source = filenames if filenames else self.file_config_store.get_all_filenames()
        longest_duration = 0
        longest_filename = ""
        for filename in source:
            config = self.file_config_store.get(filename)
            if config and config.duration_seconds > longest_duration:
                longest_duration = int(config.duration_seconds)
                longest_filename = filename
        return longest_duration, longest_filename

    def _finalize_metadata(self):
        """Set time range slider from longest video across all files (cached + scanned)."""
        longest_duration, longest_filename = self._get_longest_duration()

        if longest_filename and longest_duration > 0:
            self.time_range_slider.set_duration(longest_duration, longest_filename)

            if self._pending_time_range:
                start_str, end_str = self._pending_time_range
                self._pending_time_range = None
                self.time_range_slider.blockSignals(True)
                self.time_range_slider.set_time_range(start_str, end_str)
                self.time_range_slider.blockSignals(False)

        self._schedule_save()

    def on_folder_changed(self, path: str):
        """Handle folder content changes."""
        # Don't refresh while pipeline is running to avoid disrupting progress
        if not self.is_running:
            self.refresh_file_list()

    def on_file_selection_changed(self, filenames: list[str]):
        """Handle file selection change - update inputs to reflect selected files' config."""
        self._update_start_button_label()

        # Update slider duration: selected files or all files
        duration, ref_name = self._get_longest_duration(filenames if filenames else None)
        if ref_name and duration > 0:
            self.time_range_slider.set_duration(duration, ref_name)

        if not filenames:
            self._sync_crop_input()
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
            # Load first range into slider, populate chips
            r = config.time_ranges[0]
            self.time_range_slider.set_time_range(r[0] or "", r[1] or "")
            self._active_range_index = 0
            self.time_range_chips.set_ranges(config.time_ranges)
            self.time_range_chips.set_active(0)
        else:
            # Reset to full duration, clear chips
            self.time_range_slider.set_time_range("", "")
            self._active_range_index = -1
            self.time_range_chips.set_ranges([])
            self.time_range_chips.set_active(-1)

    def _update_start_button_label(self):
        """Update start button text to reflect selection count."""
        if self.is_running:
            return
        selected = self.file_table.get_selected_filenames()
        if selected:
            self.start_button.setText(f"Start Processing ({len(selected)})")
        else:
            self.start_button.setText("Start Processing")

    def _get_common_crop(self) -> tuple | None:
        """Return the common crop if all files share the same value, else None."""
        all_filenames = self.file_table.get_all_filenames()
        if not all_filenames:
            return None
        crops = set()
        for filename in all_filenames:
            fc = self.file_config_store.get(filename)
            if fc and fc.has_custom_crop():
                crops.add(fc.get_crop_tuple())
            else:
                crops.add(None)
        if len(crops) == 1:
            return crops.pop()
        return None

    def _sync_crop_input(self):
        """Update global crop input to reflect common crop or clear if mixed."""
        common = self._get_common_crop()
        if common:
            self.crop_input.setText(f"{common[0]}, {common[1]}, {common[2]}, {common[3]}")
        else:
            self.crop_input.clear()

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

    def _on_brightness_changed(self, value: int):
        """Handle brightness spinbox change - apply to selected files or all if none selected."""
        target_files = self._get_target_files()
        self._apply_brightness_to_files(value, target_files)

    def _on_time_range_changed(self, start: int, end: int, handle: str):
        """Handle time range slider commit - update the active range in file configs."""
        if self._active_range_index < 0:
            # No active range being edited - slider is in "preview" mode before Add
            return

        start_str, end_str = self.time_range_slider.get_time_strings()
        target_files = self._get_target_files()
        for filename in target_files:
            config = self.file_config_store.get_or_create(filename)
            if self._active_range_index < len(config.time_ranges):
                config.set_time_range(
                    self._active_range_index,
                    start_str or None,
                    end_str or None,
                )
                self.file_table.update_config_indicator(filename)
        self._refresh_time_range_chips()
        self._schedule_save()

    def _on_add_range_clicked(self):
        """Capture current slider values as a new time range."""
        start_str, end_str = self.time_range_slider.get_time_strings()

        target_files = self._get_target_files()
        for filename in target_files:
            config = self.file_config_store.get_or_create(filename)
            config.add_time_range(start_str or None, end_str or None)
            self.file_table.update_config_indicator(filename)

        self._refresh_time_range_chips()
        # Select the newly added range
        first_file = target_files[0] if target_files else None
        if first_file:
            fc = self.file_config_store.get(first_file)
            if fc:
                # Find index of the range we just added (sorted list)
                new_range = (start_str or None, end_str or None)
                for i, r in enumerate(fc.time_ranges):
                    if r == new_range:
                        self._active_range_index = i
                        self.time_range_chips.set_active(i)
                        break
        self._schedule_save()

    def _on_range_chip_selected(self, index: int):
        """Load the selected range into the slider."""
        self._active_range_index = index
        target_files = self._get_target_files()
        first_file = target_files[0] if target_files else None
        if first_file:
            fc = self.file_config_store.get(first_file)
            if fc and index < len(fc.time_ranges):
                r = fc.time_ranges[index]
                self.time_range_slider.blockSignals(True)
                self.time_range_slider.set_time_range(r[0] or "", r[1] or "")
                self.time_range_slider.blockSignals(False)

    def _on_range_chip_removed(self, index: int):
        """Remove a time range from file configs."""
        target_files = self._get_target_files()
        for filename in target_files:
            config = self.file_config_store.get_or_create(filename)
            config.remove_time_range(index)
            self.file_table.update_config_indicator(filename)

        # Adjust active index
        first_file = target_files[0] if target_files else None
        fc = self.file_config_store.get(first_file) if first_file else None
        remaining = len(fc.time_ranges) if fc else 0
        if remaining == 0:
            self._active_range_index = -1
            self.time_range_slider.set_time_range("", "")
        elif index >= remaining:
            self._active_range_index = remaining - 1
        else:
            self._active_range_index = index

        self._refresh_time_range_chips()
        # Load the new active range into slider
        if self._active_range_index >= 0 and fc and self._active_range_index < len(fc.time_ranges):
            r = fc.time_ranges[self._active_range_index]
            self.time_range_slider.blockSignals(True)
            self.time_range_slider.set_time_range(r[0] or "", r[1] or "")
            self.time_range_slider.blockSignals(False)
        self._schedule_save()

    def _refresh_time_range_chips(self):
        """Rebuild chip display from the first target file's config."""
        target_files = self._get_target_files()
        first_file = target_files[0] if target_files else None
        if first_file:
            fc = self.file_config_store.get(first_file)
            ranges = fc.time_ranges if fc else []
        else:
            ranges = []
        self.time_range_chips.set_ranges(ranges)
        self.time_range_chips.set_active(self._active_range_index)

    def _on_auto_time_range_clicked(self):
        """Run audio fingerprint analysis to auto-detect intros/outros."""
        if not self.project_path:
            QMessageBox.warning(self, "Error", "Please select a project directory first")
            return

        filenames = self.file_table.get_all_filenames()
        if len(filenames) < 2:
            QMessageBox.warning(
                self, "Not Enough Files",
                "Audio fingerprint analysis requires at least 2 video files\n"
                "to detect repeating segments (intros/outros)."
            )
            return

        # Create progress dialog
        self._audio_progress_dialog = QProgressDialog(
            "Preparing audio analysis...", "Cancel", 0, len(filenames), self
        )
        self._audio_progress_dialog.setWindowTitle("Audio Analysis")
        self._audio_progress_dialog.setMinimumWidth(400)
        self._audio_progress_dialog.setModal(True)
        self._audio_progress_dialog.setAutoClose(False)
        self._audio_progress_dialog.setAutoReset(False)
        self._audio_progress_dialog.canceled.connect(self._on_audio_analysis_cancelled)

        # Create and start worker
        self._audio_analysis_worker = AudioAnalysisWorker(
            self.project_path, filenames,
            min_segment_sec=int(self.autodetect_settings['min_segment_length']),
            merge_repeating_silences=(
                self.autodetect_settings.get('merge_repeating_silences', 'false').lower() == 'true'
            ),
        )
        self._audio_analysis_worker.phase_changed.connect(self._on_audio_phase_changed)
        self._audio_analysis_worker.file_progress.connect(self._on_audio_file_progress)
        self._audio_analysis_worker.analysis_progress.connect(self._on_audio_analysis_msg)
        self._audio_analysis_worker.error.connect(self._on_audio_analysis_error)
        self._audio_analysis_worker.finished.connect(self._on_audio_analysis_finished)

        self.auto_time_range_btn.setEnabled(False)
        self._audio_analysis_worker.start()

    def _on_audio_phase_changed(self, phase: str):
        """Update progress dialog label for phase change."""
        if self._audio_progress_dialog:
            self._audio_progress_dialog.setLabelText(phase)
            if phase != "Fingerprinting":
                self._audio_progress_dialog.setMaximum(0)  # Indeterminate spinner

    def _on_audio_file_progress(self, filename: str, current: int, total: int):
        """Update progress dialog for file processing."""
        if self._audio_progress_dialog:
            self._audio_progress_dialog.setMaximum(total)
            self._audio_progress_dialog.setValue(current)
            self._audio_progress_dialog.setLabelText(
                f"Fingerprinting: {filename} ({current}/{total})"
            )

    def _on_audio_analysis_msg(self, msg: str):
        """Update progress dialog with analysis status."""
        if self._audio_progress_dialog:
            self._audio_progress_dialog.setMaximum(0)  # Indeterminate
            self._audio_progress_dialog.setLabelText(f"Analyzing: {msg.strip()}")

    def _on_audio_analysis_error(self, msg: str):
        """Handle audio analysis error."""
        self._cleanup_audio_analysis()
        QMessageBox.critical(self, "Audio Analysis Error", msg)

    def _on_audio_analysis_cancelled(self):
        """Handle user cancellation of audio analysis."""
        if self._audio_analysis_worker:
            self._audio_analysis_worker.cancel()
        self._cleanup_audio_analysis()

    def _on_audio_analysis_finished(self, results: dict):
        """Apply auto-detected time ranges to file configs."""
        self._cleanup_audio_analysis()

        if not results:
            QMessageBox.information(
                self, "Audio Analysis",
                "No repeating segments (intros/outros) were detected."
            )
            return

        applied = 0
        for filename, ranges in results.items():
            if ranges:
                config = self.file_config_store.get_or_create(filename)
                config.time_ranges = ranges
                self.file_table.update_config_indicator(filename)
                applied += 1

        self._schedule_save()

    def _cleanup_audio_analysis(self):
        """Clean up audio analysis worker and dialog."""
        if self._audio_analysis_worker:
            self._audio_analysis_worker.cleanup()
            self._audio_analysis_worker = None
        if self._audio_progress_dialog:
            self._audio_progress_dialog.close()
            self._audio_progress_dialog = None
        if self.auto_time_range_btn:
            self.auto_time_range_btn.setEnabled(True)

    def _on_detect_subtitle_clicked(self):
        """Run subtitle detection to find frames with hardcoded subtitles."""
        if not self.project_path:
            QMessageBox.warning(self, "Error", "Please select a project directory first")
            return

        # Build list of (filename, full_path, duration) for files with known duration
        video_files = []
        for filename in self.file_table.get_all_filenames():
            config = self.file_config_store.get(filename)
            if config and config.duration_seconds > 0:
                full_path = str(Path(self.project_path) / filename)
                video_files.append((filename, full_path, config.duration_seconds))

        if not video_files:
            QMessageBox.warning(self, "Error", "No video files with known duration. Wait for scanning to finish.")
            return

        # Create progress dialog
        self._subtitle_progress_dialog = QProgressDialog(
            "Detecting subtitles...", "Cancel", 0, len(video_files), self
        )
        self._subtitle_progress_dialog.setWindowTitle("Subtitle Detection")
        self._subtitle_progress_dialog.setMinimumWidth(400)
        self._subtitle_progress_dialog.setModal(True)
        self._subtitle_progress_dialog.setAutoClose(False)
        self._subtitle_progress_dialog.setAutoReset(False)
        self._subtitle_progress_dialog.canceled.connect(self._on_subtitle_detection_cancelled)

        # Create and start worker
        self._subtitle_detection_worker = SubtitleDetectionWorker(
            video_files, automation_settings=self.automation_settings
        )
        self._subtitle_detection_worker.file_detected.connect(self._on_subtitle_file_detected)
        self._subtitle_detection_worker.progress.connect(self._on_subtitle_detection_progress)
        self._subtitle_detection_worker.error.connect(self._on_subtitle_detection_error)
        self._subtitle_detection_worker.finished.connect(self._on_subtitle_detection_finished)

        self.detect_subtitle_btn.setEnabled(False)
        self._subtitle_detection_worker.start()

    def _on_subtitle_file_detected(self, filename: str, slider_position: int,
                                    crop_x: int, crop_y: int, crop_w: int, crop_h: int):
        """Store detected subtitle position and auto-crop for a file."""
        config = self.file_config_store.get_or_create(filename)
        config.subtitle_position = slider_position
        if crop_w > 0 and crop_h > 0:
            config.set_crop(crop_x, crop_y, crop_w, crop_h)
            self.file_table.update_config_indicator(filename)

    def _on_subtitle_detection_progress(self, resolved: int, total: int):
        """Update progress dialog."""
        if not self._subtitle_progress_dialog:
            return
        self._subtitle_progress_dialog.setLabelText(
            f"Detecting subtitles... ({resolved}/{total} files)"
        )
        self._subtitle_progress_dialog.setMaximum(total)
        # setValue can trigger canceled signal (auto-close when value == max),
        # which runs cleanup and sets dialog to None, so call it last
        self._subtitle_progress_dialog.setValue(resolved)

    def _on_subtitle_detection_error(self, msg: str):
        """Handle subtitle detection error."""
        self._cleanup_subtitle_detection()
        QMessageBox.critical(self, "Subtitle Detection Error", msg)

    def _on_subtitle_detection_cancelled(self):
        """Handle user cancellation."""
        if self._subtitle_detection_worker:
            self._subtitle_detection_worker.cancel()
        self._cleanup_subtitle_detection()

    def _on_subtitle_detection_finished(self):
        """Handle subtitle detection completion."""
        self._schedule_save()
        self._cleanup_subtitle_detection()

    def _cleanup_subtitle_detection(self):
        """Clean up subtitle detection worker and dialog."""
        if self._subtitle_detection_worker:
            self._subtitle_detection_worker.cleanup()
            self._subtitle_detection_worker = None
        if self._subtitle_progress_dialog:
            self._subtitle_progress_dialog.close()
            self._subtitle_progress_dialog = None
        if self.detect_subtitle_btn:
            self.detect_subtitle_btn.setEnabled(True)

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
                    time_ranges=list(config.time_ranges),
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

        # Build subtitle positions dict and resolve initial timeline position
        subtitle_positions = self._get_subtitle_positions()
        timeline_position = self._resolve_timeline_position(video_files)

        # Build per-file crops dict from file config store
        per_file_crops = {}
        for f in video_files:
            fc = self.file_config_store.get(f.name)
            if fc and fc.has_custom_crop():
                per_file_crops[f.name] = fc.get_crop_tuple()

        dialog = CropSelectorDialog(
            [str(f) for f in video_files], existing_crop, timeline_position,
            labels_enabled=self.labels_checkbox.isChecked(),
            existing_masks=self.label_mask_crops if self.label_mask_crops else None,
            subtitle_positions=subtitle_positions,
            durations=self._get_durations_dict(),
            existing_crops=per_file_crops,
            parent=self
        )
        if dialog.exec():
            self.last_selected_episode = dialog.get_selected_episode()
            self.last_timeline_position = dialog.get_timeline_position()
            self.label_mask_crops = dialog.get_label_masks()

            x, y, w, h = dialog.frame_label.get_crop_coordinates()
            if w > 0 and h > 0:
                current_filename = dialog.get_current_filename()
                self._handle_crop_apply(x, y, w, h, current_filename)

            self._schedule_save()

    def _handle_crop_apply(self, x: int, y: int, w: int, h: int, current_filename: str):
        """Apply crop with conflict detection when target files have different crops."""
        new_crop = (x, y, w, h)
        target_files = self._get_target_files()

        files_with_different_crop = [
            f for f in target_files
            if f != current_filename
            and (fc := self.file_config_store.get(f))
            and fc.has_custom_crop()
            and fc.get_crop_tuple() != new_crop
        ]

        if files_with_different_crop:
            msg = QMessageBox(self)
            msg.setWindowTitle("Apply Crop")
            msg.setText(f"{len(files_with_different_crop)} file(s) have different crop values.")
            apply_all = msg.addButton("Apply to All", QMessageBox.ButtonRole.AcceptRole)
            apply_one = msg.addButton("This File Only", QMessageBox.ButtonRole.ActionRole)
            msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == apply_all:
                self._apply_crop_to_files(x, y, w, h, target_files)
            elif clicked == apply_one:
                self._apply_crop_to_files(x, y, w, h, [current_filename])
            else:
                return
        else:
            self._apply_crop_to_files(x, y, w, h, target_files)

        self._sync_crop_input()

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

        # Build per-file crops dict from file config store
        per_file_crops = {}
        for f in video_files:
            fc = self.file_config_store.get(f.name)
            if fc and fc.has_custom_crop():
                per_file_crops[f.name] = fc.get_crop_tuple()

        # Initial crop: from first file's config (or None)
        first_name = video_files[0].name if video_files else None
        crop_region = per_file_crops.get(first_name) if first_name else None

        # Build subtitle positions dict and resolve initial timeline position
        subtitle_positions = self._get_subtitle_positions()
        timeline_position = self._resolve_timeline_position(video_files)

        dialog = BrightnessTesterDialog(
            [str(f) for f in video_files],
            self.last_selected_episode,
            timeline_position,
            self.brightness_spin.value(),
            crop_region,
            subtitle_positions=subtitle_positions,
            durations=self._get_durations_dict(),
            existing_crops=per_file_crops,
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

    def _get_durations_dict(self) -> dict[str, float]:
        """Build dict of filename -> duration_seconds for all files with known durations."""
        durations = {}
        for filename in self.file_config_store.get_all_filenames():
            config = self.file_config_store.get(filename)
            if config and config.duration_seconds > 0:
                durations[filename] = config.duration_seconds
        return durations

    def _get_subtitle_positions(self) -> dict[str, int]:
        """Build dict of filename -> subtitle_position for all files with detected positions."""
        positions = {}
        for filename in self.file_config_store.get_all_filenames():
            config = self.file_config_store.get(filename)
            if config and config.subtitle_position is not None:
                positions[filename] = config.subtitle_position
        return positions

    def _resolve_timeline_position(self, video_files: list[Path]) -> int:
        """Resolve timeline position: use first file's detected subtitle position, else fallback."""
        if video_files:
            first_name = video_files[0].name
            config = self.file_config_store.get(first_name)
            if config and config.subtitle_position is not None:
                return config.subtitle_position
        return self.last_timeline_position

    def _open_settings(self):
        """Open the unified settings dialog."""
        dialog = SettingsDialog(
            self.videocr_settings, self.label_settings,
            self.autodetect_settings, self.automation_settings,
            self
        )
        dialog.settings_changed.connect(self._on_settings_changed)
        dialog.exec()

    def _on_settings_changed(self, ocr: dict, label: dict, autodetect: dict, automation: dict):
        """Handle unified settings changes."""
        self.videocr_settings.update(ocr)
        self.label_settings.update(label)
        self.autodetect_settings.update(autodetect)
        self.automation_settings.update(automation)
        self._update_settings_summary()
        self._schedule_save()

    def _update_settings_summary(self):
        """Update the muted settings summary label in the folder row."""
        s = self.videocr_settings
        text = f"{s['ocr_lang']} | {s['conf_threshold']} | {s['sim_threshold']} | {s['similar_image']}"
        self.settings_summary_label.setText(text)

    def _is_labels_only(self) -> bool:
        """Labels-only mode: Labels checked, Dialogue unchecked."""
        return self.labels_checkbox.isChecked() and not self.dialogue_checkbox.isChecked()

    def _on_processing_checkboxes_changed(self):
        """Handle Dialogue/Labels checkbox changes."""
        both_off = not self.dialogue_checkbox.isChecked() and not self.labels_checkbox.isChecked()
        self.processing_warning.setVisible(both_off)
        if not self.is_running:
            self.start_button.setEnabled(not both_off)

        # Update crop enabled state
        labels_only = self._is_labels_only()
        self.crop_input.setEnabled(not labels_only)
        self.crop_select_btn.setEnabled(not labels_only)


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
        # Undo Qt's auto-toggle — we manage checked state explicitly
        self.start_button.setChecked(not self.start_button.isChecked())

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
        time_start, time_end = self.time_range_slider.get_time_strings()
        if time_start or time_end:
            config.time_ranges = [(time_start, time_end)]
        config.ocr_parallel = self.parallel_slider.value()

        # Apply videocr settings
        config.ocr_lang = self.videocr_settings['ocr_lang']
        config.conf_threshold = int(self.videocr_settings['conf_threshold'])
        config.sim_threshold = int(self.videocr_settings['sim_threshold'])
        config.similar_image = float(self.videocr_settings['similar_image'])

        # Apply label settings
        config.labels_enabled = self.labels_checkbox.isChecked()
        config.labels_only = self._is_labels_only()
        config.label_min_duration = float(self.label_settings.get('label_min_duration', '0.5'))
        config.label_max_duration = float(self.label_settings.get('label_max_duration', '5.0'))
        config.label_conf_threshold = int(self.label_settings.get('label_conf_threshold', '95'))
        config.label_conf_threshold_min = int(self.label_settings.get('label_conf_threshold_min', '80'))
        config.label_mask_crops = list(self.label_mask_crops)

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

        # Check for Done files that would be overwritten
        selected_files = self.file_table.get_selected_filenames()
        project_path = Path(self.project_path)
        file_statuses = detect_file_statuses(project_path)
        targets = selected_files if selected_files else list(file_statuses.keys())
        done_files = [f for f in targets if file_statuses.get(f) == FileStatus.DONE]

        if done_files:
            answer = QMessageBox.question(
                self, "Overwrite Existing Files",
                f"{len(done_files)} file(s) already have subtitle files that will be overwritten. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            chi_dir = project_path / "chi"
            for fname in done_files:
                ass_file = chi_dir / (Path(fname).stem + ".ass")
                if ass_file.exists():
                    ass_file.unlink()

        # Create and start pipeline with file config store
        self.pipeline = Pipeline(config, self.file_config_store,
                                 selected_files=selected_files or None)
        self.pipeline.error_occurred.connect(self.on_pipeline_error)
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

        # Subtitle preview signal
        self.pipeline.ocr_subtitle_detected.connect(self._on_subtitle_detected)

        # Clear previous logs and subtitle data, enable logs button
        self.log_store.clear()
        if self._subtitle_preview:
            self._subtitle_preview.clear_all()
        self.logs_btn.setVisible(True)
        self.overall_progress.setVisible(True)

        # Update UI
        self.is_running = True
        self.start_button.setText("Stop")
        self.start_button.setChecked(False)
        self.start_button.setStyleSheet("background-color: #f38ba8; color: #000;")
        self.update_window_title()
        self.disable_ui()

        # Reset progress widgets
        self.overall_progress.setValue(0)
        self.timing_label.setText("Elapsed: 0s  •  Remaining: calculating...")

        # Start pipeline
        self.pipeline.start()

    def on_pipeline_error(self, error_msg: str):
        """Handle pipeline errors."""
        pass

    def on_pipeline_finished(self, success: bool):
        """Handle pipeline completion."""
        self.is_running = False
        self._update_start_button_label()
        self.start_button.setChecked(True)
        self.start_button.setStyleSheet("")  # Reset to theme default
        self.update_window_title()
        self.enable_ui()

        # Reset progress bar
        self.overall_progress.setValue(0)

        # Refresh file list to update statuses from filesystem
        self.current_video_files = set()  # Force refresh
        self.refresh_file_list()

        if success:
            # Show completion stats in timing label
            total_time, avg_time = self.pipeline.get_ocr_timing()
            if total_time > 0:
                total_str = self._format_duration(total_time)
                avg_str = self._format_duration(avg_time)
                self.timing_label.setText(f"Finished in {total_str}  •  Average {avg_str}/file")
                msg = f"Finished in {total_str} | Avg: {avg_str}/file"
            else:
                self.timing_label.setText("Elapsed: --")
                msg = "All phases completed"
            self._send_notification("OCR Complete", msg)
        else:
            self.timing_label.setText("Elapsed: --")
            self._send_notification("OCR Failed", "Pipeline encountered an error", "critical")

    def on_pipeline_stopped(self):
        """Handle user-initiated pipeline stop."""
        self.is_running = False
        self._update_start_button_label()
        self.start_button.setChecked(True)
        self.start_button.setStyleSheet("")  # Reset to theme default
        self.update_window_title()
        self.enable_ui()

        # Reset progress widgets
        self.overall_progress.setValue(0)
        self.timing_label.setText("Elapsed: --")

        # Refresh file list to update statuses from filesystem
        self.current_video_files = set()  # Force refresh
        self.refresh_file_list()

    def _on_ocr_log_output(self, filename: str, text: str):
        """Accumulate per-file OCR output in the log store."""
        self.log_store.append(filename, text)

    def _on_pipeline_log_output(self, text: str):
        """Accumulate pipeline status output in the log store."""
        self.log_store.append("Pipeline", text)

    def _on_logs_clicked(self):
        """Open or raise the logs dialog."""
        if self._logs_dialog is None or not self._logs_dialog.isVisible():
            self._logs_dialog = LogsDialog(self.log_store, self)
        self._logs_dialog.show()
        self._logs_dialog.raise_()
        self._logs_dialog.activateWindow()

    def _on_file_double_clicked(self, filename: str):
        """Open subtitle preview dialog for the double-clicked file."""
        if self._subtitle_preview is None or not self._subtitle_preview.isVisible():
            self._subtitle_preview = SubtitlePreviewDialog(self)
        self._subtitle_preview.show_file(filename)
        self._subtitle_preview.show()
        self._subtitle_preview.raise_()
        self._subtitle_preview.activateWindow()

    def _on_file_details_requested(self, filename: str):
        """Open read-only details dialog showing resolved settings for a file."""
        fc = self.file_config_store.get(filename)

        # Resolve per-file values mirroring OCRWorker._build_ocr_kwargs() logic
        brightness_override = fc is not None and fc.has_custom_brightness()
        brightness = fc.brightness if brightness_override else self.brightness_spin.value()

        crop_override = fc is not None and fc.has_custom_crop()
        if crop_override:
            crop = fc.get_crop_tuple()
        else:
            crop_text = self.crop_input.text()
            crop = None
            if crop_text:
                try:
                    parts = [int(x.strip()) for x in crop_text.split(',')]
                    if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
                        crop = tuple(parts)
                except ValueError:
                    pass

        time_ranges_override = fc is not None and fc.has_custom_time_range()
        if time_ranges_override:
            time_ranges = list(fc.time_ranges)
        else:
            global_start, global_end = self.time_range_slider.get_time_strings()
            if global_start or global_end:
                time_ranges = [(global_start, global_end)]
            else:
                time_ranges = []

        # Resolution/duration from file config metadata
        res_w = fc.resolution_width if fc else 0
        res_h = fc.resolution_height if fc else 0
        duration = fc.duration_seconds if fc else 0.0

        data = FileDetailsData(
            filename=filename,
            resolution_width=res_w,
            resolution_height=res_h,
            duration_seconds=duration,
            brightness=brightness,
            brightness_is_override=brightness_override,
            crop=crop,
            crop_is_override=crop_override,
            time_ranges=time_ranges,
            time_ranges_is_override=time_ranges_override,
            ocr_lang=self.videocr_settings.get('ocr_lang', 'ch'),
            conf_threshold=int(self.videocr_settings.get('conf_threshold', '95')),
            sim_threshold=int(self.videocr_settings.get('sim_threshold', '82')),
            similar_image=float(self.videocr_settings.get('similar_image', '0.3')),
            frames_to_skip=0,
            use_gpu=True,
            ocr_parallel=self.parallel_slider.value(),
            dialogue_enabled=self.dialogue_checkbox.isChecked(),
            labels_enabled=self.labels_checkbox.isChecked(),
            labels_only=self._is_labels_only(),
            label_min_duration=float(self.label_settings.get('label_min_duration', '0.5')),
            label_max_duration=float(self.label_settings.get('label_max_duration', '5.0')),
            label_conf_threshold=int(self.label_settings.get('label_conf_threshold', '95')),
            label_conf_threshold_min=int(self.label_settings.get('label_conf_threshold_min', '80')),
            mask_crops_count=len(self.label_mask_crops),
        )

        dialog = FileDetailsDialog(data, self)
        dialog.exec()

    def _on_subtitle_detected(self, filename: str, start: float, end: float, text: str):
        """Forward subtitle detection to preview dialog."""
        if self._subtitle_preview is not None:
            self._subtitle_preview.on_subtitle_detected(filename, start, end, text)

    def disable_ui(self):
        """Disable UI during processing."""
        self.folder_btn.setEnabled(False)
        self.crop_input.setEnabled(False)
        self.crop_select_btn.setEnabled(False)
        self.brightness_spin.setEnabled(False)
        self.brightness_test_btn.setEnabled(False)
        self.time_range_slider.setEnabled(False)
        self.time_range_chips.setEnabled(False)
        self.parallel_slider.setEnabled(False)
        self.dialogue_checkbox.setEnabled(False)
        self.labels_checkbox.setEnabled(False)
        self.settings_btn.setEnabled(False)
        if self.auto_time_range_btn:
            self.auto_time_range_btn.setEnabled(False)
        if self.detect_subtitle_btn:
            self.detect_subtitle_btn.setEnabled(False)
        # Note: file_table is not disabled to allow scrolling during processing

    def enable_ui(self):
        """Re-enable UI after processing."""
        self.folder_btn.setEnabled(True)
        self.brightness_spin.setEnabled(True)
        self.brightness_test_btn.setEnabled(True)
        self.time_range_slider.setEnabled(True)
        self.time_range_chips.setEnabled(True)
        self.parallel_slider.setEnabled(True)
        self.dialogue_checkbox.setEnabled(True)
        self.labels_checkbox.setEnabled(True)
        self.settings_btn.setEnabled(True)
        if self.auto_time_range_btn:
            self.auto_time_range_btn.setEnabled(True)
        if self.detect_subtitle_btn:
            self.detect_subtitle_btn.setEnabled(True)
        # Respect labels_only for crop controls
        labels_only = self._is_labels_only()
        self.crop_input.setEnabled(not labels_only)
        self.crop_select_btn.setEnabled(not labels_only)

    def _restore_geometry(self):
        """Restore window size and position from QSettings."""
        settings = QSettings("OCRManager", "OCRTool")
        geometry = settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def _save_geometry(self):
        """Save window size and position to QSettings."""
        settings = QSettings("OCRManager", "OCRTool")
        settings.setValue("window/geometry", self.saveGeometry())

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

        from videocr.pyav_adapter import PYAV_AVAILABLE, PYAV_IMPORT_ERROR
        if not PYAV_AVAILABLE:
            QMessageBox.critical(
                self, "Video backend degraded",
                "PyAV is not available, so video will be decoded by a fallback "
                "backend that is not bit-exact and estimates timestamps.\n\n"
                "Fix it with:\n"
                "    .venv/bin/pip install -U --only-binary=:all: av\n\n"
                f"Import error: {PYAV_IMPORT_ERROR}"
            )

    def closeEvent(self, event):
        """Handle window close."""
        self._save_geometry()
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
                if self._audio_analysis_worker:
                    self._audio_analysis_worker.cancel()
                    self._audio_analysis_worker.cleanup()
                if self._subtitle_detection_worker:
                    self._subtitle_detection_worker.cancel()
                    self._subtitle_detection_worker.cleanup()
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
            if self._audio_analysis_worker:
                self._audio_analysis_worker.cancel()
                self._audio_analysis_worker.cleanup()
            if self._subtitle_detection_worker:
                self._subtitle_detection_worker.cancel()
                self._subtitle_detection_worker.cleanup()
            # Stop async saver thread
            if self._async_saver:
                self._async_saver.stop()
            event.accept()


def main():
    """Application entry point."""
    app = QApplication(sys.argv)

    # Apply dark theme
    apply_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
