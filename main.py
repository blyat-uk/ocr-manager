#!/usr/bin/env python3
"""Main application entry point."""
import sys
import shutil
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QGroupBox, QLineEdit, QPushButton,
                              QSpinBox, QPlainTextEdit, QLabel, QFileDialog,
                              QMessageBox, QSplitter, QCheckBox, QMenuBar)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut

from core.config import (Config, GlobalConfig, load_project_config, save_project_config,
                         load_global_config, save_global_config, validate_config,
                         load_header_from_translate)
from core.pipeline import Pipeline
from widgets.terminal_output import TerminalOutputWidget
from widgets.crop_selector import CropSelectorDialog
from widgets.brightness_tester import BrightnessTesterDialog


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Translator")
        self.resize(1200, 900)

        # State
        self.project_path = None
        self.config = None
        self.global_config = load_global_config()
        self.pipeline = None
        self.is_running = False
        self.last_selected_episode = 0
        self.last_timeline_position = 5000  # Default 50% (on 0-10000 scale)

        # UI components
        self.crop_input = None
        self.brightness_spin = None
        self.time_start_input = None
        self.time_end_input = None
        self.ocr_parallel_spin = None
        self.remove_credits_checkbox = None
        self.header_text = None
        self.save_default_btn = None
        self.load_default_btn = None
        self.terminal = None
        self.start_button = None
        self.auto_scroll_check = None
        self.open_dir_button = None

        self.init_ui()
        self.check_dependencies()
        self.restore_last_project()

    def update_window_title(self):
        """Update window title with project name and running status."""
        base = "Translator"
        if self.project_path:
            project_name = Path(self.project_path).name
            base = f"Translator - {project_name}"

        if self.is_running:
            self.setWindowTitle(f"[Running] {base}")
        else:
            self.setWindowTitle(base)

    def init_ui(self):
        """Setup all UI components."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Splitter for config and terminal
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Configuration section
        config_widget = self.create_config_section()
        splitter.addWidget(config_widget)

        # Terminal section
        terminal_widget = self.create_terminal_section()
        splitter.addWidget(terminal_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        # Keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

    def create_config_section(self) -> QWidget:
        """Create configuration section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Project directory
        project_layout = QHBoxLayout()
        project_layout.addWidget(QLabel("Project Directory:"))
        self.project_label = QLabel("No project selected")
        project_layout.addWidget(self.project_label)
        project_layout.addStretch()

        # Open directory button (hidden initially)
        self.open_dir_button = QPushButton("Open Project Directory")
        self.open_dir_button.clicked.connect(self.open_project_directory)
        self.open_dir_button.setVisible(False)
        project_layout.addWidget(self.open_dir_button)

        select_dir_btn = QPushButton("Select Directory")
        select_dir_btn.clicked.connect(self.select_project_directory)
        project_layout.addWidget(select_dir_btn)
        layout.addLayout(project_layout)

        # OCR Parameters
        ocr_group = QGroupBox("OCR Parameters")
        ocr_layout = QVBoxLayout()

        # Crop region
        crop_layout = QHBoxLayout()
        crop_layout.addWidget(QLabel("Crop Region:"))
        self.crop_input = QLineEdit()
        self.crop_input.setPlaceholderText("x, y, width, height")
        self.crop_input.textChanged.connect(self.auto_save_config)
        crop_layout.addWidget(self.crop_input)
        crop_select_btn = QPushButton("Select")
        crop_select_btn.clicked.connect(self.on_crop_select_clicked)
        crop_layout.addWidget(crop_select_btn)
        ocr_layout.addLayout(crop_layout)

        # Brightness
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("Brightness:"))
        self.brightness_spin = QSpinBox()
        self.brightness_spin.setRange(0, 255)
        self.brightness_spin.setValue(230)
        self.brightness_spin.valueChanged.connect(self.auto_save_config)
        brightness_layout.addWidget(self.brightness_spin)
        brightness_test_btn = QPushButton("Test")
        brightness_test_btn.clicked.connect(self.on_brightness_test_clicked)
        brightness_layout.addWidget(brightness_test_btn)
        brightness_layout.addStretch()
        ocr_layout.addLayout(brightness_layout)

        # Time start/end
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time Start:"))
        self.time_start_input = QLineEdit()
        self.time_start_input.setPlaceholderText("Optional (MM:SS or HH:MM:SS)")
        self.time_start_input.textChanged.connect(self.auto_save_config)
        time_layout.addWidget(self.time_start_input)
        time_layout.addWidget(QLabel("Time End:"))
        self.time_end_input = QLineEdit()
        self.time_end_input.setPlaceholderText("Optional (MM:SS or HH:MM:SS)")
        self.time_end_input.textChanged.connect(self.auto_save_config)
        time_layout.addWidget(self.time_end_input)
        ocr_layout.addLayout(time_layout)

        # Parallel processing
        parallel_layout = QHBoxLayout()
        parallel_layout.addWidget(QLabel("Parallel:"))
        self.ocr_parallel_spin = QSpinBox()
        self.ocr_parallel_spin.setRange(1, 32)
        self.ocr_parallel_spin.setValue(4)
        self.ocr_parallel_spin.valueChanged.connect(self.auto_save_config)
        parallel_layout.addWidget(self.ocr_parallel_spin)
        parallel_layout.addStretch()
        ocr_layout.addLayout(parallel_layout)

        ocr_group.setLayout(ocr_layout)
        layout.addWidget(ocr_group)

        # Cleanup Parameters
        cleanup_group = QGroupBox("Cleanup Parameters")
        cleanup_layout = QHBoxLayout()
        self.remove_credits_checkbox = QCheckBox("Remove Credits")
        self.remove_credits_checkbox.setChecked(True)
        self.remove_credits_checkbox.stateChanged.connect(self.auto_save_config)
        cleanup_layout.addWidget(self.remove_credits_checkbox)
        cleanup_layout.addStretch()
        cleanup_group.setLayout(cleanup_layout)
        layout.addWidget(cleanup_group)

        # Header Template
        header_group = QGroupBox("Header Template")
        header_layout = QVBoxLayout()
        self.header_text = QPlainTextEdit()
        self.header_text.setPlaceholderText("Enter ASS header template...")
        self.header_text.setMaximumHeight(150)
        self.header_text.textChanged.connect(self.auto_save_config)
        header_layout.addWidget(self.header_text)

        # Header default buttons
        header_btn_layout = QHBoxLayout()
        header_btn_layout.addStretch()
        self.save_default_btn = QPushButton("Save Default")
        self.save_default_btn.setToolTip("Save current header as the global default")
        self.save_default_btn.clicked.connect(self.on_save_header_default)
        header_btn_layout.addWidget(self.save_default_btn)
        self.load_default_btn = QPushButton("Load Default")
        self.load_default_btn.setToolTip("Load the global default header template")
        self.load_default_btn.clicked.connect(self.on_load_header_default)
        header_btn_layout.addWidget(self.load_default_btn)
        header_layout.addLayout(header_btn_layout)

        header_group.setLayout(header_layout)
        layout.addWidget(header_group)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.start_button = QPushButton("Start Processing")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_button.clicked.connect(self.on_start_processing_clicked)
        button_layout.addWidget(self.start_button)

        layout.addLayout(button_layout)

        return widget

    def create_terminal_section(self) -> QWidget:
        """Create terminal output section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Terminal Output:"))

        # Terminal widget
        self.terminal = TerminalOutputWidget()
        layout.addWidget(self.terminal)

        # Control buttons
        control_layout = QHBoxLayout()

        self.auto_scroll_check = QCheckBox("Auto-scroll")
        self.auto_scroll_check.setChecked(True)
        self.auto_scroll_check.stateChanged.connect(
            lambda state: self.terminal.set_auto_scroll(state == Qt.CheckState.Checked)
        )
        control_layout.addWidget(self.auto_scroll_check)

        control_layout.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.terminal.clear)
        control_layout.addWidget(clear_btn)

        copy_btn = QPushButton("Copy Output")
        copy_btn.clicked.connect(self.terminal.copy_to_clipboard)
        control_layout.addWidget(copy_btn)

        layout.addLayout(control_layout)

        return widget

    def select_project_directory(self):
        """Select project directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Project Directory"
        )

        if directory:
            self.set_project_directory(directory)

    def set_project_directory(self, directory: str):
        """Set project directory and save to global config."""
        self.project_path = directory
        self.project_label.setText(directory)
        self.open_dir_button.setVisible(True)
        self.update_window_title()
        self.load_project_config(directory)

        # Save to global config
        self.global_config.last_project_directory = directory
        save_global_config(self.global_config)

    def restore_last_project(self):
        """Restore last project directory from global config."""
        last_dir = self.global_config.last_project_directory
        if last_dir and Path(last_dir).is_dir():
            self.project_path = last_dir
            self.project_label.setText(last_dir)
            self.open_dir_button.setVisible(True)
            self.load_project_config(last_dir)

    def open_project_directory(self):
        """Open project directory in file manager."""
        if self.project_path:
            import subprocess
            import platform

            system = platform.system()
            if system == "Linux":
                subprocess.Popen(["xdg-open", self.project_path])
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", self.project_path])
            elif system == "Windows":
                subprocess.Popen(["explorer", self.project_path])

    def load_project_config(self, project_path: str):
        """Load project configuration."""
        self.config = load_project_config(project_path)

        # Determine header template source:
        # 1. translate/header.txt (highest priority - persistent project header)
        # 2. config.header_template (from project config.json)
        # 3. global_config.default_header_template
        header_from_translate = load_header_from_translate(project_path)
        if header_from_translate is not None:
            self.config.header_template = header_from_translate
        elif not self.config.header_template and self.global_config.default_header_template:
            self.config.header_template = self.global_config.default_header_template

        # Temporarily block signals to avoid triggering auto-save during load
        self.crop_input.blockSignals(True)
        self.brightness_spin.blockSignals(True)
        self.time_start_input.blockSignals(True)
        self.time_end_input.blockSignals(True)
        self.ocr_parallel_spin.blockSignals(True)
        self.remove_credits_checkbox.blockSignals(True)
        self.header_text.blockSignals(True)

        # Update UI with config values
        if self.config.crop_width > 0:
            self.crop_input.setText(
                f"{self.config.crop_x}, {self.config.crop_y}, "
                f"{self.config.crop_width}, {self.config.crop_height}"
            )

        self.brightness_spin.setValue(self.config.brightness)
        self.time_start_input.setText(self.config.time_start)
        self.time_end_input.setText(self.config.time_end)
        self.ocr_parallel_spin.setValue(self.config.ocr_parallel)
        self.remove_credits_checkbox.setChecked(self.config.remove_credits)
        self.header_text.setPlainText(self.config.header_template)

        # Re-enable signals
        self.crop_input.blockSignals(False)
        self.brightness_spin.blockSignals(False)
        self.time_start_input.blockSignals(False)
        self.time_end_input.blockSignals(False)
        self.ocr_parallel_spin.blockSignals(False)
        self.remove_credits_checkbox.blockSignals(False)
        self.header_text.blockSignals(False)

    def on_crop_select_clicked(self):
        """Open crop selector dialog."""
        if not self.project_path:
            QMessageBox.warning(self, "Error", "Please select a project directory first")
            return

        # Find MKV files
        mkv_files = list(Path(self.project_path).glob("*.mkv"))
        if not mkv_files:
            QMessageBox.warning(self, "Error", "No MKV files found in project directory")
            return

        # Get existing crop coordinates if available
        existing_crop = None
        if self.config and self.config.crop_width > 0:
            existing_crop = (self.config.crop_x, self.config.crop_y,
                           self.config.crop_width, self.config.crop_height)

        dialog = CropSelectorDialog([str(f) for f in mkv_files], existing_crop, self.last_timeline_position, self)
        dialog.crop_selected.connect(self.on_crop_selected)
        if dialog.exec():
            self.last_selected_episode = dialog.get_selected_episode()
            self.last_timeline_position = dialog.get_timeline_position()

    def on_crop_selected(self, x: int, y: int, width: int, height: int):
        """Handle crop selection."""
        self.crop_input.setText(f"{x}, {y}, {width}, {height}")
        if self.config:
            self.config.crop_x = x
            self.config.crop_y = y
            self.config.crop_width = width
            self.config.crop_height = height

    def on_brightness_test_clicked(self):
        """Open brightness tester dialog."""
        if not self.project_path:
            QMessageBox.warning(self, "Error", "Please select a project directory first")
            return

        # Find MKV files
        mkv_files = list(Path(self.project_path).glob("*.mkv"))
        if not mkv_files:
            QMessageBox.warning(self, "Error", "No MKV files found in project directory")
            return

        dialog = BrightnessTesterDialog(
            [str(f) for f in mkv_files],
            self.last_selected_episode,
            self.last_timeline_position,
            self.brightness_spin.value(),
            self
        )
        dialog.brightness_selected.connect(self.on_brightness_selected)
        dialog.exec()

    def on_brightness_selected(self, brightness: int):
        """Handle brightness selection."""
        self.brightness_spin.setValue(brightness)
        if self.config:
            self.config.brightness = brightness

    def on_save_header_default(self):
        """Save current header template as the global default."""
        current_header = self.header_text.toPlainText()
        if not current_header.strip():
            QMessageBox.warning(self, "Warning", "Header template is empty")
            return

        self.global_config.default_header_template = current_header
        save_global_config(self.global_config)
        QMessageBox.information(self, "Success", "Header template saved as default")

    def on_load_header_default(self):
        """Load the global default header template."""
        if not self.global_config.default_header_template:
            QMessageBox.warning(self, "Warning", "No default header template saved")
            return

        self.header_text.setPlainText(self.global_config.default_header_template)

    def auto_save_config(self):
        """Automatically save configuration when any input changes."""
        if not self.config:
            return

        # Update config from UI
        crop_values = self.crop_input.text().split(',')
        if len(crop_values) == 4:
            try:
                self.config.crop_x = int(crop_values[0].strip())
                self.config.crop_y = int(crop_values[1].strip())
                self.config.crop_width = int(crop_values[2].strip())
                self.config.crop_height = int(crop_values[3].strip())
            except ValueError:
                pass  # Ignore invalid crop values during typing

        self.config.brightness = self.brightness_spin.value()
        self.config.time_start = self.time_start_input.text()
        self.config.time_end = self.time_end_input.text()
        self.config.ocr_parallel = self.ocr_parallel_spin.value()
        self.config.remove_credits = self.remove_credits_checkbox.isChecked()
        self.config.header_template = self.header_text.toPlainText()

        save_project_config(self.config)

    def on_save_config_clicked(self):
        """Manually save configuration (kept for compatibility/explicit saves)."""
        if not self.config:
            QMessageBox.warning(self, "Error", "No project loaded")
            return

        self.auto_save_config()
        QMessageBox.information(self, "Success", "Configuration saved")

    def on_start_processing_clicked(self):
        """Start or stop pipeline processing."""
        if self.is_running:
            # Stop pipeline
            if self.pipeline:
                self.pipeline.stop()
            return

        # Validate configuration
        if not self.validate_configuration():
            return

        # Config is already saved via auto-save, just ensure it's up to date
        self.auto_save_config()

        # Create and start pipeline
        self.pipeline = Pipeline(self.config, self.global_config)
        self.pipeline.command_started.connect(self.terminal.append_command)
        self.pipeline.output_received.connect(self.terminal.append_output)
        self.pipeline.error_occurred.connect(self.on_pipeline_error)
        self.pipeline.phase_completed.connect(self.on_phase_completed)
        self.pipeline.pipeline_finished.connect(self.on_pipeline_finished)

        # Update UI
        self.is_running = True
        self.start_button.setText("Stop")
        self.update_window_title()
        self.disable_ui()

        # Clear terminal and start
        self.terminal.clear()
        self.terminal.append_output("Starting pipeline...\n")
        self.pipeline.start()

    def validate_configuration(self) -> bool:
        """Validate configuration before starting."""
        if not self.config:
            QMessageBox.warning(self, "Error", "No project loaded")
            return False

        # Update config from UI first
        crop_values = self.crop_input.text().split(',')
        if len(crop_values) == 4:
            try:
                self.config.crop_x = int(crop_values[0].strip())
                self.config.crop_y = int(crop_values[1].strip())
                self.config.crop_width = int(crop_values[2].strip())
                self.config.crop_height = int(crop_values[3].strip())
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid crop values")
                return False

        self.config.brightness = self.brightness_spin.value()
        self.config.time_start = self.time_start_input.text()
        self.config.time_end = self.time_end_input.text()
        self.config.ocr_parallel = self.ocr_parallel_spin.value()
        self.config.remove_credits = self.remove_credits_checkbox.isChecked()
        self.config.header_template = self.header_text.toPlainText()

        valid, error_msg = validate_config(self.config)
        if not valid:
            QMessageBox.warning(self, "Configuration Error", error_msg)
            return False

        return True

    def on_pipeline_error(self, error_msg: str):
        """Handle pipeline errors."""
        self.terminal.append_error(f"\nERROR: {error_msg}\n")

    def on_phase_completed(self, phase_name: str):
        """Handle phase completion."""
        self.terminal.append_output(f"\n✓ {phase_name} completed\n")

    def on_pipeline_finished(self, success: bool):
        """Handle pipeline completion."""
        self.is_running = False
        self.start_button.setText("Start Processing")
        self.update_window_title()
        self.enable_ui()

        if success:
            self.terminal.append_output("\n✓ Pipeline completed successfully!\n")
            QMessageBox.information(self, "Success", "Processing completed successfully!")
        else:
            QMessageBox.critical(self, "Error", "Pipeline failed. Check terminal output for details.")

    def disable_ui(self):
        """Disable UI during processing."""
        self.crop_input.setEnabled(False)
        self.brightness_spin.setEnabled(False)
        self.time_start_input.setEnabled(False)
        self.time_end_input.setEnabled(False)
        self.ocr_parallel_spin.setEnabled(False)
        self.remove_credits_checkbox.setEnabled(False)
        self.header_text.setEnabled(False)
        self.save_default_btn.setEnabled(False)
        self.load_default_btn.setEnabled(False)

    def enable_ui(self):
        """Re-enable UI after processing."""
        self.crop_input.setEnabled(True)
        self.brightness_spin.setEnabled(True)
        self.time_start_input.setEnabled(True)
        self.time_end_input.setEnabled(True)
        self.ocr_parallel_spin.setEnabled(True)
        self.remove_credits_checkbox.setEnabled(True)
        self.header_text.setEnabled(True)
        self.save_default_btn.setEnabled(True)
        self.load_default_btn.setEnabled(True)

    def check_dependencies(self):
        """Check if required CLI tools are available."""
        tools = ['ocrp', 'ass-credits', 'ass-qafix', 'ass-header',
                 'sub-visualize', 'subs-translator', 'submerge', 'ffmpeg']

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

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
