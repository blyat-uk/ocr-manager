"""File table widget with resolution, config status, and progress columns."""

from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QProgressBar, QHeaderView, QLabel, QMenu, QAbstractItemView, QPushButton
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QAction

from core.config import FileConfig, FileConfigStore
from core.ocr_worker import FileStatus


class FileTableWidget(QWidget):
    """Table widget showing files with resolution, config status, and progress."""

    # Signals
    selection_changed = pyqtSignal(list)  # List of selected filenames
    config_action_requested = pyqtSignal(str, str)  # action, filename

    # Status colors (Catppuccin Mocha)
    STATUS_COLORS = {
        FileStatus.QUEUED: "#6c7086",      # overlay0
        FileStatus.PROCESSING: "#89b4fa",  # blue
        FileStatus.COMPLETED: "#a6e3a1",   # green
        FileStatus.FAILED: "#f38ba8",      # red
        FileStatus.DONE: "#a6e3a1",        # green (same as completed)
    }

    STATUS_TEXT = {
        FileStatus.QUEUED: "Queued",
        FileStatus.PROCESSING: "Processing",
        FileStatus.COMPLETED: "Done",
        FileStatus.FAILED: "Failed",
        FileStatus.DONE: "Done",
    }

    # Config indicator colors
    CONFIG_DEFAULT = "#6c7086"      # overlay0 - gray dot (no custom config)

    # Color palette for distinct config groups (Catppuccin Mocha)
    CONFIG_COLORS = [
        "#89b4fa",  # blue
        "#a6e3a1",  # green
        "#f9e2af",  # yellow
        "#cba6f7",  # mauve
        "#fab387",  # peach
        "#94e2d5",  # teal
        "#f38ba8",  # red
        "#eba0ac",  # maroon
        "#89dceb",  # sky
        "#f5c2e7",  # pink
    ]

    # Column indices
    COL_FILE = 0
    COL_RESOLUTION = 1
    COL_CONFIG = 2
    COL_PROGRESS = 3
    COL_STATUS = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_rows: dict[str, int] = {}  # filename -> row index
        self._file_store: FileConfigStore | None = None
        self._signature_colors: dict[tuple, str] = {}  # config signature -> color
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header
        header = QLabel("Files:")
        layout.addWidget(header)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["File", "Res", "Config", "Progress", "Status"])

        # Column sizing
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(self.COL_FILE, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(self.COL_RESOLUTION, QHeaderView.ResizeMode.Fixed)
        header_view.setSectionResizeMode(self.COL_CONFIG, QHeaderView.ResizeMode.Fixed)
        header_view.setSectionResizeMode(self.COL_PROGRESS, QHeaderView.ResizeMode.Fixed)
        header_view.setSectionResizeMode(self.COL_STATUS, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(self.COL_RESOLUTION, 60)
        self.table.setColumnWidth(self.COL_CONFIG, 60)
        self.table.setColumnWidth(self.COL_PROGRESS, 150)
        self.table.setColumnWidth(self.COL_STATUS, 180)

        # Table settings
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)

        # Connect selection change
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        # Context menu
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self.table)

        # Bottom bar with unselect button
        bottom_layout = QHBoxLayout()
        self.unselect_btn = QPushButton("Unselect all")
        self.unselect_btn.setObjectName("secondary")
        self.unselect_btn.setVisible(False)
        self.unselect_btn.clicked.connect(self.clear_selection)
        bottom_layout.addWidget(self.unselect_btn)
        bottom_layout.addStretch()
        layout.addLayout(bottom_layout)

    def set_file_store(self, store: FileConfigStore):
        """Set the file config store for config status display."""
        self._file_store = store

    def set_files(self, filenames: list[str], initial_statuses: dict[str, FileStatus] = None):
        """Initialize the table with file list and optional initial statuses."""
        self._file_rows.clear()
        self.table.setRowCount(len(filenames))

        for row, filename in enumerate(filenames):
            self._file_rows[filename] = row

            # Determine initial status for this file
            status = FileStatus.QUEUED
            if initial_statuses and filename in initial_statuses:
                status = initial_statuses[filename]

            # File column
            file_item = QTableWidgetItem(filename)
            file_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, self.COL_FILE, file_item)

            # Resolution column (will be updated by update_resolution)
            res_item = QTableWidgetItem("?")
            res_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, self.COL_RESOLUTION, res_item)

            # Config indicator column
            config_item = QTableWidgetItem("")
            config_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, self.COL_CONFIG, config_item)
            self._update_config_indicator(filename)

            # Progress bar
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(100 if status == FileStatus.DONE else 0)
            progress_bar.setTextVisible(True)
            progress_bar.setFormat("%p%")
            self.table.setCellWidget(row, self.COL_PROGRESS, progress_bar)

            # Status column
            status_item = QTableWidgetItem(self.STATUS_TEXT[status])
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            status_item.setForeground(QColor(self.STATUS_COLORS[status]))
            self.table.setItem(row, self.COL_STATUS, status_item)

    def update_resolution(self, filename: str, label: str):
        """Update resolution label for a file."""
        if filename not in self._file_rows:
            return

        row = self._file_rows[filename]
        res_item = self.table.item(row, self.COL_RESOLUTION)
        if res_item:
            res_item.setText(label)

    def update_config_indicator(self, filename: str):
        """Update the config indicator for a file.

        Since colors are based on matching configs across files,
        we refresh all indicators when any file's config changes.
        """
        self.refresh_all_config_indicators()

    def _update_config_indicator(self, filename: str):
        """Internal method to update a single config indicator using cached colors."""
        if filename not in self._file_rows:
            return

        row = self._file_rows[filename]
        config_item = self.table.item(row, self.COL_CONFIG)
        if not config_item:
            return

        # Get file config and signature
        config = self._file_store.get(filename) if self._file_store else None

        if config and config.has_any_custom():
            signature = config.config_signature()
            color = self._signature_colors.get(signature, self.CONFIG_COLORS[0])
            config_item.setText("\u25cf")  # Filled circle
            config_item.setForeground(QColor(color))
            config_item.setToolTip(self._format_config_tooltip(config))
        else:
            config_item.setText("\u25cb")  # Empty circle
            config_item.setForeground(QColor(self.CONFIG_DEFAULT))
            config_item.setToolTip("Using defaults")

    def _format_config_tooltip(self, config: FileConfig) -> str:
        """Format tooltip showing custom config details."""
        parts = []
        if config.has_custom_crop():
            crop = config.get_crop_tuple()
            parts.append(f"Crop: {crop[0]},{crop[1]} {crop[2]}x{crop[3]}")
        if config.has_custom_brightness():
            parts.append(f"Brightness: {config.brightness}")
        if config.has_custom_time_range():
            time_parts = []
            if config.time_start:
                time_parts.append(f"from {config.time_start}")
            if config.time_end:
                time_parts.append(f"to {config.time_end}")
            parts.append(f"Time: {' '.join(time_parts)}")
        return "Custom: " + ", ".join(parts) if parts else "Custom settings"

    def update_status(self, filename: str, status: FileStatus):
        """Update the status cell for a file."""
        if filename not in self._file_rows:
            return

        row = self._file_rows[filename]
        status_item = self.table.item(row, self.COL_STATUS)
        if status_item:
            status_item.setText(self.STATUS_TEXT[status])
            color = self.STATUS_COLORS[status]
            status_item.setForeground(QColor(color))

    def update_status_text(self, filename: str, text: str):
        """Update the status cell with custom text (e.g., 'Extracting dialogue').

        Uses the PROCESSING color (blue) since this is only called during processing.
        """
        if filename not in self._file_rows:
            return

        row = self._file_rows[filename]
        status_item = self.table.item(row, self.COL_STATUS)
        if status_item:
            status_item.setText(text)
            status_item.setForeground(QColor(self.STATUS_COLORS[FileStatus.PROCESSING]))

    def update_progress(self, filename: str, percent: int):
        """Update progress bar for a file."""
        if filename not in self._file_rows:
            return

        row = self._file_rows[filename]
        progress_bar = self.table.cellWidget(row, self.COL_PROGRESS)
        if isinstance(progress_bar, QProgressBar):
            progress_bar.setValue(percent)

    def get_selected_filenames(self) -> list[str]:
        """Get list of selected filenames."""
        selected_rows = set()
        for item in self.table.selectedItems():
            selected_rows.add(item.row())

        filenames = []
        for filename, row in self._file_rows.items():
            if row in selected_rows:
                filenames.append(filename)

        return filenames

    def select_file(self, filename: str):
        """Select a specific file in the table."""
        if filename not in self._file_rows:
            return

        row = self._file_rows[filename]
        self.table.selectRow(row)

    def clear_selection(self):
        """Clear current selection."""
        self.table.clearSelection()

    def _on_selection_changed(self):
        """Handle selection change."""
        filenames = self.get_selected_filenames()
        self.unselect_btn.setVisible(len(filenames) > 0)
        self.selection_changed.emit(filenames)

    def _show_context_menu(self, pos):
        """Show context menu for right-click."""
        item = self.table.itemAt(pos)
        if not item:
            return

        selected = self.get_selected_filenames()
        if not selected:
            return

        menu = QMenu(self)

        # Copy settings action
        copy_action = QAction("Copy Settings", self)
        copy_action.triggered.connect(lambda: self.config_action_requested.emit("copy", selected[0]))
        menu.addAction(copy_action)

        # Paste settings action
        paste_action = QAction("Paste Settings to Selected", self)
        paste_action.triggered.connect(lambda: self.config_action_requested.emit("paste", ""))
        menu.addAction(paste_action)

        menu.exec(self.table.viewport().mapToGlobal(pos))

    def refresh_all_config_indicators(self):
        """Refresh config indicators for all files.

        Computes config signatures, assigns colors to unique signatures,
        then updates all indicators.
        """
        # Collect unique signatures from files with custom configs
        unique_signatures: list[tuple] = []
        for filename in self._file_rows:
            config = self._file_store.get(filename) if self._file_store else None
            if config and config.has_any_custom():
                sig = config.config_signature()
                if sig not in unique_signatures:
                    unique_signatures.append(sig)

        # Assign colors to signatures (stable ordering based on first appearance)
        self._signature_colors = {}
        for i, sig in enumerate(unique_signatures):
            self._signature_colors[sig] = self.CONFIG_COLORS[i % len(self.CONFIG_COLORS)]

        # Update all indicators
        for filename in self._file_rows:
            self._update_config_indicator(filename)

    def clear(self):
        """Clear the table."""
        self.table.setRowCount(0)
        self._file_rows.clear()

    def get_all_filenames(self) -> list[str]:
        """Get all filenames in the table."""
        return list(self._file_rows.keys())
