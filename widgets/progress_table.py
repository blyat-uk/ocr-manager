"""Progress table widget with embedded progress bars for OCR status."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QProgressBar, QHeaderView, QLabel
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from core.ocr_worker import FileStatus


class ProgressTableWidget(QWidget):
    """Table widget showing OCR progress for multiple files."""

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_rows: dict[str, int] = {}  # filename -> row index
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header
        header = QLabel("OCR Progress:")
        layout.addWidget(header)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["File", "Progress", "Status"])

        # Column sizing
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 200)
        self.table.setColumnWidth(2, 100)

        # Table settings
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)

        layout.addWidget(self.table)

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
            self.table.setItem(row, 0, file_item)

            # Progress bar
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(100 if status == FileStatus.DONE else 0)
            progress_bar.setTextVisible(True)
            progress_bar.setFormat("%p%")
            self.table.setCellWidget(row, 1, progress_bar)

            # Status column
            status_item = QTableWidgetItem(self.STATUS_TEXT[status])
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            status_item.setForeground(QColor(self.STATUS_COLORS[status]))
            self.table.setItem(row, 2, status_item)

    def update_status(self, filename: str, status: FileStatus):
        """Update the status cell for a file."""
        if filename not in self._file_rows:
            return

        row = self._file_rows[filename]
        status_item = self.table.item(row, 2)
        if status_item:
            status_item.setText(self.STATUS_TEXT[status])

            # Apply status color
            color = self.STATUS_COLORS[status]
            status_item.setForeground(QColor(color))

    def update_progress(self, filename: str, percent: int):
        """Update progress bar for a file."""
        if filename not in self._file_rows:
            return

        row = self._file_rows[filename]
        progress_bar = self.table.cellWidget(row, 1)
        if isinstance(progress_bar, QProgressBar):
            progress_bar.setValue(percent)

    def clear(self):
        """Clear the table."""
        self.table.setRowCount(0)
        self._file_rows.clear()
