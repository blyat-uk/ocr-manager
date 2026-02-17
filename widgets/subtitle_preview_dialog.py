"""Real-time subtitle preview dialog for OCR processing."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt6.QtCore import Qt


def _format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS.cc or H:MM:SS.cc."""
    total_cs = int(round(seconds * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60

    if h > 0:
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
    return f"{m}:{s:02d}.{cs:02d}"


class SubtitlePreviewDialog(QDialog):
    """Non-modal dialog displaying OCR subtitle lines in real time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Subtitle Preview")
        self.resize(600, 400)
        self.setModal(False)

        # Data storage: filename -> list of (start, end, text)
        self._data: dict[str, list[tuple[float, float, str]]] = {}
        self._current_file: str | None = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header row
        header_layout = QHBoxLayout()
        self._file_label = QLabel("No file selected")
        self._file_label.setObjectName("muted")
        header_layout.addWidget(self._file_label, 1)

        self._count_label = QLabel("")
        self._count_label.setObjectName("muted")
        header_layout.addWidget(self._count_label)

        layout.addLayout(header_layout)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Start", "End", "Text"])

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.setColumnWidth(0, 90)
        self._table.setColumnWidth(1, 90)

        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)

        layout.addWidget(self._table)

    def show_file(self, filename: str):
        """Switch display to the given file, rebuilding table from stored data."""
        self._current_file = filename
        self._file_label.setText(filename)
        self._file_label.setObjectName("")
        self._file_label.style().unpolish(self._file_label)
        self._file_label.style().polish(self._file_label)

        entries = self._data.get(filename, [])
        self._rebuild_table(entries)

    def _rebuild_table(self, entries: list[tuple[float, float, str]]):
        """Rebuild the table from a list of subtitle entries."""
        self._table.setRowCount(len(entries))
        for row, (start, end, text) in enumerate(entries):
            self._set_row(row, start, end, text)
        self._update_count(len(entries))

    def _set_row(self, row: int, start: float, end: float, text: str):
        """Set data for a single table row."""
        start_item = QTableWidgetItem(_format_timestamp(start))
        start_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, 0, start_item)

        end_item = QTableWidgetItem(_format_timestamp(end))
        end_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, 1, end_item)

        # Replace \N with actual newline for display
        display_text = text.replace('\\N', '\n')
        text_item = QTableWidgetItem(display_text)
        self._table.setItem(row, 2, text_item)

    def _update_count(self, count: int):
        """Update the subtitle count label."""
        self._count_label.setText(f"{count} subtitle{'s' if count != 1 else ''}")

    def on_subtitle_detected(self, filename: str, start: float, end: float, text: str):
        """Handle a new subtitle detection from the pipeline."""
        if filename not in self._data:
            self._data[filename] = []
        self._data[filename].append((start, end, text))

        # If this file is currently displayed, append row and auto-scroll
        if filename == self._current_file:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._set_row(row, start, end, text)
            self._table.scrollToBottom()
            self._update_count(row + 1)

    def clear_all(self):
        """Clear all stored data (called when new pipeline run starts)."""
        self._data.clear()
        self._current_file = None
        self._table.setRowCount(0)
        self._file_label.setText("No file selected")
        self._file_label.setObjectName("muted")
        self._file_label.style().unpolish(self._file_label)
        self._file_label.style().polish(self._file_label)
        self._count_label.setText("")
