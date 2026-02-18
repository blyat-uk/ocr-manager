"""Logs dialog with collapsible per-file log entries."""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QWidget, QToolButton,
                              QTextEdit, QScrollArea, QSizePolicy)
from PyQt6.QtCore import Qt

from core.log_store import LogStore


class LogEntry(QWidget):
    """Single collapsible log item: clickable header + hidden text body."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header button
        self.header = QToolButton()
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.ArrowType.RightArrow)
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.toggled.connect(self._on_toggled)
        layout.addWidget(self.header)

        # Text body
        self.body = QTextEdit()
        self.body.setReadOnly(True)
        self.body.setVisible(False)
        self.body.setMinimumHeight(120)
        self.body.setMaximumHeight(300)
        self.body.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self.body)

    def _on_toggled(self, checked: bool):
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self.body.setVisible(checked)

    def set_text(self, text: str):
        self.body.setPlainText(text)

    def append_text(self, text: str):
        self.body.moveCursor(self.body.textCursor().MoveOperation.End)
        self.body.insertPlainText(text)


class LogsDialog(QDialog):
    """Non-modal dialog showing collapsible per-file log output."""

    def __init__(self, log_store: LogStore, parent=None):
        super().__init__(parent)
        self.log_store = log_store
        self._entries: dict[str, LogEntry] = {}

        self.setWindowTitle("Logs")
        self.resize(700, 500)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(4)
        self._layout.addStretch()
        scroll.setWidget(self._container)

        # Load existing logs
        self._load_existing_logs()

        # Connect for live updates
        self.log_store.log_appended.connect(self._on_log_appended)

    def _load_existing_logs(self):
        for key in self.log_store.keys():
            entry = self._get_or_create_entry(key)
            entry.set_text(self.log_store.get(key))

    def _get_or_create_entry(self, key: str) -> LogEntry:
        if key not in self._entries:
            entry = LogEntry(key)
            self._entries[key] = entry
            # Insert before the stretch
            self._layout.insertWidget(self._layout.count() - 1, entry)
        return self._entries[key]

    def _on_log_appended(self, key: str, text: str):
        entry = self._get_or_create_entry(key)
        entry.append_text(text)
