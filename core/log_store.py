"""Accumulates log output per-key for inspection in the Logs dialog."""

from PyQt6.QtCore import QObject, pyqtSignal


class LogStore(QObject):
    """Stores log text keyed by filename (or a fixed key for QA output)."""

    QA_KEY = "Quality Assurance"

    log_appended = pyqtSignal(str, str)  # key, new_text

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logs: dict[str, str] = {}
        self._order: list[str] = []

    def append(self, key: str, text: str):
        """Append text to the log for *key*, creating it if needed."""
        if key not in self._logs:
            self._logs[key] = ""
            self._order.append(key)
        self._logs[key] += text
        self.log_appended.emit(key, text)

    def get(self, key: str) -> str:
        """Return accumulated text for *key* (empty string if missing)."""
        return self._logs.get(key, "")

    def keys(self) -> list[str]:
        """Return keys in insertion order."""
        return list(self._order)

    def clear(self):
        """Remove all stored logs."""
        self._logs.clear()
        self._order.clear()
