"""Async config file saver to prevent GUI blocking."""

import json
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


class ConfigSaverWorker(QObject):
    """Worker for async config file saving."""

    finished = pyqtSignal(bool)  # success/failure

    def __init__(self):
        super().__init__()

    @pyqtSlot(str, dict)
    def save(self, config_path: str, data: dict):
        """Save config data to file (runs in worker thread)."""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.finished.emit(True)
        except IOError:
            self.finished.emit(False)


class AsyncConfigSaver(QObject):
    """Manager for async config saving with worker thread."""

    # Signal to trigger save in worker thread
    _save_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: ConfigSaverWorker | None = None

    def _ensure_thread(self):
        """Lazily initialize worker thread."""
        if self._thread is None:
            self._thread = QThread()
            self._worker = ConfigSaverWorker()
            self._worker.moveToThread(self._thread)
            # Connect signal to worker's slot for thread-safe invocation
            self._save_requested.connect(self._worker.save)
            self._thread.start()

    def save(self, config_path: str, data: dict):
        """Queue a save operation to run in background thread."""
        self._ensure_thread()
        # Emit signal to trigger save in worker thread
        self._save_requested.emit(config_path, data)

    def stop(self):
        """Stop the worker thread gracefully."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None
