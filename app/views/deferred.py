"""Coalescing: many requests in one event-loop turn run the work once."""
from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject, QTimer


class Deferred(QObject):
    """`schedule()` marks the work dirty; it runs once on the next event-loop
    turn (a zero-delay single-shot timer), however many times it was
    scheduled. `cancel()` drops a pending run (the work ran directly)."""

    def __init__(self, work: Callable[[], None], parent: QObject | None = None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(work)

    def schedule(self, *_args) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def cancel(self) -> None:
        self._timer.stop()

    def pending(self) -> bool:
        return self._timer.isActive()
