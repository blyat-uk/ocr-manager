"""Watching a project folder for videos appearing and disappearing (ruling C7).

FolderWatch wraps a QFileSystemWatcher on the folder and on the directories
its outputs are written into (a deleted or emptied output changes the file's
done state), debounced:
`settled` fires once changes have stopped for `debounce_ms`, so a burst (a
copy of several episodes, a save's temporary file) is handled once. What a
change means is the owner's business (ProjectController reconciles the file
list, and ignores changes while a run writes its outputs).

Which output directories to watch is the project layout's answer, given by
the owner: a folder's chi/, and none for an episode, whose `<stem>.zh.ass`
sits in the watched folder itself.
"""
from __future__ import annotations

import os
from collections.abc import Iterable

from PyQt6.QtCore import QFileSystemWatcher, QObject, QTimer, pyqtSignal


class FolderWatch(QObject):
    settled = pyqtSignal()

    def __init__(self, debounce_ms: int = 300, parent: QObject | None = None):
        super().__init__(parent)
        self._path: str | None = None
        self._outputs: tuple[str, ...] = ()
        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._on_changed)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(debounce_ms)
        self._timer.timeout.connect(self.settled.emit)

    def watch(self, path: str, output_dirs: Iterable[str] = ()) -> None:
        """Watch `path` and the `output_dirs` (absolute) its outputs go to."""
        self.stop()
        self._path = path
        self._outputs = tuple(output_dirs)
        self.rewatch()

    def rewatch(self) -> None:
        """Watch the folder and its output directories again where they
        exist and are not watched: chi/ once a run created it, the folder
        after it vanished and came back (the watcher drops a deleted path)."""
        if self._path is None:
            return
        watched = set(self._watcher.directories())
        for directory in (self._path, *self._outputs):
            if directory not in watched and os.path.isdir(directory):
                self._watcher.addPath(directory)

    def stop(self) -> None:
        self._path = None
        self._outputs = ()
        self._timer.stop()
        directories = self._watcher.directories()
        if directories:
            self._watcher.removePaths(directories)

    def _on_changed(self, _directory: str) -> None:
        if self._path is not None:
            self._timer.start()
