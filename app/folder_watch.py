"""Watching a project folder for videos appearing and disappearing (ruling C7).

FolderWatch wraps a QFileSystemWatcher on the folder and on the directories
its outputs are written into (a deleted or emptied output changes the file's
done state), debounced:
`settled` fires once changes have stopped for `debounce_ms`, so a burst (a
copy of several episodes, a save's temporary file) is handled once. What a
change means is the owner's business (ProjectController reconciles the file
list, and ignores changes while a run writes its outputs).

Which output directories to watch is the project layout's answer, given by
the owner: the output subfolder (`zh/`), or none when the outputs sit in the
watched folder itself. A settings change moves it (set_outputs).
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
        exist and are not watched: the output subfolder once a run created
        it, the folder after it vanished and came back (the watcher drops a
        deleted path)."""
        if self._path is None:
            return
        watched = set(self._watcher.directories())
        for directory in (self._path, *self._outputs):
            if directory not in watched and os.path.isdir(directory):
                self._watcher.addPath(directory)

    def set_outputs(self, output_dirs: Iterable[str]) -> None:
        """Watch `output_dirs` instead of the current ones, the folder itself
        still watched; nothing while no folder is watched."""
        if self._path is None:
            return
        outputs = tuple(output_dirs)
        watched = set(self._watcher.directories())
        dropped = [directory for directory in self._outputs
                   if directory not in outputs and directory != self._path and directory in watched]
        if dropped:
            self._watcher.removePaths(dropped)
        self._outputs = outputs
        self.rewatch()

    def directories(self) -> tuple[str, ...]:
        """What is being watched now: the folder and each existing output directory."""
        return tuple(self._watcher.directories())

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
