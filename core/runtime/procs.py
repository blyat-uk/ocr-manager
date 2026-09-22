"""Running the installer's child processes with streamed output and a cancel.

The children (pip, and the verify / model-prefetch checks) are separate
processes that hold no engine lease and no project state, so a cancel or a
timeout ends them with `Popen.kill()` and the caller deletes the partial
engine dir. (The project's cooperative-cancellation rule is about worker
*threads*, which may hold an engine lease; nothing here is a thread that
does.)
"""
from __future__ import annotations

import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from core.proc import TEXT_ENCODING, hidden_child

POLL_SECONDS = 0.1
TAIL_LINES = 40


class Cancelled(Exception):
    """The user cancelled while a child process ran."""


@dataclass
class ProcessResult:
    returncode: int
    tail: list[str] = field(default_factory=list)       # the last output lines, for error reports
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class ProcessRunner(Protocol):
    def run(self, command: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: Path | None = None,
            on_line: Callable[[str], None] = lambda line: None, cancel: threading.Event | None = None,
            timeout: float | None = None) -> ProcessResult:
        """Run `command` to its end, calling `on_line` per output line
        (stdout and stderr merged). Raises Cancelled when `cancel` is set."""
        ...


class SubprocessRunner:
    def run(self, command, *, env=None, cwd=None, on_line=lambda line: None, cancel=None,
            timeout=None) -> ProcessResult:
        proc = subprocess.Popen(list(command), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=None if env is None else dict(env),
                                cwd=None if cwd is None else str(cwd), text=True, bufsize=1,
                                **TEXT_ENCODING, **hidden_child())
        lines: queue.Queue[str | None] = queue.Queue()

        def pump() -> None:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    lines.put(line.rstrip("\r\n"))
            finally:
                lines.put(None)

        reader = threading.Thread(target=pump, name="runtime-output", daemon=True)
        reader.start()
        tail: deque[str] = deque(maxlen=TAIL_LINES)
        deadline = None if timeout is None else time.monotonic() + timeout
        timed_out = False
        eof = False
        try:
            while not eof:
                if cancel is not None and cancel.is_set():
                    self._end(proc)
                    raise Cancelled()
                if deadline is not None and time.monotonic() > deadline:
                    timed_out = True
                    self._end(proc)
                    break
                try:
                    line = lines.get(timeout=POLL_SECONDS)
                except queue.Empty:
                    continue
                if line is None:
                    eof = True
                    continue
                tail.append(line)
                on_line(line)
            returncode = proc.wait()
        finally:
            if proc.poll() is None:                     # an exception out of on_line
                self._end(proc)
            reader.join(timeout=5)
        return ProcessResult(returncode, list(tail), timed_out)

    @staticmethod
    def _end(proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:           # pragma: no cover - a process that ignores SIGKILL
            pass
