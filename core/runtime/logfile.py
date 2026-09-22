"""Give the process a real stdout/stderr when it was started without one.

Under pythonw.exe (Windows, no console) `sys.stdout`/`sys.stderr` are None
and file descriptors 1 and 2 are not open; a macOS app started from Finder
writes them to /dev/null. `videocr.utils.suppress_output()` os.dup()s fds 1
and 2 around every engine build, so without real descriptors every engine
build raises. `redirect_if_needed()` opens `<data>/logs/ocr-manager.log`
(the previous run's log kept as `ocr-manager.log.1`), dup2()s it onto fds 1
and 2 and rebinds sys.stdout/sys.stderr to it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def fd_is_open(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True


def fd_is_devnull(fd: int) -> bool:
    try:
        return os.path.samestat(os.fstat(fd), os.stat(os.devnull))
    except (OSError, ValueError):
        return False


def needs_redirect(*, discard_devnull: bool = False) -> bool:
    """True when stdout/stderr go nowhere usable: None streams or closed
    descriptors, and (with `discard_devnull`, used for bundles) output that
    goes to /dev/null."""
    if sys.stdout is None or sys.stderr is None:
        return True
    if not (fd_is_open(1) and fd_is_open(2)):
        return True
    return discard_devnull and (fd_is_devnull(1) or fd_is_devnull(2))


def rotate(path: Path) -> None:
    """Keep the previous run's log as `<name>.1`."""
    if path.exists():
        os.replace(path, path.with_name(path.name + ".1"))


def redirect_to(path: Path) -> Path:
    """Send fds 1 and 2, and sys.stdout/sys.stderr, to `path` (rotated first)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rotate(path)
    except OSError:
        pass                            # another instance holds it (Windows): append to it instead
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        if fd not in (1, 2):            # with 1/2 closed, os.open hands out one of them
            os.close(fd)
    sys.stdout = open(1, "w", encoding="utf-8", errors="backslashreplace", buffering=1, closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", errors="backslashreplace", buffering=1, closefd=False)
    return path


def redirect_if_needed(path: Path, *, discard_devnull: bool = False) -> Path | None:
    """`redirect_to(path)` when `needs_redirect()`; the log path, or None
    when the streams were fine and left alone. Never raises."""
    if not needs_redirect(discard_devnull=discard_devnull):
        return None
    try:
        return redirect_to(path)
    except OSError:
        return None
