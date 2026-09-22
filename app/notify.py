"""The desktop notification a finished run sends.

Linux (and the BSDs): notify-send. macOS: osascript's `display notification`.
Windows: nothing -- there is no notifier program to start there.

The notifier is started and left running: the GUI thread never waits for it.
A daemon thread does, so the finished child is reaped (no zombie), and kills
it after NOTIFY_TIMEOUT_SECONDS (notify-send can hang when no notification
daemon answers).
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading

from core.proc import hidden_child

logger = logging.getLogger(__name__)

APP_NAME = "OCR Manager"
NOTIFY_TIMEOUT_SECONDS = 10


def notify_command(title: str, body: str, urgency: str, platform: str | None = None) -> list[str] | None:
    """The notifier command line for `platform` (sys.platform by default);
    None where there is none. `urgency` is notify-send's ("normal",
    "critical"); macOS has no equivalent."""
    platform = sys.platform if platform is None else platform
    if platform == "darwin":
        script = (f'display notification "{_applescript_text(body)}" '
                  f'with title "{APP_NAME}" subtitle "{_applescript_text(title)}"')
        return ["osascript", "-e", script]
    if platform.startswith(("linux", "freebsd", "openbsd", "netbsd")):
        return ["notify-send", "-a", APP_NAME, "-u", urgency, title, body]
    return None


def send_notification(title: str, body: str, urgency: str) -> None:
    """Start the notifier without waiting for it; a missing one is ignored."""
    command = notify_command(title, body, urgency)
    if command is None:
        return
    try:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, **hidden_child())
    except OSError:
        return                                          # the notifier is not installed
    threading.Thread(target=_reap, args=(child, command[0]), name="notify-reaper", daemon=True).start()


def _reap(child: subprocess.Popen, program: str) -> None:
    try:
        child.wait(timeout=NOTIFY_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        logger.warning("%s did not return within %s s", program, NOTIFY_TIMEOUT_SECONDS)
        child.kill()
        child.wait()


def _applescript_text(text: str) -> str:
    """`text` for the inside of an AppleScript string literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')
