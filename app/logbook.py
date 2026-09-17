"""Log text per key ("Pipeline" or a file name), as today's core/log_store.py
kept it: appended text, the oldest text dropped past LOG_LIMIT characters per
key. Qt-free and pure; the controller emits log_appended for each append.
"""
from __future__ import annotations

PIPELINE_LOG = "Pipeline"
LOG_LIMIT = 512_000


class LogBook:
    def __init__(self) -> None:
        self._logs: dict[str, str] = {}

    def append(self, key: str, text: str) -> str:
        """Append `text` (a newline is added when it has none) and return what
        was appended; "" for no text."""
        if not text:
            return ""
        if not text.endswith("\n"):
            text += "\n"
        self._logs[key] = (self._logs.get(key, "") + text)[-LOG_LIMIT:]
        return text

    def keys(self) -> list[str]:
        """Keys in the order they first got text."""
        return list(self._logs)

    def text(self, key: str) -> str:
        return self._logs.get(key, "")

    def clear(self) -> None:
        self._logs.clear()
