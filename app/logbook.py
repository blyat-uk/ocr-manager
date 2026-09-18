"""Log text per key, as the v1 app's core/log_store.py kept it: appended text, the
oldest text dropped past LOG_LIMIT characters per key. Keys: "Pipeline" (the
run and the controller), "Detections" (failed detection and proof jobs: they
are not retried, so this is their only record and a run start keeps it) and
file names (a run's per-file lines). Qt-free and pure; the controller emits
log_appended for each append.
"""
from __future__ import annotations

PIPELINE_LOG = "Pipeline"
DETECTIONS_LOG = "Detections"
FIRST_KEYS = (PIPELINE_LOG, DETECTIONS_LOG)
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
        """"Pipeline", then "Detections", then the other keys in the order
        they first got text."""
        first = [key for key in FIRST_KEYS if key in self._logs]
        return first + [key for key in self._logs if key not in FIRST_KEYS]

    def text(self, key: str) -> str:
        return self._logs.get(key, "")

    def clear(self, keep: tuple[str, ...] = ()) -> None:
        """Forget every key except those in `keep`."""
        for key in [key for key in self._logs if key not in keep]:
            del self._logs[key]
