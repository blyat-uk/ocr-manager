"""The welcome screen's "Recent files & folders": what was opened, newest
first -- an episode (its video file) or a folder of episodes -- and the text
each row shows (Polygluttony's recent folders, ported).

Qt-free: the list rules and the wording only. `app/views/open_folder.py`
keeps the list in QSettings and draws the rows.

- An entry is recorded each time something opens; opening it again moves it
  to the top. The newest RECENTS_CAP are kept.
- A folder remembers how many episodes it held when it was opened, so the row
  can say so without listing the folder again.
- An entry whose path is not there is not shown, but it is kept: a drive
  that is not mounted brings its entries back when it is. It ages out of the
  list like any other.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

RECENTS_CAP = 10
FOLDER, EPISODE = "folder", "episode"
KINDS = (FOLDER, EPISODE)

# Folder names that do not say which show it is: a season folder goes by the
# nearest ancestor that does. A Windows drive ("C:") is never a show name.
_GENERIC = re.compile(r"season", re.IGNORECASE)
_DRIVE = re.compile(r"^[a-z]:$", re.IGNORECASE)


@dataclass(frozen=True)
class RecentEntry:
    path: str
    kind: str                   # FOLDER or EPISODE
    opened: float               # epoch seconds
    count: int | None = None    # a folder's episodes when it was opened; None for an episode

    def exists(self) -> bool:
        """The path is still there, and still what it was."""
        return os.path.isfile(self.path) if self.kind == EPISODE else os.path.isdir(self.path)


def _key(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def recorded(entries: list[RecentEntry], entry: RecentEntry) -> list[RecentEntry]:
    """`entries` with `entry` first, replacing any entry for the same path,
    capped at RECENTS_CAP."""
    key = _key(entry.path)
    return [entry, *(old for old in entries if _key(old.path) != key)][:RECENTS_CAP]


def without(entries: list[RecentEntry], path: str) -> list[RecentEntry]:
    key = _key(path)
    return [entry for entry in entries if _key(entry.path) != key]


def dump(entries: list[RecentEntry]) -> str:
    return json.dumps([{"path": entry.path, "kind": entry.kind, "opened": entry.opened, "count": entry.count}
                       for entry in entries])


def _read(raw) -> RecentEntry | None:
    if not isinstance(raw, dict):
        return None
    path, kind, opened, count = raw.get("path"), raw.get("kind"), raw.get("opened"), raw.get("count")
    if not isinstance(path, str) or not path or kind not in KINDS:
        return None
    if not isinstance(opened, (int, float)) or isinstance(opened, bool):
        return None
    if not isinstance(count, int) or isinstance(count, bool):
        count = None
    return RecentEntry(path, kind, float(opened), count)


def parse(text: str) -> list[RecentEntry]:
    """What `dump` wrote. Anything unreadable is an empty list, and an
    unreadable entry is skipped: a damaged setting never stops the window
    opening."""
    try:
        raw = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    return [entry for entry in map(_read, raw) if entry is not None]


def title(entry: RecentEntry) -> str:
    """What the row is called: an episode's file name without its extension;
    a folder's own name, or for a season folder the nearest ancestor that
    names the show (its own name when none does)."""
    if entry.kind == EPISODE:
        return os.path.splitext(os.path.basename(entry.path))[0] or entry.path
    parts = [part for part in re.split(r"[/\\]", entry.path) if part]
    for part in reversed(parts):
        if not _GENERIC.search(part) and not _DRIVE.match(part):
            return part
    return parts[-1] if parts else entry.path


def location(entry: RecentEntry) -> str:
    """The path under the title: the folder itself, or the folder an episode
    is in (its file name is already the title)."""
    return os.path.dirname(entry.path) if entry.kind == EPISODE else entry.path


def _count(number: int, unit: str) -> str:
    return f"{number} {unit}" if number == 1 else f"{number} {unit}s"


def time_ago(seconds: float) -> str:
    """"just now", "5 min ago", "3 h ago", "yesterday", "4 days ago",
    "2 weeks ago", "3 months ago", "1 year ago"."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} h ago"
    days = hours // 24
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if days // 7 < 5:
        return f"{_count(days // 7, 'week')} ago"
    if days // 30 < 12:
        return f"{_count(days // 30, 'month')} ago"
    return f"{_count(max(1, days // 365), 'year')} ago"


def detail(entry: RecentEntry, now: float) -> str:
    """The row's right-hand text: "12 episodes · 2 h ago" for a folder, when
    it was opened for an episode."""
    when = time_ago(now - entry.opened)
    if entry.kind != FOLDER or entry.count is None:
        return when
    episodes = "no episodes" if entry.count == 0 else _count(entry.count, "episode")
    return f"{episodes} · {when}"
