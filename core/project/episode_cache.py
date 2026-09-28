"""Bookkeeping of the episode cache root: `<root>/index.json`, the same-folder
crop seed, the remembered run speed, and pruning.

Episodes keep their state under the cache root (core.project.layout), one
directory per video key. Three things need to know about more than one of
them, and `index.json` is where they look:

    folders  {absolute video directory: [{"key", "name", "saved_at"}]},
             updated on every episode save (record_episode). A key belongs
             to one directory: an episode that moved leaves its old one.
    speed    {"video_per_wall", "measured_at"}: the last finished run's
             speed, video seconds OCR'd per wall second (record_speed), for
             the Preparing screen's estimate and the Working screen's ETA.

Same-folder seed (sibling_seed)
    A fansubber OCRs one episode a week from the same show, and its subtitles
    sit where last week's did. When an episode opens with no settings of its
    own, the newest sibling from its directory whose crop the crop pool would
    trust (core.jobs.autopilot.crop_consensus's rule: IMPORTED or MANUAL, or
    DETECTED with no crop flag, with a known media height) gives its
    (y_frac, h_frac). AutoPilot takes it as crop consensus, not as a hint, so
    an episode that disagrees is not flagged for differing alone.

Pruning (prune)
    On each episode open: every `videos/<key>/` untouched (its newest mtime,
    the directory's and every file's in it) for more than max_age_days goes;
    then, while the total is over max_bytes, the oldest go first. The key
    being opened is never removed, and only directories named like a key are
    ever touched. Removed keys leave the index too.

Never raises into the app
    A cache is a convenience. An index that cannot be read (missing, not
    JSON, the wrong shape) reads as empty, and a write or delete that fails
    is logged and skipped. Writes are atomic: tmp in the same directory,
    fsync, os.replace.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import time
from pathlib import Path

from core.project.layout import EPISODE_CONFIG_FILENAME, EPISODES_DIRNAME
from core.project.model import Source
from core.project.store import _atomic_write_text, from_json

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.json"
INDEX_VERSION = 1
MIN_SPEED_WALL_SECONDS = 5.0      # a shorter run says more about start-up than about speed
MAX_AGE_DAYS = 90
MAX_BYTES = 2 * 1024 ** 3
_KEY_NAME = re.compile(r"^[0-9a-f]{32}$")     # a directory core.project.layout.video_key named
_SEED_SOURCES = frozenset({Source.IMPORTED, Source.MANUAL})   # plus unflagged DETECTED crops


def index_path(root: str) -> Path:
    """`<root>/index.json`."""
    return Path(root) / INDEX_FILENAME


def _read_index(root: str) -> dict:
    """The index, normalised: {"folders": {dir: [record, ...]}, "speed": dict
    or None}. Anything unreadable or of the wrong shape reads as empty; a
    folder or record of the wrong shape is dropped on its own."""
    try:
        data = json.loads(index_path(root).read_bytes())
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as exc:          # JSONDecodeError and UnicodeDecodeError included
        logger.warning("Unreadable episode index %s (%s): starting from an empty one", index_path(root), exc)
        data = {}
    if not isinstance(data, dict):
        data = {}
    folders = {}
    raw_folders = data.get("folders")
    if isinstance(raw_folders, dict):
        for directory, records in raw_folders.items():
            if not isinstance(records, list):
                continue
            kept = [r for r in records if _valid_record(r)]
            if kept:
                folders[directory] = kept
    speed = data.get("speed")
    if not (isinstance(speed, dict) and _positive(speed.get("video_per_wall"))):
        speed = None
    return {"folders": folders, "speed": speed}


def _valid_record(record) -> bool:
    return (isinstance(record, dict) and isinstance(record.get("key"), str)
            and isinstance(record.get("name"), str)
            and isinstance(record.get("saved_at"), (int, float)) and not isinstance(record.get("saved_at"), bool))


def _positive(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _write_index(root: str, index: dict) -> bool:
    """Write `index` atomically; False (logged) when it could not be."""
    data = {"version": INDEX_VERSION, "folders": index["folders"]}
    if index["speed"] is not None:
        data["speed"] = index["speed"]
    try:
        Path(root).mkdir(parents=True, exist_ok=True)
        _atomic_write_text(index_path(root), json.dumps(data, ensure_ascii=False, indent=2))
    except OSError as exc:
        logger.warning("Could not write the episode index %s (%s)", index_path(root), exc)
        return False
    return True


def record_episode(root: str, video_path: str, key: str, now: float | None = None) -> None:
    """File the episode `key` (the video at `video_path`) under its
    directory, saved at `now`. Called on every episode save."""
    now = time.time() if now is None else float(now)
    video_path = os.path.abspath(video_path)
    directory, name = os.path.dirname(video_path), os.path.basename(video_path)
    index = _read_index(root)
    folders = {}
    for other, records in index["folders"].items():
        kept = [r for r in records if r["key"] != key]
        if kept:
            folders[other] = kept
    folders.setdefault(directory, []).append({"key": key, "name": name, "saved_at": now})
    index["folders"] = folders
    _write_index(root, index)


def sibling_seed(root: str, video_path: str, key: str) -> tuple[str, tuple[float, float]] | None:
    """(sibling name, (y_frac, h_frac)) of the newest other episode recorded
    in `video_path`'s directory whose saved crop the crop pool would trust;
    None when there is none (see the module docstring)."""
    directory = os.path.dirname(os.path.abspath(video_path))
    records = [r for r in _read_index(root)["folders"].get(directory, []) if r["key"] != key]
    for record in sorted(records, key=lambda r: r["saved_at"], reverse=True):
        seed = _saved_crop_fractions(root, record)
        if seed is not None:
            return record["name"], seed
    return None


def _saved_crop_fractions(root: str, record: dict) -> tuple[float, float] | None:
    """The trusted crop of the episode `record` names, as fractions of its
    height, read from its settings; None when it has none or they cannot be
    read."""
    path = Path(root) / EPISODES_DIRNAME / record["key"] / EPISODE_CONFIG_FILENAME
    try:
        data = json.loads(path.read_bytes())
        if not isinstance(data, dict) or data.get("version") != 2:
            return None
        project = from_json(data, os.path.dirname(path))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        logger.debug("Episode settings %s give no seed (%s)", path, exc)
        return None
    # Its own entry first: settings hold one file, but a renamed video's
    # entry keeps its old name until the episode is opened again.
    entries = sorted(project.files.values(), key=lambda entry: entry.name != record["name"])
    for entry in entries:
        crop, height = entry.crop, entry.media.height
        if crop is None or not _positive(height):
            continue
        if crop.source in _SEED_SOURCES or (crop.source == Source.DETECTED and not entry.flags.get("crop")):
            return crop.y / height, crop.height / height
    return None


def record_speed(root: str, video_seconds: float, wall_seconds: float, now: float | None = None) -> None:
    """Remember a finished run's speed: `video_seconds` of video OCR'd in
    `wall_seconds`. A run shorter than MIN_SPEED_WALL_SECONDS, or numbers
    that are not positive and finite, are ignored."""
    if not (_positive(video_seconds) and _positive(wall_seconds)) or wall_seconds < MIN_SPEED_WALL_SECONDS:
        return
    now = time.time() if now is None else float(now)
    index = _read_index(root)
    index["speed"] = {"video_per_wall": float(video_seconds) / float(wall_seconds), "measured_at": now}
    _write_index(root, index)


def last_speed(root: str) -> float | None:
    """The remembered speed, video seconds per wall second; None when no run
    has been measured."""
    speed = _read_index(root)["speed"]
    return None if speed is None else float(speed["video_per_wall"])


def prune(root: str, keep_key: str, now: float | None = None,
          max_age_days: float = MAX_AGE_DAYS, max_bytes: int = MAX_BYTES) -> list[str]:
    """Delete stale episode directories (see the module docstring); the keys
    removed, in the order they went."""
    now = time.time() if now is None else float(now)
    videos = Path(root) / EPISODES_DIRNAME
    try:
        candidates = [item for item in os.scandir(videos) if _KEY_NAME.match(item.name) and item.is_dir()]
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("Could not list the episode cache %s (%s)", videos, exc)
        return []
    entries = []                                  # (last touched, key, size)
    for item in candidates:
        touched, size = _usage(item.path)
        entries.append((touched, item.name, size))
    entries.sort()                                # oldest first
    total = sum(size for _, _, size in entries)
    removed: list[str] = []

    def remove(key: str, size: int) -> None:
        nonlocal total
        try:
            shutil.rmtree(videos / key)
        except OSError as exc:
            logger.warning("Could not remove the episode cache %s (%s)", videos / key, exc)
            return
        removed.append(key)
        total -= size

    cutoff = now - max_age_days * 24 * 3600.0
    for touched, key, size in entries:
        if key != keep_key and touched < cutoff:
            remove(key, size)
    for touched, key, size in entries:
        if total <= max_bytes:
            break
        if key != keep_key and key not in removed:
            remove(key, size)

    if removed:
        gone = set(removed)
        index = _read_index(root)
        folders = {}
        for directory, records in index["folders"].items():
            kept = [r for r in records if r["key"] not in gone]
            if kept:
                folders[directory] = kept
        if folders != index["folders"]:
            index["folders"] = folders
            _write_index(root, index)
    return removed


def _usage(path: str) -> tuple[float, int]:
    """(newest mtime of `path` and everything in it, total bytes of its
    files). Anything that cannot be stat'd counts as nothing."""
    try:
        newest = os.stat(path).st_mtime
    except OSError:
        newest = 0.0
    size = 0
    for directory, subdirs, files in os.walk(path):
        for names, is_file in ((subdirs, False), (files, True)):
            for name in names:
                try:
                    stat = os.stat(os.path.join(directory, name), follow_symlinks=False)
                except OSError:
                    continue
                newest = max(newest, stat.st_mtime)
                if is_file:
                    size += stat.st_size
    return newest, size
