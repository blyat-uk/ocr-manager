"""Audio fingerprint analysis worker for automatic time range detection."""
import logging
import os
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.audio_finder.audio.extract import compute_sha256
from core.audio_finder.audio.pipeline import fingerprint_file
from core.audio_finder.config import AnalysisConfig, MatchConfig
from core.audio_finder.db import repo
from core.audio_finder.db.connect import get_conn
from core.audio_finder.matching.iterative import run_analysis

logger = logging.getLogger(__name__)

# Threshold: if a segment starts within this many seconds of 0:00, treat as intro
INTRO_THRESHOLD_SEC = 30.0
# Threshold: if a segment ends within this many seconds of file end, treat as outro
OUTRO_THRESHOLD_SEC = 30.0

# Default minimum repeating-segment length (seconds)
DEFAULT_MIN_SEGMENT_SEC = 30.0


class AudioAnalysisWorker(QObject):
    """Worker that fingerprints video files and discovers repeating segments.

    Runs in a QThread. Three phases:
    1. Ingest: fingerprint each video file (SHA256 check for caching)
    2. Analyze: run iterative segment discovery
    3. Compute: determine per-file time_start/time_end from segment matches

    Signals:
        phase_changed(str): Current phase name
        file_progress(str, int, int): filename, current (1-based), total
        analysis_progress(str): Status messages from iterative analysis
        error(str): Error message
        finished(dict): {filename: (time_start_mmss | None, time_end_mmss | None)}
    """

    phase_changed = pyqtSignal(str)
    file_progress = pyqtSignal(str, int, int)
    analysis_progress = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, project_path: str, video_files: list[str],
                 min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC):
        super().__init__()
        self._project_path = project_path
        self._video_files = video_files
        self._min_segment_sec = min_segment_sec
        self._cancel_requested = False
        self._thread: QThread | None = None

    def start(self):
        """Start the worker in a new thread."""
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.start()

    def cancel(self):
        """Request cancellation."""
        self._cancel_requested = True

    def cleanup(self):
        """Stop the thread and clean up."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None

    def _run(self):
        """Execute the three-phase analysis pipeline."""
        try:
            db_path = os.path.join(self._project_path, ".audio_fingerprints.db")
            cfg = AnalysisConfig(
                match=MatchConfig(min_length_sec=self._min_segment_sec),
                db_path=db_path,
            )
            tag_name = "default"

            # Phase 1: Ingest
            self.phase_changed.emit("Fingerprinting")
            tag_id, profile_id = self._ingest(cfg, tag_name)

            if self._cancel_requested:
                self.finished.emit({})
                return

            # Phase 2: Analyze
            self.phase_changed.emit("Analyzing")
            self._analyze(cfg, tag_id, profile_id)

            if self._cancel_requested:
                self.finished.emit({})
                return

            # Phase 3: Compute time ranges
            self.phase_changed.emit("Computing time ranges")
            results = self._compute_time_ranges(cfg, tag_id, profile_id)

            self.finished.emit(results)

        except Exception as e:
            logger.exception("Audio analysis failed")
            self.error.emit(str(e))

    def _ingest(self, cfg: AnalysisConfig, tag_name: str) -> tuple[int, int]:
        """Fingerprint video files, skipping those already in DB.

        Returns (tag_id, profile_id).
        """
        with get_conn(cfg.db_path) as conn:
            tag_id = repo.get_or_create_tag(conn, tag_name)
            profile_id = repo.get_or_create_profile(conn, cfg.fingerprint_params_dict())
            conn.commit()

        total = len(self._video_files)
        for i, filename in enumerate(self._video_files):
            if self._cancel_requested:
                break

            self.file_progress.emit(filename, i + 1, total)
            full_path = os.path.join(self._project_path, filename)

            sha256 = compute_sha256(full_path)

            with get_conn(cfg.db_path) as conn:
                existing = repo.file_exists(conn, tag_id, sha256)

            if existing is not None:
                continue  # Already fingerprinted

            hashes, duration_sec = fingerprint_file(full_path, cfg)

            with get_conn(cfg.db_path) as conn:
                file_id = repo.insert_media_file(
                    conn, tag_id, filename, sha256, duration_sec
                )
                repo.bulk_insert_fingerprints(
                    conn, tag_id, profile_id, file_id, hashes
                )
                conn.commit()

        return tag_id, profile_id

    def _analyze(self, cfg: AnalysisConfig, tag_id: int, profile_id: int):
        """Run iterative segment discovery."""

        def on_progress(msg: str) -> None:
            self.analysis_progress.emit(msg)

        run_analysis(cfg, tag_id, profile_id, on_progress=on_progress)

    def _compute_time_ranges(
        self, cfg: AnalysisConfig, tag_id: int, profile_id: int
    ) -> dict[str, tuple[str | None, str | None]]:
        """Determine per-file content windows from segment matches.

        Returns {filename: (time_start_mmss | None, time_end_mmss | None)}.
        """
        results: dict[str, tuple[str | None, str | None]] = {}

        with get_conn(cfg.db_path) as conn:
            segments = repo.get_segments_for_tag(conn, tag_id, profile_id)
            files = repo.get_files_for_tag(conn, tag_id)

        # Build file info lookup: file_id -> {path (basename), duration}
        file_info: dict[int, dict] = {}
        for f in files:
            file_info[f["id"]] = {
                "path": f["path"],
                "duration_sec": f["duration_sec"] or 0.0,
            }

        # Collect all segment matches per file_id
        file_matches: dict[int, list[dict]] = {}
        for seg in segments:
            for match in seg["matches"]:
                fid = match["file_id"]
                file_matches.setdefault(fid, []).append(match)

        # For each file, sort matches by start_sec and determine intro/outro
        for fid, matches in file_matches.items():
            info = file_info.get(fid)
            if not info:
                continue

            filename = info["path"]
            duration = info["duration_sec"]
            matches.sort(key=lambda m: m["start_sec"])

            time_start: str | None = None
            time_end: str | None = None

            # Check for intro: first match starts near beginning
            first = matches[0]
            if first["start_sec"] <= INTRO_THRESHOLD_SEC:
                end_sec = first["end_sec"]
                minutes = int(end_sec) // 60
                secs = int(end_sec) % 60
                time_start = f"{minutes:02d}:{secs:02d}"

            # Check for outro: last match ends near file end
            last = matches[-1]
            if duration > 0 and (duration - last["end_sec"]) <= OUTRO_THRESHOLD_SEC:
                start_sec = last["start_sec"]
                minutes = int(start_sec) // 60
                secs = int(start_sec) % 60
                time_end = f"{minutes:02d}:{secs:02d}"

            if time_start is not None or time_end is not None:
                results[filename] = (time_start, time_end)

        return results
