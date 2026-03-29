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

# Default minimum repeating-segment length (seconds)
DEFAULT_MIN_SEGMENT_SEC = 30.0

# Minimum gap duration (seconds) to be considered a "keep" range
MIN_GAP_SEC = 5.0


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
        finished(dict): {filename: [(start_mmss | None, end_mmss | None), ...]}
    """

    phase_changed = pyqtSignal(str)
    file_progress = pyqtSignal(str, int, int)
    analysis_progress = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, project_path: str, video_files: list[str],
                 min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC,
                 merge_repeating_silences: bool = False):
        super().__init__()
        self._project_path = project_path
        self._video_files = video_files
        self._min_segment_sec = min_segment_sec
        self._merge_repeating_silences = merge_repeating_silences
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

    @staticmethod
    def _secs_to_mmss(seconds: float) -> str:
        """Convert seconds to MM:SS string."""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def _merge_silence_gaps(
        file_skip_blocks: dict[int, list[tuple[float, float]]],
        position_tolerance_sec: float = 5.0,
        max_gap_duration_sec: float = 30.0,
        min_file_ratio: float = 0.5,
    ) -> dict[int, list[tuple[float, float]]]:
        """Bridge silence gaps between skip blocks when they appear consistently across files.

        For each file, examines gaps between consecutive skip blocks. If a gap appears
        at a similar time position in more than min_file_ratio of files, the adjacent
        skip blocks are merged (bridging the silence gap).
        """
        # Collect all gaps: (file_id, gap_index, gap_start, gap_end, midpoint)
        all_gaps: list[tuple[int, int, float, float, float]] = []
        for fid, blocks in file_skip_blocks.items():
            for i in range(len(blocks) - 1):
                gap_start = blocks[i][1]
                gap_end = blocks[i + 1][0]
                gap_duration = gap_end - gap_start
                if gap_duration <= max_gap_duration_sec:
                    midpoint = (gap_start + gap_end) / 2.0
                    all_gaps.append((fid, i, gap_start, gap_end, midpoint))

        if not all_gaps:
            return file_skip_blocks

        num_files = len(file_skip_blocks)
        min_file_count = max(2, int(num_files * min_file_ratio))

        # Cluster gaps by midpoint position across files
        all_gaps.sort(key=lambda g: g[4])
        clusters: list[list[tuple[int, int, float, float, float]]] = []
        current_cluster: list[tuple[int, int, float, float, float]] = [all_gaps[0]]

        for gap in all_gaps[1:]:
            if gap[4] - current_cluster[0][4] <= position_tolerance_sec * 2:
                cluster_fids = {g[0] for g in current_cluster}
                if gap[0] not in cluster_fids:
                    current_cluster.append(gap)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [gap]
            else:
                clusters.append(current_cluster)
                current_cluster = [gap]
        clusters.append(current_cluster)

        # Identify gaps that appear in enough files
        gaps_to_bridge: set[tuple[int, int]] = set()
        for cluster in clusters:
            unique_files = {g[0] for g in cluster}
            if len(unique_files) >= min_file_count:
                for fid, gap_idx, _, _, _ in cluster:
                    gaps_to_bridge.add((fid, gap_idx))

        if not gaps_to_bridge:
            return file_skip_blocks

        # Merge skip blocks across bridged gaps
        result: dict[int, list[tuple[float, float]]] = {}
        for fid, blocks in file_skip_blocks.items():
            new_blocks: list[tuple[float, float]] = []
            i = 0
            while i < len(blocks):
                start, end = blocks[i]
                while i < len(blocks) - 1 and (fid, i) in gaps_to_bridge:
                    end = max(end, blocks[i + 1][1])
                    i += 1
                new_blocks.append((start, end))
                i += 1
            result[fid] = new_blocks

        return result

    def _compute_time_ranges(
        self, cfg: AnalysisConfig, tag_id: int, profile_id: int
    ) -> dict[str, list[tuple[str | None, str | None]]]:
        """Determine per-file keep ranges by inverting detected repeating segments.

        All detected segments (intros, outros, mid-episode bumpers, etc.) are
        merged into "skip" blocks. The gaps between them become "keep" ranges.

        Returns {filename: [(start_mmss | None, end_mmss | None), ...]}.
        """
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

        # Phase A: Build per-file merged skip blocks
        file_skip_blocks: dict[int, list[tuple[float, float]]] = {}
        for fid, matches in file_matches.items():
            if fid not in file_info:
                continue
            matches.sort(key=lambda m: m["start_sec"])
            merged: list[tuple[float, float]] = []
            for m in matches:
                s, e = m["start_sec"], m["end_sec"]
                if merged and s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            file_skip_blocks[fid] = merged

        # Phase B: Optionally bridge silence gaps across files
        if self._merge_repeating_silences and len(file_skip_blocks) >= 2:
            file_skip_blocks = self._merge_silence_gaps(file_skip_blocks)

        # Phase C: Compute keep ranges from skip blocks
        results: dict[str, list[tuple[str | None, str | None]]] = {}
        for fid, merged in file_skip_blocks.items():
            info = file_info.get(fid)
            if not info:
                continue

            filename = info["path"]
            duration = info["duration_sec"]

            keep_ranges: list[tuple[str | None, str | None]] = []
            cursor = 0.0

            for skip_start, skip_end in merged:
                gap = skip_start - cursor
                if gap >= MIN_GAP_SEC:
                    start_str = self._secs_to_mmss(cursor) if cursor > 0 else None
                    end_str = self._secs_to_mmss(skip_start)
                    keep_ranges.append((start_str, end_str))
                cursor = skip_end

            # Gap after last skip block to end of file
            if duration > 0 and (duration - cursor) >= MIN_GAP_SEC:
                start_str = self._secs_to_mmss(cursor) if cursor > 0 else None
                keep_ranges.append((start_str, None))

            if keep_ranges:
                results[filename] = keep_ranges

        return results
