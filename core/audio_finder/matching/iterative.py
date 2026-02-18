from __future__ import annotations

from collections.abc import Callable

import numpy as np

from core.audio_finder.config import AnalysisConfig
from core.audio_finder.db import repo
from core.audio_finder.db.connect import get_conn
from core.audio_finder.matching.alignment import build_offset_histogram, find_histogram_peaks
from core.audio_finder.matching.propagate import propagate_segment
from core.audio_finder.matching.segments import (
    extract_runs,
    filter_runs_by_length,
    find_contiguous_runs,
)


class MaskSet:
    def __init__(self) -> None:
        self._masks: dict[int, np.ndarray] = {}

    def init_file(self, file_id: int, num_frames: int) -> None:
        if file_id not in self._masks:
            self._masks[file_id] = np.zeros(num_frames, dtype=bool)

    def mask_range(self, file_id: int, start: int, end: int) -> None:
        if file_id in self._masks:
            s = max(0, start)
            e = min(len(self._masks[file_id]), end + 1)
            self._masks[file_id][s:e] = True

    def is_masked(self, file_id: int, frame: int) -> bool:
        if file_id not in self._masks:
            return False
        if frame < 0 or frame >= len(self._masks[file_id]):
            return False
        return bool(self._masks[file_id][frame])

    def unmasked_count(self, file_id: int) -> int:
        if file_id not in self._masks:
            return 0
        return int((~self._masks[file_id]).sum())

    def filter_hashes(
        self, file_id: int, hashes: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        if file_id not in self._masks:
            return hashes
        mask = self._masks[file_id]
        return [(h, t) for h, t in hashes if t < len(mask) and not mask[t]]


def _estimate_frames(duration_sec: float, cfg: AnalysisConfig) -> int:
    return int(duration_sec * cfg.dsp.frames_per_sec) + 1


def run_analysis(
    cfg: AnalysisConfig,
    tag_id: int,
    profile_id: int,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> int:
    def _log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    with get_conn(cfg.db_path) as conn:
        files = repo.get_files_for_tag(conn, tag_id)
        if len(files) < 2:
            return 0

        repo.delete_analysis_for_tag(conn, tag_id, profile_id)
        conn.commit()

    min_files_for_stop_words = 10
    stop_hashes: set[int] = set()

    if len(files) >= min_files_for_stop_words:
        with get_conn(cfg.db_path) as conn:
            repo.compute_hash_stats(conn, tag_id, profile_id)
            conn.commit()

        with get_conn(cfg.db_path) as conn:
            stop_hashes = repo.load_stop_hashes(
                conn, tag_id, profile_id, cfg.match.stop_word_file_ratio
            )

    _log(f"  Stop-word hashes: {len(stop_hashes):,}")

    file_hashes: dict[int, list[tuple[int, int]]] = {}
    with get_conn(cfg.db_path) as conn:
        for f in files:
            fh = repo.load_fingerprints_for_file(conn, tag_id, profile_id, f["id"])
            fh = [(h, t) for h, t in fh if h not in stop_hashes]
            file_hashes[f["id"]] = fh

    mask = MaskSet()
    for f in files:
        mask.init_file(f["id"], _estimate_frames(f["duration_sec"] or 0, cfg))

    exhausted: set[int] = set()
    segment_count = 0
    max_iterations = 50

    for iteration in range(max_iterations):
        best_pivot = None
        best_unmasked = 0
        for f in files:
            fid = f["id"]
            if fid in exhausted:
                continue
            um = mask.unmasked_count(fid)
            if um > best_unmasked:
                best_unmasked = um
                best_pivot = fid

        if best_pivot is None:
            break

        pivot_hashes = mask.filter_hashes(best_pivot, file_hashes[best_pivot])
        if len(pivot_hashes) < cfg.match.min_count:
            exhausted.add(best_pivot)
            continue

        pivot_lookup: dict[int, list[int]] = {}
        for h, t in pivot_hashes:
            pivot_lookup.setdefault(h, []).append(t)

        best_match = None

        for f in files:
            cid = f["id"]
            if cid == best_pivot:
                continue

            cand_hashes = mask.filter_hashes(cid, file_hashes[cid])
            if not cand_hashes:
                continue

            cand_hits = [(h, t) for h, t in cand_hashes if h in pivot_lookup]
            if not cand_hits:
                continue

            deltas = build_offset_histogram(pivot_lookup, cand_hits)
            peaks = find_histogram_peaks(deltas, cfg.match.min_count)

            for delta, count in peaks:
                frames = extract_runs(
                    pivot_lookup, cand_hits, delta, cfg.match.delta_tolerance
                )
                discovery_gap = cfg.match.gap_frames * 3
                runs = find_contiguous_runs(frames, discovery_gap)
                qualified = filter_runs_by_length(
                    runs, cfg.dsp, cfg.match,
                    start_pad=cfg.hash.dt_min,
                    end_pad=cfg.hash.dt_max,
                )

                for start, end, run_count in qualified:
                    if best_match is None or run_count > best_match[4]:
                        best_match = (cid, delta, start, end, run_count)

        if best_match is None:
            exhausted.add(best_pivot)
            continue

        cand_id, delta, seg_start, seg_end, _ = best_match
        seg_duration = (seg_end - seg_start) / cfg.dsp.frames_per_sec

        _log(
            f"  Segment {segment_count + 1}: "
            f"frames {seg_start}-{seg_end} ({seg_duration:.1f}s) "
            f"pivot={best_pivot}, matched with file {cand_id}"
        )

        canonical = [
            (h, t - seg_start) for h, t in pivot_hashes
            if seg_start <= t <= seg_end
        ]

        with get_conn(cfg.db_path) as conn:
            seg_id = repo.insert_segment(
                conn, tag_id, profile_id, best_pivot,
                seg_start, seg_end, seg_duration,
            )
            repo.bulk_insert_segment_fingerprints(conn, seg_id, canonical)

            for f in files:
                fid = f["id"]
                fh = mask.filter_hashes(fid, file_hashes[fid])

                result = propagate_segment(
                    canonical, fh,
                    min_count=cfg.match.min_count,
                    delta_tolerance=cfg.match.delta_tolerance,
                    threshold=cfg.match.propagation_threshold,
                    gap_frames=cfg.match.gap_frames,
                    dt_min=cfg.hash.dt_min,
                    dt_max=cfg.hash.dt_max,
                )

                if result is not None:
                    m_start, m_end, score = result
                    m_start_sec = m_start / cfg.dsp.frames_per_sec
                    m_end_sec = m_end / cfg.dsp.frames_per_sec
                    m_end_sec = min(m_end_sec, m_start_sec + seg_duration)
                    m_end = min(m_end, m_start + seg_end - seg_start)

                    repo.insert_segment_match(
                        conn, seg_id, fid,
                        m_start, m_end,
                        m_start_sec, m_end_sec,
                        score,
                    )

                    mask.mask_range(fid, m_start, m_end)

            conn.commit()

        segment_count += 1

    _log(f"  Analysis complete: {segment_count} segment(s) found.")
    return segment_count
