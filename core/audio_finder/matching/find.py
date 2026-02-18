from __future__ import annotations

from core.audio_finder.config import AnalysisConfig
from core.audio_finder.matching.alignment import build_offset_histogram, find_histogram_peaks
from core.audio_finder.matching.segments import extract_runs, find_contiguous_runs


def find_sample_in_file(
    sample_hashes: list[tuple[int, int]],
    file_hashes: list[tuple[int, int]],
    cfg: AnalysisConfig,
) -> tuple[int, int, float] | None:
    if not sample_hashes or not file_hashes:
        return None

    sample_lookup: dict[int, list[int]] = {}
    for h, t in sample_hashes:
        sample_lookup.setdefault(h, []).append(t)

    file_hits = [(h, t) for h, t in file_hashes if h in sample_lookup]
    if not file_hits:
        return None

    deltas = build_offset_histogram(sample_lookup, file_hits)
    peaks = find_histogram_peaks(deltas, min_count=max(cfg.match.min_count // 2, 5))

    if not peaks:
        return None

    best_delta, best_count = peaks[0]

    raw_score = best_count / len(sample_hashes)
    if raw_score < cfg.match.propagation_threshold:
        return None

    matching_sample_frames = extract_runs(
        sample_lookup, file_hits, best_delta, cfg.match.delta_tolerance
    )
    if not matching_sample_frames:
        return None

    total_anchors = len({t for _, t in sample_hashes})
    score = len(matching_sample_frames) / total_anchors

    find_gap = cfg.match.gap_frames * 5
    runs = find_contiguous_runs(matching_sample_frames, find_gap)
    if not runs:
        return None

    best_run = max(runs, key=lambda r: r[1] - r[0])

    start_file = max(0, best_run[0] - cfg.hash.dt_min) + best_delta
    end_file = (best_run[1] + cfg.hash.dt_max) + best_delta

    return start_file, end_file, score
