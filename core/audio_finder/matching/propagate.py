from __future__ import annotations

from core.audio_finder.matching.alignment import build_offset_histogram, find_histogram_peaks
from core.audio_finder.matching.segments import extract_runs, find_contiguous_runs


def propagate_segment(
    canonical_hashes: list[tuple[int, int]],
    file_hashes: list[tuple[int, int]],
    min_count: int,
    delta_tolerance: int,
    threshold: float,
    gap_frames: int = 40,
    dt_min: int = 10,
    dt_max: int = 65,
) -> tuple[int, int, float] | None:
    if not canonical_hashes or not file_hashes:
        return None

    canon_lookup: dict[int, list[int]] = {}
    for h, t in canonical_hashes:
        canon_lookup.setdefault(h, []).append(t)

    deltas = build_offset_histogram(canon_lookup, file_hashes)
    peaks = find_histogram_peaks(deltas, max(min_count // 2, 5))

    if not peaks:
        return None

    best_delta, best_count = peaks[0]

    matching_frames = extract_runs(canon_lookup, file_hashes, best_delta, delta_tolerance)
    if not matching_frames:
        return None

    total_anchors = len({t for _, t in canonical_hashes})
    score = len(matching_frames) / total_anchors

    if score < threshold:
        return None

    find_gap = gap_frames * 5
    runs = find_contiguous_runs(matching_frames, find_gap)
    if not runs:
        return None

    best_run = max(runs, key=lambda r: r[1] - r[0])

    start_file = max(0, best_run[0] - dt_min + best_delta)
    end_file = best_run[1] + dt_max + best_delta

    return start_file, end_file, score
