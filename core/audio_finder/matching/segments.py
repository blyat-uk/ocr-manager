from __future__ import annotations

from core.audio_finder.config import DSPConfig, MatchConfig


def extract_runs(
    pivot_hashes: dict[int, list[int]],
    candidate_hits: list[tuple[int, int]],
    delta: int,
    tolerance: int,
) -> list[int]:
    matching_frames = set()
    for h, t_cand in candidate_hits:
        if h in pivot_hashes:
            for t_piv in pivot_hashes[h]:
                if abs((t_cand - t_piv) - delta) <= tolerance:
                    matching_frames.add(t_piv)

    return sorted(matching_frames)


def find_contiguous_runs(
    frames: list[int], gap_frames: int
) -> list[tuple[int, int]]:
    if not frames:
        return []

    runs = []
    start = frames[0]
    prev = frames[0]

    for f in frames[1:]:
        if f - prev > gap_frames:
            runs.append((start, prev))
            start = f
        prev = f

    runs.append((start, prev))
    return runs


def filter_runs_by_length(
    runs: list[tuple[int, int]],
    dsp_cfg: DSPConfig,
    match_cfg: MatchConfig,
    start_pad: int | None = None,
    end_pad: int | None = None,
) -> list[tuple[int, int, int]]:
    min_frames = int(match_cfg.min_length_sec * dsp_cfg.frames_per_sec)
    s_pad = start_pad if start_pad is not None else match_cfg.pad_frames
    e_pad = end_pad if end_pad is not None else match_cfg.pad_frames
    result = []

    for start, end in runs:
        length = end - start
        if length >= min_frames:
            padded_start = max(0, start - s_pad)
            padded_end = end + e_pad
            result.append((padded_start, padded_end, length))

    return result
