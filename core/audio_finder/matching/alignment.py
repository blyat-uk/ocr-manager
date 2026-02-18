from __future__ import annotations

import numpy as np


def build_offset_histogram(
    pivot_hashes: dict[int, list[int]],
    candidate_hits: list[tuple[int, int]],
) -> np.ndarray:
    deltas = []
    for h, t_cand in candidate_hits:
        if h in pivot_hashes:
            for t_piv in pivot_hashes[h]:
                deltas.append(t_cand - t_piv)

    if not deltas:
        return np.array([], dtype=np.int64)

    return np.array(deltas, dtype=np.int64)


def find_histogram_peaks(
    deltas: np.ndarray, min_count: int
) -> list[tuple[int, int]]:
    if len(deltas) == 0:
        return []

    min_val = int(deltas.min())
    shifted = deltas - min_val
    counts = np.bincount(shifted)

    peaks = []
    for i in range(len(counts)):
        if counts[i] >= min_count:
            lo = max(0, i - 2)
            hi = min(len(counts), i + 3)
            if counts[i] == counts[lo:hi].max():
                peaks.append((i + min_val, int(counts[i])))

    peaks.sort(key=lambda x: -x[1])
    return peaks
