"""Offset matching between fingerprint sets, vectorised.

Replaces the dict-of-lists joins and Python loops of the original
``core/audio_finder/matching/`` with sorted arrays and ``searchsorted``,
producing identical values. Every function returns plain Python ints/floats
wherever the original did, because those values end up in float arithmetic
whose bits must not change.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d

_EMPTY = np.empty(0, dtype=np.int64)


class HashIndex:
    """A fingerprint set (hash, time) sorted by hash for equal-hash joins."""

    __slots__ = ("hashes", "times", "_anchor_count")

    def __init__(self, hashes: np.ndarray, times: np.ndarray):
        hashes = np.asarray(hashes, dtype=np.int64)
        times = np.asarray(times, dtype=np.int64)
        order = np.argsort(hashes, kind="stable")
        self.hashes = hashes[order]
        self.times = times[order]
        self._anchor_count: int | None = None

    def __len__(self) -> int:
        return len(self.hashes)

    @property
    def anchor_count(self) -> int:
        """Number of distinct times (the original's ``total_anchors``)."""
        if self._anchor_count is None:
            self._anchor_count = int(np.unique(self.times).size)
        return self._anchor_count


def offset_pairs(
    index: HashIndex, hashes: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Every (t_index, t_query - t_index) pair whose hashes are equal.

    The same multiset the original built by looking each query hash up in a
    dict of index times; order differs, which is irrelevant because every
    consumer is a histogram or a set.
    """
    if len(index) == 0 or len(hashes) == 0:
        return _EMPTY, _EMPTY
    lo = np.searchsorted(index.hashes, hashes, side="left")
    hi = np.searchsorted(index.hashes, hashes, side="right")
    counts = hi - lo
    total = int(counts.sum())
    if total == 0:
        return _EMPTY, _EMPTY
    first = np.cumsum(counts) - counts
    t_index = index.times[np.repeat(lo - first, counts) + np.arange(total)]
    deltas = np.repeat(np.asarray(times, dtype=np.int64), counts) - t_index
    return t_index, deltas


def histogram_peaks(deltas: np.ndarray, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """(peak deltas, peak counts), strongest first, ties in ascending delta.

    Equal to the original loop that kept bin i when counts[i] >= min_count
    and counts[i] == counts[max(0, i-2):min(len, i+3)].max(), then did a
    stable ``sort(key=-count)``: counts are non-negative, so padding the
    truncated window with zeros (mode='constant', cval=0) cannot change its
    maximum, and argsort(kind='stable') is the same stable sort.
    """
    if len(deltas) == 0:
        return _EMPTY, _EMPTY
    min_val = int(deltas.min())
    counts = np.bincount(deltas - min_val)
    window_max = maximum_filter1d(counts, size=5, mode="constant", cval=0)
    idx = np.flatnonzero((counts >= min_count) & (counts == window_max))
    idx = idx[np.argsort(-counts[idx], kind="stable")]
    return idx + min_val, counts[idx]


class DeltaLookup:
    """Index times grouped by delta, for repeated |delta - d| <= tol queries."""

    __slots__ = ("_deltas", "_times")

    def __init__(self, t_index: np.ndarray, deltas: np.ndarray):
        order = np.argsort(deltas, kind="stable")
        self._deltas = deltas[order]
        self._times = t_index[order]

    def frames(self, delta: int, tolerance: int) -> np.ndarray:
        """Sorted distinct index times whose delta is within tolerance
        (the original ``extract_runs``)."""
        lo = np.searchsorted(self._deltas, delta - tolerance, side="left")
        hi = np.searchsorted(self._deltas, delta + tolerance, side="right")
        if hi <= lo:
            return _EMPTY
        return np.unique(self._times[lo:hi])


def contiguous_runs(frames: np.ndarray, gap_frames: int) -> list[tuple[int, int]]:
    """Split sorted frames wherever consecutive frames are > gap apart."""
    n = len(frames)
    if n == 0:
        return []
    breaks = np.flatnonzero(np.diff(frames) > gap_frames)
    starts = frames[np.r_[0, breaks + 1]]
    ends = frames[np.r_[breaks, n - 1]]
    return list(zip(starts.tolist(), ends.tolist()))


def filter_runs_by_length(
    runs: list[tuple[int, int]], min_frames: int, start_pad: int, end_pad: int
) -> list[tuple[int, int, int]]:
    result = []
    for start, end in runs:
        length = end - start
        if length >= min_frames:
            result.append((max(0, start - start_pad), end + end_pad, length))
    return result


def propagate_segment(
    canonical: HashIndex,
    file_hashes: np.ndarray,
    file_times: np.ndarray,
    *,
    min_count: int,
    delta_tolerance: int,
    threshold: float,
    gap_frames: int = 40,
    dt_min: int = 10,
    dt_max: int = 65,
) -> tuple[int, int, float] | None:
    """Locate a canonical segment inside one file: (start, end, score) in
    file frames, or None."""
    if len(canonical) == 0 or len(file_hashes) == 0:
        return None
    t_canon, deltas = offset_pairs(canonical, file_hashes, file_times)
    peak_deltas, _ = histogram_peaks(deltas, max(min_count // 2, 5))
    if peak_deltas.size == 0:
        return None
    best_delta = int(peak_deltas[0])

    matching = np.unique(t_canon[np.abs(deltas - best_delta) <= delta_tolerance])
    if matching.size == 0:
        return None
    score = int(matching.size) / canonical.anchor_count
    if score < threshold:
        return None

    runs = contiguous_runs(matching, gap_frames * 5)
    if not runs:
        return None
    best_run = max(runs, key=lambda r: r[1] - r[0])
    start_file = max(0, best_run[0] - dt_min + best_delta)
    end_file = best_run[1] + dt_max + best_delta
    return start_file, end_file, score


def stop_word_hashes(fingerprints: list[np.ndarray], max_file_ratio: float) -> np.ndarray:
    """Sorted hashes present in more than int(n_files * ratio) files."""
    total = len(fingerprints)
    if total == 0:
        return _EMPTY
    per_file = [np.unique(np.asarray(fp, dtype=np.int64).reshape(-1, 2)[:, 0]) for fp in fingerprints]
    values, file_counts = np.unique(np.concatenate(per_file), return_counts=True)
    threshold = int(total * max_file_ratio)
    return values[file_counts > threshold]
