"""Equivalence tests for the vectorised time-range pipeline (core/detect/ranges).

The requirement for core/detect/ranges is IDENTICAL output to the original
SQLite/Python-loop implementation (core/audio_analysis.py +
core/audio_finder/). The oracle for every test below is a direct
transcription of that original code into this file -- NOT the new code, and
NOT an import of core/audio_finder (which a later task deletes). The
transcriptions are deliberately slow and literal; the only liberty taken is
replacing SQLite reads/writes with the equivalent in-memory lists, and each
such replacement is commented with the SQL it stands in for.

Why the vectorised forms are exact rather than approximate (each is pinned by
a property test here, and each test has been seen failing against a mutated
implementation):

* Per-chunk top-k peak selection returns a SET that the original re-sorts by
  (t, f), so the order in which it is found is irrelevant -- EXCEPT when the
  k-th and (k+1)-th largest amplitudes in a chunk are equal: then the set
  depends on the unstable np.argsort's tie-break. The new code counts those
  boundary ties and, for exactly those chunks, calls the original
  expression on the original index order, so equivalence never rests on how
  np.argsort happens to break ties.
* Because peak times are sorted and the loop skips dt < dt_min, breaks on
  dt > dt_max and stops after `fanout` pairs, it pairs peak i with exactly
  the contiguous slice [lo, min(lo + fanout, hi)) where lo/hi come from
  np.searchsorted (lo is additionally floored at i + 1, which only matters
  for dt_min <= 0).
* The truncated 5-wide window max over a histogram equals
  maximum_filter1d(size=5, mode='constant', cval=0) because counts are
  non-negative, and argsort(kind='stable') reproduces the stable
  `list.sort(key=-count)`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading
import time
import wave
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pytest
from scipy.ndimage import maximum_filter

from core.detect.ranges import fingerprint as fp
from core.detect.ranges import matching as mt
from core.detect.ranges import pipeline as pl
from core.detect.ranges.config import (
    DSPConfig,
    HashConfig,
    MatchConfig,
    PeakConfig,
    RangesConfig,
)


# ===========================================================================
# ORACLE: transcription of core/audio_finder + core/audio_analysis.py
# ===========================================================================

def _ref_pack_hash(f1, f2, dt):
    return (f1 << 20) | (f2 << 10) | dt


def _ref_extract_audio(path, sample_rate=11025):
    # core/audio_finder/audio/extract.py::extract_audio, single-stream path
    # (the synthetic media here has exactly one audio stream).
    cmd = ["ffmpeg", "-i", path]
    cmd += [
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",
        "-af", "highpass=f=80,lowpass=f=5000",
        "-loglevel", "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    samples = np.frombuffer(result.stdout, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def _ref_compute_spectrogram(samples, cfg):
    n_fft = cfg.n_fft
    hop = cfg.hop_length
    if len(samples) < n_fft:
        return np.empty((cfg.num_bins, 0), dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    pad_len = n_fft - (len(samples) % hop)
    if pad_len > 0:
        samples = np.concatenate([samples, np.zeros(pad_len, dtype=np.float32)])
    num_frames = (len(samples) - n_fft) // hop + 1
    if num_frames <= 0:
        return np.empty((cfg.num_bins, 0), dtype=np.float32)
    shape = (num_frames, n_fft)
    strides = (samples.strides[0] * hop, samples.strides[0])
    frames = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    magnitude = np.abs(spectrum).T.astype(np.float32)
    return np.log1p(magnitude)


def _ref_top_k_keep(chunk_ids, amplitudes, max_per_chunk):
    # The per-chunk loop from core/audio_finder/audio/peaks.py::find_peaks.
    keep = np.zeros(len(chunk_ids), dtype=bool)
    num_chunks = chunk_ids.max() + 1
    for c in range(num_chunks):
        mask = chunk_ids == c
        if mask.sum() <= max_per_chunk:
            keep[mask] = True
        else:
            indices = np.where(mask)[0]
            top_k = indices[np.argsort(-amplitudes[indices])[:max_per_chunk]]
            keep[top_k] = True
    return keep


def _ref_find_peaks(spectrogram, dsp_cfg, peak_cfg):
    if spectrogram.size == 0:
        return []
    neighborhood = (peak_cfg.freq_neighborhood, peak_cfg.time_neighborhood)
    local_max = maximum_filter(spectrogram, size=neighborhood)
    is_peak = spectrogram == local_max
    is_peak &= spectrogram > peak_cfg.noise_floor
    freq_bins, time_frames = np.where(is_peak)
    if len(freq_bins) == 0:
        return []
    amplitudes = spectrogram[freq_bins, time_frames]
    frames_per_sec = dsp_cfg.frames_per_sec
    max_per_chunk = peak_cfg.peaks_per_sec
    chunk_ids = (time_frames / frames_per_sec).astype(int)
    keep = _ref_top_k_keep(chunk_ids, amplitudes, max_per_chunk)
    freq_bins = freq_bins[keep]
    time_frames = time_frames[keep]
    order = np.lexsort((freq_bins, time_frames))
    return list(zip(freq_bins[order].tolist(), time_frames[order].tolist()))


def _ref_generate_fingerprints(peaks, cfg):
    result = []
    n = len(peaks)
    for i in range(n):
        f1, t1 = peaks[i]
        paired = 0
        for j in range(i + 1, n):
            if paired >= cfg.fanout:
                break
            f2, t2 = peaks[j]
            dt = t2 - t1
            if dt < cfg.dt_min:
                continue
            if dt > cfg.dt_max:
                break
            f1_c = min(f1, (1 << cfg.f1_bits) - 1)
            f2_c = min(f2, (1 << cfg.f2_bits) - 1)
            dt_c = min(dt, (1 << cfg.dt_bits) - 1)
            h = _ref_pack_hash(f1_c, f2_c, dt_c)
            result.append((h, t1))
            paired += 1
    return result


def _ref_fingerprint_file(path, cfg):
    samples = _ref_extract_audio(path, sample_rate=cfg.dsp.sample_rate)
    duration_sec = len(samples) / cfg.dsp.sample_rate
    spectrogram = _ref_compute_spectrogram(samples, cfg.dsp)
    peaks = _ref_find_peaks(spectrogram, cfg.dsp, cfg.peak)
    hashes = _ref_generate_fingerprints(peaks, cfg.hash)
    return hashes, duration_sec


def _ref_build_offset_histogram(pivot_hashes, candidate_hits):
    deltas = []
    for h, t_cand in candidate_hits:
        if h in pivot_hashes:
            for t_piv in pivot_hashes[h]:
                deltas.append(t_cand - t_piv)
    if not deltas:
        return np.array([], dtype=np.int64)
    return np.array(deltas, dtype=np.int64)


def _ref_find_histogram_peaks(deltas, min_count):
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


def _ref_extract_runs(pivot_hashes, candidate_hits, delta, tolerance):
    matching_frames = set()
    for h, t_cand in candidate_hits:
        if h in pivot_hashes:
            for t_piv in pivot_hashes[h]:
                if abs((t_cand - t_piv) - delta) <= tolerance:
                    matching_frames.add(t_piv)
    return sorted(matching_frames)


def _ref_find_contiguous_runs(frames, gap_frames):
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


def _ref_filter_runs_by_length(runs, dsp_cfg, match_cfg, start_pad=None, end_pad=None):
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


def _ref_propagate_segment(canonical_hashes, file_hashes, min_count, delta_tolerance,
                           threshold, gap_frames=40, dt_min=10, dt_max=65):
    if not canonical_hashes or not file_hashes:
        return None
    canon_lookup = {}
    for h, t in canonical_hashes:
        canon_lookup.setdefault(h, []).append(t)
    deltas = _ref_build_offset_histogram(canon_lookup, file_hashes)
    peaks = _ref_find_histogram_peaks(deltas, max(min_count // 2, 5))
    if not peaks:
        return None
    best_delta, best_count = peaks[0]
    matching_frames = _ref_extract_runs(canon_lookup, file_hashes, best_delta, delta_tolerance)
    if not matching_frames:
        return None
    total_anchors = len({t for _, t in canonical_hashes})
    score = len(matching_frames) / total_anchors
    if score < threshold:
        return None
    find_gap = gap_frames * 5
    runs = _ref_find_contiguous_runs(matching_frames, find_gap)
    if not runs:
        return None
    best_run = max(runs, key=lambda r: r[1] - r[0])
    start_file = max(0, best_run[0] - dt_min + best_delta)
    end_file = best_run[1] + dt_max + best_delta
    return start_file, end_file, score


class _RefMaskSet:
    def __init__(self):
        self._masks = {}

    def init_file(self, file_id, num_frames):
        if file_id not in self._masks:
            self._masks[file_id] = np.zeros(num_frames, dtype=bool)

    def mask_range(self, file_id, start, end):
        if file_id in self._masks:
            s = max(0, start)
            e = min(len(self._masks[file_id]), end + 1)
            self._masks[file_id][s:e] = True

    def unmasked_count(self, file_id):
        if file_id not in self._masks:
            return 0
        return int((~self._masks[file_id]).sum())

    def filter_hashes(self, file_id, hashes):
        if file_id not in self._masks:
            return hashes
        mask = self._masks[file_id]
        return [(h, t) for h, t in hashes if t < len(mask) and not mask[t]]


def _ref_run_analysis(fingerprints, durations, cfg):
    """core/audio_finder/matching/iterative.py::run_analysis against a FRESH
    database: media_files ids are 1..n in ingest order.

    Returns (segment_rows, match_rows, log) where
      segment_rows: [(seg_id, canonical_file_id, start_frame, end_frame, duration_sec)]
      match_rows:   [(seg_id, file_id, start_frame, end_frame, start_sec, end_sec, score)]
    exactly as INSERTed.
    """
    log = []
    segment_rows = []
    match_rows = []
    # SELECT ... FROM media_files WHERE tag_id = ? ORDER BY id
    files = [{"id": i + 1, "duration_sec": d} for i, d in enumerate(durations)]
    stored = {i + 1: list(fh) for i, fh in enumerate(fingerprints)}
    if len(files) < 2:
        return segment_rows, match_rows, log

    min_files_for_stop_words = 10
    stop_hashes = set()
    if len(files) >= min_files_for_stop_words:
        # compute_hash_stats: SELECT hash, COUNT(DISTINCT file_id) ... GROUP BY hash
        file_count = {}
        for fid, fh in stored.items():
            for h in {h for h, _ in fh}:
                file_count[h] = file_count.get(h, 0) + 1
        # load_stop_hashes: total = COUNT(*) media_files; file_count > int(total * ratio)
        threshold = int(len(files) * cfg.match.stop_word_file_ratio)
        stop_hashes = {h for h, c in file_count.items() if c > threshold}

    log.append(f"  Stop-word hashes: {len(stop_hashes):,}")

    file_hashes = {}
    for f in files:
        fh = stored[f["id"]]
        fh = [(h, t) for h, t in fh if h not in stop_hashes]
        file_hashes[f["id"]] = fh

    mask = _RefMaskSet()
    for f in files:
        mask.init_file(f["id"], int((f["duration_sec"] or 0) * cfg.dsp.frames_per_sec) + 1)

    exhausted = set()
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

        pivot_lookup = {}
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
            deltas = _ref_build_offset_histogram(pivot_lookup, cand_hits)
            peaks = _ref_find_histogram_peaks(deltas, cfg.match.min_count)
            for delta, count in peaks:
                frames = _ref_extract_runs(pivot_lookup, cand_hits, delta, cfg.match.delta_tolerance)
                discovery_gap = cfg.match.gap_frames * 3
                runs = _ref_find_contiguous_runs(frames, discovery_gap)
                qualified = _ref_filter_runs_by_length(
                    runs, cfg.dsp, cfg.match,
                    start_pad=cfg.hash.dt_min, end_pad=cfg.hash.dt_max,
                )
                for start, end, run_count in qualified:
                    if best_match is None or run_count > best_match[4]:
                        best_match = (cid, delta, start, end, run_count)

        if best_match is None:
            exhausted.add(best_pivot)
            continue

        cand_id, delta, seg_start, seg_end, _ = best_match
        seg_duration = (seg_end - seg_start) / cfg.dsp.frames_per_sec
        log.append(
            f"  Segment {segment_count + 1}: "
            f"frames {seg_start}-{seg_end} ({seg_duration:.1f}s) "
            f"pivot={best_pivot}, matched with file {cand_id}"
        )
        canonical = [(h, t - seg_start) for h, t in pivot_hashes if seg_start <= t <= seg_end]
        seg_id = len(segment_rows) + 1  # INSERT INTO segments
        segment_rows.append((seg_id, best_pivot, seg_start, seg_end, seg_duration))

        for f in files:
            fid = f["id"]
            fh = mask.filter_hashes(fid, file_hashes[fid])
            result = _ref_propagate_segment(
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
                # INSERT INTO segment_matches
                match_rows.append((seg_id, fid, m_start, m_end, m_start_sec, m_end_sec, score))
                mask.mask_range(fid, m_start, m_end)
        segment_count += 1

    log.append(f"  Analysis complete: {segment_count} segment(s) found.")
    return segment_rows, match_rows, log


def _ref_secs_to_mmss(seconds):
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def _ref_merge_silence_gaps(file_skip_blocks, position_tolerance_sec=5.0,
                            max_gap_duration_sec=30.0, min_file_ratio=0.5):
    all_gaps = []
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
    all_gaps.sort(key=lambda g: g[4])
    clusters = []
    current_cluster = [all_gaps[0]]
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
    gaps_to_bridge = set()
    for cluster in clusters:
        unique_files = {g[0] for g in cluster}
        if len(unique_files) >= min_file_count:
            for fid, gap_idx, _, _, _ in cluster:
                gaps_to_bridge.add((fid, gap_idx))
    if not gaps_to_bridge:
        return file_skip_blocks
    result = {}
    for fid, blocks in file_skip_blocks.items():
        new_blocks = []
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


def _ref_compute_time_ranges(segment_rows, match_rows, names, durations, merge_repeating_silences):
    """core/audio_analysis.py::_compute_time_ranges over the rows the
    reference analysis INSERTed."""
    MIN_GAP_SEC = 5.0
    # get_segments_for_tag: segments ORDER BY id; each segment's matches
    # "ORDER BY m.score DESC". SQLite's sorter keeps insertion order for equal
    # scores (verified against the real repo module on 300 randomised
    # databases, and on the real corpus by comparing dict order), which is
    # what Python's stable sort reproduces.
    segments = []
    for seg_id, *_ in segment_rows:
        rows = [m for m in match_rows if m[0] == seg_id]
        rows.sort(key=lambda m: -m[6])
        segments.append({"matches": [
            {"file_id": m[1], "start_sec": m[4], "end_sec": m[5]} for m in rows
        ]})
    files = [{"id": i + 1, "path": name, "duration_sec": d}
             for i, (name, d) in enumerate(zip(names, durations))]

    file_info = {}
    for f in files:
        file_info[f["id"]] = {"path": f["path"], "duration_sec": f["duration_sec"] or 0.0}
    file_matches = {}
    for seg in segments:
        for match in seg["matches"]:
            fid = match["file_id"]
            file_matches.setdefault(fid, []).append(match)
    file_skip_blocks = {}
    for fid, matches in file_matches.items():
        if fid not in file_info:
            continue
        matches.sort(key=lambda m: m["start_sec"])
        merged = []
        for m in matches:
            s, e = m["start_sec"], m["end_sec"]
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        file_skip_blocks[fid] = merged
    if merge_repeating_silences and len(file_skip_blocks) >= 2:
        file_skip_blocks = _ref_merge_silence_gaps(file_skip_blocks)
    results = {}
    for fid, merged in file_skip_blocks.items():
        info = file_info.get(fid)
        if not info:
            continue
        filename = info["path"]
        duration = info["duration_sec"]
        keep_ranges = []
        cursor = 0.0
        for skip_start, skip_end in merged:
            gap = skip_start - cursor
            if gap >= MIN_GAP_SEC:
                start_str = _ref_secs_to_mmss(cursor) if cursor > 0 else None
                end_str = _ref_secs_to_mmss(skip_start)
                keep_ranges.append((start_str, end_str))
            cursor = skip_end
        if duration > 0 and (duration - cursor) >= MIN_GAP_SEC:
            start_str = _ref_secs_to_mmss(cursor) if cursor > 0 else None
            keep_ranges.append((start_str, None))
        if keep_ranges:
            results[filename] = keep_ranges
    return results


# ===========================================================================
# Helpers
# ===========================================================================

def _segments_as_rows(segments):
    """New Segment objects -> the reference's (segment_rows, match_rows)."""
    seg_rows, match_rows = [], []
    for k, seg in enumerate(segments, start=1):
        seg_rows.append((k, seg.pivot + 1, seg.start_frame, seg.end_frame, seg.duration_sec))
        for m in seg.matches:
            match_rows.append((k, m.file + 1, m.start_frame, m.end_frame,
                               m.start_sec, m.end_sec, m.score))
    return seg_rows, match_rows


def _exact_rows(rows):
    """Make float comparison bit-exact (repr round-trips doubles) and assert
    every element is a plain Python scalar, as SQLite would have returned."""
    out = []
    for row in rows:
        for v in row:
            assert type(v) in (int, float), f"{v!r} is {type(v)}, expected a Python scalar"
        out.append(tuple(v.hex() if isinstance(v, float) else v for v in row))
    return out


def _random_peaks(rng, n, max_t, max_f=1100):
    """Sorted-by-(t, f) peak list with repeated times and distinct (f, t)."""
    times = np.sort(rng.integers(0, max_t, size=n))
    peaks = set()
    for t in times.tolist():
        peaks.add((int(rng.integers(0, max_f)), t))
    return sorted(peaks, key=lambda p: (p[1], p[0]))


# ===========================================================================
# Fingerprint pairing
# ===========================================================================

def test_vectorised_pairing_matches_the_nested_loop():
    rng = np.random.default_rng(1234)
    checked_pairs = 0
    clamped_dt = 0
    for trial in range(400):
        if trial % 7 == 0:
            # Sparse peaks with dt beyond 10 bits, so the dt clamp binds.
            dt_min = int(rng.integers(500, 1100))
            cfg = HashConfig(fanout=int(rng.integers(1, 9)), dt_min=dt_min,
                             dt_max=dt_min + int(rng.integers(0, 2000)))
            n = int(rng.integers(0, 100))
            max_t = int(rng.integers(2000, 20000))
        else:
            cfg = HashConfig(fanout=int(rng.integers(1, 9)), dt_min=int(rng.integers(0, 25)),
                             dt_max=int(rng.integers(0, 90)))
            n = int(rng.integers(0, 400))
            max_t = int(rng.integers(1, 2000)) if trial % 5 else int(rng.integers(1, 30))
        peaks = _random_peaks(rng, n, max_t)
        expected = _ref_generate_fingerprints(peaks, cfg)

        freqs = np.array([p[0] for p in peaks], dtype=np.int64)
        times = np.array([p[1] for p in peaks], dtype=np.int64)
        got = fp.pair_peaks(freqs, times, cfg)

        assert got.dtype == np.int64 and got.ndim == 2 and got.shape[1] == 2
        # Same pairs in the same order: (hash, t_anchor) list, i ascending then j ascending.
        assert got.tolist() == [list(p) for p in expected], f"trial {trial} cfg={cfg}"
        checked_pairs += len(expected)
        clamped_dt += sum(1 for h, _ in expected if h & 0x3FF == 0x3FF)
    assert checked_pairs > 10_000  # the generator is not vacuous
    assert clamped_dt > 100


def test_pairing_on_default_config_uses_the_contiguous_searchsorted_slice():
    # Dense peaks: many candidates inside [dt_min, dt_max], so the fanout cap,
    # the dt_min skip and the dt_max break all bind at once.
    rng = np.random.default_rng(7)
    cfg = HashConfig()
    for _ in range(50):
        peaks = _random_peaks(rng, 1500, 600, max_f=513)
        freqs = np.array([p[0] for p in peaks], dtype=np.int64)
        times = np.array([p[1] for p in peaks], dtype=np.int64)
        assert fp.pair_peaks(freqs, times, cfg).tolist() == \
            [list(p) for p in _ref_generate_fingerprints(peaks, cfg)]


# ===========================================================================
# Per-chunk top-k peak selection
# ===========================================================================

def test_top_k_selection_is_the_same_set_and_has_no_boundary_ties_on_continuous_amplitudes():
    rng = np.random.default_rng(99)
    total_ties = 0
    selected = 0
    for trial in range(300):
        n = int(rng.integers(1, 3000))
        chunk_ids = rng.integers(0, int(rng.integers(1, 80)), size=n)
        amplitudes = rng.random(n, dtype=np.float32) * 10 + 0.1
        k = int(rng.integers(1, 12))
        expected = _ref_top_k_keep(chunk_ids, amplitudes, k)
        keep, ties = fp.select_top_k_per_chunk(chunk_ids, amplitudes, k)
        total_ties += ties
        assert np.array_equal(keep, expected), f"trial {trial}"
        selected += int(keep.sum())
    assert total_ties == 0
    assert selected > 10_000


def test_top_k_boundary_ties_are_counted_and_resolved_exactly_like_the_original():
    # Quantised amplitudes in large chunks: the k-th and (k+1)-th largest
    # values are frequently equal, and np.argsort's default (unstable) sort
    # decides which one the original kept. The new code must count those
    # ties and reproduce the original's choice, not the stable order.
    rng = np.random.default_rng(5)
    total_ties = 0
    differs_from_stable = 0
    for trial in range(300):
        n = int(rng.integers(50, 4000))
        chunk_ids = rng.integers(0, int(rng.integers(1, 20)), size=n)
        amplitudes = (rng.integers(1, 6, size=n) / 2).astype(np.float32)
        k = int(rng.integers(1, 8))
        expected = _ref_top_k_keep(chunk_ids, amplitudes, k)
        keep, ties = fp.select_top_k_per_chunk(chunk_ids, amplitudes, k)
        total_ties += ties
        assert np.array_equal(keep, expected), f"trial {trial}"

        order = np.lexsort((-amplitudes, chunk_ids))
        rank = np.empty(n, dtype=np.int64)
        sc = chunk_ids[order]
        starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
        rank[order] = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
        if not np.array_equal(rank < k, expected):
            differs_from_stable += 1
    assert total_ties > 0
    # Proves this test can tell "stable order" from "the original's order".
    assert differs_from_stable > 0


def _random_spectrogram(rng, frames, quantised):
    spec = rng.random((513, frames), dtype=np.float32) * 3
    spec[:, rng.random(frames) < 0.05] = 0.05  # quiet frames below the noise floor
    if quantised:
        spec = (np.round(spec * 4) / 4).astype(np.float32)
    return spec


@pytest.mark.parametrize("quantised", [False, True])
def test_find_peaks_matches_the_original(quantised):
    rng = np.random.default_rng(2024 + quantised)
    dsp, peak = DSPConfig(), PeakConfig()
    ties = 0
    for _ in range(6):
        spec = _random_spectrogram(rng, int(rng.integers(30, 400)), quantised)
        expected = _ref_find_peaks(spec, dsp, peak)
        got = fp.find_peaks(spec, dsp, peak)
        ties += got.boundary_ties
        assert list(zip(got.freqs.tolist(), got.times.tolist())) == expected
    assert (ties > 0) == quantised


def test_find_peaks_handles_empty_and_silent_spectrograms():
    dsp, peak = DSPConfig(), PeakConfig()
    empty = fp.find_peaks(np.empty((513, 0), dtype=np.float32), dsp, peak)
    assert len(empty.freqs) == 0 and len(empty.times) == 0
    silent = fp.find_peaks(np.zeros((513, 50), dtype=np.float32), dsp, peak)
    assert len(silent.freqs) == 0


# ===========================================================================
# Histogram peaks, offset join, runs, propagation
# ===========================================================================

def _random_deltas(rng):
    n = int(rng.integers(1, 4000))
    span = int(rng.integers(1, 3000))
    base = rng.integers(-span, span + 1, size=n)
    spikes = rng.choice(base, size=int(rng.integers(0, 40))) if n else base[:0]
    reps = rng.integers(1, 60, size=len(spikes))
    return np.concatenate([base, np.repeat(spikes, reps)]).astype(np.int64)


def test_histogram_peaks_match_the_truncated_window_loop():
    rng = np.random.default_rng(31337)
    found = 0
    for trial in range(400):
        deltas = _random_deltas(rng)
        min_count = int(rng.choice([1, 2, 3, 5, 10, 20]))
        expected = _ref_find_histogram_peaks(deltas, min_count)
        got_d, got_c = mt.histogram_peaks(deltas, min_count)
        assert list(zip(got_d.tolist(), got_c.tolist())) == expected, f"trial {trial}"
        found += len(expected)
    assert found > 1000
    assert mt.histogram_peaks(np.array([], dtype=np.int64), 5)[0].size == 0


def test_histogram_peaks_edges_and_equal_heights():
    # Peaks at both truncated edges, plateaus, and hundreds of equal-height
    # peaks (the case where an unstable sort would reorder them).
    cases = [
        np.array([0], dtype=np.int64),
        np.array([0, 0, 0, 1], dtype=np.int64),
        np.array([5, 5, 6, 6, 9, 9, 9], dtype=np.int64),
    ]
    rng = np.random.default_rng(3)
    heights = rng.permutation(np.repeat([7, 7, 7, 9, 12], 120))
    centres = np.arange(len(heights)) * 5 - 900
    cases.append(np.repeat(centres, heights).astype(np.int64))
    for deltas in cases:
        for min_count in (1, 2, 7):
            expected = _ref_find_histogram_peaks(deltas, min_count)
            got_d, got_c = mt.histogram_peaks(deltas, min_count)
            assert list(zip(got_d.tolist(), got_c.tolist())) == expected


def _random_hash_list(rng, n, hash_space, max_t):
    return [(int(h), int(t)) for h, t in zip(rng.integers(0, hash_space, n), rng.integers(0, max_t, n))]


def _arrays(pairs):
    a = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    return a[:, 0].copy(), a[:, 1].copy()


def test_offset_join_and_matching_frames_match_the_dict_loops():
    rng = np.random.default_rng(11)
    for trial in range(200):
        pivot = _random_hash_list(rng, int(rng.integers(0, 600)), int(rng.integers(1, 200)), 3000)
        cand = _random_hash_list(rng, int(rng.integers(0, 600)), int(rng.integers(1, 200)), 3000)
        lookup = {}
        for h, t in pivot:
            lookup.setdefault(h, []).append(t)
        hits = [(h, t) for h, t in cand if h in lookup]
        expected = _ref_build_offset_histogram(lookup, hits)

        index = mt.HashIndex(*_arrays(pivot))
        t_index, deltas = mt.offset_pairs(index, *_arrays(cand))
        assert sorted(deltas.tolist()) == sorted(expected.tolist())
        # The pairs themselves, not just the delta multiset.
        ref_pairs = sorted((t_c - t_p, t_p) for h, t_c in hits for t_p in lookup[h])
        assert sorted(zip(deltas.tolist(), t_index.tolist())) == ref_pairs

        if len(expected):
            lookup_frames = mt.DeltaLookup(t_index, deltas)
            for delta in rng.choice(expected, size=min(10, len(expected))).tolist():
                tol = int(rng.integers(0, 4))
                assert lookup_frames.frames(delta, tol).tolist() == \
                    _ref_extract_runs(lookup, hits, delta, tol)


def test_contiguous_runs_match():
    rng = np.random.default_rng(12)
    for _ in range(300):
        frames = sorted(set(rng.integers(0, 5000, int(rng.integers(0, 300))).tolist()))
        gap = int(rng.integers(0, 300))
        assert mt.contiguous_runs(np.array(frames, dtype=np.int64), gap) == \
            _ref_find_contiguous_runs(frames, gap)


def _planted(rng, template, offset, keep_prob, noise_n, hash_space, max_t):
    out = [(h, t + offset) for h, t in template if rng.random() < keep_prob]
    out += _random_hash_list(rng, noise_n, hash_space, max_t)
    return out


def test_propagate_segment_matches():
    rng = np.random.default_rng(13)
    accepted = 0
    for trial in range(150):
        template = [(int(h), int(t)) for h, t in zip(rng.integers(10_000, 10_400, 900),
                                                     rng.integers(0, 700, 900))]
        canonical = [p for p in template if rng.random() < 0.9]
        offset = int(rng.integers(-500, 2000))
        keep_prob = float(rng.choice([0.05, 0.3, 0.9]))
        file_hashes = _planted(rng, template, offset, keep_prob, 800, 10_400, 3000)
        kwargs = dict(min_count=int(rng.choice([5, 20, 40])), delta_tolerance=2,
                      threshold=0.15, gap_frames=int(rng.choice([5, 40])), dt_min=10, dt_max=65)
        expected = _ref_propagate_segment(canonical, file_hashes, **kwargs)
        got = mt.propagate_segment(mt.HashIndex(*_arrays(canonical)), *_arrays(file_hashes), **kwargs)
        if expected is None:
            assert got is None
        else:
            accepted += 1
            assert got is not None
            assert [type(v) for v in got] == [int, int, float]
            assert got[0] == expected[0] and got[1] == expected[1]
            assert got[2].hex() == expected[2].hex()
    assert accepted > 20

    # Two equally long runs far apart: the original keeps the FIRST longest.
    ties = 0
    for trial in range(60):
        length = int(rng.integers(20, 80))
        second = int(rng.integers(400, 900))
        canonical = [(int(h), t) for t in range(0, length + 1) for h in rng.integers(20_000, 30_000, 2)]
        canonical += [(int(h), t) for t in range(second, second + length + 1) for h in rng.integers(30_000, 40_000, 2)]
        offset = int(rng.integers(0, 3000))
        file_hashes = [(h, t + offset) for h, t in canonical] + _random_hash_list(rng, 300, 20_000, 5000)
        kwargs = dict(min_count=20, delta_tolerance=2, threshold=0.15, gap_frames=40, dt_min=10, dt_max=65)
        expected = _ref_propagate_segment(canonical, file_hashes, **kwargs)
        got = mt.propagate_segment(mt.HashIndex(*_arrays(canonical)), *_arrays(file_hashes), **kwargs)
        assert expected is not None and got is not None
        assert got[:2] == expected[:2] and got[2].hex() == expected[2].hex()
        ties += expected[0] == max(0, offset - 10)
    assert ties == 60

    # Tiny sets: the best histogram peak has 3-6 votes, right at the
    # max(min_count // 2, 5) floor.
    outcomes = {True: 0, False: 0}
    for trial in range(400):
        canonical = [(int(h), int(t)) for h, t in zip(rng.integers(0, 40, 8), rng.integers(0, 60, 8))]
        offset = int(rng.integers(0, 100))
        k = int(rng.integers(3, 8))
        file_hashes = [(h, t + offset) for h, t in canonical[:k]]
        file_hashes += _random_hash_list(rng, int(rng.integers(0, 15)), 40, 200)
        kwargs = dict(min_count=int(rng.choice([1, 8, 9, 10, 11])), delta_tolerance=int(rng.integers(0, 3)),
                      threshold=0.15, gap_frames=40, dt_min=10, dt_max=65)
        expected = _ref_propagate_segment(canonical, file_hashes, **kwargs)
        got = mt.propagate_segment(mt.HashIndex(*_arrays(canonical)), *_arrays(file_hashes), **kwargs)
        outcomes[expected is not None] += 1
        if expected is None:
            assert got is None, f"trial {trial}"
        else:
            assert got is not None and got[:2] == expected[:2] and got[2].hex() == expected[2].hex()
    assert outcomes[True] > 20 and outcomes[False] > 20


def test_stop_word_threshold_matches_count_distinct():
    rng = np.random.default_rng(14)
    for n_files in (10, 11, 12, 20, 40):
        files = []
        for i in range(n_files):
            pairs = _random_hash_list(rng, 300, 400, 1000)
            files.append(np.array(pairs, dtype=np.int64).reshape(-1, 2))
        counts = {}
        for arr in files:
            for h in set(arr[:, 0].tolist()):
                counts[h] = counts.get(h, 0) + 1
        threshold = int(n_files * 0.9)
        expected = sorted(h for h, c in counts.items() if c > threshold)
        assert mt.stop_word_hashes(files, 0.9).tolist() == expected


def test_mask_range_and_filter_reproduce_python_slice_semantics():
    rng = np.random.default_rng(15)
    for _ in range(300):
        n = int(rng.integers(1, 400))
        ref = _RefMaskSet()
        ref.init_file(1, n)
        mask = np.zeros(n, dtype=bool)
        for _ in range(int(rng.integers(1, 5))):
            start = int(rng.integers(-600, 600))
            end = int(rng.integers(-600, 600))  # negative ends slice from the back, as the original did
            ref.mask_range(1, start, end)
            pl.mask_range(mask, start, end)
        assert np.array_equal(mask, ref._masks[1])
        pairs = _random_hash_list(rng, 200, 50, n + 30)  # some t beyond the mask
        h, t = _arrays(pairs)
        fh, ft = pl.unmasked_hashes(h, t, mask)
        assert list(zip(fh.tolist(), ft.tolist())) == ref.filter_hashes(1, pairs)


# ===========================================================================
# Discovery loop + keep ranges on synthetic fingerprint sets
# ===========================================================================

def _synthetic_corpus(rng, n_files, equal_durations=False):
    """Fingerprint sets with a planted intro in every file, an outro in most,
    background noise, and hashes shared by ~9/10/all files to sit on both
    sides of the stop-word threshold."""
    intro = [(int(h), int(t)) for h, t in zip(rng.integers(50_000, 52_000, 1500), rng.integers(0, 300, 1500))]
    outro = [(int(h), int(t)) for h, t in zip(rng.integers(60_000, 62_000, 1200), rng.integers(0, 250, 1200))]
    common = [int(h) for h in rng.integers(90_000, 90_050, 30)]
    fingerprints, durations = [], []
    for i in range(n_files):
        frames = int(rng.integers(900, 1300))
        pairs = _random_hash_list(rng, 2500, 4000, frames)
        io = int(rng.integers(0, 120))
        pairs += [(h, t + io) for h, t in intro if rng.random() < 0.85]
        if rng.random() < 0.8:
            oo = frames - 260 - int(rng.integers(0, 60))
            pairs += [(h, t + oo) for h, t in outro if rng.random() < 0.85]
        for k, h in enumerate(common):
            if (i + k) % n_files < n_files - (k % 3):
                pairs.append((h, int(rng.integers(0, frames))))
        rng.shuffle(pairs)
        fingerprints.append(pairs)
        durations.append(frames / DSPConfig().frames_per_sec + float(rng.random()))
    if equal_durations:
        # Every mask the same length: the first pivot is chosen by the
        # original's "first file with the most unmasked frames" tie-break.
        durations = [1400 / DSPConfig().frames_per_sec] * n_files
    return fingerprints, durations


@pytest.mark.parametrize("n_files,seed,equal", [(2, 9, False), (3, 2, False), (4, 7, True),
                                                (5, 3, False), (10, 6, False), (11, 4, False)])
@pytest.mark.parametrize("merge", [False, True])
def test_discovery_and_keep_ranges_match_the_original(n_files, seed, equal, merge):
    rng = np.random.default_rng(seed)
    cfg = RangesConfig(match=MatchConfig(min_length_sec=8.0), merge_repeating_silences=merge)
    fingerprints, durations = _synthetic_corpus(rng, n_files, equal_durations=equal)
    names = [f"ep{i:02d}.mkv" for i in range(n_files)]

    ref_segs, ref_matches, ref_log = _ref_run_analysis(fingerprints, durations, cfg)
    ref_ranges = _ref_compute_time_ranges(ref_segs, ref_matches, names, durations, merge)

    log = []
    arrays = [np.array(f, dtype=np.int64).reshape(-1, 2) for f in fingerprints]
    segments = pl.discover_segments(arrays, durations, cfg, log=log.append)
    seg_rows, match_rows = _segments_as_rows(segments)
    ranges = pl.compute_keep_ranges(segments, names, durations, merge)

    assert log == ref_log
    assert _exact_rows(seg_rows) == _exact_rows(ref_segs)
    assert _exact_rows(match_rows) == _exact_rows(ref_matches)
    assert list(ranges.items()) == list(ref_ranges.items())  # dict ORDER too

    # Non-vacuous: segments were found, and the merge option really bridged
    # a gap in every one of these corpora.
    assert len(ref_segs) >= 2
    unmerged = _ref_compute_time_ranges(ref_segs, ref_matches, names, durations, False)
    assert unmerged
    if merge:
        assert ref_ranges != unmerged
    if n_files >= 10:
        assert ref_log[0] != "  Stop-word hashes: 0"


def test_fingerprints_on_a_files_last_frame_survive_masking_like_the_original():
    # The original sized each mask int(duration * fps) + 1 and dropped
    # fingerprints with t >= len(mask). Here every file's planted segment
    # ends exactly on frame int(duration * fps), so the discovered segment's
    # end frame depends on that last frame being kept.
    rng = np.random.default_rng(21)
    fps = DSPConfig().frames_per_sec
    cfg = RangesConfig(match=MatchConfig(min_length_sec=8.0))
    template = [(int(h), t) for t in range(0, 301) for h in rng.integers(70_000, 71_000, 3)]
    fingerprints, durations, lasts = [], [], []
    for _ in range(3):
        offset = int(rng.integers(200, 400))
        last = offset + 300
        pairs = _random_hash_list(rng, 1500, 4000, last) + [(h, t + offset) for h, t in template]
        fingerprints.append(pairs)
        durations.append((last + 0.5) / fps)
        lasts.append(last)
        assert int(durations[-1] * fps) == last

    ref_segs, ref_matches, ref_log = _ref_run_analysis(fingerprints, durations, cfg)
    log = []
    segments = pl.discover_segments([np.array(f, dtype=np.int64) for f in fingerprints],
                                    durations, cfg, log=log.append)
    seg_rows, match_rows = _segments_as_rows(segments)
    assert log == ref_log
    assert _exact_rows(seg_rows) == _exact_rows(ref_segs)
    assert _exact_rows(match_rows) == _exact_rows(ref_matches)
    _, pivot_id, _, end_frame, _ = ref_segs[0]
    assert end_frame == lasts[pivot_id - 1] + cfg.hash.dt_max  # the last frame mattered


def test_keep_ranges_match_on_adversarial_segments():
    # Ties in score (dict order), ties in start_sec, negative-length
    # intervals, zero durations, silence gaps close together.
    rng = np.random.default_rng(16)
    fps = DSPConfig().frames_per_sec
    exact_gap_trials = 0
    for trial in range(600):
        n = int(rng.integers(2, 8))
        names = [f"f{i}" for i in range(n)]
        # Half the trials live on a 2.5 s grid, so gaps of exactly
        # MIN_GAP_SEC (5.0) -- including the tail gap -- occur often.
        on_grid = trial % 2 == 1
        if on_grid:
            durations = [float(rng.choice([0.0, 30.0, 32.5, 35.0, 37.5, 40.0])) for _ in range(n)]
        else:
            durations = [float(rng.choice([0.0, 60.0, 300.0, 1450.5])) for _ in range(n)]
        segments = []
        for _ in range(int(rng.integers(0, 6))):
            matches = []
            for f in range(n):
                if rng.random() < 0.6:
                    if on_grid:
                        s_sec = 2.5 * int(rng.integers(0, 14))
                        e_sec = s_sec + 2.5 * int(rng.integers(-1, 5))
                        s, e = int(s_sec * fps), int(e_sec * fps)
                    else:
                        s = int(rng.integers(0, 30_000)) if rng.random() < 0.8 else int(rng.choice([0, 1000]))
                        e = s + int(rng.integers(-500, 3000))
                        s_sec, e_sec = s / fps, e / fps
                    matches.append(pl.SegmentMatch(
                        file=f, start_frame=s, end_frame=e,
                        start_sec=s_sec, end_sec=e_sec,
                        score=float(rng.choice([1.0, 0.5, 0.25, rng.random()])),
                    ))
            segments.append(pl.Segment(pivot=0, candidate=1, start_frame=0, end_frame=1,
                                       duration_sec=1 / fps, matches=tuple(matches)))
        merge = bool(rng.integers(0, 2))
        seg_rows, match_rows = _segments_as_rows(segments)
        expected = _ref_compute_time_ranges(seg_rows, match_rows, names, durations, merge)
        got = pl.compute_keep_ranges(segments, names, durations, merge)
        assert list(got.items()) == list(expected.items()), f"trial {trial}"
        if on_grid and any(mm.start_sec == 5.0 or d - mm.end_sec == 5.0
                           for seg in segments for mm in seg.matches for d in durations):
            exact_gap_trials += 1
    assert exact_gap_trials > 20


# ===========================================================================
# File identity and cache
# ===========================================================================

def _write_bytes(path, data, mtime_ns):
    path.write_bytes(data)
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_file_identity_sees_a_change_confined_to_the_last_megabyte(tmp_path):
    rng = np.random.default_rng(17)
    data = rng.integers(0, 256, size=12 * 1024 * 1024, dtype=np.uint8).tobytes()
    changed = bytearray(data)
    changed[-512 * 1024] ^= 0xFF  # inside the last MB, outside the first 4 MB
    a, b = tmp_path / "a.mkv", tmp_path / "b.mkv"
    mtime = 1_700_000_000_123_456_789
    _write_bytes(a, data, mtime)
    _write_bytes(b, bytes(changed), mtime)  # same size, same mtime
    assert pl.file_identity(str(a)) != pl.file_identity(str(b))


def test_file_identity_is_stable_and_sees_size_and_mtime(tmp_path):
    rng = np.random.default_rng(18)
    small = rng.integers(0, 256, size=3 * 1024 * 1024, dtype=np.uint8).tobytes()
    p = tmp_path / "small.mkv"
    _write_bytes(p, small, 1_600_000_000_000_000_000)
    first = pl.file_identity(str(p))
    assert all(pl.file_identity(str(p)) == first for _ in range(3))
    os.utime(p, ns=(1_600_000_000_000_000_001,) * 2)
    assert pl.file_identity(str(p)) != first
    q = tmp_path / "small2.mkv"
    _write_bytes(q, small + b"\0", 1_600_000_000_000_000_000)
    assert pl.file_identity(str(q)) != first


def _write_wav(path, channels_samples, rate=22050):
    data = np.stack(channels_samples, axis=1)
    data = np.clip(data, -1, 1)
    pcm = (data * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def _noise(rng, seconds, rate=22050):
    n = int(seconds * rate)
    x = rng.standard_normal(n)
    # Slowly varying envelope + a few tones so spectral peaks have structure.
    t = np.arange(n) / rate
    env = 0.25 + 0.2 * np.sin(2 * np.pi * t * rng.uniform(0.2, 1.5))
    tones = sum(0.2 * np.sin(2 * np.pi * rng.uniform(200, 3000) * t + rng.uniform(0, 6))
                for _ in range(3))
    return (0.3 * x * env + tones * np.sign(np.sin(2 * np.pi * t * rng.uniform(0.3, 2)))) * 0.5


@pytest.fixture(scope="module")
def synthetic_episodes(tmp_path_factory):
    """Three stereo WAV 'episodes': shared 14 s intro, unique 22-26 s body,
    shared 12 s outro in two of them."""
    d = tmp_path_factory.mktemp("episodes")
    rng = np.random.default_rng(19)
    intro_l, intro_r = _noise(rng, 14), _noise(rng, 14)
    outro_l, outro_r = _noise(rng, 12), _noise(rng, 12)
    paths = []
    for i in range(3):
        body = float(22 + 2 * i)
        left = [intro_l, _noise(rng, body)]
        right = [intro_r, _noise(rng, body)]
        if i != 1:
            left.append(outro_l)
            right.append(outro_r)
        p = d / f"ep{i + 1:02d}.wav"
        _write_wav(p, [np.concatenate(left), np.concatenate(right)])
        paths.append(p)
    return paths


_SYNTH_CFG = RangesConfig(match=MatchConfig(min_length_sec=8.0))


def _entries(paths):
    return [pl.FileEntry(name=p.name, path=str(p)) for p in paths]


def test_fingerprint_file_matches_the_original_on_decoded_audio(synthetic_episodes):
    for p in synthetic_episodes:
        ref_hashes, ref_duration = _ref_fingerprint_file(str(p), _SYNTH_CFG)
        hashes, duration = fp.fingerprint_file(str(p), _SYNTH_CFG)
        assert hashes.dtype == np.int64 and hashes.shape == (len(ref_hashes), 2)
        assert hashes.tolist() == [list(x) for x in ref_hashes]
        assert duration.hex() == ref_duration.hex()
        assert len(ref_hashes) > 500


def test_analyse_matches_the_original_end_to_end(synthetic_episodes, tmp_path):
    ref_fps, ref_durations = zip(*(_ref_fingerprint_file(str(p), _SYNTH_CFG) for p in synthetic_episodes))
    names = [p.name for p in synthetic_episodes]
    ref_segs, ref_matches, ref_log = _ref_run_analysis(ref_fps, ref_durations, _SYNTH_CFG)
    expected = _ref_compute_time_ranges(ref_segs, ref_matches, names, ref_durations, False)
    assert ref_segs and expected  # the synthetic corpus really has repeats

    events = []
    got = pl.analyse(_entries(synthetic_episodes), _SYNTH_CFG, progress=events.append,
                     cache_dir=str(tmp_path / "cache"), workers=2)
    assert list(got.items()) == list(expected.items())
    assert [e.message for e in events if e.kind == "log"] == ref_log
    assert [e.message for e in events if e.kind == "phase"] == \
        ["Fingerprinting", "Analyzing", "Computing time ranges"]
    file_events = [e for e in events if e.kind == "file"]
    assert sorted(e.current for e in file_events) == [1, 2, 3]
    assert all(e.total == 3 for e in file_events)


def test_cache_hits_skip_decoding_and_misses_on_param_or_file_change(synthetic_episodes, tmp_path, monkeypatch):
    import shutil
    work = tmp_path / "project"
    work.mkdir()
    copies = []
    for p in synthetic_episodes:
        shutil.copy2(p, work / p.name)
        copies.append(work / p.name)
    cache = tmp_path / "cache"
    cold = pl.analyse(_entries(copies), _SYNTH_CFG, cache_dir=str(cache), workers=1)
    assert len(list(cache.rglob("*.npz"))) == len(copies)

    calls = []
    real = fp.fingerprint_file

    def counting(path, cfg):
        calls.append(os.path.basename(path))
        return real(path, cfg)

    monkeypatch.setattr(fp, "fingerprint_file", counting)

    warm = pl.analyse(_entries(copies), _SYNTH_CFG, cache_dir=str(cache), workers=1)
    assert list(warm.items()) == list(cold.items())
    assert calls == []

    # Match-only parameter change: fingerprints stay cached.
    pl.analyse(_entries(copies), RangesConfig(match=MatchConfig(min_length_sec=9.0)),
               cache_dir=str(cache), workers=1)
    assert calls == []

    # Fingerprint parameter change: every file is recomputed.
    pl.analyse(_entries(copies), RangesConfig(peak=PeakConfig(peaks_per_sec=4),
                                              match=MatchConfig(min_length_sec=8.0)),
               cache_dir=str(cache), workers=1)
    assert sorted(calls) == sorted(p.name for p in copies)

    # One file touched: only that file is recomputed.
    calls.clear()
    st = os.stat(copies[0])
    os.utime(copies[0], ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    again = pl.analyse(_entries(copies), _SYNTH_CFG, cache_dir=str(cache), workers=1)
    assert calls == [copies[0].name]
    assert list(again.items()) == list(cold.items())

    # A corrupt cache entry is a miss, not a crash.
    calls.clear()
    for f in cache.rglob("*.npz"):
        f.write_bytes(b"not an npz")
    assert list(pl.analyse(_entries(copies), _SYNTH_CFG, cache_dir=str(cache), workers=1).items()) \
        == list(cold.items())
    assert sorted(calls) == sorted(p.name for p in copies)


def test_no_cache_dir_writes_nothing(synthetic_episodes, tmp_path):
    before = sorted(os.listdir(synthetic_episodes[0].parent))
    pl.analyse(_entries(synthetic_episodes[:2]), _SYNTH_CFG, cache_dir=None, workers=1)
    assert sorted(os.listdir(synthetic_episodes[0].parent)) == before


def test_decode_failures_propagate_out_of_the_pool(synthetic_episodes, synthetic_video, tmp_path):
    garbage = tmp_path / "garbage.mkv"
    garbage.write_bytes(b"not a video")
    with pytest.raises(subprocess.CalledProcessError):
        pl.analyse(_entries(synthetic_episodes[:2]) + [pl.FileEntry("garbage.mkv", str(garbage))],
                   _SYNTH_CFG, cache_dir=None, workers=2)
    # conftest's synthetic_video has no audio stream at all.
    with pytest.raises(RuntimeError, match="No audio streams"):
        pl.analyse(_entries(synthetic_episodes[:2]) + [pl.FileEntry("silent.mp4", str(synthetic_video))],
                   _SYNTH_CFG, cache_dir=None, workers=2)


def test_cancel_raises_and_fewer_than_two_files_is_empty(synthetic_episodes, tmp_path):
    with pytest.raises(pl.AnalysisCancelled):
        pl.analyse(_entries(synthetic_episodes), _SYNTH_CFG, cache_dir=None, workers=1,
                   cancel=lambda: True)
    assert pl.analyse(_entries(synthetic_episodes[:1]), _SYNTH_CFG, cache_dir=None, workers=1) == {}


@pytest.mark.parametrize("exc", [
    ValueError("cannot find context for 'forkserver'"),
    OSError("could not start forkserver"),
    BrokenProcessPool("pool broke at start-up"),
])
def test_pool_start_failure_falls_back_to_serial_fingerprinting_with_identical_output(
    synthetic_episodes, tmp_path, monkeypatch, caplog, exc,
):
    """Task-6 ruling 2: if the process pool cannot start -- ValueError (e.g.
    get_context("forkserver") failing on a platform without it), OSError, or
    a BrokenProcessPool at start-up -- ingest() must fall back to
    fingerprinting the misses serially, in-process, using the exact same
    fingerprint.fingerprint_file() function, and log exactly one warning.
    The fallback's output must be identical to a normal pooled run.
    """
    pooled = pl.analyse(_entries(synthetic_episodes), _SYNTH_CFG, cache_dir=None, workers=2)
    assert pooled  # the synthetic corpus really has repeats -- a non-trivial result

    def broken_pool_context():
        raise exc

    monkeypatch.setattr(pl, "_pool_context", broken_pool_context)

    with caplog.at_level(logging.WARNING):
        serial = pl.analyse(_entries(synthetic_episodes), _SYNTH_CFG, cache_dir=None, workers=2)

    assert list(serial.items()) == list(pooled.items())
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one warning, got {[r.message for r in warnings]}"


class _CrashAfterFirstExecutor:
    """Fake ProcessPoolExecutor: the first submission's future resolves for
    real (`fn` is called synchronously, inline), and every later
    submission's future raises BrokenProcessPool only once its result is
    actually collected, on a short delay -- simulating a worker that died
    (e.g. OOM-killed) partway through a batch, after at least one file had
    already come back successfully. No real subprocess involved, so
    monkeypatching fingerprint.fingerprint_file in the test process
    actually takes effect (a real pool would re-import it fresh in each
    worker, defeating the monkeypatch)."""

    def __init__(self, *a, **kw):
        self._n = 0

    def submit(self, fn, *args, **kwargs):
        index = self._n
        self._n += 1
        future = Future()
        if index == 0:
            future.set_result(fn(*args, **kwargs))
        else:
            def _break_later():
                time.sleep(0.05)
                future.set_exception(BrokenProcessPool("worker died"))
            threading.Thread(target=_break_later, daemon=True).start()
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def test_pool_crash_mid_run_falls_back_only_for_unfinished_files(
    synthetic_episodes, monkeypatch, caplog,
):
    """Task-6 review, Important finding 1/2a: a BrokenProcessPool surfacing
    while COLLECTING results (not at pool start-up) -- e.g. a worker
    process OOM-killed after finishing some files -- must fall back to
    serial fingerprinting for only the files not yet finished, log a
    DISTINCT "crashed after N of M" warning (not the generic "pool
    unavailable" one used for a pool that never started), and still
    produce output identical to a clean pooled run with each file
    fingerprinted exactly once (no file redone after it already succeeded
    through the pool).
    """
    files2 = synthetic_episodes[:2]
    pooled = pl.analyse(_entries(files2), _SYNTH_CFG, cache_dir=None, workers=2)
    assert pooled

    real_fingerprint_file = fp.fingerprint_file
    calls = []

    def counting_fingerprint_file(path, cfg):
        calls.append(os.path.basename(path))
        return real_fingerprint_file(path, cfg)

    monkeypatch.setattr(fp, "fingerprint_file", counting_fingerprint_file)
    monkeypatch.setattr(pl, "ProcessPoolExecutor", lambda *a, **kw: _CrashAfterFirstExecutor())

    with caplog.at_level(logging.WARNING):
        result = pl.analyse(_entries(files2), _SYNTH_CFG, cache_dir=None, workers=2)

    assert list(result.items()) == list(pooled.items())
    assert sorted(calls) == sorted(p.name for p in files2), (
        f"each file must be fingerprinted exactly once, got {calls}"
    )

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one warning, got {[r.message for r in warnings]}"
    assert "crashed after 1 of 2" in warnings[0].message, (
        f"expected a distinct 'crashed after N of M' warning, got: {warnings[0].message!r}"
    )


class _ImmediateExecutor:
    """Fake ProcessPoolExecutor whose submit() runs `fn` synchronously and
    stores whatever it returns or raises on a real Future -- this models a
    real ProcessPoolExecutor's contract (the callable's own exception
    surfaces from future.result(), never from submit() itself) without an
    actual subprocess, so a per-file exception can be injected via a
    monkeypatched fingerprint_file and still observed exactly as
    future.result() would deliver it."""

    def __init__(self, *a, **kw):
        pass

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - mirrors future.set_exception's own contract
            future.set_exception(exc)
        else:
            future.set_result(result)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def test_genuine_per_file_worker_exception_propagates_without_a_fallback(
    synthetic_episodes, monkeypatch, caplog,
):
    """Task-6 review, Important finding 1/2b: an exception raised BY THE
    FINGERPRINT FUNCTION inside a worker -- e.g. FileNotFoundError when
    ffmpeg is missing, or any other OSError from a bad file -- must
    propagate to the caller unchanged, exactly as before Task 6: no
    fallback warning, and no serial retry (the previous wide
    `except (ValueError, OSError, BrokenProcessPool)` around the whole
    submit+collect loop misclassified this as "pool unavailable" only
    because FileNotFoundError is an OSError subclass).
    """
    files2 = synthetic_episodes[:2]
    culprit = files2[1].name
    real_fingerprint_file = fp.fingerprint_file
    calls = []

    def flaky_fingerprint_file(path, cfg):
        calls.append(os.path.basename(path))
        if os.path.basename(path) == culprit:
            raise FileNotFoundError("ffmpeg binary not found")
        return real_fingerprint_file(path, cfg)

    monkeypatch.setattr(fp, "fingerprint_file", flaky_fingerprint_file)
    monkeypatch.setattr(pl, "ProcessPoolExecutor", lambda *a, **kw: _ImmediateExecutor())

    with caplog.at_level(logging.WARNING):
        with pytest.raises(FileNotFoundError, match="ffmpeg binary not found"):
            pl.analyse(_entries(files2), _SYNTH_CFG, cache_dir=None, workers=2)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings, (
        f"a genuine per-file worker exception must not log any fallback warning, got: "
        f"{[r.message for r in warnings]}"
    )
    assert calls.count(culprit) == 1, (
        f"the failing file must not be retried after a genuine worker exception, "
        f"called {calls.count(culprit)} times"
    )


def test_fingerprint_params_cover_every_field_that_changes_fingerprints():
    params = RangesConfig().fingerprint_params()
    assert json.loads(json.dumps(params, sort_keys=True)) == params
    base = pl.cache_key("identity", RangesConfig())
    for changed in (
        RangesConfig(dsp=DSPConfig(sample_rate=8000)),
        RangesConfig(dsp=DSPConfig(highpass_freq=100)),
        RangesConfig(dsp=DSPConfig(lowpass_freq=4000)),
        RangesConfig(peak=PeakConfig(noise_floor=0.2)),
        RangesConfig(hash=HashConfig(fanout=4)),
        RangesConfig(hash=HashConfig(f1_bits=9)),
    ):
        assert pl.cache_key("identity", changed) != base
    assert pl.cache_key("identity", RangesConfig(match=MatchConfig(min_count=3))) == base
    assert pl.cache_key("other", RangesConfig()) != base
    assert hashlib.sha256(base.encode()).hexdigest()  # a plain string key
