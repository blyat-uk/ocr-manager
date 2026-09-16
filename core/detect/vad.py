"""Speech detection used to choose frames worth inspecting for subtitles.

Frames sampled at the midpoint of detected speech contain a visible subtitle
far more often than uniformly spaced probes, because uniform probes 0.5s
apart are highly autocorrelated: they sit inside the same silent action
scene together. Measured on two reference episodes (full-episode duration,
subtitle presence taken from that project's chi/*.ass OCR output, itself an
under-reporting oracle so these are lower bounds): speech-guided hit rate
0.37-0.56 vs uniform 0.5s hit rate 0.18-0.36, a 1.55-2.05x improvement. See
.superpowers/sdd/2026-09-16-stage2b-detectors/task-1-report.md for the full
measurement.
"""
from __future__ import annotations

import subprocess

import numpy as np

SAMPLE_RATE = 16000
_BAND_LOW_HZ = 300.0
_BAND_HIGH_HZ = 3400.0
_HOP_SEC = 0.020
_FLOOR_PERCENTILE = 20.0
_CEILING_PERCENTILE = 99.0
_THRESHOLD_MARGIN_DB = 6.0
_MIN_INBAND_FRACTION = 0.5
_CLOSE_GAP_SEC = 0.2
_MIN_SEGMENT_SEC = 0.25


def extract_audio_window(video_path: str, start_sec: float, duration_sec: float,
                         sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode a window of the first audio stream as mono float32 in [-1, 1]."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start_sec):.3f}", "-t", f"{max(0.0, duration_sec):.3f}",
        "-i", video_path, "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "s16le", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    samples = np.frombuffer(result.stdout, dtype=np.int16)
    return (samples.astype(np.float32) / 32768.0)


def speech_segments(samples: np.ndarray, sample_rate: int = SAMPLE_RATE
                    ) -> list[tuple[float, float]]:
    """Detect speech-band energy bursts. Returns (start, end) seconds, chronological."""
    if samples.size == 0:
        return []

    hop = max(1, int(round(_HOP_SEC * sample_rate)))
    n_frames = samples.size // hop
    if n_frames < 2:
        return []

    frames = samples[: n_frames * hop].reshape(n_frames, hop)
    window = np.hanning(hop).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(hop, d=1.0 / sample_rate)
    band = (freqs >= _BAND_LOW_HZ) & (freqs <= _BAND_HIGH_HZ)
    if not band.any():
        return []

    # Two independent gates, both required:
    #
    # 1. Loudness, relative to *this window's own* quiet frames. The floor is
    #    a low percentile (not an assumed-majority high one), so it tracks the
    #    quiet part of the window regardless of how much of the window is
    #    speech. The threshold is then capped at a *high* (not maximum)
    #    percentile: a literal max is one outlier frame away from pinning the
    #    cutoff to a single sample (e.g. a frame straddling a silence/speech
    #    boundary can transiently exceed the steady-state level). Capping
    #    means a uniformly loud window is classified as uniformly active
    #    instead of collapsing to zero when floor+margin overshoots the top
    #    of the window's own range.
    # 2. Band concentration: what fraction of this frame's total spectral
    #    energy actually sits in the speech band. Loudness alone can't tell
    #    genuine speech-band content from an out-of-band source (e.g. a bass
    #    hum) whose sidelobes leak a little energy into the band — that
    #    leakage can be "loud" relative to a silent window's own floor while
    #    still being globally dominated by out-of-band energy. Genuine
    #    speech-band content concentrates almost all of its energy in-band;
    #    leakage does not.
    inband_energy = spectrum[:, band].sum(axis=1)
    total_energy = spectrum.sum(axis=1)
    inband_fraction = inband_energy / np.maximum(total_energy, 1e-12)

    energy_db = 20.0 * np.log10(np.maximum(inband_energy, 1e-10))
    floor = np.percentile(energy_db, _FLOOR_PERCENTILE)
    ceiling = np.percentile(energy_db, _CEILING_PERCENTILE)
    threshold = min(floor + _THRESHOLD_MARGIN_DB, ceiling)

    active = (energy_db >= threshold) & (inband_fraction >= _MIN_INBAND_FRACTION)
    if not active.any():
        return []

    # Frame indices -> (start, end) runs, then close short gaps, then drop short runs.
    edges = np.diff(active.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if active[0]:
        starts.insert(0, 0)
    if active[-1]:
        ends.append(n_frames)

    runs = [(s * hop / sample_rate, e * hop / sample_rate) for s, e in zip(starts, ends)]

    merged: list[tuple[float, float]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= _CLOSE_GAP_SEC:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return [(s, e) for s, e in merged if (e - s) >= _MIN_SEGMENT_SEC]


def probe_times(video_path: str, duration_sec: float,
                window_frac: tuple[float, float] = (0.40, 0.60)) -> list[float]:
    """Absolute timestamps worth sampling, as speech-segment midpoints.

    Chronological, never longest-first: long segments are music and action.
    """
    if duration_sec <= 0:
        return []
    start = duration_sec * window_frac[0]
    length = max(0.0, duration_sec * (window_frac[1] - window_frac[0]))
    if length <= 0:
        return []

    samples = extract_audio_window(video_path, start, length)
    return [start + (s + e) / 2.0 for s, e in speech_segments(samples)]
