from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from core.audio_finder.config import DSPConfig, PeakConfig


def find_peaks(
    spectrogram: np.ndarray,
    dsp_cfg: DSPConfig,
    peak_cfg: PeakConfig,
) -> list[tuple[int, int]]:
    if spectrogram.size == 0:
        return []

    num_bins, num_frames = spectrogram.shape

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
    num_chunks = chunk_ids.max() + 1

    keep = np.zeros(len(freq_bins), dtype=bool)
    for c in range(num_chunks):
        mask = chunk_ids == c
        if mask.sum() <= max_per_chunk:
            keep[mask] = True
        else:
            indices = np.where(mask)[0]
            top_k = indices[np.argsort(-amplitudes[indices])[:max_per_chunk]]
            keep[top_k] = True

    freq_bins = freq_bins[keep]
    time_frames = time_frames[keep]

    order = np.lexsort((freq_bins, time_frames))
    return list(zip(freq_bins[order].tolist(), time_frames[order].tolist()))
