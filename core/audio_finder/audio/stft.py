from __future__ import annotations

import numpy as np

from core.audio_finder.config import DSPConfig


def compute_spectrogram(samples: np.ndarray, cfg: DSPConfig) -> np.ndarray:
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
