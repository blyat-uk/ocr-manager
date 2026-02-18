from __future__ import annotations

from core.audio_finder.audio.extract import extract_audio
from core.audio_finder.audio.fingerprint import generate_fingerprints
from core.audio_finder.audio.peaks import find_peaks
from core.audio_finder.audio.stft import compute_spectrogram
from core.audio_finder.config import AnalysisConfig


def fingerprint_file(
    path: str, cfg: AnalysisConfig
) -> tuple[list[tuple[int, int]], float]:
    samples = extract_audio(path, sample_rate=cfg.dsp.sample_rate)
    duration_sec = len(samples) / cfg.dsp.sample_rate

    spectrogram = compute_spectrogram(samples, cfg.dsp)
    peaks = find_peaks(spectrogram, cfg.dsp, cfg.peak)
    hashes = generate_fingerprints(peaks, cfg.hash)

    return hashes, duration_sec
