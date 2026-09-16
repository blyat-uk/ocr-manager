"""Parameters for the time-range pipeline.

Every default is copied verbatim from the original
``core/audio_finder/config.py``; changing any of them changes detection
output. ``RangesConfig.fingerprint_params()`` lists exactly the fields that
change a file's fingerprints, and is what the fingerprint cache is keyed on.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DSPConfig:
    sample_rate: int = 11025
    n_fft: int = 1024
    hop_length: int = 512
    highpass_freq: int = 80
    lowpass_freq: int = 5000

    @property
    def frames_per_sec(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def num_bins(self) -> int:
        return self.n_fft // 2 + 1


@dataclass(frozen=True)
class PeakConfig:
    freq_neighborhood: int = 15
    time_neighborhood: int = 9
    noise_floor: float = 0.1
    peaks_per_sec: int = 5


@dataclass(frozen=True)
class HashConfig:
    fanout: int = 5
    dt_min: int = 10
    dt_max: int = 65
    f1_bits: int = 10
    f2_bits: int = 10
    dt_bits: int = 10


@dataclass(frozen=True)
class MatchConfig:
    min_count: int = 20
    delta_tolerance: int = 2
    gap_frames: int = 40
    pad_frames: int = 5
    min_length_sec: float = 60.0
    propagation_threshold: float = 0.15
    stop_word_file_ratio: float = 0.9


@dataclass(frozen=True)
class RangesConfig:
    dsp: DSPConfig = field(default_factory=DSPConfig)
    peak: PeakConfig = field(default_factory=PeakConfig)
    hash: HashConfig = field(default_factory=HashConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    # Bridge short silence gaps that sit at the same position in most files
    # (the "merge repeating silences" setting in the UI).
    merge_repeating_silences: bool = False

    def fingerprint_params(self) -> dict:
        """Every parameter that affects a file's fingerprint array.

        A superset of the original ``fingerprint_params_dict()``: it also
        carries the hash bit widths, which the original left out of its
        profile key even though they change the packed hashes.
        """
        return {
            "sample_rate": self.dsp.sample_rate,
            "n_fft": self.dsp.n_fft,
            "hop_length": self.dsp.hop_length,
            "highpass_freq": self.dsp.highpass_freq,
            "lowpass_freq": self.dsp.lowpass_freq,
            "freq_neighborhood": self.peak.freq_neighborhood,
            "time_neighborhood": self.peak.time_neighborhood,
            "noise_floor": self.peak.noise_floor,
            "peaks_per_sec": self.peak.peaks_per_sec,
            "fanout": self.hash.fanout,
            "dt_min": self.hash.dt_min,
            "dt_max": self.hash.dt_max,
            "f1_bits": self.hash.f1_bits,
            "f2_bits": self.hash.f2_bits,
            "dt_bits": self.hash.dt_bits,
        }
