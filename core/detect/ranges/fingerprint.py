"""Audio fingerprints for one file: decode -> spectrogram -> peaks -> hashes.

Output is bit-identical to the original ``core/audio_finder/audio/`` code.
Decode, spectrogram and the local-maximum filter are the original code
unchanged; peak selection and pairing are vectorised, and each vectorised
step is exact for the reasons given in its docstring (and pinned by the
property tests in ``tests/test_detect_ranges.py``).

Do not "optimise" the ffmpeg command. In particular, downmixing to mono
before the highpass/lowpass filters is 2x faster but was measured to move
detected boundaries by 1-2 s on 10 of 12 reference files.
"""
from __future__ import annotations

import json
import subprocess
from typing import NamedTuple

import numpy as np
from scipy.ndimage import maximum_filter

from core.detect.ranges.config import DSPConfig, HashConfig, PeakConfig, RangesConfig
from core.proc import TEXT_ENCODING, hidden_child

_LOSSLESS_CODECS = frozenset({
    "flac", "alac",
    "pcm_s16le", "pcm_s16be", "pcm_s24le", "pcm_s24be",
    "pcm_s32le", "pcm_s32be", "pcm_f32le", "pcm_f32be",
    "wavpack", "truehd", "mlp", "dts",
})


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _score_audio_stream(stream: dict) -> float:
    score = 0.0
    codec = stream.get("codec_name", "").lower()
    bitrate = _safe_int(stream.get("bit_rate"))
    sample_rate = _safe_int(stream.get("sample_rate"))
    channels = _safe_int(stream.get("channels"))

    if codec in _LOSSLESS_CODECS:
        score += 50
    elif codec in ("aac", "opus"):
        score += 30
    elif codec in ("mp3", "ac3", "eac3", "vorbis"):
        score += 20
    else:
        score += 10

    if bitrate is not None and bitrate > 0:
        score += min(bitrate / 320_000 * 30, 30)
    elif codec in _LOSSLESS_CODECS:
        score += 30

    if sample_rate is not None and sample_rate > 0:
        score += min(sample_rate / 48_000 * 10, 10)

    if channels is not None and channels > 0:
        score += min(channels / 6 * 10, 10)

    return score


def select_audio_stream(path: str) -> int | None:
    """Index (among audio streams) of the best-quality audio stream, or None
    when the file has exactly one. Raises RuntimeError when it has none."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index,codec_name,sample_rate,channels,bit_rate",
        "-of", "json",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, check=True, text=True, **TEXT_ENCODING,
                            **hidden_child())
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) == 0:
        raise RuntimeError(f"No audio streams found in {path}")
    if len(streams) == 1:
        return None
    scored = [(i, _score_audio_stream(s)) for i, s in enumerate(streams)]
    return max(scored, key=lambda t: t[1])[0]


def decode_audio(path: str, dsp: DSPConfig) -> np.ndarray:
    """Mono float32 samples at ``dsp.sample_rate``, band-limited by ffmpeg.

    The filters run BEFORE the downmix/resample (ffmpeg appends the -ac/-ar
    conversion after the -af chain). Keep it that way; see module docstring.
    """
    audio_idx = select_audio_stream(path)
    cmd = ["ffmpeg", "-i", path]
    if audio_idx is not None:
        cmd += ["-map", f"0:a:{audio_idx}"]
    cmd += [
        "-vn",
        "-ac", "1",
        "-ar", str(dsp.sample_rate),
        "-f", "s16le",
        "-af", f"highpass=f={dsp.highpass_freq},lowpass=f={dsp.lowpass_freq}",
        "-loglevel", "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True, **hidden_child())
    samples = np.frombuffer(result.stdout, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def compute_spectrogram(samples: np.ndarray, dsp: DSPConfig) -> np.ndarray:
    """log1p magnitude STFT, shape (num_bins, num_frames). Original code."""
    n_fft = dsp.n_fft
    hop = dsp.hop_length

    if len(samples) < n_fft:
        return np.empty((dsp.num_bins, 0), dtype=np.float32)

    window = np.hanning(n_fft).astype(np.float32)

    pad_len = n_fft - (len(samples) % hop)
    if pad_len > 0:
        samples = np.concatenate([samples, np.zeros(pad_len, dtype=np.float32)])

    num_frames = (len(samples) - n_fft) // hop + 1
    if num_frames <= 0:
        return np.empty((dsp.num_bins, 0), dtype=np.float32)

    shape = (num_frames, n_fft)
    strides = (samples.strides[0] * hop, samples.strides[0])
    frames = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)

    windowed = frames * window
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)

    magnitude = np.abs(spectrum).T.astype(np.float32)
    return np.log1p(magnitude)


class Peaks(NamedTuple):
    freqs: np.ndarray        # int64, sorted by (time, freq)
    times: np.ndarray        # int64
    boundary_ties: int       # chunks whose top-k cut fell between equal amplitudes


def select_top_k_per_chunk(
    chunk_ids: np.ndarray, amplitudes: np.ndarray, k: int
) -> tuple[np.ndarray, int]:
    """Boolean keep-mask equal to the original per-chunk loop

        for c in range(num_chunks):
            mask = chunk_ids == c
            if mask.sum() <= k: keep[mask] = True
            else:
                indices = np.where(mask)[0]
                keep[indices[np.argsort(-amplitudes[indices])[:k]]] = True

    and the number of chunks where that loop's choice hinged on a tie.

    The kept elements of a chunk form a SET (the caller re-sorts by (t, f)),
    so any ranking that puts the k largest amplitudes first selects the same
    set -- provided the k-th and (k+1)-th largest are not equal. When they
    are equal the set depends on how np.argsort's default, unstable sort
    orders the tied values; for exactly those chunks this function re-runs
    the original expression on the original (ascending-position) index
    order, so the result never relies on a tie-break happening to agree.
    """
    if k < 1:
        raise ValueError(f"peaks per chunk must be >= 1, got {k}")
    n = len(chunk_ids)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep, 0

    # Chunk ascending, amplitude descending; lexsort is stable.
    order = np.lexsort((-amplitudes, chunk_ids))
    sorted_chunks = chunk_ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_chunks[1:] != sorted_chunks[:-1]])
    sizes = np.diff(np.r_[starts, n])
    rank = np.arange(n) - np.repeat(starts, sizes)
    keep[order[rank < k]] = True

    big = sizes > k
    big_starts, big_sizes = starts[big], sizes[big]
    if big_starts.size == 0:
        return keep, 0
    sorted_amps = amplitudes[order]
    tied = sorted_amps[big_starts + k - 1] == sorted_amps[big_starts + k]
    for start, size in zip(big_starts[tied].tolist(), big_sizes[tied].tolist()):
        indices = np.sort(order[start:start + size])  # == np.where(chunk_ids == c)[0]
        keep[indices] = False
        keep[indices[np.argsort(-amplitudes[indices])[:k]]] = True
    return keep, int(np.count_nonzero(tied))


def find_peaks(spectrogram: np.ndarray, dsp: DSPConfig, peak: PeakConfig) -> Peaks:
    """Spectral peaks, at most ``peaks_per_sec`` per one-second chunk,
    sorted by (time, freq)."""
    empty = np.empty(0, dtype=np.int64)
    if spectrogram.size == 0:
        return Peaks(empty, empty, 0)

    neighborhood = (peak.freq_neighborhood, peak.time_neighborhood)
    local_max = maximum_filter(spectrogram, size=neighborhood)
    is_peak = spectrogram == local_max
    is_peak &= spectrogram > peak.noise_floor

    freq_bins, time_frames = np.where(is_peak)
    if len(freq_bins) == 0:
        return Peaks(empty, empty, 0)

    amplitudes = spectrogram[freq_bins, time_frames]
    chunk_ids = (time_frames / dsp.frames_per_sec).astype(int)
    keep, ties = select_top_k_per_chunk(chunk_ids, amplitudes, peak.peaks_per_sec)

    freq_bins = freq_bins[keep]
    time_frames = time_frames[keep]
    order = np.lexsort((freq_bins, time_frames))
    return Peaks(
        freq_bins[order].astype(np.int64, copy=False),
        time_frames[order].astype(np.int64, copy=False),
        ties,
    )


def pair_peaks(freqs: np.ndarray, times: np.ndarray, cfg: HashConfig) -> np.ndarray:
    """(N, 2) int64 array of (hash, anchor_time), in the original loop's order.

    The original, for each anchor i, walked j = i+1.. skipping dt < dt_min,
    breaking on dt > dt_max and stopping after ``fanout`` pairs. ``times`` is
    sorted, so dt is non-decreasing in j and the pairs it makes are exactly
    the contiguous slice [lo, min(lo + fanout, hi)) with
    lo = searchsorted(times, t_i + dt_min, 'left') and
    hi = searchsorted(times, t_i + dt_max, 'right'). When dt_min > 0, lo > i
    automatically; flooring lo at i + 1 keeps it exact for dt_min <= 0 too.
    """
    freqs = np.asarray(freqs, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    n = len(times)
    if n == 0:
        return np.empty((0, 2), dtype=np.int64)

    anchors = np.arange(n)
    lo = np.searchsorted(times, times + cfg.dt_min, side="left")
    lo = np.maximum(lo, anchors + 1)
    hi = np.searchsorted(times, times + cfg.dt_max, side="right")
    counts = np.clip(np.minimum(lo + cfg.fanout, hi) - lo, 0, None)
    total = int(counts.sum())
    if total == 0:
        return np.empty((0, 2), dtype=np.int64)

    first = np.cumsum(counts) - counts
    i_idx = np.repeat(anchors, counts)
    j_idx = np.repeat(lo - first, counts) + np.arange(total)

    dt = times[j_idx] - times[i_idx]
    f1 = np.minimum(freqs[i_idx], (1 << cfg.f1_bits) - 1)
    f2 = np.minimum(freqs[j_idx], (1 << cfg.f2_bits) - 1)
    dt = np.minimum(dt, (1 << cfg.dt_bits) - 1)

    out = np.empty((total, 2), dtype=np.int64)
    out[:, 0] = (f1 << 20) | (f2 << 10) | dt
    out[:, 1] = times[i_idx]
    return out


def fingerprint_samples(samples: np.ndarray, cfg: RangesConfig) -> np.ndarray:
    spectrogram = compute_spectrogram(samples, cfg.dsp)
    peaks = find_peaks(spectrogram, cfg.dsp, cfg.peak)
    return pair_peaks(peaks.freqs, peaks.times, cfg.hash)


def fingerprint_file(path: str, cfg: RangesConfig) -> tuple[np.ndarray, float]:
    """((N, 2) int64 array of (hash, t_frame), duration in seconds)."""
    samples = decode_audio(path, cfg.dsp)
    duration_sec = len(samples) / cfg.dsp.sample_rate
    return fingerprint_samples(samples, cfg), duration_sec
