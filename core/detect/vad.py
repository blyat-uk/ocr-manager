"""Speech detection used to choose frames worth inspecting for subtitles.

Two functions here answer two different questions, deliberately:

- ``speech_segments()`` / ``probe_times()``'s sibling for the Stage 3 UI
  speech lane makes a speech/not-speech DECISION: it thresholds, and an
  empty result means "no confident speech found here" -- NOT "no speech
  present". It is a best-effort detector. Two rounds of review found two
  distinct constructions (loudness-normalised dialogue with a narrow
  internal dB spread; in-band dialogue under a concurrent broadband music
  layer) that make its threshold return zero segments even though a human
  listening would call both "speech". Both are real, both are documented
  here rather than chased further: a hard threshold over synthetic and
  real-world audio will always have *a* failure boundary somewhere, and
  this function's job -- flagging likely-speech spans for a UI a human
  reviews -- tolerates that. See git history / task-1-report.md for the
  specific repro constructions if tightening this further is ever needed.

- ``probe_times()`` does NOT call ``speech_segments()`` and does NOT
  threshold at all, on purpose: its consumer doesn't need a speech/silence
  decision, it needs candidate timestamps ranked by how likely they are to
  carry dialogue. It ranks the in-band energy envelope by magnitude and
  spreads picks across the window (non-max suppression), so it degrades
  gracefully to "least bad candidates" instead of silently returning
  nothing on content this module's threshold-based detector cannot
  confidently call. See probe_times()'s docstring for the guarantee.

Frames sampled at the midpoint of detected speech contain a visible subtitle
far more often than uniformly spaced probes, because uniform probes 0.5s
apart are highly autocorrelated: they sit inside the same silent action
scene together. Measured on two reference episodes (full-episode duration,
subtitle presence taken from that project's chi/*.ass OCR output, itself an
under-reporting oracle so these are lower bounds) -- see
.superpowers/sdd/2026-09-16-stage2b-detectors/task-1-report.md for the
measurement and its most recent (rank-based probe_times()) numbers.
"""
from __future__ import annotations

import subprocess
import time
from collections.abc import Callable

import numpy as np

from core.proc import TEXT_ENCODING, hidden_child

SAMPLE_RATE = 16000
_BAND_LOW_HZ = 300.0
_BAND_HIGH_HZ = 3400.0
_HOP_SEC = 0.020
_FLOOR_PERCENTILE = 20.0
_CEILING_PERCENTILE = 99.0
_THRESHOLD_MARGIN_DB = 6.0
# Speech-band concentration required of a frame in speech_segments(). Genuine
# in-band content measures ~1.0; a purely out-of-band source leaking through
# the band mask measures ~0.003-0.004 (see test_out_of_band_energy_is_not_speech).
# 0.15 keeps ~25x headroom below that leakage while tolerating dialogue under
# a concurrent broadband music layer as heavy as the speech itself (measured
# in-band fraction ~0.31-0.58 for music amplitude 0.1x-0.5x the dialogue's).
_MIN_INBAND_FRACTION = 0.15
_CLOSE_GAP_SEC = 0.2
_MIN_SEGMENT_SEC = 0.25
# probe_times(): minimum spacing enforced between ranked candidates, and the
# target spacing used to size how many candidates a window gets. Both are
# picked, not measured: ~0.75s is below typical dialogue-line spacing (so it
# doesn't merge distinct lines into one pick), and ~2.5s reproduces roughly
# the candidate density speech_segments() produced organically on the
# reference corpus prior to this rewrite (see task-1-report.md round 3).
_PEAK_MIN_SEPARATION_SEC = 0.75
_PEAK_TARGET_SPACING_SEC = 2.5
_SILENCE_AMPLITUDE_EPS = 1e-6
# extract_audio_window(): a bound for a hung decoder, not a performance
# target. The longest window on the reference corpus -- 40-60% of an 84.5-min
# movie, 1014 s of audio -- extracted in 0.47 s (20.7-min 4K episode: 0.14 s),
# so 120 s is ~250x the worst measured case.
AUDIO_EXTRACT_TIMEOUT_SEC = 120.0
# How often a running extraction checks cancel_check.
_CANCEL_POLL_SEC = 0.1


class AudioExtractionCancelled(Exception):
    """cancel_check fired while extract_audio_window() was running; ffmpeg was
    stopped and nothing was returned."""


def has_audio_stream(video_path: str) -> bool:
    """Whether the file has at least one audio stream (decodable or not).

    Raises CalledProcessError when ffprobe cannot read the file at all, so a
    missing or corrupt file is never mistaken for a silent one. ffmpeg's own
    exit status cannot make this distinction: extracting audio fails with
    the same status (234) for a file with no audio stream and for one whose
    audio stream cannot be decoded.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, check=True, text=True, **TEXT_ENCODING,
                            **hidden_child())
    return bool(result.stdout.strip())


def extract_audio_window(video_path: str, start_sec: float, duration_sec: float,
                         sample_rate: int = SAMPLE_RATE,
                         cancel_check: Callable[[], bool] | None = None) -> np.ndarray:
    """Decode a window of the first audio stream as mono float32 in [-1, 1].

    Raises CalledProcessError when ffmpeg fails, TimeoutExpired after
    AUDIO_EXTRACT_TIMEOUT_SEC, and AudioExtractionCancelled once
    `cancel_check` (polled every _CANCEL_POLL_SEC) returns truthy; ffmpeg is
    killed in the last two cases.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start_sec):.3f}", "-t", f"{max(0.0, duration_sec):.3f}",
        "-i", video_path, "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "s16le", "pipe:1",
    ]
    timeout = AUDIO_EXTRACT_TIMEOUT_SEC
    deadline = time.monotonic() + timeout
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **hidden_child()) as proc:
        while True:
            try:
                # Retrying communicate() after its timeout loses no output.
                stdout, stderr = proc.communicate(timeout=_CANCEL_POLL_SEC)
                break
            except subprocess.TimeoutExpired:
                cancelled = cancel_check is not None and cancel_check()
                if cancelled or time.monotonic() >= deadline:
                    proc.kill()
                    proc.communicate()
                    if cancelled:
                        raise AudioExtractionCancelled(video_path) from None
                    raise subprocess.TimeoutExpired(cmd, timeout) from None
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
    samples = np.frombuffer(stdout, dtype=np.int16)
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


def _inband_energy_envelope(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Per-frame in-band (speech-band) energy, linear magnitude sum, one value
    per _HOP_SEC frame. No threshold, no dB conversion: this is a ranking
    signal, not a decision, so a monotonic transform of it would be wasted
    work."""
    hop = max(1, int(round(_HOP_SEC * sample_rate)))
    n_frames = samples.size // hop
    if n_frames == 0:
        return np.array([], dtype=np.float64), hop

    frames = samples[: n_frames * hop].reshape(n_frames, hop)
    window = np.hanning(hop).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(hop, d=1.0 / sample_rate)
    band = (freqs >= _BAND_LOW_HZ) & (freqs <= _BAND_HIGH_HZ)
    if not band.any():
        return np.zeros(n_frames, dtype=np.float64), hop
    return spectrum[:, band].sum(axis=1), hop


def _rank_peaks(energy: np.ndarray, hop: int, sample_rate: int,
                min_separation_sec: float, max_candidates: int) -> list[float]:
    """Greedy non-max suppression over the energy envelope: take the highest
    remaining frame, suppress a min_separation_sec window around it, repeat.

    This is rank-based, not threshold-based: there is no value a frame must
    clear to be picked, only relative order. A perfectly flat envelope (equal
    energy everywhere -- the exact construction that collapses a percentile
    threshold, see task-1-report.md round 3) still yields max_candidates
    picks, spread across the window by the suppression step, because nothing
    here depends on the *magnitude* of the gap between picks.

    Returns frame-time offsets in seconds relative to the envelope's start,
    chronological. Always returns at least one pick when energy is non-empty
    and max_candidates >= 1.
    """
    n = energy.size
    if n == 0 or max_candidates < 1:
        return []

    min_sep_frames = max(1, int(round(min_separation_sec * sample_rate / hop)))
    order = np.argsort(energy)[::-1]
    taken = np.zeros(n, dtype=bool)
    picks: list[int] = []
    for idx in order:
        if taken[idx]:
            continue
        picks.append(int(idx))
        if len(picks) >= max_candidates:
            break
        lo, hi = max(0, idx - min_sep_frames), min(n, idx + min_sep_frames + 1)
        taken[lo:hi] = True

    picks.sort()
    return [p * hop / sample_rate for p in picks]


def _probe_candidates(samples: np.ndarray, sample_rate: int,
                      start: float, length: float) -> list[float]:
    """The ranking logic behind probe_times(), factored out so it can be
    driven directly with synthetic samples in tests without going through
    ffmpeg. Not a reimplementation: probe_times() calls exactly this.

    `start`/`length` are the window's absolute position (seconds); returned
    timestamps are absolute. See probe_times() for the guarantee this
    provides.
    """
    if samples.size == 0 or not np.any(np.abs(samples) > _SILENCE_AMPLITUDE_EPS):
        return []  # true digital silence: nothing to rank

    energy, hop = _inband_energy_envelope(samples, sample_rate)
    if energy.size == 0:
        # Window shorter than one analysis frame, but audio is present:
        # still honor the guarantee with the one candidate we can offer.
        return [start + length / 2.0]

    max_candidates = max(1, round(length / _PEAK_TARGET_SPACING_SEC))
    peaks = _rank_peaks(energy, hop, sample_rate, _PEAK_MIN_SEPARATION_SEC, max_candidates)
    return [start + p for p in peaks]


def probe_times(video_path: str, duration_sec: float,
                window_frac: tuple[float, float] = (0.40, 0.60),
                cancel_check: Callable[[], bool] | None = None) -> list[float]:
    """Absolute timestamps worth sampling, ranked by in-band energy.

    Deliberately NOT a speech/not-speech decision and deliberately does not
    call speech_segments(): the consumer needs candidates ranked by how
    likely they are to carry dialogue, not a threshold crossing. Candidates
    are the top local-energy picks in the speech band, spread across the
    window by non-max suppression so they are not clustered.

    Guarantee: if the window's audio is not true digital silence, this
    returns at least one candidate -- worst case "least bad candidate",
    never an empty result that silently loses the speedup this module
    exists to provide. Returns [] only when the window is true digital
    silence, when the file has no audio stream at all, or when the window
    is degenerate (non-positive duration/window). An audio stream that
    exists but cannot be extracted still raises (CalledProcessError), as
    does a file ffprobe cannot read. `cancel_check` reaches the audio
    extraction: see extract_audio_window() for it, and for its timeout.

    Chronological, never highest-energy-first: long high-energy spans are
    frequently music and action, not dialogue, so the consumer should see
    candidates in time order rather than energy order.
    """
    if duration_sec <= 0:
        return []
    start = duration_sec * window_frac[0]
    length = max(0.0, duration_sec * (window_frac[1] - window_frac[0]))
    if length <= 0:
        return []
    if not has_audio_stream(video_path):
        return []

    samples = extract_audio_window(video_path, start, length, cancel_check=cancel_check)
    return _probe_candidates(samples, SAMPLE_RATE, start, length)
