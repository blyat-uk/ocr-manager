"""Per-episode audio profile for the Stage 3 time-ranges review tab: a
waveform envelope and speech spans. Qt-free, like the rest of core/detect/.

Decodes the whole file once through vad.extract_audio_window() -- a separate
decode from the ranges pipeline's own fingerprinting pass (see
core/detect/ranges/pipeline.py's module docstring for why fingerprinting
uses its own decode); nothing here shares that one. speech_segments() is the
same threshold-based speech/silence decision core/detect/vad.py's module
docstring describes: an empty result means "no confident speech found",
not "no speech present".
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from core.detect import vad
from core.detect.ranges.pipeline import Block

# Waveform resolution: RMS bins per file, independent of the file's duration.
ENVELOPE_BINS = 600


@dataclass(frozen=True)
class AudioProfile:
    duration: float
    envelope: list[float]              # ENVELOPE_BINS RMS values normalised to 0..1 (max = 1.0; all zeros when silent)
    speech: list[tuple[float, float]]  # vad.speech_segments() over the whole file


def _rms_envelope(samples: np.ndarray, bins: int) -> list[float]:
    """RMS per chunk, `bins` equal chunks with the last absorbing whatever
    remainder doesn't divide evenly, normalised so the loudest chunk is 1.0.
    All zeros when every chunk's RMS is 0 (a silent or empty ``samples``) --
    never divides by zero. Rounded to 4 decimals so the result is
    JSON-friendly."""
    n = samples.size
    rms = np.zeros(bins, dtype=np.float64)
    if n:
        chunk_size = n // bins
        for i in range(bins):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < bins - 1 else n
            chunk = samples[start:end]
            if chunk.size:
                rms[i] = np.sqrt(np.mean(np.square(chunk, dtype=np.float64)))
    peak = float(rms.max()) if n else 0.0
    if peak > 0:
        rms = rms / peak
    return [round(float(x), 4) for x in rms]


def audio_profile(video_path: str, duration_sec: float,
                   cancel_check: Callable[[], bool] | None = None) -> AudioProfile:
    """Envelope + speech spans for one episode's full duration.

    A file with no audio stream (vad.has_audio_stream() False) returns a
    zero envelope and no speech, without raising. Otherwise decodes the
    whole file once via
    ``vad.extract_audio_window(video_path, 0.0, duration_sec, cancel_check=cancel_check)``
    -- AudioExtractionCancelled (and any other exception extract_audio_window
    raises) propagates uncaught; the caller (the job layer) turns
    cancellation into its own cancelled state.
    """
    if not vad.has_audio_stream(video_path):
        return AudioProfile(duration=duration_sec, envelope=[0.0] * ENVELOPE_BINS, speech=[])

    samples = vad.extract_audio_window(video_path, 0.0, duration_sec, cancel_check=cancel_check)
    envelope = _rms_envelope(samples, ENVELOPE_BINS)
    speech = vad.speech_segments(samples, vad.SAMPLE_RATE)
    return AudioProfile(duration=duration_sec, envelope=envelope, speech=speech)


SkipSpan = Block | Mapping | tuple[float, float] | list[float]


def _skip_bounds(skip) -> tuple[float, float]:
    """(start_sec, end_sec) of one skipped span: a Block (anything with
    start_sec/end_sec attributes), a mapping with "start_sec"/"end_sec" keys
    (the entries of a file's evidence["ranges"]["blocks"]), or a
    (start_sec, end_sec) pair. TypeError for anything else."""
    if hasattr(skip, "start_sec") and hasattr(skip, "end_sec"):
        return float(skip.start_sec), float(skip.end_sec)
    if isinstance(skip, Mapping):
        if "start_sec" not in skip or "end_sec" not in skip:
            raise TypeError(f"a skip span mapping needs start_sec and end_sec: {skip!r}")
        return float(skip["start_sec"]), float(skip["end_sec"])
    try:
        start, end = skip
    except (TypeError, ValueError) as exc:
        raise TypeError(f"not a skip span (Block, mapping or (start_sec, end_sec) pair): {skip!r}") from exc
    return float(start), float(end)


def speech_in_skips(
    speech: Iterable[tuple[float, float]],
    blocks: Iterable[SkipSpan],
    min_overlap_sec: float = 2.0,
) -> list[tuple[float, float]]:
    """Speech spans that fall inside a span about to be skipped.

    `speech` holds (start_sec, end_sec) pairs (AudioProfile.speech, or the
    lists evidence["audio"]["speech"] holds). `blocks` holds the skipped
    spans, each a Block, a mapping with "start_sec"/"end_sec" (the evidence
    dicts of evidence["ranges"]["blocks"]) or a (start_sec, end_sec) pair, in
    any iterable and mixed freely.

    For every (speech span, skipped span) pair, the overlap clipped to the
    skipped span's own [start_sec, end_sec) is kept only when its length is
    at least ``min_overlap_sec``. A speech span overlapping two skipped spans
    yields two (independently clipped) entries. Result order: ascending
    start time.
    """
    speech = [(s, e) for s, e in speech]
    warnings: list[tuple[float, float]] = []
    for skip in blocks:
        skip_start, skip_end = _skip_bounds(skip)
        for s, e in speech:
            start = max(s, skip_start)
            end = min(e, skip_end)
            if end - start >= min_overlap_sec:
                warnings.append((start, end))
    warnings.sort(key=lambda span: span[0])
    return warnings
