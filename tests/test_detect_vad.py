import subprocess

import numpy as np
import pytest

from core.detect import vad

SR = 16000


def _tone(duration_s, freq=1000.0, amp=0.3, sr=SR):
    t = np.arange(int(duration_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_s, sr=SR):
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def test_finds_two_speech_segments_separated_by_silence():
    samples = np.concatenate([_silence(1.0), _tone(2.0), _silence(1.5), _tone(1.0), _silence(1.0)])
    segs = vad.speech_segments(samples, SR)
    assert len(segs) == 2
    assert segs[0][0] == pytest.approx(1.0, abs=0.15)
    assert segs[0][1] == pytest.approx(3.0, abs=0.15)
    assert segs[1][0] == pytest.approx(4.5, abs=0.15)


def test_segments_are_chronological():
    samples = np.concatenate([_silence(0.5), _tone(1.0), _silence(1.0), _tone(1.0)])
    segs = vad.speech_segments(samples, SR)
    assert segs == sorted(segs)


def test_short_blips_are_dropped():
    samples = np.concatenate([_silence(1.0), _tone(0.1), _silence(1.0), _tone(1.0), _silence(0.5)])
    segs = vad.speech_segments(samples, SR)
    assert len(segs) == 1, f"0.1s blip should be dropped, got {segs}"


def test_out_of_band_energy_is_not_speech():
    # 60 Hz hum sits below the 300-3400 Hz speech band.
    samples = np.concatenate([_silence(0.5), _tone(2.0, freq=60.0, amp=0.9), _silence(0.5)])
    assert vad.speech_segments(samples, SR) == []


def test_silence_yields_no_segments():
    assert vad.speech_segments(_silence(3.0), SR) == []


def test_close_runs_merge_into_one_segment():
    # Gap of 0.1s is <= _CLOSE_GAP_SEC (0.2s): the two runs must merge.
    samples = np.concatenate([_silence(0.5), _tone(0.3), _silence(0.1), _tone(0.3), _silence(0.5)])
    segs = vad.speech_segments(samples, SR)
    assert len(segs) == 1, f"runs 0.1s apart should merge, got {segs}"
    assert segs[0][0] == pytest.approx(0.5, abs=0.05)
    assert segs[0][1] == pytest.approx(1.2, abs=0.05)


def test_distant_runs_stay_separate():
    # Gap of 0.4s is > _CLOSE_GAP_SEC (0.2s): the two runs must stay separate.
    samples = np.concatenate([_silence(0.5), _tone(0.3), _silence(0.4), _tone(0.3), _silence(0.5)])
    segs = vad.speech_segments(samples, SR)
    assert len(segs) == 2, f"runs 0.4s apart should stay separate, got {segs}"


@pytest.mark.parametrize("occupancy", [0.2, 0.4, 0.6, 0.8, 0.95])
def test_detected_duration_tracks_occupancy_without_collapsing(occupancy):
    # Regression test for a cliff-edge failure: a percentile-order-statistic
    # threshold that assumes speech is a small minority of the window can
    # overshoot the window's own maximum energy once speech occupancy rises
    # past the assumed minority fraction, collapsing detection from full
    # recall to zero with no signal distinguishing it from genuine silence.
    # Detected duration must track true active duration at every occupancy,
    # never collapse to zero while speech is present.
    total_sec = 10.0
    active_sec = total_sec * occupancy
    silence_sec = total_sec - active_sec
    samples = np.concatenate([
        _silence(silence_sec / 2), _tone(active_sec), _silence(silence_sec / 2),
    ])
    segs = vad.speech_segments(samples, SR)
    detected_sec = sum(e - s for s, e in segs)
    assert detected_sec > 0, f"collapsed to zero at occupancy={occupancy}"
    assert detected_sec == pytest.approx(active_sec, abs=0.3)


def test_extract_audio_window_returns_mono_float(synthetic_audio_video):
    samples = vad.extract_audio_window(str(synthetic_audio_video), 0.0, 2.0)
    assert samples.ndim == 1
    assert samples.dtype == np.float32
    assert len(samples) == pytest.approx(2.0 * SR, rel=0.1)
    assert np.abs(samples).max() <= 1.0


def test_probe_times_are_inside_the_window_and_ordered(synthetic_audio_video):
    times = vad.probe_times(str(synthetic_audio_video), duration_sec=10.0, window_frac=(0.0, 1.0))
    assert times == sorted(times)
    assert all(0.0 <= t <= 10.0 for t in times)
    # A probe_times() that always returned [] would pass the two asserts above
    # vacuously (sorted([]) == [] and all() over [] are both trivially true).
    # probe_times() is rank-based (see module docstring): it does not return
    # segment midpoints, it returns the top-ranked, spread-out energy peaks.
    # For a 10s window with _PEAK_TARGET_SPACING_SEC=2.5, that's exactly 4
    # candidates, and each must land near one of the fixture's three known
    # bursts (1-3s, 5-6s, 8-9s) -- not in a silent stretch between them.
    assert len(times) == 4, f"expected 4 ranked candidates, got {times}"
    bursts = [(1.0, 3.0), (5.0, 6.0), (8.0, 9.0)]
    for t in times:
        assert any(s - 0.5 <= t <= e + 0.5 for s, e in bursts), (
            f"candidate {t} does not fall near any known burst {bursts}"
        )


# --- Round 3: probe_times() must not inherit speech_segments()'s threshold
# collapses. These drive vad._probe_candidates() -- the exact ranking logic
# probe_times() calls, factored out so it can run on synthetic samples
# without shelling out to ffmpeg for every construction -- with the two
# constructions the round-2 re-review used to break speech_segments() twice.


@pytest.mark.parametrize("occupancy", [0.85, 0.90, 0.95, 1.0])
def test_probe_candidates_survive_compressed_dynamic_range(occupancy):
    # Reproduces the round-2 finding: speech_segments()'s threshold is
    # min(floor + 6dB margin, ceiling). Loudness-normalised/compressed
    # dialogue (routine for broadcast/streaming masters) can have an internal
    # amplitude spread under that 6dB margin, so only the top ~1% (the
    # ceiling percentile) clears the threshold, and those isolated 20ms
    # frames then fail _MIN_SEGMENT_SEC: 0 segments, despite the window being
    # almost entirely "speech". Construction matches the reviewer's repro
    # script exactly (per-frame amplitude wobbling within a narrow 3dB band).
    rng = np.random.default_rng(0)
    hop = int(0.02 * SR)

    def tone_varied(duration_s, freq=1000.0, amp_db_range=(30.0, 33.0)):
        n_frames = int(duration_s * SR / hop)
        t = np.arange(hop) / SR
        out = []
        for i in range(n_frames):
            db = rng.uniform(*amp_db_range)
            amp = 10 ** (db / 20.0) / 1000.0
            phase = i * hop / SR
            out.append((amp * np.sin(2 * np.pi * freq * (t + phase))).astype(np.float32))
        return np.concatenate(out)

    total = 10.0
    active_sec = total * occupancy
    sil_sec = total - active_sec
    samples = np.concatenate([_silence(sil_sec / 2), tone_varied(active_sec), _silence(sil_sec / 2)])

    # The trigger is real and is documented, not fixed, in speech_segments():
    assert vad.speech_segments(samples, SR) == [], (
        "if this starts passing, the compressed-dynamic-range limitation "
        "documented in the module docstring may no longer apply"
    )

    candidates = vad._probe_candidates(samples, SR, start=0.0, length=total)
    assert candidates, f"probe_times must not collapse to zero at occupancy={occupancy}"
    assert candidates == sorted(candidates)
    active_lo, active_hi = sil_sec / 2, sil_sec / 2 + active_sec
    assert all(active_lo - 0.5 <= t <= active_hi + 0.5 for t in candidates)


def test_probe_candidates_survive_dialogue_under_music():
    # Reproduces the round-2 finding: speech_segments()'s in-band-fraction
    # gate is computed against *total* spectral energy, so a concurrent
    # broadband layer (bass + hiss, standing in for background music/effects)
    # dilutes the denominator without touching the in-band numerator.
    # Construction matches the reviewer's repro script exactly.
    rng = np.random.default_rng(0)
    music_amp = 0.2  # ~2/3 the dialogue's amplitude -- "modest", not overwhelming
    speech_amp = 0.3
    duration_s = 2.0
    n = int(duration_s * SR)
    t = np.arange(n) / SR
    speech = speech_amp * np.sin(2 * np.pi * 1000.0 * t)
    bass = music_amp * np.sin(2 * np.pi * 150.0 * t)
    hiss = music_amp * 0.6 * rng.standard_normal(n)
    mixed = (speech + bass + hiss).astype(np.float32)
    samples = np.concatenate([_silence(1.0), mixed, _silence(1.0)])

    candidates = vad._probe_candidates(samples, SR, start=0.0, length=4.0)
    assert candidates, "probe_times must not collapse to zero for dialogue under modest music"
    assert candidates == sorted(candidates)
    assert all(1.0 - 0.5 <= t <= 3.0 + 0.5 for t in candidates)


@pytest.mark.parametrize("samples,label", [
    (_tone(3.0), "clean in-band tone"),
    (_tone(3.0, freq=60.0, amp=0.9), "out-of-band tone"),
    (0.3 * np.random.default_rng(1).standard_normal(SR * 3).astype(np.float32), "broadband noise"),
    (np.full(SR * 3, 1e-4, dtype=np.float32), "near-silent constant"),
])
def test_probe_candidates_never_empty_for_nonsilent_audio(samples, label):
    candidates = vad._probe_candidates(samples, SR, start=0.0, length=len(samples) / SR)
    assert candidates, f"probe_times must return a candidate for non-silent audio ({label})"


def test_probe_candidates_empty_only_for_true_digital_silence():
    assert vad._probe_candidates(_silence(3.0), SR, start=0.0, length=3.0) == []


# --- Files without usable audio ---------------------------------------------


def test_probe_times_is_empty_for_a_file_with_no_audio_stream(video_only_clip):
    """No audio stream means nothing to rank, exactly like digital silence:
    [] sends crop detection to uniform probing with the no-speech flag,
    instead of an ffmpeg error that skips the file."""
    assert vad.probe_times(str(video_only_clip), duration_sec=20.0) == []


def test_has_audio_stream_tells_video_only_files_from_files_with_audio(
        video_only_clip, synthetic_audio_video, undecodable_audio_clip):
    assert vad.has_audio_stream(str(video_only_clip)) is False
    assert vad.has_audio_stream(str(synthetic_audio_video)) is True
    assert vad.has_audio_stream(str(undecodable_audio_clip)) is True


def test_an_audio_stream_that_cannot_be_decoded_still_raises(undecodable_audio_clip):
    """Only a MISSING audio stream is a normal outcome. An audio stream that
    is there but cannot be extracted is a real failure and must surface."""
    with pytest.raises(subprocess.CalledProcessError):
        vad.probe_times(str(undecodable_audio_clip), duration_sec=4.0)


def test_probing_a_file_that_cannot_be_opened_raises(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        vad.probe_times(str(tmp_path / "missing.mp4"), duration_sec=20.0)
