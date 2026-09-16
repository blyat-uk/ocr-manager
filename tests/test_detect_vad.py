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
    # The fixture's bursts are at 1-3s, 5-6s, 8-9s: known midpoints 2.0, 5.5, 8.5.
    # A probe_times() that always returned [] would pass the two asserts above
    # vacuously (sorted([]) == [] and all() over [] are both trivially true).
    assert len(times) == 3, f"expected 3 burst midpoints, got {times}"
    assert times[0] == pytest.approx(2.0, abs=0.15)
    assert times[1] == pytest.approx(5.5, abs=0.15)
    assert times[2] == pytest.approx(8.5, abs=0.15)
