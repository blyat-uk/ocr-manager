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
