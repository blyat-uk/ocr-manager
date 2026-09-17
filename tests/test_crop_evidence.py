"""Crop evidence: the frames detect_crop() sampled, the text boxes found on
each, and which of them contributed to the box -- carried out on
CropResult.samples without changing anything detect_crop() already returned.

The box drives OCR, so carrying the evidence out must not move it by a pixel.
The identity tests below replay every scenario in IDENTITY_CASES and compare
each pre-existing CropResult field against
tests/fixtures/crop_evidence_identity.json, which was recorded from
core/detect/crop.py as it stood before CropResult.samples existed (commit
8f27649). Never re-record that fixture to make a failing identity test pass:
a difference there is a changed box, flag or time, which is the bug.
"""
import json
import random
from pathlib import Path

import pytest

from core.detect import crop
from test_detect_crop import (
    _detect_crop_at_resolution,
    _detect_crop_with_fakes,
    _detect_crop_with_mocked_rounds,
    _poly,
)

IDENTITY_FIXTURE = Path(__file__).parent / "fixtures" / "crop_evidence_identity.json"

# Every CropResult field (and property) that existed before `samples`.
PRE_EXISTING_FIELDS = ("box", "sample_pts", "envelope", "agreed", "probes_used", "flagged",
                       "hit_pts", "frame_size", "auto_applicable")

# Speech-guided probe times used by most scenarios: 20 picks 0.8 s apart,
# inside the 40-60% window of the helpers' 60 s duration.
VAD_TIMES = [round(20.0 + 0.8 * k, 3) for k in range(20)]

# Full-frame 1920x1080 geometry of the default bottom band (0.55) as a real
# grab crops and scales it: band rows 486..1080 scaled to 480 rows, 1552 wide.
BAND_GEOMETRY = (1920, 1080) + crop._crop_geometry(1920, 1080, crop.BOTTOM_HALF_CUTOFF, crop.TARGET_HEIGHT)


def _jitter(t: float) -> int:
    """Per-frame x jitter well beyond WATERMARK_TOLERANCE_FRAC (4 px at
    1080p), so no scenario reads as a watermark unless it means to."""
    return 10 * (int(round(t * 10)) % 7)


def _one_line(t):
    j = _jitter(t)
    return [1.0], [_poly(400 + j, 980, 1500 - j, 1030)]


TWO_LINE_TIMES = {VAD_TIMES[1], VAD_TIMES[6], VAD_TIMES[13]}


def _one_or_two_lines(t):
    j = _jitter(t)
    lower = _poly(400 + j, 980, 1500 - j, 1030)
    if t in TWO_LINE_TIMES:
        return [1.0, 1.0], [_poly(450 + j, 920, 1450 - j, 970), lower]
    return [1.0], [lower]


def _to_band(x0, y0, x1, y1, geometry=BAND_GEOMETRY):
    """A full-frame rectangle in the grabbed band's downscaled coordinates."""
    _w, _h, crop_w, crop_h, crop_x, crop_y, out_w, out_h = geometry
    sx, sy = out_w / crop_w, out_h / crop_h
    return _poly((x0 - crop_x) * sx, (y0 - crop_y) * sy, (x1 - crop_x) * sx, (y1 - crop_y) * sy)


def _one_or_two_lines_in_band(t):
    scores, polys = _one_or_two_lines(t)
    return scores, [_to_band(p[0][0], p[0][1], p[2][0], p[2][1]) for p in polys]


def _cancel_after(n_calls):
    calls = {"n": 0}

    def cancel_check():
        calls["n"] += 1
        return calls["n"] > n_calls

    return cancel_check


def _cancelled_during_audio_extraction(monkeypatch):
    monkeypatch.setattr(crop, "_probe_source", lambda video_path: (1920, 1080, None))

    def cancelled(video_path, duration_sec, window_frac=(0.4, 0.6), cancel_check=None):
        raise crop.vad.AudioExtractionCancelled()

    monkeypatch.setattr(crop.vad, "probe_times", cancelled)
    return crop.detect_crop("dummy.mp4", 60.0, object(), cancel_check=lambda: True)


def _cancelled_in_round_2(monkeypatch):
    state = {"cancelled": False}

    def round_1():
        return [], [1.0, 2.0, 3.0], 0, [1.0, 2.0, 3.0]

    def round_2():
        state["cancelled"] = True
        return [], [4.0, 5.0], 0, [4.0, 5.0]

    result, _calls = _detect_crop_with_mocked_rounds(monkeypatch, [round_1, round_2],
                                                     cancel_check=lambda: state["cancelled"])
    return result


def _random_scenario(seed):
    """A seeded, reproducible scenario: random probe times, random text on
    each frame (dialogue at one or two baselines, one- or two-line, stray
    detections anywhere, scores either side of DT_SCORE_THRESHOLD), and
    randomly a no-speech file, band geometry, consensus, settings or a
    cancellation. What a frame shows depends only on (seed, time), as on a
    real video, so the full-frame retry re-reads the same text."""
    rng = random.Random(f"crop-evidence-scenario:{seed}")
    duration = rng.uniform(30.0, 120.0)
    lo, hi = duration * 0.40, duration * 0.60
    if rng.random() < 0.15:
        vad_times = []
    else:
        vad_times = sorted({round(rng.uniform(lo, hi), 3) for _ in range(rng.randint(1, 45))})
    real_geometry = rng.random() < 0.4
    alt_baseline = rng.choice([1030, 1030, 850, 700])
    text_rate = rng.choice([0.0, 0.1, 0.5, 0.9, 1.0])
    top_only = rng.random() < 0.1

    def predict_fn(t):
        frame = random.Random(f"crop-evidence-frame:{seed}:{t!r}")
        scores, rects = [], []
        if frame.random() < text_rate:
            bottom = 1030 if frame.random() < 0.8 else alt_baseline
            if top_only:
                bottom = 150
            height = frame.randint(40, 60)
            x0 = frame.randint(250, 600)
            x1 = frame.randint(1300, 1650)
            rects.append((x0, bottom - height + frame.randint(-3, 3), x1, bottom + frame.randint(-3, 3)))
            scores.append(frame.uniform(0.85, 1.0))
            if frame.random() < 0.3:
                gap = frame.randint(5, 15)
                rects.append((x0 + 40, bottom - 2 * height - gap, x1 - 40, bottom - height - gap))
                scores.append(frame.uniform(0.85, 1.0))
        if frame.random() < 0.2:
            y0 = frame.randint(0, 1000)
            x0 = frame.randint(0, 1700)
            rects.append((x0, y0, x0 + frame.randint(20, 200), y0 + frame.randint(10, 70)))
            scores.append(frame.uniform(0.5, 1.0))
        if real_geometry:
            polys = [_to_band(*r) for r in rects]
        else:
            polys = [_poly(*r) for r in rects]
        return scores, polys

    kwargs = {}
    if rng.random() < 0.3:
        kwargs["consensus"] = [(0.9, 0.05), (0.91, 0.05), (0.905, 0.1)]
    if rng.random() < 0.2:
        kwargs["settings"] = {"bottom_half_cutoff": rng.choice([0.5, 0.55, 0.7]),
                              "crop_vertical_padding": rng.choice([0.0, 0.003, 0.01])}
    cancel_after = rng.randint(1, 6) if rng.random() < 0.2 else None

    def run(monkeypatch):
        cancel_check = _cancel_after(cancel_after) if cancel_after is not None else None
        return _detect_crop_with_fakes(monkeypatch, vad_times, predict_fn, duration=duration,
                                       cancel_check=cancel_check, real_geometry=real_geometry, **kwargs)

    return run


RANDOM_SEEDS = range(40)

IDENTITY_CASES = {
    "one-line": lambda mp: _detect_crop_with_fakes(mp, VAD_TIMES, _one_line),
    "two-line": lambda mp: _detect_crop_with_fakes(mp, VAD_TIMES, _one_or_two_lines),
    "two-line-band-geometry": lambda mp: _detect_crop_with_fakes(
        mp, VAD_TIMES, _one_or_two_lines_in_band, real_geometry=True),
    "watermark-confirmed": lambda mp: _detect_crop_with_fakes(
        mp, crop._uniform_probe_times(60.0), lambda t: ([1.0], [_poly(1600, 1000, 1850, 1040)])),
    "watermark-uncertain": lambda mp: _detect_crop_with_fakes(
        mp, [10.0, 10.75, 11.5, 12.25, 13.0], lambda t: ([1.0], [_poly(1600, 1000, 1850, 1040)])),
    "no-speech-fallback": lambda mp: _detect_crop_with_fakes(mp, [], _one_line),
    "speech-probes-exhausted": lambda mp: _detect_crop_with_fakes(
        mp, [1.0, 2.0, 3.0], lambda t: _one_line(t) if t >= 24.0 and int(t) % 2 == 0 else ([], [])),
    "cancelled-partial-box": lambda mp: _detect_crop_with_fakes(
        mp, [float(t) for t in range(20, 40)], lambda t: ([1.0], [_poly(400 + int(t), 980, 1500, 1030)]),
        cancel_check=_cancel_after(1)),
    "cancelled-before-first-batch": lambda mp: _detect_crop_with_fakes(
        mp, VAD_TIMES, _one_line, cancel_check=lambda: True),
    "cancelled-during-audio-extraction": _cancelled_during_audio_extraction,
    "cancelled-in-round-2-mocked": _cancelled_in_round_2,
    "full-frame-retry": lambda mp: _detect_crop_with_fakes(
        mp, [1.0, 2.0, 3.0], lambda t: ([1.0], [_poly(700 + _jitter(t), 100, 1200, 150)])),
    "full-frame-retry-band-geometry": lambda mp: _detect_crop_with_fakes(
        mp, [1.0, 2.0, 3.0], lambda t: ([1.0], [_poly(300 + _jitter(t), 2, 700, 20)]), real_geometry=True),
    "no-hits-anywhere": lambda mp: _detect_crop_with_fakes(mp, [1.0, 2.0, 3.0], lambda t: ([], [])),
    "no-speech-no-hits": lambda mp: _detect_crop_with_fakes(mp, [], lambda t: ([], [])),
    "ceiling-exceeded": lambda mp: _detect_crop_with_fakes(
        mp, crop._uniform_probe_times(60.0),
        lambda t: ([1.0], [_poly(100 + int(t * 10) % 300, 600, 1800 + int(t * 10) % 300, 1070)])),
    "outlier-discarded": lambda mp: _detect_crop_with_fakes(
        mp, VAD_TIMES, lambda t: ([1.0], [_poly(500, 700, 1400, 740)]) if t == VAD_TIMES[4] else _one_line(t)),
    "multiple-positions": lambda mp: _detect_crop_with_fakes(
        mp, VAD_TIMES,
        lambda t: ([1.0], [_poly(500 + _jitter(t), 800, 1400, 850)]) if t in VAD_TIMES[2:6] else _one_line(t)),
    "adjacent-upper-line-singleton": lambda mp: _detect_crop_with_fakes(
        mp, VAD_TIMES, lambda t: ([1.0], [_poly(450, 920, 1450, 970)]) if t == VAD_TIMES[0] else _one_line(t)),
    "low-agreement": lambda mp: _detect_crop_with_fakes(
        mp, VAD_TIMES, lambda t: _one_line(t) if t in VAD_TIMES[:2] else ([], [])),
    "scores-and-band-filtering": lambda mp: _detect_crop_with_fakes(
        mp, VAD_TIMES,
        lambda t: ([1.0, 0.89, 0.9, 0.97],
                   [_poly(400 + _jitter(t), 980, 1500, 1030), _poly(300, 880, 1600, 930),
                    _poly(600, 1035, 900, 1060), _poly(100, 200, 400, 260)])),
    "consensus": lambda mp: _detect_crop_with_fakes(
        mp, VAD_TIMES, _one_line, consensus=[(977 / 1080, 56 / 1080)] * 3),
    "settings": lambda mp: _detect_crop_with_fakes(
        mp, VAD_TIMES, _one_or_two_lines,
        settings={"bottom_half_cutoff": 0.7, "crop_vertical_padding": 0.01,
                  "crop_width_fraction": 0.8, "crop_min_height_fraction": 0.03}),
    "4k-persistent-fetch": lambda mp: _detect_crop_at_resolution(mp, (3840, 2160))[0],
    "narrow-one-shot-fetch": lambda mp: _detect_crop_at_resolution(mp, (1920, 888))[0],
    **{f"random-{seed:02d}": _random_scenario(seed) for seed in RANDOM_SEEDS},
}


def _pre_existing_fields(result: crop.CropResult) -> dict:
    """Each pre-existing field as {type, value}, JSON-able: the type name
    keeps a tuple from turning into a list unnoticed, and comparing the
    canonical JSON text keeps an int from turning into a float."""
    record = {}
    for name in PRE_EXISTING_FIELDS:
        value = getattr(result, name)
        plain = list(value) if isinstance(value, (tuple, list)) else value
        record[name] = {"type": type(value).__name__, "value": plain}
    return record


def _canonical(record) -> str:
    return json.dumps(record, sort_keys=True)


def test_identity_fixture_covers_exactly_the_identity_cases():
    recorded = json.loads(IDENTITY_FIXTURE.read_text())["cases"]
    assert sorted(recorded) == sorted(IDENTITY_CASES)


@pytest.mark.parametrize("name", sorted(IDENTITY_CASES))
def test_pre_existing_crop_result_fields_match_the_recording_from_before_evidence(monkeypatch, name):
    expected = json.loads(IDENTITY_FIXTURE.read_text())["cases"][name]
    result = IDENTITY_CASES[name](monkeypatch)
    assert _canonical(_pre_existing_fields(result)) == _canonical(expected)
