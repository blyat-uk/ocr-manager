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
    # Boundaries: a polygon centred exactly on the cutoff row (594 at 1080p)
    # is in band; a padded box of 264 px sits between 24% and 25% of 1080;
    # a second-batch frame 13 px lower moves the union by more than 1% but
    # less than CONVERGENCE_VERTICAL_TOLERANCE_FRAC (1.5%) of frame height.
    "text-centred-on-the-cutoff": lambda mp: _detect_crop_with_fakes(
        mp, VAD_TIMES, lambda t: ([1.0, 1.0], [_one_line(t)[1][0], _poly(700, 574, 900, 614)])),
    "just-under-the-height-ceiling": lambda mp: _detect_crop_with_fakes(
        mp, VAD_TIMES, lambda t: ([1.0], [_poly(400 + _jitter(t), 770, 1500, 1028)])),
    "union-drift-inside-convergence-tolerance": lambda mp: _detect_crop_with_fakes(
        mp, VAD_TIMES,
        lambda t: ([1.0], [_poly(400 + _jitter(t), 980, 1500, 1043)]) if t == crop._spread_order(VAD_TIMES)[5]
        else _one_line(t)),
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


# --- CropResult.samples ------------------------------------------------------

FULL_FRAME_GEOMETRY = (1920, 1080) + crop._crop_geometry(1920, 1080, 1.0, crop.TARGET_HEIGHT)


def _box_in_full_frame(rect, geometry):
    """(x, y, w, h) full-frame pixels of a rectangle given in a grab's
    coordinates, rounded like CropResult.envelope."""
    _w, _h, crop_w, crop_h, crop_x, crop_y, out_w, out_h = geometry
    sx, sy = crop_w / out_w, crop_h / out_h
    x0, y0 = rect[0] * sx + crop_x, rect[1] * sy + crop_y
    x1, y1 = rect[2] * sx + crop_x, rect[3] * sy + crop_y
    return (round(x0), round(y0), round(x1 - x0), round(y1 - y0))


def _xywh(x0, y0, x1, y1):
    return (x0, y0, x1 - x0, y1 - y0)


def test_samples_carry_each_probed_frames_boxes_whether_it_was_kept_and_its_text_rows(monkeypatch):
    """One sample per probed frame, in sample_pts order: the frame's accepted
    in-band boxes (score >= DT_SCORE_THRESHOLD, centre below the cutoff)
    sorted by (y, x), whether it contributed to the kept union, and how many
    text rows those boxes form."""
    two_line, same_row, empty, outlier, filtered = crop._spread_order(VAD_TIMES)[:crop.PROBE_BATCH_SIZE]
    content = {
        # two rows, listed bottom row first: samples sort by (y, x)
        two_line: ([1.0, 1.0], [(400, 980, 1500, 1030), (450, 920, 1450, 970)]),
        # two boxes on one row, the right one listed first and slightly lower
        same_row: ([1.0, 1.0], [(950, 985, 1500, 1032), (400, 980, 900, 1030)]),
        empty: ([], []),
        # its own baseline far above the subtitle's: left out of the union
        outlier: ([1.0], [(500, 700, 1400, 740)]),
        # score 0.89 is rejected, 0.9 accepted, and text above the cutoff is out of band
        filtered: ([0.89, 1.0, 0.97, 0.9],
                   [(300, 880, 1600, 930), (430, 980, 1470, 1030), (100, 200, 400, 260), (1600, 990, 1700, 1025)]),
    }

    def predict_fn(t):
        if t in content:
            scores, rects = content[t]
            return scores, [_poly(*r) for r in rects]
        j = _jitter(t)
        return [1.0], [_poly(400 + j, 980, 1500 - j, 1030)]

    result = _detect_crop_with_fakes(monkeypatch, VAD_TIMES, predict_fn)

    assert result.box is not None and crop.FLAG_OUTLIER_DISCARDED in result.flagged
    assert set(content) <= set(result.sample_pts)
    expected = {
        two_line: crop.CropSample(time=two_line, boxes=(_xywh(450, 920, 1450, 970), _xywh(400, 980, 1500, 1030)),
                                  kept=True, lines=2),
        same_row: crop.CropSample(time=same_row, boxes=(_xywh(400, 980, 900, 1030), _xywh(950, 985, 1500, 1032)),
                                  kept=True, lines=1),
        empty: crop.CropSample(time=empty, boxes=(), kept=False, lines=0),
        outlier: crop.CropSample(time=outlier, boxes=(_xywh(500, 700, 1400, 740),), kept=False, lines=1),
        filtered: crop.CropSample(time=filtered, boxes=(_xywh(430, 980, 1470, 1030), _xywh(1600, 990, 1700, 1025)),
                                  kept=True, lines=1),
    }
    for t in result.sample_pts:
        if t not in expected:
            j = _jitter(t)
            expected[t] = crop.CropSample(time=t, boxes=(_xywh(400 + j, 980, 1500 - j, 1030),), kept=True, lines=1)
    assert result.samples == [expected[t] for t in result.sample_pts]
    # No time was probed twice here, so kept is exactly hit_pts membership.
    assert [s.kept for s in result.samples] == [s.time in result.hit_pts for s in result.samples]


def test_sample_boxes_are_full_frame_pixels_not_the_grabbed_bands(monkeypatch):
    """Detection runs on the bottom band, cropped and downscaled; the boxes
    are reported in the source frame's native pixels, like `box`."""
    def rect(t):
        j = _jitter(t)
        return (200 + j, 400, 1200 - j, 441)

    result = _detect_crop_with_fakes(monkeypatch, VAD_TIMES, lambda t: ([1.0], [_poly(*rect(t))]),
                                     real_geometry=True)

    assert result.box is not None and result.samples
    for sample in result.samples:
        assert sample.boxes == (_box_in_full_frame(rect(sample.time), BAND_GEOMETRY),)
        assert sample.kept and sample.lines == 1


def test_a_time_the_full_frame_retry_probes_again_is_kept_only_where_it_contributed(monkeypatch):
    """The full-frame retry re-probes times the bottom-band rounds already
    probed, so sample_pts lists those times twice. Only the retry's frame
    contributed; the band round's frame at the same time had nothing in band
    and is not kept, even though its time is in hit_pts."""
    def rect(t):
        return (300 + _jitter(t), 3, 700, 22)   # top of the frame, in grab coordinates

    result = _detect_crop_with_fakes(monkeypatch, [1.0, 2.0, 3.0], lambda t: ([1.0], [_poly(*rect(t))]),
                                     real_geometry=True)

    assert result.box is not None and crop.FLAG_TOP_POSITIONED in result.flagged
    band_rounds = 3 + len(crop._uniform_probe_times(60.0))
    assert len(result.sample_pts) > band_rounds
    for sample in result.samples[:band_rounds]:
        assert sample.boxes == () and not sample.kept and sample.lines == 0
    for sample in result.samples[band_rounds:]:
        assert sample.boxes == (_box_in_full_frame(rect(sample.time), FULL_FRAME_GEOMETRY),)
        assert sample.kept
    assert any(s.time in result.hit_pts for s in result.samples[:band_rounds])
    assert sorted(s.time for s in result.samples if s.kept) == result.hit_pts


def test_a_probe_the_engine_left_unanswered_gets_no_boxes_rather_than_the_next_frames():
    """_run_round() records a sample for every fetched frame but analyses
    only the frames the engine returned a result for, so an engine answering
    [1, 2] of a batch [1, 2, 3] and [4] of [4, 5] leaves frame_times
    [1, 2, 4]. Each analysed frame stays with its own sample."""
    line, upper = _poly(400, 980, 1500, 1030), _poly(400, 900, 1500, 950)
    rounds = [([1.0, 2.0, 3.0, 4.0, 5.0], [[line], [line], [upper]], [1.0, 2.0, 4.0], False)]

    samples = crop._crop_samples(rounds, 1080, crop.BOTTOM_HALF_CUTOFF, kept_idx=[0, 2])

    assert samples == [
        crop.CropSample(time=1.0, boxes=(_xywh(400, 980, 1500, 1030),), kept=True, lines=1),
        crop.CropSample(time=2.0, boxes=(_xywh(400, 980, 1500, 1030),), kept=False, lines=1),
        crop.CropSample(time=3.0, boxes=(), kept=False, lines=0),
        crop.CropSample(time=4.0, boxes=(_xywh(400, 900, 1500, 950),), kept=True, lines=1),
        crop.CropSample(time=5.0, boxes=(), kept=False, lines=0),
    ]


@pytest.mark.parametrize("name", sorted(IDENTITY_CASES))
def test_samples_line_up_with_sample_pts_hit_pts_and_the_envelope(monkeypatch, name):
    result = IDENTITY_CASES[name](monkeypatch)

    assert [s.time for s in result.samples] == result.sample_pts
    kept = [s for s in result.samples if s.kept]
    assert sorted(s.time for s in kept) == result.hit_pts
    assert len(kept) == result.agreed
    for s in result.samples:
        assert all(type(v) is int for box in s.boxes for v in box)
        assert list(s.boxes) == sorted(s.boxes, key=lambda b: (b[1], b[0]))
        assert (s.lines == 0) if not s.boxes else (1 <= s.lines <= len(s.boxes))
    if not kept:
        assert result.envelope is None
        return
    assert all(s.boxes for s in kept)
    # The kept boxes rebuild the envelope, to within their own rounding.
    boxes = [b for s in kept for b in s.boxes]
    union = (min(b[0] for b in boxes), min(b[1] for b in boxes),
             max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes))
    ex, ey, ew, eh = result.envelope
    for got, want in zip(union, (ex, ey, ex + ew, ey + eh)):
        assert abs(got - want) <= 1, (union, result.envelope)


@pytest.mark.parametrize("boxes,rows", [
    ([], 0),
    ([(0, 100, 50, 40)], 1),
    ([(0, 100, 50, 40), (60, 100, 50, 40)], 1),                 # side by side
    ([(0, 920, 1000, 50), (0, 980, 1100, 50)], 2),              # a two-line subtitle
    ([(0, 0, 10, 40), (20, 20, 10, 60)], 1),                    # overlap 20 = half the smaller height
    ([(0, 0, 10, 40), (20, 21, 10, 60)], 2),                    # overlap 19: just under half
    ([(0, 0, 10, 40), (0, 20, 10, 40), (0, 40, 10, 40)], 1),    # chained through the middle box
    ([(0, 0, 10, 40), (0, 15, 10, 40), (0, 100, 10, 40)], 2),   # one row of two, one alone
    ([(0, 0, 10, 40), (0, 30, 10, 40)], 2),                     # overlap 10: a quarter
])
def test_text_rows_join_boxes_overlapping_vertically_by_at_least_half_the_smaller_height(boxes, rows):
    assert crop._count_text_rows(tuple(boxes)) == rows


def test_to_evidence_is_json_serialisable_and_carries_every_sample(monkeypatch):
    result = _detect_crop_with_fakes(monkeypatch, VAD_TIMES, _one_or_two_lines)
    evidence = result.to_evidence()

    assert json.loads(json.dumps(evidence)) == evidence
    assert set(evidence) == {"box", "envelope", "agreed", "probes_used", "flagged", "hit_pts",
                             "frame_size", "samples"}
    assert evidence["box"] == list(result.box)
    assert evidence["envelope"] == list(result.envelope)
    assert evidence["frame_size"] == list(result.frame_size)
    assert (evidence["agreed"], evidence["probes_used"], evidence["flagged"], evidence["hit_pts"]) == (
        result.agreed, result.probes_used, result.flagged, result.hit_pts)
    assert evidence["samples"] == [
        {"time": s.time, "boxes": [list(b) for b in s.boxes], "kept": s.kept, "lines": s.lines}
        for s in result.samples
    ]
    assert any(len(s["boxes"]) == 2 for s in evidence["samples"])


def test_to_evidence_of_a_result_without_a_box_is_json_serialisable(monkeypatch):
    result = _detect_crop_with_fakes(monkeypatch, VAD_TIMES, _one_line, cancel_check=lambda: True)
    evidence = result.to_evidence()

    assert json.loads(json.dumps(evidence)) == evidence
    assert evidence["box"] is None and evidence["envelope"] is None and evidence["samples"] == []


def test_to_evidence_converts_numpy_scalars():
    import numpy as np

    result = crop.CropResult(
        box=tuple(np.int64(v) for v in (288, 977, 1344, 56)), sample_pts=[np.float32(1.5)],
        envelope=tuple(np.int32(v) for v in (400, 980, 1100, 50)), agreed=np.int64(1), probes_used=1,
        hit_pts=[np.float32(1.5)], frame_size=(np.int64(1920), np.int64(1080)),
        samples=[crop.CropSample(time=np.float32(1.5), boxes=((np.int64(400), 980, 1100, 50),),
                                 kept=np.bool_(True), lines=np.int64(1))],
    )
    evidence = result.to_evidence()

    assert json.loads(json.dumps(evidence)) == evidence
    assert evidence["samples"] == [{"time": 1.5, "boxes": [[400, 980, 1100, 50]], "kept": True, "lines": 1}]
