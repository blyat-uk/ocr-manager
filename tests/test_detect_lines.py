"""Random subtitle lines for the Brightness gallery (core/detect/lines.py).

The pure tests replace the frame source (ocr_view.video_duration and
ocr_view.grab_ocr_strips_at) with synthetic strips -- a line of white glyph
blocks (a 250 core with a 180 rim) on a dark background, two such lines, a
3x3 speck, or nothing -- and PaddleOCR's detection engine with a fake that
boxes the bright pixels it is handed, so the sampler's own
brightness._glyph_level rule decides what a speck is. Each strip's kind is
chosen by the test from the requested time and its position in the round.
"""
import json

import cv2
import numpy as np
import pytest

from core.detect import brightness as B
from core.detect import lines as L
from core.detect import ocr_view as OV

H, W = 60, 640
BG, CORE, RIM = 30, 250, 180
CROP = (0, 800, 1920, 60)
DURATION = 1000.0
SPANS = [(100.0, 900.0)]       # keep_spans(DURATION, None): the file minus its first and last 10%


def _glyph_row(img, y0, height):
    for k in range(10):
        x0 = 150 + k * 30
        img[y0:y0 + height, x0:x0 + 22] = RIM
        img[y0 + 1:y0 + height - 1, x0 + 1:x0 + 21] = CORE


def _strip(kind: str) -> np.ndarray:
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    if kind == "text":
        _glyph_row(img, 10, 40)
    elif kind == "two_line":
        _glyph_row(img, 4, 22)
        _glyph_row(img, 33, 22)
    elif kind == "speck":
        img[30:33, 300:303] = CORE
    else:
        assert kind == "empty"
    return img


def _polys(strip: np.ndarray) -> list[np.ndarray]:
    """One box per row of bright blocks, 2px around the blocks' own pixels
    (glyphs in a row merge; rows and specks stay apart)."""
    bright = (strip.min(axis=2) > 100).astype(np.uint8)
    merged = cv2.dilate(bright, np.ones((1, 15), np.uint8))
    n, labels = cv2.connectedComponents(merged)
    polys = []
    for k in range(1, n):
        ys, xs = np.nonzero((labels == k) & (bright > 0))
        x0, x1, y0, y1 = xs.min() - 2, xs.max() + 2, ys.min() - 2, ys.max() + 2
        polys.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32))
    return polys


class _FakeDet:
    def __init__(self):
        self.batches: list[int] = []

    def predict(self, strips):
        self.batches.append(len(strips))
        return [{"dt_polys": _polys(s)} for s in strips]


class _Source:
    """The frame source. `kind(t, i)` -> the strip kind at time t, the i-th
    time of its round; `drop(t, i)` -> the time cannot be decoded. Records
    every round's requested times."""

    def __init__(self, monkeypatch, kind=lambda t, i: "text", drop=lambda t, i: False, duration=DURATION):
        self.kind, self.drop = kind, drop
        self.calls: list[list[float]] = []
        self.returned: list[list[float]] = []
        monkeypatch.setattr(OV, "video_duration", lambda path: duration)
        monkeypatch.setattr(OV, "grab_ocr_strips_at", self.grab)

    def grab(self, video_path, crop_box, times):
        assert video_path == "v.mp4"
        assert crop_box == CROP
        self.calls.append(list(times))
        out = [(t, _strip(self.kind(t, i))) for i, t in enumerate(times) if not self.drop(t, i)]
        self.returned.append([t for t, _ in out])
        return out

    @property
    def grabbed(self) -> list[float]:
        return [t for call in self.calls for t in call]


def _run(time_ranges=None, speech=None, *, det=None, **kwargs):
    kwargs.setdefault("seed", 1)
    return L.sample_lines("v.mp4", CROP, time_ranges, speech, det or _FakeDet(), **kwargs)


def _in_middle_half(t: float, pieces) -> bool:
    return any(s + 0.25 * (e - s) - 1e-3 <= t <= s + 0.75 * (e - s) + 1e-3 for s, e in pieces)


def _in_spans(t: float, spans) -> bool:
    return any(s <= t <= e for s, e in spans)


# --------------------------------------------------------------------------
# The fake itself
# --------------------------------------------------------------------------

def test_the_fake_strips_measure_as_the_sampler_expects():
    """Glyph rows are measurable text, a speck is boxed but too small to
    measure, an empty strip gets no box: the premise of the tests below."""
    for kind, rows in (("text", 1), ("two_line", 2)):
        strip = _strip(kind)
        polys = _polys(strip)
        assert len(polys) == rows
        assert B._glyph_level(strip, polys) is not None
    speck = _strip("speck")
    assert len(_polys(speck)) == 1
    assert B._glyph_level(speck, _polys(speck)) is None
    assert _polys(_strip("empty")) == []


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------

def test_the_same_seed_draws_the_same_lines(monkeypatch):
    kind = lambda t, i: "text" if i % 3 == 0 else "empty"
    first_source = _Source(monkeypatch, kind=kind)
    first = _run(seed=42)
    second_source = _Source(monkeypatch, kind=kind)
    second = _run(seed=42)

    assert first == second
    assert first_source.calls == second_source.calls
    assert first.seed == 42 and first.samples


def test_different_seeds_draw_different_times(monkeypatch):
    a = _Source(monkeypatch)
    _run(seed=1)
    b = _Source(monkeypatch)
    _run(seed=2)
    assert a.calls != b.calls


def test_the_seed_is_echoed_and_samples_come_in_time_order(monkeypatch):
    source = _Source(monkeypatch)
    result = _run(seed=1234)
    times = [s.time for s in result.samples]
    assert result.seed == 1234
    assert times == sorted(times)
    assert set(times) <= set(source.grabbed)


# --------------------------------------------------------------------------
# Where candidates are drawn
# --------------------------------------------------------------------------

def test_ample_speech_draws_only_from_the_middle_of_speech(monkeypatch):
    source = _Source(monkeypatch, kind=lambda t, i: "empty")
    long_piece, short_piece = (100.0, 500.0), (600.0, 640.0)
    _run(speech=[list(long_piece), list(short_piece)])

    times = source.grabbed
    assert len(times) == L.MAX_ROUNDS * L.ROUND_SIZE
    assert all(_in_middle_half(t, [long_piece, short_piece]) for t in times)
    # Weighted by length, 400 s against 40 s: about 1 in 11 from the short
    # piece. Both have room for a whole round, so an unweighted pick would
    # put about half the times there.
    in_short = sum(_in_middle_half(t, [short_piece]) for t in times)
    assert 1 <= in_short <= len(times) // 4


def test_speech_is_clipped_to_the_keep_ranges(monkeypatch):
    source = _Source(monkeypatch, kind=lambda t, i: "empty")
    # 100 s of speech, of which only 50..80 lies inside the keep range.
    _run(time_ranges=[{"start": "00:50", "end": "02:00"}], speech=[[0.0, 80.0], [130.0, 150.0]])

    assert source.grabbed
    assert all(_in_middle_half(t, [(50.0, 80.0)]) for t in source.grabbed)


def test_exactly_min_speech_sec_of_speech_is_enough(monkeypatch):
    source = _Source(monkeypatch, kind=lambda t, i: "empty")
    _run(speech=[[400.0, 400.0 + L.MIN_SPEECH_SEC]])
    assert source.grabbed
    assert all(_in_middle_half(t, [(400.0, 400.0 + L.MIN_SPEECH_SEC)]) for t in source.grabbed)


def test_thin_speech_falls_back_to_uniform_over_the_keep_spans(monkeypatch):
    thin = [[200.0, 200.0 + L.MIN_SPEECH_SEC / 2]]
    source = _Source(monkeypatch, kind=lambda t, i: "empty")
    _run(speech=thin, seed=5)
    times = source.grabbed

    assert len(times) == L.MAX_ROUNDS * L.ROUND_SIZE
    assert all(_in_spans(t, SPANS) for t in times)
    assert sum(200.0 <= t <= 205.0 for t in times) <= 2
    assert min(times) < 300.0 and max(times) > 700.0

    # No speech at all, and ample speech lying wholly outside the keep spans,
    # draw exactly the same uniform times for the same seed.
    for speech in (None, [], [[0.0, 90.0]]):
        again = _Source(monkeypatch, kind=lambda t, i: "empty")
        _run(speech=speech, seed=5)
        assert again.calls == source.calls


def test_uniform_draws_stay_inside_several_keep_spans(monkeypatch):
    ranges = [{"start": "01:00", "end": "01:30"}, {"start": "10:00", "end": "10:40"}]
    source = _Source(monkeypatch, kind=lambda t, i: "empty")
    _run(time_ranges=ranges)
    spans = [(60.0, 90.0), (600.0, 640.0)]
    assert all(_in_spans(t, spans) for t in source.grabbed)
    assert any(t < 100 for t in source.grabbed) and any(t > 500 for t in source.grabbed)


# --------------------------------------------------------------------------
# Exclusions and the minimum gap
# --------------------------------------------------------------------------

def _pairwise_gaps_ok(times) -> bool:
    return all(abs(a - b) >= L.MIN_GAP_SEC for k, a in enumerate(times) for b in times[:k])


def test_candidates_keep_their_distance_from_excluded_times_and_each_other(monkeypatch):
    source = _Source(monkeypatch, kind=lambda t, i: "empty")
    exclude = [110.0, 130.0, 150.0]
    _run(time_ranges=[{"start": "01:40", "end": "02:40"}], exclude=exclude)

    assert len(source.calls) == L.MAX_ROUNDS
    for call in source.calls:
        assert call
        assert _pairwise_gaps_ok(call)
        assert all(abs(t - x) >= L.MIN_GAP_SEC for t in call for x in exclude)


def test_later_rounds_keep_their_distance_from_lines_already_held(monkeypatch):
    # The first time of every round is a line; the rest are empty. A 30 s
    # span packs each round tight, so a collision would be likely.
    source = _Source(monkeypatch, kind=lambda t, i: "text" if i == 0 else "empty")
    result = _run(time_ranges=[(100.0, 130.0)], seed=3)

    held = [call[0] for call in source.calls]
    assert [s.time for s in result.samples] == sorted(held)
    for k, call in enumerate(source.calls):
        assert all(abs(t - line) >= L.MIN_GAP_SEC for t in call for line in held[:k])


def test_a_round_without_room_is_smaller(monkeypatch):
    source = _Source(monkeypatch, kind=lambda t, i: "empty")
    _run(time_ranges=[(100.0, 106.0)])
    assert source.calls
    for call in source.calls:
        assert 1 <= len(call) < L.ROUND_SIZE
        assert _pairwise_gaps_ok(call)


def test_no_room_at_all_grabs_nothing(monkeypatch):
    source = _Source(monkeypatch)
    det = _FakeDet()
    result = _run(time_ranges=[(100.0, 104.0)], exclude=[102.0], det=det)
    assert result.samples == () and result.tried == 0 and not result.cancelled
    assert source.calls == [] and det.batches == []


# --------------------------------------------------------------------------
# What counts as a line; rounds and stopping
# --------------------------------------------------------------------------

def test_specks_are_not_lines(monkeypatch):
    _Source(monkeypatch, kind=lambda t, i: "speck")
    result = _run()
    assert result.samples == ()
    assert result.tried == L.MAX_ROUNDS * L.ROUND_SIZE


def test_only_measurable_text_is_kept(monkeypatch):
    kinds = ("speck", "text", "empty", "two_line")
    source = _Source(monkeypatch, kind=lambda t, i: kinds[i % 4])
    result = _run()

    by_time = {t: kinds[i % 4] for call in source.calls for i, t in enumerate(call)}
    assert result.samples
    assert all(by_time[s.time] in ("text", "two_line") for s in result.samples)


def test_boxes_and_rows_come_from_the_detection_polygons(monkeypatch):
    source = _Source(monkeypatch, kind=lambda t, i: "two_line" if i == 0 else "text")
    result = _run(count=L.ROUND_SIZE)

    two_line_time = source.calls[0][0]
    for sample in result.samples:
        kind = "two_line" if sample.time == two_line_time else "text"
        expected = tuple(B._poly_box(p) for p in _polys(_strip(kind)))
        assert sample.boxes == expected
        assert sample.lines == (2 if kind == "two_line" else 1)
        assert all(isinstance(v, int) for box in sample.boxes for v in box)


def test_it_stops_once_count_lines_are_held(monkeypatch):
    source = _Source(monkeypatch)
    det = _FakeDet()
    result = _run(det=det)

    assert len(result.samples) == L.LINE_COUNT
    assert len(source.calls) == 1 and len(source.calls[0]) == L.ROUND_SIZE
    assert det.batches == [L.ROUND_SIZE]
    assert result.tried == L.ROUND_SIZE
    # The first `count` lines in draw order are kept, reported in time order.
    assert [s.time for s in result.samples] == sorted(source.calls[0][:L.LINE_COUNT])


def test_count_caps_the_lines_kept(monkeypatch):
    _Source(monkeypatch)
    assert len(_run(count=2).samples) == 2
    _Source(monkeypatch)
    assert _run(count=0) == L.LinesResult(samples=(), tried=0, seed=1)


def test_it_gives_up_after_max_rounds_with_fewer_lines(monkeypatch):
    source = _Source(monkeypatch, kind=lambda t, i: "text" if i == 0 else "empty")
    det = _FakeDet()
    result = _run(det=det)

    assert len(source.calls) == L.MAX_ROUNDS
    assert det.batches == [L.ROUND_SIZE] * L.MAX_ROUNDS
    assert len(result.samples) == L.MAX_ROUNDS < L.LINE_COUNT
    assert result.tried == L.MAX_ROUNDS * L.ROUND_SIZE
    assert not result.cancelled


def test_undecodable_times_are_not_tried(monkeypatch):
    source = _Source(monkeypatch, kind=lambda t, i: "text" if i < 2 else "empty",
                     drop=lambda t, i: i % 2 == 1)
    det = _FakeDet()
    result = _run(det=det)

    returned = [len(r) for r in source.returned]
    assert det.batches == returned
    assert result.tried == sum(returned) == L.MAX_ROUNDS * L.ROUND_SIZE // 2
    # Position 1 of every round was the second line, but it never decoded.
    assert [s.time for s in result.samples] == sorted(call[0] for call in source.calls)


def test_a_round_that_decodes_nothing_runs_no_detection(monkeypatch):
    source = _Source(monkeypatch, drop=lambda t, i: True)
    det = _FakeDet()
    result = _run(det=det)
    assert len(source.calls) == L.MAX_ROUNDS
    assert det.batches == []
    assert result.samples == () and result.tried == 0


# --------------------------------------------------------------------------
# Cancel
# --------------------------------------------------------------------------

def test_cancel_before_the_first_round_grabs_nothing(monkeypatch):
    source = _Source(monkeypatch)
    det = _FakeDet()
    result = _run(det=det, seed=9, cancel_check=lambda: True)
    assert result == L.LinesResult(samples=(), tried=0, seed=9, cancelled=True)
    assert source.calls == [] and det.batches == []


def test_cancel_between_rounds_returns_no_samples(monkeypatch):
    source = _Source(monkeypatch, kind=lambda t, i: "text" if i == 0 else "empty")
    polls = iter([False, True])
    result = _run(cancel_check=lambda: next(polls))

    assert result.cancelled
    assert result.samples == ()                 # the round-1 line is not trusted
    assert result.tried == L.ROUND_SIZE
    assert len(source.calls) == 1


def test_a_finished_call_does_not_poll_again(monkeypatch):
    _Source(monkeypatch)
    polls = []
    result = _run(cancel_check=lambda: polls.append(1) or False)
    assert len(polls) == 1
    assert len(result.samples) == L.LINE_COUNT and not result.cancelled


# --------------------------------------------------------------------------
# Nothing to sample
# --------------------------------------------------------------------------

@pytest.mark.parametrize("time_ranges", [
    [{"start": "30:00", "end": "40:00"}],       # after the end of the file
    [{"start": "10:00", "end": "05:00"}],       # ends before it starts
    [(300.0, 300.0)],                           # no length
])
def test_keep_ranges_that_select_nothing_give_no_samples(monkeypatch, time_ranges):
    source = _Source(monkeypatch)
    det = _FakeDet()
    result = _run(time_ranges=time_ranges, det=det, seed=4)
    assert result == L.LinesResult(samples=(), tried=0, seed=4)
    assert source.calls == [] and det.batches == []


def test_a_file_without_duration_gives_no_samples(monkeypatch):
    source = _Source(monkeypatch, duration=0.0)
    result = _run()
    assert result.samples == () and result.tried == 0
    assert source.calls == []


def test_a_file_that_cannot_be_opened_gives_no_samples(monkeypatch):
    source = _Source(monkeypatch)

    def unopenable(path):
        raise OSError("no such file")

    monkeypatch.setattr(OV, "video_duration", unopenable)
    det = _FakeDet()
    result = _run(det=det, seed=8)
    assert result == L.LinesResult(samples=(), tried=0, seed=8)
    assert source.calls == [] and det.batches == []


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

def test_to_evidence_is_plain_json(monkeypatch):
    _Source(monkeypatch, kind=lambda t, i: "two_line" if i == 0 else "text")
    result = _run(seed=77)
    evidence = result.to_evidence()

    assert json.loads(json.dumps(evidence)) == evidence
    assert set(evidence) == {"seed", "tried", "samples"}
    assert evidence["seed"] == 77 and evidence["tried"] == result.tried
    assert len(evidence["samples"]) == len(result.samples) == L.LINE_COUNT
    for sample, row in zip(result.samples, evidence["samples"]):
        assert set(row) == {"time", "boxes", "lines"}
        assert row["time"] == sample.time
        assert row["boxes"] == [list(box) for box in sample.boxes]
        assert all(len(box) == 4 for box in row["boxes"])
        assert row["lines"] == sample.lines


# --------------------------------------------------------------------------
# A real episode (read only)
# --------------------------------------------------------------------------

@pytest.mark.needs_media
def test_it_finds_lines_on_a_reference_episode(reference_media):
    slay = reference_media.get("slay")
    if slay is None or not slay["crop"]:
        pytest.skip("the slay reference project is not present")
    from videocr import engine_registry
    from videocr.utils import suppress_output

    # The keep ranges the pipeline would pass, from the project's own
    # .ocr.json (read only).
    config = json.loads((slay["dir"] / ".ocr.json").read_text())
    ranges = (config.get("files", {}).get(slay["video"].name) or {}).get("time_ranges") or None
    box = tuple(slay["crop"])
    with suppress_output(), engine_registry.lease_detection_engine(None, True) as det:
        result = L.sample_lines(str(slay["video"]), box, ranges, None, det, seed=2026)

    duration = OV.video_duration(str(slay["video"]))
    spans = OV.keep_spans(duration, ranges)
    print(f"slay: {len(result.samples)} line(s) in {result.tried} strip(s): "
          + ", ".join(f"{s.time:.1f}s x{s.lines}" for s in result.samples))
    assert not result.cancelled
    assert 1 <= len(result.samples) <= L.LINE_COUNT
    assert result.tried <= L.MAX_ROUNDS * L.ROUND_SIZE
    times = [s.time for s in result.samples]
    assert times == sorted(times)
    assert all(_in_spans(t, spans) for t in times)
    # Boxes lie on the strip the view will re-fetch at the same time.
    strips = dict(OV.grab_ocr_strips_at(str(slay["video"]), box, times))
    for sample in result.samples:
        h, w = strips[sample.time].shape[:2]
        assert sample.boxes and sample.lines >= 1
        for x, y, bw, bh in sample.boxes:
            assert -2 <= x and -2 <= y and x + bw <= w + 2 and y + bh <= h + 2
