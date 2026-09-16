"""Automatic brightness threshold detection (core/detect/brightness.py).

The pure tests build synthetic crop strips: white glyph-like blocks (a 250
core with a 1px anti-aliased 180 rim) on a dark background, bright
text-free strips for the Laplacian gate, and yellow-text strips for the
coloured-subtitle failure mode. Fake detection/OCR engines stand in for
PaddleOCR; the fake OCR engines read the masked image they are handed, so
nothing here depends on how a batch happens to be ordered.

The mirror tests drive the real `videocr.video.Video.run_ocr` (with a fake
capture, or a real one on encoded clips) and compare what it hands the OCR
engine against this module's own view of the same frames.
"""
import subprocess
import time

import cv2
import numpy as np
import pytest

from core.detect import brightness as B

H, W = 54, 1344
BG = 30
CORE = 250
RIM = 180
# Glyph blocks along one centred text line: 12 blocks, 25x30, 37px pitch.
GLYPH_X0, GLYPH_PITCH, GLYPH_W = 450, 37, 25
GLYPH_Y0, GLYPH_Y1 = 12, 42
N_GLYPHS = 12
TEXT_POLY = np.array([[440, 8], [900, 8], [900, 46], [440, 46]], dtype=np.float32)
CENTRE_X0 = (W - H) // 2  # the gate's centre square, as videocr/video.py computes it


def _glyph_strip(core=(CORE, CORE, CORE), rim=(RIM, RIM, RIM), bg=BG):
    img = np.full((H, W, 3), bg, dtype=np.uint8)
    for k in range(N_GLYPHS):
        x0 = GLYPH_X0 + k * GLYPH_PITCH
        img[GLYPH_Y0:GLYPH_Y1, x0:x0 + GLYPH_W] = rim
        img[GLYPH_Y0 + 1:GLYPH_Y1 - 1, x0 + 1:x0 + GLYPH_W - 1] = core
    return img


def _ramp_strip(top=240):
    """Text-free strip: a smooth horizontal ramp 100..top across the gate's
    centre square, flat dark background elsewhere."""
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    ramp = np.linspace(100, top, H).round().astype(np.uint8)
    img[:, CENTRE_X0:CENTRE_X0 + H] = ramp[None, :, None]
    return img


def _checkerboard(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((yy // 2) + (xx // 2)) % 2 * 255).astype(np.uint8)


def _checker_strip():
    """Text-free strip whose centre square holds pure-white clutter."""
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    img[:, CENTRE_X0:CENTRE_X0 + H] = _checkerboard(H, H)[..., None]
    return img


# --------------------------------------------------------------------------
# analytic_seed
# --------------------------------------------------------------------------

def test_analytic_seed_sits_just_under_the_glyph_core():
    strips = [_glyph_strip() for _ in range(5)]
    seed = B.analytic_seed(strips, [[TEXT_POLY]] * 5)
    # Glyph core 250: the seed sits just under it -- not down at the
    # anti-aliased rim (180), and not at the background.
    assert CORE - 15 <= seed < CORE


@pytest.mark.parametrize("core, expected", [
    (237, 227),   # round5(237) = 235, minus 8
    (238, 232),   # round5(238) = 240, minus 8
    (120, 112),
    (60, 100),    # clamped up
    (255, 245),   # clamped down
])
def test_seed_is_round5_of_the_median_glyph_level_minus_8_clamped(core, expected):
    strips = [_glyph_strip(core=(core,) * 3, rim=(min(core, RIM) - 20,) * 3) for _ in range(3)]
    assert B.analytic_seed(strips, [[TEXT_POLY]] * 3) == expected


def test_analytic_seed_uses_the_median_so_one_hud_frame_cannot_drag_it_down():
    clean = [_glyph_strip() for _ in range(4)]
    hud = _glyph_strip()
    hud[4:50, 950:1250] = 170  # a burned-in HUD panel the detector also boxes
    hud_poly = np.array([[950, 4], [1250, 4], [1250, 50], [950, 50]], dtype=np.float32)
    strips = clean + [hud]
    polys = [[TEXT_POLY]] * 4 + [[TEXT_POLY, hud_poly]]

    seed = B.analytic_seed(strips, polys)

    assert seed == B.analytic_seed(clean, [[TEXT_POLY]] * 4)
    assert seed >= CORE - 15, f"seed collapsed toward the HUD level: {seed}"


def test_polygon_edge_pixels_do_not_count_as_glyph_pixels():
    # One glyph, and a bright outline lying exactly on the detection
    # polygon's edge (a text-box border, a halo). Only the polygon's
    # eroded interior is glyph territory.
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    x0 = GLYPH_X0
    img[GLYPH_Y0:GLYPH_Y1, x0:x0 + GLYPH_W] = RIM
    img[GLYPH_Y0 + 1:GLYPH_Y1 - 1, x0 + 1:x0 + GLYPH_W - 1] = CORE
    img[8, 440:901] = 240
    img[46, 440:901] = 240
    img[8:47, 440] = 240
    img[8:47, 900] = 240
    assert B.analytic_seed([img], [[TEXT_POLY]]) == 242


def test_analytic_seed_ignores_strips_without_polygons():
    strips = [_glyph_strip(), _glyph_strip(), _ramp_strip()]
    assert B.analytic_seed(strips, [[TEXT_POLY], [TEXT_POLY], []]) == \
        B.analytic_seed(strips[:2], [[TEXT_POLY], [TEXT_POLY]])


def test_analytic_seed_refuses_when_no_strip_has_text():
    with pytest.raises(ValueError):
        B.analytic_seed([_ramp_strip()], [[]])


# --------------------------------------------------------------------------
# gate_floor
# --------------------------------------------------------------------------

def test_gate_floor_is_where_empty_strips_stop_tripping_the_laplacian_gate():
    empties = [_ramp_strip(top=240) for _ in range(20)]
    floor = B.gate_floor(empties)
    # Anything kept from the 100..240 ramp leaves a hard edge that trips the
    # gate; once t passes the ramp's brightest column nothing survives. At
    # very low t nothing is cut either (the whole smooth ramp is kept), so a
    # search for the FIRST quiet threshold would wrongly answer ~1 here.
    assert floor == 241
    for t in (floor, floor + 7, 254):
        assert not B._gate_fires(B._mask(empties[0], t))
    assert B._gate_fires(B._mask(empties[0], floor - 1))
    assert not B._gate_fires(B._mask(empties[0], 1))


def test_gate_floor_needs_only_95_percent_of_empty_strips():
    empties = [_ramp_strip(top=240) for _ in range(19)] + [_ramp_strip(top=250)]
    assert B.gate_floor(empties) == 241


def test_gate_floor_only_looks_at_the_centre_square_like_the_ocr_gate():
    strip = _ramp_strip(top=240)
    left = slice(0, CENTRE_X0 - 10)
    strip[:, left] = _checkerboard(H, CENTRE_X0 - 10)[..., None]
    assert B.gate_floor([strip] * 20) == 241


def test_gate_floor_is_none_when_clutter_survives_every_threshold():
    assert B.gate_floor([_checker_strip()] * 10) is None


def test_gate_floor_is_none_without_empty_strips():
    assert B.gate_floor([]) is None


# --------------------------------------------------------------------------
# verify_with_ocr / plateau selection
# --------------------------------------------------------------------------

def _ocr_item(text, conf):
    if not text:
        return {"rec_texts": [], "rec_scores": [], "rec_polys": []}
    poly = np.array([[440, 8], [900, 8], [900, 46], [440, 46]], dtype=np.int16)
    return {"rec_texts": [text], "rec_scores": [conf], "rec_polys": [poly]}


def _probe_strip(identity, n_identities):
    """A glyph strip that tells a fake OCR engine two things from pixels
    alone: which strip it is (a 255 marker at column `identity` of row -3,
    which survives any mask) and which threshold it was masked at (row -1
    holds a 1..255 ramp, whose smallest surviving value IS the threshold)."""
    img = _glyph_strip()
    img[-3, :n_identities] = 0
    img[-3, identity] = 255
    img[-1, :255] = np.arange(1, 256, dtype=np.uint8)[:, None]
    return img


def _masked_threshold(img):
    row = img[-1, :255].min(axis=1)
    kept = row[row > 0]
    return int(kept.min()) if kept.size else 256


class _ScriptedOCR:
    """Reads (strip, threshold) back out of each masked image and answers
    from a per-threshold script -- one script for every strip, or a list of
    per-strip scripts."""

    def __init__(self, script, texts):
        self.scripts = script if isinstance(script, list) else [script] * len(texts)
        self.texts = texts
        self.calls = 0

    def predict(self, images):
        self.calls += 1
        out = []
        for img in images:
            idx = int(np.argmax(img[-3, :len(self.texts), 0]))
            variant, conf = self.scripts[idx].get(_masked_threshold(img), ("", 0.0))
            base = self.texts[idx]
            text = {"ok": base, "short": base[:-1], "wrong": "口口口口", "junk": "A", "": ""}[variant]
            out.append(_ocr_item(text, conf))
        return out


TEXTS = ["你好世界", "再见朋友", "今天下雨", "明天晴天"]


def _probe_strips(n=len(TEXTS)):
    return [_probe_strip(i, n) for i in range(n)]


def test_plateau_selection_picks_a_safe_margin_below_the_top_of_the_widest_valid_run():
    script = {
        175: ("ok", 1.0),                             # the best-scoring single point
        180: ("short", 0.99), 185: ("wrong", 0.99),   # two failing thresholds in a row
        190: ("ok", 0.99), 195: ("ok", 0.99), 200: ("ok", 0.99),
        205: ("ok", 0.99), 210: ("ok", 0.99), 215: ("ok", 0.99),
        220: ("", 0.0), 225: ("", 0.0),               # text gone for good
    }
    engine = _ScriptedOCR(script, TEXTS)

    value, plateau, curve = B.verify_with_ocr(_probe_strips(), 200, engine)

    assert [t for t, _ in curve] == list(range(175, 230, 5))
    assert plateau == (190, 215)
    # Four steps below the top: end-to-end OCR lost short, fading lines
    # 0-10 below the measured top on four reference projects, none at 20.
    assert value == 195
    best_t = max(curve, key=lambda p: p[1])[0]
    assert best_t == 175
    assert engine.calls == 1, "every threshold must be OCR'd in one batch"


def test_a_short_plateau_picks_its_lowest_verified_threshold():
    # The right text is still the modal reading (4 of 11), but it only holds
    # on two consecutive thresholds; four steps below that top is outside it.
    script = {
        175: ("ok", 0.99), 180: ("junk", 0.9), 185: ("wrong", 0.99),
        190: ("", 0.0), 195: ("", 0.0), 200: ("", 0.0),
        205: ("ok", 0.99), 210: ("ok", 0.99),
        215: ("short", 0.99), 220: ("wrong", 0.99), 225: ("ok", 0.99),
    }
    value, plateau, _ = B.verify_with_ocr(_probe_strips(), 200, _ScriptedOCR(script, TEXTS))
    assert plateau == (205, 210)
    assert value == 205


def test_an_isolated_misreading_does_not_split_the_plateau():
    # One strip misreads at a single threshold (a speck read as an extra
    # stroke); the thresholds on both sides read fine. Text lost to a high
    # threshold stays lost at every higher one, so a lone dip is noise.
    steady = {t: ("ok", 0.99) for t in range(175, 230, 5)}
    glitchy = dict(steady)
    glitchy[205] = ("wrong", 0.99)
    engine = _ScriptedOCR([glitchy, steady, steady, steady], TEXTS)

    value, plateau, _ = B.verify_with_ocr(_probe_strips(), 200, engine)

    assert plateau == (175, 225)
    assert value == 205


def test_a_strip_that_reads_text_at_only_one_threshold_carries_no_evidence():
    # A logo or speck that OCR reads once and never again is not a subtitle:
    # it must not become that strip's "modal text" and then count as lost
    # text at every other threshold.
    steady = {t: ("ok", 0.99) for t in range(175, 230, 5)}
    once = {200: ("junk", 0.95)}
    engine = _ScriptedOCR([steady, steady, steady, steady, once], TEXTS + ["标志"])

    value, plateau, _ = B.verify_with_ocr(_probe_strips(5), 200, engine)

    assert plateau == (175, 225)
    assert value == 205


def test_verify_never_scans_thresholds_above_255():
    script = {t: ("ok", 0.99) for t in range(0, 256)}
    _, plateau, curve = B.verify_with_ocr(_probe_strips(), 245, _ScriptedOCR(script, TEXTS))
    assert [t for t, _ in curve] == list(range(220, 256, 5))
    assert plateau == (220, 255)


def test_verify_reports_no_plateau_when_nothing_reads_confidently():
    script = {t: ("ok", 0.80) for t in range(0, 256)}
    value, plateau, _ = B.verify_with_ocr(_probe_strips(), 200, _ScriptedOCR(script, TEXTS))
    assert plateau is None
    assert value == 200


# --------------------------------------------------------------------------
# detect_brightness (engines and frame source faked)
# --------------------------------------------------------------------------

class _FakeDet:
    """Boxes the text line when glyph block 0 is visibly bright in ANY
    channel -- a yellow subtitle is text to a detector, whatever its blue
    channel does."""

    def __init__(self, score=0.97):
        self.calls = 0
        self.score = score

    def predict(self, images):
        self.calls += 1
        out = []
        for img in images:
            probe = img[GLYPH_Y0 + 5, GLYPH_X0 + 5]
            if int(probe.max()) >= 200:
                out.append({"dt_polys": np.array([TEXT_POLY]), "dt_scores": [self.score]})
            else:
                out.append({"dt_polys": np.zeros((0, 4, 2)), "dt_scores": []})
        return out


class _GlyphReadingOCR:
    """Reads the line whenever at least half the glyph cores survive the mask."""

    CORE_PX = N_GLYPHS * (GLYPH_Y1 - GLYPH_Y0 - 2) * (GLYPH_W - 2)

    def __init__(self):
        self.calls = 0
        self.images = 0

    def predict(self, images):
        self.calls += 1
        out = []
        for img in images:
            self.images += 1
            line = img[GLYPH_Y0:GLYPH_Y1, GLYPH_X0:GLYPH_X0 + N_GLYPHS * GLYPH_PITCH]
            kept = np.count_nonzero(line.min(axis=2))
            out.append(_ocr_item("你好世界", 0.99) if kept >= self.CORE_PX // 2
                       else _ocr_item("", 0.0))
        return out


class _ExplodingOCR:
    def predict(self, images):
        raise AssertionError("the OCR engine must not be used on this path")


def _fake_source(monkeypatch, strips=None, rounds=None):
    """Stand in for the frame source: every call returns `strips` (cycled),
    or the next entry of `rounds`. Records (n, phase) per call."""
    calls = []

    def fake_sample(video_path, crop_box, time_ranges, n, phase=0.5):
        calls.append((n, phase))
        source = rounds[len(calls) - 1] if rounds else strips
        return [source[i % len(source)] for i in range(n)]

    monkeypatch.setattr(B, "_sample_strips", fake_sample)
    return calls


def _interleave(a, b):
    return [x for pair in zip(a, b) for x in pair]


CROP = (288, 786, 1344, 53)


def test_detect_measures_the_gate_floor_on_empty_strips(monkeypatch):
    text = [_glyph_strip() for _ in range(12)]
    empty = [_ramp_strip(top=200) for _ in range(12)]
    _fake_source(monkeypatch, _interleave(text, empty))

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _GlyphReadingOCR())

    # The floor comes from the EMPTY strips (ramp tops out at 200). Measured
    # on the text strips it would be 251, above the glyph cores themselves,
    # and the pick could not clear it.
    assert result.gate_floor == 201
    assert result.flagged is None
    assert result.seed == 242
    assert result.plateau == (217, 247)
    assert result.value == 227
    assert result.curve, "the verification curve is part of the evidence"


def test_a_gate_floor_above_the_pick_is_flagged_not_applied(monkeypatch):
    # Empty strips keep clutter up to 245 (floor 246). Raising the pick
    # there would erase subtitles: measured end to end, pushing picks up to
    # a gate floor lost real lines (Legend of Soldier's countdown "10"/"1",
    # Stay Low Profile's "罢了"). The value stays; the clutter is reported.
    text = [_glyph_strip() for _ in range(16)]
    empty = [_ramp_strip(top=245) for _ in range(8)]
    _fake_source(monkeypatch, text + empty)

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _GlyphReadingOCR())

    assert result.gate_floor == 246
    assert result.plateau == (217, 247)
    assert result.value == 227
    assert result.flagged == "no-clean-threshold"


def test_a_low_scoring_subtitle_still_counts_as_text(monkeypatch):
    # On thin crops the real engine scores genuine subtitles 0.84-0.95. Such
    # a strip must seed the threshold, not be treated as an empty strip whose
    # glyphs then wreck the gate floor.
    _fake_source(monkeypatch, _interleave([_glyph_strip()] * 12, [_ramp_strip(top=200)] * 12))
    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(score=0.85), _GlyphReadingOCR())
    assert result.flagged is None
    assert result.seed == 242
    assert result.gate_floor == 201


def test_detect_samples_24_frames_when_they_hold_enough_text(monkeypatch):
    calls = _fake_source(monkeypatch, [_glyph_strip()] * 16 + [_ramp_strip()] * 8)
    ocr = _GlyphReadingOCR()
    B.detect_brightness("v.mp4", CROP, None, _FakeDet(), ocr)
    assert [n for n, _ in calls] == [24]
    # 16 text strips x seed 242's thresholds 217..252
    assert ocr.images == 16 * 8


def test_too_few_text_strips_top_up_with_interleaved_rounds(monkeypatch):
    # Sparse dialogue: 6 text strips in 24 frames. Measured on the reference
    # corpus, verifying on so few strips put the plateau's top edge 5-10
    # above where more strips put it.
    sparse = [_glyph_strip()] * 6 + [_ramp_strip(top=200)] * 18
    calls = _fake_source(monkeypatch, rounds=[sparse, sparse, sparse])

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _GlyphReadingOCR())

    assert [n for n, _ in calls] == [24, 24, 24]
    assert len({phase for _, phase in calls}) == 3, "each round must sample new times"
    assert result.flagged is None


def test_top_up_stops_once_there_are_enough_text_strips(monkeypatch):
    half = [_glyph_strip()] * 10 + [_ramp_strip()] * 14
    calls = _fake_source(monkeypatch, rounds=[half, half, half])
    ocr = _GlyphReadingOCR()
    B.detect_brightness("v.mp4", CROP, None, _FakeDet(), ocr)
    assert [n for n, _ in calls] == [24, 24]
    assert ocr.images == 16 * 8


def test_yellow_text_is_flagged_rather_than_applied(monkeypatch):
    yellow = [_glyph_strip(core=(0, 250, 250), rim=(0, 180, 180)) for _ in range(12)]
    empty = [_ramp_strip(top=240) for _ in range(12)]
    _fake_source(monkeypatch, _interleave(yellow, empty))

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _GlyphReadingOCR())

    assert result.seed < 150
    assert result.flagged == "coloured-text?"
    assert result.plateau is None


def test_no_clean_threshold_is_flagged_and_the_pick_is_not_raised(monkeypatch):
    text = [_glyph_strip() for _ in range(12)]
    clutter = [_checker_strip() for _ in range(12)]
    _fake_source(monkeypatch, _interleave(text, clutter))

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _GlyphReadingOCR())

    assert result.gate_floor is None
    assert result.flagged == "no-clean-threshold"
    assert result.plateau is not None
    assert result.plateau[0] <= result.value <= result.plateau[1]


def test_missing_crop_is_flagged_without_touching_the_video(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("must not sample frames without a crop box")

    monkeypatch.setattr(B, "_sample_strips", refuse)
    result = B.detect_brightness("v.mp4", None, None, _FakeDet(), _ExplodingOCR())
    assert result.flagged == "needs-crop"
    assert result.plateau is None


def test_no_text_anywhere_is_flagged(monkeypatch):
    _fake_source(monkeypatch, [_ramp_strip()])
    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _ExplodingOCR())
    assert result.flagged is not None and "no-text" in result.flagged.split("+")
    assert result.plateau is None


# --------------------------------------------------------------------------
# Two-tier cheap path
# --------------------------------------------------------------------------

def test_cheap_path_takes_its_own_seed_when_it_lands_inside_the_folder_plateau(monkeypatch):
    dim = _glyph_strip(core=(235,) * 3)   # seed round5(235) - 8 = 227
    calls = _fake_source(monkeypatch, _interleave([dim] * 3, [_ramp_strip()] * 3))

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _ExplodingOCR(),
                                 folder_plateau=(210, 255))

    assert [n for n, _ in calls] == [6]
    assert result.flagged is None
    assert result.seed == 227
    assert result.value == 227
    assert result.plateau == (210, 255)


def test_cheap_path_never_goes_above_the_folder_plateaus_safe_pick(monkeypatch):
    # Seed 242 is inside (220, 250), but full detection on this folder would
    # pick 250 - 20 = 230: the seed itself sits where short lines were lost.
    _fake_source(monkeypatch, _interleave([_glyph_strip()] * 3, [_ramp_strip()] * 3))

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _ExplodingOCR(),
                                 folder_plateau=(220, 250))

    assert result.flagged is None
    assert result.seed == 242
    assert result.value == 230


def test_cheap_path_escalates_when_the_seed_is_outside_the_folder_plateau(monkeypatch):
    calls = _fake_source(monkeypatch, _interleave([_glyph_strip()] * 3, [_ramp_strip()] * 3))

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _ExplodingOCR(),
                                 folder_plateau=(180, 200))

    assert [n for n, _ in calls] == [6]
    assert result.flagged == "escalate"


def test_cheap_path_escalates_when_no_text_is_found(monkeypatch):
    _fake_source(monkeypatch, [_ramp_strip()])
    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _ExplodingOCR(),
                                 folder_plateau=(230, 250))
    assert result.flagged == "escalate"


# --------------------------------------------------------------------------
# Sampling times
# --------------------------------------------------------------------------

def test_sample_times_skip_the_first_and_last_tenth_without_ranges():
    times = B.sample_times(1000.0, None, 24)
    assert len(times) == 24
    assert all(100.0 <= t <= 900.0 for t in times)
    assert times == sorted(times)
    gaps = np.diff(times)
    assert gaps.min() > 0.8 * (800.0 / 24)
    assert times[0] < 150 and times[-1] > 850


def test_sample_times_stay_inside_the_keep_ranges_in_proportion():
    ranges = [("1:00", "2:00"), ("5:00", "5:30")]
    times = B.sample_times(600.0, ranges, 24)
    assert len(times) == 24
    first = [t for t in times if 60.0 <= t <= 120.0]
    second = [t for t in times if 300.0 <= t <= 330.0]
    assert len(first) + len(second) == 24
    assert len(first) == 16 and len(second) == 8


def test_later_sampling_rounds_interleave_with_the_first():
    first = B.sample_times(1000.0, None, 24)
    second = B.sample_times(1000.0, None, 24, phase=0.0)
    third = B.sample_times(1000.0, None, 24, phase=0.25)
    merged = sorted(first + second + third)
    assert len(set(merged)) == 72
    assert all(100.0 <= t <= 900.0 for t in merged)
    # every second-round time sits between two first-round times
    for a, b in zip(first, first[1:]):
        assert sum(a < t < b for t in second) == 1


def test_sample_times_open_ended_ranges_run_to_the_file_edges():
    times = B.sample_times(600.0, [(None, "1:00"), ("9:00", "")], 12)
    assert all(t <= 60.0 or t >= 540.0 for t in times)
    assert sum(t <= 60.0 for t in times) == 6


# --------------------------------------------------------------------------
# Mirror of videocr/video.py: downscale, mask and gate, pinned against the
# real run_ocr with a fake capture (no media, no models).
# --------------------------------------------------------------------------

class _RecordingOCR:
    def __init__(self):
        self.frames = []

    def predict(self, frames):
        self.frames.extend(f.copy() for f in frames)
        return [_ocr_item("", 0.0) for _ in frames]


def _run_ocr_on_frames(monkeypatch, frames, threshold, crop=None):
    """Feed `frames` through the real Video.run_ocr and return what it hands
    the OCR engine."""
    from videocr import utils
    from videocr import video as V

    h, w = frames[0].shape[:2]

    class FakeCapture:
        def __init__(self, path, use_gpu=True, decode_target_height=None, crop_rect=None):
            self._i = 0
            self._crop_slice = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_COUNT: len(frames), cv2.CAP_PROP_FPS: 25.0,
                cv2.CAP_PROP_FRAME_HEIGHT: h, cv2.CAP_PROP_FRAME_WIDTH: w,
            }.get(prop, 0)

        def set(self, prop, value):
            return True

        def read(self):
            if self._i >= len(frames):
                return False, None
            self._i += 1
            return True, frames[self._i - 1].copy()

        def get_last_pts(self):
            return (self._i - 1) / 25.0

        def get_stream_start_time(self):
            return 0.0

    recorder = _RecordingOCR()
    monkeypatch.setattr(V, "Capture", FakeCapture)
    monkeypatch.setattr(utils, "create_ocr_engine", lambda *a, **k: recorder)
    video = V.Video("fake.mp4", None, None)
    cx, cy, cw, ch = crop if crop else (None, None, None, None)
    video.run_ocr(False, "ch", "", "", 95, crop is None, threshold, 0, 25, 0, cx, cy, cw, ch)
    return recorder.frames


def _separated(frames):
    """Two blank frames after each test frame end any tracking state, so a
    frame reaches OCR exactly when it trips the gate by itself."""
    blank = np.zeros_like(frames[0])
    out = []
    for f in frames:
        out += [f, blank, blank]
    return out


def _blocky_frame(rng, h, w, block=16):
    """Grey blocks with per-channel noise: bright blocks survive an area
    downscale (plain noise would average out and never trip the gate), and
    the noise makes the min-channel mask cut inside them."""
    grey = rng.integers(0, 256, (h // block + 1, w // block + 1))
    img = np.repeat(np.repeat(grey, block, 0), block, 1)[:h, :w]
    noise = rng.integers(-12, 13, (h, w, 3))
    return np.clip(img[..., None] + noise, 0, 255).astype(np.uint8)


# 1142: int(1142 * (720 / 1142)) is 719 -- the copy must keep run_ocr's float
# arithmetic, not "fix" it to 720.
@pytest.mark.parametrize("h, w", [(53, 1344), (720, 900), (721, 900), (1080, 1920), (1142, 700), (2000, 400)])
def test_ocr_view_and_mask_match_what_run_ocr_hands_the_engine(monkeypatch, h, w):
    rng = np.random.default_rng(h * 7 + w)
    frames = [_blocky_frame(rng, h, w) for _ in range(3)]
    threshold = 200

    seen = _run_ocr_on_frames(monkeypatch, _separated(frames), threshold)

    expected = [B._mask(B._ocr_view(f), threshold) for f in frames]
    assert all(B._gate_fires(e) for e in expected)
    assert len(seen) == len(expected)
    for got, want in zip(seen, expected):
        assert got.shape == want.shape
        assert np.array_equal(got, want)


def test_gate_fires_at_exactly_the_minimum_variance_like_run_ocr(monkeypatch):
    # Ten isolated pixels of 27 in the 54x54 centre square give a Laplacian
    # variance of exactly 145800 / 2916 = 50.0: the OCR gate's ">=" fires.
    exact = np.zeros((H, W, 3), dtype=np.uint8)
    nine = np.zeros((H, W, 3), dtype=np.uint8)
    spots = [(5 + 3 * i, CENTRE_X0 + 5 + 3 * i) for i in range(10)]
    for k, (r, c) in enumerate(spots):
        exact[r, c] = 27
        if k < 9:
            nine[r, c] = 27

    seen = _run_ocr_on_frames(monkeypatch, _separated([exact, nine]), 20)

    assert B._gate_fires(B._mask(exact, 20))
    assert not B._gate_fires(B._mask(nine, 20))
    assert len(seen) == 1 and np.array_equal(seen[0], B._mask(exact, 20))


def test_gate_matches_run_ocr_at_the_variance_boundary(monkeypatch):
    # One bright pixel on black: the centre square's Laplacian variance is
    # 20*v^2/h^2, which crosses MIN_LAPLACIAN_VARIANCE between v=85 and 86
    # for h=54. A pixel just outside the centre square must never count.
    frames = []
    for v, col in [(85, CENTRE_X0 + 20), (86, CENTRE_X0 + 20), (255, CENTRE_X0 - 3),
                   (120, CENTRE_X0 + 1), (90, CENTRE_X0 + H + 2)]:
        f = np.zeros((H, W, 3), dtype=np.uint8)
        f[27, col] = v
        frames.append(f)

    seen = _run_ocr_on_frames(monkeypatch, _separated(frames), 80)

    fired = [f for f in frames if B._gate_fires(B._mask(B._ocr_view(f), 80))]
    assert [int(f.max()) for f in fired] == [86, 120]
    assert len(seen) == len(fired)
    for got, want in zip(seen, fired):
        assert np.array_equal(got, B._mask(B._ocr_view(want), 80))


# --------------------------------------------------------------------------
# Frame source: grab_ocr_strips must return exactly the pixels the OCR pass
# sees, through the same capture chain (decode downscale, in-graph crop,
# 10-bit conversion, HDR tone map).
# --------------------------------------------------------------------------

def _encode(path, size, pix_fmt, codec="libx264", frames=40, gop=15, hdr=False):
    vf = []
    if hdr:
        vf = ["-vf", "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc",
              "-color_trc", "smpte2084", "-color_primaries", "bt2020", "-colorspace", "bt2020nc"]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
           "-i", f"testsrc2=size={size}:rate=25:duration={frames / 25}",
           "-pix_fmt", pix_fmt, "-c:v", codec, "-g", str(gop), "-preset", "ultrafast",
           *vf, str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _run_ocr_frames_by_pts(monkeypatch, path, crop, time_start="", time_end=""):
    """Every frame the real run_ocr hands its OCR engine, keyed by PTS, with
    the brightness filter off so each decoded frame is handed over as-is."""
    from videocr import utils
    from videocr import video as V

    recorder = _RecordingOCR()
    monkeypatch.setattr(utils, "create_ocr_engine", lambda *a, **k: recorder)
    video = V.Video(str(path), None, None)
    x, y, w, h = crop
    video.run_ocr(False, "ch", time_start, time_end, 95, False, 0, 0, 25, 0, x, y, w, h)
    assert len(recorder.frames) == len(video.pred_frames)
    return {round(p.pts_start, 6): f for p, f in zip(video.pred_frames, recorder.frames)}


def _assert_strips_match_run_ocr(monkeypatch, path, crop, time_start="", time_end="", picks=6):
    by_pts = _run_ocr_frames_by_pts(monkeypatch, path, crop, time_start, time_end)
    pts = sorted(by_pts)
    chosen = pts[1::max(1, len(pts) // picks)][:picks]
    strips = B.grab_ocr_strips(str(path), crop, chosen)
    assert len(strips) == len(chosen)
    for t, strip in zip(chosen, strips):
        assert strip.shape == by_pts[t].shape, f"t={t}"
        assert np.array_equal(strip, by_pts[t]), f"t={t}: strip differs from the OCR pass"


@pytest.mark.parametrize("name, size, pix_fmt, hdr, crop", [
    # no filter graph: decoded as-is, sliced in Python
    ("sdr8_360p", "640x360", "yuv420p", False, (100, 290, 400, 40)),
    # PQ tone map in the graph, crop planned inside it
    ("pq10_720p", "1280x720", "yuv420p10le", True, (100, 640, 1000, 60)),
    # 4:3 decode downscale: the capture refuses the in-graph crop, Python
    # slices. The box's far edges scale to x 1575.75 / y 1041.75, which
    # run_ocr truncates.
    ("sdr10_1440p", "2560x1440", "yuv420p10le", False, (401, 1300, 1700, 89)),
])
def test_grab_ocr_strips_returns_the_pixels_run_ocr_sees(monkeypatch, tmp_path, name, size, pix_fmt, hdr, crop):
    import av
    from videocr import pyav_adapter

    path = _encode(tmp_path / f"{name}.mp4", size, pix_fmt, hdr=hdr)
    container = av.open(str(path))
    try:
        trc = int(container.streams.video[0].codec_context.color_trc)
    finally:
        container.close()
    # Otherwise the HDR case would silently exercise the SDR chain.
    assert (trc == pyav_adapter._TRC_SMPTE2084) == hdr
    _assert_strips_match_run_ocr(monkeypatch, path, crop)


@pytest.mark.parametrize("crop", [
    (576, 1892, 2688, 108),     # a subtitle band: in-graph crop, no Python downscale
    (400, 200, 3000, 1600),     # 800 rows after decode downscale: run_ocr shrinks it to 720
])
def test_grab_ocr_strips_matches_run_ocr_on_10bit_4k(monkeypatch, tmp_path, crop):
    path = _encode(tmp_path / "sdr10_4k.mp4", "3840x2160", "yuv420p10le", frames=30)
    _assert_strips_match_run_ocr(monkeypatch, path, crop)


@pytest.mark.needs_media
@pytest.mark.slow
def test_grab_ocr_strips_matches_run_ocr_on_the_real_10bit_4k_reference(monkeypatch, reference_media, detector_truth):
    entry = reference_media.get("xwz")
    if entry is None:
        pytest.skip("xwz reference project not present")
    crop = tuple(detector_truth["xwz"]["files"][entry["video"].name]["crop"])
    _assert_strips_match_run_ocr(monkeypatch, entry["video"], crop, "5:00", "5:02")


# --------------------------------------------------------------------------
# Accuracy against the user's hand-tuned values (Step 5)
# --------------------------------------------------------------------------

@pytest.mark.needs_media
@pytest.mark.slow
def test_brightness_matches_or_beats_the_hand_tuned_value(reference_media, detector_truth):
    """Not an equality check: higher within the plateau suppresses clutter,
    so the automatic pick may legitimately beat the hand-tuned value."""
    from videocr.utils import create_detection_engine, create_ocr_engine, suppress_output
    with suppress_output():
        det = create_detection_engine(None, True)
        ocr = create_ocr_engine("ch", None, None, True)

    rows = []
    for key, entry in reference_media.items():
        truth = detector_truth.get(key)
        if not truth:
            continue
        expected = truth["files"].get(entry["video"].name)
        if not expected:
            continue
        hand = expected["brightness"]
        started = time.perf_counter()
        result = B.detect_brightness(str(entry["video"]), tuple(expected["crop"]), None, det, ocr)
        rows.append((key, hand, result, time.perf_counter() - started))

    assert rows, "no reference files resolved"
    for key, hand, r, secs in rows:
        print(f"{key}: hand={hand} chosen={r.value} plateau={r.plateau} seed={r.seed} "
              f"gate_floor={r.gate_floor} flagged={r.flagged} {secs:.1f}s")
    for key, hand, r, _ in rows:
        # "no-clean-threshold" is informational (see BrightnessResult); any
        # other flag means the value must not be auto-applied.
        assert r.flagged in (None, "no-clean-threshold"), f"{key}: flagged {r.flagged}"
        in_same_plateau = (r.plateau is not None and r.plateau[0] <= hand <= r.plateau[1]
                           and r.plateau[0] <= r.value <= r.plateau[1])
        assert in_same_plateau or r.value >= hand, (
            f"{key}: chosen {r.value} is below hand-tuned {hand} and outside its plateau {r.plateau}")
