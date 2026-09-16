"""Automatic brightness threshold detection (core/detect/brightness.py).

The pure tests build synthetic crop strips: white glyph-like blocks (a 250
core with a 1px anti-aliased 180 rim) on a dark background, bright
text-free strips for the Laplacian gate, and yellow-text strips for the
coloured-subtitle failure mode. Fake detection/OCR engines stand in for
PaddleOCR; the fake OCR engines read the masked image they are handed, so
nothing here depends on how a batch happens to be ordered. Glyph blocks
cover the gate's centre square, so a strip trips the real gate exactly while
its glyph cores survive the mask.

The OCR-pass mirrors and the frame source are tested in
tests/test_detect_ocr_view.py.
"""
import json
import time

import numpy as np
import pytest

from core.detect import brightness as B
from core.detect import ocr_view as OV

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
        assert not OV.gate_fires(OV.mask(empties[0], t))
    assert OV.gate_fires(OV.mask(empties[0], floor - 1))
    assert not OV.gate_fires(OV.mask(empties[0], 1))


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


def _probe_strip(identity, n_identities, core=255):
    """A glyph strip that tells a fake OCR engine two things from pixels
    alone: which strip it is (a 255 marker at column `identity` of row -3,
    which survives any mask) and which threshold it was masked at (row -1
    holds a 1..255 ramp, whose smallest surviving value IS the threshold).
    Its glyph cores (default 255) keep the gate firing up to `core`."""
    img = _glyph_strip(core=(core,) * 3)
    img[-3, :n_identities] = 0
    img[-3, identity] = 255
    img[-1, :255] = np.arange(1, 256, dtype=np.uint8)[:, None]
    return img


def _masked_threshold(img):
    row = img[-1, :255].min(axis=1)
    kept = row[row > 0]
    return int(kept.min()) if kept.size else 256


def _identity(img, n):
    return int(np.argmax(img[-3, :n, 0]))


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
            idx = _identity(img, len(self.texts))
            variant, conf = self.scripts[idx].get(_masked_threshold(img), ("", 0.0))
            base = self.texts[idx]
            text = {"ok": base, "short": base[:-1], "fragment": base[-2:], "wrong": "口口口口",
                    "okjunk": base + "\n1", "junk": "A", "colon": ":", "mx": "MX", "m": "M", "": ""}[variant]
            out.append(_ocr_item(text, conf))
        return out


class _CoreReadingOCR:
    """Reads each strip's own line whenever at least half its glyph cores
    survive the mask -- a dim strip stops reading where its cores fade."""

    CORE_PX = N_GLYPHS * (GLYPH_Y1 - GLYPH_Y0 - 2) * (GLYPH_W - 2)

    def __init__(self, n):
        self.n = n

    def predict(self, images):
        out = []
        for img in images:
            line = img[GLYPH_Y0:GLYPH_Y1, GLYPH_X0:GLYPH_X0 + N_GLYPHS * GLYPH_PITCH]
            readable = np.count_nonzero(line.min(axis=2)) >= self.CORE_PX // 2
            out.append(_ocr_item(f"字幕第{_identity(img, self.n)}行文本", 0.99) if readable else _ocr_item("", 0.0))
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
    script = {
        200: ("short", 0.99), 205: ("ok", 0.99), 210: ("ok", 0.99),
        215: ("wrong", 0.9), 220: ("wrong", 0.9),
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


def test_a_run_bridges_at_most_one_dip():
    # Failures at 237 AND 247: bridging both would claim 217-252 and pick
    # 232; one bridge per run gives 217-242.
    steady = {t: ("ok", 0.99) for t in range(217, 256, 5)}
    bad_237, bad_247 = dict(steady), dict(steady)
    bad_237[237] = ("wrong", 0.99)
    bad_247[247] = ("wrong", 0.99)
    engine = _ScriptedOCR([bad_237, bad_247, steady, steady], TEXTS)

    value, plateau, _ = B.verify_with_ocr(_probe_strips(), 242, engine)

    assert plateau == (217, 242)
    assert value == 222


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


@pytest.mark.parametrize("junk", [
    {200: ("junk", 0.95)},                        # 'A' once
    {200: ("junk", 0.95), 205: ("colon", 0.95)},  # 'A' then ':' -- one-character readings never match each other
    {200: ("mx", 0.95), 205: ("m", 0.95)},        # 'MX' then 'M'
])
def test_one_off_junk_readings_stay_excluded_under_near_matching(junk):
    steady = {t: ("ok", 0.99) for t in range(175, 230, 5)}
    engine = _ScriptedOCR([steady, steady, steady, steady, junk], TEXTS + ["标志"])

    value, plateau, _ = B.verify_with_ocr(_probe_strips(5), 200, engine)

    assert plateau == (175, 225)
    assert value == 205


def test_an_eroding_line_that_reads_differently_at_each_threshold_is_still_evidence():
    # The line reads whole at 175, then only its last two characters at 180,
    # then nothing: no exact repeat, but the same line eroding. It must hold
    # the plateau down where it still reads, not be dropped as noise.
    steady = {t: ("ok", 0.99) for t in range(175, 230, 5)}
    eroding = {175: ("ok", 0.99), 180: ("fragment", 0.99)}
    engine = _ScriptedOCR([steady, steady, steady, eroding], TEXTS)

    _, plateau, _ = B.verify_with_ocr(_probe_strips(), 200, engine)

    assert plateau == (175, 175)


def test_near_matching_readings_tie_on_support_and_the_most_repeated_one_is_the_modal():
    # At 175 clutter adds a stray "1" line to the reading; from 180 up the line
    # reads clean. Both readings near-match each other (equal support), so the
    # modal must be the one repeated most -- not the one seen first.
    steady = {t: ("ok", 0.99) for t in range(175, 230, 5)}
    cluttered_low = dict(steady)
    cluttered_low[175] = ("okjunk", 0.99)
    engine = _ScriptedOCR([steady, steady, steady, cluttered_low], TEXTS)

    _, plateau, _ = B.verify_with_ocr(_probe_strips(), 200, engine)

    assert plateau == (180, 225)


def test_a_single_one_threshold_hole_inside_a_readable_band_is_tolerated():
    steady = {t: ("ok", 0.99) for t in range(175, 230, 5)}
    holed = {175: ("ok", 0.99), 180: ("ok", 0.99), 190: ("ok", 0.99), 195: ("ok", 0.99), 200: ("ok", 0.99)}
    engine = _ScriptedOCR([steady, steady, steady, holed], TEXTS)

    _, plateau, _ = B.verify_with_ocr(_probe_strips(), 200, engine)

    assert plateau == (175, 200)   # the holed strip is evidence: its loss from 205 up counts


def test_a_strip_whose_readings_are_scattered_carries_no_evidence():
    # Read at 175-180 and again at 215-220 with nothing between: not a band a
    # subtitle's legibility can form, so not evidence about any threshold.
    steady = {t: ("ok", 0.99) for t in range(175, 230, 5)}
    scattered = {175: ("ok", 0.99), 180: ("ok", 0.99), 215: ("ok", 0.99), 220: ("ok", 0.99)}
    engine = _ScriptedOCR([steady, steady, steady, scattered], TEXTS)

    _, plateau, _ = B.verify_with_ocr(_probe_strips(), 200, engine)

    assert plateau == (175, 225)


@pytest.mark.parametrize("n_dim", [1, 7])
def test_dim_lines_readable_only_at_the_lowest_thresholds_keep_the_pick_below_their_limit(n_dim):
    # Reviewer's scenario: seed 242 (thresholds 217..252), bright strips read
    # everywhere, a second dimmer style reads only up to 227 -- three
    # thresholds, empty at five. Those empties are lost lines, not "no
    # evidence": the pick must stay where the dim lines still read.
    n = 16
    strips = [_probe_strip(i, n, core=255 if i >= n_dim else 227) for i in range(n)]

    value, plateau, _ = B.verify_with_ocr(strips, 242, _CoreReadingOCR(n))

    assert plateau == (217, 227)
    assert value <= 227


def test_a_masked_strip_that_does_not_trip_the_gate_reads_as_empty():
    # The OCR pass never OCRs a frame whose masked centre square stays quiet.
    # Strip 0's centre holds only one 200-level speck, so from 205 up that
    # frame is skipped there -- whatever OCR would have read.
    steady = {t: ("ok", 0.99) for t in range(175, 230, 5)}
    strips = _probe_strips()
    strips[0][:, CENTRE_X0:CENTRE_X0 + H] = 0
    strips[0][27, CENTRE_X0 + 20] = 200

    _, plateau, _ = B.verify_with_ocr(strips, 200, _ScriptedOCR(steady, TEXTS))

    assert plateau == (175, 200)


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


class _ExplodingDet:
    def predict(self, images):
        raise AssertionError("the detection engine must not be used on this path")


class _GlyphReadingOCR:
    """Reads the line whenever at least half the glyph cores survive the mask."""

    CORE_PX = N_GLYPHS * (GLYPH_Y1 - GLYPH_Y0 - 2) * (GLYPH_W - 2)

    def __init__(self, conf=0.99):
        self.calls = 0
        self.images = 0
        self.conf = conf

    def predict(self, images):
        self.calls += 1
        out = []
        for img in images:
            self.images += 1
            line = img[GLYPH_Y0:GLYPH_Y1, GLYPH_X0:GLYPH_X0 + N_GLYPHS * GLYPH_PITCH]
            kept = np.count_nonzero(line.min(axis=2))
            out.append(_ocr_item("你好世界", self.conf) if kept >= self.CORE_PX // 2
                       else _ocr_item("", 0.0))
        return out


class _WindowOCR:
    """Reads the line only for thresholds in [lo, hi], told apart by the
    threshold ramp of _probe_strip."""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def predict(self, images):
        return [_ocr_item("你好世界", 0.99) if self.lo <= _masked_threshold(img) <= self.hi
                else _ocr_item("", 0.0) for img in images]


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
    assert result.auto_applicable
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
    assert result.auto_applicable


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
    # 16 text strips x the seven of seed 242's thresholds (217..247) whose
    # masked strips still trip the gate; at 252 nothing survives to OCR.
    assert ocr.images == 16 * 7


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
    assert ocr.images == 16 * 7


def test_thin_evidence_after_every_round_is_flagged(monkeypatch):
    thin = [_glyph_strip()] * 2 + [_ramp_strip(top=200)] * 22   # 6 text strips in 72 frames
    _fake_source(monkeypatch, rounds=[thin, thin, thin])

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _GlyphReadingOCR())

    assert result.flagged == "thin-evidence?"
    assert not result.auto_applicable
    assert result.plateau == (217, 247)


def test_yellow_text_is_flagged_rather_than_applied(monkeypatch):
    yellow = [_glyph_strip(core=(0, 250, 250), rim=(0, 180, 180)) for _ in range(12)]
    empty = [_ramp_strip(top=240) for _ in range(12)]
    _fake_source(monkeypatch, _interleave(yellow, empty))

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _GlyphReadingOCR())

    assert result.seed < 150
    assert result.flagged == "coloured-text?"
    assert result.plateau is None
    assert not result.auto_applicable


def test_no_clean_threshold_is_flagged_and_the_pick_is_not_raised(monkeypatch):
    text = [_glyph_strip() for _ in range(12)]
    clutter = [_checker_strip() for _ in range(12)]
    _fake_source(monkeypatch, _interleave(text, clutter))

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _GlyphReadingOCR())

    assert result.gate_floor is None
    assert result.flagged == "no-clean-threshold"
    assert result.plateau is not None
    assert result.plateau[0] <= result.value <= result.plateau[1]


def test_no_empty_strips_means_the_floor_was_not_measured_not_that_clutter_won(monkeypatch):
    _fake_source(monkeypatch, [_glyph_strip()])

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _GlyphReadingOCR())

    assert result.gate_floor is None
    assert result.flagged is None
    assert result.auto_applicable


def test_a_plateau_too_narrow_for_the_safety_margin_is_flagged(monkeypatch):
    # Text reads only at 230-245 (seed 245, thresholds 220..255): the pick
    # cannot sit 20 below the top without leaving the plateau.
    text = [_probe_strip(0, 1) for _ in range(16)]
    _fake_source(monkeypatch, text + [_ramp_strip(top=200)] * 8)

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _WindowOCR(230, 245))

    assert result.plateau == (230, 245)
    assert result.value == 230
    assert result.flagged == "narrow-plateau?"
    assert not result.auto_applicable


def test_a_plateau_exactly_as_wide_as_the_margin_is_not_narrow(monkeypatch):
    text = [_probe_strip(0, 1) for _ in range(16)]
    _fake_source(monkeypatch, text + [_ramp_strip(top=200)] * 8)

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _WindowOCR(225, 245))

    assert result.plateau == (225, 245)
    assert result.value == 225
    assert result.flagged is None


class _StyleOCR:
    """Bright strips (glyph core >= 250) read one line; dimmer strips read a
    second style, whose reading differs at each threshold where it still
    reads -- as an eroding line does."""

    def predict(self, images):
        out = []
        for img in images:
            line = img[GLYPH_Y0:GLYPH_Y1, GLYPH_X0:GLYPH_X0 + N_GLYPHS * GLYPH_PITCH]
            if np.count_nonzero(line.min(axis=2)) < _GlyphReadingOCR.CORE_PX // 2:
                out.append(_ocr_item("", 0.0))
            elif int(line.max()) >= 250:
                out.append(_ocr_item("你好世界", 0.99))
            else:
                out.append(_ocr_item("旁白第二种样式文本"[: 9 - _masked_threshold(img) % 3], 0.99))
        return out


def _level_strip(core):
    strip = _glyph_strip(core=(core,) * 3)
    strip[-1, :255] = np.arange(1, 256, dtype=np.uint8)[:, None]   # threshold readout for _StyleOCR
    return strip


def test_a_dim_style_whose_readings_vary_as_it_erodes_is_not_erased(monkeypatch):
    # Reviewer's scenario: 12 bright strips, 4 of a dimmer style readable only
    # at 217 and 222 (seed 242), reading a different string at each.
    text = [_level_strip(250) for _ in range(12)] + [_level_strip(223) for _ in range(4)]
    _fake_source(monkeypatch, text + [_ramp_strip(top=200)] * 8)

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _StyleOCR())

    # Their varying readings agree only partly with their own modal, so the
    # plateau closes where they stop reading -- or below -- and is too narrow
    # to apply without review.
    assert result.plateau[1] <= 222
    assert result.value <= 222
    assert "narrow-plateau?" in result.flagged.split("+")
    assert not result.auto_applicable


@pytest.mark.parametrize("kept, thin", [(7, True), (8, False)])
def test_thin_evidence_threshold(monkeypatch, kept, thin):
    per_round = [kept // 3 + (1 if r < kept % 3 else 0) for r in range(3)]
    rounds = [[_glyph_strip()] * n + [_ramp_strip(top=200)] * (24 - n) for n in per_round]
    _fake_source(monkeypatch, rounds=rounds)

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _GlyphReadingOCR())

    assert ("thin-evidence?" in (result.flagged or "").split("+")) is thin


def test_thin_evidence_counts_strips_usable_as_evidence_not_detected_text(monkeypatch):
    # 16 strips the detector calls text, but only 6 ever read: 10 are logos
    # or noise that verification cannot use.
    texts = [f"第{i}行字幕文本" for i in range(16)]
    strips = [_probe_strip(i, 16) for i in range(16)]
    steady = {t: ("ok", 0.99) for t in range(220, 256, 5)}
    scripts = [steady] * 6 + [{}] * 10
    _fake_source(monkeypatch, strips + [_ramp_strip(top=200)] * 8)

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _ScriptedOCR(scripts, texts))

    assert result.plateau == (220, 255)
    assert result.flagged == "thin-evidence?"


class _BrightOnlyOCR(_GlyphReadingOCR):
    """Never reads strips below the bright style: they are logos, not lines."""

    def predict(self, images):
        self.calls += 1
        out = []
        for img in images:
            line = img[GLYPH_Y0:GLYPH_Y1, GLYPH_X0:GLYPH_X0 + N_GLYPHS * GLYPH_PITCH]
            kept = np.count_nonzero(line.min(axis=2))
            out.append(_ocr_item("你好世界", 0.99) if kept >= self.CORE_PX // 2 and int(line.max()) >= 250
                       else _ocr_item("", 0.0))
        return out


@pytest.mark.parametrize("dim_core", [
    200,   # reads nowhere in the window (seed 242 -> 217..252)
    217,   # reads only at the window's lowest threshold
])
def test_text_strips_too_dim_for_the_window_are_re_read_at_their_own_level(monkeypatch, dim_core):
    # Controller's scenario: 11 bright strips set the seed; 5 strips of a
    # dimmer style never enter the plateau. Re-read at their own seed
    # threshold (round5(level) - 8) they read, so the dim style is reported.
    text = [_glyph_strip() for _ in range(11)] + [_glyph_strip(core=(dim_core,) * 3) for _ in range(5)]
    _fake_source(monkeypatch, text + [_ramp_strip(top=200)] * 8)
    ocr = _GlyphReadingOCR()

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), ocr)

    assert result.seed == 242
    assert result.plateau == (217, 247)
    assert result.flagged == "dim-text?"
    assert not result.auto_applicable
    assert ocr.calls == 2, "verification and the re-reads are one OCR batch each"


def test_a_dim_strip_that_would_not_trip_the_gate_at_its_own_level_is_not_re_read(monkeypatch):
    # The OCR pass never OCRs a frame whose masked centre stays quiet, so a
    # dim strip with nothing in the centre square is not dim TEXT to it.
    off_centre = _glyph_strip(core=(200,) * 3)
    off_centre[:, CENTRE_X0:CENTRE_X0 + H] = BG
    text = [_glyph_strip() for _ in range(11)] + [off_centre.copy() for _ in range(5)]
    _fake_source(monkeypatch, text + [_ramp_strip(top=200)] * 8)
    ocr = _GlyphReadingOCR()

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), ocr)

    assert result.flagged is None
    assert ocr.calls == 1


def test_text_strips_that_read_nothing_even_at_their_own_level_are_not_dim_text(monkeypatch):
    text = [_glyph_strip() for _ in range(11)] + [_glyph_strip(core=(200,) * 3) for _ in range(5)]
    _fake_source(monkeypatch, text + [_ramp_strip(top=200)] * 8)

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _BrightOnlyOCR())

    assert result.plateau == (217, 247)
    assert result.flagged is None
    assert result.auto_applicable


def test_detect_flags_a_verification_that_finds_no_plateau(monkeypatch):
    text = [_glyph_strip() for _ in range(16)]
    _fake_source(monkeypatch, text + [_ramp_strip(top=200)] * 8)

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _GlyphReadingOCR(conf=0.9))

    assert result.gate_floor == 201
    assert result.plateau is None
    assert result.flagged == "no-plateau?"
    assert result.value == result.seed == 242
    assert not result.auto_applicable


def test_missing_crop_is_flagged_without_touching_the_video(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("must not sample frames without a crop box")

    monkeypatch.setattr(B, "_sample_strips", refuse)
    result = B.detect_brightness("v.mp4", None, None, _FakeDet(), _ExplodingOCR())
    assert result.flagged == "needs-crop"
    assert result.plateau is None
    assert not result.auto_applicable


def test_no_text_anywhere_is_flagged(monkeypatch):
    _fake_source(monkeypatch, [_ramp_strip()])
    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _ExplodingOCR())
    assert result.flagged is not None and "no-text" in result.flagged.split("+")
    assert result.plateau is None
    assert not result.auto_applicable


def test_keep_ranges_that_select_nothing_are_flagged_not_sampled_elsewhere(synthetic_video):
    # A 0.4 s clip; ranges past its end. Sampling the rest of the file
    # instead measures text OCR will never run on (Martial Master's opening
    # lyrics broke at a different threshold than its dialogue).
    result = B.detect_brightness(str(synthetic_video), (0, 200, 320, 40), [("1:00", "2:00")],
                                 _ExplodingDet(), _ExplodingOCR())
    assert result.flagged == "ranges-empty?"
    assert not result.auto_applicable


@pytest.mark.parametrize("cancel_at_poll, rounds_sampled", [(2, 1), (1, 0)])
def test_cancellation_between_rounds_returns_a_cancelled_result(monkeypatch, cancel_at_poll, rounds_sampled):
    sparse = [_glyph_strip()] * 6 + [_ramp_strip()] * 18
    calls = _fake_source(monkeypatch, rounds=[sparse, sparse, sparse])
    polls = []

    def cancel_check():
        polls.append(1)
        return len(polls) >= cancel_at_poll

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _ExplodingOCR(), cancel_check=cancel_check)

    assert len(calls) == rounds_sampled
    assert result.flagged == "cancelled"
    assert not result.auto_applicable


def test_cancellation_before_verification_skips_the_ocr_batch(monkeypatch):
    _fake_source(monkeypatch, [_glyph_strip()] * 16 + [_ramp_strip()] * 8)
    polls = []

    def cancel_check():
        polls.append(1)
        return len(polls) >= 2   # poll 1: before the (only) round; poll 2: before OCR

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _ExplodingOCR(), cancel_check=cancel_check)

    assert result.flagged == "cancelled"
    assert result.seed == 242


@pytest.mark.parametrize("flagged, applicable", [
    (None, True),
    ("no-clean-threshold", True),
    ("narrow-plateau?", False),
    ("dim-text?", False),
    ("no-clean-threshold+narrow-plateau?", False),
    ("thin-evidence?", False),
    ("no-plateau?", False),
    ("coloured-text?", False),
    ("no-text", False),
    ("needs-crop", False),
    ("ranges-empty?", False),
    ("escalate", False),
    ("cancelled", False),
])
def test_only_clean_or_clutter_only_results_are_auto_applicable(flagged, applicable):
    result = B.BrightnessResult(227, (217, 247), 242, 201, flagged, [])
    assert result.auto_applicable is applicable


# --------------------------------------------------------------------------
# Two-tier cheap path
# --------------------------------------------------------------------------

def test_cheap_path_picks_its_own_seed_minus_the_margin(monkeypatch):
    dim = _glyph_strip(core=(235,) * 3)   # seed round5(235) - 8 = 227
    calls = _fake_source(monkeypatch, _interleave([dim] * 3, [_ramp_strip()] * 3))

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _ExplodingOCR(),
                                 folder_plateau=(200, 255))

    assert [n for n, _ in calls] == [6]
    assert result.flagged is None
    assert result.seed == 227
    assert result.value == 207
    assert result.plateau == (200, 255)


def test_cheap_path_is_not_bounded_by_the_folder_plateaus_top(monkeypatch):
    # The folder plateau is an intersection of other files' plateaus: its top
    # says nothing about this file's own top, but this file's seed sits at or
    # just under it (XWZ 173: folder top 252, own plateau 217-247).
    _fake_source(monkeypatch, _interleave([_glyph_strip()] * 3, [_ramp_strip()] * 3))

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _ExplodingOCR(),
                                 folder_plateau=(220, 250))

    assert result.flagged is None
    assert result.seed == 242
    assert result.value == 222


@pytest.mark.parametrize("folder", [(220, 235), (215, 255)])
def test_cheap_path_never_goes_below_the_folder_plateau_and_says_so(monkeypatch, folder):
    dim = _glyph_strip(core=(235,) * 3)   # seed 227; 227 - 20 = 207 would leave the plateau
    _fake_source(monkeypatch, _interleave([dim] * 3, [_ramp_strip()] * 3))

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _ExplodingOCR(),
                                 folder_plateau=folder)

    assert result.value == folder[0]
    assert result.flagged == "narrow-plateau?"


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


def test_cheap_path_reports_coloured_text_instead_of_escalating(monkeypatch):
    yellow = _glyph_strip(core=(0, 250, 250), rim=(0, 180, 180))
    _fake_source(monkeypatch, _interleave([yellow] * 3, [_ramp_strip()] * 3))

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _ExplodingOCR(),
                                 folder_plateau=(90, 250))

    assert result.seed < 150
    assert result.flagged == "coloured-text?"


# --------------------------------------------------------------------------
# Accuracy against the user's hand-tuned values (Step 5)
# --------------------------------------------------------------------------

@pytest.mark.needs_media
@pytest.mark.slow
def test_brightness_matches_or_beats_the_hand_tuned_value(reference_media, detector_truth):
    """Not an equality check. An auto-applicable result must sit inside the
    plateau that contains the hand-tuned value, or above it; a flagged one
    goes to review and is only reported."""
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
        # The keep ranges the pipeline would pass, from the project's own
        # .ocr.json (read-only); sampling outside them can meet different text.
        config = json.loads((entry["dir"] / ".ocr.json").read_text())
        ranges = (config.get("files", {}).get(entry["video"].name) or {}).get("time_ranges") or None
        started = time.perf_counter()
        result = B.detect_brightness(str(entry["video"]), tuple(expected["crop"]), ranges, det, ocr)
        rows.append((key, expected["brightness"], result, time.perf_counter() - started))

    assert rows, "no reference files resolved"
    for key, hand, r, secs in rows:
        print(f"{key}: hand={hand} chosen={r.value} plateau={r.plateau} seed={r.seed} "
              f"gate_floor={r.gate_floor} flagged={r.flagged} auto={r.auto_applicable} {secs:.1f}s")
    for key, hand, r, _ in rows:
        if not r.auto_applicable:
            continue
        in_same_plateau = (r.plateau is not None and r.plateau[0] <= hand <= r.plateau[1]
                           and r.plateau[0] <= r.value <= r.plateau[1])
        assert in_same_plateau or r.value >= hand, (
            f"{key}: chosen {r.value} is below hand-tuned {hand} and outside its plateau {r.plateau}")
