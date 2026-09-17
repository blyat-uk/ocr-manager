"""Brightness evidence (core/detect/brightness.py, core/detect/tiles.py).

detect_brightness() carries, next to its result, what the Brightness review
tab shows: one StripSample per sampled strip, the "background clutter still
firing" curve, and -- through tiles.choose_tiles() -- the zoom tiles most
likely to break. None of it may move an existing field: the identity test
replays representative fake inputs and compares every pre-existing field
(and every sampling, detection, OCR and neighbour call) with
tests/fixtures/brightness_identity.json, recorded from the module BEFORE the
evidence was added.

Engines and the frame source are the fakes of tests/test_detect_brightness.py.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from core.detect import brightness as B
from core.detect import ocr_view as OV
from core.detect.tiles import TILE_KINDS, choose_tiles
from test_detect_brightness import (
    BG,
    CROP,
    GLYPH_PITCH,
    GLYPH_W,
    GLYPH_X0,
    GLYPH_Y0,
    GLYPH_Y1,
    H,
    N_GLYPHS,
    TEXT_POLY,
    TEXTS,
    W,
    _checker_strip,
    _DimReadsOCR,
    _ExplodingOCR,
    _fake_neighbours,
    _fake_source,
    _FakeDet,
    _glyph_strip,
    _GlyphReadingOCR,
    _interleave,
    _level_strip,
    _MarkedOCR,
    _probe_strip,
    _ramp_strip,
    _real_neighbour_fetch,
    _ScriptedOCR,
    _StyleOCR,
    _WindowOCR,
)

IDENTITY_FIXTURE = Path(__file__).parent / "fixtures" / "brightness_identity.json"
PREEXISTING_FIELDS = ("value", "plateau", "seed", "gate_floor", "flagged", "curve", "auto_applicable")


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class _Counting:
    """Wraps an engine and counts predict() calls and images."""

    def __init__(self, engine):
        self.engine = engine
        self.calls = 0
        self.images = 0

    def predict(self, images):
        self.calls += 1
        self.images += len(images)
        return self.engine.predict(images)


class _PolyDet:
    """Boxes exactly the polygons registered for each strip object; strips
    it does not know hold no text."""

    def __init__(self, polys_by_strip):
        self.polys = {id(strip): polys for strip, polys in polys_by_strip}

    def predict(self, images):
        out = []
        for img in images:
            polys = self.polys.get(id(img), [])
            out.append({"dt_polys": np.array(polys, dtype=np.float32) if polys else np.zeros((0, 4, 2)),
                        "dt_scores": [0.95] * len(polys)})
        return out


def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


UPPER_LINE = _rect(440, 8, 900, 26)
LOWER_LINE = _rect(440, 28, 900, 46)
LEFT_HALF = _rect(440, 8, 660, 46)
RIGHT_HALF = _rect(680, 8, 900, 46)
TINY_POLY = _rect(450, 20, 455, 24)            # fewer eroded pixels than MIN_GLYPH_REGION_PIXELS
HUD_POLY = _rect(950, 4, 1250, 50)


def _flat_strip(level=BG):
    return np.full((H, W, 3), level, dtype=np.uint8)


def _bar_strip(bar_w, core=250, bg=BG):
    """A long horizontal bar `bar_w` rows thick (odd) across the text line,
    on a flat background: one stroke of exactly that thickness whose ends
    are too short to matter."""
    img = np.full((H, W, 3), bg, dtype=np.uint8)
    img[27 - bar_w // 2:27 + bar_w // 2 + 1, 450:891] = core
    return img


def _noisy_text_strip(rng, core):
    img = rng.integers(20, 90, (H, W, 3), dtype=np.uint8)
    for k in range(N_GLYPHS):
        x0 = GLYPH_X0 + k * GLYPH_PITCH
        img[GLYPH_Y0:GLYPH_Y1, x0:x0 + GLYPH_W] = 180
        block = core - rng.integers(0, 7, (GLYPH_Y1 - GLYPH_Y0 - 2, GLYPH_W - 2, 3))
        img[GLYPH_Y0 + 1:GLYPH_Y1 - 1, x0 + 1:x0 + GLYPH_W - 1] = np.clip(block, 0, 255).astype(np.uint8)
    return img


def _noisy_empty_strip(rng, specks):
    img = rng.integers(20, 90, (H, W, 3), dtype=np.uint8)
    x0 = (W - H) // 2
    for _ in range(specks):
        y, x = int(rng.integers(0, H)), int(rng.integers(x0, x0 + H))
        img[y, x] = int(rng.integers(150, 256))
    return img


def _split_level_strip(low):
    """Glyph blocks without a rim whose pixels alternate `low` and `low` + 1
    in equal numbers: a glyph level of exactly low + 0.5."""
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    yy, xx = np.mgrid[GLYPH_Y0:GLYPH_Y1, 0:GLYPH_W]
    block = (low + (yy + xx) % 2).astype(np.uint8)
    for k in range(N_GLYPHS):
        x0 = GLYPH_X0 + k * GLYPH_PITCH
        img[GLYPH_Y0:GLYPH_Y1, x0:x0 + GLYPH_W] = block[..., None]
    return img


def _halo_strip():
    """One glyph and a bright outline lying exactly on TEXT_POLY's edge: only
    the polygon's eroded interior is glyph territory."""
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    img[GLYPH_Y0:GLYPH_Y1, GLYPH_X0:GLYPH_X0 + GLYPH_W] = 180
    img[GLYPH_Y0 + 1:GLYPH_Y1 - 1, GLYPH_X0 + 1:GLYPH_X0 + GLYPH_W - 1] = 250
    img[8, 440:901] = img[46, 440:901] = 240
    img[8:47, 440] = img[8:47, 900] = 240
    return img


def _two_tone_band_strip():
    """A text band 441 px wide over rows 10-43: rows 10-26 at level 240, rows
    27-43 at 220, background to its left. How a sub-pixel polygon's top edge
    rounds decides whether the 240 rows outnumber the 220 rows."""
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    img[10:27, 450:891] = 240
    img[27:44, 450:891] = 220
    return img


def _detect(monkeypatch, source=None, rounds=None, det=None, ocr=None, neighbours=None, **kwargs):
    """detect_brightness over a faked frame source with counting engines.
    Returns (result, what it consumed)."""
    samples = _fake_source(monkeypatch, source, rounds=rounds)
    requested = _fake_neighbours(monkeypatch, neighbours or (lambda t: []))
    det, ocr = _Counting(det or _FakeDet()), _Counting(ocr or _GlyphReadingOCR())
    result = B.detect_brightness("v.mp4", kwargs.pop("crop", CROP), kwargs.pop("ranges", None), det, ocr, **kwargs)
    return result, dict(samples=samples, det_calls=det.calls, det_images=det.images,
                        ocr_calls=ocr.calls, ocr_images=ocr.images, neighbour_requests=requested)


# --------------------------------------------------------------------------
# Identity cases: representative inputs for every path through detection
# --------------------------------------------------------------------------

def _case_full_clean(mp):
    return _detect(mp, _interleave([_glyph_strip() for _ in range(12)], [_ramp_strip(top=200) for _ in range(12)]))


def _case_full_floor_above_pick(mp):
    return _detect(mp, [_glyph_strip() for _ in range(16)] + [_ramp_strip(top=245) for _ in range(8)])


def _case_full_no_clean_threshold(mp):
    return _detect(mp, _interleave([_glyph_strip() for _ in range(12)], [_checker_strip() for _ in range(12)]))


def _case_full_no_empty_strips(mp):
    return _detect(mp, [_glyph_strip()])


def _case_full_three_rounds(mp):
    sparse = [_glyph_strip()] * 6 + [_ramp_strip(top=200)] * 18
    return _detect(mp, rounds=[sparse, sparse, sparse])


def _case_full_thin_evidence(mp):
    thin = [_glyph_strip()] * 2 + [_ramp_strip(top=200)] * 22
    return _detect(mp, rounds=[thin, thin, thin])


def _case_full_scripted_evidence(mp):
    texts = [f"第{i}行字幕文本" for i in range(16)]
    steady = {t: ("ok", 0.99) for t in range(220, 256, 5)}
    scripts = [steady] * 6 + [{250: ("short", 0.98), 255: ("ok", 0.97)}] * 4 + [{}] * 6
    strips = [_probe_strip(i, 16) for i in range(16)]
    return _detect(mp, strips + [_ramp_strip(top=200)] * 8, ocr=_ScriptedOCR(scripts, texts))


def _case_full_scripted_curve(mp):
    # seed 222: thresholds 197..247; the glyph cores (230) are gone from 232 up
    steady = {197: ("ok", 1.0), 202: ("short", 0.99), 207: ("wrong", 0.99), 212: ("ok", 0.99),
              217: ("okjunk", 0.98), 222: ("ok", 0.985), 227: ("ok", 0.99)}
    fading = {**steady, 222: ("fragment", 0.97), 227: ("", 0.0)}
    strips = [_probe_strip(i, 4, core=230) for i in range(4)]
    return _detect(mp, strips * 4 + [_ramp_strip(top=210)] * 8,
                   ocr=_ScriptedOCR([steady, steady, fading, steady], TEXTS))


def _case_full_style_narrow(mp):
    text = [_level_strip(250) for _ in range(12)] + [_level_strip(223) for _ in range(4)]
    return _detect(mp, text + [_ramp_strip(top=200)] * 8, ocr=_StyleOCR())


def _case_narrow_plateau(mp):
    return _detect(mp, [_probe_strip(0, 1) for _ in range(16)] + [_ramp_strip(top=200)] * 8, ocr=_WindowOCR(230, 245))


def _case_no_plateau(mp):
    return _detect(mp, [_glyph_strip() for _ in range(16)] + [_ramp_strip(top=200)] * 8,
                   ocr=_GlyphReadingOCR(conf=0.9))


def _case_coloured_text(mp):
    yellow = [_glyph_strip(core=(0, 250, 250), rim=(0, 180, 180)) for _ in range(12)]
    return _detect(mp, _interleave(yellow, [_ramp_strip(top=240) for _ in range(12)]), ocr=_ExplodingOCR())


def _case_no_text(mp):
    return _detect(mp, [_ramp_strip(), _flat_strip(), _checker_strip()], ocr=_ExplodingOCR())


def _fade(mp, neighbours, **kwargs):
    dim = _glyph_strip(core=(200,) * 3)
    text = [_glyph_strip() for _ in range(11)] + [dim.copy() for _ in range(5)]
    return _detect(mp, text + [_ramp_strip(top=200)] * 8, ocr=_MarkedOCR(), neighbours=neighbours, **kwargs)


def _case_dim_text(mp):
    return _fade(mp, lambda t: [])


def _case_dim_text_fade_survives(mp):
    return _fade(mp, lambda t: [_glyph_strip()] * 4)


def _case_dim_text_capped(mp):
    strips = [_glyph_strip() for _ in range(15)] + [_glyph_strip(core=(200,) * 3) for _ in range(9)]
    return _detect(mp, strips, ocr=_MarkedOCR(), neighbours=lambda t: [_glyph_strip()] * 4)


def _case_unverified_short_dim_line(mp):
    strips = [_level_strip(223 if i == 9 else 250) for i in range(19)]
    return _detect(mp, strips + [_ramp_strip(top=200)] * 8, ocr=_DimReadsOCR(lambda t: "罢了" if t == 217 else "罢"))


def _case_mixed_polygons(mp):
    rng = np.random.default_rng(7)
    entries = []
    for i in range(20):
        strip = _glyph_strip(core=(235 + (i % 4) * 5,) * 3)
        polys = [[TEXT_POLY], [UPPER_LINE, LOWER_LINE], [LEFT_HALF, RIGHT_HALF], [TEXT_POLY, HUD_POLY]][i % 4]
        if i % 4 == 3:
            strip[4:50, 950:1250] = 170
        entries.append((strip, polys))
    entries.append((_glyph_strip(), [TINY_POLY]))
    empties = [_noisy_empty_strip(rng, specks) for specks in (0, 3, 40, 400)]
    source = [strip for strip, _ in entries] + empties
    return _detect(mp, source, det=_PolyDet(entries))


def _case_fractional_polygons(mp):
    # Real detectors return sub-pixel polygon corners: how they round decides
    # which pixels count as glyph territory.
    rng = np.random.default_rng(5)
    entries = []
    for i in range(18):
        strip = _noisy_text_strip(rng, int(rng.integers(215, 256)))
        dx, dy = rng.uniform(-1.6, 1.6, 2)
        polys = [TEXT_POLY + np.float32([dx, dy])] if i % 3 else [UPPER_LINE + np.float32([dx, 0.5]),
                                                                  LOWER_LINE + np.float32([-dx, -0.5])]
        entries.append((strip, polys))
    empties = [_noisy_empty_strip(rng, specks) for specks in (2, 20, 200, 700, 0, 60)]
    return _detect(mp, [strip for strip, _ in entries] + empties, det=_PolyDet(entries))


def _case_half_level_glyphs(mp):
    # Glyph levels 232.5 and 232: seed round5(232.25) - 8 = 222, where the
    # median of rounded levels would give 227.
    text = [_split_level_strip(232) for _ in range(8)] + [_glyph_strip(core=(232,) * 3, rim=(232,) * 3)] * 8
    return _detect(mp, _interleave(text[::2], text[1::2]) + [_ramp_strip(top=200)] * 8)


def _case_polygon_edge_halo(mp):
    return _detect(mp, [_halo_strip() for _ in range(12)] + [_glyph_strip()] * 6 + [_ramp_strip(top=200)] * 6)


def _case_subpixel_polygon_edges(mp):
    strips = [_two_tone_band_strip() for _ in range(16)]
    polys = [_rect(420.6, 9.6, 890.4, 43.4)]
    return _detect(mp, strips + [_ramp_strip(top=200)] * 8, det=_PolyDet([(s, polys) for s in strips]))


def _case_noisy_levels(mp):
    rng = np.random.default_rng(11)
    text = [_noisy_text_strip(rng, int(core)) for core in rng.integers(215, 256, 18)]
    empties = [_noisy_empty_strip(rng, specks) for specks in (0, 5, 50, 500, 2000, 1, 10, 100)]
    return _detect(mp, _interleave(text[:8], empties) + text[8:])


def _cheap(mp, strips, folder):
    return _detect(mp, strips, ocr=_ExplodingOCR(), folder_plateau=folder)


def _case_cheap_hit(mp):
    return _cheap(mp, _interleave([_glyph_strip(core=(235,) * 3)] * 3, [_ramp_strip()] * 3), (200, 255))


def _case_cheap_narrow(mp):
    return _cheap(mp, _interleave([_glyph_strip(core=(235,) * 3)] * 3, [_ramp_strip()] * 3), (220, 235))


def _case_cheap_escalate(mp):
    return _cheap(mp, _interleave([_glyph_strip()] * 3, [_ramp_strip()] * 3), (180, 200))


def _case_cheap_escalate_no_text(mp):
    return _cheap(mp, [_ramp_strip()], (230, 250))


def _case_cheap_coloured(mp):
    yellow = _glyph_strip(core=(0, 250, 250), rim=(0, 180, 180))
    return _cheap(mp, _interleave([yellow] * 3, [_ramp_strip()] * 3), (90, 250))


def _polls(fire_at):
    polls = []

    def cancel_check():
        polls.append(1)
        return len(polls) >= fire_at

    return cancel_check


def _case_cancelled_between_rounds(mp):
    sparse = [_glyph_strip()] * 6 + [_ramp_strip()] * 18
    return _detect(mp, rounds=[sparse, sparse, sparse], ocr=_ExplodingOCR(), cancel_check=_polls(2))


def _case_cancelled_before_verification(mp):
    return _detect(mp, [_glyph_strip()] * 16 + [_ramp_strip()] * 8, ocr=_ExplodingOCR(), cancel_check=_polls(2))


def _case_cancelled_in_dim_text_check(mp):
    dim = _glyph_strip(core=(200,) * 3)
    text = [_glyph_strip() for _ in range(11)] + [dim.copy() for _ in range(5)]
    samples = _fake_source(mp, text + [_ramp_strip(top=200)] * 8)
    grabs = _real_neighbour_fetch(mp, lambda t: _glyph_strip())
    det, ocr = _Counting(_FakeDet()), _Counting(_MarkedOCR())
    result = B.detect_brightness("v.mp4", CROP, None, det, ocr, cancel_check=_polls(5))
    return result, dict(samples=samples, det_calls=det.calls, det_images=det.images, ocr_calls=ocr.calls,
                        ocr_images=ocr.images, neighbour_grabs=grabs)


def _case_needs_crop(mp):
    return _detect(mp, [_glyph_strip()], crop=None)


def _case_ranges_empty(mp):
    mp.setattr(OV, "video_duration", lambda video_path: 100.0)
    return _detect(mp, [_glyph_strip()], ranges=[("5:00", "6:00")])


IDENTITY_CASES = {
    "full_clean": _case_full_clean,
    "full_floor_above_pick": _case_full_floor_above_pick,
    "full_no_clean_threshold": _case_full_no_clean_threshold,
    "full_no_empty_strips": _case_full_no_empty_strips,
    "full_three_rounds": _case_full_three_rounds,
    "full_thin_evidence": _case_full_thin_evidence,
    "full_scripted_evidence": _case_full_scripted_evidence,
    "full_scripted_curve": _case_full_scripted_curve,
    "full_style_narrow": _case_full_style_narrow,
    "narrow_plateau": _case_narrow_plateau,
    "no_plateau": _case_no_plateau,
    "coloured_text": _case_coloured_text,
    "no_text": _case_no_text,
    "dim_text": _case_dim_text,
    "dim_text_fade_survives": _case_dim_text_fade_survives,
    "dim_text_capped": _case_dim_text_capped,
    "unverified_short_dim_line": _case_unverified_short_dim_line,
    "mixed_polygons": _case_mixed_polygons,
    "fractional_polygons": _case_fractional_polygons,
    "half_level_glyphs": _case_half_level_glyphs,
    "polygon_edge_halo": _case_polygon_edge_halo,
    "subpixel_polygon_edges": _case_subpixel_polygon_edges,
    "noisy_levels": _case_noisy_levels,
    "cheap_hit": _case_cheap_hit,
    "cheap_narrow": _case_cheap_narrow,
    "cheap_escalate": _case_cheap_escalate,
    "cheap_escalate_no_text": _case_cheap_escalate_no_text,
    "cheap_coloured": _case_cheap_coloured,
    "cancelled_between_rounds": _case_cancelled_between_rounds,
    "cancelled_before_verification": _case_cancelled_before_verification,
    "cancelled_in_dim_text_check": _case_cancelled_in_dim_text_check,
    "needs_crop": _case_needs_crop,
    "ranges_empty": _case_ranges_empty,
}


def identity_record(result, consumed) -> dict:
    """Every field detect_brightness returned before evidence existed, plus
    everything the run consumed, as JSON-normalised data."""
    record = {name: getattr(result, name) for name in PREEXISTING_FIELDS}
    record["consumed"] = consumed
    return json.loads(json.dumps(record))


@pytest.mark.parametrize("case", sorted(IDENTITY_CASES))
def test_evidence_changes_no_preexisting_field(monkeypatch, case):
    expected = json.loads(IDENTITY_FIXTURE.read_text())["cases"][case]

    result, consumed = IDENTITY_CASES[case](monkeypatch)

    assert identity_record(result, consumed) == expected


def test_the_identity_snapshot_covers_every_case_and_every_path():
    snapshot = json.loads(IDENTITY_FIXTURE.read_text())
    assert sorted(snapshot["cases"]) == sorted(IDENTITY_CASES)
    flags = {flag for record in snapshot["cases"].values() for flag in (record["flagged"] or "").split("+") if flag}
    assert flags == {B.FLAG_NO_CLEAN_THRESHOLD, B.FLAG_NEEDS_CROP, B.FLAG_RANGES_EMPTY, B.FLAG_NO_TEXT,
                     B.FLAG_THIN_EVIDENCE, B.FLAG_COLOURED_TEXT, B.FLAG_NO_PLATEAU, B.FLAG_NARROW_PLATEAU,
                     B.FLAG_DIM_TEXT, B.FLAG_ESCALATE, B.FLAG_CANCELLED}
    assert any(record["flagged"] is None for record in snapshot["cases"].values())


# --------------------------------------------------------------------------
# Strip samples
# --------------------------------------------------------------------------

def test_every_sampled_strip_is_carried_with_its_text_verdict_boxes_and_lines(monkeypatch):
    two_line, side_by_side, one_line = _glyph_strip(), _glyph_strip(), _glyph_strip()
    empty = _ramp_strip(top=230)
    det = _PolyDet([(one_line, [TEXT_POLY]), (two_line, [UPPER_LINE, LOWER_LINE]),
                    (side_by_side, [LEFT_HALF, RIGHT_HALF])])
    source = [one_line] * 14 + [two_line, side_by_side] + [empty] * 8
    pixels = [strip.copy() for strip in source]

    result, _ = _detect(monkeypatch, source, det=det)

    assert all(np.array_equal(strip, before) for strip, before in zip(source, pixels)), "evidence never writes pixels"

    assert [s.time for s in result.strips] == [100.0 + i for i in range(24)]
    assert [s.is_text for s in result.strips] == [True] * 16 + [False] * 8
    assert [s.lines for s in result.strips] == [1] * 14 + [2, 1] + [0] * 8
    assert result.strips[0].boxes == ((440, 8, 461, 39),)
    assert result.strips[14].boxes == ((440, 8, 461, 19), (440, 28, 461, 19))
    assert result.strips[15].boxes == ((440, 8, 221, 39), (680, 8, 221, 39))
    assert all(s.boxes == () for s in result.strips[16:])


def test_glyph_levels_follow_each_strips_own_glyphs_and_empty_strips_have_none(monkeypatch):
    cores = [250, 235, 220, 245] * 4
    source = [_glyph_strip(core=(c,) * 3) for c in cores] + [_ramp_strip(top=200)] * 8

    result, _ = _detect(monkeypatch, source)

    assert [s.glyph_level for s in result.strips[:16]] == cores
    assert [B._glyph_level(strip, [TEXT_POLY]) for strip in source[:16]] == cores
    assert all(s.glyph_level is None and s.stroke_px is None for s in result.strips[16:])


@pytest.mark.parametrize("seed", range(6))
def test_glyph_level_is_the_level_analytic_seed_takes_its_median_over(seed):
    rng = np.random.default_rng(seed)
    strip = _noisy_text_strip(rng, int(rng.integers(150, 256)))
    polys = [[TEXT_POLY], [UPPER_LINE, LOWER_LINE], [TINY_POLY], [TEXT_POLY, HUD_POLY], [LEFT_HALF], []][seed]

    sample = B._strip_sample(1.0, strip, polys, 227)
    level = B._glyph_level(strip, polys)

    assert (sample.glyph_level is None) is (level is None)
    if level is not None:
        assert sample.glyph_level == B._round_half_up(level, 1)


@pytest.mark.parametrize("low, level", [(232, 233), (231, 232)])
def test_a_half_level_glyph_level_rounds_half_up(low, level):
    strip = _split_level_strip(low)
    assert B._glyph_level(strip, [TEXT_POLY]) == low + 0.5
    assert B._strip_sample(0.0, strip, [TEXT_POLY], 227).glyph_level == level


def test_a_text_strip_too_small_to_split_carries_no_glyph_measurements():
    sample = B._strip_sample(3.0, _glyph_strip(), [TINY_POLY], 227)
    strip = _glyph_strip()

    assert sample.is_text and sample.lines == 1
    assert sample.glyph_level is None and sample.stroke_px is None
    assert sample.background_level == pytest.approx(float(strip.min(axis=2).mean()))
    assert sample.gate_at_value is None


@pytest.mark.parametrize("bg", [30, 90])
def test_background_level_is_the_mean_outside_glyph_pixels(bg):
    sample = B._strip_sample(0.0, _glyph_strip(bg=bg), [TEXT_POLY], 227)
    assert sample.background_level == bg


def test_an_empty_strips_background_level_is_the_whole_strip():
    strip = _ramp_strip(top=230)
    sample = B._strip_sample(0.0, strip, [], 227)
    assert sample.background_level == pytest.approx(float(strip.min(axis=2).mean()))
    assert sample.lines == 0 and sample.boxes == ()


def _bar_distances(bar_w):
    """Each bar pixel's Euclidean distance to the nearest pixel outside the
    bar, by hand: for an axis-aligned rectangle on a flat background that
    pixel lies straight up, down, left or right of it."""
    rows = np.arange(27 - bar_w // 2, 27 + bar_w // 2 + 1)
    cols = np.arange(450, 891)
    r, c = np.meshgrid(rows, cols, indexing="ij")
    return np.minimum.reduce([r - rows[0] + 1, rows[-1] - r + 1, c - cols[0] + 1, cols[-1] - c + 1])


@pytest.mark.parametrize("bar_w, about", [
    # Across a long bar the distances run 1..(bar_w + 1) / 2 and back, so
    # 2 x their mean is about (bar_w + 1)^2 / (2 bar_w); the bar's short
    # ends pull it slightly lower.
    (3, 2.67),     # 1 2 1
    (5, 3.6),      # 1 2 3 2 1
    (9, 5.56),     # 1 2 3 4 5 4 3 2 1
    (13, 7.54),    # 1 2 3 4 5 6 7 6 5 4 3 2 1
])
def test_stroke_px_is_twice_the_mean_distance_to_the_strokes_edge(bar_w, about):
    sample = B._strip_sample(0.0, _bar_strip(bar_w), [TEXT_POLY], 227)
    assert sample.stroke_px == pytest.approx(2.0 * float(_bar_distances(bar_w).mean()), abs=1e-9)
    assert about - 0.1 < sample.stroke_px <= about + 0.005


def test_every_stroke_width_measures_differently():
    # The median of these distances cannot tell 5 px from 7 px (both read 4),
    # and on real 4K subtitles read 2.0 for every strip.
    widths = [B._strip_sample(0.0, _bar_strip(w), [TEXT_POLY], 227).stroke_px for w in range(1, 16, 2)]
    assert widths == sorted(widths) and len(set(widths)) == len(widths)


def test_stroke_px_counts_the_strip_border_as_outside_the_glyph():
    # A glyph mask touching the strip's edge must not measure as infinitely thick.
    strip = np.full((H, W, 3), 250, dtype=np.uint8)
    strip[:, :200] = BG
    sample = B._strip_sample(0.0, strip, [_rect(-10, -10, W + 10, H + 10)], 227)
    assert sample.stroke_px is not None and 0 < sample.stroke_px <= H


@pytest.mark.parametrize("boxes, lines", [
    ([], 0),
    ([(0, 10, 50, 20)], 1),
    ([(0, 10, 50, 20), (60, 10, 50, 20)], 1),              # side by side
    ([(0, 0, 50, 20), (0, 30, 50, 20)], 2),                # stacked
    ([(0, 0, 50, 20), (60, 10, 50, 20)], 1),               # overlap 10 = 50% of the smaller height
    ([(0, 0, 50, 20), (60, 11, 50, 20)], 2),               # overlap 9 < 50%
    ([(0, 0, 50, 40), (60, 30, 50, 20)], 1),               # 10 of the smaller box's 20: 50%
    ([(0, 0, 50, 40), (60, 31, 50, 20)], 2),
    ([(0, 0, 10, 20), (20, 8, 10, 20), (40, 16, 10, 20)], 1),   # chained overlaps are one row
    ([(0, 0, 10, 20), (20, 30, 10, 20), (40, 60, 10, 20)], 3),
])
def test_lines_cluster_boxes_whose_vertical_extents_overlap_by_half_the_smaller_height(boxes, lines):
    assert B._count_lines(boxes) == lines


def test_gate_at_value_is_measured_on_empty_strips_at_the_final_value(monkeypatch):
    loud, quiet = _ramp_strip(top=230), _flat_strip()
    source = [_glyph_strip() for _ in range(16)] + [loud, quiet] * 4

    result, _ = _detect(monkeypatch, source)

    assert result.value == 227
    assert OV.gate_fires(OV.mask(loud, 227)) and not OV.gate_fires(OV.mask(quiet, 227))
    assert [s.gate_at_value for s in result.strips] == [None] * 16 + [True, False] * 4


def test_gate_at_value_follows_the_value_actually_returned(monkeypatch):
    # Ramp top 230: loud at 227, quiet at 232. The no-plateau path returns
    # the seed, 242, so the ramp is quiet there.
    source = [_glyph_strip() for _ in range(16)] + [_ramp_strip(top=230)] * 8

    result, _ = _detect(monkeypatch, source, ocr=_GlyphReadingOCR(conf=0.9))

    assert result.flagged == "no-plateau?" and result.value == 242
    assert [s.gate_at_value for s in result.strips[16:]] == [False] * 8


def test_strips_from_every_round_are_in_time_order(monkeypatch):
    rounds = []

    def fake_sample(video_path, crop_box, time_ranges, n, phase=0.5):
        rounds.append(phase)
        # slot k holds text only in the first 6 slots
        return [((k + phase) * 10.0, _glyph_strip() if k < 6 else _ramp_strip(top=200)) for k in range(n)]

    monkeypatch.setattr(B, "_sample_strips", fake_sample)
    _fake_neighbours(monkeypatch, lambda t: [])

    result = B.detect_brightness("v.mp4", CROP, None, _FakeDet(), _GlyphReadingOCR())

    assert rounds == list(B.SAMPLE_ROUND_PHASES)
    times = [s.time for s in result.strips]
    assert len(times) == 72
    assert times == sorted((k + p) * 10.0 for p in B.SAMPLE_ROUND_PHASES for k in range(24))
    assert [s.is_text for s in result.strips] == [int(t // 10) < 6 for t in times]


def test_the_cheap_path_carries_its_six_strips_and_no_clutter_curve(monkeypatch):
    source = _interleave([_glyph_strip(core=(235,) * 3)] * 3, [_ramp_strip()] * 3)

    result, _ = _detect(monkeypatch, source, ocr=_ExplodingOCR(), folder_plateau=(200, 255))

    assert result.value == 207 and result.flagged is None
    assert [s.is_text for s in result.strips] == [True, False] * 3
    assert [s.gate_at_value for s in result.strips] == [None, True] * 3
    assert result.clutter_curve == []


@pytest.mark.parametrize("folder", [(180, 200), (220, 235)])
def test_every_cheap_outcome_carries_its_strips(monkeypatch, folder):
    source = _interleave([_glyph_strip()] * 3, [_ramp_strip()] * 3)
    result, _ = _detect(monkeypatch, source, ocr=_ExplodingOCR(), folder_plateau=folder)
    assert len(result.strips) == 6 and result.clutter_curve == []


@pytest.mark.parametrize("case", ["needs_crop", "ranges_empty", "cancelled_between_rounds",
                                  "cancelled_before_verification", "cancelled_in_dim_text_check"])
def test_paths_that_measure_nothing_carry_no_evidence(monkeypatch, case):
    result, _ = IDENTITY_CASES[case](monkeypatch)
    assert result.strips == [] and result.clutter_curve == []


# --------------------------------------------------------------------------
# Clutter curve
# --------------------------------------------------------------------------

def test_the_clutter_curve_is_the_share_of_empty_strips_that_fire_on_the_verified_grid(monkeypatch):
    # Ramp top 230 fires from 105 to 230; a flat strip never fires. The text
    # strips fire at every threshold up to their glyph cores (250): they must
    # not count.
    source = [_glyph_strip() for _ in range(16)] + [_ramp_strip(top=230), _flat_strip()] * 2 + [_glyph_strip()] * 4

    result, _ = _detect(monkeypatch, source)

    assert [t for t, _ in result.curve] == list(range(217, 253, 5))
    assert result.clutter_curve == [(217, 0.5), (222, 0.5), (227, 0.5), (232, 0.0), (237, 0.0),
                                    (242, 0.0), (247, 0.0), (252, 0.0)]


def test_without_verification_the_clutter_curve_runs_100_to_255_in_steps_of_5(monkeypatch):
    result, _ = _detect(monkeypatch, [_ramp_strip(top=230), _flat_strip()] * 2, ocr=_ExplodingOCR())

    assert "no-text" in result.flagged.split("+")
    assert [t for t, _ in result.clutter_curve] == list(range(100, 256, 5))
    curve = dict(result.clutter_curve)
    assert curve[200] == 0.5 and curve[240] == 0.0
    assert curve[100] == 0.0, "the whole smooth ramp survives: no edge to fire on"


def test_coloured_text_gets_the_unverified_grid_over_its_empty_strips_only(monkeypatch):
    yellow = [_glyph_strip(core=(0, 250, 250), rim=(0, 180, 180)) for _ in range(12)]
    loud = [_checker_strip() for _ in range(12)]

    result, _ = _detect(monkeypatch, _interleave(yellow, loud), ocr=_ExplodingOCR())

    assert result.flagged == "no-clean-threshold+coloured-text?"
    assert result.clutter_curve == [(t, 1.0) for t in range(100, 256, 5)]


def test_no_empty_strips_means_an_empty_clutter_curve(monkeypatch):
    result, _ = _detect(monkeypatch, [_glyph_strip()])
    assert result.curve and result.clutter_curve == []


# --------------------------------------------------------------------------
# to_evidence
# --------------------------------------------------------------------------

def _natively_typed(value):
    if isinstance(value, dict):
        return all(isinstance(k, str) and _natively_typed(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_natively_typed(v) for v in value)
    return value is None or type(value) in (bool, int, float, str)


@pytest.mark.parametrize("case", ["full_clean", "mixed_polygons", "noisy_levels", "coloured_text", "cheap_hit",
                                  "no_text", "needs_crop"])
def test_to_evidence_is_json(monkeypatch, case):
    result, _ = IDENTITY_CASES[case](monkeypatch)

    evidence = result.to_evidence()

    assert json.loads(json.dumps(evidence, allow_nan=False)) == evidence
    assert _natively_typed(evidence)
    assert set(evidence) == {"value", "plateau", "seed", "gate_floor", "flagged", "curve", "clutter_curve", "strips"}
    assert evidence["value"] == result.value and evidence["flagged"] == result.flagged
    assert evidence["curve"] == [list(p) for p in result.curve]
    assert evidence["clutter_curve"] == [list(p) for p in result.clutter_curve]
    assert len(evidence["strips"]) == len(result.strips)
    for record, sample in zip(evidence["strips"], result.strips):
        assert record == {"time": sample.time, "is_text": sample.is_text, "glyph_level": sample.glyph_level,
                          "background_level": sample.background_level, "stroke_px": sample.stroke_px,
                          "lines": sample.lines, "boxes": [list(b) for b in sample.boxes],
                          "gate_at_value": sample.gate_at_value}


def test_old_positional_constructors_still_build_a_result_without_evidence():
    result = B.BrightnessResult(227, (217, 247), 242, 201, None, [])
    assert result.strips == [] and result.clutter_curve == []
    assert result.to_evidence()["strips"] == []


# --------------------------------------------------------------------------
# choose_tiles
# --------------------------------------------------------------------------

def _s(time, is_text=True, glyph_level=230, background_level=40.0, stroke_px=5.0, lines=1, gate=None):
    return B.StripSample(time=time, is_text=is_text,
                         glyph_level=glyph_level if is_text else None,
                         background_level=background_level,
                         stroke_px=stroke_px if is_text else None,
                         lines=lines if is_text else 0, boxes=((440, 8, 461, 39),) if is_text else (),
                         gate_at_value=None if is_text else gate)


def _scene():
    return [
        _s(1.0, glyph_level=180, lines=1),                      # darkest glyphs (a boxed HUD, say), one line
        _s(2.0, glyph_level=220, lines=2),
        _s(3.0, glyph_level=210, lines=3),                      # darkest glyphs of the multi-line strips
        _s(4.0, glyph_level=240, background_level=120.0),       # brightest background behind text
        _s(5.0, stroke_px=2.5),                                 # thinnest strokes
        _s(6.0, glyph_level=245, background_level=110.0, stroke_px=9.0),
        _s(7.0, is_text=False, background_level=200.0, gate=False),   # brightest empty strip, but quiet
        _s(8.0, is_text=False, background_level=150.0, gate=True),    # loudest leak
        _s(9.0, is_text=False, background_level=100.0, gate=True),
        _s(10.0, glyph_level=235, background_level=8.0),        # darkest scene behind text
        _s(11.0, is_text=False, background_level=1.0, gate=True),     # darker still, but no text
    ]


def test_tile_kinds():
    assert TILE_KINDS == ("dark", "bright", "thin", "two_line", "leaking")


def test_choose_tiles_picks_each_kind():
    assert choose_tiles(_scene(), 227) == {"dark": 10.0, "bright": 4.0, "thin": 5.0, "two_line": 3.0, "leaking": 8.0}


def test_the_dark_tile_is_the_darkest_scene_not_the_darkest_glyphs():
    # 1.0 has the lowest glyph level -- on the reference 1080p file that was a
    # HUD the detector boxed -- but 10.0 is the darkest scene behind text.
    tiles = choose_tiles(_scene(), 227)
    assert tiles["dark"] == 10.0 and tiles["dark"] != min(_scene(), key=lambda s: s.glyph_level or 999).time


def test_choose_tiles_returns_kinds_in_tile_order():
    assert list(choose_tiles(_scene(), 227)) == list(TILE_KINDS)


def test_text_strips_are_never_leaking_and_empty_strips_never_text_tiles():
    strips = [_s(1.0, is_text=False, background_level=250.0, gate=True), _s(2.0, background_level=30.0),
              _s(0.5, is_text=False, background_level=0.0, gate=False)]
    strips.append(B.StripSample(time=3.0, is_text=True, glyph_level=100, background_level=255.0, stroke_px=1.0,
                                lines=2, boxes=(), gate_at_value=None))
    assert choose_tiles(strips, 227) == {"dark": 2.0, "bright": 3.0, "thin": 3.0, "two_line": 3.0, "leaking": 1.0}


@pytest.mark.parametrize("make, expected", [
    (lambda: [], {}),
    (lambda: [_s(1.0), _s(2.0, background_level=30.0)], {"dark": 2.0, "bright": 1.0, "thin": 1.0}),
    (lambda: [_s(1.0, is_text=False, gate=False), _s(2.0, is_text=False, gate=False)], {}),
    (lambda: [_s(1.0, is_text=False, gate=True)], {"leaking": 1.0}),
    # a text strip too small to split still has a background: dark and bright, never thin
    (lambda: [B.StripSample(1.0, True, None, 50.0, None, 1, (), None)], {"dark": 1.0, "bright": 1.0}),
    (lambda: [B.StripSample(1.0, True, None, 50.0, None, 2, (), None)], {"dark": 1.0, "bright": 1.0, "two_line": 1.0}),
])
def test_kinds_without_a_candidate_are_omitted(make, expected):
    assert choose_tiles(make(), 227) == expected


def test_a_multi_line_strip_without_a_glyph_level_is_chosen_only_when_nothing_better_exists():
    strips = [B.StripSample(1.0, True, None, 50.0, None, 2, (), None), _s(2.0, glyph_level=250, lines=2)]
    assert choose_tiles(strips, 227)["two_line"] == 2.0


@pytest.mark.parametrize("kind, make", [
    ("dark", lambda t: _s(t, background_level=1.0)),
    ("bright", lambda t: _s(t, background_level=250.0)),
    ("thin", lambda t: _s(t, stroke_px=0.5)),
    ("two_line", lambda t: _s(t, lines=2)),
    ("leaking", lambda t: _s(t, is_text=False, background_level=90.0, gate=True)),
])
def test_ties_break_to_the_earlier_time_whatever_the_input_order(kind, make):
    strips = [make(9.0), _s(3.0, glyph_level=240, background_level=20.0, stroke_px=8.0), make(4.0), make(6.0)]
    for order in (strips, strips[::-1], strips[2:] + strips[:2]):
        assert choose_tiles(order, 227)[kind] == 4.0


def test_choose_tiles_is_deterministic():
    scene = _scene()
    rng = np.random.default_rng(3)
    picks = {json.dumps(choose_tiles([scene[i] for i in rng.permutation(len(scene))], 227)) for _ in range(20)}
    assert len(picks) == 1


def test_tiles_from_a_real_detection_point_at_its_strips(monkeypatch):
    dark = _glyph_strip(bg=10)
    two_line = _glyph_strip(core=(240,) * 3)
    bright_bg = _glyph_strip(bg=100)
    thin = _bar_strip(3)
    leak = _ramp_strip(top=230)
    plain = _glyph_strip()
    entries = [(plain, [TEXT_POLY]), (dark, [TEXT_POLY]), (two_line, [UPPER_LINE, LOWER_LINE]),
               (bright_bg, [TEXT_POLY]), (thin, [TEXT_POLY])]
    source = [plain] * 12 + [dark, two_line, bright_bg, thin] + [_flat_strip(), leak, _flat_strip(60), leak] * 2

    result, _ = _detect(monkeypatch, source, det=_PolyDet(entries))
    tiles = choose_tiles(result.strips, result.value)

    assert tiles == {"dark": 112.0, "bright": 114.0, "thin": 115.0, "two_line": 113.0, "leaking": 117.0}
