"""Plan 3C Task 3: the Brightness review tab (app/views/brightness_view.py).

Nothing is decoded. Strips are synthetic numpy arrays delivered through
`fake_runner` exactly as a real `StripJob` would deliver them, so the tests
exercise the controller's `request_strips` / `strip` contract too.

The strips are built so every masked quantity is exact: a text strip's glyph
pixels sit at two known levels, so the percentage lost at a threshold between
them is a whole number, and the empty strip's clutter is bright noise inside
the gate's centre square, so `gate_fires` flips at a known threshold.
"""
from __future__ import annotations

import json
import time as time_mod

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from app.controller import ProjectController
from app.views.brightness_view import (
    KIND_LABELS,
    PIN_TILE_TEXT,
    BrightnessTab,
    PinTile,
    ZoomTile,
)
from app.views.tabs import evidence_tabs
from core.detect import ocr_view
from core.jobs.view_jobs import StripsResult
from core.project import Brightness, Crop, Source

NAME = "ep01.mkv"
OTHERS = ["ep02.mkv", "ep03.mkv", "ep04.mkv"]
BOX = (288, 786, 1344, 53)
OTHER_BOX = (300, 800, 1300, 60)

STRIP_W, STRIP_H = 240, 40
TEXT_BOX = (20, 8, 200, 24)                 # x, y, w, h inside the strip
TILE_TIMES = {"dark": 10.0, "bright": 20.0, "thin": 30.0, "two_line": 40.0, "leaking": 50.0}

BACKGROUND = 20
DIM_GLYPH = 200
BRIGHT_GLYPH = 240


# --------------------------------------------------------------------------
# Synthetic strips
# --------------------------------------------------------------------------

def _bgr(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)


def text_strip(dim_rows: int = 6, bright_rows: int = 6, width: int = STRIP_W,
               height: int = STRIP_H) -> np.ndarray:
    """A strip with a dark background and, inside TEXT_BOX, two bands of
    "glyph" pixels: `dim_rows` rows at DIM_GLYPH and `bright_rows` at
    BRIGHT_GLYPH. Every band is 100 px wide, so a threshold between the two
    levels loses exactly dim_rows / (dim_rows + bright_rows) of the glyphs."""
    gray = np.full((height, width), BACKGROUND, dtype=np.uint8)
    x, y, _w, _h = TEXT_BOX
    gray[y:y + dim_rows, x:x + 100] = DIM_GLYPH
    gray[y + dim_rows:y + dim_rows + bright_rows, x:x + 100] = BRIGHT_GLYPH
    return _bgr(gray)


def two_line_strip() -> np.ndarray:
    """Two rows of glyphs, all at BRIGHT_GLYPH: nothing is lost below it."""
    gray = np.full((STRIP_H, STRIP_W), BACKGROUND, dtype=np.uint8)
    x, y, _w, _h = TEXT_BOX
    gray[y:y + 5, x:x + 100] = BRIGHT_GLYPH
    gray[y + 14:y + 19, x:x + 100] = BRIGHT_GLYPH
    return _bgr(gray)


def empty_strip(level: int = 200) -> np.ndarray:
    """No text: deterministic bright noise inside the gate's centre square,
    so `gate_fires` is True while `level` survives the mask and False once
    the threshold passes it."""
    gray = np.zeros((STRIP_H, STRIP_W), dtype=np.uint8)
    start = (STRIP_W - STRIP_H) // 2
    noise = np.indices((STRIP_H, STRIP_H)).sum(axis=0) % 2
    gray[:, start:start + STRIP_H] = (noise * level).astype(np.uint8)
    return _bgr(gray)


def strips_for(times=TILE_TIMES) -> dict[float, np.ndarray]:
    """Only the "thin" strip has dim glyphs, so only it loses strokes: the
    note's "starts losing strokes" sample is unambiguous."""
    return {times["dark"]: text_strip(dim_rows=0, bright_rows=12),
            times["bright"]: text_strip(dim_rows=0, bright_rows=12),
            times["thin"]: text_strip(),
            times["two_line"]: two_line_strip(),
            times["leaking"]: empty_strip()}


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

def sample(t: float, *, is_text: bool = True, lines: int = 1, boxes=(TEXT_BOX,),
           glyph_level: int | None = 220, background_level: float = 20.0,
           stroke_px: float | None = 3.0, gate_at_value: bool | None = None) -> dict:
    return {"time": float(t), "is_text": is_text, "glyph_level": glyph_level,
            "background_level": background_level, "stroke_px": stroke_px, "lines": lines,
            "boxes": [list(box) for box in boxes], "gate_at_value": gate_at_value}


def curve_points(lo: int = 190, hi: int = 240) -> list[list]:
    return [[t, 0.9 if lo <= t <= hi else 0.4] for t in range(150, 251, 10)]


def clutter_points() -> list[list]:
    return [[t, max(0.0, 1.0 - (t - 150) / 100)] for t in range(150, 251, 10)]


def brightness_evidence(*, value: int = 209, plateau=(190, 240), flagged=None,
                        curve=None, clutter=None, tiles=None, strips=None,
                        crop_box=BOX, value_crop_box=BOX, gate_floor=None) -> dict:
    tiles = TILE_TIMES if tiles is None else tiles
    if strips is None:
        strips = [sample(TILE_TIMES["dark"]), sample(TILE_TIMES["bright"]),
                  sample(TILE_TIMES["thin"]), sample(TILE_TIMES["two_line"], lines=2),
                  sample(TILE_TIMES["leaking"], is_text=False, lines=0, boxes=(),
                         glyph_level=None, stroke_px=None, gate_at_value=True)]
    evidence = {
        "value": value,
        "plateau": None if plateau is None else list(plateau),
        "seed": value + 20,
        "gate_floor": gate_floor,
        "flagged": flagged,
        "curve": curve_points() if curve is None else curve,
        "clutter_curve": clutter_points() if clutter is None else clutter,
        "strips": strips,
        "tiles": {kind: float(t) for kind, t in tiles.items()},
        "crop_box": list(crop_box),
    }
    if value_crop_box is not None:
        evidence["value_crop_box"] = list(value_crop_box)
    return evidence


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def settle() -> None:
    QApplication.processEvents()


class Calls:
    def __init__(self, controller, name: str, *, run: bool = True):
        self.calls: list[tuple] = []
        original = getattr(controller, name)

        def wrapper(*args, **kwargs):
            self.calls.append(args)
            return original(*args, **kwargs) if run else None

        setattr(controller, name, wrapper)


@pytest.fixture
def controller(qapp, fake_runner, tmp_project):
    made = ProjectController(fake_runner, save_debounce_ms=10)
    folder = tmp_project([NAME, *OTHERS])
    made.open_folder(str(folder))
    yield made
    made.shutdown(timeout=0.5)


def give_values(controller, name: str = NAME, *, box=BOX, value: int = 209, evidence=None,
                source: Source = Source.DETECTED) -> None:
    entry = controller.entry(name)
    entry.crop = Crop(*box, source)
    entry.brightness = Brightness(value, source)
    if evidence is not None:
        entry.evidence["brightness"] = evidence


@pytest.fixture
def tab(controller):
    """A BrightnessTab over a file with full brightness evidence, laid out at
    a size the tiles can actually draw in."""
    give_values(controller, evidence=brightness_evidence())
    made = BrightnessTab(controller)
    page = made.page()
    page.resize(880, 620)
    page.show()
    made.inspector_panel().resize(322, 400)
    made.set_file(NAME)
    settle()
    yield made
    page.close()


def deliver(controller, fake_runner, *, name: str = NAME, box=BOX, strips=None) -> None:
    """Answer the newest strips job for `name` with `strips`; a no-op when it
    has already been answered (the controller caches strips per crop box, so
    a second view over the same file asks for nothing)."""
    submission = fake_runner.last("strips", name)
    if fake_runner.ended(submission):
        return
    fake_runner.finish(submission, StripsResult(name, tuple(box),
                                                strips_for() if strips is None else strips))
    controller.drain_events()
    settle()


@pytest.fixture
def loaded(tab, controller, fake_runner):
    deliver(controller, fake_runner)
    return tab


def tile_of(tab: BrightnessTab, kind: str) -> ZoomTile:
    found = [tile for tile in tab.tiles() if tile.kind == kind]
    assert found, f"no {kind} tile among {[t.kind for t in tab.tiles()]}"
    return found[0]


# --------------------------------------------------------------------------
# Tiles
# --------------------------------------------------------------------------

def test_tiles_follow_the_detectors_kind_order_and_end_with_the_pin_tile(loaded):
    assert [tile.kind for tile in loaded.tiles()] == ["dark", "bright", "thin", "two_line", "leaking"]
    assert isinstance(loaded.pin_tile(), PinTile)
    assert loaded.pin_tile().text() == PIN_TILE_TEXT
    assert loaded.grid_widgets()[-1] is loaded.pin_tile()


def test_the_pin_tile_is_on_screen_before_any_brightness_evidence_lands(controller):
    """The state of every file until brightness detection lands, and the
    permanent state of a file whose detection found no text strips: no tiles
    at all. The pin tile -- the one thing the user can still do here -- must
    be IN the grid, not merely in `grid_widgets()`: `_sync_tiles` used to add
    it only when the tile plan changed, and an empty plan never changes."""
    give_values(controller, evidence=None)
    made = BrightnessTab(controller)
    page = made.page()
    page.resize(880, 620)
    page.show()
    made.set_file(NAME)
    settle()
    assert made.tiles() == []
    pin = made.pin_tile()
    assert pin.parent() is not None, "the pin tile was never parented into a layout"
    assert pin.isVisible(), "the tiles area is an empty black rectangle"
    assert page.rect().contains(pin.geometry()), "the pin tile is not inside the page"
    page.close()


def test_a_kind_the_detector_did_not_choose_is_skipped(controller, fake_runner):
    tiles = {"dark": 10.0, "leaking": 50.0}
    give_values(controller, evidence=brightness_evidence(tiles=tiles))
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    settle()
    assert [tile.kind for tile in made.tiles()] == ["dark", "leaking"]
    made.page().close()


def test_tile_captions_name_the_time_and_the_kind(loaded):
    assert tile_of(loaded, "dark").caption_left() == "00:10 · dark scene"
    assert tile_of(loaded, "two_line").caption_left() == "00:40 · two lines"
    assert KIND_LABELS["leaking"] == "background leaking"


def test_zoom_presets_pick_the_source_rectangle(loaded):
    tile = tile_of(loaded, "dark")
    loaded.set_zoom("300%")
    settle()
    assert loaded.zoom_factor() == pytest.approx(3.0)
    rect = tile.source_rect()
    assert rect is not None
    assert rect.width() == round(tile.content_rect().width() / 3)

    loaded.set_zoom("600%")
    settle()
    assert tile.source_rect().width() == round(tile.content_rect().width() / 6)

    loaded.set_zoom("fit")
    settle()
    assert tile.source_rect().width() == STRIP_W
    assert loaded.zoom_factor() == pytest.approx(tile.content_rect().width() / STRIP_W)


def test_dragging_the_context_window_pans_every_tile_equally(loaded):
    loaded.set_zoom("300%")
    settle()
    before = [tile.source_rect().x() for tile in loaded.tiles()]
    assert len(set(before)) == 1                 # the window starts centred on the text

    context = loaded.context
    rect = context.window_rect()
    start = QPoint(int(rect.center().x()), int(context.height() / 2))
    QTest.mousePress(context, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(context, start + QPoint(40, 0))
    QTest.mouseRelease(context, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       start + QPoint(40, 0))
    settle()

    after = [tile.source_rect().x() for tile in loaded.tiles()]
    assert len(set(after)) == 1 and after[0] > before[0]
    assert loaded.x_offset() == pytest.approx(after[0], abs=1)


def test_focusing_another_tile_does_not_move_the_zoom_window(controller, fake_runner):
    """The offset is fixed once the strips are laid out: hovering a tile whose
    text sits elsewhere and then nudging the threshold must not make all six
    jump."""
    off_centre = [sample(TILE_TIMES["dark"]),
                  sample(TILE_TIMES["bright"], boxes=((0, 8, 40, 24),))]
    give_values(controller, evidence=brightness_evidence(
        strips=off_centre, tiles={"dark": TILE_TIMES["dark"], "bright": TILE_TIMES["bright"]}))
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    deliver(controller, fake_runner)
    made.refresh()
    made.set_zoom("300%")
    settle()
    before = made.x_offset()
    assert before > 0                            # centred on the dark tile's text

    made.tiles()[1].focused.emit()               # its text starts at x 0
    settle()
    made.set_preview(215)
    settle()
    assert made.x_offset() == pytest.approx(before)
    assert [tile.source_rect().x() for tile in made.tiles()] == [int(before)] * 2
    made.page().close()


def test_the_zoom_window_starts_centred_on_the_text(loaded):
    loaded.set_zoom("300%")
    settle()
    visible = tile_of(loaded, "dark").source_rect().width()
    text_centre = TEXT_BOX[0] + TEXT_BOX[2] / 2
    assert loaded.x_offset() == pytest.approx(text_centre - visible / 2, abs=1)


def test_every_threshold_change_remasks_each_tile_once(loaded, monkeypatch):
    calls: list[int] = []
    real = ocr_view.mask

    def spy(strip, t):
        calls.append(int(t))
        return real(strip, t)

    monkeypatch.setattr(ocr_view, "mask", spy)
    loaded.set_preview(205)
    settle()
    assert calls == [205] * len(loaded.tiles())

    calls.clear()
    loaded.set_preview(206)
    settle()
    assert calls == [206] * len(loaded.tiles())


def test_lost_pixels_are_counted_exactly_and_turn_the_tile_bad(loaded):
    tile = tile_of(loaded, "thin")
    loaded.set_preview(190)
    settle()
    assert tile.caption_right() == "strokes solid"
    assert tile.status_tone() == "ok" and not tile.is_bad()

    loaded.set_preview(210)                      # DIM_GLYPH rows drop out: half the glyphs
    settle()
    assert tile.caption_right() == "50% of glyph pixels lost"
    assert tile.status_tone() == "bad" and tile.is_bad()

    loaded.set_preview(245)
    settle()
    assert tile.caption_right() == "100% of glyph pixels lost"


def test_a_two_line_tile_reports_both_lines_kept(loaded):
    loaded.set_preview(210)
    settle()
    tile = tile_of(loaded, "two_line")
    assert tile.caption_right() == "both lines kept"
    assert tile.status_tone() == "ok"


def test_an_empty_tile_leaks_below_its_gate_threshold_and_is_clean_above(loaded):
    tile = tile_of(loaded, "leaking")
    loaded.set_preview(190)                      # the clutter is at level 200: still there
    settle()
    assert tile.caption_right() == "background leaking" and tile.status_tone() == "warn"

    loaded.set_preview(210)
    settle()
    assert tile.caption_right() == "clean" and tile.status_tone() == "ok"


def test_the_lost_pixel_overlay_follows_its_toggle(loaded):
    loaded.set_preview(210)
    settle()
    tile = tile_of(loaded, "thin")
    assert loaded.lost_pixels_on() is True
    assert tile.lost_pixel_count() > 0

    loaded.toggle_lost_pixels()
    settle()
    assert loaded.lost_pixels_on() is False
    assert tile.lost_pixel_count() == 0


def test_raw_and_masked_swap_the_pixels_a_tile_draws(loaded):
    loaded.set_preview(210)
    settle()
    tile = tile_of(loaded, "thin")
    assert loaded.masked_on() is True
    masked = tile.drawn_pixels()
    assert masked is not None and int(masked.max()) == BRIGHT_GLYPH

    loaded.toggle_masked()
    settle()
    assert loaded.masked_on() is False
    raw = tile.drawn_pixels()
    assert int(raw.min()) == BACKGROUND


def test_a_tile_without_pixels_yet_draws_a_placeholder(tab):
    tile = tile_of(tab, "dark")
    assert tile.source_rect() is None
    assert tile.has_pixels() is False
    assert tile.caption_left() == "00:10 · dark scene"


# --------------------------------------------------------------------------
# Context strip
# --------------------------------------------------------------------------

def test_the_context_strip_names_the_full_strip_size(loaded):
    assert loaded.context.caption() == (
        f"full strip {STRIP_W} × {STRIP_H} · drag the amber window to move the zoom")


def test_the_zoom_window_covers_the_region_the_tiles_show(loaded):
    loaded.set_zoom("300%")
    settle()
    tile = tile_of(loaded, "dark")
    rect = loaded.context.window_rect()
    share = tile.source_rect().width() / STRIP_W
    assert rect.width() / loaded.context.width() == pytest.approx(share, abs=0.02)


# --------------------------------------------------------------------------
# Curve
# --------------------------------------------------------------------------

def test_the_curve_maps_thresholds_across_its_width(loaded):
    curve = loaded.curve
    assert curve.x_for(100) == pytest.approx(0.0)
    assert curve.x_for(255) == pytest.approx(curve.width())
    assert curve.x_for(177.5) == pytest.approx(curve.width() / 2)
    assert curve.t_for(curve.x_for(211)) == 211


def test_the_plateau_band_spans_the_safe_range(loaded):
    band = loaded.curve.plateau_rect()
    assert band is not None
    assert band.left() == pytest.approx(loaded.curve.x_for(190))
    assert band.right() == pytest.approx(loaded.curve.x_for(240))


def test_the_curve_carries_both_series_and_both_markers(loaded):
    curve = loaded.curve
    assert len(curve.points()) == len(curve_points())
    assert len(curve.clutter_points()) == len(clutter_points())
    assert curve.marker_x()[0] == pytest.approx(curve.x_for(209))       # auto
    assert curve.marker_x()[1] == pytest.approx(curve.x_for(209))       # yours
    assert "■ OCR holds up" in curve.legend_texts()
    assert "┅ background clutter still firing" in curve.legend_texts()
    assert "┆ 209 auto" in curve.legend_texts()
    assert "▲ 209 yours" in curve.legend_texts()
    assert curve.axis_texts() == ("100", "255")


def test_the_clutter_caption_is_left_out_without_a_clutter_curve(controller):
    give_values(controller, evidence=brightness_evidence(clutter=[]))
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    settle()
    legend = made.curve.legend_texts()
    assert "■ OCR holds up" in legend
    assert "┅ background clutter still firing" not in legend
    made.page().close()


def test_an_empty_curve_says_it_was_not_verified(controller, fake_runner):
    give_values(controller, evidence=brightness_evidence(curve=[], clutter=[], plateau=None))
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    settle()
    assert made.curve.points() == []
    assert "not verified on this file" in made.curve.legend_texts()
    assert "┅ background clutter still firing" not in made.curve.legend_texts()
    assert made.curve.marker_x()[1] is not None
    made.page().close()


def test_dragging_the_curve_previews_without_committing(loaded, controller):
    calls = Calls(controller, "set_brightness")
    curve = loaded.curve
    x = int(curve.x_for(220))
    point = QPoint(x, curve.height() // 2)
    QTest.mousePress(curve, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    QTest.mouseMove(curve, point)
    QTest.mouseRelease(curve, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    settle()
    assert loaded.threshold() == 220
    assert calls.calls == []
    assert controller.entry(NAME).brightness.value == 209


def test_keep_commits_the_preview_and_use_commits_auto(loaded, controller):
    calls = Calls(controller, "set_brightness")
    loaded.set_preview(221)
    settle()
    assert loaded.panel.keep_button.text() == "keep 221"
    loaded.panel.keep_button.click()
    settle()
    assert calls.calls == [(NAME, 221)]
    assert controller.entry(NAME).brightness.value == 221
    assert controller.entry(NAME).brightness.source == Source.MANUAL

    assert loaded.panel.use_button.text() == "use 209"
    loaded.panel.use_button.click()
    settle()
    assert calls.calls[-1] == (NAME, 209)
    assert controller.entry(NAME).brightness.value == 209


# --------------------------------------------------------------------------
# Inspector panel
# --------------------------------------------------------------------------

def test_the_panel_shows_auto_and_yours(loaded, controller):
    assert loaded.panel.auto_row.value() == "209"
    assert loaded.panel.yours_row.value() == "209"
    assert loaded.panel.yours_row.value_tone() == "acc"
    loaded.set_preview(215)
    settle()
    assert loaded.panel.yours_row.value() == "215"


def test_the_note_names_the_safe_range_the_losing_threshold_and_the_leak(loaded):
    """auto is 209, where the thin sample has already lost its dim glyph rows
    -- that is the loss the detector picked, so it is that tile's baseline.
    The note names the threshold where some tile loses 10 points MORE than it
    had at auto: 241, where the bright rows drop out of every tile."""
    notes = " ".join(loaded.panel.notes())
    assert "Safe range 190–240." in notes
    assert "Above 241 the 00:10 sample starts losing strokes." in notes
    assert "Below 190 the background leaks and the frame gate fires on empty frames." in notes


def test_the_losing_threshold_is_calibrated_against_each_tile_at_the_auto_value(
        controller, fake_runner):
    """The same strips, read against two different auto values.

    At auto 190 nothing is lost anywhere, so the thin sample's baseline is 0
    and the note fires at 201, where its dim rows go. At auto 209 that same
    loss is already the detector's choice, so the baseline is 50% and the
    note only fires where the tile gets 10 points worse than that.
    """
    seen = {}
    for auto in (190, 209):
        give_values(controller, value=auto, evidence=brightness_evidence(value=auto))
        made = BrightnessTab(controller)
        made.page().resize(880, 620)
        made.page().show()
        made.set_file(NAME)
        deliver(controller, fake_runner)
        made.refresh()
        settle()
        seen[auto] = " ".join(made.panel.notes())
        assert made.threshold() == auto
        made.page().close()

    assert "Above 201 the 00:30 sample starts losing strokes." in seen[190]
    assert "Above 241 the 00:10 sample starts losing strokes." in seen[209]
    for auto, notes in seen.items():                  # never fires at the value itself
        assert f"Above {auto} " not in notes


def test_a_stale_brightness_says_re_detecting_while_one_is_pending(controller):
    give_values(controller, box=OTHER_BOX,
                evidence=brightness_evidence(crop_box=OTHER_BOX, value_crop_box=BOX))
    controller.entry(NAME).crop = Crop(*OTHER_BOX, Source.DETECTED)
    controller.pending_detectors = lambda: {NAME: {"brightness"}}
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    settle()
    assert made.panel.stale_text() == "measured on an earlier crop — re-detecting"
    made.page().close()


def test_flags_are_shown_as_a_warn_line(controller, fake_runner):
    give_values(controller, evidence=brightness_evidence(flagged="narrow-plateau?+dim-text?"))
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    settle()
    assert made.panel.flag_text() == "narrow safe range · dim text on some frames"
    made.page().close()


def test_a_detected_value_that_differs_from_the_stored_one_is_named(controller):
    give_values(controller, value=211, source=Source.MANUAL,
                evidence=brightness_evidence(value=209))
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    settle()
    assert "detected 209 · yours 211" in " ".join(made.panel.notes())
    made.page().close()


def test_a_brightness_measured_on_an_earlier_crop_warns(controller):
    give_values(controller, box=OTHER_BOX,
                evidence=brightness_evidence(crop_box=OTHER_BOX, value_crop_box=BOX))
    controller.entry(NAME).crop = Crop(*OTHER_BOX, Source.DETECTED)
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    settle()
    assert made.panel.stale_text() == "measured on an earlier crop — re-detect to refresh"
    assert made.crop_box() == OTHER_BOX              # the latest result's box
    made.page().close()


def test_the_panel_leaves_the_series_median_to_the_inspector(controller):
    """The cross-file summary belongs to the inspector's Detected section
    (ruling B4, ui-spec §3.7), which shows it once. This panel is about this
    episode's own threshold and never repeats it."""
    give_values(controller, evidence=brightness_evidence())
    for name, value in zip(OTHERS, (218, 213, 230), strict=True):
        controller.entry(name).brightness = Brightness(value, Source.DETECTED)
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    settle()
    assert not any("Series median" in note for note in made.panel.notes())
    made.page().close()


def test_an_unmeasurable_tile_says_so_instead_of_claiming_success(loaded, controller,
                                                                  fake_runner):
    """A strip whose boxes hold too little to split has no lost %, so it has
    nothing to report: it must not borrow the ok tone of "strokes solid"."""
    give_values(controller, evidence=brightness_evidence(
        strips=[sample(TILE_TIMES["dark"], boxes=((0, 0, 3, 2),))],
        tiles={"dark": TILE_TIMES["dark"]}))
    loaded.refresh()
    settle()
    tile = tile_of(loaded, "dark")
    assert tile.has_pixels()
    assert tile.caption_right() == "not measurable"
    assert tile.status_tone() == "dim"
    assert not tile.is_bad()


def test_a_pinned_tile_without_a_sample_shows_pixels_but_claims_no_measurement(
        loaded, controller, fake_runner):
    """A pinned frame the detector never sampled has no boxes, so there is no
    glyph region to measure -- and Otsu over the whole strip would find a
    "split" in any gradient, then report 100% of that invention lost. The pin
    is for LOOKING at a frame under the live mask, so the tile draws its
    pixels and says only that."""
    noise = np.clip(np.linspace(20, 120, STRIP_W)[None, :, None]
                    + np.random.default_rng(5).integers(0, 40, (STRIP_H, STRIP_W, 1)),
                    0, 255).astype(np.uint8).repeat(3, axis=2)
    loaded.pin_time(70.0)
    settle()
    deliver(controller, fake_runner, strips={70.0: noise})
    loaded.refresh()
    settle()
    pinned = tile_of(loaded, "pinned")
    assert pinned.has_pixels()
    for t in (150, 190, 210, 230):
        loaded.set_preview(t)
        settle()
        assert pinned.caption_right() == "pinned frame — not measured"
        assert pinned.status_tone() == "dim"
        assert not pinned.is_bad()
        assert "%" not in pinned.caption_right()
        assert pinned.drawn_pixels() is not None          # the mask is still live
        assert pinned.lost_pixel_count() == 0


def test_a_pinned_time_the_detector_sampled_is_measured_normally(controller, fake_runner):
    """Pinning a time that IS a detector sample keeps its boxes, so the tile
    measures like any other."""
    extra = 70.0
    give_values(controller, evidence=brightness_evidence(
        strips=[sample(t) for t in (*TILE_TIMES.values(), extra)]))
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    made.pin_time(extra)
    settle()
    deliver(controller, fake_runner, strips={**strips_for(), extra: text_strip()})
    made.refresh()
    made.set_preview(210)
    settle()
    pinned = tile_of(made, "pinned")
    assert pinned.caption_right() == "50% of glyph pixels lost"
    assert pinned.is_bad()
    made.page().close()


def test_unchanged_evidence_reuses_the_measured_strip(loaded):
    """The glyph split is an Otsu pass per strip: a repaint must not redo it.
    The cache key is the pixels AND the boxes, both normalised, or a list
    from evidence never equals the tuple it was stored as."""
    held = [loaded.strip_pixels(t) for t in TILE_TIMES.values()]
    assert all(pixels is not None for pixels in held)
    loaded.refresh()
    settle()
    assert [loaded.strip_pixels(t) for t in TILE_TIMES.values()] == held
    for pixels in held:
        assert isinstance(pixels.given_boxes, tuple)
        assert all(isinstance(box, tuple) for box in pixels.given_boxes)


# --------------------------------------------------------------------------
# Strip requests and pinning
# --------------------------------------------------------------------------

def test_set_file_requests_every_tile_strip(controller, fake_runner):
    give_values(controller, evidence=brightness_evidence())
    calls = Calls(controller, "request_strips")
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    settle()
    assert calls.calls
    name, box, times = calls.calls[0]
    assert name == NAME and tuple(box) == BOX
    assert sorted(times) == sorted(TILE_TIMES.values())
    made.page().close()


def test_an_evidence_box_from_another_crop_is_described_instead_of_the_files_own(controller):
    """The stored strips and their boxes were measured in the evidence crop's
    pixel frame. Re-grabbing strips for the file's new box would pair new
    pixels with boxes that no longer point at the text, so the view describes
    the evidence box until a re-detection catches up -- and says so."""
    give_values(controller, evidence=brightness_evidence(crop_box=BOX, value_crop_box=BOX))
    controller.entry(NAME).crop = Crop(*OTHER_BOX, Source.DETECTED)
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    settle()
    assert made.crop_box() == BOX                       # not OTHER_BOX
    assert made.panel.stale_text() == "measured on an earlier crop — re-detect to refresh"
    made.page().close()


def test_the_panel_says_the_tiles_are_of_the_other_crop(controller):
    """The warn line is about the stored value; this one is about what is on
    screen. Only while the tiles really are of another crop."""
    give_values(controller, evidence=brightness_evidence(crop_box=BOX, value_crop_box=BOX))
    controller.entry(NAME).crop = Crop(*OTHER_BOX, Source.DETECTED)
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    settle()
    assert made.crop_box() == BOX
    assert made.panel.crop_note_text() == "the tiles and curve show that crop, not the file's current one"
    assert made.panel.crop_note_text() in made.panel.notes()
    made.page().close()


def test_a_value_stale_on_its_own_does_not_claim_the_tiles_are_elsewhere(controller):
    """evidence["crop_box"] IS the file's crop here -- only the stored value
    was measured on an older one -- so the tiles are of what the user sees."""
    give_values(controller, box=OTHER_BOX,
                evidence=brightness_evidence(crop_box=OTHER_BOX, value_crop_box=BOX))
    controller.entry(NAME).crop = Crop(*OTHER_BOX, Source.DETECTED)
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    settle()
    assert made.panel.stale_text() == "measured on an earlier crop — re-detect to refresh"
    assert made.panel.crop_note_text() == ""
    made.page().close()


def test_a_manual_brightness_on_an_edited_crop_still_warns(controller):
    """brightness_is_stale() only judges DETECTED/HINT values, but the tiles
    are just as much from another frame when the value is MANUAL."""
    give_values(controller, source=Source.MANUAL,
                evidence=brightness_evidence(crop_box=BOX, value_crop_box=None))
    controller.entry(NAME).crop = Crop(*OTHER_BOX, Source.MANUAL)
    made = BrightnessTab(controller)
    made.page().resize(880, 620)
    made.page().show()
    made.set_file(NAME)
    settle()
    assert made.crop_box() == BOX
    assert made.panel.stale_text() == "measured on an earlier crop — re-detect to refresh"
    made.page().close()


def test_a_crop_change_requests_the_strips_again_for_the_new_box(loaded, controller, fake_runner):
    calls = Calls(controller, "request_strips")
    controller.entry(NAME).evidence["brightness"]["crop_box"] = list(OTHER_BOX)
    controller.set_crop(NAME, OTHER_BOX)
    loaded.refresh()
    settle()
    assert calls.calls, "the new crop box must be re-requested"
    assert tuple(calls.calls[-1][1]) == OTHER_BOX
    assert tile_of(loaded, "dark").has_pixels() is False


def test_pinning_a_frame_adds_a_tile_and_asks_for_its_strip(loaded, controller, fake_runner):
    calls = Calls(controller, "request_strips")
    loaded.pin_time(123.0)
    settle()
    assert loaded.pinned_times() == [123.0]
    pinned = [tile for tile in loaded.tiles() if tile.kind == "pinned"]
    assert len(pinned) == 1
    assert pinned[0].caption_left() == "02:03 · pinned"
    assert any(123.0 in list(call[2]) for call in calls.calls)
    assert loaded.grid_widgets()[-1] is loaded.pin_tile()


def test_pinned_times_are_per_file_and_survive_a_file_switch(loaded, controller):
    loaded.pin_time(123.0)
    settle()
    loaded.set_file(OTHERS[0])
    settle()
    assert loaded.pinned_times() == []
    loaded.set_file(NAME)
    settle()
    assert loaded.pinned_times() == [123.0]


def test_the_zoom_and_toggle_controls_live_in_the_stage_head(loaded):
    """Ruling B3 puts each tab's own controls in the stage head, where the
    Crop tab puts its envelope/masked/grid row too; `Stage` mounts whatever
    `toolbar()` returns. The page itself carries none of them."""
    bar = loaded.toolbar()
    assert isinstance(bar, QWidget)
    owned = [*loaded.zoom_buttons, loaded.lost_button, loaded.mask_button]
    for button in owned:
        assert button.parentWidget() is bar
    assert not any(button.parentWidget() is loaded.page() for button in owned)
    assert [button.text() for button in loaded.zoom_buttons] == ["fit", "100%", "300%", "600%"]
    assert bar.findChildren(QLabel)[0].text() == "zoom"


def test_the_tab_leaves_room_for_the_compact_timeline(loaded):
    assert loaded.timeline_slot().height() == 68


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

def test_evidence_tabs_puts_the_brightness_tab_in_the_middle(controller):
    tabs = evidence_tabs(controller)
    assert [t.title for t in tabs] == ["Crop", "Brightness", "Time ranges"]
    assert isinstance(tabs[1], BrightnessTab)


def test_the_tab_does_not_refresh_itself_on_every_model_change(loaded, controller):
    """`Stage` already calls refresh() on file_changed (app/views/stage.py).
    A second connection here would re-mask all six tiles twice per edit."""
    calls = []
    original = loaded.refresh
    loaded.refresh = lambda: (calls.append(1), original())[1]
    controller.file_changed.emit(NAME)
    settle()
    assert calls == []


def test_no_file_clears_the_view(loaded):
    loaded.set_file(None)
    settle()
    assert loaded.tiles() == []
    assert loaded.panel.auto_row.value() == "—"


# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------

def test_fifty_threshold_changes_redraw_six_tiles_quickly(controller, fake_runner):
    times = {kind: float(10 * (index + 1)) for index, kind in enumerate(TILE_TIMES)}
    give_values(controller, evidence=brightness_evidence(
        tiles=times,
        strips=[sample(times["dark"], boxes=((40, 6, 1200, 44),)),
                sample(times["bright"], boxes=((40, 6, 1200, 44),)),
                sample(times["thin"], boxes=((40, 6, 1200, 44),)),
                sample(times["two_line"], lines=2, boxes=((40, 6, 1200, 44),)),
                sample(times["leaking"], is_text=False, lines=0, boxes=(), glyph_level=None,
                       stroke_px=None, gate_at_value=True)]))
    made = BrightnessTab(controller)
    made.page().resize(1100, 700)
    made.page().show()
    made.set_file(NAME)
    settle()
    big = {t: text_strip(dim_rows=14, bright_rows=14, width=1344, height=55) for t in times.values()}
    submission = fake_runner.last("strips", NAME)
    fake_runner.finish(submission, StripsResult(NAME, BOX, big))
    controller.drain_events()
    made.pin_time(600.0)
    settle()
    submission = fake_runner.last("strips", NAME)
    fake_runner.finish(submission, StripsResult(NAME, BOX,
                                                {600.0: text_strip(width=1344, height=55)}))
    controller.drain_events()
    made.set_zoom("300%")
    settle()
    assert len(made.tiles()) == 6
    assert all(tile.has_pixels() for tile in made.tiles()), "the strips never loaded"

    started = time_mod.perf_counter()
    for step in range(50):
        made.set_preview(190 + step % 40)
        QApplication.processEvents()
    elapsed = (time_mod.perf_counter() - started) * 1000
    made.page().close()
    assert elapsed < 250, f"50 threshold changes took {elapsed:.0f} ms"


# --------------------------------------------------------------------------
# Real pixels: the losing threshold on a reference episode
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.needs_media
def test_the_losing_threshold_holds_on_a_reference_episodes_real_strips(
        reference_media, controller, fake_runner, tmp_path, capsys):
    """The self-calibrating rule, on the pixels the OCR pass really sees.

    Glyph pixels include every anti-aliased pixel above the Otsu split, and a
    real subtitle threshold always eats part of that skirt -- which is why
    the note measures each tile against its own loss at the detector's value
    instead of against an absolute bar. This runs the real detector on a
    reference episode, re-grabs its chosen tiles with
    `ocr_view.grab_ocr_strips_at`, and checks that

      (a) nothing fires at the detector's own value, and
      (b) something does fire above it.

    The reference project is never written to: the episode is symlinked into
    tmp_path and the controller opens that.
    """
    from core.detect import brightness as brightness_mod
    from core.detect import ocr_view
    from core.detect.tiles import choose_tiles
    from videocr import engine_registry

    slay = reference_media.get("slay")
    if slay is None or not slay["crop"]:
        pytest.skip("the slay reference project is not present")
    video, box = slay["video"], tuple(slay["crop"])
    ranges = (json.loads((slay["dir"] / ".ocr.json").read_text(encoding="utf-8"))
              .get("files", {}).get(video.name, {}).get("time_ranges") or None)

    with engine_registry.lease_detection_engine(None, True) as det, \
            engine_registry.lease_ocr_engine("ch", None, None, True) as ocr:
        result = brightness_mod.detect_brightness(str(video), box, ranges, det, ocr)
    tiles = choose_tiles(result.strips, result.value)
    assert tiles, "the detector chose no tiles"
    assert result.plateau is not None, f"no plateau to calibrate from (flagged {result.flagged})"

    strips = dict(ocr_view.grab_ocr_strips_at(str(video), box, sorted(set(tiles.values()))))
    assert strips, "no strips came back"

    folder = tmp_path / "reference"
    folder.mkdir()
    (folder / video.name).symlink_to(video)          # never write into the reference project
    controller.open_folder(str(folder))
    entry = controller.entry(video.name)
    entry.crop = Crop(*box, Source.DETECTED)
    entry.brightness = Brightness(result.value, Source.DETECTED)
    entry.evidence["brightness"] = {**result.to_evidence(),
                                    "tiles": {k: float(t) for k, t in tiles.items()},
                                    "crop_box": list(box), "value_crop_box": list(box)}

    made = BrightnessTab(controller)
    made.page().resize(1100, 700)
    made.page().show()
    made.set_file(video.name)
    deliver(controller, fake_runner, name=video.name, box=box, strips=strips)
    made.refresh()
    settle()

    lo, hi = result.plateau
    baselines = []
    for tile in made.tiles():
        pixels = made.strip_pixels(tile.time)
        if tile.kind == "leaking" or pixels is None or not pixels.has_glyphs():
            continue
        baselines.append((tile.kind, tile.time, pixels.lost_percent(result.value),
                          pixels.first_losing_threshold(lo, (pixels.lost_percent(result.value) or 0) + 10)))
    losing = made.losing_threshold()
    with capsys.disabled():
        print(f"\n{video.name}: auto={result.value} plateau={result.plateau} "
              f"flagged={result.flagged}")
        for kind, at, baseline, trips in baselines:
            print(f"  {kind:9s} t={at:8.2f}s  lost at auto={baseline:5.1f}%  "
                  f"+10 points at {trips}")
        print(f"  note fires at {losing}")

    assert baselines, "no text tile with measurable glyphs"
    assert losing is not None, (
        f"nothing ever trips above the auto value {result.value} (plateau {lo}-{hi})")
    assert losing > result.value, (
        f"the note fires at {losing}, at or below the detector's own value {result.value}")
