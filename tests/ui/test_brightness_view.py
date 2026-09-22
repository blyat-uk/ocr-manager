"""The Brightness review tab (app/views/brightness_view.py): the gallery of
subtitle lines (docs/superpowers/specs/2026-09-18-brightness-gallery-design.md
section 3), its per-tile zoom, the threshold curve and the inspector panel.

Nothing is decoded. Strips are synthetic numpy arrays delivered through
`fake_runner` exactly as a real `StripJob` would deliver them, so the tests
exercise the controller's `request_strips` / `strip` contract too. Lines
evidence is written straight into `entry.evidence["lines"]`, the shape
`core.jobs.apply.apply_lines` stores; `controller.request_lines` /
`shuffle_lines` are recorded, not run -- the auto-pilot behind them has its
own tests.

The strips are built so every masked quantity is exact: a text strip's glyph
pixels sit at two known levels, so the percentage lost at a threshold between
them is a whole number.
"""
from __future__ import annotations

import json
import math
import time as time_mod
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QFont, QFontMetrics, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton, QWidget

from app.controller import ProjectController
from app.masking import StripPixels
from app.views import ranges_view
from app.views.brightness_view import (
    APPLY_ALL_BRIGHTNESS_TEXT,
    APPLY_ALL_TITLE,
    CAPTION_GAP,
    CURVE_OTHER_CROP,
    EMPTY_FINDING,
    EMPTY_NO_CROP,
    EMPTY_NO_LINES,
    EMPTY_NOT_DRAWN,
    FIT_MARGIN,
    GALLERY_SIZE,
    KIND_LABELS,
    MAX_DEVICE_ZOOM,
    MIN_GAP_SEC,
    SHUFFLE_TEXT,
    TILE_LOADING,
    TILE_UNREADABLE,
    WHEEL_STEP,
    BrightnessTab,
    GalleryTile,
    ZoomTile,
    caption_texts,
    gallery_plan,
)
from app.views.stage import Stage
from app.views.tabs import evidence_tabs
from core.detect import lines as lines_mod
from core.detect import ocr_view
from core.jobs.view_jobs import StripsResult
from core.project import Brightness, Crop, Source

NAME = "ep01.mkv"
OTHERS = ["ep02.mkv", "ep03.mkv", "ep04.mkv"]
REPO_ROOT = Path(__file__).resolve().parents[2]
BOX = (288, 786, 1344, 53)
OTHER_BOX = (300, 800, 1300, 60)

STRIP_W, STRIP_H = 240, 40
TEXT_BOX = (20, 8, 200, 24)                 # x, y, w, h inside the strip
TILE_TIMES = {"dark": 10.0, "bright": 20.0, "thin": 30.0, "two_line": 40.0, "leaking": 50.0}
PICKS = ["dark", "bright", "thin", "two_line"]
# 11.0 is within MIN_GAP_SEC of the dark pick (10.0); the rest are clear of
# every pick. In time order the gallery takes 5.0 and 25.0 after the four
# picks and is full.
LINE_TIMES = (5.0, 11.0, 25.0, 61.0, 70.0, 80.0, 90.0)

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
    gray = np.zeros((STRIP_H, STRIP_W), dtype=np.uint8)
    start = (STRIP_W - STRIP_H) // 2
    noise = np.indices((STRIP_H, STRIP_H)).sum(axis=0) % 2
    gray[:, start:start + STRIP_H] = (noise * level).astype(np.uint8)
    return _bgr(gray)


def strips_for(times=TILE_TIMES, line_times=LINE_TIMES) -> dict[float, np.ndarray]:
    """Only the "thin" strip has dim glyphs among the picks, so only it loses
    strokes at the detector's value: the note's sample is unambiguous. The
    lines are all-bright text too."""
    strips = {times["dark"]: text_strip(dim_rows=0, bright_rows=12),
              times["bright"]: text_strip(dim_rows=0, bright_rows=12),
              times["thin"]: text_strip(),
              times["two_line"]: two_line_strip(),
              times["leaking"]: empty_strip()}
    for time in line_times:
        strips[float(time)] = text_strip(dim_rows=0, bright_rows=12)
    return strips


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
    }
    if crop_box is not None:
        evidence["crop_box"] = list(crop_box)
    if value_crop_box is not None:
        evidence["value_crop_box"] = list(value_crop_box)
    return evidence


def lines_evidence(times=LINE_TIMES, *, crop_box=BOX, tried: int = 24, boxes=(TEXT_BOX,),
                   lines: int = 1) -> dict:
    """`evidence["lines"]` as `apply_lines` stores it (spec section 1)."""
    evidence = {"seed": 7, "tried": tried,
                "samples": [{"time": float(t), "boxes": [list(box) for box in boxes],
                             "lines": lines} for t in times]}
    if crop_box is not None:
        evidence["crop_box"] = list(crop_box)
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
    # The auto-pilot behind these has its own tests; here they only record
    # (see `recorded`).
    made.recorded = {name: Calls(made, name, run=False) for name in ("request_lines", "shuffle_lines")}
    yield made
    made.shutdown(timeout=0.5)


def give_values(controller, name: str = NAME, *, box=BOX, value: int | None = 209, evidence=None,
                lines=None, source: Source = Source.DETECTED) -> None:
    entry = controller.entry(name)
    entry.crop = None if box is None else Crop(*box, source)
    entry.brightness = None if value is None else Brightness(value, source)
    if evidence is not None:
        entry.evidence["brightness"] = evidence
    if lines is not None:
        entry.evidence["lines"] = lines


def make_tab(controller, name: str | None = NAME, size=(880, 620)) -> BrightnessTab:
    made = BrightnessTab(controller)
    page = made.page()
    page.resize(*size)
    page.show()
    made.inspector_panel().resize(322, 400)
    made.set_file(name)
    settle()
    return made


@pytest.fixture
def tab(controller):
    """A BrightnessTab over a file with full brightness evidence and no lines
    yet, laid out at a size the tiles can actually draw in."""
    give_values(controller, evidence=brightness_evidence())
    made = make_tab(controller)
    yield made
    made.page().close()


def deliver(controller, fake_runner, *, name: str = NAME, box=BOX, strips=None) -> None:
    """Answer the newest strips job for `name` with `strips` (extra times are
    simply cached); a no-op when it has already been answered."""
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


@pytest.fixture
def mixed(controller, fake_runner):
    """Picks AND lines: the full gallery of six, strips delivered."""
    give_values(controller, evidence=brightness_evidence(), lines=lines_evidence())
    made = make_tab(controller)
    deliver(controller, fake_runner)
    yield made
    made.page().close()


def tile_of(tab: BrightnessTab, kind: str) -> ZoomTile:
    found = [tile for tile in tab.tiles() if tile.kind == kind]
    assert found, f"no {kind} tile among {[t.kind for t in tab.tiles()]}"
    return found[0]


def content_centre(tile: ZoomTile) -> QPointF:
    rect = tile.content_rect()
    return QPointF(rect.x() + rect.width() / 2, rect.y() + rect.height() / 2)


def wheel(widget: QWidget, pos: QPointF, notches: float) -> None:
    event = QWheelEvent(pos, widget.mapToGlobal(pos), QPoint(0, 0), QPoint(0, round(120 * notches)),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)
    settle()


def union_fit(tile: ZoomTile, boxes=(TEXT_BOX,), size=(STRIP_W, STRIP_H)) -> float:
    """What "fit" means (spec section 3): the union of the boxes plus
    FIT_MARGIN, clipped to the strip, fills the tile -- width- and
    height-limited, and inside the wheel's own range."""
    width, height = size
    left = max(0, min(b[0] for b in boxes) - FIT_MARGIN)
    top = max(0, min(b[1] for b in boxes) - FIT_MARGIN)
    right = min(width, max(b[0] + b[2] for b in boxes) + FIT_MARGIN)
    bottom = min(height, max(b[1] + b[3] for b in boxes) + FIT_MARGIN)
    content = tile.content_rect()
    scale = min(content.width() / (right - left), content.height() / (bottom - top))
    return min(max(scale, tile.whole_strip_zoom()), tile.max_zoom())


# --------------------------------------------------------------------------
# The gallery plan
# --------------------------------------------------------------------------

def test_the_detectors_picks_come_first_in_their_order_and_the_leaking_frame_is_left_out(loaded):
    assert [tile.kind for tile in loaded.tiles()] == PICKS
    assert [tile.time for tile in loaded.tiles()] == [TILE_TIMES[kind] for kind in PICKS]


def test_lines_fill_the_gallery_after_the_picks_in_time_order_up_to_six(mixed):
    assert GALLERY_SIZE == lines_mod.LINE_COUNT == 6
    assert [tile.kind for tile in mixed.tiles()] == [*PICKS, "line", "line"]
    # 11.0 is within MIN_GAP_SEC of the dark pick at 10.0; after 5.0 and 25.0
    # the gallery is full.
    assert [tile.time for tile in mixed.tiles()] == [10.0, 20.0, 30.0, 40.0, 5.0, 25.0]


def test_a_line_within_the_minimum_gap_of_a_pick_is_skipped():
    assert MIN_GAP_SEC == lines_mod.MIN_GAP_SEC == 2.0
    evidence = {"brightness": brightness_evidence(tiles={"dark": 10.0}),
                "lines": lines_evidence(times=(8.0, 8.1, 11.99, 12.0, 30.0))}
    plan = gallery_plan(evidence, BOX)
    # 8.0 is exactly 2.0 s away: clear. 8.1 and 11.99 are within the gap.
    assert [(tile.kind, tile.time) for tile in plan] == [
        ("dark", 10.0), ("line", 8.0), ("line", 12.0), ("line", 30.0)]


def test_lines_alone_fill_the_gallery_when_brightness_was_never_measured(controller, fake_runner):
    """An IMPORTED or MANUAL brightness is never measured, so there are no
    picks and no curve -- the file every migrated v1 project is full of. The
    lines are the whole gallery."""
    give_values(controller, source=Source.IMPORTED, lines=lines_evidence())
    made = make_tab(controller)
    assert [tile.kind for tile in made.tiles()] == ["line"] * 6
    assert [tile.time for tile in made.tiles()] == list(LINE_TIMES[:6])
    assert made.empty_text() == ""
    assert made.curve.points() == []
    made.page().close()


def test_picks_measured_on_another_crop_are_left_out(controller, fake_runner):
    """The picks' boxes are in the pixel frame of the crop the detection ran
    on; the gallery is always of the file's own crop, so they cannot come."""
    give_values(controller, box=OTHER_BOX, evidence=brightness_evidence(crop_box=BOX),
                lines=lines_evidence(crop_box=OTHER_BOX))
    requests = Calls(controller, "request_strips")
    made = make_tab(controller)
    assert [tile.kind for tile in made.tiles()] == ["line"] * 6
    assert made.crop_box() == OTHER_BOX
    assert requests.calls and all(tuple(call[1]) == OTHER_BOX for call in requests.calls)
    made.page().close()


def test_picks_with_no_crop_box_on_their_evidence_are_kept(controller):
    """An absent `crop_box` (an older cache) cannot disagree with the file."""
    give_values(controller, evidence=brightness_evidence(crop_box=None))
    made = make_tab(controller)
    assert [tile.kind for tile in made.tiles()] == PICKS
    made.page().close()


def test_lines_drawn_on_another_crop_are_left_out(controller):
    give_values(controller, evidence=brightness_evidence(), lines=lines_evidence(crop_box=OTHER_BOX))
    made = make_tab(controller)
    assert [tile.kind for tile in made.tiles()] == PICKS
    made.page().close()


def test_lines_without_a_crop_box_are_left_out():
    """Lines evidence counts only when its box IS the file's crop."""
    plan = gallery_plan({"lines": lines_evidence(crop_box=None)}, BOX)
    assert plan == []


def test_picks_that_share_a_time_are_one_tile_with_both_names(controller):
    give_values(controller, evidence=brightness_evidence(
        tiles={"dark": 10.0, "bright": 20.0, "thin": 10.0}))
    made = make_tab(controller)
    assert [(tile.kinds, tile.time) for tile in made.tiles()] == [
        (("dark", "thin"), 10.0), (("bright",), 20.0)]
    assert made.tiles()[0].caption_left() == "00:10 · dark scene · thin strokes"
    made.page().close()


def test_a_kind_the_detector_did_not_choose_is_skipped(controller):
    give_values(controller, evidence=brightness_evidence(tiles={"dark": 10.0, "leaking": 50.0}))
    made = make_tab(controller)
    assert [tile.kind for tile in made.tiles()] == ["dark"]
    made.page().close()


def test_each_tile_measures_with_its_own_boxes(controller, fake_runner):
    """A pick's boxes are the brightness sample's at its time; a line's are
    its own lines sample's."""
    pick_box, line_box = (20, 8, 100, 12), (30, 10, 90, 20)
    give_values(controller, evidence=brightness_evidence(
        tiles={"dark": 10.0}, strips=[sample(10.0, boxes=(pick_box,))]),
        lines=lines_evidence(times=(30.0,), boxes=(line_box,), lines=2))
    made = make_tab(controller)
    deliver(controller, fake_runner)
    assert made.strip_pixels(10.0).given_boxes == (pick_box,)
    assert made.strip_pixels(30.0).given_boxes == (line_box,)
    made.page().close()


def test_the_tiles_are_one_column_across_the_whole_stage(mixed):
    tiles = mixed.tiles()
    gallery = mixed.gallery()
    assert len({tile.x() for tile in tiles}) == 1
    assert all(tile.width() == gallery.width() for tile in tiles)
    assert [tile.y() for tile in tiles] == sorted(tile.y() for tile in tiles)
    heights = {tile.height() for tile in tiles}
    assert max(heights) - min(heights) <= 1                  # they share the height


def test_tile_captions_name_the_time_and_the_kind(mixed):
    assert tile_of(mixed, "dark").caption_left() == "00:10 · dark scene"
    assert tile_of(mixed, "two_line").caption_left() == "00:40 · two lines"
    assert mixed.tiles()[4].caption_left() == f"00:05 · {KIND_LABELS['line']}"
    assert "leaking" not in KIND_LABELS


def test_a_narrow_caption_elides_the_kind_rather_than_printing_through_it(qapp):
    metrics = QFontMetrics(QFont())
    left, right = "09:38 · dark scene", "28% of glyph pixels lost"
    wide = metrics.horizontalAdvance(left) + metrics.horizontalAdvance(right) + CAPTION_GAP
    assert caption_texts(metrics, wide, left, right) == (left, right)     # room for both

    narrow = wide - metrics.horizontalAdvance("dark scene")
    drawn_left, drawn_right = caption_texts(metrics, narrow, left, right, "09:38")
    assert drawn_right == right                                # the answer is never cut
    assert drawn_left == "09:38"
    assert (metrics.horizontalAdvance(drawn_left) + CAPTION_GAP
            + metrics.horizontalAdvance(drawn_right)) <= narrow

    tiny = metrics.horizontalAdvance(right) + CAPTION_GAP + metrics.horizontalAdvance("09")
    assert caption_texts(metrics, tiny, left, right, "09:38")[0].endswith("…")

    assert caption_texts(metrics, wide, left, "", "09:38") == (left, "")


# --------------------------------------------------------------------------
# Per-tile zoom
# --------------------------------------------------------------------------

def test_a_tile_opens_at_fit_on_its_text(loaded):
    tile = tile_of(loaded, "dark")
    assert tile.is_fit()
    assert tile.zoom() == pytest.approx(union_fit(tile))
    assert tile.zoom() == pytest.approx(tile.fit_zoom())
    # ... centred on the text, which is entirely on screen.
    x, y, w, h = TEXT_BOX
    assert tile.centre()[0] == pytest.approx(x + w / 2, abs=0.5)
    shown = tile.source_rect()
    assert shown.left() <= x and shown.right() >= x + w - 1
    assert shown.top() <= y and shown.bottom() >= y + h - 1


def lone_tile(width: int, height: int, boxes=(TEXT_BOX,)) -> ZoomTile:
    """A tile of its own, outside any layout: its size is the test's."""
    tile = ZoomTile(GalleryTile(("dark",), 10.0, tuple(boxes), 1))
    tile.resize(width, height)
    tile.set_sample(StripPixels(text_strip(), boxes))
    tile.show_threshold(209, masked=True, lost=True)
    return tile


def test_a_wide_short_tile_fits_by_height_and_a_tall_one_by_width(qapp):
    union_w, union_h = TEXT_BOX[2] + 2 * FIT_MARGIN, TEXT_BOX[3] + 2 * FIT_MARGIN

    wide = lone_tile(1400, 90)
    content = wide.content_rect()
    assert content.width() / union_w > content.height() / union_h
    assert wide.zoom() == pytest.approx(content.height() / union_h)

    tall = lone_tile(300, 400)
    content = tall.content_rect()
    assert content.width() / union_w < content.height() / union_h
    assert tall.zoom() == pytest.approx(content.width() / union_w)


def test_a_resize_refits_a_tile_the_user_has_not_zoomed(qapp):
    tile = lone_tile(1400, 90)
    first = tile.zoom()
    tile.resize(1400, 180)
    assert tile.is_fit()
    assert tile.zoom() > first
    assert tile.zoom() == pytest.approx(tile.content_rect().height() / (TEXT_BOX[3] + 2 * FIT_MARGIN))


def test_a_tile_without_boxes_fits_the_whole_strip(loaded, controller):
    """A pick whose sample the cache lost has no boxes: nothing to fit to but
    the strip, and nothing measured to claim."""
    give_values(controller, evidence=brightness_evidence(tiles={"dark": 10.0}, strips=[]))
    loaded.refresh()
    settle()
    tile = tile_of(loaded, "dark")
    assert tile.has_pixels()
    assert tile.zoom() == pytest.approx(tile.whole_strip_zoom())
    content = tile.content_rect()
    assert tile.whole_strip_zoom() == pytest.approx(min(content.width() / STRIP_W,
                                                        content.height() / STRIP_H))
    assert tile.source_rect().width() == STRIP_W and tile.source_rect().height() == STRIP_H
    assert tile.caption_right() == "not measurable"
    assert tile.status_tone() == "dim" and not tile.is_bad()


def test_the_wheel_zooms_around_the_cursor(loaded):
    """The strip point under the cursor stays under it. (While the whole
    strip is narrower than the tile it is simply centred, so the test first
    zooms in far enough to be inside the strip both ways.)"""
    tile = tile_of(loaded, "dark")
    fit = tile.zoom()
    wheel(tile, content_centre(tile), 4)
    assert tile.zoom() == pytest.approx(fit * WHEEL_STEP ** 4)
    assert not tile.is_fit()
    content = tile.content_rect()
    assert content.width() / tile.zoom() < STRIP_W and content.height() / tile.zoom() < STRIP_H

    before = tile.zoom()
    point = content_centre(tile) + QPointF(-content.width() / 8, 3)      # off-centre, on the text
    under = tile.strip_point_at(point)
    wheel(tile, point, 1)
    assert tile.zoom() == pytest.approx(before * WHEEL_STEP)
    after = tile.strip_point_at(point)
    assert after[0] == pytest.approx(under[0], abs=1e-6)
    assert after[1] == pytest.approx(under[1], abs=1e-6)

    wheel(tile, point, -1)
    assert tile.zoom() == pytest.approx(before)
    assert tile.strip_point_at(point)[0] == pytest.approx(under[0], abs=1e-6)


def test_the_wheel_zooms_only_the_tile_under_the_cursor(loaded):
    others = [tile.zoom() for tile in loaded.tiles()[1:]]
    first = loaded.tiles()[0]
    wheel(first, content_centre(first), 2)
    assert [tile.zoom() for tile in loaded.tiles()[1:]] == others


def test_the_wheel_stops_at_twelve_device_pixels_and_at_the_whole_strip(loaded):
    tile = tile_of(loaded, "dark")
    wheel(tile, content_centre(tile), 40)
    assert MAX_DEVICE_ZOOM == 12
    assert tile.zoom() == pytest.approx(MAX_DEVICE_ZOOM / tile.devicePixelRatioF())
    assert tile.max_zoom() == pytest.approx(tile.zoom())

    wheel(tile, content_centre(tile), -80)
    assert tile.zoom() == pytest.approx(tile.whole_strip_zoom())
    shown = tile.source_rect()
    assert (shown.x(), shown.y(), shown.width(), shown.height()) == (0, 0, STRIP_W, STRIP_H)


def test_drag_pans_the_tile(loaded):
    tile = tile_of(loaded, "dark")
    centre = content_centre(tile)
    wheel(tile, centre, 6)                             # close enough in to have room to pan
    zoom = tile.zoom()
    before = tile.centre()
    start = centre.toPoint()
    QTest.mousePress(tile, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(tile, start + QPoint(40, 0))
    QTest.mouseRelease(tile, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       start + QPoint(40, 0))
    settle()
    after = tile.centre()
    assert after[0] == pytest.approx(before[0] - 40 / zoom, abs=1e-6)   # the strip follows the hand
    assert after[1] == pytest.approx(before[1])
    assert tile.zoom() == pytest.approx(zoom)
    left = after[0] - tile.content_rect().width() / zoom / 2
    assert tile.source_rect().x() == max(0, math.floor(left))


def test_a_pan_stops_at_the_strips_edge(loaded):
    tile = tile_of(loaded, "dark")
    wheel(tile, content_centre(tile), 6)
    start = content_centre(tile).toPoint()
    QTest.mousePress(tile, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(tile, start + QPoint(5000, 0))
    QTest.mouseRelease(tile, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       start + QPoint(5000, 0))
    settle()
    assert tile.source_rect().x() == 0


def test_a_double_click_returns_the_tile_to_fit(loaded):
    tile = tile_of(loaded, "dark")
    fit = tile.zoom()
    wheel(tile, content_centre(tile), 5)
    assert tile.zoom() != pytest.approx(fit)
    QTest.mouseDClick(tile, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      content_centre(tile).toPoint())
    settle()
    assert tile.is_fit()
    assert tile.zoom() == pytest.approx(fit)


def test_zoom_survives_a_threshold_drag_and_a_refresh(loaded):
    tile = tile_of(loaded, "thin")
    wheel(tile, content_centre(tile) + QPointF(-30, 0), 3)
    zoom, centre, shown = tile.zoom(), tile.centre(), tile.source_rect()
    curve = loaded.curve
    point = QPoint(int(curve.x_for(215)), curve.height() // 2)
    QTest.mousePress(curve, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    QTest.mouseMove(curve, point + QPoint(10, 0))
    QTest.mouseRelease(curve, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       point + QPoint(10, 0))
    loaded.refresh()                                   # what the Stage does on file_changed
    settle()
    assert tile_of(loaded, "thin") is tile
    assert (tile.zoom(), tile.centre(), tile.source_rect()) == (zoom, centre, shown)


def test_zoom_resets_when_the_file_changes(loaded, controller, fake_runner):
    tile = tile_of(loaded, "dark")
    wheel(tile, content_centre(tile), 3)
    give_values(controller, OTHERS[0], evidence=brightness_evidence())
    loaded.set_file(OTHERS[0])
    settle()
    loaded.set_file(NAME)
    settle()
    assert all(tile.is_fit() for tile in loaded.tiles())


def test_the_tile_is_drawn_nearest_neighbour(loaded):
    """Every strip pixel is a solid block of device pixels: a smoothed blit
    would invent the in-between greys a lost stroke is judged by. (The
    lost-pixel tint is a blend with a fixed colour, so it is off here: only
    strip levels, and the black outside the strip, may appear.)"""
    tile = tile_of(loaded, "thin")
    loaded.toggle_masked()                             # raw: the background is a level too
    loaded.toggle_lost_pixels()
    wheel(tile, content_centre(tile), 8)
    image = tile.grab().toImage()
    rect = tile.content_rect()
    allowed = {0, BACKGROUND, DIM_GLYPH, BRIGHT_GLYPH}
    seen = set()
    for row in range(rect.y() + 2, rect.bottom() - 2, 3):
        seen |= {image.pixelColor(x, row).red() for x in range(rect.x() + 2, rect.right() - 2)}
    assert seen <= allowed, sorted(seen - allowed)
    assert len(seen - {0}) >= 2, "no edge between two levels is on screen"


# --------------------------------------------------------------------------
# Masking, live
# --------------------------------------------------------------------------

def test_every_threshold_change_remasks_each_tile_once(mixed, monkeypatch):
    calls: list[int] = []
    real = ocr_view.mask

    def spy(strip, t):
        calls.append(int(t))
        return real(strip, t)

    monkeypatch.setattr(ocr_view, "mask", spy)
    mixed.set_preview(205)
    settle()
    assert calls == [205] * len(mixed.tiles()) == [205] * 6

    calls.clear()
    mixed.set_preview(206)
    settle()
    assert calls == [206] * 6


def test_zooming_and_panning_do_not_remask(mixed, monkeypatch):
    calls: list[int] = []
    real = ocr_view.mask
    monkeypatch.setattr(ocr_view, "mask", lambda strip, t: (calls.append(int(t)), real(strip, t))[1])
    tile = mixed.tiles()[0]
    wheel(tile, content_centre(tile), 4)
    QTest.mouseDClick(tile, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      content_centre(tile).toPoint())
    settle()
    assert calls == []


def test_dragging_the_curve_remasks_every_tile_live(mixed):
    """Each drag step re-renders all six tiles synchronously, before the next
    mouse event: nothing is deferred or debounced."""
    curve = mixed.curve
    start = QPoint(int(curve.x_for(190)), curve.height() // 2)
    QTest.mousePress(curve, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    for t in (195, 210, 230):
        QTest.mouseMove(curve, QPoint(round(curve.x_for(t)), curve.height() // 2))
        seen = mixed.threshold()
        for tile in mixed.tiles():
            pixels = mixed.strip_pixels(tile.time)
            shown = tile.source_rect()
            expected = ocr_view.mask(pixels.strip, seen)[shown.y():shown.y() + shown.height(),
                                                         shown.x():shown.x() + shown.width()]
            assert np.array_equal(tile.drawn_pixels(), expected), (tile.kind, tile.time, seen)
    QTest.mouseRelease(curve, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    assert abs(mixed.threshold() - 230) <= 1


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


def test_the_status_is_the_whole_strip_not_the_zoomed_region(loaded):
    """Zooming onto the background must not turn a losing tile green."""
    tile = tile_of(loaded, "thin")
    loaded.set_preview(210)
    settle()
    corner = QPointF(tile.content_rect().right() - 2, tile.content_rect().bottom() - 2)
    wheel(tile, corner, 12)
    assert tile.caption_right() == "50% of glyph pixels lost"


def test_a_two_line_tile_reports_both_lines_kept(loaded):
    loaded.set_preview(210)
    settle()
    tile = tile_of(loaded, "two_line")
    assert tile.caption_right() == "both lines kept"
    assert tile.status_tone() == "ok"


def test_a_two_line_subtitle_line_reports_both_lines_kept(controller, fake_runner):
    give_values(controller, lines=lines_evidence(times=(40.0,), lines=2))
    made = make_tab(controller)
    deliver(controller, fake_runner, strips={40.0: two_line_strip()})
    made.set_preview(210)
    settle()
    assert made.tiles()[0].caption_right() == "both lines kept"
    made.page().close()


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


def test_an_unmeasurable_tile_says_so_instead_of_claiming_success(loaded, controller):
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


def test_unchanged_evidence_reuses_the_measured_strip(mixed):
    times = [tile.time for tile in mixed.tiles()]
    held = [mixed.strip_pixels(t) for t in times]
    assert all(pixels is not None for pixels in held)
    mixed.refresh()
    settle()
    assert [mixed.strip_pixels(t) for t in times] == held
    for pixels in held:
        assert isinstance(pixels.given_boxes, tuple)
        assert all(isinstance(box, tuple) for box in pixels.given_boxes)


# --------------------------------------------------------------------------
# Empty states
# --------------------------------------------------------------------------

def test_a_tile_without_its_strip_says_loading(tab):
    tile = tile_of(tab, "dark")
    assert tile.has_pixels() is False
    assert tile.source_rect() is None
    assert tile.placeholder_text() == TILE_LOADING == "loading…"
    assert tile.caption_left() == "00:10 · dark scene"


def test_a_tile_whose_strip_could_not_be_read_says_so(tab, controller, fake_runner):
    """A strips job that comes back without a time marks it unreadable for
    the session: it is never asked for again, so "loading…" would be a lie."""
    dark = TILE_TIMES["dark"]
    deliver(controller, fake_runner, strips={t: s for t, s in strips_for().items() if t != dark})
    assert controller.strip_unavailable(NAME, BOX, dark) is True
    assert tile_of(tab, "dark").placeholder_text() == TILE_UNREADABLE == "this frame could not be read"
    assert tile_of(tab, "bright").placeholder_text() == ""


def test_the_placeholder_goes_once_the_strip_arrives(tab, controller, fake_runner):
    deliver(controller, fake_runner)
    assert all(tile.placeholder_text() == "" for tile in tab.tiles())


def test_no_crop_says_so(controller):
    give_values(controller, box=None, value=None)
    made = make_tab(controller)
    assert made.tiles() == []
    assert made.empty_text() == EMPTY_NO_CROP == "no crop yet — set one on the Crop tab"
    assert made.empty_label().isVisible()
    made.page().close()


def test_lines_on_their_way_say_so(controller):
    give_values(controller, value=None)
    controller.pending_detectors = lambda: {NAME: {"lines"}}
    made = make_tab(controller)
    assert made.tiles() == []
    assert made.empty_text() == EMPTY_FINDING == "finding subtitle lines…"
    made.page().close()


def test_no_lines_found_names_how_many_frames_were_tried(controller):
    give_values(controller, value=None, lines=lines_evidence(times=(), tried=24))
    made = make_tab(controller)
    assert made.tiles() == []
    assert EMPTY_NO_LINES.format(tried=24) == "no subtitle lines found in 24 frames"
    assert made.empty_text().startswith("no subtitle lines found in 24 frames")
    assert "↻ shuffle" in made.empty_text()
    made.page().close()


def test_lines_neither_held_nor_coming_say_so(controller):
    """Auto-pilot off, say: nothing is on its way, and "finding…" would be a
    promise nothing keeps."""
    give_values(controller, value=None)
    made = make_tab(controller)
    assert made.empty_text() == EMPTY_NOT_DRAWN
    made.page().close()


def test_the_empty_state_is_hidden_while_there_are_tiles(loaded):
    assert loaded.empty_text() == ""
    assert not loaded.empty_label().isVisible()


def test_waiting_for_the_frames_is_gone(controller):
    """It promised frames nothing would ever deliver (spec, "Problem")."""
    source = (REPO_ROOT / "app" / "views" / "brightness_view.py").read_text(encoding="utf-8")
    assert "waiting for the frame" not in source


# --------------------------------------------------------------------------
# Shuffle
# --------------------------------------------------------------------------

def recorded(controller, name: str) -> list[tuple]:
    """The calls of `request_lines` / `shuffle_lines` the controller fixture
    recorded (the list itself, so it grows as the test goes on)."""
    return controller.recorded[name].calls


def test_shuffle_asks_the_controller_for_new_lines(mixed, controller):
    button = mixed.shuffle_button
    assert button.text() == SHUFFLE_TEXT == "↻ shuffle"
    assert button.isEnabled()
    button.click()
    settle()
    assert recorded(controller, "shuffle_lines") == [(NAME,)]


def test_shuffle_is_disabled_without_a_crop(controller):
    give_values(controller, box=None, value=None)
    made = make_tab(controller)
    assert not made.shuffle_button.isEnabled()
    made.page().close()


def test_shuffle_is_disabled_while_lines_are_on_their_way(controller):
    give_values(controller, evidence=brightness_evidence(), lines=lines_evidence())
    pending = {NAME: {"lines"}}
    controller.pending_detectors = lambda: pending
    made = make_tab(controller)
    assert not made.shuffle_button.isEnabled()
    pending.clear()                                     # the job finished ...
    controller.activity_changed.emit()                  # ... which the activity strip hears too
    settle()
    assert made.shuffle_button.isEnabled()
    made.page().close()


def test_shuffle_is_disabled_with_no_file(controller):
    made = make_tab(controller, name=None)
    assert not made.shuffle_button.isEnabled()
    made.page().close()


# --------------------------------------------------------------------------
# Asking for lines
# --------------------------------------------------------------------------

def test_a_tab_nobody_is_looking_at_does_not_ask_for_lines(controller):
    give_values(controller, evidence=brightness_evidence())
    make_tab(controller).page().close()
    assert recorded(controller, "request_lines") == []


def test_the_visible_tab_asks_for_missing_lines(controller):
    give_values(controller, evidence=brightness_evidence())
    made = make_tab(controller)
    made.set_active(True)
    settle()
    assert recorded(controller, "request_lines")
    assert set(recorded(controller, "request_lines")) == {(NAME,)}
    made.page().close()


def test_the_visible_tab_asks_for_lines_drawn_on_another_crop(controller):
    give_values(controller, evidence=brightness_evidence(), lines=lines_evidence(crop_box=OTHER_BOX))
    made = make_tab(controller)
    made.set_active(True)
    settle()
    assert set(recorded(controller, "request_lines")) == {(NAME,)}
    made.page().close()


def test_current_lines_are_not_asked_for_again(controller):
    give_values(controller, evidence=brightness_evidence(), lines=lines_evidence())
    made = make_tab(controller)
    made.set_active(True)
    made.refresh()
    settle()
    assert recorded(controller, "request_lines") == []
    made.page().close()


def test_a_file_without_a_crop_asks_for_nothing(controller):
    give_values(controller, box=None, value=None)
    made = make_tab(controller)
    made.set_active(True)
    settle()
    assert recorded(controller, "request_lines") == []
    made.page().close()


def test_the_stage_tells_the_tab_when_it_is_showing(controller):
    """The Stage owns which tab is visible: switching to Brightness asks for
    the file's lines, and a file chosen while another tab shows does not."""
    give_values(controller, evidence=brightness_evidence())
    give_values(controller, OTHERS[0], evidence=brightness_evidence())
    requests = recorded(controller, "request_lines")
    stage = Stage(controller, evidence_tabs(controller))
    try:
        stage.resize(1100, 800)
        stage.show()
        stage.set_file(NAME)
        settle()
        assert requests == []                             # the Crop tab is showing
        stage.set_current(stage.index_of("Brightness"))
        settle()
        assert requests and set(requests) == {(NAME,)}
        stage.set_file(OTHERS[0])                         # still showing: the new file is asked for
        settle()
        assert (OTHERS[0],) in requests
        seen = len(requests)
        stage.set_current(stage.index_of("Time ranges"))
        stage.set_file(OTHERS[1])
        give_values(controller, OTHERS[1], evidence=brightness_evidence())
        controller.file_changed.emit(OTHERS[1])
        settle()
        assert len(requests) == seen
    finally:
        stage.close()
        stage.deleteLater()


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------

def test_the_timeline_marks_the_gallery_tiles(mixed):
    assert mixed.timeline_slot().sample_marks() == sorted(tile.time for tile in mixed.tiles())


def test_a_timeline_click_highlights_the_nearest_tile(mixed, controller):
    controller.entry(NAME).media.duration = 100.0
    timeline = mixed.timeline_slot()
    timeline.refresh()
    settle()
    x = timeline.x_for(22.0)                           # nearest: the 20 s "bright" pick
    QTest.mouseClick(timeline, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(round(x), 20))
    settle()
    assert mixed.highlighted_tile() is tile_of(mixed, "bright")
    assert [tile.is_highlighted() for tile in mixed.tiles()].count(True) == 1

    mixed.highlight_nearest(6.0)                       # nearest: the 5 s line
    assert mixed.highlighted_tile().time == 5.0


def test_the_tab_leaves_room_for_the_compact_timeline(loaded):
    assert loaded.timeline_slot().height() == ranges_view.TIMELINE_HEIGHT


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
    made = make_tab(controller)
    legend = made.curve.legend_texts()
    assert "■ OCR holds up" in legend
    assert "┅ background clutter still firing" not in legend
    made.page().close()


def test_an_empty_curve_says_it_was_not_verified(controller):
    give_values(controller, evidence=brightness_evidence(curve=[], clutter=[], plateau=None))
    made = make_tab(controller)
    assert made.curve.points() == []
    assert "not verified on this file" in made.curve.legend_texts()
    assert "┅ background clutter still firing" not in made.curve.legend_texts()
    assert made.curve.marker_x()[1] is not None
    made.page().close()


def curve_point(curve, t: int) -> QPoint:
    """A point on the curve whose threshold is exactly `t`."""
    point = QPoint(round(curve.x_for(t)), curve.height() // 2)
    assert curve.t_for(point.x()) == t
    return point


def test_dragging_the_curve_commits_once_on_release(loaded, controller):
    """Every tile follows the drag live; the file takes the value when the
    mouse is let go -- one MANUAL write per gesture, no button."""
    calls = Calls(controller, "set_brightness")
    curve = loaded.curve
    QTest.mousePress(curve, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, curve_point(curve, 200))
    QTest.mouseMove(curve, curve_point(curve, 210))
    QTest.mouseMove(curve, curve_point(curve, 220))
    settle()
    assert loaded.threshold() == 220                        # the drag previews live ...
    assert calls.calls == []                                # ... and writes nothing yet
    QTest.mouseRelease(curve, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       curve_point(curve, 220))
    settle()
    assert calls.calls == [(NAME, 220)]
    assert controller.entry(NAME).brightness == Brightness(220, Source.MANUAL)


def test_a_click_on_the_stored_value_commits_nothing(loaded, controller):
    """A MANUAL write of the same value would freeze a detected one."""
    calls = Calls(controller, "set_brightness")
    curve = loaded.curve
    point = curve_point(curve, 209)
    QTest.mousePress(curve, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    QTest.mouseRelease(curve, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    settle()
    assert calls.calls == []
    assert controller.entry(NAME).brightness.source == Source.DETECTED


def test_use_commits_auto(loaded, controller):
    calls = Calls(controller, "set_brightness")
    controller.set_brightness(NAME, 221)
    loaded.refresh()
    settle()
    assert loaded.panel.use_button.text() == "use 209"
    loaded.panel.use_button.click()
    settle()
    assert calls.calls[-1] == (NAME, 209)
    assert controller.entry(NAME).brightness.value == 209


# --------------------------------------------------------------------------
# Apply to all files
# --------------------------------------------------------------------------

def test_apply_to_all_names_the_value_on_screen(loaded):
    assert loaded.panel.all_button.text() == "apply 209 to all files"
    loaded.set_preview(185)
    settle()
    assert loaded.panel.all_button.text() == "apply 185 to all files"
    assert loaded.panel.all_button.isEnabled()


def test_apply_to_all_asks_once_then_writes_every_file(loaded, controller):
    asked = []
    loaded.confirm = lambda title, text: asked.append((title, text)) or True
    loaded.set_preview(185)
    loaded.panel.all_button.click()
    settle()
    assert asked == [(APPLY_ALL_TITLE, APPLY_ALL_BRIGHTNESS_TEXT.format(value=185, n=4))]
    assert "all 4 files" in asked[0][1]
    for name in [NAME, *OTHERS]:
        assert controller.entry(name).brightness == Brightness(185, Source.MANUAL)
    assert loaded.threshold() == 185                      # the preview is now the file's own value


def test_apply_to_all_asks_a_yes_no_question_that_defaults_to_no(loaded, controller, monkeypatch):
    asked = []

    def question(parent, title, text, buttons, default):
        asked.append((title, default))
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", question)
    loaded.panel.all_button.click()
    settle()
    assert asked == [(APPLY_ALL_TITLE, QMessageBox.StandardButton.No)]
    assert controller.entry(OTHERS[0]).brightness is None or controller.entry(OTHERS[0]).brightness.value != 209


def test_saying_no_to_apply_to_all_changes_nothing(loaded, controller):
    before = {name: controller.entry(name).brightness for name in [NAME, *OTHERS]}
    loaded.confirm = lambda title, text: False
    loaded.set_preview(185)
    loaded.panel.all_button.click()
    settle()
    assert {name: controller.entry(name).brightness for name in [NAME, *OTHERS]} == before


# --------------------------------------------------------------------------
# Inspector panel
# --------------------------------------------------------------------------

def test_the_panel_shows_auto_and_yours(loaded):
    assert loaded.panel.auto_row.value() == "209"
    assert loaded.panel.yours_row.value() == "209"
    assert loaded.panel.yours_row.value_tone() == "acc"
    assert loaded.panel.preview_text() == ""
    # A drag writes the value itself: there is nothing left to "keep".
    assert [button.text() for button in loaded.panel.findChildren(QPushButton)
            if button.text().startswith("keep")] == []


def test_an_imported_brightness_shows_yours_and_no_auto(controller):
    give_values(controller, value=214, source=Source.IMPORTED, lines=lines_evidence())
    made = make_tab(controller)
    assert made.panel.auto_row.value() == "—"
    assert made.panel.yours_row.value() == "214"
    assert not made.panel.use_button.isEnabled()
    assert made.threshold() == 214
    made.page().close()


def test_mid_drag_the_row_shows_the_move_and_the_release_keeps_it(loaded, controller):
    """The row says where the drag is going; no "not kept yet" line comes and
    goes under it while dragging, since letting go is what keeps it."""
    curve = loaded.curve
    QTest.mousePress(curve, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, curve_point(curve, 215))
    settle()
    assert loaded.panel.yours_row.value() == "209 → 215"
    assert loaded.panel.preview_text() == ""

    QTest.mouseRelease(curve, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       curve_point(curve, 215))
    settle()
    assert controller.entry(NAME).brightness.value == 215
    assert loaded.panel.yours_row.value() == "215"
    assert loaded.panel.preview_text() == ""


def test_a_file_with_nothing_kept_says_its_threshold_is_a_preview(controller):
    give_values(controller, evidence=brightness_evidence())
    controller.entry(NAME).brightness = None
    made = make_tab(controller)
    assert made.panel.yours_row.value() == "— → 209"
    assert made.panel.preview_text() == "preview only — 209 is not kept yet"
    made.page().close()


def test_the_note_names_the_safe_range_and_the_losing_threshold(loaded):
    """auto is 209, where the thin sample has already lost its dim glyph rows
    -- that is the loss the detector picked, so it is that tile's baseline.
    The note names the threshold where some tile loses 10 points MORE than it
    had at auto: 241, where the bright rows drop out of every tile. The
    leaking tile left the gallery, and its sentence with it."""
    notes = " ".join(loaded.panel.notes())
    assert "Safe range 190–240." in notes
    assert "Above 241 the 00:10 sample starts losing strokes." in notes
    assert "leaks" not in notes and "Below" not in notes
    assert loaded.losing_threshold() == 241


def test_the_losing_threshold_is_calibrated_against_each_tile_at_the_auto_value(
        controller, fake_runner):
    seen = {}
    for auto in (190, 209):
        give_values(controller, value=auto, evidence=brightness_evidence(value=auto))
        made = make_tab(controller)
        deliver(controller, fake_runner)
        made.refresh()
        settle()
        seen[auto] = " ".join(made.panel.notes())
        assert made.threshold() == auto
        made.page().close()

    assert "Above 201 the 00:30 sample starts losing strokes." in seen[190]
    assert "Above 241 the 00:10 sample starts losing strokes." in seen[209]
    for auto, notes in seen.items():
        assert f"Above {auto} " not in notes


def test_the_losing_threshold_reads_the_subtitle_lines_too(controller, fake_runner):
    """With no measurement at all the lines are the only tiles, and the note
    still says where they start losing strokes."""
    give_values(controller, value=180, source=Source.IMPORTED, lines=lines_evidence(times=(30.0,)))
    made = make_tab(controller)
    deliver(controller, fake_runner, strips={30.0: text_strip()})
    settle()
    assert made.losing_threshold() == 201
    assert "Above 201 the 00:30 sample starts losing strokes." in " ".join(made.panel.notes())
    made.page().close()


def test_a_stale_brightness_says_re_detecting_while_one_is_pending(controller):
    give_values(controller, box=OTHER_BOX,
                evidence=brightness_evidence(crop_box=OTHER_BOX, value_crop_box=BOX))
    controller.pending_detectors = lambda: {NAME: {"brightness"}}
    made = make_tab(controller)
    assert made.panel.stale_text() == "measured on an earlier crop — re-detecting"
    made.page().close()


def test_flags_are_shown_as_a_warn_line(controller):
    give_values(controller, evidence=brightness_evidence(flagged="narrow-plateau?+dim-text?"))
    made = make_tab(controller)
    assert made.panel.flag_text() == "narrow safe range · dim text on some frames"
    made.page().close()


def test_a_detected_value_that_differs_from_the_stored_one_is_named(controller):
    give_values(controller, value=211, source=Source.MANUAL,
                evidence=brightness_evidence(value=209))
    made = make_tab(controller)
    assert "detected 209 · yours 211" in " ".join(made.panel.notes())
    made.page().close()


def test_a_brightness_measured_on_an_earlier_crop_warns(controller):
    give_values(controller, box=OTHER_BOX,
                evidence=brightness_evidence(crop_box=OTHER_BOX, value_crop_box=BOX))
    made = make_tab(controller)
    assert made.panel.stale_text() == "measured on an earlier crop — re-detect to refresh"
    assert made.crop_box() == OTHER_BOX
    made.page().close()


def test_the_panel_leaves_the_series_median_to_the_inspector(controller):
    give_values(controller, evidence=brightness_evidence())
    for name, value in zip(OTHERS, (218, 213, 230), strict=True):
        controller.entry(name).brightness = Brightness(value, Source.DETECTED)
    made = make_tab(controller)
    assert not any("Series median" in note for note in made.panel.notes())
    made.page().close()


def test_evidence_from_another_crop_is_named_and_the_gallery_uses_the_files_own(controller):
    """The stored picks, their boxes and the curve were measured in the
    evidence crop's pixel frame. The gallery drops the picks and shows lines
    of the file's own crop; the curve can only be the old crop's, and the
    panel says both things."""
    give_values(controller, box=OTHER_BOX, evidence=brightness_evidence(crop_box=BOX, value_crop_box=BOX))
    made = make_tab(controller)
    assert made.crop_box() == OTHER_BOX
    assert made.tiles() == []
    assert made.panel.stale_text() == "measured on an earlier crop — re-detect to refresh"
    assert made.panel.crop_note_text() == CURVE_OTHER_CROP
    assert CURVE_OTHER_CROP in made.panel.notes()
    made.page().close()


def test_a_value_stale_on_its_own_does_not_claim_the_curve_is_elsewhere(controller):
    give_values(controller, box=OTHER_BOX,
                evidence=brightness_evidence(crop_box=OTHER_BOX, value_crop_box=BOX))
    made = make_tab(controller)
    assert made.panel.stale_text() == "measured on an earlier crop — re-detect to refresh"
    assert made.panel.crop_note_text() == ""
    made.page().close()


def test_a_manual_brightness_on_an_edited_crop_still_warns(controller):
    give_values(controller, box=OTHER_BOX, source=Source.MANUAL,
                evidence=brightness_evidence(crop_box=BOX, value_crop_box=None))
    made = make_tab(controller)
    assert made.crop_box() == OTHER_BOX
    assert made.panel.stale_text() == "measured on an earlier crop — re-detect to refresh"
    made.page().close()


# --------------------------------------------------------------------------
# Strip requests
# --------------------------------------------------------------------------

def test_set_file_requests_every_tile_strip(controller):
    give_values(controller, evidence=brightness_evidence(), lines=lines_evidence())
    calls = Calls(controller, "request_strips")
    made = make_tab(controller)
    assert calls.calls
    name, box, times = calls.calls[0]
    assert name == NAME and tuple(box) == BOX
    assert sorted(times) == sorted(tile.time for tile in made.tiles())
    made.page().close()


def test_a_crop_change_requests_the_strips_again_for_the_new_box(loaded, controller):
    calls = Calls(controller, "request_strips")
    controller.entry(NAME).evidence["brightness"]["crop_box"] = list(OTHER_BOX)
    controller.set_crop(NAME, OTHER_BOX)
    loaded.refresh()
    settle()
    assert calls.calls, "the new crop box must be re-requested"
    assert tuple(calls.calls[-1][1]) == OTHER_BOX
    assert tile_of(loaded, "dark").has_pixels() is False


# --------------------------------------------------------------------------
# Toolbar and wiring
# --------------------------------------------------------------------------

def test_shuffle_and_the_toggles_live_in_the_stage_head(loaded):
    bar = loaded.toolbar()
    assert isinstance(bar, QWidget)
    for button in (loaded.shuffle_button, loaded.lost_button, loaded.mask_button):
        assert bar.isAncestorOf(button)
        assert not loaded.page().isAncestorOf(button)
    texts = [button.text() for button in bar.findChildren(QWidget) if hasattr(button, "text")]
    assert not any("%" in str(text) for text in texts)              # the zoom presets are gone


def test_the_toolbar_wraps_rather_than_setting_the_windows_minimum_width(loaded):
    bar = loaded.toolbar()
    bar.show()
    bar.resize(bar.one_row_width(), bar.sizeHint().height())
    settle()
    assert bar.rows() == 1

    bar.resize(bar.one_row_width() - 1, bar.sizeHint().height())
    settle()
    assert bar.rows() == 2
    assert bar.minimumSizeHint().width() < bar.one_row_width()

    bar.resize(bar.one_row_width(), bar.sizeHint().height())
    settle()
    assert bar.rows() == 1
    assert bar.sizeHint().width() == bar.one_row_width()


def test_evidence_tabs_puts_the_brightness_tab_in_the_middle(controller):
    tabs = evidence_tabs(controller)
    assert [t.title for t in tabs] == ["Crop", "Brightness", "Time ranges"]
    assert isinstance(tabs[1], BrightnessTab)


def test_the_tab_does_not_refresh_itself_on_every_model_change(loaded, controller):
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
    assert loaded.empty_text() == ""


# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------

def test_fifty_threshold_changes_redraw_six_tiles_quickly(controller, fake_runner):
    boxes = ((40, 6, 1200, 44),)
    times = {kind: float(10 * (index + 1)) for index, kind in enumerate(PICKS)}
    give_values(controller, evidence=brightness_evidence(
        tiles=times,
        strips=[sample(times["dark"], boxes=boxes), sample(times["bright"], boxes=boxes),
                sample(times["thin"], boxes=boxes), sample(times["two_line"], lines=2, boxes=boxes)]),
        lines=lines_evidence(times=(300.0, 600.0), boxes=boxes))
    made = make_tab(controller, size=(1100, 700))
    big = {t: text_strip(dim_rows=14, bright_rows=14, width=1344, height=55)
           for t in (*times.values(), 300.0, 600.0)}
    deliver(controller, fake_runner, strips=big)
    tile = made.tiles()[0]
    wheel(tile, content_centre(tile), 4)
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

    made = make_tab(controller, name=video.name, size=(1100, 700))
    deliver(controller, fake_runner, name=video.name, box=box, strips=strips)
    made.refresh()
    settle()

    lo, hi = result.plateau
    baselines = []
    for tile in made.tiles():
        pixels = made.strip_pixels(tile.time)
        if pixels is None or not pixels.has_glyphs():
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
