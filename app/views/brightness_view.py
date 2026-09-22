"""The Brightness review tab (ui-spec §3.5, ruling B1 as amended by
docs/superpowers/specs/2026-09-18-brightness-gallery-design.md).

The threshold decides which pixels the OCR pass ever sees, and at fit-width
a 1344 px strip squeezes each glyph to about 8 px: broken strokes are
invisible at that scale. So this tab is a gallery of subtitle lines, each
masked at the threshold and each zoomable on its own, and every one of them
re-masks live while the threshold curve is dragged. "Show lost pixels" tints
every glyph pixel the current threshold throws away, so erosion is seen, not
inferred.

The gallery (`gallery_plan`)
    Up to GALLERY_SIZE tiles, one column, each spanning the stage's width (a
    1344x61 strip is ~22:1, so a grid would only waste height). First the
    brightness detector's own picks (`core/detect/tiles.py`) in the order
    dark, bright, thin, two_line -- the frames a threshold is most likely to
    break on -- when that evidence was measured on the file's current crop.
    Then the file's subtitle lines (`evidence["lines"]`, drawn at random by
    `core/detect/lines.py`, cached, re-drawn by "↻ shuffle") in time order,
    skipping any within MIN_GAP_SEC of a pick, until the gallery is full.
    Lines count only when they were drawn on the file's current crop.

    A file whose brightness was IMPORTED or MANUAL is never measured, so it
    has no picks and no curve: its gallery is lines alone, which is the
    point -- every such file used to be an empty page. The detector's
    text-free "leaking" pick is not a gallery tile; the curve's clutter line
    still covers what it showed.

    Every strip is requested for the file's OWN crop, so a tile's boxes (a
    pick's come from `evidence["brightness"]["strips"]` at its time, a
    line's from its lines sample) are in the same pixel frame as its pixels.
    Evidence measured on another crop cannot be put on these strips, which
    is why it is left out of the gallery rather than shown on the wrong box.

Per-tile zoom (`ZoomTile`)
    A tile opens at *fit*: the union of its text boxes plus FIT_MARGIN fills
    the tile, limited by width and height; without boxes the whole strip
    fits. The wheel zooms the tile under the cursor, around the cursor,
    x WHEEL_STEP per notch, from "the whole strip fits" up to
    MAX_DEVICE_ZOOM device pixels per strip pixel. Drag pans; a double-click
    returns to fit. Always nearest-neighbour. The zoom is the tile's own:
    it survives threshold drags and re-renders, and resets when the file
    changes (the tiles are rebuilt).

Pixels
    Every pixel here is OCR-exact: the strips come from the controller's
    `request_strips` / `strip` (`core.detect.ocr_view.grab_ocr_strips_at`)
    for the file's crop box, and are masked with the OCR pass's own filter
    through `app/masking.py` (`app/views/*` may not import `core` -- see
    tests/ui/test_main_window.py's import check; the two numbers this view
    shares with `core.detect.lines` are mirrored below with their source
    named). `core.detect.crop`'s own frame grabber is not an OCR-pixel
    source (it reads a full-width band through a different filter order,
    see the frame-addressing table in `core/detect/__init__.py`) and nothing
    here goes near it.

Live, not generated
    Masking one strip costs about 0.07 ms, so dragging the curve re-renders
    every tile synchronously on the GUI thread: no debounce, no Generate
    button, one mask per tile per threshold change -- and none for a zoom
    or a pan, which only re-slice the strip already masked. The file takes
    the dragged value when the mouse is let go, through
    `controller.set_brightness` (MANUAL): one write per gesture, as the crop
    box and the time ranges commit theirs. "use {auto}" writes the detected
    value back.

Lines are asked for, not waited on
    While this tab is the visible one (`set_active`, which the Stage calls
    on every tab switch) and the file's lines are missing or were drawn on
    another crop, the tab calls `controller.request_lines(file)`, which
    boosts the file's lines job to the front of the GPU queue. It is
    idempotent, and a tab nobody is looking at never calls it. Until lines
    arrive the gallery says what is happening instead of promising frames:
    no crop, finding lines, none found in N frames, or nothing drawn yet.

Evidence is disposable
    `evidence["brightness"]` and `evidence["lines"]` may be partial or gone
    after the cache was deleted, so every key is read with `.get` and junk
    is skipped: no picks, no curve, the markers and "not verified on this
    file".

Layout (ruling B4 moves `tabs-hifi`'s 230 px side panel into the one
persistent inspector, so the stage keeps the full width). "↻ shuffle" and
the two toggles are `toolbar()`, which the Stage mounts in the stage head
(ruling B3); the page below it is:

    gallery        one column of ZoomTile sharing the height, or a centred
                   line saying why there is none
    ThresholdCurve the two curves, the plateau band and both markers
    Timeline       the compact, read-only timeline (ruling B5), ticked at
                   the gallery tiles' times; a click highlights the nearest
                   tile
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QPointF, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.masking import (
    DEFAULT_BRIGHTNESS,
    LOST_ALERT_PERCENT,
    LOST_RISE_POINTS,
    MAX_T,
    MIN_T,
    StripPixels,
    normalise_boxes,
)
from app.state_text import brightness_flag_text, brightness_is_stale, clock
from app.theme import tokens
from app.views.inspector_sections import (
    APPLY_ALL_TITLE,
    ask_yes_no,
    note_label,
    small_button,
)
from app.views.ranges_view import Timeline
from app.widgets.base import KvRow, SectionHeader

# --- copy ------------------------------------------------------------------

PICK_KINDS = ("dark", "bright", "thin", "two_line")       # the detector's, in gallery order
LINE_KIND = "line"                                         # a random subtitle line
KIND_LABELS = {"dark": "dark scene", "bright": "bright bg", "thin": "thin strokes",
               "two_line": "two lines", LINE_KIND: "subtitle line"}

SHUFFLE_TEXT = "↻ shuffle"
SHUFFLE_TIP = "draw other random subtitle lines (the detector's picks stay)"
LOST_TOGGLE_TEXT = "show lost pixels"
MASK_TOGGLE_TEXT = "raw ⇄ masked"
TILE_TIP = "wheel: zoom around the cursor · drag: pan · double-click: fit"
TILE_LOADING = "loading…"
TILE_UNREADABLE = "this frame could not be read"
CURVE_HEADER = "Threshold · keeps pixels where min(B,G,R) ≥ t"
CURVE_HINT = "drag anywhere — every tile redraws in under a millisecond"
CURVE_CAPTION = "■ OCR holds up"
CLUTTER_CAPTION = "┅ background clutter still firing"
NOT_VERIFIED = "not verified on this file"
NOT_MEASURABLE = "not measurable"

# The gallery's empty states, centred where the tiles would be. Each says
# what is true now; none promises something nothing is going to deliver.
EMPTY_NO_CROP = "no crop yet — set one on the Crop tab"
EMPTY_FINDING = "finding subtitle lines…"
EMPTY_NO_LINES = "no subtitle lines found in {tried} frames"
EMPTY_TRY_AGAIN = "↻ shuffle to try again"
# Lines missing and none on their way (auto-pilot off, say): "finding…"
# would be a promise nothing keeps, so this says how to get some.
EMPTY_NOT_DRAWN = "no subtitle lines drawn yet — ↻ shuffle draws some"

STALE_REDETECTING = "measured on an earlier crop — re-detecting"
STALE_REDETECT = "measured on an earlier crop — re-detect to refresh"
# Shown under it while the curve (and the safe range the note names) comes
# from evidence measured on another crop than the file's own. The tiles are
# always of the file's crop (see `gallery_plan`); the curve cannot be
# re-measured by a view, so it says whose it is. The warn line above is
# about the stored VALUE, this one about what is on screen.
CURVE_OTHER_CROP = "the curve shows that crop, not the file's current one"
NO_VALUE = "—"
# Mid-drag the row names both values ("211 → 180"): the file keeps 211 until
# the mouse is let go. With no value stored at all, the row reads "— → 209"
# and this line says the file does not have it -- otherwise the panel would
# claim a value the inspector's Detected section, 150 px below, does not.
PREVIEW_ROW = "{stored} → {preview}"
PREVIEW_NOTE = "preview only — {value} is not kept yet"
# "apply {t} to all files": the value on screen, as MANUAL, for every file not
# skipped (controller.apply_brightness_to_all), after one question.
APPLY_ALL_TEXT = "apply {value} to all files"
APPLY_ALL_BRIGHTNESS_TEXT = ("Set brightness {value} on all {n} files?\n\n"
                             "Each file's own brightness is replaced. Skipped files are left alone.")

# --- the gallery's counts ----------------------------------------------------
#
# Views may not import `core` (tests/ui/test_main_window.py), so these two
# mirror `core.detect.lines` by hand; tests/ui/test_brightness_view.py pins
# that they still agree.

GALLERY_SIZE = 6       # core.detect.lines.LINE_COUNT: a full draw fills the gallery alone
MIN_GAP_SEC = 2.0      # core.detect.lines.MIN_GAP_SEC: a line this close to a pick is the same subtitle

# --- zoom ------------------------------------------------------------------
#
# **Not scaled, deliberately.** These are measurements, not sizes: the upper
# bound is 12 DEVICE pixels per strip pixel -- how far in the user can look at
# one stroke -- whatever the UI scale or the screen's pixel ratio, and the
# fit's margin is in strip pixels. The UI scale changes how big a tile is,
# never what a strip pixel is.

WHEEL_STEP = 1.25      # x per wheel notch
WHEEL_NOTCH = 120      # QWheelEvent.angleDelta() units per notch (Qt's own eighths of a degree)
MAX_DEVICE_ZOOM = 12.0
FIT_MARGIN = 6         # strip pixels around the text boxes' union at fit

# --- geometry (the literal CSS of tabs-hifi.html figure 1) ------------------
#
# Every length here is a mockup pixel put through `tokens.px`, so the tiles
# and the curve grow with the rest of the window (app/theme/tokens.py's
# UI_SCALE); the comments name the mockup's own value, which is the number
# `tokens.px` is called with. Alphas, counts and the fractions below are not
# lengths and are left alone -- and neither the strip arrays nor the boxes
# measured on them are ever scaled.

# The mockup's tile was 96 px of glyphs over an 18 px caption in a 3-column
# grid. One column of six shares the page's height instead, so the glyph
# area only has a floor: six tiles at their minimum plus the curve and the
# timeline must still fit a 900 px screen.
GLYPHS_MIN_HEIGHT = tokens.px(24)
CAPTION_HEIGHT = tokens.px(18)  # .ztile .cap (3px padding, 9.5px text)
TILE_MIN_HEIGHT = GLYPHS_MIN_HEIGHT + CAPTION_HEIGHT
TILE_MIN_WIDTH = tokens.px(120)
TILE_BORDER = 1              # the 1 px border `content_rect` sits inside
CAPTION_PAD_X = tokens.px(6)  # the caption's text, off the tile's edges
CAPTION_GAP = tokens.px(8)    # ... and the least room between its two halves
TILE_GAP = tokens.px(6)
LOST_TINT_ALPHA = 140        # rgba(244,112,125,.55)
# A pen is not snapped to a whole device pixel, so stroke widths scale as
# floats: rounding would collapse deliberately different weights.
HIGHLIGHT_WIDTH = 2 * tokens.UI_SCALE   # the ring round the tile a timeline click chose

PLOT_HEIGHT = tokens.px(64)  # the SVG's viewBox height
LEGEND_HEIGHT = tokens.px(16)
LEGEND_AXIS_GAP = tokens.px(10)   # between an axis label and the first legend chip
MARKER_Y = tokens.px(10)          # the "yours" dot, down from the top of the plot
PLATEAU_ALPHA = 0.08         # rect ... opacity=".08"
CURVE_WIDTH = 1.5 * tokens.UI_SCALE
CLUTTER_WIDTH = 1.2 * tokens.UI_SCALE
MARKER_RADIUS = 3.5 * tokens.UI_SCALE

PAGE_MARGIN_X = tokens.px(12)     # the page's own padding
PAGE_MARGIN_TOP = tokens.px(10)
PAGE_MARGIN_BOTTOM = tokens.px(12)
PAGE_SPACING = tokens.px(10)      # between the gallery, the curve and the timeline
TOOLBAR_GAP = tokens.px(4)        # between two buttons of a group ...
TOOLBAR_SPACING = tokens.px(6)    # ... and between the two groups
PANEL_ROW_SPACING = tokens.px(5)  # between two inspector rows
PANEL_BUTTON_TOP = tokens.px(3)   # above "use {auto}"
PANEL_BUTTON_GAP = tokens.px(6)


def _alpha(colour: str, alpha: float) -> QColor:
    value = QColor(colour)
    value.setAlphaF(alpha)
    return value


def _font(size: float, weight: int | None = None) -> QFont:
    font = QFont()
    font.setFamilies(tokens.FONT_STACK)
    font.setPointSizeF(size * 0.75)          # px -> pt at Qt's 96 dpi logical baseline
    if weight is not None:
        font.setWeight(QFont.Weight(weight))
    return font


def _round_half_up(value: float) -> int:
    return int(value + 0.5) if value >= 0 else -int(-value + 0.5)


def caption_texts(metrics, width: float, left: str, right: str,
                  short: str = "") -> tuple[str, str]:
    """The tile caption's two halves as they are drawn.

    They share one rect, one flush left and one flush right, so a tile too
    narrow for both would print them through each other. The status keeps
    its words: it is the tile's whole answer, and the tile is red or green
    because of it.

    The other half gives way in two steps. `short` is the part worth keeping
    whole (the time), so a tile that cannot hold "09:38 · dark scene" reads
    "09:38" rather than "09:38 · …", which says less in the same space.
    Below even that, it elides."""
    room = float(width) - (metrics.horizontalAdvance(right) + CAPTION_GAP if right else 0.0)
    for text in (left, short):
        if text and metrics.horizontalAdvance(text) <= room:
            return text, right
    return metrics.elidedText(short or left, Qt.TextElideMode.ElideRight,
                              max(0, int(room))), right


# --------------------------------------------------------------------------
# The gallery plan
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GalleryTile:
    """One tile of the gallery: what it is, when, and the boxes its glyphs
    are measured inside (strip pixels of the file's crop)."""

    kinds: tuple[str, ...]            # PICK_KINDS entries sharing this time, or (LINE_KIND,)
    time: float
    boxes: tuple[tuple[int, int, int, int], ...] = ()
    lines: int = 1

    @property
    def kind(self) -> str:
        return self.kinds[0]

    @property
    def key(self) -> tuple:
        """What a tile IS across refreshes: the same key keeps its widget,
        and with it the user's zoom."""
        return self.kinds, self.time


def _as_box(value) -> tuple[int, int, int, int] | None:
    try:
        box = tuple(int(part) for part in value)
    except (TypeError, ValueError):
        return None
    return box if len(box) == 4 else None


def _as_time(value) -> float | None:
    try:
        time = float(value)
    except (TypeError, ValueError):
        return None
    return time if math.isfinite(time) else None


def _boxes(value) -> tuple[tuple[int, int, int, int], ...]:
    try:
        return normalise_boxes(box for box in value or () if _as_box(box) is not None)
    except TypeError:
        return ()


def _lines(value) -> int:
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def _brightness_sample(evidence: dict, time: float) -> dict:
    for sample in evidence.get("strips") or ():
        if isinstance(sample, dict) and _as_time(sample.get("time")) == time:
            return sample
    return {}


def gallery_plan(evidence: dict, crop_box) -> list[GalleryTile]:
    """The gallery for a file whose evidence is `evidence` (the whole
    `entry.evidence`) and whose crop is `crop_box` -- see the module
    docstring. Empty without a crop: nothing can be grabbed."""
    own = _as_box(crop_box) if crop_box is not None else None
    if own is None:
        return []
    plan: list[GalleryTile] = []

    brightness = evidence.get("brightness") or {}
    measured_on = brightness.get("crop_box")
    if measured_on is None or _as_box(measured_on) == own:
        chosen = brightness.get("tiles") or {}
        kinds_at: dict[float, list[str]] = {}
        for kind in PICK_KINDS:
            time = _as_time(chosen.get(kind))
            if time is not None:
                # One strip, one tile: a frame that is both the darkest and
                # the thinnest is shown once, named for both.
                kinds_at.setdefault(time, []).append(kind)
        for time, kinds in kinds_at.items():
            sample = _brightness_sample(brightness, time)
            plan.append(GalleryTile(tuple(kinds), time, _boxes(sample.get("boxes")),
                                    _lines(sample.get("lines"))))

    lines = evidence.get("lines") or {}
    if _as_box(lines.get("crop_box")) == own:
        picks = [tile.time for tile in plan]
        samples = [(time, sample) for sample in lines.get("samples") or ()
                   if isinstance(sample, dict) and (time := _as_time(sample.get("time"))) is not None]
        for time, sample in sorted(samples, key=lambda pair: pair[0]):
            if len(plan) >= GALLERY_SIZE:
                break
            if any(abs(time - pick) < MIN_GAP_SEC for pick in picks):
                continue
            plan.append(GalleryTile((LINE_KIND,), time, _boxes(sample.get("boxes")),
                                    _lines(sample.get("lines"))))
    return plan[:GALLERY_SIZE]


# --------------------------------------------------------------------------
# Tiles
# --------------------------------------------------------------------------

class ZoomTile(QWidget):
    """`.ztile` -- one strip, masked live, zoomable on its own, with its
    caption.

    The view is a scale (device-independent pixels per strip pixel) and the
    strip point at the centre of the glyph area. Neither is stored while the
    tile is at *fit*: it is recomputed from the tile's size, so a window
    resize re-fits a tile the user has not touched. The first wheel notch or
    drag makes them explicit, and from then on they are the user's until a
    double-click (or another file) puts the tile back to fit.

    Everything drawn is computed in `show_threshold` / `_redraw`,
    synchronously, so a test can read the result without the widget ever
    being painted.
    """

    def __init__(self, spec: GalleryTile, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self.setMinimumSize(TILE_MIN_WIDTH, TILE_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setToolTip(TILE_TIP)
        self._pixels: StripPixels | None = None
        self._unreadable = False             # its strip came back unreadable: say so, not "loading…"
        self._scale: float | None = None               # None: at fit
        self._centre: tuple[float, float] | None = None
        self._t = DEFAULT_BRIGHTNESS
        self._masked_on = True
        self._lost_on = True
        self._image: QImage | None = None
        self._overlay: QImage | None = None
        self._target = QRectF()
        self._source: QRect | None = None
        self._drawn: np.ndarray | None = None
        self._lost_count = 0
        self._right = ""
        self._tone = "ok"
        self._drag_from: QPointF | None = None
        self._highlighted = False

    # --- identity -----------------------------------------------------------

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def kinds(self) -> tuple[str, ...]:
        return self.spec.kinds

    @property
    def time(self) -> float:
        return self.spec.time

    @property
    def key(self) -> tuple:
        return self.spec.key

    def set_spec(self, spec: GalleryTile) -> None:
        """New boxes or line count for the same tile (a re-detection landed)."""
        self.spec = spec

    # --- state ------------------------------------------------------------

    def set_sample(self, pixels: StripPixels | None, *, unreadable: bool = False) -> None:
        """The tile's pixels, or None while its strip has not arrived --
        `unreadable` when it never will (the controller could not decode
        it). A strip of another size (another crop) puts the view back to
        fit."""
        if unreadable != self._unreadable:
            self._unreadable = unreadable
            self.update()
        if pixels is self._pixels:
            return
        old, self._pixels = self._pixels, pixels
        if pixels is None or old is None or pixels.strip.shape != old.strip.shape:
            self._scale = self._centre = None
        self.setCursor(Qt.CursorShape.OpenHandCursor if pixels is not None
                       else Qt.CursorShape.ArrowCursor)

    def has_pixels(self) -> bool:
        return self._pixels is not None

    def placeholder_text(self) -> str:
        """What the glyph area says instead of pixels: "loading…" until the
        strip arrives, then nothing -- or, for a strip the controller could
        not decode, that it never will."""
        if self._pixels is not None:
            return ""
        return TILE_UNREADABLE if self._unreadable else TILE_LOADING

    def content_rect(self) -> QRect:
        """The `.glyphs` area, inside the tile's 1 px border and above the
        caption."""
        return QRect(TILE_BORDER, TILE_BORDER, max(0, self.width() - 2 * TILE_BORDER),
                     max(0, self.height() - CAPTION_HEIGHT - TILE_BORDER))

    def source_rect(self) -> QRect | None:
        """The whole strip pixels on screen right now, or None."""
        return self._source

    def caption_left(self) -> str:
        names = " · ".join(KIND_LABELS.get(kind, kind) for kind in self.kinds)
        return f"{clock(self.time)} · {names}"

    def caption_right(self) -> str:
        return self._right

    def status_tone(self) -> str:
        return self._tone

    def is_bad(self) -> bool:
        return self._tone == "bad"

    def drawn_pixels(self) -> np.ndarray | None:
        """The BGR sub-array the tile is currently blitting (masked or raw)."""
        return self._drawn

    def lost_pixel_count(self) -> int:
        """Glyph pixels tinted BAD in the region on screen; 0 with the
        toggle off."""
        return self._lost_count

    def set_highlighted(self, on: bool) -> None:
        if on != self._highlighted:
            self._highlighted = on
            self.update()

    def is_highlighted(self) -> bool:
        return self._highlighted

    # --- the view ---------------------------------------------------------

    def whole_strip_zoom(self) -> float:
        """The scale at which the whole strip fits the glyph area -- the
        wheel's lower bound. 0.0 without pixels or room."""
        content = self.content_rect()
        if self._pixels is None or content.width() <= 0 or content.height() <= 0:
            return 0.0
        return min(content.width() / self._pixels.width, content.height() / self._pixels.height)

    def max_zoom(self) -> float:
        """MAX_DEVICE_ZOOM device pixels per strip pixel, in this widget's
        own (device-independent) pixels."""
        return MAX_DEVICE_ZOOM / max(1e-6, self.devicePixelRatioF())

    def _clamp_zoom(self, scale: float) -> float:
        top = self.max_zoom()
        return min(top, max(min(self.whole_strip_zoom(), top), scale))

    def _fit_region(self) -> tuple[float, float, float, float]:
        """(left, top, right, bottom) in strip pixels: the text boxes' union
        plus FIT_MARGIN, clipped to the strip; the whole strip without
        boxes."""
        pixels = self._pixels
        boxes = pixels.boxes
        if not boxes:
            return 0.0, 0.0, float(pixels.width), float(pixels.height)
        left = max(0, min(box[0] for box in boxes) - FIT_MARGIN)
        top = max(0, min(box[1] for box in boxes) - FIT_MARGIN)
        right = min(pixels.width, max(box[0] + box[2] for box in boxes) + FIT_MARGIN)
        bottom = min(pixels.height, max(box[1] + box[3] for box in boxes) + FIT_MARGIN)
        return float(left), float(top), float(right), float(bottom)

    def fit_zoom(self) -> float:
        """The scale at which the fit region fills the glyph area, limited
        by width and height, inside the wheel's range."""
        content = self.content_rect()
        if self._pixels is None or content.width() <= 0 or content.height() <= 0:
            return 0.0
        left, top, right, bottom = self._fit_region()
        return self._clamp_zoom(min(content.width() / max(1.0, right - left),
                                    content.height() / max(1.0, bottom - top)))

    def is_fit(self) -> bool:
        return self._scale is None

    def zoom(self) -> float:
        """Widget pixels per strip pixel right now."""
        return self.fit_zoom() if self._scale is None else self._clamp_zoom(self._scale)

    def centre(self) -> tuple[float, float]:
        """The strip point (x, y) at the centre of the glyph area."""
        if self._pixels is None:
            return 0.0, 0.0
        if self._centre is None:
            left, top, right, bottom = self._fit_region()
            wanted = ((left + right) / 2, (top + bottom) / 2)
        else:
            wanted = self._centre
        return self._clamp_centre(self.zoom(), *wanted)

    def _clamp_centre(self, scale: float, x: float, y: float) -> tuple[float, float]:
        """Keep the view on the strip. Along an axis the whole strip fits,
        the strip is centred; along one it does not, the view may not run
        past either edge."""
        content = self.content_rect()
        pixels = self._pixels
        if scale <= 0:
            return pixels.width / 2, pixels.height / 2

        def clamp(value: float, view: float, size: int) -> float:
            if view >= size:
                return size / 2
            return min(size - view / 2, max(view / 2, value))

        return (clamp(x, content.width() / scale, pixels.width),
                clamp(y, content.height() / scale, pixels.height))

    def _content_centre(self) -> QPointF:
        return QRectF(self.content_rect()).center()

    def strip_point_at(self, pos: QPointF) -> tuple[float, float]:
        """The strip point (x, y) under widget point `pos`."""
        scale = self.zoom() or 1.0
        x, y = self.centre()
        middle = self._content_centre()
        return x + (pos.x() - middle.x()) / scale, y + (pos.y() - middle.y()) / scale

    def zoom_at(self, notches: float, pos: QPointF) -> None:
        """WHEEL_STEP ** notches, keeping the strip point under `pos` where
        it is (as far as the strip's edges allow)."""
        if self._pixels is None:
            return
        scale = self._clamp_zoom(self.zoom() * WHEEL_STEP ** notches)
        x, y = self.strip_point_at(pos)
        middle = self._content_centre()
        self._scale = scale
        self._centre = self._clamp_centre(scale, x - (pos.x() - middle.x()) / scale,
                                          y - (pos.y() - middle.y()) / scale)
        self._redraw()

    def pan_by(self, dx: float, dy: float) -> None:
        """Move the strip by (dx, dy) widget pixels, as a hand would."""
        if self._pixels is None:
            return
        scale = self.zoom()
        x, y = self.centre()
        self._scale = scale
        self._centre = self._clamp_centre(scale, x - dx / scale, y - dy / scale)
        self._redraw()

    def fit(self) -> None:
        self._scale = self._centre = None
        self._redraw()

    # --- rendering --------------------------------------------------------

    def show_threshold(self, t: int, *, masked: bool, lost: bool) -> None:
        """Re-mask at `t` (one `mask()` per strip per threshold, cached by
        StripPixels) and redraw. The status is the whole strip's, never the
        zoomed region's: zooming onto the background must not turn a
        losing tile green."""
        self._t, self._masked_on, self._lost_on = int(t), masked, lost
        if self._pixels is None:
            self._right, self._tone = "", "ok"
        else:
            self._right, self._tone = self._status(self._pixels, self._t)
        self._redraw()

    def _redraw(self) -> None:
        """Slice the region on screen out of the (already masked) strip.
        No masking happens here, so zoom and pan cost a slice and a blit."""
        self._image = self._overlay = None
        self._drawn = None
        self._source = None
        self._lost_count = 0
        pixels = self._pixels
        content = QRectF(self.content_rect())
        scale = self.zoom()
        if pixels is None or content.width() <= 0 or content.height() <= 0 or scale <= 0:
            self.update()
            return
        x, y = self.centre()
        half_w, half_h = content.width() / scale / 2, content.height() / scale / 2
        left, right = max(0, math.floor(x - half_w)), min(pixels.width, math.ceil(x + half_w))
        top, bottom = max(0, math.floor(y - half_h)), min(pixels.height, math.ceil(y + half_h))
        if right <= left or bottom <= top:
            self.update()
            return
        self._source = QRect(left, top, right - left, bottom - top)
        shown = pixels.masked(self._t) if self._masked_on else pixels.strip
        self._drawn = shown[top:bottom, left:right]
        self._image = _bgr_image(self._drawn)
        middle = content.center()
        self._target = QRectF(middle.x() + (left - x) * scale, middle.y() + (top - y) * scale,
                              (right - left) * scale, (bottom - top) * scale)
        if self._lost_on:
            lost_mask = pixels.lost(self._t)
            if lost_mask is not None:
                window = lost_mask[top:bottom, left:right]
                self._lost_count = int(window.sum())
                if self._lost_count:
                    self._overlay = _tint_image(window, QColor(tokens.BAD), LOST_TINT_ALPHA)
        self.update()

    def _status(self, pixels: StripPixels, t: int) -> tuple[str, str]:
        percent = pixels.lost_percent(t)
        if percent is None:
            # No glyph mask: no boxes, fewer pixels inside them than the
            # detector's own Otsu floor, or a single level. Nothing was
            # measured, so nothing may be claimed -- "strokes solid" here
            # would be a success message about a measurement that never
            # happened, on exactly the frames least worth trusting.
            return NOT_MEASURABLE, "dim"
        if percent >= LOST_ALERT_PERCENT:
            return f"{_round_half_up(percent)}% of glyph pixels lost", "bad"
        if self.spec.lines >= 2:
            return "both lines kept", "ok"
        return "strokes solid", "ok"

    # --- the mouse ----------------------------------------------------------

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if self._pixels is None or not delta:
            event.ignore()
            return
        self.zoom_at(delta / WHEEL_NOTCH, event.position())
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._pixels is not None:
            self._drag_from = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_from is None:
            super().mouseMoveEvent(event)
            return
        position = event.position()
        delta = position - self._drag_from
        self._drag_from = position
        self.pan_by(delta.x(), delta.y())

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_from is not None:
            self._drag_from = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.fit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:
        """The view depends on the glyph area's size (fit is recomputed, an
        explicit zoom is re-clamped), so a resize re-slices. No masking."""
        super().resizeEvent(event)
        if self._pixels is not None:
            self._redraw()

    # --- painting ---------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)   # nearest-neighbour
        bounds = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(bounds, tokens.RADIUS_SEG, tokens.RADIUS_SEG)
        painter.fillPath(path, QColor("#000000"))

        content = QRectF(self.content_rect())
        painter.save()
        painter.setClipRect(content)
        if self._image is not None:
            painter.drawImage(self._target, self._image)
            if self._overlay is not None:
                painter.drawImage(self._target, self._overlay)
        elif self.placeholder_text():
            painter.setPen(QColor(tokens.DIM2))
            painter.setFont(_font(tokens.FONT_SIZE_BTN_SM))
            painter.drawText(content, Qt.AlignmentFlag.AlignCenter, self.placeholder_text())
        painter.restore()

        caption = QRectF(TILE_BORDER, self.height() - CAPTION_HEIGHT,
                         self.width() - 2 * TILE_BORDER, CAPTION_HEIGHT - TILE_BORDER)
        painter.fillRect(caption, QColor(tokens.PANEL))
        painter.setPen(QPen(QColor(tokens.LINE), 1))
        painter.drawLine(QPointF(caption.left(), caption.top()), QPointF(caption.right(), caption.top()))
        painter.setFont(_font(tokens.FONT_SIZE_SCOPE))
        text_area = caption.adjusted(CAPTION_PAD_X, 0, -CAPTION_PAD_X, 0)
        left, right = caption_texts(painter.fontMetrics(), text_area.width(),
                                    self.caption_left(), self._right, clock(self.time))
        painter.setPen(QColor(tokens.DIM2))
        painter.drawText(text_area, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                         left)
        painter.setPen(QColor(_TONE_COLOURS[self._tone]))
        painter.drawText(text_area, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                         right)

        border = tokens.TAG_BAD_BORDER if self.is_bad() else tokens.LINE
        painter.setPen(QPen(QColor(border), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        if self._highlighted:
            inset = HIGHLIGHT_WIDTH / 2
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(QColor(tokens.ACC), HIGHLIGHT_WIDTH))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(inset, inset, -inset, -inset),
                                    tokens.RADIUS_SEG, tokens.RADIUS_SEG)
        painter.end()


_TONE_COLOURS = {"ok": tokens.OK, "warn": tokens.WARN, "bad": tokens.BAD, "dim": tokens.DIM2}


def _bgr_image(array: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(array[:, :, ::-1])
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


def _tint_image(mask: np.ndarray, colour: QColor, alpha: int) -> QImage:
    """A transparent overlay that paints `colour` at `alpha` where `mask`."""
    height, width = mask.shape
    buffer = np.zeros((height, width, 4), dtype=np.uint8)      # B, G, R, A on little-endian
    buffer[mask] = (colour.blue(), colour.green(), colour.red(), alpha)
    return QImage(buffer.data, width, height, 4 * width, QImage.Format.Format_ARGB32).copy()


# --------------------------------------------------------------------------
# Threshold curve
# --------------------------------------------------------------------------

class ThresholdCurve(QWidget):
    """The two curves, the plateau band and both markers, over MIN_T..MAX_T.

    Dragging anywhere moves the preview threshold (`previewed`); letting go
    ends the drag (`released`), and the tab commits the threshold the drag
    last showed -- not one re-read from where the button came up.
    """

    previewed = pyqtSignal(int)
    released = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(PLOT_HEIGHT + LEGEND_HEIGHT)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self._curve: list[tuple[int, float]] = []
        self._clutter: list[tuple[int, float]] = []
        self._plateau: tuple[int, int] | None = None
        self._auto: int | None = None
        self._value: int | None = None
        self._dragging = False

    # --- state ------------------------------------------------------------

    def set_state(self, curve, clutter, plateau, auto: int | None, value: int | None) -> None:
        self._curve = [(int(t), float(score)) for t, score in (curve or ())]
        self._clutter = [(int(t), float(share)) for t, share in (clutter or ())]
        self._plateau = None if plateau is None else (int(plateau[0]), int(plateau[1]))
        self._auto = auto
        self._value = value
        self.update()

    def points(self) -> list[tuple[int, float]]:
        return list(self._curve)

    def clutter_points(self) -> list[tuple[int, float]]:
        return list(self._clutter)

    def verified(self) -> bool:
        return bool(self._curve)

    # --- geometry ---------------------------------------------------------

    def x_for(self, t: float) -> float:
        return (float(t) - MIN_T) / (MAX_T - MIN_T) * self.width()

    def t_for(self, x: float) -> int:
        if self.width() <= 0:
            return MIN_T
        raw = MIN_T + x / self.width() * (MAX_T - MIN_T)
        return int(min(MAX_T, max(MIN_T, _round_half_up(raw))))

    def plot_rect(self) -> QRectF:
        return QRectF(0, 0, self.width(), PLOT_HEIGHT)

    def plateau_rect(self) -> QRectF | None:
        if self._plateau is None:
            return None
        lo, hi = self._plateau
        return QRectF(self.x_for(lo), 0, self.x_for(hi) - self.x_for(lo), PLOT_HEIGHT)

    def marker_x(self) -> tuple[float | None, float | None]:
        """(auto, yours) in widget pixels; None for a marker with no value."""
        return (None if self._auto is None else self.x_for(self._auto),
                None if self._value is None else self.x_for(self._value))

    def axis_texts(self) -> tuple[str, str]:
        return str(MIN_T), str(MAX_T)

    def legend_texts(self) -> list[str]:
        texts = [CURVE_CAPTION] if self._curve else [NOT_VERIFIED]
        if self._clutter:
            texts.append(CLUTTER_CAPTION)       # no curve, no caption for it
        if self._value is not None:
            texts.append(f"▲ {self._value} yours")
        if self._auto is not None:
            texts.append(f"┆ {self._auto} auto")
        return texts

    def _legend_tones(self) -> list[str]:
        tones = [tokens.OK] if self._curve else [tokens.DIM2]
        if self._clutter:
            tones.append(tokens.WARN)
        if self._value is not None:
            tones.append(tokens.ACC)
        if self._auto is not None:
            tones.append(tokens.BLUE)
        return tones

    # --- dragging ---------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._dragging = True
        self.previewed.emit(self.t_for(event.position().x()))

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self.previewed.emit(self.t_for(event.position().x()))

    def mouseReleaseEvent(self, event) -> None:
        if not self._dragging or event.button() != Qt.MouseButton.LeftButton:
            return
        self._dragging = False
        self.released.emit()

    # --- painting ---------------------------------------------------------

    def _polyline(self, series) -> list[QPointF]:
        return [QPointF(self.x_for(t), PLOT_HEIGHT * (1.0 - min(1.0, max(0.0, value))))
                for t, value in series]

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        band = self.plateau_rect()
        if band is not None:
            painter.fillRect(band, _alpha(tokens.OK, PLATEAU_ALPHA))

        line = self._polyline(self._curve)
        if len(line) > 1:
            area = QPainterPath(QPointF(line[0].x(), PLOT_HEIGHT))
            for point in line:
                area.lineTo(point)
            area.lineTo(QPointF(line[-1].x(), PLOT_HEIGHT))
            gradient = QLinearGradient(0, 0, 0, PLOT_HEIGHT)
            gradient.setColorAt(0.0, _alpha(tokens.OK, 0.45))
            gradient.setColorAt(1.0, _alpha(tokens.OK, 0.0))
            painter.fillPath(area, gradient)
            painter.setPen(QPen(QColor(tokens.OK), CURVE_WIDTH))
            painter.drawPolyline(*line)

        clutter = self._polyline(self._clutter)
        if len(clutter) > 1:
            pen = QPen(QColor(tokens.WARN), CLUTTER_WIDTH)
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setDashPattern([4, 3])
            painter.setPen(pen)
            painter.drawPolyline(*clutter)

        auto_x, value_x = self.marker_x()
        if auto_x is not None:
            pen = QPen(QColor(tokens.BLUE), 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setDashPattern([3, 3])
            painter.setPen(pen)
            painter.drawLine(QPointF(auto_x, 0), QPointF(auto_x, PLOT_HEIGHT))
        if value_x is not None:
            painter.setPen(QPen(QColor(tokens.ACC), 2))
            painter.drawLine(QPointF(value_x, 0), QPointF(value_x, PLOT_HEIGHT))
            painter.setBrush(QColor(tokens.ACC))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(value_x, MARKER_Y), MARKER_RADIUS, MARKER_RADIUS)

        self._paint_legend(painter)
        painter.end()

    def _paint_legend(self, painter: QPainter) -> None:
        painter.setFont(_font(tokens.FONT_SIZE_SM))
        row = QRectF(0, PLOT_HEIGHT, self.width(), LEGEND_HEIGHT)
        low, high = self.axis_texts()
        painter.setPen(QColor(tokens.DIM2))
        painter.drawText(row, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), low)
        painter.drawText(row, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), high)
        metrics = painter.fontMetrics()
        texts, tones = self.legend_texts(), self._legend_tones()
        widths = [metrics.horizontalAdvance(text) for text in texts]
        left = metrics.horizontalAdvance(low) + LEGEND_AXIS_GAP
        right = self.width() - metrics.horizontalAdvance(high) - LEGEND_AXIS_GAP
        gaps = max(1, len(texts) + 1)
        spare = max(0.0, (right - left) - sum(widths))
        x = left + spare / gaps
        for text, tone, width in zip(texts, tones, widths, strict=True):
            painter.setPen(QColor(tone))
            painter.drawText(QRectF(x, PLOT_HEIGHT, width, LEGEND_HEIGHT),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text)
            x += width + spare / gaps


# --------------------------------------------------------------------------
# The stage-head toolbar
# --------------------------------------------------------------------------

class WrappingToolbar(QWidget):
    """Two groups of controls on one row where there is room for one row, and
    on two rows where there is not.

    Ruling B3 puts a tab's controls in the stage head, and the head lays its
    tab buttons and the toolbar out in a single row -- so the toolbar's
    minimum width IS part of the window's minimum width. The shell leaves the
    stage 730 px of a 1440 px window at the UI scale and the head's tab
    buttons want 287 of it, so a toolbar that could not break would be a
    window width floor -- it was one when this tab carried seven controls,
    and the window could not be opened at 1440 at all, on a screen exactly
    that wide. A second row costs the head some height, which it has and can
    grow into; a width floor the screen cannot meet is not something the user
    can do anything about.

    `sizeHint` always asks for the one-row width, so the moment the head can
    give it that much the two groups snap back onto one row. `minimumSizeHint`
    is the wider group alone, which is the narrowest this can honestly be.
    """

    def __init__(self, first: QWidget, second: QWidget, gap: int,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._first, self._second = first, second
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(gap)
        grid.setVerticalSpacing(gap)
        # The layout may not pin the widget's own minimum to the arrangement
        # it happens to be in: `minimumSizeHint` below is the honest floor,
        # and it is what lets the head make this narrow enough to wrap.
        grid.setSizeConstraint(QGridLayout.SizeConstraint.SetNoConstraint)
        self._grid = grid
        self._rows = 0
        self._arrange(1)

    def rows(self) -> int:
        """1 or 2 -- how the groups are laid out right now."""
        return self._rows

    def one_row_width(self) -> int:
        return (self._first.sizeHint().width() + self._grid.horizontalSpacing()
                + self._second.sizeHint().width())

    def _arrange(self, rows: int) -> None:
        if rows == self._rows:
            return
        self._rows = rows
        for widget in (self._first, self._second):
            self._grid.removeWidget(widget)
        self._grid.addWidget(self._first, 0, 0)
        self._grid.addWidget(self._second, *((1, 0) if rows == 2 else (0, 1)))
        self.updateGeometry()

    def sizeHint(self):
        """The one-row width whatever the current arrangement: a layout hands
        a widget its size hint before its maximum, so this is what asks for
        the room that would let the second row come back up."""
        return QSize(self.one_row_width(), super().sizeHint().height())

    def minimumSizeHint(self):
        return QSize(max(self._first.minimumSizeHint().width(),
                         self._second.minimumSizeHint().width()),
                     super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._arrange(1 if self.width() >= self.one_row_width() else 2)


# --------------------------------------------------------------------------
# Inspector panel
# --------------------------------------------------------------------------

class BrightnessInspectorPanel(QWidget):
    """The Brightness tab's slice of the inspector (ruling B4): Auto/Yours,
    the note, and "use {auto}". A curve drag writes "yours" itself, on
    release, so there is no button to keep it."""

    use_auto = pyqtSignal()
    apply_all = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(PANEL_ROW_SPACING)
        column.addWidget(SectionHeader("Brightness"))
        self.auto_row = KvRow("Auto", NO_VALUE)
        self.yours_row = KvRow("Yours", NO_VALUE, tone="acc")
        column.addWidget(self.auto_row)
        column.addWidget(self.yours_row)
        # Directly under the row it annotates, above the evidence warnings.
        self.preview_label = note_label()
        self.preview_label.setProperty("tone", "acc")
        column.addWidget(self.preview_label)
        self.stale_label = note_label()
        self.stale_label.setProperty("tone", "warn")
        self.flag_label = note_label()
        self.flag_label.setProperty("tone", "warn")
        self.crop_note = note_label()
        self.note = note_label()
        for label in (self.stale_label, self.crop_note, self.flag_label, self.note):
            column.addWidget(label)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, PANEL_BUTTON_TOP, 0, 0)
        buttons.setSpacing(PANEL_BUTTON_GAP)
        self.use_button = small_button("use")
        self.use_button.clicked.connect(self.use_auto)
        buttons.addWidget(self.use_button)
        buttons.addStretch(1)
        column.addLayout(buttons)
        # Its own row: beside "use" it would outgrow the 322 px inspector.
        bulk = QHBoxLayout()
        bulk.setContentsMargins(0, 0, 0, 0)
        self.all_button = small_button(APPLY_ALL_TEXT.format(value=NO_VALUE), "ghost")
        self.all_button.clicked.connect(self.apply_all)
        bulk.addWidget(self.all_button)
        bulk.addStretch(1)
        column.addLayout(bulk)
        column.addStretch(1)

    def set_state(self, *, auto: int | None, value: int | None, stored: int | None,
                  note: str, flags: str, stale: str, crop_note: str) -> None:
        """`value` is the threshold on screen (the preview) and `stored` the
        one the file has. They differ mid-drag -- the row then shows the move
        rather than the destination alone -- and while the file has no value
        at all, which is the one time the note line says so: a drag in
        progress is kept by letting go, so saying "not kept" then would only
        make a line flicker under the row."""
        kept = value is None or value == stored
        preview = PREVIEW_NOTE.format(value=value) if value is not None and stored is None else ""
        self.auto_row.set_value(NO_VALUE if auto is None else str(auto))
        self.yours_row.set_value(
            NO_VALUE if value is None else
            str(value) if kept else
            PREVIEW_ROW.format(stored=NO_VALUE if stored is None else stored, preview=value),
            tone="acc")
        for label, text in ((self.preview_label, preview), (self.stale_label, stale),
                            (self.crop_note, crop_note),
                            (self.flag_label, flags), (self.note, note)):
            label.setText(text)
            label.setVisible(bool(text))
        self.use_button.setText("use" if auto is None else f"use {auto}")
        self.use_button.setEnabled(auto is not None)
        self.all_button.setText(APPLY_ALL_TEXT.format(value=NO_VALUE if value is None else value))
        self.all_button.setEnabled(value is not None)

    def notes(self) -> list[str]:
        """Every note line on show, top to bottom. The series median is NOT
        among them: the cross-file summary belongs to the inspector's
        Detected section (ruling B4, ui-spec §3.7), which shows it once."""
        return [label.text() for label in (self.preview_label, self.stale_label, self.crop_note,
                                           self.flag_label, self.note) if label.text()]

    def preview_text(self) -> str:
        return self.preview_label.text()

    def flag_text(self) -> str:
        return self.flag_label.text()

    def stale_text(self) -> str:
        return self.stale_label.text()

    def crop_note_text(self) -> str:
        return self.crop_note.text()



# --------------------------------------------------------------------------
# The tab
# --------------------------------------------------------------------------

class BrightnessTab:
    """`StageTab` for Brightness."""

    title = "Brightness"

    def __init__(self, controller):
        self._controller = controller
        self._file: str | None = None
        self._preview_file: str | None = None
        self._preview = DEFAULT_BRIGHTNESS
        self._seen_stored: int | None = None
        self._pixels: dict[float, StripPixels] = {}
        self._tiles: list[ZoomTile] = []
        self._highlighted: ZoomTile | None = None
        self._lost_on = True
        self._masked_on = True
        self._facts: dict = {}
        self._facts_key: tuple | None = None
        self._crop_box: tuple[int, int, int, int] | None = None       # the file's own: the strips'
        self._evidence_box: tuple[int, int, int, int] | None = None   # the one the curve was measured on
        self._active = False               # the visible stage tab (see set_active)
        self._asking = False               # inside controller.request_lines

        self._build_page()
        # "apply {t} to all files" asks first; a test replaces this to answer.
        self.confirm: Callable[[str, str], bool] = lambda title, text: ask_yes_no(self._page, title, text)
        self.panel = BrightnessInspectorPanel()
        self.panel.use_auto.connect(self._commit_auto)
        self.panel.apply_all.connect(self._apply_to_all)
        # Only strips_ready (and the cheap activity sync below): `Stage`
        # (app/views/stage.py) already calls refresh() on file_changed for
        # every tab it hosts, and a second connection here would re-mask all
        # six tiles twice per edit. A tab mounted outside a Stage calls
        # refresh() itself.
        controller.strips_ready.connect(self._on_strips_ready)
        # A lines job queued or finished changes whether "↻ shuffle" may be
        # pressed and what an empty gallery says; neither needs a re-render.
        controller.activity_changed.connect(self._sync_lines_state)

    # --- construction -----------------------------------------------------

    def _build_page(self) -> None:
        page = QWidget()
        page.setObjectName("BrightnessPage")
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        column = QVBoxLayout(page)
        column.setContentsMargins(PAGE_MARGIN_X, PAGE_MARGIN_TOP,
                                  PAGE_MARGIN_X, PAGE_MARGIN_BOTTOM)
        column.setSpacing(PAGE_SPACING)

        shuffles = QWidget()
        shuffle_row = QHBoxLayout(shuffles)
        shuffle_row.setContentsMargins(0, 0, 0, 0)
        shuffle_row.setSpacing(TOOLBAR_GAP)
        # Leading, so a row with room to spare right-aligns its controls the
        # way the head right-aligns the toolbar.
        shuffle_row.addStretch(1)
        self.shuffle_button = small_button(SHUFFLE_TEXT, "ghost")
        self.shuffle_button.setToolTip(SHUFFLE_TIP)
        self.shuffle_button.clicked.connect(self.shuffle)
        shuffle_row.addWidget(self.shuffle_button)
        toggles = QWidget()
        toggle_row = QHBoxLayout(toggles)
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.setSpacing(TOOLBAR_GAP)
        toggle_row.addStretch(1)
        self.lost_button = small_button(LOST_TOGGLE_TEXT, "ghost")
        self.lost_button.set_toggled(True)
        self.lost_button.clicked.connect(self.toggle_lost_pixels)
        self.mask_button = small_button(MASK_TOGGLE_TEXT, "ghost")
        self.mask_button.set_toggled(True)
        self.mask_button.clicked.connect(self.toggle_masked)
        toggle_row.addWidget(self.lost_button)
        toggle_row.addWidget(self.mask_button)
        # Two groups, so the head can put them on two rows when one will not
        # fit -- see WrappingToolbar.
        self._toolbar = WrappingToolbar(shuffles, toggles, TOOLBAR_SPACING)
        self._toolbar.setObjectName("BrightnessToolbar")

        self._gallery = QWidget()
        self._gallery.setObjectName("BrightnessGallery")
        self._gallery_layout = QVBoxLayout(self._gallery)
        self._gallery_layout.setContentsMargins(0, 0, 0, 0)
        self._gallery_layout.setSpacing(TILE_GAP)
        # Where the tiles would be, and only while there are none.
        self._empty = QLabel()
        self._empty.setObjectName("Note")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        self._empty.hide()
        self._gallery_layout.addWidget(self._empty, 1)
        column.addWidget(self._gallery, 1)

        header_hint = QLabel(CURVE_HINT)
        header_hint.setObjectName("Note")
        # The title and this hint are both a line of prose, and at the UI
        # scale the pair is wider than the stage has at a 1440 px window. A
        # non-wrapping QLabel asks for its whole line as a MINIMUM, which
        # would make this row the width the window could not go under; the
        # hint wraps instead, and it is the line that can afford to.
        header_hint.setWordWrap(True)
        column.addWidget(SectionHeader(CURVE_HEADER, header_hint))
        self.curve = ThresholdCurve()
        self.curve.previewed.connect(self.set_preview)
        self.curve.released.connect(self._commit_drag)
        column.addWidget(self.curve)

        self._timeline = Timeline(self._controller, mode="compact")
        self._timeline.seek_requested.connect(self.highlight_nearest)
        column.addWidget(self._timeline)               # under the stage (ruling B5)
        self._page = page

    # --- StageTab ---------------------------------------------------------

    def page(self) -> QWidget:
        return self._page

    def inspector_panel(self) -> QWidget:
        return self.panel

    def toolbar(self) -> QWidget:
        """"↻ shuffle" and the two toggles, which the Stage mounts in the
        stage head (ruling B3) beside the tab buttons -- the same place the
        Crop tab puts envelope / masked / grid."""
        return self._toolbar

    def set_active(self, active: bool) -> None:
        """Whether this is the stage's visible tab. The Stage calls it on
        every tab switch (app/views/stage.py); a tab mounted on its own is
        inactive until told otherwise. Only the visible tab asks the
        controller for lines: boosting a file the user is not looking at
        would jump the GPU queue for nobody."""
        self._active = bool(active)
        if self._active:
            self._ask_for_lines()
            self._sync_lines_state()

    def is_active(self) -> bool:
        return self._active

    def set_file(self, name: str | None) -> None:
        if name != self._file:
            # Another file: every tile, and the zoom it carries, starts over.
            self._clear_tiles()
            self._pixels = {}
            self._facts_key = None
        self._file = name
        self._timeline.set_file(name)
        self.refresh()

    def refresh(self) -> None:
        entry = self._entry()
        self._timeline.refresh()
        evidence = (entry.evidence or {}) if entry is not None else {}
        brightness = evidence.get("brightness") or {}
        self._crop_box = self._entry_box(entry)
        self._evidence_box = _as_box(brightness.get("crop_box") or ())
        auto = brightness.get("value")
        stored = None if entry is None or entry.brightness is None else entry.brightness.value
        self._sync_preview(stored, auto)
        self._sync_tiles(gallery_plan(evidence, self._crop_box))
        self._timeline.set_marks([tile.time for tile in self._tiles])
        self._request_strips()
        self._load_pixels()
        self._measure(brightness, auto)
        self.curve.set_state(brightness.get("curve"), brightness.get("clutter_curve"),
                             brightness.get("plateau"), auto, self._preview)
        self._render()
        self._update_panel(entry, brightness, auto, stored)
        self._ask_for_lines()
        self._sync_lines_state()

    # --- state ------------------------------------------------------------

    def tiles(self) -> list[ZoomTile]:
        return list(self._tiles)

    def gallery(self) -> QWidget:
        return self._gallery

    def empty_label(self) -> QLabel:
        return self._empty

    def empty_text(self) -> str:
        """What the gallery says instead of tiles; "" while it has tiles (or
        no file)."""
        return self._empty.text()

    def timeline_slot(self) -> Timeline:
        """The compact, read-only timeline under the stage (ruling B5),
        ticked at the gallery tiles' times. A click highlights the nearest
        tile."""
        return self._timeline

    def highlighted_tile(self) -> ZoomTile | None:
        return self._highlighted

    def threshold(self) -> int:
        return self._preview

    def crop_box(self) -> tuple[int, int, int, int] | None:
        """The file's own crop box: the one every strip is grabbed with."""
        return self._crop_box

    def lost_pixels_on(self) -> bool:
        return self._lost_on

    def masked_on(self) -> bool:
        return self._masked_on

    def strip_pixels(self, time: float) -> StripPixels | None:
        """The loaded pixels of the tile at `time`, or None while its strip
        has not arrived."""
        return self._pixels.get(float(time))

    def losing_threshold(self) -> int | None:
        """The threshold the note's "Above {t} …" sentence names, or None
        when no tile ever loses LOST_RISE_POINTS more than it already had at
        the reference value (see `_measure`)."""
        losing = self._facts.get("losing")
        return None if losing is None else losing[0]

    # --- commands ---------------------------------------------------------

    def toggle_lost_pixels(self) -> None:
        self._lost_on = not self._lost_on
        self.lost_button.set_toggled(self._lost_on)
        self._render()

    def toggle_masked(self) -> None:
        self._masked_on = not self._masked_on
        self.mask_button.set_toggled(self._masked_on)
        self._render()

    def set_preview(self, t: int) -> None:
        """Move the preview threshold. Nothing is written: every tile, the
        curve marker and the panel follow, and the file takes the value when
        the drag ends (`_commit_drag`)."""
        value = int(min(MAX_T, max(MIN_T, int(t))))
        if value == self._preview:
            return
        self._preview = value
        self.curve.set_state(self.curve.points(), self.curve.clutter_points(),
                             self._plateau(), self._auto(), value)
        self._render()
        self._refresh_panel()

    def shuffle(self) -> None:
        """Draw other random lines for this file (the detector's picks stay):
        `controller.shuffle_lines`, which excludes the times on show."""
        entry = self._entry()
        if entry is None or self._crop_box is None or self._lines_pending(entry):
            return
        self._controller.shuffle_lines(entry.name)
        self._sync_lines_state()

    def highlight_nearest(self, time: float) -> None:
        """Ring the tile whose time is nearest `time` (a timeline click)."""
        if not self._tiles:
            return
        nearest = min(self._tiles, key=lambda tile: (abs(tile.time - float(time)), tile.time))
        for tile in self._tiles:
            tile.set_highlighted(tile is nearest)
        self._highlighted = nearest

    # --- internals --------------------------------------------------------

    def _entry(self):
        if self._file is None or self._file not in self._controller.names():
            return None
        return self._controller.entry(self._file)

    @staticmethod
    def _entry_box(entry) -> tuple[int, int, int, int] | None:
        crop = None if entry is None else entry.crop
        return None if crop is None else (crop.x, crop.y, crop.width, crop.height)

    def _auto(self) -> int | None:
        entry = self._entry()
        if entry is None:
            return None
        return (entry.evidence.get("brightness") or {}).get("value")

    def _plateau(self):
        entry = self._entry()
        if entry is None:
            return None
        return (entry.evidence.get("brightness") or {}).get("plateau")

    def _sync_preview(self, stored: int | None, auto: int | None) -> None:
        if self._file != self._preview_file:
            self._preview_file = self._file
            self._seen_stored = stored
            self._preview = stored if stored is not None else (
                auto if auto is not None else DEFAULT_BRIGHTNESS)
        elif stored != self._seen_stored:
            self._seen_stored = stored
            if stored is not None:
                self._preview = stored

    def _clear_tiles(self) -> None:
        for tile in self._tiles:
            self._gallery_layout.removeWidget(tile)
            tile.setParent(None)     # removeWidget alone leaves it parented and painting
            tile.deleteLater()
        self._tiles = []
        self._highlighted = None

    def _sync_tiles(self, plan: list[GalleryTile]) -> None:
        """Lay the plan out, keeping the widget -- and so the zoom -- of
        every tile whose key is still in it."""
        if [spec.key for spec in plan] != [tile.key for tile in self._tiles]:
            held = {tile.key: tile for tile in self._tiles}
            for tile in self._tiles:
                self._gallery_layout.removeWidget(tile)
            tiles = []
            for index, spec in enumerate(plan):
                tile = held.pop(spec.key, None) or ZoomTile(spec)
                self._gallery_layout.insertWidget(index, tile, 1)
                tiles.append(tile)
            for tile in held.values():
                tile.setParent(None)
                tile.deleteLater()
            self._tiles = tiles
            if self._highlighted not in tiles:
                self._highlighted = None
        for tile, spec in zip(self._tiles, plan, strict=True):
            tile.set_spec(spec)

    def _request_strips(self) -> None:
        """Ask for every tile's strip, for the file's own crop. The
        controller submits each (box, time) at most once a session, so this
        is safe on every repaint."""
        if self._file is None or self._crop_box is None or not self._tiles:
            return
        self._controller.request_strips(self._file, self._crop_box,
                                        [tile.time for tile in self._tiles])

    def _load_pixels(self) -> None:
        pixels: dict[float, StripPixels] = {}
        for tile in self._tiles:
            strip = (None if self._file is None or self._crop_box is None
                     else self._controller.strip(self._file, self._crop_box, tile.time))
            if strip is None:
                tile.set_sample(None, unreadable=self._file is not None and self._crop_box is not None
                                and self._controller.strip_unavailable(self._file, self._crop_box, tile.time))
                continue
            held = self._pixels.get(tile.time)
            boxes = tile.spec.boxes
            # Re-measured when the pixels OR the boxes change: a re-detection
            # can land new boxes on a strip that is still cached, and the
            # glyph split belongs to the pair, not to the pixels alone.
            if held is None or held.strip is not strip or held.given_boxes != boxes:
                held = StripPixels(strip, boxes)
            pixels[tile.time] = held
            tile.set_sample(held)
        self._pixels = pixels

    def _render(self) -> None:
        for tile in self._tiles:
            tile.show_threshold(self._preview, masked=self._masked_on, lost=self._lost_on)

    def _on_strips_ready(self, name: str) -> None:
        if name != self._file:
            return
        entry = self._entry()
        self._load_pixels()
        self._facts_key = None                  # the note's facts need the new pixels
        evidence = (entry.evidence.get("brightness") or {}) if entry is not None else {}
        self._measure(evidence, evidence.get("value"))
        self._render()
        self._refresh_panel()

    # --- lines --------------------------------------------------------------

    def _lines_pending(self, entry) -> bool:
        return "lines" in self._controller.pending_detectors().get(entry.name, ())

    def _lines_current(self, entry) -> bool:
        """The file has lines, drawn on the crop it has now."""
        lines = entry.evidence.get("lines") or {}
        return bool(lines) and self._crop_box is not None and \
            _as_box(lines.get("crop_box") or ()) == self._crop_box

    def _ask_for_lines(self) -> None:
        """`controller.request_lines` while this tab is on screen and the
        file's lines are missing or stale. Guarded: the controller may emit
        file_changed from inside the call, and the Stage answers that with
        refresh(), which would ask again."""
        if not self._active or self._asking:
            return
        entry = self._entry()
        if entry is None or self._crop_box is None or self._lines_current(entry):
            return
        self._asking = True
        try:
            self._controller.request_lines(entry.name)
        finally:
            self._asking = False

    def _sync_lines_state(self) -> None:
        """"↻ shuffle" and the gallery's empty state, which follow the
        lines job as well as the model."""
        entry = self._entry()
        pending = entry is not None and self._lines_pending(entry)
        self.shuffle_button.setEnabled(entry is not None and self._crop_box is not None
                                       and not pending)
        text = self._empty_state(entry, pending)
        self._empty.setText(text)
        self._empty.setVisible(bool(text))

    def _empty_state(self, entry, pending: bool) -> str:
        if self._tiles or entry is None:
            return ""
        if self._crop_box is None:
            return EMPTY_NO_CROP
        if pending:
            return EMPTY_FINDING
        if self._lines_current(entry):
            tried = (entry.evidence.get("lines") or {}).get("tried") or 0
            return f"{EMPTY_NO_LINES.format(tried=tried)}\n{EMPTY_TRY_AGAIN}"
        return EMPTY_NOT_DRAWN

    # --- the note's facts -------------------------------------------------

    def _measure(self, evidence, auto: int | None = None) -> None:
        """The fact the note needs that does not move with the preview: the
        lowest threshold at which a tile starts losing strokes.

        **The losing threshold is self-calibrating.** "Glyph pixels" are
        everything above the Otsu split inside the tile's boxes, which
        necessarily includes the anti-aliased skirt around every stroke --
        and a subtitle threshold always eats part of that skirt, so a healthy
        tile can sit at 20-30% lost with every stroke core intact. An
        absolute bar would therefore fire on every file. So each tile is
        measured against ITSELF: its lost % at the detector's auto value is
        its baseline, and the note fires at the lowest whole threshold from
        the plateau's `lo` up at which some tile has lost LOST_RISE_POINTS
        more of its glyphs than it had already lost at auto. That is the
        point where the threshold starts taking pixels the detector's own
        pick was keeping. Picks and lines alike: a line is a subtitle too.

        Without an auto value (no evidence: an IMPORTED or MANUAL
        brightness) the baseline is measured at `lo`, or MIN_T without a
        plateau either, so the rule still reads "how much worse than the
        bottom of the range". The per-tile caption is unaffected: it states
        the plain fact (LOST_ALERT_PERCENT of the glyphs gone) rather than a
        judgement about this file.

        Recomputed only when the file, the tiles' pixels or the reference
        value change -- a threshold drag must not re-measure (and must not
        mask a strip behind the tiles' backs, which is what the
        one-mask-per-tile-per-change budget is)."""
        plateau = evidence.get("plateau")
        key = (self._file, tuple(sorted((time, id(held)) for time, held in self._pixels.items())),
               None if plateau is None else tuple(plateau), auto)
        if key == self._facts_key:
            return
        self._facts_key = key
        lo = None if plateau is None else int(plateau[0])
        start = MIN_T if lo is None else lo
        reference = start if auto is None else int(auto)
        losing: tuple[int, float] | None = None
        for tile in self._tiles:
            held = self._pixels.get(tile.time)
            if held is None or not held.has_glyphs():
                continue
            baseline = held.lost_percent(reference) or 0.0
            found = held.first_losing_threshold(start, baseline + LOST_RISE_POINTS)
            if found is not None and (losing is None or found < losing[0]):
                losing = (found, tile.time)
        self._facts = {"lo": lo, "hi": None if plateau is None else int(plateau[1]),
                       "losing": losing}

    def _note_text(self, auto: int | None, stored: int | None) -> str:
        facts = self._facts
        parts: list[str] = []
        if facts.get("lo") is not None:
            parts.append(f"Safe range {facts['lo']}–{facts['hi']}.")
        losing = facts.get("losing")
        if losing is not None:
            # Shown whether or not the preview has passed it: suppressing the
            # line once the user drags past the cliff would take away the one
            # sentence that explains why a tile just turned red.
            parts.append(f"Above {losing[0]} the {clock(losing[1])} sample starts losing strokes.")
        if auto is not None and stored is not None and auto != stored:
            parts.append(f"detected {auto} · yours {stored}")      # ruling C2
        return " ".join(parts)

    @staticmethod
    def _flag_text(evidence) -> str:
        return brightness_flag_text(evidence.get("flagged"))

    def _measured_elsewhere(self, entry) -> bool:
        """The brightness evidence was measured on another crop than the
        file's own. `own is None` (a file with no crop at all) says nothing:
        there is no crop to disagree with."""
        own = self._entry_box(entry)
        return own is not None and self._evidence_box is not None and self._evidence_box != own

    def _stale_text(self, entry) -> str:
        """The warning that what the panel and the curve describe was
        measured on another crop.

        Two ways that happens, and both must say so. `brightness_is_stale`
        is the model's own rule: the stored VALUE was measured on a crop the
        file no longer has -- but it only judges DETECTED/HINT values, so a
        MANUAL brightness never trips it. The other is this view's: the
        evidence (curve, plateau, auto) is of another crop, whatever the
        value's source.

        "…re-detecting" while a brightness measurement for the file is still
        to come -- queued, running, or held by auto-pilot behind the folder's
        ranges analysis (`pending_detectors`, not `running_detectors`: a job
        that has not started yet is still on its way) -- and "…re-detect to
        refresh" when nothing is coming."""
        if entry is None:
            return ""
        if not (self._measured_elsewhere(entry) or brightness_is_stale(entry)):
            return ""
        pending = self._controller.pending_detectors().get(entry.name, frozenset())
        return STALE_REDETECTING if "brightness" in pending else STALE_REDETECT

    def _crop_note_text(self, entry, evidence) -> str:
        """One line saying the curve is not of the crop the file has now.
        The tiles always are (see `gallery_plan`); the warn line above is
        about the stored value, and a value can be stale while the evidence
        still describes the current crop, so the two are separate."""
        return CURVE_OTHER_CROP if self._measured_elsewhere(entry) and evidence.get("curve") else ""

    def _update_panel(self, entry, evidence, auto: int | None, stored: int | None) -> None:
        self.panel.set_state(auto=auto, value=None if entry is None else self._preview,
                             stored=stored,
                             note=self._note_text(auto, stored),
                             flags=self._flag_text(evidence),
                             stale=self._stale_text(entry),
                             crop_note=self._crop_note_text(entry, evidence))

    def _refresh_panel(self) -> None:
        entry = self._entry()
        evidence = (entry.evidence.get("brightness") or {}) if entry is not None else {}
        stored = None if entry is None or entry.brightness is None else entry.brightness.value
        self._update_panel(entry, evidence, evidence.get("value"), stored)

    # --- commits ----------------------------------------------------------

    def _commit_auto(self) -> None:
        auto = self._auto()
        if self._file is not None and auto is not None:
            self._controller.set_brightness(self._file, int(auto))

    def _commit_drag(self) -> None:
        """The drag ended: the threshold on screen becomes the file's MANUAL
        value, one write per gesture as the crop box and the time ranges do.
        A gesture that ends on the stored value writes nothing -- the same
        number as MANUAL would freeze a detected value against every later
        detection."""
        entry = self._entry()
        if entry is None:
            return
        stored = None if entry.brightness is None else entry.brightness.value
        if self._preview != stored:
            self._controller.set_brightness(self._file, int(self._preview))
        self._refresh_panel()     # the Stage refreshes the rest on file_changed

    def _apply_to_all(self) -> None:
        """The value on screen for every file not skipped, after one
        question naming it and the file count."""
        if self._file is None:
            return
        value = int(self._preview)
        count = len(self._controller.bulk_targets(self._file))
        if self.confirm(APPLY_ALL_TITLE, APPLY_ALL_BRIGHTNESS_TEXT.format(value=value, n=count)):
            self._controller.apply_brightness_to_all(self._file, value)
