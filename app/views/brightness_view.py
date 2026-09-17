"""The Brightness review tab (ui-spec §3.5, ruling B1; `tabs-hifi.html`
figure 1 is canonical).

The threshold decides which pixels the OCR pass ever sees, and at fit-width
a 1344 px strip squeezes each glyph to about 8 px: broken strokes are
invisible at that scale. So this tab magnifies. Five tiles the detector
chose (`core/detect/tiles.py`: the darkest scene, the brightest background,
the thinnest strokes, a two-line subtitle and a text-free frame whose
clutter still trips the OCR pass's gate) plus a sixth the user pins show the
same region of each strip at 100/300/600%, nearest-neighbour, and re-mask
live as the threshold moves. "Show lost pixels" tints every glyph pixel the
current threshold throws away, so erosion is seen, not inferred.

Pixels
    Every pixel here is OCR-exact: the strips come from the controller's
    `request_strips` / `strip` (`core.detect.ocr_view.grab_ocr_strips_at`)
    for the file's crop box, and are masked with the OCR pass's own filter
    through `app/masking.py` (`app/views/*` may not import `core` -- see
    tests/ui/test_main_window.py's import check). `core.detect.crop`'s own
    frame grabber is not an OCR-pixel source (it reads a full-width band
    through a different filter order, see the frame-addressing table in
    `core/detect/__init__.py`) and nothing here goes near it.

Live, not generated
    Masking one strip costs about 0.07 ms, so dragging the slider or the
    curve re-renders every tile synchronously on the GUI thread: no
    debounce, no Generate button. A drag only moves a *preview* threshold;
    "use {auto}" and "keep {yours}" are what write a value, through
    `controller.set_brightness` (MANUAL).

Evidence is disposable
    `evidence["brightness"]` may hold nothing but `value_crop_box` after the
    cache was deleted, so every key is read with `.get` and the view falls
    back to its no-evidence presentation: no tiles, no curve, the markers
    and "not verified on this file".

Layout (ruling B4 moves `tabs-hifi`'s 230 px side panel into the one
persistent inspector, so the stage keeps the full width):

    toolbar        zoom presets and the two toggles
    ContextStrip   the whole strip, 26 px, with the draggable amber window
    tiles          3-column grid of ZoomTile, then the dashed PinTile
    ThresholdCurve the two curves, the plateau band and both markers
    timeline slot  68 px kept free for the compact timeline (ruling B5,
                   mounted by plan 3C Task 4)
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen
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
)
from app.state_text import brightness_flag_text, brightness_is_stale, clock
from app.theme import tokens
from app.views.inspector_sections import note_label, small_button
from app.widgets.base import Button, KvRow, SectionHeader

# --- copy ------------------------------------------------------------------

KIND_ORDER = ("dark", "bright", "thin", "two_line", "leaking")
KIND_LABELS = {"dark": "dark scene", "bright": "bright bg", "thin": "thin strokes",
               "two_line": "two lines", "leaking": "background leaking", "pinned": "pinned"}

PIN_TILE_TEXT = "＋ pin a frame you are worried about"
PIN_HINT_TEXT = "click the timeline below to choose a frame"
ZOOM_LABEL = "zoom"
LOST_TOGGLE_TEXT = "show lost pixels"
MASK_TOGGLE_TEXT = "raw ⇄ masked"
CURVE_HEADER = "Threshold · keeps pixels where min(B,G,R) ≥ t"
CURVE_HINT = "drag anywhere — every tile redraws in under a millisecond"
CURVE_CAPTION = "■ OCR holds up"
CLUTTER_CAPTION = "┅ background clutter still firing"
NOT_VERIFIED = "not verified on this file"
NOT_MEASURABLE = "not measurable"
STALE_REDETECTING = "measured on an earlier crop — re-detecting"
STALE_REDETECT = "measured on an earlier crop — re-detect to refresh"
# Shown under it while the tiles and the curve describe the evidence's crop
# rather than the file's own (see BrightnessTab._box_for): the warn line
# above is about the stored VALUE, this one about what is on screen.
TILES_OTHER_CROP = "the tiles and curve show that crop, not the file's current one"
NO_VALUE = "—"

# --- geometry (the literal CSS of tabs-hifi.html figure 1) ------------------

ZOOM_PRESETS = ("fit", "100%", "300%", "600%")
DEFAULT_PRESET = "300%"
PRESET_FACTORS = {"100%": 1.0, "300%": 3.0, "600%": 6.0}

CONTEXT_HEIGHT = 26          # .ctxstrip
WINDOW_BORDER = 1.5          # .ctxstrip .win
WINDOW_FILL_ALPHA = 0.12     # rgba(255,194,71,.12)
TAG_BG = QColor(10, 12, 16, 209)      # .tag background rgba(10,12,16,.82)

GLYPHS_HEIGHT = 96           # .ztile .glyphs -- the mockup's height, and the minimum here
CAPTION_HEIGHT = 18          # .ztile .cap (3px padding, 9.5px text)
TILE_HEIGHT = GLYPHS_HEIGHT + CAPTION_HEIGHT
TILE_COLUMNS = 3
TILE_GAP = 9
LOST_TINT_ALPHA = 140        # rgba(244,112,125,.55)

PLOT_HEIGHT = 64             # the SVG's viewBox height
LEGEND_HEIGHT = 16
PLATEAU_ALPHA = 0.08         # rect ... opacity=".08"
CURVE_WIDTH = 1.5
CLUTTER_WIDTH = 1.2
MARKER_RADIUS = 3.5

TIMELINE_SLOT_HEIGHT = 68    # ruling B5: a 52 px track plus a 16 px speech lane


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


# --------------------------------------------------------------------------
# Tiles
# --------------------------------------------------------------------------

class ZoomTile(QWidget):
    """`.ztile` -- one strip magnified, masked live, with its caption.

    Everything the tile draws is computed in `render()`, synchronously, so a
    threshold change is a numpy pass and a blit, and a test can read the
    result without the widget ever being shown.
    """

    focused = pyqtSignal()
    resized = pyqtSignal()

    def __init__(self, kind: str, time: float, parent: QWidget | None = None):
        super().__init__(parent)
        self.kind = kind
        self.time = float(time)
        self.setMinimumSize(120, TILE_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self._pixels: StripPixels | None = None
        self._is_text = True
        self._lines = 1
        self._image: QImage | None = None
        self._overlay: QImage | None = None
        self._target = QRectF()
        self._source: QRect | None = None
        self._drawn: np.ndarray | None = None
        self._lost_count = 0
        self._right = ""
        self._tone = "ok"

    # --- state ------------------------------------------------------------

    def set_sample(self, pixels: StripPixels | None, *, is_text: bool = True, lines: int = 1) -> None:
        self._pixels = pixels
        self._is_text = is_text
        self._lines = int(lines)

    def has_pixels(self) -> bool:
        return self._pixels is not None

    def content_rect(self) -> QRect:
        """The `.glyphs` area, inside the tile's 1 px border.

        The mockup fixes it at 96 px, which at 300% would show 32 of a 55-row
        strip and cut the tops and bottoms off the strokes -- the one thing
        the tile exists to show. So the tiles share whatever height the page
        has left instead, never less than the mockup's."""
        return QRect(1, 1, max(0, self.width() - 2), max(0, self.height() - CAPTION_HEIGHT - 1))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.resized.emit()

    def source_rect(self) -> QRect | None:
        """The region of the strip this tile is showing, in strip pixels."""
        return self._source

    def caption_left(self) -> str:
        return f"{clock(self.time)} · {KIND_LABELS.get(self.kind, self.kind)}"

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

    # --- rendering --------------------------------------------------------

    def render(self, zoom: float, x_offset: float, t: int, *, masked: bool, lost: bool) -> None:
        self._image = self._overlay = None
        self._drawn = None
        self._source = None
        self._lost_count = 0
        pixels = self._pixels
        if pixels is None or zoom <= 0:
            self._right, self._tone = "", "ok"
            self.update()
            return

        content = self.content_rect()
        source_width = min(pixels.width, max(1, _round_half_up(content.width() / zoom)))
        source_height = min(pixels.height, max(1, _round_half_up(content.height() / zoom)))
        x = int(min(max(0.0, x_offset), max(0, pixels.width - source_width)))
        y = max(0, (pixels.height - source_height) // 2)
        self._source = QRect(x, y, source_width, source_height)

        shown = pixels.masked(t)                       # always: the gate reads it too
        if not masked:
            shown = pixels.strip
        self._drawn = shown[y:y + source_height, x:x + source_width]
        self._image = _bgr_image(self._drawn)
        # Centred when the region does not fill the tile -- at "fit" the whole
        # strip is 11 px tall, and even at 300% a 55-row strip leaves room.
        drawn_width, drawn_height = source_width * zoom, source_height * zoom
        self._target = QRectF(content.x() + max(0.0, (content.width() - drawn_width) / 2),
                              content.y() + max(0.0, (content.height() - drawn_height) / 2),
                              drawn_width, drawn_height)

        if lost:
            lost_mask = pixels.lost(t)
            if lost_mask is not None:
                window = lost_mask[y:y + source_height, x:x + source_width]
                self._lost_count = int(window.sum())
                if self._lost_count:
                    self._overlay = _tint_image(window, QColor(tokens.BAD), LOST_TINT_ALPHA)
        self._right, self._tone = self._status(pixels, t)
        self.update()

    def _status(self, pixels: StripPixels, t: int) -> tuple[str, str]:
        if not self._is_text:
            return ("background leaking", "warn") if pixels.gate(t) else ("clean", "ok")
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
        if self._lines >= 2:
            return "both lines kept", "ok"
        return "strokes solid", "ok"

    # --- painting ---------------------------------------------------------

    def enterEvent(self, event) -> None:
        self.focused.emit()
        super().enterEvent(event)

    def mousePressEvent(self, event) -> None:
        self.focused.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
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
        else:
            painter.setPen(QColor(tokens.DIM2))
            painter.setFont(_font(tokens.FONT_SIZE_BTN_SM))
            painter.drawText(content, Qt.AlignmentFlag.AlignCenter, "waiting for the frame")
        painter.restore()

        caption = QRectF(1, self.height() - CAPTION_HEIGHT, self.width() - 2, CAPTION_HEIGHT - 1)
        painter.fillRect(caption, QColor(tokens.PANEL))
        painter.setPen(QPen(QColor(tokens.LINE), 1))
        painter.drawLine(QPointF(caption.left(), caption.top()), QPointF(caption.right(), caption.top()))
        painter.setFont(_font(tokens.FONT_SIZE_SCOPE))
        text_area = caption.adjusted(6, 0, -6, 0)
        painter.setPen(QColor(tokens.DIM2))
        painter.drawText(text_area, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                         self.caption_left())
        painter.setPen(QColor(_TONE_COLOURS[self._tone]))
        painter.drawText(text_area, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                         self._right)

        border = tokens.TAG_BAD_BORDER if self.is_bad() else tokens.LINE
        painter.setPen(QPen(QColor(border), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.end()


_TONE_COLOURS = {"ok": tokens.OK, "warn": tokens.WARN, "bad": tokens.BAD, "dim": tokens.DIM2}


class PinTile(QWidget):
    """The dashed sixth tile: "＋ pin a frame you are worried about"."""

    clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(120, TILE_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._text = PIN_TILE_TEXT

    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        bounds = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        pen = QPen(QColor(tokens.LINE), 1)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(bounds, tokens.RADIUS_SEG, tokens.RADIUS_SEG)
        painter.setPen(QColor(tokens.DIM2))
        painter.setFont(_font(tokens.FONT_SIZE_BTN_SM))
        painter.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, self._text)
        painter.end()


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
# Context strip
# --------------------------------------------------------------------------

class ContextStrip(QWidget):
    """`.ctxstrip` -- the whole strip at 26 px, raw, with the amber zoom
    window over the region every tile is showing. Dragging the window pans
    the tiles (`panned` carries the new x offset, in strip pixels)."""

    panned = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(CONTEXT_HEIGHT)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self._image: QImage | None = None
        self._strip_size = (0, 0)
        self._offset = 0.0
        self._visible = 0
        self._caption = ""
        self._drag_grab: float | None = None

    def set_state(self, image: QImage | None, strip_size: tuple[int, int],
                  offset: float, visible: int, caption: str) -> None:
        self._image = image
        self._strip_size = strip_size
        self._offset = float(offset)
        self._visible = int(visible)
        self._caption = caption
        self.update()

    def caption(self) -> str:
        return self._caption

    def window_rect(self) -> QRectF:
        width, _height = self._strip_size
        if width <= 0 or self._visible <= 0:
            return QRectF(0, -1, self.width(), self.height() + 2)
        scale = self.width() / width
        return QRectF(self._offset * scale, -1, max(2.0, self._visible * scale), self.height() + 2)

    # --- panning ----------------------------------------------------------

    def _to_strip(self, x: float) -> float:
        width, _height = self._strip_size
        return x * width / self.width() if self.width() else 0.0

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        rect = self.window_rect()
        if rect.left() <= x <= rect.right():
            self._drag_grab = self._to_strip(x) - self._offset
        else:                                        # jump: centre the window here
            self._drag_grab = self._visible / 2
            self.panned.emit(self._to_strip(x) - self._drag_grab)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_grab is None:
            return
        self.panned.emit(self._to_strip(event.position().x()) - self._drag_grab)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_grab = None

    # --- painting ---------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        bounds = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(bounds, tokens.RADIUS_TAG, tokens.RADIUS_TAG)
        painter.fillPath(path, QColor("#000000"))
        painter.save()
        painter.setClipPath(path)
        if self._image is not None:
            painter.drawImage(QRectF(self.rect()), self._image)
        else:
            painter.setPen(QColor(tokens.DIM2))
            painter.setFont(_font(tokens.FONT_SIZE_SCOPE))
            painter.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, "waiting for the frames")
        window = self.window_rect()
        painter.fillRect(window, _alpha(tokens.ACC, WINDOW_FILL_ALPHA))
        painter.setPen(QPen(QColor(tokens.ACC), WINDOW_BORDER))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(window.adjusted(WINDOW_BORDER / 2, 0, -WINDOW_BORDER / 2, 0))
        painter.restore()

        if self._caption:
            painter.setFont(_font(tokens.FONT_SIZE_SCOPE))
            metrics = painter.fontMetrics()
            width = metrics.horizontalAdvance(self._caption) + 12
            tag = QRectF(6, 3, width, CONTEXT_HEIGHT - 9)
            painter.setPen(QPen(QColor(tokens.LINE2), 1))
            painter.setBrush(TAG_BG)
            painter.drawRoundedRect(tag, tokens.RADIUS_TAG, tokens.RADIUS_TAG)
            painter.setPen(QColor(tokens.DIM))
            painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, self._caption)
        painter.setPen(QPen(QColor(tokens.LINE), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.end()


# --------------------------------------------------------------------------
# Threshold curve
# --------------------------------------------------------------------------

class ThresholdCurve(QWidget):
    """The two curves, the plateau band and both markers, over MIN_T..MAX_T.

    Dragging anywhere moves the preview threshold (`previewed`); releasing
    keeps it as the preview -- nothing is committed here.
    """

    previewed = pyqtSignal(int)

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
        self._dragging = False           # the preview stays; committing is the panel's job

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
            painter.drawEllipse(QPointF(value_x, 10), MARKER_RADIUS, MARKER_RADIUS)

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
        left = metrics.horizontalAdvance(low) + 10
        right = self.width() - metrics.horizontalAdvance(high) - 10
        gaps = max(1, len(texts) + 1)
        spare = max(0.0, (right - left) - sum(widths))
        x = left + spare / gaps
        for text, tone, width in zip(texts, tones, widths, strict=True):
            painter.setPen(QColor(tone))
            painter.drawText(QRectF(x, PLOT_HEIGHT, width, LEGEND_HEIGHT),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text)
            x += width + spare / gaps


# --------------------------------------------------------------------------
# Inspector panel
# --------------------------------------------------------------------------

class BrightnessInspectorPanel(QWidget):
    """The Brightness tab's slice of the inspector (ruling B4): Auto/Yours,
    the note, and "use {auto}" / "keep {yours}"."""

    use_auto = pyqtSignal()
    keep_yours = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(5)
        column.addWidget(SectionHeader("Brightness"))
        self.auto_row = KvRow("Auto", NO_VALUE)
        self.yours_row = KvRow("Yours", NO_VALUE, tone="acc")
        column.addWidget(self.auto_row)
        column.addWidget(self.yours_row)
        self.stale_label = note_label()
        self.stale_label.setProperty("tone", "warn")
        self.flag_label = note_label()
        self.flag_label.setProperty("tone", "warn")
        self.crop_note = note_label()
        self.note = note_label()
        for label in (self.stale_label, self.crop_note, self.flag_label, self.note):
            column.addWidget(label)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 3, 0, 0)
        buttons.setSpacing(6)
        self.use_button = small_button("use")
        self.keep_button = small_button("keep", "ghost")
        self.use_button.clicked.connect(self.use_auto)
        self.keep_button.clicked.connect(self.keep_yours)
        buttons.addWidget(self.use_button)
        buttons.addWidget(self.keep_button)
        buttons.addStretch(1)
        column.addLayout(buttons)
        column.addStretch(1)

    def set_state(self, *, auto: int | None, value: int | None, note: str, flags: str,
                  stale: str, crop_note: str) -> None:
        self.auto_row.set_value(NO_VALUE if auto is None else str(auto))
        self.yours_row.set_value(NO_VALUE if value is None else str(value), tone="acc")
        for label, text in ((self.stale_label, stale), (self.crop_note, crop_note),
                            (self.flag_label, flags), (self.note, note)):
            label.setText(text)
            label.setVisible(bool(text))
        self.use_button.setText("use" if auto is None else f"use {auto}")
        self.use_button.setEnabled(auto is not None)
        self.keep_button.setText("keep" if value is None else f"keep {value}")
        self.keep_button.setEnabled(value is not None)

    def notes(self) -> list[str]:
        """Every note line on show, top to bottom. The series median is NOT
        among them: the cross-file summary belongs to the inspector's
        Detected section (ruling B4, ui-spec §3.7), which shows it once."""
        return [label.text() for label in (self.stale_label, self.crop_note,
                                           self.flag_label, self.note) if label.text()]

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
        self._pinned: dict[str, list[float]] = {}
        self._pixels: dict[float, StripPixels] = {}
        self._tiles: list[ZoomTile] = []
        self._focused = 0
        self._zoom = DEFAULT_PRESET
        self._offset = 0.0
        self._offset_ready = False           # the default offset has been measured on real pixels
        self._resize_pending = False
        self._lost_on = True
        self._masked_on = True
        self._context_image: QImage | None = None
        self._context_key: int | None = None
        self._facts: dict = {}
        self._facts_key: tuple | None = None
        self._crop_box: tuple[int, int, int, int] | None = None

        # Plan 3C Task 4 sets this to the compact timeline's current position
        # (seconds, or None when it has none); the pin tile asks it for a time.
        self.timeline_position: Callable[[], float | None] | None = None

        self._build_page()
        self.panel = BrightnessInspectorPanel()
        self.panel.use_auto.connect(self._commit_auto)
        self.panel.keep_yours.connect(self._commit_preview)
        # Only strips_ready: `Stage` (app/views/stage.py) already calls
        # refresh() on file_changed for every tab it hosts, and a second
        # connection here would re-mask all six tiles twice per edit. A tab
        # mounted outside a Stage calls refresh() itself.
        controller.strips_ready.connect(self._on_strips_ready)

    # --- construction -----------------------------------------------------

    def _build_page(self) -> None:
        page = QWidget()
        page.setObjectName("BrightnessPage")
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        column = QVBoxLayout(page)
        column.setContentsMargins(12, 10, 12, 12)
        column.setSpacing(10)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(4)
        toolbar.addStretch(1)
        zoom_label = QLabel(ZOOM_LABEL)
        zoom_label.setObjectName("Note")
        toolbar.addWidget(zoom_label)
        self.zoom_buttons: list[Button] = []
        for preset in ZOOM_PRESETS:
            button = small_button(preset, "ghost")
            button.set_toggled(preset == self._zoom)
            button.clicked.connect(lambda _checked=False, name=preset: self.set_zoom(name))
            toolbar.addWidget(button)
            self.zoom_buttons.append(button)
        toolbar.addSpacing(6)
        self.lost_button = small_button(LOST_TOGGLE_TEXT, "ghost")
        self.lost_button.set_toggled(True)
        self.lost_button.clicked.connect(self.toggle_lost_pixels)
        self.mask_button = small_button(MASK_TOGGLE_TEXT, "ghost")
        self.mask_button.set_toggled(True)
        self.mask_button.clicked.connect(self.toggle_masked)
        toolbar.addWidget(self.lost_button)
        toolbar.addWidget(self.mask_button)
        column.addLayout(toolbar)

        self.context = ContextStrip()
        self.context.panned.connect(self._on_panned)
        column.addWidget(self.context)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(TILE_GAP)
        self._grid.setVerticalSpacing(TILE_GAP)
        for index in range(TILE_COLUMNS):
            self._grid.setColumnStretch(index, 1)
        self.pin_tile_widget = PinTile()
        self.pin_tile_widget.clicked.connect(self._on_pin_clicked)
        column.addWidget(self._grid_host, 1)

        header_hint = QLabel(CURVE_HINT)
        header_hint.setObjectName("Note")
        column.addWidget(SectionHeader(CURVE_HEADER, header_hint))
        self.curve = ThresholdCurve()
        self.curve.previewed.connect(self.set_preview)
        column.addWidget(self.curve)

        self._timeline_slot = QWidget()
        self._timeline_slot.setObjectName("TimelineSlot")
        self._timeline_slot.setFixedHeight(TIMELINE_SLOT_HEIGHT)
        column.addWidget(self._timeline_slot)          # under the stage (ruling B5)
        self._page = page

    # --- StageTab ---------------------------------------------------------

    def page(self) -> QWidget:
        return self._page

    def inspector_panel(self) -> QWidget:
        return self.panel

    def set_file(self, name: str | None) -> None:
        self._file = name
        self._offset, self._offset_ready = 0.0, False
        self.refresh()

    def refresh(self) -> None:
        entry = self._entry()
        evidence = (entry.evidence.get("brightness") or {}) if entry is not None else {}
        self._crop_box = self._box_for(entry, evidence)
        auto = evidence.get("value")
        stored = None if entry is None or entry.brightness is None else entry.brightness.value
        self._sync_preview(stored, auto)
        self._sync_tiles(evidence)
        self._request_strips()
        self._load_pixels()
        self._measure(evidence, auto)
        self.curve.set_state(evidence.get("curve"), evidence.get("clutter_curve"),
                             evidence.get("plateau"), auto, self._preview)
        self._render()
        self._update_panel(entry, evidence, auto, stored)

    # --- state ------------------------------------------------------------

    def tiles(self) -> list[ZoomTile]:
        return list(self._tiles)

    def pin_tile(self) -> PinTile:
        return self.pin_tile_widget

    def grid_widgets(self) -> list[QWidget]:
        return [*self._tiles, self.pin_tile_widget]

    def timeline_slot(self) -> QWidget:
        """The 68 px ruling-B5 placeholder plan 3C Task 4 mounts into."""
        return self._timeline_slot

    def threshold(self) -> int:
        return self._preview

    def crop_box(self) -> tuple[int, int, int, int] | None:
        return self._crop_box

    def zoom_factor(self) -> float:
        preset = PRESET_FACTORS.get(self._zoom)
        if preset is not None:
            return preset
        width = self._strip_width()
        content = self._tiles[0].content_rect().width() if self._tiles else 0
        return content / width if width and content else 1.0

    def x_offset(self) -> float:
        return self._offset

    def lost_pixels_on(self) -> bool:
        return self._lost_on

    def masked_on(self) -> bool:
        return self._masked_on

    def pinned_times(self) -> list[float]:
        return list(self._pinned.get(self._file or "", ()))

    def strip_pixels(self, time: float) -> StripPixels | None:
        """The loaded pixels of the tile at `time`, or None while its strip
        has not arrived."""
        return self._pixels.get(float(time))

    def losing_threshold(self) -> int | None:
        """The threshold the note's "Above {t} …" sentence names, or None
        when no tile ever loses LOST_RISE_POINTS more than it already had at
        the detector's value (see `_measure`)."""
        losing = self._facts.get("losing")
        return None if losing is None else losing[0]

    # --- commands ---------------------------------------------------------

    def set_zoom(self, preset: str) -> None:
        if preset not in ZOOM_PRESETS:
            raise ValueError(f"unknown zoom preset {preset!r}")
        self._zoom = preset
        for button, name in zip(self.zoom_buttons, ZOOM_PRESETS, strict=True):
            button.set_toggled(name == preset)
        self._render()

    def toggle_lost_pixels(self) -> None:
        self._lost_on = not self._lost_on
        self.lost_button.set_toggled(self._lost_on)
        self._render()

    def toggle_masked(self) -> None:
        self._masked_on = not self._masked_on
        self.mask_button.set_toggled(self._masked_on)
        self._render()

    def set_preview(self, t: int) -> None:
        """Move the preview threshold. Nothing is written: the tiles, the
        curve marker and the "keep" button follow, and a commit is one of
        the panel's two buttons."""
        value = int(min(MAX_T, max(MIN_T, int(t))))
        if value == self._preview:
            return
        self._preview = value
        self.curve.set_state(self.curve.points(), self.curve.clutter_points(),
                             self._plateau(), self._auto(), value)
        self._render()
        self._refresh_panel()

    def pin_time(self, time: float) -> None:
        """Add a sixth tile for `time` (kept for this file for the session)."""
        if self._file is None:
            return
        pinned = self._pinned.setdefault(self._file, [])
        if float(time) not in pinned:
            pinned.append(float(time))
        self.pin_tile_widget.set_text(PIN_TILE_TEXT)
        self.refresh()

    # --- internals --------------------------------------------------------

    def _entry(self):
        if self._file is None or self._file not in self._controller.names():
            return None
        return self._controller.entry(self._file)

    @staticmethod
    def _entry_box(entry) -> tuple[int, int, int, int] | None:
        crop = None if entry is None else entry.crop
        return None if crop is None else (crop.x, crop.y, crop.width, crop.height)

    @classmethod
    def _box_for(cls, entry, evidence) -> tuple[int, int, int, int] | None:
        """The crop box the tiles and the curve describe -- the box the
        strips are grabbed with.

        **`evidence["crop_box"]` wins whenever it differs from the file's
        crop.** Everything measured about a tile comes from two places that
        have to agree: the pixels (re-grabbed now, for whatever box is asked
        for) and `evidence["strips"][].boxes`, which are in the pixel frame
        of the crop the detection ran on. Asking for the file's new box after
        a crop edit would pair fresh pixels with boxes that no longer point
        at the text, and the glyph split, the lost % and the red tint would
        then be measured over a region that may hold no text at all. Nothing
        about the stored evidence describes the new box, so the view keeps
        showing the frame the evidence is in and says so (`_stale_text`),
        until a re-detection replaces the evidence.

        `core/jobs/apply.py` only ever stores a result whose `crop_box` IS
        the file's crop, so the two agree in the ordinary case and this
        chooses nothing. With no evidence box at all, the file's own crop is
        the only candidate."""
        box = evidence.get("crop_box")
        if box is not None:
            return tuple(int(value) for value in box)
        return cls._entry_box(entry)

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

    def _tile_plan(self, evidence) -> list[tuple[str, float]]:
        tiles = evidence.get("tiles") or {}
        plan = [(kind, float(tiles[kind])) for kind in KIND_ORDER if kind in tiles]
        plan += [("pinned", time) for time in self.pinned_times()]
        return plan

    def _sync_tiles(self, evidence) -> None:
        plan = self._tile_plan(evidence)
        if [(tile.kind, tile.time) for tile in self._tiles] == plan:
            return
        for tile in self._tiles:
            self._grid.removeWidget(tile)
            tile.deleteLater()
        self._grid.removeWidget(self.pin_tile_widget)
        self._tiles = []
        for index, (kind, time) in enumerate(plan):
            tile = ZoomTile(kind, time)
            tile.focused.connect(lambda i=index: self._set_focus(i))
            tile.resized.connect(self._on_tile_resized)
            self._grid.addWidget(tile, index // TILE_COLUMNS, index % TILE_COLUMNS)
            self._tiles.append(tile)
        position = len(plan)
        self._grid.addWidget(self.pin_tile_widget, position // TILE_COLUMNS, position % TILE_COLUMNS)
        self._focused = min(self._focused, max(0, len(self._tiles) - 1))

    def _request_strips(self) -> None:
        """Ask for every tile's strip. The controller submits each (box,
        time) at most once a session, so this is safe on every repaint."""
        if self._file is None or self._crop_box is None or not self._tiles:
            return
        times = [tile.time for tile in self._tiles]
        self._controller.request_strips(self._file, self._crop_box, times)

    def _sample_of(self, evidence, time: float) -> dict:
        for sample in evidence.get("strips") or ():
            if float(sample.get("time", -1.0)) == float(time):
                return sample
        return {}

    def _load_pixels(self) -> None:
        entry = self._entry()
        evidence = (entry.evidence.get("brightness") or {}) if entry is not None else {}
        pixels: dict[float, StripPixels] = {}
        for tile in self._tiles:
            sample = self._sample_of(evidence, tile.time)
            strip = (None if self._file is None or self._crop_box is None
                     else self._controller.strip(self._file, self._crop_box, tile.time))
            held = self._pixels.get(tile.time)
            if strip is None:
                tile.set_sample(None)
                continue
            boxes = self._boxes_for(tile, sample, strip)
            # Re-measured when the pixels OR the boxes change: a re-detection
            # can land new boxes on a strip that is still cached, and the
            # glyph split belongs to the pair, not to the pixels alone.
            if held is None or held.strip is not strip or held.given_boxes != boxes:
                held = StripPixels(strip, boxes)
            pixels[tile.time] = held
            tile.set_sample(held, is_text=bool(sample.get("is_text", True)),
                            lines=int(sample.get("lines") or 1))
        self._pixels = pixels

    @staticmethod
    def _boxes_for(tile: ZoomTile, sample: dict, strip) -> tuple:
        """The polygon boxes to split glyphs inside.

        A pinned frame has no detector sample and so no boxes -- and it is
        the one tile the user added because they are worried about it, so it
        is split over the WHOLE strip rather than left unmeasurable. The
        strip is the crop box: on a subtitle frame its text dominates, and
        Otsu over it lands where it would inside a polygon. On a frame with
        no text the split is noise and the glyph mask usually falls below the
        detector's floor, which reads back as "not measurable" -- the honest
        answer for a frame the detector never boxed.
        """
        boxes = sample.get("boxes") or ()
        if boxes or tile.kind != "pinned":
            return tuple(boxes)
        height, width = strip.shape[:2]
        return ((0, 0, width, height),)

    def _strip_width(self) -> int:
        for tile in self._tiles:
            if tile.has_pixels():
                return self._pixels[tile.time].width
        return 0

    def _strip_size(self) -> tuple[int, int]:
        for tile in self._tiles:
            if tile.has_pixels():
                held = self._pixels[tile.time]
                return held.width, held.height
        return 0, 0

    def _on_tile_resized(self) -> None:
        """The tiles share the page's leftover height, so a window resize
        changes how much of each strip they show. Coalesced: one re-render
        per burst of six resize events."""
        if self._resize_pending:
            return
        self._resize_pending = True
        QTimer.singleShot(0, self._render_after_resize)

    def _render_after_resize(self) -> None:
        self._resize_pending = False
        self._render()

    def _set_focus(self, index: int) -> None:
        if 0 <= index < len(self._tiles) and index != self._focused:
            self._focused = index
            self._render_context()

    def _on_panned(self, offset: float) -> None:
        self._offset_ready = True
        self._offset = self._clamp_offset(offset)
        self._render()

    def _clamp_offset(self, offset: float) -> float:
        return min(max(0.0, offset), max(0.0, self._strip_width() - self._visible_width()))

    def _default_offset(self) -> float:
        """Where the zoom window sits before the user pans: centred on the
        text of the first tile that has any. A 1344 px strip at 300% shows
        about 90 px, and its left edge -- where an offset of 0 lands -- is
        empty on every subtitle frame there is."""
        width = self._strip_width()
        if not width:
            return 0.0
        centre = width / 2
        held = next((self._pixels[tile.time] for tile in self._tiles
                     if tile.has_pixels() and self._pixels[tile.time].boxes), None)
        if held is not None:
            left = min(box[0] for box in held.boxes)
            right = max(box[0] + box[2] for box in held.boxes)
            centre = (left + right) / 2
        return self._clamp_offset(centre - self._visible_width() / 2)

    def _visible_width(self) -> int:
        width = self._strip_width()
        zoom = self.zoom_factor()
        content = self._tiles[0].content_rect().width() if self._tiles else 0
        if not width or zoom <= 0 or not content:
            return width
        return min(width, max(1, _round_half_up(content / zoom)))

    def _render(self) -> None:
        zoom = self.zoom_factor()
        # Measured once, from the first tile that has pixels, and then left
        # alone: recomputing it per render would move every tile whenever the
        # focus moved to a tile whose text sits somewhere else.
        if self._offset_ready:
            self._offset = self._clamp_offset(self._offset)
        else:
            self._offset = self._default_offset()
            self._offset_ready = bool(self._strip_width())
        for tile in self._tiles:
            tile.render(zoom, self._offset, self._preview,
                        masked=self._masked_on, lost=self._lost_on)
        self._render_context()

    def _render_context(self) -> None:
        width, height = self._strip_size()
        focused = self._tiles[self._focused] if self._focused < len(self._tiles) else None
        held = None if focused is None or not focused.has_pixels() else self._pixels[focused.time]
        # The context strip shows the whole raw strip, which does not change
        # with the threshold: converting 1344x55 on every drag step would
        # cost more than masking every tile.
        key = None if held is None else id(held)
        if key != self._context_key:
            self._context_key = key
            self._context_image = None if held is None else _bgr_image(held.strip)
        image = self._context_image
        caption = ("" if not width else
                   f"full strip {width} × {height} · drag the amber window to move the zoom")
        self.context.set_state(image, (width, height), self._offset, self._visible_width(), caption)

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

    # --- the note's facts -------------------------------------------------

    def _measure(self, evidence, auto: int | None = None) -> None:
        """The two facts the note needs, which do not move with the preview:
        the lowest threshold at which a text tile starts losing strokes, and
        whether the leaking tile's gate still fires below the plateau.

        **The losing threshold is self-calibrating.** "Glyph pixels" are
        everything above the Otsu split inside the detector's boxes, which
        necessarily includes the anti-aliased skirt around every stroke --
        and a subtitle threshold always eats part of that skirt, so a healthy
        tile can sit at 20-30% lost with every stroke core intact. An
        absolute bar would therefore fire on every file. So each text tile is
        measured against ITSELF: its lost % at the detector's auto value is
        its baseline, and the note fires at the lowest whole threshold from
        the plateau's `lo` up at which some tile has lost LOST_RISE_POINTS
        more of its glyphs than it had already lost at auto. That is the
        point where the threshold starts taking pixels the detector's own
        pick was keeping.

        Without an auto value (no evidence) the baseline is measured at `lo`
        instead, so the rule still reads "how much worse than the bottom of
        the safe range". The per-tile caption is unaffected: it states the
        plain fact (LOST_ALERT_PERCENT of the glyphs gone) rather than a
        judgement about this file.

        Recomputed only when the file, the tiles, the pixels or the reference
        value change -- a threshold drag must not re-measure (and must not
        mask a strip behind the tiles' backs, which is what the
        one-mask-per-tile-per-change budget is)."""
        plateau = evidence.get("plateau")
        key = (self._file, tuple(sorted(self._pixels)),
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
            if held is None or tile.kind == "leaking" or not held.has_glyphs():
                continue
            baseline = held.lost_percent(reference) or 0.0
            found = held.first_losing_threshold(start, baseline + LOST_RISE_POINTS)
            if found is not None and (losing is None or found < losing[0]):
                losing = (found, tile.time)
        leaks = False
        if lo is not None and lo > MIN_T:
            for tile in self._tiles:
                held = self._pixels.get(tile.time)
                if tile.kind == "leaking" and held is not None:
                    leaks = held.gate(lo - 1)
        self._facts = {"lo": lo, "hi": None if plateau is None else int(plateau[1]),
                       "losing": losing, "leaks": leaks}

    def _note_text(self, auto: int | None, stored: int | None) -> str:
        facts = self._facts
        parts: list[str] = []
        if facts.get("lo") is not None:
            parts.append(f"Safe range {facts['lo']}–{facts['hi']}.")
        losing = facts.get("losing")
        if losing is not None:
            # Shown whether or not the preview has passed it. The brief
            # words this as "crosses 10 within [t, 255]", which matches the
            # mockup (the cliff sits above the pick); suppressing the line
            # once the user drags past the cliff would take away the one
            # sentence that explains why a tile just turned red.
            parts.append(f"Above {losing[0]} the {clock(losing[1])} sample starts losing strokes.")
        if facts.get("leaks"):
            parts.append(f"Below {facts['lo']} the background leaks and the frame gate "
                         "fires on empty frames.")
        if auto is not None and stored is not None and auto != stored:
            parts.append(f"detected {auto} · yours {stored}")      # ruling C2
        return " ".join(parts)

    @staticmethod
    def _flag_text(evidence) -> str:
        return brightness_flag_text(evidence.get("flagged"))

    def _stale_text(self, entry) -> str:
        """The warning that what is on screen was measured on another crop.

        Two ways that happens, and both must say so. `brightness_is_stale`
        is the model's own rule: the stored VALUE was measured on a crop the
        file no longer has -- but it only judges DETECTED/HINT values, so a
        MANUAL brightness never trips it. The other is this view's: the
        evidence the tiles and the curve describe (`_box_for`) is not the
        file's crop, whatever the value's source.

        "…re-detecting" while a brightness measurement for the file is still
        to come -- queued, running, or held by auto-pilot behind the folder's
        ranges analysis (`pending_detectors`, not `running_detectors`: a job
        that has not started yet is still on its way) -- and "…re-detect to
        refresh" when nothing is coming."""
        if entry is None:
            return ""
        own = self._entry_box(entry)
        describes_another_crop = own is not None and self._crop_box not in (None, own)
        if not (describes_another_crop or brightness_is_stale(entry)):
            return ""
        pending = self._controller.pending_detectors().get(entry.name, frozenset())
        return STALE_REDETECTING if "brightness" in pending else STALE_REDETECT

    def _crop_note_text(self, entry) -> str:
        """One line saying the tiles are not of the crop the file has now.
        The warn line above it is about the stored value; a value can be
        stale while the evidence still describes the current crop, so the
        two are separate."""
        own = self._entry_box(entry)
        return TILES_OTHER_CROP if own is not None and self._crop_box not in (None, own) else ""

    def _update_panel(self, entry, evidence, auto: int | None, stored: int | None) -> None:
        self.panel.set_state(auto=auto, value=None if entry is None else self._preview,
                             note=self._note_text(auto, stored),
                             flags=self._flag_text(evidence),
                             stale=self._stale_text(entry),
                             crop_note=self._crop_note_text(entry))

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

    def _commit_preview(self) -> None:
        if self._file is not None:
            self._controller.set_brightness(self._file, int(self._preview))

    def _on_pin_clicked(self) -> None:
        time = self.timeline_position() if self.timeline_position is not None else None
        if time is None:
            self.pin_tile_widget.set_text(PIN_HINT_TEXT)     # plan 3C Task 4 mounts the timeline
            return
        self.pin_time(float(time))
