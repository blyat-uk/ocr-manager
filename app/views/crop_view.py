"""The Crop review view (plan 3C Task 2, ui-spec §3.4, rulings B2/B3/B5).

`CropTab` is a `StageTab`: a frame canvas with a free six-handle box, the
detector's text envelope, a sample filmstrip and its own inspector panel.

Pixels
    Every frame here -- the canvas and the filmstrip thumbnails -- comes
    from `controller.request_frames` / `controller.frame`, i.e.
    `core.detect.crop.grab_frames`, at the times the crop detector itself
    recorded. That is the only source whose times re-fetch the frames the
    evidence was measured on (the frame-addressing table in
    `core/detect/__init__.py`). The "masked" toggle previews the OCR pass's
    brightness filter over the crop region of the frame already in hand; it
    is a preview of the filter, not of the OCR pass's own pixels, so nothing
    here tunes a brightness threshold -- that is the Brightness tab's job,
    and it uses `ocr_view` strips.

Evidence is a disposable cache: every `evidence["crop"]` key is read with
`.get` and missing keys fall back to the no-evidence presentation.

    - Samples are de-duplicated by time, preferring the kept entry: the
      full-frame retry re-probes times the bottom-band round already probed,
      so one time can carry both an empty and a kept entry. Every count and
      the filmstrip use the de-duplicated list.
    - A sample is "disagreeing" only when the result HAS a box. Without one
      (a confirmed watermark, the height ceiling, ...) nothing disagrees --
      there is nothing to disagree with -- and the panel shows the flag
      reason instead.
    - "⤢ fit to all N samples" re-runs the detector's own aggregation over
      the kept samples' boxes with the cutoff the detection used
      (`cutoff_frac`, 0.0 after a full-frame retry), falling back to the
      folder setting when the evidence predates that field.

Edits are MANUAL values committed through `controller.set_crop`: on mouse
release for a drag, and 400 ms after the last key nudge or spin-box change,
so one gesture is one command. Label masks are folder-level and go through
`controller.set_label_masks`; the whole affordance is hidden when the folder
has labels off.

Ruling C2: a detection never overwrites a MANUAL/IMPORTED value, but its
evidence still replaces `evidence["crop"]`. When the two differ, the amber
box stays the stored value and the detection is drawn separately, dashed and
labelled, with a panel row naming both.

Views import no `core` module (tests/ui/test_main_window.py), so the two
pure detector helpers this view needs -- `crop.aggregate_box` and
`ocr_view.mask` -- are reached through `app.masking`.
"""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6 import sip
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QRadialGradient
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.imaging import bgr_to_qimage
from app.masking import aggregate_crop_box, mask_region
from app.state_text import clock, crop_caption
from app.theme import tokens
from app.views.inspector_sections import Section, note_label
from app.views.thumbnail import GRADIENT_DEGREES, GRADIENT_END_STOP, css_gradient
from app.widgets.base import Button, KvRow, SectionHeader

COMMIT_DEBOUNCE_MS = 400          # one command per gesture: nudges and spin-box edits
DEFAULT_BRIGHTNESS = 230          # core.config.Config's default, for a file with no value yet
NUDGE_SMALL, NUDGE_LARGE = 1, 10  # video pixels, plain and with Shift
TIMELINE_HEIGHT = 68              # the compact timeline lands here (ruling B5, plan 3C Task 4)
NUDGE_NOTE = "Arrow keys nudge 1 px, ⇧ arrows nudge 10. Free rectangle — no forced centring."
COVERED_NOTE = "The amber box already covers it. Click the warned sample to inspect."
OUTSIDE_NOTE = "This sample falls outside the box."
ARROW_HINT = "◀ ▶ arrow keys"
DETECTED_TAG = "dashed grey = the latest detection"
# The detector settings "fit to all samples" re-aggregates with; the cutoff is
# added from the evidence, never from the folder alone (see the module docstring).
DETECTOR_FIELDS = ("crop_width_fraction", "crop_vertical_padding", "crop_min_height_fraction")


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _timecode(seconds: float) -> str:
    """"MM:SS.ss" -- the canvas tag's time, to the hundredth of a second."""
    total = max(0.0, float(seconds))
    minutes = int(total // 60)
    return f"{minutes:02d}:{total - minutes * 60:05.2f}"


def _box_text(box) -> str:
    return f"{box[0]}, {box[1]} · {box[2]} × {box[3]}"


def _read_box(values) -> tuple[int, int, int, int] | None:
    """A JSON (x, y, w, h) list from the evidence, or None for anything else."""
    try:
        if values is None or len(values) != 4:
            return None
        return tuple(int(value) for value in values)
    except (TypeError, ValueError):
        return None


def _union(boxes) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return (left, top, right - left, bottom - top)


def _covers(outer, inner) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[0] + outer[2] >= inner[0] + inner[2]
            and outer[1] + outer[3] >= inner[1] + inner[3])


def _with_alpha(colour: str, alpha: float) -> QColor:
    result = QColor(colour)
    result.setAlphaF(alpha)
    return result


@dataclass(frozen=True)
class CropSampleView:
    """One de-duplicated probe frame of `evidence["crop"]["samples"]`."""

    time: float
    boxes: tuple[tuple[int, int, int, int], ...]
    kept: bool
    lines: int

    @property
    def extent(self) -> tuple[int, int, int, int] | None:
        return _union(self.boxes)


def _preference(sample: CropSampleView) -> tuple[bool, bool]:
    return (sample.kept, bool(sample.boxes))


def read_samples(evidence: dict | None) -> list[CropSampleView]:
    """The evidence's samples, one per time, chronologically.

    The full-frame retry re-probes times the bottom-band rounds already
    probed, so a time can carry two entries -- one empty, one kept. The kept
    entry wins, then the one that found text."""
    best: dict[float, CropSampleView] = {}
    for raw in (evidence or {}).get("samples") or []:
        try:
            time = float(raw.get("time"))
        except (TypeError, ValueError):
            continue
        boxes = tuple(box for box in (_read_box(value) for value in raw.get("boxes") or [])
                      if box is not None)
        try:
            lines = int(raw.get("lines") or 0)
        except (TypeError, ValueError):
            lines = 0
        sample = CropSampleView(time, boxes, bool(raw.get("kept")), lines)
        current = best.get(time)
        if current is None or _preference(sample) > _preference(current):
            best[time] = sample
    return [best[time] for time in sorted(best)]


# --------------------------------------------------------------------------
# The canvas
# --------------------------------------------------------------------------

# canvas-tag corner -> (text colour, border colour): `.tag` plain, amber-tinted
# for the crop, blue-tinted for the envelope legend (ui-spec §2.5).
_TAG_TONES = {"top_left": (tokens.DIM, tokens.LINE2),
              "top_right": (tokens.ACC, tokens.ACC_DIM),
              "bottom_right": (tokens.BLUE, tokens.TAG_BLUE_BORDER),
              "bottom_left": (tokens.DIM, tokens.LINE2)}
# `.sthumb.on` / a disagreeing sample's border.
_THUMB_TONES = {"selected": tokens.ACC, "warn": tokens.WARN}

_HANDLE_EDGES = {
    "tl": ("left", "top"), "tr": ("right", "top"),
    "bl": ("left", "bottom"), "br": ("right", "bottom"),
    "tc": ("top",), "bc": ("bottom",),
}


class CropCanvas(QWidget):
    """The frame with the crop box drawn over it (ui-spec §3.4).

    Draw order, back to front: the frame (or the gradient placeholder), the
    masked crop preview, the label masks, the spotlight dimming outside the
    box, the dashed envelope, the amber box and its six handles, the 10%
    grid, the canvas tags.

    The widget owns no model: it reports edits (`commit_requested` on mouse
    release, debounced by `CropTab` for key nudges) and sample steps, and the
    tab turns them into controller commands."""

    HANDLES = ("tl", "tr", "bl", "br", "tc", "bc")
    HANDLE_SIZE = 7                # `.cropbox b`: a 7x7 amber square per handle
    HANDLE_GRAB = 13               # the square is small; this is what the mouse actually hits
    MIN_BOX = 8                    # video pixels
    MIN_MASK = 6                   # a right-drag smaller than this counts as a click
    GRID_DIVISIONS = 10            # the "grid" overlay: every 10%
    GRID_ALPHA = 0.45
    TAG_MARGIN = 8
    TAG_PAD_X, TAG_PAD_Y = 6, 2
    PLACEHOLDER_TEXT = "loading frame…"

    box_changed = pyqtSignal()
    commit_requested = pyqtSignal(tuple)       # the box to store
    nudged = pyqtSignal()                      # a key nudge: start the commit debounce
    masks_changed = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("CropCanvas")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Stretches across the stage, but only as tall as the frame it shows:
        # the figure's canvas is the frame, with the filmstrip right under it,
        # not a frame floating in a letterbox.
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setMinimumSize(240, 120)
        self._video = (1920, 1080)
        self._frame = None
        self._image: QImage | None = None
        self._box = (0, 0, 0, 0)
        self._envelope: tuple[int, int, int, int] | None = None
        self._detected: tuple[int, int, int, int] | None = None
        self._masks: list[tuple[int, int, int, int]] = []
        self._masks_enabled = False
        self._brightness = DEFAULT_BRIGHTNESS
        self._overlays = {"envelope": True, "masked": False, "grid": False}
        self._time = 0.0
        self._kept = 0
        self._drag: tuple | None = None
        self._pending = False                   # edited, not committed yet: a refresh must not undo it
        self._mask_drag: list | None = None
        self._mask_cache: tuple | None = None

    # --- state ------------------------------------------------------------------

    def video_size(self) -> tuple[int, int]:
        return self._video

    def set_video_size(self, size) -> None:
        width, height = (int(value) for value in size)
        if width > 0 and height > 0 and (width, height) != self._video:
            self._video = (width, height)
            self._mask_cache = None
            self.update()

    def set_frame(self, frame) -> None:
        """`frame`: a BGR numpy array from `controller.frame`, or None."""
        if frame is self._frame:
            return
        self._frame = frame
        self._image = bgr_to_qimage(frame)
        self._mask_cache = None
        self.update()

    def placeholder_text(self) -> str:
        return "" if self._image is not None and not self._image.isNull() else self.PLACEHOLDER_TEXT

    def box(self) -> tuple[int, int, int, int]:
        return self._box

    def set_box(self, box) -> None:
        """Show the stored box. Ignored while the user is mid-gesture or an
        edit is still waiting for its debounced commit -- a frame arriving
        must never undo what was just typed or nudged."""
        if self._drag is not None or self._pending:
            return
        self._store_box(tuple(int(value) for value in box))

    def pending(self) -> bool:
        return self._pending

    def clear_pending(self) -> None:
        """The tab has committed (or dropped) this edit."""
        self._pending = False

    def set_evidence(self, envelope, detected) -> None:
        if (envelope, detected) != (self._envelope, self._detected):
            self._envelope, self._detected = envelope, detected
            self.update()

    def detected_box(self) -> tuple[int, int, int, int] | None:
        return self._detected

    def set_meta(self, time: float, kept_samples: int) -> None:
        if (float(time), int(kept_samples)) != (self._time, self._kept):
            self._time, self._kept = float(time), int(kept_samples)
            self.update()

    def set_brightness(self, value: int) -> None:
        if int(value) != self._brightness:
            self._brightness = int(value)
            self._mask_cache = None
            self.update()

    def overlays(self) -> dict[str, bool]:
        return dict(self._overlays)

    def set_overlay(self, name: str, on: bool) -> None:
        if self._overlays.get(name) != bool(on):
            self._overlays[name] = bool(on)
            self.update()

    def masks(self) -> list[tuple[int, int, int, int]]:
        return list(self._masks)

    def masks_enabled(self) -> bool:
        return self._masks_enabled

    def set_masks(self, masks, enabled: bool) -> None:
        masks = [tuple(int(value) for value in mask) for mask in masks]
        if (masks, bool(enabled)) != (self._masks, self._masks_enabled):
            self._masks, self._masks_enabled = masks, bool(enabled)
            if not self._masks_enabled:
                self._mask_drag = None
            self.update()

    # --- geometry ---------------------------------------------------------------

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        video_width, video_height = self._video
        return max(self.minimumHeight(), round(width * video_height / video_width))

    def frame_rect(self) -> QRectF:
        """Where the video frame is drawn: its aspect ratio fitted, centred.

        The coordinate space is the NATIVE frame size (`entry.media`), not
        the decoded image's -- frames arrive scaled to at most 720 rows, and
        the crop box is in native pixels."""
        bounds = QRectF(self.rect())
        width, height = self._video
        if width <= 0 or height <= 0:
            return bounds
        scale = min(bounds.width() / width, bounds.height() / height)
        drawn_width, drawn_height = width * scale, height * scale
        return QRectF(bounds.x() + (bounds.width() - drawn_width) / 2,
                      bounds.y() + (bounds.height() - drawn_height) / 2,
                      drawn_width, drawn_height)

    def to_widget(self, x: float, y: float) -> QPointF:
        rect = self.frame_rect()
        width, height = self._video
        return QPointF(rect.x() + float(x) * rect.width() / width,
                       rect.y() + float(y) * rect.height() / height)

    def to_video(self, point: QPointF) -> tuple[float, float]:
        rect = self.frame_rect()
        width, height = self._video
        if rect.width() <= 0 or rect.height() <= 0:
            return (0.0, 0.0)
        return ((point.x() - rect.x()) * width / rect.width(),
                (point.y() - rect.y()) * height / rect.height())

    def box_rect(self) -> QRectF:
        return self.video_rect(self._box)

    def video_rect(self, box) -> QRectF:
        x, y, width, height = box
        return QRectF(self.to_widget(x, y), self.to_widget(x + width, y + height))

    def handle_rect(self, handle: str) -> QRectF:
        rect = self.box_rect()
        edges = _HANDLE_EDGES[handle]
        x = rect.center().x() if handle in ("tc", "bc") else (rect.left() if "left" in edges else rect.right())
        y = rect.top() if "top" in edges else rect.bottom()
        size = self.HANDLE_SIZE
        return QRectF(x - size / 2, y - size / 2, size, size)

    def _handle_at(self, point: QPointF) -> str | None:
        grow = (self.HANDLE_GRAB - self.HANDLE_SIZE) / 2
        for handle in self.HANDLES:
            if self.handle_rect(handle).adjusted(-grow, -grow, grow, grow).contains(point):
                return handle
        return None

    # --- editing -----------------------------------------------------------------

    def _store_box(self, box) -> None:
        box = self._clamped(box)
        if box != self._box:
            self._box = box
            self._mask_cache = None
            self.update()
            self.box_changed.emit()

    def _clamped(self, box) -> tuple[int, int, int, int]:
        video_width, video_height = self._video
        width = _clamp(int(box[2]), self.MIN_BOX, video_width)
        height = _clamp(int(box[3]), self.MIN_BOX, video_height)
        x = _clamp(int(box[0]), 0, video_width - width)
        y = _clamp(int(box[1]), 0, video_height - height)
        return (x, y, width, height)

    def _resize(self, handle: str | None, start_box, dx: float, dy: float) -> None:
        self._pending = True
        x, y, width, height = start_box
        left, top, right, bottom = x, y, x + width, y + height
        video_width, video_height = self._video
        if handle is None:
            self._store_box((round(x + dx), round(y + dy), width, height))
            return
        edges = _HANDLE_EDGES[handle]
        if "left" in edges:
            left = _clamp(round(x + dx), 0, right - self.MIN_BOX)
        if "right" in edges:
            right = _clamp(round(right + dx), left + self.MIN_BOX, video_width)
        if "top" in edges:
            top = _clamp(round(y + dy), 0, bottom - self.MIN_BOX)
        if "bottom" in edges:
            bottom = _clamp(round(bottom + dy), top + self.MIN_BOX, video_height)
        self._store_box((left, top, right - left, bottom - top))

    def mousePressEvent(self, event) -> None:
        point = event.position()
        if event.button() == Qt.MouseButton.RightButton:
            if not self._masks_enabled:
                super().mousePressEvent(event)
                return
            start = self.to_video(point)
            self._mask_drag = [start, start]
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        handle = self._handle_at(point)
        if handle is None and not self.box_rect().contains(point):
            super().mousePressEvent(event)
            return
        self._drag = (handle, self.to_video(point), self._box)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        point = event.position()
        if self._mask_drag is not None:
            self._mask_drag[1] = self.to_video(point)
            self.update()
            event.accept()
            return
        if self._drag is None:
            super().mouseMoveEvent(event)
            return
        handle, (start_x, start_y), start_box = self._drag
        moved = self.to_video(point)
        self._resize(handle, start_box, moved[0] - start_x, moved[1] - start_y)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton and self._mask_drag is not None:
            self._finish_mask()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drag is not None:
            self._drag = None
            self.commit_requested.emit(self._box)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        """All four arrows nudge the box by 1 px, Shift+arrow by 10.

        Stepping the samples is ◀ / ▶ on the filmstrip -- the two are split
        by which widget has focus, so nudging keeps both axes (Tab walks
        canvas -> filmstrip, and clicking a sample moves focus there)."""
        key = event.key()
        step = NUDGE_LARGE if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else NUDGE_SMALL
        deltas = {Qt.Key.Key_Left: (-step, 0), Qt.Key.Key_Right: (step, 0),
                  Qt.Key.Key_Up: (0, -step), Qt.Key.Key_Down: (0, step)}
        if key in deltas:
            self._nudge(*deltas[key])
            event.accept()
            return
        super().keyPressEvent(event)

    def _nudge(self, dx: int, dy: int) -> None:
        self._pending = True
        x, y, width, height = self._box
        self._store_box((x + dx, y + dy, width, height))
        self.nudged.emit()

    # --- label masks --------------------------------------------------------------

    def _mask_drag_box(self) -> tuple[int, int, int, int] | None:
        if self._mask_drag is None:
            return None
        (x0, y0), (x1, y1) = self._mask_drag
        video_width, video_height = self._video
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        left, right = _clamp(round(left), 0, video_width), _clamp(round(right), 0, video_width)
        top, bottom = _clamp(round(top), 0, video_height), _clamp(round(bottom), 0, video_height)
        return (left, top, right - left, bottom - top)

    def _finish_mask(self) -> None:
        drawn = self._mask_drag_box()
        point = self._mask_drag[1]
        self._mask_drag = None
        if drawn is None:
            return
        if drawn[2] >= self.MIN_MASK and drawn[3] >= self.MIN_MASK:
            masks = [*self._masks, drawn]
        else:                                        # a right-click: remove the mask under it
            x, y = point
            masks = [mask for mask in self._masks
                     if not (mask[0] <= x <= mask[0] + mask[2] and mask[1] <= y <= mask[1] + mask[3])]
            if masks == self._masks:
                self.update()
                return
        self._masks = masks
        self.update()
        self.masks_changed.emit(list(masks))

    # --- painting -------------------------------------------------------------------

    def tags(self) -> dict[str, str]:
        """The canvas-tag chips, by corner -- what the figure prints over the
        frame, and what the tests read instead of pixels."""
        width, height = self._video
        x, y, box_width, box_height = self._box
        tags = {"top_left": f"{width} × {height} · t {_timecode(self._time)}",
                "top_right": f"crop {x}, {y} · {box_width} × {box_height}"}
        if self._overlays["envelope"]:
            tags["bottom_right"] = f"dashed = text found across all {self._kept} samples"
            if self._detected is not None:
                tags["bottom_left"] = DETECTED_TAG
        return tags

    def paintEvent(self, event) -> None:
        # Everything is drawn inside the fitted frame, never the widget: the
        # canvas IS the frame (`.canvas`, radius 6), and the space the fit
        # leaves over is the stage's own background, not a letterbox.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        bounds = self.frame_rect()
        clip = QPainterPath()
        clip.addRoundedRect(bounds, tokens.RADIUS_BTN, tokens.RADIUS_BTN)
        painter.setClipPath(clip)
        self._paint_frame(painter, bounds)
        self._paint_masked(painter)
        self._paint_label_masks(painter)
        self._paint_spotlight(painter, bounds)
        self._paint_envelope(painter)
        self._paint_box(painter)
        self._paint_grid(painter)
        self._paint_tags(painter, bounds)
        painter.end()

    def _paint_frame(self, painter: QPainter, bounds: QRectF) -> None:
        width, height = bounds.width(), bounds.height()
        radius_x, radius_y = width * 1.2, height * 0.9          # radial-gradient(120% 90% at 30% 25%)
        if radius_x > 0 and radius_y > 0:
            gradient = QRadialGradient(QPointF(0, 0), radius_x)
            gradient.setColorAt(0.0, QColor(tokens.CANVAS_TOP))
            gradient.setColorAt(tokens.CANVAS_MID_STOP, QColor(tokens.CANVAS_MID))
            gradient.setColorAt(1.0, QColor(tokens.CANVAS_BOTTOM))
            painter.save()
            painter.translate(bounds.x() + width * 0.3, bounds.y() + height * 0.25)
            painter.scale(1.0, radius_y / radius_x)
            span = max(width, height) * 4 * radius_x / radius_y
            painter.fillRect(QRectF(-span, -span, 2 * span, 2 * span), gradient)
            painter.restore()
        if self._image is not None and not self._image.isNull():
            painter.drawImage(self.frame_rect(), self._image)
            return
        painter.setPen(QColor(tokens.DIM2))
        font = painter.font()
        font.setPixelSize(round(tokens.FONT_SIZE_SM))
        painter.setFont(font)
        painter.drawText(bounds, int(Qt.AlignmentFlag.AlignCenter), self.PLACEHOLDER_TEXT)

    def _masked_image(self) -> QImage | None:
        """The crop region of the frame in hand, through the OCR pass's
        brightness filter (`ocr_view.mask`). The frame is decoded smaller
        than the native size, so the box is scaled into it first."""
        frame = self._frame
        if frame is None:
            return None
        key = (self._box, self._brightness, id(frame))
        if self._mask_cache is not None and self._mask_cache[0] == key:
            return self._mask_cache[1]
        height, width = frame.shape[:2]
        video_width, video_height = self._video
        scale_x, scale_y = width / video_width, height / video_height
        x, y, box_width, box_height = self._box
        left, top = max(0, round(x * scale_x)), max(0, round(y * scale_y))
        right = min(width, round((x + box_width) * scale_x))
        bottom = min(height, round((y + box_height) * scale_y))
        image = None
        if right - left >= 1 and bottom - top >= 1:
            image = bgr_to_qimage(mask_region(frame[top:bottom, left:right], self._brightness))
        self._mask_cache = (key, image)
        return image

    def _paint_masked(self, painter: QPainter) -> None:
        if not self._overlays["masked"]:
            return
        image = self._masked_image()
        if image is not None and not image.isNull():
            painter.drawImage(self.box_rect(), image)

    def _paint_label_masks(self, painter: QPainter) -> None:
        if not self._masks_enabled:
            return
        for mask in self._masks:                     # what the OCR pass blacks out
            painter.fillRect(self.video_rect(mask), QColor(Qt.GlobalColor.black))
        drawing = self._mask_drag_box()
        if drawing is not None:
            painter.fillRect(self.video_rect(drawing), _with_alpha(tokens.BG, 0.75))
            painter.setPen(QPen(QColor(tokens.DIM), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.video_rect(drawing))

    def _paint_spotlight(self, painter: QPainter, bounds: QRectF) -> None:
        outside = QPainterPath()
        outside.addRect(bounds)
        inside = QPainterPath()
        inside.addRoundedRect(self.box_rect(), tokens.RADIUS_XS, tokens.RADIUS_XS)
        painter.fillPath(outside.subtracted(inside), QColor(*tokens.SPOTLIGHT))

    def _dashed(self, colour: QColor) -> QPen:
        pen = QPen(colour, 1)
        pen.setStyle(Qt.PenStyle.DashLine)
        return pen

    def _paint_envelope(self, painter: QPainter) -> None:
        if not self._overlays["envelope"]:
            return
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._envelope is not None:
            painter.setPen(self._dashed(_with_alpha(tokens.BLUE, tokens.ENVELOPE_ALPHA)))
            painter.drawRoundedRect(self.video_rect(self._envelope), tokens.RADIUS_XS, tokens.RADIUS_XS)
        if self._detected is not None:
            painter.setPen(self._dashed(QColor(tokens.DIM)))
            painter.drawRoundedRect(self.video_rect(self._detected), tokens.RADIUS_XS, tokens.RADIUS_XS)

    def _paint_box(self, painter: QPainter) -> None:
        rect = self.box_rect()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(tokens.ACC), 1.5))
        painter.drawRoundedRect(rect, tokens.RADIUS_XS, tokens.RADIUS_XS)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(tokens.ACC))
        for handle in self.HANDLES:
            painter.drawRoundedRect(self.handle_rect(handle), tokens.RADIUS_THUMB_BOX,
                                    tokens.RADIUS_THUMB_BOX)

    def _paint_grid(self, painter: QPainter) -> None:
        if not self._overlays["grid"]:
            return
        rect = self.frame_rect()
        painter.setPen(QPen(_with_alpha(tokens.LINE2, self.GRID_ALPHA), 1))
        for step in range(1, self.GRID_DIVISIONS):
            fraction = step / self.GRID_DIVISIONS
            x = rect.x() + rect.width() * fraction
            y = rect.y() + rect.height() * fraction
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

    def _paint_tags(self, painter: QPainter, bounds: QRectF) -> None:
        font = painter.font()
        font.setPixelSize(round(tokens.FONT_SIZE_SCOPE))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        for corner, text in self.tags().items():
            width = metrics.horizontalAdvance(text) + 2 * self.TAG_PAD_X
            height = metrics.ascent() + metrics.descent() + 2 * self.TAG_PAD_Y
            x = (bounds.x() + self.TAG_MARGIN if corner.endswith("left")
                 else bounds.right() - self.TAG_MARGIN - width)
            y = (bounds.y() + self.TAG_MARGIN if corner.startswith("top")
                 else bounds.bottom() - self.TAG_MARGIN - height)
            rect = QRectF(x, y, width, height)
            colour, border = _TAG_TONES[corner]
            painter.setPen(QPen(QColor(border), 1))
            painter.setBrush(QColor(*tokens.TAG_BG))
            painter.drawRoundedRect(rect, tokens.RADIUS_TAG, tokens.RADIUS_TAG)
            painter.setPen(QColor(colour))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)


# --------------------------------------------------------------------------
# The filmstrip
# --------------------------------------------------------------------------

class SampleThumbnail(QWidget):
    """`.sthumb`: one sampled frame, 64x36, with one or two white bars for
    the text rows found in it and a border for its state."""

    BAR_INSET = 0.12               # `.sthumb i`: left/right 12%
    BAR_BOTTOM = 0.20
    BAR_BOTTOM_TWO = 0.34
    BAR_HEIGHT = 3

    clicked = pyqtSignal(int)

    def __init__(self, width: int, height: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("SampleThumbnail")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(width + 2, height + 2)       # 1 px border on each side, as the CSS box does
        self._index = -1
        self._image: QImage | None = None
        self._lines = 0
        self._tone = ""

    def set_state(self, index: int, image: QImage | None, lines: int, tone: str) -> None:
        state = (index, image, lines, tone)
        if state != (self._index, self._image, self._lines, self._tone):
            self._index, self._image, self._lines, self._tone = state
            self.update()

    def sample_index(self) -> int:
        return self._index

    def lines(self) -> int:
        return self._lines

    def tone(self) -> str:
        return self._tone

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit(self._index)
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        bounds = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        clip = QPainterPath()
        clip.addRoundedRect(bounds, tokens.RADIUS_THUMB, tokens.RADIUS_THUMB)
        painter.save()
        painter.setClipPath(clip)
        gradient = css_gradient(bounds, GRADIENT_DEGREES)
        gradient.setColorAt(0.0, QColor(tokens.THUMB_TOP))
        gradient.setColorAt(GRADIENT_END_STOP, QColor(tokens.THUMB_BOTTOM))
        gradient.setColorAt(1.0, QColor(tokens.THUMB_BOTTOM))
        painter.fillRect(bounds, gradient)
        if self._image is not None and not self._image.isNull():
            painter.drawImage(bounds, self._image)
        else:
            self._paint_bars(painter, bounds)
        painter.restore()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        colour = _THUMB_TONES.get(self._tone)
        painter.setPen(QPen(QColor(colour), 1) if colour else QPen(Qt.PenStyle.NoPen))
        if colour:
            painter.drawRoundedRect(bounds, tokens.RADIUS_THUMB, tokens.RADIUS_THUMB)
        painter.end()

    def _paint_bars(self, painter: QPainter, bounds: QRectF) -> None:
        """The stand-in for text in a sample whose frame has not arrived."""
        if self._lines <= 0:
            return
        left = bounds.x() + bounds.width() * self.BAR_INSET
        width = bounds.width() * (1 - 2 * self.BAR_INSET)
        bar = QColor(Qt.GlobalColor.white)
        bar.setAlphaF(tokens.SAMPLE_BAR_ALPHA)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bar)
        for index in range(min(2, self._lines)):
            fraction = self.BAR_BOTTOM if index == 0 else self.BAR_BOTTOM_TWO
            y = bounds.bottom() - bounds.height() * fraction - self.BAR_HEIGHT
            painter.drawRoundedRect(QRectF(left, y, width, self.BAR_HEIGHT),
                                    tokens.RADIUS_XS, tokens.RADIUS_XS)


class SampleStrip(QWidget):
    """`.strip`: "samples", up to eight thumbnails, "more ▸" to page."""

    PAGE = 8
    THUMB_WIDTH = 64
    THUMB_HEIGHT = 36
    LABEL_WIDTH = 46

    selected = pyqtSignal(int)
    stepped = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("SampleStrip")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._samples: list[CropSampleView] = []
        self._page = 0
        self._selected = -1
        self._warned: set[int] = set()
        self._images: dict[int, QImage | None] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 9, 0, 0)
        layout.setSpacing(6)
        self._label = note_label("samples")
        self._label.setFixedWidth(self.LABEL_WIDTH)
        layout.addWidget(self._label)
        self._thumbs: list[SampleThumbnail] = []
        for _ in range(self.PAGE):
            thumb = SampleThumbnail(self.THUMB_WIDTH, self.THUMB_HEIGHT)
            thumb.clicked.connect(self._on_clicked)
            layout.addWidget(thumb)
            self._thumbs.append(thumb)
        self.more_button = Button("more ▸", "ghost", small=True)
        self.more_button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.more_button.clicked.connect(self.next_page)
        layout.addWidget(self.more_button)
        layout.addStretch(1)
        layout.addWidget(note_label(ARROW_HINT))

    def _on_clicked(self, index: int) -> None:
        """Picking a sample with the mouse hands the strip the keyboard too,
        so ◀ / ▶ carry on from there."""
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.selected.emit(index)

    def label_text(self) -> str:
        return self._label.text()

    def thumbnails(self) -> list[SampleThumbnail]:
        """The thumbnails currently showing a sample (up to eight)."""
        return [thumb for thumb in self._thumbs if not thumb.isHidden()]

    def page(self) -> int:
        return self._page

    def reset_page(self) -> None:
        self._page = 0

    def pages(self) -> int:
        return max(1, -(-len(self._samples) // self.PAGE))

    def next_page(self) -> None:
        self._page = (self._page + 1) % self.pages()
        self._relayout()

    def set_state(self, samples, selected: int, warned: set[int], images) -> None:
        """`images`: sample index -> a thumbnail-sized QImage, or None."""
        self._samples = list(samples)
        self._selected = selected
        self._warned = set(warned)
        self._images = dict(images)
        if 0 <= selected < len(self._samples):
            self._page = min(selected // self.PAGE, self.pages() - 1)
        self._page = min(self._page, self.pages() - 1)
        self._relayout()

    def _relayout(self) -> None:
        start = self._page * self.PAGE
        shown = self._samples[start:start + self.PAGE]
        for offset, thumb in enumerate(self._thumbs):
            if offset >= len(shown):
                thumb.hide()
                continue
            index = start + offset
            tone = "selected" if index == self._selected else ("warn" if index in self._warned else "")
            thumb.set_state(index, self._images.get(index), shown[offset].lines, tone)
            thumb.show()
        self.more_button.setVisible(len(self._samples) > self.PAGE)

    def keyPressEvent(self, event) -> None:
        """◀ / ▶ walk the samples while the filmstrip has focus (the canvas
        nudges the box with the same keys); Shift changes nothing here."""
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            self.stepped.emit(-1 if event.key() == Qt.Key.Key_Left else 1)
            event.accept()
            return
        super().keyPressEvent(event)


# --------------------------------------------------------------------------
# The inspector panel
# --------------------------------------------------------------------------

class _ClickableKvRow(KvRow):
    clicked = pyqtSignal()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class _SpinPair(QWidget):
    """A kv row whose value is two spin boxes -- "X / width", "Y / height"."""

    edited = pyqtSignal()

    def __init__(self, key: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("KvRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)
        label = QLabel(key)
        label.setProperty("kvRole", "key")
        layout.addWidget(label)
        layout.addStretch(1)
        self.first, self.second = QSpinBox(), QSpinBox()
        for spin in (self.first, self.second):
            spin.setRange(0, 1)
            spin.setFixedWidth(64)
            spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            spin.setStyleSheet(
                f"QSpinBox {{ background: {tokens.BG}; color: {tokens.TXT}; "
                f"border: 1px solid {tokens.LINE2}; border-radius: {tokens.RADIUS_XS}px; "
                f"padding: 1px 4px; font-size: {tokens.FONT_SIZE_BODY}px; }}"
                f"QSpinBox:focus {{ border-color: {tokens.ACC}; }}")
            spin.valueChanged.connect(self._on_changed)
            layout.addWidget(spin)
        self._syncing = False

    def values(self) -> tuple[int, int]:
        return (self.first.value(), self.second.value())

    def set_limits(self, first_max: int, second_max: int, second_min: int = 0) -> None:
        self._syncing = True
        self.first.setRange(0, max(0, first_max))
        self.second.setRange(second_min, max(second_min, second_max))
        self._syncing = False

    def set_values(self, first: int, second: int) -> None:
        self._syncing = True
        self.first.setValue(int(first))
        self.second.setValue(int(second))
        self._syncing = False

    def _on_changed(self, _value: int) -> None:
        if not self._syncing:
            self.edited.emit()


class CropInspectorPanel(Section):
    """The Crop tab's slice of the inspector (ruling B4): the box as two spin
    pairs, the nudge note, and the evidence rows.

    B4 drops the "◆ this episode only" header the tab-scoped mockup shows --
    the inspector's own header already says it."""

    box_edited = pyqtSignal(tuple)
    sample_requested = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.body.setSpacing(5)
        self.x_row = _SpinPair("X / width")
        self.y_row = _SpinPair("Y / height")
        for row in (self.x_row, self.y_row):
            row.edited.connect(self._on_edited)
            self.body.addWidget(row)
        self.nudge_label = note_label(NUDGE_NOTE)
        self.body.addWidget(self.nudge_label)
        self.body.addSpacing(4)
        self.evidence_header = SectionHeader("Evidence")
        self.body.addWidget(self.evidence_header)
        self._rows: list[KvRow] = []
        self._row_keys: list[str] = []
        self.note = note_label("")
        self.body.addWidget(self.note)
        self.body.addStretch(1)
        self._commit = QTimer(self)
        self._commit.setSingleShot(True)
        self._commit.setInterval(COMMIT_DEBOUNCE_MS)
        self._commit.timeout.connect(self._emit_box)

    # --- reading (the tests' view of the panel) ---------------------------------

    def spin_values(self) -> tuple[int, int, int, int]:
        """(x, y, width, height) -- the stored box's order, not the rows'."""
        x, width = self.x_row.values()
        y, height = self.y_row.values()
        return (x, y, width, height)

    def set_spin_values(self, x: int, width: int, y: int, height: int) -> None:
        """Type into the spin boxes as the user would: this starts the
        debounced commit. `show_box` is the silent counterpart."""
        self.x_row.first.setValue(int(x))
        self.x_row.second.setValue(int(width))
        self.y_row.first.setValue(int(y))
        self.y_row.second.setValue(int(height))

    def show_box(self, box) -> None:
        """Follow the canvas without committing anything of our own."""
        self.x_row.set_values(box[0], box[2])
        self.y_row.set_values(box[1], box[3])

    def rows(self) -> list[tuple[str, str]]:
        return list(zip(self._row_keys, [row.value() for row in self._rows], strict=True))

    def click_row(self, key: str) -> None:
        for name, row in zip(self._row_keys, self._rows, strict=True):
            if name == key and isinstance(row, _ClickableKvRow):
                row.clicked.emit()
                return

    def nudge_note(self) -> str:
        return self.nudge_label.text()

    def evidence_note(self) -> str:
        return self.note.text()

    # --- writing ------------------------------------------------------------------

    def set_state(self, box, video_size, rows, note: str) -> None:
        """`rows`: (key, value, tone, sample index or None) per evidence row."""
        width, height = video_size
        self.x_row.set_limits(width, width, CropCanvas.MIN_BOX)
        self.y_row.set_limits(height, height, CropCanvas.MIN_BOX)
        enabled = box is not None
        for row in (self.x_row, self.y_row):
            row.setEnabled(enabled)
        if box is not None and not self._commit.isActive():     # never overwrite an edit in flight
            self.show_box(box)
        self._set_rows(rows)
        self.note.setText(note)
        self.note.setVisible(bool(note))

    def _set_rows(self, rows) -> None:
        keys = [key for key, _value, _tone, _index in rows]
        clickable = [index is not None for _key, _value, _tone, index in rows]
        if keys != self._row_keys or [isinstance(row, _ClickableKvRow) for row in self._rows] != clickable:
            for row in self._rows:
                self.body.removeWidget(row)
                row.deleteLater()
            self._rows = []
            position = self.body.indexOf(self.evidence_header) + 1
            for offset, (key, _value, _tone, index) in enumerate(rows):
                row = _ClickableKvRow(key, "") if index is not None else KvRow(key, "")
                self.body.insertWidget(position + offset, row)
                self._rows.append(row)
            self._row_keys = keys
        for row, (_key, value, tone, index) in zip(self._rows, rows, strict=True):
            row.set_value(value, tone)
            if isinstance(row, _ClickableKvRow):
                row.setCursor(Qt.CursorShape.PointingHandCursor)
                try:
                    row.clicked.disconnect()
                except TypeError:
                    pass
                row.clicked.connect(lambda sample=index: self.sample_requested.emit(sample))

    def _on_edited(self) -> None:
        self._commit.start()

    def _emit_box(self) -> None:
        x, y, width, height = self.spin_values()
        self.box_edited.emit((x, y, width, height))


# --------------------------------------------------------------------------
# The tab
# --------------------------------------------------------------------------

class CropTab:
    """`StageTab` for "Crop": the canvas, the filmstrip and their panel."""

    title = "Crop"

    def __init__(self, controller):
        self._controller = controller
        self._file: str | None = None
        self._samples: list[CropSampleView] = []
        self._selected = 0
        self._thumbnails: dict[float, QImage] = {}

        self._page = QWidget()
        self._page.setObjectName("CropPage")
        self._page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self._page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Ruling B3: these live right-aligned in the stage head, which the
        # Stage mounts through `toolbar()` -- not on the page.
        self._toolbar = QWidget()
        self._toolbar.setObjectName("CropToolbar")
        bar = QHBoxLayout(self._toolbar)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(4)
        self.envelope_button = self._toggle("envelope", "envelope", on=True)
        self.masked_button = self._toggle("masked", "masked")
        self.grid_button = self._toggle("grid", "grid")
        for button in (self.envelope_button, self.masked_button, self.grid_button):
            bar.addWidget(button)
        bar.addSpacing(6)
        self.fit_button = Button("⤢ fit to all 0 samples", small=True)
        self.fit_button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.fit_button.clicked.connect(self.fit_to_samples)
        bar.addWidget(self.fit_button)

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(12, 12, 12, 12)
        column.setSpacing(0)
        self.canvas = CropCanvas()
        column.addWidget(self.canvas)
        self.strip = SampleStrip()
        column.addWidget(self.strip)
        self.timeline_placeholder = QWidget()
        self.timeline_placeholder.setObjectName("CompactTimelinePlaceholder")
        self.timeline_placeholder.setFixedHeight(TIMELINE_HEIGHT)
        column.addWidget(self.timeline_placeholder)
        column.addStretch(1)                     # the slack goes below, not around the frame
        outer.addWidget(body, 1)
        QWidget.setTabOrder(self.canvas, self.strip)

        self.panel = CropInspectorPanel()

        self._commit = QTimer(self._page)
        self._commit.setSingleShot(True)
        self._commit.setInterval(COMMIT_DEBOUNCE_MS)
        self._commit.timeout.connect(lambda: self._commit_box(self.canvas.box()))
        self._frames_timer = QTimer(self._page)         # coalesce a burst of frame_ready
        self._frames_timer.setSingleShot(True)
        self._frames_timer.setInterval(0)
        self._frames_timer.timeout.connect(self.refresh)

        self.canvas.commit_requested.connect(self._commit_box)
        self.canvas.nudged.connect(self._commit.start)
        self.canvas.masks_changed.connect(self._commit_masks)
        self.canvas.box_changed.connect(self._on_box_changed)
        self.strip.selected.connect(self.select)
        self.strip.stepped.connect(self.step)
        self.panel.box_edited.connect(self._commit_box)
        self.panel.sample_requested.connect(self.select)
        controller.frame_ready.connect(self._on_frame_ready)

    def _toggle(self, text: str, key: str, *, on: bool = False) -> Button:
        button = Button(text, "ghost" if not on else "default", small=True, toggled_on=on)
        button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        button.clicked.connect(lambda _checked=False, name=key: self._flip(name))
        return button

    # --- StageTab -----------------------------------------------------------------

    def page(self) -> QWidget:
        return self._page

    def inspector_panel(self) -> QWidget:
        return self.panel

    def toolbar(self) -> QWidget:
        """envelope / masked / grid and "⤢ fit to all N samples", which the
        Stage mounts in the stage head (ruling B3)."""
        return self._toolbar

    def set_file(self, name: str | None) -> None:
        if name != self._file:
            self._flush()                        # a nudge still waiting belongs to the old file
            self.canvas.clear_pending()
            self._samples = []
            self._selected = 0
            self._thumbnails.clear()
            self.strip.reset_page()
        self._file = name
        self.refresh()

    def refresh(self) -> None:
        entry = self._entry()
        if entry is None:
            self._page.setEnabled(False)
            self.strip.set_state([], -1, set(), {})
            self.panel.set_state(None, (1, 1), [], "")
            return
        self._page.setEnabled(True)
        evidence = entry.evidence.get("crop") or {}
        samples = read_samples(evidence)
        if [sample.time for sample in samples] != [sample.time for sample in self._samples]:
            self._selected = self._first_kept(samples)      # a new detection: back to its first hit
        self._samples = samples
        self._selected = max(0, min(self._selected, max(0, len(samples) - 1)))
        self._request_frames()
        self._sync_canvas(entry, evidence)
        self._sync_strip(evidence)
        self._sync_toolbar()
        self._sync_panel(entry, evidence)

    # --- reading ---------------------------------------------------------------------

    def current_file(self) -> str | None:
        return self._file

    def samples(self) -> list[CropSampleView]:
        return list(self._samples)

    def selected_index(self) -> int:
        return self._selected

    def current_time(self) -> float:
        if self._samples:
            return self._samples[min(self._selected, len(self._samples) - 1)].time
        entry = self._entry()
        if entry is None:
            return 0.0
        if entry.sample_time is not None:
            return float(entry.sample_time)
        return 0.4 * float(entry.media.duration or 0.0)

    def _entry(self):
        name = self._file
        if name is None or name not in self._controller.names():
            return None
        return self._controller.entry(name)

    @staticmethod
    def _first_kept(samples) -> int:
        return next((index for index, sample in enumerate(samples) if sample.kept), 0)

    def _video_size(self, entry, evidence) -> tuple[int, int]:
        """The coordinate space the crop box lives in: the NATIVE frame size.
        Frames arrive scaled to at most 720 rows, so the decoded image's size
        is never it; the evidence's `frame_size` stands in until the metadata
        job has run."""
        media = entry.media
        if media.width > 0 and media.height > 0:
            return (media.width, media.height)
        size = evidence.get("frame_size")
        try:
            if size is not None and len(size) == 2 and int(size[0]) > 0 and int(size[1]) > 0:
                return (int(size[0]), int(size[1]))
        except (TypeError, ValueError):
            pass
        return (1920, 1080)

    def _displayed_box(self, entry, evidence, video_size) -> tuple[int, int, int, int]:
        crop = entry.crop
        if crop is not None:
            return (crop.x, crop.y, crop.width, crop.height)
        detected = _read_box(evidence.get("box"))
        if detected is not None:
            return detected
        width, height = video_size
        return (0, int(height * 0.8), width, max(CropCanvas.MIN_BOX, int(height * 0.15)))

    def _disagreeing(self, evidence) -> list[int]:
        """Samples that found text but did not contribute -- only when the
        detection produced a box to disagree with."""
        if _read_box(evidence.get("box")) is None:
            return []
        return [index for index, sample in enumerate(self._samples)
                if not sample.kept and sample.boxes]

    # --- selection -------------------------------------------------------------------

    def select(self, index: int) -> None:
        """Show that sample's frame. Out-of-range clamps, so ◀ / ▶ stop at
        the ends rather than wrapping."""
        if not self._samples:
            return
        self._selected = max(0, min(int(index), len(self._samples) - 1))
        self.refresh()

    def step(self, delta: int) -> None:
        self.select(self._selected + int(delta))

    # --- commands ---------------------------------------------------------------------

    def _flip(self, key: str) -> None:
        on = not self.canvas.overlays()[key]
        self.canvas.set_overlay(key, on)
        button = {"envelope": self.envelope_button, "masked": self.masked_button,
                  "grid": self.grid_button}[key]
        button.set_toggled(on)
        button.set_variant("default" if on else "ghost")

    def fit_to_samples(self) -> None:
        """Re-run the detector's aggregation over the kept samples' boxes,
        with the cutoff the detection itself used (the module docstring)."""
        entry = self._entry()
        if entry is None:
            return
        evidence = entry.evidence.get("crop") or {}
        kept = [sample for sample in self._samples if sample.kept]
        if not kept:
            return
        folder = self._controller.project.folder
        settings = {field: getattr(folder, field) for field in DETECTOR_FIELDS}
        cutoff = evidence.get("cutoff_frac")
        settings["bottom_half_cutoff"] = float(folder.bottom_half_cutoff if cutoff is None else cutoff)
        box = aggregate_crop_box([sample.boxes for sample in kept],
                                 self._video_size(entry, evidence), settings,
                                 [sample.time for sample in kept])
        if box is not None:
            self._commit_box(box)

    def _commit_box(self, box) -> None:
        """The one place an edit becomes a MANUAL value (ruling C8: only the
        controller mutates the model)."""
        self._commit.stop()
        self.canvas.clear_pending()
        entry = self._entry()
        if entry is None:
            return
        box = tuple(int(value) for value in box)
        crop = entry.crop
        if crop is not None and (crop.x, crop.y, crop.width, crop.height) == box:
            return
        self._controller.set_crop(self._file, box)
        self.refresh()

    def _commit_masks(self, masks) -> None:
        if self._controller.project is None:
            return
        self._controller.set_label_masks([tuple(int(value) for value in mask) for mask in masks])
        self.refresh()

    def _flush(self) -> None:
        if self._commit.isActive():
            self._commit_box(self.canvas.box())

    def _on_box_changed(self) -> None:
        box = self.canvas.box()
        self.canvas.set_meta(self.current_time(), sum(1 for s in self._samples if s.kept))
        self.panel.show_box(box)

    # --- frames --------------------------------------------------------------------------

    def _request_frames(self) -> None:
        name = self._file
        if name is None:
            return
        times = [sample.time for sample in self._samples]
        current = self.current_time()
        if current not in times:
            times.append(current)
        if times:
            self._controller.request_frames(name, times)

    def _on_frame_ready(self, name: str, _time: float) -> None:
        """One job brings back many frames, so the signal arrives in bursts:
        coalesce them into one repaint on the event loop.

        `CropTab` is not a QObject, so Qt cannot disconnect this for us when
        the widgets go; the deleted check is what a QWidget view gets for
        free."""
        if name == self._file and not sip.isdeleted(self._page):
            self._frames_timer.start()

    def _thumbnail(self, time: float) -> QImage | None:
        cached = self._thumbnails.get(time)
        if cached is not None:
            return cached
        frame = self._controller.frame(self._file, time)
        if frame is None:
            return None
        image = bgr_to_qimage(frame)
        if image is None or image.isNull():
            return None
        image = image.scaled(SampleStrip.THUMB_WIDTH, SampleStrip.THUMB_HEIGHT,
                             Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                             Qt.TransformationMode.SmoothTransformation)
        self._thumbnails[time] = image
        return image

    # --- syncing the widgets -------------------------------------------------------------

    def _sync_canvas(self, entry, evidence) -> None:
        video_size = self._video_size(entry, evidence)
        box = self._displayed_box(entry, evidence, video_size)
        detected = _read_box(evidence.get("box"))
        self.canvas.set_video_size(video_size)
        self.canvas.set_box(box)
        self.canvas.set_evidence(_read_box(evidence.get("envelope")),
                                 None if detected == box else detected)
        self.canvas.set_brightness(entry.brightness.value if entry.brightness else DEFAULT_BRIGHTNESS)
        self.canvas.set_meta(self.current_time(), sum(1 for s in self._samples if s.kept))
        folder = self._controller.project.folder
        self.canvas.set_masks(folder.label_mask_crops, folder.labels_enabled)
        self.canvas.set_frame(self._controller.frame(self._file, self.current_time()))

    def _sync_strip(self, evidence) -> None:
        warned = set(self._disagreeing(evidence))
        images = {index: self._thumbnail(sample.time) for index, sample in enumerate(self._samples)}
        self.strip.set_state(self._samples, self._selected, warned, images)

    def _sync_toolbar(self) -> None:
        kept = sum(1 for sample in self._samples if sample.kept)
        self.fit_button.setText(f"⤢ fit to all {kept} samples")
        self.fit_button.setEnabled(kept > 0)

    def _sync_panel(self, entry, evidence) -> None:
        # The canvas is synced first, so its box IS the displayed box -- and
        # while an edit waits for its debounced commit it is the edited one,
        # which is what the spin pairs and the "covers it" note must read.
        video_size = self._video_size(entry, evidence)
        box = self.canvas.box()
        rows, note = self._panel_rows(entry, evidence, box)
        self.panel.set_state(box, video_size, rows, note)

    def _panel_rows(self, entry, evidence, box) -> tuple[list[tuple], str]:
        if not evidence:
            source = None if entry.crop is None else entry.crop.source
            return [("Source", crop_caption(None, source)[0], None, None)], ""
        total = len(self._samples)
        kept = sum(1 for sample in self._samples if sample.kept)
        envelope = _read_box(evidence.get("envelope"))
        rows = [("Samples with text", f"{kept} / {total}", None, None),
                ("Text envelope", "—" if envelope is None
                 else f"y {envelope[1]}–{envelope[1] + envelope[3]}", None, None)]
        flagged = evidence.get("flagged")
        if flagged:
            rows.append(("Flagged", str(flagged), "warn", None))
        detected = _read_box(evidence.get("box"))
        if detected is not None and detected != box:
            rows.append(("detected", f"{_box_text(detected)} · yours {_box_text(box)}", "warn", None))
        covered, disagreeing = True, self._disagreeing(evidence)
        for index in disagreeing:
            sample = self._samples[index]
            extent = sample.extent
            where = "lower" if self._below(extent, envelope) else "higher"
            rows.append((f"1 sample sits {where}", f"{clock(sample.time)} ▸", "warn", index))
            covered = covered and _covers(box, extent)
        note = "" if not disagreeing else (COVERED_NOTE if covered else OUTSIDE_NOTE)
        return rows, note

    @staticmethod
    def _below(extent, envelope) -> bool:
        if extent is None:
            return False
        if envelope is None:
            return True
        return (extent[1] + extent[3] / 2) > (envelope[1] + envelope[3] / 2)
