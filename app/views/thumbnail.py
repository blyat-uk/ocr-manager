"""A queue row's 56×32 thumbnail (`.thumb`): the file's frame, or the
mockup's gradient placeholder, with the crop box drawn in miniature at its
true position (`.thumb i`)."""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from app.theme import tokens

GRADIENT_DEGREES = 160           # linear-gradient(160deg, THUMB_TOP, THUMB_BOTTOM 70%)
GRADIENT_END_STOP = 0.7
BOX_OUTLINE_ALPHA = 0.9          # border:1px solid rgba(255,194,71,.9)
BOX_FILL_ALPHA = 0.10            # background:rgba(255,194,71,.10)
PENDING_BOX_OPACITY = 0.3        # a pending row's box fades (.thumb i opacity .3)
SKIPPED_OPACITY = 0.5            # a skipped row is dimmed (ruling B10)


def css_gradient(rect: QRectF, degrees: float) -> QLinearGradient:
    """A QLinearGradient laid out like CSS `linear-gradient(<deg>, ...)`: 0deg
    points up, 90deg right, and the line spans the box's corners."""
    angle = math.radians(degrees)
    dx, dy = math.sin(angle), -math.cos(angle)
    half = (abs(rect.width() * dx) + abs(rect.height() * dy)) / 2
    centre = rect.center()
    return QLinearGradient(QPointF(centre.x() - dx * half, centre.y() - dy * half),
                           QPointF(centre.x() + dx * half, centre.y() + dy * half))


def _with_alpha(hex_colour: str, alpha: float) -> QColor:
    colour = QColor(hex_colour)
    colour.setAlphaF(alpha)
    return colour


class Thumbnail(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(tokens.THUMB_WIDTH, tokens.THUMB_HEIGHT)
        self._image: QImage | None = None
        self._frame_size: tuple[int, int] | None = None
        self._crop: tuple[int, int, int, int] | None = None
        self._pending = False
        self._dimmed = False

    def set_state(self, image: QImage | None, frame_size: tuple[int, int] | None,
                  crop: tuple[int, int, int, int] | None, *, pending: bool = False, dimmed: bool = False) -> None:
        """`frame_size` is the video's (width, height), the crop's coordinate
        space; None falls back to the image's size."""
        state = (image, frame_size, crop, pending, dimmed)
        if state == (self._image, self._frame_size, self._crop, self._pending, self._dimmed):
            return
        self._image, self._frame_size, self._crop, self._pending, self._dimmed = state
        self.update()

    def image(self) -> QImage | None:
        return self._image

    def _source_size(self) -> tuple[int, int] | None:
        if self._frame_size and self._frame_size[0] > 0 and self._frame_size[1] > 0:
            return self._frame_size
        if self._image is not None and not self._image.isNull():
            return self._image.width(), self._image.height()
        return None

    def frame_rect(self) -> QRectF:
        """Where the frame is drawn: its aspect ratio fitted in the thumbnail."""
        bounds = QRectF(self.rect())
        size = self._source_size()
        if size is None:
            return bounds
        scale = min(bounds.width() / size[0], bounds.height() / size[1])
        width, height = size[0] * scale, size[1] * scale
        return QRectF((bounds.width() - width) / 2, (bounds.height() - height) / 2, width, height)

    def crop_rect(self) -> QRectF | None:
        size = self._source_size()
        if self._crop is None or size is None:
            return None
        frame = self.frame_rect()
        sx, sy = frame.width() / size[0], frame.height() / size[1]
        x, y, width, height = self._crop
        return QRectF(frame.x() + x * sx, frame.y() + y * sy, width * sx, height * sy)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self._dimmed:
            painter.setOpacity(SKIPPED_OPACITY)
        bounds = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(bounds, tokens.RADIUS_THUMB, tokens.RADIUS_THUMB)
        painter.setClipPath(clip)
        if self._pending:
            painter.fillRect(bounds, QColor(tokens.THUMB_PENDING))
        else:
            gradient = css_gradient(bounds, GRADIENT_DEGREES)
            gradient.setColorAt(0.0, QColor(tokens.THUMB_TOP))
            gradient.setColorAt(GRADIENT_END_STOP, QColor(tokens.THUMB_BOTTOM))
            gradient.setColorAt(1.0, QColor(tokens.THUMB_BOTTOM))
            painter.fillRect(bounds, gradient)
        if self._image is not None and not self._image.isNull():
            painter.drawImage(self.frame_rect(), self._image)
        box = self.crop_rect()
        if box is not None:
            if self._pending:
                painter.setOpacity(painter.opacity() * PENDING_BOX_OPACITY)
            painter.setPen(QPen(_with_alpha(tokens.ACC, BOX_OUTLINE_ALPHA), 1))
            painter.setBrush(_with_alpha(tokens.ACC, BOX_FILL_ALPHA))
            painter.drawRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5), tokens.RADIUS_THUMB_BOX,
                                    tokens.RADIUS_THUMB_BOX, Qt.SizeMode.AbsoluteSize)
        painter.end()
