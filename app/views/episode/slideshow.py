"""Painted frames for the episode screens.

`SlideShow` is the Working screen's picture (episode-view-v2.html, screen
4): each released line cross-fades to its frame over FADE_MS while the
frame pushes in slowly, 1.00 -> PUSH_IN over the time until the next line,
and the text OCR read rises into place in a caption band under the frame.
A line whose frame never came keeps the picture on screen and only changes
the words. The previous frame is kept, frozen at the scale it had reached,
for the fade to come from.

The burned-in subtitle is what the user watches being read, so nothing may
hide it: the whole frame is shown (fitted, never cropped to fill), the
push-in grows from the frame's bottom edge (the subtitles' side stays put
while the rest drifts), and the recognised text sits under the picture --
burned-in line above, what we read below, never one over the other.

`FramePreview` is a still frame with the crop box drawn on it: the
Preparing screen's "here is what we found" (green and solid when the box
is good to go, amber and dashed when it needs a look), and the Done
screen's look at a line the user clicked.

Both are plain painted QWidgets driven by QVariantAnimation (no QML). With
`animated=False` every animation jumps to its end value, which is what the
offscreen tests use: they assert what is shown, never how long it took.
Frames arrive at most 720 rows high (the view cache's size), so a box in
video pixels is scaled by the image over the video size, as the crop view
does it.
"""
from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPointF, QRectF, QSize, Qt, QVariantAnimation
from PyQt6.QtGui import (
    QColor,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from app.theme import tokens
from app.views.episode.common import CAPTION_SIZE, SMALL_SIZE, font

FADE_MS = 400                  # the cross-fade to a new frame
RISE_MS = 450                  # the line text rising into place
RISE_PX = 12                   # how far it rises from (mockup px)
PUSH_IN = 1.06                 # the slow zoom's end scale
MIN_PUSH_MS = 1000             # a push-in never runs faster than this
STAGE_BG = "#07090c"           # .work background (episode-view-v2.html)
CAPTION_LINES = 2              # the caption band holds two lines of recognised text
CAPTION_PAD = 10               # mockup px above and below the caption band's text
TAG_ALPHA = 179                # .ts background rgba(10,12,16,.7)


def paint_placeholder(painter: QPainter, rect: QRectF) -> None:
    """The canvas gradient the workbench draws where a frame will be:
    radial-gradient(120% 90% at 30% 25%, CANVAS_TOP, CANVAS_MID 55%, CANVAS_BOTTOM)."""
    gradient = QRadialGradient(QPointF(rect.left() + rect.width() * 0.30, rect.top() + rect.height() * 0.25),
                               max(rect.width(), rect.height()) * 0.9)
    gradient.setColorAt(0.0, QColor(tokens.CANVAS_TOP))
    gradient.setColorAt(tokens.CANVAS_MID_STOP, QColor(tokens.CANVAS_MID))
    gradient.setColorAt(1.0, QColor(tokens.CANVAS_BOTTOM))
    painter.fillRect(rect, gradient)


def pushed_rect(image_size: QSize, target: QRectF, scale: float = 1.0) -> QRectF:
    """The frame fitted whole into `target`, `scale` times larger, grown from
    the middle of its bottom edge: the bottom (where subtitles are burned in)
    never moves, and whatever grows past `target` is clipped at the top and
    the sides."""
    fitted = contain_rect(image_size, target)
    width, height = fitted.width() * scale, fitted.height() * scale
    return QRectF(fitted.center().x() - width / 2, fitted.bottom() - height, width, height)


def contain_rect(image_size: QSize, target: QRectF) -> QRectF:
    """Where to draw an image so all of it fits `target`, centred."""
    if image_size.width() <= 0 or image_size.height() <= 0:
        return QRectF(target)
    factor = min(target.width() / image_size.width(), target.height() / image_size.height())
    width, height = image_size.width() * factor, image_size.height() * factor
    return QRectF(target.center().x() - width / 2, target.center().y() - height / 2, width, height)


def caption_text(text: str) -> str:
    """ASS's "\\N" line breaks as real ones."""
    return text.replace("\\N", "\n").replace("\\n", "\n")


def _animation(parent, start: float, end: float, on_value) -> QVariantAnimation:
    animation = QVariantAnimation(parent)
    animation.setStartValue(float(start))
    animation.setEndValue(float(end))
    animation.valueChanged.connect(lambda value: on_value(float(value)))
    return animation


class SlideShow(QWidget):
    def __init__(self, parent: QWidget | None = None, *, animated: bool = True):
        super().__init__(parent)
        self.setObjectName("EpisodeSlideShow")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(tokens.px(240), tokens.px(135))
        self._animated = animated
        self._current: QPixmap | None = None
        self._previous: QPixmap | None = None
        self._previous_scale = 1.0
        self._fade = 1.0
        self._scale = 1.0
        self._rise = 1.0
        self._caption = ""
        self._stamp = ""
        self._fade_animation = _animation(self, 0.0, 1.0, self._set_fade)
        self._fade_animation.setDuration(FADE_MS)
        self._fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._push_animation = _animation(self, 1.0, PUSH_IN, self._set_scale)
        self._push_animation.setEasingCurve(QEasingCurve.Type.Linear)
        self._rise_animation = _animation(self, 0.0, 1.0, self._set_rise)
        self._rise_animation.setDuration(RISE_MS)
        self._rise_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    # --- reading (tests) --------------------------------------------------------------

    def caption(self) -> str:
        return self._caption

    def stamp(self) -> str:
        return self._stamp

    def has_picture(self) -> bool:
        return self._current is not None

    def picture_key(self) -> int | None:
        """The shown pixmap's cache key: changes exactly when the picture does."""
        return None if self._current is None else self._current.cacheKey()

    def fade(self) -> float:
        return self._fade

    def scale(self) -> float:
        return self._scale

    def rise(self) -> float:
        return self._rise

    # --- showing ----------------------------------------------------------------------

    def show_line(self, image: QImage | None, text: str, stamp: str = "", hold_seconds: float = 3.0) -> None:
        """Show a line: its frame (None keeps the current picture), its text
        and its time stamp. `hold_seconds` is how long until the next line is
        expected, the length of the push-in."""
        if image is not None and not image.isNull():
            self._previous = self._current
            self._previous_scale = self._scale
            self._current = QPixmap.fromImage(image)
            self._start(self._fade_animation, self._set_fade, 0.0, 1.0)
            self._push_animation.setDuration(max(MIN_PUSH_MS, int(hold_seconds * 1000)))
            self._start(self._push_animation, self._set_scale, 1.0, PUSH_IN)
        self._caption = caption_text(text)
        self._stamp = stamp
        self._start(self._rise_animation, self._set_rise, 0.0, 1.0)
        self.update()

    def clear(self) -> None:
        for animation in (self._fade_animation, self._push_animation, self._rise_animation):
            animation.stop()
        self._current = self._previous = None
        self._fade, self._scale, self._rise, self._previous_scale = 1.0, 1.0, 1.0, 1.0
        self._caption = self._stamp = ""
        self.update()

    def _start(self, animation: QVariantAnimation, setter, start: float, end: float) -> None:
        animation.stop()
        if not self._animated:
            setter(end)
            return
        setter(start)
        animation.start()

    def _set_fade(self, value: float) -> None:
        self._fade = value
        if value >= 1.0:
            self._previous = None                       # nothing to fade from any more
        self.update()

    def _set_scale(self, value: float) -> None:
        self._scale = value
        self.update()

    def _set_rise(self, value: float) -> None:
        self._rise = value
        self.update()

    # --- painting ---------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect())
        painter.fillRect(rect, QColor(STAGE_BG))
        picture, band = self.picture_rect(), self.caption_rect()
        painter.save()
        painter.setClipRect(picture)
        if self._current is None:
            paint_placeholder(painter, picture)
        else:
            if self._fade < 1.0:                        # what the new frame fades in over
                if self._previous is not None:
                    painter.drawPixmap(pushed_rect(self._previous.size(), picture, self._previous_scale),
                                       self._previous, QRectF(self._previous.rect()))
                else:
                    paint_placeholder(painter, contain_rect(self._current.size(), picture))
            painter.setOpacity(self._fade)
            painter.drawPixmap(pushed_rect(self._current.size(), picture, self._scale), self._current,
                               QRectF(self._current.rect()))
            painter.setOpacity(1.0)
        painter.restore()
        self._paint_stamp(painter, picture)
        self._paint_caption(painter, band)
        painter.end()

    def _caption_height(self) -> float:
        metrics = QFontMetricsF(font(CAPTION_SIZE, 600))
        return metrics.lineSpacing() * CAPTION_LINES + 2 * tokens.px(CAPTION_PAD)

    def _layout(self) -> tuple[QRectF, QRectF]:
        """(frame, caption band): the frame fitted whole into the widget above
        room for the band, the band right under it, and the two centred as one
        block -- the text read sits under the line it was read from, not
        across an empty letterbox from it."""
        band = min(self._caption_height(), self.height() / 3)
        size = self._current.size() if self._current is not None else QSize(16, 9)
        frame = contain_rect(size, QRectF(0, 0, self.width(), max(1.0, self.height() - band)))
        top = max(0.0, (self.height() - frame.height() - band) / 2)
        frame.moveTop(top)
        return frame, QRectF(0, frame.bottom(), self.width(), band)

    def caption_rect(self) -> QRectF:
        """The band under the frame the recognised text is drawn in."""
        return self._layout()[1]

    def picture_rect(self) -> QRectF:
        """Where the frame is drawn (the push-in is clipped to it)."""
        return self._layout()[0]

    def _paint_stamp(self, painter: QPainter, picture: QRectF) -> None:
        if not self._stamp:
            return
        painter.setFont(font(SMALL_SIZE))
        metrics = painter.fontMetrics()
        pad_x, pad_y = tokens.px(6), tokens.px(2)
        box = QRectF(picture.left() + tokens.px(10), picture.top() + tokens.px(9), metrics.horizontalAdvance(self._stamp) + 2 * pad_x,
                     metrics.height() + 2 * pad_y)
        path = QPainterPath()
        path.addRoundedRect(box, tokens.RADIUS_TAG, tokens.RADIUS_TAG)
        red, green, blue, _alpha = tokens.TAG_BG
        painter.fillPath(path, QColor(red, green, blue, TAG_ALPHA))
        painter.setPen(QColor(tokens.TXT))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, self._stamp)

    def _paint_caption(self, painter: QPainter, band: QRectF) -> None:
        if not self._caption:
            return
        painter.save()
        painter.setClipRect(band)
        painter.setFont(font(CAPTION_SIZE, 600))
        margin = tokens.px(24)
        area = band.adjusted(margin, 0, -margin, 0).translated(0, tokens.px(RISE_PX) * (1.0 - self._rise))
        flags = int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap)
        painter.setOpacity(self._rise)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(area, flags, self._caption)
        painter.restore()


class FramePreview(QWidget):
    """A frame, fitted whole into a 16:9 box with rounded corners, with an
    optional crop box drawn on it. Without a frame it draws the canvas
    gradient, the box still placed by the video size when that is known."""

    ASPECT = 16 / 9
    MIN_WIDTH = 160

    def __init__(self, parent: QWidget | None = None, *, width: int = 480):
        super().__init__(parent)
        self.setObjectName("EpisodeFramePreview")
        self._image: QImage | None = None
        self._box: tuple[int, int, int, int] | None = None
        self._video_size: tuple[int, int] = (0, 0)
        self._tone = "ok"
        self._width = tokens.px(width)
        self.fit(self._width)

    def fit(self, max_width: int) -> None:
        """Size the preview to `max_width` (at most its own width, at least
        MIN_WIDTH), 16:9. The screens call it as they resize, so the preview
        shrinks with a small window instead of pushing the buttons off it."""
        width = max(tokens.px(self.MIN_WIDTH), min(self._width, int(max_width)))
        self.setFixedSize(width, round(width / self.ASPECT))

    # --- state ------------------------------------------------------------------------

    def set_image(self, image: QImage | None) -> None:
        self._image = None if image is None or image.isNull() else image
        self.update()

    def has_image(self) -> bool:
        return self._image is not None

    def set_box(self, box: tuple[int, int, int, int] | None, video_size: tuple[int, int] = (0, 0),
                tone: str = "ok") -> None:
        """`box` (x, y, width, height) in video pixels; `tone` "ok" draws it
        green and solid, "warn" amber and dashed."""
        self._box = None if box is None else tuple(int(v) for v in box)
        self._video_size = (int(video_size[0]), int(video_size[1]))
        self._tone = tone
        self.update()

    def box(self) -> tuple[int, int, int, int] | None:
        return self._box

    def tone(self) -> str:
        return self._tone

    # --- painting ---------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        outer = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(outer, tokens.RADIUS_BTN, tokens.RADIUS_BTN)
        painter.setClipPath(clip)
        painter.fillRect(outer, QColor(STAGE_BG))
        if self._image is not None:
            target = contain_rect(self._image.size(), outer)
            painter.drawImage(target, self._image)
            frame = (self._image.width(), self._image.height())
        else:
            paint_placeholder(painter, outer)
            target = contain_rect(QSize(*self._video_size), outer) if all(self._video_size) else outer
            frame = self._video_size
        self._paint_box(painter, target, frame)
        painter.end()

    def _paint_box(self, painter: QPainter, target: QRectF, frame: tuple[int, int]) -> None:
        video_width, video_height = self._video_size
        if self._box is None or video_width <= 0 or video_height <= 0 or frame[0] <= 0 or frame[1] <= 0:
            return
        x, y, width, height = self._box
        sx, sy = target.width() / video_width, target.height() / video_height
        rect = QRectF(target.left() + x * sx, target.top() + y * sy, width * sx, height * sy)
        warn = self._tone == "warn"
        pen = QPen(QColor(tokens.WARN if warn else tokens.OK))
        pen.setWidthF(max(1.5, 1.5 * tokens.UI_SCALE))
        if warn:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, tokens.RADIUS_XS, tokens.RADIUS_XS)
