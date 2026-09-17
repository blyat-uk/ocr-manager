"""Thin base widgets styled by `app.theme.qss.build_stylesheet()` (Task 1).

Each widget sets the Qt dynamic properties its QSS rule selects on
(`app/theme/qss.py`) in its constructor, and updates them -- then
repolishes, since Qt does not re-evaluate style rules on a property change
by itself -- in its setters. Later Stage 3B tasks compose these into the
actual views; nothing here knows about `core.project`/`core.jobs`.
"""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from app.theme import tokens

# Dot tones -> fill colour (`.d-ok` / `.d-warn` / `.d-bad` / `.d-idle` / `.d-run`).
_DOT_TONE_COLOR = {
    "ok": tokens.OK,
    "warn": tokens.WARN,
    "bad": tokens.BAD,
    "idle": tokens.DIM2,
    "run": tokens.BLUE,
}
# ConfBar/MiniProgress fill tones (`.bar u`'s background).
_BAR_TONE_COLOR = {
    "ok": tokens.OK,
    "warn": tokens.WARN,
    "bad": tokens.BAD,
}


def _repolish(widget: QWidget) -> None:
    """Force Qt to re-evaluate `widget`'s QSS after a dynamic property
    changed at runtime: `setProperty()` alone does not repaint the new
    rule, Qt only re-polishes on show/style change."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


class Button(QPushButton):
    """`.btn` / `.btn.ghost` / `.btn.primary` / `.btn.sm` / `.btn.on`
    (the last only in `tabs-hifi.html`, used for toggle-style buttons like
    the brightness tab's zoom presets)."""

    def __init__(self, text: str, variant: str = "default", small: bool = False,
                 toggled_on: bool = False, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("variant", variant)
        self.setProperty("small", bool(small))
        self.setProperty("toggled", bool(toggled_on))

    def set_variant(self, variant: str) -> None:
        self.setProperty("variant", variant)
        _repolish(self)

    def set_toggled(self, toggled_on: bool) -> None:
        self.setProperty("toggled", bool(toggled_on))
        _repolish(self)


class Dot(QWidget):
    """`.dot` -- a small status circle. `tone`: "ok" | "warn" | "bad" |
    "idle" | "run"."""

    def __init__(self, tone: str = "idle", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Dot")
        self._tone = tone
        self.setProperty("tone", tone)
        self.setFixedSize(tokens.DOT_SIZE, tokens.DOT_SIZE)

    def set_tone(self, tone: str) -> None:
        self._tone = tone
        self.setProperty("tone", tone)
        _repolish(self)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(_DOT_TONE_COLOR.get(self._tone, tokens.DIM2)))
        painter.drawEllipse(self.rect())
        painter.end()


class Chip(QWidget):
    """`.chip` -- a dot + bold count + label pill, e.g. the top bar's
    "3 reviewed" / "1 needs you" / "1 detecting" counters. `dot`: "ok" |
    "warn" | "idle" | "run" (the tones the mockup's chipbar actually uses)."""

    def __init__(self, dot: str = "idle", count: int = 0, label: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Chip")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 3, 9, 3)
        layout.setSpacing(6)
        self._dot = Dot(dot)
        self._count_label = QLabel()
        self._count_label.setProperty("chipRole", "count")
        self._label = QLabel(label)
        self._label.setProperty("chipRole", "label")
        layout.addWidget(self._dot)
        layout.addWidget(self._count_label)
        layout.addWidget(self._label)
        self.set_count(count)

    def set_count(self, count: int) -> None:
        self._count_label.setText(str(count))

    def set_tone(self, dot: str) -> None:
        self._dot.set_tone(dot)


class Badge(QLabel):
    """`.badge` / `.badge.w` / `.badge.g` -- the review-queue state pill
    (ruling B10). `tone`: "default" | "warn" | "good" | "bad" -- "bad" has
    no hi-fi figure (no mockup ever renders a failed badge); it reuses the
    default `.badge` background with `--bad` text rather than inventing an
    unsourced colour, see `app/theme/qss.py`."""

    def __init__(self, text: str = "", tone: str = "default", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setObjectName("Badge")
        self.setProperty("badge", tone)

    def set_state(self, text: str, tone: str) -> None:
        self.setText(text)
        self.setProperty("badge", tone)
        _repolish(self)


class KvRow(QWidget):
    """`.kv` -- a key/value row, e.g. "Crop" / "288, 784 · 1344 × 55".
    `tone`: None | "warn" | "ok" | "bad" | "acc" colours the value text;
    warn/bad additionally colour the row's own border (ui-spec §2.5's
    warn-tinted kv rows -- `TAG_BAD_BORDER` is bad's equivalent, the same
    token `.ztile.bad`/bad-tinted tags use)."""

    def __init__(self, key: str, value: str, tone: str | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("KvRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(8)
        self._key_label = QLabel(key)
        self._key_label.setProperty("kvRole", "key")
        self._value_label = QLabel()
        self._value_label.setProperty("kvRole", "value")
        layout.addWidget(self._key_label)
        layout.addStretch(1)
        layout.addWidget(self._value_label)
        self.set_value(value, tone)

    def set_value(self, value: str, tone: str | None = None) -> None:
        self._value_label.setText(value)
        tone_prop = tone or ""
        self._value_label.setProperty("tone", tone_prop)
        self.setProperty("tone", tone_prop)
        _repolish(self._value_label)
        _repolish(self)


class SectionHeader(QWidget):
    """`.sec-h` -- an uppercase, letter-spaced section title (e.g.
    "DETECTED", "PROOF · REAL OCR OF 30 S"), with an optional trailing
    widget (e.g. a "re-detect" `Button`) right-aligned on the same row."""

    def __init__(self, text: str, trailing: QWidget | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("SectionHeader")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._label = QLabel()
        self._label.setProperty("sectionRole", "title")
        font = self._label.font()
        # CSS `letter-spacing:.07em` -- Qt has no CSS letter-spacing QSS
        # property, so this is set on the QFont directly. PercentageSpacing
        # scales the font's own letter spacing; 100 = unchanged.
        font.setLetterSpacing(QFont.SpacingType.PercentageSpacing,
                               100 + tokens.LETTER_SPACING_SCOPE_EM * 100)
        self._label.setFont(font)
        layout.addWidget(self._label)
        layout.addStretch(1)
        self._trailing = trailing
        if trailing is not None:
            layout.addWidget(trailing)
        self.set_text(text)

    def set_text(self, text: str) -> None:
        self._label.setText(text.upper())


class SegmentedControl(QWidget):
    """`.rail-head .seg` -- a row of exclusive text segments, e.g. "All 5" /
    "Needs you 1" / "Reviewed 3". Clicking a segment makes it current and
    emits `current_changed(index)`. `set_current()` is the programmatic
    counterpart (syncing the control from a model) and does NOT emit, so a
    caller driving both directions cannot create a feedback loop."""

    current_changed = pyqtSignal(int)

    def __init__(self, items: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("SegmentedControl")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._buttons: list[QPushButton] = []
        self._current = 0
        self.set_labels(items)

    def set_labels(self, items: list[str]) -> None:
        for button in self._buttons:
            self._layout.removeWidget(button)
            button.deleteLater()
        self._buttons = []
        if self._current >= len(items):
            self._current = 0
        for index, label in enumerate(items):
            button = QPushButton(label)
            button.setObjectName("SegmentItem")
            button.setFlat(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setProperty("on", index == self._current)
            button.clicked.connect(lambda _checked=False, i=index: self._on_clicked(i))
            self._layout.addWidget(button)
            self._buttons.append(button)

    def _on_clicked(self, index: int) -> None:
        if index == self._current:
            return
        self._current = index
        self._refresh()
        self.current_changed.emit(index)

    def set_current(self, index: int) -> None:
        if not (0 <= index < len(self._buttons)) or index == self._current:
            return
        self._current = index
        self._refresh()

    def current(self) -> int:
        return self._current

    def _refresh(self) -> None:
        for index, button in enumerate(self._buttons):
            button.setProperty("on", index == self._current)
            _repolish(button)


class _BarTrack(QWidget):
    """Shared fixed-size painted track behind `ConfBar`/`MiniProgress`: a
    rounded `TRACK_BG` background (`.bar` / `.mini`) with a solid fill rect
    clipped to `fraction` of the width (`.bar u` / `.mini u`). Not part of
    Task 1's public widget list -- a fraction-of-width fill cannot be
    expressed in static QSS, so both bars paint themselves directly."""

    def __init__(self, width: int, height: int, fill: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self._fraction = 0.0
        self._fill = fill

    def set_value(self, fraction: float, fill: str) -> None:
        self._fraction = max(0.0, min(1.0, fraction))
        self._fill = fill
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        radius = min(float(tokens.RADIUS_XS), rect.height() / 2)
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(tokens.TRACK_BG))
        painter.drawPath(path)
        if self._fraction > 0:
            painter.setClipPath(path)
            painter.setBrush(QColor(self._fill))
            painter.drawRect(0, 0, round(rect.width() * self._fraction), self.height())
        painter.end()


class ConfBar(QWidget):
    """`.conf` -- a 74x3px confidence bar (`.bar`) plus caption text, e.g.
    "12 of 12 samples agree". `tone`: "ok" | "warn" | "bad" picks the
    fill colour."""

    def __init__(self, fraction: float, tone: str = "ok", caption: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("ConfBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 8)
        layout.setSpacing(6)
        self._track = _BarTrack(tokens.BAR_WIDTH, tokens.BAR_HEIGHT, tokens.OK)
        self._caption = QLabel()
        self._caption.setStyleSheet(
            f"color: {tokens.DIM2}; font-size: {tokens.FONT_SIZE_SM}px; background: transparent;"
        )
        layout.addWidget(self._track)
        layout.addWidget(self._caption, 1)
        self.set_value(fraction, tone, caption)

    def set_value(self, fraction: float, tone: str = "ok", caption: str = "") -> None:
        self.setProperty("tone", tone)
        self._track.set_value(fraction, _BAR_TONE_COLOR.get(tone, tokens.OK))
        self._caption.setText(caption)
        _repolish(self)


class MiniProgress(QWidget):
    """`.mini` -- a 90x4px, always-blue progress bar (the activity strip)."""

    def __init__(self, fraction: float = 0.0, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("MiniProgress")
        self.setProperty("tone", "run")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._track = _BarTrack(tokens.MINI_WIDTH, tokens.MINI_HEIGHT, tokens.BLUE)
        layout.addWidget(self._track)
        self.set_value(fraction)

    def set_value(self, fraction: float) -> None:
        self._track.set_value(fraction, tokens.BLUE)
