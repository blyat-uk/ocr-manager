"""What the episode screens share: fonts and text labels in the workbench's
tokens, and the few words every screen uses.

The global stylesheet (app/theme/qss.py) styles its widgets by object name,
and this package owns none of those rules, so its labels carry their own
small style sheet built from the same tokens: one colour, one size, one
weight -- nothing a later theme change could leave behind, because every
value is read from `app.theme.tokens` at construction.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt6.QtWidgets import QLabel, QWidget

from app.theme import tokens

# Sizes in mockup pixels (episode-view-v2.html's screens, drawn at about
# half their real size, doubled); every use goes through tokens.px/pt.
HEADLINE_SIZE = 19
TITLE_SIZE = 15
BODY_SIZE = tokens.FONT_SIZE_BODY_BASE
SMALL_SIZE = tokens.FONT_SIZE_BTN_SM_BASE
CAPTION_SIZE = 22                   # the slideshow's line text

def font(size: float, weight: int | None = None) -> QFont:
    """The theme's family stack at mockup `size` px (scaled), for painting."""
    result = QFont()
    result.setFamilies(tokens.FONT_STACK)
    result.setPixelSize(max(1, round(tokens.pt(size))))
    if weight is not None:
        result.setWeight(QFont.Weight(weight))
    return result


def text_label(text: str = "", *, color: str = tokens.TXT, size: float = BODY_SIZE, weight: int | None = None,
               wrap: bool = False, center: bool = False, name: str = "",
               parent: QWidget | None = None) -> QLabel:
    """A QLabel in one colour, size and weight."""
    label = QLabel(text, parent)
    if name:
        label.setObjectName(name)
    label.setWordWrap(wrap)
    if center:
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
    set_label_style(label, color=color, size=size, weight=weight)
    return label


def set_label_style(label: QLabel, *, color: str, size: float = BODY_SIZE, weight: int | None = None) -> None:
    rules = [f"color: {color}", "background: transparent", f"font-size: {round(tokens.pt(size))}px"]
    if weight is not None:
        rules.append(f"font-weight: {weight}")
    label.setStyleSheet("; ".join(rules) + ";")
    label.setProperty("tone_color", color)


def episode_label(name: str) -> str:
    """"EP05" for "EP05.mkv": how the screens name another episode."""
    return os.path.splitext(os.path.basename(name))[0]


class FillBar(QWidget):
    """`.bar` of episode-view-v2.html: a thin TRACK_BG track filled in
    accent to `fraction` (the Preparing checklist's progress)."""

    def __init__(self, width: int = 240, height: int = 4, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(tokens.px(width), tokens.px(height))
        self._fraction = 0.0

    def set_fraction(self, fraction: float) -> None:
        fraction = max(0.0, min(1.0, float(fraction)))
        if fraction != self._fraction:
            self._fraction = fraction
            self.update()

    def fraction(self) -> float:
        return self._fraction

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        radius = rect.height() / 2
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        painter.fillPath(path, QColor(tokens.TRACK_BG))
        if self._fraction > 0:
            painter.setClipPath(path)
            painter.fillRect(QRectF(0, 0, rect.width() * self._fraction, rect.height()), QColor(tokens.ACC))
        painter.end()
