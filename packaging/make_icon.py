"""Draw the application icon: resources/app-icon.png (1024 px).

A dark video frame with a burnt-in subtitle line, and the amber crop
brackets the app draws around it. Run it with the dev venv when the design
changes; the PNG is committed, and the build derives .ico/.icns from it.

    .venv/bin/python packaging/make_icon.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QImage, QPainter, QPainterPath, QPen

OUT = Path(__file__).resolve().parent.parent / "resources" / "app-icon.png"
SIZE = 1024
CJK_FAMILIES = ("Source Han Sans CN", "Noto Sans CJK SC", "Noto Sans SC", "Microsoft YaHei", "PingFang SC")


def cjk_font() -> QFont:
    families = set(QFontDatabase.families())
    for family in CJK_FAMILIES:
        if family in families:
            font = QFont(family)
            font.setWeight(QFont.Weight.Bold)
            return font
    sys.exit(f"no CJK font found (tried {', '.join(CJK_FAMILIES)})")


def draw() -> QImage:
    img = QImage(SIZE, SIZE, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    # The tile: the app's canvas colour, a macOS-like squircle margin.
    tile = QRectF(64, 64, SIZE - 128, SIZE - 128)
    path = QPainterPath()
    path.addRoundedRect(tile, 200, 200)
    p.fillPath(path, QColor("#111317"))

    # The video frame: a 16:9 screen with a dim horizon, the picture.
    screen = QRectF(152, 232, SIZE - 304, (SIZE - 304) * 9 / 16)
    frame = QPainterPath()
    frame.addRoundedRect(screen, 36, 36)
    p.fillPath(frame, QColor("#23272f"))
    hill = QPainterPath()
    hill.moveTo(screen.left(), screen.top() + screen.height() * 0.55)
    hill.cubicTo(screen.left() + screen.width() * 0.35, screen.top() + screen.height() * 0.30,
                 screen.left() + screen.width() * 0.60, screen.top() + screen.height() * 0.62,
                 screen.right(), screen.top() + screen.height() * 0.40)
    hill.lineTo(screen.right(), screen.bottom())
    hill.lineTo(screen.left(), screen.bottom())
    hill.closeSubpath()
    p.save()
    p.setClipPath(frame)
    p.fillPath(hill, QColor("#2d323c"))
    p.restore()

    # The burnt-in subtitle: white glyphs at the bottom of the frame.
    font = cjk_font()
    font.setPixelSize(150)
    p.setFont(font)
    p.setPen(QColor("#f4f5f7"))
    line = QRectF(screen.left(), screen.bottom() - 250, screen.width(), 200)
    p.drawText(line, Qt.AlignmentFlag.AlignCenter, "字幕")

    # The crop box: amber corner brackets around the subtitle line.
    box = line.adjusted(150, 10, -150, -10)
    arm = 64
    pen = QPen(QColor("#ffc247"), 26)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    for x, y, dx, dy in ((box.left(), box.top(), 1, 1), (box.right(), box.top(), -1, 1),
                         (box.left(), box.bottom(), 1, -1), (box.right(), box.bottom(), -1, -1)):
        corner = QPainterPath()
        corner.moveTo(x + dx * arm, y)
        corner.lineTo(x, y)
        corner.lineTo(x, y + dy * arm)
        p.drawPath(corner)

    # The output: an amber subtitle line under the frame, what the OCR writes.
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#ffc247"))
    p.drawRoundedRect(QRectF(260, screen.bottom() + 70, SIZE - 520, 34), 17, 17)
    p.setBrush(QColor("#6b5426"))
    p.drawRoundedRect(QRectF(340, screen.bottom() + 134, SIZE - 680, 34), 17, 17)
    p.end()
    return img


if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not draw().save(str(OUT)):
        sys.exit(f"could not write {OUT}")
    print(OUT)
