"""The generated Qt stylesheet for the new window's base widgets (Task 1),
and `apply_theme()`, which installs it on a `QApplication` alongside a
matching dark `QPalette` and the theme's default font.

Selectors are dynamic-property based, e.g. `QPushButton[variant="primary"]`,
`QLabel[badge="warn"]` -- `app/widgets/base.py`'s widgets set these
properties in their constructors and setters. Custom `QWidget` subclasses
that need their own background/border (Chip, KvRow, SectionHeader,
SegmentedControl's segments) are matched by `objectName` (Qt resolves
`#name` selectors from `QObject.objectName()`, which every widget in this
module sets explicitly) rather than by their Python class name, since a
plain `QWidget(objectName=...)` selector is unambiguous however the class
is later refactored.

`ConfBar`, `MiniProgress` and `Dot` are not styled here: their fill is a
fraction of a fixed-size track, which static QSS cannot express per
instance, so they paint themselves directly from `tokens` in
`app/widgets/base.py`. They still carry a `tone` dynamic property, for
introspection/testing, even though nothing here selects on it.
"""
from __future__ import annotations

from app.theme import tokens


def build_stylesheet() -> str:
    """The full QSS for the base widgets in `app/widgets/base.py`. A plain
    string (no QApplication needed): every rule is built directly from
    `app.theme.tokens`, so a stylesheet and a token module can never
    silently disagree."""
    return f"""
/* -- Base surfaces (apply_theme() also sets a matching QPalette, for the
   handful of native widgets/dialogs QSS does not reach) -------------- */
QWidget {{
    background-color: {tokens.BG};
    color: {tokens.TXT};
    selection-background-color: {tokens.ACC};
    selection-color: {tokens.PRIMARY_TEXT};
}}

QToolTip {{
    background-color: {tokens.PANEL2};
    color: {tokens.TXT};
    border: 1px solid {tokens.LINE2};
    padding: 3px 6px;
}}

/* -- Button (.btn / .btn.ghost / .btn.primary / .btn.sm / .btn.on) --- */
QPushButton {{
    font-size: {tokens.FONT_SIZE_BODY}px;
    padding: 5px 11px;
    border-radius: {tokens.RADIUS_BTN}px;
    border: 1px solid {tokens.LINE2};
    background-color: {tokens.PANEL2};
    color: {tokens.TXT};
}}
QPushButton:hover {{
    background-color: {tokens.ROW_HOVER};
}}
QPushButton:disabled {{
    color: {tokens.DIM2};
    border-color: {tokens.LINE};
}}
QPushButton[variant="ghost"] {{
    background-color: transparent;
    border-color: {tokens.LINE2};
}}
QPushButton[variant="ghost"]:hover {{
    background-color: {tokens.PANEL2};
}}
QPushButton[variant="primary"] {{
    background-color: {tokens.ACC};
    border-color: {tokens.ACC};
    color: {tokens.PRIMARY_TEXT};
    font-weight: {tokens.FONT_WEIGHT_PRIMARY};
}}
QPushButton[variant="primary"]:hover {{
    background-color: {tokens.ACC};
}}
QPushButton[small="true"] {{
    font-size: {tokens.FONT_SIZE_BTN_SM}px;
    padding: 3px 8px;
}}
QPushButton[toggled="true"] {{
    border-color: {tokens.ACC};
    color: {tokens.ACC};
}}

/* -- Chip (.chip, top-bar counters) ---------------------------------- */
QWidget#Chip {{
    background-color: {tokens.PANEL2};
    border: 1px solid {tokens.LINE2};
    border-radius: {tokens.RADIUS_CHIP}px;
}}
QWidget#Chip QLabel {{
    font-size: 11px;
    color: {tokens.DIM};
    background: transparent;
}}
QWidget#Chip QLabel[chipRole="count"] {{
    color: {tokens.TXT};
    font-weight: 600;
}}

/* -- Badge (.badge / .badge.w / .badge.g / bad) ---------------------- */
QLabel[badge="default"] {{
    background-color: {tokens.BADGE_BG};
    color: {tokens.DIM};
    border-radius: {tokens.RADIUS_TAG}px;
    padding: 1px 5px;
    font-size: {tokens.FONT_SIZE_XS}px;
}}
QLabel[badge="warn"] {{
    background-color: {tokens.BADGE_WARN_BG};
    color: {tokens.WARN};
    border-radius: {tokens.RADIUS_TAG}px;
    padding: 1px 5px;
    font-size: {tokens.FONT_SIZE_XS}px;
}}
QLabel[badge="good"] {{
    background-color: {tokens.BADGE_GOOD_BG};
    color: {tokens.OK};
    border-radius: {tokens.RADIUS_TAG}px;
    padding: 1px 5px;
    font-size: {tokens.FONT_SIZE_XS}px;
}}
QLabel[badge="bad"] {{
    background-color: {tokens.BADGE_BG};
    color: {tokens.BAD};
    border-radius: {tokens.RADIUS_TAG}px;
    padding: 1px 5px;
    font-size: {tokens.FONT_SIZE_XS}px;
}}

/* -- KvRow (.kv) ------------------------------------------------------ */
QWidget#KvRow {{
    background-color: {tokens.PANEL2};
    border: 1px solid {tokens.LINE};
    border-radius: {tokens.RADIUS_BTN}px;
}}
QWidget#KvRow[tone="warn"] {{
    border-color: {tokens.KV_WARN_BORDER};
}}
QWidget#KvRow[tone="bad"] {{
    border-color: {tokens.TAG_BAD_BORDER};
}}
QWidget#KvRow QLabel {{
    background: transparent;
    font-size: {tokens.FONT_SIZE_BODY}px;
}}
QWidget#KvRow QLabel[kvRole="key"] {{
    color: {tokens.DIM};
}}
QWidget#KvRow QLabel[kvRole="value"] {{
    color: {tokens.TXT};
}}
QWidget#KvRow QLabel[kvRole="value"][tone="ok"] {{
    color: {tokens.OK};
}}
QWidget#KvRow QLabel[kvRole="value"][tone="warn"] {{
    color: {tokens.WARN};
}}
QWidget#KvRow QLabel[kvRole="value"][tone="bad"] {{
    color: {tokens.BAD};
}}
QWidget#KvRow QLabel[kvRole="value"][tone="acc"] {{
    color: {tokens.ACC};
}}

/* -- SectionHeader (.sec-h) -------------------------------------------- */
QWidget#SectionHeader QLabel[sectionRole="title"] {{
    color: {tokens.DIM2};
    font-size: {tokens.FONT_SIZE_SCOPE}px;
    background: transparent;
}}

/* -- SegmentedControl (.rail-head .seg / .seg span / .seg span.on) --- */
QWidget#SegmentedControl {{
    background: transparent;
}}
QPushButton#SegmentItem {{
    font-size: {tokens.FONT_SIZE_BTN_SM}px;
    padding: 3px 8px;
    border-radius: {tokens.RADIUS_SEG}px;
    color: {tokens.DIM};
    background: transparent;
    border: none;
}}
QPushButton#SegmentItem[on="true"] {{
    background-color: {tokens.PANEL2};
    color: {tokens.TXT};
    border: 1px solid {tokens.LINE2};
}}
""".strip("\n")


def apply_theme(app: "QApplication") -> None:  # noqa: F821 -- PyQt6 import kept local
    """Install the dark theme on `app`: the generated stylesheet
    (`build_stylesheet()`), the theme's default font (`tokens.FONT_STACK`,
    ruling C9's CJK-capable fallback), and a dark `QPalette` built from
    `tokens` so native dialogs/widgets QSS does not reach (file pickers,
    message boxes, ...) still read as dark rather than falling back to the
    platform's light default.
    """
    from PyQt6.QtGui import QColor, QFont, QPalette

    app.setStyleSheet(build_stylesheet())

    font = QFont()
    font.setFamilies(list(tokens.FONT_STACK))
    font.setPixelSize(round(tokens.FONT_SIZE_BODY))
    app.setFont(font)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(tokens.BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(tokens.TXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(tokens.PANEL))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(tokens.PANEL2))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(tokens.PANEL2))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(tokens.TXT))
    palette.setColor(QPalette.ColorRole.Text, QColor(tokens.TXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(tokens.PANEL2))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(tokens.TXT))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(tokens.BAD))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(tokens.ACC))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(tokens.PRIMARY_TEXT))
    palette.setColor(QPalette.ColorRole.Link, QColor(tokens.BLUE))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(tokens.DIM2))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(tokens.DIM2))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(tokens.DIM2))
    app.setPalette(palette)
