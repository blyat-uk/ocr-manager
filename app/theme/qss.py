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


def _font_px(size: float) -> int:
    """A scaled font size (`tokens.FONT_SIZE_*`, or `tokens.pt(...)` for the
    one size that has no token) as the whole pixels a stylesheet wants.

    Qt's QSS parser truncates a fractional `font-size:...px` -- `11.875px`
    becomes 11 -- so every half-pixel token would quietly lose most of a
    pixel at scale (9.5 px at 1.25 is 11.875, i.e. +16%, not the +25% every
    other size gets). Rounding here keeps each one at its nearest whole
    pixel, matching what `apply_theme()` does for the default font."""
    return round(size)


# Hairlines are deliberately NOT scaled. Every `1px` border below is a
# separator or a state ring (panel edges, the focus ring, the queue row's
# selection border, a menu separator): its width is part of Qt's box model,
# so widening it would push the content in and can clip a label, while a
# 1 px rule still reads at any scale. Only lengths that carry size --
# padding, margins, radii, widths, heights -- go through `tokens.px()`.


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
    padding: {tokens.px(3)}px {tokens.px(6)}px;
}}

/* -- Button (.btn / .btn.ghost / .btn.primary / .btn.sm / .btn.on) --- */
QPushButton {{
    font-size: {_font_px(tokens.FONT_SIZE_BODY)}px;
    padding: {tokens.px(5)}px {tokens.px(11)}px;
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
/* Variant rules outrank plain :disabled (same specificity, later), so the
   disabled look is restated for them. */
QPushButton[variant="primary"]:disabled {{
    background-color: {tokens.PANEL2};
    border-color: {tokens.LINE};
    color: {tokens.DIM2};
}}
QPushButton[variant="ghost"]:disabled {{
    border-color: {tokens.LINE};
    color: {tokens.DIM2};
}}
QPushButton[small="true"] {{
    font-size: {_font_px(tokens.FONT_SIZE_BTN_SM)}px;
    padding: {tokens.px(3)}px {tokens.px(8)}px;
}}
QPushButton[toggled="true"] {{
    border-color: {tokens.ACC};
    color: {tokens.ACC};
}}
/* Keyboard focus. A widget with a stylesheet gets no default focus
   rectangle from Qt, so without this rule a focused button looks exactly
   like an unfocused one -- and the window's Space/T shortcuts step aside
   for whichever widget has the focus, so the user has to be able to see
   it. Same 1 px accent border the folder-settings editors use, last so it
   outranks the variant and toggled rules above.

   The #Ids are listed because an ID selector outranks a bare pseudo-class:
   without them a focused stage tab, log header or segment would keep its
   own rule's border. `border:` rather than `border-color:` because
   #SegmentItem's is `none` (its own `[on="true"]` state already adds one
   the same way).

   The primary variant fills with ACC, so an ACC ring on it is invisible;
   it gets the dark ink it already writes its label in. */
QPushButton:focus,
QPushButton#StageTab:focus,
QPushButton#LogHeader:focus,
QPushButton#SegmentItem:focus {{
    border: 1px solid {tokens.ACC};
}}
QPushButton[variant="primary"]:focus {{
    border: 1px solid {tokens.PRIMARY_TEXT};
}}

/* -- Chip (.chip, top-bar counters) ---------------------------------- */
QWidget#Chip {{
    background-color: {tokens.PANEL2};
    border: 1px solid {tokens.LINE2};
    border-radius: {tokens.RADIUS_CHIP_QT}px;
}}
QWidget#Chip QLabel {{
    font-size: {_font_px(tokens.pt(11))}px;
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
    padding: {tokens.px(1)}px {tokens.px(5)}px;
    font-size: {_font_px(tokens.FONT_SIZE_XS)}px;
}}
QLabel[badge="warn"] {{
    background-color: {tokens.BADGE_WARN_BG};
    color: {tokens.WARN};
    border-radius: {tokens.RADIUS_TAG}px;
    padding: {tokens.px(1)}px {tokens.px(5)}px;
    font-size: {_font_px(tokens.FONT_SIZE_XS)}px;
}}
QLabel[badge="good"] {{
    background-color: {tokens.BADGE_GOOD_BG};
    color: {tokens.OK};
    border-radius: {tokens.RADIUS_TAG}px;
    padding: {tokens.px(1)}px {tokens.px(5)}px;
    font-size: {_font_px(tokens.FONT_SIZE_XS)}px;
}}
QLabel[badge="bad"] {{
    background-color: {tokens.BADGE_BG};
    color: {tokens.BAD};
    border-radius: {tokens.RADIUS_TAG}px;
    padding: {tokens.px(1)}px {tokens.px(5)}px;
    font-size: {_font_px(tokens.FONT_SIZE_XS)}px;
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
    font-size: {_font_px(tokens.FONT_SIZE_BODY)}px;
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
    font-size: {_font_px(tokens.FONT_SIZE_SCOPE)}px;
    background: transparent;
}}

/* -- SegmentedControl (.rail-head .seg / .seg span / .seg span.on) --- */
QWidget#SegmentedControl {{
    background: transparent;
}}
QPushButton#SegmentItem {{
    font-size: {_font_px(tokens.FONT_SIZE_BTN_SM)}px;
    padding: {tokens.px(3)}px {tokens.px(8)}px;
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
""".strip("\n") + "\n" + _views_stylesheet() + "\n" + _folder_settings_stylesheet() + "\n" + _run_stylesheet() \
        + "\n" + _engine_setup_stylesheet()


def _views_stylesheet() -> str:
    """The window's views (plan 3B Task 3, `app/views/`), matched by object
    name: top bar, review queue, stage head, inspector, activity strip,
    banners and the open-folder empty state (workbench-hifi.html figure 1)."""
    return f"""
/* == Views (plan 3B Task 3) =========================================== */
QLabel {{
    background: transparent;
}}

/* -- Panels: top bar, rail, inspector, activity strip ------------------ */
QWidget#TopBar {{
    background-color: {tokens.PANEL};
    border-bottom: 1px solid {tokens.LINE};
}}
QWidget#ActivityStrip {{
    background-color: {tokens.PANEL};
    border-top: 1px solid {tokens.LINE};
}}
QWidget#Queue {{
    background-color: {tokens.PANEL};
    border-right: 1px solid {tokens.LINE};
}}
QWidget#Inspector {{
    background-color: {tokens.PANEL};
    border-left: 1px solid {tokens.LINE};
}}
QWidget#QueueList, QWidget#InspectorContent,
QScrollArea#QueueScroll, QScrollArea#InspectorScroll,
QScrollArea#QueueScroll > QWidget#qt_scrollarea_viewport,
QScrollArea#InspectorScroll > QWidget#qt_scrollarea_viewport {{
    background-color: {tokens.PANEL};
    border: none;
}}

/* -- Scroll bars and menus --------------------------------------------- */
QScrollBar:vertical {{
    background: transparent;
    width: {tokens.px(8)}px;
    margin: {tokens.px(2)}px {tokens.px(1)}px;
}}
QScrollBar::handle:vertical {{
    background-color: {tokens.LINE2};
    border-radius: {tokens.px(3)}px;
    min-height: {tokens.px(24)}px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QMenu {{
    background-color: {tokens.PANEL2};
    border: 1px solid {tokens.LINE2};
    border-radius: {tokens.RADIUS_BTN}px;
    color: {tokens.TXT};
    font-size: {_font_px(tokens.FONT_SIZE_BODY)}px;
    padding: {tokens.px(4)}px;
}}
QMenu::item {{
    padding: {tokens.px(5)}px {tokens.px(14)}px;
    border-radius: {tokens.RADIUS_TAG}px;
}}
QMenu::item:selected {{
    background-color: {tokens.ROW_SELECTED};
}}
QMenu::item:disabled {{
    color: {tokens.DIM2};
}}
QMenu::separator {{
    height: 1px;
    background-color: {tokens.LINE};
    margin: {tokens.px(4)}px {tokens.px(6)}px;
}}

/* -- Top bar (.topbar / .proj / .path) --------------------------------- */
QWidget#ChipBar {{
    background: transparent;
}}
QLabel#ProjectName {{
    font-size: {_font_px(tokens.FONT_SIZE_PROJ)}px;
    font-weight: {tokens.FONT_WEIGHT_PROJ};
    color: {tokens.TXT};
}}
QLabel#ProjectPath {{
    font-size: {_font_px(tokens.pt(11))}px;
    color: {tokens.DIM2};
}}

/* -- Banners (dependency warnings, open/save errors) ------------------- */
QWidget#Banner {{
    background-color: {tokens.BADGE_WARN_BG};
    border-bottom: 1px solid {tokens.KV_WARN_BORDER};
}}
QWidget#Banner[tone="bad"] {{
    background-color: {tokens.PANEL2};
    border-bottom: 1px solid {tokens.TAG_BAD_BORDER};
}}
QLabel#BannerTitle {{
    color: {tokens.WARN};
    font-weight: {tokens.FONT_WEIGHT_PROJ};
    font-size: {_font_px(tokens.FONT_SIZE_BODY)}px;
}}
QWidget#Banner[tone="bad"] QLabel#BannerTitle {{
    color: {tokens.BAD};
}}
QLabel#BannerText {{
    color: {tokens.TXT};
    font-size: {_font_px(tokens.FONT_SIZE_BTN_SM)}px;
}}

/* -- Review queue (.rail-head / .frow / .fname / .fsub) ---------------- */
QWidget#QueueHead {{
    background-color: {tokens.PANEL};
    border-bottom: 1px solid {tokens.LINE};
}}
QWidget#QueueRow {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: {tokens.RADIUS_ROW}px;
}}
QWidget#QueueRow[selected="false"]:hover {{
    background-color: {tokens.ROW_HOVER};
}}
QWidget#QueueRow[selected="true"] {{
    background-color: {tokens.ROW_SELECTED};
    border-color: {tokens.ACC_DIM};
}}
QLabel#QueueName {{
    font-size: {_font_px(tokens.FONT_SIZE_BODY)}px;
    font-weight: {tokens.FONT_WEIGHT_FNAME};
    color: {tokens.TXT};
}}
QWidget#QueueRow[skipped="true"] QLabel#QueueName {{
    color: {tokens.DIM2};
}}
QLabel#QueueSub {{
    font-size: {_font_px(tokens.FONT_SIZE_SM)}px;
    color: {tokens.DIM2};
}}
QLabel#QueueHint {{
    background-color: {tokens.PANEL};
    border-top: 1px solid {tokens.LINE};
    color: {tokens.DIM2};
    font-size: {_font_px(tokens.FONT_SIZE_SM)}px;
    padding: {tokens.px(9)}px {tokens.px(11)}px;
}}

/* -- Stage (.stage-head / .tab / .tab.on) ------------------------------ */
QWidget#Stage, QStackedWidget#StagePages {{
    background-color: {tokens.BG};
}}
QWidget#StageHead {{
    background-color: {tokens.BG};
    border-bottom: 1px solid {tokens.LINE};
}}
QPushButton#StageTab {{
    font-size: {_font_px(tokens.FONT_SIZE_BODY)}px;
    padding: {tokens.px(5)}px {tokens.px(11)}px;
    color: {tokens.DIM};
    background: transparent;
    border: 1px solid transparent;
    border-top-left-radius: {tokens.RADIUS_BTN}px;
    border-top-right-radius: {tokens.RADIUS_BTN}px;
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
}}
QPushButton#StageTab:hover {{
    color: {tokens.TXT};
}}
QPushButton#StageTab[on="true"] {{
    color: {tokens.TXT};
    background-color: {tokens.PANEL2};
    border: 1px solid {tokens.LINE2};
    border-bottom-color: {tokens.PANEL2};
}}

/* -- Inspector (.insp-head / .sec / .note / .ocrline / .footer) -------- */
QWidget#InspectorSection {{
    background-color: {tokens.PANEL};
    border-bottom: 1px solid {tokens.LINE};
}}
QWidget#InspectorFooter {{
    background-color: {tokens.PANEL};
    border-top: 1px solid {tokens.LINE};
}}
QLabel#InspectorScope {{
    color: {tokens.ACC};
    font-size: {_font_px(tokens.FONT_SIZE_SCOPE)}px;
}}
QLabel#InspectorFile {{
    color: {tokens.TXT};
    font-size: {_font_px(tokens.FONT_SIZE_MD)}px;
    font-weight: {tokens.FONT_WEIGHT_PROJ};
}}
QLabel#InspectorSub {{
    color: {tokens.DIM2};
    font-size: {_font_px(tokens.FONT_SIZE_BTN_SM)}px;
}}
QLabel#Note {{
    color: {tokens.DIM2};
    font-size: {_font_px(tokens.FONT_SIZE_SM)}px;
}}
QLabel#Note[tone="warn"] {{
    color: {tokens.WARN};
}}
QWidget#OcrLine {{
    background: transparent;
    border-bottom: 1px dashed {tokens.LINE};
}}
QLabel#OcrTime {{
    color: {tokens.DIM2};
    font-size: {_font_px(tokens.FONT_SIZE_SM)}px;
}}
QLabel#OcrText {{
    color: {tokens.TXT};
    font-size: {_font_px(tokens.pt(11))}px;
}}

/* -- Activity strip (.activity) ---------------------------------------- */
QWidget#ActivityStrip QLabel {{
    color: {tokens.DIM};
    font-size: {_font_px(tokens.FONT_SIZE_BTN_SM)}px;
}}
QWidget#ActivityStrip QLabel#ActivityRecent {{
    color: {tokens.DIM2};
}}

/* -- Placeholder tab pages --------------------------------------------- */
QWidget#PlaceholderPage {{
    background-color: {tokens.BG};
}}

/* -- Open-folder empty state ------------------------------------------- */
QWidget#OpenFolder {{
    background-color: {tokens.BG};
}}
QLabel#OpenTitle {{
    color: {tokens.TXT};
    font-size: {_font_px(tokens.FONT_SIZE_PROJ)}px;
    font-weight: {tokens.FONT_WEIGHT_PROJ};
}}
QLabel#OpenError {{
    color: {tokens.BAD};
    font-size: {_font_px(tokens.FONT_SIZE_BTN_SM)}px;
}}
""".strip("\n")


def _run_stylesheet() -> str:
    """The run mode (plan 3B Task 5): the top bar's start error, the Run view
    (tabs-hifi.html figure 4: `.rrow`, `.feed`, the live panel) and the logs
    window."""
    return f"""
/* == Run (plan 3B Task 5) ============================================== */
QLabel#StartError {{
    color: {tokens.BAD};
    font-size: {_font_px(tokens.FONT_SIZE_BTN_SM)}px;
}}

/* -- Run view: table (.rrow), footer, live panel (.feed) --------------- */
QWidget#RunView, QWidget#RunTable, QWidget#RunRows, QWidget#FeedLines,
QScrollArea#RunScroll, QScrollArea#FeedScroll,
QScrollArea#RunScroll > QWidget#qt_scrollarea_viewport,
QScrollArea#FeedScroll > QWidget#qt_scrollarea_viewport {{
    background-color: {tokens.BG};
    border: none;
}}
QWidget#RunRow, QWidget#RunHeader {{
    background: transparent;
    border-bottom: 1px solid {tokens.LINE};
}}
QWidget#RunRow[last="true"] {{
    border-bottom: none;
}}
QLabel#RunHeaderCell {{
    color: {tokens.DIM2};
    font-size: {_font_px(tokens.FONT_SIZE_SCOPE)}px;
}}
QLabel#RunFile, QLabel#RunPhase, QLabel#RunResult {{
    color: {tokens.TXT};
    font-size: {_font_px(tokens.FONT_SIZE_BODY)}px;
}}
QLabel#RunPhase[tone="ok"], QLabel#RunResult[tone="ok"] {{
    color: {tokens.OK};
}}
QLabel#RunPhase[tone="bad"] {{
    color: {tokens.BAD};
}}
QLabel#RunPhase[tone="dim"] {{
    color: {tokens.DIM2};
}}
QWidget#RunFooter {{
    background-color: {tokens.BG};
    border-top: 1px solid {tokens.LINE};
}}
QWidget#RunFooter QLabel {{
    color: {tokens.DIM2};
    font-size: {_font_px(tokens.FONT_SIZE_BTN_SM)}px;
}}
QWidget#LivePanel {{
    background-color: {tokens.BG};
    border-left: 1px solid {tokens.LINE};
}}
QLabel#LiveTitle {{
    color: {tokens.DIM2};
    font-size: {_font_px(tokens.FONT_SIZE_SCOPE)}px;
}}
QWidget#FeedLine {{
    background: transparent;
    border-bottom: 1px dashed {tokens.LINE};
}}
QLabel#FeedTime {{
    color: {tokens.DIM2};
    font-size: {_font_px(tokens.FONT_SIZE_SM)}px;
}}
QLabel#FeedText {{
    color: {tokens.TXT};
    font-size: {_font_px(tokens.pt(11))}px;
}}

/* -- Logs window ------------------------------------------------------- */
QWidget#LogsWindow, QWidget#LogsList, QScrollArea#LogsScroll,
QScrollArea#LogsScroll > QWidget#qt_scrollarea_viewport {{
    background-color: {tokens.BG};
    border: none;
}}
QPushButton#LogHeader {{
    text-align: left;
    font-size: {_font_px(tokens.FONT_SIZE_BODY)}px;
    padding: {tokens.px(5)}px {tokens.px(9)}px;
    color: {tokens.TXT};
    background-color: {tokens.PANEL2};
    border: 1px solid {tokens.LINE};
    border-radius: {tokens.RADIUS_BTN}px;
}}
QPushButton#LogHeader:checked {{
    border-color: {tokens.LINE2};
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
}}
QPlainTextEdit#LogBody {{
    background-color: {tokens.PANEL};
    color: {tokens.TXT};
    border: 1px solid {tokens.LINE};
    border-top: none;
    font-size: {_font_px(tokens.pt(11))}px;
    selection-background-color: {tokens.ACC};
    selection-color: {tokens.PRIMARY_TEXT};
}}
""".strip("\n")


def _folder_settings_stylesheet() -> str:
    """The Folder settings sheet (plan 3B Task 4, `app/views/folder_settings.py`,
    workbench-hifi.html figure 3): its own `.topbar`-style header, the 150 px
    vertical `.seg` nav, `.sec` blocks of `.kv` rows with an editor as the
    value, and the `.drawer` edge (`border-left:1px solid var(--acc-dim)`,
    ui-spec §2.1's "folder-settings drawer border")."""
    return f"""
/* == Folder settings sheet (plan 3B Task 4) ============================ */
QWidget#FolderSettings {{
    background-color: {tokens.BG};
    border-left: 1px solid {tokens.ACC_DIM};
}}
QWidget#FolderSettingsHead {{
    background-color: {tokens.PANEL};
    border-bottom: 1px solid {tokens.LINE};
}}
QLabel#FolderSettingsTitle {{
    font-size: {_font_px(tokens.FONT_SIZE_PROJ)}px;
    font-weight: {tokens.FONT_WEIGHT_PROJ};
    color: {tokens.TXT};
}}
QLabel#FolderSettingsScope {{
    font-size: {_font_px(tokens.pt(11))}px;
    color: {tokens.DIM2};
}}
QWidget#FolderSettingsNav {{
    background-color: {tokens.BG};
    border-right: 1px solid {tokens.LINE};
}}
QWidget#SegmentedControl[orientation="vertical"] QPushButton#SegmentItem {{
    text-align: left;
    border: 1px solid transparent;
}}
QWidget#SegmentedControl[orientation="vertical"] QPushButton#SegmentItem[on="true"] {{
    border: 1px solid {tokens.LINE2};
}}
QWidget#SegmentedControl[orientation="vertical"] QPushButton#SegmentItem:hover {{
    color: {tokens.TXT};
}}
QScrollArea#FolderSettingsScroll,
QScrollArea#FolderSettingsScroll > QWidget#qt_scrollarea_viewport,
QWidget#FolderSettingsContent {{
    background-color: {tokens.BG};
    border: none;
}}
QWidget#FolderSection {{
    background-color: {tokens.BG};
}}
QWidget#FolderSection[first="false"] {{
    border-top: 1px solid {tokens.LINE};
}}

/* -- Editors in the kv rows ------------------------------------------- */
QWidget#FolderSettings QAbstractSpinBox,
QWidget#FolderSettings QComboBox {{
    background-color: {tokens.BG};
    color: {tokens.TXT};
    border: 1px solid {tokens.LINE2};
    border-radius: {tokens.RADIUS_TAG}px;
    padding: {tokens.px(1)}px {tokens.px(6)}px;
    font-size: {_font_px(tokens.FONT_SIZE_BODY)}px;
    selection-background-color: {tokens.ACC_DIM};
    selection-color: {tokens.TXT};
}}
QWidget#FolderSettings QAbstractSpinBox:hover,
QWidget#FolderSettings QComboBox:hover {{
    border-color: {tokens.DIM2};
}}
QWidget#FolderSettings QAbstractSpinBox:focus,
QWidget#FolderSettings QComboBox:focus {{
    border-color: {tokens.ACC};
}}
QWidget#FolderSettings QAbstractSpinBox::up-button,
QWidget#FolderSettings QAbstractSpinBox::down-button {{
    width: 0;
    border: none;
}}
QWidget#FolderSettings QComboBox QAbstractItemView {{
    background-color: {tokens.PANEL2};
    color: {tokens.TXT};
    border: 1px solid {tokens.LINE2};
    selection-background-color: {tokens.ROW_SELECTED};
    selection-color: {tokens.TXT};
    outline: none;
}}
""".strip("\n")


def _engine_setup_stylesheet() -> str:
    """The first-run engine setup dialog (`app/views/engine_setup.py`): the
    folder settings sheet's header and footer surfaces, option buttons that
    read as selectable cards (the toggled one takes the accent border), a
    thin blue progress track like `.mini`, and a warn/bad message line."""
    return f"""
/* == Engine setup dialog ================================================ */
QDialog#EngineSetup {{
    background-color: {tokens.BG};
}}
QWidget#EngineSetupHead {{
    background-color: {tokens.PANEL};
    border-bottom: 1px solid {tokens.LINE};
}}
QWidget#EngineSetupFoot {{
    background-color: {tokens.PANEL};
    border-top: 1px solid {tokens.LINE};
}}
QLabel#EngineSetupTitle {{
    font-size: {_font_px(tokens.FONT_SIZE_PROJ)}px;
    font-weight: {tokens.FONT_WEIGHT_PROJ};
    color: {tokens.TXT};
    background: transparent;
}}
QLabel#EngineSetupScope,
QLabel#EngineSetupNote {{
    font-size: {_font_px(tokens.FONT_SIZE_SM)}px;
    color: {tokens.DIM2};
    background: transparent;
}}
QLabel#EngineSetupStep {{
    color: {tokens.TXT};
}}
QPushButton#EngineOption {{
    text-align: left;
    padding: {tokens.px(8)}px {tokens.px(12)}px;
    background-color: {tokens.PANEL2};
    border: 1px solid {tokens.LINE};
}}
QPushButton#EngineOption:hover {{
    border-color: {tokens.LINE2};
}}
QPushButton#EngineOption[toggled="true"] {{
    border-color: {tokens.ACC};
    color: {tokens.TXT};
}}
QPushButton#EngineOption:disabled {{
    color: {tokens.DIM2};
}}
QProgressBar#EngineProgress {{
    background-color: {tokens.TRACK_BG};
    border: none;
    border-radius: {tokens.RADIUS_XS}px;
}}
QProgressBar#EngineProgress::chunk {{
    background-color: {tokens.BLUE};
    border-radius: {tokens.RADIUS_XS}px;
}}
QLabel#EngineSetupMessage {{
    color: {tokens.WARN};
    background: transparent;
}}
QLabel#EngineSetupMessage[tone="bad"] {{
    color: {tokens.BAD};
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
