"""The Folder settings sheet (rulings B8/B9, ui-spec §1.2/§3.9,
workbench-hifi.html figure 3): everything that applies to the whole folder.

Presentation (B8)
    A child overlay of the window's central widget, anchored to its right
    edge between the top bar and the activity strip (the top and height of
    `anchor`, the window's body), `min(760 scaled, window width − the rail)`
    wide so the review queue stays visible -- the 760 is a mockup length and
    grows with `tokens.UI_SCALE`, the window's own width does not. It slides
    in, follows every resize, and is neither modal nor backed by a scrim.
    "✕" closes it; so does Esc, but only while focus is inside the sheet.

Layout
    Its own header row ("⚙ Folder settings", "applies to all N files in
    {folder}", "✕"); a 150 px (scaled) nav of the five sections; on the
    right one scrolling column of `.sec` blocks. A nav click scrolls its section to the
    top and scrolling moves the nav's highlight. The Labels section and its
    nav entry are hidden entirely while labels are off.

Editors and commits
    One kv row per setting with a one-line dim explanation under it (B9).
    Every change is committed on its own with
    `controller.update_folder(**{field: value})`, which also saves (debounced)
    and lets auto-pilot decide whether detections need to run; the sheet
    never re-runs detections itself. A toggle commits on click. A spin box
    commits on editingFinished, or after 400 ms without typing, so a
    half-typed number is never stored; the language commits on Enter, focus
    out or a pick from the list. Closing the sheet commits what is still
    pending. Only edits the user made are committed: syncing an editor from
    the model (`folder_changed`) never writes back, and an editor with an
    uncommitted edit is left alone while it syncs.

    Crop fractions are shown as percentages (`crop_vertical_padding` with one
    decimal, so its 0.3 % default survives) and stored as fractions.

Validation
    Turning off the last extraction toggle is refused by the controller
    (ValueError, the source of truth): the toggle snaps back and the inline
    warn line shows the controller's message.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QEasingCurve, QEvent, QObject, QPoint, QPropertyAnimation, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.theme import tokens
from app.views.folder_settings_fields import (
    EXTRACTION_FIELDS,
    LABELS_SECTION,
    MASK_KEY,
    MASK_NOTE,
    SECTIONS,
    DoubleSpinBox,
    Field,
    LanguageCombo,
    Section,
    SpinBox,
    language_code,
    language_label,
)
from app.widgets.base import Button, ElidedLabel, SegmentedControl, Toggle

TITLE = "⚙ Folder settings"
# Lengths below are the mockup's own pixels; every use goes through
# `tokens.px()`, so the sheet grows with the rest of the window.
MAX_WIDTH = 760
NAV_WIDTH = 150
CONTENT_MARGIN = 12
EDITOR_WIDTH = 84
LANGUAGE_WIDTH = 128
COMMIT_IDLE_MS = 400          # milliseconds, not pixels: never scaled
SLIDE_MS = 160
MASKS_TEXT = "{count} drawn · draw on the Crop tab"


def scope_text(count: int, folder: str) -> str:
    if count == 1:
        return f"applies to the 1 file in {folder}"
    return f"applies to all {count} files in {folder}"


# --------------------------------------------------------------------------
# The sheet
# --------------------------------------------------------------------------

class FolderSettingsSheet(QWidget):
    closed = pyqtSignal()

    def __init__(self, controller, parent: QWidget | None = None, *, anchor: QWidget | None = None):
        super().__init__(parent)
        self._controller = controller
        self._anchor = anchor
        self._fields: dict[str, Field] = {}
        self._editors: dict[str, QWidget] = {}
        self._timers: dict[str, QTimer] = {}
        self._pending: dict[str, object] = {}          # field -> the project its uncommitted edit belongs to
        self._sections: list[Section] = []
        self._scrolling = False
        self.setObjectName("FolderSettings")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.hide()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 0, 0, 0)                 # the stylesheet's 1 px accent edge
        outer.setSpacing(0)
        outer.addWidget(self._build_head())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_nav())
        body.addWidget(self._build_content(), 1)
        outer.addLayout(body, 1)

        close_action = QAction("Close folder settings", self)
        close_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        close_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        close_action.triggered.connect(lambda _checked=False: self.close_sheet())
        self.addAction(close_action)

        self._slide = QPropertyAnimation(self, b"pos", self)
        self._slide.setDuration(SLIDE_MS)
        self._slide.setEasingCurve(QEasingCurve.Type.OutCubic)

        for watched in (parent, anchor):
            if watched is not None:
                watched.installEventFilter(self)
        self.scroll.viewport().installEventFilter(self)

        controller.folder_changed.connect(self._sync)
        controller.files_changed.connect(self._sync_scope)
        controller.project_opened.connect(self._on_project_opened)
        controller.project_closed.connect(self._on_project_closed)
        self._sync()

    # --- building -------------------------------------------------------------------------

    def _build_head(self) -> QWidget:
        head = QWidget()
        head.setObjectName("FolderSettingsHead")
        head.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row = QHBoxLayout(head)
        row.setContentsMargins(tokens.px(14), tokens.px(9), tokens.px(14), tokens.px(9))
        row.setSpacing(tokens.px(14))
        self.title_label = QLabel(TITLE)
        self.title_label.setObjectName("FolderSettingsTitle")
        self.scope_label = ElidedLabel()
        self.scope_label.setObjectName("FolderSettingsScope")
        self.close_button = Button("✕", "ghost")
        self.close_button.setToolTip("Close (Esc)")
        self.close_button.clicked.connect(self.close_sheet)
        row.addWidget(self.title_label, 0, Qt.AlignmentFlag.AlignBaseline)
        row.addWidget(self.scope_label, 1, Qt.AlignmentFlag.AlignBaseline)
        row.addWidget(self.close_button)
        return head

    def _build_nav(self) -> QWidget:
        self.nav_panel = QWidget()
        self.nav_panel.setObjectName("FolderSettingsNav")
        self.nav_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.nav_panel.setFixedWidth(tokens.px(NAV_WIDTH))
        column = QVBoxLayout(self.nav_panel)
        # 8 px of padding, plus room for the stylesheet's 1 px rule on the right
        column.setContentsMargins(tokens.px(8), tokens.px(8), tokens.px(8) + 1, tokens.px(8))
        column.setSpacing(0)
        self.nav = SegmentedControl([title for title, _fields, _note in SECTIONS],
                                    orientation=Qt.Orientation.Vertical)
        self.nav.current_changed.connect(self._scroll_to)
        column.addWidget(self.nav)
        column.addStretch(1)
        return self.nav_panel

    def _build_content(self) -> QWidget:
        self.scroll = QScrollArea()
        self.scroll.setObjectName("FolderSettingsScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        content = QWidget()
        content.setObjectName("FolderSettingsContent")
        column = QVBoxLayout(content)
        margin = tokens.px(CONTENT_MARGIN)
        column.setContentsMargins(margin, margin, margin, margin)
        column.setSpacing(0)
        for index, (title, fields, note) in enumerate(SECTIONS):
            section = Section(title, first=index == 0)
            for field in fields:
                self._fields[field.name] = field
                editor = self._make_editor(field)
                self._editors[field.name] = editor
                section.add_row(field.key, editor, field.note)
            if title == LABELS_SECTION:
                self.mask_label = QLabel()
                self.mask_label.setProperty("kvRole", "value")
                section.add_row(MASK_KEY, self.mask_label, MASK_NOTE)
            if index == 0:
                self.warning_label = QLabel()
                self.warning_label.setObjectName("Note")
                self.warning_label.setProperty("tone", "warn")
                self.warning_label.hide()
                section.body.addWidget(self.warning_label)
                section.body.addSpacing(tokens.px(4))
            if note:
                section.add_note(note)
            self._sections.append(section)
            column.addWidget(section)
        self._filler = QWidget()
        self._filler.setFixedHeight(0)
        column.addWidget(self._filler)
        column.addStretch(1)
        self.scroll.setWidget(content)
        self.scroll.verticalScrollBar().valueChanged.connect(self._follow_scroll)
        return self.scroll

    def _make_editor(self, field: Field) -> QWidget:
        if field.kind == "toggle":
            toggle = Toggle()
            toggle.clicked.connect(lambda checked, f=field: self._commit_toggle(f, checked))
            return toggle
        if field.kind == "language":
            combo = LanguageCombo()
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            combo.setFixedWidth(tokens.px(LANGUAGE_WIDTH))
            combo.lineEdit().textEdited.connect(lambda _text, name=field.name: self._mark_pending(name))
            combo.lineEdit().editingFinished.connect(self._commit_language_text)
            combo.activated.connect(self._on_language_activated)
            return combo
        spin = DoubleSpinBox() if field.decimals else SpinBox()
        if field.decimals:
            spin.setDecimals(field.decimals)
        spin.setRange(field.minimum, field.maximum)
        spin.setSingleStep(field.step)
        spin.setSuffix(field.suffix)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        spin.setFixedWidth(tokens.px(EDITOR_WIDTH))
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(COMMIT_IDLE_MS)
        timer.timeout.connect(lambda name=field.name: self._commit(name))
        self._timers[field.name] = timer
        spin.valueChanged.connect(lambda _value, name=field.name: self._on_typed(name))
        spin.editingFinished.connect(lambda name=field.name: self._commit(name))
        return spin

    # --- public ---------------------------------------------------------------------------

    def open(self) -> None:
        """Show the sheet (sliding in when it was closed) and move focus into it."""
        if self._controller.project is None:
            return
        self._sync()
        target = self.target_geometry()
        if self.isHidden():
            self.warning_label.hide()
            self._slide.stop()
            if SLIDE_MS > 0:
                self.setGeometry(target.translated(target.width(), 0))
                self._slide.setStartValue(self.pos())
                self._slide.setEndValue(target.topLeft())
            else:
                self.setGeometry(target)
            self.show()
            if SLIDE_MS > 0:
                self._slide.start()
        self.raise_()
        self._update_filler()
        self.nav.item(self.nav.current()).setFocus(Qt.FocusReason.OtherFocusReason)

    def close_sheet(self) -> None:
        """Commit pending edits and hide; `closed` is emitted once per close."""
        if self.isHidden():
            return
        self._flush_pending()
        self._slide.stop()
        self.hide()
        self.closed.emit()

    def target_geometry(self) -> QRect:
        """Right edge of the host, the anchor's top and height,
        min(the scaled 760, window width − the rail) wide: the sheet's own
        width scales, the window it sits in does not."""
        host = self.parentWidget()
        if host is None:
            return self.geometry()
        width = max(0, min(tokens.px(MAX_WIDTH), self.window().width() - tokens.RAIL_WIDTH))
        if self._anchor is not None:
            top, height = self._anchor.mapTo(host, QPoint(0, 0)).y(), self._anchor.height()
        else:
            top, height = 0, host.height()
        return QRect(host.width() - width, top, width, height)

    def contains_focus(self, widget: QWidget | None = None) -> bool:
        widget = QApplication.focusWidget() if widget is None else widget
        return widget is not None and (widget is self or self.isAncestorOf(widget))

    def focusNextPrevChild(self, forward: bool) -> bool:
        """Tab and Shift+Tab cycle inside the sheet: never onto the inspector
        buttons it covers (Space would press one unseen)."""
        chain = self.tab_chain()
        if not chain:
            return super().focusNextPrevChild(forward)
        current = QApplication.focusWidget()
        index = -1
        for position, widget in enumerate(chain):
            if current is not None and (widget is current or widget.isAncestorOf(current)):
                index = position
                break
        if index < 0:
            next_index = 0 if forward else len(chain) - 1
        else:
            next_index = (index + (1 if forward else -1)) % len(chain)
        reason = Qt.FocusReason.TabFocusReason if forward else Qt.FocusReason.BacktabFocusReason
        chain[next_index].setFocus(reason)
        return True

    def tab_chain(self) -> list[QWidget]:
        """The sheet's visible, enabled, Tab-focusable widgets in focus order."""
        widgets = []
        widget = self.nextInFocusChain()
        while widget is not None and widget is not self:
            if (self.isAncestorOf(widget) and widget.isVisible() and widget.isEnabled()
                    and widget.focusProxy() is None
                    and widget.focusPolicy().value & Qt.FocusPolicy.TabFocus.value):
                widgets.append(widget)
            widget = widget.nextInFocusChain()
        return widgets

    def fields(self) -> list[str]:
        return list(self._editors)

    def editor(self, field: str) -> QWidget:
        return self._editors[field]

    def note(self, field: str) -> str:
        return self._fields[field].note

    def section(self, title: str) -> Section:
        return next(section for section in self._sections if section.title == title)

    def row_keys(self, title: str) -> list[str]:
        return list(self.section(title).keys)

    def section_note(self, title: str) -> str:
        return self.section(title).note_text

    # --- placement ------------------------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kind = event.type()
        if kind in (QEvent.Type.Resize, QEvent.Type.Move) and watched is not self.scroll.viewport():
            if not self.isHidden():
                self._place()
        elif kind == QEvent.Type.Resize and watched is self.scroll.viewport():
            self._update_filler()
        return super().eventFilter(watched, event)

    def _place(self) -> None:
        self._slide.stop()
        self.setGeometry(self.target_geometry())

    # --- nav ------------------------------------------------------------------------------

    def _section_top(self, section: Section) -> int:
        return max(0, section.y() - tokens.px(CONTENT_MARGIN))

    def _visible_sections(self) -> list[tuple[int, Section]]:
        return [(index, section) for index, section in enumerate(self._sections) if not section.isHidden()]

    def _scroll_to(self, index: int) -> None:
        self._scrolling = True
        try:
            self.scroll.verticalScrollBar().setValue(self._section_top(self._sections[index]))
        finally:
            self._scrolling = False

    def _follow_scroll(self, *_args) -> None:
        if self._scrolling:
            return
        value = self.scroll.verticalScrollBar().value()
        visible = self._visible_sections()
        if not visible:
            return
        current = visible[0][0]
        for index, section in visible:
            if self._section_top(section) <= value + 1:
                current = index
        self.nav.set_current(current)

    def _update_filler(self) -> None:
        """Room below the last section, so every section can scroll to the top.

        The last section's REAL height, not its size hint: a word-wrapped
        note ("These tune detections that start after a change…") hints at
        the height it would take on its own and is laid out a line shorter,
        which left the scroll range a line short of the last section's top --
        so the nav could never highlight it. `activate()` makes sure the
        column has been laid out at its current width before it is measured."""
        visible = self._visible_sections()
        if not visible:
            return
        content = self.scroll.widget()
        if content.layout() is not None:
            content.layout().activate()
        last = visible[-1][1]
        spare = self.scroll.viewport().height() - last.height() - 2 * tokens.px(CONTENT_MARGIN)
        self._filler.setFixedHeight(max(0, spare))

    # --- commits --------------------------------------------------------------------------

    def _mark_pending(self, name: str) -> None:
        self._pending[name] = self._controller.project

    def _on_typed(self, name: str) -> None:
        self._mark_pending(name)
        self._timers[name].start()

    def _commit(self, name: str) -> None:
        self._timers[name].stop()
        if name not in self._pending:
            return
        project = self._pending.pop(name)
        if project is None or project is not self._controller.project:
            return
        field = self._fields[name]
        value = field.to_model(self._editors[name].value())
        if getattr(project.folder, name) != value:
            self._controller.update_folder(**{name: value})
        else:
            self._sync()

    def _commit_toggle(self, field: Field, checked: bool) -> None:
        if self._controller.project is None:
            return
        try:
            self._controller.update_folder(**{field.name: checked})
        except ValueError as exc:                       # both extractions off: the controller refuses
            self._editors[field.name].setChecked(not checked)
            self.warning_label.setText(str(exc))
            self.warning_label.show()
            return
        if field.name in EXTRACTION_FIELDS:
            self.warning_label.hide()

    def _commit_language_text(self) -> None:
        if "ocr_lang" not in self._pending:
            return
        project = self._pending.pop("ocr_lang")
        if project is not self._controller.project:
            return
        self._store_language(language_code(self._editors["ocr_lang"].currentText()))

    def _on_language_activated(self, index: int) -> None:
        self._pending.pop("ocr_lang", None)
        combo = self._editors["ocr_lang"]
        self._store_language(combo.itemData(index) or language_code(combo.itemText(index)))

    def _store_language(self, code: str) -> None:
        project = self._controller.project
        if project is None:
            return
        if code and code != project.folder.ocr_lang:
            self._controller.update_folder(ocr_lang=code)
        else:
            self._sync_language(project.folder.ocr_lang)

    def _flush_pending(self) -> None:
        for name in list(self._pending):
            if name == "ocr_lang":
                self._commit_language_text()
            else:
                self._commit(name)

    # --- syncing from the model -----------------------------------------------------------

    def _on_project_opened(self, _path: str) -> None:
        self._drop_pending()
        self._sync()

    def _on_project_closed(self) -> None:
        self._drop_pending()
        self.close_sheet()

    def _drop_pending(self) -> None:
        for timer in self._timers.values():
            timer.stop()
        self._pending.clear()

    def _sync(self) -> None:
        project = self._controller.project
        if project is None:
            return
        folder = project.folder
        for name, field in self._fields.items():
            if name in self._pending:
                continue                                 # the user is still editing it
            editor = self._editors[name]
            value = getattr(folder, name)
            if field.kind == "toggle":
                editor.setChecked(bool(value))
            elif field.kind == "language":
                self._sync_language(value)
            else:
                shown = field.to_editor(value)
                editor.blockSignals(True)
                suffix = field.suffix_for(shown)
                if editor.suffix() != suffix:
                    editor.setSuffix(suffix)
                editor.setValue(shown)
                editor.blockSignals(False)
        labels_on = bool(folder.labels_enabled)
        labels_index = [title for title, _fields, _note in SECTIONS].index(LABELS_SECTION)
        self._sections[labels_index].setVisible(labels_on)
        self.nav.set_item_visible(labels_index, labels_on)
        self.mask_label.setText(MASKS_TEXT.format(count=len(folder.label_mask_crops)))
        self._sync_scope()
        self._update_filler()
        self._follow_scroll()

    def _sync_language(self, code: str) -> None:
        combo = self._editors["ocr_lang"]
        codes = ["ch"] if code == "ch" else ["ch", code]
        if [combo.itemData(i) for i in range(combo.count())] == codes and combo.currentText() == language_label(code):
            return                                           # unchanged: keep the cursor where the user left it
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(language_label("ch"), "ch")
        if code != "ch":
            combo.addItem(language_label(code), code)
        combo.setCurrentIndex(0 if code == "ch" else 1)
        combo.blockSignals(False)

    def _sync_scope(self, *_args) -> None:
        project = self._controller.project
        if project is None:
            self.scope_label.set_full_text("")
            return
        folder = os.path.basename(project.path) or project.path
        self.scope_label.set_full_text(scope_text(len(self._controller.names()), folder))
