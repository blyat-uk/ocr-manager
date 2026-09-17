"""The review queue (`.rail`, ui-spec §3.2): a filter, one row per video
file, and the keyboard hint.

- Filter "All N" / "Needs you N" / "Reviewed N": the counts are
  `controller.counts()`, and a row belongs to "Needs you" / "Reviewed" by the
  same rule -- the badge it shows outside a run ("check …" / "reviewed").
- Row: thumbnail with the crop box at its true position, file name
  (elided), duration and badge (`state_text.badge_for`, ruling B10).
- Keys while the queue has focus: ↑/↓ move the selection over the visible
  rows. Space (toggle reviewed) and T (test OCR) are window shortcuts
  (`MainWindow.review_action` / `proof_action`), so they also work from the
  stage and the inspector.
- Right-click: the ruling B11 menu (`context_menu`).

Badges depend on which detectors are running, which changes on job
"started" events that emit only `activity_changed`: rows refresh on that
signal as well as on `file_changed` / `files_changed`.
"""
from __future__ import annotations

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMenu, QScrollArea, QVBoxLayout, QWidget

from app.run_snapshot import FAILED, QUEUED, RUNNING
from app.state_text import badge_for, can_mark_reviewed, format_duration, is_pending, is_reviewed
from app.theme import tokens
from app.views.deferred import Deferred
from app.views.thumbnail import Thumbnail
from app.widgets.base import Badge, ElidedLabel, SegmentedControl, repolish

FILTER_ALL, FILTER_NEEDS_YOU, FILTER_REVIEWED = 0, 1, 2
# Non-breaking spaces inside each hint, so the 246 px rail wraps between hints, never inside one.
HINT_HTML = (f'↑&nbsp;↓&nbsp;move · <b style="color:{tokens.DIM}">Space</b>&nbsp;mark&nbsp;reviewed · '
             f'<b style="color:{tokens.DIM}">T</b>&nbsp;test&nbsp;OCR')
RUN_BADGE_STATES = frozenset({QUEUED, RUNNING, FAILED})     # a run's transient badges (ruling B10)


class QueueRow(QWidget):
    """One file (`.frow`)."""

    clicked = pyqtSignal(str)
    context_requested = pyqtSignal(str, QPoint)

    def __init__(self, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name
        self.setObjectName("QueueRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setProperty("selected", False)
        self.setProperty("skipped", False)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(9)
        self.thumb = Thumbnail()
        layout.addWidget(self.thumb, 0, Qt.AlignmentFlag.AlignVCenter)
        meta = QVBoxLayout()
        meta.setContentsMargins(0, 0, 0, 0)
        meta.setSpacing(2)
        self.name_label = ElidedLabel(name)
        self.name_label.setObjectName("QueueName")
        meta.addWidget(self.name_label)
        sub = QHBoxLayout()
        sub.setContentsMargins(0, 0, 0, 0)
        sub.setSpacing(7)
        self.duration_label = QLabel()
        self.duration_label.setObjectName("QueueSub")
        self.badge = Badge()
        sub.addWidget(self.duration_label)
        sub.addWidget(self.badge)
        sub.addStretch(1)
        meta.addLayout(sub)
        layout.addLayout(meta, 1)

    def set_selected(self, selected: bool) -> None:
        if self.property("selected") != selected:
            self.setProperty("selected", selected)
            repolish(self)

    def update_from(self, entry, badge: tuple[str, str], thumbnail) -> None:
        text, tone = badge
        if (self.badge.text(), self.badge.property("badge")) != (text, tone):
            self.badge.set_state(text, tone)
        media = entry.media
        self.duration_label.setText(format_duration(media.duration) if media.duration > 0 else "")
        self.duration_label.setVisible(media.duration > 0)
        if self.property("skipped") != entry.skipped:
            self.setProperty("skipped", entry.skipped)
            repolish(self)
            repolish(self.name_label)
        crop = entry.crop
        frame = (media.width, media.height) if media.width > 0 and media.height > 0 else None
        self.thumb.set_state(thumbnail, frame, None if crop is None else (crop.x, crop.y, crop.width, crop.height),
                             pending=is_pending(entry) and not entry.skipped,
                             dimmed=entry.skipped)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.name)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event) -> None:
        self.context_requested.emit(self.name, event.globalPos())


class QueueView(QWidget):
    selection_changed = pyqtSignal(object)      # str | None
    proof_requested = pyqtSignal(str)
    logs_requested = pyqtSignal(str)

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__(parent)
        self._controller = controller
        self._rows: dict[str, QueueRow] = {}
        self._selected: str | None = None
        self._filter_later = Deferred(self._refresh_filter, self)
        self.setObjectName("Queue")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(tokens.RAIL_WIDTH)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 1, 0)        # the 1 px right border
        layout.setSpacing(0)

        head = QWidget()
        head.setObjectName("QueueHead")
        head.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        head_layout = QHBoxLayout(head)
        head_layout.setContentsMargins(11, 9, 11, 7)
        self.filter = SegmentedControl(["All 0", "Needs you 0", "Reviewed 0"])
        for button in self.filter.findChildren(QWidget):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.filter.current_changed.connect(self._on_filter_changed)
        head_layout.addWidget(self.filter)
        head_layout.addStretch(1)
        layout.addWidget(head)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("QueueScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list = QWidget()
        self._list.setObjectName("QueueList")
        self._list_layout = QVBoxLayout(self._list)
        self._list_layout.setContentsMargins(6, 6, 6, 6)
        self._list_layout.setSpacing(4)
        self._list_layout.addStretch(1)
        self._scroll.setWidget(self._list)
        layout.addWidget(self._scroll, 1)

        self.hint_label = QLabel(HINT_HTML)
        self.hint_label.setObjectName("QueueHint")
        self.hint_label.setTextFormat(Qt.TextFormat.RichText)
        self.hint_label.setWordWrap(True)                # two lines at 246 px rather than clipped
        layout.addWidget(self.hint_label)

        controller.project_opened.connect(self.rebuild)
        controller.project_closed.connect(self.rebuild)
        controller.files_changed.connect(self.rebuild)
        controller.file_changed.connect(self._on_file_changed)
        controller.thumbnail_ready.connect(self._on_file_changed)
        for signal in (controller.activity_changed, controller.folder_changed, controller.run_changed):
            signal.connect(self.refresh)

    # --- reading --------------------------------------------------------------------

    def row(self, name: str) -> QueueRow:
        return self._rows[name]

    def selected(self) -> str | None:
        return self._selected

    def names(self) -> list[str]:
        return list(self._rows)

    def visible_names(self) -> list[str]:
        return [name for name in self._rows if self._in_filter(name)]

    # --- building and refreshing --------------------------------------------------------

    def rebuild(self, *_args) -> None:
        names = self._controller.names()
        for name in [name for name in self._rows if name not in names]:
            row = self._rows.pop(name)
            self._list_layout.removeWidget(row)
            row.deleteLater()
        for name in names:
            if name not in self._rows:
                row = QueueRow(name, self._list)
                row.clicked.connect(self._on_row_clicked)
                row.context_requested.connect(self._on_row_context)
                self._rows[name] = row
        for index, name in enumerate(names):             # rows in name order, the stretch last
            self._list_layout.insertWidget(index, self._rows[name])
        self._rows = {name: self._rows[name] for name in names}
        self.refresh()
        if self._selected not in self._rows:
            visible = self.visible_names()
            self.select(visible[0] if visible else (names[0] if names else None))

    def refresh(self, *_args) -> None:
        """Every row now; the filter counts and row visibility on the next
        event-loop turn (coalesced)."""
        if self._controller.project is None:
            return
        for name in self._rows:
            self._refresh_row(name)
        self._filter_later.schedule()

    def _on_file_changed(self, name: str) -> None:
        if name in self._rows and self._controller.project is not None:
            self._refresh_row(name)
            self._filter_later.schedule()                # a burst of changes counts once

    def _refresh_row(self, name: str) -> None:
        controller = self._controller
        entry = controller.entry(name)
        badge = badge_for(entry, running_detectors=controller.running_detectors(name),
                          done=controller.is_done(name), run_state=self._run_state(name))
        self._rows[name].update_from(entry, badge, controller.thumbnail(name))

    def _run_state(self, name: str) -> str | None:
        snapshot = self._controller.run_snapshot()
        if snapshot is None or snapshot.finished:
            return None
        row = snapshot.row(name)
        return row.state if row is not None and row.state in RUN_BADGE_STATES else None

    def _refresh_filter(self) -> None:
        self._filter_later.cancel()
        if self._controller.project is None:
            return
        counts = self._controller.counts()
        self.filter.set_texts([f"All {len(self._rows)}", f"Needs you {counts['needs_you']}",
                               f"Reviewed {counts['reviewed']}"])
        for name, row in self._rows.items():
            row.setVisible(self._in_filter(name))

    def _in_filter(self, name: str) -> bool:
        current = self.filter.current()
        if current == FILTER_ALL:
            return True
        controller = self._controller
        text, tone = badge_for(controller.entry(name), running_detectors=controller.running_detectors(name),
                               done=controller.is_done(name), run_state=None)
        if current == FILTER_NEEDS_YOU:
            return tone == "warn"
        return text == "reviewed"

    # --- selection ----------------------------------------------------------------------

    def select(self, name: str | None) -> None:
        if name is not None and name not in self._rows:
            return
        previous = self._selected
        self._selected = name
        for row_name, row in self._rows.items():
            row.set_selected(row_name == name)
        if name is not None:
            self._scroll.ensureWidgetVisible(self._rows[name], 0, 6)
        if name != previous:
            self.selection_changed.emit(name)

    def set_filter(self, index: int) -> None:
        self.filter.set_current(index)
        self._on_filter_changed(index)

    def _on_filter_changed(self, _index: int) -> None:
        self._refresh_filter()
        visible = self.visible_names()
        if self._selected not in visible and visible:
            self.select(visible[0])

    def move_selection(self, step: int) -> None:
        names = list(self._rows)
        visible = set(self.visible_names())
        if not visible:
            return
        if self._selected in names:
            order = names[names.index(self._selected) + 1:] if step > 0 else \
                list(reversed(names[:names.index(self._selected)]))
        else:
            order = names if step > 0 else list(reversed(names))
        target = next((name for name in order if name in visible), None)
        if target is not None:
            self.select(target)

    def _on_row_clicked(self, name: str) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.select(name)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier:
            super().keyPressEvent(event)
            return
        if key == Qt.Key.Key_Down:
            self.move_selection(1)
        elif key == Qt.Key.Key_Up:
            self.move_selection(-1)
        else:
            super().keyPressEvent(event)

    # --- commands and the context menu (ruling B11) ------------------------------------

    def _toggle_reviewed(self, name: str) -> None:
        entry = self._controller.entry(name)
        if can_mark_reviewed(entry):
            self._controller.mark_reviewed(name, not is_reviewed(entry))

    def _toggle_skipped(self, name: str) -> None:
        self._controller.set_skipped(name, not self._controller.entry(name).skipped)

    def context_menu(self, name: str) -> QMenu:
        """The row's menu, built fresh so its texts and enabled states are
        current. The caller shows it (or triggers its actions)."""
        controller = self._controller
        entry = controller.entry(name)
        menu = QMenu(self)

        def add(text: str, slot, enabled: bool = True) -> QAction:
            action = menu.addAction(text)
            action.setEnabled(enabled)
            # The file may vanish while the menu is open (the folder watcher runs meanwhile).
            action.triggered.connect(lambda _checked=False: slot() if name in controller.names() else None)
            return action

        add("Copy settings", lambda: controller.copy_settings(name))
        add("Paste settings onto this file", lambda: controller.paste_settings(name), controller.can_paste())
        menu.addSeparator()
        add("Re-detect", lambda: controller.redetect(name))
        add("Test OCR (T)", lambda: self.proof_requested.emit(name))
        add("Open logs", lambda: self.logs_requested.emit(name))
        menu.addSeparator()
        reviewed = is_reviewed(entry)
        add("Mark not reviewed" if reviewed else "Mark reviewed", lambda: self._toggle_reviewed(name),
            can_mark_reviewed(entry))
        add("Include file" if entry.skipped else "Skip file", lambda: self._toggle_skipped(name))
        return menu

    def _on_row_context(self, name: str, position: QPoint) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.select(name)
        menu = self.context_menu(name)
        menu.exec(position)
        menu.deleteLater()
