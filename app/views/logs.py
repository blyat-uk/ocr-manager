"""The logs window (today's LogsDialog, current-app inventory): non-modal,
one collapsible section per log key -- "Pipeline" first, then "Detections",
then files in name order -- appended live from `controller.log_appended`.

Each section shows at most `LOG_LIMIT` characters (~500 KB), dropping the
oldest text first, as the controller's log book keeps it. A run start (or a
folder change) restarts the logs: `logs_cleared` rebuilds the sections from
`controller.log_keys()`, keeping which sections were open and where the
window was scrolled -- a run start must not collapse the log the user is
reading. `show_key(key)` expands a section and scrolls to it ("Open logs" on
a queue row).
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from app.logbook import FIRST_KEYS, LOG_LIMIT

LOGS_TITLE = "Logs"
LOGS_SIZE = (700, 500)
BODY_MIN_HEIGHT, BODY_MAX_HEIGHT = 120, 300
SCROLL_SETTLE_MS = 250          # how long show_key waits for the layout to make room before giving up
_FIRST = {key: index for index, key in enumerate(FIRST_KEYS)}     # "Pipeline", then "Detections"


def key_order(key: str) -> tuple[int, str]:
    return _FIRST.get(key, len(_FIRST)), key


class LogSection(QWidget):
    """A header button ("▸ key" / "▾ key") over a read-only text body."""

    def __init__(self, key: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.key = key
        self.setObjectName("LogSection")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = QPushButton()
        self.header.setObjectName("LogHeader")
        self.header.setCheckable(True)
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.toggled.connect(self._on_toggled)
        layout.addWidget(self.header)
        self.body = QPlainTextEdit()
        self.body.setObjectName("LogBody")
        self.body.setReadOnly(True)
        self.body.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = QFont("monospace")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.body.setFont(font)
        self.body.setVisible(False)
        layout.addWidget(self.body)
        self._fit_height()
        self._update_header()

    def title(self) -> str:
        return self.key

    def text(self) -> str:
        return self.body.toPlainText()

    def is_expanded(self) -> bool:
        return self.header.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.header.setChecked(expanded)

    def set_text(self, text: str) -> None:
        self.body.setPlainText(text[-LOG_LIMIT:])
        self._fit_height()
        self._scroll_to_end()

    def append(self, text: str) -> None:
        bar = self.body.verticalScrollBar()
        at_end = bar.value() >= bar.maximum()
        cursor = QTextCursor(self.body.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        excess = self.body.document().characterCount() - 1 - LOG_LIMIT     # the count includes a final separator
        if excess > 0:
            cursor.setPosition(0)
            cursor.setPosition(excess, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
        self._fit_height()
        if at_end:
            self._scroll_to_end()

    def _fit_height(self) -> None:
        """As tall as the text, between BODY_MIN_HEIGHT and BODY_MAX_HEIGHT; a
        fixed height, so the list scrolls instead of squeezing the bodies."""
        lines = self.body.document().blockCount()
        margins = self.body.contentsMargins()
        wanted = (lines * self.body.fontMetrics().lineSpacing() + margins.top() + margins.bottom()
                  + 2 * self.body.document().documentMargin() + 2 * self.body.frameWidth())
        self.body.setFixedHeight(max(BODY_MIN_HEIGHT, min(BODY_MAX_HEIGHT, int(wanted))))

    def _scroll_to_end(self) -> None:
        bar = self.body.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_toggled(self, expanded: bool) -> None:
        self.body.setVisible(expanded)
        self._update_header()

    def _update_header(self) -> None:
        self.header.setText(f"{'▾' if self.is_expanded() else '▸'} {self.key}")


class LogsWindow(QWidget):
    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Window)
        self._controller = controller
        self._sections: dict[str, LogSection] = {}
        self._scroll_target: str | None = None
        self._pending_position: int | None = None       # scroll position to restore after a reload
        self.setObjectName("LogsWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle(LOGS_TITLE)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(*LOGS_SIZE)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("LogsScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll_area)
        self._container = QWidget()
        self._container.setObjectName("LogsList")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(4)
        self._layout.addStretch(1)
        self.scroll_area.setWidget(self._container)

        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.setInterval(0)
        self._scroll_timer.timeout.connect(self._scroll_to_target)
        self._settle_timer = QTimer(self)                    # stops waiting for room to scroll into
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(SCROLL_SETTLE_MS)
        self._settle_timer.timeout.connect(self._forget_target)
        self.scroll_area.verticalScrollBar().rangeChanged.connect(self._on_range_changed)

        controller.log_appended.connect(self._on_log_appended)
        controller.logs_cleared.connect(self.reload)
        self.reload()

    # --- reading ----------------------------------------------------------------------------

    def keys(self) -> list[str]:
        return list(self._sections)

    def section(self, key: str) -> LogSection:
        return self._sections[key]

    # --- commands ---------------------------------------------------------------------------

    def reload(self) -> None:
        """Rebuild every section from the controller's logs, keeping the
        sections that were open open and the scroll position."""
        expanded = {key: section.is_expanded() for key, section in self._sections.items()}
        position = self.scroll_area.verticalScrollBar().value()
        for section in self._sections.values():
            self._layout.removeWidget(section)
            section.setParent(None)      # removeWidget alone leaves it parented and painting
            section.deleteLater()
        self._sections = {}
        for key in self._controller.log_keys():
            section = self._section_for(key)
            section.set_text(self._controller.log_text(key))
            section.set_expanded(expanded.get(key, False))
        if position:
            self._pending_position = position
            self._settle_timer.start()
            self._scroll_timer.start()

    def show_key(self, key: str) -> None:
        """Expand `key`'s section (an empty one when it has no log yet) and
        scroll to it once the layout has placed it."""
        self._section_for(key).set_expanded(True)
        self._scroll_target = key
        self._settle_timer.start()
        self._scroll_timer.start()

    # --- internals --------------------------------------------------------------------------

    def _section_for(self, key: str) -> LogSection:
        section = self._sections.get(key)
        if section is not None:
            return section
        section = LogSection(key, self._container)
        ordered = sorted([*self._sections, key], key=key_order)
        index = ordered.index(key)
        self._sections[key] = section
        self._sections = {name: self._sections[name] for name in ordered}
        self._layout.insertWidget(index, section)            # the stretch stays last
        return section

    def _on_log_appended(self, key: str, text: str) -> None:
        self._section_for(key).append(text)

    def _scroll_to_target(self) -> None:
        """Put the target section at the top (or restore the position a
        reload had), as far as the list scrolls. The list may not have grown
        yet: then each range change tries again, until SCROLL_SETTLE_MS."""
        bar = self.scroll_area.verticalScrollBar()
        if self._scroll_target is not None:
            section = self._sections.get(self._scroll_target)
            if section is None:
                self._forget_target()
                return
            bar.setValue(min(section.y(), bar.maximum()))
        elif self._pending_position is not None:
            bar.setValue(min(self._pending_position, bar.maximum()))
            if bar.maximum() >= self._pending_position:
                self._pending_position = None

    def _on_range_changed(self, _minimum: int, _maximum: int) -> None:
        if self._scroll_target is not None or self._pending_position is not None:
            self._scroll_to_target()

    def _forget_target(self) -> None:
        self._scroll_target = None
        self._pending_position = None
        self._settle_timer.stop()
