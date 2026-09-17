"""The centre stage (`.stage`, ui-spec §3.3): a tab bar and the active
tab's page.

Tabs are pluggable (`StageTab`): plan 3C supplies the real Crop, Brightness
and Time ranges views; until then `placeholder_tabs` shows each tab's values
as key/value rows. The stage knows nothing about any particular tab: it
forwards the selected file, calls `refresh()` when the model changes and
reports tab switches, which the inspector follows to show the active tab's
panel (ruling B4).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from app.state_text import brightness_text, crop_text, ranges_text
from app.widgets.base import KvRow, SectionHeader, repolish


@runtime_checkable
class StageTab(Protocol):
    title: str                                   # "Crop" | "Brightness" | "Time ranges"

    def page(self) -> QWidget: ...               # centre stage content

    def inspector_panel(self) -> QWidget: ...    # the active-tab section shown in the inspector (ruling B4)

    def set_file(self, name: str | None) -> None: ...

    def refresh(self) -> None: ...               # model changed for the current file


class Stage(QWidget):
    tab_changed = pyqtSignal(int)

    def __init__(self, controller, tabs: list[StageTab], parent: QWidget | None = None):
        super().__init__(parent)
        if not tabs:
            raise ValueError("the stage needs at least one tab")
        self._controller = controller
        self._tabs = list(tabs)
        self._file: str | None = None
        self._current = 0
        self.setObjectName("Stage")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        head = QWidget()
        head.setObjectName("StageHead")
        head.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        head_layout = QHBoxLayout(head)
        head_layout.setContentsMargins(12, 8, 12, 8)
        head_layout.setSpacing(2)
        self._buttons: list[QPushButton] = []
        for index, tab in enumerate(self._tabs):
            button = QPushButton(tab.title)
            button.setObjectName("StageTab")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            button.setProperty("on", index == 0)
            button.clicked.connect(lambda _checked=False, i=index: self.set_current(i))
            head_layout.addWidget(button)
            self._buttons.append(button)
        head_layout.addStretch(1)
        layout.addWidget(head)

        self._pages = QStackedWidget()
        self._pages.setObjectName("StagePages")
        for tab in self._tabs:
            self._pages.addWidget(tab.page())
        layout.addWidget(self._pages, 1)

        controller.file_changed.connect(self._on_file_changed)
        controller.files_changed.connect(self._refresh_tabs)
        controller.folder_changed.connect(self._refresh_tabs)

    def tabs(self) -> list[StageTab]:
        return list(self._tabs)

    def tab_buttons(self) -> list[QPushButton]:
        return list(self._buttons)

    def page_host(self) -> QStackedWidget:
        return self._pages

    def current_index(self) -> int:
        return self._current

    def current_tab(self) -> StageTab:
        return self._tabs[self._current]

    def current_file(self) -> str | None:
        return self._file

    def index_of(self, title: str) -> int | None:
        return next((index for index, tab in enumerate(self._tabs) if tab.title == title), None)

    def set_current(self, index: int) -> None:
        if not 0 <= index < len(self._tabs) or index == self._current:
            return
        self._current = index
        self._pages.setCurrentIndex(index)
        for position, button in enumerate(self._buttons):
            button.setProperty("on", position == index)
            repolish(button)
        self.tab_changed.emit(index)

    def set_file(self, name: str | None) -> None:
        self._file = name
        for tab in self._tabs:
            tab.set_file(name)

    def _on_file_changed(self, name: str) -> None:
        if name == self._file:
            self._refresh_tabs()

    def _refresh_tabs(self, *_args) -> None:
        if self._controller.project is None:
            return
        if self._file is not None and self._file not in self._controller.project.files:
            return                               # the queue selects another file next
        for tab in self._tabs:
            tab.refresh()


# --------------------------------------------------------------------------
# Placeholder tabs (until plan 3C)
# --------------------------------------------------------------------------

def _source_text(value) -> str:
    return "—" if value is None else value.source.value


def _crop_rows(entry) -> list[tuple[str, str]]:
    return [("Crop", crop_text(entry.crop)), ("Source", _source_text(entry.crop))]


def _brightness_rows(entry) -> list[tuple[str, str]]:
    return [("Brightness", brightness_text(entry.brightness)), ("Source", _source_text(entry.brightness))]


def _ranges_rows(entry) -> list[tuple[str, str]]:
    return [("OCR window", ranges_text(entry.time_ranges)), ("Source", _source_text(entry.time_ranges))]


class _KvList(QWidget):
    def __init__(self, heading: str, object_name: str | None = None, margins=(0, 0, 0, 0),
                 max_width: int | None = None):
        super().__init__()
        if object_name:
            self.setObjectName(object_name)
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(*margins)
        self._layout.setSpacing(5)
        self._layout.addWidget(SectionHeader(heading))
        self._rows: list[KvRow] = []
        self._keys: list[str] = []
        self._max_width = max_width
        self._layout.addStretch(1)

    def set_rows(self, rows: list[tuple[str, str]]) -> None:
        keys = [key for key, _value in rows]
        if keys != self._keys:
            for row in self._rows:
                self._layout.removeWidget(row)
                row.deleteLater()
            self._rows = []
            for index, key in enumerate(keys):
                row = KvRow(key, "")
                if self._max_width:
                    row.setMaximumWidth(self._max_width)
                self._layout.insertWidget(1 + index, row)
                self._rows.append(row)
            self._keys = keys
        for row, (_key, value) in zip(self._rows, rows, strict=True):
            row.set_value(value)

    def values(self) -> list[str]:
        return [row.value() for row in self._rows]


class PlaceholderTab:
    """A StageTab showing the selected file's values for one tab as rows."""

    def __init__(self, controller, title: str, rows):
        self.title = title
        self._controller = controller
        self._rows = rows
        self._file: str | None = None
        self._page = _KvList(title, "PlaceholderPage", margins=(16, 14, 16, 14), max_width=420)
        self._panel = _KvList(title)

    def page(self) -> QWidget:
        return self._page

    def inspector_panel(self) -> QWidget:
        return self._panel

    def current_file(self) -> str | None:
        return self._file

    def values(self) -> list[str]:
        return self._page.values()

    def set_file(self, name: str | None) -> None:
        self._file = name
        self.refresh()

    def refresh(self) -> None:
        controller = self._controller
        if self._file is None or self._file not in controller.names():
            rows = []
        else:
            rows = self._rows(controller.entry(self._file))
        self._page.set_rows(rows)
        self._panel.set_rows(rows)


def placeholder_tabs(controller) -> list[StageTab]:
    return [PlaceholderTab(controller, "Crop", _crop_rows),
            PlaceholderTab(controller, "Brightness", _brightness_rows),
            PlaceholderTab(controller, "Time ranges", _ranges_rows)]
