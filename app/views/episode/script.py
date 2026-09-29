"""The Script panel (episode-view-v2.html, screen 4): the episode's lines,
first line first, each as "H:MM:SS.mmm → H:MM:SS.mmm" over its text, with
the raw float times the run reported (or the written .ass holds).

It follows the newest line: a line appended while following scrolls into
view and flashes the drip's amber briefly. Scrolling up -- or clicking a
line to look at it -- stops following, so the list holds still while the
user reads; the "following" pill at the foot brings it back.

A list view with one painted row per line rather than a widget per line:
a full episode is several hundred lines, and the Done screen reloads them
all at once.
"""
from __future__ import annotations

from PyQt6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QRectF,
    QSize,
    Qt,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFontMetrics, QPainter
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QListView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from app.episode_feed import Line, format_ts
from app.theme import tokens
from app.views.episode.common import BODY_SIZE, SMALL_SIZE, font, text_label
from app.widgets.base import Button

TITLE = "SCRIPT"
FOLLOWING = "▾ following"
FOLLOW = "▾ follow"
FLASH_MS = 1200                 # the newest line's amber flash
FLASH_BG = "#2a2515"            # the drip's highlight (@keyframes drip, episode-view-v2.html)
ROW_PAD_X, ROW_PAD_Y, ROW_GAP = 8, 4, 1       # mockup px
LINE_ROLE = Qt.ItemDataRole.UserRole + 1


def lines_text(count: int) -> str:
    return f"{count} line" if count == 1 else f"{count} lines"


def row_text(line) -> str:
    """The row as one string (tests, tooltips): times, then the text."""
    return f"{format_ts(line[0])} → {format_ts(line[1])}  {line[2]}"


class ScriptModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._lines: list[Line] = []

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._lines)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._lines):
            return None
        line = self._lines[index.row()]
        if role == LINE_ROLE:
            return line
        if role == Qt.ItemDataRole.DisplayRole:
            return row_text(line)
        if role == Qt.ItemDataRole.ToolTipRole:
            return line.text.replace("\\N", "\n")
        return None

    def lines(self) -> list[Line]:
        return list(self._lines)

    def append(self, line: Line) -> None:
        row = len(self._lines)
        self.beginInsertRows(QModelIndex(), row, row)
        self._lines.append(line)
        self.endInsertRows()

    def set_lines(self, lines: list[Line]) -> None:
        self.beginResetModel()
        self._lines = list(lines)
        self.endResetModel()


class ScriptDelegate(QStyledItemDelegate):
    """Two painted lines a row: the times (dim, small), then the text
    (elided; the whole of it is the row's tooltip)."""

    def __init__(self, panel: ScriptPanel):
        super().__init__(panel)
        self._panel = panel
        self._time_font = font(SMALL_SIZE)
        self._text_font = font(BODY_SIZE)

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        height = (QFontMetrics(self._time_font).height() + QFontMetrics(self._text_font).height()
                  + 2 * tokens.px(ROW_PAD_Y) + tokens.px(ROW_GAP))
        return QSize(tokens.px(200), height)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        line = index.data(LINE_ROLE)
        if line is None:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(option.rect).adjusted(0, 0, 0, -tokens.px(ROW_GAP))
        background = None
        if option.state & QStyle.StateFlag.State_Selected:
            background = QColor(tokens.ROW_SELECTED)
        elif option.state & QStyle.StateFlag.State_MouseOver:
            background = QColor(tokens.ROW_HOVER)
        flash = self._panel.flash_for(index.row())
        if flash > 0:
            tint = QColor(FLASH_BG)
            tint.setAlphaF(flash)
            background = tint
        if background is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(rect, tokens.RADIUS_XS + 1, tokens.RADIUS_XS + 1)
        inner = rect.adjusted(tokens.px(ROW_PAD_X), tokens.px(ROW_PAD_Y), -tokens.px(ROW_PAD_X), -tokens.px(ROW_PAD_Y))
        painter.setFont(self._time_font)
        time_height = painter.fontMetrics().height()
        painter.setPen(QColor(tokens.DIM2))
        painter.drawText(QRectF(inner.left(), inner.top(), inner.width(), time_height),
                         int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                         f"{format_ts(line.start)} → {format_ts(line.end)}")
        painter.setFont(self._text_font)
        metrics = painter.fontMetrics()
        text = metrics.elidedText(line.text.replace("\\N", " / "), Qt.TextElideMode.ElideRight, int(inner.width()))
        painter.setPen(QColor(tokens.TXT))
        painter.drawText(QRectF(inner.left(), inner.top() + time_height, inner.width(), metrics.height()),
                         int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text)
        painter.restore()


class ScriptPanel(QWidget):
    line_clicked = pyqtSignal(float, float, str)      # start, end, text
    following_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None, *, animated: bool = True, live: bool = True):
        """`live=False` is a finished script (the Done screen): no following,
        no pill, no flash."""
        super().__init__(parent)
        self.setObjectName("EpisodeScript")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QWidget#EpisodeScript {{ background-color: {tokens.PANEL}; border-left: 1px solid {tokens.LINE}; }}"
            f"QListView#EpisodeScriptList {{ background-color: {tokens.PANEL}; border: none; }}"
            f"QWidget#EpisodeScriptHead {{ background-color: {tokens.PANEL}; border-bottom: 1px solid {tokens.LINE}; }}"
        )
        self._animated = animated and live
        self._live = live
        self._following = True
        self._total: int | None = None
        self._flash_row = -1
        self._flash = 0.0

        column = QVBoxLayout(self)
        column.setContentsMargins(1, 0, 0, 0)            # the 1 px left border
        column.setSpacing(0)

        head = QWidget()
        head.setObjectName("EpisodeScriptHead")
        head.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        head_row = QHBoxLayout(head)
        head_row.setContentsMargins(tokens.px(10), tokens.px(7), tokens.px(10), tokens.px(7))
        self.title_label = text_label(TITLE, color=tokens.DIM2, size=tokens.FONT_SIZE_SCOPE_BASE, weight=600)
        self.count_label = text_label(lines_text(0), color=tokens.DIM2, size=SMALL_SIZE)
        head_row.addWidget(self.title_label)
        head_row.addStretch(1)
        head_row.addWidget(self.count_label)
        column.addWidget(head)

        self.model = ScriptModel(self)
        self.list_view = QListView()
        self.list_view.setObjectName("EpisodeScriptList")
        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(ScriptDelegate(self))
        self.list_view.setUniformItemSizes(True)
        self.list_view.setMouseTracking(True)
        self.list_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_view.setSpacing(0)
        self.list_view.setContentsMargins(tokens.px(4), tokens.px(4), tokens.px(4), tokens.px(4))
        self.list_view.clicked.connect(self._on_clicked)
        bar = self.list_view.verticalScrollBar()
        bar.rangeChanged.connect(self._on_range_changed)
        bar.valueChanged.connect(self._on_scrolled)
        column.addWidget(self.list_view, 1)

        foot = QHBoxLayout()
        foot.setContentsMargins(*((tokens.px(6), tokens.px(4), tokens.px(6), tokens.px(6)) if live else (0,) * 4))
        self.follow_button = Button(FOLLOWING, "ghost", small=True)
        self.follow_button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.follow_button.clicked.connect(lambda: self.set_following(True))
        foot.addStretch(1)
        foot.addWidget(self.follow_button)
        self.follow_button.setVisible(live)
        foot.addStretch(1)
        column.addLayout(foot)

        self._flash_animation = QVariantAnimation(self)
        self._flash_animation.setStartValue(1.0)
        self._flash_animation.setEndValue(0.0)
        self._flash_animation.setDuration(FLASH_MS)
        self._flash_animation.valueChanged.connect(self._set_flash)
        self._sync_follow_button()

    # --- reading ----------------------------------------------------------------------

    def lines(self) -> list[Line]:
        return self.model.lines()

    def row_texts(self) -> list[str]:
        return [row_text(line) for line in self.model.lines()]

    def count(self) -> int:
        return self.model.rowCount()

    def count_text(self) -> str:
        return self.count_label.text()

    def is_following(self) -> bool:
        return self._following

    def flash_for(self, row: int) -> float:
        return self._flash if row == self._flash_row else 0.0

    # --- filling ----------------------------------------------------------------------

    def append(self, line) -> None:
        """Add a line at the end; while following, it scrolls into view and flashes."""
        line = Line(float(line[0]), float(line[1]), str(line[2]))
        self.model.append(line)
        if self._following:
            self._flash_row = self.model.rowCount() - 1
            if self._animated:
                self._flash_animation.stop()
                self._flash_animation.start()
            self.list_view.scrollToBottom()
        self._sync_count()

    def set_lines(self, lines) -> None:
        """Replace every line (the Done screen's reload); scrolled to the top."""
        self._flash_animation.stop()
        self._flash_row, self._flash = -1, 0.0
        self.model.set_lines([Line(float(s), float(e), str(t)) for s, e, t in lines])
        self._total = None
        self._sync_count()
        self.list_view.scrollToTop()

    def clear(self) -> None:
        self.set_lines([])
        self.set_following(True)

    def set_total(self, total: int | None) -> None:
        """The count the header shows when it is ahead of the rows (the
        drip is behind the run); None shows the rows'."""
        self._total = total
        self._sync_count()

    def set_following(self, following: bool) -> None:
        following = bool(following)
        if following:
            self.list_view.scrollToBottom()
        if following == self._following:
            return
        self._following = following
        self._sync_follow_button()
        self.following_changed.emit(following)

    def click_row(self, row: int) -> None:
        """As a user click on row `row` (tests)."""
        self._on_clicked(self.model.index(row))

    # --- internals ----------------------------------------------------------------------

    def _sync_count(self) -> None:
        rows = self.model.rowCount()
        self.count_label.setText(lines_text(max(rows, self._total or 0)))

    def _sync_follow_button(self) -> None:
        self.follow_button.setText(FOLLOWING if self._following else FOLLOW)
        self.follow_button.set_toggled(not self._following)

    def _set_flash(self, value) -> None:
        self._flash = float(value)
        if self._flash_row >= 0:
            self.list_view.viewport().update(self.list_view.visualRect(self.model.index(self._flash_row)))

    def _on_range_changed(self, _minimum: int, maximum: int) -> None:
        if self._following:
            self.list_view.verticalScrollBar().setValue(maximum)

    def _on_scrolled(self, value: int) -> None:
        bar = self.list_view.verticalScrollBar()
        if value < bar.maximum() and self._following:
            self._following = False
            self._sync_follow_button()
            self.following_changed.emit(False)
        elif value >= bar.maximum() and not self._following and bar.maximum() > 0:
            self._following = True
            self._sync_follow_button()
            self.following_changed.emit(True)

    def _on_clicked(self, index: QModelIndex) -> None:
        line = index.data(LINE_ROLE)
        if line is None:
            return
        self.list_view.setCurrentIndex(index)
        self.set_following(False)
        self.line_clicked.emit(line.start, line.end, line.text)
