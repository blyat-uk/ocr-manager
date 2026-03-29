"""Chip-based widget for displaying and managing multiple time ranges."""

import os

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSizePolicy, QToolButton,
)
from PyQt6.QtCore import pyqtSignal, Qt, QSize


class _RangeChip(QWidget):
    """Single chip: clickable label + small close button."""

    clicked = pyqtSignal()
    remove_clicked = pyqtSignal()

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 2, 2)
        layout.setSpacing(2)

        self._label_btn = QPushButton(label)
        self._label_btn.setFlat(True)
        self._label_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._label_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._label_btn.setStyleSheet("QPushButton { border: none; padding: 0; font-size: 11px; }")
        self._label_btn.clicked.connect(self.clicked.emit)
        layout.addWidget(self._label_btn)

        self._close_btn = QToolButton()
        self._close_btn.setText("\u00d7")
        self._close_btn.setFixedSize(QSize(16, 16))
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.setStyleSheet(
            "QToolButton { border: none; font-size: 13px; padding: 0; }"
            "QToolButton:hover { color: #f38ba8; }"
        )
        self._close_btn.clicked.connect(self.remove_clicked.emit)
        layout.addWidget(self._close_btn)

        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_active_style(self, active: bool):
        primary = os.environ.get('QTMATERIAL_PRIMARYCOLOR', '#ffd740')
        secondary_light = os.environ.get('QTMATERIAL_SECONDARYLIGHTCOLOR', '#4f5b62')
        if active:
            self.setStyleSheet(
                f"_RangeChip {{ border: 2px solid {primary}; border-radius: 10px; }}"
            )
        else:
            self.setStyleSheet(
                f"_RangeChip {{ border: 1px solid {secondary_light}; border-radius: 10px; }}"
            )


class TimeRangeChipsWidget(QWidget):
    """Horizontal row of range chips with add/remove support."""

    range_selected = pyqtSignal(int)   # index of clicked chip
    range_removed = pyqtSignal(int)    # index of removed chip
    add_clicked = pyqtSignal()         # "+" button clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ranges: list[tuple[str, str]] = []
        self._active_index: int = -1
        self._chips: list[_RangeChip] = []

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self._add_btn = QPushButton("+")
        self._add_btn.setFixedWidth(28)
        self._add_btn.setToolTip("Add current slider range")
        self._add_btn.clicked.connect(self.add_clicked.emit)

        self._layout.addWidget(self._add_btn)
        self._layout.addStretch()

    def set_ranges(self, ranges: list[tuple[str, str]]):
        """Rebuild chips from a list of (start, end) tuples."""
        self._ranges = list(ranges)
        self._rebuild_chips()

    def set_active(self, index: int):
        """Highlight the active chip."""
        self._active_index = index
        self._update_chip_styles()

    def get_ranges(self) -> list[tuple[str, str]]:
        return list(self._ranges)

    def _rebuild_chips(self):
        for chip in self._chips:
            self._layout.removeWidget(chip)
            chip.deleteLater()
        self._chips.clear()

        for i, (start, end) in enumerate(self._ranges):
            s = start or "0:00"
            e = end or "end"
            chip = _RangeChip(f"{s} - {e}")
            idx = i
            chip.clicked.connect(lambda index=idx: self._select(index))
            chip.remove_clicked.connect(lambda index=idx: self.range_removed.emit(index))
            self._chips.append(chip)
            self._layout.insertWidget(i, chip)

        self._update_chip_styles()

    def _select(self, index: int):
        self._active_index = index
        self._update_chip_styles()
        self.range_selected.emit(index)

    def _update_chip_styles(self):
        for i, chip in enumerate(self._chips):
            chip.set_active_style(i == self._active_index)
