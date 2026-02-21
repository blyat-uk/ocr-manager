"""Dual-handle time range slider widget."""

import os

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush


def format_time(seconds: int) -> str:
    """Format seconds as MM:SS."""
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def parse_time(time_str: str) -> int | None:
    """Parse MM:SS format to seconds. Returns None if invalid."""
    if not time_str:
        return None
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        pass
    return None


class RangeSlider(QWidget):
    """Custom dual-handle range slider."""

    range_changed = pyqtSignal(int, int)  # start_seconds, end_seconds (fires during drag)
    range_committed = pyqtSignal(int, int, str)  # start_seconds, end_seconds, handle ('start'/'end')

    def __init__(self, parent=None):
        super().__init__(parent)
        self._min_value = 0
        self._max_value = 100
        self._start_value = 0
        self._end_value = 100
        self._dragging = None  # 'start', 'end', or None
        self._handle_radius = 8
        self._track_height = 6

        self.setMinimumHeight(24)
        self.setMinimumWidth(200)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_range(self, min_val: int, max_val: int):
        """Set the slider range."""
        self._min_value = min_val
        self._max_value = max(max_val, min_val + 1)
        # Clamp current values
        self._start_value = max(self._min_value, min(self._start_value, self._max_value))
        self._end_value = max(self._min_value, min(self._end_value, self._max_value))
        self.update()

    def set_values(self, start: int, end: int):
        """Set start and end values."""
        self._start_value = max(self._min_value, min(start, self._max_value))
        self._end_value = max(self._min_value, min(end, self._max_value))
        # Ensure start <= end
        if self._start_value > self._end_value:
            self._start_value, self._end_value = self._end_value, self._start_value
        self.update()

    def get_values(self) -> tuple[int, int]:
        """Get current start and end values."""
        return self._start_value, self._end_value

    def _value_to_x(self, value: int) -> int:
        """Convert value to x coordinate."""
        if self._max_value == self._min_value:
            return self._handle_radius
        ratio = (value - self._min_value) / (self._max_value - self._min_value)
        usable_width = self.width() - 2 * self._handle_radius
        return int(self._handle_radius + ratio * usable_width)

    def _x_to_value(self, x: int) -> int:
        """Convert x coordinate to value."""
        usable_width = self.width() - 2 * self._handle_radius
        if usable_width <= 0:
            return self._min_value
        ratio = (x - self._handle_radius) / usable_width
        ratio = max(0, min(1, ratio))
        return int(self._min_value + ratio * (self._max_value - self._min_value))

    def _get_handle_rect(self, value: int) -> QRect:
        """Get the rectangle for a handle."""
        x = self._value_to_x(value)
        y = self.height() // 2
        return QRect(
            x - self._handle_radius,
            y - self._handle_radius,
            self._handle_radius * 2,
            self._handle_radius * 2
        )

    def paintEvent(self, event):
        """Draw the slider."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Colors from qt-material theme
        track_color = QColor(os.environ.get('QTMATERIAL_SECONDARYLIGHTCOLOR', '#4f5b62'))
        fill_color = QColor(os.environ.get('QTMATERIAL_PRIMARYCOLOR', '#ffd740'))
        handle_color = QColor(os.environ.get('QTMATERIAL_PRIMARYCOLOR', '#ffd740'))

        # Track dimensions
        y_center = self.height() // 2
        track_y = y_center - self._track_height // 2

        # Draw background track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(track_color))
        painter.drawRoundedRect(
            self._handle_radius, track_y,
            self.width() - 2 * self._handle_radius, self._track_height,
            self._track_height // 2, self._track_height // 2
        )

        # Draw filled range
        start_x = self._value_to_x(self._start_value)
        end_x = self._value_to_x(self._end_value)
        painter.setBrush(QBrush(fill_color))
        painter.drawRoundedRect(
            start_x, track_y,
            end_x - start_x, self._track_height,
            self._track_height // 2, self._track_height // 2
        )

        # Draw handles
        painter.setBrush(QBrush(handle_color))
        painter.setPen(QPen(QColor(os.environ.get('QTMATERIAL_SECONDARYDARKCOLOR', '#232629')), 2))

        # Start handle
        painter.drawEllipse(self._get_handle_rect(self._start_value))
        # End handle
        painter.drawEllipse(self._get_handle_rect(self._end_value))

    def mousePressEvent(self, event):
        """Start dragging a handle."""
        if event.button() != Qt.MouseButton.LeftButton:
            return

        x = event.position().x()
        start_rect = self._get_handle_rect(self._start_value)
        end_rect = self._get_handle_rect(self._end_value)

        # Check which handle was clicked (prefer the one closer to click)
        start_dist = abs(x - start_rect.center().x())
        end_dist = abs(x - end_rect.center().x())

        if start_rect.contains(event.position().toPoint()):
            self._dragging = 'start'
        elif end_rect.contains(event.position().toPoint()):
            self._dragging = 'end'
        elif start_dist < end_dist:
            # Click on track - move start handle
            self._dragging = 'start'
            self._update_handle(x)
        else:
            # Click on track - move end handle
            self._dragging = 'end'
            self._update_handle(x)

    def mouseMoveEvent(self, event):
        """Drag the handle."""
        if self._dragging:
            self._update_handle(event.position().x())

    def mouseReleaseEvent(self, event):
        """Stop dragging and emit committed value."""
        if self._dragging is not None:
            self.range_committed.emit(self._start_value, self._end_value, self._dragging)
        self._dragging = None

    def _update_handle(self, x: float):
        """Update handle position based on x coordinate."""
        value = self._x_to_value(int(x))

        if self._dragging == 'start':
            # Don't go past end handle
            self._start_value = min(value, self._end_value)
        elif self._dragging == 'end':
            # Don't go before start handle
            self._end_value = max(value, self._start_value)

        self.update()
        self.range_changed.emit(self._start_value, self._end_value)


class TimeRangeSlider(QWidget):
    """Time range slider with labels and reference video info."""

    range_changed = pyqtSignal(int, int)  # start_seconds, end_seconds (fires during drag)
    range_committed = pyqtSignal(int, int, str)  # start_seconds, end_seconds, handle ('start'/'end')

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration = 0
        self._reference_video = ""

        self._init_ui()

    def _init_ui(self):
        """Setup UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._slider = RangeSlider()
        self._slider.range_changed.connect(self._on_range_changed)
        self._slider.range_committed.connect(self._on_range_committed)
        layout.addWidget(self._slider, 1)

        self._time_label = QLabel("00:00 - 00:00")
        self._time_label.setMinimumWidth(90)
        layout.addWidget(self._time_label)

    def set_duration(self, duration_seconds: int, reference_video: str = "", reset_values: bool = False):
        """Set the video duration and reference filename.

        Args:
            duration_seconds: Video duration in seconds
            reference_video: Reference video filename for tooltip
            reset_values: If True, reset slider to full range. If False (default),
                         preserve existing values if they're within valid range.
        """
        old_duration = self._duration
        self._duration = max(1, duration_seconds)
        self._reference_video = reference_video

        self._slider.set_range(0, self._duration)

        # Only reset values if explicitly requested or duration changed significantly
        if reset_values or old_duration == 0:
            self._slider.set_values(0, self._duration)
        else:
            # Preserve existing values, clamped to new duration
            start, end = self._slider.get_values()
            # Clamp end to new duration
            end = min(end, self._duration)
            # Ensure start is still valid
            start = min(start, end)
            self._slider.set_values(start, end)

        self._update_labels()

    def set_values(self, start: int, end: int):
        """Set the current time range values."""
        self._slider.set_values(start, end)
        self._update_labels()

    def get_values(self) -> tuple[int, int]:
        """Get current start and end values in seconds."""
        return self._slider.get_values()

    def get_time_strings(self) -> tuple[str, str]:
        """Get start and end times as MM:SS strings."""
        start, end = self._slider.get_values()
        # Only return non-empty if not at boundaries
        start_str = format_time(start) if start > 0 else ""
        end_str = format_time(end) if end < self._duration else ""
        return start_str, end_str

    def set_time_range(self, start_str: str, end_str: str):
        """Set time range from MM:SS strings."""
        start = parse_time(start_str) if start_str else 0
        end = parse_time(end_str) if end_str else self._duration

        # Use defaults if parsing failed
        if start is None:
            start = 0
        if end is None:
            end = self._duration

        # Clamp to valid range
        start = max(0, min(start, self._duration))
        end = max(start, min(end, self._duration))

        self.set_values(start, end)

    def _on_range_changed(self, start: int, end: int):
        """Handle slider range change (during drag)."""
        self._update_labels()
        self.range_changed.emit(start, end)

    def _on_range_committed(self, start: int, end: int, handle: str):
        """Handle slider range commit (on mouse release)."""
        self.range_committed.emit(start, end, handle)

    def _update_labels(self):
        """Update time display labels and tooltip."""
        start, end = self._slider.get_values()
        self._time_label.setText(f"{format_time(start)} - {format_time(end)}")

        # Set tooltip on widget itself with video reference info
        if self._reference_video:
            self.setToolTip(
                f"{self._reference_video} (duration: {format_time(self._duration)})"
            )
        else:
            self.setToolTip(f"Duration: {format_time(self._duration)}")
