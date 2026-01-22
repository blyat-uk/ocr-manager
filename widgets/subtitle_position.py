"""Subtitle position selector dialog with frame preview."""
import re
import tempfile
from pathlib import Path
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QSlider, QComboBox, QSpinBox)
from PyQt6.QtCore import pyqtSignal, Qt, QPoint, QRect
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QFontMetrics

from core.video_utils import get_video_duration, extract_frame


class SubtitleFrameLabel(QLabel):
    """Label that displays a frame with draggable subtitle text overlay."""

    # Signal emitted when margin changes during drag
    margin_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.scale_factor = 1.0
        self.pixmap_offset = QPoint(0, 0)

        # ASS parameters
        self.play_res_y = 720
        self.play_res_x = 1280
        self.margin_v = 80
        self.font_size = 38
        self.font_name = "Ubuntu"

        # Drag state
        self.dragging = False
        self.drag_start_y = 0
        self.drag_start_margin = 0

        # Sample text
        self.sample_text = "Sample subtitle text"

        # Text hit region (for mouse detection)
        self.text_rect = QRect()

        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_ass_params(self, play_res_x: int, play_res_y: int, margin_v: int,
                       font_size: int, font_name: str):
        """Set ASS positioning parameters."""
        self.play_res_x = play_res_x
        self.play_res_y = play_res_y
        self.margin_v = margin_v
        self.font_size = font_size
        self.font_name = font_name
        self.update_display()

    def set_margin_v(self, margin_v: int):
        """Update the vertical margin and redraw."""
        self.margin_v = margin_v
        self.update_display()

    def set_font_size(self, font_size: int):
        """Update font size and redraw."""
        self.font_size = font_size
        self.update_display()

    def set_frame(self, pixmap: QPixmap):
        """Set the frame pixmap."""
        self.original_pixmap = pixmap

        # Scale to fill the label exactly (label is sized to match aspect ratio)
        self.scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.scale_factor = pixmap.width() / self.scaled_pixmap.width() if self.scaled_pixmap.width() > 0 else 1.0

        # No offset needed - pixmap fills entire label
        self.pixmap_offset = QPoint(0, 0)

        self.update_display()

    def get_display_scale(self) -> float:
        """Get scale factor from PlayResY to display height."""
        if self.scaled_pixmap is None:
            return 1.0
        return self.scaled_pixmap.height() / self.play_res_y

    def margin_v_to_y(self, margin_v: int) -> int:
        """Convert MarginV value to display Y coordinate (baseline position)."""
        scale = self.get_display_scale()
        # MarginV is distance from bottom, so Y = height - (margin * scale)
        return int(self.scaled_pixmap.height() - (margin_v * scale))

    def y_to_margin_v(self, y: int) -> int:
        """Convert display Y coordinate to MarginV value."""
        scale = self.get_display_scale()
        if scale == 0:
            return self.margin_v
        # MarginV = (height - Y) / scale
        margin = int((self.scaled_pixmap.height() - y) / scale)
        # Clamp to valid range
        return max(10, min(self.play_res_y - 50, margin))

    def to_pixmap_coords(self, pos: QPoint) -> QPoint:
        """Convert label coordinates to pixmap coordinates."""
        return QPoint(pos.x() - self.pixmap_offset.x(), pos.y() - self.pixmap_offset.y())

    def update_display(self):
        """Redraw the frame with subtitle text overlay."""
        if self.scaled_pixmap is None:
            return

        display = QPixmap(self.scaled_pixmap)
        painter = QPainter(display)

        scale = self.get_display_scale()
        display_font_size = int(self.font_size * scale)

        # Create font
        font = QFont(self.font_name, display_font_size)
        painter.setFont(font)

        # Calculate text position
        metrics = QFontMetrics(font)
        text_width = metrics.horizontalAdvance(self.sample_text)
        text_height = metrics.height()

        # Center horizontally
        text_x = (self.scaled_pixmap.width() - text_width) // 2

        # Position vertically based on MarginV (baseline from bottom)
        baseline_y = self.margin_v_to_y(self.margin_v)
        # Adjust for text height (drawText uses baseline)
        text_y = baseline_y

        # Store text rect for hit testing (expand slightly for easier clicking)
        self.text_rect = QRect(
            text_x - 10,
            text_y - text_height - 10,
            text_width + 20,
            text_height + 20
        )

        # Draw guide line
        pen = QPen(QColor(255, 255, 0, 128), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(0, baseline_y, self.scaled_pixmap.width(), baseline_y)

        # Draw text outline (black)
        outline_pen = QPen(QColor(16, 16, 16), 3)
        painter.setPen(outline_pen)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    painter.drawText(text_x + dx, text_y + dy, self.sample_text)

        # Draw text (white)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(text_x, text_y, self.sample_text)

        painter.end()
        self.setPixmap(display)

    def mousePressEvent(self, event):
        """Start dragging if clicking near text."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.to_pixmap_coords(event.pos())
            # Check if click is within text region or just start dragging anywhere
            self.dragging = True
            self.drag_start_y = pos.y()
            self.drag_start_margin = self.margin_v
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        """Update text position while dragging (vertical only)."""
        if self.dragging and self.scaled_pixmap:
            pos = self.to_pixmap_coords(event.pos())
            # Calculate Y delta and convert to MarginV change
            delta_y = pos.y() - self.drag_start_y
            scale = self.get_display_scale()
            if scale > 0:
                # Invert because moving down (positive delta_y) decreases margin
                delta_margin = -int(delta_y / scale)
                new_margin = self.drag_start_margin + delta_margin
                # Clamp to valid range
                new_margin = max(10, min(self.play_res_y - 50, new_margin))
                if new_margin != self.margin_v:
                    self.margin_v = new_margin
                    self.margin_changed.emit(self.margin_v)
                    self.update_display()

    def mouseReleaseEvent(self, event):
        """Finish dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def resizeEvent(self, event):
        """Handle resize."""
        super().resizeEvent(event)
        if self.original_pixmap:
            self.set_frame(self.original_pixmap)


class SubtitlePositionDialog(QDialog):
    """Dialog for positioning subtitles on video frame."""

    position_selected = pyqtSignal(int, int)  # margin_v, font_size

    def __init__(self, mkv_files: list, header_template: str,
                 timeline_position: int = 5000, parent=None):
        super().__init__(parent)
        # Sort files naturally
        self.mkv_files = sorted(mkv_files)
        self.header_template = header_template
        self.initial_timeline_position = timeline_position
        self.temp_dir = tempfile.mkdtemp()
        self.current_duration = 0
        self.initial_resize_done = False  # Track if initial auto-resize has been done

        # Parse header template
        self.play_res_x = 1280
        self.play_res_y = 720
        self.margin_v = 80
        self.font_size = 38
        self.font_name = "Ubuntu"
        self.parse_header_template()

        # Debounce timer for auto frame extraction
        from PyQt6.QtCore import QTimer
        self.extract_timer = QTimer(self)
        self.extract_timer.setSingleShot(True)
        self.extract_timer.timeout.connect(self.extract_frame)

        self.setWindowTitle("Position Subtitles")
        self.resize(1200, 800)
        self.setup_ui()

        if self.mkv_files:
            self.load_episode(0)
            self.on_timeline_changed(self.initial_timeline_position)

    def parse_header_template(self):
        """Extract ASS parameters from header template."""
        if not self.header_template:
            return

        # Parse PlayResX
        match = re.search(r'PlayResX:\s*(\d+)', self.header_template)
        if match:
            self.play_res_x = int(match.group(1))

        # Parse PlayResY
        match = re.search(r'PlayResY:\s*(\d+)', self.header_template)
        if match:
            self.play_res_y = int(match.group(1))

        # Parse Style line for font name, size, and MarginV
        # Format: Style: Name,Fontname,Fontsize,...,MarginL,MarginR,MarginV,Encoding
        style_match = re.search(r'Style:\s*Default,([^,]+),(\d+)', self.header_template)
        if style_match:
            self.font_name = style_match.group(1)
            self.font_size = int(style_match.group(2))

        # MarginV is the second-to-last value before Encoding
        # Style line ends with: ...,MarginL,MarginR,MarginV,Encoding
        # We need to find the MarginV which is 2nd from end
        style_line_match = re.search(r'Style:\s*Default,(.+)', self.header_template)
        if style_line_match:
            parts = style_line_match.group(1).split(',')
            if len(parts) >= 3:
                try:
                    # MarginV is at index -2 (second from last)
                    self.margin_v = int(parts[-2])
                except (ValueError, IndexError):
                    pass

    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel("Drag the text up/down to position subtitles, or use the Font Size control")
        instructions.setStyleSheet("font-weight: bold; color: #0078d4;")
        layout.addWidget(instructions)

        # Frame display
        self.frame_label = SubtitleFrameLabel()
        self.frame_label.setMinimumSize(400, 225)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setStyleSheet("background-color: #1e1e1e;")
        self.frame_label.set_ass_params(
            self.play_res_x, self.play_res_y,
            self.margin_v, self.font_size, self.font_name
        )
        self.frame_label.margin_changed.connect(self.on_margin_changed)
        layout.addWidget(self.frame_label)

        # Timeline controls
        timeline_layout = QHBoxLayout()
        timeline_layout.addWidget(QLabel("Position:"))

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 10000)
        self.timeline_slider.setValue(self.initial_timeline_position)
        self.timeline_slider.valueChanged.connect(self.on_timeline_changed)
        timeline_layout.addWidget(self.timeline_slider)

        self.time_label = QLabel("0:00.00")
        self.time_label.setMinimumWidth(80)
        timeline_layout.addWidget(self.time_label)

        layout.addLayout(timeline_layout)

        # Episode selector
        episode_layout = QHBoxLayout()
        episode_layout.addWidget(QLabel("Episode:"))

        self.episode_combo = QComboBox()
        self.episode_combo.addItems([Path(f).name for f in self.mkv_files])
        self.episode_combo.currentIndexChanged.connect(self.load_episode)
        episode_layout.addWidget(self.episode_combo)
        episode_layout.addStretch()

        layout.addLayout(episode_layout)

        # Font size and MarginV controls
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Font Size:"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(16, 100)
        self.font_size_spin.setValue(self.font_size)
        self.font_size_spin.valueChanged.connect(self.on_font_size_changed)
        controls_layout.addWidget(self.font_size_spin)

        controls_layout.addSpacing(40)

        controls_layout.addWidget(QLabel("MarginV:"))
        self.margin_label = QLabel(str(self.margin_v))
        self.margin_label.setMinimumWidth(60)
        self.margin_label.setStyleSheet("font-family: monospace; font-weight: bold;")
        controls_layout.addWidget(self.margin_label)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
            }
        """)
        apply_btn.clicked.connect(self.on_apply_clicked)
        button_layout.addWidget(apply_btn)

        layout.addLayout(button_layout)

    def get_available_frame_size(self, pixmap: QPixmap) -> tuple[int, int]:
        """Calculate optimal frame size based on pixmap aspect ratio and available space."""
        from PyQt6.QtWidgets import QApplication

        # Get screen dimensions
        screen = QApplication.primaryScreen().availableGeometry()
        max_dialog_height = int(screen.height() * 0.8)

        # Available width = dialog width minus margins (layout margins ~20px total)
        available_width = self.width() - 40

        # Calculate height based on aspect ratio
        aspect_ratio = pixmap.height() / pixmap.width()
        optimal_height = int(available_width * aspect_ratio)

        # Estimate height of other UI elements (instructions, timeline, buttons, etc.)
        ui_overhead = 200

        # Cap frame height if dialog would exceed 80% screen height
        max_frame_height = max_dialog_height - ui_overhead
        if optimal_height > max_frame_height:
            optimal_height = max_frame_height
            available_width = int(optimal_height / aspect_ratio)

        return (available_width, optimal_height)

    def load_episode(self, index: int):
        """Load episode and extract initial frame."""
        if 0 <= index < len(self.mkv_files):
            mkv_path = self.mkv_files[index]
            try:
                self.current_duration = get_video_duration(mkv_path)
            except:
                self.current_duration = 600  # Default 10 minutes
            self.extract_frame()

    def on_timeline_changed(self, value: int):
        """Update time label and trigger debounced frame extraction."""
        total_seconds = (value / 10000) * self.current_duration
        minutes = int(total_seconds) // 60
        secs = int(total_seconds) % 60
        centisecs = int((total_seconds - int(total_seconds)) * 100)
        self.time_label.setText(f"{minutes}:{secs:02d}.{centisecs:02d}")

        # Debounce: wait 300ms after slider stops moving before extracting
        self.extract_timer.start(300)

    def extract_frame(self):
        """Extract frame at current timeline position."""
        if not self.mkv_files:
            return

        mkv_path = self.mkv_files[self.episode_combo.currentIndex()]
        position = self.timeline_slider.value()
        total_seconds = (position / 10000) * self.current_duration

        hours = int(total_seconds) // 3600
        minutes = (int(total_seconds) % 3600) // 60
        secs = int(total_seconds) % 60
        centisecs = int((total_seconds - int(total_seconds)) * 100)
        timestamp = f"{hours:02d}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

        frame_path = Path(self.temp_dir) / "position_frame.png"

        try:
            extract_frame(mkv_path, timestamp, str(frame_path))

            if frame_path.exists():
                pixmap = QPixmap(str(frame_path))

                # Calculate and set optimal frame size
                optimal_width, optimal_height = self.get_available_frame_size(pixmap)
                self.frame_label.setFixedSize(optimal_width, optimal_height)

                self.frame_label.set_frame(pixmap)

                # Only auto-resize dialog on initial frame load
                if not self.initial_resize_done:
                    self.adjustSize()
                    self.initial_resize_done = True
        except Exception as e:
            print(f"Failed to extract frame: {e}")

    def on_margin_changed(self, margin_v: int):
        """Handle margin change from drag."""
        self.margin_v = margin_v
        self.margin_label.setText(str(margin_v))

    def on_font_size_changed(self, value: int):
        """Handle font size change."""
        self.font_size = value
        self.frame_label.set_font_size(value)

    def on_apply_clicked(self):
        """Apply position selection and close."""
        self.position_selected.emit(self.margin_v, self.font_size)
        self.accept()

    def get_timeline_position(self) -> int:
        """Return the current timeline slider position."""
        return self.timeline_slider.value()

    def closeEvent(self, event):
        """Clean up temp files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
        super().closeEvent(event)

    def resizeEvent(self, event):
        """Handle dialog resize - adjust frame size to fill width."""
        super().resizeEvent(event)
        if hasattr(self, 'frame_label') and self.frame_label.original_pixmap:
            optimal_width, optimal_height = self.get_available_frame_size(self.frame_label.original_pixmap)
            self.frame_label.setFixedSize(optimal_width, optimal_height)
            self.frame_label.set_frame(self.frame_label.original_pixmap)
