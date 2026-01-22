"""Crop region selector dialog with frame preview."""
import tempfile
from pathlib import Path
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QSlider, QComboBox, QWidget, QSpinBox)
from PyQt6.QtCore import pyqtSignal, Qt, QPoint, QRect
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor

from core.video_utils import get_video_duration, extract_frame


class FrameLabel(QLabel):
    """Label that allows drawing a crop rectangle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.crop_rect = QRect()
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.scale_factor = 1.0
        self.pixmap_offset = QPoint(0, 0)
        self.stored_crop = None  # Store original crop coordinates for resize
        self.click_y = 0  # Store Y position of initial click for centered drawing

    def set_frame(self, pixmap: QPixmap, existing_crop: tuple = None):
        """Set the frame pixmap and optionally set existing crop."""
        self.original_pixmap = pixmap

        # Store crop coordinates if provided (for resize events)
        if existing_crop is not None:
            self.stored_crop = existing_crop

        # Scale to fill the label exactly (label is sized to match aspect ratio)
        self.scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)
        self.scale_factor = pixmap.width() / self.scaled_pixmap.width() if self.scaled_pixmap.width() > 0 else 1.0

        # No offset needed - pixmap fills entire label
        self.pixmap_offset = QPoint(0, 0)

        # Clear previous crop when loading new frame
        self.start_point = None
        self.end_point = None
        self.crop_rect = QRect()

        # If existing crop coordinates available (either passed or stored), draw them
        crop_to_draw = existing_crop if existing_crop else self.stored_crop
        if crop_to_draw and len(crop_to_draw) == 4:
            x, y, w, h = crop_to_draw
            # Convert from original video coordinates to display coordinates
            display_x = int(x / self.scale_factor)
            display_y = int(y / self.scale_factor)
            display_w = int(w / self.scale_factor)
            display_h = int(h / self.scale_factor)

            # Set start and end points
            self.start_point = QPoint(display_x, display_y)
            self.end_point = QPoint(display_x + display_w, display_y + display_h)
            self.update_display()
        else:
            self.setPixmap(self.scaled_pixmap)

    def to_pixmap_coords(self, pos: QPoint) -> QPoint:
        """Convert label coordinates to pixmap coordinates."""
        return QPoint(pos.x() - self.pixmap_offset.x(), pos.y() - self.pixmap_offset.y())

    def mousePressEvent(self, event):
        """Start drawing rectangle from horizontal center."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Clear stored crop when user starts drawing a new one
            self.stored_crop = None
            pos = self.to_pixmap_coords(event.pos())
            # Store click Y position, start X from horizontal center
            self.click_y = pos.y()
            center_x = self.scaled_pixmap.width() // 2
            self.start_point = QPoint(center_x, self.click_y)
            self.end_point = QPoint(center_x, self.click_y)
            self.drawing = True

    def mouseMoveEvent(self, event):
        """Update rectangle while drawing - expands symmetrically from center."""
        if self.drawing:
            pos = self.to_pixmap_coords(event.pos())
            center_x = self.scaled_pixmap.width() // 2
            # Symmetric horizontal expansion from center
            dx = abs(pos.x() - center_x)
            self.start_point = QPoint(center_x - dx, self.click_y)
            self.end_point = QPoint(center_x + dx, pos.y())
            self.update_display()

    def mouseReleaseEvent(self, event):
        """Finish drawing rectangle."""
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            pos = self.to_pixmap_coords(event.pos())
            center_x = self.scaled_pixmap.width() // 2
            dx = abs(pos.x() - center_x)
            self.start_point = QPoint(center_x - dx, self.click_y)
            self.end_point = QPoint(center_x + dx, pos.y())
            self.drawing = False
            self.update_display()
            # Store the crop in original video coordinates so it persists across frame changes
            self.stored_crop = self.get_crop_coordinates()

    def update_display(self):
        """Redraw the frame with the crop rectangle."""
        if self.scaled_pixmap is None:
            return

        if self.start_point and self.end_point:
            # Calculate rectangle in pixmap coordinates
            x1 = max(0, min(self.start_point.x(), self.end_point.x()))
            y1 = max(0, min(self.start_point.y(), self.end_point.y()))
            x2 = min(self.scaled_pixmap.width(), max(self.start_point.x(), self.end_point.x()))
            y2 = min(self.scaled_pixmap.height(), max(self.start_point.y(), self.end_point.y()))
            self.crop_rect = QRect(x1, y1, x2 - x1, y2 - y1)

            # Draw rectangle on pixmap
            display = QPixmap(self.scaled_pixmap)
            painter = QPainter(display)
            pen = QPen(QColor(255, 0, 0), 3)
            painter.setPen(pen)
            painter.drawRect(self.crop_rect)
            painter.end()
            self.setPixmap(display)
        else:
            self.setPixmap(self.scaled_pixmap)

    def get_crop_coordinates(self) -> tuple:
        """Get crop coordinates scaled to original video resolution."""
        if self.crop_rect.isNull():
            return (0, 0, 0, 0)

        # Scale from display coordinates to original video coordinates
        x = int(self.crop_rect.x() * self.scale_factor)
        y = int(self.crop_rect.y() * self.scale_factor)
        w = int(self.crop_rect.width() * self.scale_factor)
        h = int(self.crop_rect.height() * self.scale_factor)

        return (x, y, w, h)

    def resizeEvent(self, event):
        """Handle resize."""
        super().resizeEvent(event)
        if self.original_pixmap:
            self.set_frame(self.original_pixmap)


class CropSelectorDialog(QDialog):
    """Dialog for selecting crop region from video frame."""

    crop_selected = pyqtSignal(int, int, int, int)

    def __init__(self, mkv_files: list, existing_crop: tuple = None, timeline_position: int = 5000, parent=None):
        super().__init__(parent)
        # Sort files naturally
        self.mkv_files = sorted(mkv_files)
        self.existing_crop = existing_crop
        self.initial_timeline_position = timeline_position
        self.temp_dir = tempfile.mkdtemp()
        self.current_duration = 0
        self.initial_resize_done = False  # Track if initial auto-resize has been done

        # Debounce timer for auto frame extraction
        from PyQt6.QtCore import QTimer
        self.extract_timer = QTimer(self)
        self.extract_timer.setSingleShot(True)
        self.extract_timer.timeout.connect(self.extract_frame)

        self.setWindowTitle("Select Crop Region")
        self.resize(1200, 800)
        self.setup_ui()

        if self.mkv_files:
            self.load_episode(0)
            # Update time label for initial position
            self.on_timeline_changed(self.initial_timeline_position)

    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel("Click and drag on the frame to draw a crop box around the subtitles")
        instructions.setStyleSheet("font-weight: bold; color: #0078d4;")
        layout.addWidget(instructions)

        # Frame display
        self.frame_label = FrameLabel()
        self.frame_label.setMinimumSize(400, 225)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setStyleSheet("background-color: #1e1e1e;")
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

        # Coordinates display
        self.coords_label = QLabel("Crop: (0, 0, 0, 0) - Draw a rectangle on the frame")
        self.coords_label.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.coords_label)

        # Update coordinates periodically
        from PyQt6.QtCore import QTimer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_coordinates)
        self.update_timer.start(100)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply Crop")
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

        frame_path = Path(self.temp_dir) / "crop_frame.png"

        try:
            extract_frame(mkv_path, timestamp, str(frame_path))

            if frame_path.exists():
                pixmap = QPixmap(str(frame_path))

                # Calculate and set optimal frame size
                optimal_width, optimal_height = self.get_available_frame_size(pixmap)
                self.frame_label.setFixedSize(optimal_width, optimal_height)

                # Pass existing crop on first load only
                self.frame_label.set_frame(pixmap, self.existing_crop)
                # Clear existing_crop after first use so it doesn't re-apply on frame changes
                self.existing_crop = None

                # Only auto-resize dialog on initial frame load
                if not self.initial_resize_done:
                    self.adjustSize()
                    self.initial_resize_done = True
        except Exception as e:
            print(f"Failed to extract frame: {e}")

    def update_coordinates(self):
        """Update coordinate display."""
        x, y, w, h = self.frame_label.get_crop_coordinates()
        self.coords_label.setText(f"Crop: ({x}, {y}, {w}, {h})")

    def on_apply_clicked(self):
        """Apply crop selection and close."""
        x, y, w, h = self.frame_label.get_crop_coordinates()
        if w > 0 and h > 0:
            self.crop_selected.emit(x, y, w, h)
            self.accept()
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Selection", "Please draw a crop rectangle on the frame first")

    def get_selected_episode(self) -> int:
        """Return the currently selected episode index."""
        return self.episode_combo.currentIndex()

    def get_timeline_position(self) -> int:
        """Return the current timeline slider position (0-1000)."""
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
