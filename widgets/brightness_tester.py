"""Brightness tester dialog with visual previews."""
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QSlider, QComboBox, QSpinBox, QScrollArea,
                              QWidget, QMessageBox, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPoint, QPointF
from PyQt6.QtGui import QPixmap, QPainter, QWheelEvent, QMouseEvent, QImage

from core.video_utils import get_video_duration, extract_frame
from core.image_processing import apply_brightness_threshold


class ZoomableImageLabel(QLabel):
    """Label that supports zooming and panning."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.original_pixmap = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.01
        self.max_zoom = 10.0
        self.pan_offset = QPointF(0, 0)

        # For panning
        self.panning = False
        self.last_pan_point = QPoint()

    def set_image(self, pixmap: QPixmap):
        """Set the image to display."""
        self.original_pixmap = pixmap
        self.update_display()

    def calculate_fit_zoom(self):
        """Calculate zoom factor to fit image within viewport."""
        if self.original_pixmap is None:
            return 1.0

        # Calculate scale factors for width and height
        width_scale = self.width() / self.original_pixmap.width()
        height_scale = self.height() / self.original_pixmap.height()

        # Use the smaller scale to ensure the entire image fits
        fit_zoom = min(width_scale, height_scale)

        # Clamp to min/max zoom
        return max(self.min_zoom, min(self.max_zoom, fit_zoom))

    def reset_view(self):
        """Reset zoom and pan to fit viewport."""
        self.zoom_factor = self.calculate_fit_zoom()
        self.pan_offset = QPointF(0, 0)
        self.update_display()

    def update_display(self):
        """Update the displayed image with current zoom and pan."""
        if self.original_pixmap is None:
            return

        # Calculate scaled size
        scaled_width = int(self.original_pixmap.width() * self.zoom_factor)
        scaled_height = int(self.original_pixmap.height() * self.zoom_factor)

        # Scale the pixmap
        scaled_pixmap = self.original_pixmap.scaled(
            scaled_width, scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Create a pixmap the size of the widget
        display_pixmap = QPixmap(self.size())
        display_pixmap.fill(Qt.GlobalColor.black)

        # Calculate position with pan offset
        x = int((self.width() - scaled_pixmap.width()) / 2 + self.pan_offset.x())
        y = int((self.height() - scaled_pixmap.height()) / 2 + self.pan_offset.y())

        # Draw the scaled image
        painter = QPainter(display_pixmap)
        painter.drawPixmap(x, y, scaled_pixmap)
        painter.end()

        self.setPixmap(display_pixmap)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if self.original_pixmap is None:
            return

        # Get zoom direction
        delta = event.angleDelta().y()
        zoom_in = delta > 0

        # Calculate new zoom factor
        zoom_step = 1.15
        if zoom_in:
            new_zoom = self.zoom_factor * zoom_step
        else:
            new_zoom = self.zoom_factor / zoom_step

        # Clamp zoom factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))

        # Get mouse position relative to widget center
        mouse_pos = event.position()
        center = QPointF(self.width() / 2, self.height() / 2)
        mouse_offset = mouse_pos - center

        # Adjust pan offset to zoom towards mouse position
        zoom_ratio = new_zoom / self.zoom_factor
        self.pan_offset = self.pan_offset * zoom_ratio - mouse_offset * (zoom_ratio - 1)

        self.zoom_factor = new_zoom
        self.update_display()

    def mousePressEvent(self, event: QMouseEvent):
        """Start panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.panning = True
            self.last_pan_point = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle panning."""
        if self.panning:
            delta = event.pos() - self.last_pan_point
            self.pan_offset += QPointF(delta.x(), delta.y())
            self.last_pan_point = event.pos()
            self.update_display()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Stop panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def resizeEvent(self, event):
        """Handle resize."""
        super().resizeEvent(event)
        self.update_display()


class BrightnessTesterDialog(QDialog):
    """Dialog for testing brightness levels with visual previews."""

    brightness_selected = pyqtSignal(int)

    def __init__(self, mkv_files: list, selected_episode: int = 0, timeline_position: int = 5000,
                 brightness: int = 230, crop_region: tuple = None, target_file: str = None,
                 subtitle_positions: dict[str, int] = None,
                 durations: dict[str, float] = None,
                 existing_crops: dict[str, tuple] = None, parent=None):
        super().__init__(parent)
        self.mkv_files = sorted(mkv_files)
        self.subtitle_positions = subtitle_positions or {}  # filename -> slider position
        self.durations = durations or {}  # filename -> cached duration
        self.existing_crops = existing_crops or {}  # filename -> (x, y, w, h)
        self.target_file = target_file  # Pre-select this file if specified
        self.initial_timeline_position = timeline_position
        self.initial_brightness = brightness
        self.crop_region = crop_region  # (x, y, width, height) or None
        self.current_duration = 0

        # Find target file index if specified, otherwise use selected_episode
        if target_file:
            self.selected_episode = 0
            for i, f in enumerate(self.mkv_files):
                if Path(f).name == target_file or f == target_file:
                    self.selected_episode = i
                    break
        else:
            self.selected_episode = selected_episode

        # Create temp directory
        self.temp_dir = Path(f"/tmp/translator-{uuid.uuid4().hex[:8]}")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.current_frame = None  # Path to the extracted (and possibly cropped) frame
        self.current_frame_array = None  # numpy array of the frame for preview generation
        self.selected_brightness = brightness
        self.initial_resize_done = False  # Track if initial auto-resize has been done

        # Carousel state
        self.preview_images = []  # List of (brightness, pixmap) tuples
        self.current_preview_index = 0
        self.has_generated_previews = False  # Track if we've generated before

        # Debounce timer for auto frame extraction
        self.extract_timer = QTimer(self)
        self.extract_timer.setSingleShot(True)
        self.extract_timer.timeout.connect(self.extract_current_frame)

        self.setWindowTitle("Test Brightness")
        self.resize(1400, 900)
        self.setup_ui()

    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel("Move slider to find a frame with subtitles, then generate previews to find optimal brightness")
        layout.addWidget(instructions)

        # Frame preview
        self.frame_preview = QLabel()
        self.frame_preview.setMinimumSize(400, 225)
        self.frame_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.frame_preview)

        # Timeline slider
        timeline_layout = QHBoxLayout()
        timeline_layout.addWidget(QLabel("Position:"))

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 10000)
        self.timeline_slider.setValue(self.initial_timeline_position)  # Already on 0-10000 scale
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
        self.episode_combo.setCurrentIndex(min(self.selected_episode, len(self.mkv_files) - 1))
        self.episode_combo.currentIndexChanged.connect(self.on_episode_changed)
        episode_layout.addWidget(self.episode_combo)
        episode_layout.addStretch()

        layout.addLayout(episode_layout)

        # Brightness controls
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Brightness:"))

        self.brightness_spin = QSpinBox()
        self.brightness_spin.setRange(0, 255)
        self.brightness_spin.setValue(self.initial_brightness)
        control_layout.addWidget(self.brightness_spin)

        control_layout.addWidget(QLabel("Range:"))
        self.range_spin = QSpinBox()
        self.range_spin.setRange(1, 50)
        self.range_spin.setValue(5)
        control_layout.addWidget(self.range_spin)

        generate_btn = QPushButton("Generate Previews")
        generate_btn.clicked.connect(self.on_generate_previews_clicked)
        control_layout.addWidget(generate_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # Carousel preview section
        carousel_label = QLabel("Preview Carousel (scroll to zoom, drag to pan, arrows to navigate):")
        layout.addWidget(carousel_label)

        # Carousel container
        carousel_container = QWidget()
        carousel_layout = QHBoxLayout(carousel_container)
        carousel_layout.setContentsMargins(0, 0, 0, 0)

        # Left arrow
        self.prev_btn = QPushButton("<")
        self.prev_btn.setFixedSize(50, 100)
        self.prev_btn.clicked.connect(self.show_previous_preview)
        self.prev_btn.setEnabled(False)
        carousel_layout.addWidget(self.prev_btn)

        # Zoomable image
        self.carousel_image = ZoomableImageLabel()
        self.carousel_image.setMinimumSize(800, 400)
        carousel_layout.addWidget(self.carousel_image, 1)

        # Right arrow
        self.next_btn = QPushButton(">")
        self.next_btn.setFixedSize(50, 100)
        self.next_btn.clicked.connect(self.show_next_preview)
        self.next_btn.setEnabled(False)
        carousel_layout.addWidget(self.next_btn)

        layout.addWidget(carousel_container, 1)

        # Brightness indicator
        self.brightness_indicator = QLabel("Brightness: --")
        self.brightness_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.brightness_indicator)

        # Navigation indicator
        self.nav_indicator = QLabel("")
        self.nav_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.nav_indicator)

        # Buttons
        button_layout = QHBoxLayout()

        reset_zoom_btn = QPushButton("Reset Zoom")
        reset_zoom_btn.clicked.connect(self.carousel_image.reset_view)
        button_layout.addWidget(reset_zoom_btn)

        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply Selected Brightness")
        apply_btn.clicked.connect(self.on_apply_clicked)
        button_layout.addWidget(apply_btn)

        layout.addLayout(button_layout)

        # Load initial episode and extract frame
        if self.mkv_files:
            self.load_episode(self.episode_combo.currentIndex())
            # Update time label for initial position
            self.on_timeline_changed(self.initial_timeline_position)
            # Auto-generate previews after a short delay to ensure frame is ready
            QTimer.singleShot(100, self.on_generate_previews_clicked)

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

        # Estimate height of other UI elements (instructions, timeline, carousel, buttons, etc.)
        ui_overhead = 450  # More overhead for brightness tester due to carousel

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
            filename = Path(mkv_path).name

            # Update crop region for this episode
            if filename in self.existing_crops:
                self.crop_region = self.existing_crops[filename]
            else:
                self.crop_region = None

            if filename in self.durations and self.durations[filename] > 0:
                self.current_duration = int(self.durations[filename])
            else:
                try:
                    self.current_duration = get_video_duration(mkv_path)
                except Exception:
                    self.current_duration = 600  # Default 10 minutes

            # Jump to detected subtitle position if available for this episode
            if filename in self.subtitle_positions:
                pos = self.subtitle_positions[filename]
                self.timeline_slider.blockSignals(True)
                self.timeline_slider.setValue(pos)
                self.timeline_slider.blockSignals(False)
                self.on_timeline_changed(pos)

            self.extract_current_frame()
            self.on_generate_previews_clicked()

    def on_timeline_changed(self, value: int):
        """Update time label and trigger debounced frame extraction."""
        total_seconds = (value / 10000) * self.current_duration
        minutes = int(total_seconds) // 60
        secs = int(total_seconds) % 60
        centisecs = int((total_seconds - int(total_seconds)) * 100)
        self.time_label.setText(f"{minutes}:{secs:02d}.{centisecs:02d}")

        # Debounce: wait 300ms after slider stops moving
        self.extract_timer.start(300)

    def on_episode_changed(self, index: int):
        """Handle episode change."""
        self.load_episode(index)

    def extract_current_frame(self):
        """Extract frame at current timeline position, applying crop if set."""
        if not self.mkv_files:
            return

        episode_idx = self.episode_combo.currentIndex()
        mkv_path = self.mkv_files[episode_idx]

        position = self.timeline_slider.value()
        total_seconds = (position / 10000) * self.current_duration

        hours = int(total_seconds) // 3600
        minutes = (int(total_seconds) % 3600) // 60
        secs = int(total_seconds) % 60
        centisecs = int((total_seconds - int(total_seconds)) * 100)
        timestamp = f"{hours:02d}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

        # Extract full frame first
        full_frame_path = self.temp_dir / "full_frame.png"
        display_frame_path = self.temp_dir / "sub.png"

        try:
            extract_frame(mkv_path, timestamp, str(full_frame_path))

            if full_frame_path.exists():
                # Load frame as numpy array
                img = cv2.imread(str(full_frame_path))

                # Apply crop if region is set
                if self.crop_region is not None:
                    x, y, w, h = self.crop_region
                    # Ensure crop is within bounds
                    img_h, img_w = img.shape[:2]
                    x = max(0, min(x, img_w - 1))
                    y = max(0, min(y, img_h - 1))
                    w = min(w, img_w - x)
                    h = min(h, img_h - y)
                    img = img[y:y+h, x:x+w]

                # Save the (possibly cropped) frame for display
                cv2.imwrite(str(display_frame_path), img)

                # Store the frame array for preview generation
                self.current_frame_array = img
                self.current_frame = str(display_frame_path)

                pixmap = QPixmap(str(display_frame_path))

                # Calculate and set optimal frame size
                optimal_width, optimal_height = self.get_available_frame_size(pixmap)
                self.frame_preview.setFixedSize(optimal_width, optimal_height)

                # Scale to fill the label exactly
                scaled_pixmap = pixmap.scaled(
                    self.frame_preview.size(),
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.frame_preview.setPixmap(scaled_pixmap)

                # Store original pixmap for resize events
                self.frame_preview.original_pixmap = pixmap

                # Only auto-resize dialog on initial frame load
                if not self.initial_resize_done:
                    self.adjustSize()
                    self.initial_resize_done = True
        except Exception as e:
            print(f"Failed to extract frame: {e}")

    def on_generate_previews_clicked(self):
        """Generate brightness preview gallery in-memory."""
        if self.current_frame_array is None:
            QMessageBox.warning(self, "Error", "No frame extracted yet. Move the slider to extract a frame first.")
            return

        base_brightness = self.brightness_spin.value()
        range_value = self.range_spin.value()

        # Generate brightness values
        brightness_values = []
        for i in range(-range_value, range_value + 1):
            b = base_brightness + i
            if 0 <= b <= 255:
                brightness_values.append(b)

        # Generate preview images in-memory
        self.preview_images.clear()
        for brightness in brightness_values:
            # Apply brightness threshold
            processed = apply_brightness_threshold(self.current_frame_array, brightness)

            # Convert BGR numpy array to QPixmap
            pixmap = self._numpy_to_pixmap(processed)
            self.preview_images.append((brightness, pixmap))

        if not self.preview_images:
            QMessageBox.warning(self, "No Previews", "No preview images were generated.")
            return

        # Only reset zoom on first generation, maintain zoom for subsequent generations
        should_reset_zoom = not self.has_generated_previews
        self.has_generated_previews = True

        # Show first preview (middle one - the base brightness)
        middle_index = len(self.preview_images) // 2
        self.current_preview_index = middle_index
        self.show_current_preview(reset_zoom=should_reset_zoom)
        self.update_navigation_buttons()

    def _numpy_to_pixmap(self, img: np.ndarray) -> QPixmap:
        """Convert a BGR numpy array to QPixmap."""
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def show_current_preview(self, reset_zoom=False):
        """Display the current preview image in the carousel."""
        if not self.preview_images or self.current_preview_index >= len(self.preview_images):
            return

        brightness, pixmap = self.preview_images[self.current_preview_index]

        # Keep current zoom and pan when changing images (unless resetting)
        current_zoom = self.carousel_image.zoom_factor
        current_pan = self.carousel_image.pan_offset

        self.carousel_image.set_image(pixmap)

        if reset_zoom:
            # Calculate zoom to fit image in viewport
            fit_zoom = self.carousel_image.calculate_fit_zoom()
            self.carousel_image.zoom_factor = fit_zoom
            self.carousel_image.pan_offset = QPointF(0, 0)
        else:
            # Restore zoom and pan
            self.carousel_image.zoom_factor = current_zoom
            self.carousel_image.pan_offset = current_pan

        self.carousel_image.update_display()

        # Update indicators
        self.brightness_indicator.setText(f"Brightness: {brightness}")
        self.nav_indicator.setText(f"{self.current_preview_index + 1} / {len(self.preview_images)}")

        # Update selected brightness
        self.selected_brightness = brightness
        self.brightness_spin.setValue(brightness)

    def show_previous_preview(self):
        """Show the previous preview image."""
        if self.current_preview_index > 0:
            self.current_preview_index -= 1
            self.show_current_preview()
            self.update_navigation_buttons()

    def show_next_preview(self):
        """Show the next preview image."""
        if self.current_preview_index < len(self.preview_images) - 1:
            self.current_preview_index += 1
            self.show_current_preview()
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """Update the enabled state of navigation buttons."""
        self.prev_btn.setEnabled(self.current_preview_index > 0)
        self.next_btn.setEnabled(self.current_preview_index < len(self.preview_images) - 1)

    def on_apply_clicked(self):
        """Apply brightness selection."""
        brightness = self.brightness_spin.value()
        self.brightness_selected.emit(brightness)
        self.accept()

    def get_current_filename(self) -> str:
        """Get the filename of the currently selected episode."""
        if self.mkv_files and 0 <= self.episode_combo.currentIndex() < len(self.mkv_files):
            return Path(self.mkv_files[self.episode_combo.currentIndex()]).name
        return ""

    def closeEvent(self, event):
        """Clean up temp files."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
        super().closeEvent(event)

    def resizeEvent(self, event):
        """Handle dialog resize - adjust frame size to fill width."""
        super().resizeEvent(event)
        if hasattr(self, 'frame_preview') and hasattr(self.frame_preview, 'original_pixmap') and self.frame_preview.original_pixmap:
            pixmap = self.frame_preview.original_pixmap
            optimal_width, optimal_height = self.get_available_frame_size(pixmap)
            self.frame_preview.setFixedSize(optimal_width, optimal_height)
            scaled_pixmap = pixmap.scaled(
                self.frame_preview.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.frame_preview.setPixmap(scaled_pixmap)
