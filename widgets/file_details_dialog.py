"""Read-only file details dialog showing resolved OCR settings."""

import os
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QFormLayout, QGroupBox)


@dataclass
class FileDetailsData:
    """Plain data container for rendering the file details dialog."""
    # File metadata
    filename: str = ""
    resolution_width: int = 0
    resolution_height: int = 0
    duration_seconds: float = 0.0

    # Per-file overridable settings (with override flags)
    brightness: int = 230
    brightness_is_override: bool = False
    crop: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h) or None
    crop_is_override: bool = False
    time_start: str = ""
    time_start_is_override: bool = False
    time_end: str = ""
    time_end_is_override: bool = False

    # Global-only OCR settings
    ocr_lang: str = "ch"
    conf_threshold: int = 95
    sim_threshold: int = 82
    similar_image: float = 0.3
    frames_to_skip: int = 0
    use_gpu: bool = True
    ocr_parallel: int = 4

    # Label/dialogue settings
    dialogue_enabled: bool = True
    labels_enabled: bool = True
    labels_only: bool = False
    label_min_duration: float = 0.5
    label_max_duration: float = 5.0
    label_conf_threshold: int = 95
    label_conf_threshold_min: int = 80
    mask_crops_count: int = 0


class FileDetailsDialog(QDialog):
    """Read-only dialog showing resolved settings for a file."""

    def __init__(self, data: FileDetailsData, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Details - {data.filename}")
        self.setMinimumWidth(420)
        self._data = data
        self._accent = os.environ.get('QTMATERIAL_PRIMARYCOLOR', '#ffd740')
        self._setup_ui()

    def _val_label(self, text: str, is_override: bool) -> QLabel:
        """Create a value label, styled bold+accent if overridden."""
        if is_override:
            suffix = " (custom)"
            label = QLabel(f"<b style='color:{self._accent}'>{text}{suffix}</b>")
        else:
            label = QLabel(f"{text} (default)")
        return label

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        d = self._data

        # --- File Info ---
        info_group = QGroupBox("File Info")
        info_form = QFormLayout(info_group)
        info_form.setSpacing(6)

        info_form.addRow("Filename:", QLabel(d.filename))

        if d.resolution_width > 0 and d.resolution_height > 0:
            res_text = f"{d.resolution_width}x{d.resolution_height} ({d.resolution_height}p)"
        else:
            res_text = "Unknown"
        info_form.addRow("Resolution:", QLabel(res_text))

        if d.duration_seconds > 0:
            total_sec = int(d.duration_seconds)
            minutes, secs = divmod(total_sec, 60)
            hours, minutes = divmod(minutes, 60)
            if hours > 0:
                dur_text = f"{hours}h {minutes}m {secs}s"
            else:
                dur_text = f"{minutes}m {secs}s"
        else:
            dur_text = "Unknown"
        info_form.addRow("Duration:", QLabel(dur_text))

        layout.addWidget(info_group)

        # --- Per-File Settings ---
        pf_group = QGroupBox("Per-File Settings")
        pf_form = QFormLayout(pf_group)
        pf_form.setSpacing(6)

        if d.crop is not None:
            crop_text = f"{d.crop[0]}, {d.crop[1]}, {d.crop[2]}x{d.crop[3]}"
        else:
            crop_text = "Not set"
        pf_form.addRow("Crop Region:", self._val_label(crop_text, d.crop_is_override))

        pf_form.addRow("Brightness:", self._val_label(str(d.brightness), d.brightness_is_override))
        pf_form.addRow("Time Start:", self._val_label(d.time_start or "0:00", d.time_start_is_override))
        pf_form.addRow("Time End:", self._val_label(d.time_end or "(full duration)", d.time_end_is_override))

        layout.addWidget(pf_group)

        # --- OCR Settings ---
        ocr_group = QGroupBox("OCR Settings")
        ocr_form = QFormLayout(ocr_group)
        ocr_form.setSpacing(6)

        ocr_form.addRow("Language:", QLabel(d.ocr_lang))
        ocr_form.addRow("Confidence:", QLabel(str(d.conf_threshold)))
        ocr_form.addRow("Similarity:", QLabel(str(d.sim_threshold)))
        ocr_form.addRow("Similar Image:", QLabel(str(d.similar_image)))
        ocr_form.addRow("Frames to Skip:", QLabel(str(d.frames_to_skip)))
        ocr_form.addRow("GPU:", QLabel("Yes" if d.use_gpu else "No"))
        ocr_form.addRow("Parallel Workers:", QLabel(str(d.ocr_parallel)))

        layout.addWidget(ocr_group)

        # --- Label Detection ---
        label_group = QGroupBox("Label Detection")
        label_form = QFormLayout(label_group)
        label_form.setSpacing(6)

        label_form.addRow("Dialogue:", QLabel("Yes" if d.dialogue_enabled else "No"))
        label_form.addRow("Labels:", QLabel("Yes" if d.labels_enabled else "No"))
        label_form.addRow("Labels Only:", QLabel("Yes" if d.labels_only else "No"))
        label_form.addRow("Min Duration:", QLabel(f"{d.label_min_duration}s"))
        label_form.addRow("Max Duration:", QLabel(f"{d.label_max_duration}s"))
        label_form.addRow("Confidence:", QLabel(str(d.label_conf_threshold)))
        label_form.addRow("Confidence Min.:", QLabel(str(d.label_conf_threshold_min)))
        label_form.addRow("Mask Crops:", QLabel(str(d.mask_crops_count)))

        layout.addWidget(label_group)

        # --- Close button ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
