"""Unified settings dialog combining OCR, Label, and Autodetect settings."""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QLineEdit, QPushButton, QFormLayout, QGroupBox,
                              QCheckBox)
from PyQt6.QtCore import pyqtSignal


class SettingsDialog(QDialog):
    """Unified settings dialog with OCR, Labels, and Autodetect sections."""

    settings_changed = pyqtSignal(dict, dict, dict, dict)  # ocr, label, autodetect, automation

    def __init__(self, ocr_settings: dict, label_settings: dict,
                 autodetect_settings: dict, automation_settings: dict | None = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(380)

        self._ocr = ocr_settings.copy()
        self._label = label_settings.copy()
        self._autodetect = autodetect_settings.copy()
        self._automation = (automation_settings or {}).copy()

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # --- Automation group ---
        auto_group = QGroupBox("Automation")
        auto_form = QFormLayout(auto_group)
        auto_form.setSpacing(8)

        self.detection_batch_input = QLineEdit(
            str(self._automation.get('detection_batch_size', '10'))
        )
        self.detection_batch_input.setFixedWidth(60)
        auto_form.addRow("Detection Batch Size:", self.detection_batch_input)

        def _pct_value(key, default_frac):
            """Convert stored float string to integer percentage for display."""
            try:
                return str(int(round(float(self._automation.get(key, str(default_frac))) * 100)))
            except (ValueError, TypeError):
                return str(int(round(default_frac * 100)))

        self.crop_width_input = QLineEdit(_pct_value('crop_width_fraction', 0.70))
        self.crop_width_input.setFixedWidth(60)
        cw_row = QHBoxLayout()
        cw_row.addWidget(self.crop_width_input)
        cw_row.addWidget(QLabel("%"))
        cw_row.addStretch()
        auto_form.addRow("Crop Width:", cw_row)

        self.vert_padding_input = QLineEdit(_pct_value('crop_vertical_padding', 0))
        self.vert_padding_input.setFixedWidth(60)
        vp_row = QHBoxLayout()
        vp_row.addWidget(self.vert_padding_input)
        vp_row.addWidget(QLabel("%"))
        vp_row.addStretch()
        auto_form.addRow("Vertical Padding:", vp_row)

        self.min_crop_height_input = QLineEdit(_pct_value('crop_min_height_fraction', 0.05))
        self.min_crop_height_input.setFixedWidth(60)
        mh_row = QHBoxLayout()
        mh_row.addWidget(self.min_crop_height_input)
        mh_row.addWidget(QLabel("%"))
        mh_row.addStretch()
        auto_form.addRow("Min Crop Height:", mh_row)

        self.bottom_cutoff_input = QLineEdit(_pct_value('bottom_half_cutoff', 0.50))
        self.bottom_cutoff_input.setFixedWidth(60)
        bc_row = QHBoxLayout()
        bc_row.addWidget(self.bottom_cutoff_input)
        bc_row.addWidget(QLabel("%"))
        bc_row.addStretch()
        auto_form.addRow("Bottom Half Cutoff:", bc_row)

        layout.addWidget(auto_group)

        # --- OCR group ---
        ocr_group = QGroupBox("OCR")
        ocr_form = QFormLayout(ocr_group)
        ocr_form.setSpacing(8)

        self.ocr_lang_input = QLineEdit(self._ocr.get('ocr_lang', 'ch'))
        self.ocr_lang_input.setPlaceholderText("e.g., ch, en, japan")
        ocr_form.addRow("Language:", self.ocr_lang_input)

        self.conf_threshold_input = QLineEdit(str(self._ocr.get('conf_threshold', '95')))
        self.conf_threshold_input.setPlaceholderText("0-100")
        ocr_form.addRow("Confidence:", self.conf_threshold_input)

        self.sim_threshold_input = QLineEdit(str(self._ocr.get('sim_threshold', '82')))
        self.sim_threshold_input.setPlaceholderText("0-100")
        ocr_form.addRow("Similarity:", self.sim_threshold_input)

        self.similar_image_input = QLineEdit(str(self._ocr.get('similar_image', '0.3')))
        self.similar_image_input.setPlaceholderText("0.0-1.0")
        ocr_form.addRow("Similar Image:", self.similar_image_input)

        layout.addWidget(ocr_group)

        # --- Labels group ---
        labels_group = QGroupBox("Labels")
        labels_form = QFormLayout(labels_group)
        labels_form.setSpacing(8)

        self.label_min_duration_input = QLineEdit(
            str(self._label.get('label_min_duration', '0.5'))
        )
        self.label_min_duration_input.setPlaceholderText("seconds")
        labels_form.addRow("Min Duration:", self.label_min_duration_input)

        self.label_max_duration_input = QLineEdit(
            str(self._label.get('label_max_duration', '5.0'))
        )
        self.label_max_duration_input.setPlaceholderText("seconds")
        labels_form.addRow("Max Duration:", self.label_max_duration_input)

        self.label_conf_input = QLineEdit(
            str(self._label.get('label_conf_threshold', '95'))
        )
        self.label_conf_input.setPlaceholderText("0-100")
        labels_form.addRow("Confidence:", self.label_conf_input)

        self.label_conf_min_input = QLineEdit(
            str(self._label.get('label_conf_threshold_min', '80'))
        )
        self.label_conf_min_input.setPlaceholderText("0-100")
        labels_form.addRow("Confidence Min.:", self.label_conf_min_input)

        layout.addWidget(labels_group)

        # --- Autodetect group ---
        autodetect_group = QGroupBox("Autodetect")
        autodetect_form = QFormLayout(autodetect_group)
        autodetect_form.setSpacing(8)

        self.min_segment_input = QLineEdit(
            str(self._autodetect.get('min_segment_length', '30'))
        )
        self.min_segment_input.setPlaceholderText("seconds")
        self.min_segment_input.setFixedWidth(80)

        # Add "s" suffix label inline
        seg_row = QHBoxLayout()
        seg_row.addWidget(self.min_segment_input)
        seg_row.addWidget(QLabel("s"))
        seg_row.addStretch()
        autodetect_form.addRow("Min Segment Length:", seg_row)

        self.merge_silences_checkbox = QCheckBox("Merge repeating silences")
        self.merge_silences_checkbox.setChecked(
            self._autodetect.get('merge_repeating_silences', 'false').lower() == 'true'
        )
        self.merge_silences_checkbox.setToolTip(
            "Bridge silence gaps between detected repeating segments "
            "when the gap appears at the same position across multiple files"
        )
        autodetect_form.addRow("", self.merge_silences_checkbox)

        layout.addWidget(autodetect_group)

        layout.addStretch()

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_apply)
        btn_layout.addWidget(apply_btn)

        layout.addLayout(btn_layout)

    def _pct_to_frac(self, text: str, default: str) -> str:
        """Convert a percentage string to a 0-1 float string."""
        try:
            return str(round(int(text) / 100, 2))
        except (ValueError, TypeError):
            return default

    def _on_apply(self):
        ocr = {
            'ocr_lang': self.ocr_lang_input.text().strip() or 'ch',
            'conf_threshold': self.conf_threshold_input.text().strip() or '95',
            'sim_threshold': self.sim_threshold_input.text().strip() or '82',
            'similar_image': self.similar_image_input.text().strip() or '0.3',
        }
        label = {
            'label_min_duration': self.label_min_duration_input.text().strip() or '0.5',
            'label_max_duration': self.label_max_duration_input.text().strip() or '5.0',
            'label_conf_threshold': self.label_conf_input.text().strip() or '95',
            'label_conf_threshold_min': self.label_conf_min_input.text().strip() or '80',
        }
        autodetect = {
            'min_segment_length': self.min_segment_input.text().strip() or '30',
            'merge_repeating_silences': str(self.merge_silences_checkbox.isChecked()).lower(),
        }
        automation = {
            'detection_batch_size': self.detection_batch_input.text().strip() or '10',
            'crop_width_fraction': self._pct_to_frac(self.crop_width_input.text().strip(), '0.70'),
            'crop_vertical_padding': self._pct_to_frac(self.vert_padding_input.text().strip(), '0'),
            'crop_min_height_fraction': self._pct_to_frac(self.min_crop_height_input.text().strip(), '0.05'),
            'bottom_half_cutoff': self._pct_to_frac(self.bottom_cutoff_input.text().strip(), '0.50'),
        }
        self.settings_changed.emit(ocr, label, autodetect, automation)
        self.accept()
