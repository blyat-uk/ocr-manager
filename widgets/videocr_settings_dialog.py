"""VideoCR settings dialog for adjusting OCR parameters."""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QLineEdit, QPushButton, QFormLayout)
from PyQt6.QtCore import pyqtSignal


class VideoCRSettingsDialog(QDialog):
    """Dialog for configuring videocr settings."""

    settings_changed = pyqtSignal(dict)

    def __init__(self, current_settings: dict, parent=None):
        super().__init__(parent)
        self.current_settings = current_settings.copy()

        self.setWindowTitle("VideoCR Settings")
        self.setMinimumWidth(350)
        self.setup_ui()

    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Instructions
        instructions = QLabel("Adjust videocr OCR parameters (temporary, resets on app restart)")
        instructions.setObjectName("muted")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Form layout for settings
        form = QFormLayout()
        form.setSpacing(8)

        # OCR Language
        self.ocr_lang_input = QLineEdit(self.current_settings.get('ocr_lang', 'ch'))
        self.ocr_lang_input.setPlaceholderText("e.g., ch, en, japan")
        form.addRow("OCR Language:", self.ocr_lang_input)

        # Confidence Threshold
        self.conf_threshold_input = QLineEdit(str(self.current_settings.get('conf_threshold', 95)))
        self.conf_threshold_input.setPlaceholderText("0-100")
        form.addRow("Confidence Threshold:", self.conf_threshold_input)

        # Similarity Threshold
        self.sim_threshold_input = QLineEdit(str(self.current_settings.get('sim_threshold', 82)))
        self.sim_threshold_input.setPlaceholderText("0-100")
        form.addRow("Similarity Threshold:", self.sim_threshold_input)

        # Similar Image
        self.similar_image_input = QLineEdit(str(self.current_settings.get('similar_image', 0.3)))
        self.similar_image_input.setPlaceholderText("0.0-1.0")
        form.addRow("Similar Image:", self.similar_image_input)

        layout.addLayout(form)
        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondary")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.on_apply_clicked)
        button_layout.addWidget(apply_btn)

        layout.addLayout(button_layout)

    def on_apply_clicked(self):
        """Apply settings and close dialog."""
        settings = {
            'ocr_lang': self.ocr_lang_input.text().strip() or 'ch',
            'conf_threshold': self.conf_threshold_input.text().strip() or '95',
            'sim_threshold': self.sim_threshold_input.text().strip() or '82',
            'similar_image': self.similar_image_input.text().strip() or '0.3',
        }
        self.settings_changed.emit(settings)
        self.accept()
