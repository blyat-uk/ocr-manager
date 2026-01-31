"""Label detection settings dialog."""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QLineEdit, QPushButton, QFormLayout, QCheckBox)
from PyQt6.QtCore import pyqtSignal


class LabelSettingsDialog(QDialog):
    """Dialog for configuring label detection parameters."""

    settings_changed = pyqtSignal(dict)

    def __init__(self, current_settings: dict, parent=None):
        super().__init__(parent)
        self.current_settings = current_settings.copy()

        self.setWindowTitle("Label Settings")
        self.setMinimumWidth(350)
        self.setup_ui()

    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Labels Only checkbox
        self.labels_only_check = QCheckBox("Labels Only (skip dialogue detection)")
        self.labels_only_check.setChecked(
            self.current_settings.get('labels_only', False)
        )
        layout.addWidget(self.labels_only_check)

        # Form layout for numeric settings
        form = QFormLayout()
        form.setSpacing(8)

        self.min_duration_input = QLineEdit(
            str(self.current_settings.get('label_min_duration', '1.0'))
        )
        self.min_duration_input.setPlaceholderText("seconds")
        form.addRow("Min Duration:", self.min_duration_input)

        self.max_duration_input = QLineEdit(
            str(self.current_settings.get('label_max_duration', '8.0'))
        )
        self.max_duration_input.setPlaceholderText("seconds")
        form.addRow("Max Duration:", self.max_duration_input)

        self.conf_threshold_input = QLineEdit(
            str(self.current_settings.get('label_conf_threshold', '95'))
        )
        self.conf_threshold_input.setPlaceholderText("0-100")
        form.addRow("Confidence:", self.conf_threshold_input)

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
            'labels_only': self.labels_only_check.isChecked(),
            'label_min_duration': self.min_duration_input.text().strip() or '1.0',
            'label_max_duration': self.max_duration_input.text().strip() or '8.0',
            'label_conf_threshold': self.conf_threshold_input.text().strip() or '95',
        }
        self.settings_changed.emit(settings)
        self.accept()
