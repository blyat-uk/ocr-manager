"""Theme management for the OCR Tool GUI."""

from dataclasses import dataclass

from PyQt6.QtWidgets import QApplication, QWidget


@dataclass(frozen=True)
class CatppuccinColors:
    """Catppuccin color palette."""

    # Base colors
    base: str
    mantle: str
    crust: str

    # Surface colors
    surface0: str
    surface1: str
    surface2: str

    # Text colors
    text: str
    subtext0: str
    subtext1: str

    # Overlay colors
    overlay0: str
    overlay1: str
    overlay2: str

    # Accent colors
    rosewater: str
    flamingo: str
    pink: str
    mauve: str
    red: str
    maroon: str
    peach: str
    yellow: str
    green: str
    teal: str
    sky: str
    sapphire: str
    blue: str
    lavender: str


# Catppuccin Mocha (dark theme)
MOCHA = CatppuccinColors(
    base="#1e1e2e",
    mantle="#181825",
    crust="#11111b",
    surface0="#313244",
    surface1="#45475a",
    surface2="#585b70",
    text="#cdd6f4",
    subtext0="#a6adc8",
    subtext1="#bac2de",
    overlay0="#6c7086",
    overlay1="#7f849c",
    overlay2="#9399b2",
    rosewater="#f5e0dc",
    flamingo="#f2cdcd",
    pink="#f5c2e7",
    mauve="#cba6f7",
    red="#f38ba8",
    maroon="#eba0ac",
    peach="#fab387",
    yellow="#f9e2af",
    green="#a6e3a1",
    teal="#94e2d5",
    sky="#89dceb",
    sapphire="#74c7ec",
    blue="#89b4fa",
    lavender="#b4befe",
)


def generate_stylesheet(c: CatppuccinColors) -> str:
    """Generate QSS stylesheet from colors."""
    return f"""
/* Main Window */
QMainWindow {{
    background-color: {c.base};
    color: {c.text};
}}

QWidget {{
    background-color: {c.base};
    color: {c.text};
    font-size: 10pt;
}}

/* Labels */
QLabel {{
    background-color: transparent;
    color: {c.text};
    padding: 2px;
}}

QLabel#heading {{
    font-size: 14pt;
    font-weight: bold;
    color: {c.text};
}}

QLabel#subheading {{
    font-size: 11pt;
    color: {c.subtext0};
}}

QLabel#muted {{
    color: {c.overlay1};
}}

/* Buttons */
QPushButton {{
    background-color: {c.blue};
    color: {c.crust};
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
    min-height: 24px;
}}

QPushButton:hover {{
    background-color: {c.lavender};
}}

QPushButton:pressed {{
    background-color: {c.sapphire};
}}

QPushButton:disabled {{
    background-color: {c.surface1};
    color: {c.overlay0};
}}

QPushButton#secondary {{
    background-color: {c.surface1};
    color: {c.text};
}}

QPushButton#secondary:hover {{
    background-color: {c.surface2};
}}

QPushButton#danger {{
    background-color: {c.red};
}}

QPushButton#danger:hover {{
    background-color: {c.maroon};
}}

/* Check boxes */
QCheckBox {{
    background-color: transparent;
    color: {c.text};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid {c.surface2};
    background-color: {c.surface0};
}}

QCheckBox::indicator:hover {{
    border-color: {c.blue};
}}

QCheckBox::indicator:checked {{
    background-color: {c.blue};
    border-color: {c.blue};
}}

/* Line edits */
QLineEdit {{
    background-color: {c.surface0};
    border: 1px solid {c.surface1};
    border-radius: 6px;
    padding: 8px;
    color: {c.text};
    selection-background-color: {c.blue};
    selection-color: {c.crust};
}}

QLineEdit:hover {{
    border-color: {c.surface2};
}}

QLineEdit:focus {{
    border-color: {c.blue};
}}

/* Spin boxes */
QSpinBox, QDoubleSpinBox {{
    background-color: {c.surface0};
    border: 1px solid {c.surface1};
    border-radius: 6px;
    padding: 6px;
    color: {c.text};
}}

QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {c.surface2};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {c.blue};
}}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {c.surface1};
    border: none;
    width: 20px;
}}

QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {c.surface2};
}}

/* Group boxes */
QGroupBox {{
    background-color: {c.surface0};
    border: 1px solid {c.surface1};
    border-radius: 8px;
    margin-top: 16px;
    padding-top: 16px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    color: {c.subtext0};
}}

/* Scroll bars */
QScrollBar:vertical {{
    background-color: {c.mantle};
    width: 12px;
    margin: 0;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {c.surface1};
    min-height: 30px;
    border-radius: 6px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {c.surface2};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {c.mantle};
    height: 12px;
    margin: 0;
    border-radius: 6px;
}}

QScrollBar::handle:horizontal {{
    background-color: {c.surface1};
    min-width: 30px;
    border-radius: 6px;
    margin: 2px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {c.surface2};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* Text edit (terminal) */
QTextEdit {{
    background-color: {c.mantle};
    border: 1px solid {c.surface1};
    border-radius: 8px;
    color: {c.text};
    selection-background-color: {c.blue};
    selection-color: {c.crust};
}}

/* Splitters */
QSplitter::handle {{
    background-color: {c.surface1};
}}

QSplitter::handle:hover {{
    background-color: {c.blue};
}}

/* Tooltips */
QToolTip {{
    background-color: {c.surface0};
    color: {c.text};
    border: 1px solid {c.surface1};
    border-radius: 4px;
    padding: 6px;
}}

/* Message boxes */
QMessageBox {{
    background-color: {c.base};
}}

QMessageBox QLabel {{
    color: {c.text};
}}

/* Dialogs */
QDialog {{
    background-color: {c.base};
    color: {c.text};
}}

/* Sliders */
QSlider::groove:horizontal {{
    background-color: {c.surface1};
    height: 6px;
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background-color: {c.blue};
    height: 6px;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background-color: {c.blue};
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background-color: {c.lavender};
}}

/* Combo boxes */
QComboBox {{
    background-color: {c.surface0};
    border: 1px solid {c.surface1};
    border-radius: 6px;
    padding: 6px 12px;
    min-height: 20px;
    color: {c.text};
}}

QComboBox:hover {{
    border-color: {c.blue};
}}

QComboBox:focus {{
    border-color: {c.blue};
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {c.text};
    margin-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {c.surface0};
    border: 1px solid {c.surface1};
    border-radius: 4px;
    selection-background-color: {c.blue};
    selection-color: {c.crust};
    padding: 4px;
}}

/* Card container - elevated with border-radius */
QFrame#card {{
    background-color: {c.surface0};
    border: 1px solid {c.surface1};
    border-radius: 12px;
    padding: 16px;
}}

/* Panel section with padding */
QFrame#panel {{
    background-color: {c.surface0};
    border: 1px solid {c.surface1};
    border-radius: 8px;
    padding: 12px;
}}

/* Drop zone - dashed border container */
QFrame#drop-zone {{
    background-color: {c.surface0};
    border: 2px dashed {c.surface2};
    border-radius: 12px;
    min-height: 80px;
}}

QFrame#drop-zone:hover {{
    border-color: {c.blue};
    background-color: {c.mantle};
}}

QFrame#drop-zone[dragOver="true"] {{
    border-color: {c.green};
    background-color: {c.mantle};
}}

QLabel#drop-zone-icon {{
    font-size: 24pt;
    background-color: transparent;
}}

QLabel#drop-zone-hint {{
    color: {c.subtext0};
    font-size: 11pt;
    background-color: transparent;
}}

QLabel#drop-zone-path {{
    color: {c.text};
    background-color: transparent;
}}

/* Phase indicator styles */
QFrame#phase-badge,
QFrame#phase-badge-pending {{
    background-color: {c.surface2};
    border: 2px solid {c.surface2};
    border-radius: 12px;
}}

QFrame#phase-badge-active {{
    background-color: {c.blue};
    border: 2px solid {c.blue};
    border-radius: 12px;
}}

QFrame#phase-badge-complete {{
    background-color: {c.green};
    border: 2px solid {c.green};
    border-radius: 12px;
}}

QFrame#phase-badge-error {{
    background-color: {c.red};
    border: 2px solid {c.red};
    border-radius: 12px;
}}

QLabel#phase-label {{
    color: {c.subtext0};
    font-size: 9pt;
    background-color: transparent;
}}

QFrame#phase-connector {{
    background-color: {c.surface2};
    border: none;
    margin-top: 11px;
    margin-bottom: 22px;
}}

/* Primary action button - emphasized */
QPushButton#primary-action {{
    background-color: {c.blue};
    color: {c.crust};
    font-size: 11pt;
    min-height: 32px;
    padding: 10px 24px;
}}

QPushButton#primary-action:hover {{
    background-color: {c.lavender};
}}

QPushButton#primary-action:pressed {{
    background-color: {c.sapphire};
}}

QPushButton#primary-action:disabled {{
    background-color: {c.surface1};
    color: {c.overlay0};
}}

/* Danger action button */
QPushButton#danger-action {{
    background-color: {c.red};
    color: {c.crust};
    font-size: 11pt;
    min-height: 32px;
    padding: 10px 24px;
}}

QPushButton#danger-action:hover {{
    background-color: {c.maroon};
}}

/* Table widget */
QTableWidget {{
    background-color: {c.mantle};
    border: 1px solid {c.surface1};
    border-radius: 8px;
    gridline-color: {c.surface1};
    color: {c.text};
}}

QTableWidget::item {{
    padding: 8px;
    border: none;
}}

QTableWidget::item:alternate {{
    background-color: {c.surface0};
}}

QHeaderView::section {{
    background-color: {c.surface0};
    color: {c.subtext0};
    padding: 8px;
    border: none;
    border-bottom: 1px solid {c.surface1};
    font-weight: bold;
}}

/* Progress bar */
QProgressBar {{
    background-color: {c.surface1};
    border: none;
    border-radius: 4px;
    height: 16px;
    text-align: center;
    color: {c.text};
}}

QProgressBar::chunk {{
    background-color: {c.blue};
    border-radius: 4px;
}}
"""


def apply_theme(widget: QWidget | QApplication) -> None:
    """Apply dark theme to a widget or application."""
    stylesheet = generate_stylesheet(MOCHA)
    widget.setStyleSheet(stylesheet)
