"""Theme management for the OCR Tool GUI using qt-material."""

import os

from qt_material import apply_stylesheet


def apply_theme(app):
    """Apply dark_amber Material Design theme to the application."""
    extra = {
        'density_scale': '-1',
        'danger': '#f38ba8',
        'warning': '#fab387',
        'success': '#a6e3a1',
    }
    apply_stylesheet(app, theme='dark_amber.xml', extra=extra)

    # Make all buttons flat by default; checked buttons stay filled (primary)
    app.setStyleSheet(app.styleSheet() + """
        QPushButton {
            background-color: transparent;
            border: none;
        }
        QPushButton:checked {
            background-color: transparent;
            border: 1px solid %s;
            color: %s;
        }
        QPushButton:checked:hover {
            background-color: rgba(255, 215, 64, 30);
        }
    """ % (
        os.environ.get('QTMATERIAL_PRIMARYCOLOR', '#ffd740'),
        os.environ.get('QTMATERIAL_PRIMARYCOLOR', '#ffd740'),
    ))


def get_theme_color(name: str) -> str:
    """Get a theme color by name from qt-material environment variables.

    Available names: primaryColor, primaryLightColor,
    secondaryColor, secondaryLightColor, secondaryDarkColor,
    primaryTextColor, secondaryTextColor.
    """
    return os.environ.get(f'QTMATERIAL_{name.upper()}', '#ffffff')
