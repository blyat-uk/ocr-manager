"""Resources module for GUI assets."""

from pathlib import Path

RESOURCES_DIR = Path(__file__).parent
ICONS_DIR = RESOURCES_DIR / "icons"


def get_icon_path(name: str) -> Path:
    """Get path to icon file."""
    return ICONS_DIR / f"{name}.svg"
