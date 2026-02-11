"""Configuration management for translator application."""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FileConfig:
    """Per-file configuration overrides."""
    filename: str
    # OCR parameters (None means use global default)
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    crop_width: Optional[int] = None
    crop_height: Optional[int] = None
    brightness: Optional[int] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    # Auto-populated metadata
    resolution_width: int = 0
    resolution_height: int = 0
    duration_seconds: float = 0.0

    def has_custom_crop(self) -> bool:
        """Check if file has custom crop settings."""
        return self.crop_width is not None and self.crop_height is not None

    def has_custom_brightness(self) -> bool:
        """Check if file has custom brightness setting."""
        return self.brightness is not None

    def has_custom_time_range(self) -> bool:
        """Check if file has custom time range."""
        return self.time_start is not None or self.time_end is not None

    def has_any_custom(self) -> bool:
        """Check if file has any custom settings."""
        return self.has_custom_crop() or self.has_custom_brightness() or self.has_custom_time_range()

    def config_signature(self) -> tuple:
        """Return a hashable signature of custom config values for comparison.

        Files with the same signature have identical custom configurations.
        Returns None tuple elements for unset values to distinguish from set values.
        """
        return (
            self.crop_x, self.crop_y, self.crop_width, self.crop_height,
            self.brightness,
            self.time_start, self.time_end
        )

    def get_crop_tuple(self) -> Optional[tuple[int, int, int, int]]:
        """Get crop as tuple (x, y, w, h) or None if not set."""
        if self.has_custom_crop():
            return (self.crop_x, self.crop_y, self.crop_width, self.crop_height)
        return None

    def set_crop(self, x: int, y: int, w: int, h: int):
        """Set crop region."""
        self.crop_x = x
        self.crop_y = y
        self.crop_width = w
        self.crop_height = h

    def clear_crop(self):
        """Clear custom crop settings."""
        self.crop_x = None
        self.crop_y = None
        self.crop_width = None
        self.crop_height = None

    def clear_all(self):
        """Clear all custom settings."""
        self.crop_x = None
        self.crop_y = None
        self.crop_width = None
        self.crop_height = None
        self.brightness = None
        self.time_start = None
        self.time_end = None

    def get_resolution_label(self) -> str:
        """Get resolution label from video height (e.g., '1080p', '800p')."""
        if self.resolution_height == 0:
            return "?"
        return f"{self.resolution_height}p"


class FileConfigStore:
    """Store for managing per-file configurations."""

    def __init__(self):
        self._configs: dict[str, FileConfig] = {}

    def get(self, filename: str) -> Optional[FileConfig]:
        """Get config for a file, or None if not set."""
        return self._configs.get(filename)

    def get_or_create(self, filename: str) -> FileConfig:
        """Get config for a file, creating if it doesn't exist."""
        if filename not in self._configs:
            self._configs[filename] = FileConfig(filename=filename)
        return self._configs[filename]

    def set(self, filename: str, config: FileConfig):
        """Set config for a file."""
        self._configs[filename] = config

    def remove(self, filename: str):
        """Remove config for a file."""
        if filename in self._configs:
            del self._configs[filename]

    def has_custom(self, filename: str) -> bool:
        """Check if file has custom config."""
        config = self._configs.get(filename)
        return config is not None and config.has_any_custom()

    def clear_custom(self, filename: str):
        """Clear custom settings for a file but keep metadata."""
        config = self._configs.get(filename)
        if config:
            config.clear_all()

    def get_all_filenames(self) -> list[str]:
        """Get all filenames with configs."""
        return list(self._configs.keys())

    def get_files_by_resolution(self, target_height: int, tolerance: int = 100) -> list[str]:
        """Get all files within tolerance of target resolution height."""
        result = []
        for filename, config in self._configs.items():
            if abs(config.resolution_height - target_height) <= tolerance:
                result.append(filename)
        return result

    def copy_settings_to_files(self, source: FileConfig, target_filenames: list[str]):
        """Copy settings from source config to target files."""
        for filename in target_filenames:
            target = self.get_or_create(filename)
            # Copy custom settings but preserve metadata
            if source.has_custom_crop():
                target.crop_x = source.crop_x
                target.crop_y = source.crop_y
                target.crop_width = source.crop_width
                target.crop_height = source.crop_height
            if source.has_custom_brightness():
                target.brightness = source.brightness
            if source.has_custom_time_range():
                target.time_start = source.time_start
                target.time_end = source.time_end

    def copy_to_same_resolution(self, source_filename: str) -> list[str]:
        """Copy settings from source file to all files with same resolution.

        Returns list of filenames that were updated.
        """
        source = self._configs.get(source_filename)
        if not source or source.resolution_height == 0:
            return []

        target_files = self.get_files_by_resolution(source.resolution_height)
        # Exclude source file
        target_files = [f for f in target_files if f != source_filename]

        self.copy_settings_to_files(source, target_files)
        return target_files

    def clear(self):
        """Clear all configs."""
        self._configs.clear()


class ProjectConfigManager:
    """Manages persistent project configuration in .ocr.json files."""

    CONFIG_FILENAME = ".ocr.json"
    CURRENT_VERSION = 1

    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.config_file = self.project_path / self.CONFIG_FILENAME

    @property
    def config_path(self) -> Path:
        """Get the config file path."""
        return self.config_file

    def exists(self) -> bool:
        """Check if config file exists."""
        return self.config_file.exists()

    def load(self) -> tuple[dict, dict, dict, dict]:
        """Load config. Returns (global_settings, videocr_settings, file_configs, labels_settings).

        Returns empty dicts if file doesn't exist or is invalid.
        """
        if not self.exists():
            return {}, {}, {}, {}

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load config file: {e}")
            return {}, {}, {}, {}

        # Extract sections with defaults
        global_settings = data.get('global', {})
        videocr_settings = data.get('videocr', {})
        file_configs = data.get('files', {})
        labels_settings = data.get('labels', {})

        return global_settings, videocr_settings, file_configs, labels_settings

    def build_save_data(self, global_settings: dict, videocr_settings: dict,
                        file_store: 'FileConfigStore', labels_settings: dict = None) -> dict:
        """Build the save data dict without writing to disk.

        Returns the dict that would be saved to .ocr.json.
        """
        # Build file configs from store
        file_configs = {}
        for filename in file_store.get_all_filenames():
            config = file_store.get(filename)
            if config and config.has_any_custom():
                file_data = {}
                if config.has_custom_crop():
                    file_data['crop'] = {
                        'x': config.crop_x,
                        'y': config.crop_y,
                        'width': config.crop_width,
                        'height': config.crop_height
                    }
                if config.has_custom_brightness():
                    file_data['brightness'] = config.brightness
                if config.time_start:
                    file_data['time_start'] = config.time_start
                if config.time_end:
                    file_data['time_end'] = config.time_end
                if file_data:
                    file_configs[filename] = file_data

        data = {
            'version': self.CURRENT_VERSION,
            'global': global_settings,
            'videocr': videocr_settings,
        }

        # Only include files section if there are per-file configs
        if file_configs:
            data['files'] = file_configs

        # Labels section
        if labels_settings:
            data['labels'] = labels_settings

        return data

    def save(self, global_settings: dict, videocr_settings: dict,
             file_store: 'FileConfigStore', labels_settings: dict = None):
        """Save current configuration to .ocr.json."""
        data = self.build_save_data(global_settings, videocr_settings, file_store, labels_settings)

        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.warning(f"Failed to save config file: {e}")


@dataclass
class Config:
    """Project-specific configuration (in-memory only, not persisted)."""
    # OCR Parameters
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 0
    crop_height: int = 0
    brightness: int = 230
    time_start: str = ""
    time_end: str = ""
    ocr_parallel: int = 4

    # Label detection
    labels_enabled: bool = True
    labels_only: bool = False
    label_min_duration: float = 1.0
    label_max_duration: float = 8.0
    label_conf_threshold: int = 95

    # videocr configuration
    videocr_python: str = "/mnt/FAST/Code/videocr-PaddleOCR-original/.venv/bin/python"
    videocr_script: str = "/mnt/FAST/Code/videocr-PaddleOCR-original/videocr.py"
    ocr_lang: str = "ch"
    conf_threshold: int = 95
    sim_threshold: int = 82
    similar_image: float = 0.3
    frames_to_skip: int = 0
    use_gpu: bool = True

    # Runtime (not persisted)
    project_path: str = ""


def validate_config(config: Config) -> tuple[bool, str]:
    """Validate configuration completeness."""
    if not config.labels_only and (config.crop_width == 0 or config.crop_height == 0):
        return False, "Crop region not set"
    return True, ""
