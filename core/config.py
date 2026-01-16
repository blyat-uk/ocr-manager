"""Configuration management for translator application."""
import json
import os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path


@dataclass
class Config:
    """Project-specific configuration."""
    version: str = "1.0"

    # OCR Parameters
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 0
    crop_height: int = 0
    brightness: int = 230
    time_start: str = ""
    time_end: str = ""
    ocr_parallel: int = 4
    ocr_width: int = 1280      # Downscale width for OCR (0 = original)
    fullframe: bool = False    # OCR full frame instead of crop

    # Cleanup
    remove_credits: bool = True  # Whether to run ass-credits --yes

    # Styling
    header_template: str = ""

    # Pipeline Control
    stop_at_phase: int = -1  # -1 = run all phases, 0-7 = stop after that phase

    # Metadata
    project_path: str = ""
    last_run: str = ""


@dataclass
class GlobalConfig:
    """Global application settings."""
    fonts_directory: str = ""
    default_header_template: str = ""
    parallel_workers_ocr: int = 4
    parallel_workers_muxing: int = 10
    recent_projects: list = field(default_factory=list)
    last_project_directory: str = ""


def load_project_config(project_path: str) -> Config:
    """Load config from .translation-project/config.json."""
    config_file = Path(project_path) / ".translation-project" / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            data = json.load(f)
        # Filter out unexpected keys from old configs
        valid_keys = {f.name for f in fields(Config)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return Config(**filtered_data)
    return Config(project_path=project_path)


def save_project_config(config: Config):
    """Save config to .translation-project/config.json."""
    config_dir = Path(config.project_path) / ".translation-project"
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)


def load_global_config() -> GlobalConfig:
    """Load from XDG_CONFIG_HOME or ~/.config/translator-gui/config.json."""
    config_home = os.environ.get('XDG_CONFIG_HOME',
                                  os.path.expanduser('~/.config'))
    config_file = Path(config_home) / "translator-gui" / "config.json"

    if config_file.exists():
        with open(config_file) as f:
            data = json.load(f)
        # Filter out unexpected keys from old configs
        valid_keys = {f.name for f in fields(GlobalConfig)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return GlobalConfig(**filtered_data)
    return GlobalConfig()


def save_global_config(config: GlobalConfig):
    """Save global configuration."""
    config_home = os.environ.get('XDG_CONFIG_HOME',
                                  os.path.expanduser('~/.config'))
    config_dir = Path(config_home) / "translator-gui"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_file = config_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)


def validate_config(config: Config) -> tuple[bool, str]:
    """Validate configuration completeness."""
    if not config.fullframe:  # Only require crop if not fullframe
        if config.crop_width == 0 or config.crop_height == 0:
            return False, "Crop region not set"
    if not config.header_template:
        return False, "Header template not set"
    return True, ""


def load_header_from_translate(project_path: str) -> str | None:
    """Load header template from translate/header.txt if it exists."""
    header_file = Path(project_path) / "translate" / "header.txt"
    if header_file.exists():
        with open(header_file, 'r', encoding='utf-8') as f:
            return f.read()
    return None


def save_header_to_translate(project_path: str, header_content: str):
    """Save header template to translate/header.txt."""
    translate_dir = Path(project_path) / "translate"
    translate_dir.mkdir(exist_ok=True)
    header_file = translate_dir / "header.txt"
    with open(header_file, 'w', encoding='utf-8') as f:
        f.write(header_content)
