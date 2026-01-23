"""Configuration management for translator application."""
from dataclasses import dataclass


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
    if config.crop_width == 0 or config.crop_height == 0:
        return False, "Crop region not set"
    return True, ""
