"""Qt-free per-project domain model for OCR Manager.

This module has no PyQt6 dependency by design: it is the shared model used
by the pipeline, the store (`core/project/store.py`), migration
(`core/project/migrate.py`) and, eventually, the Stage 3 window.
"""
from dataclasses import dataclass, field
from enum import Enum


class ReviewState(str, Enum):
    PENDING = "pending"
    PROPOSED = "proposed"
    FLAGGED = "flagged"
    REVIEWED = "reviewed"


class Source(str, Enum):
    DETECTED = "detected"
    HINT = "hint"
    MANUAL = "manual"
    IMPORTED = "imported"


@dataclass
class Crop:
    x: int
    y: int
    width: int
    height: int
    source: Source


@dataclass
class Brightness:
    value: int
    source: Source


@dataclass
class TimeRange:
    start: str | None   # "MM:SS" / "H:MM:SS"; None = start of file
    end: str | None     # None = end of file


@dataclass
class TimeRanges:
    ranges: list[TimeRange]
    source: Source


@dataclass
class Media:
    width: int = 0
    height: int = 0
    duration: float = 0.0
    fps: float = 0.0


@dataclass
class FileEntry:
    name: str
    crop: Crop | None = None
    brightness: Brightness | None = None
    time_ranges: TimeRanges | None = None          # None = whole file
    media: Media = field(default_factory=Media)
    review: ReviewState = ReviewState.PENDING
    skipped: bool = False
    sample_time: float | None = None               # seconds; crop hit_pts[0] or migrated subtitle_position
    flags: dict[str, str] = field(default_factory=dict)      # detector name -> "+"-joined flags
    evidence: dict[str, dict] = field(default_factory=dict)  # "crop" | "brightness" | "ranges" -> JSON-able dict


@dataclass
class FolderSettings:
    dialogue_enabled: bool = True
    labels_enabled: bool = True   # matches old core.config.Config default (True) / old checked-by-default UI checkbox
    ocr_lang: str = "ch"
    conf_threshold: int = 95
    sim_threshold: int = 82
    similar_image: float = 0.3
    frames_to_skip: int = 0
    use_gpu: bool = True
    label_min_duration: float = 0.5
    label_max_duration: float = 5.0
    label_conf_threshold: int = 95
    label_conf_threshold_min: int = 80
    label_mask_crops: list[tuple[int, int, int, int]] = field(default_factory=list)
    ocr_parallel: int = 4
    autopilot_enabled: bool = True
    brightness_full_detect_files: int = 3
    min_segment_length: float = 30.0
    merge_repeating_silences: bool = False
    crop_width_fraction: float = 0.70
    crop_vertical_padding: float = 0.003
    crop_min_height_fraction: float = 0.05
    bottom_half_cutoff: float = 0.55

    @property
    def labels_only(self) -> bool:
        return self.labels_enabled and not self.dialogue_enabled


@dataclass
class Project:
    path: str                                  # folder path
    folder: FolderSettings
    files: dict[str, FileEntry]                # insertion order = sorted filename order
    migrated_from_v1: bool = False             # True until the first save writes .ocr.json.v1.bak
    # Store bookkeeping, not project data: file name -> digest of the evidence
    # last loaded from or saved to its evidence cache file (core/project/store.py).
    evidence_digests: dict[str, str] = field(default_factory=dict, compare=False, repr=False)
