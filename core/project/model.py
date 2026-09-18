"""Qt-free per-project domain model for OCR Manager.

This module has no PyQt6 dependency by design: it is the shared model used
by the pipeline, the store (`core/project/store.py`), migration
(`core/project/migrate.py`) and, eventually, the Stage 3 window.

`clamp_crop_box` lives here rather than in a view because a stored crop the
file's frame cannot hold is a fidelity bug, not a drawing mistake: see its
docstring.
"""
from dataclasses import dataclass, field
from enum import Enum

MIN_CROP_SIDE = 8        # video pixels; app/views/crop_view.py's CropCanvas.MIN_BOX


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


def frame_size_known(media: "Media") -> bool:
    """Whether the file's frame size has been scanned (the metadata job ran).
    Until it is, no crop can be checked against it."""
    return media.width > 0 and media.height > 0


def clamp_crop_box(box, frame_size, minimum: int = MIN_CROP_SIDE) -> tuple[int, int, int, int]:
    """`box` (x, y, width, height) as the frame can actually hold it: inside
    (0, 0, width, height), and at least `minimum` on each side unless the
    frame itself is smaller.

    Every path that stores a crop goes through this, because the OCR pass
    clamps too and does it differently: `videocr.video.infer_crop_region`
    keeps the origin and NARROWS the box, or drops it entirely when a side
    clamps to zero (the run then reads only the bottom third of the frame).
    Either way the run would read a region the stored value does not name
    and the user never reviewed. A box this returns survives that clamp
    unchanged -- pinned by
    tests/test_ocr_kwargs.py::test_a_stored_crop_is_the_region_videocr_slices.

    A frame size that is not known yet (a zero side: the metadata job has
    not run) leaves `box` alone -- there is nothing to clamp against. The
    value is re-checked when the size arrives (`core.jobs.apply.apply_metadata`).
    """
    frame_width, frame_height = int(frame_size[0]), int(frame_size[1])
    x, y, width, height = (int(value) for value in box)
    if frame_width <= 0 or frame_height <= 0:
        return (x, y, width, height)
    width = _clamp(width, min(minimum, frame_width), frame_width)
    height = _clamp(height, min(minimum, frame_height), frame_height)
    return (_clamp(x, 0, frame_width - width), _clamp(y, 0, frame_height - height), width, height)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


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
    labels_enabled: bool = True   # matches the v1 Config default (True) / its checked-by-default UI checkbox
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
