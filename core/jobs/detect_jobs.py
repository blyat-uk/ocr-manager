"""Detector, metadata, thumbnail and proof-OCR jobs for core.jobs.runner.

Each job captures only immutable inputs when it is constructed: paths, copies
of the settings it needs, and, for the proof, the exact OCR call. run()
returns a frozen result object. No job reads or writes a Project. Results
reach the model only through core.jobs.apply, on the model owner's thread
(ruling C8). Nothing here imports Qt.

Identity
    key       f"{kind}:{file}"; the folder-wide ranges analysis is "ranges:*"
              (file None). A second submit with the same key replaces a
              queued job and waits behind a running one (see the runner).
    lane      GPU: crop, brightness, proof (they lease OCR engines).
              CPU: metadata, thumbnail, ranges, audio_profile.
    priority  0; the proof is 10, because the user is waiting for it.

Engines (ruling A1)
    CropJob, BrightnessJob and ProofOcrJob reach engines only through
    videocr.engine_registry leases. The detector jobs hold theirs for the
    whole detector call; get_subtitles takes its own. Engine keys match the
    OCR pass's (default model dirs, the folder's use_gpu, and, for the full
    OCR engine, the folder's language), so the pool reuses the same instances.

Fidelity
    Detectors get exactly the documented arguments and nothing else.
    ProofOcrJob passes ocr_call_for(entry, folder, project_dir).kwargs
    unchanged, adding only time_ranges (one 30 s window) and cancel_event,
    neither of which changes what the OCR pass reads. Its lines are parsed
    from the ASS text get_subtitles returns, i.e. what a run would write
    before QA, labels included.

Cancellation convention: a cancelled job returns None
    A job whose work was cut short by a cancel request returns None. It never
    returns a partial result and never raises. The runner then delivers
    "cancelled" with result None, and every apply_* function in
    core.jobs.apply ignores None.
    - CropJob, BrightnessJob: the detector reports cancellation in its
      result (the "cancelled" flag). The job returns None instead of that
      result. CropJob also catches AudioExtractionCancelled, which
      detect_crop handles itself today, as a safety net.
    - RangesJob: AnalysisCancelled (and AudioExtractionCancelled) is caught.
    - AudioProfileJob: AudioExtractionCancelled is caught.
    - ProofOcrJob: get_subtitles stops at cancel_event. If the event is set
      when it returns, the job returns None.
    - MetadataJob, ThumbnailJob: short. They check once, before starting.
    A job that finished its work before it looked at the request returns its
    full result, and the runner reports "finished".
"""
from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2

from core.ass_qafix import ASS_TAG_RE
from core.detect import audio_profile as _audio_profile
from core.detect import brightness as _brightness
from core.detect import crop as _crop
from core.detect import ocr_view as _ocr_view
from core.detect import tiles as _tiles
from core.detect import vad as _vad
from core.detect.ranges import pipeline as _ranges
from core.detect.ranges.config import MatchConfig, RangesConfig
from core.detect.ranges.pipeline import FileEntry as RangesFile
from core.jobs.runner import JobContext, Lane
from core.project.ocr_kwargs import ocr_call_for
from videocr import api as _api
from videocr import engine_registry
from videocr import pyav_adapter as _pyav_adapter
from videocr import utils as _videocr_utils

if TYPE_CHECKING:
    import numpy as np

    from core.detect.audio_profile import AudioProfile
    from core.detect.brightness import BrightnessResult
    from core.detect.crop import CropResult
    from core.detect.ranges.pipeline import ProgressEvent, RangesAnalysis
    from core.project.model import FileEntry, FolderSettings

THUMB_HEIGHT = 72                 # px; thumbnails are whole frames scaled to this height
PROOF_WINDOW_SEC = 30.0           # length of the proof OCR window
PROOF_START_FRACTION = 0.4        # window start without a sample time, as a fraction of the duration
PROOF_PRIORITY = 10


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MetadataResult:
    file: str
    width: int
    height: int
    duration: float     # seconds, as the OCR pass counts them (ocr_view.video_timing)
    fps: float


@dataclass(frozen=True)
class ThumbnailResult:
    file: str
    time: float
    image: np.ndarray | None   # BGR, THUMB_HEIGHT rows; None when no frame could be grabbed at `time`


@dataclass(frozen=True)
class CropJobResult:
    file: str
    result: CropResult
    hint: tuple[float, float] | None     # (y_frac, h_frac) the consensus was seeded from; None: plain detection


@dataclass(frozen=True)
class BrightnessJobResult:
    file: str
    result: BrightnessResult
    tiles: dict[str, float]              # tiles.choose_tiles(result.strips, result.value)
    hint_value: int | None               # the edited file's value this re-detection checks against
    crop_box: tuple[int, int, int, int] | None   # the crop the result was measured with; apply drops it
                                                 # when the file's crop is no longer this box


@dataclass(frozen=True)
class RangesJobResult:
    analysis: RangesAnalysis


@dataclass(frozen=True)
class AudioProfileResult:
    file: str
    profile: AudioProfile


@dataclass(frozen=True)
class ProofResult:
    file: str
    window: tuple[float, float]                 # seconds; OCR ran on format_mss() of each end
    lines: list[tuple[float, float, str]]       # (start s, end s, text) of every Dialogue line get_subtitles
                                                # returned, in document order (see proof_lines)
    seconds: float                              # wall time of the OCR call


def detector_cancelled(flagged: str | None) -> bool:
    """True when a CropResult's or BrightnessResult's `flagged` says
    cancel_check cut the detection short."""
    if not flagged:
        return False
    reasons = flagged.split("+")
    return _crop.FLAG_CANCELLED in reasons or _brightness.FLAG_CANCELLED in reasons


def crop_settings(folder: FolderSettings) -> dict:
    """The `settings` dict detect_crop reads, from the folder's fields."""
    return {
        "crop_width_fraction": folder.crop_width_fraction,
        "crop_vertical_padding": folder.crop_vertical_padding,
        "crop_min_height_fraction": folder.crop_min_height_fraction,
        "bottom_half_cutoff": folder.bottom_half_cutoff,
    }


# --------------------------------------------------------------------------
# CPU lane
# --------------------------------------------------------------------------

class MetadataJob:
    """Width, height, duration and fps of one file, without ffprobe: duration
    and fps come from ocr_view.video_timing (as the OCR pass counts them),
    width and height from a videocr Capture opened only to read them."""

    kind = "metadata"
    lane = Lane.CPU
    priority = 0

    def __init__(self, project_dir: str, file: str):
        self.file = file
        self.key = f"{self.kind}:{file}"
        self.video_path = os.path.join(project_dir, file)

    def run(self, ctx: JobContext) -> MetadataResult | None:
        if ctx.cancelled():
            return None
        duration, fps = _ocr_view.video_timing(self.video_path)
        with _pyav_adapter.Capture(self.video_path) as cap:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return MetadataResult(self.file, width, height, float(duration), float(fps))


class ThumbnailJob:
    """One whole frame at `time`, THUMB_HEIGHT rows high, through the crop
    detector's fetch layer. These are not OCR pixels (see
    core/detect/__init__.py); fine for a queue thumbnail."""

    kind = "thumbnail"
    lane = Lane.CPU
    priority = 0

    def __init__(self, project_dir: str, file: str, time: float):
        self.file = file
        self.key = f"{self.kind}:{file}"
        self.video_path = os.path.join(project_dir, file)
        self.time = float(time)

    def run(self, ctx: JobContext) -> ThumbnailResult | None:
        if ctx.cancelled():
            return None
        frames = _crop.grab_frames(self.video_path, [self.time], band_frac=1.0, target_height=THUMB_HEIGHT)
        return ThumbnailResult(self.file, self.time, frames[0] if frames else None)


class RangesJob:
    """Folder-wide keep-range analysis, configured exactly as the old app's
    core/audio_analysis.py configured it, with its fingerprint cache."""

    kind = "ranges"
    lane = Lane.CPU
    priority = 0

    def __init__(self, project_dir: str, files: list[str], folder: FolderSettings):
        self.file = None
        self.key = "ranges:*"
        self.project_dir = project_dir
        self.files = tuple(files)
        self.config = RangesConfig(
            match=MatchConfig(min_length_sec=folder.min_segment_length),
            merge_repeating_silences=folder.merge_repeating_silences,
        )

    def run(self, ctx: JobContext) -> RangesJobResult | None:
        entries = [RangesFile(name=name, path=os.path.join(self.project_dir, name)) for name in self.files]
        fraction = 0.0

        def on_progress(event: ProgressEvent) -> None:
            nonlocal fraction
            if event.kind == "file":
                fraction = event.current / event.total if event.total else fraction
                ctx.progress(fraction, event.message)
            elif event.kind == "phase":
                ctx.progress(fraction, event.message)
            elif event.kind == "log":
                ctx.log(event.message)

        try:
            analysis = _ranges.analyse_detailed(
                entries, self.config, on_progress,
                cache_dir=_ranges.default_cache_dir(self.project_dir),
                cancel=ctx.cancel_check(),
            )
        except (_ranges.AnalysisCancelled, _vad.AudioExtractionCancelled):
            return None
        return RangesJobResult(analysis)


class AudioProfileJob:
    """Waveform envelope and speech spans of one file, for the time-ranges tab."""

    kind = "audio_profile"
    lane = Lane.CPU
    priority = 0

    def __init__(self, project_dir: str, file: str, duration: float):
        self.file = file
        self.key = f"{self.kind}:{file}"
        self.video_path = os.path.join(project_dir, file)
        self.duration = float(duration)

    def run(self, ctx: JobContext) -> AudioProfileResult | None:
        try:
            profile = _audio_profile.audio_profile(self.video_path, self.duration, cancel_check=ctx.cancel_check())
        except _vad.AudioExtractionCancelled:
            return None
        return AudioProfileResult(self.file, profile)


# --------------------------------------------------------------------------
# GPU lane
# --------------------------------------------------------------------------

class CropJob:
    """detect_crop on one file, with a detection engine leased for the call.

    With `hint` (the edited file's (y_frac, h_frac)), the consensus is
    [hint] * CONSENSUS_MIN_ENTRIES instead of `consensus`, so boxes that
    disagree with the user's edit get flagged (ruling C3), and the result
    applies with source HINT.
    """

    kind = "crop"
    lane = Lane.GPU
    priority = 0

    def __init__(self, project_dir: str, file: str, duration: float,
                 consensus: list[tuple[float, float]], folder: FolderSettings,
                 hint: tuple[float, float] | None = None):
        self.file = file
        self.key = f"{self.kind}:{file}"
        self.video_path = os.path.join(project_dir, file)
        self.duration = float(duration)
        self.hint = None if hint is None else (hint[0], hint[1])
        if self.hint is not None:
            consensus = [self.hint] * _crop.CONSENSUS_MIN_ENTRIES
        self.consensus = tuple((y, h) for y, h in consensus)
        self.settings = tuple(crop_settings(folder).items())
        self.use_gpu = folder.use_gpu

    def run(self, ctx: JobContext) -> CropJobResult | None:
        try:
            with engine_registry.lease_detection_engine(None, self.use_gpu) as det_engine:
                result = _crop.detect_crop(self.video_path, self.duration, det_engine,
                                           list(self.consensus), dict(self.settings), ctx.cancel_check())
        except _vad.AudioExtractionCancelled:
            return None
        if detector_cancelled(result.flagged):
            return None
        return CropJobResult(self.file, result, self.hint)


class BrightnessJob:
    """detect_brightness on one file, with a detection engine and a full OCR
    engine leased for the call, plus the review tab's zoom tiles.

    `folder_plateau` selects the cheap path. A hint re-detection
    (`hint_value`) always runs full detection (ruling C3), so passing both is
    refused. The "differs-from-hint?" flag is added when the result is
    applied, not here. The result carries the crop box it was measured with,
    so a result that arrives after the crop changed is dropped on apply.
    """

    kind = "brightness"
    lane = Lane.GPU
    priority = 0

    def __init__(self, project_dir: str, file: str, crop_box: tuple[int, int, int, int] | None,
                 time_ranges: list[tuple[str | None, str | None]] | None, folder: FolderSettings,
                 folder_plateau: tuple[int, int] | None = None, hint_value: int | None = None):
        if hint_value is not None and folder_plateau is not None:
            raise ValueError("a brightness hint re-detection runs full detection: pass folder_plateau=None")
        self.file = file
        self.key = f"{self.kind}:{file}"
        self.video_path = os.path.join(project_dir, file)
        self.crop_box = None if crop_box is None else tuple(crop_box)
        self.time_ranges = None if time_ranges is None else tuple((start, end) for start, end in time_ranges)
        self.folder_plateau = None if folder_plateau is None else (folder_plateau[0], folder_plateau[1])
        self.hint_value = hint_value
        self.ocr_lang = folder.ocr_lang
        self.use_gpu = folder.use_gpu

    def run(self, ctx: JobContext) -> BrightnessJobResult | None:
        time_ranges = None if self.time_ranges is None else list(self.time_ranges)
        with engine_registry.lease_detection_engine(None, self.use_gpu) as det_engine, \
                engine_registry.lease_ocr_engine(self.ocr_lang, None, None, self.use_gpu) as ocr_engine:
            result = _brightness.detect_brightness(self.video_path, self.crop_box, time_ranges,
                                                   det_engine, ocr_engine, self.folder_plateau,
                                                   ctx.cancel_check())
        if detector_cancelled(result.flagged):
            return None
        tiles = _tiles.choose_tiles(result.strips, result.value)
        return BrightnessJobResult(self.file, result, tiles, self.hint_value, self.crop_box)


def proof_window(sample_time: float | None, duration: float) -> tuple[float, float]:
    """(start, end) seconds of the proof window (ruling C4).

    Starts at `sample_time`, or at PROOF_START_FRACTION of the duration
    without one. Lasts PROOF_WINDOW_SEC, clamped to [0, duration]; when the
    end would pass the file's end, the start moves back instead. Raises
    ValueError when the duration is unknown (<= 0): scan metadata first.
    """
    if not duration or duration <= 0:
        raise ValueError(f"media duration unknown ({duration!r}): scan the file's metadata first")
    start = PROOF_START_FRACTION * duration if sample_time is None else float(sample_time)
    start = min(max(start, 0.0), duration)
    if start + PROOF_WINDOW_SEC > duration:
        return max(0.0, duration - PROOF_WINDOW_SEC), duration
    return start, start + PROOF_WINDOW_SEC


def proof_lines(ass: str) -> list[tuple[float, float, str]]:
    """(start seconds, end seconds, text) of every Dialogue line in `ass`, in
    document order: dialogue and positioned labels alike. Timestamps and
    fields are parsed by videocr.utils.parse_ass_dialogue_line; override tags
    ({\\pos(...)} and the like, core.ass_qafix.ASS_TAG_RE) are removed and
    \\N becomes a newline. Empty text (no Dialogue lines) gives []."""
    lines = []
    for raw in ass.split("\n"):                    # not splitlines(): OCR text may hold other separators
        parsed = _videocr_utils.parse_ass_dialogue_line(raw.rstrip("\r"))
        if parsed is None:
            continue
        text = ASS_TAG_RE.sub("", parsed["text"]).replace("\\N", "\n")
        lines.append((parsed["start_seconds"], parsed["end_seconds"], text))
    return lines


def format_mss(seconds: float) -> str:
    """"M:SS" for a time-range string, truncated to whole seconds (a preview
    window; minutes may exceed 59, which get_frame_index reads correctly)."""
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


class ProofOcrJob:
    """Real OCR of one 30 s window of a file, with the file's exact OCR call.

    The call (ocr_call_for) and the window are resolved at construction, so
    edits made after the user asked for the proof do not change it. The
    file's own time ranges are ignored: the window is the only range. Lines
    are parsed from the ASS text get_subtitles returns (proof_lines), so they
    are what a run would write before QA, labels-only folders included.
    """

    kind = "proof"
    lane = Lane.GPU
    priority = PROOF_PRIORITY

    def __init__(self, project_dir: str, entry: FileEntry, folder: FolderSettings):
        self.file = entry.name
        self.key = f"{self.kind}:{entry.name}"
        self.window = proof_window(entry.sample_time, entry.media.duration)
        self.time_range = (format_mss(self.window[0]), format_mss(self.window[1]))
        self._kwargs = copy.deepcopy(ocr_call_for(entry, folder, project_dir).kwargs)

    def run(self, ctx: JobContext) -> ProofResult | None:
        started = time.perf_counter()
        ass = _api.get_subtitles(**copy.deepcopy(self._kwargs), time_ranges=[self.time_range],
                                 cancel_event=ctx.cancel_event)
        seconds = time.perf_counter() - started
        if ctx.cancelled():
            return None
        return ProofResult(self.file, self.window, proof_lines(ass or ""), seconds)
