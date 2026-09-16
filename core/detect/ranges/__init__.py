"""Automatic intro/outro/recap detection by audio fingerprinting.

Pure algorithm package (no Qt). ``analyse()`` takes a folder's episodes and
returns per-file keep ranges; it is a vectorised, cached rewrite of
``core/audio_analysis.py`` + ``core/audio_finder/`` with bit-identical
output.
"""
from core.detect.ranges.config import (
    DSPConfig,
    HashConfig,
    MatchConfig,
    PeakConfig,
    RangesConfig,
)
from core.detect.ranges.fingerprint import fingerprint_file
from core.detect.ranges.pipeline import (
    CACHE_DIRNAME,
    DEFAULT_MIN_SEGMENT_SEC,
    DEFAULT_WORKERS,
    MIN_GAP_SEC,
    AnalysisCancelled,
    FileEntry,
    ProgressEvent,
    Segment,
    SegmentMatch,
    analyse,
    default_cache_dir,
    file_identity,
)

__all__ = [
    "CACHE_DIRNAME",
    "DEFAULT_MIN_SEGMENT_SEC",
    "DEFAULT_WORKERS",
    "MIN_GAP_SEC",
    "AnalysisCancelled",
    "DSPConfig",
    "FileEntry",
    "HashConfig",
    "MatchConfig",
    "PeakConfig",
    "ProgressEvent",
    "RangesConfig",
    "Segment",
    "SegmentMatch",
    "analyse",
    "default_cache_dir",
    "file_identity",
    "fingerprint_file",
]
