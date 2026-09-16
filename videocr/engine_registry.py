"""Process-wide OCR engine registry.

Model construction (`videocr.utils.create_ocr_engine` /
`create_detection_engine`) is expensive -- 5-10 seconds per engine, mostly
weight loading. Historically it was paid once per parallel worker AND once
per OCR'd time range within a worker, and once more for the label scanner's
detection-only engine even when it duplicates a model already loaded for
recognition. Measured: sharing one engine across ranges cut a 3-minute range
from 21.3s to 16.4s, and sharing across four parallel workers cut their
combined load cost from 8.45s to 0.80s (~1.5 GB VRAM saved). Throughput was
unaffected: four threads sharing one engine measured the same per-image
inference time as four threads with four engines, because PaddleOCR
serialises inference internally.

This module memoises engine construction on the full argument tuple, so
distinct configurations (different language, different model dirs, CPU vs
GPU) still get their own engine, while repeated calls with identical
arguments -- across time ranges, across files, across worker threads --
share one instance.

Construction is guarded by a single lock, held across the build, so
concurrent callers requesting the same (not-yet-built) engine block on the
one build in progress rather than racing to build duplicates. The lock is
only used for construction; callers that want to serialise *inference* may
take it themselves via `engine_lock()`, but the getters below never hold it
beyond their own construction step.
"""

from __future__ import annotations

import threading

from . import utils

_lock = threading.Lock()
_ocr_engines: dict[tuple, object] = {}
_detection_engines: dict[tuple, object] = {}


def _build_ocr_engine(lang, det_model_dir, rec_model_dir, use_gpu):
    """Builder the registry calls to construct a full OCR engine.

    A thin wrapper around `videocr.utils.create_ocr_engine` so tests can
    monkeypatch construction without touching PaddleOCR itself.
    """
    return utils.create_ocr_engine(lang, det_model_dir, rec_model_dir, use_gpu)


def _build_detection_engine(det_model_dir, use_gpu):
    """Builder the registry calls to construct a detection-only engine.

    A thin wrapper around `videocr.utils.create_detection_engine` so tests
    can monkeypatch construction without touching PaddleOCR itself.
    """
    return utils.create_detection_engine(det_model_dir, use_gpu)


def get_ocr_engine(lang, det_model_dir, rec_model_dir, use_gpu):
    """Return the shared OCR engine for this argument tuple.

    Builds it on first request; every later call with the same arguments,
    from any thread, gets the same instance. Safe to call concurrently --
    only one build ever happens per distinct key.
    """
    key = (lang, det_model_dir, rec_model_dir, use_gpu)
    engine = _ocr_engines.get(key)
    if engine is not None:
        return engine
    with _lock:
        # Re-check: another thread may have built it while we waited.
        engine = _ocr_engines.get(key)
        if engine is None:
            engine = _build_ocr_engine(lang, det_model_dir, rec_model_dir, use_gpu)
            _ocr_engines[key] = engine
    return engine


def get_detection_engine(det_model_dir, use_gpu):
    """Return the shared detection-only engine for this argument tuple.

    Cached separately from `get_ocr_engine` -- same construction pattern,
    distinct namespace, since a detection-only engine is a different object
    even when built from the same model directory.
    """
    key = (det_model_dir, use_gpu)
    engine = _detection_engines.get(key)
    if engine is not None:
        return engine
    with _lock:
        engine = _detection_engines.get(key)
        if engine is None:
            engine = _build_detection_engine(det_model_dir, use_gpu)
            _detection_engines[key] = engine
    return engine


def engine_lock() -> threading.Lock:
    """The registry's construction lock, exposed for callers that also want
    to serialise inference around a shared engine. The getters above never
    hold this beyond their own construction step."""
    return _lock


def reset_registry() -> None:
    """Clear all cached engines. Tests only."""
    with _lock:
        _ocr_engines.clear()
        _detection_engines.clear()
