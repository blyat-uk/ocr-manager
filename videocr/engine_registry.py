"""Process-wide pool of OCR engines, handed out as exclusive leases.

Model construction (`videocr.utils.create_ocr_engine` /
`create_detection_engine`) is expensive -- 5-10 seconds per engine, mostly
weight loading -- so engines are kept and reused instead of being rebuilt for
every file, time range and label pass.

An engine is NOT safe to use from two threads at once. PaddleX's inference
path shares its input/output handles between calls, and OCRManager runs its
workers as QThreads in one process (default 4). Measured on the real models
with several threads calling `predict()` on one instance: TextDetection
changed 101 and 126 of 384 outputs versus serial (separate instances: 0 of
192, twice), and PaddleOCR misread 17 texts ("LINE 020 W" -> "NE", "20 V",
"V") while one thread died with `AssertionError: 3 != 2 for key rec_text`.

So an engine is only ever reached through a lease:

    with engine_registry.lease_ocr_engine(lang, det_dir, rec_dir, use_gpu) as ocr:
        ...every predict() call, and every iteration over what it returned...

A lease checks out an idle pooled instance for its argument key, or builds a
new one when every instance for that key is leased; on exit -- also when the
body raises -- the instance goes back to the pool for the next lessee. The
engine belongs to the lessee for the whole `with` block, so results must be
consumed (predict() returns a generator) inside it; never keep an engine or
its results past the block. Distinct keys (language, model dirs, CPU vs GPU)
never share instances, and detection-only engines are pooled apart from full
OCR engines.

Inference never takes a lock here: concurrent lessees hold distinct
instances, so parallel workers run in parallel on the GPU, as they did when
every call built its own engine. The pool therefore holds at most one
instance per key per concurrent lessee -- the same GPU memory as building one
per worker -- and reuses them across files for the life of the process.

Construction is serialised process-wide (one build at a time, whatever the
key). `utils.suppress_output()` redirects the process's stdout/stderr file
descriptors during a build; two overlapping builds left fd 1 and fd 2 on
/dev/null and `sys.stdout` on a StringIO for the rest of the process in 3 of
3 runs. Serial construction of 4 OCR + 2 detection engines took 4.6 s against
2.8 s in parallel, a one-time cost per process now that engines are pooled.
A lessee waiting for a build also takes an instance that is returned while it
waits, instead of building its own.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

from . import utils

_cond = threading.Condition()
_constructing = False
# Bumped by reset_registry(): an engine leased before a reset is dropped on
# return instead of re-entering the fresh pool.
_generation = 0
_idle_ocr_engines: dict[tuple, list] = {}
_idle_detection_engines: dict[tuple, list] = {}


def _build_ocr_engine(lang, det_model_dir, rec_model_dir, use_gpu):
    """Builder the pool calls to construct a full OCR engine.

    A thin wrapper around `videocr.utils.create_ocr_engine` so tests can
    monkeypatch construction without touching PaddleOCR itself.
    """
    return utils.create_ocr_engine(lang, det_model_dir, rec_model_dir, use_gpu)


def _build_detection_engine(det_model_dir, use_gpu):
    """Builder the pool calls to construct a detection-only engine.

    A thin wrapper around `videocr.utils.create_detection_engine` so tests
    can monkeypatch construction without touching PaddleOCR itself.
    """
    return utils.create_detection_engine(det_model_dir, use_gpu)


def _checkout(idle: dict, key: tuple, build):
    """Take an idle engine for `key`, or build one. Returns (engine, token);
    the token goes back to `_checkin` with the engine."""
    global _constructing
    with _cond:
        while True:
            free = idle.get(key)
            if free:
                return free.pop(), _generation
            if not _constructing:
                _constructing = True
                generation = _generation
                break
            _cond.wait()
    try:
        engine = build()
    finally:
        with _cond:
            _constructing = False
            _cond.notify_all()
    return engine, generation


def _checkin(idle: dict, key: tuple, engine, token) -> None:
    """Return a leased engine to the pool (unless the pool was reset since)."""
    with _cond:
        if token == _generation:
            idle.setdefault(key, []).append(engine)
            _cond.notify_all()


@contextmanager
def lease_ocr_engine(lang, det_model_dir, rec_model_dir, use_gpu):
    """Exclusive use of a full OCR engine for this argument tuple, for the
    duration of the `with` block. See the module docstring."""
    key = (lang, det_model_dir, rec_model_dir, use_gpu)
    engine, token = _checkout(
        _idle_ocr_engines, key,
        lambda: _build_ocr_engine(lang, det_model_dir, rec_model_dir, use_gpu))
    try:
        yield engine
    finally:
        _checkin(_idle_ocr_engines, key, engine, token)


@contextmanager
def lease_detection_engine(det_model_dir, use_gpu):
    """Exclusive use of a detection-only engine for this argument tuple, for
    the duration of the `with` block. See the module docstring."""
    key = (det_model_dir, use_gpu)
    engine, token = _checkout(
        _idle_detection_engines, key,
        lambda: _build_detection_engine(det_model_dir, use_gpu))
    try:
        yield engine
    finally:
        _checkin(_idle_detection_engines, key, engine, token)


def reset_registry() -> None:
    """Drop every pooled engine. Tests only."""
    global _generation
    with _cond:
        _generation += 1
        _idle_ocr_engines.clear()
        _idle_detection_engines.clear()
        _cond.notify_all()
