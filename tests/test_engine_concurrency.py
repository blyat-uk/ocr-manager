"""Parallel OCR workers must never run inference on one engine at once.

OCRManager runs its workers as QThreads in one process (default 4). A
PaddleOCR / TextDetection instance shares its input and output handles
between calls, so two threads inside `predict()` on the same instance
corrupt each other's results -- measured on the real engines: 101-126 of 384
detection outputs changed, 17 recognised texts were wrong and one thread
died with `AssertionError: 3 != 2 for key rec_text`. These tests drive the
real OCR entry points from several threads with fake engines that notice
exactly that: a second thread entering an instance while another thread's
call -- including the lazy consumption of the results it returned -- is
still in progress.
"""
from __future__ import annotations

import subprocess
import threading
import time

import numpy as np
import pytest

from videocr import api, engine_registry, utils
from videocr import video as video_mod

THREADS = 4

# Dialogue band and one label, each as a white box on a dark background:
#   dialogue 0-2 s (w=200) and 3-5.6 s (w=400) at the bottom,
#   a label 1-5 s at the top-left, outside the dialogue crop.
CLIP_FILTER = (
    "drawbox=x=160:y=300:w=200:h=30:color=white@1.0:t=fill:enable='between(n,0,49)',"
    "drawbox=x=100:y=300:w=400:h=30:color=white@1.0:t=fill:enable='between(n,75,139)',"
    "drawbox=x=40:y=40:w=160:h=30:color=white@1.0:t=fill:enable='between(n,25,124)'"
)
OCR_KWARGS = dict(
    lang="ch", conf_threshold=95, sim_threshold=82, brightness_threshold=None,
    similar_image_threshold=0.3, similar_pixel_threshold=25, frames_to_skip=0,
    crop_x=0, crop_y=290, crop_width=640, crop_height=50, use_gpu=False,
)


@pytest.fixture(scope="module")
def labelled_clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("media") / "labelled.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=#202020:size=640x360:rate=25:duration=6",
        "-vf", CLIP_FILTER, "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "25", str(out),
    ], check=True, capture_output=True)
    return out


def _bright_rect(image):
    gray = image.max(axis=2) if image.ndim == 3 else image
    ys, xs = np.nonzero(gray > 128)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _poly(rect):
    x0, y0, x1, y1 = rect
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


class _ContentEngine:
    """Reads bright boxes like a (very simple) real engine, in the PaddleOCR
    3.x result format, and records every time a second thread enters it
    while another thread's call is still in progress.

    A call is in progress from `predict()` until the generator it returns
    is exhausted, which is how PaddleX's own predict behaves: results are
    produced as the caller iterates. Sleeping inside widens the window so a
    race that exists shows up on every run, not occasionally.
    """

    def __init__(self, kind):
        self.kind = kind
        self._lock = threading.Lock()
        self._holder = None
        self._depth = 0
        self.calls = 0
        self.overlaps = 0

    def _enter(self):
        me = threading.get_ident()
        with self._lock:
            self.calls += 1
            if self._holder is not None and self._holder != me:
                self.overlaps += 1
            else:
                self._holder = me
            self._depth += 1

    def _exit(self):
        with self._lock:
            self._depth -= 1
            if self._depth == 0:
                self._holder = None

    def _read(self, image):
        rect = _bright_rect(image)
        if self.kind == "det":
            return {"dt_polys": [] if rect is None else [_poly(rect)]}
        if rect is None:
            return {"rec_texts": [], "rec_scores": [], "rec_polys": []}
        width_tag = (rect[2] - rect[0] + 5) // 10
        return {"rec_texts": [f"字{width_tag}"], "rec_scores": [0.99], "rec_polys": [_poly(rect)]}

    def predict(self, images):
        self._enter()
        try:
            batch = images if isinstance(images, list) else [images]
            time.sleep(0.002)
            results = [self._read(image) for image in batch]
        except BaseException:
            self._exit()
            raise

        def lazily():
            try:
                for result in results:
                    time.sleep(0.0005)
                    yield result
            finally:
                self._exit()

        return lazily()


@pytest.fixture
def content_engines(monkeypatch):
    built = []

    def build(kind):
        def builder(*args, **kwargs):
            engine = _ContentEngine(kind)
            built.append(engine)
            return engine
        return builder

    monkeypatch.setattr(utils, "create_ocr_engine", build("ocr"))
    monkeypatch.setattr(utils, "create_detection_engine", build("det"))
    return built


def _in_threads(target, count=THREADS):
    outputs = [None] * count
    errors = []

    def run(i):
        try:
            outputs[i] = target()
        except Exception as exc:
            errors.append(repr(exc))

    threads = [threading.Thread(target=run, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return outputs, errors


def _run_ocr_text(path):
    v = video_mod.Video(str(path), None, None)
    v.run_ocr(
        use_gpu=False, lang="ch", time_start="0:00", time_end="",
        conf_threshold=95, use_fullframe=False, brightness_threshold=None,
        similar_image_threshold=0.3, similar_pixel_threshold=25, frames_to_skip=0,
        crop_x=0, crop_y=290, crop_width=640, crop_height=50,
    )
    return v.get_subtitles(82)


ENTRY_POINTS = {
    "get_subtitles-with-labels": lambda path: api.get_subtitles(str(path), detect_labels=True, **OCR_KWARGS),
    "get_subtitles-multirange": lambda path: api.get_subtitles(
        str(path), detect_labels=True, time_ranges=[("0:00", "0:03"), ("0:03", "0:06")], **OCR_KWARGS),
    "run_ocr": _run_ocr_text,
}


@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
def test_concurrent_ocr_never_enters_one_engine_from_two_threads(labelled_clip, content_engines, entry):
    call = ENTRY_POINTS[entry]
    serial = call(labelled_clip)
    assert "Dialogue:" in serial
    if entry.startswith("get_subtitles"):
        assert ",Label," in serial, "the clip's label must reach the output, or the label engines went unused"

    outputs, errors = _in_threads(lambda: call(labelled_clip))

    overlaps = {f"{e.kind}#{i}": e.overlaps for i, e in enumerate(content_engines) if e.overlaps}
    assert not overlaps, f"threads ran inference on one engine at the same time: {overlaps}"
    assert not errors
    assert outputs == [serial] * THREADS


def test_one_file_runs_on_one_engine_pair(labelled_clip, content_engines):
    """Every range's dialogue pass and label pass for a file share the one
    OCR engine and one detection engine get_subtitles leased for it."""
    api.get_subtitles(str(labelled_clip), detect_labels=True,
                      time_ranges=[("0:00", "0:03"), ("0:03", "0:06")], **OCR_KWARGS)
    assert sorted(e.kind for e in content_engines) == ["det", "ocr"]
    assert all(e.calls > 0 for e in content_engines)


# --- every engine call happens inside the caller's own lease -----------------


@pytest.fixture
def lease_tracking(monkeypatch):
    """Tracks which thread holds each engine's lease, straight from the
    registry's checkout/checkin, so an engine can tell whether the call it
    is serving (or the results it is still producing) belongs to a lessee."""
    holders = {}
    real_checkout = engine_registry._checkout
    real_checkin = engine_registry._checkin

    def checkout(*args, **kwargs):
        engine, token = real_checkout(*args, **kwargs)
        holders[id(engine)] = threading.get_ident()
        return engine, token

    def checkin(idle, key, engine, token):
        holders[id(engine)] = None
        return real_checkin(idle, key, engine, token)

    monkeypatch.setattr(engine_registry, "_checkout", checkout)
    monkeypatch.setattr(engine_registry, "_checkin", checkin)
    return holders


class _LeaseCheckingEngine(_ContentEngine):
    """Behaves like a pooled engine whose results stop being valid once its
    lease is returned: anything produced for a thread that does not hold the
    lease comes back empty, and is recorded."""

    def __init__(self, kind, holders):
        super().__init__(kind)
        self.holders = holders
        self.violations = []

    def _leased_here(self):
        return self.holders.get(id(self)) == threading.get_ident()

    def predict(self, images):
        if not self._leased_here():
            self.violations.append("predict() called without a lease")
        inner = super().predict(images)

        def checked():
            for result in inner:
                if not self._leased_here():
                    self.violations.append("results consumed after the lease was returned")
                    result = {key: [] for key in result}
                yield result

        return checked()


def test_label_scan_consumes_every_engine_result_inside_the_lease(labelled_clip, monkeypatch, lease_tracking):
    built = []

    def build(kind):
        def builder(*args, **kwargs):
            engine = _LeaseCheckingEngine(kind, lease_tracking)
            built.append(engine)
            return engine
        return builder

    monkeypatch.setattr(utils, "create_ocr_engine", build("ocr"))
    monkeypatch.setattr(utils, "create_detection_engine", build("det"))

    outputs, errors = _in_threads(
        lambda: api.get_subtitles(str(labelled_clip), detect_labels=True, **OCR_KWARGS))

    assert not errors
    assert {e.kind for e in built} == {"ocr", "det"}
    violations = {f"{e.kind}#{i}": sorted(set(e.violations)) for i, e in enumerate(built) if e.violations}
    assert not violations
    assert all(",Label," in out for out in outputs)
    assert len(set(outputs)) == 1
