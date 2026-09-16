"""Qt-adapter tests for SubtitleDetectionWorker.

SubtitleDetectionWorker is a thin Qt adapter over the pure
core.detect.crop.detect_crop() -- these tests drive it with a stubbed
detect_crop() (and a stubbed detection-engine constructor) so they never
touch ffmpeg, ffprobe or a real PaddleOCR model. Every test bounds its
event loop with a QTimer so a regression that hangs the worker fails the
test instead of hanging the suite.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer

import core.subtitle_detector as subtitle_detector
from core.detect.crop import CropResult
from core.subtitle_detector import SubtitleDetectionWorker

EVENT_LOOP_TIMEOUT_MS = 5000


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """A QCoreApplication is required for QThread/QTimer/QEventLoop to
    actually function (without one, Qt prints "Cannot be used without
    QCoreApplication" and cross-thread signal delivery silently doesn't
    happen) -- autouse so every test in this module gets one regardless
    of whether it names the fixture directly."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


@pytest.fixture(autouse=True)
def stub_engine_creation(monkeypatch):
    """Never let the worker actually load a PaddleOCR model."""
    monkeypatch.setattr(
        "videocr.utils.create_detection_engine",
        lambda det_model_dir, use_gpu: object(),
    )
    monkeypatch.setattr(subtitle_detector, "_probe_dimensions", lambda video_path: (1920, 1080))


def _run_worker(worker, timeout_ms=EVENT_LOOP_TIMEOUT_MS):
    """Start `worker`, pump a bounded event loop until finished/error fires
    (or `timeout_ms` elapses), and return the collected signal emissions.
    """
    detected = []
    progress_calls = []
    errors = []
    state = {"finished_count": 0, "timed_out": False}

    def on_detected(*args):
        detected.append(args)

    def on_progress(resolved, total):
        progress_calls.append((resolved, total))

    def on_error(msg):
        errors.append(msg)

    loop = QEventLoop()

    def on_finished():
        state["finished_count"] += 1
        loop.quit()

    def on_error_quit(_msg):
        loop.quit()

    def on_timeout():
        state["timed_out"] = True
        loop.quit()

    worker.file_detected.connect(on_detected)
    worker.progress.connect(on_progress)
    worker.error.connect(on_error)
    worker.error.connect(on_error_quit)
    worker.finished.connect(on_finished)

    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(on_timeout)
    timer.start(timeout_ms)

    worker.start()
    loop.exec()
    timer.stop()
    worker.cleanup()

    return {
        "detected": detected,
        "progress": progress_calls,
        "errors": errors,
        "finished_count": state["finished_count"],
        "timed_out": state["timed_out"],
    }


def _make_video_files(names, duration=100.0):
    return [(name, f"/videos/{name}", duration) for name in names]


def test_one_file_detected_per_resolved_file_with_right_arguments(monkeypatch):
    results = {
        "a.mp4": CropResult(
            box=(10, 900, 1000, 80), sample_pts=[42.5, 50.0],
            agreed=5, probes_used=10, flagged=None,
        ),
        "b.mp4": CropResult(
            box=(20, 910, 1000, 90), sample_pts=[30.0],
            agreed=4, probes_used=8, flagged="low-agreement",
        ),
    }
    calls = []

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None, settings=None):
        filename = os.path.basename(video_path)
        calls.append({
            "video_path": video_path,
            "duration_sec": duration_sec,
            "consensus": list(consensus) if consensus is not None else None,
        })
        return results[filename]

    monkeypatch.setattr(subtitle_detector, "detect_crop", fake_detect_crop)

    worker = SubtitleDetectionWorker(_make_video_files(["a.mp4", "b.mp4"]))
    outcome = _run_worker(worker)

    assert not outcome["timed_out"], "worker did not finish within the bounded event loop"
    assert outcome["finished_count"] == 1
    assert not outcome["errors"]

    detected = {args[0]: args for args in outcome["detected"]}
    assert set(detected) == {"a.mp4", "b.mp4"}

    # a.mp4: slider = sample_pts[0] / duration * 10000 = 42.5/100*10000 = 4250
    assert detected["a.mp4"] == ("a.mp4", 4250, 10, 900, 1000, 80)
    # b.mp4: slider = 30.0/100*10000 = 3000
    assert detected["b.mp4"] == ("b.mp4", 3000, 20, 910, 1000, 90)

    # progress: monotonically non-decreasing, ends at (2, 2)
    resolved_seq = [r for r, _t in outcome["progress"]]
    assert resolved_seq == sorted(resolved_seq)
    assert outcome["progress"][-1] == (2, 2)
    assert outcome["progress"][0] == (0, 2)

    # consensus list threads resolved files' (y_frac, h_frac) into the next call
    assert calls[0]["consensus"] == []
    assert calls[1]["consensus"] == [(900 / 1080, 80 / 1080)]


def test_box_none_is_skipped_not_emitted_with_a_zero_box(monkeypatch, caplog):
    results = {
        "a.mp4": CropResult(box=(10, 900, 1000, 80), sample_pts=[10.0], agreed=5, probes_used=5),
        "b.mp4": CropResult(box=None, sample_pts=[10.0], agreed=0, probes_used=30, flagged="no-speech"),
    }

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None, settings=None):
        return results[os.path.basename(video_path)]

    monkeypatch.setattr(subtitle_detector, "detect_crop", fake_detect_crop)

    worker = SubtitleDetectionWorker(_make_video_files(["a.mp4", "b.mp4"]))
    with caplog.at_level("INFO"):
        outcome = _run_worker(worker)

    assert not outcome["timed_out"]
    assert outcome["finished_count"] == 1
    assert not outcome["errors"]

    detected_names = [args[0] for args in outcome["detected"]]
    assert detected_names == ["a.mp4"], "a file with box=None must never emit file_detected"
    assert outcome["progress"][-1] == (2, 2), "a skipped file still counts toward progress"

    assert any("b.mp4" in rec.message and "no-speech" in rec.message for rec in caplog.records), (
        "the flag for a box=None result must be logged so it's visible"
    )


def test_cancel_stops_further_work(monkeypatch):
    """detect_crop() runs synchronously on the worker's background thread
    with no artificial delay here, so the only race-free way to pin
    "cancel() stops further work" is to request cancellation from inside
    the detect_crop call itself (as if the user clicked Cancel while file
    1 was still being processed) rather than from a cross-thread signal
    handler racing the background thread's next loop iteration."""
    results = {
        name: CropResult(box=(0, 900, 100, 50), sample_pts=[1.0], agreed=5, probes_used=5)
        for name in ("a.mp4", "b.mp4", "c.mp4")
    }

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None, settings=None):
        if os.path.basename(video_path) == "a.mp4":
            worker.cancel()
        return results[os.path.basename(video_path)]

    monkeypatch.setattr(subtitle_detector, "detect_crop", fake_detect_crop)

    worker = SubtitleDetectionWorker(_make_video_files(["a.mp4", "b.mp4", "c.mp4"]))
    outcome = _run_worker(worker)

    assert not outcome["timed_out"]
    assert outcome["finished_count"] == 1, "cancellation must still reach a clean finished()"
    detected_names = [args[0] for args in outcome["detected"]]
    assert detected_names == ["a.mp4"], (
        f"cancel() must stop further files from being processed, got {detected_names}"
    )


def test_exception_in_one_file_does_not_abort_others_or_emit_error(monkeypatch):
    good_a = CropResult(box=(0, 900, 100, 50), sample_pts=[1.0], agreed=5, probes_used=5)
    good_c = CropResult(box=(0, 910, 100, 60), sample_pts=[1.0], agreed=5, probes_used=5)

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None, settings=None):
        name = os.path.basename(video_path)
        if name == "b.mp4":
            raise RuntimeError("boom")
        return good_a if name == "a.mp4" else good_c

    monkeypatch.setattr(subtitle_detector, "detect_crop", fake_detect_crop)

    worker = SubtitleDetectionWorker(_make_video_files(["a.mp4", "b.mp4", "c.mp4"]))
    outcome = _run_worker(worker)

    assert not outcome["timed_out"]
    assert outcome["finished_count"] == 1
    assert not outcome["errors"], "a per-file exception must not surface as the fatal error() signal"

    detected_names = sorted(args[0] for args in outcome["detected"])
    assert detected_names == ["a.mp4", "c.mp4"], "the other files must still resolve"
    assert outcome["progress"][-1] == (3, 3), "the failed file must still count toward progress"
