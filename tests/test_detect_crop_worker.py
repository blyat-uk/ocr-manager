"""Qt-adapter tests for SubtitleDetectionWorker.

SubtitleDetectionWorker is a thin Qt adapter over the pure
core.detect.crop.detect_crop() -- these tests drive it with a stubbed
detect_crop() (and a stubbed detection-engine constructor) so they never
touch ffmpeg, ffprobe or a real PaddleOCR model. Every test bounds its
event loop with a QTimer so a regression that hangs the worker fails the
test instead of hanging the suite -- and worker.cleanup() itself is now
also called with a bound (see CLEANUP_TIMEOUT_MS / ruling B below), since
an unbounded QThread.wait() right after a bounded event loop would defeat
the point of bounding the event loop at all.
"""
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer

import core.subtitle_detector as subtitle_detector
from core.detect.crop import CropResult
from core.subtitle_detector import SubtitleDetectionWorker

EVENT_LOOP_TIMEOUT_MS = 5000
CLEANUP_TIMEOUT_MS = 5000

FRAME_SIZE = (1920, 1080)


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


def _run_worker(worker, timeout_ms=EVENT_LOOP_TIMEOUT_MS, cleanup_timeout_ms=CLEANUP_TIMEOUT_MS):
    """Start `worker`, pump a bounded event loop until finished/error fires
    (or `timeout_ms` elapses), then clean up with a bound of its own (see
    ruling B: an unbounded cleanup() defeats the event-loop bound above),
    and return the collected signal emissions.
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
    cleanup_finished = worker.cleanup(timeout_ms=cleanup_timeout_ms)

    return {
        "detected": detected,
        "progress": progress_calls,
        "errors": errors,
        "finished_count": state["finished_count"],
        "timed_out": state["timed_out"],
        "cleanup_finished": cleanup_finished,
    }


def _make_video_files(names, duration=100.0):
    return [(name, f"/videos/{name}", duration) for name in names]


def _crop_result(box, hit_pts, sample_pts=None, flagged=None, frame_size=FRAME_SIZE, **kw):
    """Build a CropResult with the fields the adapter now actually reads
    (hit_pts, frame_size -- see rulings C/D), defaulting sample_pts to
    something that deliberately does NOT start with hit_pts[0], so any
    test using this helper would catch a regression back to deriving the
    slider position from sample_pts[0]."""
    if sample_pts is None:
        sample_pts = [0.1] + list(hit_pts)
    return CropResult(box=box, sample_pts=sample_pts, hit_pts=list(hit_pts),
                       flagged=flagged, frame_size=frame_size, **kw)


def test_one_file_detected_per_resolved_file_with_right_arguments(monkeypatch):
    results = {
        "a.mp4": _crop_result(
            box=(10, 900, 1000, 80), hit_pts=[42.5], sample_pts=[10.0, 42.5, 50.0],
            agreed=5, probes_used=10, flagged=None,
        ),
        "b.mp4": _crop_result(
            box=(20, 910, 1000, 90), hit_pts=[30.0], sample_pts=[5.0, 30.0],
            agreed=4, probes_used=8, flagged="no-speech",
        ),
    }
    calls = []

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None,
                          settings=None, cancel_check=None):
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
    assert outcome["cleanup_finished"] is True
    assert outcome["finished_count"] == 1
    assert not outcome["errors"]

    detected = {args[0]: args for args in outcome["detected"]}
    assert set(detected) == {"a.mp4", "b.mp4"}

    # a.mp4: slider = hit_pts[0] / duration * 10000 = 42.5/100*10000 = 4250
    # (NOT sample_pts[0]=10.0, which would give 1000 -- see ruling D)
    assert detected["a.mp4"] == ("a.mp4", 4250, 10, 900, 1000, 80)
    # b.mp4: slider = hit_pts[0]=30.0 -> 3000 (not sample_pts[0]=5.0 -> 500)
    assert detected["b.mp4"] == ("b.mp4", 3000, 20, 910, 1000, 90)

    # progress: monotonically non-decreasing, ends at (2, 2)
    resolved_seq = [r for r, _t in outcome["progress"]]
    assert resolved_seq == sorted(resolved_seq)
    assert outcome["progress"][-1] == (2, 2)
    assert outcome["progress"][0] == (0, 2)

    # consensus list threads resolved files' (y_frac, h_frac) into the next call
    assert calls[0]["consensus"] == []
    assert calls[1]["consensus"] == [(900 / 1080, 80 / 1080)]


def test_slider_position_is_derived_from_hit_pts_not_sample_pts(monkeypatch):
    """Task-3 review ruling D, isolated: sample_pts[0] is the full probe
    history's first entry -- after a fallback round it is guaranteed to be
    a frame with no detected text (see core/detect/crop.py). The slider
    position the worker emits must come from hit_pts[0] (a real hit), not
    sample_pts[0]."""
    result = CropResult(
        box=(0, 900, 100, 50),
        sample_pts=[1.0, 2.0, 3.0],   # would give slider=100 if used (wrong)
        hit_pts=[75.0],               # must give slider=7500
        agreed=5, probes_used=3, flagged=None, frame_size=FRAME_SIZE,
    )

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None,
                          settings=None, cancel_check=None):
        return result

    monkeypatch.setattr(subtitle_detector, "detect_crop", fake_detect_crop)

    worker = SubtitleDetectionWorker(_make_video_files(["a.mp4"]))
    outcome = _run_worker(worker)

    assert not outcome["timed_out"]
    assert len(outcome["detected"]) == 1
    filename, slider_pos, cx, cy, cw, ch = outcome["detected"][0]
    assert slider_pos == 7500, f"slider must come from hit_pts[0]=75.0, got {slider_pos}"


def test_box_none_is_skipped_not_emitted_with_a_zero_box(monkeypatch, caplog):
    results = {
        "a.mp4": _crop_result(box=(10, 900, 1000, 80), hit_pts=[10.0], agreed=5, probes_used=5),
        "b.mp4": CropResult(box=None, sample_pts=[10.0], agreed=0, probes_used=30, flagged="no-speech"),
    }

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None,
                          settings=None, cancel_check=None):
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
    the detect_crop call itself (as if the user clicked Cancel while a
    file was still being processed) rather than from a cross-thread signal
    handler racing the background thread's next loop iteration.

    Cancellation is requested while b.mp4 is "in flight": a.mp4 (already
    fully processed beforehand) is still emitted, b.mp4's own result is
    discarded (a cancelled-mid-file result is not trustworthy -- see
    SubtitleDetectionWorker._run()), and c.mp4 is never started.
    """
    results = {
        name: _crop_result(box=(0, 900, 100, 50), hit_pts=[1.0], agreed=5, probes_used=5)
        for name in ("a.mp4", "b.mp4", "c.mp4")
    }

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None,
                          settings=None, cancel_check=None):
        if os.path.basename(video_path) == "b.mp4":
            worker.cancel()
        return results[os.path.basename(video_path)]

    monkeypatch.setattr(subtitle_detector, "detect_crop", fake_detect_crop)

    worker = SubtitleDetectionWorker(_make_video_files(["a.mp4", "b.mp4", "c.mp4"]))
    outcome = _run_worker(worker)

    assert not outcome["timed_out"]
    assert outcome["finished_count"] == 1, "cancellation must still reach a clean finished()"
    detected_names = [args[0] for args in outcome["detected"]]
    assert detected_names == ["a.mp4"], (
        f"cancel() must stop further files from being processed (and discard the "
        f"in-flight file's own result), got {detected_names}"
    )


def test_exception_in_one_file_does_not_abort_others_or_emit_error(monkeypatch):
    good_a = _crop_result(box=(0, 900, 100, 50), hit_pts=[1.0], agreed=5, probes_used=5)
    good_c = _crop_result(box=(0, 910, 100, 60), hit_pts=[1.0], agreed=5, probes_used=5)

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None,
                          settings=None, cancel_check=None):
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


def test_cancel_takes_effect_within_one_probe_batch_not_only_between_files(monkeypatch):
    """Task-3 review ruling A: detect_crop() now takes a cancel_check hook
    that the worker must wire to its own cancellation flag, checked
    between probe batches -- not just between files (previously, cancel()
    only took effect once the CURRENT file's detect_crop() call returned
    on its own, which could mean waiting out up to 3 internal probe
    rounds x 30 probes each; main.py's cancel handler calls cleanup()
    synchronously right after cancel(), so that unbounded wait froze the
    whole GUI). Simulates a detect_crop() that would run many "batches"
    (several seconds) unless cancelled, cancels shortly after it starts,
    and asserts the worker thread finishes well under the time the full,
    uncancelled detection would have taken.
    """
    BATCH_DELAY_S = 0.05
    MANY_BATCHES = 100  # 100 * 0.05s = 5s if cancellation is not honoured

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None,
                          settings=None, cancel_check=None):
        assert cancel_check is not None, "the worker must pass a cancel_check into detect_crop"
        for _ in range(MANY_BATCHES):
            if cancel_check():
                break
            time.sleep(BATCH_DELAY_S)
        return CropResult(box=None, sample_pts=[])

    monkeypatch.setattr(subtitle_detector, "detect_crop", fake_detect_crop)

    worker = SubtitleDetectionWorker(_make_video_files(["a.mp4"]))
    worker.start()
    time.sleep(BATCH_DELAY_S * 3)  # let a few "batches" happen before cancelling
    worker.cancel()

    start = time.monotonic()
    finished = worker.cleanup(timeout_ms=2000)
    elapsed = time.monotonic() - start

    assert finished is True, "the worker thread must actually finish once cancelled"
    bound = (MANY_BATCHES * BATCH_DELAY_S) / 2
    assert elapsed < bound, (
        f"cancellation must take effect within roughly one batch, not run to "
        f"completion (took {elapsed:.2f}s, bound {bound:.2f}s)"
    )


def test_cleanup_returns_promptly_even_if_run_never_finishes(monkeypatch):
    """Task-3 review ruling B (and round-2 finding M6): the worker-test
    event loop is bounded by a QTimer, but the very next call after it
    used to be an UNBOUNDED QThread.wait() -- if a regression hung
    _run(), the test wouldn't fail fast; the only backstop left would be
    pytest-timeout's global 900s. cleanup() now accepts an optional
    timeout_ms (main.py's own no-argument call is unaffected -- it still
    waits indefinitely) so a hung worker thread fails this test in
    seconds.

    Round 2 found that the test itself still trusted cleanup() to honour
    its own timeout: calling `worker.cleanup(timeout_ms=500)` directly on
    the test thread means a regression that makes cleanup() IGNORE
    timeout_ms (e.g. reverting to a bare `self._thread.wait()`) hangs this
    test too, caught only by the global 900s. The call under test must
    not be able to block the test thread: it's invoked from a helper
    thread, joined with a short, hard bound independent of cleanup()'s own
    behaviour -- if cleanup() ignores its timeout, the helper thread is
    still alive when the join bound expires, and `not helper.is_alive()`
    fails within seconds.
    """
    hang = threading.Event()  # never set by this test until explicitly released below

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None,
                          settings=None, cancel_check=None):
        hang.wait()  # simulates a genuinely stuck _run()
        return CropResult(box=None, sample_pts=[])

    monkeypatch.setattr(subtitle_detector, "detect_crop", fake_detect_crop)

    worker = SubtitleDetectionWorker(_make_video_files(["a.mp4"]))
    worker.start()

    call_result = {}

    def call_cleanup():
        t0 = time.monotonic()
        call_result["finished"] = worker.cleanup(timeout_ms=500)
        call_result["elapsed"] = time.monotonic() - t0

    helper = threading.Thread(target=call_cleanup, daemon=True)
    helper.start()
    helper.join(timeout=2.0)  # hard bound on the TEST, independent of cleanup()'s own behaviour

    assert not helper.is_alive(), (
        "cleanup(timeout_ms=500) did not return within 2s -- it must have "
        "ignored its timeout_ms argument and blocked on an unbounded wait()"
    )
    assert call_result.get("finished") is False, "the hung thread must not actually have finished"
    assert call_result["elapsed"] < 2.0, (
        f"cleanup(timeout_ms=500) must return promptly, took {call_result['elapsed']:.2f}s"
    )

    # Let the background thread actually exit before the process does, and
    # tidy up its QThread -- via another bounded helper, so a regression
    # here can't hang the suite either.
    hang.set()
    tidy = threading.Thread(target=lambda: worker.cleanup(timeout_ms=3000), daemon=True)
    tidy.start()
    tidy.join(timeout=5.0)


@pytest.mark.parametrize("flagged", [
    "top-positioned?", "low-agreement", "static-content?", "multiple-positions?",
    "outlier-discarded?", "cancelled", "no-speech+top-positioned?", "speech-probes-exhausted+low-agreement",
])
def test_a_box_that_is_not_auto_applicable_is_logged_not_emitted(monkeypatch, caplog, flagged):
    """M8: the old bottom-half detector could never return a top-positioned?
    box, and none of these flags' boxes may be applied without review. Today's
    UI has nowhere to show a flag, so such a box is withheld and logged with
    its flags; the files around it are unaffected."""
    results = {
        "a.mp4": _crop_result(box=(10, 900, 1000, 80), hit_pts=[10.0], agreed=5, probes_used=5),
        "b.mp4": _crop_result(box=(10, 100, 1000, 80), hit_pts=[20.0], agreed=5, probes_used=5, flagged=flagged),
        "c.mp4": _crop_result(box=(10, 905, 1000, 75), hit_pts=[30.0], agreed=5, probes_used=5,
                              flagged="no-speech"),
    }

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None,
                          settings=None, cancel_check=None):
        return results[os.path.basename(video_path)]

    monkeypatch.setattr(subtitle_detector, "detect_crop", fake_detect_crop)

    with caplog.at_level("INFO"):
        outcome = _run_worker(SubtitleDetectionWorker(_make_video_files(["a.mp4", "b.mp4", "c.mp4"])))

    assert not outcome["timed_out"] and not outcome["errors"]
    assert [args[0] for args in outcome["detected"]] == ["a.mp4", "c.mp4"]
    assert outcome["progress"][-1] == (3, 3), "a withheld file still counts toward progress"
    assert any("b.mp4" in rec.getMessage() and flagged in rec.getMessage() for rec in caplog.records), (
        "a withheld box must be logged with its flags")


def test_flagged_results_are_excluded_from_the_consensus_pool(monkeypatch):
    """Task-3 review ruling C: a resolved, boxed file with a non-None flag
    (multiple-positions?, static-content?, low-agreement, ...) is exactly
    the kind of uncertain result consensus must not learn from -- it must
    not enter the consensus pool passed to later files. That includes a box
    with only an informational flag (no-speech), which is emitted.

    4 files: b.mp4 resolves with a wildly different, flagged shape between
    two clean files; c.mp4 carries an informational flag. Asserts b's shape
    never appears in the consensus passed to c.mp4 or d.mp4, and that the
    pool used for d.mp4 reflects only a.mp4.
    """
    def crop_result(y, h, flagged=None):
        return _crop_result(box=(0, y, 100, h), hit_pts=[1.0], agreed=5, probes_used=5, flagged=flagged)

    results = {
        "a.mp4": crop_result(900, 50),
        "b.mp4": crop_result(100, 800, flagged="multiple-positions?"),  # wildly different, flagged
        "c.mp4": crop_result(905, 55, flagged="no-speech"),
        "d.mp4": crop_result(895, 45),
    }
    calls = []

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None,
                          settings=None, cancel_check=None):
        filename = os.path.basename(video_path)
        calls.append(list(consensus) if consensus is not None else None)
        return results[filename]

    monkeypatch.setattr(subtitle_detector, "detect_crop", fake_detect_crop)

    worker = SubtitleDetectionWorker(_make_video_files(["a.mp4", "b.mp4", "c.mp4", "d.mp4"]))
    outcome = _run_worker(worker)

    assert not outcome["timed_out"]
    assert outcome["finished_count"] == 1

    detected_names = sorted(args[0] for args in outcome["detected"])
    assert detected_names == ["a.mp4", "c.mp4", "d.mp4"], (
        "only auto-applicable results are emitted (see the test above)"
    )

    assert calls[0] == []
    assert calls[1] == [(900 / 1080, 50 / 1080)]
    # c.mp4's call: b.mp4 was flagged, so its (100/1080, 800/1080) shape
    # must be absent here -- the pool must still be just a.mp4's entry.
    assert calls[2] == [(900 / 1080, 50 / 1080)], (
        f"the flagged file (b.mp4) must not have entered the consensus pool: {calls[2]}"
    )
    # d.mp4's call: only a.mp4 (unflagged) has contributed.
    assert calls[3] == [(900 / 1080, 50 / 1080)], (
        f"consensus for d.mp4 must reflect only the unflagged a.mp4 result: {calls[3]}"
    )


def test_worker_detects_only_on_an_engine_it_holds_a_lease_on(monkeypatch):
    """OCR workers run as threads in this process too, and a detection engine
    must never serve two threads at once: every detect_crop() call has to run
    on an engine this worker's thread has leased, and the lease has to be
    back in the pool once the run is over."""
    from videocr import engine_registry

    holders = {}
    real_checkout, real_checkin = engine_registry._checkout, engine_registry._checkin

    def checkout(*args, **kwargs):
        engine, token = real_checkout(*args, **kwargs)
        holders[id(engine)] = threading.get_ident()
        return engine, token

    def checkin(idle, key, engine, token):
        holders[id(engine)] = None
        return real_checkin(idle, key, engine, token)

    monkeypatch.setattr(engine_registry, "_checkout", checkout)
    monkeypatch.setattr(engine_registry, "_checkin", checkin)

    leased_during_detection = []

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None,
                          settings=None, cancel_check=None):
        leased_during_detection.append(holders.get(id(det_engine)) == threading.get_ident())
        return _crop_result(box=(10, 900, 1000, 80), hit_pts=[42.5])

    monkeypatch.setattr(subtitle_detector, "detect_crop", fake_detect_crop)

    outcome = _run_worker(SubtitleDetectionWorker(_make_video_files(["a.mp4", "b.mp4"])))
    assert not outcome["timed_out"]
    assert not outcome["errors"]
    assert leased_during_detection == [True, True]
    assert set(holders.values()) == {None}, "the worker kept its lease after finishing"


def test_detection_engine_is_shared_across_worker_runs(monkeypatch):
    """Opening a folder twice must not rebuild the detection model: the worker
    leases its engine from videocr.engine_registry's pool, which keeps it for
    the next run (5-10 s per build on real PaddleOCR)."""
    builds = []

    def counting_builder(det_model_dir, use_gpu):
        engine = object()
        builds.append(engine)
        return engine

    monkeypatch.setattr("videocr.utils.create_detection_engine", counting_builder)

    seen_engines = []

    def fake_detect_crop(video_path, duration_sec, det_engine, consensus=None,
                          settings=None, cancel_check=None):
        seen_engines.append(det_engine)
        return _crop_result(box=(10, 900, 1000, 80), hit_pts=[42.5])

    monkeypatch.setattr(subtitle_detector, "detect_crop", fake_detect_crop)

    for _ in range(2):
        outcome = _run_worker(SubtitleDetectionWorker(_make_video_files(["a.mp4"])))
        assert not outcome["timed_out"]
        assert not outcome["errors"]

    assert len(builds) == 1
    assert len(seen_engines) == 2
    assert seen_engines[0] is seen_engines[1] is builds[0]
