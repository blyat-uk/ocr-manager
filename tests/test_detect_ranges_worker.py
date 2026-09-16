"""Qt-adapter tests for AudioAnalysisWorker.

AudioAnalysisWorker is a thin Qt adapter over the pure
core.detect.ranges.pipeline.analyse() -- these tests drive it with a
stubbed analyse() so they never touch ffmpeg or a real fingerprinting
pipeline. Every test bounds its event loop with a QTimer so a regression
that hangs the worker fails the test instead of hanging the suite, and
cleanup() itself is called with a bound for the same reason.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer

import core.audio_analysis as audio_analysis
from core.audio_analysis import AudioAnalysisWorker
from core.detect.ranges.pipeline import AnalysisCancelled, ProgressEvent

EVENT_LOOP_TIMEOUT_MS = 5000
CLEANUP_TIMEOUT_MS = 5000


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """A QCoreApplication is required for QThread/QTimer/QEventLoop to
    actually function -- autouse so every test in this module gets one
    regardless of whether it names the fixture directly."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def _run_worker(worker, timeout_ms=EVENT_LOOP_TIMEOUT_MS, cleanup_timeout_ms=CLEANUP_TIMEOUT_MS):
    """Start `worker`, pump a bounded event loop until finished/error fires
    (or `timeout_ms` elapses), then clean up with a bound of its own, and
    return the collected signal emissions."""
    phases = []
    file_progress = []
    analysis_msgs = []
    errors = []
    finished_results = []
    state = {"timed_out": False}

    def on_phase(p):
        phases.append(p)

    def on_file_progress(*args):
        file_progress.append(args)

    def on_analysis_progress(msg):
        analysis_msgs.append(msg)

    def on_error(msg):
        errors.append(msg)

    loop = QEventLoop()

    def on_finished(res):
        finished_results.append(res)
        loop.quit()

    def on_error_quit(_msg):
        loop.quit()

    def on_timeout():
        state["timed_out"] = True
        loop.quit()

    worker.phase_changed.connect(on_phase)
    worker.file_progress.connect(on_file_progress)
    worker.analysis_progress.connect(on_analysis_progress)
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
        "phases": phases,
        "file_progress": file_progress,
        "analysis_msgs": analysis_msgs,
        "errors": errors,
        "finished_results": finished_results,
        "timed_out": state["timed_out"],
        "cleanup_finished": cleanup_finished,
    }


def test_finished_carries_the_analyse_result_dict_unchanged(monkeypatch):
    expected = {"a.mkv": [("00:05", "01:00")], "b.mkv": [(None, "00:30")]}

    def fake_analyse(files, cfg, progress=None, *, cache_dir, workers, cancel):
        return expected

    monkeypatch.setattr(audio_analysis, "analyse", fake_analyse)

    worker = AudioAnalysisWorker("/proj", ["a.mkv", "b.mkv"])
    outcome = _run_worker(worker)

    assert not outcome["timed_out"], "worker did not finish within the bounded event loop"
    assert outcome["cleanup_finished"] is True
    assert not outcome["errors"]
    assert outcome["finished_results"] == [expected]


def test_phase_changed_fires_for_each_phase_analyse_reports(monkeypatch):
    phase_names = ["Fingerprinting", "Analyzing", "Computing time ranges"]

    def fake_analyse(files, cfg, progress=None, *, cache_dir, workers, cancel):
        for name in phase_names:
            progress(ProgressEvent("phase", name))
        return {}

    monkeypatch.setattr(audio_analysis, "analyse", fake_analyse)

    worker = AudioAnalysisWorker("/proj", ["a.mkv"])
    outcome = _run_worker(worker)

    assert not outcome["timed_out"]
    assert outcome["phases"] == phase_names


def test_cancel_before_completion_emits_finished_with_empty_dict(monkeypatch):
    """analyse() raises AnalysisCancelled when its cancel callback returns
    True -- the worker must turn that into finished({}), the existing
    contract, not error()."""
    calls = {}

    def fake_analyse(files, cfg, progress=None, *, cache_dir, workers, cancel):
        worker.cancel()
        calls["cancel_result"] = cancel()
        raise AnalysisCancelled()

    monkeypatch.setattr(audio_analysis, "analyse", fake_analyse)

    worker = AudioAnalysisWorker("/proj", ["a.mkv", "b.mkv"])
    outcome = _run_worker(worker)

    assert not outcome["timed_out"]
    assert calls["cancel_result"] is True, "the worker's cancel flag must reach analyse()'s cancel callback"
    assert not outcome["errors"]
    assert outcome["finished_results"] == [{}]


def test_exception_in_analyse_emits_error_exactly_once_and_does_not_raise_out_of_the_thread(monkeypatch):
    def fake_analyse(files, cfg, progress=None, *, cache_dir, workers, cancel):
        raise RuntimeError("boom")

    monkeypatch.setattr(audio_analysis, "analyse", fake_analyse)

    worker = AudioAnalysisWorker("/proj", ["a.mkv"])
    outcome = _run_worker(worker)

    assert not outcome["timed_out"], "an exception inside analyse() must not hang the worker thread"
    assert outcome["cleanup_finished"] is True, "the thread must still exit cleanly after the exception"
    assert outcome["errors"] == ["boom"]
    assert not outcome["finished_results"], "an error must not also emit finished()"
