"""Fixtures for tests/ui (the new PyQt6 window under `app/`).

pytest-qt is NOT installed (task brief) -- this file provides the one
fixture it would otherwise give us (a session-scoped QApplication), and
tests drive widgets directly with PyQt6.QtTest.QTest instead of pytest-qt's
`qtbot`. tests/conftest.py's autouse `_reset_engine_registry` fixture still
applies here (pytest collects parent conftests for a subdirectory), which
is harmless: nothing under tests/ui touches the OCR engine registry.

This directory is named `tests/ui`, not `tests/app`, specifically so it
never shares a name with the real top-level `app/` package -- see
test_theme_widgets.py's test_app_package_resolves_to_real_source_package.

`tmp_project` and `fake_runner` (plan 3B Task 2) let the controller and the
views be driven without decoding a frame: placeholder video files, and a
runner that runs nothing but records what was submitted and lets the test
deliver the events a real JobRunner would.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

from core.jobs.runner import JobEvent, Lane

V1_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "ocr_json_v1"
DEFAULT_NAMES = ("ep01.mkv", "ep02.mkv")
TERMINAL = ("finished", "failed", "cancelled")


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """The one QApplication instance for the whole test session -- Qt does
    not allow more than one per process, and offscreen tests still need one
    to construct any QWidget."""
    app = QApplication.instance() or QApplication([])
    return app


# --------------------------------------------------------------------------
# tmp_project
# --------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path):
    """Factory for a project folder under tmp_path:

        tmp_project(names=None, *, fixture=None, config=None) -> Path

    - names: placeholder video files to create (a few bytes each, not
      decodable). Default: the fixture's file names with `fixture`, else
      DEFAULT_NAMES.
    - fixture: "slay" or "dragon", a v1 `.ocr.json` from
      tests/fixtures/ocr_json_v1 copied in (it migrates on open).
    - config: a dict written as `.ocr.json` (JSON), or a str written verbatim.

    Each call makes a new folder, so one test can have several projects.
    """
    counter = itertools.count(1)

    def make(names=None, *, fixture: str | None = None, config=None) -> Path:
        folder = tmp_path / f"project{next(counter)}"
        folder.mkdir()
        if fixture is not None:
            data = json.loads((V1_FIXTURES / f"{fixture}.json").read_text(encoding="utf-8"))
            (folder / ".ocr.json").write_text(json.dumps(data), encoding="utf-8")
            if names is None:
                names = list(data.get("files", {}))
        if config is not None:
            text = config if isinstance(config, str) else json.dumps(config)
            (folder / ".ocr.json").write_text(text, encoding="utf-8")
        for name in (DEFAULT_NAMES if names is None else names):
            (folder / name).write_bytes(b"placeholder video")
        return folder

    return make


# --------------------------------------------------------------------------
# fake_runner
# --------------------------------------------------------------------------

@dataclass
class Submission:
    job: object
    job_id: int


class FakeRunner:
    """Stands in for core.jobs.runner.JobRunner, and is its own factory:
    pass the instance as ProjectController(runner_factory=fake_runner).

    It runs nothing. submit() records the job with a job_id (1, 2, ... like
    the real runner) and delivers "queued" synchronously, as the real runner
    does. Like the real runner, a submit replaces a queued (not yet started)
    job with the same key: that job gets "cancelled" (result None) first. A
    started job with the same key is left alone. cancel/cancel_where/pause/
    resume/shutdown are only recorded. The test delivers every other event
    with emit/start/progress/finish, which call the controller's listener
    exactly as a worker thread would (the listener only enqueues; the
    controller drains on its timer or drain_events()). Events after a job's
    terminal event are refused (the real runner drops them), so a test cannot
    rely on a sequence that never happens.
    """

    def __init__(self):
        self.on_event = None
        self.factory_calls = 0
        self.submissions: list[Submission] = []
        self.cancelled_keys: list[str] = []
        self.cancel_predicates: list = []
        self.pauses: list[tuple[Lane, object]] = []
        self.resumes: list[Lane] = []
        self.shutdown_calls: list[float] = []
        self.closed = False
        self._ids = itertools.count(1)
        self._started: set[int] = set()
        self._ended: set[int] = set()

    # --- the JobRunner surface ------------------------------------------------

    def __call__(self, on_event):
        self.factory_calls += 1
        self.on_event = on_event
        return self

    def submit(self, job) -> None:
        if self.closed:
            raise RuntimeError("JobRunner has been shut down")
        for queued in self.queued():
            if queued.job.key == job.key:
                self.emit(queued, "cancelled", result=None)
        submission = Submission(job, next(self._ids))
        self.submissions.append(submission)
        self.emit(submission, "queued")

    def cancel(self, key: str) -> None:
        self.cancelled_keys.append(key)

    def cancel_where(self, predicate) -> None:
        self.cancel_predicates.append(predicate)

    def pause(self, lane, *, only=None) -> None:
        self.pauses.append((Lane(lane), only))

    def resume(self, lane) -> None:
        self.resumes.append(Lane(lane))

    def shutdown(self, timeout: float = 10.0) -> bool:
        self.closed = True
        self.shutdown_calls.append(timeout)
        return True

    # --- test side --------------------------------------------------------------

    def emit(self, submission: Submission, event_type: str, **fields) -> None:
        assert submission.job_id not in self._ended, \
            f"{event_type!r} after the terminal event of {submission.job.key} (job {submission.job_id})"
        if event_type in TERMINAL:
            self._ended.add(submission.job_id)
        job = submission.job
        file = fields.pop("file", job.file)
        self.on_event(JobEvent(event_type, job.key, job.kind, file, job_id=submission.job_id, **fields))

    def start(self, submission: Submission) -> None:
        if submission.job_id not in self._started:
            self._started.add(submission.job_id)
            self.emit(submission, "started")

    def progress(self, submission: Submission, fraction: float, message: str = "") -> None:
        self.start(submission)
        self.emit(submission, "progress", progress=fraction, message=message)

    def finish(self, submission: Submission, result=None, event_type: str = "finished", **fields) -> None:
        """"started" (once), then the terminal event."""
        assert event_type in TERMINAL
        self.start(submission)
        self.emit(submission, event_type, result=result, **fields)

    def queued(self) -> list[Submission]:
        """Submissions neither started nor ended."""
        return [s for s in self.submissions if s.job_id not in self._started and s.job_id not in self._ended]

    def ended(self, submission: Submission) -> bool:
        return submission.job_id in self._ended

    def of_kind(self, kind: str, file: str | None = None) -> list[Submission]:
        return [s for s in self.submissions
                if s.job.kind == kind and (file is None or s.job.file == file)]

    def last(self, kind: str, file: str | None = None) -> Submission:
        matching = self.of_kind(kind, file)
        assert matching, f"no {kind} submission for {file}"
        return matching[-1]


@pytest.fixture
def fake_runner() -> FakeRunner:
    return FakeRunner()
