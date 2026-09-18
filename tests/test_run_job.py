"""Behaviour of core.jobs.run.RunJob: the OCR run over a snapshot of files.

Pinned here (rulings C5/C6, task-9-brief.md):

- Atomic output. Each file writes chi/<stem>.ass.partial, runs
  core.ass_qafix.process_file on it, then os.replace()s it onto chi/<stem>.ass.
  A stopped or failed file deletes only its .partial: an existing
  chi/<stem>.ass is never deleted or modified except by that replace.
- The old OCRWorker call shape. No range or one range goes through
  save_subtitles_to_file(time_start=, time_end=); several go through
  get_subtitles(time_ranges=) and the text is written only if there is some
  and the file was not stopped.
- File workers. At most `parallel` files run at once. Pause starts no new
  file (in-flight files continue). Stop reaches every in-flight file through
  its own cancel event, never kills anything, and files it kept from starting
  are reported cancelled.

videocr.api and core.ass_qafix.process_file are fakes in the fast tests. The
slow test runs real OCR on two fidelity cases and compares the run's output
with the old OCRWorker flow -- transcribed into _todays_worker_output below,
because core/ocr_worker.py itself was deleted in plan 3B Task 6 -- and with
the goldens.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import ass_qafix
from core.jobs import JobContext, JobEvent, JobRunner, Lane
from core.jobs import run as run_module
from core.jobs.run import RunFile, RunJob, RunSummary
from core.project.model import (
    Brightness,
    Crop,
    FileEntry,
    FolderSettings,
    Source,
    TimeRange,
    TimeRanges,
)
from core.project.ocr_kwargs import ocr_call_for
from videocr import api

pytestmark = pytest.mark.timeout(60)

WAIT = 5.0      # bounds what must happen promptly, so a regression fails instead of hanging
QUIET = 0.3     # window in which something must NOT happen (a held file starting)
PROMPT = 1.0    # a stopped run (paused or not) must return within this

REAL_PROCESS_FILE = ass_qafix.process_file
QAStats = ass_qafix.QAStats
CALLBACK_KEYS = {"progress_callback", "subtitle_callback", "cancel_event"}

HEADER = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize\n"
    "Style: Default,Arial,20\n"
    "\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)
QA_LINE = "Dialogue: 0,0:00:09.00,0:00:09.50,Default,,0,0,0,,QA\n"
OLD_FINAL = "OLD 字幕\r\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,旧\r\n".encode("utf-8")


def ass_text(name: str) -> str:
    return (HEADER
            + f"Dialogue: 0,0:00:01.00,0:00:01.50,Default,,0,0,0,,字幕一 {name}\n"
            + "Comment: 0,0:00:01.50,0:00:02.00,Default,,0,0,0,,not a dialogue line\n"
            + f"Dialogue: 0,0:00:02.00,0:00:02.50,Default,,0,0,0,,字幕二 {name}\n")


def stem(name: str) -> str:
    return Path(name).stem


def partial_of(project: Path, name: str) -> Path:
    return project / "chi" / f"{stem(name)}.ass.partial"


def final_of(project: Path, name: str) -> Path:
    return project / "chi" / f"{stem(name)}.ass"


def write_old_final(project: Path, name: str) -> Path:
    final = final_of(project, name)
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(OLD_FINAL)
    return final


def run_file(project: Path, name: str, ranges=None) -> RunFile:
    """A RunFile built the way the app builds one: ocr_call_for on an entry."""
    entry = FileEntry(
        name,
        crop=Crop(288, 786, 1344, 53, Source.MANUAL),
        brightness=Brightness(209, Source.MANUAL),
        time_ranges=(None if ranges is None
                     else TimeRanges([TimeRange(s, e) for s, e in ranges], Source.MANUAL)),
    )
    return RunFile(name, ocr_call_for(entry, FolderSettings(labels_enabled=False), str(project)))


# --------------------------------------------------------------------------
# Fakes and helpers
# --------------------------------------------------------------------------

class FakeOcr:
    """Stands in for videocr.api.save_subtitles_to_file / get_subtitles.

    Accepts keyword arguments only (RunJob passes everything by keyword).
    Records each call, reports progress and one subtitle, then runs the
    file's hook (keyed by video file name), which may block, raise, or return
    the text to produce instead of ass_text(name).
    """

    def __init__(self, monkeypatch):
        self.calls: list[tuple[str, dict]] = []
        self.hooks: dict[str, object] = {}
        self._cond = threading.Condition()
        self.active = 0
        self.max_active = 0
        monkeypatch.setattr(api, "save_subtitles_to_file", self.save_subtitles_to_file)
        monkeypatch.setattr(api, "get_subtitles", self.get_subtitles)

    def _ocr(self, function: str, kwargs: dict) -> str:
        name = os.path.basename(kwargs["video_path"])
        with self._cond:
            self.calls.append((function, dict(kwargs)))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self._cond.notify_all()
        try:
            kwargs["progress_callback"]("Extracting dialogue", 0)
            kwargs["subtitle_callback"](1.0, 1.5, f"字幕 {name}")
            kwargs["progress_callback"]("Extracting dialogue", 100)
            text = ass_text(name)
            hook = self.hooks.get(name)
            if hook is not None:
                produced = hook(kwargs)
                if produced is not None:
                    text = produced
            return text
        finally:
            with self._cond:
                self.active -= 1
                self._cond.notify_all()

    def wait_active(self, count: int, timeout: float = WAIT) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self.active >= count, timeout)

    def save_subtitles_to_file(self, **kwargs) -> None:
        text = self._ocr("save_subtitles_to_file", kwargs)
        # Like the real one: file_path is written whatever happened, even after a cancel.
        with open(kwargs["file_path"], "w+", encoding="utf-8") as f:
            f.write(text)

    def get_subtitles(self, **kwargs) -> str:
        return self._ocr("get_subtitles", kwargs)


class FakeQa:
    """Stands in for core.ass_qafix.process_file: records what it saw, appends QA_LINE."""

    def __init__(self, monkeypatch):
        self.calls: list[SimpleNamespace] = []
        self.raise_for: dict[str, Exception] = {}
        self.stats = QAStats()
        monkeypatch.setattr(ass_qafix, "process_file", self)

    def __call__(self, *args, **kwargs):
        path = args[0]
        final = path[: -len(".partial")] if path.endswith(".partial") else None
        self.calls.append(SimpleNamespace(
            args=args, kwargs=kwargs, data=Path(path).read_bytes(),
            final=(Path(final).read_bytes() if final and os.path.exists(final) else None)))
        error = self.raise_for.get(os.path.basename(path))
        if error is not None:
            raise error
        with open(path, "a", encoding="utf-8") as f:
            f.write(QA_LINE)
        return self.stats


@pytest.fixture
def ocr(monkeypatch) -> FakeOcr:
    return FakeOcr(monkeypatch)


@pytest.fixture
def qa(monkeypatch) -> FakeQa:
    return FakeQa(monkeypatch)


class Collector:
    """on_event listener for a JobContext or a JobRunner."""

    def __init__(self):
        self._cond = threading.Condition()
        self.events: list[JobEvent] = []

    def __call__(self, event: JobEvent) -> None:
        with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    def of(self, file: str) -> list[JobEvent]:
        with self._cond:
            return [e for e in self.events if e.file == file]

    def types(self, file: str) -> list[str]:
        return [e.type for e in self.of(file)]

    def wait_for(self, predicate, timeout: float = WAIT) -> bool:
        with self._cond:
            return self._cond.wait_for(predicate, timeout)

    def finished(self, file: str) -> bool:
        return any(e.type == "run_file_finished" for e in self.of(file))

    def started(self, file: str) -> bool:
        return any(e.type == "run_file_started" for e in self.of(file))


class Running:
    """RunJob.run(ctx) on a background thread."""

    def __init__(self, job: RunJob, ctx: JobContext):
        self.result = None
        self.error: BaseException | None = None
        self.returned_at: float | None = None
        self.thread = threading.Thread(target=self._body, args=(job, ctx), daemon=True)
        self.thread.start()

    def _body(self, job, ctx):
        try:
            self.result = job.run(ctx)
        except BaseException as exc:  # noqa: BLE001 - reported by join()
            self.error = exc
        finally:
            self.returned_at = time.monotonic()

    def join(self, timeout: float = WAIT):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "run() did not return"
        if self.error is not None:
            raise self.error
        return self.result


def make_ctx() -> tuple[JobContext, Collector]:
    events = Collector()
    return JobContext("run", "run", None, events), events


def run_now(project: Path, files: list[RunFile], parallel: int = 1):
    ctx, events = make_ctx()
    summary = RunJob(str(project), files, parallel).run(ctx)
    return summary, events


def blocking_until_cancelled(entered: threading.Event, seen: list | None = None, produce=None):
    """A hook that waits (bounded) for its file's cancel event, like OCR does."""
    def hook(kwargs):
        if seen is not None:
            seen.append(kwargs["cancel_event"])
        entered.set()
        kwargs["cancel_event"].wait(WAIT)
        return produce
    return hook


# --------------------------------------------------------------------------
# Identity and construction
# --------------------------------------------------------------------------

def test_identity_matches_the_run_lane_contract(tmp_path):
    job = RunJob(str(tmp_path), [], 1)
    assert (job.kind, job.lane, job.priority, job.file, job.key) == ("run", Lane.RUN, 0, None, "run")


@pytest.mark.parametrize("parallel", [0, -1])
def test_parallel_must_be_at_least_one(tmp_path, parallel):
    with pytest.raises(ValueError):
        RunJob(str(tmp_path), [], parallel)


@pytest.mark.parametrize("names, message", [
    (["a.mp4", "a.mkv"], "a.mkv and a.mp4 both write chi/a.ass"),
    (["a.mp4", "b.mp4", "a.mp4"], "a.mp4 and a.mp4 both write chi/a.ass"),
    (["x.mp4", "a.mp4", "a.mkv", "x.mkv", "a.avi"],
     "a.avi, a.mkv and a.mp4 both write chi/a.ass; x.mkv and x.mp4 both write chi/x.ass"),
    (["[1080p] ep.01.mkv", "[1080p] ep.01.mp4"],
     "[1080p] ep.01.mkv and [1080p] ep.01.mp4 both write chi/[1080p] ep.01.ass"),
])
def test_run_files_that_write_the_same_output_are_refused(tmp_path, ocr, names, message):
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError) as raised:
        RunJob(str(project), [run_file(project, n) for n in names], 2)
    assert str(raised.value) == message
    assert list(project.iterdir()) == []
    assert ocr.calls == []


def test_distinct_output_stems_are_accepted(tmp_path, ocr, qa):
    names = ["a.mp4", "A.mp4", "a.b.mp4", "a .mp4", "ab.mkv"]     # case-sensitive, like the filesystem
    job = RunJob(str(tmp_path), [run_file(tmp_path, n) for n in names], 2)
    ctx, _ = make_ctx()
    summary = job.run(ctx)
    assert summary.succeeded == names
    assert sorted(os.listdir(tmp_path / "chi")) == sorted(["a.ass", "A.ass", "a.b.ass", "a .ass", "ab.ass"])


def test_the_file_list_is_a_snapshot_taken_at_construction(tmp_path, ocr, qa):
    a = run_file(tmp_path, "a.mp4")
    files = [a]
    job = RunJob(str(tmp_path), files, 1)
    wanted = dict(a.call.kwargs)
    files.append(run_file(tmp_path, "b.mp4"))
    a.call.kwargs["brightness_threshold"] = 1
    a.call.time_ranges.append(("1:00", "2:00"))

    ctx, _ = make_ctx()
    summary = job.run(ctx)

    assert summary.succeeded == ["a.mp4"]
    (function, kwargs), = ocr.calls
    assert function == "save_subtitles_to_file"
    assert {k: kwargs[k] for k in wanted} == wanted
    assert (kwargs["time_start"], kwargs["time_end"]) == ("0:00", "")


def test_run_file_and_run_summary_are_frozen(tmp_path):
    rf = run_file(tmp_path, "a.mp4")
    with pytest.raises(dataclasses.FrozenInstanceError):
        rf.name = "b.mp4"
    summary = RunSummary([], {}, [], 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.seconds = 1.0


# --------------------------------------------------------------------------
# Directories and output
# --------------------------------------------------------------------------

def test_output_dirs_are_created_before_any_file(tmp_path, ocr, qa):
    (tmp_path / "eng").mkdir()
    (tmp_path / "eng" / "keep.ass").write_text("keep")
    summary, events = run_now(tmp_path, [])
    for sub in ("chi", "eng", "translate"):
        assert (tmp_path / sub).is_dir()
    assert (tmp_path / "eng" / "keep.ass").read_text() == "keep"
    assert (summary.succeeded, summary.failed, summary.cancelled) == ([], {}, [])
    assert events.events == []

    seen = []
    ocr.hooks["a.mp4"] = lambda kwargs: seen.append(
        all((tmp_path / sub).is_dir() for sub in ("chi", "eng", "translate")))
    shutil.rmtree(tmp_path / "chi")
    run_now(tmp_path, [run_file(tmp_path, "a.mp4")])
    assert seen == [True]


def test_output_lands_in_chi_only_after_qa_replacing_the_old_file(tmp_path, ocr, qa):
    write_old_final(tmp_path, "a.mp4")
    during_ocr = []
    ocr.hooks["a.mp4"] = lambda kwargs: during_ocr.append(final_of(tmp_path, "a.mp4").read_bytes())

    summary, _ = run_now(tmp_path, [run_file(tmp_path, "a.mp4"), run_file(tmp_path, "b.mkv")])

    assert summary.succeeded == ["a.mp4", "b.mkv"]
    assert during_ocr == [OLD_FINAL]
    assert [(c.args, c.kwargs) for c in qa.calls] == [
        ((str(partial_of(tmp_path, "a.mp4")),), {}),
        ((str(partial_of(tmp_path, "b.mkv")),), {}),
    ]
    # QA saw the OCR output in the .partial while the old final was still in place.
    assert qa.calls[0].data == ass_text("a.mp4").encode("utf-8")
    assert qa.calls[0].final == OLD_FINAL
    assert qa.calls[1].data == ass_text("b.mkv").encode("utf-8")
    assert qa.calls[1].final is None
    for name in ("a.mp4", "b.mkv"):
        assert final_of(tmp_path, name).read_bytes() == (ass_text(name) + QA_LINE).encode("utf-8")
    assert sorted(os.listdir(tmp_path / "chi")) == ["a.ass", "b.ass"]


def test_final_is_exactly_what_the_real_qa_makes_of_the_ocr_text(tmp_path, ocr):
    project = tmp_path / "project"
    project.mkdir()
    messy = (HEADER
             + "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,字幕\n"
             + "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,字幕\n"
             + "Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,  \n")
    ocr.hooks["a.mp4"] = lambda kwargs: messy
    ocr.hooks["b.mp4"] = lambda kwargs: messy

    summary, _ = run_now(project, [run_file(project, "a.mp4"),
                                   run_file(project, "b.mp4", [("0:00", "1:00"), ("2:00", "3:00")])])

    expected = tmp_path / "expected.ass"
    expected.write_text(messy, encoding="utf-8")
    REAL_PROCESS_FILE(str(expected))
    assert summary.succeeded == ["a.mp4", "b.mp4"]
    assert final_of(project, "a.mp4").read_bytes() == expected.read_bytes()
    assert final_of(project, "b.mp4").read_bytes() == expected.read_bytes()
    assert expected.read_bytes() != messy.encode("utf-8")


def test_lines_counts_dialogue_lines_of_the_final_file(tmp_path, ocr, qa):
    _, events = run_now(tmp_path, [run_file(tmp_path, "a.mp4")])
    finished = [e for e in events.of("a.mp4") if e.type == "run_file_finished"]
    # two OCR Dialogue lines + the line QA added; the Comment line does not count
    assert [e.result for e in finished] == [{"ok": True, "lines": 3, "error": ""}]


# --------------------------------------------------------------------------
# Call shape (the old OCRWorker._run_ocr)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ranges, start, end", [
    (None, "0:00", ""),
    ([], "0:00", ""),
    ([("1:00", "2:30")], "1:00", "2:30"),
    ([(None, "2:30")], "0:00", "2:30"),
    ([("1:00", None)], "1:00", ""),
])
def test_no_range_or_one_range_calls_save_subtitles_to_file(tmp_path, ocr, qa, ranges, start, end):
    rf = run_file(tmp_path, "a.mp4", ranges)
    ctx, _ = make_ctx()
    RunJob(str(tmp_path), [rf], 1).run(ctx)

    (function, kwargs), = ocr.calls
    assert function == "save_subtitles_to_file"
    assert set(kwargs) == set(rf.call.kwargs) | {"file_path", "time_start", "time_end"} | CALLBACK_KEYS
    assert {k: kwargs[k] for k in rf.call.kwargs} == rf.call.kwargs
    assert kwargs["file_path"] == str(partial_of(tmp_path, "a.mp4"))
    assert (kwargs["time_start"], kwargs["time_end"]) == (start, end)
    assert callable(kwargs["progress_callback"]) and callable(kwargs["subtitle_callback"])
    assert isinstance(kwargs["cancel_event"], threading.Event)
    assert kwargs["cancel_event"] is not ctx.cancel_event
    assert not kwargs["cancel_event"].is_set()


def test_several_ranges_call_get_subtitles_and_write_the_text_as_utf8(tmp_path, ocr, qa):
    ranges = [("0:00", "1:00"), ("5:00", None), (None, "9:00")]
    rf = run_file(tmp_path, "a.mp4", ranges)
    text = ass_text("a.mp4").replace("字幕一", "「字幕」—…")
    ocr.hooks["a.mp4"] = lambda kwargs: text
    ctx, _ = make_ctx()
    summary = RunJob(str(tmp_path), [rf], 1).run(ctx)

    (function, kwargs), = ocr.calls
    assert function == "get_subtitles"
    assert set(kwargs) == set(rf.call.kwargs) | {"time_ranges"} | CALLBACK_KEYS
    assert {k: kwargs[k] for k in rf.call.kwargs} == rf.call.kwargs
    assert kwargs["time_ranges"] == [("0:00", "1:00"), ("5:00", ""), ("0:00", "9:00")]
    assert isinstance(kwargs["cancel_event"], threading.Event)
    assert kwargs["cancel_event"] is not ctx.cancel_event
    assert summary.succeeded == ["a.mp4"]
    (call,) = qa.calls
    assert call.args == (str(partial_of(tmp_path, "a.mp4")),)
    assert call.data == text.encode("utf-8")
    assert final_of(tmp_path, "a.mp4").read_bytes() == (text + QA_LINE).encode("utf-8")


def test_several_ranges_with_no_text_produce_no_file_and_keep_the_old_final(tmp_path, ocr, qa):
    write_old_final(tmp_path, "a.mp4")
    ocr.hooks["a.mp4"] = lambda kwargs: ""
    summary, events = run_now(tmp_path, [run_file(tmp_path, "a.mp4", [("0:00", "1:00"), ("2:00", "3:00")])])

    assert summary.succeeded == [] and summary.cancelled == []
    assert summary.failed == {"a.mp4": "no subtitles produced"}
    assert qa.calls == []
    assert final_of(tmp_path, "a.mp4").read_bytes() == OLD_FINAL
    assert sorted(os.listdir(tmp_path / "chi")) == ["a.ass"]
    assert events.of("a.mp4")[-1].result == {"ok": False, "lines": 0, "error": "no subtitles produced"}


def test_a_stale_partial_from_an_earlier_run_is_never_promoted(tmp_path, ocr, qa):
    write_old_final(tmp_path, "a.mp4")
    partial_of(tmp_path, "a.mp4").write_text("STALE", encoding="utf-8")
    ocr.hooks["a.mp4"] = lambda kwargs: ""
    summary, _ = run_now(tmp_path, [run_file(tmp_path, "a.mp4", [("0:00", "1:00"), ("2:00", "3:00")])])

    assert summary.failed == {"a.mp4": "no subtitles produced"}
    assert qa.calls == []
    assert final_of(tmp_path, "a.mp4").read_bytes() == OLD_FINAL
    assert not partial_of(tmp_path, "a.mp4").exists()


# --------------------------------------------------------------------------
# Stop
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ranges", [None, [("0:00", "1:00"), ("2:00", "3:00")]],
                         ids=["save_subtitles_to_file", "get_subtitles"])
def test_stopping_mid_file_keeps_the_old_final_and_removes_the_partial(tmp_path, ocr, qa, ranges):
    write_old_final(tmp_path, "a.mp4")
    entered = threading.Event()
    ocr.hooks["a.mp4"] = blocking_until_cancelled(entered, produce=ass_text("partial"))
    ctx, events = make_ctx()
    running = Running(RunJob(str(tmp_path), [run_file(tmp_path, "a.mp4", ranges),
                                             run_file(tmp_path, "b.mp4")], 1), ctx)
    assert entered.wait(WAIT)
    ctx.cancel_event.set()
    summary = running.join()

    assert final_of(tmp_path, "a.mp4").read_bytes() == OLD_FINAL
    assert sorted(os.listdir(tmp_path / "chi")) == ["a.ass"]
    assert qa.calls == []
    assert [fn for fn, _ in ocr.calls] == ["save_subtitles_to_file" if ranges is None else "get_subtitles"]
    assert (summary.succeeded, summary.failed, summary.cancelled) == ([], {}, ["a.mp4", "b.mp4"])
    assert events.of("a.mp4")[0].type == "run_file_started"
    assert events.of("a.mp4")[-1].type == "run_file_finished"
    assert events.of("a.mp4")[-1].result == {"ok": False, "lines": 0, "error": "cancelled"}
    assert events.of("b.mp4") == []


def test_stop_reaches_every_in_flight_file_through_its_own_event(tmp_path, ocr, qa):
    seen: list[threading.Event] = []
    entered_a, entered_b = threading.Event(), threading.Event()
    ocr.hooks["a.mp4"] = blocking_until_cancelled(entered_a, seen)
    ocr.hooks["b.mp4"] = blocking_until_cancelled(entered_b, seen)
    ctx, events = make_ctx()
    files = [run_file(tmp_path, n) for n in ("a.mp4", "b.mp4", "c.mp4")]
    running = Running(RunJob(str(tmp_path), files, 2), ctx)
    assert entered_a.wait(WAIT) and entered_b.wait(WAIT)
    stopped_at = time.monotonic()
    ctx.cancel_event.set()
    summary = running.join()

    assert running.returned_at - stopped_at < PROMPT
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert all(event.is_set() and event is not ctx.cancel_event for event in seen)
    assert summary.cancelled == ["a.mp4", "b.mp4", "c.mp4"]
    assert events.of("c.mp4") == []
    assert not (tmp_path / "chi" / "a.ass.partial").exists()
    assert os.listdir(tmp_path / "chi") == []


def test_a_run_stopped_before_it_starts_starts_nothing(tmp_path, ocr, qa):
    ctx, events = make_ctx()
    ctx.cancel_event.set()
    summary = RunJob(str(tmp_path), [run_file(tmp_path, "a.mp4"), run_file(tmp_path, "b.mp4")], 2).run(ctx)
    assert ocr.calls == []
    assert events.events == []
    assert (summary.succeeded, summary.failed, summary.cancelled) == ([], {}, ["a.mp4", "b.mp4"])
    assert (tmp_path / "chi").is_dir()


# --------------------------------------------------------------------------
# Failures
# --------------------------------------------------------------------------

def test_an_exception_in_one_file_does_not_stop_the_others(tmp_path, ocr, qa):
    write_old_final(tmp_path, "b.mp4")

    def explode(kwargs):
        Path(kwargs["file_path"]).write_text("half written", encoding="utf-8")
        raise RuntimeError("decoder exploded\nat frame 12")

    ocr.hooks["b.mp4"] = explode
    files = [run_file(tmp_path, n) for n in ("a.mp4", "b.mp4", "c.mp4")]
    summary, events = run_now(tmp_path, files)

    assert summary.succeeded == ["a.mp4", "c.mp4"]
    assert summary.failed == {"b.mp4": "decoder exploded"}
    assert summary.cancelled == []
    assert final_of(tmp_path, "b.mp4").read_bytes() == OLD_FINAL
    assert sorted(os.listdir(tmp_path / "chi")) == ["a.ass", "b.ass", "c.ass"]
    assert [c.args[0] for c in qa.calls] == [str(partial_of(tmp_path, n)) for n in ("a.mp4", "c.mp4")]
    assert events.of("b.mp4")[-1].result == {"ok": False, "lines": 0, "error": "decoder exploded"}
    logs = [e.message for e in events.events if e.type == "log"]
    assert any("b.mp4" in line and "Traceback" in line and "RuntimeError: decoder exploded" in line
               for line in logs), logs


def test_an_exception_without_a_message_reports_its_type(tmp_path, ocr, qa):
    def explode(kwargs):
        raise KeyError()

    ocr.hooks["a.mp4"] = explode
    summary, _ = run_now(tmp_path, [run_file(tmp_path, "a.mp4")])
    assert summary.failed == {"a.mp4": "KeyError"}


def test_a_qa_failure_keeps_the_old_final_and_removes_the_partial(tmp_path, ocr, qa):
    write_old_final(tmp_path, "a.mp4")
    qa.raise_for["a.ass.partial"] = ValueError("qa broke")
    summary, events = run_now(tmp_path, [run_file(tmp_path, "a.mp4"), run_file(tmp_path, "b.mp4")])

    assert summary.failed == {"a.mp4": "qa broke"}
    assert summary.succeeded == ["b.mp4"]
    assert final_of(tmp_path, "a.mp4").read_bytes() == OLD_FINAL
    assert sorted(os.listdir(tmp_path / "chi")) == ["a.ass", "b.ass"]
    assert events.of("a.mp4")[-1].result == {"ok": False, "lines": 0, "error": "qa broke"}


def test_a_failed_replace_keeps_the_old_final(tmp_path, ocr, qa, monkeypatch):
    write_old_final(tmp_path, "a.mp4")
    real_replace = os.replace

    def failing_replace(src, dst, *args, **kwargs):
        if str(src).endswith("a.ass.partial"):
            raise OSError("disk full")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", failing_replace)
    summary, events = run_now(tmp_path, [run_file(tmp_path, "a.mp4")])
    assert summary.failed == {"a.mp4": "disk full"}
    assert final_of(tmp_path, "a.mp4").read_bytes() == OLD_FINAL
    assert os.listdir(tmp_path / "chi") == ["a.ass"]
    assert events.of("a.mp4")[-1].result == {"ok": False, "lines": 0, "error": "disk full"}


def test_lines_are_counted_on_the_qad_partial_before_the_final_is_replaced(tmp_path, ocr, qa, monkeypatch):
    write_old_final(tmp_path, "a.mp4")
    real_count = run_module._count_dialogue_lines
    counted = []

    def count(path):
        counted.append((path, Path(path).read_bytes(), final_of(tmp_path, "a.mp4").read_bytes()))
        return real_count(path)

    monkeypatch.setattr(run_module, "_count_dialogue_lines", count)
    summary, events = run_now(tmp_path, [run_file(tmp_path, "a.mp4")])

    assert summary.succeeded == ["a.mp4"]
    qad = (ass_text("a.mp4") + QA_LINE).encode("utf-8")
    assert counted == [(str(partial_of(tmp_path, "a.mp4")), qad, OLD_FINAL)]
    assert final_of(tmp_path, "a.mp4").read_bytes() == qad
    assert events.of("a.mp4")[-1].result == {"ok": True, "lines": 3, "error": ""}


def test_failed_always_means_the_final_was_not_replaced(tmp_path, ocr, qa, monkeypatch):
    write_old_final(tmp_path, "a.mp4")

    def count(path):
        raise OSError("read back failed")

    monkeypatch.setattr(run_module, "_count_dialogue_lines", count)
    summary, events = run_now(tmp_path, [run_file(tmp_path, "a.mp4")])

    assert summary.failed == {"a.mp4": "read back failed"}
    assert final_of(tmp_path, "a.mp4").read_bytes() == OLD_FINAL
    assert os.listdir(tmp_path / "chi") == ["a.ass"]
    assert events.of("a.mp4")[-1].result == {"ok": False, "lines": 0, "error": "read back failed"}


def test_a_file_worker_that_cannot_start_stops_and_joins_the_others_before_raising(
        tmp_path, ocr, qa, monkeypatch):
    entered = threading.Event()
    ocr.hooks["a.mp4"] = blocking_until_cancelled(entered)
    real_start = threading.Thread.start

    def start(self):
        if self.name == "run-file-1":
            assert entered.wait(WAIT)          # the first worker is mid-file
            raise RuntimeError("can't start new thread")
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    ctx, events = make_ctx()
    with pytest.raises(RuntimeError, match="can't start new thread"):
        RunJob(str(tmp_path), [run_file(tmp_path, n) for n in ("a.mp4", "b.mp4")], 2).run(ctx)

    assert events.of("a.mp4")[-1].result == {"ok": False, "lines": 0, "error": "cancelled"}
    assert events.of("b.mp4") == []
    assert not [t for t in threading.enumerate() if t.name.startswith("run-file-")]


def test_summary_sorts_every_file_into_one_outcome_in_snapshot_order(tmp_path, ocr, qa):
    def explode(kwargs):
        raise RuntimeError("boom")

    ocr.hooks["b.mp4"] = explode
    ocr.hooks["c.mp4"] = lambda kwargs: ""
    files = [run_file(tmp_path, "d.mp4"),
             run_file(tmp_path, "b.mp4"),
             run_file(tmp_path, "c.mp4", [("0:00", "1:00"), ("2:00", "3:00")]),
             run_file(tmp_path, "a.mp4")]
    started = time.monotonic()
    summary, _ = run_now(tmp_path, files, parallel=3)
    elapsed = time.monotonic() - started

    assert isinstance(summary, RunSummary)
    assert summary.succeeded == ["d.mp4", "a.mp4"]
    assert summary.failed == {"b.mp4": "boom", "c.mp4": "no subtitles produced"}
    assert list(summary.failed) == ["b.mp4", "c.mp4"]
    assert summary.cancelled == []
    assert 0 < summary.seconds <= elapsed


# --------------------------------------------------------------------------
# Parallel files and pause
# --------------------------------------------------------------------------

@pytest.mark.parametrize("parallel", [1, 2, 3])
def test_parallel_files_run_at_once_and_never_more(tmp_path, ocr, qa, parallel):
    names = [f"{i}.mp4" for i in range(3 * parallel + 1)]
    reached = []

    def first(kwargs):
        # The first `parallel` files are taken by `parallel` workers and none of
        # them ends before all are running, so this wait must succeed.
        reached.append(ocr.wait_active(parallel, timeout=WAIT))
        # Hold a moment, so a worker too many would be running meanwhile.
        extra = ocr.wait_active(parallel + 1, timeout=QUIET)
        reached.append(not extra)

    def later(kwargs):
        time.sleep(0.01)

    for index, name in enumerate(names):
        ocr.hooks[name] = first if index < parallel else later
    summary, _ = run_now(tmp_path, [run_file(tmp_path, n) for n in names], parallel)

    assert summary.succeeded == names
    assert reached == [True] * (2 * parallel)
    assert ocr.max_active == parallel


def test_more_workers_than_files_is_fine(tmp_path, ocr, qa):
    summary, _ = run_now(tmp_path, [run_file(tmp_path, "a.mp4")], parallel=8)
    assert summary.succeeded == ["a.mp4"]


def test_pause_holds_the_next_file_until_resume(tmp_path, ocr, qa):
    entered, release = threading.Event(), threading.Event()

    def hold(kwargs):
        entered.set()
        assert release.wait(WAIT)

    ocr.hooks["a.mp4"] = hold
    ctx, events = make_ctx()
    job = RunJob(str(tmp_path), [run_file(tmp_path, n) for n in ("a.mp4", "b.mp4", "c.mp4")], 1)
    running = Running(job, ctx)
    assert entered.wait(WAIT)
    job.pause()
    release.set()
    assert events.wait_for(lambda: events.finished("a.mp4"))
    time.sleep(QUIET)
    assert not events.started("b.mp4")
    assert events.of("a.mp4")[-1].result["ok"] is True      # the in-flight file finished normally

    job.resume()
    summary = running.join()
    assert summary.succeeded == ["a.mp4", "b.mp4", "c.mp4"]


def test_pause_lets_every_in_flight_file_finish(tmp_path, ocr, qa):
    entered = {n: threading.Event() for n in ("a.mp4", "b.mp4")}
    release = threading.Event()

    def hold(kwargs):
        entered[os.path.basename(kwargs["video_path"])].set()
        assert release.wait(WAIT)

    ocr.hooks["a.mp4"] = ocr.hooks["b.mp4"] = hold
    ctx, events = make_ctx()
    job = RunJob(str(tmp_path), [run_file(tmp_path, n) for n in ("a.mp4", "b.mp4", "c.mp4")], 2)
    running = Running(job, ctx)
    assert entered["a.mp4"].wait(WAIT) and entered["b.mp4"].wait(WAIT)
    job.pause()
    release.set()
    assert events.wait_for(lambda: events.finished("a.mp4") and events.finished("b.mp4"))
    time.sleep(QUIET)
    assert not events.started("c.mp4")
    assert final_of(tmp_path, "a.mp4").exists() and final_of(tmp_path, "b.mp4").exists()

    job.resume()
    assert running.join().succeeded == ["a.mp4", "b.mp4", "c.mp4"]


def test_pause_before_the_run_holds_the_first_file(tmp_path, ocr, qa):
    ctx, events = make_ctx()
    job = RunJob(str(tmp_path), [run_file(tmp_path, "a.mp4")], 1)
    job.pause()
    running = Running(job, ctx)
    time.sleep(QUIET)
    assert events.events == [] and ocr.calls == []
    assert (tmp_path / "chi").is_dir()
    job.resume()
    assert running.join().succeeded == ["a.mp4"]


def test_a_paused_run_that_is_stopped_returns_promptly_with_the_rest_cancelled(tmp_path, ocr, qa):
    entered, release = threading.Event(), threading.Event()

    def hold(kwargs):
        entered.set()
        assert release.wait(WAIT)

    ocr.hooks["a.mp4"] = hold
    ctx, events = make_ctx()
    job = RunJob(str(tmp_path), [run_file(tmp_path, n) for n in ("a.mp4", "b.mp4", "c.mp4")], 1)
    running = Running(job, ctx)
    assert entered.wait(WAIT)
    job.pause()
    release.set()
    assert events.wait_for(lambda: events.finished("a.mp4"))
    time.sleep(QUIET)

    stopped_at = time.monotonic()
    ctx.cancel_event.set()
    summary = running.join()
    assert running.returned_at - stopped_at < PROMPT
    assert summary.succeeded == ["a.mp4"]
    assert summary.cancelled == ["b.mp4", "c.mp4"]
    assert events.of("b.mp4") == [] and events.of("c.mp4") == []


def test_a_run_paused_from_the_start_and_stopped_cancels_everything(tmp_path, ocr, qa):
    ctx, events = make_ctx()
    job = RunJob(str(tmp_path), [run_file(tmp_path, n) for n in ("a.mp4", "b.mp4")], 2)
    job.pause()
    running = Running(job, ctx)
    time.sleep(QUIET)
    stopped_at = time.monotonic()
    ctx.cancel_event.set()
    summary = running.join()
    assert running.returned_at - stopped_at < PROMPT
    assert (summary.succeeded, summary.failed, summary.cancelled) == ([], {}, ["a.mp4", "b.mp4"])
    assert events.events == [] and ocr.calls == []


# --------------------------------------------------------------------------
# set_parallel: how many files run at once, changed while the run goes on
# --------------------------------------------------------------------------

def file_workers() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name.startswith("run-file-")]


class HeldFiles:
    """OCR hooks that hold each file until the test releases it."""

    def __init__(self, ocr: FakeOcr, names: list[str]):
        self.entered = {name: threading.Event() for name in names}
        self.release = {name: threading.Event() for name in names}
        for name in names:
            ocr.hooks[name] = self._hold

    def _hold(self, kwargs):
        name = os.path.basename(kwargs["video_path"])
        self.entered[name].set()
        assert self.release[name].wait(WAIT)

    def release_all(self) -> None:
        for event in self.release.values():
            event.set()


@pytest.mark.parametrize("parallel", [0, -1])
def test_set_parallel_must_be_at_least_one(tmp_path, parallel):
    job = RunJob(str(tmp_path), [], 2)
    with pytest.raises(ValueError):
        job.set_parallel(parallel)
    assert job.parallel == 2


def test_set_parallel_before_the_run_sets_how_many_files_start(tmp_path, ocr, qa):
    names = ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]
    held = HeldFiles(ocr, names)
    ctx, events = make_ctx()
    job = RunJob(str(tmp_path), [run_file(tmp_path, n) for n in names], 1)
    job.set_parallel(3)
    assert job.parallel == 3
    running = Running(job, ctx)
    assert all(held.entered[n].wait(WAIT) for n in names[:3])
    time.sleep(QUIET)
    assert not events.started("d.mp4")
    held.release_all()
    assert running.join().succeeded == names
    assert ocr.max_active == 3


def test_raising_parallel_mid_run_starts_files_not_yet_started(tmp_path, ocr, qa):
    names = ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]
    held = HeldFiles(ocr, names)
    ctx, events = make_ctx()
    job = RunJob(str(tmp_path), [run_file(tmp_path, n) for n in names], 1)
    running = Running(job, ctx)
    assert held.entered["a.mp4"].wait(WAIT)
    time.sleep(QUIET)
    assert not events.started("b.mp4")

    job.set_parallel(3)
    assert job.parallel == 3
    assert held.entered["b.mp4"].wait(WAIT) and held.entered["c.mp4"].wait(WAIT)
    assert ocr.active == 3                                  # a.mp4 is still in flight: nothing was stopped
    time.sleep(QUIET)
    assert not events.started("d.mp4")                      # never more than the new limit

    held.release_all()
    summary = running.join()
    assert summary.succeeded == names and summary.cancelled == []
    assert ocr.max_active == 3
    assert file_workers() == []                             # every worker, the added ones too, was joined


def test_lowering_parallel_mid_run_keeps_in_flight_files_and_limits_new_starts(tmp_path, ocr, qa):
    names = ["a.mp4", "b.mp4", "c.mp4", "d.mp4", "e.mp4"]
    held = HeldFiles(ocr, names)
    ctx, events = make_ctx()
    job = RunJob(str(tmp_path), [run_file(tmp_path, n) for n in names], 3)
    running = Running(job, ctx)
    assert all(held.entered[n].wait(WAIT) for n in names[:3])

    job.set_parallel(1)
    time.sleep(QUIET)
    assert ocr.active == 3                                  # lowering stops nothing
    held.release["a.mp4"].set()
    held.release["b.mp4"].set()
    assert events.wait_for(lambda: events.finished("a.mp4") and events.finished("b.mp4"))
    time.sleep(QUIET)
    assert not events.started("d.mp4")                      # c.mp4 still runs: at the new limit
    assert events.of("a.mp4")[-1].result["ok"] and events.of("b.mp4")[-1].result["ok"]

    held.release["c.mp4"].set()
    assert held.entered["d.mp4"].wait(WAIT)
    time.sleep(QUIET)
    assert not events.started("e.mp4")                      # one at a time from now on

    held.release_all()
    summary = running.join()
    assert summary.succeeded == names and summary.cancelled == []
    assert file_workers() == []


def test_raising_parallel_while_paused_starts_nothing_until_resume(tmp_path, ocr, qa):
    names = ["a.mp4", "b.mp4", "c.mp4"]
    held = HeldFiles(ocr, names)
    ctx, events = make_ctx()
    job = RunJob(str(tmp_path), [run_file(tmp_path, n) for n in names], 1)
    running = Running(job, ctx)
    assert held.entered["a.mp4"].wait(WAIT)
    job.pause()
    job.set_parallel(3)
    time.sleep(QUIET)
    assert not events.started("b.mp4") and not events.started("c.mp4")

    job.resume()
    assert held.entered["b.mp4"].wait(WAIT) and held.entered["c.mp4"].wait(WAIT)
    held.release_all()
    assert running.join().succeeded == names


def test_set_parallel_after_a_stop_or_the_end_starts_nothing(tmp_path, ocr, qa):
    entered = threading.Event()
    ocr.hooks["a.mp4"] = blocking_until_cancelled(entered)
    ctx, events = make_ctx()
    job = RunJob(str(tmp_path), [run_file(tmp_path, n) for n in ("a.mp4", "b.mp4", "c.mp4")], 1)
    running = Running(job, ctx)
    assert entered.wait(WAIT)
    ctx.cancel_event.set()
    job.set_parallel(3)
    summary = running.join()
    assert summary.cancelled == ["a.mp4", "b.mp4", "c.mp4"]
    assert events.of("b.mp4") == [] and events.of("c.mp4") == []

    job.set_parallel(5)                                     # the run is over: only the number changes
    assert job.parallel == 5
    assert file_workers() == []


def test_set_parallel_from_many_threads_keeps_the_limit(tmp_path, ocr, qa):
    names = [f"{i:02d}.mp4" for i in range(16)]
    for name in names:
        ocr.hooks[name] = lambda kwargs: time.sleep(0.02)
    ctx, events = make_ctx()
    job = RunJob(str(tmp_path), [run_file(tmp_path, n) for n in names], 1)
    running = Running(job, ctx)

    def churn(seed: int) -> None:
        for step in range(40):
            job.set_parallel(1 + (seed + step) % 4)
            time.sleep(0.001)

    callers = [threading.Thread(target=churn, args=(seed,)) for seed in range(6)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(WAIT)
    job.set_parallel(4)
    summary = running.join()
    assert summary.succeeded == names
    assert ocr.max_active <= 4
    assert file_workers() == []


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

def test_events_per_file_are_emitted_in_order(tmp_path, ocr, qa):
    def more_progress(kwargs):
        progress = kwargs["progress_callback"]
        progress("Extracting dialogue", 100)       # a repeat of the last report: not re-emitted
        progress("Extracting labels", 0)
        progress("Extracting labels", 40)
        kwargs["subtitle_callback"](3.25, 4.0, "标签")

    ocr.hooks["a.mp4"] = ocr.hooks["b.mp4"] = more_progress
    _, events = run_now(tmp_path, [run_file(tmp_path, "a.mp4"), run_file(tmp_path, "b.mp4")], parallel=2)

    for name in ("a.mp4", "b.mp4"):
        got = [(e.type, e.progress, e.message, e.result) for e in events.of(name)]
        assert got == [
            ("run_file_started", None, "", None),
            ("run_file_log", None, f"Starting OCR: {name}\n", None),
            ("run_file_progress", 0.0, "Extracting dialogue", None),
            ("run_subtitle", None, "", (1.0, 1.5, f"字幕 {name}")),
            ("run_file_progress", 1.0, "Extracting dialogue", None),
            ("run_file_progress", 0.0, "Extracting labels", None),
            ("run_file_progress", 0.4, "Extracting labels", None),
            ("run_subtitle", None, "", (3.25, 4.0, "标签")),
            ("run_file_log", None, "QA: 0 dialogues, 0 fixed, 0 deduped\n", None),
            ("run_file_log", None, "OCR completed successfully.\n", None),
            ("run_file_finished", None, "", {"ok": True, "lines": 3, "error": ""}),
        ]
    assert {(e.key, e.kind) for e in events.events} == {("run", "run")}


def file_log(events: Collector, name: str) -> list[str]:
    return [e.message for e in events.of(name) if e.type == "run_file_log"]


def test_a_file_logs_what_todays_worker_logs_on_success(tmp_path, ocr, qa):
    qa.stats = QAStats(dialogue_lines=12, fixed_lines=3, duplicates_removed=2)
    _, events = run_now(tmp_path, [run_file(tmp_path, "a b.mp4")])
    assert file_log(events, "a b.mp4") == [
        "Starting OCR: a b.mp4\n",
        "QA: 12 dialogues, 3 fixed, 2 deduped\n",
        "OCR completed successfully.\n",
    ]


def test_the_qa_line_reports_the_real_qa_stats(tmp_path, ocr):
    project = tmp_path / "project"
    project.mkdir()
    messy = (HEADER
             + "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,字幕\n"
             + "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,字幕\n")
    ocr.hooks["a.mp4"] = lambda kwargs: messy
    expected = tmp_path / "expected.ass"
    expected.write_text(messy, encoding="utf-8")
    stats = REAL_PROCESS_FILE(str(expected))

    _, events = run_now(project, [run_file(project, "a.mp4")])

    assert stats.duplicates_removed > 0
    assert file_log(events, "a.mp4")[1] == (
        f"QA: {stats.dialogue_lines} dialogues, {stats.fixed_lines} fixed, {stats.duplicates_removed} deduped\n")


@pytest.mark.parametrize("ranges, lines", [
    (None, []),
    ([("02:33", "21:20")], []),                                      # one range: no range lines, as today
    ([("02:33", "21:20"), ("22:00", None)], ["Range 1/2: 02:33 - 21:20\n", "Range 2/2: 22:00 - end\n"]),
    ([(None, "1:00"), ("2:00", "3:00"), ("4:00", None)],
     ["Range 1/3: 0:00 - 1:00\n", "Range 2/3: 2:00 - 3:00\n", "Range 3/3: 4:00 - end\n"]),
])
def test_several_ranges_log_one_line_per_range_before_ocr_starts(tmp_path, ocr, qa, ranges, lines):
    _, events = run_now(tmp_path, [run_file(tmp_path, "a.mp4", ranges)])
    assert file_log(events, "a.mp4") == (["Starting OCR: a.mp4\n"] + lines
                                         + ["QA: 0 dialogues, 0 fixed, 0 deduped\n", "OCR completed successfully.\n"])
    types = [(e.type, e.message) for e in events.of("a.mp4")]
    first_progress = next(i for i, (t, _) in enumerate(types) if t == "run_file_progress")
    assert [m for t, m in types[:first_progress] if t == "run_file_log"] == ["Starting OCR: a.mp4\n"] + lines


def test_a_stopped_file_logs_ocr_cancelled(tmp_path, ocr, qa):
    entered = threading.Event()
    ocr.hooks["a.mp4"] = blocking_until_cancelled(entered, produce=ass_text("partial"))
    ctx, events = make_ctx()
    running = Running(RunJob(str(tmp_path), [run_file(tmp_path, "a.mp4")], 1), ctx)
    assert entered.wait(WAIT)
    ctx.cancel_event.set()
    running.join()
    assert file_log(events, "a.mp4") == ["Starting OCR: a.mp4\n", "OCR cancelled.\n"]
    assert events.of("a.mp4")[-1].type == "run_file_finished"


def test_a_failed_file_logs_the_error_and_its_traceback(tmp_path, ocr, qa):
    def explode(kwargs):
        raise RuntimeError("decoder exploded\nat frame 12")

    ocr.hooks["a.mp4"] = explode
    _, events = run_now(tmp_path, [run_file(tmp_path, "a.mp4")])

    starting, failed = file_log(events, "a.mp4")
    assert starting == "Starting OCR: a.mp4\n"
    assert failed.startswith("OCR failed: decoder exploded\nat frame 12\nTraceback (most recent call last):\n")
    assert failed.endswith("RuntimeError: decoder exploded\nat frame 12\n")
    assert events.of("a.mp4")[-1].type == "run_file_finished"
    assert any(e.type == "log" and "Traceback" in e.message for e in events.events)    # still logged as before


def test_a_qa_failure_logs_no_qa_line(tmp_path, ocr, qa):
    qa.raise_for["a.ass.partial"] = ValueError("bad ass")
    _, events = run_now(tmp_path, [run_file(tmp_path, "a.mp4")])
    log = file_log(events, "a.mp4")
    assert [line.split("\n")[0] for line in log] == ["Starting OCR: a.mp4", "OCR failed: bad ass"]


def test_a_file_that_produced_nothing_logs_the_failure(tmp_path, ocr, qa):
    ocr.hooks["a.mp4"] = lambda kwargs: ""
    _, events = run_now(tmp_path, [run_file(tmp_path, "a.mp4", [("0:00", "1:00"), ("2:00", "3:00")])])
    assert file_log(events, "a.mp4")[-1] == "OCR failed: no subtitles produced\n"


def test_output_name_is_the_file_a_run_writes_into_chi(tmp_path, ocr, qa):
    from core.jobs.run import output_name

    assert output_name("ZS2_-_11_[1080p]TXHBR.mp4") == "ZS2_-_11_[1080p]TXHBR.ass"
    assert output_name("a.b.mkv") == "a.b.ass"
    assert output_name("第一集.mp4") == "第一集.ass"
    names = ["a.b.mkv", "第一集.mp4"]
    run_now(tmp_path, [run_file(tmp_path, name) for name in names])
    assert sorted(os.listdir(tmp_path / "chi")) == sorted(output_name(name) for name in names)


def test_progress_is_a_fraction_clamped_to_0_1(tmp_path, ocr, qa):
    def odd_progress(kwargs):
        kwargs["progress_callback"]("Extracting labels", 250)
        kwargs["progress_callback"]("Extracting labels", -5)

    ocr.hooks["a.mp4"] = odd_progress
    _, events = run_now(tmp_path, [run_file(tmp_path, "a.mp4")])
    assert [e.progress for e in events.of("a.mp4") if e.type == "run_file_progress"] == [0.0, 1.0, 1.0, 0.0]


# --------------------------------------------------------------------------
# Through the runner
# --------------------------------------------------------------------------

def test_through_the_runner_a_run_finishes_with_its_summary(tmp_path, ocr, qa):
    events = Collector()
    runner = JobRunner(events)
    try:
        runner.submit(RunJob(str(tmp_path), [run_file(tmp_path, "a.mp4"), run_file(tmp_path, "b.mp4")], 2))
        assert events.wait_for(lambda: any(e.type in ("finished", "failed", "cancelled") for e in events.events))
    finally:
        assert runner.shutdown(WAIT)
    terminal = events.events[-1]
    assert terminal.type == "finished", terminal.error
    assert terminal.result.succeeded == ["a.mp4", "b.mp4"]
    assert [e.type for e in events.of("a.mp4")][0] == "run_file_started"


def test_through_the_runner_cancelling_the_run_stops_it_with_its_summary(tmp_path, ocr, qa):
    entered = threading.Event()
    ocr.hooks["a.mp4"] = blocking_until_cancelled(entered)
    events = Collector()
    runner = JobRunner(events)
    try:
        runner.submit(RunJob(str(tmp_path), [run_file(tmp_path, "a.mp4"), run_file(tmp_path, "b.mp4")], 1))
        assert entered.wait(WAIT)
        runner.cancel("run")
        assert events.wait_for(lambda: any(e.type in ("finished", "failed", "cancelled") for e in events.events))
    finally:
        assert runner.shutdown(WAIT)
    terminal = events.events[-1]
    assert terminal.type == "cancelled"
    assert terminal.result.cancelled == ["a.mp4", "b.mp4"]


def test_run_module_imports_no_qt():
    code = ("import sys, core.jobs.run; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('PyQt6', 'PyQt5', 'PySide6')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False,
                          cwd=Path(__file__).resolve().parent.parent)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------------------------
# Fidelity: real OCR on the reference episode
# --------------------------------------------------------------------------

FIDELITY_CASES = {                     # case -> pinned golden digest prefix (as in test_concurrent_fidelity)
    "slay_1080p_dialogue": "c76ee2af4ecd",
    "slay_1080p_multirange": "273ba6ac6c03",
}


def _todays_worker_output(ref_dir: Path, name: str, case) -> bytes:
    """chi/<stem>.ass as the old `core/ocr_worker.py` flow wrote it.

    That module was deleted in plan 3B Task 6, so its call sequence lives here
    instead, transcribed from `OCRWorker._run_ocr` and run synchronously on
    this thread in `ref_dir`, a tmp dir of its own:

      - the OCR output goes NEXT TO THE VIDEO, `<video dir>/<stem>.ass`;
      - no range or one range -> `videocr.api.save_subtitles_to_file(**kwargs,
        file_path=, time_start=, time_end=, progress_callback=,
        subtitle_callback=, cancel_event=)`;
      - several ranges -> `videocr.api.get_subtitles(**kwargs, time_ranges=,
        <same three>)`, and the returned text is written (UTF-8) only if there
        is some and the cancel event is not set;
      - then `core.ass_qafix.process_file` on that file, then
        `shutil.move` into `<video dir>/chi/<stem>.ass`.

    `kwargs` come from `core.project.ocr_kwargs.ocr_call_for`, as they did
    when this reference still built a `Config`/`FileConfig` for `OCRWorker`:
    `tests/test_ocr_kwargs.py` pins that resolution literally, this test pins
    what the call sequence around it produces.
    """
    entry = FileEntry(
        name,
        crop=Crop(*case.crop, Source.MANUAL),
        brightness=Brightness(case.brightness, Source.MANUAL),
        time_ranges=TimeRanges([TimeRange(s, e) for s, e in case.time_ranges], Source.MANUAL),
    )
    call = ocr_call_for(entry, FolderSettings(labels_enabled=case.detect_labels), str(ref_dir))
    kwargs, time_ranges = dict(call.kwargs), list(call.time_ranges)
    ass_source = ref_dir / f"{stem(name)}.ass"
    cancel_event = threading.Event()

    def progress_callback(_phase, _percent):
        pass

    def subtitle_callback(_start, _end, _text):
        pass

    # api.* are still the real functions here: the caller monkeypatches them
    # only after this reference has produced its output.
    if len(time_ranges) <= 1:
        api.save_subtitles_to_file(
            **kwargs,
            file_path=str(ass_source),
            time_start=time_ranges[0][0] if time_ranges else "0:00",
            time_end=time_ranges[0][1] if time_ranges else "",
            progress_callback=progress_callback,
            subtitle_callback=subtitle_callback,
            cancel_event=cancel_event,
        )
    else:
        result = api.get_subtitles(
            **kwargs,
            time_ranges=time_ranges,
            progress_callback=progress_callback,
            subtitle_callback=subtitle_callback,
            cancel_event=cancel_event,
        )
        if result and not cancel_event.is_set():
            with open(ass_source, "w", encoding="utf-8") as f:
                f.write(result)

    assert ass_source.exists(), f"{name}: the reference flow produced no .ass"
    REAL_PROCESS_FILE(str(ass_source))
    destination = ref_dir / "chi" / f"{stem(name)}.ass"
    shutil.move(str(ass_source), str(destination))
    return destination.read_bytes()


@pytest.mark.slow
@pytest.mark.needs_media
@pytest.mark.timeout(1800)
def test_run_output_is_byte_identical_to_todays_worker_flow_and_matches_the_goldens(tmp_path, monkeypatch):
    from tools.fidelity_check import GOLDEN_DIR, digest, load_cases, media_root

    cases = [case for case in load_cases() if case.name in FIDELITY_CASES]
    assert [case.name for case in cases] == list(FIDELITY_CASES)
    root = media_root()
    run_dir, ref_dir = tmp_path / "run", tmp_path / "ref"
    (ref_dir / "chi").mkdir(parents=True)
    run_dir.mkdir()
    names = {}
    for case in cases:
        source = root / case.project / case.file
        if not source.exists():
            pytest.skip(f"missing input for {case.name}: {source}")
        # One symlink per case (same episode), so both run in one RunJob; the
        # reference media stays read-only.
        names[case.name] = f"{case.name}{source.suffix}"
        (run_dir / names[case.name]).symlink_to(source)
        (ref_dir / names[case.name]).symlink_to(source)

    expected = {case.name: _todays_worker_output(ref_dir, names[case.name], case) for case in cases}

    calls: list[tuple[str, tuple, dict]] = []
    real_save, real_get = api.save_subtitles_to_file, api.get_subtitles

    def save_subtitles_to_file(*args, **kwargs):
        calls.append(("save_subtitles_to_file", args, kwargs))
        return real_save(*args, **kwargs)

    def get_subtitles(*args, **kwargs):
        calls.append(("get_subtitles", args, kwargs))
        return real_get(*args, **kwargs)

    pre_qa: dict[str, str] = {}

    def process_file(path, *args, **kwargs):
        pre_qa[os.path.basename(path)] = Path(path).read_bytes().decode("utf-8")
        return REAL_PROCESS_FILE(path, *args, **kwargs)

    monkeypatch.setattr(api, "save_subtitles_to_file", save_subtitles_to_file)
    monkeypatch.setattr(api, "get_subtitles", get_subtitles)
    monkeypatch.setattr(ass_qafix, "process_file", process_file)

    files = []
    for case in cases:
        entry = FileEntry(
            names[case.name],
            crop=Crop(*case.crop, Source.MANUAL),
            brightness=Brightness(case.brightness, Source.MANUAL),
            time_ranges=TimeRanges([TimeRange(s, e) for s, e in case.time_ranges], Source.MANUAL),
        )
        folder = FolderSettings(labels_enabled=case.detect_labels)
        files.append(RunFile(entry.name, ocr_call_for(entry, folder, str(run_dir))))
    ctx, events = make_ctx()
    started = time.perf_counter()
    summary = RunJob(str(run_dir), files, parallel=2).run(ctx)
    print(f"run: {time.perf_counter() - started:.1f}s {summary}")

    assert summary.succeeded == [names[case.name] for case in cases], summary
    assert sorted(os.listdir(run_dir / "chi")) == sorted(f"{stem(n)}.ass" for n in names.values())

    # Which branch of the old call shape each file took.
    def calls_for(name):
        video = str(run_dir / name)
        top = [(fn, kw) for fn, args, kw in calls if kw.get("video_path") == video]
        inner = [(fn, kw) for fn, args, kw in calls if args and args[0] == video]
        return top, inner

    top, inner = calls_for(names["slay_1080p_dialogue"])
    assert [fn for fn, _ in top] == ["save_subtitles_to_file"]
    assert (top[0][1]["time_start"], top[0][1]["time_end"]) == ("9:30", "11:30")
    assert top[0][1]["file_path"] == str(partial_of(run_dir, names["slay_1080p_dialogue"]))
    assert "time_ranges" not in top[0][1]
    assert [fn for fn, _ in inner] == ["get_subtitles"]         # save_subtitles_to_file's own call
    assert "time_ranges" not in inner[0][1]

    top, inner = calls_for(names["slay_1080p_multirange"])
    assert [fn for fn, _ in top] == ["get_subtitles"]
    assert top[0][1]["time_ranges"] == [("9:30", "10:00"), ("14:00", "14:30")]
    assert inner == []

    for case in cases:
        name = names[case.name]
        produced = final_of(run_dir, name).read_bytes()
        assert produced == expected[case.name], f"{case.name}: run output differs from the old OCRWorker flow"
        golden = digest((GOLDEN_DIR / f"{case.name}.ass").read_text(encoding="utf-8"))
        assert golden.startswith(FIDELITY_CASES[case.name]), case.name
        got = digest(pre_qa[f"{stem(name)}.ass.partial"])
        assert got == golden, f"{case.name}: golden {golden[:12]} != produced {got[:12]}"
        lines = produced.decode("utf-8").count("\nDialogue:")
        finished = [e.result for e in events.of(name) if e.type == "run_file_finished"]
        assert finished == [{"ok": True, "lines": lines, "error": ""}] and lines > 0
        print(f"{case.name}: {lines} lines, pre-QA digest {got[:12]}, {len(produced)} bytes identical to OCRWorker")
