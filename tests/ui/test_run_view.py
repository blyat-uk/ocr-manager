"""Plan 3B Task 5: Start with the overwrite question, the run mode of the
window (top bar, Run view, live feed, GPU and idle-worker footer) and the
logs window.

Every test drives a real ProjectController over `fake_runner`
(tests/ui/conftest.py): the run job never runs, and the test delivers the
events a real RunJob sends, then drains them with
`controller.drain_events()`. QSettings point at a per-test directory.
"""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton

from app.controller import ProjectController
from app.logbook import LOG_LIMIT
from app.main_window import MODE_REVIEW, MODE_RUN, OVERWRITE_TEXT, MainWindow
from app.run_snapshot import (
    CANCELLED,
    DONE,
    FAILED,
    QUEUED,
    RUNNING,
    RunFileRow,
    RunSnapshot,
    eta_seconds,
    run_status_text,
)
from app.views import run_view as run_view_module
from app.views.run_view import LIVE_NOTE, feed_time, parse_gpu_utilisation, phase_text
from core.jobs.run import RunSummary
from core.project import (
    Brightness,
    Crop,
    FileEntry,
    FolderSettings,
    Media,
    Project,
    ReviewState,
    Source,
    TimeRanges,
    save_project,
)

NAMES = ["ep01.mkv", "ep02.mkv", "ep03.mkv", "ep04.mkv", "ep05.mkv"]
WAIT_MS = 5000
Yes, No = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No


# --------------------------------------------------------------------------
# Helpers and fixtures
# --------------------------------------------------------------------------

def wait_for(predicate, timeout_ms: int = WAIT_MS) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        QTest.qWait(10)
    return True


def settle() -> None:
    QApplication.processEvents()


def ready_entry(name: str) -> FileEntry:
    """A reviewed file with known media, crop, brightness and a whole-file range choice."""
    return FileEntry(name, media=Media(1920, 888, 1628.0, 25.0), review=ReviewState.REVIEWED,
                     crop=Crop(288, 786, 1344, 53, Source.MANUAL), brightness=Brightness(209, Source.MANUAL),
                     time_ranges=TimeRanges([], Source.MANUAL), sample_time=578.0)


class Recorder:
    """Wraps a controller command on the instance: records (args, kwargs), then runs it."""

    def __init__(self, controller, name: str):
        self.calls: list[tuple[tuple, dict]] = []
        original = getattr(controller, name)

        def wrapper(*args, **kwargs):
            self.calls.append((args, kwargs))
            return original(*args, **kwargs)

        setattr(controller, name, wrapper)


@pytest.fixture(autouse=True)
def settings_dir(tmp_path):
    path = tmp_path / "qsettings"
    path.mkdir()
    for fmt in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, str(path))
    return path


@pytest.fixture
def controller(qapp, fake_runner):
    made = ProjectController(fake_runner, save_debounce_ms=10)
    yield made
    made.shutdown(timeout=0.5)


@pytest.fixture
def notifications(monkeypatch):
    import subprocess

    sent = []

    def fake_run(args, **kwargs):
        sent.append(list(args))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return sent


@pytest.fixture
def make_window(controller, notifications):
    windows = []

    def make(names=NAMES, *, done=(), tmp_project=None, **folder) -> MainWindow:
        folder_path = tmp_project(list(names))
        folder.setdefault("labels_enabled", False)
        save_project(Project(path=str(folder_path), folder=FolderSettings(**folder),
                             files={name: ready_entry(name) for name in names}))
        for name in done:
            (folder_path / "chi").mkdir(exist_ok=True)
            (folder_path / "chi" / f"{Path(name).stem}.ass").write_text("Dialogue: old\n", encoding="utf-8")
        window = MainWindow(controller)
        window.resize(1440, 900)
        windows.append(window)
        window.open_folder(str(folder_path))
        settle()
        return window

    yield make
    for window in windows:
        window.close()
        window.deleteLater()
    settle()


@pytest.fixture
def window(make_window, tmp_project):
    return make_window(tmp_project=tmp_project, ocr_parallel=4)


def answer(monkeypatch, reply) -> list[tuple]:
    asked = []

    def question(parent, title, text, buttons, default):
        asked.append((parent, title, text, buttons, default))
        return reply

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return asked


def start(window: MainWindow, fake_runner):
    """Click Start (no done files, so no question) and return the run submission."""
    window.topbar.start_button.click()
    settle()
    return fake_runner.last("run")


def deliver(window: MainWindow) -> None:
    window.controller.drain_events()
    settle()


def switch_to(window: MainWindow, index: int) -> None:
    buttons = window.topbar.run_switch.findChildren(QPushButton)
    buttons[index].click()
    settle()


# --------------------------------------------------------------------------
# Start and the overwrite question
# --------------------------------------------------------------------------

@pytest.mark.parametrize("reply, expected", [
    (No, ["ep01.mkv", "ep03.mkv", "ep05.mkv"]),
    (Yes, NAMES),
])
def test_start_asks_before_replacing_done_files(make_window, tmp_project, fake_runner, monkeypatch, reply, expected):
    window = make_window(tmp_project=tmp_project, done=("ep02.mkv", "ep04.mkv"))
    folder = Path(window.controller.project.path)
    asked = answer(monkeypatch, reply)
    started = Recorder(window.controller, "start_run")
    assert window.topbar.start_button.text() == "▶ Start 3 ready files"

    window.topbar.start_button.click()
    settle()

    assert len(asked) == 1
    parent, _title, text, buttons, default = asked[0]
    assert parent is window
    assert text == "2 file(s) already have subtitles in chi/. Re-run and replace them when their new output is ready?"
    assert text == OVERWRITE_TEXT.format(n=2)
    assert buttons == Yes | No and default == No
    assert started.calls == [((expected,), {})]
    assert fake_runner.last("run").job.files[0].name == "ep01.mkv"
    assert (folder / "chi" / "ep02.ass").read_text(encoding="utf-8") == "Dialogue: old\n"    # nothing deleted
    assert window.mode() == MODE_RUN


def test_start_without_done_files_does_not_ask(window, fake_runner, monkeypatch):
    asked = answer(monkeypatch, No)
    run = start(window, fake_runner)
    assert asked == []
    assert [run_file.name for run_file in run.job.files] == NAMES
    assert window.mode() == MODE_RUN


def test_only_done_files_can_be_re_run_and_no_starts_nothing(make_window, tmp_project, fake_runner, monkeypatch):
    window = make_window(["a.mkv", "b.mkv"], tmp_project=tmp_project, done=("a.mkv", "b.mkv"))
    asked = answer(monkeypatch, No)
    button = window.topbar.start_button
    assert button.text() == "▶ Re-run 2 done files" and button.isEnabled()
    button.click()
    settle()
    assert asked and asked[0][2] == OVERWRITE_TEXT.format(n=2)
    assert fake_runner.of_kind("run") == []
    assert window.mode() == MODE_REVIEW and window.topbar.run_switch.isHidden()


def test_start_says_re_run_for_one_done_file_and_stays_disabled_with_nothing_to_run(make_window, tmp_project):
    window = make_window(["a.mkv", "b.mkv"], tmp_project=tmp_project, done=("a.mkv",))
    button = window.topbar.start_button
    assert button.text() == "▶ Start 1 ready file" and button.isEnabled()

    window.controller.set_skipped("b.mkv", True)
    settle()
    assert button.text() == "▶ Re-run 1 done file" and button.isEnabled()

    window.controller.set_skipped("a.mkv", True)                # nothing ready, nothing to re-run
    settle()
    assert button.text() == "▶ Start 0 ready files" and not button.isEnabled()


def test_the_logs_window_opens_during_a_run(window, fake_runner):
    start(window, fake_runner)
    top = window.topbar
    assert not top.logs_button.isHidden() and top.settings_button.isHidden()
    top.logs_button.click()
    settle()
    assert window.logs_window is not None and window.logs_window.isVisible()


def test_a_refused_start_is_shown_beside_start_without_switching_modes(make_window, tmp_project, fake_runner):
    window = make_window(["a.mkv", "a.mp4"], tmp_project=tmp_project)
    top = window.topbar
    top.start_button.click()
    settle()
    assert fake_runner.of_kind("run") == []
    assert not top.start_error_label.isHidden()
    assert top.start_error_label.full_text() == "a.mkv and a.mp4 both write chi/a.ass"
    assert window.mode() == MODE_REVIEW and top.run_switch.isHidden()

    window.controller.set_skipped("a.mp4", True)
    settle()
    top.start_button.click()
    settle()
    assert top.start_error_label.isHidden()
    assert window.mode() == MODE_RUN


# --------------------------------------------------------------------------
# The window's run mode and the top bar
# --------------------------------------------------------------------------

def test_the_run_switch_appears_and_toggles_the_centre_and_right_area(window, fake_runner, make_window,
                                                                      tmp_project):
    top = window.topbar
    assert top.run_switch.isHidden() and window.mode() == MODE_REVIEW
    assert window.modes.currentWidget() is window.review_area
    run = start(window, fake_runner)

    assert not top.run_switch.isHidden()
    assert top.run_switch.labels() == ["Review", "Run"] and top.run_switch.current() == MODE_RUN
    assert window.modes.currentWidget() is window.run_view
    assert window.queue.parentWidget() is window.workbench and window.run_view.parentWidget() is window.modes
    for hidden in (top.settings_button, top.start_button, top._chips):
        assert hidden.isHidden()
    assert not top.logs_button.isHidden()                       # logs stay reachable during a run
    assert not top.pause_button.isHidden() and not top.stop_button.isHidden()
    assert top.pause_button.text() == "⏸ pause" and top.stop_button.text() == "■ stop"

    switch_to(window, MODE_REVIEW)
    assert window.mode() == MODE_REVIEW and window.modes.currentWidget() is window.review_area
    switch_to(window, MODE_RUN)
    assert window.modes.currentWidget() is window.run_view

    fake_runner.emit(run, "run_file_started", file="ep01.mkv")
    fake_runner.emit(run, "run_file_finished", file="ep01.mkv", result={"ok": True, "lines": 3, "error": ""})
    fake_runner.finish(run, RunSummary(["ep01.mkv"], {}, NAMES[1:], 5.0))
    deliver(window)
    for shown in (top.settings_button, top.logs_button, top.start_button, top._chips):
        assert not shown.isHidden()                             # the top bar is back to itself
    assert top.pause_button.isHidden() and top.stop_button.isHidden()
    assert not top.run_switch.isHidden()                        # the switch stays until another folder opens
    assert window.mode() == MODE_RUN
    assert window.run_view.row("ep01.mkv").phase_label.text() == "done"      # the finished run's table stays

    window.open_folder(str(tmp_project(["x.mkv"])))
    settle()
    assert top.run_switch.isHidden() and window.mode() == MODE_REVIEW
    assert window.run_view.names() == []


def test_top_bar_status_text_follows_the_run(window, fake_runner, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    run = start(window, fake_runner)
    fake_runner.emit(run, "run_file_started", file="ep01.mkv")
    fake_runner.emit(run, "run_file_started", file="ep02.mkv")
    deliver(window)
    clock[0] = 1042.0
    window.topbar.refresh()
    assert window.topbar.path_label.full_text() == "· running · 0 of 5 done · 0:42 elapsed"   # no ETA yet

    clock[0] = 1100.0
    fake_runner.emit(run, "run_file_finished", file="ep01.mkv", result={"ok": True, "lines": 3, "error": ""})
    deliver(window)
    clock[0] = 1862.0
    window.topbar.refresh()
    # average 100 s x 4 remaining / 1 active worker = 400 s
    assert window.topbar.path_label.full_text() == "· running · 1 of 5 done · 14:22 elapsed · ~7 min left"

    window.topbar.pause_button.click()
    settle()
    assert window.topbar.path_label.full_text().startswith("· paused · 1 of 5 done")


def test_eta_is_average_finished_time_times_remaining_over_active_workers():
    rows = (RunFileRow("a", DONE, started_at=0.0, finished_at=100.0),
            RunFileRow("b", FAILED, started_at=50.0, finished_at=250.0),
            RunFileRow("c", RUNNING, started_at=260.0),
            RunFileRow("d", RUNNING, started_at=270.0),
            RunFileRow("e", QUEUED))
    snapshot = RunSnapshot(rows, started_at=0.0, parallel=2)
    assert eta_seconds(snapshot) == pytest.approx(150.0 * 3 / 2)
    assert run_status_text(snapshot, 842.0) == "· running · 1 of 5 done · 14:02 elapsed · ~4 min left"

    one_running = RunSnapshot(rows[:3] + (RunFileRow("d", QUEUED), rows[4]), started_at=0.0, parallel=2)
    assert eta_seconds(one_running) == pytest.approx(150.0 * 3 / 1)
    none_running = RunSnapshot(rows[:2] + (RunFileRow("c"), RunFileRow("d"), RunFileRow("e")), started_at=0.0,
                               parallel=2, paused=True)
    assert eta_seconds(none_running) == pytest.approx(150.0 * 3)           # at least one worker
    assert run_status_text(none_running, 3725.0) == "· paused · 1 of 5 done · 1:02:05 elapsed · ~8 min left"

    cancelled_only = RunSnapshot((RunFileRow("a", CANCELLED, started_at=0.0, finished_at=10.0), RunFileRow("b")),
                                 started_at=0.0, parallel=1)
    assert eta_seconds(cancelled_only) is None                              # a stopped file is no measure
    assert eta_seconds(RunSnapshot(rows[2:], started_at=0.0, parallel=2)) is None
    assert eta_seconds(RunSnapshot(rows[:2], started_at=0.0, parallel=2)) is None     # nothing left
    short = RunSnapshot((RunFileRow("a", DONE, started_at=0.0, finished_at=20.0), RunFileRow("b", RUNNING)),
                        started_at=0.0, parallel=1, stopping=True)
    assert run_status_text(short, 30.0) == "· stopping · 1 of 2 done · 0:30 elapsed"        # no ETA when stopping
    assert run_status_text(RunSnapshot(short.files, started_at=0.0, parallel=1), 30.0) == \
        "· running · 1 of 2 done · 0:30 elapsed · ~20 s left"


def test_pause_resume_and_stop_buttons_call_the_controller(window, fake_runner):
    controller = window.controller
    paused, resumed, stopped = (Recorder(controller, name) for name in ("pause_run", "resume_run", "stop_run"))
    run = start(window, fake_runner)
    top = window.topbar

    top.pause_button.click()
    settle()
    assert paused.calls == [((), {})] and run.job._paused
    assert top.pause_button.text() == "▶ resume"
    top.pause_button.click()
    settle()
    assert resumed.calls == [((), {})] and not run.job._paused
    assert top.pause_button.text() == "⏸ pause"

    top.stop_button.click()
    settle()
    assert stopped.calls == [((), {})] and fake_runner.cancelled_keys == ["run"]
    assert not top.stop_button.isEnabled() and not top.pause_button.isEnabled()
    assert top.path_label.full_text().startswith("· stopping ·")


# --------------------------------------------------------------------------
# The Run view
# --------------------------------------------------------------------------

def test_run_events_render_rows_progress_results_failures_and_cancelled(window, fake_runner):
    run = start(window, fake_runner)
    view = window.run_view
    assert view.header_texts() == ["FILE", "PHASE", "PROGRESS", "RESULT"]
    assert view.names() == NAMES

    fake_runner.emit(run, "run_file_started", file="ep01.mkv")
    fake_runner.emit(run, "run_file_progress", file="ep01.mkv", progress=0.44, message="Extracting dialogue")
    fake_runner.emit(run, "run_subtitle", file="ep01.mkv", result=(578.0, 580.0, "你竟掌握了鲲鹏道法"))
    fake_runner.emit(run, "run_subtitle", file="ep01.mkv", result=(581.0, 583.0, "我早已不是当年的我"))
    fake_runner.emit(run, "run_file_started", file="ep02.mkv")
    fake_runner.emit(run, "run_file_progress", file="ep02.mkv", progress=0.71, message="Extracting labels")
    fake_runner.emit(run, "run_file_started", file="ep03.mkv")
    fake_runner.emit(run, "run_file_finished", file="ep03.mkv", result={"ok": True, "lines": 412, "error": ""})
    fake_runner.emit(run, "run_file_started", file="ep04.mkv")
    fake_runner.emit(run, "run_file_progress", file="ep04.mkv", progress=0.3, message="Extracting dialogue")
    fake_runner.emit(run, "run_file_finished", file="ep04.mkv",
                     result={"ok": False, "lines": 0, "error": "decoder exploded"})
    deliver(window)

    dialogue, labels, done, failed, queued = (view.row(name) for name in NAMES)
    assert (dialogue.phase_label.text(), dialogue.phase_tone()) == ("dialogue", "run")
    assert not dialogue.phase_dot.isHidden()
    assert (dialogue.progress.fraction(), dialogue.progress.tone()) == (pytest.approx(0.44), "run")
    assert dialogue.result_label.text() == "2 lines"
    assert labels.phase_label.text() == "labels" and labels.result_label.text() == ""
    assert (done.phase_label.text(), done.phase_tone()) == ("done", "ok")
    assert done.phase_dot.isHidden()
    assert (done.progress.fraction(), done.progress.tone()) == (1.0, "done")
    assert (done.result_label.text(), done.result_tone()) == ("412 lines", "ok")
    assert (failed.phase_label.text(), failed.phase_tone()) == ("failed", "bad")
    assert failed.phase_label.toolTip() == "decoder exploded"
    assert failed.progress.tone() == "bad"
    assert (queued.phase_label.text(), queued.phase_tone()) == ("queued", "dim")
    assert queued.progress.fraction() == 0.0 and queued.result_label.text() == ""

    window.controller.stop_run()
    fake_runner.emit(run, "run_file_finished", file="ep01.mkv", result={"ok": False, "lines": 0, "error": "cancelled"})
    fake_runner.emit(run, "run_file_finished", file="ep02.mkv", result={"ok": False, "lines": 0, "error": "cancelled"})
    fake_runner.finish(run, RunSummary(["ep03.mkv"], {"ep04.mkv": "decoder exploded"},
                                       ["ep01.mkv", "ep02.mkv", "ep05.mkv"], 9.0), "cancelled")
    deliver(window)
    for name in ("ep01.mkv", "ep02.mkv", "ep05.mkv"):
        row = view.row(name)
        assert (row.phase_label.text(), row.phase_tone()) == ("cancelled", "dim")
        assert row.progress.tone() == "dim"
    assert view.row("ep03.mkv").phase_label.text() == "done"


def test_phase_and_feed_texts():
    assert phase_text("Extracting dialogue") == "dialogue"
    assert phase_text("Extracting labels") == "labels"
    assert phase_text("") == "starting"
    assert phase_text("QA fixing") == "qa fixing"
    assert feed_time(578.9) == "09:38"
    assert feed_time(3725.0) == "62:05"


def test_the_live_feed_follows_the_newest_file_and_the_menu_switches(window, fake_runner):
    run = start(window, fake_runner)
    view = window.run_view
    assert view.live_title.full_text() == "LIVE"
    assert view.note_label.text() == LIVE_NOTE == ("You can keep reviewing other episodes while this runs — nothing "
                                                  "is blocked, and edits apply to files that haven't started yet.")
    assert view.live_panel.minimumWidth() == view.live_panel.maximumWidth() == 300

    fake_runner.emit(run, "run_file_started", file="ep01.mkv")
    fake_runner.emit(run, "run_subtitle", file="ep01.mkv", result=(578.0, 580.0, "你竟掌握了鲲鹏道法"))
    fake_runner.emit(run, "run_subtitle", file="ep01.mkv", result=(581.0, 583.0, "我早已不是当年的我"))
    deliver(window)
    assert view.followed() == "ep01.mkv"
    assert view.live_title.full_text() == "LIVE · ep01.mkv"
    assert view.feed_lines() == ["09:38 你竟掌握了鲲鹏道法", "09:41 我早已不是当年的我"]

    fake_runner.emit(run, "run_file_started", file="ep02.mkv")
    fake_runner.emit(run, "run_subtitle", file="ep02.mkv", result=(10.0, 12.0, "今日便让你见识见识"))
    fake_runner.emit(run, "run_subtitle", file="ep01.mkv", result=(584.0, 586.0, "纵使千难万险"))
    deliver(window)
    assert view.followed() == "ep02.mkv"
    assert view.feed_lines() == ["00:10 今日便让你见识见识"]

    menu = view.follow_menu()
    actions = {action.text(): action for action in menu.actions() if not action.isSeparator()}
    assert list(actions) == ["Newest file", "ep01.mkv", "ep02.mkv"]
    assert actions["Newest file"].isChecked()
    actions["ep01.mkv"].trigger()
    settle()
    assert view.followed() == "ep01.mkv" and view.live_title.full_text() == "LIVE · ep01.mkv"
    assert view.feed_lines() == ["09:38 你竟掌握了鲲鹏道法", "09:41 我早已不是当年的我", "09:44 纵使千难万险"]

    fake_runner.emit(run, "run_file_started", file="ep03.mkv")
    deliver(window)
    assert view.followed() == "ep01.mkv"                        # a chosen file stays followed
    menu = view.follow_menu()
    actions = {action.text(): action for action in menu.actions() if not action.isSeparator()}
    assert actions["ep01.mkv"].isChecked() and not actions["Newest file"].isChecked()
    actions["Newest file"].trigger()
    settle()
    assert view.followed() == "ep03.mkv" and view.feed_lines() == []


def test_the_idle_hint_offers_to_raise_parallel_files(window, fake_runner):
    controller = window.controller
    run = start(window, fake_runner)
    view = window.run_view
    fake_runner.emit(run, "run_file_started", file="ep01.mkv")
    fake_runner.emit(run, "run_file_started", file="ep02.mkv")
    deliver(window)

    assert not view.hint_label.isHidden() and not view.raise_button.isHidden()
    assert view.hint_label.text() == "2 workers idle →"
    assert view.raise_button.text() == "raise to 4"
    assert view.hint_text() == "2 workers idle → raise to 4"
    assert "not started yet" in view.raise_button.toolTip()
    updates = Recorder(controller, "update_folder")
    view.raise_button.click()
    settle()
    assert updates.calls == [((), {"ocr_parallel": 4})]

    controller.pause_run()                                      # a paused run starts nothing: no hint
    settle()
    assert view.hint_label.isHidden() and view.raise_button.isHidden()
    assert view.hint_text() == ""
    controller.resume_run()
    for name in NAMES[2:]:
        fake_runner.emit(run, "run_file_started", file=name)
    fake_runner.emit(run, "run_file_finished", file="ep01.mkv", result={"ok": True, "lines": 1, "error": ""})
    deliver(window)
    assert view.hint_label.isHidden()                           # nothing queued: nothing to raise for


def test_one_idle_worker_reads_singular(make_window, tmp_project, fake_runner):
    window = make_window(tmp_project=tmp_project, ocr_parallel=2)
    run = start(window, fake_runner)
    fake_runner.emit(run, "run_file_started", file="ep01.mkv")
    deliver(window)
    assert window.run_view.hint_label.text() == "1 worker idle →"
    assert window.run_view.raise_button.text() == "raise to 2"


def test_queued_files_offer_a_higher_parallel_that_reaches_the_run_job(make_window, tmp_project, fake_runner):
    """Every worker is busy and files wait: the offer is parallel + 2 (B13 as amended)."""
    window = make_window(tmp_project=tmp_project, ocr_parallel=2)
    controller, view = window.controller, window.run_view
    run = start(window, fake_runner)
    fake_runner.emit(run, "run_file_started", file="ep01.mkv")
    fake_runner.emit(run, "run_file_started", file="ep02.mkv")
    deliver(window)

    assert view.hint_label.text() == "3 files queued ·"
    assert view.raise_button.text() == "raise to 4"
    assert view.hint_text() == "3 files queued · raise to 4"
    assert "not started yet" in view.raise_button.toolTip()
    updates = Recorder(controller, "update_folder")
    view.raise_button.click()
    deliver(window)

    assert updates.calls == [((), {"ocr_parallel": 4})]
    assert run.job.parallel == 4                                # it reached the running job
    assert controller.run_snapshot().parallel == 4
    assert controller.project.folder.ocr_parallel == 4
    assert view.hint_text() == "2 workers idle → raise to 4"    # the literal B13 case, now that 4 > 2 running

    fake_runner.emit(run, "run_file_started", file="ep03.mkv")
    fake_runner.emit(run, "run_file_started", file="ep04.mkv")
    deliver(window)
    assert view.hint_text() == "1 file queued · raise to 6"     # singular, and two above the folder's 4


def test_no_queued_hint_at_the_parallel_cap(make_window, tmp_project, fake_runner):
    names = [f"ep{index:02d}.mkv" for index in range(1, 11)]
    window = make_window(names, tmp_project=tmp_project, ocr_parallel=8)
    run = start(window, fake_runner)
    for name in names[:8]:
        fake_runner.emit(run, "run_file_started", file=name)
    deliver(window)
    assert window.run_view.hint_label.isHidden() and window.run_view.hint_text() == ""


def test_parse_gpu_utilisation():
    assert run_view_module.GPU_PROGRAM == "nvidia-smi"
    assert run_view_module.GPU_ARGUMENTS == ["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
    assert run_view_module.GPU_POLL_MS == 2000
    assert parse_gpu_utilisation("74\n") == 74
    assert parse_gpu_utilisation(" 12\n80\n") == 80
    assert parse_gpu_utilisation("[N/A]\n") is None
    assert parse_gpu_utilisation("") is None
    assert parse_gpu_utilisation("150\n") == 100


def fake_nvidia_smi(tmp_path: Path, body: str) -> str:
    script = tmp_path / "nvidia-smi"
    script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def test_gpu_text_is_shown_from_nvidia_smi_while_the_run_view_shows_an_active_run(window, fake_runner, monkeypatch,
                                                                                  tmp_path):
    arguments_file = tmp_path / "args.txt"
    monkeypatch.setattr(run_view_module, "GPU_PROGRAM",
                        fake_nvidia_smi(tmp_path, f'echo "$@" > {arguments_file}\necho 74'))
    window.show()
    meter = window.run_view.gpu_meter
    assert not meter.is_polling()
    run = start(window, fake_runner)
    assert window.mode() == MODE_RUN and meter.is_polling()
    assert wait_for(lambda: not window.run_view.gpu_label.isHidden())
    assert window.run_view.gpu_label.text() == "GPU 74%"
    assert arguments_file.read_text(encoding="utf-8").split() == run_view_module.GPU_ARGUMENTS

    switch_to(window, MODE_REVIEW)
    assert not meter.is_polling()
    switch_to(window, MODE_RUN)
    assert meter.is_polling()
    fake_runner.finish(run, RunSummary([], {}, NAMES, 1.0), "cancelled")
    deliver(window)
    assert not meter.is_polling() and window.run_view.gpu_label.isHidden()
    assert wait_for(lambda: not meter.busy())


@pytest.mark.parametrize("body", [None, "exit 3", "echo '[N/A]'"], ids=["missing", "exit-code", "unparsable"])
def test_gpu_text_is_hidden_when_nvidia_smi_fails(window, fake_runner, monkeypatch, tmp_path, body):
    program = str(tmp_path / "no-such-nvidia-smi") if body is None else fake_nvidia_smi(tmp_path, body)
    monkeypatch.setattr(run_view_module, "GPU_PROGRAM", program)
    window.show()
    start(window, fake_runner)
    meter = window.run_view.gpu_meter
    assert wait_for(lambda: meter.queries() >= 1 and not meter.busy())
    settle()
    assert meter.value() is None
    assert window.run_view.gpu_label.isHidden()


def test_gpu_polls_run_every_interval_one_at_a_time(window, fake_runner, monkeypatch, tmp_path):
    monkeypatch.setattr(run_view_module, "GPU_PROGRAM", fake_nvidia_smi(tmp_path, "echo 5"))
    monkeypatch.setattr(run_view_module, "GPU_POLL_MS", 50)
    window.show()
    start(window, fake_runner)
    meter = window.run_view.gpu_meter
    assert wait_for(lambda: meter.queries() >= 3)
    assert window.run_view.gpu_label.text() == "GPU 5%"


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------

def test_logs_window_is_non_modal_ordered_live_and_capped(window, fake_runner):
    controller = window.controller
    controller.append_log("ep02.mkv", "Starting OCR: ep02.mkv")
    window.topbar.logs_button.click()
    settle()
    logs = window.logs_window
    assert logs is not None and logs.isVisible()
    assert logs.windowModality() == Qt.WindowModality.NonModal and not logs.isModal()
    assert logs.windowTitle() == "Logs"

    controller.append_log("ep01.mkv", "Starting OCR: ep01.mkv")
    controller.append_log("Detections", "ep03.mkv: crop failed: boom")
    controller.append_log("Pipeline", "OCR completed: 1/2 files successful")
    controller.append_log("ep02.mkv", "OCR completed successfully.")
    settle()
    assert logs.keys() == ["Pipeline", "Detections", "ep01.mkv", "ep02.mkv"]
    assert logs.section("ep02.mkv").text() == "Starting OCR: ep02.mkv\nOCR completed successfully.\n"
    assert logs.section("Pipeline").title() == "Pipeline"

    controller.append_log("Pipeline", "x" * (LOG_LIMIT - 10))
    controller.append_log("Pipeline", "the latest line")
    settle()
    text = logs.section("Pipeline").text()
    assert len(text) == LOG_LIMIT and text.endswith("the latest line\n")
    assert text == controller.log_text("Pipeline")

    window.topbar.logs_button.click()                           # a second click raises the same window
    settle()
    assert window.logs_window is logs

    start(window, fake_runner)                                  # a run start keeps only "Detections"
    assert logs.keys() == ["Detections"]
    assert logs.section("Detections").text() == "ep03.mkv: crop failed: boom\n"


def test_open_logs_from_the_queue_scrolls_to_that_files_section(window):
    controller = window.controller
    for name in NAMES:
        controller.append_log(name, "\n".join(f"{name} line {i}" for i in range(40)))
    controller.append_log("Pipeline", "\n".join(f"pipeline line {i}" for i in range(40)))
    window.show()
    logs_actions = {action.text(): action for action in window.queue.context_menu("ep05.mkv").actions()}
    logs_actions["Open logs"].trigger()
    settle()
    logs = window.logs_window
    for key in logs.keys():
        logs.section(key).set_expanded(True)
    window.open_logs("ep05.mkv")
    section = logs.section("ep05.mkv")
    assert section.is_expanded()
    bar = logs.scroll_area.verticalScrollBar()
    viewport = logs.scroll_area.viewport()

    def shown() -> bool:
        top = section.mapTo(logs.scroll_area.widget(), section.rect().topLeft()).y() - bar.value()
        return bar.value() > 0 and 0 <= top < viewport.height()

    assert wait_for(shown)


def test_open_logs_for_a_file_without_logs_shows_its_empty_section(window):
    window.open_logs("ep03.mkv")
    settle()
    logs = window.logs_window
    assert logs.keys() == ["ep03.mkv"] and logs.section("ep03.mkv").text() == ""
    assert logs.section("ep03.mkv").is_expanded()


def test_closing_the_window_closes_the_logs_window(controller, make_window, tmp_project):
    window = make_window(tmp_project=tmp_project)
    window.open_logs(None)
    settle()
    logs = window.logs_window
    assert logs.isVisible()
    window.close()
    settle()
    assert not logs.isVisible()


def test_run_view_imports_no_core_modules():
    import ast

    root = Path(__file__).resolve().parents[2]
    for path in (root / "app" / "views" / "run_view.py", root / "app" / "views" / "logs.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in ("core", "videocr"), f"{path.name}: {node.module}"
            elif isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in ("core", "videocr") for alias in node.names)
    assert os.path.exists(root / "app" / "views" / "run_view.py")
