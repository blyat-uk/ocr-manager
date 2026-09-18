"""Plan 3B Task 6: the feature-parity checklist of the design spec's §11.

Nothing in §11 may be lost in the rewrite, so every row of that table has a
row in `PARITY` below naming the tests that keep it, and a test here that
exercises it through the new window (or, where the behaviour lives below the
window, through the object the window drives).

`test_every_parity_row_names_tests_that_exist` re-reads those names from the
test files, so renaming or deleting a cited test breaks the checklist instead
of quietly emptying it.

Two rows are not automatable and are recorded as manual checks in
`MANUAL_CHECKS` (the Task 6 report spells out what a human should do):
look-and-feel of the window, and the desktop notification actually appearing
on the desktop.

Like the other tests/ui files, every test drives a real ProjectController
over `fake_runner` (tests/ui/conftest.py), and QSettings point at a per-test
directory.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

from app.controller import ProjectController
from app.logbook import PIPELINE_LOG
from app.main_window import MODE_RUN, MainWindow
from app.views import open_folder as open_folder_module
from app.views.run_view import MAX_PARALLEL
from core.jobs.run import OUTPUT_DIRS
from core.jobs.runner import JobContext
from core.project import (
    Brightness,
    Crop,
    FileEntry,
    FolderSettings,
    Media,
    Project,
    ReviewState,
    Source,
    TimeRange,
    TimeRanges,
    load_project,
    ocr_call_for,
    save_project,
)

NAMES = ["ep01.mkv", "ep02.mkv", "ep03.mkv", "ep04.mkv", "ep05.mkv"]
WAIT_MS = 5000
REPO_ROOT = Path(__file__).resolve().parents[2]
BOX = (288, 786, 1344, 53)
Yes, No = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No

# Source files the old window lived in. They are gone (plan 3B Task 6,
# ruling C10) and nothing may import them again.
REMOVED_FILES = [
    "widgets",
    "theme.py",
    "core/pipeline.py",
    "core/ocr_manager.py",
    "core/ocr_worker.py",
    "core/subtitle_detector.py",
    "core/audio_analysis.py",
    "core/config.py",
    "core/config_saver.py",
    "core/log_store.py",
]
REMOVED_MODULES = [name.removesuffix(".py").replace("/", ".") for name in REMOVED_FILES if name.endswith(".py")]
# Dependencies only the old window needed (rulings C9/C10). They may still sit
# in a developer's virtualenv, so what is pinned is that the project no longer
# asks for them.
REMOVED_REQUIREMENTS = ["qt-material", "qtawesome"]
# Import roots nothing may name again: the removed modules, the removed
# top-level `widgets` package, and the dependencies only the old window used.
# `app.widgets` / `app.theme` are the NEW window's own packages and are not
# these (see imports_a_removed_module: a root matches only as a whole path or
# as the parent of one).
REMOVED_IMPORT_ROOTS = [*REMOVED_MODULES, "widgets", "qt_material", "qtawesome"]


def imports_a_removed_module(module: str) -> bool:
    """True when the dotted `module` IS one of REMOVED_IMPORT_ROOTS or lives
    under one. Whole-path matching, so `import widgets` and
    `from widgets import x` are caught as well as `import widgets.file_table`,
    and `app.widgets.base` is not."""
    return any(module == root or module.startswith(root + ".") for root in REMOVED_IMPORT_ROOTS)

# §11 row -> the tests that keep it. "file::test" is looked up in that file;
# a bare name is a test in this file.
PARITY = {
    "Crash recovery / resume": [
        "test_done_files_are_badged_done_and_left_out_of_a_start",
        "tests/ui/test_controller.py::test_startable_files_and_overwrite",
        "tests/ui/test_controller.py::test_run_end_re_derives_done_states",
    ],
    "Overwrite confirmation before re-running finished files": [
        "test_replacing_done_files_is_asked_once_and_deletes_nothing",
        "tests/ui/test_run_view.py::test_start_asks_before_replacing_done_files",
        "tests/test_run_job.py::test_stopping_mid_file_keeps_the_old_final_and_removes_the_partial",
        "tests/test_run_job.py::test_output_lands_in_chi_only_after_qa_replacing_the_old_file",
    ],
    "Folder watcher": [
        "test_the_folder_watcher_adds_new_episodes_and_drops_vanished_ones",
        "tests/ui/test_controller.py::test_watcher_adds_new_videos_and_drops_vanished_ones",
        "tests/ui/test_controller.py::test_watcher_changes_wait_for_the_run_to_end",
    ],
    "Desktop notification on completion": [
        "test_a_finished_run_sends_a_desktop_notification",
        "tests/ui/test_controller.py::test_a_run_with_a_failed_file_notifies_critical",
        "tests/ui/test_controller.py::test_notification_text_matches_todays_window",
    ],
    "Per-file logs viewer, pipeline log": [
        "test_per_file_logs_and_the_pipeline_log_are_in_the_logs_window",
        "tests/ui/test_run_view.py::test_logs_window_is_non_modal_ordered_live_and_capped",
        "tests/ui/test_run_view.py::test_open_logs_from_the_queue_scrolls_to_that_files_section",
    ],
    "Live subtitle feed while running": [
        "test_the_live_feed_shows_subtitles_as_the_run_finds_them",
        "tests/ui/test_run_view.py::test_the_live_feed_follows_the_newest_file_and_the_menu_switches",
    ],
    "File details (resolved settings) -> the inspector": [
        "test_the_inspector_shows_the_resolved_settings_the_details_dialog_showed",
        "tests/ui/test_main_window.py::test_selecting_a_file_updates_the_inspector_header_and_detected_rows",
    ],
    "Copy / paste settings between files": [
        "test_copy_and_paste_settings_from_a_rows_context_menu",
        "tests/ui/test_controller.py::test_copy_and_paste_settings",
        "tests/ui/test_main_window.py::test_context_menu_offers_b11_actions_and_invokes_commands",
    ],
    "Label mask regions": [
        "test_label_mask_regions_round_trip_through_the_folder",
        "tests/ui/test_controller.py::test_set_label_masks",
        "tests/ui/test_folder_settings.py::test_mask_count_follows_the_folder",
    ],
    "Labels-only mode": [
        "test_labels_only_needs_no_crop_and_reaches_the_ocr_call",
        "tests/ui/test_folder_settings.py::test_turning_off_the_last_extraction_is_refused_with_a_warning",
        "tests/test_ocr_kwargs.py::test_labels_only_sets_only_labels_true",
    ],
    "Parallel worker count": [
        "test_parallel_files_is_a_folder_setting_the_run_view_can_raise",
        "tests/ui/test_folder_settings.py::test_each_editor_round_trips_its_folder_setting",
        "tests/ui/test_run_view.py::test_queued_files_offer_a_higher_parallel_that_reaches_the_run_job",
        "tests/ui/test_controller.py::test_parallel_files_changed_during_a_run_reach_the_run_job",
    ],
    "Window geometry persistence": [
        "test_window_geometry_is_remembered",
        "tests/ui/test_main_window.py::test_geometry_and_last_path_are_persisted",
    ],
    "Native folder picker (kdialog when present)": [
        "test_the_folder_picker_is_kdialog_started_at_the_remembered_path",
        "tests/ui/test_main_window.py::test_choose_folder_runs_kdialog_from_the_last_path",
        "tests/ui/test_main_window.py::test_choose_folder_uses_the_qt_picker_without_kdialog",
        "tests/ui/test_main_window.py::test_kdialog_that_fails_to_start_falls_back_to_the_qt_picker",
    ],
    "Multi-range OCR per file": [
        "test_a_files_ranges_reach_the_run_as_one_get_subtitles_call",
        "tests/test_run_job.py::test_several_ranges_call_get_subtitles_and_write_the_text_as_utf8",
        "tests/test_multirange.py::test_time_ranges_matches_old_per_range_calls_and_golden",
    ],
    "Creating chi/, eng/, translate/ output directories": [
        "test_a_run_creates_the_three_output_directories",
        "tests/test_run_job.py::test_output_dirs_are_created_before_any_file",
    ],
    "HDR->SDR tone mapping (PQ and HLG)": [
        "test_detector_frames_and_ocr_frames_share_one_decode_and_tone_map_path",
        "tests/test_capture_hdr.py::test_hdr_pq_source_is_tone_mapped",
        "tests/test_detect_ocr_view.py::test_grab_ocr_strips_returns_the_pixels_run_ocr_sees",
    ],
    # The timeline that replaces the old chips is plan 3C; what it edits (a
    # file's time ranges, manual, whole-file included) is here already and is
    # what the OCR call reads.
    "Time range chips / active-range editing": [
        "test_time_ranges_are_editable_and_reach_the_ocr_call",
        "tests/ui/test_main_window.py::test_selecting_a_file_updates_the_inspector_header_and_detected_rows",
    ],
}

MANUAL_CHECKS = {
    # name -> what a human does, recorded in the Task 6 report
    "look and feel": "run `.venv/bin/python main.py`, open a folder of episodes and look at the window",
    "desktop notification": "finish a real run and check the notification appears on the desktop",
}


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


def ready_entry(name: str, *, crop=BOX, ranges=()) -> FileEntry:
    return FileEntry(name, media=Media(1920, 888, 1628.0, 25.0), review=ReviewState.REVIEWED,
                     crop=None if crop is None else Crop(*crop, Source.MANUAL),
                     brightness=Brightness(209, Source.MANUAL),
                     time_ranges=TimeRanges([TimeRange(s, e) for s, e in ranges], Source.MANUAL),
                     sample_time=578.0)


@pytest.fixture(autouse=True)
def settings_dir(tmp_path):
    path = tmp_path / "qsettings"
    path.mkdir()
    for fmt in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, str(path))
    return path


@pytest.fixture
def notifications(monkeypatch):
    """notify-send never reaches the desktop from a test."""
    sent = []

    def fake_run(args, **kwargs):
        sent.append(list(args))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return sent


@pytest.fixture
def controller(qapp, fake_runner):
    made = ProjectController(fake_runner, save_debounce_ms=10, watch_debounce_ms=10)
    yield made
    made.shutdown(timeout=0.5)


@pytest.fixture
def make_window(controller, notifications, tmp_project):
    windows = []

    def make(names=NAMES, *, done=(), entries=None, open_it=True, **folder) -> MainWindow:
        folder_path = tmp_project(list(names))
        folder.setdefault("labels_enabled", False)
        files = entries if entries is not None else {name: ready_entry(name) for name in names}
        save_project(Project(path=str(folder_path), folder=FolderSettings(**folder), files=files))
        for name in done:
            (folder_path / "chi").mkdir(exist_ok=True)
            (folder_path / "chi" / f"{Path(name).stem}.ass").write_text("Dialogue: old\n", encoding="utf-8")
        window = MainWindow(controller)
        window.resize(1440, 900)
        windows.append(window)
        if open_it:
            window.open_folder(str(folder_path))
            settle()
        return window

    yield make
    for window in windows:
        # The teardown is not a user: it never answers closeEvent's questions
        # (a run in progress, settings that could not be saved).
        window._closing = True
        window.close()
        window.deleteLater()
    settle()


@pytest.fixture
def window(make_window):
    return make_window(ocr_parallel=4)


def answer(monkeypatch, reply) -> list[tuple]:
    asked = []

    def question(parent, title, text, buttons, default):
        asked.append((title, text, buttons, default))
        return reply

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return asked


def deliver(window: MainWindow) -> None:
    window.controller.drain_events()
    settle()


def start_run(window: MainWindow, fake_runner):
    window.topbar.start_button.click()
    settle()
    return fake_runner.last("run")


class FakeOcr:
    """videocr.api stand-in for a real RunJob.run(): writes an ASS file."""

    HEADER = ("[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
              "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

    def __init__(self, monkeypatch):
        from videocr import api

        self.calls: list[tuple[str, dict]] = []
        monkeypatch.setattr(api, "save_subtitles_to_file", self.save_subtitles_to_file)
        monkeypatch.setattr(api, "get_subtitles", self.get_subtitles)

    def _text(self, kwargs: dict) -> str:
        self.calls.append((kwargs.pop("_function"), dict(kwargs)))
        return self.HEADER + "Dialogue: 0,0:00:01.00,0:00:01.50,Default,,0,0,0,,字幕\n"

    def save_subtitles_to_file(self, **kwargs) -> None:
        path = kwargs.pop("file_path")
        text = self._text({**kwargs, "_function": "save_subtitles_to_file"})
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def get_subtitles(self, **kwargs) -> str:
        return self._text({**kwargs, "_function": "get_subtitles"})


# --------------------------------------------------------------------------
# The checklist itself
# --------------------------------------------------------------------------

def _test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}


def test_every_parity_row_names_tests_that_exist():
    """A cited test that was renamed or deleted must break the checklist, not
    silently leave its §11 row without evidence."""
    here = _test_names(Path(__file__))
    missing = []
    for row, citations in PARITY.items():
        assert citations, row
        for citation in citations:
            if "::" in citation:
                file_name, test = citation.split("::")
                path = REPO_ROOT / file_name
                if not path.exists() or test not in _test_names(path):
                    missing.append(f"{row}: {citation}")
            elif citation not in here:
                missing.append(f"{row}: {citation}")
    assert missing == []


def test_the_old_window_and_its_dead_support_code_are_gone():
    """Ruling C10: `main.py` launches the new window, and the old one, the
    dead widgets and the dead config code are deleted."""
    import importlib.util

    assert [name for name in REMOVED_FILES if (REPO_ROOT / name).exists()] == []
    assert [name for name in REMOVED_MODULES if importlib.util.find_spec(name) is not None] == []

    source = (REPO_ROOT / "main.py").read_text(encoding="utf-8")
    assert "from app.__main__ import main" in source
    assert "multiprocessing.freeze_support()" in source
    assert "QMainWindow" not in source


def test_the_old_windows_own_dependencies_are_no_longer_required():
    """Ruling C9: the new window does not use qt-material, and nothing needs
    qtawesome any more."""
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for name in REMOVED_REQUIREMENTS:
        assert name not in requirements, name
        assert name not in pyproject, name


def test_main_py_launches_the_new_window(tmp_path):
    """`main.py` is the new window's launcher, folder argument and
    `--quit-after` hook included (in its own process, so its theme, settings
    and hook never touch this session)."""
    folder = tmp_path / "episodes"
    folder.mkdir()
    (folder / "ep01.mkv").write_bytes(b"placeholder video")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", XDG_CONFIG_HOME=str(tmp_path / "config"))
    result = subprocess.run([sys.executable, str(REPO_ROOT / "main.py"), str(folder), "--quit-after", "1"],
                            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120,
                            check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    assert (folder / ".ocr.json").exists()              # the folder really opened
    assert (tmp_path / "config" / "OCRManager" / "OCRTool.conf").exists()


def test_main_py_and_python_m_app_both_describe_themselves(tmp_path):
    """--help names whichever way the app was started."""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", XDG_CONFIG_HOME=str(tmp_path / "config"))

    def usage(args):
        result = subprocess.run([sys.executable, *args, "--help"], cwd=str(REPO_ROOT), env=env,
                                capture_output=True, text=True, timeout=120, check=False)
        assert result.returncode == 0, result.stderr[-2000:]
        return result.stdout.splitlines()[0]

    assert usage([str(REPO_ROOT / "main.py")]).startswith("usage: main.py")
    assert usage(["-m", "app"]).startswith("usage: python -m app")


def test_the_removed_module_matcher_matches_whole_paths():
    """A bare `import widgets` must be caught, and the new window's own
    `app.widgets` must not (an earlier version matched the prefix "widgets."
    and missed the bare form)."""
    for caught in ("widgets", "widgets.file_table", "theme", "core.config", "core.config.Config",
                   "core.ocr_worker", "qt_material", "qtawesome"):
        assert imports_a_removed_module(caught), caught
    for allowed in ("app", "app.widgets", "app.widgets.base", "app.theme", "app.theme.tokens",
                    "core", "core.project", "core.jobs.run", "widgetsmith", "themes"):
        assert not imports_a_removed_module(allowed), allowed


def test_no_module_imports_the_removed_ones():
    """The cut-over's grep, as a test: no source file outside .venv may name
    a removed module again."""
    offenders = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT)
        # Skip everything that is not this project's source: virtualenvs,
        # worktrees, build output and any other dot-directory.
        if any(part.startswith(".") for part in relative.parts) or set(relative.parts) & {"build", "dist"}:
            continue
        if relative.as_posix() == "tests/test_ocr_kwargs.py":
            continue            # its skip guard imports them on purpose, inside try/except ImportError
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            offenders += [f"{relative}: {module}" for module in modules if imports_a_removed_module(module)]
    assert offenders == []


def test_nothing_in_the_new_app_force_terminates_a_thread_or_opens_a_progress_dialog():
    """Plan exit criteria: cooperative cancellation only (ruling C6), and no
    QProgressDialog anywhere in the new window."""
    offenders = []
    for root, needles in ((REPO_ROOT / "app", ("terminate()", "QProgressDialog")),
                          (REPO_ROOT / "core", ("terminate()",))):
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            offenders += [f"{path.relative_to(REPO_ROOT)}: {needle}" for needle in needles if needle in text]
    assert offenders == []


# --------------------------------------------------------------------------
# Crash recovery / resume, and the overwrite question
# --------------------------------------------------------------------------

def test_done_files_are_badged_done_and_left_out_of_a_start(make_window, monkeypatch, fake_runner):
    """A non-empty `chi/<stem>.ass` is the old resume rule: the file shows
    `done` and a start leaves it out unless the user says to replace it."""
    window = make_window(done=("ep02.mkv", "ep04.mkv"))
    controller = window.controller
    folder = Path(controller.project.path)
    (folder / "chi" / "ep03.ass").write_text("", encoding="utf-8")       # empty: not done
    controller._refresh_done()
    settle()

    assert [name for name in NAMES if controller.is_done(name)] == ["ep02.mkv", "ep04.mkv"]
    assert window.queue.row("ep02.mkv").badge.text() == "done"
    assert window.queue.row("ep03.mkv").badge.text() != "done"
    assert controller.startable_files() == ["ep01.mkv", "ep03.mkv", "ep05.mkv"]
    assert controller.startable_files(include_done=True) == NAMES
    assert controller.files_needing_overwrite(NAMES) == ["ep02.mkv", "ep04.mkv"]

    answer(monkeypatch, No)
    submission = start_run(window, fake_runner)
    assert [run_file.name for run_file in submission.job.files] == ["ep01.mkv", "ep03.mkv", "ep05.mkv"]


def test_replacing_done_files_is_asked_once_and_deletes_nothing(make_window, monkeypatch, fake_runner):
    """The old window deleted the existing ASS files before the run started.
    Now one question decides whether they are re-run, and the existing files
    stay untouched until each file's new output is ready (ruling C5)."""
    window = make_window(done=("ep02.mkv",))
    folder = Path(window.controller.project.path)
    existing = folder / "chi" / "ep02.ass"
    before = existing.read_bytes()

    asked = answer(monkeypatch, Yes)
    submission = start_run(window, fake_runner)

    assert len(asked) == 1
    title, text, _buttons, default = asked[0]
    assert title == "Replace existing subtitles?"
    assert "1 file(s) already have subtitles" in text
    assert default == No                                  # the safe answer is the default
    assert [run_file.name for run_file in submission.job.files] == NAMES
    assert existing.exists() and existing.read_bytes() == before
    assert window.mode() == MODE_RUN


# --------------------------------------------------------------------------
# Folder watcher, notification, logs and the live feed
# --------------------------------------------------------------------------

def test_the_folder_watcher_adds_new_episodes_and_drops_vanished_ones(window):
    folder = Path(window.controller.project.path)
    (folder / "ep06.mkv").write_bytes(b"placeholder video")
    (folder / "ep01.mkv").unlink()

    assert wait_for(lambda: "ep06.mkv" in window.queue.names() and "ep01.mkv" not in window.queue.names())
    assert window.controller.names() == ["ep02.mkv", "ep03.mkv", "ep04.mkv", "ep05.mkv", "ep06.mkv"]


def test_a_finished_run_sends_a_desktop_notification(window, fake_runner, notifications):
    from core.jobs.run import RunSummary

    submission = start_run(window, fake_runner)
    fake_runner.finish(submission, result=RunSummary(NAMES, {}, [], 12.0))
    deliver(window)

    assert len(notifications) == 1
    command, body = notifications[0][:-1], notifications[0][-1]
    assert command == ["notify-send", "-a", "OCR Manager", "-u", "normal", "OCR Complete"]
    assert body.startswith("Finished in ") and "/file" in body


def test_per_file_logs_and_the_pipeline_log_are_in_the_logs_window(window, fake_runner):
    submission = start_run(window, fake_runner)
    fake_runner.emit(submission, "run_file_started", file="ep01.mkv")
    fake_runner.emit(submission, "run_file_log", file="ep01.mkv", message="Starting OCR: ep01.mkv\n")
    deliver(window)
    window.controller.append_log(PIPELINE_LOG, "a pipeline line\n")
    settle()

    window.open_logs("ep01.mkv")
    logs = window.logs_window
    assert logs is not None and logs.isVisible()
    assert not logs.isModal()
    keys = logs.keys()
    assert PIPELINE_LOG in keys and "ep01.mkv" in keys
    assert "Starting OCR: ep01.mkv" in logs.section("ep01.mkv").text()
    assert "a pipeline line" in logs.section(PIPELINE_LOG).text()


def test_the_live_feed_shows_subtitles_as_the_run_finds_them(window, fake_runner):
    submission = start_run(window, fake_runner)
    fake_runner.emit(submission, "run_file_started", file="ep01.mkv")
    deliver(window)
    fake_runner.emit(submission, "run_subtitle", file="ep01.mkv", result=(61.0, 62.5, "字幕一"))
    fake_runner.emit(submission, "run_subtitle", file="ep01.mkv", result=(70.0, 71.0, "字幕二"))
    deliver(window)

    assert window.mode() == MODE_RUN
    assert [line.text() for line in window.run_view._feed_lines()] == ["01:01 字幕一", "01:10 字幕二"]
    assert window.controller.run_subtitles("ep01.mkv") == [(61.0, 62.5, "字幕一"), (70.0, 71.0, "字幕二")]


# --------------------------------------------------------------------------
# Inspector, copy/paste, masks, labels-only, parallel
# --------------------------------------------------------------------------

def test_the_inspector_shows_the_resolved_settings_the_details_dialog_showed(make_window):
    """The old File details dialog listed the values a run would use. The
    inspector shows the same resolved values for the selected file."""
    entries = {name: ready_entry(name) for name in NAMES}
    entries["ep02.mkv"] = ready_entry("ep02.mkv", ranges=(("9:30", "11:30"),))
    window = make_window(entries=entries)
    window.queue.select("ep02.mkv")
    settle()
    inspector = window.inspector

    assert inspector.file_label.full_text() == "ep02.mkv"
    assert inspector.media_label.text() == "1920×888 · 27:08 · 25 fps"
    assert inspector.crop_row.value() == "288, 786 · 1344 × 53"
    assert inspector.brightness_row.value() == "209"
    assert inspector.window_row.value() == "9:30 → 11:30"

    window.queue.select("ep01.mkv")
    settle()
    assert inspector.window_row.value() == "whole file"


def test_copy_and_paste_settings_from_a_rows_context_menu(make_window):
    entries = {name: ready_entry(name) for name in NAMES}
    entries["ep01.mkv"] = ready_entry("ep01.mkv", crop=(10, 20, 30, 40), ranges=(("1:00", "2:00"),))
    window = make_window(entries=entries)
    controller = window.controller

    def trigger(name: str, text: str) -> None:
        menu = window.queue.context_menu(name)
        actions = {action.text(): action for action in menu.actions() if not action.isSeparator()}
        assert text in actions and actions[text].isEnabled(), sorted(actions)
        actions[text].trigger()
        menu.deleteLater()

    assert not controller.can_paste()
    trigger("ep01.mkv", "Copy settings")
    assert controller.can_paste()
    trigger("ep03.mkv", "Paste settings onto this file")
    settle()

    pasted = controller.entry("ep03.mkv")
    assert (pasted.crop.x, pasted.crop.y, pasted.crop.width, pasted.crop.height) == (10, 20, 30, 40)
    assert pasted.crop.source == Source.MANUAL
    assert [(r.start, r.end) for r in pasted.time_ranges.ranges] == [("1:00", "2:00")]


def test_label_mask_regions_round_trip_through_the_folder(make_window):
    window = make_window(labels_enabled=True)
    controller = window.controller
    masks = [(10, 20, 30, 40), (50, 60, 70, 80)]

    controller.set_label_masks(masks)
    settle()
    assert controller.project.folder.label_mask_crops == masks

    folder = controller.project.path
    controller.close_folder()
    assert load_project(folder).folder.label_mask_crops == masks

    entry = ready_entry("ep01.mkv")
    call = ocr_call_for(entry, FolderSettings(labels_enabled=True, label_mask_crops=masks), folder)
    assert call.kwargs["label_mask_crops"] == masks


def test_labels_only_needs_no_crop_and_reaches_the_ocr_call(make_window, fake_runner):
    """Dialogue off + labels on is the old labels-only mode: a file needs no
    crop to be ready, Start is enabled, and the OCR call carries
    `only_labels`."""
    entries = {name: ready_entry(name, crop=None) for name in NAMES}
    window = make_window(entries=entries, dialogue_enabled=False, labels_enabled=True)
    controller = window.controller
    assert controller.project.folder.labels_only

    assert controller.startable_files() == NAMES
    assert window.topbar.start_button.isEnabled()

    submission = start_run(window, fake_runner)
    kwargs = submission.job.files[0].call.kwargs
    assert kwargs["detect_labels"] is True and kwargs["only_labels"] is True
    assert "crop_x" not in kwargs


def test_parallel_files_is_a_folder_setting_the_run_view_can_raise(make_window, fake_runner):
    window = make_window(ocr_parallel=2)
    controller = window.controller
    window.show()
    sheet = window.folder_settings
    window.topbar.settings_button.click()
    assert wait_for(lambda: sheet.isVisible() and sheet.geometry() == sheet.target_geometry())

    editor = sheet.editor("ocr_parallel")
    editor.setValue(3)
    editor.editingFinished.emit()
    settle()
    assert controller.project.folder.ocr_parallel == 3

    submission = start_run(window, fake_runner)
    assert submission.job.parallel == 3
    raised = []
    submission.job.set_parallel = raised.append
    for index, name in enumerate(NAMES):
        if index < 3:
            fake_runner.emit(submission, "run_file_started", file=name)
    deliver(window)

    assert window.run_view.raise_button.isVisible()
    window.run_view.raise_button.click()
    settle()
    assert raised == [5] and controller.project.folder.ocr_parallel == 5
    assert 5 <= MAX_PARALLEL


# --------------------------------------------------------------------------
# Window geometry and the folder picker
# --------------------------------------------------------------------------

def test_window_geometry_is_remembered(make_window, controller):
    window = make_window()
    window.resize(1100, 720)
    window.close()
    settle()
    saved = QSettings("OCRManager", "OCRTool").value("window/geometry")
    assert saved is not None and not saved.isEmpty()

    reopened = MainWindow(controller)
    try:
        # The offscreen screen is 800 px wide, so only the height comes back
        # unclamped (as tests/ui/test_main_window.py's geometry test notes).
        assert reopened.height() == 720
    finally:
        reopened.close()
        reopened.deleteLater()


def test_the_folder_picker_is_kdialog_started_at_the_remembered_path(make_window, monkeypatch, tmp_path):
    """Today's picker: kdialog when it is installed, started at the last
    folder opened rather than a hardcoded directory."""
    window = make_window()
    assert open_folder_module.last_path() == window.controller.project.path

    started = []
    monkeypatch.setattr(open_folder_module.shutil, "which", lambda name: "/usr/bin/kdialog")
    monkeypatch.setattr(open_folder_module.QProcess, "start",
                        lambda self, program, args: started.append((program, args)))
    window.choose_folder()

    assert started == [("/usr/bin/kdialog", ["--getexistingdirectory", window.controller.project.path])]


# --------------------------------------------------------------------------
# The run itself: ranges, output directories, tone mapping
# --------------------------------------------------------------------------

def test_a_files_ranges_reach_the_run_as_one_get_subtitles_call(make_window, fake_runner, monkeypatch, tmp_path):
    """Several ranges per file survive as one `get_subtitles(time_ranges=)`
    call -- one engine session for the whole file, not one per range."""
    entries = {name: ready_entry(name) for name in NAMES}
    entries["ep01.mkv"] = ready_entry("ep01.mkv", ranges=(("9:30", "10:00"), ("14:00", "14:30")))
    window = make_window(entries=entries)
    window.controller.start_run(["ep01.mkv"])
    settle()
    job = fake_runner.last("run").job
    assert job.files[0].call.time_ranges == [("9:30", "10:00"), ("14:00", "14:30")]

    ocr = FakeOcr(monkeypatch)
    summary = job.run(JobContext("run", "run"))

    assert summary.succeeded == ["ep01.mkv"]
    assert [function for function, _kwargs in ocr.calls] == ["get_subtitles"]
    assert ocr.calls[0][1]["time_ranges"] == [("9:30", "10:00"), ("14:00", "14:30")]


def test_a_run_creates_the_three_output_directories(make_window, fake_runner, monkeypatch):
    window = make_window()
    folder = Path(window.controller.project.path)
    assert not (folder / "eng").exists()

    window.controller.start_run(["ep01.mkv"])
    settle()
    FakeOcr(monkeypatch)
    fake_runner.last("run").job.run(JobContext("run", "run"))

    assert OUTPUT_DIRS == ("chi", "eng", "translate")
    assert all((folder / name).is_dir() for name in OUTPUT_DIRS)
    assert (folder / "chi" / "ep01.ass").exists()


def test_detector_frames_and_ocr_frames_share_one_decode_and_tone_map_path():
    """HDR->SDR tone mapping is applied consistently: everything that looks at
    what OCR reads goes through the same `videocr.pyav_adapter.Capture`, whose
    tone-map chain is pinned by tests/test_capture_hdr.py."""
    from core.detect import ocr_view
    from videocr import pyav_adapter, video

    assert ocr_view.Capture is pyav_adapter.Capture
    assert video.Capture is pyav_adapter.Capture
    assert ocr_view.DECODE_TARGET_HEIGHT is pyav_adapter.DECODE_TARGET_HEIGHT


def test_time_ranges_are_editable_and_reach_the_ocr_call(make_window, fake_runner):
    """The timeline that replaces the old chips is plan 3C; what it edits is
    already here and is what a run reads."""
    window = make_window()
    controller = window.controller

    controller.set_time_ranges("ep01.mkv", [("9:30", "11:30"), ("14:00", None)])
    settle()
    entry = controller.entry("ep01.mkv")
    assert [(r.start, r.end) for r in entry.time_ranges.ranges] == [("9:30", "11:30"), ("14:00", None)]
    assert entry.time_ranges.source == Source.MANUAL

    controller.start_run(["ep01.mkv"])
    settle()
    assert fake_runner.last("run").job.files[0].call.time_ranges == [("9:30", "11:30"), ("14:00", "")]

    controller.stop_run()
    settle()


def test_manual_checks_are_named():
    """The two rows a headless test cannot cover are named here so the report
    can record who checked them and how."""
    assert set(MANUAL_CHECKS) == {"look and feel", "desktop notification"}
    assert all(text and text[0].islower() for text in MANUAL_CHECKS.values())
