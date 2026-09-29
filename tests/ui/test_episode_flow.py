"""The episode view, wired (docs/superpowers/plans/2026-09-28-episode-view.md,
Stream B): the controller's episode opener and contract, and the window's
Preparing -> Review -> Working -> Done flow.

Like the other tests/ui files, everything runs over a real ProjectController
and `fake_runner`: nothing is decoded, the test delivers the job events. The
episode cache root is the per-test $OCR_MANAGER_CACHE_DIR of tests/conftest.py,
and QSettings point at a per-test directory.

Placeholder videos get different bytes per name: an episode's key is its
content (core.project.layout.video_key), and two episodes with the same bytes
would share one settings file.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from PyQt6.QtCore import QMimeData, QPoint, QPointF, QSettings, Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

from app.controller import ProjectController
from app.main_window import MODE_REVIEW, MODE_RUN, MainWindow
from app.views import open_folder as open_folder_module
from core.detect.brightness import FLAG_NARROW_PLATEAU, BrightnessResult, StripSample
from core.detect.crop import CONSENSUS_MIN_ENTRIES, FLAG_LOW_AGREEMENT, CropResult
from core.jobs.autopilot import AutoPilot
from core.jobs.detect_jobs import BrightnessJobResult, CropJobResult, MetadataResult
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
    TimeRange,
    TimeRanges,
    save_project,
)
from core.project.episode_cache import index_path, last_speed, record_speed
from core.project.layout import episode_layout

BOX = (288, 786, 1344, 53)
WAIT_MS = 5000
Yes, No = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No
ASS_HEADER = ("[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
              "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")


# --------------------------------------------------------------------------
# Helpers and fixtures
# --------------------------------------------------------------------------

def settle() -> None:
    QApplication.processEvents()


def wait_for(predicate, timeout_ms: int = WAIT_MS) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        QTest.qWait(10)
    return True


def cache_root() -> Path:
    return Path(os.environ["OCR_MANAGER_CACHE_DIR"])


def listing(folder: Path) -> list[str]:
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*"))


@pytest.fixture
def videos(tmp_path):
    """videos(names, folder="show") -> Path: a folder of placeholder videos,
    each with bytes of its own (see the module docstring)."""
    def make(names, folder: str = "show") -> Path:
        directory = tmp_path / folder
        directory.mkdir(exist_ok=True)
        for name in names:
            (directory / name).write_bytes(f"placeholder video {name}".encode())
        return directory
    return make


def ready_entry(name: str, *, crop=BOX, review=ReviewState.REVIEWED, ranges=(), flags=None) -> FileEntry:
    return FileEntry(name, media=Media(1920, 1080, 1400.0, 23.976), review=review,
                     crop=None if crop is None else Crop(*crop, Source.MANUAL),
                     brightness=Brightness(209, Source.MANUAL),
                     time_ranges=TimeRanges([TimeRange(s, e) for s, e in ranges], Source.MANUAL),
                     flags=dict(flags or {}), sample_time=300.0)


def save_episode(video: Path, entry: FileEntry, **folder) -> None:
    """Save `entry` as the episode `video`'s settings, in the cache root."""
    layout = episode_layout(str(video), str(cache_root()))
    folder.setdefault("labels_enabled", False)
    save_project(Project(path=str(video.parent), folder=FolderSettings(**folder), files={entry.name: entry},
                         layout=layout))


@pytest.fixture(autouse=True)
def settings_dir(tmp_path):
    path = tmp_path / "qsettings"
    path.mkdir()
    for fmt in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, str(path))
    return path


@pytest.fixture
def notifications(monkeypatch):
    sent = []

    class Notifier:
        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda args, **kwargs: sent.append(list(args)) or Notifier())
    return sent


@pytest.fixture
def controller(qapp, fake_runner, notifications):
    made = ProjectController(fake_runner, save_debounce_ms=10, watch_debounce_ms=10)
    yield made
    made.shutdown(timeout=0.5)


@pytest.fixture
def make_window(controller):
    windows = []

    def make() -> MainWindow:
        window = MainWindow(controller)
        window.resize(1440, 900)
        windows.append(window)
        return window

    yield make
    for window in windows:
        window._closing = True
        window.close()
        window.deleteLater()
    settle()


def answer(monkeypatch, reply) -> list[tuple]:
    asked = []

    def question(parent, title, text, buttons, default):
        asked.append((title, text))
        return reply

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return asked


def deliver(controller) -> None:
    controller.drain_events()
    settle()


def run_detections(fake_runner, controller, name: str, *, crop_flag=None, brightness_flag=None,
                   hold=()) -> None:
    """Finish every queued detection of `name` the way the detectors would
    on a clean episode, until nothing is left to run -- leaving the kinds in
    `hold` queued, for the test to finish."""
    for _ in range(40):
        pending = [s for s in fake_runner.queued() if s.job.file == name
                   and s.job.kind not in ("frames", "strips", *hold)]
        if not pending:
            return
        submission = pending[0]
        job = submission.job
        if job.kind == "metadata":
            result = MetadataResult(name, 1920, 1080, 1400.0, 23.976)
        elif job.kind == "crop":
            result = CropJobResult(name, CropResult(box=BOX, sample_pts=[300.0], envelope=BOX, agreed=3,
                                                    probes_used=3, flagged=crop_flag, hit_pts=[300.0],
                                                    frame_size=(1920, 1080)), job.hint)
        elif job.kind == "brightness":
            strips = [StripSample(512.0, True, 200, 20.0, 3.0, 1, ((10, 10, 80, 30),), None)]
            result = BrightnessJobResult(name, BrightnessResult(value=205, plateau=(180, 230), seed=225,
                                                                gate_floor=None, flagged=brightness_flag, curve=[],
                                                                strips=strips),
                                         {}, job.hint_value, job.crop_box)
        else:
            result = None
        fake_runner.finish(submission, result)
        deliver(controller)
    raise AssertionError("detections never settled")


def write_output(path: str, lines) -> None:
    body = "".join(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n" for start, end, text in lines)
    Path(path).parent.mkdir(exist_ok=True)                   # the output subfolder a run would have made
    Path(path).write_text(ASS_HEADER + body, encoding="utf-8")


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def test_open_path_opens_a_video_file_as_an_episode(controller, videos):
    folder = videos(["EP06.mkv", "EP07.mkv"])
    controller.open_path(str(folder / "EP06.mkv"))
    assert controller.is_episode
    assert controller.episode_name() == "EP06.mkv"
    assert controller.names() == ["EP06.mkv"]            # the sibling is not part of it
    assert controller.project.path == str(folder)
    assert controller.output_path("EP06.mkv") == str(folder / "zh" / "EP06.zh.ass")


def test_open_path_routes_folders_by_their_video_count(controller, videos):
    one = videos(["EP06.mkv"], folder="one")
    controller.open_path(str(one))
    assert controller.is_episode and controller.episode_name() == "EP06.mkv"

    two = videos(["EP06.mkv", "EP07.mkv"], folder="two")
    controller.open_path(str(two))
    assert not controller.is_episode
    assert controller.episode_name() is None
    assert controller.names() == ["EP06.mkv", "EP07.mkv"]
    assert controller.output_path("EP06.mkv") == str(two / "zh" / "EP06.zh.ass")


def test_open_folder_stays_a_folder_even_with_one_video(controller, videos):
    """open_folder is the workbench's own opener: the routing is open_path's."""
    folder = videos(["EP06.mkv"])
    controller.open_folder(str(folder))
    assert not controller.is_episode
    assert controller.episode_name() is None


def test_project_opened_carries_the_video_path_for_an_episode(controller, videos):
    folder = videos(["EP06.mkv"])
    opened = []
    controller.project_opened.connect(opened.append)
    controller.open_path(str(folder))
    assert opened == [str(folder / "EP06.mkv")]


# --------------------------------------------------------------------------
# Where an episode's state lives
# --------------------------------------------------------------------------

def test_an_episode_writes_nothing_next_to_the_video_but_its_output(controller, fake_runner, videos):
    folder = videos(["EP06.mkv"])
    before = listing(folder)
    controller.open_path(str(folder / "EP06.mkv"))
    run_detections(fake_runner, controller, "EP06.mkv")
    controller.set_brightness("EP06.mkv", 211)
    controller.close_folder()
    assert listing(folder) == before

    layout = episode_layout(str(folder / "EP06.mkv"), str(cache_root()))
    settings = json.loads(Path(layout.config_path).read_text(encoding="utf-8"))
    assert settings["files"]["EP06.mkv"]["brightness"]["value"] == 211

    controller.open_path(str(folder / "EP06.mkv"))           # reloaded from the cache dir
    assert controller.entry("EP06.mkv").brightness.value == 211
    assert controller.entry("EP06.mkv").crop is not None


def test_view_jobs_and_the_run_use_the_episode_layout(controller, fake_runner, videos):
    folder = videos(["EP06.mkv"])
    save_episode(folder / "EP06.mkv", ready_entry("EP06.mkv"))
    controller.open_path(str(folder / "EP06.mkv"))
    layout = controller.project.layout
    controller.request_frames("EP06.mkv", [12.0])
    assert fake_runner.last("frames", "EP06.mkv").job.cache_dir == layout.cache_dir
    controller.request_strips("EP06.mkv", BOX, [12.0])
    assert fake_runner.last("strips", "EP06.mkv").job.cache_dir == layout.cache_dir
    controller.start_run(["EP06.mkv"])
    assert fake_runner.last("run").job.layout == layout


def test_a_saved_episode_is_recorded_and_seeds_its_sibling(controller, fake_runner, videos, monkeypatch):
    folder = videos(["EP05.mkv", "EP06.mkv"])
    save_episode(folder / "EP05.mkv", ready_entry("EP05.mkv"))
    controller.open_path(str(folder / "EP05.mkv"))
    controller.set_brightness("EP05.mkv", 212)                # an edit: saved, which records it
    controller.close_folder()
    index = json.loads(index_path(str(cache_root())).read_text(encoding="utf-8"))
    assert [record["name"] for record in index["folders"][str(folder)]] == ["EP05.mkv"]

    seeds = []
    original = AutoPilot.__init__

    def spy(self, *args, **kwargs):
        seeds.append(kwargs.get("seed_consensus"))
        original(self, *args, **kwargs)

    monkeypatch.setattr(AutoPilot, "__init__", spy)
    controller.open_path(str(folder / "EP06.mkv"))
    seed = (786 / 1080, 53 / 1080)
    assert seeds == [[seed] * CONSENSUS_MIN_ENTRIES]
    assert controller.episode_seed_source() == "EP05.mkv"

    metadata = fake_runner.last("metadata", "EP06.mkv")
    fake_runner.finish(metadata, MetadataResult("EP06.mkv", 1920, 1080, 1400.0, 23.976))
    deliver(controller)
    crop = fake_runner.last("crop", "EP06.mkv").job
    assert crop.hint is None                                   # a consensus, not a hint


def test_an_episode_with_a_crop_of_its_own_takes_no_seed(controller, videos, monkeypatch):
    folder = videos(["EP05.mkv", "EP06.mkv"])
    save_episode(folder / "EP05.mkv", ready_entry("EP05.mkv"))
    controller.open_path(str(folder / "EP05.mkv"))
    controller.set_brightness("EP05.mkv", 212)
    controller.close_folder()
    save_episode(folder / "EP06.mkv", ready_entry("EP06.mkv", crop=(300, 800, 1300, 60)))

    seeds = []
    original = AutoPilot.__init__
    monkeypatch.setattr(AutoPilot, "__init__",
                        lambda self, *a, **k: seeds.append(k.get("seed_consensus")) or original(self, *a, **k))
    controller.open_path(str(folder / "EP06.mkv"))
    assert seeds == [None]
    assert controller.episode_seed_source() is None


def test_opening_an_episode_prunes_the_cache_but_never_itself(controller, videos):
    folder = videos(["EP06.mkv"])
    stale = cache_root() / "videos" / ("0" * 32)
    stale.mkdir(parents=True)
    (stale / "settings.json").write_text("{}", encoding="utf-8")
    old = time.time() - 200 * 24 * 3600
    os.utime(stale / "settings.json", (old, old))
    os.utime(stale, (old, old))
    controller.open_path(str(folder / "EP06.mkv"))
    assert not stale.exists()


# --------------------------------------------------------------------------
# The contract the episode views read
# --------------------------------------------------------------------------

def test_kept_duration_estimate_and_speed(controller, fake_runner, videos):
    folder = videos(["EP06.mkv"])
    save_episode(folder / "EP06.mkv", ready_entry("EP06.mkv", ranges=(("1:00", "11:00"), ("20:00", None))))
    controller.open_path(str(folder / "EP06.mkv"))
    assert controller.episode_kept_duration("EP06.mkv") == pytest.approx(600.0 + 200.0)
    assert controller.episode_speed_hint() is None
    assert controller.episode_estimate_seconds() is None
    with pytest.raises(KeyError):
        controller.episode_kept_duration("nope.mkv")

    record_speed(str(cache_root()), 400.0, 100.0)            # 4x real time
    assert controller.episode_speed_hint() == pytest.approx(4.0)
    assert controller.episode_estimate_seconds() == pytest.approx(200.0)

    controller.set_time_ranges("EP06.mkv", [])                # the whole file
    assert controller.episode_kept_duration("EP06.mkv") == pytest.approx(1400.0)


def test_a_finished_episode_run_remembers_its_speed(controller, fake_runner, videos):
    folder = videos(["EP06.mkv"])
    save_episode(folder / "EP06.mkv", ready_entry("EP06.mkv"))
    controller.open_path(str(folder / "EP06.mkv"))
    controller.start_run(["EP06.mkv"])
    run = fake_runner.last("run")
    fake_runner.emit(run, "run_file_started", file="EP06.mkv")
    fake_runner.emit(run, "run_file_finished", file="EP06.mkv", result={"ok": True, "lines": 2, "error": ""})
    fake_runner.finish(run, RunSummary(["EP06.mkv"], {}, [], 350.0))
    deliver(controller)
    assert last_speed(str(cache_root())) == pytest.approx(1400.0 / 350.0)


def test_a_stopped_episode_run_remembers_no_speed(controller, fake_runner, videos):
    folder = videos(["EP06.mkv"])
    save_episode(folder / "EP06.mkv", ready_entry("EP06.mkv"))
    controller.open_path(str(folder / "EP06.mkv"))
    controller.start_run(["EP06.mkv"])
    run = fake_runner.last("run")
    controller.stop_run()
    fake_runner.finish(run, RunSummary([], {}, ["EP06.mkv"], 350.0))
    deliver(controller)
    assert last_speed(str(cache_root())) is None


def test_output_lines_parse_the_written_output(controller, videos):
    folder = videos(["EP06.mkv"])
    save_episode(folder / "EP06.mkv", ready_entry("EP06.mkv"))
    controller.open_path(str(folder / "EP06.mkv"))
    assert controller.output_lines("EP06.mkv") == []
    assert not controller.is_done("EP06.mkv")
    write_output(controller.output_path("EP06.mkv"), [("0:00:01.00", "0:00:02.50", "你好"),
                                                      ("0:01:00.00", "0:01:01.00", "{\\an8}再见")])
    assert controller.output_lines("EP06.mkv") == [(1.0, 2.5, "你好"), (60.0, 61.0, "再见")]


def test_a_folder_and_an_episode_name_their_output_by_the_same_rule(controller, fake_runner, videos):
    """`<tag>/<stem>.<tag>.ass`, or `<stem>.<tag>.ass` next to the video with
    "Create subfolder" off -- in the workbench and the episode view alike."""
    folder = videos(["EP06.mkv", "EP07.mkv"])
    controller.open_path(str(folder))
    assert controller.output_path("EP07.mkv") == str(folder / "zh" / "EP07.zh.ass")
    assert controller.episode_seed_source() is None
    controller.update_folder(output_subfolder=False, ocr_lang="japan")
    assert controller.output_path("EP07.mkv") == str(folder / "EP07.ja.ass")

    controller.open_path(str(folder / "EP06.mkv"))
    assert controller.is_episode
    assert controller.output_path("EP06.mkv") == str(folder / "zh" / "EP06.zh.ass")
    assert controller.output_label("EP06.mkv") == "zh/EP06.zh.ass"
    controller.update_folder(output_subfolder=False)
    assert controller.output_path("EP06.mkv") == str(folder / "EP06.zh.ass")
    assert controller.output_label("EP06.mkv") == "EP06.zh.ass"


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------

def test_a_video_opens_the_episode_view_on_prepare(make_window, videos):
    window = make_window()
    folder = videos(["EP06.mkv", "EP07.mkv"])
    window.open_path(str(folder / "EP06.mkv"))
    settle()
    assert window.centre.currentWidget() is window.workbench
    assert window.queue.isHidden()
    assert window.episode_page() == "prepare"
    assert window.prepare_view.file() == "EP06.mkv"
    assert window.stage.current_file() == "EP06.mkv"
    topbar = window.topbar
    assert topbar.project_label.full_text() == "EP06.mkv"
    assert topbar.path_label.full_text() == f"· {folder}"
    assert topbar.start_button.text() == "▶ Start OCR"
    assert topbar._chips.isHidden()
    assert window.windowTitle() == "OCR Manager — EP06.mkv"


def test_a_folder_with_two_videos_opens_the_workbench(make_window, videos):
    window = make_window()
    folder = videos(["EP06.mkv", "EP07.mkv"])
    window.open_path(str(folder))
    settle()
    assert window.episode_page() is None
    assert not window.queue.isHidden()
    assert window.mode() == MODE_REVIEW
    assert window.modes.currentWidget() is window.review_area
    assert window.topbar.start_button.text().startswith("▶ Start")
    assert window.topbar.start_button.text() != "▶ Start OCR"


def test_going_from_an_episode_back_to_a_folder_restores_the_workbench(make_window, videos):
    window = make_window()
    window.open_path(str(videos(["EP06.mkv"], folder="one")))
    settle()
    assert window.episode_page() == "prepare"
    window.open_path(str(videos(["EP06.mkv", "EP07.mkv"], folder="two")))
    settle()
    assert window.episode_page() is None
    assert not window.queue.isHidden()
    assert window.modes.currentWidget() is window.review_area
    assert window.topbar.run_switch.labels() == ["Review", "Run"]


def test_the_whole_episode_flow(make_window, fake_runner, videos, monkeypatch):
    window = make_window()
    controller = window.controller
    folder = videos(["EP06.mkv"])
    before = listing(folder)
    window.open_path(str(folder / "EP06.mkv"))
    settle()
    assert window.episode_page() == "prepare"
    assert window.prepare_view.mode() == "working"

    run_detections(fake_runner, controller, "EP06.mkv")
    window.prepare_view.refresh()
    assert window.prepare_view.mode() == "ready"
    assert window.topbar.start_button.isEnabled()

    window.prepare_view.start_requested.emit()
    settle()
    assert window.episode_page() == "working"
    assert window.mode() == MODE_RUN
    assert window.topbar.run_switch.labels() == ["Review", "Working"]
    run = fake_runner.last("run")
    assert run.job.files[0].name == "EP06.mkv"

    fake_runner.emit(run, "run_file_started", file="EP06.mkv")
    for start, end, text in [(1.0, 2.5, "你好"), (60.0, 61.0, "再见")]:
        fake_runner.emit(run, "run_subtitle", file="EP06.mkv", result=(start, end, text))
    deliver(controller)
    assert controller.run_subtitles("EP06.mkv") == [(1.0, 2.5, "你好"), (60.0, 61.0, "再见")]

    write_output(controller.output_path("EP06.mkv"), [("0:00:01.00", "0:00:02.50", "你好"),
                                                      ("0:01:00.00", "0:01:01.00", "再见")])
    fake_runner.emit(run, "run_file_finished", file="EP06.mkv", result={"ok": True, "lines": 2, "error": ""})
    fake_runner.finish(run, RunSummary(["EP06.mkv"], {}, [], 60.0))
    deliver(controller)
    assert wait_for(lambda: window.episode_page() == "done")
    assert window.done_view.outcome() == "done"
    assert controller.output_lines("EP06.mkv") == [(1.0, 2.5, "你好"), (60.0, 61.0, "再见")]
    assert listing(folder) == sorted(before + ["zh", "zh/EP06.zh.ass"])

    window.done_view.review_requested.emit()
    settle()
    assert window.episode_page() == "review"
    assert window.mode() == MODE_REVIEW
    window.set_mode(MODE_RUN)                                 # the switch goes back to Done
    assert window.episode_page() == "done"

    asked = answer(monkeypatch, Yes)
    window.done_view.retry_requested.emit()                   # again: the output exists, so it asks
    settle()
    assert asked == [("Replace existing subtitles?",
                      "zh/EP06.zh.ass already exists. Replace it when the new one is ready?")]
    assert window.episode_page() == "working"
    assert len(fake_runner.of_kind("run")) == 2


def test_declining_the_overwrite_starts_nothing(make_window, fake_runner, videos, monkeypatch):
    window = make_window()
    folder = videos(["EP06.mkv"])
    save_episode(folder / "EP06.mkv", ready_entry("EP06.mkv"))
    write_output(str(folder / "zh" / "EP06.zh.ass"), [("0:00:01.00", "0:00:02.00", "旧")])
    window.open_path(str(folder / "EP06.mkv"))
    settle()
    assert window.controller.is_done("EP06.mkv")
    assert window.topbar.start_button.isEnabled()             # done files can be run again
    asked = answer(monkeypatch, No)
    window.topbar.start_button.click()
    settle()
    assert len(asked) == 1
    assert fake_runner.of_kind("run") == []
    assert window.episode_page() == "prepare"


def test_a_flagged_episode_reviews_on_its_flagged_tab(make_window, fake_runner, videos):
    window = make_window()
    folder = videos(["EP06.mkv"])
    window.open_path(str(folder / "EP06.mkv"))
    run_detections(fake_runner, window.controller, "EP06.mkv", crop_flag=FLAG_LOW_AGREEMENT)
    window.prepare_view.refresh()
    assert window.prepare_view.mode() == "flagged"
    assert not window.topbar.start_button.isEnabled()
    window.stage.set_current(window.stage.index_of("Brightness"))
    window.prepare_view.review_requested.emit()
    settle()
    assert window.episode_page() == "review"
    assert window.stage.current_tab().title == "Crop"


def test_space_and_t_act_on_the_episode_file(make_window, videos):
    window = make_window()
    folder = videos(["EP06.mkv"])
    save_episode(folder / "EP06.mkv", ready_entry("EP06.mkv", review=ReviewState.PROPOSED))
    window.open_path(str(folder / "EP06.mkv"))
    settle()
    window._sync_actions()
    assert window.review_action.isEnabled()
    assert window.proof_action.isEnabled()
    window.review_action.trigger()
    assert window.controller.entry("EP06.mkv").review == ReviewState.REVIEWED


def test_dropping_a_video_opens_it_as_an_episode(make_window, videos):
    window = make_window()
    folder = videos(["EP06.mkv", "EP07.mkv"])
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(folder / "EP07.mkv"))])
    enter = QDragEnterEvent(QPoint(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
    window.dragEnterEvent(enter)
    assert enter.isAccepted()
    window.dropEvent(QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                                Qt.KeyboardModifier.NoModifier))
    assert window.controller.episode_name() == "EP07.mkv"


def test_open_episode_starts_the_picker_at_the_last_dir(make_window, videos, monkeypatch):
    folder = videos(["EP06.mkv", "EP07.mkv"])
    asked = []
    monkeypatch.setattr(open_folder_module.shutil, "which", lambda tool: None)
    monkeypatch.setattr(open_folder_module.QFileDialog, "getOpenFileName",
                        lambda parent, caption, start, filter_: asked.append(start) or (str(folder / "EP07.mkv"), ""))
    open_folder_module.remember_path(str(folder / "EP06.mkv"))     # last opened: an episode's video
    window = make_window()
    window.open_view.episode_button.click()
    assert asked == [str(folder)]
    assert wait_for(lambda: window.controller.episode_name() == "EP07.mkv")

    window.open_action.trigger()                                    # Ctrl+O: the episode picker again
    assert asked == [str(folder), str(folder)]
    assert window.open_action.shortcut().toString() == "Ctrl+O"
    assert window.open_folder_action.shortcut().toString() == "Ctrl+Shift+O"


def test_done_offers_another_episode(make_window, videos, monkeypatch):
    folder = videos(["EP06.mkv", "EP07.mkv"])
    picked = []
    window = make_window()
    monkeypatch.setattr(window._episode_picker, "pick", picked.append)
    window.open_path(str(folder / "EP06.mkv"))
    window.done_view.open_another_requested.emit()
    assert picked == [str(folder)]


def test_the_settings_sheet_says_an_episodes_settings_are_its_own(make_window, controller, videos):
    """In a folder the sheet's settings apply to every file in it; an
    episode's live in its own cache entry, so naming the folder would be
    wrong -- the sheet names the episode."""
    folder = videos(["EP06.mkv", "notes.txt"])
    window = make_window()
    window.open_path(str(folder / "EP06.mkv"))
    window.open_folder_settings()
    settle()
    assert window.folder_settings.scope_label.full_text() == "applies to EP06.mkv only"


def test_a_doubted_brightness_keeps_preparing_until_its_confirm_answers(make_window, controller, fake_runner, videos):
    """The detector's doubt makes the file FLAGGED at once, but the confirm
    stage is already on its way to answer it: the Preparing screen goes on
    working ("Double-checking the brightness") rather than telling the user
    to take a look detection had not finished with -- and turns green when
    the confirm reads the strip."""
    from core.detect.confirm import ConfirmResult, Rung
    from core.jobs.detect_jobs import ConfirmJobResult

    folder = videos(["EP06.mkv"])
    window = make_window()
    window.open_path(str(folder / "EP06.mkv"))
    run_detections(fake_runner, controller, "EP06.mkv", brightness_flag=FLAG_NARROW_PLATEAU, hold=("confirm",))
    assert controller.entry("EP06.mkv").review == ReviewState.FLAGGED
    (confirm,) = [s for s in fake_runner.queued() if s.job.kind == "confirm"]
    settle()
    assert window.prepare_view.mode() == "working"
    assert window.prepare_view.checklist()[-1] == ("Double-checking the brightness", "active")

    job = confirm.job
    rung = Rung(threshold=job.start_value, gated=True, text="字幕", confidence=0.99, passed=True)
    result = ConfirmResult(value=job.start_value, probe_time=job.probe_time, rungs=(rung,), cancelled=False)
    fake_runner.finish(confirm, ConfirmJobResult(job.file, result, job.crop_box, job.start_value, job.conf_threshold))
    deliver(controller)
    assert controller.entry("EP06.mkv").review == ReviewState.PROPOSED
    assert window.prepare_view.mode() == "ready"
