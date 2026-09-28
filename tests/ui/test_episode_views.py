"""The episode view's screens (app/views/episode/), offscreen, over a stub
controller.

The stub is a QObject carrying the controller's real signals and the
methods the episode views call -- the existing ones with ProjectController's
signatures, and the new episode ones from the plan's controller contract,
which the real controller does not have yet. Nothing decodes: a "frame" is
a small numpy array the test delivers with `deliver_frame`, emitting
`frame_ready` exactly as the controller does. Clocks are fake; animations
are off (`animated=False`), so every assertion is about what is shown, not
when."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app.episode_feed import Line, LineFeed, format_ts
from app.run_snapshot import CANCELLED, DONE, FAILED, RUNNING, RunFileRow, RunSnapshot
from app.views.episode import DoneView, PrepareView, ScriptPanel, WorkingView
from app.views.episode import prepare as prepare_module
from app.views.episode.done import OUTCOME_DONE, OUTCOME_FAILED, OUTCOME_STOPPED
from app.views.episode.prepare import (
    ACTIVE,
    HEADLINE_FLAGGED,
    HEADLINE_READY,
    HEADLINE_WORKING,
    MISSING,
    MODE_FLAGGED,
    MODE_READY,
    MODE_WORKING,
    WAITING,
)
from app.views.episode.prepare import (
    DONE as CHECK_DONE,
)
from app.views.episode.working import TITLE_DIALOGUE, TITLE_LABELS
from core.detect import crop as crop_detect
from core.project import (
    Brightness,
    Crop,
    FileEntry,
    FolderSettings,
    Media,
    ReviewState,
    Source,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NAME = "EP06.mkv"
VIDEO_DIR = "/videos/anime/S2"


def settle() -> None:
    QApplication.processEvents()


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _key(name: str, time_value: float) -> tuple:
    return name, round(float(time_value), 3)


class StubController(QObject):
    """ProjectController's signals and the calls the episode views make."""

    project_opened = pyqtSignal(str)
    project_closed = pyqtSignal()
    files_changed = pyqtSignal()
    file_changed = pyqtSignal(str)
    folder_changed = pyqtSignal()
    activity_changed = pyqtSignal()
    frame_ready = pyqtSignal(str, float)
    run_changed = pyqtSignal()
    run_subtitle = pyqtSignal(str, float, float, str)

    def __init__(self, entry: FileEntry, folder: FolderSettings | None = None):
        super().__init__()
        self.project = SimpleNamespace(path=VIDEO_DIR, folder=folder or FolderSettings(), files={entry.name: entry})
        self.pending: dict[str, set[str]] = {}
        self.running: dict[str, set[str]] = {}
        self.startable: list[str] = []
        self.frames: dict[tuple, np.ndarray] = {}
        self.requests: list[tuple[str, list[float]]] = []
        self.snapshot: RunSnapshot | None = None
        self.subtitles: list[tuple[float, float, str]] = []
        self.done: set[str] = set()
        self.seed: str | None = None
        self.estimate: float | None = None
        self.kept = 1200.0
        self.speed: float | None = None
        self.output: list[tuple[float, float, str]] = []

    # existing controller API
    def entry(self, name: str) -> FileEntry:
        return self.project.files[name]

    def pending_detectors(self) -> dict[str, set[str]]:
        return {name: set(kinds) for name, kinds in self.pending.items()}

    def running_detectors(self, name: str) -> set[str]:
        return set(self.running.get(name, set()))

    def startable_files(self, *, include_done: bool = False) -> list[str]:
        return list(self.startable)

    def request_frames(self, name: str, times: list[float], exact: bool = False) -> None:
        self.requests.append((name, list(times)))

    def frame(self, name: str, time: float, exact: bool = False):
        return self.frames.get(_key(name, time))

    def run_snapshot(self) -> RunSnapshot | None:
        return self.snapshot

    def run_subtitles(self, name: str) -> list[tuple[float, float, str]]:
        return list(self.subtitles)

    def is_done(self, name: str) -> bool:
        return name in self.done

    # the plan's new episode API
    def episode_name(self) -> str | None:
        return NAME

    def episode_seed_source(self) -> str | None:
        return self.seed

    def episode_estimate_seconds(self) -> float | None:
        return self.estimate

    def episode_kept_duration(self, name: str) -> float:
        return self.kept

    def episode_speed_hint(self) -> float | None:
        return self.speed

    def output_path(self, name: str) -> str:
        return f"{VIDEO_DIR}/{Path(name).stem}.zh.ass"

    def output_lines(self, name: str) -> list[tuple[float, float, str]]:
        return list(self.output)

    # test helpers
    def deliver_frame(self, name: str, time_value: float) -> None:
        self.frames[_key(name, time_value)] = np.full((72, 128, 3), 40, dtype=np.uint8)
        self.frame_ready.emit(name, time_value)

    def requested_times(self) -> list[float]:
        return [t for _name, times in self.requests for t in times]


def pending_entry() -> FileEntry:
    return FileEntry(NAME, media=Media(1920, 1080, 1420.0, 23.976), review=ReviewState.PENDING)


def ready_entry(review=ReviewState.PROPOSED) -> FileEntry:
    return FileEntry(NAME, media=Media(1920, 1080, 1420.0, 23.976), review=review,
                     crop=Crop(288, 900, 1344, 120, Source.DETECTED), brightness=Brightness(210, Source.DETECTED),
                     sample_time=578.0, evidence={"audio": {"speech": [[1.0, 2.0]]}})


def flagged_entry() -> FileEntry:
    entry = ready_entry(ReviewState.FLAGGED)
    entry.flags["crop"] = crop_detect.FLAG_LOW_AGREEMENT
    return entry


def run_snapshot(state=RUNNING, progress=0.0, phase="Extracting dialogue", *, lines=0, finished=False,
                 error="", started_at=0.0, finished_at=None) -> RunSnapshot:
    row = RunFileRow(NAME, state=state, phase=phase, progress=progress, lines=lines, error=error,
                     started_at=started_at, finished_at=finished_at)
    return RunSnapshot(files=(row,), started_at=started_at, parallel=1, finished=finished,
                       finished_at=finished_at if finished else None)


def batch(count: int, first: float = 100.0) -> list[tuple[float, float, str]]:
    return [(first + 3.0 * i, first + 3.0 * i + 2.0, f"第{i}句") for i in range(count)]


# --------------------------------------------------------------------------
# The import boundary
# --------------------------------------------------------------------------

def test_episode_views_import_no_core_modules():
    offenders = []
    for path in sorted((REPO_ROOT / "app" / "views" / "episode").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            offenders += [f"{path.name}: {m}" for m in modules if m.split(".")[0] in ("core", "videocr")]
    assert offenders == []


# --------------------------------------------------------------------------
# PrepareView
# --------------------------------------------------------------------------

def test_prepare_ticks_the_checklist_from_the_entry_and_pending_detectors(qapp):
    controller = StubController(pending_entry())
    controller.pending[NAME] = {"audio_profile", "crop", "brightness"}
    controller.running[NAME] = {"audio_profile"}
    view = PrepareView(controller)
    view.set_file(NAME)
    assert view.mode() == MODE_WORKING
    assert view.headline.text() == HEADLINE_WORKING
    assert [state for _label, state in view.checklist()] == [CHECK_DONE, ACTIVE, WAITING, WAITING]
    assert [label for label, _state in view.checklist()] == [
        "Read the video", "Listened for speech", "Finding the subtitle area", "Measuring subtitle brightness"]
    assert not view.start_button.isEnabled()
    assert view.preview.isHidden()
    assert view.progress.fraction() == pytest.approx(0.25)


def test_prepare_all_green(qapp):
    controller = StubController(ready_entry())
    controller.startable = [NAME]
    controller.estimate = 360.0
    controller.seed = "EP05.mkv"
    view = PrepareView(controller)
    view.set_file(NAME)
    assert view.mode() == MODE_READY
    assert view.headline.text() == HEADLINE_READY
    assert view.subline.text() == "Found the subtitles and tuned the brightness · about 6 min on this computer"
    assert view.seed_note.text() == "Starting from EP05's settings"
    assert view.start_button.isEnabled()
    assert view.primary_button() is view.start_button
    assert view.start_button.property("variant") == "primary"
    assert view.review_button.property("variant") == "default"
    assert view.reason.isHidden()
    assert view.preview.box() == (288, 900, 1344, 120) and view.preview.tone() == "ok"
    assert controller.requested_times() == [578.0]            # the crop's first hit
    controller.deliver_frame(NAME, 578.0)
    assert view.preview.has_image()


def test_prepare_leaves_out_the_estimate_without_a_remembered_speed(qapp):
    controller = StubController(ready_entry(ReviewState.REVIEWED))
    controller.startable = [NAME]
    view = PrepareView(controller)
    view.set_file(NAME)
    assert view.subline.text() == "Found the subtitles and tuned the brightness"
    assert view.seed_note.isHidden()


def test_prepare_enter_starts_when_all_green(qapp):
    controller = StubController(ready_entry())
    controller.startable = [NAME]
    view = PrepareView(controller)
    view.set_file(NAME)
    started = []
    view.start_requested.connect(lambda: started.append(True))
    QTest.keyClick(view, Qt.Key.Key_Return)
    assert started == [True]


def test_prepare_needs_a_look(qapp):
    controller = StubController(flagged_entry())
    view = PrepareView(controller)
    view.set_file(NAME)
    assert view.mode() == MODE_FLAGGED
    assert view.headline.text() == HEADLINE_FLAGGED
    assert view.reason.text() == "⚠ Subtitle area: few frames agreed"
    assert not view.start_button.isEnabled()
    assert view.primary_button() is view.review_button
    assert view.review_button.property("variant") == "primary"
    assert view.start_button.property("variant") == "default"
    assert view.preview.tone() == "warn"
    assert view.flagged_field() == "crop"
    reviews = []
    view.review_requested.connect(lambda: reviews.append(True))
    QTest.keyClick(view, Qt.Key.Key_Return)                  # Enter presses the primary button: Review
    view.review_button.click()
    assert reviews == [True, True]


def test_prepare_start_follows_startable_whatever_the_screen_says(qapp):
    controller = StubController(ready_entry())
    view = PrepareView(controller)
    view.set_file(NAME)
    assert view.mode() == MODE_READY and not view.start_button.isEnabled()
    controller.startable = [NAME]
    controller.run_changed.emit()
    settle()
    assert view.start_button.isEnabled()


def test_prepare_moves_to_all_green_when_detection_finishes(qapp):
    entry = pending_entry()
    controller = StubController(entry)
    controller.pending[NAME] = {"crop", "brightness"}
    controller.running[NAME] = {"crop"}
    view = PrepareView(controller)
    view.set_file(NAME)
    assert view.mode() == MODE_WORKING
    ready = ready_entry()
    controller.project.files[NAME] = ready
    controller.pending.clear()
    controller.running.clear()
    controller.startable = [NAME]
    controller.file_changed.emit(NAME)
    settle()
    assert view.mode() == MODE_READY and view.start_button.isEnabled()


def test_prepare_a_job_that_left_no_value_does_not_hang_the_checklist(qapp):
    entry = ready_entry()
    entry.evidence.clear()                                   # the audio profile found no audio
    controller = StubController(entry)
    view = PrepareView(controller)
    view.set_file(NAME)
    assert dict(view.checklist())["Listened for speech"] == MISSING
    assert view.mode() == MODE_READY


def test_prepare_labels_only_needs_no_crop_or_brightness(qapp):
    entry = FileEntry(NAME, media=Media(1920, 1080, 1420.0, 23.976), review=ReviewState.PROPOSED,
                      evidence={"audio": {}} )
    entry.evidence["audio"] = {"speech": []}
    controller = StubController(entry, FolderSettings(dialogue_enabled=False, labels_enabled=True))
    view = PrepareView(controller)
    view.set_file(NAME)
    assert [state for _label, state in view.checklist()] == [CHECK_DONE] * 4
    assert view.subline.text() == prepare_module.READY_LABELS


# --------------------------------------------------------------------------
# ScriptPanel
# --------------------------------------------------------------------------

def test_script_rows_follow_and_click(qapp):
    panel = ScriptPanel(animated=False)
    panel.resize(300, 200)
    panel.show()
    for line in batch(40):
        panel.append(line)
    settle()
    assert panel.count() == 40 and panel.count_text() == "40 lines"
    assert panel.row_texts()[0] == f"{format_ts(100.0)} → {format_ts(102.0)}  第0句"
    bar = panel.list_view.verticalScrollBar()
    assert panel.is_following() and bar.value() == bar.maximum()
    bar.setValue(0)                                          # the user scrolls up
    assert not panel.is_following()
    panel.append((500.0, 501.0, "late"))
    assert bar.value() == 0                                  # held still while reading
    clicked = []
    panel.line_clicked.connect(lambda s, e, t: clicked.append((s, e, t)))
    panel.click_row(2)
    assert clicked == [(106.0, 108.0, "第2句")]
    panel.follow_button.click()
    settle()
    assert panel.is_following() and bar.value() == bar.maximum()
    panel.set_total(60)
    assert panel.count_text() == "60 lines"
    panel.close()


# --------------------------------------------------------------------------
# WorkingView
# --------------------------------------------------------------------------

def working(controller: StubController, clock: FakeClock) -> WorkingView:
    view = WorkingView(controller, clock=clock, animated=False)
    view.set_file(NAME)
    return view


def test_a_batch_drips_each_line_only_after_its_frame(qapp):
    controller = StubController(ready_entry())
    controller.snapshot = run_snapshot(progress=0.1)
    clock = FakeClock()
    view = working(controller, clock)
    lines = batch(32)
    for line in lines:
        controller.run_subtitle.emit(NAME, *line)
    settle()                                                 # one event-loop turn: one batch
    assert len(controller.requests) == 1
    assert controller.requests[0][1] == [Line(*line).mid for line in lines]
    assert view.script.count() == 0 and view.script.count_text() == "32 lines"   # counts are the real state
    view.tick()
    assert view.script.count() == 0                          # no frame yet
    for index, line in enumerate(lines):
        clock.advance(view.feed.interval)
        view.tick()
        assert view.script.count() == index                  # line `index` still waits for its frame
        controller.deliver_frame(NAME, Line(*line).mid)
        view.tick()
        assert view.script.count() == index + 1
        assert view.shown_line() == Line(*line)
        assert view.slideshow.caption() == line[2] and view.slideshow.has_picture()
    assert view.script.lines() == [Line(*line) for line in lines]


def test_frames_already_held_count_as_ready(qapp):
    controller = StubController(ready_entry())
    controller.snapshot = run_snapshot(progress=0.1)
    clock = FakeClock()
    view = working(controller, clock)
    line = batch(1)[0]
    controller.frames[_key(NAME, Line(*line).mid)] = np.zeros((72, 128, 3), dtype=np.uint8)
    controller.run_subtitle.emit(NAME, *line)
    settle()
    view.tick()
    assert view.script.lines() == [Line(*line)]


def test_a_frame_that_never_comes_releases_the_line_after_the_timeout(qapp):
    controller = StubController(ready_entry())
    controller.snapshot = run_snapshot(progress=0.1)
    clock = FakeClock()
    view = working(controller, clock)
    line = batch(1)[0]
    controller.run_subtitle.emit(NAME, *line)
    settle()
    view.tick()
    assert view.script.count() == 0
    clock.advance(LineFeed.FRAME_TIMEOUT)
    view.tick()
    assert view.script.lines() == [Line(*line)]
    assert view.slideshow.caption() == line[2] and not view.slideshow.has_picture()


def test_other_files_lines_are_ignored(qapp):
    controller = StubController(ready_entry())
    view = working(controller, FakeClock())
    controller.run_subtitle.emit("other.mkv", 1.0, 2.0, "x")
    settle()
    assert view.received() == [] and controller.requests == []


def test_the_run_ending_flushes_the_queue(qapp):
    controller = StubController(ready_entry())
    controller.snapshot = run_snapshot(progress=0.5)
    view = working(controller, FakeClock())
    lines = batch(10)
    for line in lines:
        controller.run_subtitle.emit(NAME, *line)
    settle()
    assert view.script.count() == 0
    controller.snapshot = run_snapshot(DONE, 1.0, lines=10, finished=True, finished_at=100.0)
    controller.run_changed.emit()
    settle()
    assert view.script.lines() == [Line(*line) for line in lines]
    assert view.feed.pending == 0


def test_hud_shows_phase_position_eta_and_speed(qapp):
    controller = StubController(ready_entry())
    controller.kept = 1200.0
    controller.speed = 4.0                                   # remembered: 4x, so 300 s for the file
    clock = FakeClock(0.0)
    controller.snapshot = run_snapshot(progress=0.0)
    view = working(controller, clock)
    controller.run_changed.emit()
    settle()
    texts = view.hud_texts()
    assert texts["title"] == TITLE_DIALOGUE
    assert texts["position"] == "0:00 of 20:00"
    assert texts["eta"] == "≈ 5 min"
    assert texts["speed"] == "4.0× real time"
    clock.advance(30.0)
    controller.snapshot = run_snapshot(progress=0.5)
    controller.run_changed.emit()
    settle()
    texts = view.hud_texts()
    assert texts["position"] == "10:00 of 20:00"
    assert texts["eta"] == "≈ 30 s"                          # 50 % in 30 s: measured only past 5 %
    assert texts["speed"] == "20.0× real time"
    controller.snapshot = run_snapshot(progress=0.1, phase="Extracting labels")
    controller.run_changed.emit()
    settle()
    assert view.hud_texts()["title"] == TITLE_LABELS


def test_hud_ticks_every_reported_line_at_once(qapp):
    controller = StubController(ready_entry())
    controller.snapshot = run_snapshot(progress=0.2)
    view = working(controller, FakeClock())
    controller.run_subtitle.emit(NAME, 710.0, 712.0, "a")
    settle()
    assert view.hud.ticks == [pytest.approx(710.0 / 1420.0)]


def test_clicking_a_line_holds_its_frame_until_following_resumes(qapp):
    controller = StubController(ready_entry())
    controller.snapshot = run_snapshot(progress=0.1)
    clock = FakeClock()
    view = working(controller, clock)
    lines = batch(3)
    for line in lines:
        controller.frames[_key(NAME, Line(*line).mid)] = np.zeros((72, 128, 3), dtype=np.uint8)
        controller.run_subtitle.emit(NAME, *line)
    settle()
    view.tick()
    assert view.shown_line() == Line(*lines[0])
    view.script.click_row(0)
    clock.advance(view.feed.interval)
    view.tick()
    assert view.script.count() == 2
    assert view.shown_line() == Line(*lines[0])              # held
    view.script.follow_button.click()
    clock.advance(view.feed.interval)
    view.tick()
    assert view.shown_line() == Line(*lines[2])


def test_reset_takes_lines_the_run_reported_before_the_view_came_up(qapp):
    controller = StubController(ready_entry())
    controller.subtitles = batch(3)
    view = working(controller, FakeClock())
    assert view.script.count() == 3
    controller.subtitles = []
    view.reset()
    assert view.script.count() == 0 and view.received() == []


# --------------------------------------------------------------------------
# DoneView
# --------------------------------------------------------------------------

def done_view(controller: StubController, opened: list) -> DoneView:
    view = DoneView(controller, open_url=lambda url: opened.append(url.toLocalFile()), animated=False)
    view.set_file(NAME)
    return view


def test_done_loads_the_written_output(qapp):
    controller = StubController(ready_entry())
    controller.output = batch(3)
    controller.done = {NAME}
    controller.snapshot = run_snapshot(DONE, 1.0, lines=3, finished=True, started_at=10.0, finished_at=382.0)
    opened = []
    view = done_view(controller, opened)
    assert view.outcome() == OUTCOME_DONE
    assert view.title.text() == "Subtitles unburned"
    assert "3 lines in 6 min 12 s · saved as" in view.summary.text()
    assert "EP06.zh.ass" in view.summary.text()
    assert view.script.lines() == [Line(*line) for line in batch(3)]
    assert view.retry_button.isHidden()
    view.folder_button.click()
    assert opened == [VIDEO_DIR]


def test_done_clicking_a_line_shows_its_frame(qapp):
    controller = StubController(ready_entry())
    controller.output = batch(3)
    controller.done = {NAME}
    view = done_view(controller, [])
    view.script.click_row(1)
    mid = Line(*batch(3)[1]).mid
    assert controller.requested_times() == [mid]
    assert view.preview_line() == Line(*batch(3)[1]) and not view.preview.has_image()
    controller.deliver_frame(NAME, mid)
    assert view.preview.has_image()


def test_done_without_a_run_this_session_counts_the_file(qapp):
    controller = StubController(ready_entry())
    controller.output = batch(5)
    controller.done = {NAME}
    view = done_view(controller, [])
    assert view.summary.text().startswith("5 lines · saved as")


def test_done_signals(qapp):
    controller = StubController(ready_entry())
    view = done_view(controller, [])
    seen = []
    view.open_another_requested.connect(lambda: seen.append("another"))
    view.review_requested.connect(lambda: seen.append("review"))
    view.another_button.click()
    view.review_link.click()
    assert seen == ["another", "review"]


@pytest.mark.parametrize("state, outcome, title", [
    (FAILED, OUTCOME_FAILED, "OCR failed"),
    (CANCELLED, OUTCOME_STOPPED, "OCR stopped"),
])
def test_a_failed_or_stopped_run_offers_try_again(qapp, state, outcome, title):
    controller = StubController(ready_entry())
    controller.done = {NAME}                                 # an earlier output is still there
    controller.snapshot = run_snapshot(state, 0.4, finished=True, error="boom" if state == FAILED else "",
                                       finished_at=50.0)
    view = done_view(controller, [])
    assert view.outcome() == outcome and view.title.text() == title
    assert view.summary.text() == ("boom" if state == FAILED else "The run was stopped before it finished.")
    assert view.detail.text() == "The earlier EP06.zh.ass is unchanged."
    assert not view.retry_button.isHidden()
    retries = []
    view.retry_requested.connect(lambda: retries.append(True))
    view.retry_button.click()
    assert retries == [True]
