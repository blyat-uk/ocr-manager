"""Plan 3B Task 3: the workbench window -- top bar, review queue, stage,
inspector, activity strip, the open-folder empty state and `python -m app`.

Every test drives a real ProjectController over `fake_runner`
(tests/ui/conftest.py): nothing is decoded, and job events are delivered by
the test, then drained with `controller.drain_events()` (the controller's
30 ms timer slot) so each step is deterministic.

QSettings is pointed at a per-test directory (autouse `settings_dir`) before
any window is built, so no test reads or writes the user's real settings.
"""
from __future__ import annotations

import importlib
import os
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from PyQt6.QtCore import QMimeData, QPoint, QPointF, QSettings, Qt, QTimer, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget

from app.controller import ProjectController
from app.controller import UnsupportedProjectVersion
from app.main_window import WARM_LOOKAHEAD, OPEN_FAILED_TITLE, SAVE_FAILED_TITLE, MainWindow
from app.theme import tokens
from app.views import folder_settings, folder_settings_fields, open_folder as open_folder_module, run_view
from app.views.inspector_sections import button_text_budget
from app.views.stage import Stage, StageTab, placeholder_tabs
from app.widgets.base import KvRow
from core.detect.brightness import BrightnessResult
from core.detect.crop import FLAG_LOW_AGREEMENT, CropResult
from core.jobs.detect_jobs import BrightnessJobResult, CropJobResult, MetadataResult, ProofResult
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

SLAY_NAMES = [
    "ZS2_-_11_[1080p]TXHBR.mp4",
    "ZS2_-_12_[1080p]TXHBR.mp4",
    "ZS2_-_13_[1080p]TXHBR.mp4",
    "ZS2_-_14_[1080p]TXHBR.mp4",
    "ZS2_-_15_[1080p]TXHBR.mp4",
]
BOX = (288, 786, 1344, 53)
WAIT_MS = 5000
REPO_ROOT = Path(__file__).resolve().parents[2]
PROOF_WINDOW_TEXT = "09:37–10:07"      # the first slay file's 30 s proof window, from its sample time
NEWER_VERSION_TEXT = ("This folder was saved by a newer version of OCR Manager (project version 99). "
                      "Update the app to open it.")


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
    """Run the zero-delay refreshes the views coalesce bursts into."""
    QApplication.processEvents()


def entry(name, *, crop=BOX, brightness=209, review=ReviewState.REVIEWED, source=Source.DETECTED,
          flags=None, skipped=False, ranges=()) -> FileEntry:
    """A file with known media; `ranges=()` stores a MANUAL whole-file choice,
    so the folder's ranges analysis never holds the file PENDING."""
    item = FileEntry(name, media=Media(1920, 888, 1628.0, 25.0), review=review, skipped=skipped,
                     flags=dict(flags or {}), sample_time=578.0)
    if crop is not None:
        item.crop = Crop(*crop, source)
    if brightness is not None:
        item.brightness = Brightness(brightness, source)
    if ranges is not None:
        item.time_ranges = TimeRanges([TimeRange(start, end) for start, end in ranges], Source.MANUAL)
    return item


def write_project(folder: Path, entries: list[FileEntry], **folder_settings) -> Path:
    """Save a v2 project (config plus evidence cache) into `folder`."""
    save_project(Project(path=str(folder), folder=FolderSettings(**folder_settings),
                         files={item.name: item for item in entries}))
    return folder


def crop_result(submission, box=BOX, flagged=None) -> CropJobResult:
    result = CropResult(box=box, sample_pts=[578.0], envelope=box, agreed=3, probes_used=3,
                        flagged=flagged, hit_pts=[578.0], frame_size=(1920, 888))
    return CropJobResult(submission.job.file, result, submission.job.hint)


def brightness_result(submission, value=209) -> BrightnessJobResult:
    job = submission.job
    result = BrightnessResult(value=value, plateau=(190, 230), seed=value + 20, gate_floor=None,
                              flagged=None, curve=[])
    return BrightnessJobResult(job.file, result, {}, job.hint_value, job.crop_box)


class Calls:
    """Wraps a controller command on the instance: records each call, then
    runs the real command."""

    def __init__(self, controller, name: str, *, run: bool = True):
        self.calls: list[tuple] = []
        original = getattr(controller, name)

        def wrapper(*args, **kwargs):
            self.calls.append(args)
            return original(*args, **kwargs) if run else None

        setattr(controller, name, wrapper)


@pytest.fixture(autouse=True)
def settings_dir(tmp_path):
    path = tmp_path / "qsettings"
    path.mkdir()
    for fmt in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, str(path))
    return path


def app_settings() -> QSettings:
    return QSettings("OCRManager", "OCRTool")


@pytest.fixture
def controller(qapp, fake_runner):
    made = ProjectController(fake_runner, save_debounce_ms=10)
    yield made
    made.shutdown(timeout=0.5)


@pytest.fixture
def make_window(controller):
    windows = []

    def make(**kwargs) -> MainWindow:
        window = MainWindow(controller, **kwargs)
        window.resize(1440, 900)
        windows.append(window)
        return window

    yield make
    for window in windows:
        # The teardown is not a user: it never answers closeEvent's questions
        # (a run in progress, settings that could not be saved).
        window._closing = True
        window.close()
        window.deleteLater()


@pytest.fixture
def slay_window(make_window, tmp_project):
    window = make_window()
    window.open_folder(str(tmp_project(fixture="slay")))
    settle()
    return window


@pytest.fixture
def detected_window(make_window, tmp_project):
    """Five reviewed files whose crop and brightness were detected (so a new
    detection result may flag them)."""
    folder = write_project(tmp_project(SLAY_NAMES), [entry(name) for name in SLAY_NAMES], labels_enabled=False)
    window = make_window()
    window.open_folder(str(folder))
    settle()
    return window


def activate(window: MainWindow) -> None:
    """Show the window and make it active: window shortcuts only fire in the
    active window."""
    window.show()
    window.activateWindow()
    assert wait_for(lambda: QApplication.activeWindow() is window)


class EditorTab:
    """A StageTab whose page holds a focusable surface and a spin box and a
    line edit (plan 3C's tabs have such editors)."""

    def __init__(self, title: str):
        self.title = title
        self._page = QWidget()
        layout = QVBoxLayout(self._page)
        self.surface = QWidget()
        self.surface.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.spin = QSpinBox()
        self.line = QLineEdit()
        for widget in (self.surface, self.spin, self.line):
            layout.addWidget(widget)
        self._panel = QWidget()

    def page(self) -> QWidget:
        return self._page

    def inspector_panel(self) -> QWidget:
        return self._panel

    def set_file(self, name) -> None:
        pass

    def refresh(self) -> None:
        pass


def menu_actions(menu) -> dict:
    return {action.text(): action for action in menu.actions() if not action.isSeparator()}


# --------------------------------------------------------------------------
# Window, top bar and fixed columns
# --------------------------------------------------------------------------

def test_window_shows_counts_start_and_fixed_columns(slay_window):
    window = slay_window
    top = window.topbar
    assert top.reviewed_chip.text() == "5 reviewed"
    assert top.reviewed_chip.tone() == "ok"
    assert top.needs_chip.text() == "0 needs you"
    assert top.needs_chip.tone() == "warn"
    assert top.detecting_chip.text() == "0 detecting"
    assert top.start_button.text() == "▶ Start 5 ready files"
    assert top.start_button.isEnabled()
    assert top.settings_button.text() == "⚙ Folder settings"
    assert top.logs_button.text() == "⤓ Logs"
    assert top.run_switch.isHidden()
    folder = window.controller.project.path
    assert top.project_label.text() == os.path.basename(folder)
    assert top.path_label.text() == f"· {folder}"
    assert window.windowTitle() == f"OCR Manager — {os.path.basename(folder)}"
    # The tokens, never the numbers: the rail and the inspector are the
    # mockup's 246 and 322 px at the current UI scale, and a literal put back
    # here (or in the views) fails this at any scale but 1.0.
    assert window.queue.minimumWidth() == window.queue.maximumWidth() == tokens.RAIL_WIDTH == tokens.px(246)
    assert (window.inspector.minimumWidth() == window.inspector.maximumWidth()
            == tokens.INSPECTOR_WIDTH == tokens.px(322))
    assert window.queue.filter.labels() == ["All 5", "Needs you 0", "Reviewed 5"]
    assert window.queue.visible_names() == SLAY_NAMES
    assert window.queue.selected() == SLAY_NAMES[0]
    row = window.queue.row(SLAY_NAMES[0])
    assert (row.badge.text(), row.badge.property("badge")) == ("reviewed", "good")
    assert row.duration_label.text() == "23:38"


@contextmanager
def ui_scale(value: float):
    """Rebuild the token module at `value` for the body of the `with`.

    Every view reaches its sizes through the `tokens` MODULE (`from app.theme
    import tokens`), so reloading it in place is enough for widgets built
    inside the block to be laid out at that scale -- and it is the only way
    to reach the sizes a module computed at import time."""
    before = os.environ.get("OCR_MANAGER_UI_SCALE")
    os.environ["OCR_MANAGER_UI_SCALE"] = str(value)
    try:
        importlib.reload(tokens)
        yield
    finally:
        if before is None:
            os.environ.pop("OCR_MANAGER_UI_SCALE", None)
        else:
            os.environ["OCR_MANAGER_UI_SCALE"] = before
        importlib.reload(tokens)


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_the_shell_is_laid_out_at_whatever_the_ui_scale_says(make_window, tmp_project, scale):
    """Nothing in the shell may depend on the default scale being 1.25: at
    1.0 the window is the mockup's own size, at 2.0 every length is doubled.
    A hard-coded pixel anywhere in the rail, the inspector, the thumbnails,
    the sheet or the run view breaks one of these."""
    with ui_scale(scale):
        assert tokens.UI_SCALE == scale
        assert tokens.RAIL_WIDTH == round(246 * scale)
        assert tokens.INSPECTOR_WIDTH == round(322 * scale)
        window = make_window(tabs_factory=placeholder_tabs)
        window.open_folder(str(tmp_project(fixture="slay")))
        settle()

        assert window.queue.width() == tokens.RAIL_WIDTH
        assert window.inspector.width() == tokens.INSPECTOR_WIDTH
        thumb = window.queue.row(SLAY_NAMES[0]).thumb
        assert (thumb.width(), thumb.height()) == (tokens.THUMB_WIDTH, tokens.THUMB_HEIGHT)
        assert button_text_budget() == tokens.INSPECTOR_WIDTH - 1 - tokens.px(2 * 12) - tokens.px(18)

        # The sheet: min(the scaled 760, the window's own width less the rail).
        window.open_folder_settings()
        settle()
        sheet = window.folder_settings
        assert sheet.width() == min(tokens.px(folder_settings.MAX_WIDTH), 1440 - tokens.RAIL_WIDTH)
        assert sheet.nav_panel.width() == tokens.px(folder_settings.NAV_WIDTH)
        assert sheet.x() >= window.queue.width()                   # the rail stays visible at every scale
        # ... and its last section still scrolls to the top of the viewport:
        # the filler that makes room below it has to follow the scale too.
        titles = [title for title, _fields, _note in folder_settings_fields.SECTIONS]
        last = [title for title in titles if not sheet.section(title).isHidden()][-1]
        sheet.nav.item(titles.index(last)).click()
        settle()
        assert sheet.section(last).mapTo(sheet.scroll.viewport(), QPoint(0, 0)).y() \
            == tokens.px(folder_settings.CONTENT_MARGIN)
        sheet.close_sheet()

        assert window.run_view.live_panel.width() == tokens.px(run_view.LIVE_WIDTH)


def test_a_window_over_an_already_open_controller_shows_its_folder(controller, make_window, tmp_project):
    folder = tmp_project(fixture="slay")
    controller.open_folder(str(folder))
    window = make_window()
    assert window.centre.currentWidget() is window.workbench
    assert window.windowTitle() == f"OCR Manager — {folder.name}"
    assert window.queue.names() == SLAY_NAMES
    assert window.queue.selected() == SLAY_NAMES[0]
    assert window.inspector.file_label.full_text() == SLAY_NAMES[0]
    assert window.topbar.reviewed_chip.text() == "5 reviewed"


def test_flagging_a_file_updates_chips_badge_and_start(detected_window, fake_runner):
    window, controller = detected_window, detected_window.controller
    name = SLAY_NAMES[1]
    row = window.queue.row(name)
    top = window.topbar
    assert top.start_button.text() == "▶ Start 5 ready files"

    controller.redetect(name)
    crop = fake_runner.last("crop", name)
    fake_runner.start(crop)
    controller.drain_events()
    settle()
    assert top.detecting_chip.tone() == "run"              # B7: a detection is running
    fake_runner.finish(crop, crop_result(crop, flagged=FLAG_LOW_AGREEMENT))
    controller.drain_events()
    settle()
    assert row.badge.text() == "waiting"                   # brightness follows the re-detect, queued
    assert top.detecting_chip.text() == "1 detecting"
    assert top.detecting_chip.tone() == "idle"             # B7: only queued

    brightness = fake_runner.last("brightness", name)
    fake_runner.start(brightness)                           # "started" emits only activity_changed
    controller.drain_events()
    settle()
    assert row.badge.text() == "measuring brightness…"
    assert top.detecting_chip.tone() == "run"

    fake_runner.finish(brightness, brightness_result(brightness))
    controller.drain_events()
    settle()
    assert (row.badge.text(), row.badge.property("badge")) == ("check crop", "warn")
    assert top.reviewed_chip.text() == "4 reviewed"
    assert top.needs_chip.text() == "1 needs you"
    assert top.detecting_chip.text() == "0 detecting"
    assert top.start_button.text() == "▶ Start 4 ready files"
    assert window.queue.filter.labels() == ["All 5", "Needs you 1", "Reviewed 4"]


def test_start_is_disabled_without_ready_files_or_extraction(make_window, tmp_project):
    folder = write_project(tmp_project(["a.mkv", "b.mkv"]),
                           [entry("a.mkv", review=ReviewState.PROPOSED), entry("b.mkv", review=ReviewState.PROPOSED)])
    window = make_window()
    window.open_folder(str(folder))
    settle()
    controller, start = window.controller, window.topbar.start_button
    assert start.text() == "▶ Start 2 ready files" and start.isEnabled()

    controller.update_folder(labels_enabled=False)
    settle()
    assert start.isEnabled()
    with pytest.raises(ValueError, match="At least one of dialogue or labels must be on."):
        controller.update_folder(dialogue_enabled=False)          # both off is refused, Start stays usable
    settle()
    assert start.isEnabled() and controller.project.folder.dialogue_enabled
    controller.set_skipped("a.mkv", True)
    controller.set_skipped("b.mkv", True)
    settle()
    assert start.text() == "▶ Start 0 ready files"
    assert not start.isEnabled()

    both_off = write_project(tmp_project(["c.mkv"]), [entry("c.mkv", review=ReviewState.PROPOSED)],
                             dialogue_enabled=False, labels_enabled=False)   # e.g. a hand-edited .ocr.json
    window.open_folder(str(both_off))
    settle()
    assert start.text() == "▶ Start 1 ready file"
    assert not start.isEnabled()


# --------------------------------------------------------------------------
# Queue
# --------------------------------------------------------------------------

@pytest.fixture
def mixed_window(make_window, tmp_project):
    names = ["a.mkv", "b.mkv", "c.mkv", "d.mkv", "e.mkv"]
    flagged = entry("b.mkv", review=ReviewState.PROPOSED, flags={"crop": FLAG_LOW_AGREEMENT})
    flagged.evidence["crop"] = {"agreed": 1, "probes_used": 3, "flagged": FLAG_LOW_AGREEMENT}
    entries = [
        entry("a.mkv"),                                                        # reviewed
        flagged,                                                               # flagged
        entry("c.mkv", review=ReviewState.PROPOSED),                           # ready
        entry("d.mkv"),                                                        # reviewed
        entry("e.mkv", skipped=True),                                          # skipped
    ]
    window = make_window()
    window.open_folder(str(write_project(tmp_project(names), entries)))
    settle()
    return window


def test_queue_filter_counts_and_filtering(mixed_window):
    queue = mixed_window.queue
    assert queue.filter.labels() == ["All 5", "Needs you 1", "Reviewed 2"]
    assert [queue.row(name).badge.text() for name in queue.visible_names()] == [
        "reviewed", "check crop", "ready", "reviewed", "skipped"]
    segments = queue.filter.findChildren(QPushButton)

    segments[1].click()
    assert queue.visible_names() == ["b.mkv"]
    assert queue.selected() == "b.mkv"                      # the selection follows into the filter
    assert queue.row("a.mkv").isHidden()

    segments[2].click()
    assert queue.visible_names() == ["a.mkv", "d.mkv"]

    mixed_window.controller.mark_reviewed("b.mkv")           # accepted: leaves "Needs you"
    settle()
    assert queue.filter.labels() == ["All 5", "Needs you 0", "Reviewed 3"]
    assert queue.visible_names() == ["a.mkv", "b.mkv", "d.mkv"]

    segments[0].click()
    assert queue.visible_names() == ["a.mkv", "b.mkv", "c.mkv", "d.mkv", "e.mkv"]


def test_a_filter_that_matches_nothing_says_so_and_drops_the_selection(mixed_window):
    """An empty rail while the stage and the inspector still describe a file
    the rail does not list is a window at odds with itself. Switching filter
    is navigation, so the selection goes where the filter goes -- nowhere."""
    window = mixed_window
    queue = window.queue
    window.controller.mark_reviewed("b.mkv")                 # nothing needs the user now
    settle()
    assert queue.filter.labels()[1] == "Needs you 0"
    assert queue.empty_label.isHidden()

    queue.filter.findChildren(QPushButton)[1].click()        # "Needs you"
    settle()
    assert queue.visible_names() == []
    assert not queue.empty_label.isHidden()
    assert queue.empty_label.text() == "No files match this filter."
    assert queue.selected() is None
    assert window.stage.current_file() is None
    assert window.inspector.file_label.full_text() == "No file selected"

    queue.filter.findChildren(QPushButton)[0].click()        # back to "All"
    settle()
    assert queue.empty_label.isHidden()
    assert queue.selected() == "a.mkv"


def test_queue_keyboard_moves_marks_and_proves(slay_window, fake_runner):
    window, controller = slay_window, slay_window.controller
    queue = window.queue
    activate(window)
    queue.setFocus()
    marks = Calls(controller, "mark_reviewed")
    proofs = Calls(controller, "run_proof")
    assert queue.selected() == SLAY_NAMES[0]

    QTest.keyClick(queue, Qt.Key.Key_Down)
    assert queue.selected() == SLAY_NAMES[1]
    assert window.inspector.file_label.full_text() == SLAY_NAMES[1]
    assert window.stage.current_file() == SLAY_NAMES[1]
    QTest.keyClick(queue, Qt.Key.Key_Down)
    QTest.keyClick(queue, Qt.Key.Key_Up)
    assert queue.selected() == SLAY_NAMES[1]
    QTest.keyClick(queue, Qt.Key.Key_Up)
    QTest.keyClick(queue, Qt.Key.Key_Up)                    # stays on the first row
    assert queue.selected() == SLAY_NAMES[0]

    QTest.keyClick(queue, Qt.Key.Key_Space)                 # reviewed -> not reviewed
    assert marks.calls == [(SLAY_NAMES[0], False)]
    assert controller.entry(SLAY_NAMES[0]).review != ReviewState.REVIEWED
    assert queue.row(SLAY_NAMES[0]).badge.text() == "ready"
    QTest.keyClick(queue, Qt.Key.Key_Space)
    assert marks.calls[-1] == (SLAY_NAMES[0], True)
    assert controller.entry(SLAY_NAMES[0]).review == ReviewState.REVIEWED

    QTest.keyClick(queue, Qt.Key.Key_T)
    assert proofs.calls == [(SLAY_NAMES[0],)]
    assert fake_runner.last("proof", SLAY_NAMES[0])
    assert window.inspector.proof_status.text() == f"running on {PROOF_WINDOW_TEXT}…"


def test_moving_the_selection_hands_the_new_file_to_the_controller(slay_window):
    """So the frames and strips of the file being left stop competing for the
    CPU lane (ProjectController.set_view_file)."""
    window, queue = slay_window, slay_window.queue
    view_files = Calls(window.controller, "set_view_file")
    activate(window)
    queue.setFocus()

    QTest.keyClick(queue, Qt.Key.Key_Down)
    QTest.keyClick(queue, Qt.Key.Key_Down)
    assert [call[0] for call in view_files.calls] == [SLAY_NAMES[1], SLAY_NAMES[2]]
    # ... and the files that selection is heading towards, so the view cache
    # warms ahead of the cursor instead of in name order (AutoPilot.boost_warm).
    assert [call[1] for call in view_files.calls] == [
        queue.visible_names()[1:1 + 1 + WARM_LOOKAHEAD],
        queue.visible_names()[2:2 + 1 + WARM_LOOKAHEAD],
    ]


def test_clicking_a_row_selects_it(slay_window):
    queue = slay_window.queue
    row = queue.row(SLAY_NAMES[3])
    QTest.mouseClick(row, Qt.MouseButton.LeftButton)
    assert queue.selected() == SLAY_NAMES[3]
    assert row.property("selected") is True
    assert queue.row(SLAY_NAMES[0]).property("selected") is False


def test_context_menu_offers_b11_actions_and_invokes_commands(slay_window, fake_runner):
    window, controller = slay_window, slay_window.controller
    queue = window.queue
    name, other = SLAY_NAMES[2], SLAY_NAMES[3]
    logs_requested = []
    queue.logs_requested.connect(logs_requested.append)
    copies = Calls(controller, "copy_settings")
    pastes = Calls(controller, "paste_settings")
    redetects = Calls(controller, "redetect")
    proofs = Calls(controller, "run_proof")
    marks = Calls(controller, "mark_reviewed")
    skips = Calls(controller, "set_skipped")

    actions = menu_actions(queue.context_menu(name))
    assert list(actions) == ["Copy settings", "Paste settings onto this file", "Re-detect", "Test OCR (T)",
                             "Open logs", "Mark not reviewed", "Skip file"]
    assert not actions["Paste settings onto this file"].isEnabled()     # nothing copied yet

    actions["Copy settings"].trigger()
    assert copies.calls == [(name,)]
    actions = menu_actions(queue.context_menu(other))
    assert actions["Paste settings onto this file"].isEnabled()
    actions["Paste settings onto this file"].trigger()
    assert pastes.calls == [(other,)]
    assert controller.entry(other).crop.source == Source.MANUAL

    actions = menu_actions(queue.context_menu(name))
    actions["Re-detect"].trigger()
    assert redetects.calls == [(name,)]
    assert fake_runner.last("crop", name)
    actions["Test OCR (T)"].trigger()
    assert proofs.calls == [(name,)]
    actions["Open logs"].trigger()
    assert logs_requested == [name]
    actions["Mark not reviewed"].trigger()
    assert marks.calls == [(name, False)]
    actions["Skip file"].trigger()
    assert skips.calls == [(name, True)]
    assert queue.row(name).badge.text() == "skipped"

    actions = menu_actions(queue.context_menu(name))
    assert "Mark reviewed" in actions and "Include file" in actions
    actions["Mark reviewed"].trigger()
    actions["Include file"].trigger()
    assert marks.calls[-1] == (name, True)
    assert skips.calls[-1] == (name, False)


def test_thumbnail_and_crop_overlay_follow_the_model(slay_window, fake_runner):
    import numpy as np

    from core.jobs.detect_jobs import ThumbnailResult

    window, controller = slay_window, slay_window.controller
    name = SLAY_NAMES[0]
    thumb = window.queue.row(name).thumb
    assert thumb.image() is None
    box = thumb.crop_rect()
    # 1920x888 fitted into the thumbnail (the mockup's 56x32 at the current
    # UI scale) letterboxes it; the crop (288, 786, 1344, 53) sits at its
    # true place inside that frame, whatever the scale.
    wide, high = tokens.THUMB_WIDTH, tokens.THUMB_HEIGHT
    assert box.x() == pytest.approx(288 / 1920 * wide, abs=0.01)
    assert box.width() == pytest.approx(1344 / 1920 * wide, abs=0.01)
    assert box.y() == pytest.approx((high - 888 * wide / 1920) / 2 + 786 * wide / 1920, abs=0.01)

    submission = fake_runner.last("thumbnail", name)
    fake_runner.finish(submission, ThumbnailResult(name, submission.job.time, np.zeros((36, 64, 3), np.uint8)))
    controller.drain_events()
    assert thumb.image() is not None


# --------------------------------------------------------------------------
# Stage and inspector
# --------------------------------------------------------------------------

def test_stage_hosts_tabs_and_switches_pages(slay_window):
    stage = slay_window.stage
    assert isinstance(stage, Stage)
    assert [tab.title for tab in stage.tabs()] == ["Crop", "Brightness", "Time ranges"]
    assert stage.current_index() == 0
    changes = []
    stage.tab_changed.connect(changes.append)
    stage.tab_buttons()[2].click()
    assert changes == [2]
    assert stage.current_tab().title == "Time ranges"
    assert stage.page_host().currentWidget() is stage.current_tab().page()
    assert slay_window.inspector.current_panel() is stage.current_tab().inspector_panel()


def test_placeholder_tabs_follow_the_selected_file(qapp, controller, make_window, tmp_project):
    window = make_window(tabs_factory=placeholder_tabs)
    window.open_folder(str(tmp_project(fixture="slay")))
    crop_tab = window.stage.tabs()[0]
    window.queue.select(SLAY_NAMES[1])
    assert crop_tab.current_file() == SLAY_NAMES[1]
    assert crop_tab.values() == ["288, 784 · 1344 × 55", "imported"]
    window.controller.set_crop(SLAY_NAMES[1], (300, 790, 1300, 50))
    assert crop_tab.values() == ["300, 790 · 1300 × 50", "manual"]
    assert isinstance(crop_tab, StageTab)


def test_replaced_rows_are_unparented_not_just_unlaid_out(qapp, controller, make_window,
                                                          tmp_project):
    """`removeWidget` leaves a row a child of the list, at whatever size it
    had -- a row that was never laid out keeps QWidget's default 640x480 and
    paints over everything beneath it until it is really deleted.

    `held` stands in for what keeps a replaced row alive in real use: a row
    with a signal connection of its own outlives the call that dropped it,
    because the two reference each other (app/views/ranges_view.py's rows do,
    and one of them hid a row of buttons)."""
    window = make_window(tabs_factory=placeholder_tabs)
    window.open_folder(str(tmp_project(fixture="slay")))
    crop_tab = window.stage.tabs()[0]
    window.queue.select(SLAY_NAMES[1])
    page = crop_tab.page()
    held = page.findChildren(KvRow)
    assert len(held) == 2
    crop_tab.set_file(None)                        # no file: both rows go
    assert page.findChildren(KvRow) == []
    crop_tab.set_file(SLAY_NAMES[1])               # and come back, without the old ones
    assert len(page.findChildren(KvRow)) == 2
    assert not set(page.findChildren(KvRow)) & set(held)


def test_selecting_a_file_updates_the_inspector_header_and_detected_rows(slay_window):
    window = slay_window
    inspector = window.inspector
    window.queue.select(SLAY_NAMES[1])
    assert inspector.scope_label.text() == "◆ THIS EPISODE ONLY"
    assert inspector.file_label.full_text() == SLAY_NAMES[1]
    assert inspector.media_label.text() == "1920×888 · 27:08"            # fps unknown: omitted
    assert inspector.crop_row.value() == "288, 784 · 1344 × 55"
    assert inspector.crop_conf.caption() == "imported from the previous version"
    assert inspector.brightness_row.value() == "209"
    assert inspector.window_row.value() == "2:33 → 23:05"
    assert inspector.review_button.text() == "Mark not reviewed"
    assert inspector.skip_button.text() == "skip file"

    window.controller.set_time_ranges(SLAY_NAMES[1], None)               # "use whole file": an empty MANUAL list
    assert inspector.window_row.value() == "whole file"
    assert inspector.window_conf.caption() == "set by you"


def test_inspector_detected_rows_show_flagged_values_in_warn(mixed_window):
    inspector = mixed_window.inspector
    mixed_window.queue.select("b.mkv")
    assert inspector.media_label.text() == "1920×888 · 27:08 · 25 fps"
    assert inspector.crop_row.value_tone() == "warn"
    assert (inspector.crop_conf.caption(), inspector.crop_conf.property("tone")) == ("1 of 3 samples agree", "warn")
    assert inspector.brightness_row.value_tone() == ""
    assert inspector.review_button.text() == "✓ Mark reviewed (Space)"
    inspector.review_button.click()
    assert mixed_window.controller.entry("b.mkv").review == ReviewState.REVIEWED
    inspector.skip_button.click()
    assert mixed_window.controller.entry("b.mkv").skipped
    assert inspector.skip_button.text() == "include file"


def test_clicking_the_brightness_detected_row_switches_the_stage_tab(slay_window):
    window = slay_window
    QTest.mouseClick(window.inspector.brightness_row, Qt.MouseButton.LeftButton)
    assert window.stage.current_tab().title == "Brightness"
    assert window.inspector.current_panel() is window.stage.current_tab().inspector_panel()
    QTest.mouseClick(window.inspector.window_row, Qt.MouseButton.LeftButton)
    assert window.stage.current_tab().title == "Time ranges"


def test_redetect_button_redetects_the_selected_file(slay_window):
    calls = Calls(slay_window.controller, "redetect")
    slay_window.queue.select(SLAY_NAMES[4])
    slay_window.inspector.redetect_button.click()
    assert calls.calls == [(SLAY_NAMES[4],)]


def test_change_offer_appears_after_a_manual_crop_edit(detected_window):
    window, controller = detected_window, detected_window.controller
    inspector = window.inspector
    name = SLAY_NAMES[0]
    hints = Calls(controller, "redetect_others_with_hint")
    assert inspector.offer_section.isHidden()

    controller.set_crop(name, (290, 780, 1340, 60))
    assert not inspector.offer_section.isHidden()
    assert inspector.hint_buttons["crop"].text().replace("\n", " ") == (
        "↻ re-detect the other 4 using this crop as a hint")
    assert inspector.offer_note.text() == (
        "Corrections are never copied verbatim to other episodes. Instead the app offers:")
    inspector.hint_buttons["crop"].click()
    assert hints.calls == [(name, "crop")]
    assert inspector.hint_buttons["crop"].isHidden()
    assert inspector.offer_status.text() == "re-detecting 4 files…"   # until those jobs end

    controller.set_brightness(name, 215)
    assert not inspector.hint_buttons["brightness"].isHidden()
    inspector.this_file_only_button.click()
    assert inspector.hint_buttons["brightness"].isHidden()
    window.queue.select(SLAY_NAMES[1])
    window.queue.select(name)
    assert inspector.hint_buttons["brightness"].isHidden()             # dismissed stays dismissed


def test_change_offer_ignores_accepting_a_flagged_value(mixed_window):
    controller, inspector = mixed_window.controller, mixed_window.inspector
    mixed_window.queue.select("b.mkv")
    controller.mark_reviewed("b.mkv")                  # the flagged detected crop becomes MANUAL, same box
    assert controller.entry("b.mkv").crop.source == Source.MANUAL
    assert inspector.offer_section.isHidden()
    controller.set_brightness("b.mkv", 209)             # same value: nothing changed
    assert inspector.offer_section.isHidden()
    controller.set_brightness("b.mkv", 214)
    assert not inspector.offer_section.isHidden()


def test_proof_section_shows_running_then_lines(slay_window, fake_runner):
    window, controller = slay_window, slay_window.controller
    inspector = window.inspector
    name = SLAY_NAMES[0]
    inspector.proof_button.click()
    assert inspector.proof_status.text() == f"running on {PROOF_WINDOW_TEXT}…"
    assert not inspector.proof_status.isHidden()
    submission = fake_runner.last("proof", name)
    lines = [(578.0, 580.0, "你竟掌握了鲲鹏道法"), (581.0, 583.0, "我早已不是当年的我"),
             (584.0, 586.0, "今日便让你见识见识"), (588.0, 590.0, "纵使千难万险")]
    fake_runner.finish(submission, ProofResult(name, (578.0, 608.0), lines, 4.1))
    controller.drain_events()
    assert inspector.proof_status.isHidden()
    assert inspector.proof_texts() == [
        "09:38 你竟掌握了鲲鹏道法", "09:41 我早已不是当年的我", "09:44 今日便让你见识见识", "09:48 纵使千难万险"]
    assert inspector.proof_note.text() == "4 lines · took 4.1 s"


def test_t_is_not_offered_before_the_file_has_been_scanned(make_window, tmp_project, fake_runner):
    """proof_window refuses a file whose duration is unknown, so T is gated
    the way it is for a proof already running rather than offered and
    refused."""
    window = make_window()
    window.open_folder(str(tmp_project(["new.mkv"])))
    activate(window)
    window.queue.setFocus()
    assert not window.proof_action.isEnabled()

    QTest.keyClick(window.queue, Qt.Key.Key_T)
    assert fake_runner.of_kind("proof") == []
    assert window.inspector.proof_note.text() == ""

    metadata = fake_runner.last("metadata", "new.mkv")
    fake_runner.finish(metadata, MetadataResult("new.mkv", 1920, 1080, 1400.0, 23.976))
    window.controller.drain_events()
    settle()
    assert window.proof_action.isEnabled()


# --------------------------------------------------------------------------
# Activity strip
# --------------------------------------------------------------------------

def test_activity_strip_follows_running_events(slay_window, fake_runner):
    window, controller = slay_window, slay_window.controller
    strip = window.activity_strip
    assert strip.text_label.text() == "idle"
    assert strip.progress.isHidden()

    name = SLAY_NAMES[4]
    audio = fake_runner.last("audio_profile", name)
    fake_runner.start(audio)
    controller.drain_events()
    assert strip.dot.property("tone") == "run"
    assert strip.text_label.text() == f"{name} · audio analysis"
    assert not strip.progress.isHidden() and strip.progress.is_indeterminate()
    assert strip.percent_label.isHidden()

    fake_runner.progress(audio, 0.62, "reading audio")
    controller.drain_events()
    assert strip.text_label.text() == f"{name} · reading audio"
    assert not strip.progress.is_indeterminate()
    assert strip.percent_label.text() == "62%"

    fake_runner.finish(audio, None)
    controller.drain_events()
    assert strip.text_label.text() == "idle"
    assert strip.dot.property("tone") == "idle"
    assert strip.recent_label.text() == "· audio analysis done 0 s ago"


def test_pause_autopilot_toggles(slay_window):
    strip, controller = slay_window.activity_strip, slay_window.controller
    assert strip.pause_button.text() == "pause auto-pilot"
    strip.pause_button.click()
    assert controller.activity().paused
    assert strip.pause_button.text() == "resume auto-pilot"
    strip.pause_button.click()
    assert not controller.activity().paused
    assert strip.pause_button.text() == "pause auto-pilot"


# --------------------------------------------------------------------------
# Opening folders, settings, dependencies
# --------------------------------------------------------------------------

def test_empty_state_before_a_folder_is_open(make_window):
    window = make_window()
    assert window.windowTitle() == "OCR Manager"
    assert window.centre.currentWidget() is window.open_view
    assert window.open_view.title_label.text() == "Open a folder of episodes"
    assert window.open_view.choose_button.text() == "Choose folder…"


def test_drag_and_drop_of_a_directory_opens_it(make_window, tmp_project, tmp_path):
    window = make_window()
    folder = tmp_project(fixture="slay")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(folder))])
    enter = QDragEnterEvent(QPoint(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
    window.dragEnterEvent(enter)
    assert enter.isAccepted()
    drop = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                      Qt.KeyboardModifier.NoModifier)
    window.dropEvent(drop)
    assert window.controller.project.path == str(folder)
    assert window.centre.currentWidget() is not window.open_view

    not_a_folder = QMimeData()
    not_a_folder.setUrls([QUrl.fromLocalFile(str(folder / SLAY_NAMES[0]))])
    enter = QDragEnterEvent(QPoint(10, 10), Qt.DropAction.CopyAction, not_a_folder, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
    enter.ignore()
    window.dragEnterEvent(enter)
    assert not enter.isAccepted()


def test_choose_folder_uses_the_qt_picker_without_kdialog(make_window, tmp_project, monkeypatch):
    folder = tmp_project(fixture="slay")
    asked = []
    monkeypatch.setattr(open_folder_module.shutil, "which", lambda tool: None)
    monkeypatch.setattr(open_folder_module.QFileDialog, "getExistingDirectory",
                        lambda parent, caption, start: asked.append(start) or str(folder))
    window = make_window()
    window.open_view.choose_button.click()
    assert asked == [str(Path.home())]                     # default: the user's home directory
    assert wait_for(lambda: window.controller.project is not None)
    assert window.controller.project.path == str(folder)


def test_choose_folder_runs_kdialog_from_the_last_path(make_window, tmp_project, tmp_path, monkeypatch):
    folder = tmp_project(fixture="slay")
    last = tmp_project(["x.mkv"])
    app_settings().setValue("project/last_path", str(last))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "kdialog-args"
    script = bin_dir / "kdialog"
    script.write_text(f'#!/bin/sh\necho "$@" > "{record}"\necho "{folder}"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    window = make_window()
    window.choose_folder()
    assert wait_for(lambda: window.controller.project is not None)
    assert window.controller.project.path == str(folder)
    assert record.read_text().strip() == f"--getexistingdirectory {last}"


def test_geometry_and_last_path_are_persisted(controller, make_window, tmp_project, monkeypatch):
    folder = tmp_project(fixture="slay")
    window = make_window()
    window.resize(1111, 640)
    window.open_folder(str(folder))
    window.close()
    settings = app_settings()
    assert settings.value("project/last_path") == str(folder)
    saved_geometry = settings.value("window/geometry")
    assert saved_geometry is not None

    restored = []
    original = MainWindow.restoreGeometry
    monkeypatch.setattr(MainWindow, "restoreGeometry",
                        lambda self, geometry: restored.append(bytes(geometry)) or original(self, geometry))
    again = MainWindow(ProjectController(type(controller._runner)(), save_debounce_ms=10))
    try:
        assert restored == [bytes(saved_geometry)]
        # The offscreen screen is 800 px wide, so only the height comes back unclamped.
        assert again.height() == 640
    finally:
        again.close()


def test_unsupported_version_is_reported_without_closing_the_open_folder(make_window, tmp_project):
    good = tmp_project(fixture="slay")
    bad = tmp_project(["ep01.mkv"], config={"version": 99, "files": {}})
    window = make_window()

    window.open_folder(str(bad))                            # nothing open: inline in the empty state
    assert window.centre.currentWidget() is window.open_view
    assert window.open_view.error_label.text() == NEWER_VERSION_TEXT
    assert not window.open_view.error_label.isHidden()

    window.open_folder(str(good))
    assert window.open_view.error_label.isHidden()
    window.open_folder(str(bad))                            # a folder open: a banner, the folder stays
    assert window.controller.project.path == str(good)
    assert window.error_banner.text() == NEWER_VERSION_TEXT
    assert not window.error_banner.isHidden()
    assert window.windowTitle().endswith(os.path.basename(good))


def test_a_failed_save_is_reported_in_a_banner(slay_window):
    slay_window.controller.save_failed.emit("Could not save /x/.ocr.json: disk full")
    assert not slay_window.error_banner.isHidden()
    assert slay_window.error_banner.text() == "Could not save /x/.ocr.json: disk full"


def test_the_not_saved_banner_goes_when_the_next_save_works(slay_window):
    """Only reopening or the ✕ cleared it, so a folder that saved fine a
    second later still read "Not saved"."""
    window = slay_window
    window.controller.save_failed.emit("Could not save /x/.ocr.json: disk full")
    assert not window.error_banner.isHidden()

    window.controller.project_saved.emit()
    assert window.error_banner.isHidden()


def test_a_successful_save_leaves_an_open_failure_on_screen(slay_window, tmp_project):
    window = slay_window
    window.open_folder(str(tmp_project(["ep01.mkv"], config={"version": 99, "files": {}})))
    assert window.error_banner.title() == OPEN_FAILED_TITLE

    window.controller.project_saved.emit()
    assert not window.error_banner.isHidden()


def test_closing_warns_again_while_saving_is_blocked(make_window, tmp_project, monkeypatch):
    """_save_blocked latches, so every later save -- close_folder's and
    shutdown's -- returned silently and the session's work went with it."""
    from PyQt6.QtGui import QCloseEvent
    from PyQt6.QtWidgets import QMessageBox

    from app.main_window import BLOCKED_SAVE_TITLE
    from core.project import store as store_module

    window = make_window()
    window.open_folder(str(tmp_project(fixture="slay")))
    settle()

    def refuse(project):
        raise UnsupportedProjectVersion(project.path, 99)

    monkeypatch.setattr(store_module, "save_project", refuse)
    window.controller.set_brightness(SLAY_NAMES[0], 150)
    assert wait_for(lambda: not window.error_banner.isHidden())
    assert window.controller.save_blocked

    asked = []

    def question(parent, title, text, buttons, default):
        asked.append(title)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    event = QCloseEvent()
    window.closeEvent(event)
    assert asked == [BLOCKED_SAVE_TITLE]
    assert not event.isAccepted()                       # Cancel: the window stays, the values stay

    asked.clear()
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *args: QMessageBox.StandardButton.Yes))
    window.closeEvent(QCloseEvent())
    assert window.error_banner.title() == SAVE_FAILED_TITLE
    assert "without saving" in window.error_banner.text()


def test_files_appearing_and_disappearing_update_the_queue(qapp, fake_runner, tmp_project):
    folder = tmp_project(fixture="slay")
    controller = ProjectController(fake_runner, save_debounce_ms=10, watch_debounce_ms=10)
    window = MainWindow(controller)
    try:
        window.open_folder(str(folder))
        window.queue.select(SLAY_NAMES[2])
        (folder / "ZS2_-_16_[1080p]TXHBR.mp4").write_bytes(b"placeholder video")
        assert wait_for(lambda: "ZS2_-_16_[1080p]TXHBR.mp4" in window.queue.names())
        assert window.queue.filter.labels()[0] == "All 6"
        (folder / SLAY_NAMES[2]).unlink()
        assert wait_for(lambda: SLAY_NAMES[2] not in window.queue.names())
        assert window.queue.selected() == SLAY_NAMES[0]
        assert window.inspector.file_label.full_text() == SLAY_NAMES[0]
    finally:
        window.close()


def test_dependency_banner_when_pyav_is_missing(make_window, monkeypatch):
    import app.main_window as main_window_module
    import videocr.pyav_adapter as pyav

    monkeypatch.setattr(main_window_module.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(pyav, "PYAV_AVAILABLE", False)
    monkeypatch.setattr(pyav, "PYAV_IMPORT_ERROR", "No module named 'av'")
    window = make_window()
    banner = window.dependency_banner
    assert not banner.isHidden()
    assert banner.title() == "Video backend degraded"
    assert "PyAV is not available" in banner.text()
    assert ".venv/bin/pip install -U --only-binary=:all: av" in banner.text()
    assert "Import error: No module named 'av'" in banner.text()


def test_dependency_banner_when_ffmpeg_is_missing(make_window, monkeypatch):
    import app.main_window as main_window_module
    import videocr.pyav_adapter as pyav

    monkeypatch.setattr(main_window_module.shutil, "which", lambda tool: None)
    monkeypatch.setattr(pyav, "PYAV_AVAILABLE", True)
    window = make_window()
    assert window.dependency_banner.title() == "Missing Dependencies"
    assert "Required tools not found in PATH:\nffmpeg" in window.dependency_banner.text()


def test_dependency_banner_names_a_missing_ffprobe(make_window, monkeypatch):
    import app.main_window as main_window_module
    import videocr.pyav_adapter as pyav

    monkeypatch.setattr(main_window_module.shutil, "which", lambda tool: None if tool == "ffprobe" else tool)
    monkeypatch.setattr(pyav, "PYAV_AVAILABLE", True)
    window = make_window()
    assert window.dependency_banner.title() == "Missing Dependencies"
    assert "Required tools not found in PATH:\nffprobe" in window.dependency_banner.text()


def test_dependency_banner_in_a_bundle_without_an_ocr_engine(make_window, monkeypatch, tmp_path):
    import app.main_window as main_window_module
    import videocr.pyav_adapter as pyav

    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "bundle.json").write_text('{"version": "1.0.0", "os": "linux", "arch": "x86_64"}')
    monkeypatch.setenv("OCR_MANAGER_BUNDLE", str(tmp_path / "bundle"))
    monkeypatch.setenv("OCR_MANAGER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(main_window_module.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(pyav, "PYAV_AVAILABLE", True)
    window = make_window()
    assert window.dependency_banner.title() == "OCR engine missing"
    assert "--setup-engine" in window.dependency_banner.text()


def test_no_dependency_banner_when_everything_is_there(make_window, monkeypatch):
    import app.main_window as main_window_module
    import videocr.pyav_adapter as pyav

    monkeypatch.setattr(main_window_module.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(pyav, "PYAV_AVAILABLE", True)
    assert make_window().dependency_banner.isHidden()


def test_ctrl_q_closes_and_close_shuts_the_controller_down(controller, fake_runner, make_window):
    window = make_window()
    window.show()
    assert window.quit_action.shortcut().toString() == "Ctrl+Q"
    window.quit_action.trigger()
    assert not window.isVisible()
    assert fake_runner.shutdown_calls


# --------------------------------------------------------------------------
# python -m app
# --------------------------------------------------------------------------

def test_python_m_app_smoke(tmp_path):
    """A real launch in its own process: its theme, settings and hook never
    touch this test session."""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", XDG_CONFIG_HOME=str(tmp_path / "config"))
    result = subprocess.run([sys.executable, "-m", "app", "--quit-after", "1"], cwd=str(REPO_ROOT), env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert (tmp_path / "config" / "OCRManager" / "OCRTool.conf").exists()     # geometry saved there


def test_main_restores_the_excepthook(qapp, monkeypatch):
    import app.__main__ as entry

    themed = []
    monkeypatch.setattr(entry, "apply_theme", themed.append)   # keep the session's app unthemed
    hook = sys.excepthook
    assert entry.main(["--quit-after", "0.2"]) == 0
    assert themed == [qapp]
    assert sys.excepthook is hook


# --------------------------------------------------------------------------
# Follow-up: window-level shortcuts, crash guard, pending review button
# --------------------------------------------------------------------------

def test_space_and_t_act_from_the_queue_and_stage_but_step_aside_for_key_consumers(make_window, tmp_project,
                                                                                   fake_runner):
    window = make_window(tabs_factory=lambda controller: [EditorTab("Crop"), EditorTab("Brightness")])
    window.open_folder(str(tmp_project(fixture="slay")))
    controller, name = window.controller, SLAY_NAMES[0]
    marks = Calls(controller, "mark_reviewed")
    proofs = Calls(controller, "run_proof")
    redetects = Calls(controller, "redetect")
    activate(window)
    tab = window.stage.current_tab()

    tab.surface.setFocus()                                     # a stage page
    QTest.keyClick(tab.surface, Qt.Key.Key_Space)
    assert marks.calls == [(name, False)]
    QTest.keyClick(tab.surface, Qt.Key.Key_T)
    assert proofs.calls == [(name,)]
    fake_runner.finish(fake_runner.last("proof", name), None)  # T is disabled while that proof runs
    controller.drain_events()
    window.queue.setFocus()                                    # the queue
    QTest.keyClick(window.queue, Qt.Key.Key_Space)
    assert marks.calls == [(name, False), (name, True)]

    button = window.inspector.redetect_button                  # a focused button presses on Space
    button.setFocus()
    assert wait_for(lambda: QApplication.focusWidget() is button)
    assert not window.review_action.isEnabled() and not window.proof_action.isEnabled()
    QTest.keyClick(button, Qt.Key.Key_Space)
    QTest.keyClick(button, Qt.Key.Key_T)
    assert redetects.calls == [(name,)]
    assert len(marks.calls) == 2 and len(proofs.calls) == 1

    for editor in (tab.spin, tab.line):
        editor.setFocus()
        assert wait_for(lambda editor=editor: QApplication.focusWidget() is editor)
        assert not window.review_action.isEnabled() and not window.proof_action.isEnabled()
        QTest.keyClick(editor, Qt.Key.Key_Space)
        QTest.keyClick(editor, Qt.Key.Key_T)
        assert len(marks.calls) == 2 and len(proofs.calls) == 1
    assert tab.line.text() == " t"                             # the keys went to the editor

    tab.surface.setFocus()
    assert window.review_action.isEnabled() and window.proof_action.isEnabled()
    QTest.keyClick(tab.surface, Qt.Key.Key_T)
    assert len(proofs.calls) == 2


def test_key_consumer_detection():
    from PyQt6.QtWidgets import QCheckBox, QComboBox, QListWidget, QPlainTextEdit, QSlider, QTextEdit

    from app.main_window import consumes_keys

    for widget in (QPushButton(), QCheckBox(), QLineEdit(), QTextEdit(), QPlainTextEdit(), QSpinBox(),
                   QComboBox(), QSlider(), QListWidget()):
        assert consumes_keys(widget), type(widget).__name__
    assert not consumes_keys(QWidget()) and not consumes_keys(None)


def test_an_exception_in_a_slot_is_reported_instead_of_aborting(slay_window, capsys):
    from app.__main__ import install_excepthook

    window = slay_window
    assert window.crash_banner.isHidden()

    def boom():
        raise RuntimeError("boom in a slot")

    previous = sys.excepthook
    replaced = install_excepthook(window)
    try:
        assert replaced is previous
        QTimer.singleShot(0, boom)
        QTest.qWait(30)
    finally:
        sys.excepthook = previous
    assert not window.crash_banner.isHidden()
    assert window.crash_banner.title() == "Something went wrong — details are in Logs (Pipeline)."
    log = window.controller.log_text("Pipeline")
    assert "Traceback (most recent call last)" in log and "RuntimeError: boom in a slot" in log
    assert "RuntimeError: boom in a slot" in capsys.readouterr().err
    window.crash_banner.dismiss_button.click()
    assert window.crash_banner.isHidden()


def test_the_review_tooltip_names_a_missing_value_rather_than_a_wait(make_window, tmp_project,
                                                                     fake_runner):
    """"Mark reviewed" is refused for two different reasons -- detections are
    still running, or a value the file needs is missing -- and only the first
    is a wait. A FLAGGED file with no crop was being told "waiting for
    detections to finish" while nothing was coming."""
    folder = write_project(tmp_project(["a.mkv"]), [entry("a.mkv", crop=None)])
    window = make_window()
    window.open_folder(str(folder))
    settle()
    controller, button = window.controller, window.inspector.review_button
    assert controller.entry("a.mkv").review == ReviewState.PENDING
    assert button.toolTip() == "waiting for detections to finish"        # ... and here it IS a wait

    crop = fake_runner.last("crop", "a.mkv")                 # the detector found no box
    fake_runner.finish(crop, crop_result(crop, box=None, flagged="static-content"))
    controller.drain_events()
    settle()
    assert controller.entry("a.mkv").review == ReviewState.FLAGGED
    assert controller.entry("a.mkv").crop is None
    assert not button.isEnabled()
    assert button.toolTip() == "set a crop first"            # nothing is coming; say so

    controller.set_crop("a.mkv", BOX)
    settle()
    assert button.isEnabled() and button.toolTip() == ""


def test_mark_reviewed_is_disabled_while_the_selected_file_is_pending(make_window, tmp_project, fake_runner):
    names = ["a.mkv", "b.mkv"]
    folder = write_project(tmp_project(names), [entry("a.mkv", crop=None, brightness=None, review=ReviewState.PENDING),
                                                entry("b.mkv")])
    window = make_window()
    window.open_folder(str(folder))
    controller, inspector, queue = window.controller, window.inspector, window.queue
    marks = Calls(controller, "mark_reviewed")
    activate(window)
    queue.setFocus()
    assert queue.selected() == "a.mkv" and controller.entry("a.mkv").review == ReviewState.PENDING

    assert not inspector.review_button.isEnabled()
    assert inspector.review_button.toolTip() == "waiting for detections to finish"
    assert not window.review_action.isEnabled()
    assert not menu_actions(queue.context_menu("a.mkv"))["Mark reviewed"].isEnabled()
    QTest.keyClick(queue, Qt.Key.Key_Space)
    assert marks.calls == []

    crop = fake_runner.last("crop", "a.mkv")
    fake_runner.finish(crop, crop_result(crop))
    controller.drain_events()
    brightness = fake_runner.last("brightness", "a.mkv")
    fake_runner.finish(brightness, brightness_result(brightness))
    controller.drain_events()
    assert controller.entry("a.mkv").review == ReviewState.PROPOSED
    assert inspector.review_button.isEnabled() and inspector.review_button.toolTip() == ""
    assert window.review_action.isEnabled()
    assert menu_actions(queue.context_menu("a.mkv"))["Mark reviewed"].isEnabled()
    QTest.keyClick(queue, Qt.Key.Key_Space)
    assert marks.calls == [("a.mkv", True)]


# --------------------------------------------------------------------------
# Follow-up 2: detection dot, view imports, refresh coalescing, footer hint,
# kdialog failing to start, long names in the activity strip
# --------------------------------------------------------------------------

def test_detecting_dot_ignores_metadata_and_thumbnail_jobs(slay_window, fake_runner):
    window, controller = slay_window, slay_window.controller
    thumbnail = fake_runner.last("thumbnail", SLAY_NAMES[0])
    fake_runner.start(thumbnail)
    controller.drain_events()
    settle()
    assert controller.activity().running == (("thumbnail", SLAY_NAMES[0]),)
    assert window.topbar.detecting_chip.tone() == "idle"
    audio = fake_runner.last("audio_profile", SLAY_NAMES[1])
    fake_runner.start(audio)
    controller.drain_events()
    settle()
    assert window.topbar.detecting_chip.tone() == "run"


def test_controller_names_the_detection_kinds(controller):
    assert controller.DETECTION_KINDS == frozenset({"crop", "brightness", "ranges", "audio_profile", "lines"})
    assert controller.is_detection_kind("audio_profile") and not controller.is_detection_kind("thumbnail")


def test_views_import_no_core_modules():
    import ast

    sources = sorted((REPO_ROOT / "app" / "views").glob("*.py")) + [REPO_ROOT / "app" / "main_window.py"]
    offenders = []
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            for module in modules:
                if module == "videocr" or module.startswith("videocr.pyav_adapter"):
                    continue                                # the brief's dependency check
                if module.split(".")[0] in ("core", "videocr"):
                    offenders.append(f"{path.name}: {module}")
    assert offenders == []


def test_a_burst_of_file_changes_refreshes_counts_once_per_view(slay_window):
    controller = slay_window.controller
    settle()
    counts = Calls(controller, "counts")
    startable = Calls(controller, "startable_files")
    for _ in range(20):
        for name in SLAY_NAMES:
            controller.file_changed.emit(name)
    assert counts.calls == [] and startable.calls == []        # deferred to the event loop
    settle()
    assert len(counts.calls) == 2                               # the queue filter and the top bar, once each
    assert len(startable.calls) == 1                            # the top bar
    settle()
    assert len(counts.calls) == 2


def test_queue_hint_wraps_instead_of_clipping(slay_window):
    from PyQt6.QtGui import QTextDocument

    hint = slay_window.queue.hint_label
    assert hint.wordWrap()
    document = QTextDocument()
    document.setHtml(hint.text())
    assert document.toPlainText().replace("\xa0", " ") == "↑ ↓ move · Space mark reviewed · T test OCR"
    hint.parentWidget().show()
    assert wait_for(lambda: hint.height() >= hint.heightForWidth(hint.width()) > 0)
    hint.parentWidget().hide()


def test_kdialog_that_fails_to_start_falls_back_to_the_qt_picker(make_window, tmp_project, tmp_path, monkeypatch):
    folder = tmp_project(fixture="slay")
    asked = []
    monkeypatch.setattr(open_folder_module.shutil, "which", lambda tool: str(tmp_path / "no-such-kdialog"))
    monkeypatch.setattr(open_folder_module.QFileDialog, "getExistingDirectory",
                        lambda parent, caption, start: asked.append(start) or str(folder))
    window = make_window()
    window.choose_folder()
    assert wait_for(lambda: window.controller.project is not None)
    assert window.controller.project.path == str(folder)
    assert len(asked) == 1
    window.controller.close_folder()
    window.choose_folder()                                      # the failed process does not block later picks
    assert wait_for(lambda: len(asked) == 2)


def test_long_file_names_do_not_widen_the_activity_strip(make_window, tmp_project, fake_runner):
    name = "E" * 100 + ".mkv"
    window = make_window()
    window.open_folder(str(tmp_project([name])))
    submission = fake_runner.last("metadata", name)
    fake_runner.start(submission)
    window.controller.drain_events()
    settle()
    strip = window.activity_strip
    assert strip.text_label.full_text() == f"{name} · metadata"
    assert strip.minimumSizeHint().width() < 400
    fake_runner.finish(submission, None)
    window.controller.drain_events()
    assert strip.minimumSizeHint().width() < 400
