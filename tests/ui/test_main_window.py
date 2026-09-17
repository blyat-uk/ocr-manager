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

import os
import stat
import time
from pathlib import Path

import pytest
from PyQt6.QtCore import QMimeData, QPoint, QPointF, QSettings, Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QPushButton

from app.controller import ProjectController
from app.main_window import MainWindow
from app.views import open_folder as open_folder_module
from app.views.stage import Stage, StageTab, placeholder_tabs
from core.detect.brightness import BrightnessResult
from core.detect.crop import FLAG_LOW_AGREEMENT, CropResult
from core.jobs.detect_jobs import BrightnessJobResult, CropJobResult, ProofResult
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
        window.close()
        window.deleteLater()


@pytest.fixture
def slay_window(make_window, tmp_project):
    window = make_window()
    window.open_folder(str(tmp_project(fixture="slay")))
    return window


@pytest.fixture
def detected_window(make_window, tmp_project):
    """Five reviewed files whose crop and brightness were detected (so a new
    detection result may flag them)."""
    folder = write_project(tmp_project(SLAY_NAMES), [entry(name) for name in SLAY_NAMES], labels_enabled=False)
    window = make_window()
    window.open_folder(str(folder))
    return window


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
    assert window.queue.minimumWidth() == window.queue.maximumWidth() == 246
    assert window.inspector.minimumWidth() == window.inspector.maximumWidth() == 322
    assert window.queue.filter.labels() == ["All 5", "Needs you 0", "Reviewed 5"]
    assert window.queue.visible_names() == SLAY_NAMES
    assert window.queue.selected() == SLAY_NAMES[0]
    row = window.queue.row(SLAY_NAMES[0])
    assert (row.badge.text(), row.badge.property("badge")) == ("reviewed", "good")
    assert row.duration_label.text() == "23:38"


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
    assert top.detecting_chip.tone() == "run"              # B7: a detection is running
    fake_runner.finish(crop, crop_result(crop, flagged=FLAG_LOW_AGREEMENT))
    controller.drain_events()
    assert row.badge.text() == "waiting"                   # brightness follows the re-detect, queued
    assert top.detecting_chip.text() == "1 detecting"
    assert top.detecting_chip.tone() == "idle"             # B7: only queued

    brightness = fake_runner.last("brightness", name)
    fake_runner.start(brightness)                           # "started" emits only activity_changed
    controller.drain_events()
    assert row.badge.text() == "measuring brightness…"
    assert top.detecting_chip.tone() == "run"

    fake_runner.finish(brightness, brightness_result(brightness))
    controller.drain_events()
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
    controller, start = window.controller, window.topbar.start_button
    assert start.text() == "▶ Start 2 ready files" and start.isEnabled()

    controller.project.folder.dialogue_enabled = False
    controller.project.folder.labels_enabled = False
    controller.folder_changed.emit()
    assert not start.isEnabled()

    controller.project.folder.dialogue_enabled = True
    controller.folder_changed.emit()
    assert start.isEnabled()
    controller.set_skipped("a.mkv", True)
    controller.set_skipped("b.mkv", True)
    assert start.text() == "▶ Start 0 ready files"
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
    assert queue.filter.labels() == ["All 5", "Needs you 0", "Reviewed 3"]
    assert queue.visible_names() == ["a.mkv", "b.mkv", "d.mkv"]

    segments[0].click()
    assert queue.visible_names() == ["a.mkv", "b.mkv", "c.mkv", "d.mkv", "e.mkv"]


def test_queue_keyboard_moves_marks_and_proves(slay_window, fake_runner):
    window, controller = slay_window, slay_window.controller
    queue = window.queue
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
    assert window.inspector.proof_status.text() == "running…"


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
    # 1920x888 fitted into 56x32 -> 56x25.9 at y 3.05; the crop (288, 786, 1344, 53) sits at its true place
    assert box.x() == pytest.approx(288 / 1920 * 56, abs=0.01)
    assert box.width() == pytest.approx(1344 / 1920 * 56, abs=0.01)
    assert box.y() == pytest.approx((32 - 888 * 56 / 1920) / 2 + 786 * 56 / 1920, abs=0.01)

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
    assert inspector.hint_button.text() == "↻ re-detect the other 4 using this as a hint"
    assert inspector.offer_note.text() == (
        "Corrections are never copied verbatim to other episodes. Instead the app offers:")
    inspector.hint_button.click()
    assert hints.calls == [(name, "crop")]
    assert inspector.offer_section.isHidden()

    controller.set_brightness(name, 215)
    assert not inspector.offer_section.isHidden()
    inspector.this_file_only_button.click()
    assert inspector.offer_section.isHidden()
    window.queue.select(SLAY_NAMES[1])
    window.queue.select(name)
    assert inspector.offer_section.isHidden()                          # dismissed stays dismissed


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
    assert inspector.proof_status.text() == "running…"
    assert not inspector.proof_status.isHidden()
    submission = fake_runner.last("proof", name)
    lines = [(578.0, 580.0, "你竟掌握了鲲鹏道法"), (581.0, 583.0, "我早已不是当年的我"),
             (584.0, 586.0, "今日便让你见识见识"), (588.0, 590.0, "纵使千难万险")]
    fake_runner.finish(submission, ProofResult(name, (578.0, 608.0), lines, 4.1))
    controller.drain_events()
    assert inspector.proof_status.isHidden()
    assert inspector.proof_texts() == [
        "09:38 你竟掌握了鲲鹏道法", "09:41 我早已不是当年的我", "09:44 今日便让你见识见识"]
    assert inspector.proof_note.text() == "4 lines · took 4.1 s"


def test_proof_with_unknown_duration_says_so_instead_of_raising(make_window, tmp_project):
    window = make_window()
    window.open_folder(str(tmp_project(["new.mkv"])))
    QTest.keyClick(window.queue, Qt.Key.Key_T)
    assert "duration unknown" in window.inspector.proof_note.text()


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
    assert "unsupported project version 99" in window.open_view.error_label.text()
    assert not window.open_view.error_label.isHidden()

    window.open_folder(str(good))
    assert window.open_view.error_label.isHidden()
    window.open_folder(str(bad))                            # a folder open: a banner, the folder stays
    assert window.controller.project.path == str(good)
    assert "unsupported project version 99" in window.error_banner.text()
    assert not window.error_banner.isHidden()
    assert window.windowTitle().endswith(os.path.basename(good))


def test_a_failed_save_is_reported_in_a_banner(slay_window):
    slay_window.controller.save_failed.emit("Could not save /x/.ocr.json: disk full")
    assert not slay_window.error_banner.isHidden()
    assert slay_window.error_banner.text() == "Could not save /x/.ocr.json: disk full"


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

def test_python_m_app_smoke(qapp, tmp_project):
    from app.__main__ import main

    assert main(["--quit-after", "1"]) == 0
