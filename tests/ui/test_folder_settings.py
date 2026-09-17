"""Plan 3B Task 4: the Folder settings sheet (rulings B8/B9, ui-spec §1.2/§3.9).

Every test drives a real ProjectController over `fake_runner` inside a real
MainWindow, so the sheet is exercised where it lives: opened from the top
bar, anchored between the top bar and the activity strip, committing through
`controller.update_folder`.

QSettings is pointed at a per-test directory (autouse `settings_dir`) before
any window is built.
"""
from __future__ import annotations

import dataclasses
import time

import pytest
from PyQt6.QtCore import QPoint, QPointF, QSettings, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QAbstractButton, QAbstractSpinBox, QApplication, QComboBox, QDoubleSpinBox, QSpinBox

from app.controller import ProjectController
from app.main_window import MainWindow
from app.views.folder_settings import FolderSettingsSheet
from app.widgets.base import Toggle
from core.project import Brightness, Crop, FileEntry, FolderSettings, Media, Project, ReviewState, Source, save_project

NAMES = ["ep01.mkv", "ep02.mkv", "ep03.mkv", "ep04.mkv", "ep05.mkv"]
WAIT_MS = 5000
BOTH_OFF = "At least one of dialogue or labels must be on."
SECTIONS = ["What to extract", "OCR engine", "Labels", "Performance", "Auto-pilot"]
ROW_KEYS = {
    "What to extract": ["Dialogue subtitles", "Positioned labels / nameplates"],
    "OCR engine": ["Language", "Confidence threshold", "Merge similar lines above", "Similar-frame threshold"],
    # No "Confidence threshold": `label_conf_threshold` has no editor, see
    # NOT_IN_THE_SHEET and app/views/folder_settings_fields.py.
    "Labels": ["Minimum duration", "Maximum duration", "Minimum confidence", "Mask regions"],
    "Performance": ["Parallel files"],
    "Auto-pilot": ["Run detections when a folder opens", "Full brightness detection on the first",
                   "Minimum repeating segment", "Merge repeating silences", "Crop width (% of frame)",
                   "Crop vertical padding (% of frame height)", "Crop minimum height (% of frame height)",
                   "Subtitle band starts at (% from top)"],
}
# FolderSettings fields with no editor: B9 does not list frames_to_skip / use_gpu
# (the old app never exposed them either), label_mask_crops is the read-only mask
# count, and label_conf_threshold changes nothing -- videocr/label_scanner.py
# stores it (line 153) but filters only on conf_threshold_min (lines 1474, 1586).
NOT_IN_THE_SHEET = {"frames_to_skip", "use_gpu", "label_mask_crops", "label_conf_threshold"}


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


def reviewed(name: str) -> FileEntry:
    return FileEntry(name, crop=Crop(288, 786, 1344, 53, Source.DETECTED), brightness=Brightness(209, Source.DETECTED),
                     media=Media(1920, 1080, 1400.0, 25.0), review=ReviewState.REVIEWED, sample_time=500.0)


def write_project(folder, names, **folder_settings):
    save_project(Project(path=str(folder), folder=FolderSettings(**folder_settings),
                         files={name: reviewed(name) for name in names}))
    return folder


def activate(window: MainWindow) -> None:
    window.show()
    window.activateWindow()
    assert wait_for(lambda: QApplication.activeWindow() is window)


def focus(widget) -> None:
    widget.setFocus(Qt.FocusReason.OtherFocusReason)
    assert wait_for(lambda: QApplication.focusWidget() is widget)


class Recorder:
    """Wraps a controller command: records each call's (args, kwargs), then
    runs the real command."""

    def __init__(self, controller, name: str):
        self.calls: list[tuple[tuple, dict]] = []
        original = getattr(controller, name)

        def wrapper(*args, **kwargs):
            self.calls.append((args, kwargs))
            return original(*args, **kwargs)

        setattr(controller, name, wrapper)

    def kwargs(self) -> list[dict]:
        return [kwargs for _args, kwargs in self.calls]


@pytest.fixture(autouse=True)
def settings_dir(tmp_path):
    path = tmp_path / "qsettings"
    path.mkdir()
    for fmt in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, str(path))
    return path


@pytest.fixture
def controller(qapp, fake_runner):
    made = ProjectController(fake_runner, save_debounce_ms=10, watch_debounce_ms=10)
    yield made
    made.shutdown(timeout=0.5)


@pytest.fixture
def make_window(controller):
    windows = []

    def make(folder=None) -> MainWindow:
        window = MainWindow(controller)
        window.resize(1440, 900)
        windows.append(window)
        if folder is not None:
            window.open_folder(str(folder))
            settle()
        return window

    yield make
    for window in windows:
        # The teardown is not a user: it never answers closeEvent's questions
        # (a run in progress, settings that could not be saved).
        window._closing = True
        window.close()
        window.deleteLater()


@pytest.fixture
def folder(tmp_project):
    return write_project(tmp_project(NAMES), NAMES)


@pytest.fixture
def window(make_window, folder):
    window = make_window(folder)
    window.show()
    return window


def open_sheet(window: MainWindow) -> FolderSettingsSheet:
    window.topbar.settings_button.click()
    sheet = window.folder_settings
    assert wait_for(lambda: sheet.isVisible() and sheet.geometry() == sheet.target_geometry())
    return sheet


def enter(sheet: FolderSettingsSheet, field: str, value) -> None:
    """Change one editor the way a user finishes an edit."""
    editor = sheet.editor(field)
    if isinstance(editor, QAbstractButton):
        if editor.isChecked() != value:
            editor.click()
    elif isinstance(editor, QComboBox):
        editor.setEditText(value)
        editor.lineEdit().textEdited.emit(value)
        editor.lineEdit().editingFinished.emit()
    else:
        editor.setValue(value)
        editor.editingFinished.emit()


def shown(sheet: FolderSettingsSheet, field: str):
    editor = sheet.editor(field)
    if isinstance(editor, QAbstractButton):
        return editor.isChecked()
    if isinstance(editor, QComboBox):
        return editor.currentText()
    return editor.value()


# --------------------------------------------------------------------------
# Presentation (B8)
# --------------------------------------------------------------------------

def test_sheet_opens_from_the_top_bar_between_the_top_bar_and_the_activity_strip(window, folder):
    sheet = window.folder_settings
    assert not sheet.isVisible()
    sheet = open_sheet(window)
    central = window.centralWidget()
    assert sheet.parentWidget() is central
    assert sheet.width() == 760                                   # min(760, 1440 - 246)
    assert sheet.geometry().right() == central.width() - 1
    assert sheet.y() == window.centre.y()
    assert sheet.geometry().bottom() == window.activity_strip.geometry().top() - 1
    assert sheet.x() >= window.queue.mapTo(central, QPoint(window.queue.width(), 0)).x()   # queue stays visible
    assert sheet.title_label.text() == "⚙ Folder settings"
    assert sheet.scope_label.full_text() == f"applies to all 5 files in {folder.name}"
    assert sheet.close_button.text() == "✕"
    assert sheet.nav.labels() == SECTIONS
    assert sheet.nav_panel.width() == 150


def test_resizing_the_window_recomputes_the_sheet(window):
    sheet = open_sheet(window)
    central = window.centralWidget()
    window.resize(900, 700)
    assert wait_for(lambda: sheet.width() == 900 - 246 and sheet.geometry().right() == central.width() - 1)
    assert sheet.geometry().bottom() == window.activity_strip.geometry().top() - 1
    window.resize(1200, 800)
    assert wait_for(lambda: sheet.width() == 760 and sheet.geometry().right() == central.width() - 1)
    assert sheet.geometry() == sheet.target_geometry()


def test_a_banner_pushes_the_sheet_down_instead_of_covering_it(window):
    sheet = open_sheet(window)
    window.error_banner.show_message("Not saved", "disk full", "bad")
    assert wait_for(lambda: sheet.y() == window.centre.y() and window.error_banner.isVisible())
    assert sheet.y() > window.error_banner.y()


def test_escape_closes_only_while_focus_is_inside_the_sheet(window):
    activate(window)
    sheet = open_sheet(window)
    closed = []
    sheet.closed.connect(lambda: closed.append(True))
    assert sheet.isAncestorOf(QApplication.focusWidget())         # opening moves focus into the sheet

    focus(window.queue)
    QTest.keyClick(window.queue, Qt.Key.Key_Escape)
    assert sheet.isVisible() and closed == []

    editor = sheet.editor("conf_threshold")
    focus(editor)
    QTest.keyClick(editor, Qt.Key.Key_Escape)
    assert not sheet.isVisible() and closed == [True]
    assert wait_for(lambda: QApplication.focusWidget() is window.queue)   # focus goes back to the queue


def test_the_close_button_closes_it(window):
    sheet = open_sheet(window)
    closed = []
    sheet.closed.connect(lambda: closed.append(True))
    sheet.close_button.click()
    assert not sheet.isVisible() and closed == [True]
    sheet.close_sheet()                                           # already closed: no second signal
    assert closed == [True]
    assert open_sheet(window) is sheet


def test_closing_the_folder_closes_the_sheet(window):
    sheet = open_sheet(window)
    window.controller.close_folder()
    settle()
    assert not sheet.isVisible()


def test_nav_scrolls_to_each_section_and_follows_scrolling(window):
    sheet = open_sheet(window)
    bar = sheet.scroll.verticalScrollBar()
    for index, title in enumerate(SECTIONS):
        sheet.nav.item(index).click()
        settle()
        assert sheet.nav.current() == index
        assert sheet.section(title).visibleRegion().boundingRect().top() == 0, title   # its top is in view
    bar.setValue(0)
    settle()
    assert sheet.nav.current() == 0
    bar.setValue(sheet.section("Performance").y())
    settle()
    assert sheet.nav.current() == SECTIONS.index("Performance")


# --------------------------------------------------------------------------
# Sections, rows and copy (B9)
# --------------------------------------------------------------------------

def test_sections_rows_and_explanations_follow_b9(window):
    sheet = open_sheet(window)
    for title, keys in ROW_KEYS.items():
        assert sheet.row_keys(title) == keys, title
    for field in sheet.fields():
        note = sheet.note(field)
        assert note and "\n" not in note, field
    assert sheet.note("conf_threshold") == "retry until this confident"
    assert sheet.section_note("What to extract") == \
        "With labels off, the label mask regions and their thresholds are hidden entirely."


def test_every_folder_setting_b9_names_has_an_editor(window):
    sheet = open_sheet(window)
    expected = {field.name for field in dataclasses.fields(FolderSettings)} - NOT_IN_THE_SHEET
    assert set(sheet.fields()) == expected


def test_editor_kinds_and_ranges(window):
    sheet = open_sheet(window)
    for field in ("dialogue_enabled", "labels_enabled", "autopilot_enabled", "merge_repeating_silences"):
        assert isinstance(sheet.editor(field), Toggle), field
    for field in ("conf_threshold", "sim_threshold", "label_conf_threshold_min"):
        editor = sheet.editor(field)
        assert isinstance(editor, QSpinBox) and (editor.minimum(), editor.maximum()) == (0, 100), field
        assert editor.suffix() == " %"
    for field in ("crop_width_fraction", "crop_min_height_fraction", "bottom_half_cutoff"):
        editor = sheet.editor(field)
        assert isinstance(editor, QSpinBox) and editor.suffix() == " %" and editor.maximum() <= 100, field
    padding = sheet.editor("crop_vertical_padding")                # 0.003 needs a decimal to survive
    assert isinstance(padding, QDoubleSpinBox) and padding.decimals() == 1 and padding.suffix() == " %"
    similar = sheet.editor("similar_image")
    assert isinstance(similar, QDoubleSpinBox) and similar.decimals() == 2
    for field in ("label_min_duration", "label_max_duration", "min_segment_length"):
        editor = sheet.editor(field)
        assert isinstance(editor, QDoubleSpinBox) and editor.decimals() == 1 and editor.suffix() == " s", field
    assert (sheet.editor("ocr_parallel").minimum(), sheet.editor("ocr_parallel").maximum()) == (1, 8)
    files = sheet.editor("brightness_full_detect_files")
    assert (files.minimum(), files.maximum()) == (1, 20)
    segment = sheet.editor("min_segment_length")
    assert (segment.minimum(), segment.maximum()) == (5.0, 300.0)
    language = sheet.editor("ocr_lang")
    assert isinstance(language, QComboBox) and language.isEditable()


def test_default_values_are_shown(window):
    sheet = open_sheet(window)
    assert shown(sheet, "ocr_lang") == "Chinese (ch)"
    assert shown(sheet, "conf_threshold") == 95
    assert shown(sheet, "sim_threshold") == 82
    assert sheet.editor("similar_image").text() == "0.30 %"
    assert shown(sheet, "crop_width_fraction") == 70
    assert shown(sheet, "crop_vertical_padding") == pytest.approx(0.3)
    assert shown(sheet, "crop_min_height_fraction") == 5
    assert shown(sheet, "bottom_half_cutoff") == 55
    assert shown(sheet, "min_segment_length") == 30.0
    assert sheet.editor("brightness_full_detect_files").text() == "3 files"
    assert sheet.mask_label.text() == "0 drawn · draw on the Crop tab"


# field, what the user enters, the stored value, a value stored elsewhere, what the editor then shows
ROUND_TRIPS = [
    ("dialogue_enabled", False, False, True, True),
    ("labels_enabled", False, False, True, True),
    ("ocr_lang", "japan", "japan", "en", "en"),
    ("conf_threshold", 90, 90, 97, 97),
    ("sim_threshold", 75, 75, 88, 88),
    ("similar_image", 0.45, 0.45, 1.25, 1.25),
    ("label_min_duration", 1.5, 1.5, 0.8, 0.8),
    ("label_max_duration", 7.5, 7.5, 4.0, 4.0),
    ("label_conf_threshold_min", 70, 70, 85, 85),
    ("ocr_parallel", 6, 6, 2, 2),
    ("autopilot_enabled", False, False, True, True),
    ("brightness_full_detect_files", 5, 5, 12, 12),
    ("min_segment_length", 45.5, 45.5, 120.0, 120.0),
    ("merge_repeating_silences", True, True, False, False),
    ("crop_width_fraction", 65, 0.65, 0.8, 80),
    ("crop_vertical_padding", 0.5, 0.005, 0.003, 0.3),
    ("crop_min_height_fraction", 8, 0.08, 0.06, 6),
    ("bottom_half_cutoff", 60, 0.6, 0.45, 45),
]


def test_round_trips_cover_every_editor(window):
    sheet = open_sheet(window)
    assert {case[0] for case in ROUND_TRIPS} == set(sheet.fields())


@pytest.mark.parametrize("field, entered, stored, external, displayed", ROUND_TRIPS,
                         ids=[case[0] for case in ROUND_TRIPS])
def test_each_editor_round_trips_its_folder_setting(window, field, entered, stored, external, displayed):
    controller = window.controller
    sheet = open_sheet(window)
    updates = Recorder(controller, "update_folder")
    default = getattr(FolderSettings(), field)

    enter(sheet, field, entered)
    assert updates.kwargs() == [{field: stored}]
    value = getattr(controller.project.folder, field)
    assert value == stored and type(value) is type(default)

    controller.update_folder(**{field: external})
    settle()
    if isinstance(displayed, float):
        assert shown(sheet, field) == pytest.approx(displayed)
    else:
        assert shown(sheet, field) == displayed
    assert len(updates.calls) == 2                                # syncing the editor commits nothing


def test_language_combo_offers_chinese_plus_the_current_value(window):
    controller = window.controller
    sheet = open_sheet(window)
    combo = sheet.editor("ocr_lang")
    assert [combo.itemText(i) for i in range(combo.count())] == ["Chinese (ch)"]
    controller.update_folder(ocr_lang="en")
    settle()
    assert [combo.itemText(i) for i in range(combo.count())] == ["Chinese (ch)", "en"]
    assert combo.currentText() == "en"
    combo.activated.emit(0)
    assert controller.project.folder.ocr_lang == "ch"
    enter(sheet, "ocr_lang", "   ")                                # blank: nothing stored, the value comes back
    assert controller.project.folder.ocr_lang == "ch" and combo.currentText() == "Chinese (ch)"


# --------------------------------------------------------------------------
# Commit timing (controller ruling 1) and detections (ruling 4)
# --------------------------------------------------------------------------

def test_typing_commits_after_400_ms_idle_not_per_keystroke(window):
    activate(window)
    controller = window.controller
    sheet = open_sheet(window)
    updates = Recorder(controller, "update_folder")
    spin = sheet.editor("conf_threshold")
    focus(spin)
    spin.selectAll()
    QTest.keyClicks(spin, "85")
    typed = time.monotonic()
    assert spin.value() == 85 and updates.calls == []
    assert wait_for(lambda: updates.calls, 2000)
    assert time.monotonic() - typed >= 0.35                       # ~400 ms later (Qt timers may be 5% early)
    assert updates.kwargs() == [{"conf_threshold": 85}]           # once, with the finished number
    QTest.qWait(450)
    assert len(updates.calls) == 1


def test_editing_finished_commits_at_once(window):
    activate(window)
    controller = window.controller
    sheet = open_sheet(window)
    updates = Recorder(controller, "update_folder")
    spin = sheet.editor("sim_threshold")
    focus(spin)
    spin.selectAll()
    QTest.keyClicks(spin, "70")
    QTest.keyClick(spin, Qt.Key.Key_Return)
    assert updates.kwargs() == [{"sim_threshold": 70}]
    QTest.qWait(450)
    assert len(updates.calls) == 1                                # the idle timer was dropped
    focus(window.queue)                                           # leaving the untouched editor commits nothing
    assert len(updates.calls) == 1


def test_closing_the_sheet_commits_a_pending_edit(window):
    activate(window)
    controller = window.controller
    sheet = open_sheet(window)
    spin = sheet.editor("ocr_parallel")
    focus(spin)
    spin.selectAll()
    QTest.keyClicks(spin, "7")
    QTest.keyClick(spin, Qt.Key.Key_Escape)
    assert not sheet.isVisible()
    assert controller.project.folder.ocr_parallel == 7


def test_an_outside_change_does_not_overwrite_what_the_user_is_typing(window):
    activate(window)
    controller = window.controller
    sheet = open_sheet(window)
    spin = sheet.editor("conf_threshold")
    focus(spin)
    spin.selectAll()
    QTest.keyClicks(spin, "88")
    controller.update_folder(ocr_parallel=6)                       # e.g. the run view's "raise to 6"
    settle()
    assert shown(sheet, "ocr_parallel") == 6
    assert spin.value() == 88
    assert wait_for(lambda: controller.project.folder.conf_threshold == 88, 1000)


def test_an_out_of_range_stored_value_is_not_rewritten_by_leaving_its_editor(make_window, tmp_project):
    folder = write_project(tmp_project(NAMES), NAMES, brightness_full_detect_files=0)
    window = make_window(folder)
    activate(window)
    sheet = open_sheet(window)
    spin = sheet.editor("brightness_full_detect_files")
    focus(spin)
    focus(window.queue)
    assert window.controller.project.folder.brightness_full_detect_files == 0


def test_detector_settings_only_commit(window, fake_runner):
    controller = window.controller
    sheet = open_sheet(window)
    redetects = Recorder(controller, "redetect")
    hinted = Recorder(controller, "redetect_others_with_hint")
    submitted = len(fake_runner.submissions)
    for field, value in (("crop_width_fraction", 60), ("crop_vertical_padding", 1.0), ("bottom_half_cutoff", 50),
                         ("crop_min_height_fraction", 7), ("min_segment_length", 20.0)):
        enter(sheet, field, value)
    controller.drain_events()
    assert redetects.calls == [] and hinted.calls == []
    assert len(fake_runner.submissions) == submitted
    folder = controller.project.folder
    assert (folder.crop_width_fraction, folder.crop_vertical_padding, folder.bottom_half_cutoff,
            folder.crop_min_height_fraction, folder.min_segment_length) == (0.6, 0.01, 0.5, 0.07, 20.0)


def test_scrolling_over_an_unfocused_editor_does_not_change_it(window):
    activate(window)
    window.controller.update_folder(ocr_lang="en")                 # two languages: the wheel could move
    sheet = open_sheet(window)
    focus(sheet.close_button)
    for field in ("conf_threshold", "similar_image", "ocr_lang"):
        editor = sheet.editor(field)
        before = shown(sheet, field)
        event = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(editor, event)
        assert shown(sheet, field) == before, field


# --------------------------------------------------------------------------
# Labels section and the both-off refusal
# --------------------------------------------------------------------------

def test_labels_section_is_hidden_while_labels_are_off(window):
    controller = window.controller
    sheet = open_sheet(window)
    labels = SECTIONS.index("Labels")
    assert sheet.section("Labels").isVisible() and sheet.nav.is_item_visible(labels)

    sheet.editor("labels_enabled").click()
    assert controller.project.folder.labels_enabled is False
    assert sheet.section("Labels").isHidden() and not sheet.nav.is_item_visible(labels)

    sheet.editor("labels_enabled").click()
    assert sheet.section("Labels").isVisible() and sheet.nav.is_item_visible(labels)

    controller.update_folder(labels_enabled=False)
    settle()
    assert sheet.section("Labels").isHidden()


def test_mask_count_follows_the_folder(window):
    controller = window.controller
    sheet = open_sheet(window)
    controller.set_label_masks([(10, 20, 300, 40), (1500, 20, 200, 60)])
    settle()
    assert sheet.mask_label.text() == "2 drawn · draw on the Crop tab"


@pytest.mark.parametrize("turn_off, other", [("dialogue_enabled", "labels_enabled"),
                                             ("labels_enabled", "dialogue_enabled")])
def test_turning_off_the_last_extraction_is_refused_with_a_warning(window, turn_off, other):
    controller = window.controller
    sheet = open_sheet(window)
    enter(sheet, other, False)
    before = dataclasses.replace(controller.project.folder)
    assert sheet.warning_label.isHidden()

    sheet.editor(turn_off).click()
    assert sheet.editor(turn_off).isChecked()                     # snapped back
    assert controller.project.folder == before
    assert sheet.warning_label.isVisible() and sheet.warning_label.text() == BOTH_OFF
    settle()
    assert window.topbar.start_button.isEnabled()

    enter(sheet, other, True)                                     # a successful extraction change clears it
    assert sheet.warning_label.isHidden()


# --------------------------------------------------------------------------
# Header follows the files; keys never reach the queue shortcuts
# --------------------------------------------------------------------------

def test_file_count_follows_files_changed(window, folder):
    sheet = open_sheet(window)
    (folder / "ep06.mkv").write_bytes(b"placeholder video")
    assert wait_for(lambda: sheet.scope_label.full_text() == f"applies to all 6 files in {folder.name}")
    for name in NAMES[:4]:
        (folder / name).unlink()
    assert wait_for(lambda: sheet.scope_label.full_text() == f"applies to all 2 files in {folder.name}")
    (folder / NAMES[4]).unlink()
    assert wait_for(lambda: sheet.scope_label.full_text() == f"applies to the 1 file in {folder.name}")


def test_space_and_t_never_reach_the_queue_shortcuts_from_the_sheet(window):
    activate(window)
    controller = window.controller
    sheet = open_sheet(window)
    marks = Recorder(controller, "mark_reviewed")
    proofs = Recorder(controller, "run_proof")

    focusable = [sheet.close_button, *(sheet.nav.item(i) for i in range(len(SECTIONS)))]
    focusable += [sheet.editor(field) for field in sheet.fields()]
    for widget in focusable:
        focus(widget)
        assert not window.review_action.isEnabled() and not window.proof_action.isEnabled(), widget

    for field in ("conf_threshold", "similar_image", "ocr_lang", "merge_repeating_silences"):
        editor = sheet.editor(field)
        focus(editor)
        QTest.keyClick(editor, Qt.Key.Key_Space)
        QTest.keyClick(editor, Qt.Key.Key_T)
    assert controller.project.folder.merge_repeating_silences is True   # Space pressed the focused toggle

    sheet.setFocus(Qt.FocusReason.OtherFocusReason)               # any focus inside the sheet, even a plain area
    assert wait_for(lambda: QApplication.focusWidget() is sheet)
    assert not window.review_action.isEnabled() and not window.proof_action.isEnabled()
    QTest.keyClick(sheet, Qt.Key.Key_Space)
    QTest.keyClick(sheet, Qt.Key.Key_T)
    assert marks.calls == [] and proofs.calls == []

    sheet.close_button.click()
    assert wait_for(lambda: window.review_action.isEnabled())     # back on the queue, the shortcuts act again


def test_tab_cycles_inside_the_sheet(window):
    activate(window)
    sheet = open_sheet(window)
    chain = sheet.tab_chain()
    assert chain[0] is sheet.close_button and sheet.nav.item(0) in chain
    assert sheet.editor("bottom_half_cutoff") is chain[-1]
    focus(chain[-1])
    for key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
        for _ in range(len(chain) + 2):
            QTest.keyClick(QApplication.focusWidget(), key)
            assert sheet.contains_focus(), QApplication.focusWidget()
    focus(chain[-1])
    QTest.keyClick(chain[-1], Qt.Key.Key_Tab)
    assert QApplication.focusWidget() is chain[0]                 # wraps round to the ✕ button
    QTest.keyClick(chain[0], Qt.Key.Key_Backtab)
    assert QApplication.focusWidget() is chain[-1]


# --------------------------------------------------------------------------
# Toggle
# --------------------------------------------------------------------------

def test_toggle_is_a_checkable_key_consuming_button(qapp):
    from app.main_window import consumes_keys

    toggle = Toggle(True)
    assert isinstance(toggle, QAbstractButton) and toggle.isCheckable() and toggle.isChecked()
    assert toggle.state_text() == "on"
    toggle.click()
    assert not toggle.isChecked() and toggle.state_text() == "off"
    assert consumes_keys(toggle)
    hint = toggle.sizeHint()
    assert hint.width() > 0 and hint.height() > 0
    assert toggle.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert not isinstance(toggle, QAbstractSpinBox)
