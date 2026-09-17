"""Plan 3C Task 5: the inspector's Proof panel and the hint re-detect offers
(rulings B4, C3, C4; ui-spec §3.7).

Every test drives a real ProjectController over `fake_runner`
(tests/ui/conftest.py): no OCR runs, and the test delivers the events a real
ProofOcrJob or detection job would, then drains them with
`controller.drain_events()`. QSettings point at a per-test directory.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app.controller import ProjectController
from app.main_window import MainWindow
from app.state_text import series_median_brightness
from app.theme import tokens
from app.views.inspector_sections import (
    BUTTON_TEXT_BUDGET,
    NO_LINES_TEXT,
    PROOF_LINES_SHOWN,
    SECTION_MARGIN_X,
    SHOW_ALL_TEXT,
    STALE_TEXT,
    VALUES_NOTE,
    hint_text,
    wrap_to_width,
)
from core.detect.brightness import BrightnessResult
from core.detect.crop import CropResult
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
    TimeRanges,
    save_project,
)

NAMES = ["ep01.mkv", "ep02.mkv", "ep03.mkv", "ep04.mkv", "ep05.mkv"]
BOX = (288, 786, 1344, 53)
DURATION = 1628.0
SAMPLE_TIME = 578.0                 # a 30 s proof window of 09:38–10:08
WINDOW_TEXT = "09:38–10:08"
WAIT_MS = 5000
WHOLE_FILE = object()               # entry(ranges=...) default: a MANUAL "use whole file" choice
LINES = [(578.0, 580.0, "你竟掌握了鲲鹏道法"), (581.0, 583.0, "我早已不是当年的我"),
         (584.0, 586.0, "今日便让你见识见识"), (588.0, 590.0, "纵使千难万险"),
         (591.0, 593.0, "我也要闯一闯"), (594.0, 596.0, "这条路没有回头"),
         (597.0, 599.0, "你我之间早有定数"), (601.0, 603.0, "就此别过")]
LINE_TEXTS = ["09:38 你竟掌握了鲲鹏道法", "09:41 我早已不是当年的我", "09:44 今日便让你见识见识",
              "09:48 纵使千难万险", "09:51 我也要闯一闯", "09:54 这条路没有回头",
              "09:57 你我之间早有定数", "10:01 就此别过"]


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


def entry(name: str, *, crop=BOX, brightness=209, source=Source.DETECTED, review=ReviewState.REVIEWED,
          skipped=False, sample_time=SAMPLE_TIME, ranges=WHOLE_FILE) -> FileEntry:
    """A file with known media and, by default, a MANUAL whole-file range
    choice, so the folder's ranges analysis never runs. `ranges=None` leaves
    the ranges unset, which makes auto-pilot analyse the folder."""
    item = FileEntry(name, media=Media(1920, 888, DURATION, 25.0), review=review, skipped=skipped,
                     sample_time=sample_time,
                     time_ranges=TimeRanges([], Source.MANUAL) if ranges is WHOLE_FILE else ranges)
    if crop is not None:
        item.crop = Crop(*crop, source)
    if brightness is not None:
        item.brightness = Brightness(brightness, source)
    return item


def crop_result(submission, box=BOX) -> CropJobResult:
    result = CropResult(box=box, sample_pts=[SAMPLE_TIME], envelope=box, agreed=3, probes_used=3,
                        flagged=None, hit_pts=[SAMPLE_TIME], frame_size=(1920, 888))
    return CropJobResult(submission.job.file, result, submission.job.hint)


def brightness_result(submission, value=209) -> BrightnessJobResult:
    job = submission.job
    result = BrightnessResult(value=value, plateau=(190, 230), seed=value + 20, gate_floor=None,
                              flagged=None, curve=[])
    return BrightnessJobResult(job.file, result, {}, job.hint_value, job.crop_box)


class Calls:
    """Records each call of a controller command, then runs the real one."""

    def __init__(self, controller, name: str):
        self.calls: list[tuple] = []
        original = getattr(controller, name)

        def wrapper(*args, **kwargs):
            self.calls.append(args)
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
def make_window(controller, tmp_project):
    windows = []

    def make(entries=None, **folder) -> MainWindow:
        entries = [entry(name) for name in NAMES] if entries is None else entries
        folder.setdefault("labels_enabled", False)
        path = tmp_project([item.name for item in entries])
        save_project(Project(path=str(path), folder=FolderSettings(**folder),
                             files={item.name: item for item in entries}))
        window = MainWindow(controller)
        window.resize(1440, 900)
        windows.append(window)
        window.open_folder(str(path))
        settle()
        return window

    yield make
    for window in windows:
        window.close()
        window.deleteLater()
    settle()


@pytest.fixture
def window(make_window) -> MainWindow:
    return make_window()


def activate(window: MainWindow) -> None:
    window.show()
    window.activateWindow()
    assert wait_for(lambda: QApplication.activeWindow() is window)


def unwrapped(button) -> str:
    """A hint button's label with its wrap undone: the verbatim copy."""
    return button.text().replace("\n", " ")


def finish_proof(window, fake_runner, name: str, lines=(), seconds: float = 4.1,
                 window_seconds=(SAMPLE_TIME, SAMPLE_TIME + 30.0)) -> None:
    submission = fake_runner.last("proof", name)
    fake_runner.finish(submission, ProofResult(name, window_seconds, list(lines), seconds))
    window.controller.drain_events()


# --------------------------------------------------------------------------
# Proof: running, lines, notes
# --------------------------------------------------------------------------

def test_the_button_and_the_t_key_run_the_proof_and_are_disabled_while_it_runs(window, fake_runner):
    controller, inspector = window.controller, window.inspector
    name = NAMES[0]
    proofs = Calls(controller, "run_proof")
    activate(window)
    window.queue.setFocus()
    assert inspector.proof_button.text() == "T run"
    assert inspector.proof_button.isEnabled()

    inspector.proof_button.click()
    assert proofs.calls == [(name,)]
    assert fake_runner.last("proof", name)
    assert inspector.proof_status.text() == f"running on {WINDOW_TEXT}…"
    assert not inspector.proof_status.isHidden()
    assert not inspector.proof_button.isEnabled()
    assert not window.proof_action.isEnabled()

    inspector.proof_button.click()                       # the button is disabled while it runs
    QTest.keyClick(window.queue, Qt.Key.Key_T)           # and so is T
    assert proofs.calls == [(name,)]

    finish_proof(window, fake_runner, name, LINES[:2])
    assert inspector.proof_button.isEnabled()
    assert window.proof_action.isEnabled()
    QTest.keyClick(window.queue, Qt.Key.Key_T)           # the T key does what the button does
    assert proofs.calls == [(name,), (name,)]
    assert inspector.proof_status.text() == f"running on {WINDOW_TEXT}…"


def test_the_queue_menu_does_not_queue_a_second_proof_of_a_running_file(window, fake_runner):
    controller, name = window.controller, NAMES[0]
    proofs = Calls(controller, "run_proof")
    actions = {action.text(): action for action in window.queue.context_menu(name).actions()
               if not action.isSeparator()}
    actions["Test OCR (T)"].trigger()
    assert proofs.calls == [(name,)]
    actions["Test OCR (T)"].trigger()                    # the same window is already being OCR'd
    assert proofs.calls == [(name,)]

    finish_proof(window, fake_runner, name, LINES[:1])
    actions["Test OCR (T)"].trigger()
    assert proofs.calls == [(name,), (name,)]


def test_a_finished_proof_shows_its_lines_the_count_and_a_show_all_link(window, fake_runner):
    inspector = window.inspector
    name = NAMES[0]
    inspector.proof_button.click()
    finish_proof(window, fake_runner, name, LINES)

    assert inspector.proof_status.isHidden()
    assert PROOF_LINES_SHOWN == 6
    assert inspector.proof_texts() == LINE_TEXTS[:PROOF_LINES_SHOWN]
    assert inspector.proof_note.text() == "8 lines · took 4.1 s"
    assert inspector.proof_note.property("tone") == ""
    assert not inspector.proof_show_all.isHidden()
    assert inspector.proof_show_all.text() == SHOW_ALL_TEXT

    inspector.proof_show_all.click()
    assert inspector.proof_texts() == LINE_TEXTS
    assert inspector.proof_show_all.isHidden()

    inspector.proof_button.click()                       # a new proof collapses the list again
    finish_proof(window, fake_runner, name, LINES[:2], seconds=2.0)
    assert inspector.proof_texts() == LINE_TEXTS[:2]
    assert inspector.proof_note.text() == "2 lines · took 2.0 s"
    assert inspector.proof_show_all.isHidden()


def test_a_proof_with_no_lines_warns_about_the_crop_and_the_brightness(window, fake_runner):
    inspector = window.inspector
    inspector.proof_button.click()
    finish_proof(window, fake_runner, NAMES[0], [])
    assert inspector.proof_texts() == []
    assert inspector.proof_note.text() == NO_LINES_TEXT
    assert inspector.proof_note.text() == (
        "No subtitles recognised in this window — check crop and brightness.")
    assert inspector.proof_note.property("tone") == "warn"


def test_a_proof_result_is_kept_per_file_for_the_session(window, fake_runner):
    inspector, queue = window.inspector, window.queue
    inspector.proof_button.click()
    finish_proof(window, fake_runner, NAMES[0], LINES[:3])

    queue.select(NAMES[1])
    assert inspector.proof_texts() == []
    assert inspector.proof_note.text() == ""
    assert inspector.proof_status.isHidden()

    queue.select(NAMES[0])
    assert inspector.proof_texts() == LINE_TEXTS[:3]
    assert inspector.proof_note.text() == "3 lines · took 4.1 s"


def test_a_proof_of_a_file_whose_duration_is_unknown_says_so(make_window):
    window = make_window([entry(NAMES[0])] + [entry(name) for name in NAMES[1:]])
    window.controller.entry(NAMES[0]).media = Media(0, 0, 0.0, 0.0)
    window.inspector.proof_button.click()
    assert "duration unknown" in window.inspector.proof_note.text()
    assert window.inspector.proof_note.property("tone") == "warn"


# --------------------------------------------------------------------------
# Proof staleness
# --------------------------------------------------------------------------

@pytest.mark.parametrize("edit", ["crop", "brightness", "ranges"])
def test_an_edit_makes_the_stored_proof_stale_and_keeps_its_lines(window, fake_runner, edit):
    controller, inspector = window.controller, window.inspector
    name = NAMES[0]
    inspector.proof_button.click()
    finish_proof(window, fake_runner, name, LINES[:3])
    assert inspector.proof_note.text() == "3 lines · took 4.1 s"

    controller.file_changed.emit(name)                   # evidence only: nothing the proof used changed
    assert inspector.proof_note.text() == "3 lines · took 4.1 s"

    if edit == "crop":
        controller.set_crop(name, (290, 780, 1340, 60))
    elif edit == "brightness":
        controller.set_brightness(name, 214)
    else:
        controller.set_time_ranges(name, [("02:33", "23:05")])
    assert inspector.proof_note.text() == STALE_TEXT
    assert inspector.proof_note.text() == "settings changed — run again"
    assert inspector.proof_note.property("tone") == ""
    assert inspector.proof_texts() == LINE_TEXTS[:3]     # the old lines stay visible

    inspector.proof_button.click()                       # running again clears the mark
    finish_proof(window, fake_runner, name, LINES[:2], seconds=2.0)
    assert inspector.proof_note.text() == "2 lines · took 2.0 s"


def test_another_files_edit_leaves_the_proof_alone(window, fake_runner):
    controller, inspector = window.controller, window.inspector
    inspector.proof_button.click()
    finish_proof(window, fake_runner, NAMES[0], LINES[:3])
    controller.set_brightness(NAMES[1], 214)
    assert inspector.proof_note.text() == "3 lines · took 4.1 s"


def test_an_edit_while_the_proof_runs_marks_its_result_stale(window, fake_runner):
    controller, inspector = window.controller, window.inspector
    name = NAMES[0]
    inspector.proof_button.click()
    controller.set_brightness(name, 214)                 # the job already holds the old settings
    assert inspector.proof_status.text() == f"running on {WINDOW_TEXT}…"
    finish_proof(window, fake_runner, name, LINES[:3])
    assert inspector.proof_note.text() == STALE_TEXT
    assert inspector.proof_texts() == LINE_TEXTS[:3]


# --------------------------------------------------------------------------
# The hint re-detect offer
# --------------------------------------------------------------------------

def test_the_offer_appears_only_after_a_manual_edit_with_one_button_per_kind(window):
    controller, inspector = window.controller, window.inspector
    name = NAMES[0]
    hints = Calls(controller, "redetect_others_with_hint")
    assert inspector.offer_section.isHidden()
    assert inspector.offer_note.text() == (
        "Corrections are never copied verbatim to other episodes. Instead the app offers:")

    controller.set_crop(name, (290, 780, 1340, 60))
    assert not inspector.offer_section.isHidden()
    assert not inspector.hint_buttons["crop"].isHidden()
    assert hint_text(4, "crop") == "↻ re-detect the other 4 using this crop as a hint"
    assert unwrapped(inspector.hint_buttons["crop"]) == hint_text(4, "crop")
    assert inspector.hint_buttons["brightness"].isHidden()

    controller.set_brightness(name, 214)
    assert not inspector.hint_buttons["brightness"].isHidden()
    assert hint_text(4, "brightness") == "↻ re-detect the other 4 using this brightness as a hint"
    assert unwrapped(inspector.hint_buttons["brightness"]) == hint_text(4, "brightness")

    inspector.hint_buttons["crop"].click()
    assert hints.calls == [(name, "crop")]
    assert inspector.hint_buttons["crop"].isHidden()
    assert not inspector.hint_buttons["brightness"].isHidden()

    inspector.hint_buttons["brightness"].click()
    assert hints.calls == [(name, "crop"), (name, "brightness")]
    assert inspector.hint_buttons["brightness"].isHidden()


def test_the_offer_belongs_to_the_file_that_was_edited(window):
    controller, inspector, queue = window.controller, window.inspector, window.queue
    controller.set_crop(NAMES[0], (290, 780, 1340, 60))
    assert not inspector.offer_section.isHidden()
    queue.select(NAMES[1])
    assert inspector.offer_section.isHidden()
    queue.select(NAMES[0])
    assert not inspector.offer_section.isHidden()


def test_the_counts_are_the_files_the_re_detect_would_submit(make_window):
    entries = [entry(NAMES[0]),                                      # the edited file
               entry(NAMES[1], source=Source.MANUAL),                # never re-measured
               entry(NAMES[2], source=Source.IMPORTED),              # never re-measured
               entry(NAMES[3], skipped=True),                        # skipped
               entry(NAMES[4])]                                      # the only target
    window = make_window(entries)
    controller, inspector = window.controller, window.inspector
    name = NAMES[0]

    controller.set_crop(name, (290, 780, 1340, 60))
    controller.set_brightness(name, 214)
    assert controller.hint_targets(name, "crop") == [NAMES[4]]
    assert controller.hint_targets(name, "brightness") == [NAMES[4]]
    assert unwrapped(inspector.hint_buttons["crop"]) == hint_text(1, "crop")
    assert unwrapped(inspector.hint_buttons["brightness"]) == hint_text(1, "brightness")


def test_a_hint_button_with_nothing_to_re_detect_is_disabled(make_window):
    window = make_window([entry(NAMES[0])])
    controller, inspector = window.controller, window.inspector
    controller.set_crop(NAMES[0], (290, 780, 1340, 60))
    assert not inspector.offer_section.isHidden()
    assert unwrapped(inspector.hint_buttons["crop"]) == hint_text(0, "crop")
    assert not inspector.hint_buttons["crop"].isEnabled()


def test_the_hint_buttons_wrap_instead_of_widening_the_column(window):
    """The one-line copy is wider than the whole 322 px inspector, which used
    to push the scroll content out and clip every section. It is measured and
    wrapped, so the section -- and with it the column -- still fits."""
    controller, inspector = window.controller, window.inspector
    name = NAMES[0]
    controller.set_crop(name, (290, 780, 1340, 60))
    controller.set_brightness(name, 214)
    settle()

    limit = tokens.INSPECTOR_WIDTH - 1 - 2 * SECTION_MARGIN_X          # the inspector's 1 px border
    for what, button in inspector.hint_buttons.items():
        assert button.fontMetrics().horizontalAdvance(hint_text(4, what)) > BUTTON_TEXT_BUDGET or \
            "\n" not in button.text()                                  # only wrapped when it must be
        for line in button.text().split("\n"):
            assert button.fontMetrics().horizontalAdvance(line) <= BUTTON_TEXT_BUDGET
        assert button.sizeHint().width() <= limit
    assert inspector.offer_section.minimumSizeHint().width() <= tokens.INSPECTOR_WIDTH - 1
    assert inspector.minimumSizeHint().width() <= tokens.INSPECTOR_WIDTH


def test_wrap_to_width_keeps_the_words_and_balances_the_lines(window):
    metrics = window.inspector.hint_buttons["crop"].fontMetrics()
    text = hint_text(4, "brightness")
    wrapped = wrap_to_width(text, metrics, BUTTON_TEXT_BUDGET)
    assert wrapped.replace("\n", " ") == text                          # nothing added or lost
    lines = wrapped.split("\n")
    assert len(lines) == 2
    assert all(metrics.horizontalAdvance(line) <= BUTTON_TEXT_BUDGET for line in lines)
    assert len(lines[-1].split(" ")) > 1                               # no single-word last line
    assert wrap_to_width("short enough", metrics, BUTTON_TEXT_BUDGET) == "short enough"


def test_apply_to_this_file_only_hides_the_offer(window):
    controller, inspector, queue = window.controller, window.inspector, window.queue
    name = NAMES[0]
    hints = Calls(controller, "redetect_others_with_hint")
    controller.set_crop(name, (290, 780, 1340, 60))
    controller.set_brightness(name, 214)
    assert inspector.this_file_only_button.text() == "apply to this file only"

    inspector.this_file_only_button.click()
    assert hints.calls == []
    assert inspector.offer_section.isHidden()
    queue.select(NAMES[1])
    queue.select(name)
    assert inspector.offer_section.isHidden()            # dismissed stays dismissed

    controller.set_brightness(name, 216)                 # a new edit offers again
    assert not inspector.offer_section.isHidden()


def test_the_re_detecting_line_follows_the_jobs(window, fake_runner):
    controller, inspector = window.controller, window.inspector
    name = NAMES[0]
    controller.set_crop(name, (290, 780, 1340, 60))
    targets = controller.hint_targets(name, "crop")
    assert targets == NAMES[1:]

    inspector.hint_buttons["crop"].click()
    assert inspector.offer_status.text() == "re-detecting 4 files…"
    assert not inspector.offer_status.isHidden()
    assert not inspector.offer_section.isHidden()
    controller.drain_events()                            # the jobs are queued, none has started
    assert inspector.offer_status.text() == "re-detecting 4 files…"

    for target in targets[:-1]:
        submission = fake_runner.last("crop", target)
        fake_runner.finish(submission, crop_result(submission))
        controller.drain_events()
        assert not inspector.offer_status.isHidden()

    submission = fake_runner.last("crop", targets[-1])
    fake_runner.finish(submission, crop_result(submission))
    controller.drain_events()
    assert inspector.offer_status.isHidden()
    assert inspector.offer_section.isHidden()            # nothing left to offer


def test_a_brightness_hint_waiting_for_the_ranges_analysis_keeps_the_line(make_window, fake_runner):
    """Auto-pilot holds a brightness job until the folder's ranges analysis
    ends (detect_brightness samples inside the keep ranges), so the files the
    hint re-detect covers have no job of their own yet. The line follows
    auto-pilot's pending kinds, not the runner's queue, and stays up."""
    window = make_window([entry(name, ranges=None) for name in NAMES])
    controller, inspector = window.controller, window.inspector
    name = NAMES[0]
    assert fake_runner.last("ranges")                    # outstanding: brightness waits for it

    controller.set_brightness(name, 214)
    targets = controller.hint_targets(name, "brightness")
    assert targets == NAMES[1:]
    inspector.hint_buttons["brightness"].click()
    assert fake_runner.of_kind("brightness") == []       # nothing submitted: the jobs wait
    assert inspector.offer_status.text() == "re-detecting 4 files…"
    controller.drain_events()
    assert not inspector.offer_status.isHidden()

    for target in targets:                               # other jobs come and go meanwhile
        audio = fake_runner.last("audio_profile", target)
        fake_runner.finish(audio, None)
        controller.drain_events()
        assert not inspector.offer_status.isHidden()

    fake_runner.finish(fake_runner.last("ranges"), None)  # the analysis ends: the jobs go out
    controller.drain_events()
    assert [s.job.file for s in fake_runner.of_kind("brightness")] == targets
    assert not inspector.offer_status.isHidden()

    for target in targets:
        assert not inspector.offer_status.isHidden()
        submission = fake_runner.last("brightness", target)
        fake_runner.finish(submission, brightness_result(submission))
        controller.drain_events()
    assert inspector.offer_status.isHidden()
    assert inspector.offer_section.isHidden()


def test_the_re_detecting_line_survives_a_failed_job(window, fake_runner):
    controller, inspector = window.controller, window.inspector
    name = NAMES[0]
    controller.set_brightness(name, 214)
    targets = controller.hint_targets(name, "brightness")
    inspector.hint_buttons["brightness"].click()
    assert inspector.offer_status.text() == f"re-detecting {len(targets)} files…"
    controller.drain_events()

    for target in targets[:-1]:
        submission = fake_runner.last("brightness", target)
        fake_runner.finish(submission, brightness_result(submission))
        controller.drain_events()
    submission = fake_runner.last("brightness", targets[-1])
    fake_runner.finish(submission, None, event_type="failed", message="boom", error="")
    controller.drain_events()
    assert inspector.offer_status.isHidden()


# --------------------------------------------------------------------------
# The Detected rows' series-median note
# --------------------------------------------------------------------------

def test_the_brightness_note_reports_the_series_median(make_window):
    entries = [entry(name, brightness=value)
               for name, value in zip(NAMES, (205, 207, 209, 211, 213))]
    window = make_window(entries)
    assert series_median_brightness(entries) == 209
    assert window.inspector.detected_note.text() == (
        f"{VALUES_NOTE} Series median brightness is 209 — this episode keeps its own.")


def test_the_series_median_note_needs_three_measured_files(make_window):
    entries = [entry(NAMES[0], brightness=205), entry(NAMES[1], brightness=211),
               entry(NAMES[2], brightness=None, review=ReviewState.PROPOSED)]
    window = make_window(entries)
    assert series_median_brightness(entries) is None
    assert window.inspector.detected_note.text() == VALUES_NOTE


# --------------------------------------------------------------------------
# Files vanishing, the folder closing
# --------------------------------------------------------------------------

def vanish(controller, name: str) -> None:
    """Delete `name` from the folder and let the folder watcher notice."""
    (Path(controller.project.path) / name).unlink()
    assert wait_for(lambda: name not in controller.names())
    settle()


def test_a_vanished_target_leaves_no_trace_and_the_rest_hold_the_line(window, fake_runner):
    controller, inspector = window.controller, window.inspector
    name, victim = NAMES[0], NAMES[2]
    controller.set_crop(name, (290, 780, 1340, 60))
    targets = controller.hint_targets(name, "crop")
    assert victim in targets
    inspector.hint_buttons["crop"].click()
    assert inspector.offer_status.text() == f"re-detecting {len(targets)} files…"

    vanish(controller, victim)

    for book in (inspector._seen, inspector._edited, inspector._proof_keys, inspector._stale_proofs):
        assert victim not in book
    assert all(victim not in files for files in inspector._redetecting.values())
    assert inspector.offer_status.text() == f"re-detecting {len(targets) - 1} files…"

    for target in [name for name in targets if name != victim]:
        submission = fake_runner.last("crop", target)
        fake_runner.finish(submission, crop_result(submission))
        controller.drain_events()
    assert inspector.offer_status.isHidden()
    assert inspector.offer_section.isHidden()


def test_a_vanished_source_file_takes_its_offer_and_its_line_with_it(window, fake_runner):
    controller, inspector = window.controller, window.inspector
    name = NAMES[0]
    controller.set_crop(name, (290, 780, 1340, 60))
    controller.set_brightness(name, 214)                 # an offer still standing for the other kind
    inspector.proof_button.click()
    finish_proof(window, fake_runner, name, LINES[:2])
    controller.set_time_ranges(name, [("02:33", "23:05")])   # ... and a stale proof
    inspector.hint_buttons["crop"].click()
    assert not inspector.offer_status.isHidden()
    assert not inspector.hint_buttons["brightness"].isHidden()

    vanish(controller, name)

    for book in (inspector._seen, inspector._edited, inspector._proof_keys, inspector._stale_proofs):
        assert name not in book
    assert all(source != name for source, _kind in inspector._redetecting)
    assert inspector.current_file() != name              # the queue moved on
    assert inspector.offer_section.isHidden()            # the file it belonged to is gone
    assert inspector.proof_texts() == []


def test_closing_the_folder_forgets_the_session(window, fake_runner):
    controller, inspector = window.controller, window.inspector
    name = NAMES[0]
    controller.set_crop(name, (290, 780, 1340, 60))
    inspector.hint_buttons["crop"].click()
    inspector.proof_button.click()
    finish_proof(window, fake_runner, name, LINES[:2])
    controller.set_brightness(name, 214)                 # the proof is stale, the offer stands again
    assert inspector._edited and inspector._proof_keys and inspector._stale_proofs and inspector._redetecting

    controller.close_folder()
    settle()

    assert inspector._seen == {}
    assert inspector._edited == {}
    assert inspector._proof_keys == {}
    assert inspector._stale_proofs == set()
    assert inspector._redetecting == {}
    assert inspector.current_file() is None
    assert inspector.offer_section.isHidden()
    assert inspector.proof_texts() == []
