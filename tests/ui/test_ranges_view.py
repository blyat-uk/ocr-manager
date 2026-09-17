"""Plan 3C Task 4: the Time ranges tab and the shared timeline
(app/views/ranges_view.py).

Nothing is decoded: the timeline draws from `entry.time_ranges`,
`evidence["ranges"]` and `evidence["audio"]`, all constructed here. The
numbers are the brief's: a 27:08 episode with a 0:00-2:33 intro block and a
23:05-27:08 outro block, both matched in 5 files, and a speech span at
23:20-24:15 that falls inside the outro.
"""
from __future__ import annotations

import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

from app.controller import ProjectController
from app.views import ranges_view
from app.views.crop_view import CropTab
from app.views.ranges_view import (
    NO_DURATION_TEXT,
    NOTE_TEXT,
    UNREADABLE_TEXT,
    KeepRow,
    RangesTab,
    Timeline,
    WarningRow,
    with_added_range,
)
from app.views.stage import Stage, StageTab
from app.views.tabs import evidence_tabs
from core.project import Source, TimeRange, TimeRanges
from core.project.ocr_kwargs import ocr_call_for

NAME = "ep01.mkv"
OTHERS = ["ep02.mkv", "ep03.mkv"]
DURATION = 1628.0                      # 27:08 exactly ...
FRACTIONAL = 1628.44                   # ... and as a real file has it: frames / fps
INTRO_END = 153.0                      # 2:33
OUTRO_START = 1385.0                   # 23:05
OTHER_DURATIONS = {"ep02.mkv": 1418.0, "ep03.mkv": 1628.0}     # 23:38, 27:08

SPEECH = [(200.0, 260.0), (400.0, 470.0), (700.0, 760.0),
          (1000.0, 1080.0), (1300.0, 1360.0), (1400.0, 1455.0)]
WARN_SPEECH = (1400.0, 1455.0)         # 23:20-24:15, inside the outro block


def blocks() -> list[dict]:
    return [{"start_sec": 0.0, "end_sec": INTRO_END, "kind": "intro",
             "matched_files": 5, "score": 0.98},
            {"start_sec": OUTRO_START, "end_sec": DURATION, "kind": "outro",
             "matched_files": 5, "score": 0.96}]


def envelope(bins: int = 600) -> list[float]:
    """A deterministic 0..1 envelope: loud in the middle, quiet at the ends."""
    return [round(0.2 + 0.8 * abs(((index % 40) / 20.0) - 1.0), 4) for index in range(bins)]


def ranges_evidence(*, duration: float = DURATION, block_list=None) -> dict:
    return {"blocks": blocks() if block_list is None else block_list, "duration": float(duration)}


def audio_evidence(*, speech=SPEECH, duration: float = DURATION) -> dict:
    return {"envelope": envelope(), "speech": [[s, e] for s, e in speech], "duration": float(duration)}


def crop_evidence(times=(120.0, 600.0, 1200.0)) -> dict:
    return {"frame_size": [1920, 1080],
            "samples": [{"time": float(t), "kept": True, "lines": 1,
                         "boxes": [[288, 784, 1344, 55]]} for t in times]}


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def settle() -> None:
    QApplication.processEvents()


class Calls:
    """Record every call of one controller method (and still run it)."""

    def __init__(self, controller, name: str):
        self.calls: list[tuple] = []
        original = getattr(controller, name)

        def wrapper(*args, **kwargs):
            self.calls.append(args)
            return original(*args, **kwargs)

        setattr(controller, name, wrapper)

    def last(self) -> tuple:
        assert self.calls, "no call recorded"
        return self.calls[-1]


@pytest.fixture(params=[DURATION, FRACTIONAL], ids=["whole-second", "fractional"])
def duration(request) -> float:
    """Every test runs at both durations. A real one is frames / fps and
    lands between whole seconds, which is exactly where "to the end of the
    file" used to be unreachable: a boundary snapped to 1628 could never
    equal a duration of 1628.44."""
    return request.param


@pytest.fixture
def controller(qapp, fake_runner, tmp_project, duration):
    made = ProjectController(fake_runner, save_debounce_ms=10)
    made.open_folder(str(tmp_project([NAME, *OTHERS])))
    made.entry(NAME).media.duration = duration
    yield made
    made.shutdown(timeout=0.5)


def give_values(controller, name: str = NAME, *, keep=((INTRO_END, OUTRO_START),),
                duration: float | None = None, ranges=True, audio=True, crop=False) -> None:
    """`duration=None` keeps whatever the `duration` fixture put on the file."""
    entry = controller.entry(name)
    if duration is not None:
        entry.media.duration = duration
    duration = entry.media.duration
    if keep is not None:
        entry.time_ranges = TimeRanges(
            [TimeRange(_clock(start), _clock(end)) for start, end in keep], Source.DETECTED)
    if ranges:
        entry.evidence["ranges"] = ranges_evidence(duration=duration)
    if audio:
        entry.evidence["audio"] = audio_evidence(duration=duration)
    if crop:
        entry.evidence["crop"] = crop_evidence()
    for other, other_duration in OTHER_DURATIONS.items():
        controller.entry(other).media.duration = other_duration


def _clock(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


@pytest.fixture
def tab(controller):
    give_values(controller)
    made = RangesTab(controller)
    page = made.page()
    page.resize(880, 420)
    page.show()
    made.inspector_panel().resize(322, 400)
    made.set_file(NAME)
    made.timeline.resize(800, 68)
    settle()
    yield made
    page.close()
    page.deleteLater()


# --------------------------------------------------------------------------
# Mouse helpers (tests/ui/test_crop_view.py's, verbatim in spirit)
# --------------------------------------------------------------------------

def _send(widget, kind, point: QPointF, button, buttons) -> None:
    event = QMouseEvent(kind, point, point, QPointF(widget.mapToGlobal(point.toPoint())),
                        button, buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


def press(widget, point: QPointF) -> None:
    _send(widget, QEvent.Type.MouseButtonPress, point, Qt.MouseButton.LeftButton,
          Qt.MouseButton.LeftButton)


def drag_to(widget, point: QPointF) -> None:
    _send(widget, QEvent.Type.MouseMove, point, Qt.MouseButton.NoButton,
          Qt.MouseButton.LeftButton)


def release(widget, point: QPointF) -> None:
    _send(widget, QEvent.Type.MouseButtonRelease, point, Qt.MouseButton.LeftButton,
          Qt.MouseButton.NoButton)


def double_click(widget, point: QPointF) -> None:
    press(widget, point)
    _send(widget, QEvent.Type.MouseButtonDblClick, point, Qt.MouseButton.LeftButton,
          Qt.MouseButton.LeftButton)
    release(widget, point)


def drag_grip(timeline: Timeline, from_time: float, to_x: float) -> None:
    start = QPointF(timeline.x_for(from_time), 26.0)
    end = QPointF(float(to_x), 26.0)
    press(timeline, start)
    drag_to(timeline, end)
    release(timeline, end)


def click_at(timeline: Timeline, time: float) -> None:
    point = QPointF(timeline.x_for(time), 26.0)
    press(timeline, point)
    release(timeline, point)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

@pytest.fixture
def bare(controller):
    """A Timeline of its own, outside any layout: its width is the test's."""
    give_values(controller)
    made = Timeline(controller, mode="edit")
    made.set_file(NAME)
    made.resize(800, 68)
    yield made
    made.deleteLater()


def test_time_maps_to_x_across_the_full_width_at_any_width(bare, duration):
    timeline = bare
    for width in (800, 431):
        timeline.resize(width, 68)
        assert timeline.x_for(0.0) == pytest.approx(0.0)
        assert timeline.x_for(duration) == pytest.approx(width)
        assert timeline.x_for(duration / 2) == pytest.approx(width / 2)
        assert timeline.time_at(width / 2) == pytest.approx(duration / 2)
        assert timeline.time_at(timeline.x_for(OUTRO_START)) == pytest.approx(OUTRO_START)


def test_out_of_range_times_and_positions_clamp(bare, duration):
    timeline = bare
    assert timeline.x_for(-40.0) == pytest.approx(0.0)
    assert timeline.x_for(duration + 500) == pytest.approx(800.0)
    assert timeline.time_at(-20.0) == pytest.approx(0.0)
    assert timeline.time_at(9000.0) == pytest.approx(duration)


# --------------------------------------------------------------------------
# Spans, labels and sub-lines
# --------------------------------------------------------------------------

def test_the_track_shows_the_intro_keep_and_outro_with_the_mockups_copy(tab, duration):
    spans = tab.timeline.spans()
    assert [(span.start, span.end, span.keep) for span in spans] == [
        (0.0, INTRO_END, False), (INTRO_END, OUTRO_START, True), (OUTRO_START, duration, False)]
    assert [span.label for span in spans] == [
        "INTRO · 0:00–2:33", "KEEP · 2:33–23:05", "OUTRO · 23:05–27:08"]
    assert [span.detail for span in spans] == [
        "matches 4 episodes · 98%", "20:32 of OCR", "matches 4 episodes · 96%"]


def test_a_skip_span_takes_the_block_it_overlaps_most(controller):
    """Two blocks inside one skip span: the bigger overlap names it."""
    give_values(controller, keep=((600.0, 1200.0),), ranges=False)
    controller.entry(NAME).evidence["ranges"] = ranges_evidence(block_list=[
        {"start_sec": 0.0, "end_sec": 60.0, "kind": "repeat", "matched_files": 3, "score": 0.5},
        {"start_sec": 100.0, "end_sec": 500.0, "kind": "intro", "matched_files": 5, "score": 0.9},
    ])
    made = RangesTab(controller)
    made.set_file(NAME)
    made.timeline.resize(800, 68)
    settle()
    first = made.timeline.spans()[0]
    assert first.label == "INTRO · 0:00–10:00"
    assert first.detail == "matches 4 episodes · 90%"


def test_a_skip_span_with_no_block_is_still_drawn_and_labelled(controller):
    give_values(controller, keep=((60.0, 600.0),), ranges=False, audio=False)
    made = RangesTab(controller)
    made.set_file(NAME)
    made.timeline.resize(800, 68)
    settle()
    spans = made.timeline.spans()
    assert [span.keep for span in spans] == [False, True, False]
    assert spans[0].label == "SKIP · 0:00–1:00"
    assert spans[0].detail == ""


def test_a_whole_file_entry_with_no_evidence_is_one_keep_span_and_no_warnings(controller, duration):
    entry = controller.entry(NAME)
    entry.time_ranges = None
    made = RangesTab(controller)
    made.set_file(NAME)
    made.timeline.resize(800, 68)
    settle()
    spans = made.timeline.spans()
    assert [(span.start, span.end, span.keep) for span in spans] == [(0.0, duration, True)]
    assert spans[0].label == "KEEP · 0:00–27:08"
    assert made.timeline.speech_segments() == []
    assert made.timeline.envelope() == []
    assert made.warning_texts() == []
    assert made.inspector_panel().keep_rows() == [("Keep", "whole file")]


def test_an_empty_manual_range_list_is_the_whole_file_too(controller, duration):
    give_values(controller, keep=())
    made = RangesTab(controller)
    made.set_file(NAME)
    made.timeline.resize(800, 68)
    settle()
    assert [(span.start, span.end) for span in made.timeline.spans()] == [(0.0, duration)]
    assert made.timeline.spans()[0].keep


def test_overlapping_stored_ranges_are_drawn_as_one_keep(controller):
    """Sorting alone leaves the boundaries out of order, and a grip between
    two of them snaps backwards."""
    give_values(controller, keep=((INTRO_END, 700.0), (600.0, OUTRO_START)))
    made = RangesTab(controller)
    made.set_file(NAME)
    settle()
    assert made.timeline.keeps() == [(INTRO_END, OUTRO_START)]
    assert made.timeline.grips() == [INTRO_END, OUTRO_START]
    assert made.inspector_panel().keep_rows() == [("Keep", "2:33 → 23:05")]


def test_a_stored_range_that_cannot_be_read_is_named_never_redrawn(controller):
    """The one thing the timeline must not do is disagree with the run: an
    unreadable range is left alone and said out loud, not quietly turned into
    a range that starts at 0:00."""
    give_values(controller, keep=None)
    controller.entry(NAME).time_ranges = TimeRanges(
        [TimeRange("nonsense", "10:00"), TimeRange("2:33", "23:05")], Source.MANUAL)
    made = RangesTab(controller)
    made.set_file(NAME)
    settle()
    assert made.timeline.keeps() == [(INTRO_END, OUTRO_START)]
    assert made.status_text() == UNREADABLE_TEXT.format(values="nonsense → 10:00")
    assert made.status_tone() == "warn"
    assert made.inspector_panel().keep_rows() == [("Keep", "2:33 → 23:05")]


def test_an_inverted_stored_range_is_named_too_and_is_not_whole_file(controller):
    give_values(controller, keep=None)
    controller.entry(NAME).time_ranges = TimeRanges([TimeRange("10:00", "2:00")], Source.MANUAL)
    made = RangesTab(controller)
    made.set_file(NAME)
    settle()
    assert made.timeline.keeps() == []
    assert made.status_text() == UNREADABLE_TEXT.format(values="10:00 → 2:00")
    assert made.inspector_panel().keep_rows() == []       # stored, but not "whole file"


def test_a_readable_file_says_nothing(tab):
    assert tab.status_text() == ""


def test_without_a_duration_the_tab_says_so_rather_than_drawing_nothing(controller):
    give_values(controller, ranges=False, audio=False)
    controller.entry(NAME).media.duration = 0.0
    made = RangesTab(controller)
    made.set_file(NAME)
    settle()
    assert made.timeline.spans() == []
    assert made.status_text() == NO_DURATION_TEXT
    assert made.status_tone() == ""


def test_the_duration_falls_back_to_the_ranges_evidence(controller, duration):
    give_values(controller)
    controller.entry(NAME).media.duration = 0.0
    made = RangesTab(controller)
    made.set_file(NAME)
    settle()
    assert made.timeline.duration() == pytest.approx(duration)


# --------------------------------------------------------------------------
# Grips and editing
# --------------------------------------------------------------------------

def test_the_grips_sit_at_every_keep_boundary_in_edit_mode(tab):
    assert tab.timeline.grips() == [INTRO_END, OUTRO_START]


def test_dragging_a_grip_commits_one_snapped_range(tab, controller):
    calls = Calls(controller, "set_time_ranges")
    timeline = tab.timeline
    drag_grip(timeline, INTRO_END, timeline.x_for(300.4))
    assert calls.calls == [(NAME, [("5:00", "23:05")])]


def test_a_grip_dragged_past_the_duration_stores_an_open_end(tab, controller):
    calls = Calls(controller, "set_time_ranges")
    drag_grip(tab.timeline, OUTRO_START, 5_000.0)
    assert calls.calls == [(NAME, [("2:33", None)])]


def test_a_keep_that_reaches_the_end_leaves_no_sliver(tab, controller, duration):
    """A duration is frames / fps: a boundary snapped to whole seconds can
    land just short of it, which would store a closed end, draw an
    ungrabbable skip sliver and stop the run early."""
    drag_grip(tab.timeline, OUTRO_START, 5_000.0)
    assert [(span.start, span.end, span.keep) for span in tab.timeline.spans()] == [
        (0.0, INTRO_END, False), (INTRO_END, duration, True)]
    project = controller.project
    assert ocr_call_for(project.files[NAME], project.folder, project.path).time_ranges == [("2:33", "")]


def test_a_boundary_dragged_within_a_second_of_the_end_takes_the_end(tab, controller, duration):
    calls = Calls(controller, "set_time_ranges")
    drag_grip(tab.timeline, OUTRO_START, tab.timeline.x_for(duration - 0.4))
    assert calls.calls == [(NAME, [("2:33", None)])]


def test_a_grip_dragged_before_zero_stores_an_open_start(tab, controller):
    calls = Calls(controller, "set_time_ranges")
    drag_grip(tab.timeline, INTRO_END, -400.0)
    assert calls.calls == [(NAME, [(None, "23:05")])]


def test_a_grip_is_clamped_by_its_neighbouring_boundary(tab, controller, duration):
    calls = Calls(controller, "set_time_ranges")
    drag_grip(tab.timeline, INTRO_END, tab.timeline.x_for(duration))
    assert calls.calls == [(NAME, [("23:04", "23:05")])]


def test_a_drag_commits_once_on_release_not_on_every_move(tab, controller):
    calls = Calls(controller, "set_time_ranges")
    timeline = tab.timeline
    press(timeline, QPointF(timeline.x_for(INTRO_END), 26.0))
    for time in (200.0, 240.0, 300.0):
        drag_to(timeline, QPointF(timeline.x_for(time), 26.0))
    assert calls.calls == []
    assert timeline.spans()[1].start == pytest.approx(300.0)     # the drag previews live
    release(timeline, QPointF(timeline.x_for(300.0), 26.0))
    assert calls.calls == [(NAME, [("5:00", "23:05")])]


def test_taking_a_grip_without_moving_it_commits_nothing(tab, controller):
    """A MANUAL write of the same times would freeze the detected value."""
    calls = Calls(controller, "set_time_ranges")
    point = QPointF(tab.timeline.x_for(INTRO_END), 26.0)
    press(tab.timeline, point)
    release(tab.timeline, point)
    assert calls.calls == []
    assert controller.entry(NAME).time_ranges.source == Source.DETECTED


def test_a_press_away_from_a_grip_does_not_edit(tab, controller):
    calls = Calls(controller, "set_time_ranges")
    timeline = tab.timeline
    drag_grip(timeline, 700.0, timeline.x_for(800.0))
    assert calls.calls == []


def test_the_edited_ranges_round_trip_into_the_ocr_call(tab, controller):
    drag_grip(tab.timeline, INTRO_END, tab.timeline.x_for(300.0))
    settle()
    project = controller.project
    call = ocr_call_for(project.files[NAME], project.folder, project.path)
    assert call.time_ranges == [("5:00", "23:05")]
    drag_grip(tab.timeline, OUTRO_START, 5_000.0)
    settle()
    call = ocr_call_for(project.files[NAME], project.folder, project.path)
    assert call.time_ranges == [("5:00", "")]


# --------------------------------------------------------------------------
# Speech lane and warnings
# --------------------------------------------------------------------------

def test_the_speech_lane_draws_every_span_and_warns_only_inside_a_skip(tab):
    timeline = tab.timeline
    assert timeline.speech_segments() == [tuple(span) for span in SPEECH]
    assert timeline.warn_spans() == [WARN_SPEECH]


def test_a_speech_span_crossing_a_boundary_warns_only_for_the_part_inside(controller):
    give_values(controller, audio=False)
    controller.entry(NAME).evidence["audio"] = audio_evidence(speech=[(1370.0, 1420.0)])
    made = RangesTab(controller)
    made.set_file(NAME)
    made.timeline.resize(800, 68)
    settle()
    assert made.timeline.speech_segments() == [(1370.0, 1420.0)]
    assert made.timeline.warn_spans() == [(OUTRO_START, 1420.0)]


def test_the_warning_row_names_the_span_and_the_block(tab):
    assert tab.warning_texts() == ["⚠ Speech at 23:20–24:15 falls inside the outro block"]
    assert tab.note.text() == NOTE_TEXT


def test_extend_keep_covers_the_speech_and_commits_whole_seconds(tab, controller):
    calls = Calls(controller, "set_time_ranges")
    tab.extend_buttons()[0].click()
    assert calls.calls == [(NAME, [("2:33", "24:15")])]


def test_extend_keep_reaches_back_when_the_speech_is_before_the_keep(controller):
    give_values(controller, audio=False)
    controller.entry(NAME).evidence["audio"] = audio_evidence(speech=[(100.4, 140.2)])
    made = RangesTab(controller)
    made.set_file(NAME)
    made.timeline.resize(800, 68)
    settle()
    calls = Calls(controller, "set_time_ranges")
    assert made.warning_texts() == ["⚠ Speech at 1:40–2:20 falls inside the intro block"]
    made.extend_buttons()[0].click()
    assert calls.calls == [(NAME, [("1:40", "23:05")])]


def test_no_warning_row_when_every_speech_span_is_kept(controller):
    give_values(controller, audio=False)
    controller.entry(NAME).evidence["audio"] = audio_evidence(speech=[(400.0, 470.0)])
    made = RangesTab(controller)
    made.set_file(NAME)
    settle()
    assert made.warning_texts() == []
    assert made.extend_buttons() == []


def test_the_speech_in_skip_check_runs_once_per_state(tab, monkeypatch):
    """`warnings()` is read from paintEvent as well as from the tab, so it
    must not re-run the check on every repaint."""
    original = ranges_view.speech_in_skips
    calls: list[tuple] = []

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(ranges_view, "speech_in_skips", counted)
    tab.timeline.warnings()
    tab.timeline.warnings()
    tab.timeline.grab()
    assert calls == []                               # nothing has changed since the tab was built
    tab.timeline.commit([(INTRO_END, 900.0)])        # a new state: checked once ...
    tab.timeline.warnings()
    tab.timeline.grab()
    assert len(calls) == 1                           # ... and not again per repaint


# --------------------------------------------------------------------------
# The inspector panel
# --------------------------------------------------------------------------

def test_the_panel_lists_one_row_per_keep_range(controller):
    give_values(controller, keep=((INTRO_END, 600.0), (700.0, OUTRO_START)))
    made = RangesTab(controller)
    made.set_file(NAME)
    settle()
    assert made.inspector_panel().keep_rows() == [("Keep", "2:33 → 10:00"), ("Keep", "11:40 → 23:05")]


def test_removing_a_range_commits_the_rest(controller):
    give_values(controller, keep=((INTRO_END, 600.0), (700.0, OUTRO_START)))
    made = RangesTab(controller)
    made.set_file(NAME)
    settle()
    calls = Calls(controller, "set_time_ranges")
    made.inspector_panel().remove_buttons()[1].click()
    assert calls.calls == [(NAME, [("2:33", "10:00")])]


def test_a_second_refresh_leaves_no_row_behind(tab):
    """A row only removed from the layout stays a child of the panel, at its
    default size, painting over everything under it -- which is how the
    panel's buttons went missing. `held` stands in for whatever outlives the
    refresh that dropped the row (a row's own signal connections do)."""
    panel, page = tab.inspector_panel(), tab.page()
    held = panel.findChildren(KeepRow) + page.findChildren(WarningRow)
    assert len(held) == 2
    tab.refresh()
    tab.refresh()
    rows = panel.findChildren(KeepRow) + page.findChildren(WarningRow)
    assert len(panel.findChildren(KeepRow)) == 1
    assert len(page.findChildren(WarningRow)) == 1
    assert not set(rows) & set(held)
    assert not panel.add_button.isHidden()


def test_add_range_puts_a_minute_in_the_largest_skip_gap(tab, controller):
    calls = Calls(controller, "set_time_ranges")
    tab.inspector_panel().add_button.click()
    assert calls.calls == [(NAME, [("2:33", "23:05"), ("24:36", "25:36")])]


def test_add_range_skips_a_gap_too_small_to_hold_a_range():
    """The 0.4 s between these two keeps cannot hold a range at all: adding a
    zero-length one there is worse than adding none."""
    keeps = [(0.0, 100.0), (100.4, 200.0)]
    assert with_added_range(keeps, 200.0) == keeps


def test_add_range_never_starts_inside_the_keep_before_it():
    assert with_added_range([(0.0, 2.5)], 40.0) == [(0.0, 2.5), (3.0, 40.0)]


def test_add_range_on_a_whole_file_keeps_the_first_minute(controller):
    give_values(controller, keep=())
    made = RangesTab(controller)
    made.set_file(NAME)
    settle()
    calls = Calls(controller, "set_time_ranges")
    made.inspector_panel().add_button.click()
    assert calls.calls == [(NAME, [(None, "1:00")])]


def test_use_whole_file_commits_none(tab, controller):
    calls = Calls(controller, "set_time_ranges")
    tab.inspector_panel().whole_file_button.click()
    assert calls.calls == [(NAME, None)]
    assert controller.entry(NAME).time_ranges.source == Source.MANUAL
    assert controller.entry(NAME).time_ranges.ranges == []


# --------------------------------------------------------------------------
# The header
# --------------------------------------------------------------------------

def test_the_header_names_the_file_its_duration_and_the_other_episodes(tab):
    assert tab.header_text() == "ep01.mkv · 27:08 · other episodes 23:38 – 27:08"


def test_the_header_shows_one_known_duration_once(controller):
    give_values(controller)
    for other in OTHERS:
        controller.entry(other).media.duration = 1418.0
    made = RangesTab(controller)
    made.set_file(NAME)
    settle()
    assert made.header_text() == "ep01.mkv · 27:08 · other episodes 23:38"


def test_the_header_drops_the_other_episodes_when_none_is_known(controller):
    give_values(controller)
    for other in OTHERS:
        controller.entry(other).media.duration = 0.0
    made = RangesTab(controller)
    made.set_file(NAME)
    settle()
    assert made.header_text() == "ep01.mkv · 27:08"


def test_the_header_is_the_stage_head_toolbar(controller):
    give_values(controller)
    stage = Stage(controller, evidence_tabs(controller))
    try:
        ranges_tab = stage.tabs()[2]
        assert isinstance(ranges_tab, RangesTab)
        stage.set_current(2)
        assert stage.current_toolbar() is ranges_tab.toolbar()
        assert ranges_tab.toolbar().parentWidget() is stage.head()
    finally:
        stage.deleteLater()


# --------------------------------------------------------------------------
# Compact mode
# --------------------------------------------------------------------------

@pytest.fixture
def compact(controller):
    give_values(controller, crop=True)
    made = Timeline(controller, mode="compact")
    made.resize(800, 68)
    made.show()
    made.set_file(NAME)
    settle()
    yield made
    made.close()
    made.deleteLater()


def test_compact_mode_draws_the_same_spans_without_grips(compact):
    assert [span.keep for span in compact.spans()] == [False, True, False]
    assert compact.grips() == []
    assert compact.height() == 68


def test_compact_mode_drops_the_lane_label(compact, tab):
    assert tab.timeline.lane_label() == "speech"
    assert compact.lane_label() == ""


def test_compact_mode_marks_the_crop_samples(compact):
    assert compact.sample_marks() == [120.0, 600.0, 1200.0]


def test_compact_mode_is_read_only(compact, controller):
    calls = Calls(controller, "set_time_ranges")
    drag_grip(compact, INTRO_END, compact.x_for(300.0))
    assert calls.calls == []
    assert compact.spans()[1].start == pytest.approx(INTRO_END)


def test_a_compact_click_asks_for_a_seek_and_remembers_the_position(compact):
    seeks: list[float] = []
    compact.seek_requested.connect(seeks.append)
    click_at(compact, 900.0)
    assert seeks == [pytest.approx(900.0, abs=3.0)]
    assert compact.position() == pytest.approx(900.0, abs=3.0)


def test_a_compact_double_click_asks_for_a_pin(compact):
    pins: list[float] = []
    compact.pin_requested.connect(pins.append)
    double_click(compact, QPointF(compact.x_for(640.0), 26.0))
    assert pins == [pytest.approx(640.0, abs=3.0)]


def test_the_edit_timeline_ignores_clicks_away_from_a_grip(tab):
    seeks: list[float] = []
    tab.timeline.seek_requested.connect(seeks.append)
    click_at(tab.timeline, 900.0)
    assert seeks == []


# --------------------------------------------------------------------------
# The other two tabs mount it (ruling B5)
# --------------------------------------------------------------------------

def test_the_crop_tab_selects_the_nearest_sample_on_a_seek(controller):
    give_values(controller, crop=True)
    tabs = evidence_tabs(controller)
    crop_tab = tabs[0]
    assert isinstance(crop_tab, CropTab)
    crop_tab.page().resize(880, 620)
    crop_tab.set_file(NAME)
    crop_tab.timeline.resize(800, 68)
    settle()
    assert crop_tab.timeline.mode == "compact"
    click_at(crop_tab.timeline, 1100.0)
    settle()
    assert crop_tab.selected_index() == 2                     # 1200 s is nearest to 1100 s
    click_at(crop_tab.timeline, 300.0)
    settle()
    assert crop_tab.selected_index() == 0                     # 120 s


def test_the_brightness_tab_pins_a_tile_on_a_double_click(controller):
    give_values(controller, crop=True)
    tabs = evidence_tabs(controller)
    brightness_tab = tabs[1]
    brightness_tab.page().resize(880, 620)
    brightness_tab.set_file(NAME)
    timeline = brightness_tab.timeline_slot()
    timeline.resize(800, 68)
    settle()
    double_click(timeline, QPointF(timeline.x_for(640.0), 26.0))
    settle()
    assert brightness_tab.pinned_times() == [pytest.approx(640.0, abs=3.0)]


def test_the_brightness_pin_tile_uses_the_timelines_last_click(controller):
    give_values(controller, crop=True)
    tabs = evidence_tabs(controller)
    brightness_tab = tabs[1]
    brightness_tab.page().resize(880, 620)
    brightness_tab.set_file(NAME)
    timeline = brightness_tab.timeline_slot()
    timeline.resize(800, 68)
    settle()
    click_at(timeline, 420.0)
    brightness_tab.pin_tile().clicked.emit()
    settle()
    assert brightness_tab.pinned_times() == [pytest.approx(420.0, abs=3.0)]


def test_evidence_tabs_ends_with_the_real_time_ranges_tab(controller):
    tabs = evidence_tabs(controller)
    assert [tab.title for tab in tabs] == ["Crop", "Brightness", "Time ranges"]
    assert isinstance(tabs[2], RangesTab)
    assert all(isinstance(tab, StageTab) for tab in tabs)


def test_the_tab_follows_the_selected_file(controller):
    give_values(controller)
    give_values(controller, OTHERS[0], keep=((60.0, 600.0),), duration=1418.0)
    made = RangesTab(controller)
    made.set_file(NAME)
    settle()
    assert made.timeline.spans()[1].start == pytest.approx(INTRO_END)
    made.set_file(OTHERS[0])
    settle()
    assert made.current_file() == OTHERS[0]
    assert made.timeline.duration() == pytest.approx(1418.0)
    assert made.timeline.spans()[1].start == pytest.approx(60.0)
    made.set_file(None)
    settle()
    assert made.timeline.spans() == []
    assert made.inspector_panel().keep_rows() == []       # no file: nothing to keep
    assert made.header_text() == ""
    assert not made.page().isEnabled()
