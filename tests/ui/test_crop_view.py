"""Plan 3C Task 2: the Crop review view (`app/views/crop_view.py`).

Nothing is decoded. `fake_runner` (tests/ui/conftest.py) records the frame
jobs the view asks for and the test hands back a synthetic numpy frame, so
the canvas draws real pixels without a video file. Mouse gestures are sent
as QMouseEvents (deterministic, unlike a synthetic cursor), keys with QTest.

The evidence in `crop_evidence()` is the shape `CropResult.to_evidence()`
writes: 1920x888 frames, 13 raw samples over 12 distinct times (the
full-frame retry re-probes a time it already had, leaving one empty and one
kept entry for it), 11 of them kept, one disagreeing sample sitting lower
and one two-line sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, QSettings, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget

from app import masking
from app.controller import ProjectController
from app.main_window import MainWindow
from app.views.crop_view import CropTab
from app.views.stage import Stage, StageTab
from app.views.tabs import evidence_tabs
from core.detect import crop as crop_mod
from core.detect import ocr_view
from core.jobs.view_jobs import FramesResult
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

NAMES = ["ZS2_-_11_[1080p]TXHBR.mp4", "ZS2_-_12_[1080p]TXHBR.mp4"]
FRAME_SIZE = (1920, 888)
BOX = (288, 784, 1344, 55)
ENVELOPE = (300, 791, 1320, 45)
BRIGHTNESS = 214
SAMPLE_TIMES = [60.0 * step for step in range(1, 13)]
LOWER_INDEX = 4                     # 5:00 -- not kept, text sits lower than the envelope
TWO_LINE_INDEX = 7                  # 8:00 -- two text rows
CANVAS_SIZE = (800, 400)
COMMIT_WAIT_MS = 700                # > crop_view.COMMIT_DEBOUNCE_MS


# --------------------------------------------------------------------------
# Evidence and project fixtures
# --------------------------------------------------------------------------

def line_boxes(y: int = 791, height: int = 45, lines: int = 1):
    if lines == 2:
        return ((320, y - 50, 1280, 44), (300, y, 1320, height))
    return ((300, y, 1320, height),)


def sample(time: float, *, kept: bool = True, boxes=None, lines: int = 1) -> dict:
    boxes = line_boxes() if boxes is None else boxes
    return {"time": float(time), "boxes": [list(box) for box in boxes], "kept": kept, "lines": lines}


def default_samples() -> list[dict]:
    """13 raw entries over 12 times, 11 kept -- see the module docstring."""
    raw = [sample(time) for time in SAMPLE_TIMES]
    raw[LOWER_INDEX] = sample(SAMPLE_TIMES[LOWER_INDEX], kept=False, boxes=((310, 845, 1300, 40),))
    raw[TWO_LINE_INDEX] = sample(SAMPLE_TIMES[TWO_LINE_INDEX], boxes=line_boxes(lines=2), lines=2)
    # The full-frame retry's empty probe of a time the bottom-band round kept.
    return [sample(SAMPLE_TIMES[0], kept=False, boxes=())] + raw


def crop_evidence(*, box=BOX, envelope=ENVELOPE, samples=None, cutoff: float | None = 0.55,
                  flagged: str | None = None, frame_size=FRAME_SIZE) -> dict:
    samples = default_samples() if samples is None else samples
    kept = [item for item in samples if item["kept"]]
    evidence = {
        "box": None if box is None else list(box),
        "envelope": None if envelope is None else list(envelope),
        "agreed": len(kept),
        "probes_used": len(samples),
        "flagged": flagged,
        "hit_pts": [item["time"] for item in kept],
        "frame_size": list(frame_size),
        "samples": samples,
    }
    if cutoff is not None:
        evidence["cutoff_frac"] = cutoff
    return evidence


def make_entry(name: str, *, crop=BOX, source=Source.MANUAL) -> FileEntry:
    item = FileEntry(name, media=Media(*FRAME_SIZE, 1628.0, 25.0), review=ReviewState.REVIEWED,
                     sample_time=578.0)
    if crop is not None:
        item.crop = Crop(*crop, source)
    item.brightness = Brightness(BRIGHTNESS, Source.MANUAL)
    item.time_ranges = TimeRanges([], Source.MANUAL)
    return item


def frame_array(width: int = 1557, height: int = 720) -> np.ndarray:
    """A synthetic BGR frame, smaller than the native 1920x888 as FrameJob's
    720-row cap makes real ones."""
    image = np.zeros((height, width, 3), np.uint8)
    image[:, :, 0] = 60
    image[int(height * 0.85):, :, :] = 250          # a bright "subtitle" band the mask keeps
    return image


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

@dataclass
class Harness:
    tab: CropTab
    controller: ProjectController
    runner: object
    name: str

    def entry(self):
        return self.controller.entry(self.name)

    def crop(self) -> tuple[int, int, int, int] | None:
        crop = self.entry().crop
        return None if crop is None else (crop.x, crop.y, crop.width, crop.height)

    def deliver_frames(self, array: np.ndarray | None = None) -> None:
        array = frame_array() if array is None else array
        for submission in self.runner.of_kind("frames", self.name):
            if not self.runner.ended(submission):
                self.runner.finish(submission, FramesResult(self.name, dict.fromkeys(submission.job.times,
                                                                                     array)))
        self.controller.drain_events()
        QApplication.processEvents()

    def requested_times(self) -> set[float]:
        return {time for submission in self.runner.of_kind("frames", self.name)
                for time in submission.job.times}

    def size_canvas(self, width: int = CANVAS_SIZE[0], height: int = CANVAS_SIZE[1]) -> None:
        self.tab.canvas.resize(width, height)


@pytest.fixture(autouse=True)
def settings_dir(tmp_path):
    """No test may read or write the user's real QSettings."""
    path = tmp_path / "qsettings"
    path.mkdir()
    for fmt in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, str(path))
    return path


@pytest.fixture
def make_tab(qapp, fake_runner, tmp_project):
    made: list[Harness] = []

    def make(*, crop=BOX, source=Source.MANUAL, evidence=..., labels_enabled=False, **folder) -> Harness:
        path = tmp_project(NAMES)
        entries = [make_entry(name, crop=crop, source=source) for name in NAMES]
        save_project(Project(path=str(path),
                             folder=FolderSettings(labels_enabled=labels_enabled, **folder),
                             files={item.name: item for item in entries}))
        controller = ProjectController(fake_runner, save_debounce_ms=10)
        controller.open_folder(str(path))
        built = crop_evidence() if evidence is ... else evidence
        if built is not None:
            for name in NAMES:
                controller.entry(name).evidence["crop"] = built
        tab = CropTab(controller)
        page = tab.page()
        page.resize(960, 560)
        page.show()
        QApplication.processEvents()
        tab.set_file(NAMES[0])
        QApplication.processEvents()
        harness = Harness(tab, controller, fake_runner, NAMES[0])
        harness.size_canvas()
        made.append(harness)
        return harness

    yield make
    for harness in made:
        harness.tab.page().close()
        harness.tab.page().deleteLater()
        harness.controller.shutdown(timeout=0.5)


# --------------------------------------------------------------------------
# Mouse and key helpers
# --------------------------------------------------------------------------

def _send(widget, kind, point: QPointF, button, buttons, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
    event = QMouseEvent(kind, point, point, QPointF(widget.mapToGlobal(point.toPoint())),
                        button, buttons, modifiers)
    QApplication.sendEvent(widget, event)


def press(widget, point: QPointF, button=Qt.MouseButton.LeftButton) -> None:
    _send(widget, QEvent.Type.MouseButtonPress, point, button, button)


def drag_to(widget, point: QPointF, button=Qt.MouseButton.LeftButton) -> None:
    _send(widget, QEvent.Type.MouseMove, point, Qt.MouseButton.NoButton, button)


def release(widget, point: QPointF, button=Qt.MouseButton.LeftButton) -> None:
    _send(widget, QEvent.Type.MouseButtonRelease, point, button, Qt.MouseButton.NoButton)


def gesture(widget, start: QPointF, end: QPointF, button=Qt.MouseButton.LeftButton) -> None:
    press(widget, start, button)
    drag_to(widget, end, button)
    release(widget, end, button)


def arrow(canvas, key, shift: bool = False) -> None:
    modifier = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    QTest.keyClick(canvas, key, modifier)


# --------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------

def test_widget_and_video_coordinates_convert_at_two_sizes(make_tab):
    canvas = make_tab().tab.canvas
    for width, height in ((800, 400), (541, 317)):
        canvas.resize(width, height)
        rect = canvas.frame_rect()
        assert rect.width() / rect.height() == pytest.approx(FRAME_SIZE[0] / FRAME_SIZE[1], rel=1e-6)
        assert rect.width() <= width + 1e-6 and rect.height() <= height + 1e-6
        assert canvas.to_widget(0, 0).x() == pytest.approx(rect.x())
        assert canvas.to_widget(0, 0).y() == pytest.approx(rect.y())
        assert canvas.to_widget(*FRAME_SIZE).x() == pytest.approx(rect.right())
        for point in ((0.0, 0.0), (960.0, 444.0), (1919.0, 887.0)):
            back = canvas.to_video(canvas.to_widget(*point))
            assert back[0] == pytest.approx(point[0], abs=1e-6)
            assert back[1] == pytest.approx(point[1], abs=1e-6)


def test_the_canvas_uses_the_native_media_size_not_the_frames_size(make_tab):
    harness = make_tab()
    harness.deliver_frames()
    assert harness.tab.canvas.video_size() == FRAME_SIZE


# --------------------------------------------------------------------------
# Editing: drag, handles, nudge, clamping
# --------------------------------------------------------------------------

def test_dragging_the_body_moves_the_box_and_commits_on_release(make_tab):
    harness = make_tab()
    canvas = harness.tab.canvas
    start = canvas.to_widget(BOX[0] + BOX[2] / 2, BOX[1] + BOX[3] / 2)
    end = canvas.to_widget(BOX[0] + BOX[2] / 2 - 40, BOX[1] + BOX[3] / 2 - 25)
    press(canvas, start)
    drag_to(canvas, end)
    assert canvas.box() == (BOX[0] - 40, BOX[1] - 25, BOX[2], BOX[3])
    assert harness.crop() == BOX                       # nothing committed while dragging
    release(canvas, end)
    assert harness.crop() == (BOX[0] - 40, BOX[1] - 25, BOX[2], BOX[3])
    assert harness.entry().crop.source == Source.MANUAL


@pytest.mark.parametrize(("handle", "expected"), [
    ("tl", (288 - 30, 784 - 20, 1344 + 30, 55 + 20)),
    ("tr", (288, 784 - 20, 1344 - 30, 55 + 20)),
    ("bl", (288 - 30, 784, 1344 + 30, 55 - 20)),
    ("br", (288, 784, 1344 - 30, 55 - 20)),
    ("tc", (288, 784 - 20, 1344, 55 + 20)),
    ("bc", (288, 784, 1344, 55 - 20)),
])
def test_each_handle_resizes_its_own_edges(make_tab, handle, expected):
    harness = make_tab()
    canvas = harness.tab.canvas
    start = canvas.handle_rect(handle).center()
    end = canvas.to_widget(canvas.to_video(start)[0] - 30, canvas.to_video(start)[1] - 20)
    gesture(canvas, start, end)
    assert canvas.box() == expected
    assert harness.crop() == expected


@pytest.mark.parametrize(("handle", "expected"), [("tc", (288, 796, 1344, 43)), ("bc", (288, 784, 1344, 67))])
def test_the_centre_handles_change_height_only(make_tab, handle, expected):
    harness = make_tab()
    canvas = harness.tab.canvas
    start = canvas.handle_rect(handle).center()
    end = canvas.to_widget(canvas.to_video(start)[0] + 120, canvas.to_video(start)[1] + 12)
    gesture(canvas, start, end)
    assert canvas.box() == expected                    # a big horizontal move changes neither x nor width


def test_all_four_arrows_nudge_one_pixel_on_the_canvas_and_shift_ten(make_tab):
    harness = make_tab()
    canvas = harness.tab.canvas
    arrow(canvas, Qt.Key.Key_Up)
    assert canvas.box() == (BOX[0], BOX[1] - 1, BOX[2], BOX[3])
    arrow(canvas, Qt.Key.Key_Right)                     # horizontal nudging is 1 px too
    assert canvas.box() == (BOX[0] + 1, BOX[1] - 1, BOX[2], BOX[3])
    arrow(canvas, Qt.Key.Key_Left)
    arrow(canvas, Qt.Key.Key_Down, shift=True)
    assert canvas.box() == (BOX[0], BOX[1] + 9, BOX[2], BOX[3])
    arrow(canvas, Qt.Key.Key_Right, shift=True)
    assert canvas.box() == (BOX[0] + 10, BOX[1] + 9, BOX[2], BOX[3])
    assert harness.tab.selected_index() == 0            # the canvas never steps samples


def test_a_burst_of_nudges_commits_once(make_tab):
    harness = make_tab()
    canvas = harness.tab.canvas
    calls = []
    original = harness.controller.set_crop

    def record(name, box):
        calls.append(box)
        original(name, box)

    harness.controller.set_crop = record
    for _ in range(4):
        arrow(canvas, Qt.Key.Key_Up)
    assert calls == []                                  # debounced
    QTest.qWait(COMMIT_WAIT_MS)
    assert calls == [(BOX[0], BOX[1] - 4, BOX[2], BOX[3])]
    assert harness.crop() == (BOX[0], BOX[1] - 4, BOX[2], BOX[3])


def test_a_frame_arriving_does_not_undo_an_uncommitted_nudge(make_tab):
    harness = make_tab()
    canvas = harness.tab.canvas
    arrow(canvas, Qt.Key.Key_Up)
    harness.deliver_frames()                            # frame_ready refreshes the whole tab
    assert canvas.box() == (BOX[0], BOX[1] - 1, BOX[2], BOX[3])
    assert harness.tab.panel.spin_values() == (BOX[0], BOX[1] - 1, BOX[2], BOX[3])
    QTest.qWait(COMMIT_WAIT_MS)
    assert harness.crop() == (BOX[0], BOX[1] - 1, BOX[2], BOX[3])


def test_the_box_clamps_to_the_frame_edges(make_tab):
    harness = make_tab()
    canvas = harness.tab.canvas
    centre = canvas.to_widget(BOX[0] + BOX[2] / 2, BOX[1] + BOX[3] / 2)
    gesture(canvas, centre, canvas.to_widget(-4000, 4000))
    x, y, width, height = canvas.box()
    assert (x, y, width, height) == (0, FRAME_SIZE[1] - BOX[3], BOX[2], BOX[3])

    harness.controller.set_crop(harness.name, BOX)
    harness.tab.refresh()
    start = canvas.handle_rect("bc").center()
    gesture(canvas, start, canvas.to_widget(BOX[0], -4000))      # drag the bottom edge above the top
    _x, _y, _w, height = canvas.box()
    assert height == canvas.MIN_BOX


# --------------------------------------------------------------------------
# Toolbar
# --------------------------------------------------------------------------

def test_the_toolbar_toggles_change_the_painted_overlays(make_tab):
    harness = make_tab()
    tab, canvas = harness.tab, harness.tab.canvas
    assert canvas.overlays() == {"envelope": True, "masked": False, "grid": False}
    assert tab.envelope_button.property("toggled") is True
    for button, key in ((tab.grid_button, "grid"), (tab.masked_button, "masked")):
        button.click()
        assert canvas.overlays()[key] is True
        assert button.property("toggled") is True
    tab.envelope_button.click()
    assert canvas.overlays() == {"envelope": False, "masked": True, "grid": True}
    canvas.grab()                                       # every combination paints


def test_masked_previews_the_mask_with_the_files_brightness(make_tab, monkeypatch):
    harness = make_tab()
    harness.deliver_frames()
    thresholds = []
    real = ocr_view.mask

    def spy(region, threshold):
        thresholds.append(int(threshold))
        return real(region, threshold)

    monkeypatch.setattr(ocr_view, "mask", spy)
    harness.tab.canvas.grab()
    assert thresholds == []                             # masked is off by default
    harness.tab.masked_button.click()
    harness.tab.canvas.grab()
    assert thresholds and set(thresholds) == {BRIGHTNESS}


def test_fit_to_all_samples_aggregates_the_kept_boxes_and_commits(make_tab, monkeypatch):
    harness = make_tab()
    calls = []

    def spy(polys, frame_size, band_frac=0.55, settings=None, sample_times=None):
        calls.append((polys, tuple(frame_size), dict(settings or {}), list(sample_times or [])))
        return (200, 700, 1500, 90)

    monkeypatch.setattr(crop_mod, "aggregate_box", spy)
    assert harness.tab.fit_button.text() == "⤢ fit to all 11 samples"
    assert harness.tab.fit_button.isEnabled()
    harness.tab.fit_button.click()
    polys, frame_size, settings, times = calls[0]
    kept = [item for item in harness.tab.samples() if item.kept]
    assert len(polys) == 11
    assert times == [item.time for item in kept]
    assert frame_size == FRAME_SIZE
    assert settings["bottom_half_cutoff"] == 0.55
    assert settings["crop_width_fraction"] == FolderSettings().crop_width_fraction
    assert polys[0] == [[(300, 791), (1620, 791), (1620, 836), (300, 836)]]
    assert harness.crop() == (200, 700, 1500, 90)
    assert harness.entry().crop.source == Source.MANUAL


def test_fit_uses_the_cutoff_the_detection_used(make_tab, monkeypatch):
    calls = []

    def spy(_polys, _frame_size, band_frac=0.55, settings=None, sample_times=None):
        calls.append(dict(settings or {}))          # nothing to fit: no box is committed

    monkeypatch.setattr(crop_mod, "aggregate_box", spy)
    retry = make_tab(evidence=crop_evidence(cutoff=0.0))
    retry.tab.fit_button.click()
    assert calls[-1]["bottom_half_cutoff"] == 0.0
    older = make_tab(evidence=crop_evidence(cutoff=None), bottom_half_cutoff=0.6)
    older.tab.fit_button.click()
    assert calls[-1]["bottom_half_cutoff"] == 0.6       # falls back to the folder setting


def test_fit_is_disabled_without_kept_samples(make_tab):
    harness = make_tab(evidence=crop_evidence(samples=[sample(60.0, kept=False, boxes=())]))
    assert not harness.tab.fit_button.isEnabled()
    assert harness.tab.fit_button.text() == "⤢ fit to all 0 samples"


# --------------------------------------------------------------------------
# Filmstrip
# --------------------------------------------------------------------------

def test_samples_are_deduplicated_by_time_preferring_the_kept_entry(make_tab):
    harness = make_tab()
    samples = harness.tab.samples()
    assert len(samples) == 12
    assert [item.time for item in samples] == SAMPLE_TIMES
    assert sum(1 for item in samples if item.kept) == 11
    assert samples[0].kept and samples[0].boxes                 # the kept entry won, not the empty retry
    assert samples[TWO_LINE_INDEX].lines == 2


def test_the_filmstrip_shows_eight_thumbnails_and_pages(make_tab):
    harness = make_tab()
    strip = harness.tab.strip
    assert strip.label_text() == "samples"
    assert len(strip.thumbnails()) == 8
    assert strip.more_button.text() == "more ▸"
    assert strip.more_button.isVisible()
    assert [thumb.sample_index() for thumb in strip.thumbnails()] == list(range(8))
    strip.more_button.click()
    assert [thumb.sample_index() for thumb in strip.thumbnails()] == list(range(8, 12))
    strip.more_button.click()
    assert [thumb.sample_index() for thumb in strip.thumbnails()] == list(range(8))


def test_the_selected_and_disagreeing_samples_are_bordered(make_tab):
    harness = make_tab()
    strip = harness.tab.strip
    assert harness.tab.selected_index() == 0                   # the first kept sample
    tones = {thumb.sample_index(): thumb.tone() for thumb in strip.thumbnails()}
    assert tones[0] == "selected"
    assert tones[LOWER_INDEX] == "warn"
    assert tones[1] == ""


def test_no_sample_is_disagreeing_when_the_detection_found_no_box(make_tab):
    harness = make_tab(evidence=crop_evidence(box=None, envelope=None, flagged="static-content"))
    tones = {thumb.sample_index(): thumb.tone() for thumb in harness.tab.strip.thumbnails()}
    assert "warn" not in tones.values()
    assert any("static-content" in text for _key, text in harness.tab.panel.rows())


def test_clicking_a_thumbnail_selects_it_and_takes_the_keyboard(make_tab):
    harness = make_tab()
    strip = harness.tab.strip
    harness.tab.canvas.setFocus()
    gesture(strip.thumbnails()[3], QPointF(10, 10), QPointF(10, 10))
    assert harness.tab.selected_index() == 3
    assert harness.tab.current_time() == SAMPLE_TIMES[3]
    assert strip.hasFocus()                             # ◀ / ▶ carry on from the strip


def test_the_filmstrip_steps_samples_with_the_arrow_keys(make_tab):
    harness = make_tab()
    strip = harness.tab.strip
    harness.tab.select(3)
    arrow(strip, Qt.Key.Key_Right)
    assert harness.tab.selected_index() == 4
    for _ in range(2):
        arrow(strip, Qt.Key.Key_Left)
    assert harness.tab.selected_index() == 2
    arrow(strip, Qt.Key.Key_Left, shift=True)           # Shift changes nothing on the strip
    assert harness.tab.selected_index() == 1
    assert harness.tab.canvas.box() == BOX              # the strip never nudges the box


def test_tab_walks_from_the_canvas_to_the_filmstrip(make_tab):
    harness = make_tab()
    canvas, strip = harness.tab.canvas, harness.tab.strip
    canvas.setFocus()
    assert harness.tab.page().focusNextPrevChild(True)
    assert strip.hasFocus()


def test_the_two_line_sample_draws_two_bars(make_tab):
    harness = make_tab()
    harness.tab.select(TWO_LINE_INDEX)
    thumb = next(t for t in harness.tab.strip.thumbnails() if t.sample_index() == TWO_LINE_INDEX)
    assert thumb.lines() == 2
    assert thumb.grab().width() == harness.tab.strip.THUMB_WIDTH + 2


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

def test_request_frames_covers_every_sample_time_on_set_file(make_tab):
    harness = make_tab()
    assert harness.requested_times() >= set(SAMPLE_TIMES)


def test_the_canvas_shows_the_placeholder_until_the_frame_arrives(make_tab):
    harness = make_tab()
    assert harness.tab.canvas.placeholder_text() == "loading frame…"
    harness.deliver_frames()
    assert harness.tab.canvas.placeholder_text() == ""
    harness.tab.canvas.grab()


def test_an_unreadable_time_is_asked_for_once(make_tab):
    harness = make_tab()
    for submission in harness.runner.of_kind("frames", harness.name):
        harness.runner.finish(submission, FramesResult(harness.name, {}))
    harness.controller.drain_events()
    before = len(harness.runner.of_kind("frames", harness.name))
    harness.tab.refresh()
    harness.tab.select(3)
    assert len(harness.runner.of_kind("frames", harness.name)) == before
    assert harness.tab.canvas.placeholder_text() == "loading frame…"


# --------------------------------------------------------------------------
# Inspector panel
# --------------------------------------------------------------------------

def test_the_panel_shows_the_box_evidence_and_the_nudge_note(make_tab):
    harness = make_tab()
    panel = harness.tab.panel
    assert panel.spin_values() == BOX
    rows = dict(panel.rows())
    assert rows["Samples with text"] == "11 / 12"
    assert rows["Text envelope"] == "y 791–836"
    assert panel.nudge_note() == ("Arrow keys nudge 1 px, ⇧ arrows nudge 10. "
                                  "Free rectangle — no forced centring.")
    assert "◆ this episode only" not in [key.lower() for key, _value in panel.rows()]


def test_the_spin_pairs_commit_through_set_crop(make_tab):
    harness = make_tab()
    panel = harness.tab.panel
    panel.set_spin_values(300, 1300, 800, 60)
    QTest.qWait(COMMIT_WAIT_MS)
    assert harness.crop() == (300, 800, 1300, 60)
    assert harness.entry().crop.source == Source.MANUAL
    assert harness.tab.canvas.box() == (300, 800, 1300, 60)


def test_a_disagreeing_sample_gets_a_warn_row_that_selects_it(make_tab):
    harness = make_tab()
    panel = harness.tab.panel
    rows = dict(panel.rows())
    assert "1 sample sits lower" in rows
    assert rows["1 sample sits lower"] == "05:00 ▸"
    assert panel.evidence_note() == "This sample falls outside the box."
    panel.click_row("1 sample sits lower")
    assert harness.tab.selected_index() == LOWER_INDEX


def test_a_covered_disagreeing_sample_reads_as_covered(make_tab):
    covered = crop_evidence(samples=default_samples())
    covered["samples"][1 + LOWER_INDEX] = sample(SAMPLE_TIMES[LOWER_INDEX], kept=False,
                                                 boxes=((320, 790, 1200, 30),))
    harness = make_tab(evidence=covered)
    assert dict(harness.tab.panel.rows())["1 sample sits higher"] == "05:00 ▸"
    assert harness.tab.panel.evidence_note() == ("The amber box already covers it. "
                                                 "Click the warned sample to inspect.")


def test_without_evidence_the_panel_names_the_source(make_tab):
    imported = make_tab(evidence=None, source=Source.IMPORTED)
    assert dict(imported.tab.panel.rows())["Source"] == "imported from the previous version"
    mine = make_tab(evidence=None, source=Source.MANUAL)
    assert dict(mine.tab.panel.rows())["Source"] == "set by you"


def test_a_detection_that_differs_from_your_box_is_shown_as_the_detection(make_tab):
    harness = make_tab(crop=(300, 800, 1300, 60), source=Source.MANUAL)
    rows = dict(harness.tab.panel.rows())
    assert rows["detected"] == "288, 784 · 1344 × 55 · yours 300, 800 · 1300 × 60"
    assert harness.tab.canvas.detected_box() == BOX
    assert harness.tab.canvas.box() == (300, 800, 1300, 60)


# --------------------------------------------------------------------------
# Label masks
# --------------------------------------------------------------------------

def test_right_drag_adds_a_label_mask_and_right_click_removes_it(make_tab):
    harness = make_tab(labels_enabled=True)
    canvas = harness.tab.canvas
    gesture(canvas, canvas.to_widget(400, 100), canvas.to_widget(700, 200),
            Qt.MouseButton.RightButton)
    assert harness.controller.project.folder.label_mask_crops == [(400, 100, 300, 100)]
    inside = canvas.to_widget(500, 150)
    gesture(canvas, inside, inside, Qt.MouseButton.RightButton)
    assert harness.controller.project.folder.label_mask_crops == []


def test_label_masks_are_hidden_when_the_folder_has_labels_off(make_tab):
    harness = make_tab(labels_enabled=False)
    canvas = harness.tab.canvas
    assert not canvas.masks_enabled()
    gesture(canvas, canvas.to_widget(400, 100), canvas.to_widget(700, 200),
            Qt.MouseButton.RightButton)
    assert harness.controller.project.folder.label_mask_crops == []


def test_label_masks_are_drawn_on_every_file(make_tab):
    harness = make_tab(labels_enabled=True, label_mask_crops=[(10, 20, 30, 40)])
    assert harness.tab.canvas.masks() == [(10, 20, 30, 40)]
    harness.tab.set_file(NAMES[1])
    assert harness.tab.canvas.masks() == [(10, 20, 30, 40)]


# --------------------------------------------------------------------------
# Tags and the timeline placeholder
# --------------------------------------------------------------------------

def test_the_canvas_tags_name_the_frame_the_crop_and_the_envelope(make_tab):
    harness = make_tab()
    tags = harness.tab.canvas.tags()
    assert tags["top_left"] == "1920 × 888 · t 01:00.00"
    assert tags["top_right"] == "crop 288, 784 · 1344 × 55"
    assert tags["bottom_right"] == "dashed = text found across all 11 samples"
    harness.tab.envelope_button.click()
    assert "bottom_right" not in harness.tab.canvas.tags()


def test_the_page_leaves_room_for_the_compact_timeline(make_tab):
    placeholder = make_tab().tab.timeline_placeholder
    assert placeholder.height() == 68


# --------------------------------------------------------------------------
# The stage tab factory
# --------------------------------------------------------------------------

def test_evidence_tabs_gives_the_three_stage_tabs(qapp, fake_runner):
    controller = ProjectController(fake_runner, save_debounce_ms=10)
    try:
        tabs = evidence_tabs(controller)
        assert [tab.title for tab in tabs] == ["Crop", "Brightness", "Time ranges"]
        assert isinstance(tabs[0], CropTab)
        assert all(isinstance(tab, StageTab) for tab in tabs)
    finally:
        controller.shutdown(timeout=0.5)


def test_the_window_can_be_built_with_the_evidence_tabs(qapp, fake_runner, tmp_project, tmp_path):
    controller = ProjectController(fake_runner, save_debounce_ms=10)
    window = MainWindow(controller, tabs_factory=evidence_tabs)
    try:
        window.resize(1440, 900)
        path = tmp_project(NAMES)
        save_project(Project(path=str(path), folder=FolderSettings(labels_enabled=False),
                             files={name: make_entry(name) for name in NAMES}))
        window.open_folder(str(path))
        QApplication.processEvents()
        assert isinstance(window.stage.tabs()[0], CropTab)
        assert window.stage.current_tab().title == "Crop"
        window.stage.set_file(NAMES[0])
        QApplication.processEvents()
        assert window.stage.tabs()[0].canvas.box() == BOX
    finally:
        window.close()
        window.deleteLater()
        controller.shutdown(timeout=0.5)


def test_the_toolbar_is_mounted_in_the_stage_head_and_swaps_with_the_tab(qapp, fake_runner, tmp_project):
    """Ruling B3: the per-tab controls live in `.stage-head`, right of the
    tab buttons -- and a tab without a toolbar leaves the head clean."""
    controller = ProjectController(fake_runner, save_debounce_ms=10)
    stage = Stage(controller, evidence_tabs(controller))
    try:
        crop_tab = stage.tabs()[0]
        toolbar = crop_tab.toolbar()
        assert stage.current_toolbar() is toolbar
        assert toolbar.parentWidget() is stage.head()
        for button in (crop_tab.envelope_button, crop_tab.fit_button):
            assert button.parentWidget() is toolbar
        stage.set_current(1)                            # a PlaceholderTab: no toolbar
        assert stage.current_toolbar() is None
        assert toolbar.isHidden()
        stage.set_current(0)
        assert stage.current_toolbar() is toolbar
    finally:
        stage.deleteLater()
        controller.shutdown(timeout=0.5)


def test_a_tab_without_a_toolbar_attribute_still_works(qapp, fake_runner):
    """`toolbar()` is optional: the Stage asks with getattr, so a tab written
    before it (or a test's stand-in) is fine."""
    class Bare:
        title = "Bare"

        def __init__(self):
            self._page, self._panel = QWidget(), QWidget()

        def page(self):
            return self._page

        def inspector_panel(self):
            return self._panel

        def set_file(self, name):
            pass

        def refresh(self):
            pass

    controller = ProjectController(fake_runner, save_debounce_ms=10)
    stage = Stage(controller, [Bare(), Bare()])
    try:
        assert stage.current_toolbar() is None
        stage.set_current(1)
        assert stage.current_toolbar() is None
    finally:
        stage.deleteLater()
        controller.shutdown(timeout=0.5)


# --------------------------------------------------------------------------
# The core bridge (app/masking.py)
# --------------------------------------------------------------------------

def test_masking_turns_boxes_into_polygons_for_the_detector(monkeypatch):
    calls = []

    def spy(polys, frame_size, band_frac=0.55, settings=None, sample_times=None):
        calls.append((polys, frame_size, settings, sample_times))
        return [200, 700, 1500, 90]

    monkeypatch.setattr(crop_mod, "aggregate_box", spy)
    box = masking.aggregate_crop_box([((10, 20, 30, 40),), ()], (1920, 888),
                                     {"bottom_half_cutoff": 0.0}, [1.0, 2.0])
    polys, frame_size, settings, times = calls[0]
    assert polys == [[[(10, 20), (40, 20), (40, 60), (10, 60)]], []]
    assert frame_size == (1920, 888) and settings == {"bottom_half_cutoff": 0.0}
    assert times == [1.0, 2.0]
    assert box == (200, 700, 1500, 90)          # tuples of ints, whatever the detector returned


def test_masking_passes_none_through(monkeypatch):
    monkeypatch.setattr(crop_mod, "aggregate_box", lambda *args, **kwargs: None)
    assert masking.aggregate_crop_box([], (1920, 888)) is None


def test_masking_masks_a_region_with_the_ocr_pass_filter():
    region = np.zeros((4, 4, 3), np.uint8)
    region[1, 1] = 240
    masked = masking.mask_region(region, 230)
    assert masked[1, 1].tolist() == [240, 240, 240]
    assert masked[0, 0].tolist() == [0, 0, 0]


def test_masking_dispatches_through_the_module_so_spies_see_it(monkeypatch):
    """The wrappers must not bind `mask` at import time, or monkeypatching
    `core.detect.ocr_view.mask` would miss the call."""
    seen = []
    monkeypatch.setattr(ocr_view, "mask", lambda region, threshold: seen.append(threshold) or region)
    masking.mask_region(np.zeros((2, 2, 3), np.uint8), 211)
    assert seen == [211]


def test_state_text_no_longer_carries_the_detector_wrappers():
    """`app/state_text.py` stays pure badge and caption text."""
    from app import state_text

    assert not hasattr(state_text, "mask_region")
    assert not hasattr(state_text, "aggregate_crop_box")


def test_the_crop_view_never_reaches_for_ocr_strips():
    """The crop canvas and filmstrip draw crop.grab_frames pixels only
    (core/detect/__init__.py's frame-addressing table)."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "app" / "views" / "crop_view.py").read_text("utf-8")
    assert "grab_ocr_strips_at" not in source
    assert "request_strips" not in source
