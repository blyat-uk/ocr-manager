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

import random
from dataclasses import dataclass

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, QRectF, QSettings, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget

from app import masking
from app.controller import ProjectController
from app.main_window import MainWindow
from app.views import crop_view, ranges_view
from app.views.crop_view import DETECTED_TAG, PAGE_MARGIN, CropCanvas, CropTab, SampleStrip
from app.views.ranges_view import Timeline
from app.views.stage import Stage, StageTab
from app.views.tabs import evidence_tabs
from core.detect import crop as crop_mod
from core.detect import ocr_view
from core.jobs.view_jobs import FramesResult
from core.project.model import clamp_crop_box
from core.project import (
    MIN_CROP_SIDE,
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


@pytest.mark.parametrize("box, frame", [
    ((288, 784, 1344, 55), (1920, 888)),        # already inside: unchanged by both
    ((-40, 9000, 4000, 4000), (1920, 888)),     # outside on every side
    ((0, 0, 100, 100), (4, 4)),                 # a frame smaller than MIN_BOX
    ((0, 0, 100, 100), (0, 0)),                 # no frame size known yet
    ((10, 10, 2, 2), (1920, 888)),              # smaller than MIN_BOX
])
def test_the_views_clamp_is_the_models_clamp(box, frame):
    """The view clamps before it commits and `core.project` clamps before it
    stores. Two spellings of one rule meant the view could draw and report a
    box the model would then quietly change under it -- they differed for a
    frame under MIN_BOX, and for a frame whose size is not known yet."""
    assert CropCanvas.MIN_BOX is MIN_CROP_SIDE              # one floor, not two
    assert crop_view.clamp_box(box, frame, CropCanvas.MIN_BOX) == clamp_crop_box(box, frame)
    assert crop_view.clamp_box(box, frame) == clamp_crop_box(box, frame)


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


def widen(strip, thumbs: int = SampleStrip.PAGE) -> None:
    """Resize the page until the row has the width `thumbs` thumbnails need.
    A thumbnail is a fixed size, so how many are on screen is a question
    about the width the stage has -- at the UI scale the mockup's eight need
    more than 1440 px of window leaves it."""
    page = strip.window()
    chrome = page.width() - strip.width()
    page.resize(strip.minimumSizeHint().width() + (thumbs - 1) * strip._thumb_step() + chrome,
                page.height())
    QApplication.processEvents()


def test_the_filmstrip_shows_eight_thumbnails_and_pages(make_tab):
    harness = make_tab()
    strip = harness.tab.strip
    widen(strip)
    assert strip.label_text() == "samples"
    assert strip.fits() == 8
    assert len(strip.thumbnails()) == 8
    assert strip.more_button.text() == "more ▸"
    assert strip.more_button.isVisible()
    assert [thumb.sample_index() for thumb in strip.thumbnails()] == list(range(8))
    strip.more_button.click()
    assert [thumb.sample_index() for thumb in strip.thumbnails()] == list(range(8, 12))
    strip.more_button.click()
    assert [thumb.sample_index() for thumb in strip.thumbnails()] == list(range(8))


def test_the_filmstrip_shows_what_fits_and_pages_through_the_rest(make_tab):
    """A thumbnail keeps its size; the row does not keep all eight. Eight of
    them at the UI scale are wider than the stage is at 1440 px, and a row of
    fixed-size children would make that the window's own minimum width -- the
    window could then not be opened on a 1440 px screen at all. So the row
    shows the ones its width holds and "more ▸" reaches every other sample."""
    harness = make_tab()
    strip = harness.tab.strip
    widen(strip, 4)
    assert strip.fits() == 4
    assert [thumb.sample_index() for thumb in strip.thumbnails()] == list(range(4))
    assert strip.more_button.isVisible()
    assert strip.pages() == 3
    seen = []
    for _ in range(strip.pages()):
        seen += [thumb.sample_index() for thumb in strip.thumbnails()]
        strip.more_button.click()
    assert seen == list(range(12))                   # every sample is reachable
    widen(strip)
    assert len(strip.thumbnails()) == 8              # ... and eight again where eight fit


def test_the_filmstrip_never_sets_the_windows_minimum_width(make_tab):
    """One thumbnail's worth, whatever the detection found: the evidence
    view may not decide how narrow the window can be."""
    strip = make_tab().tab.strip
    one = strip.minimumSizeHint().width()
    assert one < 8 * strip._thumb_step()              # never all eight ...
    assert one == make_tab(evidence=crop_evidence(samples=[])).tab.strip.minimumSizeHint().width()
    widen(strip, 1)
    assert strip.fits() == 1
    assert len(strip.thumbnails()) == 1               # ... and it still shows one
    assert strip.pages() == 12


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
    assert dict(harness.tab.panel.rows())["Detection"] == "a watermark, not subtitles"


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


def test_the_canvas_and_the_filmstrip_paint_a_focus_ring(make_tab):
    """The arrows mean "nudge the box" on the canvas and "step samples" on
    the filmstrip, so which of the two holds the keyboard is the difference
    between two commands. A stylesheet suppresses Qt's own focus rectangle,
    so each surface paints its own."""
    harness = make_tab()
    canvas, strip = harness.tab.canvas, harness.tab.strip
    canvas.setFocus()
    canvas.grab(), strip.grab()
    assert canvas.focus_ring_painted() is True
    assert strip.focus_ring_painted() is False

    strip.setFocus()
    canvas.grab(), strip.grab()
    assert canvas.focus_ring_painted() is False
    assert strip.focus_ring_painted() is True


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
# Tags and the compact timeline
# --------------------------------------------------------------------------

def test_the_canvas_tags_name_the_frame_the_crop_and_the_envelope(make_tab):
    harness = make_tab()
    tags = harness.tab.canvas.tags()
    assert tags["top_left"] == "1920 × 888 · t 01:00.00"
    assert tags["top_right"] == "crop 288, 784 · 1344 × 55"
    assert tags["bottom_right"] == "dashed = text found across all 11 samples"
    harness.tab.envelope_button.click()
    assert "bottom_right" not in harness.tab.canvas.tags()


def test_a_canvas_tag_steps_clear_of_the_crop_box(make_tab):
    """A subtitle crop is a band along the bottom of the frame, which is
    where the bottom tags land. At the UI scale the tag is taller and the
    canvas, in a 1440 px window, is shorter, and the legend was printing
    through the subtitle it names."""
    harness = make_tab(crop=(0, 700, 1920, 188))        # the whole bottom fifth
    canvas = harness.tab.canvas
    harness.size_canvas(700, 324)                      # the stage's width at 1440x900
    QApplication.processEvents()
    box = canvas.box_rect()
    rects = canvas.tag_rects()
    assert set(rects) == set(canvas.tags())
    for corner, rect in rects.items():
        assert not rect.intersects(box), corner
        assert canvas.frame_rect().contains(rect), corner
    assert rects["bottom_right"].bottom() <= box.top()  # ... above it, not over it
    assert rects["top_left"].top() < box.top()          # the top pair never moved


def test_the_envelope_legend_is_left_out_when_there_is_no_envelope(make_tab):
    """The legend explains a dashed rectangle. With no envelope none is
    drawn, and "dashed = text found across all 0 samples" points at nothing
    while telling the user the detector found nothing -- twice over."""
    harness = make_tab(evidence=crop_evidence(box=None, envelope=None, samples=[]))
    canvas = harness.tab.canvas
    assert canvas.overlays()["envelope"] is True        # the toggle is still on
    assert "bottom_right" not in canvas.tags()
    canvas.grab()


def test_the_page_centres_its_content_instead_of_leaving_a_void_below(make_tab):
    """The canvas is as tall as its aspect ratio makes it at the stage's
    width, so the leftover height cannot go into the frame -- a taller canvas
    would only letterbox it. It is split above and below instead, which at
    1440x900 turns ~230 px of flat black under the timeline into margin."""
    harness = make_tab()
    tab = harness.tab
    page = tab.page()
    page.resize(960, 800)
    QApplication.processEvents()
    above = tab.canvas.y()
    below = page.height() - (tab.timeline.y() + tab.timeline.height())
    assert above > PAGE_MARGIN                      # not pinned to the top any more
    assert abs(above - below) <= 3
    assert tab.canvas.height() == tab.canvas.heightForWidth(tab.canvas.width())


def test_the_page_mounts_the_compact_timeline_under_the_stage(make_tab):
    """Ruling B5: the same timeline the Time ranges tab hosts, read-only."""
    timeline = make_tab().tab.timeline
    assert isinstance(timeline, Timeline)
    assert timeline.mode == "compact"
    assert timeline.grips() == []
    # The compact slot and the widget in it are the same scaled constant, so
    # the strip cannot outgrow the room the page leaves it.
    assert timeline.height() == ranges_view.TIMELINE_HEIGHT


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
        brightness_toolbar = stage.tabs()[1].toolbar()   # Brightness has one too
        stage.set_current(1)
        assert stage.current_toolbar() is brightness_toolbar
        assert toolbar.isHidden()
        ranges_toolbar = stage.tabs()[2].toolbar()      # Time ranges: its dim header line
        stage.set_current(2)
        assert stage.current_toolbar() is ranges_toolbar
        assert toolbar.isHidden() and brightness_toolbar.isHidden()
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


# --------------------------------------------------------------------------
# Fix round 1
# --------------------------------------------------------------------------

def test_a_typed_x_keeps_its_value_and_the_width_gives_way(make_tab):
    """A stored crop the frame cannot hold is a fidelity bug: videocr slices
    frame[y:y+h, x:x+w] and numpy clips silently, so the OCR pass would read
    a narrower band than the box says. x is the only field this changes, so
    x is what survives and the width shrinks to fit beside it."""
    harness = make_tab()
    panel = harness.tab.panel
    panel.set_spin_values(600, BOX[2], BOX[1], BOX[3])      # BOX[2] is already the width
    panel.flush()
    resolved = (600, BOX[1], FRAME_SIZE[0] - 600, BOX[3])
    assert harness.crop() == resolved
    assert harness.tab.canvas.box() == resolved
    assert panel.spin_values() == resolved


def test_a_typed_width_that_does_not_fit_moves_x_rather_than_losing_the_width(make_tab):
    """Last-edited-wins: `set_spin_values` types the width after the x, so
    the width is the value the user meant and x gives way."""
    harness = make_tab()
    panel = harness.tab.panel
    panel.set_spin_values(BOX[0], 1800, BOX[1], BOX[3])
    panel.flush()
    resolved = (FRAME_SIZE[0] - 1800, BOX[1], 1800, BOX[3])
    assert harness.crop() == resolved
    assert harness.tab.canvas.box() == resolved
    assert panel.spin_values() == resolved


def test_typed_pairs_always_store_exactly_what_is_shown_inside_the_frame(make_tab):
    harness = make_tab()
    panel = harness.tab.panel
    width, height = FRAME_SIZE
    rng = random.Random(20260917)
    for _ in range(16):
        panel.set_spin_values(rng.randrange(-300, 2400), rng.randrange(-80, 2400),
                              rng.randrange(-300, 1400), rng.randrange(-80, 1400))
        panel.flush()
        stored = harness.crop()
        assert stored == harness.tab.canvas.box() == panel.spin_values()
        x, y, box_width, box_height = stored
        assert 0 <= x and 0 <= y
        assert x + box_width <= width and y + box_height <= height
        assert box_width >= CropCanvas.MIN_BOX and box_height >= CropCanvas.MIN_BOX


def test_fit_to_samples_always_commits_even_when_the_box_is_unchanged(make_tab, monkeypatch):
    """The aggregation reproducing the detector's own box is the normal case
    on an unreviewed file; the click must still make the value yours."""
    monkeypatch.setattr(crop_mod, "aggregate_box", lambda *args, **kwargs: list(BOX))
    harness = make_tab(source=Source.DETECTED)
    assert harness.entry().crop.source == Source.DETECTED
    harness.tab.fit_button.click()
    assert harness.crop() == BOX
    assert harness.entry().crop.source == Source.MANUAL


def test_a_gesture_that_changes_nothing_commits_nothing(make_tab):
    harness = make_tab(source=Source.DETECTED)
    calls = []
    original = harness.controller.set_crop
    harness.controller.set_crop = lambda name, box: calls.append(box) or original(name, box)
    canvas = harness.tab.canvas
    point = canvas.to_widget(BOX[0] + BOX[2] / 2, BOX[1] + BOX[3] / 2)
    gesture(canvas, point, point)
    assert calls == []
    assert harness.entry().crop.source == Source.DETECTED


def test_paging_survives_a_frame_ready_burst(make_tab):
    harness = make_tab()
    strip = harness.tab.strip
    widen(strip)
    strip.more_button.click()
    assert [thumb.sample_index() for thumb in strip.thumbnails()] == list(range(8, 12))
    harness.deliver_frames()
    assert [thumb.sample_index() for thumb in strip.thumbnails()] == list(range(8, 12))
    harness.tab.select(0)                               # the selection leaving the page re-pages
    assert [thumb.sample_index() for thumb in strip.thumbnails()] == list(range(8))


def test_partial_evidence_falls_back_to_the_no_evidence_row(make_tab):
    """A deleted evidence cache can leave a dict with nothing in it worth
    showing -- 'Samples with text 0 / 0' would be a lie."""
    harness = make_tab(evidence={"frame_size": list(FRAME_SIZE), "cutoff_frac": 0.55})
    assert dict(harness.tab.panel.rows()) == {"Source": "set by you"}


def test_an_informational_flag_reads_plainly_and_does_not_warn(make_tab):
    harness = make_tab(evidence=crop_evidence(flagged="no-speech"))
    panel = harness.tab.panel
    assert dict(panel.rows())["Detection"] == "no speech — probed evenly"
    assert panel.row_tone("Detection") == ""


def test_a_blocking_flag_reads_in_human_words_and_warns(make_tab):
    harness = make_tab(evidence=crop_evidence(box=None, envelope=None, flagged="static-content"))
    panel = harness.tab.panel
    assert dict(panel.rows())["Detection"] == "a watermark, not subtitles"
    assert panel.row_tone("Detection") == "warn"


def test_thumbnails_centre_crop_instead_of_squashing(make_tab):
    harness = make_tab()
    harness.deliver_frames()
    thumb = harness.tab.strip.thumbnails()[0]
    target = QRectF(0, 0, SampleStrip.THUMB_WIDTH, SampleStrip.THUMB_HEIGHT)
    source = thumb.source_rect(target)
    image = thumb.image()
    assert image is not None and source is not None
    assert source.width() / source.height() == pytest.approx(target.width() / target.height())
    assert source.center().x() == pytest.approx(image.width() / 2)
    assert source.center().y() == pytest.approx(image.height() / 2)


def test_the_detected_overlay_is_independent_of_the_envelope_toggle(make_tab):
    harness = make_tab(crop=(300, 800, 1300, 60))
    canvas = harness.tab.canvas
    assert canvas.tags()["bottom_left"] == DETECTED_TAG
    harness.tab.envelope_button.click()
    assert canvas.tags()["bottom_left"] == DETECTED_TAG      # ruling C2 is not a toggle
    assert "bottom_right" not in canvas.tags()               # only the envelope legend went
    assert canvas.detected_box() == BOX
    canvas.grab()


def test_two_disagreeing_samples_each_get_their_own_clickable_row(make_tab):
    samples = default_samples()
    samples[1 + 6] = sample(SAMPLE_TIMES[6], kept=False, boxes=((310, 860, 1300, 40),))
    harness = make_tab(evidence=crop_evidence(samples=samples))
    panel = harness.tab.panel
    keys = [key for key, _value in panel.rows()]
    assert keys.count("1 sample sits lower") == 2
    panel.click_sample_row(6)
    assert harness.tab.selected_index() == 6
    panel.click_sample_row(LOWER_INDEX)
    assert harness.tab.selected_index() == LOWER_INDEX


def test_closing_the_page_flushes_a_pending_nudge(make_tab):
    harness = make_tab()
    arrow(harness.tab.canvas, Qt.Key.Key_Up)
    assert harness.crop() == BOX                            # still inside the debounce
    harness.tab.page().close()
    assert harness.crop() == (BOX[0], BOX[1] - 1, BOX[2], BOX[3])


def test_closing_the_page_flushes_a_typed_value(make_tab):
    harness = make_tab()
    harness.tab.panel.set_spin_values(300, 1300, 800, 60)
    assert harness.crop() == BOX
    harness.tab.page().close()
    assert harness.crop() == (300, 800, 1300, 60)


def test_closing_the_window_flushes_a_pending_nudge(qapp, fake_runner, tmp_project):
    """MainWindow.closeEvent shuts the controller down, so the flush has to
    happen before it -- a hide afterwards would be too late."""
    controller = ProjectController(fake_runner, save_debounce_ms=10)
    window = MainWindow(controller, tabs_factory=evidence_tabs)
    try:
        path = tmp_project(NAMES)
        save_project(Project(path=str(path), folder=FolderSettings(labels_enabled=False),
                             files={name: make_entry(name) for name in NAMES}))
        window.show()                       # a pending crop edit implies a visible crop page
        window.open_folder(str(path))
        window.stage.set_file(NAMES[0])
        QApplication.processEvents()
        tab = window.stage.tabs()[0]
        arrow(tab.canvas, Qt.Key.Key_Up)
        assert controller.entry(NAMES[0]).crop.y == BOX[1]
        window.close()
        assert controller.entry(NAMES[0]).crop.y == BOX[1] - 1
    finally:
        window.deleteLater()
        controller.shutdown(timeout=0.5)


# --------------------------------------------------------------------------
# Fix round 2 -- last-edited-wins instead of coupled spin ranges
# --------------------------------------------------------------------------

FULL_WIDTH_BOX = (0, 0, FRAME_SIZE[0], 60)


def test_typing_x_then_a_width_that_fits_stores_exactly_what_was_typed(make_tab):
    """1000 + 100 fits in 1920, so neither field may be touched."""
    harness = make_tab(crop=FULL_WIDTH_BOX)
    panel = harness.tab.panel
    panel.x_row.first.setValue(1000)
    panel.x_row.second.setValue(100)
    panel.flush()
    assert harness.crop() == (1000, 0, 100, 60)
    assert harness.tab.canvas.box() == (1000, 0, 100, 60)
    assert panel.spin_values() == (1000, 0, 100, 60)


def test_the_x_field_takes_a_value_with_no_prior_width_edit(make_tab):
    """With the width still full-frame, X was unusable: its maximum was
    derived from the width, so no nonzero digit was accepted."""
    harness = make_tab(crop=FULL_WIDTH_BOX)
    panel = harness.tab.panel
    panel.x_row.first.setValue(1000)
    assert panel.x_row.first.value() == 1000          # the field accepts it in the first place
    panel.flush()
    assert harness.crop() == (1000, 0, FRAME_SIZE[0] - 1000, 60)
    assert panel.spin_values() == (1000, 0, FRAME_SIZE[0] - 1000, 60)


def test_a_typed_origin_that_leaves_no_room_is_clamped_itself(make_tab):
    """The partner can only shrink to MIN_BOX, so the typed field gives way
    -- and the panel simply shows the result."""
    harness = make_tab(crop=FULL_WIDTH_BOX)
    panel = harness.tab.panel
    panel.x_row.first.setValue(FRAME_SIZE[0] - 4)
    panel.flush()
    biggest = FRAME_SIZE[0] - CropCanvas.MIN_BOX
    assert harness.crop() == (biggest, 0, CropCanvas.MIN_BOX, 60)
    assert panel.spin_values() == (biggest, 0, CropCanvas.MIN_BOX, 60)


def test_the_last_typed_field_survives_whenever_the_frame_can_hold_it(make_tab):
    harness = make_tab()
    panel = harness.tab.panel
    frame_width, frame_height = FRAME_SIZE
    rng = random.Random(20260918)
    for _ in range(24):
        harness.controller.set_crop(harness.name, BOX)
        harness.tab.refresh()
        horizontal = rng.random() < 0.5
        row, extent = (panel.x_row, frame_width) if horizontal else (panel.y_row, frame_height)
        origin_field = rng.random() < 0.5
        assert row.first.maximum() == extent - CropCanvas.MIN_BOX     # independent ranges:
        assert row.second.maximum() == extent                          # neither follows the other
        spin = row.first if origin_field else row.second
        spin.setValue(rng.randrange(-400, extent + 600))
        typed = spin.value()                          # what the field itself accepted
        panel.flush()
        stored = harness.crop()
        assert stored == harness.tab.canvas.box() == panel.spin_values()
        x, y, box_width, box_height = stored
        assert 0 <= x and 0 <= y
        assert x + box_width <= frame_width and y + box_height <= frame_height
        assert box_width >= CropCanvas.MIN_BOX and box_height >= CropCanvas.MIN_BOX
        origin, size = (x, box_width) if horizontal else (y, box_height)
        assert (origin if origin_field else size) == typed      # the typed field survived
