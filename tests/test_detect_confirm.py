"""OCR-confirmation of a doubted brightness threshold (core/detect/confirm.py).

A file the brightness detector measured but doubts is FLAGGED. The confirm
stage masks one strip the file is known to hold a subtitle on and asks the
OCR engine to read it, stepping the threshold down by STEP until it reads
confidently or the ladder hits FLOOR. What is pinned here:

- the ladder is descending, never repeats FLOOR and always ends on it;
- a rung passes only on non-empty text at the folder's own conf_threshold --
  an engine that reads nothing can never confirm a file, whatever confidence
  comes back with the nothing (PredictedFrames scores an empty frame with a
  100 sentinel, and brightness._reading maps that to 0.0);
- a rung whose masked strip would not trip the OCR pass's gate is failed
  without being sent to the engine, because the run would not OCR it either;
- the highest passing rung wins, and the ladder costs one engine call when
  the stored value is fine and two when it is not;
- cancellation and an unreadable strip both yield a result with nothing to
  apply, never a partial one.

The strips are synthetic, like tests/test_detect_brightness.py's: glyph
blocks over the gate's centre square (so the real gate fires exactly while
the glyphs survive the mask) plus a 1..255 ramp row outside it, from which a
fake engine reads back the threshold each image it is handed was masked at.
No video is decoded and no real engine is built.
"""
import json

import numpy as np
import pytest

from core.detect import brightness as B
from core.detect import confirm as C
from core.detect import ocr_view as OV
from videocr.models import PredictedFrames

CROP = (288, 786, 1344, 53)
PROBE_TIME = 120.5
CONF_THRESHOLD = 95          # the FolderSettings default
TEXT = "你好世界"

H, W = 54, 1344
BG = 30
GLYPH_X0, GLYPH_PITCH, GLYPH_W = 450, 37, 25
GLYPH_Y0, GLYPH_Y1 = 12, 42
N_GLYPHS = 12
CENTRE_X0 = (W - H) // 2     # the gate's centre square, as videocr/video.py cuts it


# --------------------------------------------------------------------------
# Synthetic strips and fake engines
# --------------------------------------------------------------------------

def _strip(core: int = 255) -> np.ndarray:
    """A crop strip whose glyph blocks cover the gate's centre square at
    level `core` -- so the strip trips the real gate exactly while `core`
    survives the mask -- and whose last row holds a 1..255 ramp, well clear
    of the centre square, from which _masked_threshold reads the threshold
    the image was masked at back out of the pixels."""
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    for k in range(N_GLYPHS):
        x0 = GLYPH_X0 + k * GLYPH_PITCH
        img[GLYPH_Y0:GLYPH_Y1, x0:x0 + GLYPH_W] = core
    img[-1, :255] = np.arange(1, 256, dtype=np.uint8)[:, None]
    return img


def _masked_threshold(img: np.ndarray) -> int:
    """The threshold `img` was masked at: the smallest ramp level that
    survived."""
    row = img[-1, :255].min(axis=1)
    kept = row[row > 0]
    return int(kept.min()) if kept.size else 256


def _ocr_item(text, conf):
    if not text:
        return {"rec_texts": [], "rec_scores": [], "rec_polys": []}
    poly = np.array([[440, 8], [900, 8], [900, 46], [440, 46]], dtype=np.int16)
    return {"rec_texts": [text], "rec_scores": [conf], "rec_polys": [poly]}


class _ScriptedOCR:
    """Answers each image from `script`, keyed by the threshold the image was
    masked at; a threshold the script does not name reads as nothing.
    Records every batch it was handed, so a test can assert both how many
    calls were made and which thresholds each carried."""

    def __init__(self, script: dict):
        self.script = script
        self.batches: list[list[int]] = []

    @property
    def calls(self) -> int:
        return len(self.batches)

    @property
    def thresholds(self) -> list[int]:
        return [t for batch in self.batches for t in batch]

    def predict(self, images):
        batch = [_masked_threshold(img) for img in images]
        self.batches.append(batch)
        return [_ocr_item(*self.script.get(t, ("", 0.0))) for t in batch]


class _SilentOCR:
    """PaddleOCR reading nothing at all: no texts, no scores, no polygons."""

    def __init__(self):
        self.batches = []

    @property
    def calls(self) -> int:
        return len(self.batches)

    def predict(self, images):
        self.batches.append(len(images))
        return [_ocr_item("", 1.0) for _ in images]


class _ExplodingOCR:
    def predict(self, images):
        raise AssertionError("the OCR engine must not be used on this path")


def _confirm(engine, start_value=170, conf_threshold=CONF_THRESHOLD, strip=None, **kwargs):
    return C.confirm_brightness("v.mkv", CROP, PROBE_TIME, start_value, conf_threshold, engine,
                                strip=_strip() if strip is None else strip, **kwargs)


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------

def test_the_ladder_steps_down_by_ten_and_stops_at_the_floor():
    assert C.ladder(230) == [230, 220, 210, 200, 190, 180, 170, 160, 150, 140]
    assert (C.STEP, C.FLOOR) == (10, 140)


def test_a_start_off_the_step_still_ends_on_exactly_the_floor():
    rungs = C.ladder(237)

    assert rungs[0] == 237
    assert rungs[-1] == C.FLOOR
    assert rungs[-2] == 147
    assert min(rungs) == C.FLOOR, "no rung may sit below the hard stop"
    assert rungs == [237, 227, 217, 207, 197, 187, 177, 167, 157, 147, 140]


def test_the_rung_one_step_above_the_floor_does_not_repeat_it():
    assert C.ladder(150) == [150, 140]


def test_a_start_at_the_floor_is_the_only_rung_it_has():
    assert C.ladder(140) == [140]


def test_a_start_below_the_floor_yields_only_itself():
    assert C.ladder(135) == [135]


@pytest.mark.parametrize("start", [255, 245, 237, 200, 151, 150, 149, 141, 140, 135, 100])
def test_the_ladder_only_ever_goes_down(start):
    rungs = C.ladder(start)

    assert rungs[0] == start
    assert rungs == sorted(rungs, reverse=True)
    assert len(set(rungs)) == len(rungs), "a threshold is never probed twice"
    assert all(rung >= min(start, C.FLOOR) for rung in rungs)


# --------------------------------------------------------------------------
# The pass rule
# --------------------------------------------------------------------------

@pytest.mark.parametrize("confidence, value", [(0.95, 140), (0.94, None)])
def test_a_reading_passes_exactly_at_the_folders_confidence_threshold(confidence, value):
    # conf_threshold is 0-100, a reading's confidence 0.0-1.0: 95 means 0.95,
    # and 0.94 is a reading the run itself would drop.
    engine = _ScriptedOCR({140: (TEXT, confidence)})

    result = _confirm(engine, start_value=140, conf_threshold=95)

    assert result.value == value
    assert result.confirmed is (value is not None)
    (rung,) = result.rungs
    assert (rung.text, rung.confidence, rung.passed) == (TEXT, confidence, value is not None)


def test_a_confident_reading_under_a_stricter_folder_threshold_does_not_pass():
    engine = _ScriptedOCR({140: (TEXT, 0.90)})

    assert _confirm(engine, start_value=140, conf_threshold=90).value == 140
    assert _confirm(_ScriptedOCR({140: (TEXT, 0.90)}), start_value=140, conf_threshold=91).value is None


def test_an_empty_reading_never_confirms_a_file_whatever_confidence_comes_back():
    # The dangerous one: PredictedFrames scores "PaddleOCR returned nothing"
    # with a 100 sentinel, which would clear any threshold on earth.
    # brightness._reading maps it to ("", 0.0) and the rung must fail.
    assert PredictedFrames(0, [_ocr_item("", 1.0)], 0, "ch").confidence == 100
    engine = _SilentOCR()

    result = _confirm(engine, start_value=150)

    assert engine.calls > 0, "the engine was asked -- it simply read nothing"
    assert result.value is None
    assert result.confirmed is False
    assert [(rung.text, rung.confidence, rung.passed) for rung in result.rungs] == \
           [("", 0.0, False), ("", 0.0, False)]


def test_a_word_below_the_engines_own_garbage_floor_is_not_text():
    # PredictedFrames drops words under MIN_WORD_CONFIDENCE before lines are
    # assembled, so such a rung reads as empty however low the folder sets
    # its own threshold.
    engine = _ScriptedOCR({140: (TEXT, 0.4)})

    result = _confirm(engine, start_value=140, conf_threshold=10)

    assert result.value is None
    assert result.rungs[0].text == ""


# --------------------------------------------------------------------------
# The gate short-circuit
# --------------------------------------------------------------------------

def test_a_rung_whose_masked_strip_would_not_trip_the_gate_is_never_sent_to_ocr():
    # Glyph cores at 145: masked at 150 the centre square goes black, so the
    # OCR pass would never read that frame either.
    strip = _strip(core=145)
    assert not OV.gate_fires(OV.mask(strip, 150))
    assert OV.gate_fires(OV.mask(strip, 140))
    engine = _ScriptedOCR({150: (TEXT, 0.99), 140: (TEXT, 0.99)})

    result = _confirm(engine, start_value=150, strip=strip)

    assert engine.thresholds == [140], "the ungated rung was handed to the engine"
    ungated, gated = result.rungs
    assert (ungated.threshold, ungated.gated, ungated.text, ungated.confidence, ungated.passed) == \
           (150, False, "", 0.0, False)
    assert (gated.threshold, gated.gated, gated.passed) == (140, True, True)
    assert result.value == 140


def test_a_strip_no_rung_can_light_up_never_reaches_the_engine_at_all():
    strip = _strip(core=100)   # black at every rung from 150 down to 140
    assert not any(OV.gate_fires(OV.mask(strip, t)) for t in C.ladder(150))

    result = _confirm(_ExplodingOCR(), start_value=150, strip=strip)

    assert result.value is None
    assert result.confirmed is False
    assert [rung.gated for rung in result.rungs] == [False, False]


# --------------------------------------------------------------------------
# Which rung wins, and what it costs
# --------------------------------------------------------------------------

def test_the_highest_passing_rung_is_the_files_answer():
    # A brighter threshold lets less burned-in scenery through, so among
    # rungs that read the line the brightest one is the safest.
    engine = _ScriptedOCR({160: (TEXT, 0.99), 150: (TEXT, 0.99), 140: (TEXT, 0.99)})

    result = _confirm(engine, start_value=170)

    assert result.value == 160


def test_the_rungs_below_the_one_that_passed_are_not_reported():
    engine = _ScriptedOCR({160: (TEXT, 0.99), 150: (TEXT, 0.99), 140: (TEXT, 0.99)})

    result = _confirm(engine, start_value=170)

    assert [rung.threshold for rung in result.rungs] == [170, 160]
    assert [rung.passed for rung in result.rungs] == [False, True]


def test_a_stored_value_that_is_simply_fine_costs_one_image_and_one_call():
    engine = _ScriptedOCR({170: (TEXT, 0.99), 160: (TEXT, 0.99), 150: (TEXT, 0.99), 140: (TEXT, 0.99)})

    result = _confirm(engine, start_value=170)

    assert engine.batches == [[170]], "the whole ladder went out for a value that was already good"
    assert result.value == 170
    assert len(result.rungs) == 1


def test_a_stored_value_that_fails_sends_the_rest_of_the_ladder_in_one_batch():
    engine = _ScriptedOCR({150: (TEXT, 0.99)})

    result = _confirm(engine, start_value=170)

    assert engine.calls == 2
    assert engine.batches == [[170], [160, 150, 140]]
    assert result.value == 150


def test_a_ladder_that_passes_nowhere_leaves_the_value_alone():
    engine = _ScriptedOCR({})

    result = _confirm(engine, start_value=170)

    assert result.value is None
    assert result.confirmed is False
    assert [rung.threshold for rung in result.rungs] == [170, 160, 150, 140]
    assert all(rung.gated and not rung.passed for rung in result.rungs)


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------

def _cancels(*answers):
    """A cancel_check that answers `answers` in turn, then True forever."""
    pending = list(answers)

    def cancel_check():
        return pending.pop(0) if pending else True

    return cancel_check


def test_a_cancel_before_the_first_engine_call_probes_nothing():
    result = _confirm(_ExplodingOCR(), start_value=170, cancel_check=lambda: True)

    assert result.cancelled is True
    assert result.value is None
    assert result.rungs == ()
    assert result.confirmed is False


def test_a_cancel_between_the_stored_value_and_the_batch_stops_the_ladder():
    engine = _ScriptedOCR({150: (TEXT, 0.99)})

    result = _confirm(engine, start_value=170, cancel_check=_cancels(False, True))

    assert engine.batches == [[170]], "the batch went out after the cancel"
    assert result.cancelled is True
    assert result.value is None
    assert result.confirmed is False


def test_a_cancelled_result_is_never_confirmed_even_with_a_value():
    assert C.ConfirmResult(value=180, probe_time=PROBE_TIME, cancelled=True).confirmed is False
    assert C.ConfirmResult(value=180, probe_time=PROBE_TIME).confirmed is True


def test_no_cancel_check_at_all_walks_the_whole_ladder():
    engine = _ScriptedOCR({140: (TEXT, 0.99)})

    result = C.confirm_brightness("v.mkv", CROP, PROBE_TIME, 170, CONF_THRESHOLD, engine, strip=_strip())

    assert result.cancelled is False
    assert result.value == 140


# --------------------------------------------------------------------------
# Where the strip comes from
# --------------------------------------------------------------------------

def _no_grabbing(monkeypatch):
    def grab(*args, **kwargs):
        raise AssertionError("the strip was already in hand: nothing may be decoded")

    monkeypatch.setattr(OV, "grab_ocr_strips", grab)


def test_a_strip_handed_in_is_never_decoded_again(monkeypatch):
    _no_grabbing(monkeypatch)
    engine = _ScriptedOCR({170: (TEXT, 0.99)})

    result = _confirm(engine, start_value=170)

    assert result.value == 170
    assert result.probe_time == PROBE_TIME


def test_without_a_strip_one_is_grabbed_exactly_as_the_ocr_pass_sees_it(monkeypatch):
    calls = []

    def grab(video_path, crop_box, times):
        calls.append((video_path, tuple(crop_box), list(times)))
        return [_strip()]

    monkeypatch.setattr(OV, "grab_ocr_strips", grab)
    engine = _ScriptedOCR({170: (TEXT, 0.99)})

    result = C.confirm_brightness("v.mkv", CROP, PROBE_TIME, 170, CONF_THRESHOLD, engine)

    assert calls == [("v.mkv", CROP, [PROBE_TIME])]
    assert result.value == 170


def test_a_file_with_no_time_known_to_hold_a_subtitle_is_not_probed(monkeypatch):
    _no_grabbing(monkeypatch)

    result = C.confirm_brightness("v.mkv", CROP, None, 170, CONF_THRESHOLD, _ExplodingOCR())

    assert (result.value, result.probe_time, result.rungs) == (None, None, ())
    assert result.cancelled is False


@pytest.mark.parametrize("grabbed", [[], [None], [np.zeros((0, 0, 3), dtype=np.uint8)]])
def test_a_strip_that_comes_back_empty_is_not_probed(monkeypatch, grabbed):
    monkeypatch.setattr(OV, "grab_ocr_strips", lambda *a, **k: grabbed)

    result = C.confirm_brightness("v.mkv", CROP, PROBE_TIME, 170, CONF_THRESHOLD, _ExplodingOCR())

    assert result.value is None
    assert result.probe_time == PROBE_TIME
    assert result.rungs == ()


@pytest.mark.parametrize("error", [ValueError("no such stream"), OSError("gone"), EOFError()])
def test_a_file_that_will_not_open_returns_no_value_rather_than_raising(monkeypatch, error):
    assert isinstance(error, OV.FETCH_ERRORS)

    def grab(*args, **kwargs):
        raise error

    monkeypatch.setattr(OV, "grab_ocr_strips", grab)

    result = C.confirm_brightness("v.mkv", CROP, PROBE_TIME, 170, CONF_THRESHOLD, _ExplodingOCR())

    assert result.value is None
    assert result.probe_time == PROBE_TIME
    assert result.confirmed is False


# --------------------------------------------------------------------------
# The evidence record
# --------------------------------------------------------------------------

def _record(engine=None, start_value=170, conf_threshold=CONF_THRESHOLD):
    engine = _ScriptedOCR({150: (TEXT, 0.97)}) if engine is None else engine
    result = _confirm(engine, start_value=start_value, conf_threshold=conf_threshold)
    return result, C.record(result, start_value=start_value, conf_threshold=conf_threshold, crop_box=CROP)


def test_a_record_carries_the_whole_probe_it_came_from():
    result, stored = _record()

    assert stored["start_value"] == 170
    assert stored["conf_threshold"] == CONF_THRESHOLD
    assert stored["crop_box"] == list(CROP)
    assert stored["probe_time"] == PROBE_TIME
    assert stored["value"] == 150
    assert stored["rungs"] == [
        {"threshold": 170, "gated": True, "text": "", "confidence": 0.0, "passed": False},
        {"threshold": 160, "gated": True, "text": "", "confidence": 0.0, "passed": False},
        {"threshold": 150, "gated": True, "text": TEXT, "confidence": 0.97, "passed": True},
    ]
    assert [rung.threshold for rung in result.rungs] == [170, 160, 150]


def test_a_ladder_that_passed_nowhere_is_recorded_too():
    _, stored = _record(engine=_ScriptedOCR({}))

    assert stored["value"] is None
    assert [rung["threshold"] for rung in stored["rungs"]] == [170, 160, 150, 140]


def test_the_record_survives_a_trip_through_json():
    # It is written into .ocr-cache evidence, so it must be JSON data and
    # nothing else -- no numpy scalars, no tuples that come back as lists.
    _, stored = _record()

    assert json.loads(json.dumps(stored)) == stored


def test_a_record_matches_the_probe_it_came_from():
    _, stored = _record()

    assert C.matches(stored, start_value=170, conf_threshold=CONF_THRESHOLD, crop_box=CROP,
                     probe_time=PROBE_TIME) is True


@pytest.mark.parametrize("changed", [
    {"start_value": 180},
    {"conf_threshold": 90},
    {"crop_box": (288, 786, 1344, 60)},
    {"probe_time": 120.6},
    {"probe_time": None},
])
def test_a_record_does_not_match_a_probe_that_differs_in_any_input(changed):
    _, stored = _record()
    probe = {"start_value": 170, "conf_threshold": CONF_THRESHOLD, "crop_box": CROP,
             "probe_time": PROBE_TIME, **changed}

    assert C.matches(stored, **probe) is False


def test_a_record_of_a_file_with_no_probe_time_matches_only_that():
    result = C.confirm_brightness("v.mkv", CROP, None, 170, CONF_THRESHOLD, _ExplodingOCR())
    stored = C.record(result, start_value=170, conf_threshold=CONF_THRESHOLD, crop_box=CROP)

    assert stored["probe_time"] is None
    assert C.matches(stored, start_value=170, conf_threshold=CONF_THRESHOLD, crop_box=CROP,
                     probe_time=None) is True
    assert C.matches(stored, start_value=170, conf_threshold=CONF_THRESHOLD, crop_box=CROP,
                     probe_time=PROBE_TIME) is False


@pytest.mark.parametrize("stored", [
    None,
    [],
    "not a record",
    42,
    {},
    {"start_value": 170},                                                   # missing everything else
    {"start_value": 170, "conf_threshold": 95, "crop_box": list(CROP)},     # no probe_time
    {"start_value": "junk", "conf_threshold": 95, "crop_box": list(CROP), "probe_time": 120.5},
    {"start_value": 170, "conf_threshold": None, "crop_box": list(CROP), "probe_time": 120.5},
    {"start_value": 170, "conf_threshold": 95, "crop_box": None, "probe_time": 120.5},
    {"start_value": 170, "conf_threshold": 95, "crop_box": "abcd", "probe_time": 120.5},
    {"start_value": 170, "conf_threshold": 95, "crop_box": list(CROP), "probe_time": "soon"},
    {"start_value": 170, "conf_threshold": 95, "crop_box": [1, 2], "probe_time": 120.5},
])
def test_a_record_that_came_back_as_junk_never_counts_as_already_probed(stored):
    # Evidence is a disposable cache. Probing again costs one strip; a wrong
    # "already probed" strands a flagged file forever.
    assert C.matches(stored, start_value=170, conf_threshold=CONF_THRESHOLD, crop_box=CROP,
                     probe_time=PROBE_TIME) is False


def test_the_readings_a_rung_records_are_the_ones_the_ocr_pass_would_emit():
    # brightness._reading is the bridge: the same joining and the same
    # garbage floor the run applies, so a rung's text is a run's text.
    engine = _ScriptedOCR({170: (TEXT, 0.99)})

    result = _confirm(engine, start_value=170)

    assert (result.rungs[0].text, result.rungs[0].confidence) == B._reading(_ocr_item(TEXT, 0.99))
