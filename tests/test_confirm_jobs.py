"""The confirm stage's job and its strip choice (core.jobs.detect_jobs).

`probe_time_for` picks the one moment a ConfirmJob probes, out of the
evidence the file already carries: a gallery line first (text detection
confirmed a subtitle there and the view cache holds the strip), then a
brightness zoom tile -- never "leaking", which is chosen among the strips
with NO text and so would fail every rung by construction -- and last the
earliest strip the detector called text, which costs a decode.

`ConfirmJob` then walks that file's ladder: the strip comes from the view
cache when it is there and from a decode when it is not, only an OCR engine
is leased, everything the result is judged against is captured at
construction, and a cancelled job returns None like every other job in that
module.

Engines, the decoder and the detector are fakes here; the view cache is the
real one, over an ordinary file standing in for the episode (it only stat()s
its source).
"""
import inspect
from pathlib import Path

import numpy as np
import pytest

from core.detect import confirm as confirm_mod
from core.detect import ocr_view
from core.detect.confirm import ConfirmResult, Rung
from core.detect.tiles import TILE_KINDS
from core.jobs import JobContext, Lane
from core.jobs import view_cache
from core.jobs.detect_jobs import ConfirmJob, ConfirmJobResult, probe_time_for
from core.project import FileEntry, FolderSettings
from videocr import engine_registry

NAME = "Episode 01.mkv"
BOX = (288, 786, 1344, 53)
PROBE_TIME = 120.5
TEXT = "你好世界"

H, W = 54, 1344
BG = 30
GLYPH_X0, GLYPH_PITCH, GLYPH_W = 450, 37, 25
GLYPH_Y0, GLYPH_Y1 = 12, 42
N_GLYPHS = 12


# --------------------------------------------------------------------------
# probe_time_for
# --------------------------------------------------------------------------

def _entry(evidence) -> FileEntry:
    return FileEntry(name=NAME, evidence=evidence)


def _line(time) -> dict:
    return {"time": time, "boxes": [[440, 8, 460, 38]], "lines": 1}


def _strip_sample(time, is_text=True) -> dict:
    return {"time": time, "is_text": is_text, "glyph_level": 240, "background_level": 30.0,
            "stroke_px": 3.0, "lines": 1, "boxes": [], "gate_at_value": None}


FULL_EVIDENCE = {
    "lines": {"samples": [_line(610.0), _line(900.0)], "crop_box": list(BOX)},
    "brightness": {"tiles": {"dark": 300.0, "bright": 420.0, "leaking": 55.0},
                   "strips": [_strip_sample(30.0), _strip_sample(90.0)]},
}


def test_the_first_gallery_line_is_the_strip_a_confirm_probes():
    assert probe_time_for(_entry(FULL_EVIDENCE)) == 610.0


def test_a_brightness_zoom_tile_answers_when_there_are_no_gallery_lines():
    evidence = {**FULL_EVIDENCE, "lines": {"samples": []}}

    # "dark" comes first in TILE_KINDS, so it is the tile taken.
    assert probe_time_for(_entry(evidence)) == 300.0
    assert TILE_KINDS[0] == "dark"


def test_the_leaking_tile_is_never_the_strip_a_confirm_probes():
    # "leaking" is picked among the strips with NO text (core.detect.tiles):
    # probing it would fail every rung by construction and strand the file.
    evidence = {"brightness": {"tiles": {"leaking": 55.0},
                               "strips": [_strip_sample(90.0)]}}

    assert probe_time_for(_entry(evidence)) == 90.0

    alone = {"brightness": {"tiles": {"leaking": 55.0}, "strips": []}}
    assert probe_time_for(_entry(alone)) is None


def test_the_earliest_text_strip_answers_when_there_is_neither_a_line_nor_a_tile():
    evidence = {"brightness": {"strips": [_strip_sample(30.0), _strip_sample(90.0)]}}

    assert probe_time_for(_entry(evidence)) == 30.0


def test_a_strip_the_detector_found_no_text_on_is_never_the_one_probed():
    evidence = {"brightness": {"strips": [_strip_sample(12.0, is_text=False),
                                          _strip_sample(30.0, is_text=False),
                                          _strip_sample(90.0)]}}

    assert probe_time_for(_entry(evidence)) == 90.0


@pytest.mark.parametrize("evidence", [
    {},
    None,
    {"lines": {}, "brightness": {}},
    {"lines": {"samples": []}, "brightness": {"tiles": {}, "strips": []}},
    {"brightness": {"strips": [_strip_sample(30.0, is_text=False)]}},
    {"brightness": {"tiles": {"leaking": 55.0}}},
])
def test_a_file_whose_evidence_names_no_subtitle_bearing_time_is_not_probed(evidence):
    assert probe_time_for(_entry(evidence)) is None


@pytest.mark.parametrize("evidence", [
    {"lines": {"samples": ["not a sample", 7]}},
    {"lines": {"samples": [{"boxes": []}]}},                 # a sample with no time
    {"lines": {"samples": [{"time": "soon"}]}},
    {"lines": {"samples": [{"time": None}]}},
    {"brightness": {"tiles": "x"}},
    {"brightness": {"tiles": {"dark": "soon", "bright": None}}},
    {"brightness": {"tiles": []}},
    {"brightness": {"strips": ["junk", None]}},
    {"brightness": {"strips": [{"is_text": True}]}},         # a text strip with no time
    {"brightness": {"strips": [{"time": "soon", "is_text": True}]}},
])
def test_evidence_that_came_back_malformed_contributes_no_candidate(evidence):
    assert probe_time_for(_entry(evidence)) is None


@pytest.mark.parametrize("evidence", [
    {"lines": "junk"},
    {"lines": ["junk"]},
    {"lines": {"samples": 7}},
    {"brightness": "junk"},
    {"brightness": ["junk"]},
    {"brightness": {"strips": 7}},
])
def test_evidence_of_the_wrong_type_yields_no_time_rather_than_raising(evidence):
    assert probe_time_for(_entry(evidence)) is None


# --------------------------------------------------------------------------
# Fakes: the strip, the engine and the decoder
# --------------------------------------------------------------------------

def _strip(core: int = 255) -> np.ndarray:
    """A crop strip whose glyph blocks cover the gate's centre square, so the
    real gate fires, with a 1..255 ramp in its last row the fake engine reads
    the masking threshold back out of (see tests/test_detect_confirm.py)."""
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    for k in range(N_GLYPHS):
        x0 = GLYPH_X0 + k * GLYPH_PITCH
        img[GLYPH_Y0:GLYPH_Y1, x0:x0 + GLYPH_W] = core
    img[-1, :255] = np.arange(1, 256, dtype=np.uint8)[:, None]
    return img


def _masked_threshold(img: np.ndarray) -> int:
    row = img[-1, :255].min(axis=1)
    kept = row[row > 0]
    return int(kept.min()) if kept.size else 256


def _ocr_item(text, conf):
    if not text:
        return {"rec_texts": [], "rec_scores": [], "rec_polys": []}
    poly = np.array([[440, 8], [900, 8], [900, 46], [440, 46]], dtype=np.int16)
    return {"rec_texts": [text], "rec_scores": [conf], "rec_polys": [poly]}


class _ScriptedOCR:
    """Reads every image at whatever threshold it was masked at, answering
    from `script`; anything unscripted reads as nothing."""

    def __init__(self, script):
        self.script = script
        self.batches = []
        self.build_args = None

    def predict(self, images):
        self.batches.append([_masked_threshold(img) for img in images])
        return [_ocr_item(*self.script.get(t, ("", 0.0))) for t in self.batches[-1]]


@pytest.fixture
def ocr_engine(monkeypatch):
    """The one OCR engine the registry hands out, recording how it was
    built, so a test can pin the lease's arguments."""
    engine = _ScriptedOCR({230: (TEXT, 0.99), 220: (TEXT, 0.99)})

    def build(lang, det_model_dir, rec_model_dir, use_gpu):
        engine.build_args = (lang, det_model_dir, rec_model_dir, use_gpu)
        return engine

    monkeypatch.setattr(engine_registry, "_build_ocr_engine", build)
    return engine


def _no_decoding(monkeypatch):
    def grab(*args, **kwargs):
        raise AssertionError("the view cache held this strip: nothing may be decoded")

    monkeypatch.setattr(ocr_view, "grab_ocr_strips", grab)


def _decodes(monkeypatch, strip=None):
    """Stand in for the decoder; records (video_path, crop_box, times)."""
    calls = []

    def grab(video_path, crop_box, times):
        calls.append((video_path, tuple(crop_box), list(times)))
        return [_strip() if strip is None else strip]

    monkeypatch.setattr(ocr_view, "grab_ocr_strips", grab)
    return calls


def _project(tmp_path: Path) -> str:
    """A project folder holding an ordinary file where the episode goes."""
    (tmp_path / NAME).write_bytes(b"\0" * 4096)
    return str(tmp_path)


def _job(project_dir, folder=None, start_value=230, crop_box=BOX, probe_time=PROBE_TIME) -> ConfirmJob:
    return ConfirmJob(project_dir, NAME, crop_box, probe_time, start_value,
                      folder if folder is not None else FolderSettings(use_gpu=False))


def _ctx(job, events=None) -> JobContext:
    return JobContext(job.key, job.kind, job.file, None if events is None else events.append)


# --------------------------------------------------------------------------
# ConfirmJob: identity and what it captures
# --------------------------------------------------------------------------

def test_a_confirm_job_is_a_gpu_job_keyed_by_its_file(tmp_path):
    job = _job(_project(tmp_path))

    assert (job.kind, job.lane, job.key, job.file, job.priority) == \
           ("confirm", Lane.GPU, f"confirm:{NAME}", NAME, 0)


def test_a_confirm_job_refuses_to_be_built_without_a_crop_box(tmp_path):
    with pytest.raises(ValueError):
        _job(_project(tmp_path), crop_box=None)


def test_a_confirm_job_captures_its_inputs_so_a_later_edit_cannot_change_it(tmp_path, ocr_engine, monkeypatch):
    _decodes(monkeypatch)
    folder = FolderSettings(ocr_lang="chinese_cht", conf_threshold=90, use_gpu=False)
    box = [288, 786, 1344, 53]

    job = ConfirmJob(_project(tmp_path), NAME, box, PROBE_TIME, 230, folder)

    box.append(999)
    folder.conf_threshold = 10
    folder.ocr_lang = "en"
    folder.use_gpu = True

    assert job.crop_box == BOX
    assert job.probe_time == PROBE_TIME
    assert job.start_value == 230
    assert job.conf_threshold == 90
    assert (job.ocr_lang, job.use_gpu) == ("chinese_cht", False)

    job.run(_ctx(job))

    assert ocr_engine.build_args == ("chinese_cht", None, None, False)


def test_a_confirm_job_calls_confirm_brightness_with_what_it_captured(tmp_path, ocr_engine, monkeypatch):
    signature = inspect.signature(confirm_mod.confirm_brightness)
    calls = []

    def fake(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(dict(bound.arguments))
        # The lease is held for the whole call.
        assert ocr_engine not in [e for pool in engine_registry._idle_ocr_engines.values() for e in pool]
        return ConfirmResult(value=220, probe_time=PROBE_TIME,
                             rungs=(Rung(230, True, "", 0.0, False), Rung(220, True, TEXT, 0.99, True)))

    monkeypatch.setattr(confirm_mod, "confirm_brightness", fake)
    project_dir = _project(tmp_path)
    job = _job(project_dir, FolderSettings(conf_threshold=90, use_gpu=False))

    out = job.run(_ctx(job))

    (call,) = calls
    assert call["video_path"] == str(tmp_path / NAME)
    assert tuple(call["crop_box"]) == BOX
    assert call["probe_time"] == PROBE_TIME
    assert call["start_value"] == 230
    assert call["conf_threshold"] == 90
    assert call["ocr_engine"] is ocr_engine
    cancel = call["cancel_check"]
    assert cancel() is False
    assert out.result.value == 220


# --------------------------------------------------------------------------
# ConfirmJob: the strip, the result and cancellation
# --------------------------------------------------------------------------

def test_a_confirm_job_reads_its_strip_out_of_the_view_cache_rather_than_decoding(tmp_path, ocr_engine,
                                                                                  monkeypatch):
    project_dir = _project(tmp_path)
    cache = view_cache.FileViewCache(project_dir, NAME, str(tmp_path / NAME))
    assert cache.write_strip(BOX, PROBE_TIME, _strip()) is True
    _no_decoding(monkeypatch)
    job = _job(project_dir)

    out = job.run(_ctx(job))

    assert ocr_engine.batches == [[230]], "the cached strip was read at the stored value"
    assert out.result.value == 230


def test_a_confirm_job_decodes_its_strip_when_the_cache_does_not_hold_it(tmp_path, ocr_engine, monkeypatch):
    project_dir = _project(tmp_path)
    calls = _decodes(monkeypatch)
    job = _job(project_dir)

    out = job.run(_ctx(job))

    assert calls == [(str(tmp_path / NAME), BOX, [PROBE_TIME])]
    assert out.result.value == 230


def test_a_strip_cached_for_another_crop_box_is_a_miss_not_the_old_pixels(tmp_path, ocr_engine, monkeypatch):
    project_dir = _project(tmp_path)
    cache = view_cache.FileViewCache(project_dir, NAME, str(tmp_path / NAME))
    cache.write_strip((0, 0, 640, 48), PROBE_TIME, _strip())
    calls = _decodes(monkeypatch)

    _job(project_dir).run(_ctx(_job(project_dir)))

    assert calls, "a strip cut with another crop box was used as this one's"


def test_a_confirm_job_returns_the_file_crop_and_question_it_was_built_with(tmp_path, ocr_engine, monkeypatch):
    _decodes(monkeypatch)
    project_dir = _project(tmp_path)
    job = _job(project_dir, FolderSettings(conf_threshold=90, use_gpu=False), start_value=230)

    out = job.run(_ctx(job))

    assert isinstance(out, ConfirmJobResult)
    assert out.file == NAME
    assert out.crop_box == BOX
    assert out.start_value == 230
    assert out.conf_threshold == 90
    assert isinstance(out.result, ConfirmResult)


def test_a_cancelled_confirm_job_returns_no_partial_result(tmp_path, ocr_engine, monkeypatch):
    _decodes(monkeypatch)
    project_dir = _project(tmp_path)
    job = _job(project_dir)
    ctx = _ctx(job)
    ctx.cancel_event.set()

    assert job.run(ctx) is None
    assert ocr_engine.batches == [], "a cancelled job still walked its ladder"


def test_a_confirm_job_that_cannot_read_its_strip_reports_no_value_rather_than_failing(tmp_path, ocr_engine,
                                                                                       monkeypatch):
    def grab(*args, **kwargs):
        raise ValueError("no such stream")

    monkeypatch.setattr(ocr_view, "grab_ocr_strips", grab)
    project_dir = _project(tmp_path)
    job = _job(project_dir)

    out = job.run(_ctx(job))

    assert out is not None
    assert out.result.value is None
    assert out.result.confirmed is False


def test_a_confirm_job_leases_only_an_ocr_engine(tmp_path, ocr_engine, monkeypatch):
    # The strip is a time the file's evidence already named, so there is
    # nothing for a detection engine to find.
    def build_detection(det_model_dir, use_gpu):
        raise AssertionError("a confirm needs no text detection")

    monkeypatch.setattr(engine_registry, "_build_detection_engine", build_detection)
    _decodes(monkeypatch)
    project_dir = _project(tmp_path)
    job = _job(project_dir)

    job.run(_ctx(job))

    idle = [e for pool in engine_registry._idle_ocr_engines.values() for e in pool]
    assert idle == [ocr_engine], "the lease was not returned"
