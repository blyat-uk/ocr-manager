"""Frame and strip jobs for the review views (core.jobs.view_jobs).

Two frame sources, never mixed (the table in core/detect/__init__.py):

- FrameJob fetches the crop canvas's frames through core.detect.crop.grab_frames
  (whole frame, band_frac=1.0) -- the first frame at or after the requested
  time rounded to whole ms, and NOT the pixels the OCR pass sees;
- StripJob fetches OCR-exact crop strips through
  core.detect.ocr_view.grab_ocr_strips_at -- index round(t * fps), the same
  Capture the OCR pass uses, the pixels the brightness mask is applied to.

Both are Qt-free and run on the CPU lane. The frame sources are fakes here;
the one slow test runs the real ones on a reference episode and compares the
jobs' output with a direct call.
"""
import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core.detect import crop as crop_mod
from core.detect import ocr_view
from core.jobs.runner import JobContext, Lane
from core.jobs.view_jobs import FRAME_HEIGHT, FrameJob, FramesResult, StripJob, StripsResult

PROJECT_DIR = "/proj"
BOX = (288, 786, 1344, 53)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _ctx(job) -> JobContext:
    return JobContext(job.key, job.kind, job.file)


def _image(value: int, height: int = 4, width: int = 6) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _dimensions(video_path: str) -> tuple[int, int]:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video_path],
                           capture_output=True, text=True, check=True)
    width, height = probe.stdout.strip().split("x")
    return int(width), int(height)


class FakeGrabFrames:
    """crop.grab_frames: frames for the times it knows, in request order,
    with the unknown ones dropped -- exactly as the real one does (it returns
    frames only, so a dropped time shifts every later frame)."""

    def __init__(self, frames: dict[float, np.ndarray], during=None):
        self.signature = inspect.signature(crop_mod.grab_frames)
        self.frames = frames
        self.during = during
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        call = dict(bound.arguments)
        self.calls.append(call)
        if self.during is not None:
            self.during(call)
        return [self.frames[time] for time in call["times"] if time in self.frames]


class FakeGrabStrips:
    """ocr_view.grab_ocr_strips_at: (requested time, strip) pairs, dropping
    the times it has no strip for."""

    def __init__(self, strips: dict[float, np.ndarray], during=None):
        self.signature = inspect.signature(ocr_view.grab_ocr_strips_at)
        self.strips = strips
        self.during = during
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        call = dict(bound.arguments)
        self.calls.append(call)
        if self.during is not None:
            self.during(call)
        return [(time, self.strips[time]) for time in call["times"] if time in self.strips]


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

def test_a_frame_job_is_a_cpu_job_keyed_by_file_and_times():
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.5, 2.0])

    assert (job.kind, job.lane, job.priority, job.file) == ("frames", Lane.CPU, 0, "a.mp4")
    assert job.key == f"frames:a.mp4:{hash((1.5, 2.0))}"
    assert FrameJob(PROJECT_DIR, "a.mp4", [1.5, 2.0]).key == job.key
    assert FrameJob(PROJECT_DIR, "a.mp4", [2.0, 1.5]).key != job.key       # order is part of the request
    assert FrameJob(PROJECT_DIR, "b.mp4", [1.5, 2.0]).key != job.key


def test_a_strip_job_is_a_cpu_job_keyed_by_file_crop_box_and_times():
    job = StripJob(PROJECT_DIR, "a.mp4", BOX, [1.5, 2.0])

    assert (job.kind, job.lane, job.priority, job.file) == ("strips", Lane.CPU, 0, "a.mp4")
    assert job.key == f"strips:a.mp4:{BOX}:{hash((1.5, 2.0))}"
    assert StripJob(PROJECT_DIR, "a.mp4", (0, 0, 10, 10), [1.5, 2.0]).key != job.key
    assert StripJob(PROJECT_DIR, "a.mp4", BOX, [1.5]).key != job.key


def test_both_jobs_take_their_video_from_the_project_directory():
    assert FrameJob(PROJECT_DIR, "a.mp4", [1.0]).video_path == os.path.join(PROJECT_DIR, "a.mp4")
    assert StripJob(PROJECT_DIR, "a.mp4", BOX, [1.0]).video_path == os.path.join(PROJECT_DIR, "a.mp4")


# --------------------------------------------------------------------------
# FrameJob
# --------------------------------------------------------------------------

def test_a_frame_job_grabs_whole_frames_at_the_requested_times(monkeypatch):
    first, second = _image(10), _image(20)
    fake = FakeGrabFrames({1.5: first, 2.0: second})
    monkeypatch.setattr(crop_mod, "grab_frames", fake)
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.5, 2.0])

    out = job.run(_ctx(job))

    assert fake.calls == [{"video_path": "/proj/a.mp4", "times": [1.5, 2.0],
                           "band_frac": 1.0, "target_height": FRAME_HEIGHT}]
    assert isinstance(out, FramesResult)
    assert out.file == "a.mp4"
    assert out.frames == {1.5: first, 2.0: second}
    assert out.frames[1.5] is first and out.frames[2.0] is second


def test_a_frame_job_can_be_asked_for_another_height(monkeypatch):
    fake = FakeGrabFrames({1.5: _image(10)})
    monkeypatch.setattr(crop_mod, "grab_frames", fake)
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.5], target_height=180)

    job.run(_ctx(job))

    assert fake.calls[0]["target_height"] == 180


def test_frames_that_could_not_be_grabbed_are_left_out_without_shifting_the_others(monkeypatch):
    first, third = _image(10), _image(30)
    fake = FakeGrabFrames({1.0: first, 3.0: third})            # 2.0 cannot be grabbed
    monkeypatch.setattr(crop_mod, "grab_frames", fake)
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.0, 2.0, 3.0])

    out = job.run(_ctx(job))

    # The batch came back short, so the times no longer pair by position: the
    # job re-fetches one at a time rather than mislabel a frame.
    assert [call["times"] for call in fake.calls] == [[1.0, 2.0, 3.0], [1.0], [2.0], [3.0]]
    assert out.frames == {1.0: first, 3.0: third}
    assert out.frames[1.0] is first and out.frames[3.0] is third


def test_a_frame_job_asked_for_nothing_grabs_nothing(monkeypatch):
    fake = FakeGrabFrames({})
    monkeypatch.setattr(crop_mod, "grab_frames", fake)
    job = FrameJob(PROJECT_DIR, "a.mp4", [])

    out = job.run(_ctx(job))

    assert fake.calls == []
    assert out == FramesResult("a.mp4", {})


def test_a_frame_job_cancelled_before_it_starts_returns_nothing(monkeypatch):
    fake = FakeGrabFrames({1.0: _image(10)})
    monkeypatch.setattr(crop_mod, "grab_frames", fake)
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.0])
    ctx = _ctx(job)
    ctx.cancel_event.set()

    assert job.run(ctx) is None
    assert fake.calls == []


def test_a_frame_job_cancelled_while_it_grabs_returns_nothing(monkeypatch):
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.0, 2.0])
    ctx = _ctx(job)
    fake = FakeGrabFrames({1.0: _image(10), 2.0: _image(20)}, during=lambda call: ctx.cancel_event.set())
    monkeypatch.setattr(crop_mod, "grab_frames", fake)

    assert job.run(ctx) is None


# --------------------------------------------------------------------------
# StripJob
# --------------------------------------------------------------------------

def test_a_strip_job_grabs_ocr_exact_strips_at_the_requested_times(monkeypatch):
    first, second = _image(10), _image(20)
    fake = FakeGrabStrips({1.5: first, 2.0: second})
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", fake)
    job = StripJob(PROJECT_DIR, "a.mp4", BOX, [1.5, 2.0])

    out = job.run(_ctx(job))

    assert fake.calls == [{"video_path": "/proj/a.mp4", "crop_box": BOX, "times": [1.5, 2.0]}]
    assert isinstance(out, StripsResult)
    assert (out.file, out.crop_box) == ("a.mp4", BOX)
    assert out.strips == {1.5: first, 2.0: second}
    assert out.strips[1.5] is first and out.strips[2.0] is second


def test_strips_that_could_not_be_grabbed_are_left_out(monkeypatch):
    second = _image(20)
    fake = FakeGrabStrips({2.0: second})
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", fake)
    job = StripJob(PROJECT_DIR, "a.mp4", BOX, [1.5, 2.0])

    out = job.run(_ctx(job))

    assert [call["times"] for call in fake.calls] == [[1.5, 2.0]]     # the pairs say which time survived
    assert out.strips == {2.0: second}


def test_a_strip_job_asked_for_nothing_grabs_nothing(monkeypatch):
    fake = FakeGrabStrips({})
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", fake)
    job = StripJob(PROJECT_DIR, "a.mp4", BOX, [])

    out = job.run(_ctx(job))

    assert fake.calls == []
    assert out == StripsResult("a.mp4", BOX, {})


def test_a_strip_job_cancelled_before_it_starts_returns_nothing(monkeypatch):
    fake = FakeGrabStrips({1.0: _image(10)})
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", fake)
    job = StripJob(PROJECT_DIR, "a.mp4", BOX, [1.0])
    ctx = _ctx(job)
    ctx.cancel_event.set()

    assert job.run(ctx) is None
    assert fake.calls == []


def test_a_strip_job_cancelled_while_it_grabs_returns_nothing(monkeypatch):
    job = StripJob(PROJECT_DIR, "a.mp4", BOX, [1.0])
    ctx = _ctx(job)
    fake = FakeGrabStrips({1.0: _image(10)}, during=lambda call: ctx.cancel_event.set())
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", fake)

    assert job.run(ctx) is None


def test_view_jobs_import_no_qt():
    code = ("import sys, core.jobs.view_jobs; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('PyQt6', 'PyQt5', 'PySide6')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False,
                          cwd=Path(__file__).resolve().parent.parent)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------------------------
# Real frame sources on a reference episode
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.needs_media
def test_view_jobs_on_a_reference_episode_match_a_direct_grab(reference_media, tmp_path):
    slay = reference_media.get("slay")
    if slay is None or not slay["crop"]:
        pytest.skip("the slay reference project is not present")
    video = slay["video"]
    (tmp_path / video.name).symlink_to(video)                  # never write into the reference folder
    path = str(tmp_path / video.name)
    box = tuple(slay["crop"])
    times = [300.0, 420.5]

    strip_job = StripJob(str(tmp_path), video.name, box, times)
    strips = strip_job.run(_ctx(strip_job))
    expected_strips = ocr_view.grab_ocr_strips_at(path, box, list(times))

    assert isinstance(strips, StripsResult) and strips.crop_box == box
    assert sorted(strips.strips) == sorted(time for time, _ in expected_strips) == times
    for time, strip in expected_strips:
        assert np.array_equal(strips.strips[time], strip), f"strip at {time} differs from a direct grab"

    frame_job = FrameJob(str(tmp_path), video.name, times)
    frames = frame_job.run(_ctx(frame_job))
    expected_frames = crop_mod.grab_frames(path, list(times), band_frac=1.0, target_height=FRAME_HEIGHT)

    assert sorted(frames.frames) == times
    assert len(expected_frames) == len(times)
    for time, expected in zip(times, expected_frames):
        got = frames.frames[time]
        assert got.shape == expected.shape and got.shape[0] == FRAME_HEIGHT and got.shape[2] == 3
        assert np.array_equal(got, expected), f"frame at {time} differs from a direct grab"
    # Whole frames, not a bottom band: the source scaled to FRAME_HEIGHT rows,
    # keeping its full width (grab_frames rounds the width down to an even one).
    source_width, source_height = _dimensions(path)
    width = round(source_width * FRAME_HEIGHT / source_height)
    assert frames.frames[times[0]].shape[:2] == (FRAME_HEIGHT, width - width % 2)
    # And not the strips: the two sources are never interchangeable.
    assert frames.frames[times[0]].shape != strips.strips[times[0]].shape
