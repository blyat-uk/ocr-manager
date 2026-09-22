"""Frame and strip jobs for the review views (core.jobs.view_jobs).

Two frame sources, never mixed (the table in core/detect/__init__.py):

- FrameJob fetches the crop canvas's frames through core.detect.crop.grab_frames
  (whole frame, band_frac=1.0) -- the first frame at or after the requested
  time rounded to whole ms, and NOT the pixels the OCR pass sees;
- StripJob fetches OCR-exact crop strips through
  core.detect.ocr_view.grab_ocr_strips_at -- index round(t * fps), the same
  Capture the OCR pass uses, the pixels the brightness mask is applied to.

Both fetch disk-first: `core.jobs.view_cache` answers what it holds and only
the misses reach a decoder. WarmJob decodes a file's whole `Wanted` set into
that cache ahead of time and keeps none of it.

Both are Qt-free and run on the CPU lane. The frame sources and the cache are
fakes here (the cache stands in as an in-memory one, so these tests say what
the jobs do with it rather than what it does on disk -- that is
tests/test_view_cache.py's job); the one slow test runs the real frame
sources on a reference episode and compares the jobs' output with a direct
call.
"""
import gc
import inspect
import os
import subprocess
import sys
import time
import weakref
from pathlib import Path

import numpy as np
import pytest

from core.detect import crop as crop_mod
from core.detect import ocr_view
from core.jobs.autopilot import PRIORITY, REDETECT_BOOST
from core.jobs.detect_jobs import PROOF_PRIORITY
from core.jobs.runner import JobContext, JobRunner, Lane
from core.jobs import view_jobs
from core.jobs.view_cache import FileViewCache, Wanted
from core.jobs.view_jobs import (
    FRAME_HEIGHT,
    VIEW_PRIORITY,
    WARM_PRIORITY,
    FrameJob,
    FramesResult,
    StripJob,
    StripsResult,
    WarmJob,
    WarmResult,
)

PROJECT_DIR = "/proj"
BOX = (288, 786, 1344, 53)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _ctx(job) -> JobContext:
    return JobContext(job.key, job.kind, job.file)


def _image(value: int, height: int = 4, width: int = 6) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


class _Background:
    """A background CPU job (an auto-pilot kind's priority), recording when
    it ran."""

    kind = "metadata"
    lane = Lane.CPU
    priority = max(PRIORITY.values())

    def __init__(self, key: str, order: list):
        self.key = key
        self.file = None
        self._order = order

    def run(self, ctx: JobContext) -> None:
        self._order.append(self.key)


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


class FakeViewCache:
    """core.jobs.view_cache.FileViewCache, in memory.

    It is its own factory: `monkeypatch.setattr(view_jobs, "FileViewCache",
    cache)` makes every job in a test build this one, which is what lets a
    test say what was on disk before the job ran and read back what the job
    put there. It keeps the real contract -- a miss is None, a write returns
    whether it landed, nothing raises -- and `writable=False` is the
    read-only project the jobs must still work on.
    """

    readable = True

    def __init__(self, frames=None, strips=None, writable: bool = True, trimmed: int = 0):
        self.frames = dict(frames or {})                     # time -> frame
        self.strips = dict(strips or {})                     # (box, time) -> strip
        self.writable = writable
        self.trimmed = trimmed
        self.built: list[tuple[str, str, str]] = []          # (project_dir, file, video_path) per construction
        self.reads: list[float] = []                         # frame times read, in order
        self.strip_reads: list[tuple[tuple, float]] = []
        self.writes: list[float] = []                        # frame times written, in order
        self.strip_writes: list[tuple[tuple, float]] = []
        self.trims: list[Wanted] = []                        # what each trim was told to keep

    def __call__(self, project_dir: str, file: str, video_path: str) -> "FakeViewCache":
        self.built.append((project_dir, file, video_path))
        return self

    def read_frame(self, time):
        self.reads.append(time)
        return self.frames.get(time)

    def write_frame(self, time, image) -> bool:
        self.writes.append(time)
        if not self.writable:
            return False
        self.frames[time] = image
        return True

    def read_strip(self, crop_box, time):
        self.strip_reads.append((tuple(crop_box), time))
        return self.strips.get((tuple(crop_box), time))

    def write_strip(self, crop_box, time, image) -> bool:
        self.strip_writes.append((tuple(crop_box), time))
        if not self.writable:
            return False
        self.strips[(tuple(crop_box), time)] = image
        return True

    def trim(self, keep) -> int:
        self.trims.append(keep)
        return self.trimmed


@pytest.fixture(autouse=True)
def cache(monkeypatch) -> FakeViewCache:
    """The cache every job in this module builds: empty unless the test fills
    it, so a test that says nothing about the disk describes a cold one.

    It is autouse because a job's first act is now to open its cache: no test
    here, not even the slow one on real media, may touch a real project
    directory to do it.
    """
    fake = FakeViewCache()
    monkeypatch.setattr(view_jobs, "FileViewCache", fake)
    return fake


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

def test_a_frame_job_is_a_cpu_job_keyed_by_file_and_times():
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.5, 2.0])

    assert (job.kind, job.lane, job.priority, job.file) == ("frames", Lane.CPU, VIEW_PRIORITY, "a.mp4")
    assert job.key == f"frames:a.mp4:{hash((1.5, 2.0))}:cached"
    assert FrameJob(PROJECT_DIR, "a.mp4", [1.5, 2.0]).key == job.key
    assert FrameJob(PROJECT_DIR, "a.mp4", [2.0, 1.5]).key != job.key       # order is part of the request
    assert FrameJob(PROJECT_DIR, "b.mp4", [1.5, 2.0]).key != job.key
    # An exact request is a different job: the runner replaces an identical
    # key, and a queued cached request must not answer for it.
    assert FrameJob(PROJECT_DIR, "a.mp4", [1.5, 2.0], exact=True).key != job.key


def test_a_strip_job_is_a_cpu_job_keyed_by_file_crop_box_and_times():
    job = StripJob(PROJECT_DIR, "a.mp4", BOX, [1.5, 2.0])

    assert (job.kind, job.lane, job.priority, job.file) == ("strips", Lane.CPU, VIEW_PRIORITY, "a.mp4")
    assert job.key == f"strips:a.mp4:{BOX}:{hash((1.5, 2.0))}"
    assert StripJob(PROJECT_DIR, "a.mp4", (0, 0, 10, 10), [1.5, 2.0]).key != job.key
    assert StripJob(PROJECT_DIR, "a.mp4", BOX, [1.5]).key != job.key


def test_a_warm_job_is_a_cpu_job_keyed_by_file_alone():
    """One warm pass per file: the times it warms come from that file's
    evidence, so a second request for the same file is the same job (the
    runner replaces the queued one with the newer `keep`)."""
    keep = Wanted((1.0, 2.0), BOX, (3.0,))
    job = WarmJob(PROJECT_DIR, "a.mp4", keep)

    assert (job.kind, job.lane, job.priority, job.file) == ("warm", Lane.CPU, WARM_PRIORITY, "a.mp4")
    assert job.key == "warm:a.mp4"
    assert WarmJob(PROJECT_DIR, "a.mp4", Wanted((9.0,), None, ())).key == job.key
    assert WarmJob(PROJECT_DIR, "b.mp4", keep).key != job.key
    assert job.video_path == os.path.join(PROJECT_DIR, "a.mp4")


def test_warming_yields_to_every_detection_job():
    """Nobody is waiting on a warm pass: it may only use a CPU worker no
    detection job wants, and AutoPilot holds it outright during a run."""
    assert WARM_PRIORITY < min(PRIORITY.values())
    assert WARM_PRIORITY < VIEW_PRIORITY


def test_view_jobs_outrank_background_cpu_work_and_yield_to_the_proof():
    """Someone is looking at these pixels: they must not queue behind a
    folder's worth of metadata, thumbnails and audio profiles."""
    assert VIEW_PRIORITY > max(PRIORITY.values()) + REDETECT_BOOST
    assert VIEW_PRIORITY < PROOF_PRIORITY


def test_a_frame_job_starts_before_background_work_queued_first(monkeypatch):
    order = []
    monkeypatch.setattr(crop_mod, "grab_frames",
                        FakeGrabFrames({1.0: _image(1)}, during=lambda call: order.append("frames")))
    runner = JobRunner(lambda event: None, cpu_workers=1)
    try:
        runner.pause(Lane.CPU)                                 # queue them all, then let one worker loose
        runner.submit(_Background("metadata:a.mp4", order))
        runner.submit(FrameJob(PROJECT_DIR, "a.mp4", [1.0]))
        runner.submit(_Background("metadata:b.mp4", order))
        runner.resume(Lane.CPU)
        assert _wait_for(lambda: len(order) == 3), order
    finally:
        assert runner.shutdown(5.0)

    assert order == ["frames", "metadata:a.mp4", "metadata:b.mp4"]


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
# FrameJob and the disk cache
# --------------------------------------------------------------------------

def test_a_frame_job_opens_the_cache_of_its_own_file(cache, monkeypatch):
    monkeypatch.setattr(crop_mod, "grab_frames", FakeGrabFrames({1.0: _image(10)}))
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.0])

    job.run(_ctx(job))

    assert cache.built == [(PROJECT_DIR, "a.mp4", os.path.join(PROJECT_DIR, "a.mp4"))]


def test_frames_the_cache_holds_are_never_decoded(cache, monkeypatch):
    """The whole point of the cache: a second visit to a file the user has
    already looked at costs a WebP read, not a seek and a decode."""
    first, second = _image(10), _image(20)
    cache.frames.update({1.5: first, 2.0: second})
    fake = FakeGrabFrames({1.5: _image(11), 2.0: _image(21)})
    monkeypatch.setattr(crop_mod, "grab_frames", fake)
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.5, 2.0])

    out = job.run(_ctx(job))

    assert fake.calls == []                                    # the decoder was never woken
    assert out.frames[1.5] is first and out.frames[2.0] is second
    assert out.lossy == frozenset({1.5, 2.0})
    assert cache.writes == []                                  # nothing new to store


def test_a_frame_job_decodes_only_the_times_the_cache_misses(cache, monkeypatch):
    cached, decoded = _image(10), _image(30)
    cache.frames[1.5] = cached
    fake = FakeGrabFrames({3.0: decoded})
    monkeypatch.setattr(crop_mod, "grab_frames", fake)
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.5, 3.0])

    out = job.run(_ctx(job))

    assert [call["times"] for call in fake.calls] == [[3.0]]
    assert out.frames == {1.5: cached, 3.0: decoded}
    assert out.frames[1.5] is cached and out.frames[3.0] is decoded


def test_a_frame_job_says_which_of_its_frames_came_off_the_lossy_disk(cache, monkeypatch):
    """`lossy` is the crop view's licence to filter on real pixel levels: a
    time is in it when, and only when, its frame came back off the disk."""
    cache.frames[1.5] = _image(10)
    monkeypatch.setattr(crop_mod, "grab_frames", FakeGrabFrames({3.0: _image(30)}))
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.5, 3.0])

    out = job.run(_ctx(job))

    assert out.lossy == frozenset({1.5})                       # not 3.0: that one is the decoder's own pixels


def test_a_frame_job_that_decoded_everything_reports_nothing_lossy(cache, monkeypatch):
    monkeypatch.setattr(crop_mod, "grab_frames", FakeGrabFrames({1.5: _image(10)}))
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.5])

    assert job.run(_ctx(job)).lossy == frozenset()


def test_frames_a_frame_job_decoded_are_written_back(cache, monkeypatch):
    """A view that decoded a frame has already paid for it: the next view of
    the same file should not pay again."""
    cached, decoded = _image(10), _image(30)
    cache.frames[1.5] = cached
    monkeypatch.setattr(crop_mod, "grab_frames", FakeGrabFrames({3.0: decoded}))
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.5, 3.0])

    job.run(_ctx(job))

    assert cache.writes == [3.0]                               # only the one that was missing
    assert cache.frames[3.0] is decoded


def test_frames_that_could_not_be_decoded_are_not_written(cache, monkeypatch):
    monkeypatch.setattr(crop_mod, "grab_frames", FakeGrabFrames({1.0: _image(10)}))
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.0, 2.0])           # 2.0 cannot be grabbed

    out = job.run(_ctx(job))

    assert cache.writes == [1.0]
    assert out.frames == {1.0: cache.frames[1.0]}


def test_an_exact_frame_job_ignores_what_is_on_disk_but_still_fills_it(cache, monkeypatch):
    """The masked preview measures pixel levels, so it asks for the decoder's
    own frame however warm the cache is -- and the warm copy it writes back is
    still worth having for everyone drawing the same frame."""
    stale, fresh = _image(10), _image(30)
    cache.frames[1.5] = stale
    fake = FakeGrabFrames({1.5: fresh})
    monkeypatch.setattr(crop_mod, "grab_frames", fake)
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.5], exact=True)

    out = job.run(_ctx(job))

    assert [call["times"] for call in fake.calls] == [[1.5]]
    assert cache.reads == []                                   # the disk was not even consulted
    assert out.frames[1.5] is fresh
    assert out.lossy == frozenset()
    assert cache.writes == [1.5] and cache.frames[1.5] is fresh


def test_a_frame_job_on_a_project_it_cannot_write_to_still_answers(cache, monkeypatch):
    """A read-only mount or a full disk degrades to today's decode-every-time
    behaviour; it never fails a view."""
    cache.writable = False
    decoded = _image(30)
    monkeypatch.setattr(crop_mod, "grab_frames", FakeGrabFrames({1.5: decoded}))
    job = FrameJob(PROJECT_DIR, "a.mp4", [1.5])

    out = job.run(_ctx(job))

    assert cache.writes == [1.5]                               # it tried
    assert out == FramesResult("a.mp4", {1.5: decoded}, frozenset())


def test_a_frame_job_asked_for_nothing_does_not_even_open_the_cache(cache, monkeypatch):
    monkeypatch.setattr(crop_mod, "grab_frames", FakeGrabFrames({}))
    job = FrameJob(PROJECT_DIR, "a.mp4", [])

    assert job.run(_ctx(job)) == FramesResult("a.mp4", {})
    assert cache.built == []


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


# --------------------------------------------------------------------------
# StripJob and the disk cache
# --------------------------------------------------------------------------

def test_strips_the_cache_holds_are_never_decoded(cache, monkeypatch):
    first, second = _image(10), _image(20)
    cache.strips.update({(BOX, 1.5): first, (BOX, 2.0): second})
    fake = FakeGrabStrips({1.5: _image(11), 2.0: _image(21)})
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", fake)
    job = StripJob(PROJECT_DIR, "a.mp4", BOX, [1.5, 2.0])

    out = job.run(_ctx(job))

    assert fake.calls == []
    assert out.strips[1.5] is first and out.strips[2.0] is second
    assert cache.strip_writes == []


def test_a_cached_strip_is_as_exact_as_a_decoded_one(cache, monkeypatch):
    """Strips are stored losslessly, so there is no provenance to report: a
    strip off the disk is the strip the OCR pass reads, byte for byte, and
    StripsResult says nothing about where it came from."""
    cache.strips[(BOX, 1.5)] = _image(10)
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", FakeGrabStrips({2.0: _image(20)}))
    job = StripJob(PROJECT_DIR, "a.mp4", BOX, [1.5, 2.0])

    out = job.run(_ctx(job))

    assert not hasattr(out, "lossy")
    assert set(out.strips) == {1.5, 2.0}


def test_a_strip_job_decodes_only_the_times_the_cache_misses(cache, monkeypatch):
    cached, decoded = _image(10), _image(20)
    cache.strips[(BOX, 1.5)] = cached
    fake = FakeGrabStrips({2.0: decoded})
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", fake)
    job = StripJob(PROJECT_DIR, "a.mp4", BOX, [1.5, 2.0])

    out = job.run(_ctx(job))

    assert [call["times"] for call in fake.calls] == [[2.0]]
    assert out.strips == {1.5: cached, 2.0: decoded}
    assert out.strips[1.5] is cached and out.strips[2.0] is decoded


def test_strips_a_strip_job_decoded_are_written_back_under_their_box(cache, monkeypatch):
    decoded = _image(20)
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", FakeGrabStrips({2.0: decoded}))
    job = StripJob(PROJECT_DIR, "a.mp4", BOX, [2.0])

    job.run(_ctx(job))

    assert cache.strip_writes == [(BOX, 2.0)]
    assert cache.strips[(BOX, 2.0)] is decoded


def test_a_strip_cached_for_another_box_is_a_miss(cache, monkeypatch):
    """A strip is only valid for the box it was cut with: moving the crop
    invalidates every one of them, and the cache is asked with the box the
    job was given."""
    other = (0, 0, 10, 10)
    cache.strips[(other, 1.5)] = _image(10)
    fake = FakeGrabStrips({1.5: _image(20)})
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", fake)
    job = StripJob(PROJECT_DIR, "a.mp4", BOX, [1.5])

    out = job.run(_ctx(job))

    assert cache.strip_reads == [(BOX, 1.5)]
    assert [call["times"] for call in fake.calls] == [[1.5]]
    assert out.strips[1.5] is not cache.strips[(other, 1.5)]


def test_a_strip_job_on_a_project_it_cannot_write_to_still_answers(cache, monkeypatch):
    cache.writable = False
    decoded = _image(20)
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", FakeGrabStrips({2.0: decoded}))
    job = StripJob(PROJECT_DIR, "a.mp4", BOX, [2.0])

    out = job.run(_ctx(job))

    assert cache.strip_writes == [(BOX, 2.0)]
    assert out == StripsResult("a.mp4", BOX, {2.0: decoded})


def test_a_strip_job_asked_for_nothing_does_not_even_open_the_cache(cache, monkeypatch):
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", FakeGrabStrips({}))
    job = StripJob(PROJECT_DIR, "a.mp4", BOX, [])

    assert job.run(_ctx(job)) == StripsResult("a.mp4", BOX, {})
    assert cache.built == []


# --------------------------------------------------------------------------
# WarmJob
# --------------------------------------------------------------------------

def test_a_warm_job_puts_a_file_s_frames_and_strips_on_disk(cache, monkeypatch):
    frames = FakeGrabFrames({1.0: _image(10), 2.0: _image(20)})
    strips = FakeGrabStrips({3.0: _image(30)})
    monkeypatch.setattr(crop_mod, "grab_frames", frames)
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", strips)
    job = WarmJob(PROJECT_DIR, "a.mp4", Wanted((1.0, 2.0), BOX, (3.0,)))

    out = job.run(_ctx(job))

    assert cache.built == [(PROJECT_DIR, "a.mp4", os.path.join(PROJECT_DIR, "a.mp4"))]
    assert cache.writes == [1.0, 2.0]
    assert cache.strip_writes == [(BOX, 3.0)]
    assert [call["target_height"] for call in frames.calls] == [FRAME_HEIGHT]
    assert [call["crop_box"] for call in strips.calls] == [BOX]
    assert out == WarmResult("a.mp4", 2, 1, 0)


def test_a_warm_result_carries_counts_rather_than_pixels(cache, monkeypatch):
    """432 files' worth of warm results may sit in the event queue: none of
    them may carry a frame."""
    monkeypatch.setattr(crop_mod, "grab_frames", FakeGrabFrames({1.0: _image(10)}))
    job = WarmJob(PROJECT_DIR, "a.mp4", Wanted((1.0,), None, ()))

    out = job.run(_ctx(job))

    assert [type(value) for value in (out.frames_written, out.strips_written, out.trimmed)] == [int, int, int]
    assert not any(isinstance(value, np.ndarray) for value in vars(out).values())


def test_a_warm_job_decodes_nothing_that_is_already_on_disk(cache, monkeypatch):
    """The second warm pass over a folder is nearly free: it reads the cache,
    finds everything there and decodes not one frame."""
    cache.frames.update({1.0: _image(10), 2.0: _image(20)})
    cache.strips[(BOX, 3.0)] = _image(30)
    frames = FakeGrabFrames({})
    strips = FakeGrabStrips({})
    monkeypatch.setattr(crop_mod, "grab_frames", frames)
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", strips)
    job = WarmJob(PROJECT_DIR, "a.mp4", Wanted((1.0, 2.0), BOX, (3.0,)))

    out = job.run(_ctx(job))

    assert frames.calls == [] and strips.calls == []
    assert cache.writes == [] and cache.strip_writes == []
    assert out == WarmResult("a.mp4", 0, 0, 0)


def test_a_warm_job_decodes_only_the_times_that_are_missing(cache, monkeypatch):
    cache.frames[1.0] = _image(10)
    cache.strips[(BOX, 3.0)] = _image(30)
    frames = FakeGrabFrames({2.0: _image(20)})
    strips = FakeGrabStrips({4.0: _image(40)})
    monkeypatch.setattr(crop_mod, "grab_frames", frames)
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", strips)
    job = WarmJob(PROJECT_DIR, "a.mp4", Wanted((1.0, 2.0), BOX, (3.0, 4.0)))

    out = job.run(_ctx(job))

    assert [call["times"] for call in frames.calls] == [[2.0]]
    assert [call["times"] for call in strips.calls] == [[4.0]]
    assert out == WarmResult("a.mp4", 1, 1, 0)


def test_a_warm_job_without_a_crop_box_warms_frames_alone(cache, monkeypatch):
    """A file with no crop has no strips to hold: they are keyed by the box,
    and there is none."""
    frames = FakeGrabFrames({1.0: _image(10)})
    strips = FakeGrabStrips({3.0: _image(30)})
    monkeypatch.setattr(crop_mod, "grab_frames", frames)
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", strips)
    job = WarmJob(PROJECT_DIR, "a.mp4", Wanted((1.0,), None, (3.0,)))

    out = job.run(_ctx(job))

    assert strips.calls == [] and cache.strip_reads == []
    assert out == WarmResult("a.mp4", 1, 0, 0)


def test_a_warm_job_finishes_by_trimming_what_it_was_not_asked_to_keep(cache, monkeypatch):
    """Evidence moves -- a re-detect samples other times, a new crop box
    orphans every strip -- and the warm pass is where the old pixels go."""
    cache.trimmed = 7
    keep = Wanted((1.0,), BOX, (3.0,))
    monkeypatch.setattr(crop_mod, "grab_frames", FakeGrabFrames({1.0: _image(10)}))
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", FakeGrabStrips({3.0: _image(30)}))
    job = WarmJob(PROJECT_DIR, "a.mp4", keep)

    out = job.run(_ctx(job))

    assert cache.trims == [keep]
    assert out.trimmed == 7


def test_a_warm_job_with_nothing_to_keep_still_trims(cache):
    """Wanted() empty is a file whose evidence is gone: the cache should go
    with it."""
    keep = Wanted((), None, ())
    job = WarmJob(PROJECT_DIR, "a.mp4", keep)

    out = job.run(_ctx(job))

    assert cache.trims == [keep]
    assert out == WarmResult("a.mp4", 0, 0, 0)


def test_a_warm_job_counts_only_the_images_the_cache_took(cache, monkeypatch):
    """A cache that cannot be written to wasted the decode, and says so: the
    counts are what landed on disk, not what was decoded."""
    cache.writable = False
    monkeypatch.setattr(crop_mod, "grab_frames", FakeGrabFrames({1.0: _image(10)}))
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", FakeGrabStrips({3.0: _image(30)}))
    job = WarmJob(PROJECT_DIR, "a.mp4", Wanted((1.0,), BOX, (3.0,)))

    out = job.run(_ctx(job))

    assert cache.writes == [1.0] and cache.strip_writes == [(BOX, 3.0)]
    assert out == WarmResult("a.mp4", 0, 0, 0)


def test_a_warm_job_cancelled_before_it_starts_returns_nothing(cache, monkeypatch):
    frames = FakeGrabFrames({1.0: _image(10)})
    monkeypatch.setattr(crop_mod, "grab_frames", frames)
    job = WarmJob(PROJECT_DIR, "a.mp4", Wanted((1.0,), None, ()))
    ctx = _ctx(job)
    ctx.cancel_event.set()

    assert job.run(ctx) is None
    assert frames.calls == [] and cache.built == []


def test_a_warm_job_stops_between_batches_when_a_run_starts(cache, monkeypatch):
    """Warming decodes in batches so that a cancel -- the run lane taking the
    machine, or the user pausing -- lands in the gap between two of them
    rather than at the end of the file."""
    batch = view_jobs.WARM_BATCH
    times = tuple(float(index) for index in range(batch * 3))
    job = WarmJob(PROJECT_DIR, "a.mp4", Wanted(times, None, ()))
    ctx = _ctx(job)
    frames = FakeGrabFrames({time: _image(int(time)) for time in times},
                            during=lambda call: ctx.cancel_event.set())
    monkeypatch.setattr(crop_mod, "grab_frames", frames)

    assert job.run(ctx) is None
    assert [call["times"] for call in frames.calls] == [list(times[:batch])]
    assert cache.writes == list(times[:batch])                 # what it did decode is still on disk
    assert cache.trims == []                                   # and it did not stay to tidy up


def test_a_warm_job_cancelled_between_its_two_kinds_returns_nothing(cache, monkeypatch):
    job = WarmJob(PROJECT_DIR, "a.mp4", Wanted((1.0,), BOX, (3.0,)))
    ctx = _ctx(job)
    frames = FakeGrabFrames({1.0: _image(10)}, during=lambda call: ctx.cancel_event.set())
    strips = FakeGrabStrips({3.0: _image(30)})
    monkeypatch.setattr(crop_mod, "grab_frames", frames)
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", strips)

    assert job.run(ctx) is None
    assert strips.calls == []


def test_a_warm_job_holds_on_to_none_of_the_pixels_it_decoded(cache, monkeypatch):
    """The pixels are the disk's, not the job's: a folder-wide warm pass must
    not grow with the number of frames it has warmed."""
    cache.writable = False                                     # so only the job could be holding them
    alive: list[weakref.ref] = []

    def grab_frames(video_path, times, band_frac=1.0, target_height=FRAME_HEIGHT):
        made = [_image(int(time)) for time in times]
        alive.extend(weakref.ref(image) for image in made)
        return made

    def grab_strips(video_path, crop_box, times):
        made = [(time, _image(int(time))) for time in times]
        alive.extend(weakref.ref(image) for _, image in made)
        return made

    monkeypatch.setattr(crop_mod, "grab_frames", grab_frames)
    monkeypatch.setattr(ocr_view, "grab_ocr_strips_at", grab_strips)
    keep = Wanted(tuple(float(index) for index in range(view_jobs.WARM_BATCH * 2)), BOX, (99.0,))
    job = WarmJob(PROJECT_DIR, "a.mp4", keep)

    out = job.run(_ctx(job))

    assert len(alive) == view_jobs.WARM_BATCH * 2 + 1          # every image really was decoded
    gc.collect()
    assert [reference for reference in alive if reference() is not None] == []
    assert out == WarmResult("a.mp4", 0, 0, 0)


# --------------------------------------------------------------------------
# The stand-in cache against the real one
# --------------------------------------------------------------------------

def test_the_fake_view_cache_answers_the_real_one_s_calls():
    """These tests are only worth anything while the stand-in takes the same
    calls as core.jobs.view_cache.FileViewCache. Arity, not names: the jobs
    call it positionally."""
    for name in ("read_frame", "write_frame", "read_strip", "write_strip", "trim"):
        real = inspect.signature(getattr(FileViewCache, name)).parameters
        fake = inspect.signature(getattr(FakeViewCache, name)).parameters
        assert len(fake) == len(real), name
    assert isinstance(getattr(FileViewCache, "readable"), property)


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
    assert sorted(strips.strips) == sorted(at for at, _ in expected_strips) == times
    for at, strip in expected_strips:
        assert np.array_equal(strips.strips[at], strip), f"strip at {at} differs from a direct grab"

    frame_job = FrameJob(str(tmp_path), video.name, times)
    frames = frame_job.run(_ctx(frame_job))
    expected_frames = crop_mod.grab_frames(path, list(times), band_frac=1.0, target_height=FRAME_HEIGHT)

    assert sorted(frames.frames) == times
    assert len(expected_frames) == len(times)
    for at, expected in zip(times, expected_frames):
        got = frames.frames[at]
        assert got.shape == expected.shape and got.shape[0] == FRAME_HEIGHT and got.shape[2] == 3
        assert np.array_equal(got, expected), f"frame at {at} differs from a direct grab"
    # Whole frames, not a bottom band: the source scaled to FRAME_HEIGHT rows,
    # keeping its full width (grab_frames rounds the width down to an even one).
    source_width, source_height = _dimensions(path)
    width = round(source_width * FRAME_HEIGHT / source_height)
    assert frames.frames[times[0]].shape[:2] == (FRAME_HEIGHT, width - width % 2)
    # And not the strips: the two sources are never interchangeable.
    assert frames.frames[times[0]].shape != strips.strips[times[0]].shape
