"""AutoPilot driving the real JobRunner and the real job classes.

Only the expensive edges are faked: decoding and metadata, the detectors, the
ranges analysis, the audio profile and engine construction. Everything
between them is real: jobs run on the runner's threads, events arrive in
whatever order the threads produce, and the test plays the model owner
exactly as AutoPilot's class docstring prescribes, on its own thread:

    is_current(event) -> apply the result if current -> on_job_event(event)
    -> recompute_all(project, pending=..., ranges_pending=...)

with random user actions (re-detects, hint re-detects, manual edits, marks,
skips, pause/resume, files added, cancels, files removed) interleaved while
jobs run. Once nothing is queued, running or undelivered, the folder must be
at rest: nothing pending or outstanding in AutoPilot, no stale brightness,
and no file left PENDING.

Idle is exact, not timed: the owner counts every job from its "queued" event
(delivered to the queue before submit returns) to its terminal event, and
only the owner's thread submits jobs.
"""
from __future__ import annotations

import itertools
import queue
import random
import threading
import time

import cv2
import numpy as np
import pytest

from core.detect import audio_profile as audio_profile_module
from core.detect import brightness as brightness_module
from core.detect import crop as crop_module
from core.detect import ocr_view
from core.detect.brightness import BrightnessResult, StripSample
from core.detect.crop import CropResult, CropSample
from core.detect.ranges import pipeline as ranges_module
from core.jobs import apply
from core.jobs.autopilot import AUTOPILOT_KINDS, AutoPilot
from core.jobs.runner import JobRunner
from core.project.model import (
    FileEntry,
    FolderSettings,
    Project,
    ReviewState,
    Source,
)
from videocr import engine_registry, pyav_adapter

pytestmark = pytest.mark.timeout(60)

TERMINAL = ("finished", "failed", "cancelled")
APPLY = {
    "metadata": apply.apply_metadata,
    "crop": apply.apply_crop,
    "brightness": apply.apply_brightness,
    "ranges": apply.apply_ranges,
    "audio_profile": apply.apply_audio_profile,
}
DURATION = 1500.0
FRAME = (1920, 1080)
SETTLE_SECONDS = 30.0      # a run that has not come to rest by then is stuck


class Fakes:
    """The faked edges. Detectors take a few sub-millisecond steps and honour
    their cancel check, like the real ones between batches."""

    def __init__(self, seed: int, hint_shift: bool):
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.hint_shift = hint_shift

    def _random(self) -> float:
        with self._lock:
            return self._rng.random()

    def _step(self) -> None:
        time.sleep(self._random() * 0.001)

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(ocr_view, "video_timing", self.video_timing)
        monkeypatch.setattr(pyav_adapter, "Capture", FakeCapture)
        monkeypatch.setattr(crop_module, "grab_frames", self.grab_frames)
        monkeypatch.setattr(crop_module, "detect_crop", self.detect_crop)
        monkeypatch.setattr(brightness_module, "detect_brightness", self.detect_brightness)
        monkeypatch.setattr(ranges_module, "analyse_detailed", self.analyse_detailed)
        monkeypatch.setattr(ranges_module, "default_cache_dir", lambda project_dir: None)
        monkeypatch.setattr(audio_profile_module, "audio_profile", self.audio_profile)
        monkeypatch.setattr(engine_registry, "_build_ocr_engine", lambda *args: object())
        monkeypatch.setattr(engine_registry, "_build_detection_engine", lambda *args: object())

    def video_timing(self, path):
        self._step()
        return DURATION, 23.976

    def grab_frames(self, path, times, band_frac=1.0, target_height=72):
        self._step()
        return [np.zeros((target_height, 128, 3), np.uint8) for _ in times]

    def detect_crop(self, path, duration, det_engine, consensus, settings, cancel_check):
        for _ in range(3):
            self._step()
            if cancel_check and cancel_check():
                return CropResult(box=None, flagged=crop_module.FLAG_CANCELLED, frame_size=FRAME)
        box = (288, 780 + sum(map(ord, path)) % 7, 1344, 60)
        if self.hint_shift and consensus and self._random() < 0.3:
            box = (box[0], box[1] + 40, box[2], box[3])        # disagrees with a hint: flagged, legitimately
        return CropResult(box=box, sample_pts=[700.0, 710.0], envelope=box, agreed=2, probes_used=2,
                          flagged=None, hit_pts=[700.0, 710.0], frame_size=FRAME,
                          samples=[CropSample(700.0, (box,), True, 1), CropSample(710.0, (box,), True, 1)],
                          cutoff_frac=0.55)

    def detect_brightness(self, path, crop_box, time_ranges, det_engine, ocr_engine, folder_plateau, cancel_check):
        for _ in range(3):
            self._step()
            if cancel_check and cancel_check():
                return BrightnessResult(230, None, 230, None, brightness_module.FLAG_CANCELLED, [])
        strips = [StripSample(701.0, True, 240, 30.0, 3.0, 1, ((1, 1, 10, 10),), None)]
        if folder_plateau is not None:
            if self._random() < 0.3:
                return BrightnessResult(250, None, 250, None, brightness_module.FLAG_ESCALATE, [], strips=strips)
            lo, hi = folder_plateau
            return BrightnessResult(max(lo, hi - 20), (lo, hi), hi, None, None, [], strips=strips)
        lo = 180 + int(self._random() * 20)
        return BrightnessResult(lo + 20, (lo, lo + 40), lo + 40, 150, None, [(lo, 0.99)], strips=strips)

    def analyse_detailed(self, files, config, progress=None, *, cache_dir=None, workers=None, cancel=None):
        for _ in range(5):
            self._step()
            if cancel and cancel():
                raise ranges_module.AnalysisCancelled()
        return ranges_module.RangesAnalysis(keep={f.name: [("1:30", "23:00")] for f in files}, blocks={},
                                            durations={f.name: DURATION for f in files})

    def audio_profile(self, path, duration, cancel_check=None):
        self._step()
        return audio_profile_module.AudioProfile(duration, [0.0] * 4, [])


class FakeCapture:
    def __init__(self, path):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, prop):
        return FRAME[0] if prop == cv2.CAP_PROP_FRAME_WIDTH else FRAME[1]


class Owner:
    """The model owner: the only thread that touches the project or AutoPilot."""

    def __init__(self, tmp_path, names: list[str], folder: FolderSettings):
        self.project = Project(str(tmp_path), folder, {name: FileEntry(name) for name in names})
        self.events: queue.SimpleQueue = queue.SimpleQueue()
        self.runner = JobRunner(self.events.put, cpu_workers=2)
        self.autopilot = AutoPilot(self.runner, lambda: self.project)
        self.live: set[int] = set()          # job ids queued and not yet ended, as drained
        self.problems: list = []

    def recompute(self) -> None:
        apply.recompute_all(self.project, pending=self.autopilot.pending(),
                            ranges_pending=self.autopilot.ranges_pending())

    def drain_one(self, timeout: float) -> bool:
        try:
            event = self.events.get(timeout=timeout)
        except queue.Empty:
            return False
        if event.type == "queued":
            self.live.add(event.job_id)
        elif event.type in TERMINAL:
            self.live.discard(event.job_id)
            if event.type == "failed":
                self.problems.append(("job failed", event.key, event.error))
            if event.kind in AUTOPILOT_KINDS:
                if self.autopilot.is_current(event) and event.kind in APPLY:
                    APPLY[event.kind](self.project, event.result)
                self.autopilot.on_job_event(event)
                self.recompute()
        return True

    def idle(self) -> bool:
        return not self.live and self.events.empty()


def _act(owner: Owner, rnd: random.Random, counter, *, cancels: bool) -> None:
    project, autopilot, runner = owner.project, owner.autopilot, owner.runner
    names = list(project.files)
    if not names:
        return
    name = rnd.choice(names)
    choices = ["redetect", "crop_hint", "brightness_hint", "manual_crop", "manual_brightness", "pause",
               "add", "mark", "skip"]
    if cancels:
        choices += ["cancel", "remove"]
    action = rnd.choice(choices)
    if action == "redetect":
        autopilot.redetect(name)
    elif action == "crop_hint":
        autopilot.redetect_others_with_crop_hint(name)
    elif action == "brightness_hint":
        autopilot.redetect_others_with_brightness_hint(name)
    elif action == "manual_crop":
        apply.set_manual_crop(project, name, (288, 800 + rnd.randrange(5), 1344, 55))
        autopilot.on_crop_changed(name)
    elif action == "manual_brightness":
        apply.set_manual_brightness(project, name, 200 + rnd.randrange(20))
    elif action == "pause":
        if rnd.random() < 0.5:
            autopilot.pause()
        else:
            autopilot.resume()
    elif action == "add":
        new = f"new{next(counter)}.mp4"
        project.files[new] = FileEntry(new)
        project.files = dict(sorted(project.files.items()))
        autopilot.on_files_added([new])
    elif action == "mark":
        apply.mark_reviewed(project, name, rnd.random() < 0.7)
    elif action == "skip":
        apply.set_skipped(project, name, not project.files[name].skipped)
    elif action == "cancel":
        kind = rnd.choice(["crop", "brightness", "metadata", "audio_profile", "thumbnail", "ranges"])
        runner.cancel("ranges:*" if kind == "ranges" else f"{kind}:{name}")
    elif action == "remove":
        del project.files[name]                                       # reconcile_files removed it
        runner.cancel_where(lambda job, name=name: job.file == name)
        autopilot.on_files_removed([name])
    owner.recompute()


def simulate(monkeypatch, tmp_path, seed: int, *, actions: int, cancels: bool, hint_shift: bool) -> Owner:
    Fakes(seed, hint_shift).install(monkeypatch)
    rnd = random.Random(seed * 7 + 1)
    names = [f"ep{i:02d}.mp4" for i in range(3 + seed % 5)]
    owner = Owner(tmp_path, names, FolderSettings(brightness_full_detect_files=rnd.choice([0, 1, 3])))
    counter = itertools.count(100)
    try:
        owner.autopilot.on_open()
        owner.recompute()
        deadline = time.monotonic() + SETTLE_SECONDS
        left = actions
        while True:
            assert time.monotonic() < deadline, ("did not come to rest", owner.runner.queued(),
                                                 owner.runner.running(), owner.autopilot.pending())
            owner.drain_one(0.002)
            if left and rnd.random() < 0.2:
                _act(owner, rnd, counter, cancels=cancels)
                left -= 1
                if not left:
                    owner.autopilot.resume()                           # nothing may stay held at rest
            elif not left and owner.idle():
                break
        owner.recompute()
    finally:
        assert owner.runner.shutdown(5)
    return owner


def assert_at_rest(owner: Owner) -> None:
    autopilot, project = owner.autopilot, owner.project
    assert owner.problems == []
    assert owner.runner.queued() == [] and owner.runner.running() == []
    assert autopilot.pending() == {}
    assert autopilot.ranges_pending() is False
    assert autopilot._outstanding == {}            # every submission's terminal event was consumed
    for name, entry in project.files.items():
        assert not apply.brightness_is_stale(entry), (name, entry.crop, entry.evidence.get("brightness"))
        assert entry.review != ReviewState.PENDING, (name, entry)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_folder_left_alone_comes_to_rest_proposed(monkeypatch, tmp_path, seed):
    owner = simulate(monkeypatch, tmp_path, seed, actions=0, cancels=False, hint_shift=False)
    assert_at_rest(owner)
    for entry in owner.project.files.values():
        assert entry.review == ReviewState.PROPOSED
        assert entry.crop.source == Source.DETECTED
        assert entry.brightness is not None and entry.brightness.source == Source.DETECTED
        assert "audio" in entry.evidence and entry.time_ranges is not None


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_user_actions_while_jobs_run_leave_the_folder_at_rest(monkeypatch, tmp_path, seed):
    owner = simulate(monkeypatch, tmp_path, seed, actions=40, cancels=False, hint_shift=False)
    assert_at_rest(owner)
    for name, entry in owner.project.files.items():            # nothing went missing or got flagged
        assert entry.review in (ReviewState.PROPOSED, ReviewState.REVIEWED), (name, entry)


@pytest.mark.parametrize("seed", [6, 7, 8])
def test_cancels_removals_and_disagreeing_hints_still_leave_the_folder_at_rest(monkeypatch, tmp_path, seed):
    owner = simulate(monkeypatch, tmp_path, seed, actions=40, cancels=True, hint_shift=True)
    assert_at_rest(owner)
