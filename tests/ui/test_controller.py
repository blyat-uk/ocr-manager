"""Plan 3B Task 2: app/controller.py's ProjectController, the Qt bridge.

Nothing is decoded: `fake_runner` (tests/ui/conftest.py) records what the
controller and its AutoPilot submit, and each test delivers the events a
real JobRunner would, with constructed results. Events reach the controller
through its listener (a queue); `drain_events()` is what the controller's
30 ms timer calls, so tests call it directly for determinism, and wait (with
the event loop running) only for what really is asynchronous: the debounced
save and the folder watcher. PyQt6 does not expose QTest.qWaitFor, so
wait_for() below polls with QTest.qWait.
"""
from __future__ import annotations

import json
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtTest import QTest

from app.activity import ActivitySnapshot
from app.controller import CPU_WORKERS, VIEW_KINDS, ProjectController, default_runner_factory
from app.logbook import LOG_LIMIT, LogBook
from app.run_snapshot import RunSnapshot, RunTracker, notification_for, notify_duration
from app.state_text import badge_for
from core.detect.audio_profile import AudioProfile
from core.detect.brightness import FLAG_DIM_TEXT, BrightnessResult
from core.detect.confirm import ConfirmResult, Rung, ladder
from core.detect.crop import FLAG_LOW_AGREEMENT, CropResult
from core.detect.lines import LineSample, LinesResult
from core.detect.ranges.pipeline import RangesAnalysis
from core.jobs import apply as apply_mod
from core.jobs.autopilot import LINES_BOOST, AutoPilot, is_held_job
from core.jobs.detect_jobs import (
    PROOF_PRIORITY,
    AudioProfileResult,
    BrightnessJobResult,
    ConfirmJobResult,
    CropJobResult,
    LinesJob,
    LinesJobResult,
    MetadataResult,
    ProofOcrJob,
    ProofResult,
    RangesJobResult,
    ThumbnailResult,
)
from core.jobs.run import RunFile, RunJob, RunSummary
from core.jobs.runner import JobRunner, Lane
from core.jobs.view_cache import wanted
from core.jobs.view_jobs import FramesResult, StripsResult, WarmJob, WarmResult
from core.project import (
    Brightness,
    Crop,
    FileEntry,
    FolderSettings,
    Media,
    Project,
    ReviewState,
    Source,
    TimeRange,
    TimeRanges,
    UnsupportedProjectVersion,
    ocr_call_for,
    store,
    to_json,
)

SLAY_NAMES = [
    "ZS2_-_11_[1080p]TXHBR.mp4",
    "ZS2_-_12_[1080p]TXHBR.mp4",
    "ZS2_-_13_[1080p]TXHBR.mp4",
    "ZS2_-_14_[1080p]TXHBR.mp4",
    "ZS2_-_15_[1080p]TXHBR.mp4",
]
BOX = (288, 786, 1344, 53)
OTHER_BOX = (300, 800, 1300, 60)
DURATION = 1400.0
WAIT_MS = 5000


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def wait_for(predicate, timeout_ms: int) -> bool:
    """QTest.qWaitFor: run the event loop until `predicate()` or the timeout."""
    deadline = time.monotonic() + timeout_ms / 1000
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        QTest.qWait(10)
    return True


class Spy:
    """Records every emission of a pyqtSignal."""

    def __init__(self, signal):
        self.calls: list[tuple] = []
        signal.connect(self._record)

    def _record(self, *args):
        self.calls.append(args)

    @property
    def firsts(self) -> list:
        return [call[0] for call in self.calls]


def make_entry(name, *, crop=None, brightness=None, ranges=None, review=ReviewState.PROPOSED,
               crop_source=Source.MANUAL, brightness_source=Source.MANUAL, flags=None,
               skipped=False, duration=DURATION) -> FileEntry:
    """An entry with known media; a detected brightness is recorded as
    measured on the entry's crop, as apply_brightness leaves it."""
    entry = FileEntry(name, media=Media(1920, 1080, duration, 23.976), review=review,
                      skipped=skipped, flags=dict(flags or {}), sample_time=300.0)
    if crop is not None:
        entry.crop = Crop(*crop, crop_source)
    if brightness is not None:
        entry.brightness = Brightness(brightness, brightness_source)
        if brightness_source in (Source.DETECTED, Source.HINT) and crop is not None:
            entry.evidence["brightness"] = {"crop_box": list(crop), "value_crop_box": list(crop)}
    if ranges is not None:
        entry.time_ranges = TimeRanges([TimeRange(start, end) for start, end in ranges], Source.MANUAL)
    return entry


def v2_config(entries: list[FileEntry], **folder) -> dict:
    project = Project(path="unused", folder=FolderSettings(**folder), files={e.name: e for e in entries})
    return to_json(project, include_evidence=False)


def manual_config(names, **folder) -> dict:
    """Every file ready to run: MANUAL crop and brightness, reviewed."""
    return v2_config([make_entry(name, crop=BOX, brightness=209, review=ReviewState.REVIEWED) for name in names],
                     **folder)


def saved(folder) -> dict:
    return json.loads((folder / ".ocr.json").read_text(encoding="utf-8"))


def crop_result(submission, box=BOX, flagged=None) -> CropJobResult:
    result = CropResult(box=box, sample_pts=[300.0], envelope=box, agreed=3, probes_used=3,
                        flagged=flagged, hit_pts=[300.0] if box else [], frame_size=(1920, 1080))
    return CropJobResult(submission.job.file, result, submission.job.hint)


def brightness_result(submission, value=205, plateau=(180, 230), flagged=None) -> BrightnessJobResult:
    job = submission.job
    result = BrightnessResult(value=value, plateau=plateau, seed=value + 20, gate_floor=None,
                              flagged=flagged, curve=[])
    return BrightnessJobResult(job.file, result, {}, job.hint_value, job.crop_box)


def run_to_end(fake_runner, run, summary: RunSummary, event_type="finished") -> None:
    for name in summary.succeeded:
        fake_runner.emit(run, "run_file_started", file=name)
        fake_runner.emit(run, "run_file_finished", file=name, result={"ok": True, "lines": 3, "error": ""})
    for name, error in summary.failed.items():
        fake_runner.emit(run, "run_file_started", file=name)
        fake_runner.emit(run, "run_file_finished", file=name, result={"ok": False, "lines": 0, "error": error})
    fake_runner.finish(run, summary, event_type)


@pytest.fixture
def make_controller(qapp, fake_runner):
    made = []

    def make(**kwargs) -> ProjectController:
        kwargs.setdefault("save_debounce_ms", 10)
        controller = ProjectController(fake_runner, **kwargs)
        made.append(controller)
        return controller

    yield make
    for controller in made:
        controller.shutdown(timeout=0.5)


@pytest.fixture
def notifications(monkeypatch):
    sent = []

    def fake_run(args, **kwargs):
        sent.append(list(args))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return sent


def spy_method(monkeypatch, cls, name, log: list, tag: str | None = None):
    """Wrap cls.name so each call is appended to `log` (arguments after self)."""
    original = getattr(cls, name)

    def wrapper(self, *args, **kwargs):
        log.append((tag, *args) if tag else args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, name, wrapper)


# --------------------------------------------------------------------------
# Opening a folder
# --------------------------------------------------------------------------

def test_open_migrates_a_v1_project_and_starts_autopilot(make_controller, fake_runner, tmp_project):
    folder = tmp_project(fixture="slay")
    controller = make_controller()
    opened = Spy(controller.project_opened)
    files = Spy(controller.files_changed)

    controller.open_folder(str(folder))

    assert opened.calls == [(str(folder),)]
    assert files.calls
    assert controller.project.path == str(folder)
    assert controller.names() == SLAY_NAMES
    first = controller.entry(SLAY_NAMES[0])
    assert first.crop == Crop(288, 786, 1344, 53, Source.IMPORTED)
    assert first.brightness == Brightness(209, Source.IMPORTED)
    # C1: imported crop and brightness -> reviewed
    assert {controller.entry(name).review for name in SLAY_NAMES} == {ReviewState.REVIEWED}
    # Auto-pilot fills only what is missing: thumbnails and audio profiles.
    assert fake_runner.factory_calls == 1
    assert sorted((s.job.kind, s.job.file) for s in fake_runner.submissions) == sorted(
        (kind, name) for name in SLAY_NAMES for kind in ("thumbnail", "audio_profile"))


def test_open_refuses_an_unsupported_version_without_touching_anything(make_controller, fake_runner, tmp_project):
    good = tmp_project(["ep01.mkv"], config=manual_config(["ep01.mkv"]))
    bad = tmp_project(["ep01.mkv"], config={"version": 99, "files": {}})
    bad_text = (bad / ".ocr.json").read_text(encoding="utf-8")
    controller = make_controller(save_debounce_ms=60_000)
    controller.open_folder(str(good))
    controller.set_brightness("ep01.mkv", 150)                 # an edit waiting for its debounced save
    good_config = good / ".ocr.json"
    good_bytes, good_mtime = good_config.read_bytes(), good_config.stat().st_mtime_ns
    project = controller.project
    closed = Spy(controller.project_closed)
    opened = Spy(controller.project_opened)
    submitted = len(fake_runner.submissions)

    with pytest.raises(UnsupportedProjectVersion):
        controller.open_folder(str(bad))

    assert controller.project is project and project.path == str(good)     # still open
    assert controller.entry("ep01.mkv").brightness.value == 150
    assert good_config.read_bytes() == good_bytes and good_config.stat().st_mtime_ns == good_mtime
    assert closed.calls == [] and opened.calls == []
    assert len(fake_runner.submissions) == submitted
    controller.shutdown()
    assert saved(good)["files"]["ep01.mkv"]["brightness"]["value"] == 150
    assert (bad / ".ocr.json").read_text(encoding="utf-8") == bad_text
    assert sorted(p.name for p in bad.iterdir()) == [".ocr.json", "ep01.mkv"]


def test_reopening_the_open_folder_keeps_unsaved_edits(make_controller, fake_runner, tmp_project):
    folder = tmp_project(["ep01.mkv"], config=manual_config(["ep01.mkv"]))
    controller = make_controller(save_debounce_ms=60_000)
    controller.open_folder(str(folder))
    old = controller.project
    controller.set_brightness("ep01.mkv", 150)

    controller.open_folder(str(folder))

    assert controller.project is not old
    assert controller.entry("ep01.mkv").brightness == Brightness(150, Source.MANUAL)
    assert saved(folder)["files"]["ep01.mkv"]["brightness"]["value"] == 150


def test_open_on_an_unsupported_version_with_nothing_open(make_controller, fake_runner, tmp_project):
    bad = tmp_project(["ep01.mkv"], config={"version": 3})
    controller = make_controller()
    with pytest.raises(UnsupportedProjectVersion):
        controller.open_folder(str(bad))
    assert controller.project is None and controller.names() == []
    assert fake_runner.submissions == []
    controller.shutdown()
    assert json.loads((bad / ".ocr.json").read_text(encoding="utf-8")) == {"version": 3}


# --------------------------------------------------------------------------
# The event drain
# --------------------------------------------------------------------------

def test_detection_results_are_applied_recomputed_and_saved(make_controller, fake_runner, tmp_project):
    folder = tmp_project(["ep01.mkv"])
    controller = make_controller()
    controller.open_folder(str(folder))
    changed = Spy(controller.file_changed)
    name = "ep01.mkv"
    entry = controller.entry(name)

    meta = fake_runner.last("metadata", name)
    fake_runner.finish(meta, MetadataResult(name, 1920, 1080, DURATION, 23.976))
    controller.drain_events()
    assert entry.media == Media(1920, 1080, DURATION, 23.976)
    assert name in changed.firsts
    assert wait_for(lambda: (folder / ".ocr.json").exists()
                          and saved(folder)["files"][name]["media"]["duration"] == DURATION, WAIT_MS)

    crop = fake_runner.last("crop", name)
    changed.calls.clear()
    fake_runner.finish(crop, crop_result(crop, BOX))
    controller.drain_events()
    assert entry.crop == Crop(*BOX, Source.DETECTED)
    assert name in changed.firsts
    # on_job_event ran after the apply: brightness is measured on the applied box
    brightness = fake_runner.last("brightness", name)
    assert brightness.job.crop_box == BOX
    assert entry.review == ReviewState.PENDING

    fake_runner.finish(brightness, brightness_result(brightness, 205))
    controller.drain_events()
    assert entry.brightness == Brightness(205, Source.DETECTED)
    assert entry.review == ReviewState.PROPOSED            # recomputed: nothing pending any more
    assert wait_for(lambda: saved(folder)["files"][name]["review"] == "proposed", WAIT_MS)
    assert saved(folder)["files"][name]["crop"]["source"] == "detected"


def test_other_results_are_applied_by_type(make_controller, fake_runner, tmp_project):
    names = ["ep01.mkv", "ep02.mkv"]
    config = v2_config([make_entry(name) for name in names])
    folder = tmp_project(names, config=config)
    controller = make_controller()
    controller.open_folder(str(folder))
    changed = Spy(controller.file_changed)

    ranges = fake_runner.last("ranges")
    analysis = RangesAnalysis(keep={"ep01.mkv": [("02:00", "20:00")]}, blocks={},
                              durations={name: DURATION for name in names})
    fake_runner.finish(ranges, RangesJobResult(analysis))
    audio = fake_runner.last("audio_profile", "ep02.mkv")
    fake_runner.finish(audio, AudioProfileResult("ep02.mkv", AudioProfile(DURATION, [0.0, 1.0], [(1.0, 2.0)])))
    controller.drain_events()

    assert controller.entry("ep01.mkv").time_ranges == TimeRanges([TimeRange("02:00", "20:00")], Source.DETECTED)
    assert controller.entry("ep02.mkv").time_ranges is None
    assert controller.entry("ep02.mkv").evidence["audio"]["speech"] == [[1.0, 2.0]]
    assert set(changed.firsts) >= set(names)


def test_thumbnail_results_become_qimage_copies(make_controller, fake_runner, tmp_project):
    folder = tmp_project(fixture="slay")
    controller = make_controller()
    controller.open_folder(str(folder))
    ready = Spy(controller.thumbnail_ready)
    name, other = SLAY_NAMES[0], SLAY_NAMES[1]
    assert controller.thumbnail(name) is None

    image = np.zeros((72, 128, 3), np.uint8)
    image[0, 0] = (255, 0, 0)                                  # BGR: blue
    thumb = fake_runner.last("thumbnail", name)
    fake_runner.finish(thumb, ThumbnailResult(name, thumb.job.time, image))
    no_frame = fake_runner.last("thumbnail", other)
    fake_runner.finish(no_frame, ThumbnailResult(other, no_frame.job.time, None))
    controller.drain_events()

    qimage = controller.thumbnail(name)
    assert isinstance(qimage, QImage)
    assert (qimage.width(), qimage.height()) == (128, 72)
    assert qimage.pixelColor(0, 0) == QColor(0, 0, 255)
    image[0, 0] = (0, 0, 0)
    assert qimage.pixelColor(0, 0) == QColor(0, 0, 255)       # a copy, not a view
    assert ready.calls == [(name,)]
    assert controller.thumbnail(other) is None                 # no frame: keep the placeholder


def test_a_superseded_crop_result_is_not_applied(make_controller, fake_runner, tmp_project):
    name = "ep01.mkv"
    folder = tmp_project([name], config=v2_config([make_entry(name, review=ReviewState.PENDING)]))
    controller = make_controller()
    controller.open_folder(str(folder))
    first = fake_runner.last("crop", name)
    fake_runner.start(first)                                   # running: a re-detect waits behind it
    controller.redetect(name)
    second = fake_runner.last("crop", name)
    assert second.job_id != first.job_id and not fake_runner.ended(first)

    fake_runner.finish(first, crop_result(first, OTHER_BOX))
    controller.drain_events()
    assert controller.entry(name).crop is None

    fake_runner.finish(second, crop_result(second, BOX))
    controller.drain_events()
    assert controller.entry(name).crop == Crop(*BOX, Source.DETECTED)


def test_a_redetect_replacing_a_queued_crop_leaves_nothing_outstanding(make_controller, fake_runner, tmp_project):
    name = "ep01.mkv"
    folder = tmp_project([name], config=v2_config([make_entry(name, review=ReviewState.PENDING)]))
    controller = make_controller()
    controller.open_folder(str(folder))
    first = fake_runner.last("crop", name)

    controller.redetect(name)                                  # the queued crop is replaced: "cancelled" at once
    second = fake_runner.last("crop", name)
    assert fake_runner.ended(first) and not fake_runner.ended(second)
    controller.drain_events()
    assert controller.entry(name).crop is None
    assert "crop" in controller._autopilot.pending()[name]    # the replacement is still outstanding

    fake_runner.finish(second, crop_result(second, BOX))
    controller.drain_events()
    entry = controller.entry(name)
    assert entry.crop == Crop(*BOX, Source.DETECTED)
    assert "crop" not in controller._autopilot.pending().get(name, set())
    brightness = fake_runner.last("brightness", name)          # the chain moved on
    fake_runner.finish(brightness, brightness_result(brightness))
    controller.drain_events()
    assert entry.review == ReviewState.PROPOSED


def test_a_failed_detection_is_logged_and_not_retried(make_controller, fake_runner, tmp_project):
    name = "ep01.mkv"
    folder = tmp_project([name], config=v2_config([make_entry(name, review=ReviewState.PENDING)]))
    controller = make_controller()
    controller.open_folder(str(folder))
    logs = Spy(controller.log_appended)
    crop = fake_runner.last("crop", name)

    fake_runner.finish(crop, None, "failed", message="decoder exploded", error="Traceback: decoder exploded\n")
    controller.drain_events()

    assert fake_runner.of_kind("crop", name) == [crop]
    assert controller.entry(name).review == ReviewState.FLAGGED
    assert any(key == "Detections" and "ep01.mkv: crop failed: decoder exploded" in text for key, text in logs.calls)
    assert "Traceback: decoder exploded" in controller.log_text("Detections")


def test_a_run_start_keeps_the_detection_failures_log(make_controller, fake_runner, tmp_project, notifications):
    names = ["ep01.mkv", "ep02.mkv"]
    entries = [make_entry("ep01.mkv", review=ReviewState.PENDING),
               make_entry("ep02.mkv", crop=BOX, brightness=209, review=ReviewState.REVIEWED)]
    folder = tmp_project(names, config=v2_config(entries))
    controller = make_controller()
    controller.open_folder(str(folder))
    crop = fake_runner.last("crop", "ep01.mkv")
    fake_runner.finish(crop, None, "failed", message="decoder exploded", error="Traceback: decoder exploded\n")
    controller.drain_events()
    controller.start_run(["ep02.mkv"])
    run = fake_runner.last("run")
    fake_runner.emit(run, "run_file_log", file="ep02.mkv", message="Starting OCR: ep02.mkv\n")
    fake_runner.emit(run, "log", message="run note")
    controller.drain_events()
    assert controller.log_keys() == ["Pipeline", "Detections", "ep02.mkv"]
    cleared = Spy(controller.logs_cleared)

    controller.stop_run()
    fake_runner.finish(run, RunSummary([], {}, ["ep02.mkv"], 1.0), "cancelled")
    controller.drain_events()
    controller.start_run(["ep02.mkv"])                          # a new run: logs restart, except Detections

    assert cleared.calls == [()]
    assert controller.log_keys() == ["Detections"]
    assert "ep01.mkv: crop failed: decoder exploded" in controller.log_text("Detections")
    assert controller.log_text("ep02.mkv") == "" and controller.log_text("Pipeline") == ""


def test_an_apply_that_raises_still_releases_the_job(make_controller, fake_runner, tmp_project, monkeypatch):
    name = "ep01.mkv"
    folder = tmp_project([name], config=v2_config([make_entry(name, review=ReviewState.PENDING)]))
    controller = make_controller()
    controller.open_folder(str(folder))
    crop = fake_runner.last("crop", name)
    original = apply_mod.apply_crop

    def broken(project, result):
        raise RuntimeError("apply exploded")

    monkeypatch.setattr(apply_mod, "apply_crop", broken)
    fake_runner.finish(crop, crop_result(crop, OTHER_BOX))
    controller.drain_events()

    assert "RuntimeError: apply exploded" in controller.log_text("Pipeline")
    assert "crop" not in controller._autopilot.pending().get(name, set())    # on_job_event still ran
    assert controller.entry(name).review == ReviewState.FLAGGED               # and the recompute

    monkeypatch.setattr(apply_mod, "apply_crop", original)
    controller.redetect(name)
    again = fake_runner.last("crop", name)
    fake_runner.finish(again, crop_result(again, BOX))
    controller.drain_events()
    assert controller.entry(name).crop == Crop(*BOX, Source.DETECTED)      # current, not "superseded"


def test_events_of_a_closed_folder_are_ignored(make_controller, fake_runner, tmp_project):
    name = "ep01.mkv"
    config = v2_config([make_entry(name, review=ReviewState.PENDING)])
    first = tmp_project([name], config=config)
    second = tmp_project([name], config=config)
    controller = make_controller()
    controller.open_folder(str(first))
    old_crop = fake_runner.last("crop", name)
    fake_runner.start(old_crop)                                # still running when the folder closes
    closed = Spy(controller.project_closed)

    controller.open_folder(str(second))
    assert closed.calls == [()]
    assert fake_runner.cancel_predicates and fake_runner.cancel_predicates[-1](old_crop.job)
    fake_runner.finish(old_crop, crop_result(old_crop, OTHER_BOX))
    controller.drain_events()
    assert controller.entry(name).crop is None

    new_crop = fake_runner.last("crop", name)
    assert new_crop.job_id > old_crop.job_id
    fake_runner.finish(new_crop, crop_result(new_crop, BOX))
    controller.drain_events()
    assert controller.entry(name).crop == Crop(*BOX, Source.DETECTED)


def test_unknown_job_kinds_are_logged_to_pipeline(make_controller, fake_runner, tmp_project):
    folder = tmp_project(["ep01.mkv"])
    controller = make_controller()
    controller.open_folder(str(folder))
    logs = Spy(controller.log_appended)
    stray = SimpleNamespace(key="mystery:*", kind="mystery", file=None)
    fake_runner.finish(SimpleNamespace(job=stray, job_id=999), "whatever")
    controller.drain_events()
    assert any(key == "Pipeline" and "mystery" in text for key, text in logs.calls)


# --------------------------------------------------------------------------
# Activity
# --------------------------------------------------------------------------

def test_activity_follows_queued_running_and_finished_jobs(make_controller, fake_runner, tmp_project):
    name = "ep01.mkv"
    folder = tmp_project([name], config=v2_config([make_entry(name, review=ReviewState.PENDING)]))
    controller = make_controller()
    controller.open_folder(str(folder))
    activity = Spy(controller.activity_changed)
    controller.drain_events()
    assert activity.calls
    snapshot = controller.activity()
    assert isinstance(snapshot, ActivitySnapshot)
    assert snapshot.current is None and not snapshot.paused and snapshot.running == ()

    crop = fake_runner.last("crop", name)
    fake_runner.start(crop)
    controller.drain_events()
    assert controller.activity().current[:3] == ("crop", name, None)   # no progress yet: indeterminate
    assert controller.running_detectors(name) == {"crop"}
    fake_runner.progress(crop, 0.5, "probing")
    controller.drain_events()
    assert controller.activity().current == ("crop", name, 0.5, "probing")

    fake_runner.finish(crop, crop_result(crop))
    controller.drain_events()
    snapshot = controller.activity()
    assert controller.running_detectors(name) == set()
    assert [(kind, file) for kind, file, _ in snapshot.recent] == [("crop", name)]

    controller.pause_autopilot()
    assert controller.activity().paused
    assert {lane for lane, only in fake_runner.pauses if only is is_held_job} == {Lane.GPU, Lane.CPU}
    controller.resume_autopilot()
    assert not controller.activity().paused
    assert set(fake_runner.resumes) == {Lane.GPU, Lane.CPU}


# --------------------------------------------------------------------------
# Edits
# --------------------------------------------------------------------------

def edit_project(tmp_project):
    """ep01: detected crop and brightness, no flags. ep02: the same with a
    blocking crop flag. Auto-pilot off, so only thumbnails are submitted."""
    entries = [
        make_entry("ep01.mkv", crop=BOX, brightness=205, crop_source=Source.DETECTED,
                   brightness_source=Source.DETECTED, flags={"crop": "", "brightness": ""}),
        make_entry("ep02.mkv", crop=BOX, brightness=205, crop_source=Source.DETECTED,
                   brightness_source=Source.DETECTED, flags={"crop": FLAG_LOW_AGREEMENT, "brightness": ""},
                   review=ReviewState.FLAGGED),
    ]
    return tmp_project(["ep01.mkv", "ep02.mkv"], config=v2_config(entries, autopilot_enabled=False))


def test_edits_store_manual_values_and_review_unless_another_field_is_flagged(
        make_controller, fake_runner, tmp_project, monkeypatch):
    crop_changes = []
    spy_method(monkeypatch, AutoPilot, "on_crop_changed", crop_changes)
    folder = edit_project(tmp_project)
    controller = make_controller()
    controller.open_folder(str(folder))
    assert controller.entry("ep01.mkv").review == ReviewState.PROPOSED
    assert controller.entry("ep02.mkv").review == ReviewState.FLAGGED
    changed = Spy(controller.file_changed)

    controller.set_brightness("ep01.mkv", 200)
    assert controller.entry("ep01.mkv").brightness == Brightness(200, Source.MANUAL)
    assert controller.entry("ep01.mkv").review == ReviewState.REVIEWED

    controller.set_brightness("ep02.mkv", 200)                 # the crop is still flagged
    assert controller.entry("ep02.mkv").brightness == Brightness(200, Source.MANUAL)
    assert controller.entry("ep02.mkv").review == ReviewState.FLAGGED

    controller.set_crop("ep02.mkv", OTHER_BOX)
    assert controller.entry("ep02.mkv").crop == Crop(*OTHER_BOX, Source.MANUAL)
    assert controller.entry("ep02.mkv").review == ReviewState.REVIEWED
    assert crop_changes == [("ep02.mkv",)]
    controller.set_crop("ep02.mkv", OTHER_BOX)                 # same box: nothing to re-measure
    assert crop_changes == [("ep02.mkv",)]

    controller.set_time_ranges("ep01.mkv", [("1:00", None)])
    assert controller.entry("ep01.mkv").time_ranges == TimeRanges([TimeRange("1:00", None)], Source.MANUAL)
    controller.set_time_ranges("ep01.mkv", None)
    assert controller.entry("ep01.mkv").time_ranges == TimeRanges([], Source.MANUAL)

    assert changed.firsts.count("ep01.mkv") >= 3 and changed.firsts.count("ep02.mkv") >= 3
    assert wait_for(lambda: saved(folder)["files"]["ep02.mkv"]["crop"]["source"] == "manual", WAIT_MS)


def test_set_crop_on_a_detected_brightness_asks_autopilot_to_remeasure(
        make_controller, fake_runner, tmp_project, monkeypatch):
    crop_changes = []
    spy_method(monkeypatch, AutoPilot, "on_crop_changed", crop_changes)
    folder = edit_project(tmp_project)
    controller = make_controller()
    controller.open_folder(str(folder))

    controller.set_crop("ep01.mkv", OTHER_BOX)

    assert crop_changes == [("ep01.mkv",)]
    remeasure = fake_runner.last("brightness", "ep01.mkv")
    assert remeasure.job.crop_box == OTHER_BOX
    # recomputed after the submission: the stale brightness is being measured again
    assert controller.entry("ep01.mkv").review == ReviewState.PENDING


def test_bulk_targets_are_every_file_not_skipped(make_controller, fake_runner, tmp_project):
    folder = edit_project(tmp_project)
    controller = make_controller()
    controller.open_folder(str(folder))
    assert controller.bulk_targets("ep01.mkv") == ["ep01.mkv", "ep02.mkv"]
    controller.set_skipped("ep02.mkv", True)
    assert controller.bulk_targets("ep01.mkv") == ["ep01.mkv"]


def test_apply_brightness_to_all_writes_every_target_and_saves(make_controller, fake_runner, tmp_project):
    folder = edit_project(tmp_project)
    controller = make_controller()
    controller.open_folder(str(folder))
    changed = Spy(controller.file_changed)

    assert controller.apply_brightness_to_all("ep01.mkv", 185) == ["ep01.mkv", "ep02.mkv"]

    for name in ("ep01.mkv", "ep02.mkv"):
        assert controller.entry(name).brightness == Brightness(185, Source.MANUAL)
    assert controller.entry("ep01.mkv").review == ReviewState.REVIEWED
    assert controller.entry("ep02.mkv").review == ReviewState.FLAGGED      # its crop is still flagged
    assert set(changed.firsts) >= {"ep01.mkv", "ep02.mkv"}
    assert wait_for(lambda: all(saved(folder)["files"][name]["brightness"] == {"value": 185, "source": "manual"}
                                for name in ("ep01.mkv", "ep02.mkv")), WAIT_MS)


def test_apply_crop_to_all_re_measures_and_drops_strips_per_file(
        make_controller, fake_runner, tmp_project, monkeypatch):
    crop_changes = []
    spy_method(monkeypatch, AutoPilot, "on_crop_changed", crop_changes)
    folder = edit_project(tmp_project)
    controller = make_controller()
    controller.open_folder(str(folder))
    controller.request_strips("ep02.mkv", BOX, [10.0])
    fake_runner.finish(fake_runner.last("strips", "ep02.mkv"), StripsResult("ep02.mkv", BOX, {10.0: _frame(3)}))
    controller.drain_events()
    changed = Spy(controller.file_changed)

    assert controller.apply_crop_to_all("ep01.mkv", OTHER_BOX) == ["ep01.mkv", "ep02.mkv"]

    for name in ("ep01.mkv", "ep02.mkv"):
        assert controller.entry(name).crop == Crop(*OTHER_BOX, Source.MANUAL)
        assert fake_runner.last("brightness", name).job.crop_box == OTHER_BOX   # the detected value is stale
    assert sorted(crop_changes) == [("ep01.mkv",), ("ep02.mkv",)]
    assert controller.strip("ep02.mkv", BOX, 10.0) is None
    assert set(changed.firsts) >= {"ep01.mkv", "ep02.mkv"}


def test_apply_crop_to_all_leaves_a_file_already_on_that_box_alone(
        make_controller, fake_runner, tmp_project, monkeypatch):
    crop_changes = []
    spy_method(monkeypatch, AutoPilot, "on_crop_changed", crop_changes)
    folder = edit_project(tmp_project)
    controller = make_controller()
    controller.open_folder(str(folder))
    controller.set_crop("ep02.mkv", OTHER_BOX)
    crop_changes.clear()

    controller.apply_crop_to_all("ep01.mkv", OTHER_BOX)

    assert crop_changes == [("ep01.mkv",)]                    # ep02's box did not move: nothing to re-measure


def test_copy_and_paste_settings(make_controller, fake_runner, tmp_project, monkeypatch):
    crop_changes = []
    spy_method(monkeypatch, AutoPilot, "on_crop_changed", crop_changes)
    folder = edit_project(tmp_project)
    controller = make_controller()
    controller.open_folder(str(folder))
    assert controller.can_paste() is False
    assert controller.paste_settings("ep02.mkv") is False     # empty clipboard

    controller.set_crop("ep01.mkv", OTHER_BOX)
    controller.set_brightness("ep01.mkv", 190)
    crop_changes.clear()
    controller.copy_settings("ep01.mkv")
    assert controller.can_paste() is True
    changed = Spy(controller.file_changed)
    assert controller.paste_settings("ep02.mkv") is True

    target = controller.entry("ep02.mkv")
    assert target.crop == Crop(*OTHER_BOX, Source.MANUAL)
    assert target.brightness == Brightness(190, Source.MANUAL)
    assert target.review == ReviewState.REVIEWED
    assert crop_changes == [("ep02.mkv",)]
    assert "ep02.mkv" in changed.firsts


def test_mark_not_reviewed_on_an_idle_folder_is_proposed_not_pending(make_controller, fake_runner, tmp_project):
    names = ["ep01.mkv", "ep02.mkv"]
    folder = tmp_project(names, config=manual_config(names, autopilot_enabled=False))
    controller = make_controller()
    controller.open_folder(str(folder))
    for submission in list(fake_runner.submissions):          # thumbnails only: finish them
        fake_runner.finish(submission, None)
    controller.drain_events()
    assert controller.activity().running == () and controller.activity().current is None

    controller.mark_reviewed("ep01.mkv", False)
    assert controller.entry("ep01.mkv").review == ReviewState.PROPOSED
    controller.mark_reviewed("ep01.mkv")
    assert controller.entry("ep01.mkv").review == ReviewState.REVIEWED


def test_set_skipped(make_controller, fake_runner, tmp_project):
    names = ["ep01.mkv", "ep02.mkv"]
    folder = tmp_project(names, config=manual_config(names, autopilot_enabled=False))
    controller = make_controller()
    controller.open_folder(str(folder))
    changed = Spy(controller.file_changed)
    controller.set_skipped("ep02.mkv", True)
    assert controller.entry("ep02.mkv").skipped is True
    assert changed.firsts == ["ep02.mkv"]
    assert wait_for(lambda: saved(folder)["files"]["ep02.mkv"]["skipped"] is True, WAIT_MS)


# --------------------------------------------------------------------------
# Folder settings
# --------------------------------------------------------------------------

def test_update_folder_validates_and_hands_a_snapshot_of_the_old_settings_over(
        make_controller, fake_runner, tmp_project, monkeypatch):
    names = ["ep01.mkv", "ep02.mkv"]
    entries = [make_entry(name, review=ReviewState.REVIEWED) for name in names]    # labels-only: nothing required
    folder = tmp_project(names, config=v2_config(entries, dialogue_enabled=False, labels_enabled=True))
    controller = make_controller()
    controller.open_folder(str(folder))
    assert controller.entry("ep01.mkv").review == ReviewState.REVIEWED
    folder_changed = Spy(controller.folder_changed)

    with pytest.raises(ValueError):
        controller.update_folder(labels_enabled=False)          # both extraction toggles off
    with pytest.raises(TypeError):
        controller.update_folder(no_such_setting=1)
    assert controller.project.folder.labels_enabled is True
    assert folder_changed.calls == []

    calls = []
    spy_method(monkeypatch, AutoPilot, "on_folder_changed", calls, "autopilot")
    original = apply_mod.apply_folder_change

    def apply_spy(project, old, new):
        calls.append(("apply", old, new))
        return original(project, old, new)

    monkeypatch.setattr(apply_mod, "apply_folder_change", apply_spy)

    controller.update_folder(dialogue_enabled=True)

    assert [call[0] for call in calls] == ["apply", "autopilot"]
    _, old, new = calls[0]
    assert old is not new and new is controller.project.folder
    assert old.dialogue_enabled is False and new.dialogue_enabled is True
    assert calls[1][1] is old and calls[1][2] is new
    assert folder_changed.calls == [()]
    # Newly required crops: the files are re-opened and their crops detected.
    assert controller.entry("ep01.mkv").review == ReviewState.PENDING
    assert fake_runner.of_kind("crop")
    assert wait_for(lambda: saved(folder)["folder"]["dialogue_enabled"] is True, WAIT_MS)


def test_update_folder_signals_every_file_it_re_opens(make_controller, fake_runner, tmp_project):
    names = ["ep01.mkv", "ep02.mkv"]
    entries = [make_entry(name, review=ReviewState.REVIEWED) for name in names]
    folder = tmp_project(names, config=v2_config(entries, dialogue_enabled=False, autopilot_enabled=False))
    controller = make_controller()
    controller.open_folder(str(folder))
    changed = Spy(controller.file_changed)

    controller.update_folder(dialogue_enabled=True)

    # apply_folder_change re-opened them; with auto-pilot off the recompute keeps them FLAGGED
    assert {controller.entry(name).review for name in names} == {ReviewState.FLAGGED}
    assert changed.firsts == names


def test_set_label_masks(make_controller, fake_runner, tmp_project):
    folder = tmp_project(["ep01.mkv"])
    controller = make_controller()
    controller.open_folder(str(folder))
    folder_changed = Spy(controller.folder_changed)
    controller.set_label_masks([(10, 20, 30, 40), [1, 2, 3, 4]])
    assert controller.project.folder.label_mask_crops == [(10, 20, 30, 40), (1, 2, 3, 4)]
    assert folder_changed.calls == [()]


# --------------------------------------------------------------------------
# Detections the user asks for
# --------------------------------------------------------------------------

def test_hint_redetects_and_proof(make_controller, fake_runner, tmp_project):
    names = ["ep01.mkv", "ep02.mkv", "ep03.mkv"]
    entries = [make_entry("ep01.mkv", crop=BOX, brightness=200, review=ReviewState.REVIEWED)] + [
        make_entry(name, crop=OTHER_BOX, brightness=205, crop_source=Source.DETECTED,
                   brightness_source=Source.DETECTED) for name in names[1:]]
    folder = tmp_project(names, config=v2_config(entries, autopilot_enabled=False))
    controller = make_controller()
    controller.open_folder(str(folder))

    assert controller.hint_targets("ep01.mkv", "crop") == ["ep02.mkv", "ep03.mkv"]
    controller.redetect_others_with_hint("ep01.mkv", "crop")
    hinted = fake_runner.of_kind("crop")
    assert [s.job.file for s in hinted] == ["ep02.mkv", "ep03.mkv"]
    assert all(s.job.hint == (BOX[1] / 1080, BOX[3] / 1080) for s in hinted)
    with pytest.raises(ValueError):
        controller.redetect_others_with_hint("ep01.mkv", "ranges")

    started, finished = Spy(controller.proof_started), Spy(controller.proof_finished)
    controller.run_proof("ep01.mkv")
    proof = fake_runner.last("proof", "ep01.mkv")
    assert isinstance(proof.job, ProofOcrJob) and proof.job.priority == PROOF_PRIORITY
    assert started.calls == [("ep01.mkv",)] and controller.proof_pending("ep01.mkv")
    result = ProofResult("ep01.mkv", (300.0, 330.0), [(301.0, 303.0, "你好")], 4.2)
    fake_runner.finish(proof, result)
    controller.drain_events()
    assert controller.proof_result("ep01.mkv") == result
    assert finished.calls == [("ep01.mkv",)] and not controller.proof_pending("ep01.mkv")


# --------------------------------------------------------------------------
# Gallery lines (the Brightness tab's random subtitle lines)
# --------------------------------------------------------------------------

def lines_result(submission, times=(512.0, 900.0), tried=24) -> LinesJobResult:
    job = submission.job
    samples = tuple(LineSample(float(t), ((10, 5, 200, 30),), 1) for t in times)
    return LinesJobResult(job.file, LinesResult(samples, tried, job.seed), job.crop_box)


def lines_project(tmp_project, names=("ep01.mkv",), **folder):
    """Files ready to run (MANUAL crop and brightness, reviewed, the user's
    whole file as ranges), with no audio profile yet."""
    entries = [make_entry(name, crop=BOX, brightness=209, ranges=[], review=ReviewState.REVIEWED)
               for name in names]
    return tmp_project(list(names), config=v2_config(entries, **folder))


def test_lines_results_are_applied_as_evidence_and_saved(make_controller, fake_runner, tmp_project):
    name = "ep01.mkv"
    folder = lines_project(tmp_project)
    controller = make_controller()
    controller.open_folder(str(folder))
    assert fake_runner.of_kind("lines") == []                   # the draw waits for the audio profile
    assert "lines" in controller.pending_detectors()[name]
    audio = fake_runner.last("audio_profile", name)
    fake_runner.finish(audio, AudioProfileResult(name, AudioProfile(DURATION, [0.0], [(10.0, 20.0)])))
    controller.drain_events()
    lines = fake_runner.last("lines", name)
    assert isinstance(lines.job, LinesJob) and lines.job.crop_box == BOX and lines.job.priority == 1
    assert [tuple(span) for span in lines.job.speech] == [(10.0, 20.0)]
    changed = Spy(controller.file_changed)

    fake_runner.finish(lines, lines_result(lines))
    controller.drain_events()

    entry = controller.entry(name)
    assert entry.evidence["lines"]["crop_box"] == list(BOX)
    assert [sample["time"] for sample in entry.evidence["lines"]["samples"]] == [512.0, 900.0]
    assert name in changed.firsts
    assert entry.review == ReviewState.REVIEWED
    assert "lines" not in controller.pending_detectors().get(name, set())
    assert wait_for(lambda: "lines" in store.load_project(str(folder)).files[name].evidence, WAIT_MS)


def test_lines_drawn_on_an_old_crop_are_dropped(make_controller, fake_runner, tmp_project):
    name = "ep01.mkv"
    controller = make_controller()
    controller.open_folder(str(lines_project(tmp_project, autopilot_enabled=False)))
    controller.request_lines(name)
    old = fake_runner.last("lines", name)
    fake_runner.start(old)
    controller.drain_events()
    controller.set_crop(name, OTHER_BOX)                        # the user's draw follows the crop
    new = fake_runner.last("lines", name)
    assert new is not old and new.job.crop_box == OTHER_BOX and new.job.priority == LINES_BOOST

    fake_runner.finish(old, lines_result(old))
    controller.drain_events()
    assert "lines" not in controller.entry(name).evidence       # superseded: not applied
    fake_runner.finish(new, lines_result(new, times=(33.0,)))
    controller.drain_events()
    assert controller.entry(name).evidence["lines"]["crop_box"] == list(OTHER_BOX)


def test_request_lines_boosts_the_draw_once(make_controller, fake_runner, tmp_project, monkeypatch):
    boosts = []
    spy_method(monkeypatch, AutoPilot, "boost_lines", boosts)
    name = "ep01.mkv"
    controller = make_controller()
    controller.open_folder(str(lines_project(tmp_project)))
    assert fake_runner.of_kind("lines") == []

    controller.request_lines(name)                              # does not wait for the audio profile
    (boosted,) = fake_runner.of_kind("lines", name)
    assert boosted.job.priority == LINES_BOOST and boosted.job.crop_box == BOX
    controller.request_lines(name)                              # the tab may ask on every refresh
    controller.request_lines(name)
    assert boosts == [(name,)] * 3
    assert fake_runner.of_kind("lines", name) == [boosted]
    assert controller.entry(name).review == ReviewState.REVIEWED

    controller.request_lines("missing.mkv")                     # a view never raises from a refresh
    assert boosts == [(name,)] * 3


def test_request_lines_leaves_a_running_draw_alone(make_controller, fake_runner, tmp_project):
    name = "ep01.mkv"
    controller = make_controller()
    controller.open_folder(str(lines_project(tmp_project)))
    audio = fake_runner.last("audio_profile", name)
    fake_runner.finish(audio, AudioProfileResult(name, AudioProfile(DURATION, [0.0], [])))
    controller.drain_events()
    auto = fake_runner.last("lines", name)
    fake_runner.start(auto)
    controller.drain_events()
    assert controller.activity().current[:2] == ("lines", name)
    assert controller.activity().current[3] == "finding subtitle lines"
    assert controller.running_detectors(name) == {"lines"}

    controller.request_lines(name)
    assert fake_runner.of_kind("lines", name) == [auto]


def test_request_lines_without_a_folder_or_after_shutdown_does_nothing(make_controller, fake_runner, tmp_project):
    controller = make_controller()
    controller.request_lines("ep01.mkv")
    controller.open_folder(str(lines_project(tmp_project)))
    controller.shutdown(timeout=0.5)
    controller.request_lines("ep01.mkv")
    assert fake_runner.of_kind("lines") == []


def test_shuffle_lines_excludes_the_lines_and_tiles_shown_now(make_controller, fake_runner, tmp_project,
                                                              monkeypatch):
    shuffles = []
    spy_method(monkeypatch, AutoPilot, "shuffle_lines", shuffles)
    name = "ep01.mkv"
    controller = make_controller()
    controller.open_folder(str(lines_project(tmp_project, autopilot_enabled=False)))
    entry = controller.entry(name)
    entry.evidence["lines"] = {"crop_box": list(BOX), "seed": 3, "tried": 12,
                               "samples": [{"time": 512.0, "boxes": [], "lines": 1},
                                           {"time": 900.5, "boxes": [], "lines": 1}]}
    entry.evidence["brightness"] = {"tiles": {"dark": 120.0, "bright": 640.0}}

    controller.shuffle_lines(name)

    assert shuffles == [(name, [512.0, 900.5, 120.0, 640.0])]
    (shuffled,) = fake_runner.of_kind("lines", name)
    assert shuffled.job.exclude == (512.0, 900.5, 120.0, 640.0)
    assert shuffled.job.priority == LINES_BOOST and shuffled.job.crop_box == BOX
    controller.shuffle_lines(name)                              # every shuffle is a new draw
    first, second = fake_runner.of_kind("lines", name)
    assert first.job.seed != second.job.seed                    # fresh seeds (31 random bits each)
    with pytest.raises(KeyError):
        controller.shuffle_lines("missing.mkv")


def test_shuffle_lines_needs_an_open_folder(make_controller):
    with pytest.raises(RuntimeError):
        make_controller().shuffle_lines("ep01.mkv")


def test_lines_never_change_review_states_badges_or_counts(make_controller, fake_runner, tmp_project):
    folder, names = counts_project(tmp_project)                # every row has a crop: a draw for each
    controller = make_controller()
    controller.open_folder(str(folder))

    def seen():
        return ({name: controller.entry(name).review for name in names},
                {name: badge_for(controller.entry(name), running_detectors=controller.running_detectors(name),
                                 done=controller.is_done(name), run_state=None) for name in names},
                controller.counts())

    controller.redetect(names[0])                               # a real pending state among them
    before = seen()
    assert {state for state in before[0].values()} >= {ReviewState.PENDING, ReviewState.FLAGGED}

    for name in names:
        controller.request_lines(name)
    draws = fake_runner.of_kind("lines")
    assert len(draws) == len(names)
    assert all("lines" in controller.pending_detectors()[name] for name in names)
    assert seen() == before                                     # queued
    for draw in draws:
        fake_runner.start(draw)
    controller.drain_events()
    assert all("lines" in controller.running_detectors(name) for name in names)
    assert seen() == before                                     # running
    for draw in draws:
        fake_runner.finish(draw, lines_result(draw))
    controller.drain_events()
    assert all("lines" in controller.entry(name).evidence for name in names)
    assert seen() == before                                     # applied


# --------------------------------------------------------------------------
# Confirming a doubted brightness (the "confirm" stage)
# --------------------------------------------------------------------------
# The detector measures a threshold it then doubts, so the file is FLAGGED and
# the queue badges it "check brightness". AutoPilot answers most of those
# doubts without the user: one masked strip, read by the OCR engine at the
# folder's own conf_threshold, stepping the threshold down until it reads
# (core/detect/confirm.py). A confirmed file becomes PROPOSED and badges
# "ready" -- there is deliberately no badge, chip or marker of its own, so
# these tests pin the absence of one as hard as the presence of "ready".

CONFIRM_TIME = 512.0              # the gallery line's time: what probe_time_for picks


def draw_lines(entry, time: float = CONFIRM_TIME, box=BOX) -> None:
    """Give `entry` the one piece of evidence a confirm needs: a gallery line
    on the file's own crop, which `detect_jobs.probe_time_for` reads the probe
    strip's time from. It also settles the file's lines, which a confirm waits
    for like any other detection of its own.

    Set on the loaded entry rather than written into the folder, because
    evidence lives in the disposable `.ocr-cache/` and `v2_config` (to_json
    with include_evidence=False) does not carry it."""
    entry.evidence["lines"] = {"crop_box": list(box), "seed": 7, "tried": 12,
                               "samples": [{"time": float(time), "boxes": [], "lines": 1}]}


def doubted_project(tmp_project, names=("ep01.mkv",), *, value=209, **folder):
    """A folder of files the brightness detector measured and then doubted: a
    DETECTED brightness measured on the file's own crop, flagged "dim-text?"
    -- a reason that doubts a reading which WAS taken, which is exactly what a
    confirm can retire -- and everything else (crop, ranges) already settled by
    the user."""
    entries = [make_entry(name, crop=BOX, brightness=value, brightness_source=Source.DETECTED,
                          ranges=[], flags={"brightness": FLAG_DIM_TEXT})
               for name in names]
    return tmp_project(list(names), config=v2_config(entries, **folder))


def open_doubted(make_controller, fake_runner, tmp_project, names=("ep01.mkv",), **kwargs):
    """(controller, folder) on a doubted-brightness folder, with the confirm
    chain free to run: the thumbnail and audio-profile jobs the open submits
    are finished and the gallery lines are already drawn, because AutoPilot
    never confirms a file that still has a detection of its own outstanding
    (its values, and the strip the probe reads, are still moving)."""
    folder = doubted_project(tmp_project, names, **kwargs)
    controller = make_controller()
    controller.open_folder(str(folder))
    for name in controller.names():
        draw_lines(controller.entry(name))
    for submission in [*fake_runner.of_kind("thumbnail"), *fake_runner.of_kind("audio_profile")]:
        if submission.job.kind == "thumbnail":
            fake_runner.finish(submission, ThumbnailResult(submission.job.file, submission.job.time, None))
        else:
            fake_runner.finish(submission, AudioProfileResult(submission.job.file,
                                                              AudioProfile(DURATION, [0.0], [])))
    controller.drain_events()
    return controller, folder


def confirm_result(submission, *, value=None, text="第一行") -> ConfirmJobResult:
    """What `core.detect.confirm.confirm_brightness` returns for this job:
    every rung of the ladder down to `value`, which is the one that read the
    strip. `value=None` is a ladder that passed nowhere."""
    job = submission.job
    rungs = []
    for threshold in ladder(job.start_value):
        passed = threshold == value
        rungs.append(Rung(threshold=threshold, gated=True, text=text if passed else "",
                          confidence=0.97 if passed else 0.0, passed=passed))
        if passed:
            break                             # the rungs below the one that passed are never probed
    result = ConfirmResult(value=value, probe_time=job.probe_time, rungs=tuple(rungs))
    return ConfirmJobResult(job.file, result, job.crop_box, job.start_value, job.conf_threshold)


def badge(controller, name) -> tuple[str, str]:
    return badge_for(controller.entry(name), running_detectors=controller.running_detectors(name),
                     done=controller.is_done(name), run_state=None)


def test_a_confirm_result_is_applied_on_the_gui_thread_when_the_drain_runs(make_controller, fake_runner,
                                                                           tmp_project, monkeypatch):
    """The runner's listener only enqueues: a confirm reaches
    core.jobs.apply.apply_confirm when drain_events() runs, never from the
    worker thread that finished the job."""
    applied = []
    original = apply_mod.apply_confirm

    def spy(project, r):
        applied.append(r)
        return original(project, r)

    monkeypatch.setattr(apply_mod, "apply_confirm", spy)
    name = "ep01.mkv"
    controller, _folder = open_doubted(make_controller, fake_runner, tmp_project)

    confirm = fake_runner.last("confirm", name)
    assert confirm.job.crop_box == BOX and confirm.job.probe_time == CONFIRM_TIME
    assert confirm.job.start_value == 209

    fake_runner.finish(confirm, confirm_result(confirm, value=209))
    assert applied == []                                        # queued, not applied
    assert controller.entry(name).review == ReviewState.FLAGGED

    controller.drain_events()

    assert [r.file for r in applied] == [name]
    assert controller.entry(name).review == ReviewState.PROPOSED


def test_a_confirmed_file_stops_asking_for_the_user_and_badges_ready(make_controller, fake_runner, tmp_project):
    """The whole point of the stage, end to end through the controller: a file
    FLAGGED on a doubted brightness that the engine reads at the stored value
    becomes PROPOSED, badges "ready" and moves from the "needs you" chip to
    the "ready" one. Its value is untouched -- the ladder passed on the first
    rung -- and nothing on screen says it was confirmed."""
    name = "ep01.mkv"
    controller, folder = open_doubted(make_controller, fake_runner, tmp_project)
    assert controller.entry(name).review == ReviewState.FLAGGED
    assert badge(controller, name) == ("check brightness", "warn")
    assert controller.counts()["needs_you"] == 1 and controller.counts()["ready"] == 0
    changed = Spy(controller.file_changed)

    confirm = fake_runner.last("confirm", name)
    fake_runner.finish(confirm, confirm_result(confirm, value=209))
    controller.drain_events()

    entry = controller.entry(name)
    assert entry.brightness == Brightness(209, Source.DETECTED)     # the rung that passed was the stored one
    assert entry.flags["brightness"] == ""                          # the doubt is retired
    assert entry.review == ReviewState.PROPOSED
    assert badge(controller, name) == ("ready", "default")
    assert controller.counts()["needs_you"] == 0 and controller.counts()["ready"] == 1
    assert name in changed.firsts
    assert controller.startable_files() == [name]                   # a FLAGGED file never was
    assert wait_for(lambda: saved(folder)["files"][name]["review"] == "proposed", WAIT_MS)


def test_a_confirm_that_passed_lower_down_leaves_the_lower_brightness(make_controller, fake_runner, tmp_project):
    """The ladder steps down by 10 until the engine reads the line, and the
    rung that read it is the file's value from then on -- still DETECTED, so
    the file reads as a detected value the user never had to touch."""
    name = "ep01.mkv"
    controller, folder = open_doubted(make_controller, fake_runner, tmp_project)
    confirm = fake_runner.last("confirm", name)
    assert ladder(209)[:3] == [209, 199, 189]

    fake_runner.finish(confirm, confirm_result(confirm, value=189))
    controller.drain_events()

    entry = controller.entry(name)
    assert entry.brightness == Brightness(189, Source.DETECTED)
    assert entry.review == ReviewState.PROPOSED
    assert badge(controller, name) == ("ready", "default")
    assert wait_for(lambda: saved(folder)["files"][name]["brightness"]["value"] == 189, WAIT_MS)


def test_a_failed_confirm_leaves_the_file_flagged_and_still_checking_brightness(make_controller, fake_runner,
                                                                                tmp_project):
    """A ladder that passes nowhere proved nothing: the value, the flag and
    the badge are exactly as the detector left them, and the file still waits
    for the user."""
    name = "ep01.mkv"
    controller, _folder = open_doubted(make_controller, fake_runner, tmp_project)
    confirm = fake_runner.last("confirm", name)

    fake_runner.finish(confirm, confirm_result(confirm, value=None))
    controller.drain_events()

    entry = controller.entry(name)
    assert entry.brightness == Brightness(209, Source.DETECTED)
    assert entry.flags["brightness"] == FLAG_DIM_TEXT
    assert entry.review == ReviewState.FLAGGED
    assert badge(controller, name) == ("check brightness", "warn")
    assert controller.counts()["needs_you"] == 1 and controller.startable_files() == []


def test_applying_a_confirm_schedules_the_save(make_controller, fake_runner, tmp_project):
    """Every applied confirm is saved, as every other applied result is. The
    failing ladder is the test case that isolates it: it changes no value and
    no review state, so the record in evidence["brightness"]["confirm"] -- the
    memo that stops the folder re-probing the same ladder on every open -- is
    the only thing there is to write."""
    name = "ep01.mkv"
    controller, folder = open_doubted(make_controller, fake_runner, tmp_project)
    confirm = fake_runner.last("confirm", name)

    fake_runner.finish(confirm, confirm_result(confirm, value=None))
    controller.drain_events()

    def recorded():
        evidence = store.load_project(str(folder)).files[name].evidence
        return (evidence.get("brightness") or {}).get("confirm")

    assert wait_for(lambda: recorded() is not None, WAIT_MS)
    assert recorded()["start_value"] == 209 and recorded()["value"] is None
    assert recorded()["probe_time"] == CONFIRM_TIME
    assert saved(folder)["files"][name]["review"] == "flagged"


def test_a_running_confirm_shows_in_the_activity_strip_but_changes_no_badge(make_controller, fake_runner,
                                                                            tmp_project):
    """A confirm is named where the machine's work is named, and nowhere else.

    The activity strip says "checking brightness" because it is real GPU work
    that a folder of flagged files spends minutes in, and a strip reading
    "idle" through it would be a lie. But it is not a pending detection: it
    gives the file no pending badge and does not appear in
    pending_detectors(), so a folder of 180 flagged rows does not churn
    through a transient "waiting" that tells the user nothing."""
    name = "ep01.mkv"
    controller, _folder = open_doubted(make_controller, fake_runner, tmp_project)
    before = badge(controller, name)
    confirm = fake_runner.last("confirm", name)

    fake_runner.start(confirm)
    controller.drain_events()

    assert controller.activity().current[:2] == ("confirm", name)
    assert controller.activity().current[3] == "checking brightness"
    assert controller.running_detectors(name) == {"confirm"}     # an auto-pilot kind, so it is counted
    assert "confirm" not in controller.pending_detectors().get(name, set())
    assert not controller.is_detection_kind("confirm")           # the top bar's "detecting" dot stays dark
    assert badge(controller, name) == before == ("check brightness", "warn")
    assert controller.entry(name).review == ReviewState.FLAGGED


# --------------------------------------------------------------------------
# Counts, done, startable files
# --------------------------------------------------------------------------

def mixed_project(tmp_project):
    names = ["a.mkv", "b.mkv", "c.mkv", "d.mkv", "e.mkv", "f.mkv", "g.mkv"]
    folder = tmp_project(names, config=manual_config(names, autopilot_enabled=False))
    (folder / "chi").mkdir()
    (folder / "chi" / "f.ass").write_text("[Script Info]\n", encoding="utf-8")
    (folder / "chi" / "g.ass").write_bytes(b"")                 # empty: not done
    return folder


def set_states(controller):
    states = {"a.mkv": ReviewState.REVIEWED, "b.mkv": ReviewState.PROPOSED, "c.mkv": ReviewState.FLAGGED,
              "d.mkv": ReviewState.PENDING, "e.mkv": ReviewState.REVIEWED, "f.mkv": ReviewState.PROPOSED,
              "g.mkv": ReviewState.PROPOSED}
    for name, state in states.items():
        controller.entry(name).review = state
    controller.entry("e.mkv").skipped = True


def test_counts_for_a_mix_of_states(make_controller, fake_runner, tmp_project):
    controller = make_controller()
    controller.open_folder(str(mixed_project(tmp_project)))
    set_states(controller)
    assert controller.is_done("f.mkv") and not controller.is_done("g.mkv") and not controller.is_done("a.mkv")
    # e (reviewed) is skipped and f (proposed) is done: their badges say so, and no chip counts them
    assert controller.counts() == {"reviewed": 1, "needs_you": 1, "detecting": 1, "ready": 3}


# (review state, skipped, done) -> (badge text, chip it counts under or None, ready)
COUNT_CASES = [
    (ReviewState.REVIEWED, False, False, "reviewed", "reviewed", True),
    (ReviewState.REVIEWED, True, False, "skipped", None, False),
    (ReviewState.REVIEWED, False, True, "done", None, False),
    (ReviewState.REVIEWED, True, True, "skipped", None, False),
    (ReviewState.FLAGGED, False, False, "check crop", "needs_you", False),
    (ReviewState.FLAGGED, True, False, "skipped", None, False),
    (ReviewState.FLAGGED, False, True, "done", None, False),
    (ReviewState.FLAGGED, True, True, "skipped", None, False),
    (ReviewState.PENDING, False, False, "waiting", "detecting", False),
    (ReviewState.PENDING, True, False, "skipped", None, False),
    (ReviewState.PENDING, False, True, "done", None, False),
    (ReviewState.PENDING, True, True, "skipped", None, False),
    (ReviewState.PROPOSED, False, False, "ready", None, True),
    (ReviewState.PROPOSED, True, False, "skipped", None, False),
    (ReviewState.PROPOSED, False, True, "done", None, False),
    (ReviewState.PROPOSED, True, True, "skipped", None, False),
]


def counts_project(tmp_project):
    """One file per COUNT_CASES row; the crop is DETECTED with a blocking flag,
    so a FLAGGED row's badge is "check crop"."""
    names = [f"f{index:02d}.mkv" for index in range(len(COUNT_CASES))]
    entries = [make_entry(name, crop=BOX, brightness=209, crop_source=Source.DETECTED,
                          flags={"crop": FLAG_LOW_AGREEMENT}) for name in names]
    folder = tmp_project(names, config=v2_config(entries, autopilot_enabled=False))
    (folder / "chi").mkdir()
    for name, (_state, _skipped, done, *_rest) in zip(names, COUNT_CASES, strict=True):
        if done:
            (folder / "chi" / name.replace(".mkv", ".ass")).write_text("Dialogue: x\n", encoding="utf-8")
    return folder, names


def apply_count_cases(controller, names):
    for name, (state, skipped, *_rest) in zip(names, COUNT_CASES, strict=True):
        controller.entry(name).review = state
        controller.entry(name).skipped = skipped


def expected_counts(cases) -> dict[str, int]:
    expected = {"reviewed": 0, "needs_you": 0, "detecting": 0, "ready": 0}
    for *_flags, _badge, chip, ready in cases:
        if chip is not None:
            expected[chip] += 1
        expected["ready"] += ready
    return expected


@pytest.mark.parametrize("index", range(len(COUNT_CASES)))
def test_counts_follow_the_row_badges(make_controller, fake_runner, tmp_project, index):
    folder, names = counts_project(tmp_project)
    controller = make_controller()
    controller.open_folder(str(folder))
    apply_count_cases(controller, names)
    name, case = names[index], COUNT_CASES[index]
    state, skipped, done, badge, _chip, _ready = case
    entry = controller.entry(name)
    assert (entry.review, entry.skipped, controller.is_done(name)) == (state, skipped, done)
    assert badge_for(entry, running_detectors=controller.running_detectors(name), done=done,
                     run_state=None)[0] == badge
    for other in names:                                         # this row alone
        if other != name:
            controller.entry(other).skipped = True
    assert controller.counts() == expected_counts([case])


def test_counts_keep_run_rows_in_their_review_state_bucket(make_controller, fake_runner, tmp_project,
                                                           notifications):
    folder, names = counts_project(tmp_project)
    controller = make_controller()
    controller.open_folder(str(folder))
    apply_count_cases(controller, names)
    before = controller.counts()
    assert before == expected_counts(COUNT_CASES) == {"reviewed": 1, "needs_you": 1, "detecting": 1, "ready": 2}

    controller.start_run(names)
    run = fake_runner.last("run")
    fake_runner.emit(run, "run_file_started", file=names[0])                     # reviewed row: "running"
    fake_runner.emit(run, "run_file_started", file=names[4])                     # flagged row: then "failed"
    fake_runner.emit(run, "run_file_finished", file=names[4], result={"ok": False, "lines": 0, "error": "boom"})
    fake_runner.emit(run, "run_file_started", file=names[8])                     # pending row: "running"
    controller.drain_events()
    assert controller.run_snapshot().row(names[0]).state == "running"
    assert controller.run_snapshot().row(names[4]).state == "failed"
    assert controller.counts() == before


def test_startable_files_and_overwrite(make_controller, fake_runner, tmp_project):
    controller = make_controller()
    controller.open_folder(str(mixed_project(tmp_project)))
    set_states(controller)

    assert controller.startable_files() == ["a.mkv", "b.mkv", "g.mkv"]
    assert controller.startable_files(include_done=True) == ["a.mkv", "b.mkv", "f.mkv", "g.mkv"]
    assert controller.files_needing_overwrite(["a.mkv", "f.mkv", "c.mkv"]) == ["f.mkv"]

    controller.start_run(["a.mkv"])
    assert controller.startable_files() == ["b.mkv", "g.mkv"]   # a is in the run


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

def test_start_run_submits_a_snapshot_and_holds_autopilot(make_controller, fake_runner, tmp_project):
    names = ["ep01.mkv", "ep02.mkv", "ep03.mkv"]
    config = manual_config(names, labels_enabled=False, ocr_parallel=3)
    config["files"]["ep02.mkv"]["time_ranges"] = {"ranges": [{"start": "1:00", "end": "2:00"},
                                                             {"start": "3:00", "end": None}],
                                                  "source": "manual"}
    folder = tmp_project(names, config=config)
    controller = make_controller()
    controller.open_folder(str(folder))
    run_changed = Spy(controller.run_changed)
    assert controller.run_snapshot() is None

    controller.start_run(["ep01.mkv", "ep02.mkv"])

    run = fake_runner.last("run")
    job = run.job
    project = controller.project
    assert isinstance(job, RunJob)
    assert job.files == tuple(RunFile(name, ocr_call_for(project.files[name], project.folder, project.path))
                              for name in ["ep01.mkv", "ep02.mkv"])
    assert job.files[1].call.time_ranges == [("1:00", "2:00"), ("3:00", "")]
    assert job.parallel == 3 and job.project_dir == project.path
    assert {(lane, only) for lane, only in fake_runner.pauses} == {(Lane.GPU, is_held_job),
                                                                   (Lane.CPU, is_held_job)}
    snapshot = controller.run_snapshot()
    assert isinstance(snapshot, RunSnapshot)
    assert [(row.name, row.state) for row in snapshot.files] == [("ep01.mkv", "queued"), ("ep02.mkv", "queued")]
    assert not snapshot.finished and not snapshot.paused
    assert run_changed.calls

    controller.set_brightness("ep01.mkv", 150)                  # edits do not reach the running snapshot
    assert job.files[0].call.kwargs["brightness_threshold"] == 209
    with pytest.raises(RuntimeError):
        controller.start_run(["ep03.mkv"])


def test_start_run_refuses_files_that_write_the_same_output(make_controller, fake_runner, tmp_project):
    names = ["a.mkv", "a.mp4"]
    folder = tmp_project(names, config=manual_config(names))
    controller = make_controller()
    controller.open_folder(str(folder))
    with pytest.raises(ValueError, match=r"a\.mkv and a\.mp4 both write chi/a\.ass"):
        controller.start_run(names)
    assert fake_runner.of_kind("run") == [] and fake_runner.pauses == []
    assert controller.run_snapshot() is None
    controller.pause_autopilot()
    controller.resume_autopilot()                               # no run hold left behind
    assert set(fake_runner.resumes) == {Lane.GPU, Lane.CPU}


def test_run_events_update_the_snapshot_logs_and_live_feed(make_controller, fake_runner, tmp_project):
    names = ["ep01.mkv", "ep02.mkv"]
    folder = tmp_project(names, config=manual_config(names))
    controller = make_controller()
    controller.open_folder(str(folder))
    controller.start_run(names)
    run = fake_runner.last("run")
    subtitles, logs = Spy(controller.run_subtitle), Spy(controller.log_appended)
    run_changed = Spy(controller.run_changed)

    fake_runner.start(run)
    fake_runner.emit(run, "run_file_started", file="ep01.mkv")
    fake_runner.emit(run, "run_file_log", file="ep01.mkv", message="Starting OCR: ep01.mkv\n")
    fake_runner.emit(run, "run_file_progress", file="ep01.mkv", progress=0.25, message="Extracting dialogue")
    fake_runner.emit(run, "run_subtitle", file="ep01.mkv", result=(1.5, 3, "你好"))
    controller.drain_events()

    row = controller.run_snapshot().row("ep01.mkv")
    assert (row.state, row.phase, row.progress, row.lines) == ("running", "Extracting dialogue", 0.25, 1)
    assert row.started_at is not None
    assert controller.run_snapshot().row("ep02.mkv").state == "queued"
    assert subtitles.calls == [("ep01.mkv", 1.5, 3.0, "你好")]
    assert controller.run_subtitles("ep01.mkv") == [(1.5, 3.0, "你好")]
    assert ("ep01.mkv", "Starting OCR: ep01.mkv\n") in logs.calls
    assert run_changed.calls

    fake_runner.emit(run, "run_file_finished", file="ep01.mkv", result={"ok": True, "lines": 12, "error": ""})
    fake_runner.emit(run, "run_file_started", file="ep02.mkv")
    fake_runner.emit(run, "run_file_finished", file="ep02.mkv", result={"ok": False, "lines": 0, "error": "boom"})
    fake_runner.emit(run, "log", message="ep02.mkv: failed\nTraceback (most recent call last): boom\n")
    controller.drain_events()

    snapshot = controller.run_snapshot()
    done, failed = snapshot.row("ep01.mkv"), snapshot.row("ep02.mkv")
    assert (done.state, done.lines, done.progress) == ("done", 12, 1.0) and done.finished_at is not None
    assert (failed.state, failed.error) == ("failed", "boom")
    assert any(key == "Pipeline" and "Traceback" in text for key, text in logs.calls)


@pytest.mark.parametrize("user_paused", [False, True])
def test_run_end_releases_autopilot_only_without_a_user_pause(
        make_controller, fake_runner, tmp_project, notifications, user_paused):
    names = ["ep01.mkv", "ep02.mkv"]
    folder = tmp_project(names, config=manual_config(names))
    controller = make_controller()
    controller.open_folder(str(folder))
    if user_paused:
        controller.pause_autopilot()
    controller.start_run(["ep01.mkv"])
    run = fake_runner.last("run")
    assert controller.autopilot_held() and fake_runner.resumes == []

    run_to_end(fake_runner, run, RunSummary(["ep01.mkv"], {}, [], 42.0))
    controller.drain_events()

    snapshot = controller.run_snapshot()
    assert snapshot.finished and snapshot.summary == RunSummary(["ep01.mkv"], {}, [], 42.0)
    if user_paused:
        assert fake_runner.resumes == []
        assert controller.activity().paused and controller.autopilot_held()
        controller.resume_autopilot()
    assert set(fake_runner.resumes) == {Lane.GPU, Lane.CPU}
    assert not controller.activity().paused and not controller.autopilot_held()
    assert len(notifications) == 1
    assert notifications[0][:6] == ["notify-send", "-a", "OCR Manager", "-u", "normal", "OCR Complete"]
    assert notifications[0][6].startswith("Finished in 42s | Avg: ")


def test_resuming_autopilot_during_a_run_keeps_the_run_hold(
        make_controller, fake_runner, tmp_project, notifications):
    names = ["ep01.mkv", "ep02.mkv"]
    folder = tmp_project(names, config=manual_config(names))
    controller = make_controller()
    controller.open_folder(str(folder))
    controller.pause_autopilot()
    controller.start_run(["ep01.mkv"])
    run = fake_runner.last("run")

    controller.resume_autopilot()
    assert fake_runner.resumes == []
    assert not controller.activity().paused and controller.autopilot_held()

    run_to_end(fake_runner, run, RunSummary(["ep01.mkv"], {}, [], 1.0))
    controller.drain_events()
    assert set(fake_runner.resumes) == {Lane.GPU, Lane.CPU}
    assert not controller.autopilot_held()


def test_a_run_with_a_failed_file_notifies_critical(make_controller, fake_runner, tmp_project, notifications):
    names = ["ep01.mkv", "ep02.mkv"]
    folder = tmp_project(names, config=manual_config(names))
    controller = make_controller()
    controller.open_folder(str(folder))
    logs = Spy(controller.log_appended)
    controller.start_run(names)
    run = fake_runner.last("run")
    run_to_end(fake_runner, run, RunSummary(["ep01.mkv"], {"ep02.mkv": "boom"}, [], 9.0))
    controller.drain_events()
    assert notifications == [["notify-send", "-a", "OCR Manager", "-u", "critical",
                              "OCR Failed", "Pipeline encountered an error"]]
    pipeline = "".join(text for key, text in logs.calls if key == "Pipeline")
    assert "OCR completed: 1/2 files successful" in pipeline
    assert "Warning: 1 file(s) failed OCR" in pipeline


def test_a_missing_notify_send_is_ignored(make_controller, fake_runner, tmp_project, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("notify-send")

    monkeypatch.setattr(subprocess, "run", missing)
    names = ["ep01.mkv"]
    folder = tmp_project(names, config=manual_config(names))
    controller = make_controller()
    controller.open_folder(str(folder))
    controller.start_run(names)
    run_to_end(fake_runner, fake_runner.last("run"), RunSummary(names, {}, [], 3.0))
    controller.drain_events()
    assert controller.run_snapshot().finished


def test_a_failed_run_job_fails_its_unfinished_files(make_controller, fake_runner, tmp_project, notifications):
    names = ["ep01.mkv", "ep02.mkv"]
    folder = tmp_project(names, config=manual_config(names))
    controller = make_controller()
    controller.open_folder(str(folder))
    controller.start_run(names)
    run = fake_runner.last("run")
    fake_runner.emit(run, "run_file_started", file="ep01.mkv")
    fake_runner.finish(run, None, "failed", message="disk full", error="Traceback: disk full\n")
    controller.drain_events()

    snapshot = controller.run_snapshot()
    assert snapshot.finished and snapshot.summary is None and snapshot.error == "disk full"
    assert [(row.state, row.error) for row in snapshot.files] == [("failed", "disk full"), ("failed", "disk full")]
    assert notifications == [["notify-send", "-a", "OCR Manager", "-u", "critical",
                              "OCR Failed", "Pipeline encountered an error"]]
    assert "Traceback: disk full" in controller.log_text("Pipeline")
    assert not controller.autopilot_held()


def test_run_end_re_derives_done_states(make_controller, fake_runner, tmp_project, notifications):
    names = ["ep01.mkv", "ep02.mkv"]
    folder = tmp_project(names, config=manual_config(names))
    controller = make_controller()
    controller.open_folder(str(folder))
    controller.start_run(names)
    run = fake_runner.last("run")
    changed = Spy(controller.file_changed)
    (folder / "chi").mkdir()
    (folder / "chi" / "ep01.ass").write_text("Dialogue: x\n", encoding="utf-8")
    run_to_end(fake_runner, run, RunSummary(["ep01.mkv"], {}, ["ep02.mkv"], 3.0))
    controller.drain_events()
    assert controller.is_done("ep01.mkv") and not controller.is_done("ep02.mkv")
    assert "ep01.mkv" in changed.firsts
    snapshot = controller.run_snapshot()
    assert snapshot.row("ep02.mkv").state == "cancelled"
    assert controller.startable_files() == ["ep02.mkv"]


def test_parallel_files_changed_during_a_run_reach_the_run_job(
        make_controller, fake_runner, tmp_project, notifications, monkeypatch):
    names = ["ep01.mkv", "ep02.mkv", "ep03.mkv"]
    folder = tmp_project(names, config=manual_config(names, ocr_parallel=2))
    controller = make_controller()
    controller.open_folder(str(folder))
    calls = []
    spy_method(monkeypatch, RunJob, "set_parallel", calls)

    controller.update_folder(ocr_parallel=3)                    # no run: nothing to reach
    assert calls == []
    controller.start_run(names)
    run = fake_runner.last("run")
    assert run.job.parallel == 3 and controller.run_snapshot().parallel == 3
    run_changed = Spy(controller.run_changed)

    controller.update_folder(ocr_parallel=4)
    assert calls == [(4,)]
    assert run.job.parallel == 4 and controller.project.folder.ocr_parallel == 4
    assert controller.run_snapshot().parallel == 4 and run_changed.calls

    with pytest.raises(ValueError):
        controller.update_folder(ocr_parallel=0)
    assert run.job.parallel == 4 and controller.project.folder.ocr_parallel == 4
    controller.update_folder(labels_enabled=not controller.project.folder.labels_enabled, ocr_parallel=4)
    assert calls == [(4,)]                                      # parallel unchanged: not sent again

    run_to_end(fake_runner, run, RunSummary(names, {}, [], 3.0))
    controller.drain_events()
    controller.update_folder(ocr_parallel=1)                    # the run is over
    assert run.job.parallel == 4 and controller.run_snapshot().parallel == 4


def test_a_run_job_that_cannot_add_workers_is_logged_and_the_setting_kept(
        make_controller, fake_runner, tmp_project, monkeypatch):
    names = ["ep01.mkv", "ep02.mkv"]
    folder = tmp_project(names, config=manual_config(names, ocr_parallel=1))
    controller = make_controller()
    controller.open_folder(str(folder))
    controller.start_run(names)

    def refuse(self, parallel):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(RunJob, "set_parallel", refuse)
    controller.update_folder(ocr_parallel=2)
    assert controller.project.folder.ocr_parallel == 2
    assert "can't start new thread" in controller.log_text("Pipeline")


def test_pause_resume_and_stop_the_run(make_controller, fake_runner, tmp_project, notifications):
    names = ["ep01.mkv", "ep02.mkv"]
    folder = tmp_project(names, config=manual_config(names))
    controller = make_controller()
    controller.open_folder(str(folder))
    controller.start_run(names)
    run = fake_runner.last("run")
    run_changed = Spy(controller.run_changed)

    controller.pause_run()
    assert run.job._paused and controller.run_snapshot().paused
    controller.resume_run()
    assert not run.job._paused and not controller.run_snapshot().paused

    controller.stop_run()
    assert fake_runner.cancelled_keys == ["run"]
    assert controller.run_snapshot().stopping
    assert run_changed.calls

    fake_runner.emit(run, "run_file_started", file="ep01.mkv")
    fake_runner.emit(run, "run_file_finished", file="ep01.mkv", result={"ok": False, "lines": 0, "error": "cancelled"})
    fake_runner.finish(run, RunSummary([], {}, names, 2.0), "cancelled")
    controller.drain_events()
    snapshot = controller.run_snapshot()
    assert snapshot.finished
    assert [row.state for row in snapshot.files] == ["cancelled", "cancelled"]
    assert notifications == []                                  # a user stop does not notify (as today)
    assert set(fake_runner.resumes) == {Lane.GPU, Lane.CPU}


# --------------------------------------------------------------------------
# Folder watcher
# --------------------------------------------------------------------------

def test_watcher_adds_new_videos_and_drops_vanished_ones(make_controller, fake_runner, tmp_project, monkeypatch):
    added, removed = [], []
    spy_method(monkeypatch, AutoPilot, "on_files_added", added)
    spy_method(monkeypatch, AutoPilot, "on_files_removed", removed)
    folder = tmp_project(["ep01.mkv", "ep02.mkv"])
    controller = make_controller()
    controller.open_folder(str(folder))
    files = Spy(controller.files_changed)

    (folder / "ep03.mkv").write_bytes(b"placeholder video")
    assert wait_for(lambda: "ep03.mkv" in controller.names(), WAIT_MS)
    assert files.calls
    assert added == [(["ep03.mkv"],)]
    assert fake_runner.of_kind("metadata", "ep03.mkv")

    (folder / "ep01.mkv").unlink()
    assert wait_for(lambda: "ep01.mkv" not in controller.names(), WAIT_MS)
    assert removed == [(["ep01.mkv"],)]
    predicate = fake_runner.cancel_predicates[-1]
    assert predicate(SimpleNamespace(file="ep01.mkv"))
    assert not predicate(SimpleNamespace(file="ep02.mkv")) and not predicate(SimpleNamespace(file=None))
    assert wait_for(lambda: (folder / ".ocr.json").exists()
                          and "ep01.mkv" not in saved(folder)["files"], WAIT_MS)


def test_watcher_changes_wait_for_the_run_to_end(make_controller, fake_runner, tmp_project, notifications):
    names = ["ep01.mkv", "ep02.mkv"]
    folder = tmp_project(names, config=manual_config(names))
    controller = make_controller(watch_debounce_ms=20)
    controller.open_folder(str(folder))
    controller.start_run(["ep01.mkv"])
    run = fake_runner.last("run")

    (folder / "ep03.mkv").write_bytes(b"placeholder video")
    QTest.qWait(400)
    assert "ep03.mkv" not in controller.names()

    run_to_end(fake_runner, run, RunSummary(["ep01.mkv"], {}, [], 1.0))
    controller.drain_events()
    assert "ep03.mkv" in controller.names()


def test_a_removed_files_proof_is_never_reported(make_controller, fake_runner, tmp_project):
    names = ["ep01.mkv", "ep02.mkv"]
    folder = tmp_project(names, config=manual_config(names, autopilot_enabled=False))
    controller = make_controller(watch_debounce_ms=20)
    controller.open_folder(str(folder))
    finished = Spy(controller.proof_finished)
    controller.run_proof("ep02.mkv")
    old_proof = fake_runner.last("proof", "ep02.mkv")
    fake_runner.start(old_proof)

    (folder / "ep02.mkv").unlink()
    assert wait_for(lambda: "ep02.mkv" not in controller.names(), WAIT_MS)
    assert not controller.proof_pending("ep02.mkv")

    (folder / "ep02.mkv").write_bytes(b"placeholder video")    # the same name comes back
    assert wait_for(lambda: "ep02.mkv" in controller.names(), WAIT_MS)
    metadata = fake_runner.last("metadata", "ep02.mkv")
    fake_runner.finish(metadata, MetadataResult("ep02.mkv", 1920, 1080, DURATION, 23.976))
    controller.drain_events()
    controller.run_proof("ep02.mkv")
    new_proof = fake_runner.last("proof", "ep02.mkv")
    stale = ProofResult("ep02.mkv", (560.0, 590.0), [(561.0, 562.0, "old")], 1.0)
    fake_runner.finish(old_proof, stale)                       # the removed entry's proof ends first
    controller.drain_events()
    assert finished.calls == [] and controller.proof_result("ep02.mkv") is None
    assert controller.proof_pending("ep02.mkv")

    fresh = ProofResult("ep02.mkv", (560.0, 590.0), [(561.0, 562.0, "new")], 1.0)
    fake_runner.finish(new_proof, fresh)
    controller.drain_events()
    assert finished.calls == [("ep02.mkv",)] and controller.proof_result("ep02.mkv") == fresh


def test_a_proof_ending_in_the_batch_that_removes_its_file_emits_nothing(
        make_controller, fake_runner, tmp_project, notifications):
    names = ["ep01.mkv", "ep02.mkv"]
    folder = tmp_project(names, config=manual_config(names, autopilot_enabled=False))
    controller = make_controller(watch_debounce_ms=20)
    controller.open_folder(str(folder))
    finished = Spy(controller.proof_finished)
    controller.start_run(["ep01.mkv"])
    run = fake_runner.last("run")
    controller.run_proof("ep02.mkv")                           # a proof runs during a run (ruling C4)
    proof = fake_runner.last("proof", "ep02.mkv")
    (folder / "ep02.mkv").unlink()                             # ignored while the run is on
    QTest.qWait(100)
    assert "ep02.mkv" in controller.names()

    # One drain: the proof ends while ep02 is still listed, then the run's end reconciles it away.
    fake_runner.finish(proof, ProofResult("ep02.mkv", (560.0, 590.0), [], 1.0))
    run_to_end(fake_runner, run, RunSummary(["ep01.mkv"], {}, [], 1.0))
    controller.drain_events()

    assert "ep02.mkv" not in controller.names()
    assert finished.calls == [] and controller.proof_result("ep02.mkv") is None
    assert not controller.proof_pending("ep02.mkv")


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------

def test_close_folder_saves_cancels_and_forgets(make_controller, fake_runner, tmp_project):
    names = ["ep01.mkv"]
    folder = tmp_project(names, config=manual_config(names))
    controller = make_controller(save_debounce_ms=60_000)
    controller.open_folder(str(folder))
    controller.set_brightness("ep01.mkv", 177)
    closed = Spy(controller.project_closed)

    controller.close_folder()

    assert closed.calls == [()]
    assert controller.project is None and controller.names() == []
    assert saved(folder)["files"]["ep01.mkv"]["brightness"] == {"value": 177, "source": "manual"}
    assert fake_runner.cancel_predicates[-1](SimpleNamespace(file=None, kind="run"))


def test_shutdown_saves_and_shuts_the_runner_down(make_controller, fake_runner, tmp_project):
    names = ["ep01.mkv"]
    folder = tmp_project(names, config=manual_config(names))
    controller = make_controller(save_debounce_ms=60_000)
    controller.open_folder(str(folder))
    controller.set_brightness("ep01.mkv", 177)
    assert saved(folder)["files"]["ep01.mkv"]["brightness"]["value"] == 209    # debounced: not yet

    assert controller.shutdown(timeout=2.5) is True

    assert saved(folder)["files"]["ep01.mkv"]["brightness"] == {"value": 177, "source": "manual"}
    assert fake_runner.shutdown_calls == [2.5]


def test_open_folder_needs_a_directory(make_controller, tmp_path):
    controller = make_controller()
    with pytest.raises(NotADirectoryError):
        controller.open_folder(str(tmp_path / "missing"))
    assert controller.project is None


def test_a_failed_save_is_reported_and_keeps_the_values(make_controller, fake_runner, tmp_project, monkeypatch):
    names = ["ep01.mkv"]
    folder = tmp_project(names, config=manual_config(names))
    controller = make_controller()
    controller.open_folder(str(folder))
    failures = Spy(controller.save_failed)

    def refuse(project):
        raise PermissionError("read-only folder")

    monkeypatch.setattr(store, "save_project", refuse)
    controller.set_brightness("ep01.mkv", 150)
    assert wait_for(lambda: failures.calls, WAIT_MS)
    assert "read-only folder" in failures.calls[0][0]
    assert "read-only folder" in controller.log_text("Pipeline")
    assert controller.entry("ep01.mkv").brightness.value == 150


def test_notification_text_matches_todays_window():
    assert (notify_duration(42.9), notify_duration(65), notify_duration(3725)) == ("42s", "1m 5s", "1h 2m")
    tracker = RunTracker(["a.mkv"], 1, 100.0)
    tracker.file_started("a.mkv", 100.0)
    tracker.file_finished("a.mkv", {"ok": True, "lines": 2, "error": ""}, 130.0)
    tracker.finish(RunSummary(["a.mkv"], {}, [], 65.0), "", 165.0)
    assert notification_for(tracker.snapshot()) == ("OCR Complete", "Finished in 1m 5s | Avg: 30s/file", "normal")
    empty = RunTracker([], 1, 5.0)
    empty.finish(RunSummary([], {}, [], 0.0), "", 5.0)
    assert notification_for(empty.snapshot()) == ("OCR Complete", "All phases completed", "normal")


def test_logbook_caps_each_key_like_todays_log_store():
    book = LogBook()
    assert book.append("a.mkv", "") == ""
    assert book.append("a.mkv", "line") == "line\n"
    book.append("Pipeline", "x" * (LOG_LIMIT + 10))
    assert len(book.text("Pipeline")) == LOG_LIMIT
    book.append("Detections", "b.mkv: crop failed")
    assert book.keys() == ["Pipeline", "Detections", "a.mkv"]
    book.clear(keep=("Detections",))
    assert book.keys() == ["Detections"] and book.text("a.mkv") == ""


def test_default_runner_factory_builds_a_job_runner_with_cpu_workers_for_the_machine():
    """One GPU worker and one run worker, whatever the machine; the CPU lane
    scales with it (CPU_WORKERS), because thumbnails, audio profiles and the
    views' own fetches all queue there and the lane is latency-bound."""
    runner = default_runner_factory(lambda event: None)
    try:
        assert isinstance(runner, JobRunner)
        assert 2 <= CPU_WORKERS <= 8
        names = sorted(t.name for t in runner._threads)
        assert names == sorted([f"jobs-cpu-{i}" for i in range(CPU_WORKERS)] + ["jobs-gpu-0", "jobs-run-0"])
    finally:
        assert runner.shutdown(timeout=2.0)


def test_append_log_records_text_under_a_key(make_controller, tmp_project):
    controller = make_controller()
    controller.open_folder(str(tmp_project(["ep01.mkv"])))
    logs = Spy(controller.log_appended)
    controller.append_log("Pipeline", "Unexpected error:\nTraceback ...")
    assert "Traceback ..." in controller.log_text("Pipeline")
    assert logs.calls and logs.calls[-1][0] == "Pipeline"


# --------------------------------------------------------------------------
# Frames and strips for the review views (plan 3C Task 1)
# --------------------------------------------------------------------------

def _frames_controller(make_controller, tmp_project, names=("ep01.mkv", "ep02.mkv")):
    names = list(names)
    folder = tmp_project(names, config=manual_config(names, autopilot_enabled=False))
    controller = make_controller()
    controller.open_folder(str(folder))
    return controller, folder


def _frame(value: int, size: int = 8) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_request_frames_submits_one_job_for_the_times_that_are_missing(
        make_controller, fake_runner, tmp_project):
    controller, _ = _frames_controller(make_controller, tmp_project)

    controller.request_frames("ep01.mkv", [10.0, 20.0])
    controller.request_frames("ep01.mkv", [10.0, 20.0])          # still in flight: no second job

    jobs = [s.job for s in fake_runner.of_kind("frames", "ep01.mkv")]
    assert len(jobs) == 1
    assert jobs[0].times == (10.0, 20.0)
    assert jobs[0].lane is Lane.CPU

    fake_runner.finish(fake_runner.last("frames", "ep01.mkv"),
                       FramesResult("ep01.mkv", {10.0: _frame(10), 20.0: _frame(20)}))
    controller.drain_events()

    controller.request_frames("ep01.mkv", [10.0, 20.0, 30.0])    # only 30.0 is missing now
    jobs = [s.job for s in fake_runner.of_kind("frames", "ep01.mkv")]
    assert len(jobs) == 2 and jobs[1].times == (30.0,)


def test_a_finished_frame_job_fills_the_cache_and_announces_each_frame(
        make_controller, fake_runner, tmp_project):
    controller, _ = _frames_controller(make_controller, tmp_project)
    ready = Spy(controller.frame_ready)
    image = _frame(42)

    controller.request_frames("ep01.mkv", [10.0, 20.0])
    assert controller.frame("ep01.mkv", 10.0) is None
    fake_runner.finish(fake_runner.last("frames", "ep01.mkv"),
                       FramesResult("ep01.mkv", {10.0: image}))   # 20.0 could not be grabbed
    controller.drain_events()

    assert controller.frame("ep01.mkv", 10.0) is image
    assert controller.frame("ep01.mkv", 20.0) is None
    assert ready.calls == [("ep01.mkv", 10.0)]


def test_a_frame_job_that_fails_is_not_asked_for_again(make_controller, fake_runner, tmp_project):
    """A failure marks every time it was asked for: one attempt per session,
    not one per repaint."""
    controller, _ = _frames_controller(make_controller, tmp_project)

    controller.request_frames("ep01.mkv", [10.0, 20.0])
    fake_runner.finish(fake_runner.last("frames", "ep01.mkv"), None,
                       event_type="failed", message="decode error", error="Traceback")
    controller.drain_events()

    controller.request_frames("ep01.mkv", [10.0, 20.0])
    assert len(fake_runner.of_kind("frames", "ep01.mkv")) == 1
    assert controller.frame("ep01.mkv", 10.0) is None
    assert "decode error" in controller.log_text("Pipeline")


def test_a_time_that_could_not_be_grabbed_is_not_asked_for_again(make_controller, fake_runner, tmp_project):
    """A finished job marks only the times missing from its result."""
    controller, _ = _frames_controller(make_controller, tmp_project)
    ready = Spy(controller.frame_ready)

    controller.request_frames("ep01.mkv", [10.0, 20.0])
    fake_runner.finish(fake_runner.last("frames", "ep01.mkv"),
                       FramesResult("ep01.mkv", {10.0: _frame(1)}))       # 20.0 could not be read
    controller.drain_events()

    controller.request_frames("ep01.mkv", [10.0, 20.0])                   # a repaint asks again
    controller.request_frames("ep01.mkv", [20.0])
    assert len(fake_runner.of_kind("frames", "ep01.mkv")) == 1
    assert controller.frame("ep01.mkv", 20.0) is None                     # the view draws its placeholder
    assert ready.calls == [("ep01.mkv", 10.0)]

    controller.request_frames("ep01.mkv", [30.0])                         # a time never tried still goes
    assert [s.job.times for s in fake_runner.of_kind("frames", "ep01.mkv")] == [(10.0, 20.0), (30.0,)]


def test_a_strip_that_could_not_be_grabbed_is_tried_again_for_a_new_crop_box(
        make_controller, fake_runner, tmp_project):
    controller, _ = _frames_controller(make_controller, tmp_project)

    controller.request_strips("ep01.mkv", BOX, [10.0])
    fake_runner.finish(fake_runner.last("strips", "ep01.mkv"), StripsResult("ep01.mkv", BOX, {}))
    controller.drain_events()

    controller.request_strips("ep01.mkv", BOX, [10.0])                    # same box: already tried
    assert len(fake_runner.of_kind("strips", "ep01.mkv")) == 1

    controller.request_strips("ep01.mkv", OTHER_BOX, [10.0])              # another box: a new question
    assert [s.job.crop_box for s in fake_runner.of_kind("strips", "ep01.mkv")] == [BOX, OTHER_BOX]


def test_a_file_that_comes_back_is_tried_again(make_controller, fake_runner, tmp_project):
    """clear_file drops the markers with the frames: a re-added file is not
    stuck with the verdicts of the one that vanished."""
    controller, folder = _frames_controller(make_controller, tmp_project)
    controller.request_frames("ep01.mkv", [10.0])
    fake_runner.finish(fake_runner.last("frames", "ep01.mkv"),
                       FramesResult("ep01.mkv", {}), event_type="failed", message="decode error")
    controller.drain_events()
    assert len(fake_runner.of_kind("frames", "ep01.mkv")) == 1

    (folder / "ep01.mkv").unlink()
    assert wait_for(lambda: "ep01.mkv" not in controller.names(), WAIT_MS)
    (folder / "ep01.mkv").write_bytes(b"placeholder video")
    assert wait_for(lambda: "ep01.mkv" in controller.names(), WAIT_MS)

    controller.request_frames("ep01.mkv", [10.0])
    assert len(fake_runner.of_kind("frames", "ep01.mkv")) == 2


def test_a_cancelled_frame_job_lets_a_later_request_try_again(make_controller, fake_runner, tmp_project):
    """Cancellation is not "unavailable": nothing was learned about the time."""
    controller, _ = _frames_controller(make_controller, tmp_project)

    controller.request_frames("ep01.mkv", [10.0])
    fake_runner.finish(fake_runner.last("frames", "ep01.mkv"), None, event_type="cancelled")
    controller.drain_events()

    controller.request_frames("ep01.mkv", [10.0])
    assert len(fake_runner.of_kind("frames", "ep01.mkv")) == 2


# --------------------------------------------------------------------------
# Warming the view cache
# --------------------------------------------------------------------------

def _warm(controller, fake_runner, folder, name="ep01.mkv", **finish):
    """Submit and end a warm job for `name`, the way AutoPilot's chain does."""
    fake_runner.submit(WarmJob(str(folder), name, wanted(controller.entry(name))))
    fake_runner.finish(fake_runner.last("warm", name),
                       finish.pop("result", WarmResult(name, 3, 2, 0)), **finish)
    controller.drain_events()


def test_warming_never_shows_up_as_activity(make_controller, fake_runner, tmp_project):
    """It must not push the detectors out of the five-deep history, any more
    than a view's own fetches do."""
    controller, folder = _frames_controller(make_controller, tmp_project)
    before = controller.activity()

    _warm(controller, fake_runner, folder)

    after = controller.activity()
    assert after.current == before.current
    assert after.running == before.running
    assert [kind for kind, _file, _at in after.recent] == [kind for kind, _file, _at in before.recent]


def test_a_warm_event_is_handled_without_an_internal_error(make_controller, fake_runner, tmp_project):
    """drain_events swallows and logs a handler that raises, so a routing
    mistake here would be invisible: pin that nothing was logged."""
    controller, folder = _frames_controller(make_controller, tmp_project)

    _warm(controller, fake_runner, folder)

    assert "Internal error" not in controller.log_text("Pipeline")
    assert "Unexpected" not in controller.log_text("Pipeline")


def test_a_failed_warm_job_is_logged_and_changes_no_state(make_controller, fake_runner, tmp_project):
    controller, folder = _frames_controller(make_controller, tmp_project)
    before = controller.entry("ep01.mkv").review

    _warm(controller, fake_runner, folder, result=None, event_type="failed", message="disk full")

    assert "Could not warm the view cache for ep01.mkv" in controller.log_text("Pipeline")
    assert controller.entry("ep01.mkv").review is before


def test_warming_reports_no_pending_detector(make_controller, fake_runner, tmp_project):
    """Warming is not detection: it must not make a file look busy."""
    controller, folder = _frames_controller(make_controller, tmp_project)
    fake_runner.submit(WarmJob(str(folder), "ep01.mkv", wanted(controller.entry("ep01.mkv"))))
    fake_runner.start(fake_runner.last("warm", "ep01.mkv"))
    controller.drain_events()

    assert "warm" not in controller.pending_detectors().get("ep01.mkv", set())
    assert "warm" not in controller.running_detectors("ep01.mkv")
    assert not controller.is_detection_kind("warm")


# --------------------------------------------------------------------------
# Exact frames: the crop view's "masked" preview vs. the lossy disk cache
# --------------------------------------------------------------------------

def _finish_frames(fake_runner, controller, times, lossy=(), file="ep01.mkv"):
    """Answer the outstanding frames job for `file` with a frame per time,
    `lossy` naming the ones the job read back from the on-disk view cache."""
    frames = {time: _frame(int(time)) for time in times}
    fake_runner.finish(fake_runner.last("frames", file),
                       FramesResult(file, frames, lossy=frozenset(lossy)))
    controller.drain_events()


def test_an_exact_request_carries_exact_to_the_job(make_controller, fake_runner, tmp_project):
    controller, _ = _frames_controller(make_controller, tmp_project)

    controller.request_frames("ep01.mkv", [10.0], exact=True)

    job = fake_runner.last("frames", "ep01.mkv").job
    assert job.exact is True and job.times == (10.0,)


def test_a_lossy_frame_is_refetched_for_an_exact_request(make_controller, fake_runner, tmp_project):
    """The masked preview filters on pixel levels, and the disk cache's WebP
    moved them: a cached lossy frame is a miss for `exact`."""
    controller, _ = _frames_controller(make_controller, tmp_project)
    controller.request_frames("ep01.mkv", [10.0])
    _finish_frames(fake_runner, controller, [10.0], lossy=[10.0])

    assert controller.frame("ep01.mkv", 10.0) is not None          # good enough to look at
    assert controller.frame("ep01.mkv", 10.0, exact=True) is None  # not to filter

    controller.request_frames("ep01.mkv", [10.0], exact=True)
    jobs = [s.job for s in fake_runner.of_kind("frames", "ep01.mkv")]
    assert len(jobs) == 2 and jobs[1].exact is True


def test_an_exact_frame_satisfies_both_kinds_of_request(make_controller, fake_runner, tmp_project):
    controller, _ = _frames_controller(make_controller, tmp_project)
    controller.request_frames("ep01.mkv", [10.0])
    _finish_frames(fake_runner, controller, [10.0])                # decoded, so exact

    assert controller.frame("ep01.mkv", 10.0, exact=True) is not None

    controller.request_frames("ep01.mkv", [10.0])
    controller.request_frames("ep01.mkv", [10.0], exact=True)
    assert len(fake_runner.of_kind("frames", "ep01.mkv")) == 1     # nothing to fetch


def test_an_exact_fetch_under_way_answers_a_plain_request(make_controller, fake_runner, tmp_project):
    controller, _ = _frames_controller(make_controller, tmp_project)

    controller.request_frames("ep01.mkv", [10.0], exact=True)
    controller.request_frames("ep01.mkv", [10.0])                  # exact is strictly better: wait for it

    assert len(fake_runner.of_kind("frames", "ep01.mkv")) == 1


def test_a_plain_fetch_under_way_does_not_answer_an_exact_request(
        make_controller, fake_runner, tmp_project):
    """The other way round does not hold: the plain fetch may answer off the
    disk, and the masked preview cannot use that."""
    controller, _ = _frames_controller(make_controller, tmp_project)

    controller.request_frames("ep01.mkv", [10.0])
    controller.request_frames("ep01.mkv", [10.0], exact=True)

    jobs = [s.job for s in fake_runner.of_kind("frames", "ep01.mkv")]
    assert len(jobs) == 2 and [job.exact for job in jobs] == [False, True]
    assert jobs[0].key != jobs[1].key             # or the runner would replace one with the other


def test_an_exact_frame_upgrades_the_cached_lossy_one(make_controller, fake_runner, tmp_project):
    controller, _ = _frames_controller(make_controller, tmp_project)
    controller.request_frames("ep01.mkv", [10.0])
    _finish_frames(fake_runner, controller, [10.0], lossy=[10.0])

    controller.request_frames("ep01.mkv", [10.0], exact=True)
    _finish_frames(fake_runner, controller, [10.0])

    assert controller.frame("ep01.mkv", 10.0, exact=True) is not None
    controller.request_frames("ep01.mkv", [10.0], exact=True)
    assert len(fake_runner.of_kind("frames", "ep01.mkv")) == 2     # nothing left to fetch


def test_a_time_known_unreadable_is_not_asked_for_exactly_either(
        make_controller, fake_runner, tmp_project):
    """A decoder that could not read the time would not read it exactly
    either: the once-per-session rule holds for both kinds of request."""
    controller, _ = _frames_controller(make_controller, tmp_project)
    controller.request_frames("ep01.mkv", [10.0])
    _finish_frames(fake_runner, controller, [])                    # finished without the frame

    controller.request_frames("ep01.mkv", [10.0], exact=True)
    assert len(fake_runner.of_kind("frames", "ep01.mkv")) == 1


def test_a_mixed_result_marks_only_the_disk_sourced_frames_lossy(
        make_controller, fake_runner, tmp_project):
    """One job may answer some times off the disk and decode the rest; only
    the former are lossy."""
    controller, _ = _frames_controller(make_controller, tmp_project)
    controller.request_frames("ep01.mkv", [10.0, 20.0])
    _finish_frames(fake_runner, controller, [10.0, 20.0], lossy=[10.0])

    assert controller.frame("ep01.mkv", 10.0, exact=True) is None
    assert controller.frame("ep01.mkv", 20.0, exact=True) is not None


def test_a_cancelled_strip_job_lets_a_later_request_try_again(make_controller, fake_runner, tmp_project):
    controller, _ = _frames_controller(make_controller, tmp_project)

    controller.request_strips("ep01.mkv", BOX, [10.0])
    fake_runner.finish(fake_runner.last("strips", "ep01.mkv"), None, event_type="cancelled")
    controller.drain_events()

    controller.request_strips("ep01.mkv", BOX, [10.0])
    assert len(fake_runner.of_kind("strips", "ep01.mkv")) == 2


def test_request_strips_keys_the_cache_by_crop_box(make_controller, fake_runner, tmp_project):
    controller, _ = _frames_controller(make_controller, tmp_project)
    ready = Spy(controller.strips_ready)
    strip = _frame(7)

    controller.request_strips("ep01.mkv", BOX, [10.0])
    controller.request_strips("ep01.mkv", BOX, [10.0])            # in flight: no second job
    controller.request_strips("ep01.mkv", OTHER_BOX, [10.0])      # another box is another request

    jobs = [s.job for s in fake_runner.of_kind("strips", "ep01.mkv")]
    assert [(job.crop_box, job.times) for job in jobs] == [(BOX, (10.0,)), (OTHER_BOX, (10.0,))]

    fake_runner.finish(fake_runner.of_kind("strips", "ep01.mkv")[0],
                       StripsResult("ep01.mkv", BOX, {10.0: strip}))
    controller.drain_events()

    assert controller.strip("ep01.mkv", BOX, 10.0) is strip
    assert controller.strip("ep01.mkv", OTHER_BOX, 10.0) is None
    assert ready.calls == [("ep01.mkv",)]


def test_changing_the_crop_drops_the_files_strips(make_controller, fake_runner, tmp_project):
    controller, _ = _frames_controller(make_controller, tmp_project)
    controller.request_frames("ep01.mkv", [10.0])
    controller.request_strips("ep01.mkv", BOX, [10.0])
    controller.request_strips("ep02.mkv", BOX, [10.0])
    fake_runner.finish(fake_runner.last("frames", "ep01.mkv"), FramesResult("ep01.mkv", {10.0: _frame(1)}))
    fake_runner.finish(fake_runner.last("strips", "ep01.mkv"), StripsResult("ep01.mkv", BOX, {10.0: _frame(2)}))
    fake_runner.finish(fake_runner.last("strips", "ep02.mkv"), StripsResult("ep02.mkv", BOX, {10.0: _frame(3)}))
    controller.drain_events()

    controller.set_crop("ep01.mkv", OTHER_BOX)

    assert controller.strip("ep01.mkv", BOX, 10.0) is None        # measured on a box that is gone
    assert controller.strip("ep02.mkv", BOX, 10.0) is not None
    assert controller.frame("ep01.mkv", 10.0) is not None         # frames do not depend on the crop


def test_a_removed_file_loses_its_frames(make_controller, fake_runner, tmp_project):
    controller, folder = _frames_controller(make_controller, tmp_project)
    controller.request_frames("ep01.mkv", [10.0])
    controller.request_strips("ep01.mkv", BOX, [10.0])
    fake_runner.finish(fake_runner.last("frames", "ep01.mkv"), FramesResult("ep01.mkv", {10.0: _frame(1)}))
    fake_runner.finish(fake_runner.last("strips", "ep01.mkv"), StripsResult("ep01.mkv", BOX, {10.0: _frame(2)}))
    controller.drain_events()

    (folder / "ep01.mkv").unlink()
    assert wait_for(lambda: "ep01.mkv" not in controller.names(), WAIT_MS)

    assert controller.frame("ep01.mkv", 10.0) is None
    assert controller.strip("ep01.mkv", BOX, 10.0) is None


def test_a_frame_of_a_removed_file_is_never_cached(make_controller, fake_runner, tmp_project):
    controller, folder = _frames_controller(make_controller, tmp_project)
    controller.request_frames("ep01.mkv", [10.0])
    submission = fake_runner.last("frames", "ep01.mkv")

    (folder / "ep01.mkv").unlink()
    assert wait_for(lambda: "ep01.mkv" not in controller.names(), WAIT_MS)
    fake_runner.finish(submission, FramesResult("ep01.mkv", {10.0: _frame(1)}))
    controller.drain_events()

    assert controller.frame("ep01.mkv", 10.0) is None


def test_closing_the_folder_forgets_every_frame(make_controller, fake_runner, tmp_project):
    controller, _ = _frames_controller(make_controller, tmp_project)
    controller.request_frames("ep01.mkv", [10.0])
    fake_runner.finish(fake_runner.last("frames", "ep01.mkv"), FramesResult("ep01.mkv", {10.0: _frame(1)}))
    controller.drain_events()

    controller.close_folder()

    assert controller.frame("ep01.mkv", 10.0) is None


def test_frames_and_strips_are_not_activity(make_controller, fake_runner, tmp_project):
    """A view repainting must not push the detectors out of the activity
    strip (or its five-deep "recent" list)."""
    controller, _ = _frames_controller(make_controller, tmp_project)
    controller.drain_events()
    before = controller.activity()
    changed = Spy(controller.activity_changed)

    controller.request_frames("ep01.mkv", [10.0])
    submission = fake_runner.last("frames", "ep01.mkv")
    fake_runner.start(submission)
    controller.drain_events()
    assert controller.activity() == before

    fake_runner.finish(submission, FramesResult("ep01.mkv", {10.0: _frame(1)}))
    controller.drain_events()
    assert controller.activity() == before
    assert changed.calls == []


def test_a_frame_the_cache_evicted_is_not_taken_for_unreadable(make_controller, fake_runner, tmp_project):
    """"Unavailable" means the grab failed, never "the LRU dropped it": the
    times that could not be read come from the result, not from what is still
    in the cache when the last frame of the batch has been stored."""
    controller, _ = _frames_controller(make_controller, tmp_project)
    controller._frames.max_bytes = 100                        # two of these frames fit, not three

    controller.request_frames("ep01.mkv", [10.0, 20.0, 30.0])
    fake_runner.finish(fake_runner.last("frames", "ep01.mkv"),
                       FramesResult("ep01.mkv", {time: _frame(int(time), size=4) for time in (10.0, 20.0, 30.0)}))
    controller.drain_events()
    assert controller.frame("ep01.mkv", 10.0) is None         # evicted, not unreadable
    assert controller.frame("ep01.mkv", 30.0) is not None

    controller.request_frames("ep01.mkv", [10.0])
    assert [s.job.times for s in fake_runner.of_kind("frames", "ep01.mkv")] == [(10.0, 20.0, 30.0), (10.0,)]


def test_requesting_frames_without_an_open_folder_does_nothing(make_controller, fake_runner, tmp_project):
    """Views ask from paintEvent: a repaint between close_folder() and the
    view hearing about it must not raise."""
    controller = make_controller()
    controller.request_frames("ep01.mkv", [10.0])             # no folder has ever been open
    controller.request_strips("ep01.mkv", BOX, [10.0])

    controller.open_folder(str(tmp_project(["ep01.mkv"], config=manual_config(["ep01.mkv"]))))
    controller.close_folder()
    controller.request_frames("ep01.mkv", [10.0])
    controller.request_strips("ep01.mkv", BOX, [10.0])

    controller.shutdown(timeout=0.5)
    controller.request_frames("ep01.mkv", [10.0])             # and on the way out

    assert fake_runner.of_kind("frames") == [] and fake_runner.of_kind("strips") == []
    assert controller.frame("ep01.mkv", 10.0) is None
    assert controller.strip("ep01.mkv", BOX, 10.0) is None


def test_requesting_frames_for_an_unknown_file_is_refused(make_controller, tmp_project):
    controller, _ = _frames_controller(make_controller, tmp_project)
    with pytest.raises(KeyError):
        controller.request_frames("nope.mkv", [10.0])
    with pytest.raises(KeyError):
        controller.request_strips("nope.mkv", BOX, [10.0])


# --------------------------------------------------------------------------
# A stored crop is one the file's frame can hold
# --------------------------------------------------------------------------

SMALL_BOX = (0, 665, 1280, 55)            # BOX's band as a 1280x720 frame holds it


def _unscanned(names):
    """Entries with no media at all: the metadata job has not run."""
    entries = [make_entry(name, brightness=205) for name in names]
    for entry in entries:
        entry.media = Media()
    return entries


def test_set_crop_clamps_to_the_files_frame(make_controller, fake_runner, tmp_project):
    from core.jobs.apply import FLAG_CROP_CLAMPED

    names = ["ep01.mkv"]
    entries = [make_entry(name, brightness=205) for name in names]
    entries[0].media = Media(1280, 720, DURATION, 23.976)
    folder = tmp_project(names, config=v2_config(entries, autopilot_enabled=False))
    controller = make_controller()
    controller.open_folder(str(folder))

    controller.set_crop("ep01.mkv", (288, 784, 1344, 55))
    entry = controller.entry("ep01.mkv")
    assert (entry.crop.x, entry.crop.y, entry.crop.width, entry.crop.height) == SMALL_BOX
    assert FLAG_CROP_CLAMPED in entry.flags["crop"]
    assert badge_for(entry, running_detectors=set(), done=False, run_state=None)[0] == "check crop"


def test_pasting_across_resolutions_fits_the_target_and_doubts_the_brightness(
        make_controller, fake_runner, tmp_project):
    from core.jobs.apply import FLAG_BRIGHTNESS_OTHER_CROP, FLAG_CROP_CLAMPED

    names = ["ep01.mkv", "ep02.mkv"]
    entries = [make_entry("ep01.mkv", crop=(288, 784, 1344, 55), brightness=190),
               make_entry("ep02.mkv")]
    entries[1].media = Media(1280, 720, DURATION, 23.976)          # the 720p target
    folder = tmp_project(names, config=v2_config(entries, autopilot_enabled=False))
    controller = make_controller()
    controller.open_folder(str(folder))

    controller.copy_settings("ep01.mkv")
    assert controller.paste_settings("ep02.mkv") is True
    target = controller.entry("ep02.mkv")
    assert (target.crop.x, target.crop.y, target.crop.width, target.crop.height) == SMALL_BOX
    assert FLAG_CROP_CLAMPED in target.flags["crop"]
    assert target.brightness == Brightness(190, Source.MANUAL)
    assert FLAG_BRIGHTNESS_OTHER_CROP in target.flags["brightness"]
    assert target.review == ReviewState.FLAGGED
    assert badge_for(target, running_detectors=set(), done=False,
                     run_state=None)[0] == "check crop + brightness"


def test_a_crop_edited_before_the_metadata_lands_is_cut_when_it_arrives(
        make_controller, fake_runner, tmp_project):
    """The crop canvas has to assume a frame size until the metadata job
    reports one; whatever it assumed, the stored box is re-checked here."""
    from core.jobs.apply import FLAG_CROP_CLAMPED

    names = ["ep01.mkv"]
    folder = tmp_project(names, config=v2_config(_unscanned(names), autopilot_enabled=False))
    controller = make_controller()
    controller.open_folder(str(folder))

    controller.set_crop("ep01.mkv", (288, 784, 1344, 55))          # drawn against a guessed 1920x1080
    entry = controller.entry("ep01.mkv")
    assert (entry.crop.x, entry.crop.y, entry.crop.width, entry.crop.height) == (288, 784, 1344, 55)
    assert not (entry.flags.get("crop") or "")                     # nothing to check it against yet
    controller.mark_reviewed("ep01.mkv")
    assert entry.review == ReviewState.REVIEWED

    changed = Spy(controller.file_changed)
    metadata = fake_runner.last("metadata", "ep01.mkv")
    fake_runner.finish(metadata, MetadataResult("ep01.mkv", 1280, 720, DURATION, 23.976))
    controller.drain_events()

    assert (entry.crop.x, entry.crop.y, entry.crop.width, entry.crop.height) == SMALL_BOX
    assert FLAG_CROP_CLAMPED in entry.flags["crop"]
    assert entry.review == ReviewState.FLAGGED
    assert "ep01.mkv" in changed.firsts


def test_metadata_that_fits_the_stored_crop_changes_nothing(make_controller, fake_runner, tmp_project):
    names = ["ep01.mkv"]
    entries = _unscanned(names)
    entries[0].crop = Crop(*BOX, Source.MANUAL)
    entries[0].brightness = Brightness(205, Source.MANUAL)
    entries[0].review = ReviewState.REVIEWED
    folder = tmp_project(names, config=v2_config(entries, autopilot_enabled=False))
    controller = make_controller()
    controller.open_folder(str(folder))

    metadata = fake_runner.last("metadata", "ep01.mkv")
    fake_runner.finish(metadata, MetadataResult("ep01.mkv", 1920, 1080, DURATION, 23.976))
    controller.drain_events()

    entry = controller.entry("ep01.mkv")
    assert (entry.crop.x, entry.crop.y, entry.crop.width, entry.crop.height) == BOX
    assert entry.review == ReviewState.REVIEWED


# --------------------------------------------------------------------------
# View jobs follow the selection
# --------------------------------------------------------------------------

class Cancels:
    """Runs the cancel_where predicates the controller records against the
    fake runner's queue, as the real runner does: a matching queued job ends
    "cancelled". Only jobs queued when a predicate is recorded can match it,
    so the predicates are applied in the order they arrive."""

    def __init__(self, controller, fake_runner):
        self._controller = controller
        self._runner = fake_runner
        self._seen = len(fake_runner.cancel_predicates)

    def settle(self) -> None:
        for predicate in self._runner.cancel_predicates[self._seen:]:
            for submission in list(self._runner.queued()):
                if predicate(submission.job):
                    self._runner.emit(submission, "cancelled", result=None)
        self._seen = len(self._runner.cancel_predicates)
        self._controller.drain_events()

    def view_jobs(self) -> list:
        self.settle()
        return [s.job for s in self._runner.queued() if s.job.kind in VIEW_KINDS]


def test_walking_the_queue_leaves_only_the_current_files_view_jobs(
        make_controller, fake_runner, tmp_project):
    """VIEW_PRIORITY outranks every auto-pilot job on a two-worker lane, so
    frames for files the user has walked past would decode ahead of the
    metadata and thumbnails of the file they are looking at."""
    names = [f"ep{index:02d}.mkv" for index in range(1, 6)]
    controller, _ = _frames_controller(make_controller, tmp_project, names)
    cancels = Cancels(controller, fake_runner)

    for name in names:
        controller.set_view_file(name)
        cancels.settle()
        controller.request_frames(name, [10.0, 20.0])
        controller.request_strips(name, BOX, [10.0])

    live = cancels.view_jobs()
    assert {job.file for job in live} == {"ep05.mkv"}
    assert sorted(job.kind for job in live) == ["frames", "strips"]


def test_leaving_every_file_cancels_all_view_work(make_controller, fake_runner, tmp_project):
    controller, _ = _frames_controller(make_controller, tmp_project)
    cancels = Cancels(controller, fake_runner)
    controller.set_view_file("ep01.mkv")
    controller.request_frames("ep01.mkv", [10.0])
    controller.request_strips("ep01.mkv", BOX, [10.0])

    controller.set_view_file(None)
    assert cancels.view_jobs() == []


def test_set_view_file_never_cancels_anything_but_view_jobs(make_controller, fake_runner, tmp_project):
    controller, _ = _frames_controller(make_controller, tmp_project)
    controller.request_frames("ep01.mkv", [10.0])
    controller.set_view_file("ep02.mkv")

    other = [job for job in (s.job for s in fake_runner.submissions) if job.kind not in VIEW_KINDS]
    assert other, "the folder submits thumbnails at least"
    for predicate in fake_runner.cancel_predicates:
        assert not any(predicate(job) for job in other)


def test_a_cancelled_view_job_releases_its_times_for_the_next_visit(
        make_controller, fake_runner, tmp_project):
    """_on_view_event marks nothing for a cancelled job: coming back to the
    file must fetch those times again rather than draw the placeholder."""
    controller, _ = _frames_controller(make_controller, tmp_project)
    controller.set_view_file("ep01.mkv")
    controller.request_frames("ep01.mkv", [10.0])
    submission = fake_runner.last("frames", "ep01.mkv")

    controller.set_view_file("ep02.mkv")
    fake_runner.emit(submission, "cancelled", result=None)
    controller.drain_events()

    controller.set_view_file("ep01.mkv")
    controller.request_frames("ep01.mkv", [10.0])
    assert len(fake_runner.of_kind("frames", "ep01.mkv")) == 2


def test_set_view_file_without_a_folder_does_nothing(make_controller):
    controller = make_controller()
    controller.set_view_file("ep01.mkv")
    controller.set_view_file(None)
