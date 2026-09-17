"""Tests for core.jobs.autopilot: which detection jobs AutoPilot submits, and when.

No detection runs here. A fake runner records submissions; each test plays
the model owner (plan 3B's controller): it feeds synthetic terminal events
with constructed results, applies a result with core.jobs.apply only when
AutoPilot says the event is current, then calls on_job_event and
recompute_all, in that order.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.detect import crop as crop_mod
from core.detect.audio_profile import AudioProfile
from core.detect.brightness import FLAG_ESCALATE, BrightnessResult
from core.detect.crop import CONSENSUS_MIN_ENTRIES, CropResult
from core.detect.ranges.pipeline import RangesAnalysis
from core.jobs.apply import (
    apply_audio_profile,
    apply_brightness,
    apply_crop,
    apply_folder_change,
    apply_metadata,
    apply_ranges,
    recompute_all,
    set_manual_brightness,
    set_manual_crop,
)
from core.jobs.autopilot import (
    DETECTION_KINDS,
    AutoPilot,
    crop_consensus,
    intersect_plateaus,
    is_autopilot_job,
    is_detection_job,
)
from core.jobs.detect_jobs import (
    AudioProfileJob,
    AudioProfileResult,
    BrightnessJob,
    BrightnessJobResult,
    CropJob,
    CropJobResult,
    MetadataJob,
    MetadataResult,
    RangesJob,
    RangesJobResult,
    ThumbnailJob,
    ThumbnailResult,
)
from core.jobs.runner import JobEvent, Lane
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
    migrate_v1,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ocr_json_v1"
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
HEIGHT = 1080


# --------------------------------------------------------------------------
# The fake runner and the model owner
# --------------------------------------------------------------------------

@dataclass
class Submission:
    job: object
    job_id: int


class FakeRunner:
    """Records submit/pause/resume; job ids are assigned like the runner's."""

    def __init__(self):
        self.submissions: list[Submission] = []
        self.pauses: list[tuple[Lane, object]] = []
        self.resumes: list[Lane] = []
        self._ids = itertools.count(1)

    def submit(self, job):
        self.submissions.append(Submission(job, next(self._ids)))

    def pause(self, lane, *, only=None):
        self.pauses.append((lane, only))

    def resume(self, lane):
        self.resumes.append(lane)


APPLY = {
    "metadata": apply_metadata,
    "crop": apply_crop,
    "brightness": apply_brightness,
    "ranges": apply_ranges,
    "audio_profile": apply_audio_profile,
}


class Owner:
    """The model owner, driving AutoPilot as the class docstring prescribes."""

    def __init__(self, project: Project):
        self.project = project
        self.runner = FakeRunner()
        self.autopilot = AutoPilot(self.runner, lambda: self.project)
        self._seen = 0

    def take(self) -> list[Submission]:
        """Submissions made since the last take()."""
        new = self.runner.submissions[self._seen:]
        self._seen = len(self.runner.submissions)
        return new

    def event(self, submission: Submission, result=None, type="finished") -> JobEvent:
        job = submission.job
        return JobEvent(type, job.key, job.kind, job.file, job_id=submission.job_id, result=result)

    def deliver(self, submission: Submission, result=None, type="finished") -> bool:
        """Drain one terminal event; True when its result was applied."""
        event = self.event(submission, result, type)
        current = self.autopilot.is_current(event)
        if current and job_kind(submission) in APPLY:
            APPLY[job_kind(submission)](self.project, result)
        self.autopilot.on_job_event(event)
        self.recompute()
        return current

    def recompute(self):
        recompute_all(self.project, pending=self.autopilot.pending(),
                      ranges_pending=self.autopilot.ranges_pending())

    def state(self, name: str) -> ReviewState:
        return self.project.files[name].review


def job_kind(submission: Submission) -> str:
    return submission.job.kind


def of_kind(submissions: list[Submission], kind: str) -> list[Submission]:
    return [s for s in submissions if s.job.kind == kind]


def pairs(submissions: list[Submission]) -> list[tuple[str, str | None]]:
    return [(s.job.kind, s.job.file) for s in submissions]


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def _project(tmp_path, names, folder: FolderSettings | None = None) -> Project:
    return Project(path=str(tmp_path), folder=folder or FolderSettings(),
                   files={name: FileEntry(name=name) for name in names})


def _media(entry: FileEntry, duration: float = DURATION, height: int = HEIGHT) -> FileEntry:
    entry.media = Media(width=1920, height=height, duration=duration, fps=23.976)
    return entry


def _brightness_measured_on(entry: FileEntry, value: int, source: Source, box) -> None:
    """A brightness value as apply_brightness leaves it: with the crop it was measured on."""
    entry.brightness = Brightness(value, source)
    entry.evidence["brightness"] = {"crop_box": list(box), "value_crop_box": list(box)}


def metadata_done(sub: Submission, duration: float = DURATION, height: int = HEIGHT) -> MetadataResult:
    return MetadataResult(sub.job.file, 1920, height, duration, 23.976)


def crop_done(sub: Submission, box=BOX, hit: float = 300.0, flagged=None, frame=(1920, HEIGHT)) -> CropJobResult:
    result = CropResult(box=box, sample_pts=[hit], envelope=box, agreed=1 if box else 0, probes_used=1,
                        flagged=flagged, hit_pts=[hit] if box else [], frame_size=frame)
    return CropJobResult(sub.job.file, result, sub.job.hint)


def brightness_done(sub: Submission, value: int = 200, plateau=(180, 230), flagged=None) -> BrightnessJobResult:
    job = sub.job
    result = BrightnessResult(value=value, plateau=plateau, seed=value + 20, gate_floor=None,
                              flagged=flagged, curve=[])
    return BrightnessJobResult(job.file, result, {}, job.hint_value, job.crop_box)


def ranges_done(sub: Submission, keep=None) -> RangesJobResult:
    return RangesJobResult(RangesAnalysis(keep=keep or {}, blocks={},
                                          durations={name: DURATION for name in sub.job.files}))


def audio_done(sub: Submission) -> AudioProfileResult:
    return AudioProfileResult(sub.job.file, AudioProfile(duration=DURATION, envelope=[0.0, 1.0], speech=[]))


def thumbnail_done(sub: Submission) -> ThumbnailResult:
    return ThumbnailResult(sub.job.file, sub.job.time, None)


def finish_side_jobs(owner: Owner, submissions: list[Submission]) -> None:
    """Deliver thumbnails and audio profiles (they drive nothing)."""
    for sub in submissions:
        if sub.job.kind == "thumbnail":
            owner.deliver(sub, thumbnail_done(sub))
        elif sub.job.kind == "audio_profile":
            owner.deliver(sub, audio_done(sub))


# --------------------------------------------------------------------------
# Opening a folder
# --------------------------------------------------------------------------

def test_open_on_a_fresh_folder_submits_every_detection_in_dependency_order(tmp_path):
    names = [f"ep{i:02d}.mkv" for i in range(1, 6)]
    owner = Owner(_project(tmp_path, names))
    owner.autopilot.on_open()
    owner.recompute()

    opened = owner.take()
    assert pairs(opened) == [("ranges", None)] + [("metadata", name) for name in names]
    assert [s.job.priority for s in opened] == [0] + [5] * 5
    ranges = opened[0]
    assert list(ranges.job.files) == names
    assert {owner.state(name) for name in names} == {ReviewState.PENDING}

    for sub in opened[1:]:
        owner.deliver(sub, metadata_done(sub))
    after_metadata = owner.take()
    assert sorted(pairs(after_metadata)) == sorted(
        [("crop", names[0])] + [(kind, name) for name in names for kind in ("thumbnail", "audio_profile")])
    for sub in of_kind(after_metadata, "thumbnail"):
        assert sub.job.priority == 1 and sub.job.time == pytest.approx(0.4 * DURATION)
    for sub in of_kind(after_metadata, "audio_profile"):
        assert sub.job.priority == 1 and sub.job.duration == DURATION
    finish_side_jobs(owner, after_metadata)
    assert owner.take() == []

    # Crops run one at a time, in name order, each seeded with the unflagged results before it.
    boxes = {name: (288, 780 + index, 1344, 50 + index) for index, name in enumerate(names)}
    hits = {name: 100.0 + index for index, name in enumerate(names)}
    (crop,) = of_kind(after_metadata, "crop")
    full, rethumbs = [], []
    for index, name in enumerate(names):
        assert crop.job.file == name and crop.job.priority == 3 and crop.job.hint is None
        assert crop.job.duration == DURATION
        assert list(crop.job.consensus) == [(boxes[n][1] / HEIGHT, boxes[n][3] / HEIGHT) for n in names[:index]]
        assert all("crop" in owner.autopilot.pending()[n] for n in names[index:])
        owner.deliver(crop, crop_done(crop, box=boxes[name], hit=hits[name]))
        new = owner.take()
        crops = of_kind(new, "crop")
        assert len(crops) == (1 if index < len(names) - 1 else 0)
        full += of_kind(new, "brightness")
        rethumbs += of_kind(new, "thumbnail")
        assert len(new) == len(crops) + len(of_kind(new, "brightness")) + len(of_kind(new, "thumbnail"))
        crop = crops[0] if crops else None
    assert full == []                                          # brightness waits for the folder's ranges
    assert {(s.job.file, s.job.time) for s in rethumbs} == set(hits.items())
    assert all("crop" not in kinds for kinds in owner.autopilot.pending().values())
    # Files waiting for ranges or for the folder plateau are pending, not flagged.
    assert {owner.state(name) for name in names} == {ReviewState.PENDING}
    assert all("brightness" in owner.autopilot.pending()[name] for name in names)
    finish_side_jobs(owner, rethumbs)

    keep = {names[0]: [("02:00", "20:00")], names[3]: [("01:30", None)]}
    owner.deliver(ranges, ranges_done(ranges, keep=keep))
    full = owner.take()
    assert pairs(full) == [("brightness", name) for name in names[:3]]
    for sub in full:
        assert sub.job.folder_plateau is None and sub.job.hint_value is None
        assert sub.job.crop_box == boxes[sub.job.file] and sub.job.priority == 2
    assert [s.job.time_ranges for s in full] == [(("02:00", "20:00"),), None, None]   # the applied ranges

    plateaus = [(180, 230), (190, 240), (170, 225)]
    for sub, plateau in zip(full[:2], plateaus[:2]):
        owner.deliver(sub, brightness_done(sub, plateau=plateau))
        assert owner.take() == []
    owner.deliver(full[2], brightness_done(full[2], plateau=plateaus[2]))
    cheap = owner.take()
    assert pairs(cheap) == [("brightness", name) for name in names[3:]]
    for sub in cheap:
        assert sub.job.folder_plateau == (190, 225) and sub.job.crop_box == boxes[sub.job.file]
        assert sub.job.priority == 2
    assert [s.job.time_ranges for s in cheap] == [(("01:30", None),), None]

    for sub in cheap:
        owner.deliver(sub, brightness_done(sub, value=205, plateau=(190, 225)))
    assert owner.take() == []
    assert len(of_kind(owner.runner.submissions, "ranges")) == 1
    assert {owner.state(name) for name in names} == {ReviewState.PROPOSED}
    assert owner.autopilot.pending() == {}
    assert owner.autopilot.ranges_pending() is False


def test_an_imported_project_gets_only_thumbnails_and_audio_profiles(tmp_path):
    data = json.loads((FIXTURES / "slay.json").read_text(encoding="utf-8"))
    project = migrate_v1(data, str(tmp_path), SLAY_NAMES)
    owner = Owner(project)
    owner.autopilot.on_open()

    submitted = owner.take()
    assert sorted(pairs(submitted)) == sorted(
        [("thumbnail", name) for name in SLAY_NAMES] + [("audio_profile", name) for name in SLAY_NAMES])
    for sub in of_kind(submitted, "thumbnail"):
        assert sub.job.time == project.files[sub.job.file].sample_time
    finish_side_jobs(owner, submitted)
    assert owner.take() == []
    assert {owner.state(name) for name in SLAY_NAMES} == {ReviewState.REVIEWED}


def test_autopilot_disabled_runs_only_metadata_and_thumbnails_but_redetect_works(tmp_path):
    project = _project(tmp_path, ["a.mkv", "b.mkv", "c.mkv"], FolderSettings(autopilot_enabled=False))
    _media(project.files["a.mkv"]).crop = Crop(*BOX, Source.DETECTED)
    b = _media(project.files["b.mkv"])
    b.crop = Crop(*OTHER_BOX, Source.IMPORTED)
    b.brightness = Brightness(209, Source.IMPORTED)
    owner = Owner(project)

    owner.autopilot.on_open()
    opened = owner.take()
    assert sorted(pairs(opened)) == [("metadata", "c.mkv"), ("thumbnail", "a.mkv"), ("thumbnail", "b.mkv")]
    assert owner.autopilot.pending() == {"a.mkv": {"thumbnail"}, "b.mkv": {"thumbnail"}, "c.mkv": {"metadata"}}
    finish_side_jobs(owner, opened)
    (metadata,) = of_kind(opened, "metadata")
    owner.deliver(metadata, metadata_done(metadata))
    assert pairs(owner.take()) == [("thumbnail", "c.mkv")]              # no crop, audio or brightness
    assert "crop" not in owner.autopilot.pending().get("c.mkv", set())
    assert owner.state("c.mkv") == ReviewState.FLAGGED                 # nothing will detect its crop

    project.files["d.mkv"] = FileEntry(name="d.mkv")
    owner.autopilot.on_files_added(["d.mkv"])
    added = owner.take()
    assert pairs(added) == [("metadata", "d.mkv")]                      # no ranges either
    owner.autopilot.redetect("d.mkv")                                   # waits for the queued metadata
    assert owner.take() == []
    assert owner.autopilot.pending()["d.mkv"] >= {"metadata", "crop", "brightness"}
    owner.deliver(added[0], metadata_done(added[0]))
    after = owner.take()
    assert sorted(pairs(after)) == [("crop", "d.mkv"), ("thumbnail", "d.mkv")]
    (crop,) = of_kind(after, "crop")
    assert crop.job.priority == 13
    owner.deliver(crop, crop_done(crop))
    brightness = of_kind(owner.take(), "brightness")
    assert pairs(brightness) == [("brightness", "d.mkv")]
    assert brightness[0].job.priority == 12 and brightness[0].job.crop_box == BOX

    owner.autopilot.redetect("b.mkv")
    crop = owner.take()
    assert pairs(crop) == [("crop", "b.mkv")]
    assert crop[0].job.priority == 13
    assert crop[0].job.consensus == ((BOX[1] / HEIGHT, BOX[3] / HEIGHT),) * 2   # a and d: the others' crops
    owner.deliver(crop[0], crop_done(crop[0], box=BOX))
    assert project.files["b.mkv"].crop == Crop(*OTHER_BOX, Source.IMPORTED)   # the user's value stays
    brightness = of_kind(owner.take(), "brightness")
    assert pairs(brightness) == [("brightness", "b.mkv")]                     # sources ignored
    assert brightness[0].job.priority == 12
    assert brightness[0].job.folder_plateau is None and brightness[0].job.crop_box == OTHER_BOX


def test_a_labels_only_folder_gets_no_crop_or_brightness(tmp_path):
    names = ["a.mkv", "b.mkv"]
    owner = Owner(_project(tmp_path, names, FolderSettings(dialogue_enabled=False, labels_enabled=True)))
    owner.autopilot.on_open()
    opened = owner.take()
    assert pairs(opened) == [("ranges", None), ("metadata", "a.mkv"), ("metadata", "b.mkv")]
    assert owner.autopilot.pending() == {"a.mkv": {"metadata"}, "b.mkv": {"metadata"}}

    for sub in opened[1:]:
        owner.deliver(sub, metadata_done(sub))
    after = owner.take()
    assert sorted(pairs(after)) == sorted(
        [("thumbnail", name) for name in names] + [("audio_profile", name) for name in names])

    owner.autopilot.redetect("a.mkv")
    owner.autopilot.on_crop_changed("a.mkv")
    assert owner.take() == []
    finish_side_jobs(owner, after)
    owner.deliver(opened[0], ranges_done(opened[0]))
    assert {owner.state(name) for name in names} == {ReviewState.PROPOSED}


def _ready_folder(tmp_path, names, folder: FolderSettings | None = None, *, open_ranges: bool = False) -> Project:
    """Files with known media and a detected crop, and no brightness. Unless
    `open_ranges`, every file's time ranges are the user's whole file, so no
    ranges analysis runs and brightness does not wait for one."""
    project = _project(tmp_path, names, folder)
    for entry in project.files.values():
        _media(entry).crop = Crop(*BOX, Source.DETECTED)
        entry.sample_time = 300.0
        entry.evidence["audio"] = {}
        if not open_ranges:
            entry.time_ranges = TimeRanges([], Source.MANUAL)
    return project


def test_opening_twice_submits_nothing_new(tmp_path):
    owner = Owner(_ready_folder(tmp_path, ["a.mkv", "b.mkv"]))
    owner.autopilot.on_open()
    opened = owner.take()
    assert sorted(pairs(opened)) == sorted([("brightness", "a.mkv"), ("brightness", "b.mkv"),
                                            ("thumbnail", "a.mkv"), ("thumbnail", "b.mkv")])   # audio already known
    owner.autopilot.on_open()
    assert owner.take() == []


def test_open_measures_a_detected_brightness_whose_crop_has_changed(tmp_path):
    project = _ready_folder(tmp_path, ["a.mkv", "b.mkv"])
    _brightness_measured_on(project.files["a.mkv"], 209, Source.DETECTED, OTHER_BOX)   # stale
    _brightness_measured_on(project.files["b.mkv"], 209, Source.DETECTED, BOX)         # current
    owner = Owner(project)
    owner.autopilot.on_open()
    brightness = of_kind(owner.take(), "brightness")
    assert pairs(brightness) == [("brightness", "a.mkv")] and brightness[0].job.crop_box == BOX


def test_a_full_run_measured_on_a_replaced_crop_does_not_set_the_folder_plateau(tmp_path):
    project = _ready_folder(tmp_path, ["a.mkv", "b.mkv"], FolderSettings(brightness_full_detect_files=1))
    owner = Owner(project)
    owner.autopilot.on_open()
    (full,) = of_kind(owner.take(), "brightness")
    set_manual_brightness(project, "a.mkv", 215)
    set_manual_crop(project, "a.mkv", OTHER_BOX)
    owner.autopilot.on_crop_changed("a.mkv")               # the user's brightness: nothing to re-measure
    assert owner.take() == []
    assert owner.deliver(full, brightness_done(full, plateau=(10, 20))) is True
    (next_full,) = owner.take()
    assert next_full.job.file == "b.mkv" and next_full.job.folder_plateau is None


def test_an_escalated_cheap_result_is_resubmitted_as_full_detection(tmp_path):
    owner = Owner(_ready_folder(tmp_path, ["a.mkv", "b.mkv"], FolderSettings(brightness_full_detect_files=1)))
    owner.autopilot.on_open()
    full = of_kind(owner.take(), "brightness")
    assert pairs(full) == [("brightness", "a.mkv")] and full[0].job.folder_plateau is None

    owner.deliver(full[0], brightness_done(full[0], plateau=(190, 230)))
    cheap = owner.take()
    assert pairs(cheap) == [("brightness", "b.mkv")] and cheap[0].job.folder_plateau == (190, 230)

    owner.deliver(cheap[0], brightness_done(cheap[0], value=250, plateau=None, flagged=FLAG_ESCALATE))
    escalated = owner.take()
    assert pairs(escalated) == [("brightness", "b.mkv")]
    assert escalated[0].job.folder_plateau is None
    assert escalated[0].job.crop_box == BOX and escalated[0].job.priority == 2
    assert owner.state("b.mkv") == ReviewState.PENDING

    owner.deliver(escalated[0], brightness_done(escalated[0], value=212))
    assert owner.take() == []
    assert owner.project.files["b.mkv"].brightness == Brightness(212, Source.DETECTED)


@pytest.mark.parametrize("plateaus", [
    [(180, 230), None],            # one full run verified nothing
    [(150, 170), (180, 230)],      # no threshold both files read
])
def test_without_a_common_plateau_every_later_file_runs_full(tmp_path, plateaus):
    names = ["a.mkv", "b.mkv", "c.mkv", "d.mkv"]
    owner = Owner(_ready_folder(tmp_path, names, FolderSettings(brightness_full_detect_files=2)))
    owner.autopilot.on_open()
    full = of_kind(owner.take(), "brightness")
    assert [s.job.file for s in full] == ["a.mkv", "b.mkv"]

    for sub, plateau in zip(full, plateaus):
        owner.deliver(sub, brightness_done(sub, plateau=plateau, flagged=None if plateau else "no-plateau?"))
    later = owner.take()
    assert [s.job.file for s in later] == ["c.mkv", "d.mkv"]
    assert all(s.job.folder_plateau is None for s in later)


@pytest.mark.parametrize("plateaus, expected", [
    ([(180, 230), (190, 240), (170, 225)], (190, 225)),
    ([(180, 230)], (180, 230)),
    ([(180, 200), (200, 240)], (200, 200)),
    ([(180, 199), (200, 240)], None),
    ([(180, 230), None], None),
    ([], None),
])
def test_intersect_plateaus(plateaus, expected):
    assert intersect_plateaus(plateaus) == expected


def test_the_folder_plateau_comes_from_this_sessions_full_runs_not_imported_evidence(tmp_path):
    names = ["a.mkv", "b.mkv", "c.mkv", "d.mkv"]
    project = _ready_folder(tmp_path, names, FolderSettings(brightness_full_detect_files=2))
    for name in ("a.mkv", "b.mkv"):
        entry = project.files[name]
        entry.brightness = Brightness(150, Source.IMPORTED)
        entry.evidence["brightness"] = {"plateau": [10, 20], "value_crop_box": list(BOX)}
    owner = Owner(project)
    owner.autopilot.on_open()
    full = of_kind(owner.take(), "brightness")
    assert [s.job.file for s in full] == ["c.mkv", "d.mkv"]
    owner.deliver(full[0], brightness_done(full[0], plateau=(180, 230)))
    owner.deliver(full[1], brightness_done(full[1], plateau=(190, 240)))
    assert owner.take() == []                      # every file needing detection was in the full tier

    project.files["e.mkv"] = _media(FileEntry(name="e.mkv"))
    project.files["e.mkv"].crop = Crop(*BOX, Source.DETECTED)
    project.files["e.mkv"].time_ranges = TimeRanges([], Source.MANUAL)
    owner.autopilot.on_files_added(["e.mkv"])
    cheap = of_kind(owner.take(), "brightness")
    assert pairs(cheap) == [("brightness", "e.mkv")] and cheap[0].job.folder_plateau == (190, 230)


def test_a_full_tier_file_that_gets_no_crop_is_replaced_by_the_next_file(tmp_path):
    names = ["a.mkv", "b.mkv", "c.mkv"]
    project = _project(tmp_path, names, FolderSettings(brightness_full_detect_files=2))
    for entry in project.files.values():
        _media(entry).time_ranges = TimeRanges([], Source.MANUAL)
    owner = Owner(project)
    owner.autopilot.on_open()
    (crop_a,) = of_kind(owner.take(), "crop")

    owner.deliver(crop_a, crop_done(crop_a, box=None, flagged=crop_mod.FLAG_LOW_AGREEMENT))
    after_a = owner.take()
    assert of_kind(after_a, "brightness") == []
    assert owner.state("a.mkv") == ReviewState.FLAGGED
    (crop_b,) = of_kind(after_a, "crop")
    owner.deliver(crop_b, crop_done(crop_b))
    after_b = owner.take()
    (crop_c,) = of_kind(after_b, "crop")
    owner.deliver(crop_c, crop_done(crop_c))
    full = of_kind(after_b + owner.take(), "brightness")
    assert [s.job.file for s in full] == ["b.mkv", "c.mkv"]
    assert all(s.job.folder_plateau is None for s in full)


def test_a_full_tier_file_whose_detection_fails_is_replaced_by_the_next_file(tmp_path):
    owner = Owner(_ready_folder(tmp_path, ["a.mkv", "b.mkv", "c.mkv"], FolderSettings(brightness_full_detect_files=2)))
    owner.autopilot.on_open()
    full = of_kind(owner.take(), "brightness")
    assert [s.job.file for s in full] == ["a.mkv", "b.mkv"]
    owner.deliver(full[0], type="failed")
    (promoted,) = owner.take()
    assert promoted.job.file == "c.mkv" and promoted.job.folder_plateau is None
    owner.deliver(full[1], brightness_done(full[1], plateau=(180, 230)))
    assert owner.take() == []
    owner.deliver(promoted, brightness_done(promoted, plateau=(190, 240)))
    assert owner.take() == []
    assert "brightness" not in owner.autopilot.pending().get("a.mkv", set())
    assert owner.state("a.mkv") == ReviewState.FLAGGED           # no value, nothing will measure it


def test_without_a_full_tier_every_file_runs_full(tmp_path):
    names = ["a.mkv", "b.mkv", "c.mkv"]
    owner = Owner(_ready_folder(tmp_path, names, FolderSettings(brightness_full_detect_files=0)))
    owner.autopilot.on_open()
    full = of_kind(owner.take(), "brightness")
    assert [s.job.file for s in full] == names
    assert all(s.job.folder_plateau is None for s in full)


def test_waiting_files_run_full_when_the_tier_empties_and_cannot_be_refilled(tmp_path):
    project = _ready_folder(tmp_path, ["a.mkv", "b.mkv", "c.mkv"], FolderSettings(brightness_full_detect_files=1))
    owner = Owner(project)
    owner.autopilot.on_open()
    (full,) = of_kind(owner.take(), "brightness")
    project.folder = replace(project.folder, brightness_full_detect_files=0)
    owner.deliver(full, type="failed")
    released = owner.take()
    assert pairs(released) == [("brightness", "b.mkv"), ("brightness", "c.mkv")]
    assert all(s.job.folder_plateau is None for s in released)


def test_turning_dialogue_off_while_the_full_tier_runs_does_not_close_it(tmp_path):
    names = ["a.mkv", "b.mkv", "c.mkv", "d.mkv", "e.mkv"]
    project = _ready_folder(tmp_path, names)
    owner = Owner(project)
    owner.autopilot.on_open()
    full = {s.job.file: s for s in of_kind(owner.take(), "brightness")}
    assert sorted(full) == names[:3]
    owner.deliver(full["a.mkv"], brightness_done(full["a.mkv"], plateau=(100, 120)))

    old = replace(project.folder)
    project.folder = replace(project.folder, dialogue_enabled=False)
    apply_folder_change(project, old, project.folder)
    owner.autopilot.on_folder_changed(old, project.folder)
    owner.deliver(full["b.mkv"], type="cancelled")
    owner.deliver(full["c.mkv"], type="cancelled")
    assert owner.take() == []

    old = replace(project.folder)
    project.folder = replace(project.folder, dialogue_enabled=True)
    apply_folder_change(project, old, project.folder)
    owner.autopilot.on_folder_changed(old, project.folder)
    again = of_kind(owner.take(), "brightness")
    assert pairs(again) == [("brightness", "b.mkv"), ("brightness", "c.mkv")]
    assert all(s.job.folder_plateau is None for s in again)
    owner.deliver(again[0], brightness_done(again[0], plateau=(110, 130)))
    owner.deliver(again[1], brightness_done(again[1], plateau=(105, 125)))
    cheap = owner.take()
    assert pairs(cheap) == [("brightness", "d.mkv"), ("brightness", "e.mkv")]
    assert all(s.job.folder_plateau == (110, 120) for s in cheap)


def test_the_tier_settles_when_dialogue_comes_back_even_with_autopilot_off(tmp_path):
    project = _ready_folder(tmp_path, ["a.mkv", "b.mkv", "c.mkv"], FolderSettings(brightness_full_detect_files=1))
    owner = Owner(project)
    owner.autopilot.on_open()
    (full,) = of_kind(owner.take(), "brightness")

    def change(**fields):
        old = replace(project.folder)
        project.folder = replace(project.folder, **fields)
        apply_folder_change(project, old, project.folder)
        owner.autopilot.on_folder_changed(old, project.folder)
        owner.recompute()

    change(dialogue_enabled=False)
    owner.deliver(full, type="cancelled")
    change(autopilot_enabled=False)
    change(dialogue_enabled=True)
    (promoted,) = of_kind(owner.take(), "brightness")          # a cannot report: b takes its place
    assert promoted.job.file == "b.mkv" and promoted.job.folder_plateau is None
    assert "brightness" not in owner.autopilot.pending().get("a.mkv", set())
    assert owner.state("a.mkv") == ReviewState.FLAGGED


# --------------------------------------------------------------------------
# Brightness waits for ranges
# --------------------------------------------------------------------------

@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_brightness_waiting_for_ranges_is_released_when_they_fail_or_are_cancelled(tmp_path, outcome):
    owner = Owner(_ready_folder(tmp_path, ["a.mkv", "b.mkv"], open_ranges=True))
    owner.autopilot.on_open()
    opened = owner.take()
    assert of_kind(opened, "brightness") == []
    (ranges,) = of_kind(opened, "ranges")
    assert all("brightness" in owner.autopilot.pending()[name] for name in ("a.mkv", "b.mkv"))
    assert owner.state("a.mkv") == ReviewState.PENDING
    owner.deliver(ranges, type=outcome)
    released = of_kind(owner.take(), "brightness")
    assert pairs(released) == [("brightness", "a.mkv"), ("brightness", "b.mkv")]
    assert all(s.job.time_ranges is None and s.job.folder_plateau is None for s in released)


@pytest.mark.parametrize("names, sources", [
    (["a.mkv"], [None]),                                      # a single file: no analysis
    (["a.mkv", "b.mkv"], [Source.MANUAL, Source.IMPORTED]),   # every file's ranges are the user's
])
def test_brightness_does_not_wait_when_no_ranges_analysis_is_needed(tmp_path, names, sources):
    project = _ready_folder(tmp_path, names, open_ranges=True)
    for entry, source in zip(project.files.values(), sources):
        if source is not None:
            entry.time_ranges = TimeRanges([TimeRange("01:00", "20:00")], source)
    owner = Owner(project)
    owner.autopilot.on_open()
    opened = owner.take()
    assert of_kind(opened, "ranges") == []
    brightness = of_kind(opened, "brightness")
    assert [s.job.file for s in brightness] == names
    expected = [None if source is None else (("01:00", "20:00"),) for source in sources]
    assert [s.job.time_ranges for s in brightness] == expected


def test_a_brightness_waiting_for_ranges_is_dropped_once_the_user_sets_the_value(tmp_path):
    project = _ready_folder(tmp_path, ["a.mkv", "b.mkv"], open_ranges=True)
    owner = Owner(project)
    owner.autopilot.on_open()
    (ranges,) = of_kind(owner.take(), "ranges")
    set_manual_brightness(project, "a.mkv", 215)
    assert "brightness" not in owner.autopilot.pending().get("a.mkv", set())
    assert "brightness" in owner.autopilot.pending()["b.mkv"]
    owner.deliver(ranges, ranges_done(ranges))
    assert pairs(of_kind(owner.take(), "brightness")) == [("brightness", "b.mkv")]


def test_a_file_waiting_for_the_tier_is_not_measured_once_the_user_sets_its_value(tmp_path):
    project = _ready_folder(tmp_path, ["a.mkv", "b.mkv", "c.mkv"], FolderSettings(brightness_full_detect_files=1))
    owner = Owner(project)
    owner.autopilot.on_open()
    (full,) = of_kind(owner.take(), "brightness")
    set_manual_brightness(project, "b.mkv", 215)
    assert "brightness" not in owner.autopilot.pending().get("b.mkv", set())
    assert "brightness" in owner.autopilot.pending()["c.mkv"]
    owner.deliver(full, brightness_done(full))
    assert pairs(of_kind(owner.take(), "brightness")) == [("brightness", "c.mkv")]


@pytest.mark.parametrize("with_hint", [False, True])
def test_brightness_asked_for_while_ranges_run_waits_for_them(tmp_path, with_hint):
    project = _ready_folder(tmp_path, ["a.mkv", "b.mkv", "c.mkv"], open_ranges=True)
    _brightness_measured_on(project.files["a.mkv"], 209, Source.DETECTED, BOX)
    _brightness_measured_on(project.files["b.mkv"], 209, Source.DETECTED, BOX)
    project.files["c.mkv"].brightness = Brightness(220, Source.MANUAL)
    owner = Owner(project)
    owner.autopilot.on_open()
    (ranges,) = of_kind(owner.take(), "ranges")

    set_manual_crop(project, "a.mkv", OTHER_BOX)
    owner.autopilot.on_crop_changed("a.mkv")                   # the crop change's re-measure waits
    owner.autopilot.redetect("b.mkv")
    (crop_b,) = of_kind(owner.take(), "crop")                  # crops do not wait
    owner.deliver(crop_b, crop_done(crop_b))                   # the re-detect's brightness step waits
    if with_hint:
        owner.autopilot.redetect_others_with_brightness_hint("c.mkv")
    assert of_kind(owner.take(), "brightness") == []
    assert all("brightness" in owner.autopilot.pending()[name] for name in ("a.mkv", "b.mkv"))

    owner.deliver(ranges, ranges_done(ranges, keep={"a.mkv": [("00:30", "10:00")]}))
    released = {s.job.file: s.job for s in of_kind(owner.take(), "brightness")}
    assert sorted(released) == ["a.mkv", "b.mkv"]
    a, b = released["a.mkv"], released["b.mkv"]
    assert a.crop_box == OTHER_BOX and a.time_ranges == (("00:30", "10:00"),)
    assert b.crop_box == BOX and b.time_ranges is None and b.priority == 12
    if with_hint:
        assert (a.hint_value, a.priority, b.hint_value) == (220, 12, 220)
    else:
        assert (a.hint_value, a.priority, b.hint_value) == (None, 2, None)


# --------------------------------------------------------------------------
# Ranges and added files
# --------------------------------------------------------------------------

def test_added_files_get_their_jobs_and_ranges_again(tmp_path):
    owner = Owner(_project(tmp_path, ["a.mkv", "b.mkv"]))
    owner.autopilot.on_open()
    ranges = of_kind(owner.take(), "ranges")
    assert len(ranges) == 1
    owner.deliver(ranges[0], None, type="cancelled")
    owner.autopilot.on_open()                          # a cancelled analysis has not completed
    (ranges,) = of_kind(owner.take(), "ranges")
    owner.deliver(ranges, ranges_done(ranges))
    owner.autopilot.on_open()                          # already analysed this session
    assert of_kind(owner.take(), "ranges") == []

    owner.project.files["c.mkv"] = FileEntry(name="c.mkv")
    owner.autopilot.on_files_added(["c.mkv"])
    added = owner.take()
    assert pairs(added) == [("ranges", None), ("metadata", "c.mkv")]
    assert list(added[0].job.files) == ["a.mkv", "b.mkv", "c.mkv"]

    owner.project.files["d.mkv"] = FileEntry(name="d.mkv")
    owner.autopilot.on_files_added(["d.mkv"])          # the queued analysis is replaced
    again = owner.take()
    assert pairs(again) == [("ranges", None), ("metadata", "d.mkv")]
    assert owner.deliver(added[0], None, type="cancelled") is False
    assert owner.autopilot.ranges_pending() is True
    assert owner.deliver(again[0], ranges_done(again[0])) is True
    assert owner.autopilot.ranges_pending() is False


def test_a_file_that_disappears_and_comes_back_gets_its_thumbnail_again(tmp_path):
    project = _ready_folder(tmp_path, ["a.mkv", "b.mkv"])
    owner = Owner(project)
    owner.autopilot.on_open()
    assert len(of_kind(owner.take(), "thumbnail")) == 2
    entry = project.files.pop("b.mkv")
    project.files["b.mkv"] = entry
    owner.autopilot.on_files_added(["b.mkv"])
    (thumbnail,) = of_kind(owner.take(), "thumbnail")
    assert thumbnail.job.file == "b.mkv" and thumbnail.job.time == 300.0


@pytest.mark.parametrize("names, sources, submitted", [
    (["a.mkv"], [None], False),
    (["a.mkv", "b.mkv"], [Source.MANUAL, Source.IMPORTED], False),
    (["a.mkv", "b.mkv"], [Source.MANUAL, Source.DETECTED], True),
    (["a.mkv", "b.mkv"], [Source.IMPORTED, None], True),
])
def test_ranges_run_only_for_two_or_more_files_with_ranges_open_to_detection(tmp_path, names, sources, submitted):
    project = _project(tmp_path, names)
    for entry, source in zip(project.files.values(), sources):
        if source is not None:
            entry.time_ranges = TimeRanges([TimeRange("01:00", "20:00")], source)
    owner = Owner(project)
    owner.autopilot.on_open()
    assert bool(of_kind(owner.take(), "ranges")) is submitted


# --------------------------------------------------------------------------
# Consensus and hints
# --------------------------------------------------------------------------

def test_crop_consensus_follows_the_old_adapters_pool_rule_in_name_order(tmp_path):
    names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    project = _project(tmp_path, names)
    for entry in project.files.values():
        _media(entry)
    files = project.files
    files["a"].crop = Crop(0, 810, 1920, 60, Source.DETECTED)
    files["b"].crop = Crop(0, 900, 1920, 50, Source.IMPORTED)
    files["b"].media.height = 1000
    files["b"].flags["crop"] = crop_mod.FLAG_LOW_AGREEMENT      # flags describe a detection, not the import
    files["c"].crop = Crop(0, 700, 1920, 40, Source.DETECTED)
    files["c"].flags["crop"] = crop_mod.FLAG_LOW_AGREEMENT      # flagged
    files["d"].crop = Crop(0, 972, 1920, 54, Source.DETECTED)
    files["d"].flags["crop"] = crop_mod.FLAG_NO_SPEECH          # informational, but still a flag
    files["e"].crop = Crop(0, 600, 1920, 40, Source.MANUAL)
    files["f"].crop = Crop(0, 610, 1920, 40, Source.HINT)
    files["g"].crop = Crop(0, 620, 1920, 40, Source.DETECTED)
    files["g"].media.height = 0                                 # height unknown
    files["i"].crop = Crop(0, 540, 1920, 108, Source.IMPORTED)
    files["j"].crop = None
    expected = [(810 / 1080, 60 / 1080), (900 / 1000, 50 / 1000), (600 / 1080, 40 / 1080), (540 / 1080, 108 / 1080)]

    assert crop_consensus(project) == expected
    assert crop_consensus(project, exclude="b") == [expected[0], expected[2], expected[3]]

    owner = Owner(project)
    owner.autopilot.on_open()
    (first,) = of_kind(owner.take(), "crop")                   # h now, j once h ends
    assert first.job.file == "h" and list(first.job.consensus) == expected
    owner.autopilot.redetect("a")                              # outside the chain: now, with the current pool
    (redetect,) = of_kind(owner.take(), "crop")
    assert redetect.job.file == "a" and list(redetect.job.consensus) == expected[1:]


def test_auto_fill_crops_run_one_at_a_time_and_grow_the_consensus(tmp_path):
    names = ["a.mkv", "b.mkv", "c.mkv", "d.mkv", "e.mkv", "f.mkv", "g.mkv"]
    project = _project(tmp_path, names)
    for entry in project.files.values():
        _media(entry).evidence["audio"] = {}
        entry.sample_time = 300.0
    project.files["a.mkv"].crop = Crop(0, 972, 1920, 54, Source.IMPORTED)
    project.files["c.mkv"].crop = Crop(0, 900, 1920, 60, Source.MANUAL)
    project.files["c.mkv"].media.height = 1000
    pool = [(972 / 1080, 54 / 1080), (900 / 1000, 60 / 1000)]
    owner = Owner(project)

    def next_crops():
        return of_kind(owner.take(), "crop")

    owner.autopilot.on_open()
    (b,) = next_crops()
    assert b.job.file == "b.mkv" and b.job.priority == 3 and list(b.job.consensus) == pool
    assert all("crop" in owner.autopilot.pending()[name] for name in ("d.mkv", "e.mkv", "f.mkv", "g.mkv"))

    owner.autopilot.redetect("g.mkv")                          # submitted now, outside the chain
    (g,) = next_crops()
    assert g.job.file == "g.mkv" and g.job.priority == 13 and list(g.job.consensus) == pool

    owner.deliver(b, crop_done(b, box=(0, 950, 1920, 50), flagged=crop_mod.FLAG_NO_SPEECH))
    assert project.files["b.mkv"].crop == Crop(0, 950, 1920, 50, Source.DETECTED)   # applied, not pooled
    (d,) = next_crops()
    assert d.job.file == "d.mkv" and list(d.job.consensus) == pool

    owner.deliver(d, type="failed")
    (e,) = next_crops()
    assert e.job.file == "e.mkv" and list(e.job.consensus) == pool

    owner.deliver(e, type="cancelled")
    (f,) = next_crops()
    assert f.job.file == "f.mkv" and list(f.job.consensus) == pool

    owner.deliver(g, type="failed")
    assert next_crops() == []                                  # the chain waits for f
    owner.deliver(f, crop_done(f, box=(0, 940, 1920, 52)))
    assert next_crops() == []                                  # g's re-detect replaced its auto-fill: done
    assert all("crop" not in kinds for kinds in owner.autopilot.pending().values())
    assert {owner.state(name) for name in ("d.mkv", "e.mkv", "g.mkv")} == {ReviewState.FLAGGED}

    owner.autopilot.redetect("d.mkv")
    (again,) = next_crops()
    assert list(again.job.consensus) == pool + [(940 / 1080, 52 / 1080)]


def test_the_crop_chain_skips_a_file_the_user_gave_a_crop_while_it_waited(tmp_path):
    project = _project(tmp_path, ["a.mkv", "b.mkv", "c.mkv"])
    for entry in project.files.values():
        _media(entry).evidence["audio"] = {}
        entry.sample_time = 300.0
    owner = Owner(project)
    owner.autopilot.on_open()
    (crop_a,) = of_kind(owner.take(), "crop")
    set_manual_crop(project, "b.mkv", OTHER_BOX)
    owner.autopilot.on_crop_changed("b.mkv")
    owner.take()
    owner.deliver(crop_a, crop_done(crop_a))
    (crop_c,) = of_kind(owner.take(), "crop")
    assert crop_c.job.file == "c.mkv"
    assert list(crop_c.job.consensus) == [(BOX[1] / HEIGHT, BOX[3] / HEIGHT),
                                          (OTHER_BOX[1] / HEIGHT, OTHER_BOX[3] / HEIGHT)]


def test_hint_redetects_replace_the_auto_fill_crops_still_waiting(tmp_path):
    project = _project(tmp_path, ["a.mkv", "b.mkv", "c.mkv"])
    for entry in project.files.values():
        _media(entry).evidence["audio"] = {}
        entry.sample_time = 300.0
    project.files["a.mkv"].crop = Crop(*BOX, Source.MANUAL)
    owner = Owner(project)
    owner.autopilot.on_open()
    (auto_b,) = of_kind(owner.take(), "crop")                  # c waits in the chain
    owner.autopilot.redetect_others_with_crop_hint("a.mkv")
    hinted = {s.job.file: s for s in of_kind(owner.take(), "crop")}
    assert sorted(hinted) == ["b.mkv", "c.mkv"] and all(s.job.hint for s in hinted.values())
    assert owner.deliver(auto_b, None, type="cancelled") is False   # replaced while queued
    owner.deliver(hinted["c.mkv"], type="failed")
    owner.deliver(hinted["b.mkv"], type="failed")
    assert of_kind(owner.take(), "crop") == []
    assert all("crop" not in kinds for kinds in owner.autopilot.pending().values())


def test_hint_redetects_seed_every_other_file_from_the_edited_file(tmp_path):
    names = ["a.mkv", "b.mkv", "c.mkv", "d.mkv", "e.mkv"]
    project = _project(tmp_path, names)
    files = project.files
    for name in names[:4]:
        _media(files[name])
    files["a.mkv"].crop = Crop(0, 972, 1920, 54, Source.MANUAL)
    files["a.mkv"].brightness = Brightness(215, Source.MANUAL)
    files["b.mkv"].crop = Crop(*BOX, Source.DETECTED)
    files["c.mkv"].crop = Crop(*OTHER_BOX, Source.IMPORTED)
    files["c.mkv"].brightness = Brightness(209, Source.IMPORTED)
    owner = Owner(project)
    hint = (972 / 1080, 54 / 1080)

    owner.autopilot.redetect_others_with_crop_hint("a.mkv")
    crops = owner.take()
    assert pairs(crops) == [("crop", "b.mkv"), ("crop", "c.mkv"), ("crop", "d.mkv"), ("metadata", "e.mkv")]
    for sub in crops[:3]:
        assert sub.job.hint == hint
        assert sub.job.consensus == (hint,) * CONSENSUS_MIN_ENTRIES
        assert sub.job.priority == 13
    assert crops[3].job.priority == 15
    owner.deliver(crops[3], metadata_done(crops[3]))
    (deferred,) = owner.take()
    assert deferred.job.kind == "crop" and deferred.job.file == "e.mkv"
    assert deferred.job.hint == hint and deferred.job.priority == 13

    owner.autopilot.redetect_others_with_brightness_hint("a.mkv")
    brightness = owner.take()
    assert pairs(brightness) == [("brightness", "b.mkv"), ("brightness", "c.mkv")]   # d and e have no crop
    for sub, box in zip(brightness, (BOX, OTHER_BOX)):
        assert sub.job.hint_value == 215 and sub.job.folder_plateau is None
        assert sub.job.crop_box == box and sub.job.priority == 12


def test_hint_redetects_need_a_value_to_seed_from(tmp_path):
    project = _project(tmp_path, ["a.mkv", "b.mkv"])
    for entry in project.files.values():
        _media(entry).crop = Crop(*BOX, Source.DETECTED)
    project.files["a.mkv"].crop = None
    owner = Owner(project)
    owner.autopilot.redetect_others_with_crop_hint("a.mkv")
    owner.autopilot.redetect_others_with_brightness_hint("a.mkv")
    owner.autopilot.redetect_others_with_crop_hint("missing.mkv")
    assert owner.take() == []


# --------------------------------------------------------------------------
# Pause, pending and superseded results
# --------------------------------------------------------------------------

def test_pause_and_resume_hold_only_detection_jobs_on_the_gpu_and_cpu_lanes(tmp_path):
    owner = Owner(_project(tmp_path, ["a.mkv"]))
    owner.autopilot.pause()
    owner.autopilot.pause()                               # idempotent
    assert [lane for lane, _ in owner.runner.pauses] == [Lane.GPU, Lane.CPU]
    for _, only in owner.runner.pauses:
        assert only is is_detection_job
    folder = FolderSettings()
    held = [AudioProfileJob("/p", "a", 10.0), RangesJob("/p", ["a", "b"], folder),
            CropJob("/p", "a", 10.0, [], folder), BrightnessJob("/p", "a", BOX, None, folder)]
    assert all(is_detection_job(job) for job in held)
    assert {job.kind for job in held} == DETECTION_KINDS
    never_held = [MetadataJob("/p", "a"), ThumbnailJob("/p", "a", 1.0),
                  SimpleNamespace(kind="proof", lane=Lane.GPU), SimpleNamespace(kind="run", lane=Lane.RUN)]
    assert not any(is_detection_job(job) for job in never_held)
    assert is_autopilot_job(never_held[0]) and is_autopilot_job(never_held[1])
    assert not is_autopilot_job(never_held[2])

    owner.autopilot.resume()
    owner.autopilot.resume()
    assert owner.runner.resumes == [Lane.GPU, Lane.CPU]


def test_pending_tracks_queued_running_and_finished_jobs(tmp_path):
    owner = Owner(_project(tmp_path, ["a.mkv", "b.mkv"]))
    autopilot = owner.autopilot
    assert autopilot.pending() == {} and autopilot.ranges_pending() is False
    autopilot.on_open()
    opened = owner.take()
    assert autopilot.pending() == {name: {"metadata", "crop", "brightness"} for name in ("a.mkv", "b.mkv")}
    assert autopilot.ranges_pending() is True

    ranges, metadata_a, metadata_b = opened
    owner.deliver(metadata_a, metadata_done(metadata_a))
    assert autopilot.pending()["a.mkv"] == {"crop", "thumbnail", "audio_profile", "brightness"}
    owner.deliver(metadata_b, type="failed")
    assert "b.mkv" not in autopilot.pending()
    assert owner.state("b.mkv") == ReviewState.FLAGGED

    after = owner.take()
    (crop,) = of_kind(after, "crop")
    owner.deliver(crop, crop_done(crop))
    after_crop = owner.take()
    assert of_kind(after_crop, "brightness") == []                      # waits for the ranges
    assert autopilot.pending()["a.mkv"] == {"thumbnail", "audio_profile", "brightness"}
    assert owner.state("a.mkv") == ReviewState.PENDING
    owner.deliver(ranges, ranges_done(ranges))
    assert autopilot.ranges_pending() is False
    (brightness,) = of_kind(owner.take(), "brightness")
    assert autopilot.pending()["a.mkv"] == {"thumbnail", "audio_profile", "brightness"}
    owner.deliver(brightness, brightness_done(brightness))
    finish_side_jobs(owner, after + after_crop)
    assert autopilot.pending() == {}
    assert owner.state("a.mkv") == ReviewState.PROPOSED


def test_a_superseded_result_is_not_current_and_keeps_the_kind_pending(tmp_path):
    project = _project(tmp_path, ["a.mkv"])
    _media(project.files["a.mkv"]).evidence["audio"] = {}
    project.files["a.mkv"].sample_time = 300.0
    owner = Owner(project)
    owner.autopilot.on_open()
    (old,) = of_kind(owner.take(), "crop")
    owner.autopilot.redetect("a.mkv")          # its "queued" is never drained before old's "finished"
    (new,) = of_kind(owner.take(), "crop")

    assert owner.autopilot.is_current(owner.event(old, crop_done(old, box=OTHER_BOX))) is False
    assert owner.deliver(old, crop_done(old, box=OTHER_BOX)) is False
    assert project.files["a.mkv"].crop is None                        # the older result was not applied
    assert "crop" in owner.autopilot.pending()["a.mkv"]
    assert owner.take() == []
    assert owner.state("a.mkv") == ReviewState.PENDING

    assert owner.deliver(new, crop_done(new, box=BOX)) is True
    assert project.files["a.mkv"].crop == Crop(*BOX, Source.DETECTED)
    assert "crop" not in owner.autopilot.pending().get("a.mkv", set())
    (brightness,) = owner.take()
    assert brightness.job.kind == "brightness" and brightness.job.priority == 12   # the re-detect's


def test_a_queued_job_replaced_by_a_newer_one_is_not_current(tmp_path):
    project = _ready_folder(tmp_path, ["a.mkv"])
    project.files["a.mkv"].crop = Crop(*BOX, Source.MANUAL)
    project.files["a.mkv"].brightness = Brightness(209, Source.MANUAL)
    owner = Owner(project)
    owner.autopilot.redetect_others_with_brightness_hint("missing.mkv")
    owner.autopilot.redetect("a.mkv")
    (first,) = owner.take()
    owner.autopilot.redetect("a.mkv")
    (second,) = owner.take()
    assert owner.deliver(first, None, type="cancelled") is False
    assert owner.take() == []
    assert owner.deliver(second, crop_done(second)) is True
    (brightness,) = owner.take()
    assert brightness.job.kind == "brightness" and brightness.job.priority == 12


def test_a_newer_job_cancelled_while_an_older_one_runs_leaves_the_older_one_current(tmp_path):
    project = _project(tmp_path, ["a.mkv", "b.mkv"])
    for entry in project.files.values():
        _media(entry).evidence["audio"] = {}
        entry.sample_time = 300.0
        entry.time_ranges = TimeRanges([], Source.MANUAL)
    owner = Owner(project)
    owner.autopilot.on_open()
    (older,) = of_kind(owner.take(), "crop")                   # a's auto-fill crop, running
    owner.autopilot.redetect("a.mkv")
    (newer,) = owner.take()
    # Cancelled while queued, the newer job's terminal event arrives before the older one's.
    assert owner.deliver(newer, None, type="cancelled") is False
    assert owner.take() == []
    assert "crop" in owner.autopilot.pending()["a.mkv"]
    assert owner.deliver(older, crop_done(older)) is True      # it ends the key's last submission
    assert project.files["a.mkv"].crop == Crop(*BOX, Source.DETECTED)
    after = owner.take()
    assert pairs(of_kind(after, "crop")) == [("crop", "b.mkv")]          # the chain moves on
    assert "crop" not in owner.autopilot.pending().get("a.mkv", set())


def test_events_autopilot_did_not_submit_are_current_and_ignored(tmp_path):
    owner = Owner(_project(tmp_path, ["a.mkv", "b.mkv"]))
    owner.autopilot.on_open()
    owner.take()
    before = owner.autopilot.pending()
    for event in (JobEvent("finished", "proof:a.mkv", "proof", "a.mkv", job_id=99, result=object()),
                  JobEvent("finished", "run", "run", None, job_id=100),
                  JobEvent("finished", "crop:b.mkv", "crop", "b.mkv", job_id=101)):
        assert owner.autopilot.is_current(event) is True
        owner.autopilot.on_job_event(event)
    assert owner.take() == []
    assert owner.autopilot.pending() == before


def test_non_terminal_events_change_nothing(tmp_path):
    owner = Owner(_project(tmp_path, ["a.mkv"]))
    owner.autopilot.on_open()
    (metadata,) = owner.take()
    for event_type in ("queued", "started", "progress", "log"):
        owner.autopilot.on_job_event(owner.event(metadata, type=event_type))
    assert owner.autopilot.pending() == {"a.mkv": {"metadata", "crop", "brightness"}}


# --------------------------------------------------------------------------
# Crop and folder changes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("source, requeued", [
    (None, True),
    (Source.DETECTED, True),
    (Source.HINT, True),
    (Source.MANUAL, False),
    (Source.IMPORTED, False),
])
def test_a_crop_change_requeues_detected_brightness_with_the_new_box(tmp_path, source, requeued):
    project = _ready_folder(tmp_path, ["a.mkv"])
    entry = project.files["a.mkv"]
    entry.crop = Crop(*OTHER_BOX, Source.DETECTED)
    if source is not None:
        _brightness_measured_on(entry, 209, source, OTHER_BOX)
    owner = Owner(project)

    set_manual_crop(project, "a.mkv", BOX)
    owner.autopilot.on_crop_changed("a.mkv")
    submitted = owner.take()
    if not requeued:
        assert submitted == []
        return
    assert pairs(submitted) == [("brightness", "a.mkv")]
    assert submitted[0].job.crop_box == BOX and submitted[0].job.priority == 2
    owner.autopilot.on_crop_changed("a.mkv")                  # already measuring this box
    assert owner.take() == []


def test_a_crop_change_while_brightness_is_measured_on_the_old_box_resubmits_it(tmp_path):
    project = _ready_folder(tmp_path, ["a.mkv", "b.mkv"])
    project.files["b.mkv"].brightness = Brightness(220, Source.MANUAL)
    owner = Owner(project)
    owner.autopilot.redetect_others_with_brightness_hint("b.mkv")
    (hinted,) = owner.take()
    set_manual_crop(project, "a.mkv", OTHER_BOX)
    owner.autopilot.on_crop_changed("a.mkv")
    (again,) = owner.take()
    assert again.job.crop_box == OTHER_BOX
    assert again.job.hint_value == 220 and again.job.priority == 12 and again.job.folder_plateau is None
    assert owner.deliver(hinted, brightness_done(hinted)) is False
    assert owner.deliver(again, brightness_done(again)) is True
    assert project.files["a.mkv"].brightness == Brightness(200, Source.HINT)


@pytest.mark.parametrize("result_box, requeued", [(BOX, True), (OTHER_BOX, False)])
def test_an_applied_crop_result_requeues_brightness_measured_on_the_old_crop(tmp_path, result_box, requeued):
    project = _ready_folder(tmp_path, ["a.mkv", "b.mkv"])
    a = project.files["a.mkv"]
    a.crop = Crop(*OTHER_BOX, Source.DETECTED)
    _brightness_measured_on(a, 209, Source.DETECTED, OTHER_BOX)
    b = project.files["b.mkv"]
    b.crop = Crop(*BOX, Source.MANUAL)
    b.brightness = Brightness(209, Source.MANUAL)
    owner = Owner(project)

    owner.autopilot.redetect_others_with_crop_hint("b.mkv")
    (crop,) = owner.take()
    owner.deliver(crop, crop_done(crop, box=result_box, hit=300.0))
    assert a.crop == Crop(*result_box, Source.HINT)
    submitted = owner.take()
    if requeued:
        assert pairs(submitted) == [("brightness", "a.mkv")] and submitted[0].job.crop_box == BOX
    else:
        assert submitted == []


def test_the_thumbnail_reruns_only_when_the_crop_moves_the_sample_time(tmp_path):
    project = _project(tmp_path, ["a.mkv", "b.mkv"])
    for entry in project.files.values():
        _media(entry).evidence["audio"] = {}
    project.files["b.mkv"].sample_time = 50.0
    owner = Owner(project)
    owner.autopilot.on_open()
    opened = owner.take()
    thumbs = {s.job.file: s.job.time for s in of_kind(opened, "thumbnail")}
    assert thumbs == {"a.mkv": pytest.approx(0.4 * DURATION), "b.mkv": 50.0}
    (crop_a,) = of_kind(opened, "crop")

    owner.deliver(crop_a, crop_done(crop_a, hit=123.0))
    after_a = owner.take()
    (rethumb,) = of_kind(after_a, "thumbnail")
    assert rethumb.job.file == "a.mkv" and rethumb.job.time == 123.0 and rethumb.job.priority == 1
    (crop_b,) = of_kind(after_a, "crop")
    owner.deliver(crop_b, crop_done(crop_b, box=None, flagged=crop_mod.FLAG_LOW_AGREEMENT))
    assert of_kind(owner.take(), "thumbnail") == []         # nothing written, sample time kept


def test_labels_only_to_dialogue_submits_crop_then_brightness(tmp_path):
    names = ["a.mkv", "b.mkv"]
    project = _project(tmp_path, names, FolderSettings(dialogue_enabled=False, labels_enabled=True))
    for entry in project.files.values():
        _media(entry).evidence["audio"] = {}
        entry.sample_time = 10.0
    owner = Owner(project)
    owner.autopilot.on_open()
    opened = owner.take()
    assert of_kind(opened, "crop") == [] and of_kind(opened, "brightness") == []
    finish_side_jobs(owner, opened)
    (ranges,) = of_kind(opened, "ranges")
    owner.deliver(ranges, ranges_done(ranges))

    old = replace(project.folder)
    project.folder = replace(project.folder, dialogue_enabled=True)
    apply_folder_change(project, old, project.folder)
    owner.autopilot.on_folder_changed(old, project.folder)
    crops = owner.take()
    assert pairs(crops) == [("crop", "a.mkv")]                  # b follows when a ends
    owner.recompute()
    assert all(owner.autopilot.pending()[name] == {"crop", "brightness"} for name in names)
    assert {owner.state(name) for name in names} == {ReviewState.PENDING}

    owner.deliver(crops[0], crop_done(crops[0]))
    after_a = owner.take()
    (crop_b,) = of_kind(after_a, "crop")
    owner.deliver(crop_b, crop_done(crop_b))
    assert pairs(of_kind(after_a + owner.take(), "brightness")) == [("brightness", name) for name in names]

    back = replace(project.folder)
    project.folder = replace(project.folder, dialogue_enabled=False)
    owner.autopilot.on_folder_changed(back, project.folder)
    assert owner.take() == []


def test_turning_autopilot_on_schedules_detections_and_off_schedules_nothing(tmp_path):
    project = _project(tmp_path, ["a.mkv", "b.mkv"], FolderSettings(autopilot_enabled=False))
    owner = Owner(project)
    owner.autopilot.on_open()
    opened = owner.take()
    assert pairs(opened) == [("metadata", "a.mkv"), ("metadata", "b.mkv")]
    assert owner.autopilot.pending() == {"a.mkv": {"metadata"}, "b.mkv": {"metadata"}}

    old = replace(project.folder)
    project.folder = replace(project.folder, autopilot_enabled=True)
    owner.autopilot.on_folder_changed(old, project.folder)
    assert pairs(owner.take()) == [("ranges", None)]                     # metadata is already queued
    assert owner.autopilot.pending()["a.mkv"] == {"metadata", "crop", "brightness"}
    for sub in opened:
        owner.deliver(sub, metadata_done(sub))
    assert sorted(pairs(owner.take())) == sorted(
        [("crop", "a.mkv")] + [(kind, name) for name in ("a.mkv", "b.mkv") for kind in ("thumbnail", "audio_profile")])

    old = replace(project.folder)
    project.folder = replace(project.folder, autopilot_enabled=False, labels_enabled=False)
    owner.autopilot.on_folder_changed(old, project.folder)
    assert owner.take() == []


def test_autopilot_module_imports_no_qt():
    code = ("import sys, core.jobs.autopilot; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('PyQt6', 'PyQt5', 'PySide6')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False,
                          cwd=Path(__file__).resolve().parent.parent)
    assert proc.returncode == 0, proc.stdout + proc.stderr
