"""Auto-pilot: which detection jobs to submit for a project folder, and when.

Pure decisions over the Qt-free model plus JobRunner.submit() calls. AutoPilot
owns no threads, imports no Qt (ruling C8) and never mutates the project: it
reads it through `project_getter` on every call, and results reach the model
only through core.jobs.apply, applied by the model owner.

Order (ruling C1: fill missing values only; re-detects ignore sources)
    1. metadata for every file whose duration is unknown;
    2. once the duration is known: thumbnail (at sample_time, else
       THUMBNAIL_FRACTION of the duration), audio profile (no "audio"
       evidence) and crop (no crop, folder not labels-only). Auto-fill crops
       run as a chain, one at a time in name order: the next is submitted when
       the previous one's terminal event arrives (after the owner applied it),
       so each is seeded with every result before it (below). The chain costs
       no throughput (the GPU lane runs one job at a time), but it does not
       keep brightness behind crops: the GPU worker starts whatever is queued
       as soon as a crop's terminal event is sent, while the next chained crop
       is submitted only when the owner drains that event, so brightness jobs
       and chained crops interleave. Files waiting in the chain are pending
       "crop";
    3. once the crop is known: brightness (missing or stale, dialogue on), in
       two tiers (below), and only once the folder's ranges are settled
       (below). An applied crop result that moves sample_time re-runs the
       thumbnail there;
    4. ranges once per session for the whole folder, when it has >= 2 files
       and some file's ranges are still open to detection (None, DETECTED or
       HINT). on_files_added resubmits it.
    Priorities (within a lane): metadata 5 > crop 3 > brightness 2 >
    thumbnail, audio 1 > ranges 0; re-detects get +REDETECT_BOOST.

Crop consensus (crop_consensus)
    The old adapter's pool (core/subtitle_detector.py), in name order:
    (y / height, h / height) of every other file whose crop is IMPORTED or
    MANUAL, or DETECTED from a result with no flag at all (an informational
    flag such as no-speech keeps it out), with media height known. Taken when
    the job is built; the file being detected is never part of its own
    consensus. A consensus of >= CONSENSUS_MIN_ENTRIES that agrees with the
    raw union lets detect_crop stop early, which is why the chain grows it.
    redetect() is not part of the chain: it is submitted at once, with the
    pool as it stands. Hint re-detects use the hint consensus (CropJob).

Brightness waits for ranges
    detect_brightness samples inside the file's keep ranges, so no brightness
    job is submitted while ranges_pending() (a ranges analysis is queued or
    running): auto-fill, full tier and cheap, escalation, on_crop_changed,
    redetect's brightness step and the brightness hint re-detect alike. Each
    request waits (the newest per file) and is submitted when the analysis
    ends (finished, failed or cancelled), with the file's time_ranges as they
    are then; a waiting auto-pilot request whose value no longer needs
    measuring (the user set it) is dropped, explicit re-detects are not.
    When no analysis runs (one file, every file's ranges the user's, or
    auto-pilot off) nothing waits. Waiting files are pending "brightness". A
    later change of ranges does not re-measure brightness.

Two-tier brightness
    The first folder.brightness_full_detect_files files in name order that
    need detection this session form the full tier and run full detection.
    Once every tier file has a result, the folder plateau is the
    intersection of their plateaus, read from this session's full-run
    BrightnessJobResults (None when any is None or they do not overlap), and
    the tier closes. Files needing brightness after that (or waiting for it)
    run cheap with that plateau, or full when it is None. A cheap result
    flagged "escalate" is resubmitted full. A tier file whose full run cannot
    report a plateau (no crop found; its metadata or brightness job failed or
    was cancelled; its brightness set by the user before a run was submitted;
    its run measured a crop that has since been replaced; the file removed)
    leaves the tier, and the first waiting file in name order takes its
    place. When no file can take it and the tier is empty (the setting fell
    to 0), waiting files run full. With brightness_full_detect_files <= 0
    every file runs full. Waiting files count as pending (they are not
    flagged while they wait). While dialogue is off the tier is left as it
    is (nothing is dropped, promoted or closed), so switching dialogue off
    and on again cannot close it on a partial plateau.

Superseded jobs
    Submissions are counted per key: +1 on submit, -1 on the terminal event
    for that key. The runner delivers exactly one terminal event per
    submission, so a terminal event is current iff it ends the last
    outstanding submission for its key. Same-key terminal events normally
    arrive in submission order, and then that is the newest job. They do not
    when a newer job is removed from the queue (cancel, cancel_where,
    shutdown) while an older one with the same key runs: the newer job's
    "cancelled" (result None, nothing to apply) arrives first and is not
    current, and the older job's terminal event, which ends the key's work,
    is current and drives what follows. This does not depend on "queued"
    events having been drained. pending() reports a kind for a file while any
    submission for that key is outstanding. Every job of an AUTOPILOT_KINDS
    kind must be submitted through AutoPilot; events for keys it never
    submitted (proof, run) are current and otherwise ignored.

autopilot_enabled
    Gates the detections AutoPilot starts on its own (crop, brightness,
    ranges, audio profile) where they are first wanted: on_open,
    on_files_added, after a scheduled file's metadata, and on_folder_changed
    (turning it on schedules the folder's detections). Metadata and
    thumbnails always run on on_open/on_files_added: the queue needs
    durations and thumbnails. Explicit requests (redetect, the hint
    re-detects), on_crop_changed and the continuation of work already
    started (the crop chain, brightness after a crop, escalation, the tier)
    run regardless.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from core.detect.brightness import FLAG_ESCALATE
from core.jobs.apply import DETECTION_SOURCES, brightness_is_stale
from core.jobs.detect_jobs import (
    AudioProfileJob,
    BrightnessJob,
    BrightnessJobResult,
    CropJob,
    MetadataJob,
    RangesJob,
    ThumbnailJob,
)
from core.jobs.runner import Job, JobEvent, JobRunner, Lane
from core.project.model import FileEntry, FolderSettings, Project, Source

AUTOPILOT_KINDS = frozenset({"metadata", "thumbnail", "crop", "brightness", "ranges", "audio_profile"})
DETECTION_KINDS = frozenset({"crop", "brightness", "ranges", "audio_profile"})   # what pause() holds
TERMINAL_EVENTS = frozenset({"finished", "failed", "cancelled"})
PRIORITY = {"metadata": 5, "crop": 3, "brightness": 2, "thumbnail": 1, "audio_profile": 1, "ranges": 0}
REDETECT_BOOST = 10
THUMBNAIL_FRACTION = 0.4          # thumbnail time without a sample time, as a fraction of the duration
RANGES_KEY = "ranges:*"

Box = tuple[int, int, int, int]
Plateau = tuple[int, int]


def is_autopilot_job(job: Job) -> bool:
    """A job of a kind AutoPilot owns (and must be submitted through it)."""
    return getattr(job, "kind", None) in AUTOPILOT_KINDS


def is_detection_job(job: Job) -> bool:
    """The `only` predicate pause() holds: detection jobs. Metadata and
    thumbnails are never held (ruling C6 holds detection jobs)."""
    return getattr(job, "kind", None) in DETECTION_KINDS


CONSENSUS_SOURCES = frozenset({Source.IMPORTED, Source.MANUAL})   # plus unflagged DETECTED crops


def crop_consensus(project: Project, exclude: str | None = None) -> list[tuple[float, float]]:
    """(y_frac, h_frac) of every file except `exclude` whose crop is IMPORTED
    or MANUAL, or DETECTED with no crop flag at all, and whose media height is
    known, in name order."""
    consensus = []
    for name, entry in project.files.items():
        crop, height = entry.crop, entry.media.height
        if name == exclude or crop is None or not height or height <= 0:
            continue
        if crop.source in CONSENSUS_SOURCES or (crop.source == Source.DETECTED and not entry.flags.get("crop")):
            consensus.append((crop.y / height, crop.height / height))
    return consensus


def intersect_plateaus(plateaus: Iterable[Plateau | None]) -> Plateau | None:
    """The thresholds every plateau contains: None when there are none, any
    plateau is None, or they do not overlap."""
    plateaus = list(plateaus)
    if not plateaus or any(plateau is None for plateau in plateaus):
        return None
    lo = max(int(plateau[0]) for plateau in plateaus)
    hi = min(int(plateau[1]) for plateau in plateaus)
    return (lo, hi) if lo <= hi else None


def _box(values) -> Box:
    x, y, width, height = values
    return int(x), int(y), int(width), int(height)


def _crop_box(entry: FileEntry) -> Box:
    crop = entry.crop
    return _box((crop.x, crop.y, crop.width, crop.height))


def _time_ranges(entry: FileEntry) -> list[tuple[str | None, str | None]] | None:
    if entry.time_ranges is None or not entry.time_ranges.ranges:
        return None
    return [(r.start, r.end) for r in entry.time_ranges.ranges]


def _crop_wanted(folder: FolderSettings, entry: FileEntry) -> bool:
    """An auto-fill crop is wanted: the file has none and the folder is not
    labels-only."""
    return entry.crop is None and not folder.labels_only


def _brightness_missing(entry: FileEntry) -> bool:
    """None, or a detected/hinted value measured on another crop."""
    return entry.brightness is None or brightness_is_stale(entry)


def _brightness_unmeasured(entry: FileEntry) -> bool:
    """None, or a detected/hinted value not known to be measured on the
    file's current crop (a missing value_crop_box record counts as not)."""
    brightness = entry.brightness
    if brightness is None:
        return True
    if brightness.source not in DETECTION_SOURCES or entry.crop is None:
        return brightness.source in DETECTION_SOURCES
    measured = (entry.evidence.get("brightness") or {}).get("value_crop_box")
    return measured is None or _box(measured) != _crop_box(entry)


@dataclass(frozen=True)
class _BrightnessRequest:
    """A brightness job for a file, submitted or waiting for the ranges."""
    box: Box
    plateau: Plateau | None        # None: full detection
    hint_value: int | None
    priority: int
    explicit: bool = False         # a user request (re-detect, hint): runs whatever the value's source


@dataclass(frozen=True)
class _CropRequest:
    """A crop detection waiting for the file's metadata."""
    hint: tuple[float, float] | None
    boost: bool


_UNRESOLVED = object()   # a full-tier file whose full run has not reported yet


class AutoPilot:
    """Decides which jobs to submit for a project, in dependency order. Pure
    decisions over the model plus submit() calls; owns no threads. The model
    owner calls every method on its own thread.

    The owner's protocol, for every JobEvent it drains:
        1. current = autopilot.is_current(event)   -- before on_job_event,
           which consumes the event's submission;
        2. if current: apply the result with core.jobs.apply (never apply
           a result that is not current);
        3. autopilot.on_job_event(event)           -- AFTER the apply, for
           every event, current or not, so the scheduler sees the applied
           values (e.g. the applied crop, which brightness is then measured
           on);
        4. recompute_all(project, pending=autopilot.pending(),
                         ranges_pending=autopilot.ranges_pending()).
    After a user edit changes a file's crop (set_manual_crop, paste_settings):
    on_crop_changed(file). After a folder change: apply_folder_change(project,
    old, new), then on_folder_changed(old, new). After files appear:
    on_files_added(names). Calling on_crop_changed after an applied crop
    result is harmless: on_job_event already did the same, and a brightness
    job already measuring the file's current box is not submitted again.

    pause()/resume() hold and release queued DETECTION_KINDS jobs on the GPU
    and CPU lanes (running jobs finish, ruling C6); metadata and thumbnails
    are never held. JobRunner.resume() clears every hold on a lane, so
    resume() also releases holds others placed there.
    """

    def __init__(self, runner: JobRunner, project_getter: Callable[[], Project]):
        self._runner = runner
        self._project_getter = project_getter
        self._outstanding: dict[str, int] = {}                    # key -> submissions without a terminal event
        self._identity: dict[str, tuple[str, str | None]] = {}    # key -> (kind, file), while outstanding
        self._brightness_requests: dict[str, _BrightnessRequest] = {}   # the newest submitted, per file
        self._ranges_waits: dict[str, _BrightnessRequest] = {}          # brightness waiting for ranges
        self._deferred_crops: dict[str, _CropRequest] = {}
        self._crop_chain: set[str] = set()        # files waiting for their auto-fill crop
        self._crop_active: str | None = None      # the file whose auto-fill crop is outstanding
        self._redetect: set[str] = set()          # files whose crop re-detect is followed by brightness
        self._scheduled: set[str] = set()         # files the folder schedule covers (metadata -> everything)
        self._thumbnail_times: dict[str, float] = {}
        self._ranges_done = False
        self._tier: dict[str, object] = {}        # full-tier file -> plateau | None | _UNRESOLVED
        self._waiting: set[str] = set()           # files waiting for the tier to close
        self._tier_closed = False
        self._folder_plateau: Plateau | None = None
        self._paused = False

    # --- triggers ---------------------------------------------------------------

    def on_open(self) -> None:
        """Schedule the folder: metadata and thumbnails always, detections when
        folder.autopilot_enabled."""
        self._schedule_folder(self._project())

    def on_files_added(self, names: list[str]) -> None:
        """Schedule the added files (metadata and thumbnails always) and, when
        autopilot_enabled, their detections and the ranges analysis again."""
        project = self._project()
        if project.folder.autopilot_enabled:
            self._schedule_ranges(project, force=True)      # first: brightness waits for it
        added = set(names)
        for name in project.files:
            if name in added:
                self._thumbnail_times.pop(name, None)      # a file that came back has no thumbnail
                self._schedule_file(project, name)
        self._settle_tier(project)

    def on_job_event(self, event: JobEvent) -> None:
        """Consume a terminal event of a job AutoPilot submitted and, when it is
        the newest for its key, submit what it unblocks. Call after applying
        the result (see the class docstring). Other events are ignored."""
        if event.type not in TERMINAL_EVENTS:
            return
        count = self._outstanding.get(event.key, 0)
        if count <= 0:
            return
        if count > 1:
            self._outstanding[event.key] = count - 1
            return
        del self._outstanding[event.key]
        kind, file = self._identity.pop(event.key)
        project = self._project()
        if kind == "metadata":
            self._after_metadata_event(project, file)
        elif kind == "crop":
            self._after_crop_event(project, file)
            self._advance_crop_chain(project)
        elif kind == "brightness":
            self._after_brightness_event(project, file, event)
        elif kind == "ranges":
            if event.type != "cancelled":
                self._ranges_done = True
            self._release_ranges_waits(project)
        self._settle_tier(project)

    def redetect(self, file: str) -> None:
        """User "re-detect": crop, then full brightness, for one file, whatever
        its values' sources (results still obey the apply rules). Metadata
        first when the duration is unknown. Respects the extraction toggles."""
        project = self._project()
        if file not in project.files or project.folder.labels_only:
            return
        self._redetect.add(file)
        self._crop_chain.discard(file)                     # this detection replaces the auto-fill one
        self._submit_crop(project, file, boost=True)
        self._settle_tier(project)

    def redetect_others_with_crop_hint(self, source_file: str) -> None:
        """Re-detect every other file's crop with the consensus seeded from
        `source_file`'s crop (ruling C3). Needs that crop and its media height."""
        project = self._project()
        source = project.files.get(source_file)
        if source is None or source.crop is None or project.folder.labels_only:
            return
        height = source.media.height
        if not height or height <= 0:
            return
        hint = (source.crop.y / height, source.crop.height / height)
        for name in project.files:
            if name != source_file:
                self._crop_chain.discard(name)
                self._submit_crop(project, name, hint=hint, boost=True)
        self._settle_tier(project)

    def redetect_others_with_brightness_hint(self, source_file: str) -> None:
        """Full brightness detection on every other file with a crop, checked
        against `source_file`'s value (ruling C3)."""
        project = self._project()
        source = project.files.get(source_file)
        if source is None or source.brightness is None or not project.folder.dialogue_enabled:
            return
        value = int(source.brightness.value)
        for name, entry in project.files.items():
            if name != source_file and entry.crop is not None:
                self._submit_brightness(project, name, _crop_box(entry), plateau=None, hint_value=value,
                                        priority=PRIORITY["brightness"] + REDETECT_BOOST, explicit=True)
        self._settle_tier(project)

    def on_crop_changed(self, file: str) -> None:
        """The file's crop changed (an edit, a paste, an applied result):
        measure brightness on the new box unless the value is the user's
        (MANUAL/IMPORTED) or is already measured, or being measured, on it."""
        project = self._project()
        self._remeasure_brightness(project, file, unrecorded=True)
        self._settle_tier(project)

    def on_folder_changed(self, old: FolderSettings, new: FolderSettings) -> None:
        """Submit what the new settings newly require: the folder schedule when
        auto-pilot was just turned on; crop and brightness for files missing
        them when extraction starts needing them. Call after
        core.jobs.apply.apply_folder_change(project, old, new). The full tier is
        settled whatever changed (it is left alone while dialogue is off)."""
        project = self._project()
        if new.autopilot_enabled and not old.autopilot_enabled:
            self._schedule_folder(project)
        elif new.autopilot_enabled and (
                (old.labels_only and not new.labels_only) or (new.dialogue_enabled and not old.dialogue_enabled)):
            self._schedule_newly_required(project, new)
        self._settle_tier(project)

    def _schedule_newly_required(self, project: Project, new: FolderSettings) -> None:
        for name, entry in project.files.items():
            self._scheduled.add(name)
            needs_crop = _crop_wanted(new, entry)
            if entry.media.duration <= 0:
                if needs_crop and not self._is_outstanding("metadata", name):
                    self._submit(MetadataJob(project.path, name), PRIORITY["metadata"])
            elif needs_crop and not self._is_outstanding("crop", name):
                self._crop_chain.add(name)
            if new.dialogue_enabled and _brightness_missing(entry):
                self._want_brightness(project, name)
        self._advance_crop_chain(project)

    # --- pause ------------------------------------------------------------------

    def pause(self) -> None:
        """Hold queued detection jobs (DETECTION_KINDS) on the GPU and CPU
        lanes; metadata and thumbnails keep running."""
        if self._paused:
            return
        self._paused = True
        for lane in (Lane.GPU, Lane.CPU):
            self._runner.pause(lane, only=is_detection_job)

    def resume(self) -> None:
        if not self._paused:
            return
        self._paused = False
        for lane in (Lane.GPU, Lane.CPU):
            self._runner.resume(lane)

    # --- queries ----------------------------------------------------------------

    def pending(self) -> dict[str, set[str]]:
        """file -> kinds with a queued or running submission, plus the crop and
        brightness detections AutoPilot will submit once those finish (behind
        metadata or a crop, or waiting for the full tier). For
        compute_review_state; a fresh dict."""
        project = self._project()
        folder = project.folder
        pending: dict[str, set[str]] = {}
        for kind, file in self._identity.values():
            if file is not None:
                pending.setdefault(file, set()).add(kind)
        for name, entry in project.files.items():
            kinds = pending.get(name, set())
            crop_pending = self._crop_pending(project, name, entry)
            if crop_pending:
                kinds.add("crop")
            if folder.dialogue_enabled and "brightness" not in kinds:
                after_crop = crop_pending and self._brightness_follows_crop(name, entry)
                for_tier = self._waits_for_tier(name, entry) and (entry.crop is not None or crop_pending)
                if after_crop or for_tier or self._waits_for_ranges(name, entry):
                    kinds.add("brightness")
            if kinds:
                pending[name] = kinds
        return pending

    def ranges_pending(self) -> bool:
        return self._outstanding.get(RANGES_KEY, 0) > 0

    def is_current(self, event: JobEvent) -> bool:
        """True when the event ends the last outstanding submission for its
        key (see "Superseded jobs"): False while another one, normally a newer
        one, is outstanding. Call before on_job_event for the same event. True
        for keys AutoPilot never submitted."""
        return self._outstanding.get(event.key, 0) <= 1

    # --- scheduling -------------------------------------------------------------

    def _project(self) -> Project:
        return self._project_getter()

    def _is_outstanding(self, kind: str, file: str) -> bool:
        return self._outstanding.get(f"{kind}:{file}", 0) > 0

    def _submit(self, job, priority: int) -> None:
        job.priority = int(priority)
        key = job.key
        self._outstanding[key] = self._outstanding.get(key, 0) + 1
        self._identity[key] = (job.kind, job.file)
        try:
            self._runner.submit(job)
        except BaseException:
            remaining = self._outstanding[key] - 1
            if remaining:
                self._outstanding[key] = remaining
            else:
                del self._outstanding[key]
                del self._identity[key]
            raise

    def _schedule_folder(self, project: Project) -> None:
        if project.folder.autopilot_enabled:
            self._schedule_ranges(project, force=False)     # first: brightness waits for it
        for name in list(project.files):
            self._schedule_file(project, name)
        self._settle_tier(project)

    def _schedule_file(self, project: Project, name: str) -> None:
        self._scheduled.add(name)
        entry = project.files[name]
        folder = project.folder
        if entry.media.duration <= 0:
            if not self._is_outstanding("metadata", name):
                self._submit(MetadataJob(project.path, name), PRIORITY["metadata"])
            if folder.autopilot_enabled and folder.dialogue_enabled and _brightness_missing(entry):
                self._want_brightness(project, name)     # takes its place in the tier, in name order
            return
        self._after_metadata(project, name)

    def _after_metadata(self, project: Project, name: str, *, crop: bool = True) -> None:
        """A scheduled file's duration is known: its thumbnail, and, when
        autopilot_enabled, its detections."""
        entry = project.files[name]
        folder = project.folder
        self._submit_thumbnail(project, name)
        if not self._detects_automatically(folder, name):
            return
        if crop and _crop_wanted(folder, entry) and not self._is_outstanding("crop", name):
            self._crop_chain.add(name)
            self._advance_crop_chain(project)
        if "audio" not in entry.evidence and not self._is_outstanding("audio_profile", name):
            self._submit(AudioProfileJob(project.path, name, entry.media.duration), PRIORITY["audio_profile"])
        if folder.dialogue_enabled and _brightness_missing(entry):
            self._want_brightness(project, name)

    def _schedule_ranges(self, project: Project, *, force: bool) -> None:
        names = list(project.files)
        if len(names) < 2:
            return
        if not any(entry.time_ranges is None or entry.time_ranges.source in DETECTION_SOURCES
                   for entry in project.files.values()):
            return
        if not force and (self._ranges_done or self.ranges_pending()):
            return
        self._submit(RangesJob(project.path, names, project.folder), PRIORITY["ranges"])

    def _submit_crop(self, project: Project, name: str, *, hint: tuple[float, float] | None = None,
                     boost: bool = False) -> None:
        """A crop detection, or, while the duration is unknown, metadata first
        and the crop once it arrives."""
        entry = project.files[name]
        bonus = REDETECT_BOOST if boost else 0
        if entry.media.duration <= 0:
            self._deferred_crops[name] = _CropRequest(hint, boost)
            if not self._is_outstanding("metadata", name):
                self._submit(MetadataJob(project.path, name), PRIORITY["metadata"] + bonus)
            return
        consensus = [] if hint is not None else crop_consensus(project, exclude=name)
        self._submit(CropJob(project.path, name, entry.media.duration, consensus, project.folder, hint=hint),
                     PRIORITY["crop"] + bonus)

    def _advance_crop_chain(self, project: Project) -> None:
        """Submit the next auto-fill crop, in name order, unless one is still
        outstanding. Files that no longer need one (a crop arrived, another
        crop detection is outstanding, the folder went labels-only, removed)
        leave the chain."""
        if self._crop_active is not None and self._is_outstanding("crop", self._crop_active):
            return
        self._crop_active = None
        for name in [n for n in project.files if n in self._crop_chain]:
            self._crop_chain.discard(name)
            entry = project.files[name]
            if (_crop_wanted(project.folder, entry) and entry.media.duration > 0
                    and not self._is_outstanding("crop", name)):
                self._submit_crop(project, name)
                self._crop_active = name
                return
        self._crop_chain.clear()

    def _submit_thumbnail(self, project: Project, name: str) -> None:
        entry = project.files[name]
        if entry.sample_time is not None:
            time = float(entry.sample_time)
        elif entry.media.duration > 0:
            time = THUMBNAIL_FRACTION * entry.media.duration
        else:
            return
        if self._thumbnail_times.get(name) == time:
            return
        self._submit(ThumbnailJob(project.path, name, time), PRIORITY["thumbnail"])
        self._thumbnail_times[name] = time

    def _submit_brightness(self, project: Project, name: str, box: Box, *, plateau: Plateau | None,
                           hint_value: int | None, priority: int, explicit: bool = False) -> None:
        """Every brightness job goes through here: submitted now, or, while a
        ranges analysis is pending, kept (replacing an earlier wait) until it
        ends, so the job samples the file's keep ranges."""
        request = _BrightnessRequest(box, plateau, hint_value, priority, explicit)
        if self.ranges_pending():
            self._ranges_waits[name] = request
            return
        self._ranges_waits.pop(name, None)
        entry = project.files[name]
        self._submit(BrightnessJob(project.path, name, box, _time_ranges(entry), project.folder,
                                   folder_plateau=plateau, hint_value=hint_value), priority)
        self._brightness_requests[name] = request

    def _release_ranges_waits(self, project: Project) -> None:
        """The ranges analysis ended: submit the brightness requests that waited,
        in name order, on the file's crop and ranges as they are now."""
        if self.ranges_pending():
            return
        for name in [n for n in project.files if n in self._ranges_waits]:
            entry = project.files[name]
            if not project.folder.dialogue_enabled or not self._waits_for_ranges(name, entry):
                continue
            request = self._ranges_waits.pop(name)
            self._submit_brightness(project, name, _crop_box(entry), plateau=request.plateau,
                                    hint_value=request.hint_value, priority=request.priority,
                                    explicit=request.explicit)
        self._ranges_waits.clear()

    def _brightness_request(self, name: str) -> _BrightnessRequest | None:
        """The file's newest brightness request: waiting for ranges, else outstanding."""
        waiting = self._ranges_waits.get(name)
        if waiting is not None:
            return waiting
        return self._brightness_requests.get(name) if self._is_outstanding("brightness", name) else None

    def _want_brightness(self, project: Project, name: str) -> None:
        """The file needs its brightness measured: place it in the two tiers,
        and submit when its crop is known (the caller checked the need)."""
        entry = project.files.get(name)
        folder = project.folder
        if entry is None or not folder.dialogue_enabled:
            return
        plateau = None
        if name not in self._tier:
            if not self._tier_closed and folder.brightness_full_detect_files > 0:
                if len(self._tier) >= folder.brightness_full_detect_files:
                    self._waiting.add(name)
                    return
                self._tier[name] = _UNRESOLVED
            elif self._tier_closed:
                plateau = self._folder_plateau
        if entry.crop is None:
            return
        box = _crop_box(entry)
        request = self._brightness_request(name)
        if request is not None and request.box == box:
            return
        self._submit_brightness(project, name, box, plateau=plateau, hint_value=None,
                                priority=PRIORITY["brightness"])

    def _remeasure_brightness(self, project: Project, name: str, *, unrecorded: bool) -> None:
        """After a crop change: brightness on the new box when the value is
        missing, or detected/hinted and measured on another box (`unrecorded`:
        also when no measurement box is recorded). An outstanding request on
        another box is resubmitted as it was (tier, hint, priority)."""
        entry = project.files.get(name)
        if entry is None or not project.folder.dialogue_enabled or entry.crop is None:
            return
        if not (_brightness_unmeasured(entry) if unrecorded else _brightness_missing(entry)):
            return
        box = _crop_box(entry)
        request = self._brightness_request(name)
        if request is not None:
            if request.box != box:
                self._submit_brightness(project, name, box, plateau=request.plateau, hint_value=request.hint_value,
                                        priority=request.priority, explicit=request.explicit)
            return
        self._want_brightness(project, name)

    # --- job events -------------------------------------------------------------

    def _after_metadata_event(self, project: Project, name: str) -> None:
        request = self._deferred_crops.pop(name, None)
        entry = project.files.get(name)
        if entry is None or entry.media.duration <= 0:
            self._redetect.discard(name)
            return
        if name in self._scheduled:
            self._after_metadata(project, name, crop=request is None)
        if request is not None:
            self._submit_crop(project, name, hint=request.hint, boost=request.boost)

    def _after_crop_event(self, project: Project, name: str) -> None:
        entry = project.files.get(name)
        if entry is None:
            self._redetect.discard(name)
            return
        if (project.folder.dialogue_enabled and entry.crop is not None
                and self._brightness_follows_crop(name, entry)):
            if name in self._redetect:
                self._submit_brightness(project, name, _crop_box(entry), plateau=None, hint_value=None,
                                        priority=PRIORITY["brightness"] + REDETECT_BOOST, explicit=True)
            else:
                self._remeasure_brightness(project, name, unrecorded=False)
        self._redetect.discard(name)
        if (name in self._thumbnail_times and entry.sample_time is not None
                and float(entry.sample_time) != self._thumbnail_times[name]):
            self._submit_thumbnail(project, name)

    def _after_brightness_event(self, project: Project, name: str, event: JobEvent) -> None:
        request = self._brightness_requests.pop(name, None)
        entry = project.files.get(name)
        result = event.result
        if entry is None or event.type != "finished" or not isinstance(result, BrightnessJobResult):
            return
        if FLAG_ESCALATE in (result.result.flagged or "").split("+"):
            brightness = entry.brightness
            if (project.folder.dialogue_enabled and entry.crop is not None
                    and (brightness is None or brightness.source in DETECTION_SOURCES)):
                priority = request.priority if request is not None else PRIORITY["brightness"]
                self._submit_brightness(project, name, _crop_box(entry), plateau=None, hint_value=None,
                                        priority=priority)
            return
        if (self._tier.get(name) is _UNRESOLVED and (request is None or request.plateau is None)
                and entry.crop is not None and result.crop_box is not None
                and _box(result.crop_box) == _crop_box(entry)):
            plateau = result.result.plateau
            self._tier[name] = None if plateau is None else (int(plateau[0]), int(plateau[1]))

    # --- conditions shared by the scheduler and pending() -----------------------

    def _detects_automatically(self, folder: FolderSettings, name: str) -> bool:
        """The folder schedule covers the file and auto-pilot is on: its missing
        values are detected once its duration is known."""
        return name in self._scheduled and folder.autopilot_enabled

    def _brightness_follows_crop(self, name: str, entry: FileEntry) -> bool:
        """A crop event for the file (with a crop known) is followed by a
        brightness job: its re-detect's step, or a value missing or stale."""
        return name in self._redetect or _brightness_missing(entry)

    def _waits_for_ranges(self, name: str, entry: FileEntry) -> bool:
        """A brightness request for the file waits for the ranges analysis and
        will be submitted when it ends: an explicit one always, an auto-pilot
        one while the value still needs measuring."""
        request = self._ranges_waits.get(name)
        return (request is not None and entry.crop is not None
                and (request.explicit or _brightness_unmeasured(entry)))

    def _waits_for_tier(self, name: str, entry: FileEntry) -> bool:
        """The file waits for the full tier to close and still needs measuring."""
        return name in self._waiting and _brightness_unmeasured(entry)

    def _crop_pending(self, project: Project, name: str, entry: FileEntry) -> bool:
        """A crop detection is outstanding, waits in the auto-fill chain, or
        will follow outstanding metadata."""
        folder = project.folder
        if self._is_outstanding("crop", name):
            return True
        if name in self._crop_chain and _crop_wanted(folder, entry):
            return True
        if not self._is_outstanding("metadata", name):
            return False
        return name in self._deferred_crops or (
            self._detects_automatically(folder, name) and _crop_wanted(folder, entry))

    # --- the full tier ----------------------------------------------------------

    def _member_alive(self, project: Project, name: str) -> bool:
        """A tier file that may still report a full-run plateau."""
        entry = project.files.get(name)
        if entry is None:
            return False
        if self._brightness_request(name) is not None:        # submitted, or waiting for ranges
            return True
        if not _brightness_unmeasured(entry):
            return False
        return self._crop_pending(project, name, entry) and (entry.crop is None or name in self._redetect)

    def _settle_tier(self, project: Project) -> None:
        """Drop tier files that can no longer be measured, fill their places
        from the waiting files, and close the tier once every file in it has
        reported: then the folder plateau is fixed and waiting files are
        released. Nothing happens while dialogue is off."""
        if self._tier_closed or not project.folder.dialogue_enabled:
            return
        limit = project.folder.brightness_full_detect_files
        self._waiting.intersection_update(project.files)
        while True:
            for name in [n for n, plateau in self._tier.items()
                         if plateau is _UNRESOLVED and not self._member_alive(project, n)]:
                del self._tier[name]
            promoted = False
            for name in [n for n in project.files if n in self._waiting]:
                if len(self._tier) >= limit:
                    break
                entry = project.files[name]
                wanted = self._waits_for_tier(name, entry)
                self._waiting.discard(name)
                if wanted:
                    self._want_brightness(project, name)
                    promoted = True
            if not promoted:
                break
        if any(plateau is _UNRESOLVED for plateau in self._tier.values()):
            return
        if not self._tier and not self._waiting:       # nothing has needed brightness yet
            return
        # An empty tier with files still waiting (the limit fell to 0) closes without a plateau.
        self._tier_closed = True
        self._folder_plateau = intersect_plateaus(self._tier.values())
        for name in [n for n in project.files if n in self._waiting]:
            entry = project.files[name]
            if self._waits_for_tier(name, entry):
                self._waiting.discard(name)
                self._want_brightness(project, name)
        self._waiting.clear()
