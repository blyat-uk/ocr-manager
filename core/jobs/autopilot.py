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
       HINT). on_files_added resubmits it;
    5. the Brightness tab's gallery lines (see "Gallery lines");
    6. an OCR confirmation of every file whose brightness was measured but
       doubted, one job at a time for the whole folder (see "Confirming a
       doubted brightness");
    7. the view cache of every file whose own detections have settled, one
       job at a time for the whole folder (see "Warming the view cache").
    Priorities (within a lane): metadata 5 > crop 3 > brightness 2 >
    thumbnail, audio, lines 1 > ranges, confirm 0 > warming -1
    (WARM_PRIORITY, under every detection there is); re-detects get
    +REDETECT_BOOST. On the GPU lane that leaves confirm under crop,
    brightness and lines -- the lines last of those on purpose, because a
    confirm probes a strip the gallery draw chose, so the draw goes first.
    The lines of the file the Brightness tab shows (boost_lines,
    shuffle_lines) get LINES_BOOST (20), above every other priority given
    here (at most 15) and below the proof's PROOF_PRIORITY (100).

Gallery lines (docs/superpowers/specs/2026-09-18-brightness-gallery-design.md)
    Random subtitle lines of a file, grabbed on its crop (LinesJob), for the
    Brightness tab. View evidence, not a value: a "lines" job never holds a
    review state, a badge or a chip count. Lines are wanted (_lines_wanted)
    while dialogue is extracted, for a file not skipped with a crop and a
    known duration whose evidence["lines"] is missing or was grabbed on
    another crop. An auto-pilot draw is submitted once the folder's ranges
    are settled (ranges_pending() false) and the file's audio profile is not
    outstanding (the draw samples the keep ranges, weighted by speech);
    until then the file is pending "lines". It is re-evaluated after
    metadata, crop, ranges and audio-profile events, on_open,
    on_files_added, on_crop_changed (the old lines show another region: a
    new draw supersedes one outstanding on the old crop) and
    on_folder_changed. Nothing is resubmitted on a lines event: a draw that
    failed is not drawn again on that crop, automatically or by
    boost_lines, and a cancelled one waits for the next trigger.
    boost_lines is the tab asking for the file it shows: submitted at once
    at LINES_BOOST, and idempotent (see there). shuffle_lines is the user's
    new draw away from the lines shown. Each draw takes a new seed from
    `seed_source`.

Confirming a doubted brightness (core.detect.confirm, ConfirmJob)
    A brightness the detector measured but doubts flags the file: it lands in
    FLAGGED and the queue badges it "check brightness". Most of those doubts
    can be answered without the user -- mask one strip the file is known to
    hold a subtitle on and ask the OCR engine to read it at the folder's own
    conf_threshold, stepping the threshold down until it reads or the ladder
    hits the hard stop. A rung that reads is a threshold a run would read
    that line at, so the doubt is retired; a ladder that passes nowhere
    leaves the file exactly as the detector left it, for the user. On the
    folders this was asked for (2026-09-20) that is the difference between
    reviewing ~180 files by hand and reviewing the handful the engine could
    not read.

    A file wants a confirm (_confirm_wanted) when all of:
      - folder.autopilot_enabled. This stage MOVES a value, so it is gated
        where every other detection AutoPilot starts on its own is gated.
        (Warming is not gated: it detects nothing and only fills a cache the
        tabs would fill themselves. This writes to the model.)
      - folder.dialogue_enabled: brightness is a required field only then,
        so only then is there a doubt worth retiring;
      - the file is not skipped, and has both a crop and a brightness;
      - that brightness is DETECTED or HINT (DETECTION_SOURCES): a MANUAL or
        IMPORTED value is the user's, and detection never overwrites it
        (ruling C2) -- nor does this;
      - it is not stale (brightness_is_stale). A value measured against a
        crop the file no longer has must be RE-MEASURED, not confirmed:
        confirming it would bless a reading of another region;
      - the flag actually blocks (apply.counts_flagged): an informational
        reason such as no-clean-threshold is how the result was reached, not
        a doubt about it, and there is nothing there to retire;
      - the reasons do not name core.detect.brightness.NOTHING_MEASURED_FLAGS
        (needs-crop, ranges-empty?, no-text, escalate, cancelled). Those
        results hold no reading of the file at all -- the stored value is a
        placeholder -- and there is no reading to confirm;
      - the reasons do not name apply.SOURCE_INDEPENDENT_FLAGS["brightness"]
        (brightness-other-crop). Those describe the stored value rather than
        a detection of it, they count whoever set it, and only the user
        answers them;
      - the crop is not blocking too (apply.counts_flagged(entry, "crop") is
        False). If the user has to open the file for its crop anyway,
        clearing the brightness half of that row saves them no visit at all,
        and costs a GPU job that a file nobody has to open could have had;
      - pending() reports nothing for the file: no detection of its own is
        outstanding or foreseen. Its values are still moving, and so are its
        gallery lines -- which is where the probe strip comes from. The same
        rule as _warm_wanted, for the same reason;
      - its evidence names a strip to probe (detect_jobs.probe_time_for);
      - it has not been probed on exactly this question already
        (confirm.matches over evidence["brightness"]["confirm"]: the same
        starting value, conf_threshold, crop box and strip).

    The chain and the memo
        One confirm is outstanding for the whole folder, in name order, the
        next submitted when the previous one's terminal event arrives,
        whatever that event says -- the auto-fill crops' rule and the warm
        chain's, for the warm chain's reason: a folder of hundreds of flagged
        files must not queue hundreds of GPU jobs, and pending() must stay
        worth reading. The chain is re-evaluated wherever a file's flags,
        crop, evidence or the folder's settings can have moved: on_open,
        on_files_added, on_files_removed, on_crop_changed, on_folder_changed
        and after every terminal event.
        The evidence record is written by core.jobs.apply.apply_confirm when
        a result arrives, but a job that FAILED with an exception, or was
        cancelled, leaves no record at all, and its file would be offered
        again for ever. So the probe each job was submitted for is remembered
        in session (self._confirmed), and a file is offered again only when
        that probe changes -- another crop box, another strip, another stored
        value, another conf_threshold. The memo goes the way the warm memo
        goes: dropped when the file leaves (on_files_removed) and when it
        comes back (on_files_added), whose evidence is not the old entry's,
        and pruned to the project's files when the chain advances.

    What a confirm is not
        It holds no value of its own, no review state and no badge, and
        pending() never reports "confirm": a file keeps its honest "check
        brightness" badge until a confirm actually clears it, rather than
        churning 180 rows through a transient "waiting" that tells the user
        nothing. is_detection_job is False for the same reason it is False
        for warming -- a confirm can move a value, but it can only ever
        REMOVE a reason to stop, so a folder with confirms outstanding is a
        folder that has been detected and the top bar must not claim
        otherwise. It is held with detection all the same (HELD_KINDS,
        is_held_job, ruling C6): it is GPU work that leases an OCR engine,
        and the run the user is waiting for must never queue behind it.
        A confirm can only ever LOWER a file's value or drop its flag. It
        never raises a value, never adds a doubt and never blocks anything;
        the worst it can do is leave the file exactly as it found it.

Warming the view cache (core.jobs.view_cache, core.jobs.view_jobs)
    A WarmJob decodes the frames and strips a file's Crop and Brightness tabs
    will draw into <project>/.ocr-cache/view/ and throws the pixels away: the
    point is the disk, so that opening the file later costs a read and not a
    decode. Warming is not detection. It holds no value, no review state and
    no badge, and pending() never reports "warm", or the top bar would claim
    the folder is still being detected; it is held with detection all the
    same (HELD_KINDS, is_held_job), because it is CPU decoding and the run
    the user is waiting for must never queue behind it. A file is warmed once
    its own detections have settled: it is not skipped, pending() reports
    nothing for it, and view_cache.wanted(entry) names pixels to hold (a file
    with no evidence yet is not worth a job). autopilot_enabled does not gate
    it: warming detects nothing, and the tabs draw the same pixels whether or
    not auto-pilot fills the folder's values. Like the auto-fill crops, warm
    jobs run as a chain: one outstanding for the whole folder, the next
    submitted when the previous one's terminal event arrives, whatever that
    event says. A folder of hundreds of files must not queue hundreds of jobs,
    and pending() must stay worth reading. The chain is re-evaluated wherever
    a file's evidence can have moved: on_open, on_files_added,
    on_files_removed, on_crop_changed, on_folder_changed and after every
    terminal event. Warming a file that is already warm is cheap (the job
    reads the directory and decodes nothing), but resubmitting it forever
    would not be: each file's Wanted is remembered when its job is submitted,
    and the file is warmed again only when that Wanted changes -- new crop
    samples, new tiles, new lines, another crop box. A warm job that failed or
    was cancelled is not retried on the same Wanted (but see boost_warm). The
    memo is per file, and goes the way the file's thumbnail time goes: dropped
    when the file leaves (on_files_removed) and when it comes back
    (on_files_added), whose pixels are not the old entry's.

    Warming follows the cursor (boost_warm)
        Name order is the one order the user is NOT working in. A filter such
        as "Needs you" picks a scattered handful out of a big folder, so a
        chain plodding from the first episode warms nothing the user opens:
        measured on a 432-file 4K folder, the frontier sat around episode 100
        while every flagged file the user actually reviewed -- all 209 of them
        -- decoded its strips live, 2-4 s each time. The window therefore
        hands the queue's own visible order to boost_warm on every selection
        (ProjectController.set_view_file), and the chain takes those files
        first and the rest of the folder afterwards. A warm job running for a
        file nobody is heading towards is cancelled to free the worker; that
        one file, and only that one, drops its memo so it is warmed later --
        every other cancellation keeps its memo, which is what bounds
        resubmission.

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

Hint re-detects (ruling C3)
    redetect_others_with_crop_hint / _brightness_hint submit jobs for exactly
    hint_targets(source, what): the other files, not skipped, whose target
    value detection may still write (None, DETECTED or HINT), with that
    detection enabled by the extraction toggles (brightness also needs a
    crop). MANUAL and IMPORTED values are never re-measured for a hint.

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
    later change of ranges does not re-measure brightness. A waiting request
    is the newest for its file: it supersedes brightness jobs already
    outstanding for it (see "Superseded jobs").

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
    is current and drives what follows. A brightness request waiting for the
    ranges counts as a newer outstanding submission: while one waits, older
    brightness events for that file are not current. (A lines draw waiting
    for the ranges or the audio profile does not: an older draw that ends
    meanwhile is current, and apply_lines drops it when the crop moved.) A
    non-current event releases its own count and changes nothing else (no
    escalation, no resubmission, no tier or chain change). This does not
    depend on "queued" events having been drained. pending() reports a kind
    for a file while any submission for that key is outstanding. Every job
    of an AUTOPILOT_KINDS kind must be submitted through AutoPilot; events
    for keys it never submitted (proof, run) are current and otherwise
    ignored. "started" events only record which job of a key runs (see
    _lines_running).

autopilot_enabled
    Gates the detections AutoPilot starts on its own (crop, brightness,
    ranges, audio profile, gallery lines, brightness confirms) where they are
    first wanted:
    on_open, on_files_added, after a scheduled file's metadata, and
    on_folder_changed (turning it on schedules the folder's detections).
    Metadata and thumbnails always run on on_open/on_files_added: the queue
    needs durations and thumbnails. Explicit requests (redetect, the hint
    re-detects, boost_lines, shuffle_lines), on_crop_changed and the
    continuation of work already started (the crop chain, brightness after a
    crop, escalation, the tier, a boosted or shuffled draw following a crop
    change) run regardless. Auto-pilot lines draws are gated everywhere,
    on_crop_changed included: with auto-pilot off, the Brightness tab's
    boost draws the lines of the file it shows.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from core.detect import confirm
from core.detect.brightness import FLAG_ESCALATE, NOTHING_MEASURED_FLAGS
from core.jobs import apply
from core.jobs.apply import DETECTION_SOURCES, brightness_is_stale
from core.jobs.detect_jobs import (
    AudioProfileJob,
    BrightnessJob,
    BrightnessJobResult,
    ConfirmJob,
    CropJob,
    LinesJob,
    MetadataJob,
    RangesJob,
    ThumbnailJob,
    probe_time_for,
)
from core.jobs.runner import Job, JobEvent, JobRunner, Lane
from core.jobs.view_cache import Wanted, wanted
from core.jobs.view_jobs import WARM_PRIORITY, WarmJob
from core.project.model import FileEntry, FolderSettings, Project, Source

AUTOPILOT_KINDS = frozenset({"metadata", "thumbnail", "crop", "brightness", "ranges", "audio_profile", "lines",
                             "confirm", "warm"})
DETECTION_KINDS = frozenset({"crop", "brightness", "ranges", "audio_profile", "lines"})
# What pause() holds. Neither warming the view cache nor confirming a doubted
# brightness is detection -- neither holds a review state, a badge or a
# pending chip, and `is_detection_job` must go on answering False for both, or
# the top bar would claim the folder is still being detected -- but both are
# work the user's run must never wait behind (warming decodes on the CPU, a
# confirm leases a GPU OCR engine), so both are held by the same lever
# (ruling C6).
HELD_KINDS = DETECTION_KINDS | {"confirm", "warm"}
TERMINAL_EVENTS = frozenset({"finished", "failed", "cancelled"})
# Kinds pending() never reports: they move no value the queue is waiting on
# (see "Warming the view cache" and "Confirming a doubted brightness").
UNREPORTED_KINDS = frozenset({"confirm", "warm"})
PRIORITY = {"metadata": 5, "crop": 3, "brightness": 2, "thumbnail": 1, "audio_profile": 1, "lines": 1, "ranges": 0,
            "confirm": 0}
REDETECT_BOOST = 10
# The gallery lines of the file the Brightness tab shows (boost_lines, shuffle_lines): the user is
# looking at an empty gallery. Above every other priority AutoPilot gives (at most 15, a boosted
# re-detect's metadata; on the GPU lane 13, a boosted crop), below the proof's PROOF_PRIORITY (100).
LINES_BOOST = 20
THUMBNAIL_FRACTION = 0.4          # thumbnail time without a sample time, as a fraction of the duration
RANGES_KEY = "ranges:*"

Box = tuple[int, int, int, int]
Plateau = tuple[int, int]


def is_autopilot_job(job: Job) -> bool:
    """A job of a kind AutoPilot owns (and must be submitted through it)."""
    return getattr(job, "kind", None) in AUTOPILOT_KINDS


def is_detection_job(job: Job) -> bool:
    """A detection job: one whose result can move a value or a review state.
    Metadata and thumbnails are not, and neither is warming (see HELD_KINDS).

    Confirming a doubted brightness is the deliberate exception: a confirm
    CAN move a value, but it only ever REMOVES a reason to stop -- it never
    blocks a review state, never raises a value and never adds a doubt -- so
    a folder with confirms outstanding is a folder that has been detected,
    and the top bar must not claim otherwise (the same reasoning as "warm";
    see "Confirming a doubted brightness")."""
    return getattr(job, "kind", None) in DETECTION_KINDS


def is_held_job(job: Job) -> bool:
    """The `only` predicate pause() holds: detection plus view-cache warming
    and brightness confirms (HELD_KINDS). Metadata and thumbnails are never
    held (ruling C6)."""
    return getattr(job, "kind", None) in HELD_KINDS


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


def _crop_hint(entry: FileEntry) -> tuple[float, float] | None:
    """(y_frac, h_frac) of the file's crop, or None without a crop or a known media height."""
    height = entry.media.height
    if entry.crop is None or not height or height <= 0:
        return None
    return (entry.crop.y / height, entry.crop.height / height)


def _speech(entry: FileEntry) -> list | None:
    """The file's speech spans (evidence["audio"]["speech"]), None without an audio profile."""
    return (entry.evidence.get("audio") or {}).get("speech")


def _lines_wanted(folder: FolderSettings, entry: FileEntry) -> bool:
    """The file's gallery lines are to be drawn: dialogue is extracted (the
    Brightness tab has a threshold to show), the file is not skipped, has a
    crop and a known duration, and its evidence["lines"] is missing or was
    grabbed on another crop (a record without its crop box counts as
    another)."""
    if not folder.dialogue_enabled or entry.skipped or entry.crop is None or entry.media.duration <= 0:
        return False
    drawn_on = (entry.evidence.get("lines") or {}).get("crop_box")
    return drawn_on is None or _box(drawn_on) != _crop_box(entry)


_SEEDS = random.SystemRandom()


def default_seed() -> int:
    """A fresh seed for a lines draw (AutoPilot's default seed_source)."""
    return _SEEDS.randrange(1 << 31)


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
class _LinesRequest:
    """The newest lines job submitted for a file."""
    box: Box
    priority: int
    explicit: bool                 # boost_lines or shuffle_lines: the user is looking at the gallery


@dataclass(frozen=True)
class _ConfirmProbe:
    """The question one ConfirmJob asks: this strip, cut with this crop box,
    read from this stored value down, judged at this confidence threshold.

    It is what the chain's in-session memo holds, and it carries every field
    confirm.matches compares, so "has this file been asked this already?"
    has one answer whether it is read off the file's evidence or off the
    memo."""
    box: Box
    probe_time: float
    start_value: int
    conf_threshold: int


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
    A "started" event goes to on_job_event too (it only records that the job
    runs, so boost_lines leaves a running draw alone; without it a boost
    replaces a draw that is already running).
    After a user edit changes a file's crop (set_manual_crop, paste_settings):
    on_crop_changed(file). After a folder change: apply_folder_change(project,
    old, new), then on_folder_changed(old, new). After files appear:
    on_files_added(names). After reconcile_files removed entries: cancel the
    removed files' jobs (runner.cancel_where on job.file), then
    on_files_removed(names); their cancelled events are drained like any
    other. Calling on_crop_changed after an applied crop
    result is harmless: on_job_event already did the same, and a brightness
    job already measuring the file's current box is not submitted again.

    pause()/resume() hold and release queued HELD_KINDS jobs on the GPU and
    CPU lanes (running jobs finish, ruling C6), boosted lines draws,
    brightness confirms and view-cache warming included; metadata and
    thumbnails are never held.
    JobRunner.resume() clears every hold on a lane, so resume() also releases
    holds others placed there.
    """

    def __init__(self, runner: JobRunner, project_getter: Callable[[], Project], *,
                 seed_source: Callable[[], int] | None = None):
        self._runner = runner
        self._project_getter = project_getter
        self._seed_source = default_seed if seed_source is None else seed_source   # one seed per lines draw
        self._outstanding: dict[str, int] = {}                    # key -> submissions without a terminal event
        self._identity: dict[str, tuple[str, str | None]] = {}    # key -> (kind, file), while outstanding
        self._started: dict[str, int] = {}                        # key -> job_id of its job that runs
        self._lines_requests: dict[str, _LinesRequest] = {}       # the newest submitted, per file
        self._lines_failed: dict[str, Box] = {}                   # file -> the crop its last draw failed on
        self._brightness_requests: dict[str, _BrightnessRequest] = {}   # the newest submitted, per file
        self._ranges_waits: dict[str, _BrightnessRequest] = {}          # brightness waiting for ranges
        self._deferred_crops: dict[str, _CropRequest] = {}
        self._crop_chain: set[str] = set()        # files waiting for their auto-fill crop
        self._crop_active: str | None = None      # the file whose auto-fill crop is outstanding
        self._redetect: set[str] = set()          # files whose crop re-detect is followed by brightness
        self._redetect_superseded: dict[str, int] = {}   # re-detected file -> highest job_id of a
                                                         # non-current crop event since the re-detect
        self._scheduled: set[str] = set()         # files the folder schedule covers (metadata -> everything)
        self._readded: set[str] = set()           # files added back while jobs for their old entry were outstanding
        self._thumbnail_times: dict[str, float] = {}
        self._ranges_done = False
        self._ranges_files: tuple[str, ...] = ()   # the files of the newest ranges analysis submitted
        self._tier: dict[str, object] = {}        # full-tier file -> plateau | None | _UNRESOLVED
        self._waiting: set[str] = set()           # files waiting for the tier to close
        self._tier_closed = False
        self._folder_plateau: Plateau | None = None
        self._confirm_active: str | None = None   # the file whose brightness confirm is outstanding
        self._confirmed: dict[str, _ConfirmProbe] = {}   # file -> the probe its last confirm was submitted for
        self._warm_active: str | None = None      # the file whose view-cache warm job is outstanding
        self._warmed: dict[str, Wanted] = {}      # file -> the Wanted its last warm job was submitted for
        self._warm_ahead: tuple[str, ...] = ()    # boost_warm: the files the user is heading towards, in order
        self._warm_displaced: str | None = None   # the warm job boost_warm cancelled: retry it, unlike other cancels
        self._paused = False

    # --- triggers ---------------------------------------------------------------

    def on_open(self) -> None:
        """Schedule the folder: metadata and thumbnails always, detections when
        folder.autopilot_enabled, an OCR confirmation of the files flagged on
        a doubted brightness, and warm the view cache of the files that need
        nothing detected."""
        project = self._project()
        self._schedule_folder(project)
        self._advance_confirm_chain(project)
        self._advance_warm_chain(project)

    def on_files_added(self, names: list[str]) -> None:
        """Schedule the added files (metadata and thumbnails always) and, when
        autopilot_enabled, their detections and the ranges analysis again. A
        file added back while jobs of its removed entry are still outstanding
        is scheduled again, as here (without another ranges analysis), when
        the last outstanding job for it ends; until then it is pending "crop"
        when it needs one."""
        project = self._project()
        wanted = set(names)
        added = [name for name in project.files if name in wanted]
        for name in added:
            if self._file_outstanding(name):
                # Jobs of the removed entry still count, so the checks below
                # would skip what they cover. Schedule it again once they end.
                self._readded.add(name)
            self._thumbnail_times.pop(name, None)          # a file that came back has no thumbnail
            self._warmed.pop(name, None)                   # nor the view cache of the entry that left
            self._confirmed.pop(name, None)                # nor the evidence its confirm memo speaks for
            self._scan_metadata(project, name)
        if project.folder.autopilot_enabled:
            self._schedule_ranges(project, force=True)      # before any brightness: it waits for ranges
        for name in added:
            self._schedule_file(project, name)
        self._settle_tier(project)
        self._advance_confirm_chain(project)
        self._advance_warm_chain(project)

    def on_files_removed(self, names: list[str]) -> None:
        """Forget files that left the project: they leave the crop chain, the
        full tier and the tier's waiting files; their deferred crops,
        brightness requests (waiting for the ranges or submitted), lines
        requests, re-detect intents, thumbnail times and their warm and
        confirm memos are dropped. Their outstanding submissions stay counted until their
        terminal events arrive (the owner cancels those jobs), but pending()
        no longer reports them.
        When the outstanding ranges analysis covers a removed file, a fresh
        analysis of the remaining files supersedes it if ranges are still
        wanted (>= 2 files, some file's ranges open to detection); otherwise
        no new analysis is submitted and ranges are not needed. The chain
        and the tier then move on (a waiting file takes a removed tier
        file's place)."""
        project = self._project()
        removed = [name for name in names if name not in project.files]
        if not removed:
            return
        for name in removed:
            self._readded.discard(name)
            self._crop_chain.discard(name)
            self._deferred_crops.pop(name, None)
            self._drop_redetect(name)
            self._scheduled.discard(name)
            self._thumbnail_times.pop(name, None)
            self._warmed.pop(name, None)
            self._confirmed.pop(name, None)
            self._brightness_requests.pop(name, None)
            self._ranges_waits.pop(name, None)
            self._lines_requests.pop(name, None)
            self._lines_failed.pop(name, None)
            self._tier.pop(name, None)
            self._waiting.discard(name)
        if self._crop_active in removed:
            self._crop_active = None
        if self._confirm_active in removed:
            self._confirm_active = None
        if self._warm_active in removed:
            self._warm_active = None
        if self.ranges_pending() and not set(removed).isdisjoint(self._ranges_files):
            self._schedule_ranges(project, force=True)
        self._advance_crop_chain(project)
        self._settle_tier(project)
        self._advance_confirm_chain(project)
        self._advance_warm_chain(project)

    def on_job_event(self, event: JobEvent) -> None:
        """Consume a terminal event of a job AutoPilot submitted and, when it is
        current (is_current), submit what it unblocks. A non-current event only
        releases its own submission. Call after applying the result (see the
        class docstring). A "started" event only records that the job runs
        (boost_lines leaves a running draw alone); other events are ignored."""
        if event.type == "started":
            if self._outstanding.get(event.key, 0) > 0:
                self._started[event.key] = event.job_id
            return
        if event.type not in TERMINAL_EVENTS:
            return
        if self._started.get(event.key) == event.job_id:
            del self._started[event.key]
        count = self._outstanding.get(event.key, 0)
        if count <= 0:
            return
        current = self.is_current(event)
        kind, file = self._identity[event.key]
        if count > 1:
            self._outstanding[event.key] = count - 1
        else:
            del self._outstanding[event.key]
            del self._identity[event.key]
        if not current:
            if kind == "crop" and file in self._redetect:
                # Remember it, so the key's last crop event can tell whether it
                # is the re-detect's own job (the newest has the highest job_id).
                self._redetect_superseded[file] = max(self._redetect_superseded.get(file, 0), event.job_id)
            self._schedule_readded(self._project(), file)
            return
        project = self._project()
        if kind == "metadata":
            self._after_metadata_event(project, file)
        elif kind == "crop":
            self._after_crop_event(project, file, event)
            self._want_lines(project, file)
            self._advance_crop_chain(project)
        elif kind == "brightness":
            self._after_brightness_event(project, file, event)
        elif kind == "ranges":
            if event.type != "cancelled":
                self._ranges_done = True
            self._release_ranges_waits(project)
            for name in list(project.files):
                self._want_lines(project, name)
        elif kind == "audio_profile":
            self._want_lines(project, file)
        elif kind == "lines":
            self._after_lines_event(file, event)
        elif kind == "confirm":
            # Said explicitly rather than left to fall through: a confirm
            # unblocks nothing of its own. Nothing is scheduled behind one,
            # and whatever it changed -- a value lowered, a flag dropped --
            # the owner has already applied (apply_confirm) and the chains
            # advanced at the end of this method pick up. The event's only
            # job here is to free the chain's one slot.
            pass
        elif kind == "warm" and event.type == "cancelled" and file == self._warm_displaced:
            # boost_warm took this file's worker away for one the user is
            # heading towards. It learned nothing about the file, so it is
            # offered again rather than left memoed as warmed. Every other
            # cancellation keeps its memo: that is what stops a folder being
            # resubmitted for ever (see "Warming the view cache").
            self._warmed.pop(file, None)
            self._warm_displaced = None
        self._schedule_readded(project, file)
        self._settle_tier(project)
        self._advance_confirm_chain(project)  # a confirm's own event moves its chain on too
        self._advance_warm_chain(project)     # a warm job's own event moves the chain on too

    def redetect(self, file: str) -> None:
        """User "re-detect": crop, then full brightness, for one file, whatever
        its values' sources (results still obey the apply rules). Metadata
        first when the duration is unknown. Respects the extraction toggles.
        The brightness step follows only the re-detect's own crop job, and only
        when it finishes: a crop job cancelled, failed or superseded (a later
        crop hint, or an older same-key job that ends last) drops it."""
        project = self._project()
        if file not in project.files or project.folder.labels_only:
            return
        self._redetect.add(file)
        self._redetect_superseded.pop(file, None)
        self._crop_chain.discard(file)                     # this detection replaces the auto-fill one
        self._submit_crop(project, file, boost=True)
        self._settle_tier(project)

    def boost_lines(self, file: str) -> None:
        """The Brightness tab shows `file`: draw its gallery lines now, at
        LINES_BOOST, when they are wanted (_lines_wanted), whatever
        autopilot_enabled says and without waiting for the ranges or the
        audio profile. Idempotent, so the tab may call it on every refresh:
        nothing is submitted while the file's newest lines job draws on its
        current crop and is either boosted already or running, nor after a
        draw on that crop failed (a view refreshing on every event would
        otherwise resubmit a failing job forever; shuffle_lines retries). A
        queued auto-pilot draw is replaced by the boosted one (same key)."""
        project = self._project()
        entry = project.files.get(file)
        if entry is None or not _lines_wanted(project.folder, entry):
            return
        box = _crop_box(entry)
        if self._lines_failed.get(file) == box:
            return
        request = self._lines_request(file)
        if request is not None and request.box == box and (
                request.priority >= LINES_BOOST or self._lines_running(file)):
            return
        self._submit_lines(project, file, priority=LINES_BOOST, explicit=True)

    def shuffle_lines(self, file: str, exclude: list[float]) -> None:
        """The user's "shuffle": draw new gallery lines for `file` with a new
        seed, away from the times in `exclude` (the lines it shows now), at
        LINES_BOOST. Explicit: it needs only a crop, and runs whatever
        autopilot_enabled, the file's lines evidence or an earlier failure
        say. A draw already outstanding is superseded (see "Superseded
        jobs")."""
        project = self._project()
        entry = project.files.get(file)
        if entry is None or entry.crop is None:
            return
        self._submit_lines(project, file, priority=LINES_BOOST, explicit=True, exclude=exclude)

    def hint_targets(self, source_file: str, what: str) -> list[str]:
        """The files a hint re-detect from `source_file` submits jobs for, in
        name order; `what` is "crop" or "brightness" (ValueError otherwise).

        Every other file that is not skipped, whose target value is None or
        DETECTED/HINT (detection can never write MANUAL or IMPORTED values,
        so re-measuring them would only replace their evidence), and for
        which that detection is enabled: crop unless the folder is
        labels-only, brightness only while dialogue is extracted, and only
        for files with a crop to measure on. Empty when `source_file` cannot
        seed the hint: not in the project, no crop or no media height (crop),
        no brightness (brightness)."""
        if what not in ("crop", "brightness"):
            raise ValueError(f"hint re-detects are for 'crop' or 'brightness', not {what!r}")
        project = self._project()
        folder = project.folder
        source = project.files.get(source_file)
        if source is None:
            return []
        if what == "crop":
            if folder.labels_only or _crop_hint(source) is None:
                return []
        elif not folder.dialogue_enabled or source.brightness is None:
            return []
        targets = []
        for name, entry in project.files.items():
            if name == source_file or entry.skipped:
                continue
            value = entry.crop if what == "crop" else entry.brightness
            if value is not None and value.source not in DETECTION_SOURCES:
                continue
            if what == "brightness" and entry.crop is None:
                continue
            targets.append(name)
        return targets

    def redetect_others_with_crop_hint(self, source_file: str) -> None:
        """Re-detect the crop of every file in hint_targets(source_file,
        "crop") with the consensus seeded from `source_file`'s crop (ruling
        C3)."""
        project = self._project()
        targets = self.hint_targets(source_file, "crop")
        if not targets:
            return
        hint = _crop_hint(project.files[source_file])
        for name in targets:
            self._crop_chain.discard(name)
            self._drop_redetect(name)                      # this crop job supersedes a re-detect's
            self._submit_crop(project, name, hint=hint, boost=True)
        self._settle_tier(project)

    def redetect_others_with_brightness_hint(self, source_file: str) -> None:
        """Full brightness detection on every file in
        hint_targets(source_file, "brightness"), checked against
        `source_file`'s value (ruling C3)."""
        project = self._project()
        targets = self.hint_targets(source_file, "brightness")
        if not targets:
            return
        value = int(project.files[source_file].brightness.value)
        for name in targets:
            self._submit_brightness(project, name, _crop_box(project.files[name]), plateau=None, hint_value=value,
                                    priority=PRIORITY["brightness"] + REDETECT_BOOST, explicit=True)
        self._settle_tier(project)

    def on_crop_changed(self, file: str) -> None:
        """The file's crop changed (an edit, a paste, an applied result):
        measure brightness on the new box unless the value is the user's
        (MANUAL/IMPORTED) or is already measured, or being measured, on it;
        and draw the gallery lines on it (the old ones show another region,
        see _want_lines)."""
        project = self._project()
        self._remeasure_brightness(project, file, unrecorded=True)
        self._want_lines(project, file)
        self._settle_tier(project)
        self._advance_confirm_chain(project)  # the strip a confirm probes is cut with the crop box
        self._advance_warm_chain(project)     # strips are keyed by the crop box

    def on_folder_changed(self, old: FolderSettings, new: FolderSettings) -> None:
        """Submit what the new settings newly require: the folder schedule when
        auto-pilot was just turned on; crop, brightness and gallery lines for
        files missing them when extraction starts needing them. Call after
        core.jobs.apply.apply_folder_change(project, old, new). The full tier is
        settled whatever changed (it is left alone while dialogue is off)."""
        project = self._project()
        if new.autopilot_enabled and not old.autopilot_enabled:
            self._schedule_folder(project)
        elif new.autopilot_enabled and (
                (old.labels_only and not new.labels_only) or (new.dialogue_enabled and not old.dialogue_enabled)):
            self._schedule_newly_required(project, new)
        for name in list(project.files):
            self._want_lines(project, name)
        self._settle_tier(project)
        self._advance_confirm_chain(project)   # conf_threshold and the toggles decide what a confirm asks
        self._advance_warm_chain(project)

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
        """Hold the queued jobs a run must not wait behind (HELD_KINDS:
        detection plus brightness confirms and view-cache warming) on the GPU
        and CPU lanes; metadata and thumbnails keep running."""
        if self._paused:
            return
        self._paused = True
        for lane in (Lane.GPU, Lane.CPU):
            self._runner.pause(lane, only=is_held_job)

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
        metadata or a crop, or waiting for the full tier), and "lines" for a
        draw waiting for the ranges or the file's audio profile. Never
        UNREPORTED_KINDS: warming holds no value and no review state, and a
        folder whose view cache is filling is a folder that has been detected
        (see "Warming the view cache"); a confirm does hold a value, but only
        one it may take AWAY a doubt from, so the file keeps its honest
        "check brightness" badge until the confirm clears it rather than
        churning a whole folder of flagged rows through a transient "waiting"
        that tells the user nothing (see "Confirming a doubted brightness").
        For compute_review_state (which never waits for "lines" either: they
        are view evidence) and the views; a fresh dict."""
        project = self._project()
        folder = project.folder
        pending: dict[str, set[str]] = {}
        for kind, file in self._identity.values():
            if kind in UNREPORTED_KINDS:
                continue
            if file is not None and file in project.files:     # not a removed file's job still ending
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
            if self._lines_due(project, name, entry) and self._lines_blocked(name):
                kinds.add("lines")
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
        count = self._outstanding.get(event.key, 0)
        if count > 1:
            return False
        return count == 0 or not self._request_waits(event.key)

    def _request_waits(self, key: str) -> bool:
        """A brightness request newer than every submission for `key` waits for the ranges."""
        kind, file = self._identity.get(key, (None, None))
        return kind == "brightness" and file in self._ranges_waits

    # --- scheduling -------------------------------------------------------------

    def _project(self) -> Project:
        return self._project_getter()

    def _is_outstanding(self, kind: str, file: str) -> bool:
        return self._outstanding.get(f"{kind}:{file}", 0) > 0

    def _file_outstanding(self, name: str) -> bool:
        """Any submission for the file is outstanding."""
        return any(file == name for _kind, file in self._identity.values())

    def _schedule_readded(self, project: Project, name: str | None) -> None:
        """A file added back while jobs for it were outstanding: once the last
        of them has ended, schedule it as on_files_added would (without
        another ranges analysis)."""
        if name not in self._readded or self._file_outstanding(name):
            return
        self._readded.discard(name)
        if name not in project.files:
            return
        self._scan_metadata(project, name)
        self._schedule_file(project, name)
        self._settle_tier(project)

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
        """Metadata first (a CPU worker may start whatever is queued at once,
        and the ranges analysis has the lowest priority), then the ranges
        analysis (before any brightness: it waits for ranges), then each
        file's jobs."""
        for name in list(project.files):
            self._scan_metadata(project, name)
        if project.folder.autopilot_enabled:
            self._schedule_ranges(project, force=False)
        for name in list(project.files):
            self._schedule_file(project, name)
        self._settle_tier(project)

    def _scan_metadata(self, project: Project, name: str) -> None:
        if project.files[name].media.duration <= 0 and not self._is_outstanding("metadata", name):
            self._submit(MetadataJob(project.path, name), PRIORITY["metadata"])

    def _schedule_file(self, project: Project, name: str) -> None:
        self._scheduled.add(name)
        entry = project.files[name]
        folder = project.folder
        if entry.media.duration <= 0:
            self._scan_metadata(project, name)
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
        self._want_lines(project, name)                  # after the audio profile: the draw waits for it

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
        self._ranges_files = tuple(names)

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

    @staticmethod
    def _thumbnail_time(entry: FileEntry) -> float | None:
        """The frame this file's queue row shows, or None while there is not
        enough known to choose one.

        The one place that rule lives: the submit path below picks the frame
        with it, and the warm pass names the same frame in its Wanted, so
        `FileViewCache.trim` keeps the thumbnail on disk instead of deleting
        the one image the queue needs most."""
        if entry.sample_time is not None:
            return float(entry.sample_time)
        if entry.media.duration > 0:
            return THUMBNAIL_FRACTION * entry.media.duration
        return None

    def _submit_thumbnail(self, project: Project, name: str) -> None:
        entry = project.files[name]
        time = self._thumbnail_time(entry)
        if time is None:
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

    def _lines_request(self, name: str) -> _LinesRequest | None:
        """The file's newest lines request while it is outstanding."""
        return self._lines_requests.get(name) if self._is_outstanding("lines", name) else None

    def _lines_running(self, name: str) -> bool:
        """The file's newest lines job runs: a "started" event arrived for the
        key and it is the key's only outstanding submission (a job never
        starts while an older one with its key runs, so the one that started
        is the oldest outstanding; it is the newest only when it is alone)."""
        key = f"lines:{name}"
        return key in self._started and self._outstanding.get(key, 0) == 1

    def _lines_blocked(self, name: str) -> bool:
        """An auto-pilot draw waits: a ranges analysis is pending (the keep
        ranges it samples may still change) or the file's audio profile is
        outstanding (its speech spans weight the draw)."""
        return self.ranges_pending() or self._is_outstanding("audio_profile", name)

    def _lines_due(self, project: Project, name: str, entry: FileEntry) -> bool:
        """Auto-pilot draws the file's lines (now, or once _lines_blocked
        clears): the folder schedule covers the file with auto-pilot on, the
        lines are wanted, no lines job is outstanding on the file's crop, and
        the last draw on that crop did not fail."""
        if not self._detects_automatically(project.folder, name) or not _lines_wanted(project.folder, entry):
            return False
        box = _crop_box(entry)
        request = self._lines_request(name)
        return (request is None or request.box != box) and self._lines_failed.get(name) != box

    def _want_lines(self, project: Project, name: str | None) -> None:
        """Re-evaluate the file's gallery lines (after its metadata, crop,
        audio profile, the ranges, a crop change, a folder change). A draw the
        user asked for (boost or shuffle) still outstanding on another crop is
        resubmitted on the current one at once, as it was; otherwise an
        auto-pilot draw is submitted when _lines_due and not _lines_blocked
        (while blocked, pending() reports it and a later trigger submits it).
        An outstanding draw on another crop is superseded by the new one; if
        it ends first, apply_lines drops its lines."""
        entry = project.files.get(name) if name is not None else None
        if entry is None or not _lines_wanted(project.folder, entry):
            return
        request = self._lines_request(name)
        if request is not None and request.explicit and request.box != _crop_box(entry):
            self._submit_lines(project, name, priority=request.priority, explicit=True)
            return
        if self._lines_due(project, name, entry) and not self._lines_blocked(name):
            self._submit_lines(project, name, priority=PRIORITY["lines"], explicit=False)

    def _submit_lines(self, project: Project, name: str, *, priority: int, explicit: bool,
                      exclude: Iterable[float] = ()) -> None:
        """Every lines job goes through here: a new seed, the file's crop,
        keep ranges and speech spans as they are now."""
        entry = project.files[name]
        box = _crop_box(entry)
        self._submit(LinesJob(project.path, name, box, _time_ranges(entry), _speech(entry), project.folder,
                              seed=self._seed_source(), exclude=tuple(exclude)), priority)
        self._lines_requests[name] = _LinesRequest(box, int(priority), explicit)
        self._lines_failed.pop(name, None)

    # --- confirming a doubted brightness ----------------------------------------

    def _confirm_wanted(self, folder: FolderSettings, name: str, entry: FileEntry,
                        pending: dict[str, set[str]]) -> _ConfirmProbe | None:
        """The probe a ConfirmJob for the file would ask, or None when it is
        not to be confirmed now.

        The full rule and the reasoning behind each clause are in "Confirming
        a doubted brightness"; in short, the file must be flagged on a
        brightness that was actually MEASURED, that detection still owns, on
        a crop that is not itself in doubt, with nothing of its own still
        moving, a strip in its evidence to read -- and this exact question
        must not have been asked already, on disk (confirm.matches) or in
        this session (self._confirmed).
        """
        if not folder.autopilot_enabled or not folder.dialogue_enabled:
            return None
        if entry.skipped or entry.crop is None or entry.brightness is None:
            return None
        if entry.brightness.source not in DETECTION_SOURCES or brightness_is_stale(entry):
            return None
        if not apply.counts_flagged(entry, "brightness"):
            return None
        reasons = set((entry.flags.get("brightness") or "").split("+"))
        if reasons & NOTHING_MEASURED_FLAGS:
            return None
        if reasons & apply.SOURCE_INDEPENDENT_FLAGS["brightness"]:
            return None
        if apply.counts_flagged(entry, "crop"):
            return None
        if name in pending:
            return None
        probe_time = probe_time_for(entry)
        if probe_time is None:
            return None
        probe = _ConfirmProbe(_crop_box(entry), float(probe_time), int(entry.brightness.value),
                              int(folder.conf_threshold))
        if self._confirmed.get(name) == probe:
            return None
        # Evidence comes back from the disposable .ocr-cache/ and may be
        # anything at all, while the value and flags this file was picked on
        # come from .ocr.json and are sound. A junk evidence dict must cost
        # this file its memo, not raise out of a scheduling decision.
        stored = entry.evidence.get("brightness") if isinstance(entry.evidence, dict) else None
        recorded = stored.get("confirm") if isinstance(stored, dict) else None
        if confirm.matches(recorded, start_value=probe.start_value, conf_threshold=probe.conf_threshold,
                           crop_box=probe.box, probe_time=probe.probe_time):
            return None
        return probe

    def _advance_confirm_chain(self, project: Project) -> None:
        """Submit the next file's brightness confirm, in name order, unless
        one is still outstanding: one job at a time for the whole folder (see
        "Confirming a doubted brightness"). The probe it was submitted for is
        remembered, so a file is asked again only when the question changes,
        whatever the job's terminal event said -- a job that failed or was
        cancelled writes no evidence record, and without the memo its file
        would be offered again on every event for ever."""
        if self._confirm_active is not None and self._is_outstanding("confirm", self._confirm_active):
            return
        self._confirm_active = None
        self._confirmed = {name: probe for name, probe in self._confirmed.items() if name in project.files}
        folder = project.folder
        if not folder.autopilot_enabled or not folder.dialogue_enabled:
            return                                  # nothing to ask: skip pending()'s work entirely
        pending = self.pending()
        for name, entry in project.files.items():
            probe = self._confirm_wanted(folder, name, entry, pending)
            if probe is None:
                continue
            self._submit(ConfirmJob(project.path, name, probe.box, probe.probe_time, probe.start_value, folder),
                         PRIORITY["confirm"])
            self._confirmed[name] = probe
            self._confirm_active = name
            return

    # --- warming the view cache -------------------------------------------------

    def _warm_wanted(self, name: str, entry: FileEntry, pending: dict[str, set[str]]) -> Wanted | None:
        """What a warm job for the file would hold, or None when it is not to
        be warmed now: it is skipped, a detection of its own is outstanding or
        foreseen (its evidence is still moving), its evidence names nothing to
        hold, or it was warmed for exactly this Wanted already."""
        if entry.skipped or name in pending:
            return None
        keep = wanted(entry, self._thumbnail_time(entry))
        if not keep or self._warmed.get(name) == keep:
            return None
        return keep

    def boost_warm(self, files: Iterable[str]) -> None:
        """The user is on `files[0]` and heading down `files[1:]`: warm these
        next, in this order, before the rest of the folder.

        The queue hands its own visible order (a filter such as "Needs you"
        selects a scattered handful of a big folder), because name order is
        exactly the order the user is NOT working in: warming would plod from
        the first episode while the user jumps to the seventy-fourth, and
        every file they opened would decode its strips from scratch.

        Only an ordering, not a queue of its own -- the chain still runs one
        job at a time and still skips a file that is already warm for its
        Wanted, so calling this on every selection costs nothing. A warm job
        already running for a file nobody is heading towards is cancelled,
        since the user is waiting on the boosted one and a half-warmed cache
        is still a correct cache.
        """
        self._warm_ahead = tuple(dict.fromkeys(files))
        if self._warm_active is not None and self._warm_active not in self._warm_ahead:
            # Remembered so its memo can be dropped when the cancellation
            # arrives: a job the USER displaced must still be warmed later,
            # while an ordinary cancelled one stays memoed, or a folder whose
            # warm jobs keep being cancelled would resubmit for ever.
            self._warm_displaced = self._warm_active
            self._runner.cancel(f"warm:{self._warm_active}")
        self._advance_warm_chain(self._project())

    def _warm_order(self, project: Project) -> Iterable[str]:
        """The files to consider warming: the ones the user is heading
        towards first, then the rest of the folder in name order."""
        ahead = [name for name in self._warm_ahead if name in project.files]
        rest = [name for name in project.files if name not in set(ahead)]
        return ahead + rest

    def _advance_warm_chain(self, project: Project) -> None:
        """Submit the next file's view-cache warm job, in name order, unless
        one is still outstanding: one job at a time for the whole folder (see
        "Warming the view cache"). The Wanted it was submitted for is
        remembered, so a file is warmed again only when its evidence or its
        crop box moves, whatever the job's terminal event said."""
        if self._warm_active is not None and self._is_outstanding("warm", self._warm_active):
            return
        self._warm_active = None
        self._warmed = {name: keep for name, keep in self._warmed.items() if name in project.files}
        pending = self.pending()
        for name in self._warm_order(project):
            keep = self._warm_wanted(name, project.files[name], pending)
            if keep is None:
                continue
            self._submit(WarmJob(project.path, name, keep), WARM_PRIORITY)
            self._warmed[name] = keep
            self._warm_active = name
            return

    # --- job events -------------------------------------------------------------

    def _after_metadata_event(self, project: Project, name: str) -> None:
        request = self._deferred_crops.pop(name, None)
        entry = project.files.get(name)
        if entry is None or entry.media.duration <= 0:
            self._drop_redetect(name)
            return
        if name in self._scheduled:
            self._after_metadata(project, name, crop=request is None)
        if request is not None:
            self._submit_crop(project, name, hint=request.hint, boost=request.boost)

    def _after_crop_event(self, project: Project, name: str, event: JobEvent) -> None:
        entry = project.files.get(name)
        # The re-detect's brightness step needs its own crop job, finished: this
        # event ends the key's last submission, and no superseded event of the
        # key since the re-detect had a higher job_id (a newer job, cancelled).
        if not (event.type == "finished" and event.job_id >= self._redetect_superseded.get(name, 0)):
            self._drop_redetect(name)
        if entry is None:
            self._drop_redetect(name)
            return
        if (project.folder.dialogue_enabled and entry.crop is not None
                and self._brightness_follows_crop(name, entry)):
            if name in self._redetect:
                self._submit_brightness(project, name, _crop_box(entry), plateau=None, hint_value=None,
                                        priority=PRIORITY["brightness"] + REDETECT_BOOST, explicit=True)
            else:
                self._remeasure_brightness(project, name, unrecorded=False)
        self._drop_redetect(name)
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
                    and name not in self._ranges_waits              # never replaces a waiting request
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

    def _after_lines_event(self, name: str | None, event: JobEvent) -> None:
        """The file's newest lines job ended. Nothing is resubmitted: a draw
        that failed is remembered with its crop, so neither auto-pilot nor
        boost_lines submits it again on that crop (shuffle_lines does)."""
        request = self._lines_requests.pop(name, None)
        if event.type == "failed" and request is not None:
            self._lines_failed[name] = request.box

    def _drop_redetect(self, name: str) -> None:
        self._redetect.discard(name)
        self._redetect_superseded.pop(name, None)

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
        """A crop detection is outstanding, waits in the auto-fill chain, will
        follow outstanding metadata, or will be scheduled for a file added back
        once its old jobs end."""
        folder = project.folder
        if self._is_outstanding("crop", name):
            return True
        if name in self._crop_chain and _crop_wanted(folder, entry):
            return True
        if name in self._readded and self._detects_automatically(folder, name) and _crop_wanted(folder, entry):
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
