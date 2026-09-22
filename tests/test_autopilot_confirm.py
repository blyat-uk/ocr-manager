"""AutoPilot's confirm stage: which doubted brightnesses get an OCR probe.

No OCR runs here. The fake runner and the model owner of test_autopilot are
reused as they stand, with core.jobs.apply.apply_confirm wired into deliver()
-- that module has no applier for a kind it never submits.

The folders built here are folders the detectors have finished with: every
file has its media, a detected crop, a detected brightness the detector
MEASURED but doubts, and a strip in its evidence to read. What is left to
decide is scheduling: whose doubt a probe could retire, in what order, and
which files must be left to the user.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from core.detect import confirm as confirm_mod
from core.detect import crop as crop_mod
from core.detect.brightness import (
    FLAG_NARROW_PLATEAU,
    FLAG_NO_CLEAN_THRESHOLD,
    NOTHING_MEASURED_FLAGS,
)
from core.detect.confirm import ConfirmResult, Rung
from core.jobs.apply import (
    FLAG_BRIGHTNESS_OTHER_CROP,
    apply_confirm,
    apply_folder_change,
)
from core.jobs.autopilot import (
    AUTOPILOT_KINDS,
    DETECTION_KINDS,
    HELD_KINDS,
    PRIORITY,
    is_autopilot_job,
    is_detection_job,
    is_held_job,
)
from core.jobs.detect_jobs import ConfirmJob, ConfirmJobResult
from core.jobs.runner import Lane
from core.project import (
    Brightness,
    Crop,
    FolderSettings,
    Project,
    ReviewState,
    Source,
    TimeRanges,
)
from test_autopilot import (
    BOX,
    OTHER_BOX,
    Owner as _Owner,
    Submission,
    _media,
    _project,
    finish_side_jobs,
    lines_done,
    of_kind,
    pairs,
)

BRIGHTNESS = 200                 # the value the detector measured and doubted
PROBE_TIME = 512.0               # the detector's earliest text strip (evidence["brightness"]["strips"])
DOUBT = FLAG_NARROW_PLATEAU      # a blocking flag about a reading that WAS taken


# --------------------------------------------------------------------------
# The owner, the builders
# --------------------------------------------------------------------------

class Owner(_Owner):
    """test_autopilot's owner, plus apply_confirm.

    Its APPLY table covers the kinds that module submits, and "confirm" is
    not one of them; everything else about the protocol (is_current, then the
    apply, then on_job_event, then recompute_all) is unchanged.
    """

    def deliver(self, submission: Submission, result=None, type="finished") -> bool:
        if submission.job.kind != "confirm":
            return super().deliver(submission, result, type)
        event = self.event(submission, result, type)
        current = self.autopilot.is_current(event)
        if current:
            apply_confirm(self.project, result)
        self.autopilot.on_job_event(event)
        self.recompute()
        return current


def _doubted(entry, *, value: int = BRIGHTNESS, source: Source = Source.DETECTED, flag: str | None = DOUBT,
             measured_on=BOX, probe: float | None = PROBE_TIME):
    """A file the brightness detector measured and then doubted: the value it
    read, the crop it read it on, the strips it read (where the probe time
    comes from without gallery lines) and the flag it raised."""
    entry.brightness = Brightness(value, source)
    evidence = {"crop_box": list(BOX), "value_crop_box": list(measured_on), "flagged": flag}
    if probe is not None:
        evidence["strips"] = [{"time": probe, "is_text": True}]
    entry.evidence["brightness"] = evidence
    if flag is not None:
        entry.flags["brightness"] = flag
    return entry


def _confirm_folder(tmp_path, names, folder: FolderSettings | None = None) -> Project:
    """Files with nothing left to detect but a doubt to retire: known media, a
    detected crop, the user's whole file for ranges, an audio profile, gallery
    lines drawn on the current crop -- and a detected, flagged brightness.

    The lines evidence names no samples on purpose: a file whose evidence
    names pixels to hold earns a warm job too, and the warm chain is another
    module's rule. The probe time here comes from the detector's own strips.
    """
    project = _project(tmp_path, names, folder)
    for entry in project.files.values():
        _media(entry).crop = Crop(*BOX, Source.DETECTED)
        entry.sample_time = 300.0
        entry.time_ranges = TimeRanges([], Source.MANUAL)
        entry.evidence["audio"] = {"envelope": [0.0], "speech": [], "duration": 1400.0}
        entry.evidence["lines"] = {"crop_box": list(BOX), "seed": 7, "tried": 12, "samples": []}
        _doubted(entry)
    return project


def _opened(owner: Owner) -> list[Submission]:
    """Open the folder and finish the thumbnails it always submits: a confirm
    waits behind every job of its own file, and pending() reports thumbnails
    like anything else."""
    owner.autopilot.on_open()
    opened = owner.take()
    assert of_kind(opened, "confirm") == [], pairs(opened)   # never before the file's own jobs
    finish_side_jobs(owner, opened)
    return opened


def confirms(submissions: list[Submission]) -> list[Submission]:
    return of_kind(submissions, "confirm")


def confirm_done(sub: Submission, value: int | None = None, *, cancelled: bool = False) -> ConfirmJobResult:
    """A result for the ladder `sub` was submitted for: `value` is the rung
    that read the strip, None when none did."""
    job = sub.job
    walked = confirm_mod.ladder(job.start_value)
    if value is not None:
        walked = walked[:walked.index(value) + 1]
    rungs = tuple(Rung(threshold=rung, gated=True, text="字幕" if rung == value else "",
                       confidence=0.99 if rung == value else 0.0, passed=rung == value)
                  for rung in walked)
    result = ConfirmResult(value=value, probe_time=job.probe_time, rungs=rungs, cancelled=cancelled)
    return ConfirmJobResult(job.file, result, job.crop_box, job.start_value, job.conf_threshold)


def _left_alone(owner: Owner) -> None:
    """Open the folder and assert nothing about it is confirmed: whatever else
    its files need, no doubt here is this stage's to retire."""
    _opened(owner)
    assert confirms(owner.take()) == []
    assert confirms(owner.runner.submissions) == []


# --------------------------------------------------------------------------
# The stage itself
# --------------------------------------------------------------------------

def test_confirming_is_held_work_of_its_own_kind_and_never_detection():
    assert "confirm" in AUTOPILOT_KINDS and "confirm" in HELD_KINDS
    assert "confirm" not in DETECTION_KINDS and HELD_KINDS == DETECTION_KINDS | {"confirm", "warm"}
    # On the GPU lane: under crop, brightness and lines. Lines first of all,
    # because the draw is where the probe strip comes from.
    assert PRIORITY["confirm"] == 0 < PRIORITY["lines"] < PRIORITY["brightness"] < PRIORITY["crop"]


def test_a_confirm_job_is_held_with_detection_but_is_not_detection(tmp_path):
    job = ConfirmJob(str(tmp_path), "a.mkv", BOX, PROBE_TIME, BRIGHTNESS, FolderSettings())
    assert is_autopilot_job(job) and is_held_job(job)
    assert not is_detection_job(job)      # it can only ever REMOVE a reason to stop


def test_a_file_flagged_on_a_measured_but_doubted_brightness_is_confirmed(tmp_path):
    project = _confirm_folder(tmp_path, ["a.mkv"])
    owner = Owner(project)
    opened = _opened(owner)
    assert pairs(opened) == [("thumbnail", "a.mkv")]
    assert owner.state("a.mkv") == ReviewState.FLAGGED

    (sub,) = owner.take()
    job = sub.job
    assert (job.kind, job.file, job.lane, job.priority) == ("confirm", "a.mkv", Lane.GPU, 0)
    assert job.key == "confirm:a.mkv" and job.project_dir == project.path
    assert job.crop_box == BOX                       # the strip is cut with the file's own crop
    assert job.probe_time == PROBE_TIME              # a strip the detector already read text on
    assert job.start_value == BRIGHTNESS             # the ladder starts at the stored value
    assert job.conf_threshold == project.folder.conf_threshold


def test_a_confirmed_file_stops_asking_for_the_user(tmp_path):
    project = _confirm_folder(tmp_path, ["a.mkv"])
    owner = Owner(project)
    _opened(owner)
    (sub,) = owner.take()

    assert owner.deliver(sub, confirm_done(sub, BRIGHTNESS)) is True
    entry = project.files["a.mkv"]
    assert entry.brightness == Brightness(BRIGHTNESS, Source.DETECTED)   # the value stands
    assert entry.flags.get("brightness") == ""                           # the doubt is retired
    assert owner.state("a.mkv") == ReviewState.PROPOSED
    assert owner.take() == []                                            # and it is never asked again


def test_a_ladder_that_passes_nowhere_leaves_the_file_to_the_user(tmp_path):
    project = _confirm_folder(tmp_path, ["a.mkv"])
    owner = Owner(project)
    _opened(owner)
    (sub,) = owner.take()

    owner.deliver(sub, confirm_done(sub, None))
    entry = project.files["a.mkv"]
    assert entry.brightness == Brightness(BRIGHTNESS, Source.DETECTED)
    assert entry.flags["brightness"] == DOUBT        # 140 was not shown to beat the detector's pick
    assert owner.state("a.mkv") == ReviewState.FLAGGED
    assert owner.take() == []                        # the record it wrote stops the folder re-asking


def test_the_probe_reads_the_gallery_line_once_the_draw_has_landed(tmp_path):
    """The lines draw is where the probe strip comes from, and the view cache
    holds it losslessly: a warmed file costs no decode at all."""
    project = _confirm_folder(tmp_path, ["a.mkv"])
    project.files["a.mkv"].evidence["lines"] = {"crop_box": list(OTHER_BOX)}   # drawn on an old crop
    owner = Owner(project)
    opened = _opened(owner)

    (draw,) = of_kind(opened, "lines")
    assert confirms(owner.take()) == []               # "lines" is pending: the file's evidence still moves
    assert "lines" in owner.autopilot.pending()["a.mkv"]
    owner.deliver(draw, lines_done(draw, times=(407.5, 900.0)))

    (sub,) = confirms(owner.take())
    assert sub.job.probe_time == 407.5                # the gallery's first line, not the detector's strip


# --------------------------------------------------------------------------
# The chain
# --------------------------------------------------------------------------

def test_confirms_run_one_at_a_time_for_the_whole_folder_in_name_order(tmp_path):
    names = [f"ep{index:02d}.mkv" for index in range(1, 5)]
    owner = Owner(_confirm_folder(tmp_path, names))
    _opened(owner)
    for name in names:
        outstanding = owner.take()
        assert pairs(outstanding) == [("confirm", name)]   # one GPU job for the whole folder
        owner.deliver(outstanding[0], confirm_done(outstanding[0], BRIGHTNESS))
    assert owner.take() == []


@pytest.mark.parametrize("outcome", ["finished", "failed", "cancelled"])
def test_the_next_confirm_follows_the_previous_ones_terminal_event_whatever_it_says(tmp_path, outcome):
    names = ["a.mkv", "b.mkv"]
    owner = Owner(_confirm_folder(tmp_path, names))
    _opened(owner)
    (first,) = owner.take()
    assert first.job.file == "a.mkv"

    result = confirm_done(first, None) if outcome == "finished" else None
    owner.deliver(first, result, type=outcome)
    assert pairs(owner.take()) == [("confirm", "b.mkv")]


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_a_confirm_that_never_reported_is_not_asked_again_this_session(tmp_path, outcome):
    """A failed or cancelled job writes no evidence record, so only the
    in-session memo stops its file being offered on every event for ever."""
    names = ["a.mkv", "b.mkv"]
    project = _confirm_folder(tmp_path, names)
    owner = Owner(project)
    _opened(owner)
    (first,) = owner.take()
    owner.deliver(first, None, type=outcome)
    assert "confirm" not in (project.files["a.mkv"].evidence.get("brightness") or {})

    (second,) = owner.take()
    assert second.job.file == "b.mkv"
    owner.deliver(second, confirm_done(second, BRIGHTNESS))
    assert owner.take() == []

    folder = project.folder
    owner.autopilot.on_folder_changed(folder, folder)     # every trigger the chain listens on
    owner.autopilot.on_crop_changed("a.mkv")
    owner.autopilot.on_open()
    assert confirms(owner.take()) == []


def test_a_removed_files_confirm_does_not_stall_the_chain(tmp_path):
    names = ["a.mkv", "b.mkv"]
    owner = Owner(_confirm_folder(tmp_path, names))
    _opened(owner)
    (first,) = owner.take()
    assert first.job.file == "a.mkv"

    del owner.project.files["a.mkv"]
    owner.autopilot.on_files_removed(["a.mkv"])
    owner.recompute()
    assert pairs(owner.take()) == [("confirm", "b.mkv")]
    assert owner.deliver(first, None, type="cancelled") is True    # the owner cancelled it at removal
    assert owner.take() == []


def test_a_file_that_comes_back_is_confirmed_again(tmp_path):
    """Its memo went with it: the evidence a new entry carries is not the old
    entry's, and neither is the answer to the question it asks."""
    project = _confirm_folder(tmp_path, ["a.mkv"])
    owner = Owner(project)
    _opened(owner)
    (first,) = owner.take()
    owner.deliver(first, None, type="failed")
    assert owner.take() == []

    entry = project.files.pop("a.mkv")
    owner.autopilot.on_files_removed(["a.mkv"])
    project.files["a.mkv"] = entry
    owner.autopilot.on_files_added(["a.mkv"])
    finish_side_jobs(owner, owner.take())
    assert pairs(confirms(owner.take())) == [("confirm", "a.mkv")]


# --------------------------------------------------------------------------
# Who is left alone
# --------------------------------------------------------------------------

def test_a_skipped_file_is_never_confirmed(tmp_path):
    project = _confirm_folder(tmp_path, ["a.mkv"])
    project.files["a.mkv"].skipped = True
    _left_alone(Owner(project))


def test_a_file_without_a_crop_is_never_confirmed(tmp_path):
    """There is no strip to cut without one -- and the crop is being detected,
    so nothing about this file has settled."""
    project = _confirm_folder(tmp_path, ["a.mkv"])
    project.files["a.mkv"].crop = None
    owner = Owner(project)
    opened = _opened(owner)
    assert pairs(of_kind(opened, "crop")) == [("crop", "a.mkv")]
    assert confirms(owner.take()) == []


@pytest.mark.parametrize("source", [Source.MANUAL, Source.IMPORTED])
def test_the_users_own_brightness_is_never_confirmed(tmp_path, source):
    """Detection may not overwrite it (ruling C2), and this stage writes by
    the same rule as detection."""
    project = _confirm_folder(tmp_path, ["a.mkv"])
    _doubted(project.files["a.mkv"], source=source)
    _left_alone(Owner(project))


def test_a_stale_brightness_is_re_measured_and_not_confirmed(tmp_path):
    """Its value was read on a crop the file no longer has: confirming it
    would bless a reading of another region."""
    project = _confirm_folder(tmp_path, ["a.mkv"])
    _doubted(project.files["a.mkv"], measured_on=OTHER_BOX)
    owner = Owner(project)
    opened = _opened(owner)
    assert pairs(of_kind(opened, "brightness")) == [("brightness", "a.mkv")]
    assert confirms(owner.take()) == []


def test_a_file_whose_brightness_is_not_flagged_is_never_confirmed(tmp_path):
    """There is no doubt there to retire."""
    project = _confirm_folder(tmp_path, ["a.mkv"])
    project.files["a.mkv"].flags.pop("brightness")
    owner = Owner(project)
    _left_alone(owner)
    assert owner.state("a.mkv") == ReviewState.PROPOSED


def test_an_informational_flag_is_not_a_doubt_and_is_never_confirmed(tmp_path):
    """no-clean-threshold says how the result was reached, not that it is
    wrong: the file is not flagged, so there is nothing to answer."""
    project = _confirm_folder(tmp_path, ["a.mkv"])
    _doubted(project.files["a.mkv"], flag=FLAG_NO_CLEAN_THRESHOLD)
    owner = Owner(project)
    _left_alone(owner)
    assert owner.state("a.mkv") == ReviewState.PROPOSED


@pytest.mark.parametrize("flag", sorted(NOTHING_MEASURED_FLAGS))
def test_a_result_that_measured_nothing_has_no_reading_to_confirm(tmp_path, flag):
    """needs-crop, ranges-empty?, no-text, escalate and cancelled all mean the
    stored value is a placeholder rather than a reading of this file."""
    project = _confirm_folder(tmp_path, ["a.mkv"])
    _doubted(project.files["a.mkv"], flag=flag)
    owner = Owner(project)
    _opened(owner)
    assert confirms(owner.take()) == []


def test_a_reason_about_the_stored_value_is_not_this_stages_to_withdraw(tmp_path):
    """brightness-other-crop describes the value itself, counts whoever set
    it, and only the user answers it -- even beside a doubt a probe could
    otherwise retire."""
    for flag in (FLAG_BRIGHTNESS_OTHER_CROP, f"{DOUBT}+{FLAG_BRIGHTNESS_OTHER_CROP}"):
        project = _confirm_folder(tmp_path, ["a.mkv"])
        _doubted(project.files["a.mkv"], flag=flag)
        _left_alone(Owner(project))


def test_a_file_whose_crop_is_flagged_too_is_left_whole_for_the_user(tmp_path):
    """The user has to open it for its crop anyway: clearing the brightness
    half of that row saves them no visit, and costs a GPU job a file nobody
    has to open could have had."""
    project = _confirm_folder(tmp_path, ["a.mkv", "b.mkv"])
    project.files["a.mkv"].flags["crop"] = crop_mod.FLAG_LOW_AGREEMENT
    owner = Owner(project)
    _opened(owner)
    assert owner.state("a.mkv") == ReviewState.FLAGGED
    assert pairs(confirms(owner.take())) == [("confirm", "b.mkv")]


def test_a_file_is_not_confirmed_while_a_detection_of_its_own_is_pending(tmp_path):
    """Its values are still moving, and so are its gallery lines."""
    project = _confirm_folder(tmp_path, ["a.mkv", "b.mkv"])
    project.files["a.mkv"].evidence["lines"] = {"crop_box": list(OTHER_BOX)}   # a redraws on its new crop
    owner = Owner(project)
    opened = _opened(owner)
    (draw,) = of_kind(opened, "lines")
    assert draw.job.file == "a.mkv"

    # b has settled and goes first; a waits, because the draw outstanding for
    # it is the very thing the probe strip would come from.
    (first,) = confirms(owner.take())
    assert first.job.file == "b.mkv"
    assert "lines" in owner.autopilot.pending()["a.mkv"]
    owner.deliver(first, confirm_done(first, BRIGHTNESS))
    assert confirms(owner.take()) == []

    owner.deliver(draw, lines_done(draw))
    assert pairs(confirms(owner.take())) == [("confirm", "a.mkv")]


def test_a_file_with_no_strip_to_read_is_never_confirmed(tmp_path):
    """Nothing in its evidence names a time the file is known to hold a
    subtitle at, so there is nothing to mask and read."""
    project = _confirm_folder(tmp_path, ["a.mkv"])
    _doubted(project.files["a.mkv"], probe=None)
    _left_alone(Owner(project))


@pytest.mark.parametrize("off", ["autopilot_enabled", "dialogue_enabled"])
def test_the_stage_is_gated_by_the_folders_toggles(tmp_path, off):
    """It moves a value, so it is gated where every other detection AutoPilot
    starts on its own is gated; and brightness is a required field only while
    dialogue is extracted."""
    folder = replace(FolderSettings(), **{off: False})
    _left_alone(Owner(_confirm_folder(tmp_path, ["a.mkv"], folder)))


# --------------------------------------------------------------------------
# Asking the same question twice
# --------------------------------------------------------------------------

def _recorded(entry, *, conf_threshold: int = 95, value: int | None = None) -> None:
    """The evidence record apply_confirm writes for a walked ladder."""
    result = ConfirmResult(value=value, probe_time=PROBE_TIME)
    entry.evidence["brightness"]["confirm"] = confirm_mod.record(
        result, start_value=BRIGHTNESS, conf_threshold=conf_threshold, crop_box=BOX)


def test_a_file_whose_evidence_already_answers_this_probe_is_not_asked_again(tmp_path):
    project = _confirm_folder(tmp_path, ["a.mkv"])
    _recorded(project.files["a.mkv"])
    _left_alone(Owner(project))


def test_a_new_confidence_threshold_is_a_new_question(tmp_path):
    """The rungs were judged against the folder's own conf_threshold: change
    it and the recorded answer is an answer to another question."""
    project = _confirm_folder(tmp_path, ["a.mkv"])
    _recorded(project.files["a.mkv"], conf_threshold=95)
    owner = Owner(project)
    _left_alone(owner)

    old = replace(project.folder)
    project.folder = replace(project.folder, conf_threshold=80)
    apply_folder_change(project, old, project.folder)
    owner.autopilot.on_folder_changed(old, project.folder)

    (sub,) = confirms(owner.take())
    assert sub.job.conf_threshold == 80 and sub.job.start_value == BRIGHTNESS


def test_a_crop_change_is_a_new_question(tmp_path):
    """The strip is cut with the crop box, so a recorded answer says nothing
    about the region the file uses now."""
    project = _confirm_folder(tmp_path, ["a.mkv"])
    entry = project.files["a.mkv"]
    _recorded(entry)
    owner = Owner(project)
    _left_alone(owner)

    entry.crop = Crop(*OTHER_BOX, Source.MANUAL)
    entry.evidence["brightness"]["value_crop_box"] = list(OTHER_BOX)   # re-measured there
    owner.autopilot.on_crop_changed("a.mkv")
    (draw,) = of_kind(owner.take(), "lines")       # the gallery redraws on the new region first
    owner.deliver(draw, lines_done(draw))

    (sub,) = confirms(owner.take())
    assert sub.job.crop_box == OTHER_BOX


# --------------------------------------------------------------------------
# What a confirm does not touch
# --------------------------------------------------------------------------

def test_pending_never_reports_confirm_and_the_badge_stays_honest(tmp_path):
    """A file keeps its "check brightness" badge until the probe clears it:
    180 flagged rows must not churn through a transient state of their own."""
    project = _confirm_folder(tmp_path, ["a.mkv", "b.mkv"])
    owner = Owner(project)
    _opened(owner)

    (sub,) = owner.take()
    owner.recompute()
    assert all("confirm" not in kinds for kinds in owner.autopilot.pending().values())
    assert owner.autopilot.pending() == {}
    assert owner.state("a.mkv") == ReviewState.FLAGGED

    owner.start(sub)
    owner.recompute()
    assert owner.autopilot.pending() == {}
    assert owner.state("a.mkv") == ReviewState.FLAGGED

    owner.deliver(sub, confirm_done(sub, BRIGHTNESS))
    assert owner.state("a.mkv") == ReviewState.PROPOSED
    assert owner.autopilot.pending() == {}


def test_a_confirm_can_only_lower_a_value(tmp_path):
    """The ladder only ever steps down, and a rung that passes is written
    with the source the detector gave the value."""
    project = _confirm_folder(tmp_path, ["a.mkv"])
    owner = Owner(project)
    _opened(owner)
    (sub,) = owner.take()
    assert confirm_mod.ladder(sub.job.start_value)[0] == BRIGHTNESS
    assert max(confirm_mod.ladder(sub.job.start_value)) == BRIGHTNESS

    owner.deliver(sub, confirm_done(sub, BRIGHTNESS - confirm_mod.STEP))
    entry = project.files["a.mkv"]
    assert entry.brightness == Brightness(BRIGHTNESS - confirm_mod.STEP, Source.DETECTED)
    assert owner.state("a.mkv") == ReviewState.PROPOSED
    assert owner.take() == []                 # its own new value is not a new question


def test_pause_holds_confirm_jobs(tmp_path):
    """GPU work that leases an OCR engine: the run the user is waiting for
    must never queue behind it (ruling C6)."""
    owner = Owner(_confirm_folder(tmp_path, ["a.mkv"]))
    _opened(owner)
    (sub,) = owner.take()
    owner.autopilot.pause()
    assert [lane for lane, _only in owner.runner.pauses] == [Lane.GPU, Lane.CPU]
    assert all(only(sub.job) for _lane, only in owner.runner.pauses)
