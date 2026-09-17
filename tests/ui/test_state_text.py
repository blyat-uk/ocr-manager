"""Task 1: app/state_text.py -- pure, Qt-free badge/caption text (ruling B10
and the Task 1 brief's caption rules). Table-driven: every B10 row and
every caption branch (including the narrow-plateau prefix and IMPORTED/
MANUAL text) gets its own case.

No PyQt6 import anywhere: verified in a subprocess (test_state_text_module_
imports_no_qt below) rather than by asserting on this test process's own
sys.modules, since PyQt6 is already imported here by the time this file's
other tests run (conftest.py's qapp fixture, and tests/ui/test_theme_
widgets.py earlier in the session) -- only a fresh interpreter can prove
`import app.state_text` alone never pulls it in.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from app import state_text
from core.detect.brightness import BrightnessResult
from core.detect.crop import FLAG_LOW_AGREEMENT, CropResult
from core.jobs import apply as apply_mod
from core.jobs.detect_jobs import BrightnessJobResult, CropJobResult
from core.project.model import (
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
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _entry(**kwargs) -> FileEntry:
    kwargs.setdefault("name", "ZS2_-_12.mp4")
    return FileEntry(**kwargs)


# --------------------------------------------------------------------------
# badge_for() -- every B10 row
# --------------------------------------------------------------------------

def test_badge_pending_crop_running():
    entry = _entry(review=ReviewState.PENDING)
    assert state_text.badge_for(entry, running_detectors={"crop"}, done=False,
                                run_state=None) == ("finding subtitles…", "default")


def test_badge_pending_brightness_running():
    entry = _entry(review=ReviewState.PENDING)
    assert state_text.badge_for(entry, running_detectors={"brightness"}, done=False,
                                run_state=None) == ("measuring brightness…", "default")


def test_badge_pending_ranges_running():
    entry = _entry(review=ReviewState.PENDING)
    assert state_text.badge_for(entry, running_detectors={"ranges"}, done=False,
                                run_state=None) == ("matching intro/outro…", "default")


def test_badge_pending_nothing_running_waits():
    entry = _entry(review=ReviewState.PENDING)
    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=None) == ("waiting", "default")


def test_badge_pending_prefers_crop_over_brightness_and_ranges():
    # core.jobs.autopilot.PRIORITY order: crop > brightness > ranges.
    entry = _entry(review=ReviewState.PENDING)
    text, tone = state_text.badge_for(entry, running_detectors={"ranges", "brightness", "crop"},
                                      done=False, run_state=None)
    assert (text, tone) == ("finding subtitles…", "default")


def test_badge_proposed():
    entry = _entry(review=ReviewState.PROPOSED)
    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=None) == ("ready", "default")


def test_badge_flagged_crop_missing():
    entry = _entry(review=ReviewState.FLAGGED, crop=None,
                   brightness=Brightness(200, Source.MANUAL))
    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=None) == ("check crop", "warn")


def test_badge_flagged_brightness_missing():
    entry = _entry(review=ReviewState.FLAGGED, crop=Crop(0, 0, 10, 10, Source.DETECTED),
                   brightness=None)
    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=None) == ("check brightness", "warn")


def test_badge_flagged_crop_and_brightness_missing():
    entry = _entry(review=ReviewState.FLAGGED, crop=None, brightness=None)
    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=None) == ("check crop + brightness", "warn")


def test_badge_flagged_crop_present_but_blocking_flag():
    entry = _entry(review=ReviewState.FLAGGED, crop=Crop(0, 0, 10, 10, Source.DETECTED),
                   brightness=Brightness(200, Source.MANUAL),
                   flags={"crop": "low-agreement"})
    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=None) == ("check crop", "warn")


def test_badge_flagged_crop_present_informational_flag_only_not_flagged_text():
    # An informational-only crop flag does not itself make crop "blocking";
    # this entry is only FLAGGED because brightness is missing.
    entry = _entry(review=ReviewState.FLAGGED, crop=Crop(0, 0, 10, 10, Source.DETECTED),
                   brightness=None, flags={"crop": "no-speech"})
    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=None) == ("check brightness", "warn")


def test_badge_flagged_time_ranges_synthetic():
    # core/jobs/apply.py never writes entry.flags["ranges"] today (ranges is
    # not a required field), but badge_for() still recognises it -- see the
    # module docstring -- so this constructs the FileEntry directly rather
    # than going through apply.py. The ranges value must be DETECTED/HINT
    # sourced for the flag to count (mirroring core.jobs.apply's own
    # DETECTED/HINT gating, fix round 1) -- a MANUAL/IMPORTED one would not.
    entry = _entry(review=ReviewState.FLAGGED, crop=Crop(0, 0, 10, 10, Source.DETECTED),
                   brightness=Brightness(200, Source.DETECTED),
                   time_ranges=TimeRanges([], Source.DETECTED),
                   flags={"ranges": "some-future-blocking-reason"})
    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=None) == ("check time ranges", "warn")


def test_badge_flagged_time_ranges_ignored_when_manual():
    entry = _entry(review=ReviewState.FLAGGED, crop=Crop(0, 0, 10, 10, Source.DETECTED),
                   brightness=Brightness(200, Source.DETECTED),
                   time_ranges=TimeRanges([], Source.MANUAL),
                   flags={"ranges": "some-future-blocking-reason"})
    # brightness has no issue of its own here, and crop is clean too, so
    # with the leftover ranges flag correctly ignored the file should read
    # as if nothing at all were flagged for these three fields -- exercised
    # indirectly through _blocking_fields() since badge_for() only reaches
    # _flagged_text() when entry.review is already FLAGGED.
    assert "ranges" not in state_text._blocking_fields(entry)


def test_badge_reviewed():
    entry = _entry(review=ReviewState.REVIEWED)
    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=None) == ("reviewed", "good")


def test_badge_done_overrides_review_state():
    entry = _entry(review=ReviewState.PROPOSED)
    assert state_text.badge_for(entry, running_detectors=set(), done=True,
                                run_state=None) == ("done", "good")


def test_badge_skipped_overrides_everything_outside_a_run():
    entry = _entry(review=ReviewState.PROPOSED, skipped=True)
    assert state_text.badge_for(entry, running_detectors=set(), done=True,
                                run_state=None) == ("skipped", "default")


@pytest.mark.parametrize("run_state, expected", [
    ("running", ("running", "default")),
    ("failed", ("failed", "bad")),
    ("queued", ("queued", "default")),
    ("done", ("done", "good")),
])
def test_badge_run_states(run_state, expected):
    entry = _entry(review=ReviewState.PENDING)
    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=run_state) == expected


def test_badge_run_state_overrides_skipped():
    entry = _entry(review=ReviewState.PENDING, skipped=True)
    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state="running") == ("running", "default")


# --------------------------------------------------------------------------
# badge_for() driven through the REAL core.jobs.apply functions (fix round
# 1: a leftover blocking flag on a value apply.py no longer counts as
# DETECTED/HINT -- e.g. once mark_reviewed() has accepted it as MANUAL --
# must not make badge_for() name that field).
# --------------------------------------------------------------------------

def _project(entry: FileEntry) -> Project:
    return Project(path="/tmp/proj", folder=FolderSettings(), files={entry.name: entry})


def test_badge_matches_apply_rules_when_a_flagged_detected_value_becomes_manual():
    """Regression for the coordinator's fix-round-1 repro: crop DETECTED +
    "low-agreement", brightness DETECTED + stale + "differs-from-hint?",
    then apply.mark_reviewed(project, name, True). crop becomes MANUAL (its
    blocking flag string is left behind, unread -- core/jobs/apply.py never
    clears entry.flags on acceptance); brightness stays DETECTED and stale,
    so the file stays FLAGGED for brightness only. Before the fix,
    badge_for() also reported "check crop" because it looked at
    entry.flags["crop"] alone, ignoring that crop's value was no longer
    DETECTED/HINT."""
    entry = FileEntry(name="ep01.mp4")
    project = _project(entry)
    box1 = (10, 780, 1300, 60)
    box2 = (12, 782, 1300, 58)

    # 1. Clean crop detection -> DETECTED, box1.
    apply_mod.apply_crop(project, CropJobResult(
        file=entry.name,
        result=CropResult(box=box1, agreed=12, probes_used=12, flagged=None, frame_size=(1920, 1080)),
        hint=None,
    ))
    assert entry.crop == Crop(*box1, Source.DETECTED)

    # 2. Brightness measured against box1 -> DETECTED, tied to box1.
    apply_mod.apply_brightness(project, BrightnessJobResult(
        file=entry.name,
        result=BrightnessResult(value=210, plateau=(200, 220), seed=210, gate_floor=180,
                                flagged=None, curve=[(200, 0.9)]),
        tiles={}, hint_value=None, crop_box=box1,
    ))
    assert entry.brightness == Brightness(210, Source.DETECTED)

    # 3. A clean re-detect moves the crop to box2 -- auto-applicable, so it
    #    overwrites the box. Brightness's stored value was measured on
    #    box1, so it is now stale.
    apply_mod.apply_crop(project, CropJobResult(
        file=entry.name,
        result=CropResult(box=box2, agreed=12, probes_used=12, flagged=None, frame_size=(1920, 1080)),
        hint=None,
    ))
    assert entry.crop == Crop(*box2, Source.DETECTED)
    assert apply_mod.brightness_is_stale(entry)

    # 4. A later re-detect on box2 disagrees (low-agreement) -- NOT
    #    auto-applicable, so the box is untouched, but the blocking flag is
    #    stored.
    apply_mod.apply_crop(project, CropJobResult(
        file=entry.name,
        result=CropResult(box=box2, agreed=3, probes_used=12, flagged="low-agreement", frame_size=(1920, 1080)),
        hint=None,
    ))
    assert entry.crop == Crop(*box2, Source.DETECTED)
    assert entry.flags["crop"] == "low-agreement"

    # 5. A hint re-detect on brightness disagrees too ("dim-text?" is not
    #    auto-applicable either), so the stored value (still measured on
    #    box1) is untouched, but "differs-from-hint?" is composed onto
    #    entry.flags["brightness"].
    apply_mod.apply_brightness(project, BrightnessJobResult(
        file=entry.name,
        result=BrightnessResult(value=210, plateau=None, seed=210, gate_floor=180,
                                flagged="dim-text?", curve=[]),
        tiles={}, hint_value=150, crop_box=box2,
    ))
    assert entry.brightness == Brightness(210, Source.DETECTED)
    assert "differs-from-hint?" in entry.flags["brightness"]
    assert apply_mod.brightness_is_stale(entry)

    apply_mod.mark_reviewed(project, entry.name, True)
    assert entry.crop.source == Source.MANUAL
    assert entry.flags["crop"] == "low-agreement"          # leftover, no longer read
    assert entry.brightness.source == Source.DETECTED      # stale: not accepted
    assert entry.review == ReviewState.FLAGGED

    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=None) == ("check brightness", "warn")


def test_badge_ignores_leftover_flag_on_a_manual_value():
    entry = FileEntry(name="ep02.mp4")
    project = _project(entry)

    # A low-agreement crop result is never auto-applicable, so nothing is
    # written -- entry.crop stays None, but the blocking flag is stored.
    apply_mod.apply_crop(project, CropJobResult(
        file=entry.name,
        result=CropResult(box=(10, 780, 1300, 60), agreed=2, probes_used=12,
                          flagged="low-agreement", frame_size=(1920, 1080)),
        hint=None,
    ))
    assert entry.crop is None
    assert entry.flags["crop"] == "low-agreement"

    # The user sets the crop by hand; core/jobs/apply.py never clears the
    # leftover flag string.
    apply_mod.set_manual_crop(project, entry.name, (12, 782, 1300, 58))
    assert entry.crop.source == Source.MANUAL
    assert entry.flags["crop"] == "low-agreement"

    apply_mod.recompute_all(project, pending={}, ranges_pending=False)
    assert entry.review == ReviewState.FLAGGED   # brightness is still missing
    assert "crop" not in state_text._blocking_fields(entry)

    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=None) == ("check brightness", "warn")


def test_badge_missing_value_is_flagged_without_any_detector_flag():
    entry = FileEntry(name="ep03.mp4")
    project = _project(entry)

    apply_mod.set_manual_brightness(project, entry.name, 210)   # crop stays None
    apply_mod.recompute_all(project, pending={}, ranges_pending=False)
    assert entry.review == ReviewState.FLAGGED
    assert entry.flags.get("crop") is None   # never even ran a crop detector

    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=None) == ("check crop", "warn")


def test_badge_stale_brightness_is_flagged_even_though_still_detected():
    entry = FileEntry(name="ep04.mp4")
    project = _project(entry)
    box1 = (10, 780, 1300, 60)
    box2 = (12, 782, 1300, 58)

    apply_mod.apply_crop(project, CropJobResult(
        file=entry.name,
        result=CropResult(box=box1, agreed=12, probes_used=12, flagged=None, frame_size=(1920, 1080)),
        hint=None,
    ))
    apply_mod.apply_brightness(project, BrightnessJobResult(
        file=entry.name,
        result=BrightnessResult(value=210, plateau=(200, 220), seed=210, gate_floor=180,
                                flagged=None, curve=[(200, 0.9)]),
        tiles={}, hint_value=None, crop_box=box1,
    ))
    assert entry.brightness.source == Source.DETECTED
    assert not apply_mod.brightness_is_stale(entry)

    # Re-detecting the crop onto a different box makes the stored
    # brightness value stale, even though it is still DETECTED and carries
    # no blocking flag of its own.
    apply_mod.apply_crop(project, CropJobResult(
        file=entry.name,
        result=CropResult(box=box2, agreed=12, probes_used=12, flagged=None, frame_size=(1920, 1080)),
        hint=None,
    ))
    assert apply_mod.brightness_is_stale(entry)
    assert not entry.flags.get("brightness")   # no blocking flag string at all

    apply_mod.recompute_all(project, pending={}, ranges_pending=False)
    assert entry.review == ReviewState.FLAGGED

    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=None) == ("check brightness", "warn")


# --------------------------------------------------------------------------
# crop_caption()
# --------------------------------------------------------------------------

def test_crop_caption_manual_ignores_evidence():
    evidence = {"agreed": 3, "probes_used": 12, "flagged": "low-agreement"}
    assert state_text.crop_caption(evidence, Source.MANUAL) == ("set by you", 1.0, "ok")


def test_crop_caption_manual_no_evidence():
    assert state_text.crop_caption(None, Source.MANUAL) == ("set by you", 1.0, "ok")


def test_crop_caption_imported_no_evidence():
    assert state_text.crop_caption(None, Source.IMPORTED) == (
        "imported from the previous version", 1.0, "ok")


def test_crop_caption_no_evidence_no_source_falls_back():
    assert state_text.crop_caption(None, None) == ("not detected yet", 0.0, "default")


def test_crop_caption_no_evidence_detected_source_falls_back():
    # Should not happen via core/jobs/apply.py (evidence is always stored
    # alongside a written value), but crop_caption() stays defensive.
    assert state_text.crop_caption(None, Source.DETECTED) == ("not detected yet", 0.0, "default")


def test_crop_caption_evidence_informational_flag_is_ok():
    evidence = {"agreed": 12, "probes_used": 12, "flagged": "no-speech"}
    assert state_text.crop_caption(evidence, Source.DETECTED) == (
        "12 of 12 samples agree", 1.0, "ok")


def test_crop_caption_evidence_no_flag_is_ok():
    evidence = {"agreed": 12, "probes_used": 12, "flagged": None}
    assert state_text.crop_caption(evidence, Source.DETECTED) == (
        "12 of 12 samples agree", 1.0, "ok")


def test_crop_caption_evidence_blocking_flag_is_warn():
    evidence = {"agreed": 5, "probes_used": 12, "flagged": "low-agreement"}
    caption, bar, tone = state_text.crop_caption(evidence, Source.HINT)
    assert caption == "5 of 12 samples agree"
    assert bar == pytest.approx(5 / 12)
    assert tone == "warn"


def test_crop_caption_zero_probes_used_avoids_division_by_zero():
    evidence = {"agreed": 0, "probes_used": 0, "flagged": None}
    assert state_text.crop_caption(evidence, Source.DETECTED) == ("0 of 0 samples agree", 0.0, "ok")


# --------------------------------------------------------------------------
# brightness_caption()
# --------------------------------------------------------------------------

def test_brightness_caption_manual_ignores_evidence():
    evidence = {"plateau": None}
    assert state_text.brightness_caption(evidence, Source.MANUAL) == ("set by you", 1.0, "ok")


def test_brightness_caption_imported_no_evidence():
    assert state_text.brightness_caption(None, Source.IMPORTED) == (
        "imported from the previous version", 1.0, "ok")


def test_brightness_caption_no_evidence_falls_back():
    assert state_text.brightness_caption(None, None) == ("not detected yet", 0.0, "default")


def test_brightness_caption_no_plateau_is_not_verified():
    evidence = {"plateau": None, "flagged": "no-plateau?"}
    assert state_text.brightness_caption(evidence, Source.DETECTED) == ("not verified", 0.0, "warn")


def test_brightness_caption_wide_plateau_no_flag_is_ok_no_narrow_prefix():
    # A non-empty curve marks this a full (not cheap-path) detection.
    evidence = {"plateau": [180, 220], "flagged": None, "curve": [[200, 0.98]]}
    caption, bar, tone = state_text.brightness_caption(evidence, Source.DETECTED)
    assert caption == "safe range 180–220"
    assert bar == pytest.approx(40 / 60)
    assert tone == "ok"


def test_brightness_caption_narrow_plateau_prefixes_and_warns():
    evidence = {"plateau": [203, 216], "flagged": None, "curve": [[210, 0.98]]}
    caption, bar, tone = state_text.brightness_caption(evidence, Source.DETECTED)
    assert caption == "narrow safe range 203–216"
    assert bar == pytest.approx(13 / 60)
    assert tone == "warn"


def test_brightness_caption_wide_plateau_but_blocking_flag_still_warns():
    evidence = {"plateau": [180, 230], "flagged": "dim-text?", "curve": [[200, 0.98]]}
    caption, bar, tone = state_text.brightness_caption(evidence, Source.DETECTED)
    assert caption == "safe range 180–230"  # not narrow: 230-180 == 50 >= 20
    assert tone == "warn"


def test_brightness_caption_wide_plateau_informational_flag_is_ok():
    evidence = {"plateau": [180, 230], "flagged": "no-clean-threshold", "curve": [[200, 0.98]]}
    _, _, tone = state_text.brightness_caption(evidence, Source.HINT)
    assert tone == "ok"


def test_brightness_caption_bar_clamps_to_one():
    evidence = {"plateau": [100, 200], "flagged": None, "curve": [[150, 0.98]]}
    _, bar, _ = state_text.brightness_caption(evidence, Source.DETECTED)
    assert bar == 1.0


# --------------------------------------------------------------------------
# brightness_caption() -- cheap-path (folder plateau) evidence
# --------------------------------------------------------------------------

def test_brightness_caption_cheap_path_wide_plateau_reads_folder_safe_range():
    # detect_brightness(..., folder_plateau=...) never populates `curve` --
    # see core/detect/brightness.py's cheap-path branch -- so an empty (or
    # missing) curve marks this a cheap-path result whose plateau is the
    # folder's, not one measured on this file.
    evidence = {"plateau": [180, 220], "flagged": None, "curve": []}
    caption, bar, tone = state_text.brightness_caption(evidence, Source.DETECTED)
    assert caption == "folder safe range 180–220"
    assert bar == pytest.approx(40 / 60)
    assert tone == "ok"


def test_brightness_caption_cheap_path_narrow_plateau_has_no_narrow_prefix_but_warns():
    # The "narrow " prefix is reserved for the per-file (full-detection)
    # form; the cheap-path caption always reads "folder safe range ...", but
    # still turns warn when narrow.
    evidence = {"plateau": [203, 216], "flagged": None, "curve": []}
    caption, bar, tone = state_text.brightness_caption(evidence, Source.DETECTED)
    assert caption == "folder safe range 203–216"
    assert tone == "warn"


def test_brightness_caption_cheap_path_missing_curve_key_is_also_cheap_path():
    evidence = {"plateau": [180, 220], "flagged": None}
    caption, _, tone = state_text.brightness_caption(evidence, Source.DETECTED)
    assert caption == "folder safe range 180–220"
    assert tone == "ok"


# --------------------------------------------------------------------------
# crop_caption() / brightness_caption() / ranges_caption() -- blocking=True
# --------------------------------------------------------------------------

def test_crop_caption_blocking_forces_warn_even_when_evidence_is_clean():
    evidence = {"agreed": 12, "probes_used": 12, "flagged": None}
    caption, bar, tone = state_text.crop_caption(evidence, Source.HINT, blocking=True)
    assert caption == "12 of 12 samples agree"
    assert tone == "warn"


def test_crop_caption_blocking_false_default_keeps_clean_evidence_ok():
    evidence = {"agreed": 12, "probes_used": 12, "flagged": None}
    _, _, tone = state_text.crop_caption(evidence, Source.HINT)
    assert tone == "ok"


def test_crop_caption_manual_ignores_blocking():
    caption, bar, tone = state_text.crop_caption(None, Source.MANUAL, blocking=True)
    assert (caption, bar, tone) == ("set by you", 1.0, "ok")


def test_brightness_caption_blocking_forces_warn_on_wide_clean_plateau():
    evidence = {"plateau": [180, 220], "flagged": None, "curve": [[200, 0.98]]}
    caption, bar, tone = state_text.brightness_caption(evidence, Source.HINT, blocking=True)
    assert caption == "safe range 180–220"  # blocking does not add "narrow "
    assert tone == "warn"


def test_brightness_caption_blocking_forces_warn_on_cheap_path_too():
    evidence = {"plateau": [180, 220], "flagged": None, "curve": []}
    caption, _, tone = state_text.brightness_caption(evidence, Source.HINT, blocking=True)
    assert caption == "folder safe range 180–220"
    assert tone == "warn"


def test_brightness_caption_manual_ignores_blocking():
    assert state_text.brightness_caption(None, Source.MANUAL, blocking=True) == ("set by you", 1.0, "ok")


def test_ranges_caption_blocking_forces_warn_with_blocks():
    evidence = {"blocks": [
        {"kind": "intro", "matched_files": 4, "score": 0.98, "start_sec": 0.0, "end_sec": 1.0},
        {"kind": "outro", "matched_files": 4, "score": 0.96, "start_sec": 2.0, "end_sec": 3.0},
    ]}
    caption, bar, tone = state_text.ranges_caption(evidence, Source.HINT, blocking=True)
    assert caption == "intro+outro matched in 3 episodes"
    assert tone == "warn"


def test_ranges_caption_blocking_forces_warn_without_blocks():
    evidence = {"blocks": []}
    caption, bar, tone = state_text.ranges_caption(evidence, Source.HINT, blocking=True)
    assert caption == "no repeating intro/outro found"
    assert tone == "warn"


def test_ranges_caption_manual_ignores_blocking():
    assert state_text.ranges_caption({"blocks": []}, Source.MANUAL, blocking=True) == ("set by you", 1.0, "ok")


# --------------------------------------------------------------------------
# ranges_caption()
# --------------------------------------------------------------------------

def test_ranges_caption_manual_ignores_evidence():
    evidence = {"blocks": []}
    assert state_text.ranges_caption(evidence, Source.MANUAL) == ("set by you", 1.0, "ok")


def test_ranges_caption_imported_no_evidence():
    assert state_text.ranges_caption(None, Source.IMPORTED) == (
        "imported from the previous version", 1.0, "ok")


def test_ranges_caption_no_evidence_falls_back():
    assert state_text.ranges_caption(None, None) == ("not detected yet", 0.0, "default")


def test_ranges_caption_empty_blocks():
    evidence = {"blocks": []}
    assert state_text.ranges_caption(evidence, Source.DETECTED) == (
        "no repeating intro/outro found", 0.0, "default")


def test_ranges_caption_only_repeat_kind_blocks_counts_as_no_intro_outro():
    evidence = {"blocks": [{"kind": "repeat", "matched_files": 3, "score": 0.9,
                            "start_sec": 0.0, "end_sec": 1.0}]}
    assert state_text.ranges_caption(evidence, Source.DETECTED) == (
        "no repeating intro/outro found", 0.0, "default")


def test_ranges_caption_intro_and_outro_blocks():
    evidence = {"blocks": [
        {"kind": "intro", "matched_files": 4, "score": 0.98, "start_sec": 0.0, "end_sec": 153.0},
        {"kind": "outro", "matched_files": 4, "score": 0.96, "start_sec": 1200.0, "end_sec": 1400.0},
    ]}
    caption, bar, tone = state_text.ranges_caption(evidence, Source.DETECTED)
    # n = max(matched_files) - 1 = 4 - 1 = 3 (matched_files counts this file
    # itself, per core/detect/ranges/pipeline.py's Block docstring).
    assert caption == "intro+outro matched in 3 episodes"
    assert bar == pytest.approx(0.98)
    assert tone == "ok"


def test_ranges_caption_uses_max_matched_files_and_max_score_independently():
    evidence = {"blocks": [
        {"kind": "intro", "matched_files": 5, "score": 0.90, "start_sec": 0.0, "end_sec": 1.0},
        {"kind": "outro", "matched_files": 3, "score": 0.99, "start_sec": 2.0, "end_sec": 3.0},
    ]}
    caption, bar, tone = state_text.ranges_caption(evidence, Source.DETECTED)
    assert caption == "intro+outro matched in 4 episodes"
    assert bar == pytest.approx(0.99)


# --------------------------------------------------------------------------
# format_duration()
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seconds, expected", [
    (0, "0:00"),
    (8, "0:08"),
    (68, "1:08"),
    (1628, "27:08"),
    (3600, "1:00:00"),
    (3723, "1:02:03"),
    (67.6, "1:08"),
    (-5, "0:00"),
])
def test_format_duration(seconds, expected):
    assert state_text.format_duration(seconds) == expected


# --------------------------------------------------------------------------
# No PyQt6 import
# --------------------------------------------------------------------------

def test_state_text_module_imports_no_qt():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys\nimport app.state_text\nassert 'PyQt6' not in sys.modules"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"


# --------------------------------------------------------------------------
# Task 3: value texts for the inspector and the placeholder tabs
# --------------------------------------------------------------------------

def test_crop_and_brightness_texts():
    from app.state_text import brightness_text, crop_text

    assert crop_text(Crop(288, 784, 1344, 55, Source.IMPORTED)) == "288, 784 · 1344 × 55"
    assert crop_text(None) == "—"
    assert brightness_text(Brightness(211, Source.DETECTED)) == "211"
    assert brightness_text(None) == "—"


@pytest.mark.parametrize("ranges, expected", [
    (None, "whole file"),
    (TimeRanges([], Source.MANUAL), "whole file"),
    (TimeRanges([TimeRange("02:33", "23:05")], Source.DETECTED), "2:33 → 23:05"),
    (TimeRanges([TimeRange(None, "21:20")], Source.IMPORTED), "0:00 → 21:20"),
    (TimeRanges([TimeRange("1:02:03", None)], Source.MANUAL), "1:02:03 → end"),
    (TimeRanges([TimeRange("02:33", "21:20"), TimeRange("23:40", "24:00")], Source.IMPORTED),
     "2:33 → 21:20, 23:40 → 24:00"),
    (TimeRanges([TimeRange("0:10", "0:20"), TimeRange("0:30", "0:40"), TimeRange("0:50", "1:00")],
                Source.MANUAL), "0:10 → 0:20, 0:30 → 0:40 +1"),
    (TimeRanges([TimeRange("soon", "later")], Source.MANUAL), "soon → later"),
])
def test_ranges_text(ranges, expected):
    from app.state_text import ranges_text

    assert ranges_text(ranges) == expected


@pytest.mark.parametrize("media, expected", [
    (Media(1920, 888, 1628.0, 25.0), "1920×888 · 27:08 · 25 fps"),
    (Media(1920, 1080, 1418.0, 23.976), "1920×1080 · 23:38 · 23.976 fps"),
    (Media(1920, 888, 1628.0, 0.0), "1920×888 · 27:08"),
    (Media(0, 0, 0.0, 0.0), ""),
    (Media(0, 0, 3725.0, 0.0), "1:02:05"),
])
def test_media_text_omits_unknown_parts(media, expected):
    from app.state_text import media_text

    assert media_text(media) == expected


def test_clock():
    from app.state_text import clock

    assert clock(578.4) == "09:38"
    assert clock(0) == "00:00"
    assert clock(3725) == "62:05"
    assert clock(-3) == "00:00"


def test_field_blocking_follows_value_source():
    from app.state_text import field_blocking

    detected = FileEntry("a.mkv", crop=Crop(1, 2, 3, 4, Source.DETECTED), flags={"crop": FLAG_LOW_AGREEMENT})
    assert field_blocking(detected, "crop")
    manual = FileEntry("a.mkv", crop=Crop(1, 2, 3, 4, Source.MANUAL), flags={"crop": FLAG_LOW_AGREEMENT})
    assert not field_blocking(manual, "crop")
    informational = FileEntry("a.mkv", crop=Crop(1, 2, 3, 4, Source.DETECTED), flags={"crop": "no-speech"})
    assert not field_blocking(informational, "crop")
    assert not field_blocking(FileEntry("a.mkv"), "brightness")
    hinted = FileEntry("a.mkv", brightness=Brightness(200, Source.HINT), flags={"brightness": "differs-from-hint?"})
    assert field_blocking(hinted, "brightness")


def _reviewable(review) -> FileEntry:
    """A file with both required values, in `review`."""
    return FileEntry("a.mkv", crop=Crop(1, 2, 3, 4, Source.DETECTED),
                     brightness=Brightness(200, Source.DETECTED), review=review)


@pytest.mark.parametrize("review, expected", [
    (ReviewState.PENDING, False),
    (ReviewState.PROPOSED, True),
    (ReviewState.FLAGGED, True),
    (ReviewState.REVIEWED, True),
])
def test_can_mark_reviewed_waits_for_pending_files(review, expected):
    assert state_text.can_mark_reviewed(_reviewable(review)) is expected
    assert state_text.REVIEW_WAIT_TOOLTIP == "waiting for detections to finish"


@pytest.mark.parametrize("entry, missing", [
    (FileEntry("a.mkv", brightness=Brightness(200, Source.DETECTED), review=ReviewState.FLAGGED), ["crop"]),
    (FileEntry("a.mkv", crop=Crop(1, 2, 3, 4, Source.DETECTED), review=ReviewState.FLAGGED), ["brightness"]),
    (FileEntry("a.mkv", review=ReviewState.FLAGGED), ["crop", "brightness"]),
])
def test_mark_reviewed_is_refused_while_a_required_value_is_missing(entry, missing):
    """core/jobs/apply.py will not store REVIEWED while one of these is
    missing, so the button must not offer it (and must say why)."""
    assert state_text.missing_required_values(entry) == missing
    assert state_text.can_mark_reviewed(entry) is False
    assert state_text.mark_reviewed_tooltip(entry) == state_text.REVIEW_MISSING_TOOLTIP.format(
        what=" and ".join({"crop": "crop", "brightness": "brightness"}[name] for name in missing))


def test_mark_reviewed_is_refused_while_the_brightness_is_stale():
    entry = FileEntry("a.mkv", crop=Crop(9, 9, 9, 9, Source.MANUAL),
                      brightness=Brightness(200, Source.DETECTED), review=ReviewState.FLAGGED,
                      evidence={"brightness": {"value_crop_box": [1, 2, 3, 4]}})
    assert apply_mod.brightness_is_stale(entry)
    assert state_text.missing_required_values(entry) == ["brightness"]
    assert state_text.can_mark_reviewed(entry) is False


def test_a_labels_only_file_with_no_crop_can_still_be_reviewed():
    """A labels-only folder requires neither value, so its files are never
    FLAGGED for a missing one -- which is how this reads the requirement
    without being handed the folder."""
    entry = FileEntry("a.mkv", review=ReviewState.PROPOSED)
    assert state_text.missing_required_values(entry) == []
    assert state_text.can_mark_reviewed(entry) is True
    assert state_text.mark_reviewed_tooltip(entry) == ""


@pytest.mark.parametrize("entry", [
    FileEntry("a.mkv", review=ReviewState.FLAGGED),
    FileEntry("a.mkv", crop=Crop(1, 2, 3, 4, Source.DETECTED), review=ReviewState.FLAGGED),
    FileEntry("a.mkv", brightness=Brightness(200, Source.DETECTED), review=ReviewState.FLAGGED),
    FileEntry("a.mkv", crop=Crop(9, 9, 9, 9, Source.MANUAL), brightness=Brightness(200, Source.DETECTED),
              review=ReviewState.FLAGGED, evidence={"brightness": {"value_crop_box": [1, 2, 3, 4]}}),
    FileEntry("a.mkv", crop=Crop(1, 2, 3, 4, Source.DETECTED), brightness=Brightness(200, Source.DETECTED),
              review=ReviewState.FLAGGED, flags={"crop": FLAG_LOW_AGREEMENT}),
    FileEntry("a.mkv", crop=Crop(1, 2, 3, 4, Source.MANUAL), brightness=Brightness(200, Source.MANUAL),
              review=ReviewState.PROPOSED),
])
def test_can_mark_reviewed_agrees_with_what_the_apply_rules_will_store(entry):
    """Through the real rules: the button is offered exactly when pressing it
    would leave the file REVIEWED."""
    project = _project(entry)
    offered = state_text.can_mark_reviewed(entry)
    apply_mod.mark_reviewed(project, entry.name)
    apply_mod.recompute_all(project, pending={}, ranges_pending=False)
    assert (entry.review == ReviewState.REVIEWED) is offered


def test_mark_reviewed_tooltip_names_the_wait_while_pending():
    assert state_text.mark_reviewed_tooltip(_reviewable(ReviewState.PENDING)) == state_text.REVIEW_WAIT_TOOLTIP
    assert state_text.mark_reviewed_tooltip(_reviewable(ReviewState.FLAGGED)) == ""
