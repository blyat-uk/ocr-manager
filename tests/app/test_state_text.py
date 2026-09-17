"""Task 1: app/state_text.py -- pure, Qt-free badge/caption text (ruling B10
and the Task 1 brief's caption rules). Table-driven: every B10 row and
every caption branch (including the narrow-plateau prefix and IMPORTED/
MANUAL text) gets its own case.

No PyQt6 import anywhere: verified in a subprocess (test_state_text_module_
imports_no_qt below) rather than by asserting on this test process's own
sys.modules, since PyQt6 is already imported here by the time this file's
other tests run (conftest.py's qapp fixture, and tests/app/test_theme_
widgets.py earlier in the session) -- only a fresh interpreter can prove
`import app.state_text` alone never pulls it in.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from app import state_text
from core.project.model import (
    Brightness,
    Crop,
    FileEntry,
    Media,
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
    # than going through apply.py.
    entry = _entry(review=ReviewState.FLAGGED, crop=Crop(0, 0, 10, 10, Source.DETECTED),
                   brightness=Brightness(200, Source.DETECTED),
                   flags={"ranges": "some-future-blocking-reason"})
    assert state_text.badge_for(entry, running_detectors=set(), done=False,
                                run_state=None) == ("check time ranges", "warn")


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
    evidence = {"plateau": [180, 220], "flagged": None}
    caption, bar, tone = state_text.brightness_caption(evidence, Source.DETECTED)
    assert caption == "safe range 180–220"
    assert bar == pytest.approx(40 / 60)
    assert tone == "ok"


def test_brightness_caption_narrow_plateau_prefixes_and_warns():
    evidence = {"plateau": [203, 216], "flagged": None}
    caption, bar, tone = state_text.brightness_caption(evidence, Source.DETECTED)
    assert caption == "narrow safe range 203–216"
    assert bar == pytest.approx(13 / 60)
    assert tone == "warn"


def test_brightness_caption_wide_plateau_but_blocking_flag_still_warns():
    evidence = {"plateau": [180, 230], "flagged": "dim-text?"}
    caption, bar, tone = state_text.brightness_caption(evidence, Source.DETECTED)
    assert caption == "safe range 180–230"  # not narrow: 230-180 == 50 >= 20
    assert tone == "warn"


def test_brightness_caption_wide_plateau_informational_flag_is_ok():
    evidence = {"plateau": [180, 230], "flagged": "no-clean-threshold"}
    _, _, tone = state_text.brightness_caption(evidence, Source.HINT)
    assert tone == "ok"


def test_brightness_caption_bar_clamps_to_one():
    evidence = {"plateau": [100, 200], "flagged": None}
    _, bar, _ = state_text.brightness_caption(evidence, Source.DETECTED)
    assert bar == 1.0


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
