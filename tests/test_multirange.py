"""Pins get_subtitles(time_ranges=...) as output-neutral against the old
per-range approach: one get_subtitles() call per range, merged by hand.

Runs the slay_1080p_multirange fidelity case both ways -- through two
get_subtitles() calls merged by videocr.utils.merge_ass_documents (the
merge core/ocr_worker.py used to do itself before this task), and through
one get_subtitles() call with time_ranges -- and asserts the digest() of
both matches each other and the committed golden. This is the proof that
sharing one Video/engine session across ranges does not change output.
"""
import pytest

from tools.fidelity_check import GOLDEN_DIR, digest, load_cases
from videocr.api import get_subtitles
from videocr.utils import merge_ass_documents

CASE_NAME = "slay_1080p_multirange"


def _case():
    cases = {c.name: c for c in load_cases()}
    return cases[CASE_NAME]


def _kwargs(case):
    return dict(
        lang="ch", conf_threshold=95, sim_threshold=82,
        brightness_threshold=case.brightness,
        similar_image_threshold=0.3, similar_pixel_threshold=25,
        frames_to_skip=0,
        crop_x=case.crop[0], crop_y=case.crop[1],
        crop_width=case.crop[2], crop_height=case.crop[3],
        detect_labels=case.detect_labels,
    )


@pytest.mark.needs_media
@pytest.mark.slow
def test_time_ranges_matches_old_per_range_calls_and_golden(reference_media):
    case = _case()
    video = reference_media["slay"]["video"]
    kwargs = _kwargs(case)

    # Old path: one get_subtitles() call per range, merged by hand.
    old_parts = [
        get_subtitles(str(video), time_start=start, time_end=end, **kwargs)
        for start, end in case.time_ranges
    ]
    old_merged = merge_ass_documents(old_parts)

    # New path: one call, ranges handled internally in one engine session.
    new_merged = get_subtitles(
        str(video),
        time_ranges=[tuple(r) for r in case.time_ranges],
        **kwargs,
    )

    golden = (GOLDEN_DIR / f"{case.name}.ass").read_text(encoding="utf-8")

    assert digest(new_merged) == digest(old_merged) == digest(golden)
