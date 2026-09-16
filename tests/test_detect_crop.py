import numpy as np
import pytest

from core.detect import crop


def _poly(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


FRAME = (1920, 1080)  # width, height


def test_union_covers_a_two_line_subtitle():
    one_line = [_poly(400, 980, 1500, 1030)]
    two_line = [_poly(400, 920, 1500, 970), _poly(400, 980, 1500, 1030)]
    box = crop.aggregate_box([one_line, two_line], FRAME, band_frac=0.55, settings=None)
    x, y, w, h = box
    assert y <= 920, "union must reach the upper line"
    assert y + h >= 1030, "union must reach the lower line"


def test_tightest_single_frame_would_have_clipped():
    one_line = [_poly(400, 980, 1500, 1030)]
    box_single = crop.aggregate_box([one_line], FRAME, band_frac=0.55, settings=None)
    two_line = [_poly(400, 920, 1500, 970), _poly(400, 980, 1500, 1030)]
    box_union = crop.aggregate_box([one_line, two_line], FRAME, band_frac=0.55, settings=None)
    assert box_union[3] > box_single[3]


def test_width_is_the_configured_fraction_and_centred():
    box = crop.aggregate_box([[_poly(400, 980, 1500, 1030)]], FRAME, 0.55, None)
    x, _, w, _ = box
    assert w == int(FRAME[0] * 0.70)
    assert x == (FRAME[0] - w) // 2


def test_minimum_height_floor_is_applied():
    tiny = [[_poly(900, 1000, 1000, 1004)]]
    _, _, _, h = crop.aggregate_box(tiny, FRAME, 0.55, None)
    assert h >= int(FRAME[1] * 0.05)


def test_ceiling_rejects_an_absurd_box():
    huge = [[_poly(100, 600, 1800, 1070)]]
    assert crop.aggregate_box(huge, FRAME, 0.55, None) is None


def test_no_polys_returns_none():
    assert crop.aggregate_box([[], []], FRAME, 0.55, None) is None


def test_static_content_across_all_frames_is_rejected_as_a_watermark():
    # An identical box in every frame is a logo, not a subtitle.
    same = _poly(1600, 1000, 1850, 1040)
    box = crop.aggregate_box([[same], [same], [same], [same], [same]], FRAME, 0.55, None)
    assert box is None


def test_padding_is_applied_before_the_floor():
    # Pins the exact padded value (56px) as distinct from the unpadded
    # value (54px, which happens to equal the 5%-of-1080 floor and so
    # would pass this same input even with padding deleted entirely).
    box = crop.aggregate_box([[_poly(400, 980, 1500, 1030)]], FRAME, 0.55, None)
    assert box == (288, 977, 1344, 56)


def test_watermark_five_vad_minimum_spaced_picks_is_not_confirmed_static():
    # vad.probe_times() guarantees only 0.75s minimum separation between
    # picks (_PEAK_MIN_SEPARATION_SEC), so 5 chronological picks can span
    # as little as (5-1)*0.75 = 3.0s -- well under WATERMARK_MIN_SPAN_SEC
    # (label_max_duration + margin, 6.0s by default). This is precisely
    # the boundary a prior version of this rule got wrong: it must NOT be
    # confirmed as static content, and the box must still be returned.
    same = _poly(1600, 1000, 1850, 1040)
    polys_per_frame = [[same]] * 5
    times = [10.0, 10.75, 11.5, 12.25, 13.0]
    assert times[-1] - times[0] < crop.WATERMARK_MIN_SPAN_SEC

    _union, _agreed, status = crop._union_extent(polys_per_frame, FRAME[1], 0.55, times)
    assert status == "uncertain"
    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=times)
    assert box is not None


def test_watermark_three_minimum_spaced_picks_is_not_silently_accepted():
    # 3 identical-extent picks at minimum VAD spacing span only 1.5s --
    # too little evidence to confirm a watermark, but also too little to
    # silently wave through as an ordinary clean detection: it must carry
    # the "could not judge" status/flag rather than either extreme.
    same = _poly(1600, 1000, 1850, 1040)
    polys_per_frame = [[same]] * 3
    times = [10.0, 10.75, 11.5]

    _union, _agreed, status = crop._union_extent(polys_per_frame, FRAME[1], 0.55, times)
    assert status == "uncertain"
    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=times)
    assert box is not None


def test_watermark_wide_span_is_still_confirmed_and_rejected():
    # A genuine watermark sampled across a span no single subtitle could
    # plausibly last must still be rejected -- the fix for the false
    # positive/negative boundaries must not have gutted the rule entirely.
    same = _poly(1600, 1000, 1850, 1040)
    polys_per_frame = [[same]] * 5
    times = [10.0, 30.0, 50.0, 70.0, 90.0]  # span 80s
    assert times[-1] - times[0] >= crop.WATERMARK_MIN_SPAN_SEC

    _union, _agreed, status = crop._union_extent(polys_per_frame, FRAME[1], 0.55, times)
    assert status == "confirmed"
    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=times)
    assert box is None


@pytest.mark.needs_media
@pytest.mark.slow
def test_crop_does_not_drift_from_previously_accepted_values(reference_media, detector_truth):
    """Drift guard, not an accuracy oracle: the stored crops came from the OLD
    detector, so this asserts the band did not move and the new box still
    contains the old one. A TALLER box is expected wherever two-line subtitles
    occur - that is the union rule working."""
    from videocr.utils import create_detection_engine, suppress_output
    with suppress_output():
        engine = create_detection_engine(None, True)

    deviations = []
    containment_pairs = []
    for key, entry in reference_media.items():
        truth = detector_truth.get(key)
        if not truth:
            continue
        name = entry["video"].name
        expected = truth["files"].get(name)
        if not expected:
            continue
        result = crop.detect_crop(str(entry["video"]), entry["duration"], engine)
        assert result is not None and result.box is not None, f"{key}: no crop found"
        _, y, _, h = result.box
        deviations.append((key, abs(y - expected["crop"][1]), abs(h - expected["crop"][3])))
        containment_pairs.append((key, result.box, tuple(expected["crop"])))

    assert deviations, "no reference files resolved"
    worst_y = max(d[1] for d in deviations)
    for key, dy, dh in deviations:
        print(f"{key}: dy={dy}px dh={dh}px")
    assert worst_y <= 40, f"subtitle band moved: {deviations}"

    # The new box must still contain the old one; growth is allowed, shrinkage is not.
    for key, new_box, old_box in containment_pairs:
        nx, ny, nw, nh = new_box
        ox, oy, ow, oh = old_box
        assert ny <= oy and ny + nh >= oy + oh, (
            f"{key}: new box {new_box} does not contain previously accepted {old_box}"
        )
        if nh > oh:
            print(f"{key}: box grew {oh}px -> {nh}px (expected where two-line subtitles occur)")
