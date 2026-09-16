import subprocess

import av
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

    _union, _agreed, status, _position_flag = crop._union_extent(polys_per_frame, FRAME[1], 0.55, times)
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

    _union, _agreed, status, _position_flag = crop._union_extent(polys_per_frame, FRAME[1], 0.55, times)
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

    _union, _agreed, status, _position_flag = crop._union_extent(polys_per_frame, FRAME[1], 0.55, times)
    assert status == "confirmed"
    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=times)
    assert box is None


def test_convergence_captures_a_two_line_subtitle_regardless_of_probe_order(monkeypatch):
    """Regression test for the round-2 fix's own regression: _spread_order()
    reorders which candidates get probed first, and with a fixed hit-count
    stop (PROBE_BATCH_SIZE == STOP_HITS == 5), whichever batch happened to
    land first -- content-blind -- became the ENTIRE contributing set.

    10 candidate timestamps t=0..9, exactly one (t=3) carries a two-line
    subtitle, the rest single-line. _spread_order(range(10))'s first batch
    is [0, 9, 4, 2, 6] -- confirmed by direct computation, and matching the
    reviewer's own reproduction -- which excludes t=3 entirely and, under
    the old fixed-count stop, would resolve as a confident one-line box
    with agreed=5 and flagged=None: the second line silently dropped. The
    union must capture the two-line extent regardless of which batch t=3
    lands in.
    """
    one_line = [_poly(400, 980, 1500, 1030)]
    two_line = [_poly(400, 920, 1500, 970), _poly(400, 980, 1500, 1030)]
    times_all = [float(t) for t in range(10)]

    assert crop._spread_order(times_all)[:5] == [0.0, 9.0, 4.0, 2.0, 6.0], (
        "test assumes this exact first batch -- if _spread_order()'s "
        "algorithm changes, update or re-derive this expectation"
    )

    last_chunk_times: list[float] = []

    def fake_grab_frames_with_times(video_path, times, band_frac, target_height, known_dims=None):
        # Identity geometry: crop==out, no offset, so _map_poly_to_full_frame
        # passes the canned polys through unchanged for simple assertions.
        last_chunk_times[:] = times
        pairs = [(t, np.zeros((4, 4, 3), dtype=np.uint8)) for t in times]
        geometry = (1920, 1080, 1920, 1080, 0, 0, 1920, 1080)
        return pairs, geometry

    class FakeEngine:
        def predict(self, frames):
            results = []
            for t in last_chunk_times:
                polys = two_line if t == 3.0 else one_line
                results.append({"dt_scores": [1.0] * len(polys), "dt_polys": polys})
            return results

    monkeypatch.setattr(crop, "_grab_frames_with_times", fake_grab_frames_with_times)

    polys_per_frame, sample_pts, raw_hits, frame_times = crop._run_round(
        "dummy.mp4", times_all, FakeEngine(), band_frac=0.55, consensus=None,
        frame_size=FRAME, settings=None,
    )
    assert 3.0 in sample_pts, "the batch containing the two-line subtitle must actually get probed"

    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=frame_times)
    assert box is not None
    x, y, w, h = box
    assert y <= 920, "union must still reach the upper line even though its batch arrived second"
    assert y + h >= 1030, "union must still reach the lower line"


def test_repositioned_subtitle_does_not_silently_vanish():
    """Round-5 regression: Y-centre-based outlier rejection (round 4)
    could drop a subtitle legitimately repositioned to avoid on-screen
    graphics, since it's a Y-centre-position minority indistinguishable
    from noise by that signal alone. Baseline (bottom-edge) clustering
    must either include it (if enough samples support it) or discard it
    with an explicit flag -- never silently."""
    normal = _poly(400, 980, 1500, 1030)        # baseline (bottom edge) 1030
    repositioned = _poly(400, 650, 1500, 700)   # baseline 700 -- still in-band (centre 675 >= 594)
    polys_per_frame = [[normal]] * 9 + [[repositioned]]
    times = [float(t) for t in range(10)]

    union, agreed, watermark_status, position_flag = crop._union_extent(
        polys_per_frame, FRAME[1], 0.55, times,
    )
    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=times)

    assert box is not None, "the dominant (normal) cluster alone must still produce a valid box"
    if position_flag is not None and crop.FLAG_OUTLIER_DISCARDED in position_flag:
        pass  # discarded, but flagged -- acceptable per the ruling
    else:
        # included: the union must actually reach the repositioned instance
        assert union[1] <= 650, f"repositioned instance silently excluded: {union}"
        assert position_flag is not None, "inclusion of a second cluster must still be flagged"
        assert crop.FLAG_MULTIPLE_POSITIONS in position_flag


def test_two_noise_detections_do_not_let_the_noise_pair_decide_the_baseline():
    """2 detections that happen to cluster together at a shared (noise)
    baseline must not out-vote 1 genuine hit just by being a 2-vs-1
    majority at this tiny a sample size -- BASELINE_CLUSTER_MIN_DOMINANT_SIZE
    gates confident exclusion on requiring the SAME baseline to recur at
    least that many times, not just "most of what's been seen so far"."""
    noise1 = _poly(300, 850, 500, 900)    # baseline 900
    noise2 = _poly(1300, 860, 1500, 910)  # baseline 910 -- clusters with noise1 (within tolerance)
    genuine = _poly(400, 980, 1500, 1030)  # baseline 1030 -- its own cluster, size 1
    polys_per_frame = [[noise1], [noise2], [genuine]]
    times = [1.0, 2.0, 3.0]

    union, agreed, watermark_status, position_flag = crop._union_extent(
        polys_per_frame, FRAME[1], 0.55, times,
    )
    assert union is not None
    assert union[3] >= 1030, f"the noise pair (only 2 members) must not decide the baseline: {union}"

    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=times)
    assert box is not None
    x, y, w, h = box
    assert y + h >= 1030


def test_two_line_frame_is_admitted_once_baseline_clustering_is_active():
    """Once there's enough data for baseline clustering's dominant-size
    gate to activate (>=BASELINE_CLUSTER_MIN_DOMINANT_SIZE), a genuine
    two-line frame sharing the SAME baseline as one-line frames from the
    same track must still be admitted by construction -- clustering by
    bottom edge doesn't care that its top edge differs."""
    one_line = [_poly(400, 980, 1500, 1030)]   # baseline 1030
    two_line = [_poly(400, 920, 1500, 970), _poly(400, 980, 1500, 1030)]  # per-frame union also baseline 1030
    polys_per_frame = [one_line] * 8 + [two_line]
    times = [float(t) for t in range(9)]

    union, agreed, watermark_status, position_flag = crop._union_extent(
        polys_per_frame, FRAME[1], 0.55, times,
    )
    assert union is not None
    assert union[1] <= 920, "the two-line frame's upper line must be admitted into the dominant cluster"
    assert position_flag is None, "sharing one baseline leaves nothing else to flag"

    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=times)
    assert box is not None
    x, y, w, h = box
    assert y <= 920
    assert y + h >= 1030


def test_discarding_does_not_shorten_probing_relative_to_unfiltered_union(monkeypatch):
    """The convergence stop (see _run_round()) must read the RAW,
    unfiltered union -- baseline clustering only runs once, afterward, in
    _union_extent()/aggregate_box(). If clustering ran inside the probing
    loop instead, discarding an outlier before the convergence check could
    make the FILTERED union look stable while the RAW union was still
    changing, stopping probing early -- exactly what a re-reviewer
    demonstrated against an earlier version of this file.

    Runs the real _run_round() twice, identical except one probe's canned
    detection is an outlier far from the rest: the outlier run must use
    at least as many probes as the no-outlier run, since its contribution
    to the raw union can only ever extend convergence, never shorten it.
    """
    normal = _poly(400, 980, 1500, 1030)
    outlier = _poly(300, 650, 500, 700)
    times_all = [float(t) for t in range(25)]
    outlier_time = 9.0  # lands in the 2nd batch under _spread_order(range(25))

    def run(with_outlier):
        last_chunk_times: list[float] = []

        def fake_grab_frames_with_times(video_path, times, band_frac, target_height, known_dims=None):
            last_chunk_times[:] = times
            pairs = [(t, np.zeros((4, 4, 3), dtype=np.uint8)) for t in times]
            geometry = (1920, 1080, 1920, 1080, 0, 0, 1920, 1080)
            return pairs, geometry

        class FakeEngine:
            def predict(self, frames):
                results = []
                for t in last_chunk_times:
                    is_outlier = with_outlier and t == outlier_time
                    polys = [outlier] if is_outlier else [normal]
                    results.append({"dt_scores": [1.0], "dt_polys": polys})
                return results

        monkeypatch.setattr(crop, "_grab_frames_with_times", fake_grab_frames_with_times)
        _polys, sample_pts, _raw_hits, _frame_times = crop._run_round(
            "dummy.mp4", times_all, FakeEngine(), band_frac=0.55, consensus=None,
            frame_size=FRAME, settings=None,
        )
        return sample_pts

    without_outlier = run(with_outlier=False)
    with_outlier = run(with_outlier=True)

    assert outlier_time in with_outlier, "the batch containing the outlier must actually get probed"
    # Strict `>`, not `>=`: a mutant that filters before the convergence
    # check again makes both runs converge identically (15 vs 15),
    # trivially satisfying `>=` while reintroducing the exact regression
    # this test is named for. Current code: 15 without the outlier, 20
    # with it -- genuinely more, not merely "not fewer".
    assert len(with_outlier) > len(without_outlier), (
        f"discarding-eligible evidence failed to extend probing: "
        f"{len(with_outlier)} probes with the outlier vs {len(without_outlier)} without"
    )


def test_second_cluster_with_multiple_members_is_included():
    """Guard against a mutant that discards every non-dominant cluster
    with a flag regardless of size: a second cluster with >=2 members,
    recurring at a baseline far enough from the dominant cluster that it
    could not qualify as an "adjacent singleton" either, is substantial
    evidence of a genuinely repositioned subtitle and MUST be included,
    not merely "possibly" included."""
    normal = _poly(400, 980, 1500, 1030)         # baseline 1030, line height 50, dominant (8 members)
    repositioned = _poly(400, 850, 1500, 900)    # baseline 900 -- gap to dominant top (980) is 80px,
                                                  # >1 line height (50px), so NOT eligible via adjacency;
                                                  # only the >=2-member rule can admit it here
    polys_per_frame = [[normal]] * 8 + [[repositioned]] * 2
    times = [float(t) for t in range(10)]

    union, agreed, watermark_status, position_flag = crop._union_extent(
        polys_per_frame, FRAME[1], 0.55, times,
    )
    assert union is not None
    assert union[1] <= 850, f"a second cluster with >=2 members must be included: {union}"
    assert position_flag is not None
    assert crop.FLAG_MULTIPLE_POSITIONS in position_flag

    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=times)
    assert box is not None


def test_consensus_relaxes_convergence_but_never_stops_on_hit_count(monkeypatch):
    """Round-6 fix 1, reproducing the reviewer's exact scenario: 10
    candidates, a two-line subtitle only at t=3 (which _spread_order()
    places in the 2nd batch, not the 1st -- see
    test_convergence_captures_a_two_line_subtitle_regardless_of_probe_order),
    consensus set to match the ONE-LINE shape. The old consensus fast
    path stopped after a single batch (5 probes) the moment raw_hits
    reached 2 and a provisional one-line box matched consensus -- zero
    stability confirmed, and using a box that had already been through
    baseline clustering rather than the raw evidence. Must now use the
    same probe count and produce the same box as the no-consensus case.
    """
    def one_line_for(t):
        # Varying width per probe (like different dialogue lines) avoids
        # the watermark check's "identical extent everywhere" trigger,
        # without changing the subtitle's Y-position at all.
        w = 300 + int(t) * 37 % 400
        x0 = 400 + int(t) * 13 % 200
        return [_poly(x0, 980, x0 + w, 1030)]

    two_line = [_poly(400, 920, 1500, 970), _poly(400, 980, 1500, 1030)]
    times_all = [float(t) for t in range(10)]
    consensus = [(977.0 / 1080, 56.0 / 1080)] * 3  # matches the one-line shape

    last_chunk_times: list[float] = []

    def fake_grab_frames_with_times(video_path, times, band_frac, target_height, known_dims=None):
        last_chunk_times[:] = times
        pairs = [(t, np.zeros((4, 4, 3), dtype=np.uint8)) for t in times]
        geometry = (1920, 1080, 1920, 1080, 0, 0, 1920, 1080)
        return pairs, geometry

    class FakeEngine:
        def predict(self, frames):
            results = []
            for t in last_chunk_times:
                polys = two_line if t == 3.0 else one_line_for(t)
                results.append({"dt_scores": [1.0] * len(polys), "dt_polys": polys})
            return results

    monkeypatch.setattr(crop, "_grab_frames_with_times", fake_grab_frames_with_times)
    polys_per_frame, sample_pts, raw_hits, frame_times = crop._run_round(
        "dummy.mp4", times_all, FakeEngine(), band_frac=0.55, consensus=consensus,
        frame_size=FRAME, settings=None,
    )
    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=frame_times)

    assert len(sample_pts) > 5, "must not stop after a single batch on hit count alone"
    assert 3.0 in sample_pts, "the batch containing the two-line subtitle must actually get probed"
    assert box is not None
    x, y, w, h = box
    assert y <= 920, "the two-line subtitle's upper line must not be clipped"
    assert y + h >= 1030


def test_consensus_stop_reads_the_unfiltered_union_not_a_clustered_box(monkeypatch):
    """Round-6 fix 1, reproducing the reviewer's second scenario: the
    first batch of 5 holds 4 one-line frames and 1 upper-line-only frame
    (a singleton, non-adjacent under a mutant / an earlier version of
    baseline clustering that discarded it outright). Consensus set to
    match the one-line shape. Evaluating the consensus check through
    aggregate_box() (which clusters) sees only the 4 one-line frames and
    stops after 5 probes with the upper line clipped; evaluating it on
    the raw union catches the mismatch (the raw union is taller than
    consensus expects) and keeps probing.
    """
    def one_line_for(t):
        w = 300 + int(t) * 37 % 400
        x0 = 400 + int(t) * 13 % 200
        return [_poly(x0, 980, x0 + w, 1030)]

    upper_only = [_poly(400, 920, 1500, 970)]
    times_all = [float(t) for t in range(10)]
    outlier_time = 4.0  # lands in the 1st batch under _spread_order(range(10)) == [0,9,4,2,6,...]
    consensus = [(977.0 / 1080, 56.0 / 1080)] * 3

    last_chunk_times: list[float] = []

    def fake_grab_frames_with_times(video_path, times, band_frac, target_height, known_dims=None):
        last_chunk_times[:] = times
        pairs = [(t, np.zeros((4, 4, 3), dtype=np.uint8)) for t in times]
        geometry = (1920, 1080, 1920, 1080, 0, 0, 1920, 1080)
        return pairs, geometry

    class FakeEngine:
        def predict(self, frames):
            results = []
            for t in last_chunk_times:
                polys = upper_only if t == outlier_time else one_line_for(t)
                results.append({"dt_scores": [1.0] * len(polys), "dt_polys": polys})
            return results

    monkeypatch.setattr(crop, "_grab_frames_with_times", fake_grab_frames_with_times)
    polys_per_frame, sample_pts, raw_hits, frame_times = crop._run_round(
        "dummy.mp4", times_all, FakeEngine(), band_frac=0.55, consensus=consensus,
        frame_size=FRAME, settings=None,
    )
    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=frame_times)

    assert len(sample_pts) > 5, "must not stop after a single batch on a clustered provisional box"
    assert box is not None
    x, y, w, h = box
    assert y <= 920, "the upper-line-only frame must not be clipped from the final box"
    assert y + h >= 1030


def test_consensus_never_relaxes_toward_a_union_shorter_than_the_series(monkeypatch):
    """Round-7 fix: `_consistent_with_consensus()` only rejected a union
    that was too TALL relative to consensus (`h_frac > med_h *
    CONSENSUS_MAX_HEIGHT_RATIO`) -- it never rejected one that was too
    SHORT. Reproduces the review's exact numbers: consensus fixed at the
    padded TWO-LINE shape (y_frac=917/1080, h_frac=116/1080 -- see
    test_union_covers_a_two_line_subtitle's fixture), checked against a
    raw ONE-LINE union (y_frac=977/1080, h_frac=56/1080). That one-line
    union clears both the too-tall check (56/1080 is nowhere near
    1.5x116/1080) and the Y-deviation check (|0.905-0.849| < 0.10), so the
    old, one-sided rule returned True -- RELAXING the stop requirement
    exactly when the consensus is evidence the box should be TALLER, not
    proof it's safe to stop early.

    End-to-end: 15 candidate timestamps, a two-line subtitle only at
    t=6.0, which _spread_order() places in the 3rd batch (not the 1st or
    2nd). Before the fix: the run incorrectly stabilizes after 2 batches
    (10 probes) on the one-line-only evidence seen so far, NEVER reaching
    the batch containing the two-line frame, and resolves to the clipped
    one-line box (288, 977, 1344, 56) with no signal anything was missed.
    After the fix, the relaxed (CONSENSUS_STABLE_ROUNDS) stop must not
    fire on a union that disagrees with consensus in the "too short"
    direction either, so probing continues, the two-line batch is
    reached, and the box matches the no-consensus result exactly:
    (288, 917, 1344, 116).
    """
    def one_line_for(t):
        # Varying width per probe (like different dialogue lines) avoids
        # the watermark check's "identical extent everywhere" trigger,
        # without changing the subtitle's Y-position at all.
        w = 300 + int(t) * 37 % 400
        x0 = 400 + int(t) * 13 % 200
        return [_poly(x0, 980, x0 + w, 1030)]

    two_line = [_poly(400, 920, 1500, 970), _poly(400, 980, 1500, 1030)]
    times_all = [float(t) for t in range(15)]
    two_line_time = 6.0  # lands in the 3rd batch under _spread_order(range(15))
    assert crop._spread_order(times_all)[10:15] == [6.0, 8.0, 9.0, 11.0, 13.0], (
        "test assumes this exact 3rd-batch composition -- if _spread_order()'s "
        "algorithm changes, update or re-derive this expectation"
    )
    consensus = [(917.0 / 1080, 116.0 / 1080)] * 3  # matches the TWO-LINE shape

    last_chunk_times: list[float] = []

    def fake_grab_frames_with_times(video_path, times, band_frac, target_height, known_dims=None):
        last_chunk_times[:] = times
        pairs = [(t, np.zeros((4, 4, 3), dtype=np.uint8)) for t in times]
        geometry = (1920, 1080, 1920, 1080, 0, 0, 1920, 1080)
        return pairs, geometry

    class FakeEngine:
        def predict(self, frames):
            results = []
            for t in last_chunk_times:
                polys = two_line if t == two_line_time else one_line_for(t)
                results.append({"dt_scores": [1.0] * len(polys), "dt_polys": polys})
            return results

    monkeypatch.setattr(crop, "_grab_frames_with_times", fake_grab_frames_with_times)
    polys_per_frame, sample_pts, raw_hits, frame_times = crop._run_round(
        "dummy.mp4", times_all, FakeEngine(), band_frac=0.55, consensus=consensus,
        frame_size=FRAME, settings=None,
    )
    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=frame_times)

    assert two_line_time in sample_pts, (
        "consensus must not relax the stop toward a union SHORTER than the "
        "series -- the batch containing the two-line subtitle must still "
        "get probed"
    )
    assert box == (288, 917, 1344, 116), (
        f"the two-line subtitle must not be silently clipped to the "
        f"one-line box: got {box}"
    )


def test_consistent_with_consensus_rejects_a_union_too_short_for_the_series():
    """Unit-level pin, isolated from _run_round()'s batching: the exact
    numbers from the review. A one-line union must NOT be judged
    consistent with a two-line consensus."""
    assert crop._consistent_with_consensus(
        y_frac=977.0 / 1080, h_frac=56.0 / 1080,
        consensus=[(917.0 / 1080, 116.0 / 1080)],
    ) is False


def test_adjacent_singleton_upper_line_is_included_not_clipped():
    """Round-6 fix 2: an isolated singleton hit directly adjacent to the
    dominant cluster (within ~1 detected line height) is a missing line,
    not noise -- an earlier version discarded every singleton
    unconditionally, clipping a genuine two-line subtitle back to one
    line whenever OCR only caught the upper line on a single probe."""
    normal = [_poly(400, 980, 1500, 1030)]      # baseline 1030, line height 50, dominant (8 members)
    upper_only = [_poly(400, 920, 1500, 970)]   # baseline 970 -- gap to dominant top (980) is 10px, adjacent
    polys_per_frame = [normal] * 8 + [upper_only]
    times = [float(t) for t in range(9)]

    union, agreed, watermark_status, position_flag = crop._union_extent(
        polys_per_frame, FRAME[1], 0.55, times,
    )
    assert union is not None
    assert union[1] <= 920, f"the adjacent upper line must be included, not clipped: {union}"
    assert position_flag is not None
    assert crop.FLAG_MULTIPLE_POSITIONS in position_flag

    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=times)
    assert box is not None
    x, y, w, h = box
    assert y <= 920
    assert y + h >= 1030


def test_slay_stray_hit_geometry_stays_rejected_after_adjacency_fix():
    """Regression guard for the adjacency fix above: a genuinely unrelated
    detection, geometrically far from the dominant cluster, must still be
    discarded. Slay's real stray hit (the one that originally motivated
    outlier rejection -- see task-2-report.md) measured ~2.4 detected
    line heights above the dominant cluster's top edge; this reproduces
    that ratio synthetically to confirm the adjacency exception does not
    reopen the original noise-vs-signal failure."""
    normal = [_poly(400, 980, 1500, 1030)]  # baseline 1030, line height 50 (1030-980), dominant
    # 2.4 line heights above the dominant union's top edge (980):
    # 980 - 2.4*50 = 860.
    stray = [_poly(686, 830, 704, 860)]     # baseline 860, in-band (centre 845 >= 594)
    polys_per_frame = [normal] * 8 + [stray]
    times = [float(t) for t in range(9)]

    union, agreed, watermark_status, position_flag = crop._union_extent(
        polys_per_frame, FRAME[1], 0.55, times,
    )
    assert union is not None
    assert union[1] >= 970, f"a genuinely distant stray hit must stay rejected: {union}"
    assert position_flag is not None
    assert crop.FLAG_OUTLIER_DISCARDED in position_flag

    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=times)
    assert box is not None


def test_run_round_cancel_check_stops_within_a_couple_of_batches(monkeypatch):
    """Task-3 review ruling A: cancellation must be checked BETWEEN probe
    batches, not only after the whole candidate list (or
    MAX_PROBES_PER_ROUND) is exhausted. 200 candidate timestamps, no text
    ever detected (nothing to converge on, so without cancellation this
    would run every batch up to MAX_PROBES_PER_ROUND=30 probes);
    cancel_check() reports cancelled starting on its 3rd call, so at most
    2 batches worth of probes (10) should be fetched."""
    times_all = [float(t) for t in range(200)]

    def fake_grab_frames_with_times(video_path, times, band_frac, target_height, known_dims=None):
        pairs = [(t, np.zeros((4, 4, 3), dtype=np.uint8)) for t in times]
        geometry = (1920, 1080, 1920, 1080, 0, 0, 1920, 1080)
        return pairs, geometry

    class FakeEngine:
        def predict(self, frames):
            return [{"dt_scores": [], "dt_polys": []} for _ in frames]

    monkeypatch.setattr(crop, "_grab_frames_with_times", fake_grab_frames_with_times)

    calls = {"n": 0}

    def cancel_check():
        calls["n"] += 1
        return calls["n"] > 2

    polys_per_frame, sample_pts, raw_hits, frame_times = crop._run_round(
        "dummy.mp4", times_all, FakeEngine(), band_frac=0.55, consensus=None,
        frame_size=FRAME, settings=None, cancel_check=cancel_check,
    )

    assert len(sample_pts) <= crop.PROBE_BATCH_SIZE * 3, (
        f"cancellation must stop probing within a couple of batches, got "
        f"{len(sample_pts)} probes ({len(sample_pts) / crop.PROBE_BATCH_SIZE} batches)"
    )


def test_hit_pts_reflects_a_real_hit_after_a_fallback_round(monkeypatch):
    """Task-3 review ruling D: CropResult.hit_pts must be the timestamps
    that actually contributed to the kept union, NOT the full probe
    history (sample_pts). Forces the speech-guided round to find zero
    hits (so detect_crop() falls back to uniform probing, flagged
    speech-probes-exhausted), and only ONE of the uniform-fallback probes
    -- not the first one tried -- carries real text. sample_pts[0] is
    therefore guaranteed to be a no-text frame by construction; hit_pts[0]
    must not be.
    """
    hit_text = [_poly(400, 980, 1500, 1030)]
    duration = 60.0
    uniform_times = crop._uniform_probe_times(duration)
    hit_time = 25.5
    assert hit_time in uniform_times and hit_time != uniform_times[0]

    monkeypatch.setattr(crop, "_probe_dimensions", lambda video_path: (1920, 1080))
    monkeypatch.setattr(
        crop.vad, "probe_times",
        lambda video_path, duration_sec, window_frac=(0.4, 0.6): [1.0, 2.0, 3.0],
    )

    last_chunk_times: list[float] = []

    def fake_grab_frames_with_times(video_path, times, band_frac, target_height, known_dims=None):
        last_chunk_times[:] = times
        pairs = [(t, np.zeros((4, 4, 3), dtype=np.uint8)) for t in times]
        geometry = (1920, 1080, 1920, 1080, 0, 0, 1920, 1080)
        return pairs, geometry

    class FakeEngine:
        def predict(self, frames):
            results = []
            for t in last_chunk_times:
                if t == hit_time:
                    results.append({"dt_scores": [1.0], "dt_polys": hit_text})
                else:
                    results.append({"dt_scores": [], "dt_polys": []})
            return results

    monkeypatch.setattr(crop, "_grab_frames_with_times", fake_grab_frames_with_times)

    result = crop.detect_crop("dummy.mp4", duration, FakeEngine())

    assert result.flagged is not None and crop.FLAG_SPEECH_PROBES_EXHAUSTED in result.flagged
    assert result.box is not None
    assert result.sample_pts[0] != hit_time, (
        "sanity check: the old (buggy) sample_pts[0] basis is NOT the hit frame"
    )
    assert result.hit_pts, "hit_pts must not be empty when a box was found"
    assert result.hit_pts[0] == hit_time, (
        f"hit_pts[0] must be the real hit timestamp ({hit_time}), got {result.hit_pts[0]}"
    )
    assert result.frame_size == (1920, 1080)


def _detect_crop_with_fakes(monkeypatch, vad_times, predict_fn, duration=60.0, cancel_check=None):
    """Shared plumbing for the end-to-end box=None/flag tests below: drives
    the real detect_crop() orchestration with vad.probe_times(),
    _probe_dimensions() and _grab_frames_with_times() faked out, and
    `predict_fn(t) -> (scores, polys)` controlling what the engine "sees"
    at each probed timestamp."""
    monkeypatch.setattr(crop, "_probe_dimensions", lambda video_path: (1920, 1080))
    monkeypatch.setattr(
        crop.vad, "probe_times",
        lambda video_path, duration_sec, window_frac=(0.4, 0.6): vad_times,
    )

    last_chunk_times: list[float] = []

    def fake_grab_frames_with_times(video_path, times, band_frac, target_height, known_dims=None):
        last_chunk_times[:] = times
        pairs = [(t, np.zeros((4, 4, 3), dtype=np.uint8)) for t in times]
        geometry = (1920, 1080, 1920, 1080, 0, 0, 1920, 1080)
        return pairs, geometry

    class FakeEngine:
        def predict(self, frames):
            results = []
            for t in last_chunk_times:
                scores, polys = predict_fn(t)
                results.append({"dt_scores": scores, "dt_polys": polys})
            return results

    monkeypatch.setattr(crop, "_grab_frames_with_times", fake_grab_frames_with_times)

    return crop.detect_crop("dummy.mp4", duration, FakeEngine(), cancel_check=cancel_check)


def _detect_crop_with_mocked_rounds(monkeypatch, round_outcomes, vad_times=None, duration=60.0,
                                     cancel_check=None):
    """Mocks _run_round() itself (not the frame-grab/engine layer), so
    each of detect_crop()'s up-to-3 sequential round calls can be
    scripted independently, including side effects -- used for the
    cancellation-mid-round-2/3 tests below, where exactly WHEN
    cancellation becomes visible to detect_crop() matters more than
    _run_round()'s own internal batching (already covered by
    test_run_round_cancel_check_stops_within_a_couple_of_batches).

    `round_outcomes`: list of zero-arg callables, one per expected
    _run_round() call, each returning
    (polys_per_frame, sample_pts, raw_hits, frame_times); may have side
    effects (e.g. flipping a cancellation flag). Returns (result,
    number_of_run_round_calls) -- the latter lets a test assert a later
    round was never reached at all.
    """
    monkeypatch.setattr(crop, "_probe_dimensions", lambda video_path: (1920, 1080))
    monkeypatch.setattr(
        crop.vad, "probe_times",
        lambda video_path, duration_sec, window_frac=(0.4, 0.6): (
            vad_times if vad_times is not None else [1.0, 2.0, 3.0]
        ),
    )
    calls = {"n": 0}

    def fake_run_round(video_path, times, det_engine, band_frac, consensus, frame_size,
                        settings, known_dims=None, cancel_check=None, fetcher=None):
        idx = calls["n"]
        calls["n"] += 1
        assert idx < len(round_outcomes), f"_run_round called more times than scripted ({idx + 1})"
        return round_outcomes[idx]()

    monkeypatch.setattr(crop, "_run_round", fake_run_round)
    result = crop.detect_crop("dummy.mp4", duration, object(), cancel_check=cancel_check)
    return result, calls["n"]


def test_confirmed_watermark_sets_static_content_flag_end_to_end(monkeypatch):
    """Task-3 round-2 review, finding 1: _union_extent_detailed()'s
    confirmed-watermark branch returns an empty kept_idx (by design --
    nothing is "kept" into a real union), which made detect_crop()'s old
    `agreed > 0` gate silently skip FLAG_STATIC_CONTENT for exactly the
    case it exists to flag: a confirmed watermark left result.flagged as
    None instead of "static-content" -- a silent no-box-no-reason result.
    Drives detect_crop() end-to-end (not just the pure _union_extent()
    layer, which existing tests already cover but discard the count) with
    an identical extent on every sampled frame, spread across well more
    than WATERMARK_MIN_SPAN_SEC.
    """
    same = _poly(1600, 1000, 1850, 1040)
    watermark_times = crop._uniform_probe_times(60.0)  # 24.0..36.0, 12s span

    result = _detect_crop_with_fakes(
        monkeypatch, watermark_times, lambda t: ([1.0], [same]),
    )

    assert result.box is None, "a confirmed watermark must not produce a box"
    assert result.flagged is not None, (
        "a confirmed watermark must never resolve with flagged=None -- "
        "a silent no-box-no-reason result"
    )
    assert crop.FLAG_STATIC_CONTENT in result.flagged, (
        f"expected {crop.FLAG_STATIC_CONTENT!r} in flagged, got {result.flagged!r}"
    )
    assert crop.FLAG_UNKNOWN_REJECTION not in result.flagged, (
        "a KNOWN no-box scenario must never reach the generic safety-net fallback"
    )


def test_ceiling_exceeded_sets_a_flag_end_to_end(monkeypatch):
    """The other box=None exit that depends on `agreed > 0`: real
    (non-watermark) hits exist, but the resulting box is too tall.
    Varying x-position per probe (like the existing consensus tests) keeps
    this from also being classified as a confirmed watermark."""
    def predict_fn(t):
        x0 = 100 + (int(t * 10) % 300)
        return [1.0], [_poly(x0, 600, x0 + 1700, 1070)]  # ~470px tall, way over the 25% ceiling

    ceiling_times = crop._uniform_probe_times(60.0)
    result = _detect_crop_with_fakes(monkeypatch, ceiling_times, predict_fn)

    assert result.box is None, "an absurdly tall box must still be rejected"
    assert result.flagged is not None, "ceiling-exceeded must never resolve with flagged=None"
    assert crop.FLAG_CEILING_EXCEEDED in result.flagged, (
        f"expected {crop.FLAG_CEILING_EXCEEDED!r} in flagged, got {result.flagged!r}"
    )
    assert crop.FLAG_UNKNOWN_REJECTION not in result.flagged, (
        "a KNOWN no-box scenario must never reach the generic safety-net fallback"
    )


def test_no_hits_anywhere_sets_a_flag_end_to_end(monkeypatch):
    """Third box=None exit: nothing is ever detected, even after every
    fallback round (uniform probing, then the full-frame retry). Already
    covered indirectly by the fallback-flag composition logic (independent
    of the agreed>0 gate this round's regression was in), included here so
    every box=None exit of detect_crop() is checked the same, explicit
    way."""
    result = _detect_crop_with_fakes(monkeypatch, [1.0, 2.0, 3.0], lambda t: ([], []))

    assert result.box is None
    assert result.flagged is not None, "exhausting every fallback must never resolve with flagged=None"
    assert crop.FLAG_SPEECH_PROBES_EXHAUSTED in result.flagged
    assert crop.FLAG_TOP_POSITIONED in result.flagged
    assert crop.FLAG_UNKNOWN_REJECTION not in result.flagged, (
        "a KNOWN no-box scenario must never reach the generic safety-net fallback"
    )


def test_no_speech_available_and_no_hits_sets_a_flag_end_to_end(monkeypatch):
    """Fourth box=None exit: vad.probe_times() finds true digital silence
    (returns []), so detect_crop() falls back to uniform probing
    immediately (flagged="no-speech") -- and nothing is found there
    either, so the full-frame retry also runs and also finds nothing."""
    result = _detect_crop_with_fakes(monkeypatch, [], lambda t: ([], []))

    assert result.box is None
    assert result.flagged is not None, "no-speech + no hits must never resolve with flagged=None"
    assert crop.FLAG_NO_SPEECH in result.flagged
    assert crop.FLAG_TOP_POSITIONED in result.flagged
    assert crop.FLAG_UNKNOWN_REJECTION not in result.flagged, (
        "a KNOWN no-box scenario must never reach the generic safety-net fallback"
    )


def test_cancellation_during_round_1_sets_cancelled_flag_end_to_end(monkeypatch):
    """Task-3 review round 3, the reported regression: cancel_check()
    returning True from the very start makes _run_round() break before
    any batch on round 1, leaving raw_hits==0 -- both the round-2 and
    round-3 entry gates then evaluate False *because* cancellation is
    True, so neither FLAG_SPEECH_PROBES_EXHAUSTED nor FLAG_TOP_POSITIONED
    is ever composed. Before this fix, nothing else explained the
    resulting box=None: result.flagged came back None. FLAG_CANCELLED
    exists precisely to name this case explicitly rather than leaving it
    to be inferred from an absent flag."""
    result = _detect_crop_with_fakes(
        monkeypatch, [1.0, 2.0, 3.0], lambda t: ([], []),
        cancel_check=lambda: True,
    )

    assert result.box is None
    assert result.flagged is not None, (
        "cancellation during round 1 must never resolve with flagged=None"
    )
    assert crop.FLAG_CANCELLED in result.flagged, (
        f"expected {crop.FLAG_CANCELLED!r} in flagged, got {result.flagged!r}"
    )
    assert crop.FLAG_UNKNOWN_REJECTION not in result.flagged, (
        "a KNOWN no-box scenario must never reach the generic safety-net fallback"
    )


def test_cancellation_during_round_2_sets_cancelled_flag_end_to_end(monkeypatch):
    """Cancellation taking effect specifically during round 2 (the
    uniform-probing fallback): round 1 completes normally with zero hits
    (not cancelled), round 2 starts, and is cut short by cancellation
    partway through -- round 3 must then be skipped (cancellation, not
    "found something"), and the result must carry both
    FLAG_SPEECH_PROBES_EXHAUSTED (round 2 did run) and FLAG_CANCELLED
    (it didn't finish on its own)."""
    cancel_state = {"cancelled": False}

    def round_1():
        return [], [1.0, 2.0, 3.0], 0, [1.0, 2.0, 3.0]

    def round_2():
        cancel_state["cancelled"] = True  # cancellation happens DURING this round
        return [], [4.0, 5.0], 0, [4.0, 5.0]

    result, n_calls = _detect_crop_with_mocked_rounds(
        monkeypatch, [round_1, round_2],
        cancel_check=lambda: cancel_state["cancelled"],
    )

    assert n_calls == 2, "round 3 must be skipped once cancellation is visible after round 2"
    assert result.box is None
    assert result.flagged is not None
    assert crop.FLAG_SPEECH_PROBES_EXHAUSTED in result.flagged
    assert crop.FLAG_CANCELLED in result.flagged, (
        f"expected {crop.FLAG_CANCELLED!r} in flagged, got {result.flagged!r}"
    )
    assert crop.FLAG_TOP_POSITIONED not in result.flagged, "round 3 never ran"
    assert crop.FLAG_UNKNOWN_REJECTION not in result.flagged, (
        "a KNOWN no-box scenario must never reach the generic safety-net fallback"
    )


def test_cancellation_during_round_3_sets_cancelled_flag_end_to_end(monkeypatch):
    """Cancellation taking effect specifically during round 3 (the
    full-frame retry): rounds 1 and 2 both complete normally with zero
    hits (not cancelled), round 3 starts and is cut short -- the result
    must carry FLAG_SPEECH_PROBES_EXHAUSTED, FLAG_TOP_POSITIONED (round 3
    did run) AND FLAG_CANCELLED (it didn't finish on its own)."""
    cancel_state = {"cancelled": False}

    def round_1():
        return [], [1.0, 2.0, 3.0], 0, [1.0, 2.0, 3.0]

    def round_2():
        return [], [4.0, 5.0], 0, [4.0, 5.0]

    def round_3():
        cancel_state["cancelled"] = True  # cancellation happens DURING this round
        return [], [6.0], 0, [6.0]

    result, n_calls = _detect_crop_with_mocked_rounds(
        monkeypatch, [round_1, round_2, round_3],
        cancel_check=lambda: cancel_state["cancelled"],
    )

    assert n_calls == 3
    assert result.box is None
    assert result.flagged is not None
    assert crop.FLAG_SPEECH_PROBES_EXHAUSTED in result.flagged
    assert crop.FLAG_TOP_POSITIONED in result.flagged
    assert crop.FLAG_CANCELLED in result.flagged, (
        f"expected {crop.FLAG_CANCELLED!r} in flagged, got {result.flagged!r}"
    )
    assert crop.FLAG_UNKNOWN_REJECTION not in result.flagged, (
        "a KNOWN no-box scenario must never reach the generic safety-net fallback"
    )


def test_safety_net_fallback_flag_fires_for_a_truly_unanticipated_no_box_path(monkeypatch, caplog):
    """Task-3 review round 3, rulings 2 and 4: test the safety net itself,
    not a specific case. Forces raw_hits > 0 (so none of the fallback-round
    flags fire) while the internal union/aggregation machinery is
    monkeypatched to report "nothing found" regardless (so neither
    FLAG_STATIC_CONTENT nor FLAG_CEILING_EXCEEDED fires either) -- i.e. a
    box=None outcome that NO currently-known branch accounts for,
    simulating a future no-box path nobody remembered to flag. The
    structural post-condition just before CropResult is built must catch
    this and compose FLAG_UNKNOWN_REJECTION, and log a warning naming the
    file."""
    monkeypatch.setattr(crop, "_probe_dimensions", lambda video_path: (1920, 1080))
    monkeypatch.setattr(
        crop.vad, "probe_times",
        lambda video_path, duration_sec, window_frac=(0.4, 0.6): [1.0, 2.0, 3.0],
    )

    def fake_run_round(video_path, times, det_engine, band_frac, consensus, frame_size,
                        settings, known_dims=None, cancel_check=None, fetcher=None):
        # raw_hits > 0 so neither fallback round ever runs -- only
        # _union_extent_detailed()/aggregate_box() decide box/flag from
        # here, and those are the ones forced to "find nothing" below.
        return [["something"]], [1.0], 1, [1.0]

    monkeypatch.setattr(crop, "_run_round", fake_run_round)
    monkeypatch.setattr(
        crop, "_union_extent_detailed",
        lambda polys_per_frame, frame_h, cutoff_frac, sample_times=None: (None, [], None, None),
    )
    monkeypatch.setattr(
        crop, "aggregate_box",
        lambda polys_per_frame, frame_size, band_frac, settings, sample_times=None: None,
    )

    with caplog.at_level("WARNING"):
        result = crop.detect_crop("dummy-unflagged-path.mp4", 60.0, object())

    assert result.box is None
    assert result.flagged is not None
    assert crop.FLAG_UNKNOWN_REJECTION in result.flagged, (
        f"expected the safety-net fallback flag, got {result.flagged!r}"
    )
    assert any(
        "dummy-unflagged-path.mp4" in rec.message and rec.levelname == "WARNING"
        for rec in caplog.records
    ), "a warning naming the file must be logged when the fallback fires"


# --------------------------------------------------------------------------
# Task 2b: tolerant convergence
# --------------------------------------------------------------------------

def _drive_run_round_by_rank(monkeypatch, n_candidates, polys_for_rank, consensus=None):
    """Runs the real _run_round() over `n_candidates` timestamps, with the
    frame-grab layer faked out and the engine answering by each probe's
    VISITATION rank (its index in _spread_order()), so a test can script
    exactly what each successive batch of PROBE_BATCH_SIZE probes sees."""
    times_all = [float(t) for t in range(n_candidates)]
    rank = {t: i for i, t in enumerate(crop._spread_order(times_all))}
    last_chunk_times: list[float] = []

    def fake_grab_frames_with_times(video_path, times, band_frac, target_height, known_dims=None):
        last_chunk_times[:] = times
        pairs = [(t, np.zeros((4, 4, 3), dtype=np.uint8)) for t in times]
        return pairs, (1920, 1080, 1920, 1080, 0, 0, 1920, 1080)

    class FakeEngine:
        def predict(self, frames):
            results = []
            for t in last_chunk_times:
                polys = polys_for_rank(rank[t])
                results.append({"dt_scores": [1.0] * len(polys), "dt_polys": polys})
            return results

    monkeypatch.setattr(crop, "_grab_frames_with_times", fake_grab_frames_with_times)
    return crop._run_round(
        "dummy.mp4", times_all, FakeEngine(), band_frac=0.55, consensus=consensus,
        frame_size=FRAME, settings=None,
    )


def _dialogue_line(rank, top=980.0, bottom=1030.0):
    """One subtitle line whose WIDTH differs on every probe, the way
    successive dialogue lines do -- each later probe is a little wider than
    any before it, so the raw union's horizontal extent grows every batch."""
    half = 150 + 6 * rank
    return [_poly(960 - half, top, 960 + half, bottom)]


def _vertical_tolerance_px():
    return crop.CONVERGENCE_VERTICAL_TOLERANCE_FRAC * FRAME[1]


# The largest batch-to-batch vertical growth of the raw union measured on the
# reference corpus without a new line appearing -- pure edge jitter (XWZ 169,
# 12.4px of a 2160px frame -- see task-2b-report.md).
LARGEST_MEASURED_BATCH_JITTER_FRAC = 12.4 / 2160.0


def test_dialogue_width_variance_and_edge_jitter_converge_before_the_probe_cap(monkeypatch):
    """Task 2b: exact-equality convergence never fired on real dialogue (slay
    hit the 30-probe cap on every run) because each new line's width nudges
    the union's horizontal edges. The crop's width is a fixed centred fraction
    of the frame, so horizontal growth cannot change the box; and the
    vertical edges jitter by a few px between frames of the same line. Both
    must count as stable: here the width grows on every probe, and batches 2
    and 3 each move the top and bottom edges by the largest jitter measured
    on the corpus, so the run must stop after exactly
    CONVERGENCE_STABLE_ROUNDS stable batches, well under the cap."""
    jitter = LARGEST_MEASURED_BATCH_JITTER_FRAC * FRAME[1]

    def polys_for_rank(rank):
        batch = min(rank // crop.PROBE_BATCH_SIZE, 2)
        return _dialogue_line(rank, top=980.0 - jitter * batch, bottom=1030.0 + jitter * batch)

    _polys, sample_pts, _hits, _times = _drive_run_round_by_rank(monkeypatch, 60, polys_for_rank)

    expected = crop.PROBE_BATCH_SIZE * (crop.CONVERGENCE_STABLE_ROUNDS + 1)
    assert len(sample_pts) == expected < crop.MAX_PROBES_PER_ROUND, (
        f"ordinary width variance and edge jitter must converge after {expected} probes, "
        f"got {len(sample_pts)}"
    )


# The smallest single-line height measured on the reference corpus (XWZ 168
# and 169, 87px of a 2160px frame -- see task-2b-report.md). A second subtitle line
# grows the union by at least this much, so no tolerance may swallow it.
SMALLEST_MEASURED_LINE_HEIGHT_FRAC = 87.0 / 2160.0


@pytest.mark.parametrize("edge,growth", [
    ("top", "just-over-tolerance"),
    ("bottom", "just-over-tolerance"),
    ("top", "second-line"),
])
def test_growth_beyond_the_convergence_tolerance_is_never_treated_as_stable(monkeypatch, edge, growth):
    """Task 2b gate: a union that grows by more than the tolerance across
    batches must not count as stable. Batches 1-2 hold steady (1 stable
    round); batch 3 grows one vertical edge by either 1px more than the
    tolerance or by one second subtitle line (the smallest line height
    measured on the corpus). That batch must reset stability, so probing
    cannot stop before batch 5 -- a tolerance loose enough to absorb it would
    stop after batch 3, exactly the "width-variance-sized tolerance swallows
    a second line" failure this guards."""
    tol = _vertical_tolerance_px()
    amount = tol + 1.0 if growth == "just-over-tolerance" else SMALLEST_MEASURED_LINE_HEIGHT_FRAC * FRAME[1]

    def polys_for_rank(rank):
        if rank == 12:  # batch 3
            if edge == "top":
                return _dialogue_line(rank, top=980.0 - amount)
            return _dialogue_line(rank, bottom=1030.0 + amount)
        return _dialogue_line(rank)

    polys_per_frame, sample_pts, _hits, frame_times = _drive_run_round_by_rank(monkeypatch, 60, polys_for_rank)

    assert len(sample_pts) >= crop.PROBE_BATCH_SIZE * 5, (
        f"growth of {amount:.1f}px (tolerance {tol:.1f}px) on the {edge} edge was treated as "
        f"stable: probing stopped after {len(sample_pts)} probes"
    )
    union, *_ = crop._union_extent(polys_per_frame, FRAME[1], 0.55, frame_times)
    if edge == "top":
        assert union[1] == pytest.approx(980.0 - amount, abs=0.01)
    else:
        assert union[3] == pytest.approx(1030.0 + amount, abs=0.01)


def test_convergence_tolerance_is_cumulative_so_creeping_growth_cannot_mask_a_late_second_line(monkeypatch):
    """Task 2b gate, the per-batch loophole: if stability compared each batch
    only against the batch before it, a union creeping upward by 0.6x the
    tolerance per batch would look stable twice in a row and stop after 3
    batches -- never reaching the two-line frame in batch 4. Growth is
    measured against the union at the START of the stable streak, so 1.2x
    the tolerance of cumulative creep resets it and the second line is
    captured."""
    tol = _vertical_tolerance_px()
    two_line = [_poly(400, 920, 1500, 970), _poly(400, 980, 1500, 1030)]

    def polys_for_rank(rank):
        if rank == 7:    # batch 2
            return _dialogue_line(rank, top=980.0 - 0.6 * tol)
        if rank == 12:   # batch 3
            return _dialogue_line(rank, top=980.0 - 1.2 * tol)
        if rank == 17:   # batch 4
            return two_line
        return _dialogue_line(rank)

    polys_per_frame, sample_pts, _hits, frame_times = _drive_run_round_by_rank(monkeypatch, 60, polys_for_rank)
    box = crop.aggregate_box(polys_per_frame, FRAME, 0.55, None, sample_times=frame_times)

    assert len(sample_pts) > crop.PROBE_BATCH_SIZE * 3, (
        f"creeping growth (1.2x tolerance over two batches) was treated as stable: stopped after "
        f"{len(sample_pts)} probes"
    )
    assert box is not None and box[1] <= 920, f"the late second line was clipped: {box}"


# --------------------------------------------------------------------------
# Task 2b: probe fetching -- recorded timestamps and the persistent path
# --------------------------------------------------------------------------

def test_failed_grabs_are_not_recorded_but_still_spend_the_probe_budget(monkeypatch):
    """A grab that fails returns no frame, so nothing was analysed at its time
    and it must not appear in sample_pts -- but it was still attempted, so it
    counts against MAX_PROBES_PER_ROUND, or an unreadable file would walk its
    whole candidate list."""
    times_all = [float(t) for t in range(200)]
    requested: list[float] = []

    def half_failing_grab(video_path, times, band_frac, target_height, known_dims=None):
        requested.extend(times)
        pairs = [(t, np.zeros((4, 4, 3), dtype=np.uint8)) for t in times if int(t) % 2 == 0]
        return pairs, (1920, 1080, 1920, 1080, 0, 0, 1920, 1080)

    class NoTextEngine:
        def predict(self, frames):
            return [{"dt_scores": [], "dt_polys": []} for _ in frames]

    monkeypatch.setattr(crop, "_grab_frames_with_times", half_failing_grab)
    _polys, sample_pts, _hits, _times = crop._run_round(
        "dummy.mp4", times_all, NoTextEngine(), band_frac=0.55, consensus=None,
        frame_size=FRAME, settings=None,
    )

    assert len(requested) == crop.MAX_PROBES_PER_ROUND, (
        f"failed grabs must spend the probe budget: {len(requested)} probes requested"
    )
    assert sample_pts == [t for t in requested if int(t) % 2 == 0], (
        "sample_pts must list exactly the probes that returned a frame"
    )


class _RecordingFetcher:
    """Stands in for crop._PersistentFrameFetcher: returns a frame for every
    requested time, recorded under that requested time, with identity
    geometry at the source's own resolution."""
    instances: list = []

    def __init__(self, video_path, known_dims, pool_size=crop.PERSISTENT_POOL_SIZE):
        self.video_path = video_path
        self.known_dims = known_dims
        self.pool_size = pool_size
        self.closed = False
        self.fetched: list[float] = []
        self.last_pairs: list = []
        _RecordingFetcher.instances.append(self)

    def fetch(self, times, band_frac, target_height):
        w, h = self.known_dims
        pairs = [(t, np.zeros((4, 4, 3), dtype=np.uint8)) for t in times]
        self.fetched.extend(t for t, _ in pairs)
        self.last_pairs = pairs
        return pairs, (w, h, w, h, 0, 0, w, h)

    def close(self):
        self.closed = True


def _detect_crop_at_resolution(monkeypatch, dims, predict=None):
    """detect_crop() with vad/ffprobe faked, a spy on the one-shot grab
    layer and _RecordingFetcher standing in for the persistent one. Returns
    (result, one_shot_calls)."""
    _RecordingFetcher.instances = []
    w, h = dims
    requested = [float(t) for t in range(10, 40)]
    one_shot_calls = {"n": 0}
    last_pairs: list = []

    monkeypatch.setattr(crop, "_probe_dimensions", lambda video_path: dims)
    monkeypatch.setattr(crop.vad, "probe_times",
                        lambda video_path, duration_sec, window_frac=(0.4, 0.6): list(requested))
    monkeypatch.setattr(crop, "_PersistentFrameFetcher", _RecordingFetcher, raising=False)

    def fake_grab_frames_with_times(video_path, times, band_frac, target_height, known_dims=None):
        one_shot_calls["n"] += 1
        pairs = [(t, np.zeros((4, 4, 3), dtype=np.uint8)) for t in times]
        last_pairs[:] = pairs
        return pairs, (w, h, w, h, 0, 0, w, h)

    monkeypatch.setattr(crop, "_grab_frames_with_times", fake_grab_frames_with_times)

    class FakeEngine:
        def predict(self, frames):
            if predict is not None:
                predict()
            pairs = _RecordingFetcher.instances[-1].last_pairs if _RecordingFetcher.instances else last_pairs
            line_top, line_bottom = h * 0.92, h * 0.96
            return [
                {"dt_scores": [1.0], "dt_polys": [_poly(w * 0.4 - int(t), line_top, w * 0.6 + int(t), line_bottom)]}
                for t, _ in pairs
            ]

    result = crop.detect_crop("dummy-4k.mp4", 60.0, FakeEngine())
    return result, one_shot_calls["n"]


@pytest.mark.parametrize("dims,persistent", [
    ((1920, 888), False),    # slay: one-shot grabs measured faster
    ((1920, 1080), False),   # the crossover boundary itself stays on one-shot grabs
    ((3840, 2160), True),    # xwz: persistent containers measured faster
])
def test_fetch_strategy_follows_the_measured_resolution_crossover(monkeypatch, dims, persistent):
    """Task 2b requirement 2: persistent containers only where they measurably
    win. Above the crossover every probe goes through the persistent fetcher
    (never a one-shot ffmpeg grab), what it returns is what the result
    records, and it is closed once detection finishes; at or below it,
    nothing is opened and one-shot grabs are used exactly as before."""
    result, one_shot_calls = _detect_crop_at_resolution(monkeypatch, dims)

    assert result.box is not None
    if persistent:
        assert len(_RecordingFetcher.instances) == 1, "one persistent fetcher per detect_crop() call"
        fetcher = _RecordingFetcher.instances[0]
        assert one_shot_calls == 0, f"{dims}: {one_shot_calls} batches went through one-shot grabs"
        assert result.sample_pts == fetcher.fetched
        assert set(result.hit_pts) <= set(fetcher.fetched)
        assert fetcher.closed, "the persistent containers must be released when detection ends"
    else:
        assert _RecordingFetcher.instances == [], f"{dims}: persistent containers opened below the crossover"
        assert one_shot_calls > 0


def test_persistent_fetcher_is_closed_even_when_detection_raises(monkeypatch):
    """Containers hold several decoded 4K reference frames each; an exception
    from the detection engine (which detect_crop() lets propagate to the
    adapter) must not leak them."""
    def boom():
        raise RuntimeError("engine failed")

    with pytest.raises(RuntimeError, match="engine failed"):
        _detect_crop_at_resolution(monkeypatch, (3840, 2160), predict=boom)

    assert len(_RecordingFetcher.instances) == 1
    assert _RecordingFetcher.instances[0].closed


def test_persistent_fetcher_open_failure_falls_back_to_one_shot_grabs(monkeypatch, caplog):
    """If the persistent containers cannot be opened (PyAV cannot read what
    the ffmpeg CLI can), detection must still run on one-shot grabs, and say
    so in the log rather than fail the file."""
    class FailingFetcher:
        def __init__(self, video_path, known_dims, pool_size=None):
            raise OSError("cannot open container")

    monkeypatch.setattr(crop, "_probe_dimensions", lambda video_path: (3840, 2160))
    monkeypatch.setattr(crop.vad, "probe_times",
                        lambda video_path, duration_sec, window_frac=(0.4, 0.6): [float(t) for t in range(10, 40)])
    monkeypatch.setattr(crop, "_PersistentFrameFetcher", FailingFetcher, raising=False)
    last_chunk: list[float] = []

    def fake_grab_frames_with_times(video_path, times, band_frac, target_height, known_dims=None):
        last_chunk[:] = times
        pairs = [(t, np.zeros((4, 4, 3), dtype=np.uint8)) for t in times]
        return pairs, (3840, 2160, 3840, 2160, 0, 0, 3840, 2160)

    class FakeEngine:
        def predict(self, frames):
            return [{"dt_scores": [1.0], "dt_polys": [_poly(1500 - int(t), 1990, 2300 + int(t), 2080)]}
                    for t in last_chunk]

    monkeypatch.setattr(crop, "_grab_frames_with_times", fake_grab_frames_with_times)
    with caplog.at_level("WARNING"):
        result = crop.detect_crop("dummy-unopenable.mp4", 60.0, FakeEngine())

    assert result.box is not None
    assert any("dummy-unopenable.mp4" in rec.message and rec.levelname == "WARNING" for rec in caplog.records)


def test_grab_frames_opens_no_more_persistent_containers_than_requested_times(monkeypatch):
    """Each persistent 4K container holds several decoded reference frames;
    grabbing one frame must not open a pool of five."""
    opened: list[int] = []

    class CountingFetcher:
        def __init__(self, video_path, known_dims, pool_size=crop.PERSISTENT_POOL_SIZE):
            opened.append(pool_size)

        def fetch(self, times, band_frac, target_height):
            pairs = [(t, np.zeros((4, 4, 3), dtype=np.uint8)) for t in times]
            return pairs, (3840, 2160, 3840, 2160, 0, 0, 3840, 2160)

        def close(self):
            pass

    monkeypatch.setattr(crop, "_probe_dimensions", lambda video_path: (3840, 2160))
    monkeypatch.setattr(crop, "_PersistentFrameFetcher", CountingFetcher, raising=False)

    assert len(crop.grab_frames("dummy-4k.mp4", [12.0])) == 1
    assert opened == [1], f"containers opened for a single requested time: {opened}"


# Real PyAV decoding against real one-shot ffmpeg grabs, on small synthetic
# clips: 3s each with a GOP of 50 frames, so an accurate seek has to decode
# forward from a keyframe (skipping non-reference frames or not) to reach
# every requested time below. Each clip lists encoder choices in order of
# preference; the first one this ffmpeg build accepts is used.
_CLIPS = {
    "h264-bframes": dict(size="320x240", rate="25", pix_fmt="yuv420p",
                         encoders=[["-c:v", "libx264", "-g", "50", "-bf", "3"]]),
    # 24000/1001 fps in mp4 gets a 1/24000 time base: frame times are not
    # whole milliseconds, which is what a recorded time has to survive.
    "h264-ntsc-film": dict(size="320x240", rate="24000/1001", pix_fmt="yuv420p",
                           encoders=[["-c:v", "libx264", "-g", "50", "-bf", "3"]]),
    "hevc-bpyramid": dict(size="320x240", rate="25", pix_fmt="yuv420p",
                          encoders=[["-c:v", "libx265", "-x265-params",
                                     "keyint=50:bframes=4:b-pyramid=1:log-level=error"]]),
    "hevc-temporal-layers": dict(size="320x240", rate="25", pix_fmt="yuv420p",
                                 encoders=[["-c:v", "libx265", "-x265-params",
                                            "keyint=50:bframes=4:b-pyramid=1:temporal-layers=3:log-level=error"]]),
    # 10-bit like the 4K reference sources, a non-millisecond time base, and
    # large enough for an odd crop plus a real downscale (see
    # test_persistent_fetcher_matches_one_shot_on_10bit_hevc_with_an_odd_crop_and_a_real_downscale).
    "hevc10-ntsc-film": dict(size="640x360", rate="24000/1001", pix_fmt="yuv420p10le",
                             encoders=[["-c:v", "libx265", "-x265-params",
                                        "keyint=50:bframes=4:b-pyramid=1:log-level=error"]]),
    # Identity on a codec that decodes every frame. Whether non-reference
    # skipping stays OFF for AV1 is guarded separately, by
    # test_non_reference_skipping_stays_off_where_it_returns_a_different_frame.
    "av1": dict(size="320x240", rate="25", pix_fmt="yuv420p",
                encoders=[["-c:v", "libsvtav1", "-g", "50"],
                          ["-c:v", "libaom-av1", "-cpu-used", "8", "-g", "50"]]),
}

# Requested probe times, deliberately out of order (so a reused decoder seeks
# backward). On the 24000/1001 clips, 0.9, 2.2 and 0.3 land on frames whose
# own timestamps (0.917583, 2.210542, 0.333667) round UP to the next
# millisecond -- recording the frame's timestamp instead of the requested
# time would re-fetch the NEXT frame.
_ROUND_TRIP_TIMES = [0.9, 2.2, 0.3, 1.37, 0.05, 2.03, 0.21, 2.9]


@pytest.fixture(scope="module")
def encoded_clips(tmp_path_factory):
    """name -> (path, None), or (None, why no encoder could produce it)."""
    out_dir = tmp_path_factory.mktemp("fetch-clips")
    clips = {}
    for name, spec in _CLIPS.items():
        out = out_dir / f"{name}.mp4"
        tried = []
        for codec_args in spec["encoders"]:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", f"testsrc2=size={spec['size']}:rate={spec['rate']}:duration=3",
                 "-pix_fmt", spec["pix_fmt"], *codec_args, str(out)],
                capture_output=True,
            )
            if proc.returncode == 0:
                clips[name] = (out, None)
                break
            last_line = (proc.stderr.decode(errors="replace").strip().splitlines() or ["no output"])[-1]
            tried.append(f"{codec_args[1]}: {last_line}")
        else:
            clips[name] = (None, "; ".join(tried))
    return clips


def _clip(encoded_clips, name):
    path, reason = encoded_clips[name]
    if path is None:
        pytest.skip(f"no ffmpeg encoder could produce the {name} clip (tried {reason})")
    return path


def _one_shot(video, t, geometry):
    _w, _h, cw, ch, cx, cy, ow, oh = geometry
    return crop._grab_one(str(video), t, cw, ch, cx, cy, ow, oh)


@pytest.mark.parametrize("pool_size", [crop.PERSISTENT_POOL_SIZE, 1])
@pytest.mark.parametrize("clip", sorted(_CLIPS))
def test_persistent_fetcher_frames_match_one_shot_and_recorded_times_refetch_them(encoded_clips, clip, pool_size):
    """Task 2b requirement 4, on real decoding. Every frame the persistent
    fetcher returns is pixel-identical to the one-shot accurate grab at the
    requested time (switching fetch strategy cannot change what is analysed),
    and the time it records re-fetches exactly that frame -- through the
    persistent path and through the one-shot path. pool_size=1 reuses a
    single decoder across the backward seeks in _ROUND_TRIP_TIMES."""
    video = _clip(encoded_clips, clip)
    dims = crop._probe_dimensions(str(video))

    fetcher = crop._PersistentFrameFetcher(str(video), dims, pool_size=pool_size)
    try:
        pairs, geometry = fetcher.fetch(_ROUND_TRIP_TIMES, 0.55, crop.TARGET_HEIGHT)
        again, _ = fetcher.fetch([t for t, _ in pairs], 0.55, crop.TARGET_HEIGHT)
    finally:
        fetcher.close()

    assert len(pairs) == len(again) == len(_ROUND_TRIP_TIMES)
    for req, (recorded, frame), (_t, refetched) in zip(_ROUND_TRIP_TIMES, pairs, again):
        assert np.array_equal(frame, _one_shot(video, req, geometry)), (
            f"{clip}: frame for t={req} differs from the one-shot accurate grab"
        )
        assert np.array_equal(refetched, frame), (
            f"{clip}: recorded time {recorded} (requested {req}) re-fetches a different frame"
        )
        assert np.array_equal(_one_shot(video, recorded, geometry), frame), (
            f"{clip}: recorded time {recorded} (requested {req}) re-fetches a different frame one-shot"
        )
        assert recorded == req, f"{clip}: recorded {recorded}, requested {req}"
        # testsrc2 repeats the odd frame at 24000/1001 fps, so require the
        # frame to differ from at least one neighbour, not both.
        assert not (np.array_equal(frame, _one_shot(video, req + 0.05, geometry))
                    and np.array_equal(frame, _one_shot(video, max(0.0, req - 0.05), geometry))), (
            f"{clip}: clip frames are not distinguishable around t={req}, test proves nothing"
        )


# AV1 encoders in order of preference for the skipping guard below. Only a
# clip that actually contains non-reference frames can catch skipping being
# enabled for AV1: svt-av1's hierarchical mini-GOPs and av1_nvenc's B-frames
# do; libaom-av1 emitted none in any mode tried (good quality, realtime,
# 16/35 lag frames, pyramid height 4), so it is last and only ever produces
# a visible skip, never a silent pass.
_AV1_SKIP_GUARD_ENCODERS = [
    ["-c:v", "libsvtav1", "-g", "50"],
    ["-c:v", "av1_nvenc", "-bf", "3", "-g", "50"],
    ["-c:v", "libaom-av1", "-cpu-used", "8", "-g", "50"],
]


def test_non_reference_skipping_stays_off_where_it_returns_a_different_frame(monkeypatch, tmp_path):
    """libdav1d (AV1) never outputs a frame it was told to skip, so skipping
    non-reference frames on the way to an accurate-seek target silently lands
    on a LATER frame. The persistent fetcher must decode every AV1 frame. The
    clip is first checked to be able to show the difference -- skipping
    forced on must lose at least one frame -- otherwise the guard would pass
    no matter what, and the test skips saying why for each encoder tried."""
    outcomes = []
    for codec_args in _AV1_SKIP_GUARD_ENCODERS:
        encoder = codec_args[1]
        video = tmp_path / f"{encoder}.mp4"
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=3",
             "-pix_fmt", "yuv420p", *codec_args, str(video)],
            capture_output=True,
        )
        if proc.returncode != 0:
            last_line = (proc.stderr.decode(errors="replace").strip().splitlines() or ["no output"])[-1]
            outcomes.append(f"{encoder}: cannot encode ({last_line})")
            continue
        dims = crop._probe_dimensions(str(video))
        with av.open(str(video)) as container:
            decoder_name = container.streams.video[0].codec_context.name

        def lost_frames():
            fetcher = crop._PersistentFrameFetcher(str(video), dims)
            try:
                pairs, geometry = fetcher.fetch(_ROUND_TRIP_TIMES, 0.55, crop.TARGET_HEIGHT)
            finally:
                fetcher.close()
            assert len(pairs) == len(_ROUND_TRIP_TIMES)
            return sum(not np.array_equal(frame, _one_shot(video, t, geometry)) for t, frame in pairs)

        with monkeypatch.context() as forced:
            forced.setattr(crop, "NONREF_SKIP_EXACT_CODECS", crop.NONREF_SKIP_EXACT_CODECS | {decoder_name})
            lost_when_forced = lost_frames()
        if lost_when_forced == 0:
            outcomes.append(f"{encoder}: clip has no non-reference frames, cannot show the difference")
            continue

        assert lost_frames() == 0, (
            f"{decoder_name} ({encoder} clip): the persistent fetcher returned a different frame than the "
            f"one-shot grab -- non-reference skipping must stay off for this decoder"
        )
        return
    pytest.skip("no AV1 clip able to expose non-reference skipping could be encoded here: " + "; ".join(outcomes))


def test_persistent_fetcher_matches_one_shot_on_10bit_hevc_with_an_odd_crop_and_a_real_downscale(encoded_clips):
    """The identity above only exercises a no-resampling geometry. PyAV
    bundles its own FFmpeg while one-shot grabs use the system ffmpeg, so
    guard the conversions that can drift between them: 10-bit to bgr24, an
    odd crop height, and a real downscale."""
    video = _clip(encoded_clips, "hevc10-ntsc-film")
    dims = crop._probe_dimensions(str(video))
    fetcher = crop._PersistentFrameFetcher(str(video), dims)
    try:
        pairs, geometry = fetcher.fetch(_ROUND_TRIP_TIMES, 0.37, 97)
    finally:
        fetcher.close()

    assert geometry == (640, 360, 640, 133, 0, 227, 466, 97), "expected an odd crop and a real downscale"
    assert len(pairs) == len(_ROUND_TRIP_TIMES)
    for req, (_t, frame) in zip(_ROUND_TRIP_TIMES, pairs):
        assert np.array_equal(frame, _one_shot(video, req, geometry)), (
            f"t={req}: persistent frame differs from the one-shot grab"
        )


class _FrameRecordingEngine:
    """Detects one in-band text line in every frame it is given (slightly
    different width each time, so nothing looks like a watermark), and keeps
    a copy of every frame in the order it analysed them."""

    def __init__(self):
        self.analysed: list[np.ndarray] = []

    def predict(self, frames):
        results = []
        for frame in frames:
            self.analysed.append(frame.copy())
            h, w = frame.shape[:2]
            k = len(self.analysed)
            results.append({"dt_scores": [1.0], "dt_polys": [_poly(w * 0.2 + k, h * 0.75, w * 0.8 - k, h * 0.9)]})
        return results


@pytest.mark.parametrize("path", ["one-shot", "persistent"])
@pytest.mark.parametrize("clip", ["h264-ntsc-film", "hevc10-ntsc-film"])
def test_recorded_sample_and_hit_pts_refetch_exactly_the_analysed_frames(monkeypatch, encoded_clips, clip, path):
    """Task 2b requirement 4, end to end: every time detect_crop() records in
    sample_pts (and hit_pts, a subset) must re-fetch, through the same fetch
    path, exactly the frame that was analysed -- on a time base where frame
    timestamps are not whole milliseconds. The UI slider, the filmstrip and
    the full-frame retry all re-request these times."""
    video = _clip(encoded_clips, clip)
    monkeypatch.setattr(crop, "_prefers_persistent_fetch", lambda frame_size: path == "persistent", raising=False)
    monkeypatch.setattr(crop.vad, "probe_times",
                        lambda video_path, duration_sec, window_frac=(0.4, 0.6): list(_ROUND_TRIP_TIMES))
    engine = _FrameRecordingEngine()

    result = crop.detect_crop(str(video), 3.0, engine)

    assert result.box is not None
    assert len(result.sample_pts) == len(engine.analysed) == len(_ROUND_TRIP_TIMES)
    assert result.hit_pts and set(result.hit_pts) <= set(result.sample_pts)
    refetched = crop.grab_frames(str(video), result.sample_pts)
    assert len(refetched) == len(result.sample_pts)
    for t, analysed, again in zip(result.sample_pts, engine.analysed, refetched):
        assert np.array_equal(again, analysed), (
            f"{path}: recorded time {t} re-fetches a different frame than the one analysed"
        )


def test_persistent_path_falls_back_to_one_shot_grabs_when_a_whole_batch_fails(monkeypatch, encoded_clips, caplog):
    """PyAV can open a file it then cannot decode (e.g. "cannot decode unknown
    codec"). If every probe of a batch fails on the persistent path, that
    batch and the rest of the file must go through one-shot grabs, with a
    warning -- not be silently dropped and misreported as
    speech-probes-exhausted / top-positioned?."""
    video = _clip(encoded_clips, "h264-bframes")
    monkeypatch.setattr(crop, "_prefers_persistent_fetch", lambda frame_size: True, raising=False)
    monkeypatch.setattr(crop.vad, "probe_times",
                        lambda video_path, duration_sec, window_frac=(0.4, 0.6): list(_ROUND_TRIP_TIMES))
    attempts = {"n": 0}

    def undecodable(self, t, geometry):
        attempts["n"] += 1
        raise ValueError("cannot decode unknown codec")

    monkeypatch.setattr(crop._PersistentDecoder, "grab", undecodable)

    with caplog.at_level("WARNING"):
        result = crop.detect_crop(str(video), 3.0, _FrameRecordingEngine())

    assert result.box is not None
    for flag in (crop.FLAG_SPEECH_PROBES_EXHAUSTED, crop.FLAG_TOP_POSITIONED, crop.FLAG_UNKNOWN_REJECTION):
        assert flag not in (result.flagged or ""), f"misreported as {result.flagged!r}"
    assert result.sample_pts and set(result.sample_pts) == set(_ROUND_TRIP_TIMES)
    assert attempts["n"] <= crop.PROBE_BATCH_SIZE, (
        f"{attempts['n']} persistent grabs attempted: after a wholly failed batch the rest of the "
        f"file must use one-shot grabs"
    )
    assert any(
        rec.levelname == "WARNING" and str(video) in rec.getMessage() and "one-shot" in rec.getMessage()
        for rec in caplog.records
    ), "falling back to one-shot grabs must be logged"


def test_persistent_fetcher_honours_the_container_start_offset(offset_video):
    """Probe times, like `ffmpeg -ss`, are relative to the container's start;
    the offset_video fixture's timestamps all begin at 1.5s. A persistent
    decoder seeking in raw stream time would land on the wrong frame."""
    requested = [0.05, 0.21, 0.33]
    dims = crop._probe_dimensions(str(offset_video))
    fetcher = crop._PersistentFrameFetcher(str(offset_video), dims)
    try:
        pairs, geometry = fetcher.fetch(requested, 0.55, crop.TARGET_HEIGHT)
    finally:
        fetcher.close()

    assert [t for t, _ in pairs] == requested
    for req, (_t, frame) in zip(requested, pairs):
        assert np.array_equal(frame, _one_shot(offset_video, req, geometry))


def test_persistent_fetcher_drops_and_reports_a_failed_grab_keeping_request_order(encoded_clips, caplog):
    """grab_frames()'s contract carries over: a failed grab (here, past the
    end of the clip) is dropped and logged, not raised, and the frames that
    did succeed stay in request order."""
    video = _clip(encoded_clips, "h264-bframes")
    dims = crop._probe_dimensions(str(video))
    fetcher = crop._PersistentFrameFetcher(str(video), dims)
    try:
        with caplog.at_level("WARNING"):
            pairs, geometry = fetcher.fetch([1.01, 99.0, 0.5], 0.55, crop.TARGET_HEIGHT)
    finally:
        fetcher.close()

    assert [t for t, _ in pairs] == [1.01, 0.5]
    for t, frame in pairs:
        assert np.array_equal(frame, _one_shot(video, t, geometry))
    assert any("99.000" in rec.message and rec.levelname == "WARNING" for rec in caplog.records)


def test_grab_frames_uses_persistent_containers_above_the_crossover_with_identical_frames(
        monkeypatch, encoded_clips):
    """grab_frames() (public) gains the persistent path without changing its
    contract: frames in request order, identical to one-shot grabs. The clip
    is small, so the crossover policy is forced on to exercise the path."""
    video = _clip(encoded_clips, "hevc-bpyramid")
    times = [2.03, 0.21, 1.37]
    expected = crop.grab_frames(str(video), times)

    one_shot_calls = {"n": 0}
    real_grab_one = crop._grab_one

    def counting_grab_one(*args, **kwargs):
        one_shot_calls["n"] += 1
        return real_grab_one(*args, **kwargs)

    monkeypatch.setattr(crop, "_grab_one", counting_grab_one)
    monkeypatch.setattr(crop, "_prefers_persistent_fetch", lambda frame_size: True, raising=False)
    frames = crop.grab_frames(str(video), times)

    assert one_shot_calls["n"] == 0, "above the crossover grab_frames() must not spawn one-shot grabs"
    assert len(frames) == len(expected) == len(times)
    assert all(np.array_equal(a, b) for a, b in zip(frames, expected))


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


# --- Files without an audio stream ------------------------------------------


class _BrightBoxDetector:
    """Detection engine stand-in: one polygon around the bright pixels of
    each frame it is given, scored 1.0; nothing on a dark frame."""

    def __init__(self):
        self.frames_seen = 0

    def predict(self, frames):
        results = []
        for frame in frames:
            self.frames_seen += 1
            ys, xs = np.nonzero(frame.max(axis=2) > 128)
            if len(xs) == 0:
                results.append({"dt_scores": [], "dt_polys": []})
            else:
                results.append({"dt_scores": [1.0],
                                "dt_polys": [_poly(xs.min(), ys.min(), xs.max(), ys.max())]})
        return results


def test_detect_crop_on_a_video_only_file_probes_uniformly_and_flags_no_speech(video_only_clip):
    """A file with no audio stream at all gets what spec 7.1 promises files
    with no speech: uniform probing over the 40-60% window, a box, and the
    no-speech flag -- not an ffmpeg error that skips the file."""
    engine = _BrightBoxDetector()
    result = crop.detect_crop(str(video_only_clip), 20.0, engine)

    assert result.flagged == crop.FLAG_NO_SPEECH
    assert result.box is not None
    _, y, _, h = result.box
    assert y <= 300 and y + h >= 330, f"box {result.box} must contain the bar (y 300-330)"
    uniform = set(crop._uniform_probe_times(20.0))
    assert result.sample_pts and set(result.sample_pts) <= uniform
    assert result.hit_pts and all(int(t) % 2 == 0 for t in result.hit_pts)


def test_detect_crop_still_raises_when_the_audio_stream_cannot_be_decoded(undecodable_audio_clip):
    with pytest.raises(subprocess.CalledProcessError):
        crop.detect_crop(str(undecodable_audio_clip), 4.0, _BrightBoxDetector())
