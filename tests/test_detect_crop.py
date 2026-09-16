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
