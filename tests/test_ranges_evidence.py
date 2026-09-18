"""Tests for the ranges pipeline's evidence surface (Block/RangesAnalysis via
analyse_detailed()) and the Qt-free per-episode audio profile
(core/detect/audio_profile.py).

Reuses the equivalence oracle and synthetic corpus helpers from
tests/test_detect_ranges.py rather than duplicating them: `_synthetic_corpus`
(planted-repeat fingerprint/duration lists, no ffmpeg needed) for broad
property coverage of compute_blocks(), and `synthetic_episodes` /
`_entries` / `_SYNTH_CFG` (real decoded WAV "episodes") for the
analyse_detailed().keep == analyse() end-to-end equivalence check.
`synthetic_audio_video` and `video_only_clip` come from tests/conftest.py
and need no import -- pytest injects fixtures from conftest.py by name.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.detect import audio_profile as ap
from core.detect import vad
from core.detect.ranges import pipeline as pl
from core.detect.ranges.config import MatchConfig, RangesConfig

from test_detect_ranges import _entries, _SYNTH_CFG, _synthetic_corpus, synthetic_episodes


# ===========================================================================
# analyse_detailed(): keep-range fidelity
# ===========================================================================

def test_analyse_detailed_keep_equals_analyse_end_to_end(synthetic_episodes, tmp_path):
    entries = _entries(synthetic_episodes)
    events_a = []
    keep = pl.analyse(entries, _SYNTH_CFG, progress=events_a.append,
                       cache_dir=str(tmp_path / "cache_a"), workers=2)
    events_b = []
    result = pl.analyse_detailed(entries, _SYNTH_CFG, progress=events_b.append,
                                  cache_dir=str(tmp_path / "cache_b"), workers=2)

    assert isinstance(result, pl.RangesAnalysis)
    assert result.keep == keep
    assert keep  # non-vacuous: synthetic_episodes really has repeats
    assert [e.kind for e in events_a] == [e.kind for e in events_b]

    names = {e.name for e in entries}
    assert set(result.durations) == names
    assert all(d > 0 for d in result.durations.values())
    assert any(result.blocks.values())  # some file has at least one block
    assert set(result.blocks) <= names


def test_analyse_detailed_raises_analysis_cancelled_like_analyse(synthetic_episodes):
    with pytest.raises(pl.AnalysisCancelled):
        pl.analyse_detailed(_entries(synthetic_episodes), _SYNTH_CFG, cache_dir=None,
                             workers=1, cancel=lambda: True)


def test_analyse_detailed_no_cache_dir_writes_nothing(synthetic_episodes):
    import os
    before = sorted(os.listdir(synthetic_episodes[0].parent))
    pl.analyse_detailed(_entries(synthetic_episodes[:2]), _SYNTH_CFG, cache_dir=None, workers=1)
    assert sorted(os.listdir(synthetic_episodes[0].parent)) == before


# ===========================================================================
# compute_blocks(): raw matched segments per file, unmerged, sorted
# ===========================================================================

@pytest.mark.parametrize("n_files,seed", [(3, 2), (5, 3), (10, 6)])
def test_compute_blocks_preserves_every_raw_match_without_merging(n_files, seed):
    rng = np.random.default_rng(seed)
    cfg = RangesConfig(match=MatchConfig(min_length_sec=8.0))
    fingerprints, durations = _synthetic_corpus(rng, n_files)
    names = [f"ep{i:02d}.mkv" for i in range(n_files)]
    arrays = [np.array(f, dtype=np.int64).reshape(-1, 2) for f in fingerprints]
    segments = pl.discover_segments(arrays, durations, cfg)
    assert segments  # the synthetic corpus really has repeats

    blocks = pl.compute_blocks(segments, names, durations)

    total_matches = sum(len(seg.matches) for seg in segments)
    total_blocks = sum(len(v) for v in blocks.values())
    assert total_blocks == total_matches  # one Block per SegmentMatch, none merged/dropped

    for file_blocks in blocks.values():
        starts = [b.start_sec for b in file_blocks]
        assert starts == sorted(starts)


def test_blocks_carry_kind_matched_files_and_score_for_intro_and_outro():
    names = ["a.mkv", "b.mkv", "c.mkv"]
    durations = [100.0, 100.0, 100.0]
    intro_matches = (
        pl.SegmentMatch(file=0, start_frame=0, end_frame=100, start_sec=0.0, end_sec=4.0, score=0.9),
        pl.SegmentMatch(file=1, start_frame=0, end_frame=100, start_sec=1.0, end_sec=5.0, score=0.8),
        pl.SegmentMatch(file=2, start_frame=0, end_frame=100, start_sec=2.0, end_sec=6.0, score=0.7),
    )
    outro_matches = (
        pl.SegmentMatch(file=0, start_frame=0, end_frame=100, start_sec=94.0, end_sec=100.0, score=0.5),
        pl.SegmentMatch(file=1, start_frame=0, end_frame=100, start_sec=95.0, end_sec=99.0, score=0.4),
    )
    segments = [
        pl.Segment(pivot=0, candidate=1, start_frame=0, end_frame=100, duration_sec=4.0, matches=intro_matches),
        pl.Segment(pivot=0, candidate=1, start_frame=0, end_frame=100, duration_sec=6.0, matches=outro_matches),
    ]

    blocks = pl.compute_blocks(segments, names, durations)

    assert [b.kind for b in blocks["a.mkv"]] == ["intro", "outro"]
    assert blocks["a.mkv"][0].matched_files == 3
    assert blocks["a.mkv"][0].score == 0.9
    assert blocks["a.mkv"][1].matched_files == 2
    assert blocks["a.mkv"][1].score == 0.5

    assert [b.kind for b in blocks["b.mkv"]] == ["intro", "outro"]
    assert [b.kind for b in blocks["c.mkv"]] == ["intro"]  # c has no outro match


@pytest.mark.parametrize("duration,edge", [(100.0, 5.0), (40.0, 5.0), (1000.0, 50.0)])
def test_kind_heuristic_uses_edge_max_5_or_5pct_duration_with_inclusive_boundaries(duration, edge):
    names = ["f.mkv"]
    durations = [duration]

    def kind_for(start_sec, end_sec):
        seg = pl.Segment(
            pivot=0, candidate=1, start_frame=0, end_frame=1, duration_sec=1.0,
            matches=(pl.SegmentMatch(file=0, start_frame=0, end_frame=1,
                                      start_sec=start_sec, end_sec=end_sec, score=1.0),),
        )
        return pl.compute_blocks([seg], names, durations)["f.mkv"][0].kind

    # start boundary: exactly at edge -> intro ( <= ); just past it -> not intro
    assert kind_for(edge, edge + 20.0) == "intro"
    assert kind_for(edge + 0.001, duration / 2) == "repeat"
    # end boundary: exactly at duration - edge -> outro ( >= ); just short -> not outro
    assert kind_for(duration / 2, duration - edge) == "outro"
    assert kind_for(duration / 2, duration - edge - 0.001) == "repeat"


def test_matched_files_counts_distinct_files_not_raw_match_count():
    names = ["a.mkv", "b.mkv"]
    durations = [100.0, 100.0]
    # Two matches for the SAME file inside one segment. discover_segments()
    # never produces this itself (one match per file per segment), but
    # compute_blocks() must not rely on that invariant to get the count
    # right -- it counts distinct files, not len(matches).
    matches = (
        pl.SegmentMatch(file=0, start_frame=0, end_frame=10, start_sec=10.0, end_sec=20.0, score=0.9),
        pl.SegmentMatch(file=0, start_frame=100, end_frame=110, start_sec=50.0, end_sec=60.0, score=0.6),
        pl.SegmentMatch(file=1, start_frame=0, end_frame=10, start_sec=10.0, end_sec=20.0, score=0.5),
    )
    seg = pl.Segment(pivot=0, candidate=1, start_frame=0, end_frame=10, duration_sec=10.0, matches=matches)

    blocks = pl.compute_blocks([seg], names, durations)

    assert len(matches) == 3
    assert all(b.matched_files == 2 for b in blocks["a.mkv"])
    assert blocks["b.mkv"][0].matched_files == 2


def test_overlapping_matches_in_the_same_file_are_both_kept_as_separate_blocks():
    names = ["a.mkv"]
    durations = [100.0]
    matches1 = (pl.SegmentMatch(file=0, start_frame=0, end_frame=200, start_sec=10.0, end_sec=30.0, score=0.9),)
    matches2 = (pl.SegmentMatch(file=0, start_frame=0, end_frame=200, start_sec=20.0, end_sec=40.0, score=0.7),)
    segments = [
        pl.Segment(pivot=0, candidate=1, start_frame=0, end_frame=200, duration_sec=20.0, matches=matches1),
        pl.Segment(pivot=0, candidate=1, start_frame=0, end_frame=200, duration_sec=20.0, matches=matches2),
    ]

    blocks = pl.compute_blocks(segments, names, durations)

    assert len(blocks["a.mkv"]) == 2
    assert [(b.start_sec, b.end_sec) for b in blocks["a.mkv"]] == [(10.0, 30.0), (20.0, 40.0)]


# ===========================================================================
# audio_profile(): envelope + speech
# ===========================================================================

def test_rms_envelope_splits_into_equal_chunks_with_remainder_in_the_last():
    # 10 samples, 3 bins: chunk_size = 10 // 3 = 3 -> sizes 3, 3, 4.
    samples = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.float32)
    env = ap._rms_envelope(samples, 3)
    assert len(env) == 3
    assert env == [0.0, 0.5, 1.0]  # normalised so the loudest chunk (rms=2) is 1.0


def test_rms_envelope_is_all_zero_for_silence():
    samples = np.zeros(1000, dtype=np.float32)
    assert ap._rms_envelope(samples, 600) == [0.0] * 600


def test_rms_envelope_rounds_to_four_decimals():
    rng = np.random.default_rng(5)
    samples = (rng.standard_normal(9000).astype(np.float32) * 0.3)
    env = ap._rms_envelope(samples, 600)
    assert all(v == round(v, 4) for v in env)


def test_audio_profile_envelope_length_and_normalisation(synthetic_audio_video):
    profile = ap.audio_profile(str(synthetic_audio_video), 10.0)
    assert len(profile.envelope) == ap.ENVELOPE_BINS
    assert profile.duration == 10.0
    assert max(profile.envelope) == 1.0
    assert min(profile.envelope) >= 0.0


def test_audio_profile_speech_covers_each_tone_burst(synthetic_audio_video):
    # conftest's synthetic_audio_video has 1 kHz tone bursts at 1-3s, 5-6s, 8-9s.
    profile = ap.audio_profile(str(synthetic_audio_video), 10.0)
    assert profile.speech
    for w_start, w_end in [(1.0, 3.0), (5.0, 6.0), (8.0, 9.0)]:
        assert any(s <= w_start + 0.5 and e >= w_end - 0.5 for s, e in profile.speech), profile.speech


def test_audio_profile_on_video_without_audio_returns_zeros_without_raising(video_only_clip):
    profile = ap.audio_profile(str(video_only_clip), 20.0)
    assert profile.envelope == [0.0] * ap.ENVELOPE_BINS
    assert profile.speech == []
    assert profile.duration == 20.0


def test_audio_profile_passes_cancel_check_through_and_propagates_cancellation(
    synthetic_audio_video, monkeypatch,
):
    received = {}

    def fake_extract(video_path, start_sec, duration_sec, sample_rate=vad.SAMPLE_RATE, cancel_check=None):
        received["args"] = (video_path, start_sec, duration_sec)
        received["cancel_check"] = cancel_check
        raise vad.AudioExtractionCancelled(video_path)

    monkeypatch.setattr(ap.vad, "extract_audio_window", fake_extract)

    def my_cancel():
        return True

    with pytest.raises(vad.AudioExtractionCancelled):
        ap.audio_profile(str(synthetic_audio_video), 10.0, cancel_check=my_cancel)

    assert received["cancel_check"] is my_cancel
    assert received["args"] == (str(synthetic_audio_video), 0.0, 10.0)


# ===========================================================================
# speech_in_skips(): overlap, clipping, minimum length
# ===========================================================================

def test_speech_in_skips_returns_overlap_clipped_to_the_block():
    block = pl.Block(start_sec=10.0, end_sec=20.0, kind="repeat", matched_files=2, score=0.8)
    speech = [(5.0, 12.0), (18.0, 25.0), (13.0, 14.0), (0.0, 5.0)]
    result = ap.speech_in_skips(speech, [block], min_overlap_sec=2.0)
    assert result == [(10.0, 12.0), (18.0, 20.0)]


def test_speech_in_skips_respects_min_overlap_boundary():
    block = pl.Block(start_sec=0.0, end_sec=10.0, kind="repeat", matched_files=1, score=1.0)
    assert ap.speech_in_skips([(8.0, 10.0)], [block], min_overlap_sec=2.0) == [(8.0, 10.0)]
    assert ap.speech_in_skips([(8.001, 10.0)], [block], min_overlap_sec=2.0) == []


def test_speech_in_skips_handles_multiple_blocks_independently():
    blocks = [
        pl.Block(0.0, 10.0, "intro", 1, 1.0),
        pl.Block(50.0, 60.0, "outro", 1, 1.0),
    ]
    speech = [(5.0, 55.0)]  # spans across both blocks and the gap between them
    result = ap.speech_in_skips(speech, blocks, min_overlap_sec=2.0)
    assert result == [(5.0, 10.0), (50.0, 55.0)]


@pytest.mark.parametrize("as_spans", [
    lambda blocks: [(b.start_sec, b.end_sec) for b in blocks],                 # skip spans (tuples)
    lambda blocks: [[b.start_sec, b.end_sec] for b in blocks],                 # JSON-loaded spans (lists)
    lambda blocks: ((b.start_sec, b.end_sec) for b in blocks),                 # any iterable, even a generator
    lambda blocks: [{"start_sec": b.start_sec, "end_sec": b.end_sec, "kind": b.kind,
                     "matched_files": b.matched_files, "score": b.score} for b in blocks],   # evidence dicts
])
def test_speech_in_skips_accepts_spans_and_evidence_blocks_like_blocks(as_spans):
    blocks = [
        pl.Block(0.0, 10.0, "intro", 1, 1.0),
        pl.Block(50.0, 60.0, "outro", 1, 1.0),
    ]
    speech = [(5.0, 55.0), (58.5, 59.0)]
    expected = ap.speech_in_skips(speech, blocks, min_overlap_sec=2.0)
    assert expected == [(5.0, 10.0), (50.0, 55.0)]
    assert ap.speech_in_skips(speech, as_spans(blocks), min_overlap_sec=2.0) == expected


def test_speech_in_skips_mixes_blocks_and_spans_and_takes_json_speech():
    speech = [[5.0, 55.0]]                                                     # evidence["audio"]["speech"]
    skips = [pl.Block(0.0, 10.0, "intro", 1, 1.0), (50.0, 60.0)]
    assert ap.speech_in_skips(speech, skips, min_overlap_sec=2.0) == [(5.0, 10.0), (50.0, 55.0)]


@pytest.mark.parametrize("bad", [[(1.0,)], [(1.0, 2.0, 3.0)], [{"start_sec": 1.0}], [None], [1.0]])
def test_speech_in_skips_refuses_what_is_not_a_span(bad):
    with pytest.raises((TypeError, ValueError)):
        ap.speech_in_skips([(0.0, 10.0)], bad)


# ===========================================================================
# Optional: real-project equivalence (slow, needs_media)
# ===========================================================================

@pytest.mark.slow
@pytest.mark.needs_media
def test_analyse_detailed_keep_matches_analyse_on_a_real_reference_project(tmp_path):
    """analyse_detailed(...).keep == analyse(...) on real episodes, read-only
    against /mnt/FAST/work -- cache_dir is under tmp_path, so nothing is
    written to the reference project. Not part of the fast suite: fingerprints
    a couple of full 1080p episodes."""
    manifest = json.loads((Path(__file__).parent / "fixtures" / "media_manifest.json").read_text())
    root = Path(manifest["root"])
    if not root.exists():
        pytest.skip(f"reference media root {root} not present")
    project_dir = root / manifest["projects"]["slay"]["dir"]
    if not project_dir.is_dir():
        pytest.skip(f"{project_dir} not present")
    videos = sorted(p for p in project_dir.iterdir() if p.suffix.lower() in (".mp4", ".mkv"))[:2]
    if len(videos) < 2:
        pytest.skip("need at least 2 episode files for repeat detection")

    entries = [pl.FileEntry(name=p.name, path=str(p)) for p in videos]
    cfg = RangesConfig()
    keep = pl.analyse(entries, cfg, cache_dir=str(tmp_path / "cache_a"), workers=2)
    result = pl.analyse_detailed(entries, cfg, cache_dir=str(tmp_path / "cache_b"), workers=2)

    assert result.keep == keep
