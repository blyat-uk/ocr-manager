"""Phase 4 must put each label boundary on the exact frame the label appears
or disappears on.

Phase 4 scans outward from phase 3's first and last reading in
LabelScanner.TIMING_SCAN_INTERVAL (0.2 s) steps and stops after two
consecutive samples without the label. That scan only brackets a boundary:
between the last sample that shows the label and the first sample that does
not, so on its own a start can be up to 0.2 s late and an end up to 0.2 s
early. The frames inside the bracket are then read in order, after one
display-time seek, and the boundary moves outward over them while they show
the label.

Times follow the dialogue path's convention (videocr/video.py,
Video.get_subtitles): a label starts at the PTS of the first frame it is on
and ends at the PTS of the last frame it is on plus one frame (1 / fps).

The detection engine here finds the label on a chosen set of frames. It tells
frames apart by the detection input it is given -- the whole region above the
dialogue cutoff, label masks applied, which differs for every frame -- so each
decision is tied to the pixels actually analysed, not to whatever the capture
read last. Frame PTS and pixels come from one independent sequential decode
(tests/test_label_frame_identity.py), and the clips are that file's: a 0.021 s
video start after a zero container start (Youxia Zhanji), a 1.5 s container
start, a zero start, HEVC in MP4 (seeks that land after their target) and
23.976 fps (a 0.2 s step is not a whole number of frames).
"""
import itertools
import logging
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from videocr.label_scanner import LabelResult, _ScanBracket
from videocr.pyav_adapter import PyAVCapture
from test_label_frame_identity import (
    CLIPS,
    ON_SCREEN_TOLERANCE,
    SIZES,
    VIDEO_START,
    _Counting,
    _decode_reference,
    _encode,
    _ExhaustingSeeks,
    _have_encoder,
    _matrix,
    _on_screen,
    _phase34_scanner,
    _region_box,
)

# 25 fps with one frame left out after frame GAP_AFTER, keeping every other
# frame's timestamp, so the frame after it comes two frame durations later.
GAP_CLIP = "dropped-frame-h264"
GAP_AFTER = 69

# 23.976 fps in MKV's millisecond time base: frame durations are rounded to
# 41 or 42 ms, so a frame's PTS plus 1 / fps can pass the next frame's PTS.
MS_CLIP = "zero-start-h264-23.976fps-mkv"
EXTRA_CLIPS = {MS_CLIP: ("mkv", "yuv420p", "libx264", "24000/1001", [])}

BOUNDARY_CLIPS = [
    "video-start-0.021s-mkv",
    "offset-h264",
    "zero-start-h264",
    "zero-start-h265-10bit",
    "offset-h264-23.976fps",
]


class _Clip:
    """A clip, its reference decode, and the detection input of each frame."""

    def __init__(self, path):
        self.path = path
        self.reference = _decode_reference(path)
        self.pts = [p for p, _ in self.reference]
        probe = self.scanner()
        self.fps = probe.fps
        self.box = _region_box(probe)
        self.frame_of_input = {}
        for i, (_, frame) in enumerate(self.reference):
            frame = frame.copy()
            probe._apply_label_masks(frame)
            roi, _, _ = probe._crop_roi_for_detection(frame[: probe.dialogue_cutoff_y, :], self.box)
            if roi.shape[0] > probe.SCAN_HEIGHT:
                roi, _ = probe._downscale(roi, probe.SCAN_HEIGHT)
            key = self.key(roi)
            self.region_input_shape = roi.shape
            assert key not in self.frame_of_input, (
                f"frames {self.frame_of_input.get(key)} and {i} give detection the same input")
            self.frame_of_input[key] = i

    def scanner(self, min_duration=0.0, max_duration=1000.0):
        """A label scanner with label masks. Durations are pinned by their own
        test; elsewhere every label is kept."""
        scanner = _phase34_scanner(self.path)
        scanner.label_min_duration = min_duration
        scanner.label_max_duration = max_duration
        return scanner

    @staticmethod
    def key(image):
        return image.shape, image.tobytes()

    def segment(self, first_reading, last_reading, text="label"):
        """A segment as phase 3 hands it over, read at these times."""
        return {"box": self.box, "text": text, "confidence": 1.0,
                "start_pts": first_reading, "end_pts": last_reading}

    def end_time(self, i):
        """End time of a label whose last frame is frame `i`."""
        return self.pts[i] + 1.0 / self.fps


class _LabelOn:
    """Finds the label -- one box covering the whole detection input -- on the
    frames in `shown` and nowhere else, and records which frame each call
    analysed. With `elsewhere_nothing`, an input cut around some other box
    finds nothing (and is not recorded) instead of failing the test."""

    def __init__(self, clip, shown, elsewhere_nothing=False):
        self.clip = clip
        self.shown = set(shown)
        self.elsewhere_nothing = elsewhere_nothing
        self.calls = []

    def predict(self, image):
        i = self.clip.frame_of_input.get(self.clip.key(image))
        if i is None and self.elsewhere_nothing and image.shape != self.clip.region_input_shape:
            return []
        assert i is not None, "detection was given an input that is not any frame's detection input"
        self.calls.append(i)
        if i not in self.shown:
            return []
        h, w = image.shape[:2]
        return [{"dt_polys": [[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]]}]


def _ffmpeg_has_option(name):
    """Whether this ffmpeg lists `-name` among its options (as "-name <arg>"
    or "-name[:<stream_spec>] <arg>")."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-h", "full"], capture_output=True, text=True).stdout
    return re.search(rf"^-{re.escape(name)}[\s\[]", out, re.MULTILINE) is not None


@pytest.fixture(scope="module")
def clip_for(tmp_path_factory):
    root = tmp_path_factory.mktemp("label_boundaries")
    made = {}

    def get(clip_id):
        if clip_id == GAP_CLIP:
            if not _have_encoder("libx264"):
                pytest.skip(f"libx264 encoder not available for {clip_id}")
            if not _ffmpeg_has_option("fps_mode"):
                pytest.skip(f"this ffmpeg has no -fps_mode, needed to keep {clip_id}'s timestamp gap")
            if clip_id not in made:
                path = root / f"{clip_id}.mp4"
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25", "-frames:v", "100",
                     "-vf", f"select='not(eq(n\\,{GAP_AFTER + 1}))'", "-fps_mode", "passthrough",
                     "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "10", str(path)],
                    check=True, capture_output=True)
                made[clip_id] = _Clip(path)
            return made[clip_id]
        ext, pix_fmt, codec, rate, extra = EXTRA_CLIPS.get(clip_id) or CLIPS[clip_id]
        if not _have_encoder(codec):
            pytest.skip(f"{codec} encoder not available for {clip_id}")
        if clip_id not in made:
            path = root / f"{clip_id}.{ext}"
            _encode(path, pix_fmt, codec, rate, extra, SIZES.get(clip_id, "320x240"), VIDEO_START.get(clip_id))
            made[clip_id] = _Clip(path)
        return made[clip_id]

    return get


# --- what phase 4 must do, from the scan's rules and the frames' PTS ----------

def _scan(clip, scanner, shown, discovery, direction, bound, skipped):
    """The 0.2 s scan from `discovery` outward, as the scan's own rules run it:
    it stops past `bound`, past the last frame (forward), or after two
    consecutive samples without the label; a time whose seek runs out of
    retries (`skipped`) is passed over without counting as an absence.

    Returns ([(time, frame index, shows the label)] per sample analysed, seeks made).
    """
    step = scanner.TIMING_SCAN_INTERVAL
    samples, seeks, absent_run = [], 0, 0
    t = discovery + direction * step
    while (t >= bound) if direction < 0 else (t <= bound):
        seeks += 1
        if skipped(t):
            t += direction * step
            continue
        i = _on_screen(clip.reference, t, clip.fps)
        if i is None:
            break
        samples.append((t, i, i in shown))
        absent_run = 0 if i in shown else absent_run + 1
        if absent_run == 2:
            break
        t += direction * step
    return samples, seeks


def _bracket(clip, samples, discovery):
    """(last sample time showing the label -- the discovery time if none did
    --, the frame on screen then, first frame after it without the label or
    None)."""
    present, first_absent = discovery, None
    for t, i, shows in samples:
        if shows:
            present, first_absent = t, None
        elif first_absent is None:
            first_absent = i
    return present, _on_screen(clip.reference, present, clip.fps), first_absent


def _expected(clip, scanner, shown, segment, lower_bound=None, upper_bound=None, skipped=lambda t: False):
    """What phase 4 must produce for one segment.

    Inside each bracket the frames between the scan's last frame showing the
    label and its first frame without it are analysed. With no frame without
    it (the scan stopped at its bound or past the last frame), the bracket
    reaches the bound instead: back to the first frame at or after the backward
    bound, forward to the last frame before the forward bound. Backward, every
    frame in the bracket is analysed; forward, frames are analysed until the
    first without the label. A frame at or after an adjacent segment's bound
    belongs to the later segment, the scan's own last frame included.

    Returns a namespace: start and end times; the frames detection runs on, in
    order; the display-time seeks made; and, per side, the frames the 0.2 s
    scan analysed (`backward`, `forward`), the frame it alone ends on
    (`scanned_start`, `scanned_end`) and the first frame it found without the
    label after that (`absent_before`, `absent_after`, or None).
    """
    tol = ON_SCREEN_TOLERANCE
    first_reading, last_reading = segment["start_pts"], segment["end_pts"]
    got = SimpleNamespace(analysed=[], seeks=1)
    mid = _on_screen(clip.reference, (first_reading + last_reading) / 2, clip.fps)
    if mid is not None:
        got.analysed.append(mid)  # the reference box

    low = max(0, first_reading - scanner.TIMING_SCAN_MAX_DURATION)
    bound = lower_bound if lower_bound is not None and lower_bound >= low else None
    if lower_bound is not None:
        low = max(low, lower_bound)
    samples, n = _scan(clip, scanner, shown, first_reading, -1, low, skipped)
    got.backward = [i for _, i, _ in samples]
    got.analysed += got.backward
    present, got.scanned_start, got.absent_before = _bracket(clip, samples, first_reading)
    inside = [i for i in range(got.scanned_start)
              if (i > got.absent_before if got.absent_before is not None else clip.pts[i] >= low - tol)]
    got.analysed += inside
    got.seeks += n + 1
    start = got.scanned_start
    for i in reversed(inside):
        if i not in shown:
            break
        start = i
    if bound is not None and clip.pts[start] < bound - tol:
        start += 1  # the scan's frame began before the bound: it is the earlier segment's

    high = last_reading + scanner.TIMING_SCAN_MAX_DURATION
    bound = upper_bound if upper_bound is not None and upper_bound <= high else None
    if upper_bound is not None:
        high = min(high, upper_bound)
    samples, n = _scan(clip, scanner, shown, last_reading, 1, high, skipped)
    got.forward = [i for _, i, _ in samples]
    got.analysed += got.forward
    present, got.scanned_end, got.absent_after = _bracket(clip, samples, last_reading)
    got.seeks += n + 1
    got.start = clip.pts[start]
    if bound is not None and clip.pts[got.scanned_end] >= bound - tol:
        # The scan's frame begins at the bound: it is the later segment's, and
        # this label ends where that frame replaces the one before it.
        got.end = clip.pts[got.scanned_end]
        return got
    end = got.scanned_end
    for i in range(got.scanned_end + 1, len(clip.pts)):
        if (i >= got.absent_after) if got.absent_after is not None else (clip.pts[i] >= high - tol):
            break
        got.analysed.append(i)
        if i not in shown:
            break
        end = i
    got.end = clip.end_time(end)
    return got


def _run(clip, scanner, segments, detector, monkeypatch):
    seeks = _Counting(monkeypatch, "seek_to_display_time")
    labels = scanner._phase4_find_timing(segments, detector)
    return labels, seeks.calls


# --- exact frames -------------------------------------------------------------

# (first frame shown, last frame shown, frame of phase 3's first reading, of its last).
# At 25 fps the backward scan from frame 40 samples frames 35, 30, 25 and the
# forward scan from frame 60 samples 65, 70, 75: these spans start 1 and 4
# frames after an absent sample or 1 and 4 frames before phase 3's first
# reading, and end likewise; the last lies on that grid, a control where the
# frames inside each bracket must all be found without the label.
SPANS = [
    pytest.param(31, 69, 40, 60, id="shown-31-69"),
    pytest.param(34, 66, 40, 60, id="shown-34-66"),
    pytest.param(36, 64, 40, 60, id="shown-36-64"),
    pytest.param(39, 61, 40, 60, id="shown-39-61"),
    pytest.param(35, 65, 40, 60, id="shown-35-65"),
]


@pytest.mark.parametrize("clip_id,first,last,first_reading,last_reading", _matrix(
    [(c, span) for c in BOUNDARY_CLIPS for span in SPANS],
    fast={"video-start-0.021s-mkv-shown-31-69", "video-start-0.021s-mkv-shown-39-61",
          "offset-h264-shown-34-66", "offset-h264-shown-36-64",
          "video-start-0.021s-mkv-shown-35-65"}))
def test_a_label_starts_and_ends_on_the_exact_frames(clip_for, monkeypatch, clip_id, first, last,
                                                    first_reading, last_reading):
    clip = clip_for(clip_id)
    scanner = clip.scanner()
    shown = range(first, last + 1)
    segment = clip.segment(clip.pts[first_reading], clip.pts[last_reading])
    detector = _LabelOn(clip, shown)

    labels, seeks = _run(clip, scanner, [segment], detector, monkeypatch)

    want = _expected(clip, scanner, shown, segment)
    assert (want.start, want.end) == (clip.pts[first], clip.end_time(last)), "the model of phase 4 is wrong"
    if (first, last) != (35, 65):
        assert (want.scanned_start, want.scanned_end) != (first, last), (
            "the 0.2 s scan alone finds both boundaries here, so this case proves nothing")
    assert len(labels) == 1
    (label,) = labels
    assert (label.start_pts, label.end_pts) == (clip.pts[first], clip.end_time(last)), (
        f"label shown on frames {first}-{last} (PTS {clip.pts[first]:.6f} to {clip.pts[last]:.6f}, so ending "
        f"{clip.end_time(last):.6f}) came out as {label.start_pts:.6f}-{label.end_pts:.6f}")
    assert detector.calls == want.analysed, "phase 4 ran detection on other frames than the scan and its brackets"
    assert seeks == want.seeks, "the frames inside a bracket were not read in order after one seek"


def test_a_label_ends_one_frame_after_its_last_frame_even_where_the_next_frame_comes_later(clip_for, monkeypatch):
    """The end is the last frame's PTS plus 1 / fps, as the dialogue path ends
    a subtitle (Video.get_subtitles), not the PTS of the frame after it. The
    two differ where the next frame comes later than one frame duration."""
    clip = clip_for(GAP_CLIP)
    gap = clip.pts[GAP_AFTER + 1] - clip.pts[GAP_AFTER]
    assert gap > 1.5 / clip.fps, f"the clip has no timestamp gap after frame {GAP_AFTER}: {gap}"
    scanner = clip.scanner()
    shown = range(31, GAP_AFTER + 1)
    segment = clip.segment(clip.pts[40], clip.pts[60])
    detector = _LabelOn(clip, shown)

    labels, seeks = _run(clip, scanner, [segment], detector, monkeypatch)

    assert [(l.start_pts, l.end_pts) for l in labels] == [(clip.pts[31], clip.end_time(GAP_AFTER))]
    assert clip.end_time(GAP_AFTER) < clip.pts[GAP_AFTER + 1]
    want = _expected(clip, scanner, set(shown), segment)
    assert detector.calls == want.analysed
    assert seeks == want.seeks


@pytest.mark.parametrize("clip_id", _matrix(BOUNDARY_CLIPS, fast={"video-start-0.021s-mkv", "offset-h264"}))
def test_a_missed_detection_inside_a_bracket_never_moves_a_boundary_past_the_scan(clip_for, monkeypatch, clip_id):
    """A frame inside the bracket that detection misses stops the boundary
    there: it can end up anywhere between the true frame and the frame the
    0.2 s scan found, never outside that, and never past the bracket."""
    clip = clip_for(clip_id)
    # The last frame shown is two frames into the forward bracket at 25 fps
    # and three at 23.976 fps, where a 0.2 s step drifts off whole frames.
    first, last = 31, 67
    shown = set(range(first, last + 1))
    segment = clip.segment(clip.pts[40], clip.pts[60])
    want = _expected(clip, clip.scanner(), shown, segment)
    start_inside = [i for i in range(want.absent_before + 1, want.scanned_start) if i in shown]
    end_inside = [i for i in range(want.scanned_end + 1, want.absent_after) if i in shown]
    assert start_inside and end_inside, "no frame showing the label inside a bracket, so nothing to miss"

    outcomes = []
    for missed in start_inside:
        (label,), _ = _run(clip, clip.scanner(), [segment], _LabelOn(clip, shown - {missed}), monkeypatch)
        assert clip.pts[first] <= label.start_pts <= clip.pts[want.scanned_start], (
            f"a miss on frame {missed} put the start at {label.start_pts}, outside "
            f"[{clip.pts[first]}, {clip.pts[want.scanned_start]}]")
        outcomes.append((missed, label.start_pts == clip.pts[missed + 1]))
        assert label.end_pts == clip.end_time(last)
    for missed in end_inside:
        (label,), _ = _run(clip, clip.scanner(), [segment], _LabelOn(clip, shown - {missed}), monkeypatch)
        assert clip.end_time(want.scanned_end) <= label.end_pts <= clip.end_time(last), (
            f"a miss on frame {missed} put the end at {label.end_pts}, outside "
            f"[{clip.end_time(want.scanned_end)}, {clip.end_time(last)}]")
        outcomes.append((missed, label.end_pts == clip.end_time(missed - 1)))
        assert label.start_pts == clip.pts[first]
    assert all(exact for _, exact in outcomes), (
        f"a miss must stop the boundary on the frame next to it (missed frame, stopped there): {outcomes}")


@pytest.mark.parametrize("clip_id", _matrix(BOUNDARY_CLIPS, fast={"video-start-0.021s-mkv", "offset-h264"}))
def test_the_bracket_edge_is_the_first_absent_sample_after_the_last_present_one(clip_for, monkeypatch, clip_id):
    """Detection misses the label at the first sample on each side, and finds
    it again at the next. The scan goes on past that one absence, as it always
    has, so the bracket runs from the next sample to the first absent sample
    after it -- not from the absence the scan passed over."""
    clip = clip_for(clip_id)
    scanner = clip.scanner()
    segment = clip.segment(clip.pts[40], clip.pts[60])
    first, last = 28, 72
    whole = set(range(first, last + 1))
    plain = _expected(clip, scanner, whole, segment)
    shown = whole - {plain.backward[0], plain.forward[0]}
    want = _expected(clip, scanner, shown, segment)
    assert want.scanned_start == plain.backward[1] and want.scanned_end == plain.forward[1], (
        "the scan did not find the label again after the miss, so this case proves nothing")
    assert want.absent_before < first < want.scanned_start and want.scanned_end < last < want.absent_after
    detector = _LabelOn(clip, shown)

    labels, seeks = _run(clip, scanner, [segment], detector, monkeypatch)

    assert [(l.start_pts, l.end_pts) for l in labels] == [(clip.pts[first], clip.end_time(last))]
    assert detector.calls == want.analysed
    assert seeks == want.seeks


# --- where the scan stops without an absent sample ----------------------------

@pytest.mark.parametrize("clip_id,second_reading", _matrix(
    [(c, r) for c in BOUNDARY_CLIPS for r in (
        pytest.param(52, id="bound-between-frames"), pytest.param(53, id="bound-on-a-frame"))],
    fast={"video-start-0.021s-mkv-bound-between-frames", "offset-h264-bound-on-a-frame"}))
def test_back_to_back_segments_never_refine_across_their_bound(clip_for, monkeypatch, clip_id, second_reading):
    """Two segments at the same position with the label found on every frame
    between them, so only the bound between them (the midpoint of the first's
    last reading and the second's first) keeps them apart. The first ends on
    the last frame before the bound and the second starts on the first frame
    at or after it: no frame is analysed for both, and they do not overlap."""
    clip = clip_for(clip_id)
    scanner = clip.scanner()
    shown = range(20, 81)
    first = clip.segment(clip.pts[25], clip.pts[45])
    second = clip.segment(clip.pts[second_reading], clip.pts[70])
    bound = (first["end_pts"] + second["start_pts"]) / 2
    before = _before_bound(clip, bound)
    detector = _LabelOn(clip, shown)

    labels, seeks = _run(clip, scanner, [first, second], detector, monkeypatch)

    assert [(l.start_pts, l.end_pts) for l in labels] == [
        (clip.pts[20], clip.end_time(before)), (clip.pts[before + 1], clip.end_time(80))]
    want_first = _expected(clip, scanner, shown, first, upper_bound=bound)
    want_second = _expected(clip, scanner, shown, second, lower_bound=bound)
    assert detector.calls == want_first.analysed + want_second.analysed
    assert seeks == want_first.seeks + want_second.seeks


def _before_bound(clip, bound):
    """The last frame that begins before `bound`: the earlier segment's last."""
    return max(i for i, p in enumerate(clip.pts) if p < bound - ON_SCREEN_TOLERANCE)


def _scan_segments(scanner, segments, detector):
    """scan() with phases 1-3 standing in for `segments`, so phase 4 and all of
    scan()'s post-processing run on them (container start 0)."""
    scanner._phase1_find_text_frames = lambda *a, **k: [(0, 0.0, [])]
    scanner._batch_ocr_text_frames = lambda text_frames, *a, **k: text_frames
    scanner._phase2_group_by_position = lambda *a, **k: [None]
    scanner._phase3_ocr_and_segment = lambda *a, **k: [dict(s) for s in segments]
    return scanner.scan(detector, None, "", "", 0.0)


# The earlier segment's last reading on four frame phases; the later segment's
# first reading 0.20-1.00 s after it. Gaps that are multiples of 0.4 s put the
# bound on a sample of both scans, and on a frame where readings are frames.
SWEEP_LAST_READINGS = [(40, 0.0), (41, 0.5), (42, 0.0), (43, 0.5)]
SWEEP_GAPS = {
    "every-0.01s": [g / 100 for g in range(20, 101)],
    "around-0.4s-and-0.8s": [g / 100 for g in list(range(20, 101, 7)) + list(range(36, 46)) + list(range(76, 86))],
}


@pytest.mark.parametrize("clip_id,gaps", _matrix(
    [(c, g) for c in BOUNDARY_CLIPS + [MS_CLIP] for g in SWEEP_GAPS],
    fast={"video-start-0.021s-mkv-around-0.4s-and-0.8s", f"{MS_CLIP}-around-0.4s-and-0.8s"}))
def test_back_to_back_labels_at_one_position_share_no_frame_and_lose_no_span(clip_for, clip_id, gaps):
    """Two segments at one position with the label found on every frame from 5
    to 95, through phase 4 and scan()'s post-processing. The bound between
    them is the only thing that tells them apart, and every frame belongs to
    exactly one side of it: with different texts the two labels share no
    frame and together cover the whole run; with the same text they come out
    as one label covering the whole run (an overlap would make them
    "duplicates" and drop one)."""
    clip = clip_for(clip_id)
    shown = range(5, 96)
    half = 0.5 / clip.fps
    whole = (clip.pts[shown[0]], clip.end_time(shown[-1]))
    failures, checked = [], 0
    for frame, offset in SWEEP_LAST_READINGS:
        for gap in SWEEP_GAPS[gaps]:
            last_reading = clip.pts[frame] + offset
            second_reading = last_reading + gap
            for texts in (("甲", "乙乙乙乙"), ("第三十七集", "第三十七集")):
                segments = [clip.segment(clip.pts[20], last_reading, texts[0]),
                            clip.segment(second_reading, second_reading + 0.6, texts[1])]
                labels = _scan_segments(clip.scanner(), segments, _LabelOn(clip, shown))
                spans = sorted((l.start_pts, l.end_pts) for l in labels)
                checked += 1
                if texts[0] == texts[1]:
                    if spans != [whole]:
                        failures.append((frame, offset, gap, texts[0], spans))
                    continue
                claimed = [{i for i, p in enumerate(clip.pts) if a - half <= p < b - half} for a, b in spans]
                if (len(spans) != 2 or claimed[0] & claimed[1] or claimed[0] | claimed[1] != set(shown)
                        or (spans[0][0], spans[1][1]) != whole):
                    failures.append((frame, offset, gap, "shared " + str(sorted(claimed[0] & claimed[1]))
                                     if len(spans) == 2 else "", spans))
    assert checked == 2 * len(SWEEP_LAST_READINGS) * len(SWEEP_GAPS[gaps])
    assert not failures, (f"{len(failures)} of {checked} back-to-back pairs (last reading frame, offset, gap, "
                          f"texts or shared frames, labels): {failures[:8]}")


def test_remove_duplicates_ignores_an_overlap_under_half_a_frame_but_not_a_real_one(clip_for):
    """Labels timed back to back can overlap by float noise, since an end is a
    frame's PTS plus 1 / fps, or by under a millisecond where frame durations
    are rounded. That is no overlap; half a frame or more still is (and the
    label kept then covers both, see the test below)."""
    clip = clip_for("video-start-0.021s-mkv")
    scanner = clip.scanner()
    frame = 1.0 / scanner.fps

    def label(start, end, text="第三十七集"):
        return LabelResult(start_pts=start, end_pts=end, text=text, pos_x=160, pos_y=200,
                           bbox_x_min=0.0, bbox_y_min=0.0, bbox_x_max=319.0, bbox_y_max=160.0)

    def kept(a, b):
        return [(l.start_pts, l.end_pts) for l in scanner._remove_duplicates([label(*a), label(*b)])]

    assert 1.086 + frame > 1.126  # float noise: the frame at 1.086 ends a little after the one at 1.126 starts
    assert kept((0.5, 1.086 + frame), (1.126, 2.5)) == [(0.5, 1.086 + frame), (1.126, 2.5)]
    assert kept((0.8, 1.3 + 0.4 * frame), (1.3, 2.5)) == [(0.8, 1.3 + 0.4 * frame), (1.3, 2.5)]
    # Duplicates: one label, the longer, covering both.
    assert kept((0.8, 1.3 + 0.75 * frame), (1.3, 2.5)) == [(0.8, 2.5)], "two labels overlapping by 3/4 frame"
    assert kept((0.8, 1.3 + 1.5 * frame), (1.3, 2.5)) == [(0.8, 2.5)]
    assert kept((1.0, 3.0), (1.5, 2.5)) == [(1.0, 3.0)]
    assert len(scanner._remove_duplicates([label(1.0, 3.0), label(1.5, 2.5, "乙乙乙乙")])) == 2


@pytest.mark.parametrize("clip_id", _matrix(BOUNDARY_CLIPS, fast={"video-start-0.021s-mkv", "offset-h264"}))
def test_bounds_come_from_the_neighbours_in_time_at_the_same_position(clip_for, monkeypatch, clip_id):
    """Phase 3 hands segments over in cluster order, not time order. Each
    segment is bounded by the segment just before it and just after it in
    time at the same position, wherever those are in the list. The label is
    found on every frame 10-85, so the bounds alone split it three ways."""
    clip = clip_for(clip_id)
    scanner = clip.scanner()
    shown = set(range(10, 86))
    first = clip.segment(clip.pts[15], clip.pts[20], "甲")
    second = clip.segment(clip.pts[45], clip.pts[50], "乙乙")
    third = clip.segment(clip.pts[75], clip.pts[80], "丙丙丙")
    one_two = (first["end_pts"] + second["start_pts"]) / 2
    two_three = (second["end_pts"] + third["start_pts"]) / 2
    detector = _LabelOn(clip, shown)

    labels, seeks = _run(clip, scanner, [third, first, second], detector, monkeypatch)

    assert [(l.text, l.start_pts, l.end_pts) for l in labels] == [
        ("丙丙丙", clip.pts[_before_bound(clip, two_three) + 1], clip.end_time(85)),
        ("甲", clip.pts[10], clip.end_time(_before_bound(clip, one_two))),
        ("乙乙", clip.pts[_before_bound(clip, one_two) + 1], clip.end_time(_before_bound(clip, two_three)))]
    want = [_expected(clip, scanner, shown, third, lower_bound=two_three),
            _expected(clip, scanner, shown, first, upper_bound=one_two),
            _expected(clip, scanner, shown, second, lower_bound=one_two, upper_bound=two_three)]
    assert detector.calls == [i for w in want for i in w.analysed]
    assert seeks == sum(w.seeks for w in want)


@pytest.mark.parametrize("clip_id", _matrix(BOUNDARY_CLIPS, fast={"video-start-0.021s-mkv"}))
def test_a_segment_elsewhere_between_them_in_time_does_not_separate_same_position_neighbours(
        clip_for, monkeypatch, clip_id):
    """A segment at another position (a small box in the corner, never found)
    read between two segments at the same position is not their neighbour:
    the bound between those two still applies to both."""
    clip = clip_for(clip_id)
    scanner = clip.scanner()
    shown = set(range(10, 71))
    corner = np.array([[0, 0], [40, 0], [40, 20], [0, 20]], dtype=np.float32)
    assert not scanner._boxes_overlap(clip.box, corner)
    first = clip.segment(clip.pts[15], clip.pts[20], "甲")
    elsewhere = dict(clip.segment(clip.pts[30], clip.pts[35], "角"), box=corner)
    second = clip.segment(clip.pts[45], clip.pts[50], "乙乙")
    bound = (first["end_pts"] + second["start_pts"]) / 2

    labels = scanner._phase4_find_timing([second, elsewhere, first], _LabelOn(clip, shown, elsewhere_nothing=True))

    by_text = {l.text: (l.start_pts, l.end_pts) for l in labels}
    assert by_text["甲"] == (clip.pts[10], clip.end_time(_before_bound(clip, bound)))
    assert by_text["乙乙"] == (clip.pts[_before_bound(clip, bound) + 1], clip.end_time(70))


@pytest.mark.parametrize("clip_id", _matrix(BOUNDARY_CLIPS, fast={"video-start-0.021s-mkv", "offset-h264"}))
def test_segments_whose_readings_overlap_do_not_bound_each_other(clip_for, monkeypatch, clip_id):
    """The earlier segment's last reading comes after the later one's first:
    both were read at once, so no bound between them can separate them (a
    midpoint would lie inside both). Neither bounds the other, and each is
    refined as if alone."""
    clip = clip_for(clip_id)
    scanner = clip.scanner()
    shown = set(range(10, 71))
    earlier = clip.segment(clip.pts[15], clip.pts[40], "甲")
    later = clip.segment(clip.pts[30], clip.pts[60], "乙乙乙乙")
    detector = _LabelOn(clip, shown)

    labels, seeks = _run(clip, scanner, [later, earlier], detector, monkeypatch)

    assert [(l.text, l.start_pts, l.end_pts) for l in labels] == [
        ("乙乙乙乙", clip.pts[10], clip.end_time(70)), ("甲", clip.pts[10], clip.end_time(70))]
    want_later = _expected(clip, scanner, shown, later)
    want_earlier = _expected(clip, scanner, shown, earlier)
    assert detector.calls == want_later.analysed + want_earlier.analysed
    assert seeks == want_later.seeks + want_earlier.seeks


CORNER_BOX = np.array([[0, 0], [40, 0], [40, 20], [0, 20]], dtype=np.float32)


def _lost_and_shared(clip, labels, whole):
    """(seconds of `whole` no label covers, frames two labels both claim,
    labels reaching outside `whole`), for the labels at the region box. A
    label claims a frame whose PTS lies in [start, end), and a gap between
    labels counts as lost only from half a frame on: an end is a frame's PTS
    plus 1 / fps, which in a millisecond time base can fall a fraction of a
    millisecond short of the next frame's PTS without leaving that frame
    unclaimed (the claims below check that)."""
    half = 0.5 / clip.fps
    spans = sorted((l.start_pts, l.end_pts) for l in labels if l.text != "角")
    t, lost = whole[0], 0.0
    for a, b in spans:
        if a - t >= half or t == whole[0]:
            lost += max(0.0, a - t)
        t = max(t, b)
    lost += max(0.0, whole[1] - t)
    claimed = [{i for i, p in enumerate(clip.pts) if a - half <= p < b - half} for a, b in spans]
    shared = sorted({i for x in range(len(claimed)) for y in range(x) for i in claimed[x] & claimed[y]})
    whole_frames = {i for i, p in enumerate(clip.pts) if whole[0] - half <= p < whole[1] - half}
    unclaimed = sorted(whole_frames - set().union(*claimed)) if claimed else sorted(whole_frames)
    if unclaimed:
        lost = max(lost, len(unclaimed) / clip.fps)
    outside = [(a, b) for a, b in spans if a < whole[0] - 1e-9 or b > whole[1] + 1e-9]
    return lost, shared, outside


def _orders(a, b, clip, elsewhere_reading):
    x = dict(clip.segment(*elsewhere_reading, "角"), box=CORNER_BOX)
    return {"earlier-first": [a, b], "later-first": [b, a], "elsewhere-between": [a, x, b]}


@pytest.mark.parametrize("clip_id,shapes", _matrix(
    [(c, "all-shapes") for c in BOUNDARY_CLIPS + [MS_CLIP]]
    + [(c, "some-shapes") for c in ("video-start-0.021s-mkv", MS_CLIP)],
    fast={"video-start-0.021s-mkv-some-shapes", f"{MS_CLIP}-some-shapes"}))
def test_same_position_segments_read_at_once_lose_no_span_in_any_list_order(clip_for, clip_id, shapes):
    """Two segments at one position whose readings overlap -- by 0.05-1.0 s,
    or true duplicates read over (nearly) the same time -- with the label
    found on every frame 5-95, through phase 4 and scan()'s post-processing,
    listed earlier first, later first, and with a segment at another
    position listed between them. With the same text or not, the labels at
    that position cover exactly the label's frames: none is lost, none is
    claimed twice, and nothing lies outside them; with the same text, they
    come out as one label with that text, once. The fast cases run every
    duplicate shape, and the overlapping ones after one of the two frames."""
    clip = clip_for(clip_id)
    shown = range(5, 96)
    whole = (clip.pts[shown[0]], clip.end_time(shown[-1]))
    every = shapes == "all-shapes"
    cases = []
    for frame in ((40, 41) if every else (41,)):
        for phase in (0.0, 0.5):
            last_reading = clip.pts[frame] + phase / clip.fps
            for overlap in (0.05, 0.2, 0.5, 1.0):
                for earlier_len, later_len in ((1.2, 0.6), (0.6, 1.2)):
                    first_reading = last_reading - overlap
                    for texts in (("第三十七集", "第三十七集"), ("甲", "乙乙乙乙")):
                        a = clip.segment(last_reading - earlier_len, last_reading, texts[0])
                        b = clip.segment(first_reading, first_reading + later_len, texts[1])
                        cases.append((f"overlap {overlap} s, lengths {earlier_len}/{later_len}, {texts[1]}", a, b,
                                      (first_reading, last_reading)))
    start = clip.pts[30]
    for shift_first in (0.0, 0.1, 0.5):
        for shift_last in (0.0, -0.1, -0.5):
            a = clip.segment(start + shift_first, start + 2.0, "第三十七集")
            b = clip.segment(start, start + 2.0 + shift_last, "第三十七集")
            cases.append((f"duplicates {shift_first}/{shift_last}", a, b, (start + 0.5, start + 1.0)))
    failures = []
    for name, a, b, elsewhere_reading in cases:
        for order, segments in _orders(a, b, clip, elsewhere_reading).items():
            labels = _scan_segments(clip.scanner(), segments, _LabelOn(clip, shown, elsewhere_nothing=True))
            lost, shared, outside = _lost_and_shared(clip, labels, whole)
            texts = [l.text for l in labels if l.text != "角"]
            if a["text"] == b["text"] and texts != [a["text"]]:
                failures.append((name, order, "same text did not come out as one label with that text", texts))
            if lost > 1e-9 or shared or outside:
                failures.append((name, order, round(lost, 3), shared[:3], outside,
                                 [(l.text, round(l.start_pts, 3), round(l.end_pts, 3)) for l in labels]))
    assert len(cases) == (2 if every else 1) * 2 * 4 * 2 * 2 + 9
    assert not failures, f"{len(failures)} of {3 * len(cases)} (case, order, lost s, shared frames, outside, labels): {failures[:6]}"


# Three segments at one position, as (first reading frame, last reading
# frame): one read inside (or across the end of) another's readings, and a
# third read apart from both, just after, well after, or before them.
NESTED_SHAPES = {
    "B inside A, C after": [(15, 45), (25, 30), (60, 80)],
    "B inside A, C just after": [(15, 45), (25, 30), (50, 80)],
    "B across the end of A, C after": [(15, 45), (40, 42), (55, 80)],
    "C before, B inside A": [(40, 80), (55, 60), (10, 25)],
    "C just before, B inside A": [(40, 80), (55, 60), (10, 35)],
}


@pytest.mark.parametrize("clip_id", _matrix(BOUNDARY_CLIPS + [MS_CLIP], fast={"video-start-0.021s-mkv", MS_CLIP}))
def test_nested_readings_at_one_position_keep_one_label_per_text_and_lose_no_span(clip_for, clip_id):
    """Three segments at one position, the label found on every frame 5-95,
    one segment read while another was, in every list order, through
    scan(). Each segment is bounded by the segment whose readings ended last
    before its own began and the one whose readings began first after its
    own ended, so the bounds the two sides of a gap use agree. Nothing is
    lost or claimed twice, and with one text there is one label, with that
    text once."""
    clip = clip_for(clip_id)
    shown = range(5, 96)
    whole = (clip.pts[shown[0]], clip.end_time(shown[-1]))
    failures, runs = [], 0
    for name, readings in NESTED_SHAPES.items():
        for texts in (("第三十七集",) * 3, ("甲甲甲", "乙乙乙", "丙丙丙")):
            segments = [clip.segment(clip.pts[a], clip.pts[b], t) for (a, b), t in zip(readings, texts)]
            for order in itertools.permutations(range(3)):
                runs += 1
                labels = _scan_segments(clip.scanner(), [segments[i] for i in order], _LabelOn(clip, shown))
                lost, shared, outside = _lost_and_shared(clip, labels, whole)
                out = [(l.text, round(l.start_pts, 3), round(l.end_pts, 3)) for l in labels]
                if texts[0] == texts[1] and [l.text for l in labels] != [texts[0]]:
                    failures.append((name, order, "not one label with the text once", out))
                if lost > 1e-9 or shared or outside:
                    failures.append((name, order, round(lost, 3), shared[:4], outside, out))
    assert runs == len(NESTED_SHAPES) * 2 * 6
    assert not failures, f"{len(failures)} failures in {runs} runs (shape, order, ...): {failures[:6]}"


def _label(start, end, text="第三十七集", pos_x=160, bbox_x_min=0.0):
    return LabelResult(start_pts=start, end_pts=end, text=text, pos_x=pos_x, pos_y=200,
                       bbox_x_min=bbox_x_min, bbox_y_min=0.0, bbox_x_max=319.0, bbox_y_max=160.0)


def test_a_removed_duplicate_extends_the_label_kept_to_both_spans(clip_for):
    """_remove_duplicates keeps the same label as before (the longer, the
    earlier on a tie) with its own text and position, but its span becomes
    the union of every duplicate it took the place of, directly or through a
    label that was itself removed later. Labels that are not duplicates keep
    their spans."""
    scanner = clip_for("video-start-0.021s-mkv").scanner()

    def spans(labels):
        return [(l.text, l.pos_x, l.bbox_x_min, l.start_pts, l.end_pts) for l in labels]

    shorter_first = [_label(1.0, 2.0, pos_x=150, bbox_x_min=5.0), _label(1.5, 3.5, pos_x=160)]
    assert spans(scanner._remove_duplicates(shorter_first)) == [("第三十七集", 160, 0.0, 1.0, 3.5)]
    longer_first = [_label(1.5, 3.5, pos_x=160), _label(1.0, 2.0, pos_x=150, bbox_x_min=5.0)]
    assert spans(scanner._remove_duplicates(longer_first)) == [("第三十七集", 160, 0.0, 1.0, 3.5)]
    tie = [_label(1.0, 2.0, pos_x=150), _label(1.5, 2.5, pos_x=160)]
    assert spans(scanner._remove_duplicates(tie)) == [("第三十七集", 150, 0.0, 1.0, 2.5)]
    # 0 is replaced by 1, and 1 by 2: 2 covers all three.
    chain = [_label(1.0, 2.0), _label(1.5, 2.6, pos_x=161), _label(2.4, 5.0, pos_x=162)]
    assert spans(scanner._remove_duplicates(chain)) == [("第三十七集", 162, 0.0, 1.0, 5.0)]

    # Not duplicates: both kept as they came (compared with fresh copies, as
    # _remove_duplicates may change the labels it is given).
    for other in (_label(1.5, 3.5, text="乙乙乙乙"), _label(1.5, 3.5, pos_x=300), _label(2.0, 3.5)):
        before = [_label(1.0, 2.0), LabelResult(**vars(other))]
        assert spans(scanner._remove_duplicates([_label(1.0, 2.0), other])) == spans(before), other


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))), ids=lambda o: "-".join(map(str, o)))
def test_a_chain_of_duplicates_comes_out_of_post_processing_as_one_label(clip_for, order):
    """Three same-text labels at one position, each overlapping the next:
    (1.0, 2.0), (1.9, 2.8), (2.3, 2.9). A label kept in place of a duplicate
    takes the duplicate's span at once, so the later comparisons see it: the
    first, widened to (1.0, 2.8), also removes the third. Widened only after
    every decision, it would stay beside the third, overlapping it, and the
    split-label merge after it would join the two as one label with the text
    twice. Run through the post-processing scan() applies after phase 4."""
    scanner = clip_for("video-start-0.021s-mkv").scanner()
    spans = [(1.0, 2.0), (1.9, 2.8), (2.3, 2.9)]
    labels = [_label(*spans[i]) for i in order]

    labels = scanner._merge_adjacent_labels(scanner._merge_split_labels(scanner._remove_duplicates(labels)))

    assert [(l.text, l.start_pts, l.end_pts) for l in labels] == [("第三十七集", 1.0, 2.9)]


@pytest.mark.parametrize("shape", [
    pytest.param([(1.0, 2.0), (1.9, 2.8), (2.3, 2.9)], id="each-overlapping-the-next"),
    pytest.param([(1.0, 3.0), (1.5, 2.0), (2.5, 4.0)], id="one-inside-another"),
])
@pytest.mark.parametrize("order", list(itertools.permutations(range(3))), ids=lambda o: "-".join(map(str, o)))
def test_labels_with_different_texts_are_left_alone_by_duplicate_removal(clip_for, shape, order):
    """Duplicate removal only ever acts on similar texts, so for labels with
    different texts it changes nothing -- not their spans, not which come
    out -- and what post-processing makes of them is exactly what it makes
    without that step (whatever _merge_split_labels then joins)."""
    scanner = clip_for("video-start-0.021s-mkv").scanner()
    texts = ["甲甲甲", "乙乙乙", "丙丙丙"]

    def make():
        return [_label(*shape[i], text=texts[i]) for i in order]

    labels = make()
    kept = scanner._remove_duplicates(labels)
    assert len(kept) == len(labels) and all(a is b for a, b in zip(kept, labels))
    assert [(l.start_pts, l.end_pts) for l in kept] == [shape[i] for i in order]

    def result(labels):
        return [(l.text, l.start_pts, l.end_pts, l.pos_x) for l in labels]

    with_removal = scanner._merge_adjacent_labels(scanner._merge_split_labels(scanner._remove_duplicates(make())))
    without = scanner._merge_adjacent_labels(scanner._merge_split_labels(make()))
    assert result(with_removal) == result(without)


@pytest.mark.parametrize("order", ["shorter-first", "longer-first"])
def test_duplicates_with_different_spans_come_out_as_one_label_covering_both(clip_for, order):
    """Two segments at one position with the same text, read at overlapping
    times. With a 0.6 s scan limit their refined spans differ (each reaches
    0.6 s past its own readings), and scan() must return one label covering
    both, not only the longer."""
    clip = clip_for("video-start-0.021s-mkv")
    scanner = clip.scanner()
    scanner.TIMING_SCAN_MAX_DURATION = 0.6
    shown = set(range(5, 96))
    shorter = clip.segment(clip.pts[25], clip.pts[30], "第三十七集")
    longer = clip.segment(clip.pts[28], clip.pts[50], "第三十七集")
    want = [_expected(clip, scanner, shown, s) for s in (shorter, longer)]
    assert want[0].start < want[1].start and want[0].end < want[1].end
    assert want[0].end - want[0].start < want[1].end - want[1].start
    segments = [shorter, longer] if order == "shorter-first" else [longer, shorter]

    labels = _scan_segments(scanner, segments, _LabelOn(clip, shown))

    assert [(l.start_pts, l.end_pts) for l in labels] == [(want[0].start, want[1].end)]


@pytest.mark.parametrize("clip_id", _matrix(BOUNDARY_CLIPS, fast={"video-start-0.021s-mkv", "offset-h264"}))
def test_a_label_on_the_first_or_last_frame_of_the_file(clip_for, monkeypatch, clip_id):
    """From the first frame, where the backward scan stops at time 0 (on the
    1.5 s offset clip, long before the first frame); and to the last frame,
    where the forward scan stops past the end of the stream."""
    clip = clip_for(clip_id)
    last_frame = len(clip.pts) - 1
    for shown, readings in ((range(0, 30), (8, 20)), (range(70, last_frame + 1), (80, 92))):
        scanner = clip.scanner()
        segment = clip.segment(clip.pts[readings[0]], clip.pts[readings[1]])
        detector = _LabelOn(clip, shown)

        labels, seeks = _run(clip, scanner, [segment], detector, monkeypatch)

        assert [(l.start_pts, l.end_pts) for l in labels] == [(clip.pts[shown[0]], clip.end_time(shown[-1]))]
        want = _expected(clip, scanner, set(shown), segment)
        assert detector.calls == want.analysed
        assert seeks == want.seeks


@pytest.mark.parametrize("clip_id", ["video-start-0.021s-mkv", "zero-start-h264"])
def test_a_sample_whose_seek_runs_out_of_retries_is_not_a_bracket_edge(clip_for, monkeypatch, caplog, clip_id):
    """A sample skipped because its seek ran out of retries is neither present
    nor absent, so the bracket reaches the next sample the scan could analyse:
    here 0.4 s, and the label's boundaries lie beyond the skipped sample."""
    clip = clip_for(clip_id)
    scanner = clip.scanner()
    step = scanner.TIMING_SCAN_INTERVAL
    shown = set(range(33, 68))
    segment = clip.segment(clip.pts[40], clip.pts[60])
    failing = [segment["start_pts"] - step, segment["end_pts"] + step]

    def skipped(t):
        return any(abs(t - f) < 1e-9 for f in failing)

    caplog.set_level(logging.WARNING, logger="videocr.pyav_adapter")
    exhausting = _ExhaustingSeeks(monkeypatch, skipped)
    detector = _LabelOn(clip, shown)
    labels = scanner._phase4_find_timing([segment], detector)

    assert [result for t, result in exhausting.calls if skipped(t)] == [False, False]
    assert sum(r.levelno == logging.WARNING for r in caplog.records) == 2
    assert [(l.start_pts, l.end_pts) for l in labels] == [(clip.pts[33], clip.end_time(67))]
    want = _expected(clip, scanner, shown, segment, skipped=skipped)
    assert detector.calls == want.analysed
    assert len(exhausting.calls) == want.seeks


def test_duration_limits_apply_to_the_refined_boundaries(clip_for, monkeypatch):
    """Frames 31-55 are exactly 1.00 s; the 0.2 s scan alone saw 0.6 s of it.
    Frames 31-69 are 1.56 s; the scan alone saw 1.2 s."""
    clip = clip_for("video-start-0.021s-mkv")

    kept, _ = _run(clip, clip.scanner(min_duration=0.9, max_duration=5.0),
                   [clip.segment(clip.pts[40], clip.pts[50])], _LabelOn(clip, range(31, 56)), monkeypatch)
    assert [(l.start_pts, l.end_pts) for l in kept] == [(clip.pts[31], clip.end_time(55))]

    dropped, _ = _run(clip, clip.scanner(min_duration=0.5, max_duration=1.4),
                      [clip.segment(clip.pts[40], clip.pts[60])], _LabelOn(clip, range(31, 70)), monkeypatch)
    assert dropped == [], "a label 1.56 s long was kept with a 1.4 s maximum"


def _bracket_of(present, limit, present_pts=None, absent=None, absent_pts=None, bound=None):
    return _ScanBracket(present=present, present_pts=present_pts, absent=absent, absent_pts=absent_pts,
                        limit=limit, bound=bound)


def test_a_start_whose_bracket_cannot_be_read_stays_on_the_scan_frame(clip_for, monkeypatch, caplog):
    """The one seek refinement makes that the scan did not make before is to a
    bound before the first reading. If it runs out of retries, the start is
    the frame on screen at the reading, read from there -- a frame's PTS, as
    every label time is -- and the frames before it stay unrefined."""
    clip = clip_for("video-start-0.021s-mkv")
    scanner = clip.scanner()
    first = clip.segment(clip.pts[25], clip.pts[45])
    second = clip.segment(clip.pts[52] + 0.01, clip.pts[70])  # read between two frames
    bound = (first["end_pts"] + second["start_pts"]) / 2
    caplog.set_level(logging.WARNING, logger="videocr.pyav_adapter")
    exhausting = _ExhaustingSeeks(monkeypatch, lambda t: abs(t - bound) < 1e-9)
    shown = range(20, 81)

    labels = scanner._phase4_find_timing([first, second], _LabelOn(clip, shown))

    assert [result for t, result in exhausting.calls if abs(t - bound) < 1e-9] == [False]
    assert sum(r.levelno == logging.WARNING for r in caplog.records) == 1
    assert [(l.start_pts, l.end_pts) for l in labels] == [
        (clip.pts[20], clip.end_time(_before_bound(clip, bound))), (clip.pts[52], clip.end_time(80))]


def test_a_boundary_stays_where_the_scan_left_it_when_its_frame_cannot_be_read(clip_for, monkeypatch):
    """If the frame on screen at the scan's last present time cannot be read,
    there is nothing to extend from, and the boundary is the frame the scan
    analysed there (its PTS, or its PTS + 1 / fps for an end). Unreachable in
    practice: that frame was read before, by the scan or, for a reading's
    time, by phase 3, with the same deterministic seek. For a reading's time
    the scan has no frame PTS, and the boundary is that time. A start whose
    last present time is on the last frame is still refined up to it (the
    stream ends right after that frame)."""
    clip = clip_for("offset-h264")
    scanner = clip.scanner()
    fps = clip.fps
    past_end = clip.pts[-1] + 1.0
    detector = _LabelOn(clip, range(len(clip.pts)))
    with PyAVCapture(str(clip.path)) as cap:
        assert scanner._refine_end(cap, detector, clip.box, clip.box,
                                   _bracket_of(past_end, past_end + 5.0)) == past_end
        assert scanner._refine_start(cap, detector, clip.box, clip.box,
                                     _bracket_of(past_end, clip.pts[90])) == past_end
        # With the frame PTS the scan read there, that frame's PTS and end.
        assert scanner._refine_start(cap, detector, clip.box, clip.box,
                                     _bracket_of(past_end, clip.pts[90], present_pts=clip.pts[97])) == clip.pts[97]
        assert scanner._refine_end(cap, detector, clip.box, clip.box,
                                   _bracket_of(past_end, past_end + 5.0, present_pts=clip.pts[97])) == clip.end_time(97)
        on_last_frame = clip.pts[-1] + 0.5 / fps
        assert scanner._refine_start(cap, detector, clip.box, clip.box,
                                     _bracket_of(on_last_frame, clip.pts[95])) == clip.pts[95]

    # The scan's own last present sample, whose seek fails the second time.
    clip = clip_for("video-start-0.021s-mkv")
    scanner = clip.scanner()
    segment = clip.segment(clip.pts[40], clip.pts[60])
    step = scanner.TIMING_SCAN_INTERVAL
    sample = segment["end_pts"] + step
    seen = []

    def second_seek_at_sample(t):
        seen.append(abs(t - sample) < 1e-9)
        return seen[-1] and seen.count(True) == 2

    _ExhaustingSeeks(monkeypatch, second_seek_at_sample)
    (label,) = scanner._phase4_find_timing([segment], _LabelOn(clip, range(31, 70)))
    assert seen.count(True) == 2
    assert (label.start_pts, label.end_pts) == (clip.pts[31], clip.end_time(65))
