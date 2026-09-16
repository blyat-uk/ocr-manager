"""Label phases must analyse exactly the frame each decision is recorded under.

Phase 1.5 must OCR exactly the frames phase 1 detected text in, and phases 3
and 4 must analyse the frame on screen at each time they record a reading or
a boundary under (see the section on phases 3 and 4 below).

Phase 1 samples a frame, runs detection on it and records
(frame_idx, pts, boxes). Phase 1.5 then fetches that frame again at native
resolution, crops each box out of it and OCRs the crop; phase 2 groups
boxes using that text. If the re-fetch lands on a different frame, a
label's text is read from a different moment than the detection that
located it.

These tests drive the real `_phase1_find_text_frames` and
`_batch_ocr_text_frames` with a detector that reports one large box on
every sample and an OCR engine that records what it is given, and assert
that for every phase-1 sample, phase 1.5 read a frame with the *same PTS*
and *byte-identical pixels*, and OCR'd the same crop.

Clips cover a zero start time (what every golden is) and a 1.5 s
container start time, in H.264, 10-bit HEVC, 23.976 fps and MKV. 1.5 s at
25 fps is a half-frame position (37.5 frames), which makes any
"position = round(pts * fps)" scheme ambiguous between adjacent frames,
and every-frame sampling makes sure those ambiguous frames, and the frames
just before each keyframe, are actually sampled. Zero-start H.264 is the
control that the old frame-index re-fetch also passed; zero-start 10-bit
HEVC in MP4 is not -- see its entry in CLIPS.
"""
import bisect
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from videocr import pyav_adapter
from videocr.label_scanner import LabelScanner
from videocr.pyav_adapter import FFmpegNVDECCapture, PyAVCapture

FRAMES = 100

# id -> (container ext, pix_fmt, codec, rate, extra ffmpeg args)
CLIPS = {
    "zero-start-h264": ("mp4", "yuv420p", "libx264", "25", []),
    # Zero start, but in MP4 a seek to one of the frames just before an
    # HEVC keyframe's PTS lands on that keyframe (the demuxer finds
    # keyframes by DTS, which B-frame reordering puts ahead of the PTS).
    "zero-start-h265-10bit": ("mp4", "yuv420p10le", "libx265", "25", []),
    "offset-h264": ("mp4", "yuv420p", "libx264", "25", ["-output_ts_offset", "1.5"]),
    "offset-h265-10bit": ("mp4", "yuv420p10le", "libx265", "25", ["-output_ts_offset", "1.5"]),
    "offset-h264-23.976fps": ("mp4", "yuv420p", "libx264", "24000/1001", ["-output_ts_offset", "1.5"]),
    "offset-h264-mkv": ("mkv", "yuv420p", "libx264", "25", ["-output_ts_offset", "1.5"]),
    # Taller than LabelScanner.SCAN_HEIGHT above the dialogue cutoff, so
    # phase 1 detects on a downscaled copy rather than on the frame itself.
    "zero-start-h264-960p": ("mp4", "yuv420p", "libx264", "25", []),
    # Like Youxia Zhanji (4K HEVC MKV): the video stream starts 0.021 s --
    # 0.525 of a frame at 25 fps -- after an audio stream at 0, so the
    # container starts at 0 but every frame's PTS * fps is k + 0.525.
    "video-start-0.021s-mkv": ("mkv", "yuv420p", "libx264", "25", []),
    "video-start-0.021s-h265-10bit-mkv": ("mkv", "yuv420p10le", "libx265", "25", []),
}
SIZES = {"zero-start-h264-960p": "1280x960"}
# Clips whose video stream is remuxed to start this long after the audio.
VIDEO_START = {"video-start-0.021s-mkv": 0.021, "video-start-0.021s-h265-10bit-mkv": 0.021}


def _have_encoder(name):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines())


# Phase 1's own sampling (every 0.5 s), and every frame.
SAMPLING = [pytest.param(None, id="every-0.5s"), pytest.param("every-frame", id="every-frame")]

# The second range starts before the offset clips' first frame (1.5 s); the
# third starts inside them, mid-GOP.
RANGES = [
    pytest.param(None, None, id="whole-clip"),
    pytest.param("0:01", "0:03.5", id="from-1s"),
    pytest.param("0:02.3", "0:03.9", id="from-2.3s"),
]


def _matrix(cases, fast):
    """Parametrize over `cases` (single values, or pytest.param tuples whose
    ids are joined), keeping the cases whose id is in `fast` in the fast suite
    and marking the rest of the matrix slow. `fast` holds at least one case
    per property a test exists for; `-m slow` runs the remainder."""
    params = []
    for case in cases:
        parts = case if isinstance(case, tuple) else (case,)
        values, ids = [], []
        for part in parts:
            if hasattr(part, "values"):
                values.extend(part.values)
                ids.append(part.id)
            else:
                values.append(part)
                ids.append(str(part))
        case_id = "-".join(ids)
        marks = () if case_id in fast else (pytest.mark.slow,)
        params.append(pytest.param(*values, id=case_id, marks=marks))
    unknown = set(fast) - {p.id for p in params}
    assert not unknown, f"fast cases that are not in the matrix: {unknown}"
    return params


def _encode(path, pix_fmt, codec, rate, extra, size="320x240", video_start=None):
    gop = (["-x265-params", "keyint=10:min-keyint=10:log-level=error"]
           if codec == "libx265" else ["-g", "10"])
    if video_start is None:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}",
             "-frames:v", str(FRAMES), "-pix_fmt", pix_fmt, "-c:v", codec,
             *gop, *extra, str(path)],
            check=True, capture_output=True,
        )
        return
    # Encode video and audio both starting at 0 (with a millisecond encoder
    # time base, so the offset is not rounded to a frame), then remux with
    # the video input delayed.
    plain = path.with_name(f"{path.stem}-plain{path.suffix}")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}",
         "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
         "-map", "0:v", "-map", "1:a", "-frames:v", str(FRAMES), "-t", str(FRAMES / 25 + 0.2),
         "-pix_fmt", pix_fmt, "-c:v", codec, *gop, "-enc_time_base:v", "1:1000",
         "-c:a", "aac", *extra, str(plain)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-itsoffset", str(video_start), "-i", str(plain), "-i", str(plain),
         "-map", "0:v", "-map", "1:a", "-c", "copy", str(path)],
        check=True, capture_output=True,
    )
    plain.unlink()


def _start_time(path):
    container = av.open(str(path))
    try:
        return (container.start_time or 0) / 1_000_000
    finally:
        container.close()


def _video_start_time(path):
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        return float(stream.start_time * stream.time_base)
    finally:
        container.close()


def _assert_adjacent_frames_differ(path):
    """A re-fetch one frame early or late must never compare equal by
    accident, so every frame must differ from the next in the region
    phase 1.5 crops from (above the dialogue cutoff, the top 80%)."""
    container = av.open(str(path))
    try:
        previous = None
        count = 0
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="bgr24")
            top = img[: int(img.shape[0] * 0.8)]
            assert previous is None or not np.array_equal(previous, top), path
            previous = top
            count += 1
        return count
    finally:
        container.close()


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    """Clip id -> path. A clip whose encoder is missing is left out, and the
    tests that use it skip (see _clip)."""
    root = tmp_path_factory.mktemp("label_frame_identity")
    out = {}
    for cid, (ext, pix_fmt, codec, rate, extra) in CLIPS.items():
        if not _have_encoder(codec):
            continue
        path = root / f"{cid}.{ext}"
        _encode(path, pix_fmt, codec, rate, extra, SIZES.get(cid, "320x240"), VIDEO_START.get(cid))
        expected_start = 1.5 if extra else 0.0
        assert _start_time(path) == pytest.approx(expected_start, abs=1e-3), cid
        assert _video_start_time(path) == pytest.approx(VIDEO_START.get(cid, expected_start), abs=1e-6), cid
        assert _assert_adjacent_frames_differ(path) == FRAMES, cid
        out[cid] = path
    return out


def _clip(clips, clip_id):
    if clip_id not in clips:
        pytest.skip(f"{CLIPS[clip_id][2]} encoder not available for {clip_id}")
    return clips[clip_id]


def _scanner(path, sampling):
    # Metadata exactly as videocr.video.Video probes it (an MKV has no
    # stream frame count, so this is not the same as stream.frames).
    with PyAVCapture(str(path)) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scanner = LabelScanner(str(path), fps, width, height, num_frames, None, None, None, None)
    if sampling == "every-frame":
        # int(fps * (1 / fps)) == 1 for every rate used here.
        scanner.SAMPLE_INTERVAL_SECONDS = 1.0 / fps
        assert max(1, int(scanner.fps * scanner.SAMPLE_INTERVAL_SECONDS)) == 1
    return scanner


class CpuFFmpegCapture(FFmpegNVDECCapture):
    """The subprocess fallback, decoding on the CPU."""

    def __init__(self, video_path, **kwargs):
        super().__init__(video_path, use_gpu=False, **kwargs)


class _WholeRegionDetector:
    """One box covering the whole detection input, on every sample, so
    every sample becomes a text frame and its crop is most of the frame."""

    def predict(self, frame):
        h, w = frame.shape[:2]
        return [{"dt_polys": [[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]]}]


class _Recorder:
    """Records every frame a capture hands out, and what OCR receives."""

    def __init__(self):
        self.reads = {}        # pts -> frame, for every frame read
        self.last = None       # (pts, frame) of the most recent read
        self.ocr_calls = []    # (pts, frame, crop) at the time of each OCR call

    def install(self, monkeypatch, capture_cls):
        real_read = capture_cls.read
        recorder = self

        def recording_read(cap):
            ok, frame = real_read(cap)
            if ok and frame is not None:
                pts = cap.get_last_pts()
                recorder.last = (pts, frame.copy())
                recorder.reads.setdefault(pts, frame.copy())
            return ok, frame

        monkeypatch.setattr(capture_cls, "read", recording_read)

    def ocr(self):
        recorder = self

        class _RecordingOCR:
            def predict(self, crop):
                pts, frame = recorder.last
                recorder.ocr_calls.append((pts, frame, crop.copy()))
                return []

        return _RecordingOCR()


def _run_phases(scanner, recorder, time_start, time_end):
    text_frames = scanner._phase1_find_text_frames(_WholeRegionDetector(), time_start, time_end)
    phase1_frames = dict(recorder.reads)
    recorder.last = None
    augmented = scanner._batch_ocr_text_frames(text_frames, recorder.ocr())
    return text_frames, phase1_frames, augmented


def _assert_phase15_matches_phase1(scanner, recorder, text_frames, phase1_frames, augmented):
    assert len(augmented) == len(text_frames)
    assert len(recorder.ocr_calls) == len(text_frames), (
        f"phase 1.5 OCR'd {len(recorder.ocr_calls)} crops for {len(text_frames)} text frames "
        "(a frame it failed to fetch is skipped)"
    )
    cutoff = scanner.dialogue_cutoff_y
    mismatches = []
    for (frame_idx, pts, boxes), (got_pts, got_frame, got_crop) in zip(text_frames, recorder.ocr_calls):
        (box,) = boxes
        sampled = phase1_frames[pts]
        expected_crop, _ = scanner._crop_box_region(sampled[:cutoff, :], box)
        expected_crop, _ = scanner._resize_max_dimension(expected_crop, scanner.RECOGNIZE_HEIGHT)
        same_pts = got_pts == pts
        same_frame = got_frame.shape == sampled.shape and np.array_equal(got_frame, sampled)
        same_crop = got_crop.shape == expected_crop.shape and np.array_equal(got_crop, expected_crop)
        if not (same_pts and same_frame and same_crop):
            mismatches.append(
                f"frame_idx {frame_idx}: phase 1 sampled PTS {pts:.4f}, phase 1.5 read PTS "
                f"{got_pts:.4f} (pixels identical: {same_frame}, OCR crop identical: {same_crop})"
            )
    assert not mismatches, (
        f"{len(mismatches)} of {len(text_frames)} phase-1.5 fetches are not the frame phase 1 "
        "sampled:\n  " + "\n  ".join(mismatches)
    )


@pytest.mark.parametrize("clip_id,sampling,time_start,time_end", _matrix(
    [(c, sampling, r) for c in CLIPS for sampling in SAMPLING for r in RANGES],
    fast={
        # Container start: index origin, and half-frame ties after phase 1's seek.
        "offset-h264-every-frame-whole-clip", "offset-h264-every-0.5s-whole-clip",
        "offset-h264-every-frame-from-2.3s",
        # HEVC in MP4: seeks landing after the target.
        "zero-start-h265-10bit-every-frame-whole-clip",
        # Video 0.525 frame after a zero container start, with a seek.
        "video-start-0.021s-mkv-every-frame-from-1s",
    }))
def test_phase15_reads_the_frames_phase1_sampled(clips, monkeypatch, clip_id, sampling, time_start, time_end):
    scanner = _scanner(_clip(clips, clip_id), sampling)
    recorder = _Recorder()
    recorder.install(monkeypatch, PyAVCapture)

    text_frames, phase1_frames, augmented = _run_phases(scanner, recorder, time_start, time_end)

    # Non-vacuous: several samples, and every-frame sampling really sampled
    # adjacent frames.
    assert len(text_frames) >= 3
    if sampling == "every-frame":
        assert len(text_frames) >= 30
    _assert_phase15_matches_phase1(scanner, recorder, text_frames, phase1_frames, augmented)


@pytest.mark.parametrize("sampling", SAMPLING)
def test_phase15_reads_the_frames_phase1_sampled_on_the_offset_fixture(offset_video, monkeypatch, sampling):
    """The shared 1.5 s-offset fixture (10 frames), whole clip."""
    scanner = _scanner(offset_video, sampling)
    recorder = _Recorder()
    recorder.install(monkeypatch, PyAVCapture)

    text_frames, phase1_frames, augmented = _run_phases(scanner, recorder, None, None)

    assert len(text_frames) >= (10 if sampling == "every-frame" else 1)
    _assert_phase15_matches_phase1(scanner, recorder, text_frames, phase1_frames, augmented)


@pytest.mark.parametrize("clip_id,sampling", _matrix(
    [(c, sampling) for c in ["zero-start-h264", "offset-h264"] for sampling in SAMPLING],
    fast={"offset-h264-every-frame"}))
def test_ffmpeg_fallback_phase15_reads_the_frames_phase1_sampled(clips, monkeypatch, clip_id, sampling):
    """The subprocess backend positions by frame ordinal and estimates PTS
    as ordinal / fps + start time -- a different frame-index meaning from
    PyAVCapture's. Phase 1.5 must still land on phase 1's frames through it."""
    if not pyav_adapter.FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg CLI not available")
    monkeypatch.setattr("videocr.label_scanner.Capture", CpuFFmpegCapture)
    scanner = _scanner(_clip(clips, clip_id), sampling)
    recorder = _Recorder()
    recorder.install(monkeypatch, FFmpegNVDECCapture)

    text_frames, phase1_frames, augmented = _run_phases(scanner, recorder, None, None)

    assert len(text_frames) >= 3
    _assert_phase15_matches_phase1(scanner, recorder, text_frames, phase1_frames, augmented)


def test_phase15_never_ocrs_a_frame_other_than_the_one_asked_for(clips, monkeypatch):
    """If the capture lands on a frame with a different PTS, phase 1.5 keeps
    the box without text (as for an unreadable frame) rather than reading
    the label off the wrong frame."""
    real_seek = PyAVCapture.seek_to_pts

    def seek_one_frame_late(cap, pts):
        return real_seek(cap, pts + 1.0 / cap.get(cv2.CAP_PROP_FPS))

    scanner = _scanner(_clip(clips, "zero-start-h264"), None)
    text_frames = scanner._phase1_find_text_frames(_WholeRegionDetector(), None, None)
    assert len(text_frames) >= 3

    monkeypatch.setattr(PyAVCapture, "seek_to_pts", seek_one_frame_late)
    recorder = _Recorder()
    augmented = scanner._batch_ocr_text_frames(text_frames, recorder.ocr())

    assert recorder.ocr_calls == []
    assert [(idx, pts) for idx, pts, _ in augmented] == [(idx, pts) for idx, pts, _ in text_frames]
    assert all(entry["text"] is None for _, _, entries in augmented for entry in entries)


# --- seek_to_pts(): the capture-level contract phase 1.5 relies on -----------

def _every_frame(capture_cls, path):
    with capture_cls(str(path)) as cap:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append((cap.get_last_pts(), frame.copy()))
    assert len(frames) == FRAMES
    return frames


def _visit_order(n, stride=1):
    """Forwards, then backwards, then an interleaved jump pattern, so every
    seek direction and distance (including the next frame) is covered."""
    forwards = list(range(0, n, stride))
    backwards = forwards[::-1]
    jumps = [i for pair in zip(forwards, backwards) for i in pair]
    return forwards + backwards + jumps


def _assert_seek_lands(cap, frames, order):
    wrong = []
    for i in order:
        pts, expected = frames[i]
        assert cap.seek_to_pts(pts)
        ok, frame = cap.read()
        if not (ok and cap.get_last_pts() == pts and np.array_equal(frame, expected)):
            wrong.append((i, pts, cap.get_last_pts() if ok else None))
    assert not wrong, f"seek_to_pts landed on the wrong frame (index, wanted PTS, got PTS): {wrong[:10]}"


@pytest.mark.parametrize("clip_id", _matrix(
    list(CLIPS), fast={"offset-h264", "zero-start-h265-10bit", "video-start-0.021s-mkv"}))
def test_seek_to_pts_lands_on_exactly_that_frame(clips, clip_id):
    frames = _every_frame(PyAVCapture, _clip(clips, clip_id))
    with PyAVCapture(str(_clip(clips, clip_id))) as cap:
        _assert_seek_lands(cap, frames, _visit_order(len(frames)))


@pytest.mark.parametrize("clip_id", _matrix(["zero-start-h264", "offset-h264"], fast={"offset-h264"}))
def test_ffmpeg_fallback_seek_to_pts_lands_on_exactly_that_frame(clips, clip_id):
    if not pyav_adapter.FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg CLI not available")
    frames = _every_frame(CpuFFmpegCapture, _clip(clips, clip_id))
    with CpuFFmpegCapture(str(_clip(clips, clip_id))) as cap:
        # Every seek restarts ffmpeg, so visit a subset, including frame 0
        # and consecutive frames (which must not restart it).
        _assert_seek_lands(cap, frames, _visit_order(len(frames), stride=7) + [40, 41, 42, 0])


@pytest.mark.parametrize("clip_id", ["zero-start-h264", "offset-h264", "offset-h265-10bit"])
def test_seek_to_pts_between_and_outside_frames(clips, clip_id):
    frames = _every_frame(PyAVCapture, _clip(clips, clip_id))
    pts = [p for p, _ in frames]
    with PyAVCapture(str(_clip(clips, clip_id))) as cap:
        # Between two frames: the later one.
        assert cap.seek_to_pts((pts[30] + pts[31]) / 2)
        ok, frame = cap.read()
        assert ok and cap.get_last_pts() == pts[31] and np.array_equal(frame, frames[31][1])
        # Before the first frame: the first frame.
        assert cap.seek_to_pts(pts[0] - 1.0)
        ok, frame = cap.read()
        assert ok and cap.get_last_pts() == pts[0] and np.array_equal(frame, frames[0][1])
        # After a seek parked a frame, a seek past the end must not leave
        # that frame behind for the next read().
        assert cap.seek_to_pts(pts[50])
        assert not cap.seek_to_pts(pts[-1] + 1.0)
        assert cap.read() == (False, None)
        # And the capture is still usable afterwards.
        assert cap.seek_to_pts(pts[10])
        ok, frame = cap.read()
        assert ok and cap.get_last_pts() == pts[10] and np.array_equal(frame, frames[10][1])


class _LateSeekingContainer:
    """Wraps a PyAV container so every seek lands `late_seconds` after the
    timestamp asked for -- what a demuxer that indexes keyframes by DTS can
    do for a target just before a keyframe's PTS."""

    def __init__(self, container, stream, late_seconds):
        self._container = container
        self._late = int(round(late_seconds / stream.time_base))
        self.seeks = []

    def __getattr__(self, name):
        return getattr(self._container, name)

    def seek(self, offset, **kwargs):
        self.seeks.append(offset)
        return self._container.seek(offset + self._late, **kwargs)


@pytest.mark.parametrize("clip_id", ["zero-start-h264", "offset-h264"])
def test_seek_to_pts_retries_from_further_back_when_a_seek_lands_late(clips, clip_id):
    frames = _every_frame(PyAVCapture, _clip(clips, clip_id))
    with PyAVCapture(str(_clip(clips, clip_id))) as cap:
        late = _LateSeekingContainer(cap.container, cap.stream, late_seconds=1.5)
        cap.container = late
        # Frames far enough in that the first seek lands after them.
        _assert_seek_lands(cap, frames, [60, 75, 99, 62])
        assert len(late.seeks) > 4, "the late seeks never needed a retry, so this proves nothing"


# --- retained crops: phase 1.5 without re-fetching ----------------------------

# A label mask inside the region phase 1.5 crops from, so crops taken before
# the mask is applied could not compare equal.
MASKS = [(20, 30, 90, 40)]


def _masked_scanner(path, sampling):
    scanner = _scanner(path, sampling)
    scanner.label_mask_crops = MASKS
    return scanner


class _TwoBandDetector:
    """Two separate wide boxes per sample (so neither covers the whole
    region, and a frame's crops are more than one)."""

    def predict(self, frame):
        h, w = frame.shape[:2]
        return [{"dt_polys": [
            [[0, 0], [w - 1, 0], [w - 1, h // 3], [0, h // 3]],
            [[w // 4, h // 2], [w - 1, h // 2], [w - 1, h - 1], [w // 4, h - 1]],
        ]}]


class _Counting:
    """Counts calls to a PyAVCapture method."""

    def __init__(self, monkeypatch, name):
        self.calls = 0
        real = getattr(PyAVCapture, name)
        counter = self

        def counted(cap, *args, **kwargs):
            counter.calls += 1
            return real(cap, *args, **kwargs)

        monkeypatch.setattr(PyAVCapture, name, counted)


def _expected_crops(scanner, text_frames, phase1_frames):
    """What phase 1.5 must OCR, computed from the frames phase 1 read."""
    expected = []
    for _, pts, boxes in text_frames:
        frame = phase1_frames[pts].copy()
        scanner._apply_label_masks(frame)
        for box in boxes:
            crop, _ = scanner._crop_box_region(frame[: scanner.dialogue_cutoff_y, :], box)
            crop, _ = scanner._resize_max_dimension(crop, scanner.RECOGNIZE_HEIGHT)
            expected.append(crop)
    return expected


def _assert_same_crops(got, expected):
    assert len(got) == len(expected)
    for i, (g, e) in enumerate(zip(got, expected)):
        assert g.shape == e.shape and np.array_equal(g, e), f"OCR crop {i} differs"


@pytest.mark.parametrize("clip_id,sampling", _matrix(
    [(c, sampling) for c in CLIPS for sampling in SAMPLING],
    # A container start, and a frame phase 1 detects on a downscaled copy of.
    fast={"offset-h264-every-frame", "zero-start-h264-960p-every-0.5s"}))
def test_retained_crops_are_what_phase15_would_have_fetched(clips, monkeypatch, clip_id, sampling):
    from videocr.label_scanner import _RetainedCrops

    scanner = _masked_scanner(_clip(clips, clip_id), sampling)
    recorder = _Recorder()
    recorder.install(monkeypatch, PyAVCapture)
    retained = _RetainedCrops(budget_bytes=1 << 30)

    text_frames = scanner._phase1_find_text_frames(_TwoBandDetector(), None, None, retain=retained)
    phase1_frames = dict(recorder.reads)
    assert len(text_frames) >= 3 and retained.refused == 0

    seeks = _Counting(monkeypatch, "seek_to_pts")
    ocr = recorder.ocr()
    recorder.last = (None, None)
    augmented = scanner._batch_ocr_text_frames(text_frames, ocr, retained=retained)
    assert seeks.calls == 0, "phase 1.5 fetched frames although phase 1 kept every crop"
    assert retained.nbytes == 0, "phase 1.5 did not release the crops it used"
    retained_crops = [crop for _, _, crop in recorder.ocr_calls]

    # The same text frames through the fetch path, which the tests above
    # pin to phase 1's frames.
    recorder.ocr_calls.clear()
    fetched = scanner._batch_ocr_text_frames(text_frames, ocr)
    assert seeks.calls == len(text_frames)
    fetched_crops = [crop for _, _, crop in recorder.ocr_calls]

    expected = _expected_crops(scanner, text_frames, phase1_frames)
    _assert_same_crops(retained_crops, expected)
    _assert_same_crops(fetched_crops, expected)
    assert [(i, p, [e["text"] for e in b]) for i, p, b in augmented] == \
        [(i, p, [e["text"] for e in b]) for i, p, b in fetched]


@pytest.mark.parametrize("clip_id", ["zero-start-h264", "offset-h265-10bit"])
def test_crops_over_the_budget_are_fetched_by_pts(clips, monkeypatch, clip_id):
    from videocr.label_scanner import _RetainedCrops

    scanner = _masked_scanner(_clip(clips, clip_id), "every-frame")
    recorder = _Recorder()
    recorder.install(monkeypatch, PyAVCapture)

    # Size one text frame's crops, then allow a little under a third of them.
    probe = _RetainedCrops(budget_bytes=1 << 30)
    text_frames = scanner._phase1_find_text_frames(_TwoBandDetector(), None, None, retain=probe)
    per_frame = probe.nbytes // len(text_frames)
    budget = per_frame * len(text_frames) // 3 + per_frame // 2

    recorder.reads.clear()
    retained = _RetainedCrops(budget_bytes=budget)
    text_frames = scanner._phase1_find_text_frames(_TwoBandDetector(), None, None, retain=retained)
    phase1_frames = dict(recorder.reads)
    assert retained.refused > 0 and retained.refused < len(text_frames)
    assert retained.peak_nbytes <= budget

    seeks = _Counting(monkeypatch, "seek_to_pts")
    recorder.last = (None, None)
    scanner._batch_ocr_text_frames(text_frames, recorder.ocr(), retained=retained)
    assert seeks.calls == retained.refused
    _assert_same_crops([crop for _, _, crop in recorder.ocr_calls],
                       _expected_crops(scanner, text_frames, phase1_frames))


def test_retained_crops_are_only_used_for_the_boxes_they_were_cut_for(clips):
    from videocr.label_scanner import _RetainedCrops

    boxes = [np.zeros((4, 2), dtype=np.float32)]
    crops = [np.ones((3, 5, 3), dtype=np.uint8)]
    retained = _RetainedCrops(budget_bytes=100)
    assert retained.offer(12, 0.48, boxes, crops)
    assert retained.nbytes == 45
    assert not retained.offer(24, 0.96, boxes, [np.ones((10, 10, 3), dtype=np.uint8)])
    assert retained.refused == 1

    # A different boxes list for the same frame (even an equal one) is a miss.
    assert retained.take(12, 0.48, [b.copy() for b in boxes]) is None
    assert retained.nbytes == 0
    assert retained.offer(12, 0.48, boxes, crops)
    assert retained.take(12, 0.48, boxes) is crops
    assert retained.take(12, 0.48, boxes) is None


def test_scan_serves_phase15_from_phase1_without_opening_the_video_again(clips, monkeypatch):
    scanner = _masked_scanner(_clip(clips, "offset-h264"), None)
    opens = _Counting(monkeypatch, "__enter__")
    seeks = _Counting(monkeypatch, "seek_to_pts")
    monkeypatch.setattr(LabelScanner, "_phase2_group_by_position", lambda self, text_frames, progress=None: [])

    recorder = _Recorder()
    recorder.install(monkeypatch, PyAVCapture)
    recorder.last = (None, None)
    scanner.scan(_TwoBandDetector(), recorder.ocr(), "", "", 1.5)

    assert len(recorder.ocr_calls) >= 3
    assert opens.calls == 1, "phase 1.5 opened the video although phase 1 kept every crop"
    assert seeks.calls == 0


# --- phases 3 and 4: the frame on screen at each decision's time -------------
#
# Phase 3 OCRs a cluster every 0.5 s from its first PTS and records each
# reading under that sample time; phase 4 runs detection 0.2 s apart around
# each segment and records the label's start and end under those times. Each
# must therefore analyse the frame on screen at the time: the last frame
# whose PTS is at most t (within ON_SCREEN_TOLERANCE), the first frame when t
# precedes it, and no frame once t is past the last frame's duration.
#
# The reference is a plain sequential decode of the clip, independent of any
# capture seek. Phase 3 is driven with groups as phase 2 hands them over
# (first/last PTS are frame PTS) and phase 4 with segments as phase 3 hands
# them over (start/end are phase 3 sample times), through the real sampling
# loops, with engines that record the frame the capture last read and the
# exact input they were given.

ON_SCREEN_TOLERANCE = 1e-6

# Zero start; video 0.021 s after a zero container start (0.525 frame, as on
# Youxia Zhanji); 1.5 s container start (half-frame ties at 25 fps); HEVC in
# MP4, whose seeks can land after the target; 23.976 fps; MKV.
PHASE34_FAST = {"video-start-0.021s-mkv", "offset-h264", "zero-start-h265-10bit"}
PHASE34_CLIPS = [
    "zero-start-h264",
    "video-start-0.021s-mkv",
    "video-start-0.021s-h265-10bit-mkv",
    "offset-h264",
    "zero-start-h265-10bit",
    "offset-h265-10bit",
    "offset-h264-23.976fps",
    "offset-h264-mkv",
]


def _decode_reference(path):
    """[(PTS, BGR frame)] for every frame, from one sequential PyAV decode."""
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        frames = [(float(frame.pts * stream.time_base), frame.to_ndarray(format="bgr24"))
                  for frame in container.decode(stream)]
    finally:
        container.close()
    pts = [p for p, _ in frames]
    assert all(a < b for a, b in zip(pts, pts[1:])), f"{path}: PTS not strictly increasing"
    return frames


def _on_screen(reference, t, fps):
    """Index of the frame on screen at `t` in `reference`, or None past the end."""
    pts = [p for p, _ in reference]
    i = bisect.bisect_right(pts, t + ON_SCREEN_TOLERANCE) - 1
    if i < 0:
        return 0
    if i == len(pts) - 1 and t + ON_SCREEN_TOLERANCE >= pts[i] + 1.0 / fps:
        return None
    return i


def _region_box(scanner):
    """The whole region above the dialogue cutoff, as a 4x2 box."""
    w, h = scanner.width, scanner.dialogue_cutoff_y
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


class _Mismatches:
    """Samples whose analysed frame is not the one on screen at their time."""

    def __init__(self, what):
        self.what = what
        self.checked = 0
        self.early = 0
        self.late = 0
        self.lines = []

    def check(self, t, reference, fps, got_pts, got_frame, got_input, expected_input):
        self.checked += 1
        want_pts, want_frame = reference[_on_screen(reference, t, fps)]
        same_frame = got_frame is not None and got_frame.shape == want_frame.shape and np.array_equal(got_frame, want_frame)
        same_input = got_input.shape == expected_input.shape and np.array_equal(got_input, expected_input)
        if got_pts == want_pts and same_frame and same_input:
            return
        if got_pts is not None and got_pts < want_pts:
            self.early += 1
        elif got_pts is not None and got_pts > want_pts:
            self.late += 1
        self.lines.append(f"t={t:.6f}: on screen PTS {want_pts:.6f}, analysed PTS "
                          f"{got_pts if got_pts is None else format(got_pts, '.6f')} "
                          f"(pixels identical: {same_frame}, engine input identical: {same_input})")

    def missing(self, message):
        self.lines.append(message)

    def assert_none(self):
        assert not self.lines, (
            f"{self.what}: {len(self.lines)} of {self.checked} samples did not analyse the frame on "
            f"screen at their time ({self.early} early, {self.late} late):\n  " + "\n  ".join(self.lines[:40])
        )


class _Phase3OCR:
    """Reads one confident line covering the whole crop, named after the call
    ("call0", "call1", ...), so each reading phase 3 records under a time can
    be traced to the frame the capture had read and the crop OCR was given."""

    def __init__(self, recorder):
        self.recorder = recorder
        self.calls = []  # (pts, frame, crop)

    def predict(self, crop):
        pts, frame = self.recorder.last
        self.calls.append((pts, frame, crop.copy()))
        h, w = crop.shape[:2]
        poly = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        return [{"rec_texts": [f"call{len(self.calls) - 1}"], "rec_scores": [1.0], "rec_polys": [poly]}]


class _Phase4Detector:
    """Finds the whole detection input as one box on every call, so every
    phase 4 scan runs to its time bound, and records the frame the capture had
    read and the input it was given."""

    def __init__(self, recorder):
        self.recorder = recorder
        self.calls = []  # (pts, frame, input)

    def predict(self, image):
        pts, frame = self.recorder.last
        self.calls.append((pts, frame, image.copy()))
        h, w = image.shape[:2]
        return [{"dt_polys": [[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]]}]


def _phase3_group_runs(scanner, reference):
    """Lists of groups for separate phase 3 calls. Groups in one call are
    disjoint in time, so each is its own cluster. One call has a group per
    frame (sampled once, at that frame's PTS); the others have groups spanning
    25 frames (sampled at +0, +0.5 and +1.0 s), starting at every frame."""
    pts = [p for p, _ in reference]
    box = _region_box(scanner)
    runs = [[{"first_pts": p, "last_pts": p, "encompassing_box": box} for p in pts]]
    span = min(25, len(pts) - 1)
    for first in range(span + 1):
        groups = [{"first_pts": pts[i], "last_pts": pts[i + span], "encompassing_box": box}
                  for i in range(first, len(pts) - span, span + 1)]
        if groups:
            runs.append(groups)
    return runs


def _phase3_sample_times(scanner, group):
    """The times phase 3 records a group's readings under."""
    times, t = [], group["first_pts"]
    while t <= group["last_pts"]:
        times.append(t)
        t += scanner.SAMPLE_INTERVAL_SECONDS
    return times


def _phase4_segments(scanner, reference):
    """Segments starting at every other frame's PTS, alternately 0.5 s later,
    each 0.5 s long; plus starts of 0.62 s and 0.61 s, whose backward scans
    reach 0.02 s and 0.01 s (inside the first frame's duration)."""
    pts = [p for p, _ in reference]
    box = _region_box(scanner)
    starts = [pts[i] + 0.5 if n % 2 else pts[i] for n, i in enumerate(range(0, len(pts), 2))]
    starts += [0.62, 0.61]
    return [{"box": box, "text": "label", "confidence": 1.0, "start_pts": s, "end_pts": s + 0.5}
            for s in starts]


def _phase4_sample_times(scanner, segment, reference):
    """The times phase 4 runs detection at for one segment, in order: the
    reference box at the midpoint (if a frame is on screen then), the
    backward scan, and the forward scan, which ends at its time bound or at
    the first time past the last frame, whichever comes first."""
    fps, step = scanner.fps, scanner.TIMING_SCAN_INTERVAL
    start, end = segment["start_pts"], segment["end_pts"]
    times = [(start + end) / 2]
    t, low = start - step, max(0, start - scanner.TIMING_SCAN_MAX_DURATION)
    while t >= low and int(t * fps) >= 0:
        times.append(t)
        t -= step
    t, high = end + step, end + scanner.TIMING_SCAN_MAX_DURATION
    while t <= high and _on_screen(reference, t, fps) is not None:
        times.append(t)
        t += step
    return [t for t in times if _on_screen(reference, t, fps) is not None]


def _check_phase3(scanner, reference, recorder, monkeypatch):
    fps = scanner.fps
    readings = []
    real_best = LabelScanner._best_single_reading

    def recording_best(self, ocr_results, fallback_box):
        readings.append([(t, text) for t, text, _, _ in ocr_results])
        return real_best(self, ocr_results, fallback_box)

    monkeypatch.setattr(LabelScanner, "_best_single_reading", recording_best)
    result = _Mismatches("phase 3")
    for groups in _phase3_group_runs(scanner, reference):
        ocr = _Phase3OCR(recorder)
        readings.clear()
        scanner._phase3_ocr_and_segment(groups, ocr)
        assert len(readings) == len(groups)
        for group, got in zip(groups, readings):
            box = group["encompassing_box"]
            for t, text in got:
                pts, frame, crop = ocr.calls[int(text[len("call"):])]
                i = _on_screen(reference, t, fps)
                if i is None:
                    result.missing(f"t={t:.6f} is past the last frame but was OCR'd (PTS {pts})")
                    continue
                want = reference[i][1].copy()
                scanner._apply_label_masks(want)
                expected, _ = scanner._crop_cluster_region(want, box, padding=50)
                expected, _ = scanner._resize_max_dimension(expected, scanner.RECOGNIZE_HEIGHT)
                result.check(t, reference, fps, pts, frame, crop, expected)
            want_times = [t for t in _phase3_sample_times(scanner, group)
                          if _on_screen(reference, t, fps) is not None]
            if [t for t, _ in got] != want_times:
                result.missing(f"group {group['first_pts']:.6f}-{group['last_pts']:.6f}: readings at "
                               f"{[t for t, _ in got]}, expected one at each of {want_times}")
    return result


def _check_phase4(scanner, reference, recorder):
    fps = scanner.fps
    result = _Mismatches("phase 4")
    times_checked = []
    for segment in _phase4_segments(scanner, reference):
        detector = _Phase4Detector(recorder)
        scanner._phase4_find_timing([segment], detector)
        want_times = _phase4_sample_times(scanner, segment, reference)
        if len(detector.calls) != len(want_times):
            result.missing(f"segment {segment['start_pts']:.6f}-{segment['end_pts']:.6f}: "
                           f"{len(detector.calls)} detections for {len(want_times)} sample times")
        for t, (pts, frame, image) in zip(want_times, detector.calls):
            want = reference[_on_screen(reference, t, fps)][1].copy()
            scanner._apply_label_masks(want)
            roi, _, _ = scanner._crop_roi_for_detection(want[: scanner.dialogue_cutoff_y, :], segment["box"])
            if roi.shape[0] > scanner.SCAN_HEIGHT:
                roi, _ = scanner._downscale(roi, scanner.SCAN_HEIGHT)
            result.check(t, reference, fps, pts, frame, image, roi)
            times_checked.append(t)
    return result, times_checked


def _phase34_scanner(path):
    scanner = _scanner(path, None)
    scanner.label_mask_crops = MASKS
    return scanner


@pytest.mark.parametrize("clip_id", _matrix(PHASE34_CLIPS, PHASE34_FAST))
def test_phase3_reads_the_frame_on_screen_at_each_sample_time(clips, monkeypatch, clip_id):
    path = _clip(clips, clip_id)
    reference = _decode_reference(path)
    scanner = _phase34_scanner(path)
    recorder = _Recorder()
    recorder.install(monkeypatch, PyAVCapture)

    result = _check_phase3(scanner, reference, recorder, monkeypatch)

    assert result.checked >= 2 * len(reference), "too few phase 3 samples to mean anything"
    result.assert_none()


@pytest.mark.parametrize("clip_id", _matrix(PHASE34_CLIPS, PHASE34_FAST))
def test_phase4_reads_the_frame_on_screen_at_each_sample_time(clips, monkeypatch, clip_id):
    path = _clip(clips, clip_id)
    reference = _decode_reference(path)
    scanner = _phase34_scanner(path)
    recorder = _Recorder()
    recorder.install(monkeypatch, PyAVCapture)

    result, times = _check_phase4(scanner, reference, recorder)

    assert result.checked >= 5 * len(reference), "too few phase 4 samples to mean anything"
    # A backward scan reached the first frame's duration from time 0, where a
    # frame-index seek to position 0 would not move the capture at all.
    assert any(0 <= t < 1.0 / scanner.fps for t in times)
    result.assert_none()


def test_phases_3_and_4_read_the_frame_on_screen_on_the_offset_fixture(offset_video, monkeypatch):
    """The shared 1.5 s-offset fixture (10 frames). Every frame here is at
    position int(PTS * fps) 37 or later of a 10-frame count, so an end-of-stream
    check against the frame count would skip every phase 3 sample; phase 3
    must analyse one per group, and phase 4 must scan from before the first
    frame to past the last."""
    reference = _decode_reference(offset_video)
    scanner = _phase34_scanner(offset_video)
    recorder = _Recorder()
    recorder.install(monkeypatch, PyAVCapture)

    phase3 = _check_phase3(scanner, reference, recorder, monkeypatch)
    phase4, times = _check_phase4(scanner, reference, recorder)

    assert phase3.checked >= len(reference), "phase 3 skipped samples that have a frame on screen"
    assert phase4.checked >= 5 * len(reference)
    assert any(t < reference[0][0] for t in times), "no phase 4 sample before the first frame"
    assert any(reference[0][0] <= t for t in times), "no phase 4 sample inside the clip"
    phase3.assert_none()
    phase4.assert_none()


def _fallback_phase34_run(scanner, reference, recorder, monkeypatch, capture_cls):
    """A reduced phase 3 and phase 4 run for a fallback backend, which restarts
    a decoder for most seeks: groups spanning 14 frames from every 14th frame
    (sampled at +0 and +0.5 s), and three segments."""
    fps = scanner.fps
    box = _region_box(scanner)
    pts = [p for p, _ in reference]
    readings = []
    real_best = LabelScanner._best_single_reading

    def recording_best(self, ocr_results, fallback_box):
        readings.append([(t, text) for t, text, _, _ in ocr_results])
        return real_best(self, ocr_results, fallback_box)

    monkeypatch.setattr(LabelScanner, "_best_single_reading", recording_best)
    monkeypatch.setattr("videocr.label_scanner.Capture", capture_cls)
    phase3 = _Mismatches("phase 3")
    groups = [{"first_pts": pts[i], "last_pts": pts[i + 13], "encompassing_box": box}
              for i in range(0, len(pts) - 13, 14)]
    ocr = _Phase3OCR(recorder)
    scanner._phase3_ocr_and_segment(groups, ocr)
    for group, got in zip(groups, readings):
        for t, text in got:
            got_pts, frame, crop = ocr.calls[int(text[len("call"):])]
            want = reference[_on_screen(reference, t, fps)][1].copy()
            scanner._apply_label_masks(want)
            expected, _ = scanner._crop_cluster_region(want, box, padding=50)
            phase3.check(t, reference, fps, got_pts, frame, crop, expected)

    phase4 = _Mismatches("phase 4")
    for start in (0.62, pts[40], pts[61] + 0.5):
        segment = {"box": box, "text": "label", "confidence": 1.0, "start_pts": start, "end_pts": start + 0.5}
        detector = _Phase4Detector(recorder)
        scanner._phase4_find_timing([segment], detector)
        want_times = _phase4_sample_times(scanner, segment, reference)
        if len(detector.calls) != len(want_times):
            phase4.missing(f"segment at {start}: {len(detector.calls)} detections for {len(want_times)} times")
        for t, (got_pts, frame, image) in zip(want_times, detector.calls):
            want = reference[_on_screen(reference, t, fps)][1].copy()
            scanner._apply_label_masks(want)
            roi, _, _ = scanner._crop_roi_for_detection(want[: scanner.dialogue_cutoff_y, :], box)
            phase4.check(t, reference, fps, got_pts, frame, image, roi)
    return phase3, phase4


@pytest.mark.parametrize("clip_id", _matrix(["zero-start-h264", "offset-h264"], fast={"offset-h264"}))
def test_ffmpeg_fallback_phases_3_and_4_read_the_frame_on_screen(clips, monkeypatch, clip_id):
    """The subprocess backend positions by frame ordinal and reports PTS as
    ordinal / fps + container start; the frame on screen is judged by those
    PTS and that backend's own frames. Label masks are applied to its frames
    in place, so they must be writable."""
    if not pyav_adapter.FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg CLI not available")
    path = _clip(clips, clip_id)
    reference = _every_frame(CpuFFmpegCapture, path)
    scanner = _phase34_scanner(path)
    assert scanner.label_mask_crops
    recorder = _Recorder()
    recorder.install(monkeypatch, FFmpegNVDECCapture)

    phase3, phase4 = _fallback_phase34_run(scanner, reference, recorder, monkeypatch, CpuFFmpegCapture)

    assert phase3.checked >= 8 and phase4.checked >= 30
    phase3.assert_none()
    phase4.assert_none()


# --- seek_to_display_time(): the capture-level contract phases 3 and 4 use ----

def _display_time_probes(reference, fps):
    """Times on every frame's PTS, just inside and just outside the tolerance
    below it, a third of a frame after it, before the first frame, inside the
    last frame's duration and past it."""
    pts = [p for p, _ in reference]
    probes = []
    for p in pts:
        probes += [p, p - ON_SCREEN_TOLERANCE / 2, p - 2 * ON_SCREEN_TOLERANCE, p + 1.0 / (3 * fps)]
    probes += [0.0, -1.0, pts[0] - 1e-3, pts[-1] + 0.5 / fps, pts[-1] + 1.0 / fps, pts[-1] + 10.0]
    return probes


def _assert_display_time_seeks(cap, reference, fps, times, read_on=True):
    wrong = []
    for t in times:
        i = _on_screen(reference, t, fps)
        ok = cap.seek_to_display_time(t)
        if i is None:
            if ok or cap.read() != (False, None):
                wrong.append((t, "past the end", "a frame"))
            continue
        read_ok, frame = cap.read()
        got = cap.get_last_pts() if read_ok else None
        if not (ok and read_ok and got == reference[i][0] and np.array_equal(frame, reference[i][1])):
            wrong.append((t, reference[i][0], got))
            continue
        if read_on and i + 1 < len(reference):
            # Reading on continues with the next frame.
            read_ok, frame = cap.read()
            got = cap.get_last_pts() if read_ok else None
            if not (read_ok and got == reference[i + 1][0] and np.array_equal(frame, reference[i + 1][1])):
                wrong.append((t, "then", reference[i + 1][0], got))
    assert not wrong, (f"{len(wrong)} of {len(times)} display-time seeks read the wrong frame "
                       f"(time, frame on screen PTS, PTS read): {wrong[:10]}")


@pytest.mark.parametrize("clip_id", _matrix(PHASE34_CLIPS, PHASE34_FAST))
def test_seek_to_display_time_reads_the_frame_on_screen(clips, clip_id):
    path = _clip(clips, clip_id)
    reference = _decode_reference(path)
    with PyAVCapture(str(path)) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        probes = _display_time_probes(reference, fps)
        # Forwards, backwards and jumping, so every seek direction is covered.
        _assert_display_time_seeks(cap, reference, fps, probes)
        _assert_display_time_seeks(cap, reference, fps, probes[::-3], read_on=False)
        _assert_display_time_seeks(cap, reference, fps, [p for pair in zip(probes[::5], probes[::-5]) for p in pair],
                                   read_on=False)


def test_seek_to_display_time_on_the_offset_fixture(offset_video):
    reference = _decode_reference(offset_video)
    with PyAVCapture(str(offset_video)) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        probes = _display_time_probes(reference, fps)
        _assert_display_time_seeks(cap, reference, fps, probes + probes[::-1])


@pytest.mark.parametrize("clip_id", ["zero-start-h264", "video-start-0.021s-mkv", "offset-h264"])
def test_seek_to_display_time_repositions_to_the_first_frame(clips, clip_id):
    """set(CAP_PROP_POS_FRAMES, 0) does not move the capture; a display time
    inside or before the first frame's duration must."""
    path = _clip(clips, clip_id)
    reference = _decode_reference(path)
    with PyAVCapture(str(path)) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        for t in (0.0, reference[0][0] + 0.5 / fps, reference[0][0] - 1.0):
            assert cap.seek_to_display_time(reference[60][0])
            for _ in range(5):
                assert cap.read()[0]
            assert cap.seek_to_display_time(t)
            ok, frame = cap.read()
            assert ok and cap.get_last_pts() == reference[0][0] and np.array_equal(frame, reference[0][1]), t


@pytest.mark.parametrize("clip_id", ["zero-start-h264", "video-start-0.021s-mkv", "offset-h264"])
def test_seek_to_display_time_retries_from_further_back_when_a_seek_lands_late(clips, clip_id):
    path = _clip(clips, clip_id)
    reference = _decode_reference(path)
    with PyAVCapture(str(path)) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        late = _LateSeekingContainer(cap.container, cap.stream, late_seconds=1.5)
        cap.container = late
        times = [reference[60][0], reference[75][0] + 0.5 / fps, reference[99][0], reference[62][0] - 1e-7]
        _assert_display_time_seeks(cap, reference, fps, times)
        assert len(late.seeks) > len(times), "the late seeks never needed a retry, so this proves nothing"


@pytest.mark.parametrize("seek_name", ["seek_to_pts", "seek_to_display_time"])
def test_seeks_report_no_frame_when_the_retries_run_out(clips, monkeypatch, caplog, seek_name):
    """When every allowed seek lands after the target, neither seek may hand
    out the later frame as if it were the one asked for: both return False,
    log a warning, and the next read() fails. That is not the end of the
    stream (seek_past_end stays False), even straight after a seek that was."""
    path = _clip(clips, "zero-start-h264")
    reference = _decode_reference(path)
    monkeypatch.setattr(PyAVCapture, "_SEEK_MAX_RETRIES", 1)
    caplog.set_level(logging.WARNING, logger="videocr.pyav_adapter")
    target = reference[60][0]
    with PyAVCapture(str(path)) as cap:
        seek = getattr(cap, seek_name)
        assert seek(reference[-1][0] + 1.0) is False and cap.seek_past_end is True
        cap.container = _LateSeekingContainer(cap.container, cap.stream, late_seconds=1.5)
        assert seek(target) is False
        assert cap.seek_past_end is False
        assert cap.read() == (False, None)
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        # Phase 1.5's fetch treats it as an unreadable frame.
        assert LabelScanner._read_frame_at_pts(cap, target) is None
        # And the capture is still usable.
        cap.container = cap.container._container
        assert seek(target) is True and cap.seek_past_end is False
        ok, frame = cap.read()
        assert ok and cap.get_last_pts() == target and np.array_equal(frame, reference[60][1])


@pytest.mark.parametrize("seek_name", ["seek_to_pts", "seek_to_display_time"])
def test_seeks_use_their_last_allowed_retry(clips, monkeypatch, seek_name):
    """With one retry allowed, a seek that lands late once and then lands
    before its target still finds the frame."""
    path = _clip(clips, "zero-start-h264")
    reference = _decode_reference(path)
    monkeypatch.setattr(PyAVCapture, "_SEEK_MAX_RETRIES", 1)
    target = reference[60][0]
    with PyAVCapture(str(path)) as cap:
        late = _LateSeekingContainer(cap.container, cap.stream, late_seconds=0.5)
        cap.container = late
        assert getattr(cap, seek_name)(target) is True
        ok, frame = cap.read()
        assert ok and cap.get_last_pts() == target and np.array_equal(frame, reference[60][1])
        assert len(late.seeks) == 2, "expected exactly one retry"


@pytest.mark.parametrize("seek_name", ["seek_to_pts", "seek_to_display_time"])
def test_seeks_past_the_end_report_no_frame_without_retrying(clips, caplog, seek_name):
    """A seek that decodes frames up to the end of the stream without finding
    one has its answer: no frame, from that one seek, and nothing logged."""
    path = _clip(clips, "offset-h264")
    reference = _decode_reference(path)
    caplog.set_level(logging.WARNING, logger="videocr.pyav_adapter")
    with PyAVCapture(str(path)) as cap:
        counting = _LateSeekingContainer(cap.container, cap.stream, late_seconds=0.0)
        cap.container = counting
        assert getattr(cap, seek_name)(reference[-1][0] + 1.0) is False
        assert cap.seek_past_end is True
        assert len(counting.seeks) == 1
        assert cap.read() == (False, None)
        assert getattr(cap, seek_name)(reference[10][0]) is True and cap.seek_past_end is False
    assert not caplog.records


@pytest.mark.parametrize("clip_id", _matrix(["zero-start-h264", "offset-h264"], fast={"offset-h264"}))
def test_ffmpeg_fallback_seek_to_display_time_reads_the_frame_on_screen(clips, clip_id):
    if not pyav_adapter.FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg CLI not available")
    path = _clip(clips, clip_id)
    reference = _every_frame(CpuFFmpegCapture, path)
    with CpuFFmpegCapture(str(path)) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        probes = _display_time_probes(reference, fps)
        # Every seek restarts ffmpeg, so visit a subset: every seventh probe
        # both ways, the ones around time 0 and the end, and a run of
        # consecutive frames (which must not restart it).
        subset = probes[::7] + probes[::-7] + probes[-6:] + probes[:12] + [0.0]
        _assert_display_time_seeks(cap, reference, fps, subset, read_on=False)


_OPENCV_PROBE = r"""
import json, shutil, sys
sys.modules["av"] = None
_which = shutil.which
shutil.which = lambda name, *a, **k: None if name in ("ffmpeg", "ffprobe") else _which(name, *a, **k)
import numpy as np
from videocr import pyav_adapter
assert pyav_adapter.Capture.__name__ == "OpenCVCapture", pyav_adapter.Capture
path, times = sys.argv[1], json.loads(sys.argv[2])
with pyav_adapter.Capture(path) as cap:
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append((cap.get_last_pts(), frame.copy()))
    out = []
    for t in times:
        found = cap.seek_to_display_time(t)
        past_end = cap.seek_past_end
        ok, frame = cap.read()
        pts = cap.get_last_pts() if ok else None
        index = next((i for i, (p, f) in enumerate(frames) if p == pts and np.array_equal(f, frame)), None) if ok else None
        out.append([t, found, past_end, ok, index])
    assert out[-1][2], "the probe must end past the last frame"
    cap.seek_to_pts(frames[3][0])
    after_pts_seek = cap.seek_past_end
print(json.dumps({"pts": [p for p, _ in frames], "reads": out, "after_pts_seek": after_pts_seek}))
"""


def test_opencv_fallback_seek_to_display_time_reads_the_frame_on_screen(clips):
    """OpenCVCapture only exists when neither PyAV nor the ffmpeg CLI does, so
    it is exercised in a subprocess that hides both. It reports PTS as
    (ordinal + 1) / fps; the frame on screen is judged by those. Past the last
    frame it returns False and the next read fails, as the other backends do."""
    path = _clip(clips, "zero-start-h264")
    fps = 25.0
    times = [k / fps for k in range(0, 101, 3)] + [k / fps - 2e-6 for k in range(1, 101, 5)]
    times += [k / fps + 0.5 / fps for k in range(0, 100, 4)] + [0.0, -1.0, 0.5 / fps, 4.02, 4.04 - 2e-6, 4.04, 4.1, 5.0]
    probe = subprocess.run([sys.executable, "-c", _OPENCV_PROBE, str(path), json.dumps(times)],
                           capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert probe.returncode == 0, probe.stderr[-2000:]
    result = json.loads(probe.stdout.strip().splitlines()[-1])
    reference = [(p, None) for p in result["pts"]]
    assert len(reference) == FRAMES
    wrong = []
    for t, found, past_end, read_ok, index in result["reads"]:
        i = _on_screen(reference, t, fps)
        if i is None:
            if found or not past_end or read_ok:
                wrong.append((t, "past the end", found, past_end, read_ok))
        elif not (found and not past_end and read_ok and index == i):
            wrong.append((t, reference[i][0], found, past_end, read_ok, index))
    assert not wrong, f"(time, frame on screen PTS, seek result, seek_past_end, read ok, frame index read): {wrong}"
    assert result["after_pts_seek"] is False, "seek_to_pts() left seek_past_end set"
    assert any(_on_screen(reference, t, fps) is None for t, *_ in result["reads"])


# --- phase 1.5 cancellation, and detection input that retained crops share ----

class _CancellingOCR:
    """Records each crop it is given and sets `cancel` on the second."""

    def __init__(self):
        self.cancel = threading.Event()
        self.calls = 0

    def predict(self, crop):
        self.calls += 1
        if self.calls == 2:
            self.cancel.set()
        return []


@pytest.mark.parametrize("retain", [False, True], ids=["fetched", "retained"])
def test_phase15_stops_once_cancelled(clips, monkeypatch, retain):
    """A cancel during phase 1.5 stops it before the next text frame: no more
    OCR, and no more fetching."""
    from videocr.label_scanner import _RetainedCrops

    scanner = _masked_scanner(_clip(clips, "offset-h264"), None)
    store = _RetainedCrops(budget_bytes=1 << 30) if retain else None
    text_frames = scanner._phase1_find_text_frames(_WholeRegionDetector(), None, None, retain=store)
    assert len(text_frames) >= 6
    seeks = _Counting(monkeypatch, "seek_to_pts")
    ocr = _CancellingOCR()

    augmented = scanner._batch_ocr_text_frames(text_frames, ocr, retained=store, cancel_event=ocr.cancel)

    assert ocr.calls == 2, "phase 1.5 went on reading text frames after the cancel"
    assert len(augmented) == 2
    assert seeks.calls == (0 if retain else 2)


def test_scan_returns_nothing_after_a_cancel_in_phase15(clips, monkeypatch):
    scanner = _masked_scanner(_clip(clips, "offset-h264"), None)
    phase2_calls = []
    monkeypatch.setattr(LabelScanner, "_phase2_group_by_position",
                        lambda self, text_frames, progress=None: phase2_calls.append(text_frames) or [])
    ocr = _CancellingOCR()

    assert scanner.scan(_WholeRegionDetector(), ocr, "", "", 1.5, cancel_event=ocr.cancel) == []
    assert ocr.calls == 2
    assert phase2_calls == [], "scan went on to phase 2 after a cancel in phase 1.5"


class _WritingDetector(_WholeRegionDetector):
    """A detection engine that (wrongly) writes into the image it is given."""

    def predict(self, frame):
        frame[:] = 0
        return super().predict(frame)


def test_detection_cannot_write_into_the_frame_retained_crops_are_cut_from(clips):
    """At or below SCAN_HEIGHT phase 1 hands detection the frame itself, and
    then cuts the crops phase 1.5 will OCR from that same memory. An engine
    writing to its input must fail loudly rather than blank those crops."""
    from videocr.label_scanner import _RetainedCrops

    scanner = _masked_scanner(_clip(clips, "zero-start-h264"), None)
    with pytest.raises(ValueError, match="read-only"):
        scanner._phase1_find_text_frames(_WritingDetector(), None, None, retain=_RetainedCrops(1 << 30))


def test_detection_sees_the_shared_frame_read_only_and_a_downscaled_copy_as_it_is(clips, monkeypatch):
    """At or below SCAN_HEIGHT detection is given a read-only view and phase 1
    still retains every crop. Above it detection gets its own downscaled copy,
    which shares nothing with the crops: an engine writing to that copy does
    not raise and does not change them."""
    from videocr.label_scanner import _RetainedCrops

    seen = []

    class _RecordingDetector(_WholeRegionDetector):
        def predict(self, frame):
            seen.append(frame.flags.writeable)
            return super().predict(frame)

    scanner = _masked_scanner(_clip(clips, "zero-start-h264"), None)
    recorder = _Recorder()
    recorder.install(monkeypatch, PyAVCapture)
    retained = _RetainedCrops(1 << 30)
    text_frames = scanner._phase1_find_text_frames(_RecordingDetector(), None, None, retain=retained)
    assert seen and not any(seen), "detection input shared with the retained crops was writable"
    crops = [crop for idx, pts, boxes in text_frames for crop in retained.take(idx, pts, boxes)]
    assert len(crops) == len(text_frames)

    scanner = _masked_scanner(_clip(clips, "zero-start-h264-960p"), None)
    recorder.reads.clear()
    retained = _RetainedCrops(1 << 30)
    text_frames = scanner._phase1_find_text_frames(_WritingDetector(), None, None, retain=retained)
    phase1_frames = dict(recorder.reads)
    assert len(text_frames) >= 3
    got = [crop for idx, pts, boxes in text_frames for crop in retained.take(idx, pts, boxes)]
    expected = []
    for _, pts, boxes in text_frames:
        frame = phase1_frames[pts].copy()
        scanner._apply_label_masks(frame)
        expected += [scanner._crop_box_region(frame[: scanner.dialogue_cutoff_y, :], box)[0] for box in boxes]
    _assert_same_crops(got, expected)


@pytest.mark.parametrize("clip_id", ["offset-h264", "video-start-0.021s-mkv"])
def test_phase4_forward_scan_runs_to_the_last_frame_and_stops_past_it(clips, monkeypatch, clip_id):
    """A label still on screen at the last frame ends at the last scan time
    that has a frame, even on a file whose first frame is after time 0 (where
    int(t * fps) reaches the frame count up to the start offset early). The
    scan stops at the first time past the last frame instead of seeking on
    to its time bound."""
    path = _clip(clips, clip_id)
    reference = _decode_reference(path)
    scanner = _phase34_scanner(path)
    fps, box = scanner.fps, _region_box(scanner)
    discovery = reference[-1][0] - 1.6
    step = scanner.TIMING_SCAN_INTERVAL
    with_frame, t = [], discovery + step
    while _on_screen(reference, t, fps) is not None:
        with_frame.append(t)
        t += step
    assert len(with_frame) >= 7

    seeks = _Counting(monkeypatch, "seek_to_display_time")
    detector = _Phase4Detector(_Recorder())
    detector.recorder.install(monkeypatch, PyAVCapture)
    with PyAVCapture(str(path)) as cap:
        end = scanner._scan_for_end(cap, detector, box, discovery, ref_box=box)

    assert end == with_frame[-1]
    assert len(detector.calls) == len(with_frame)
    assert seeks.calls == len(with_frame) + 1, "the forward scan went on seeking past the last frame"


@pytest.mark.parametrize("target", ["the position before the seek", "position 0"])
def test_ffmpeg_fallback_set_restarts_the_pipe_a_seek_past_the_end_stopped(clips, target):
    """A display-time seek past the last frame stops the pipe. A later
    set(CAP_PROP_POS_FRAMES, n) must start it again even when n is the
    position the pipe had before, or 0 -- positions set() otherwise treats as
    "already there"."""
    if not pyav_adapter.FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg CLI not available")
    path = _clip(clips, "zero-start-h264")
    reference = _every_frame(CpuFFmpegCapture, path)
    with CpuFFmpegCapture(str(path)) as cap:
        assert cap.seek_to_display_time(reference[40][0])
        assert cap.read()[0]
        position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        assert position == 41 and cap.seek_past_end is False
        assert not cap.seek_to_display_time(reference[-1][0] + 1.0)
        assert cap.seek_past_end is True
        assert cap.read() == (False, None)
        assert cap.seek_to_pts(reference[5][0]) and cap.seek_past_end is False
        assert not cap.seek_to_display_time(reference[-1][0] + 1.0) and cap.seek_past_end is True

        n = position if target == "the position before the seek" else 0
        assert cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = cap.read()
        assert ok and cap.get_last_pts() == reference[n][0] and np.array_equal(frame, reference[n][1])


def test_ffmpeg_fallback_frames_are_writable_so_label_masks_apply(clips):
    """Every label phase blacks out mask regions in the frame it read."""
    if not pyav_adapter.FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg CLI not available")
    path = _clip(clips, "zero-start-h264")
    scanner = _phase34_scanner(path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("videocr.label_scanner.Capture", CpuFFmpegCapture)
        text_frames = scanner._phase1_find_text_frames(_WholeRegionDetector(), None, "0:01")
    assert len(text_frames) >= 2
    with CpuFFmpegCapture(str(path)) as cap:
        ok, frame = cap.read()
        assert ok and frame.flags.writeable
        scanner._apply_label_masks(frame)
        x, y, w, h = MASKS[0]
        assert not frame[y:y + h, x:x + w].any()
        ok, following = cap.read()
        assert ok and following[y:y + h, x:x + w].any(), "masking one frame changed the next"


class _NothingDecodingContainer(_LateSeekingContainer):
    """Seeks normally, but decodes nothing after them -- what a seek that
    fails to find its keyframe looks like to the capture."""

    def decode(self, *args, **kwargs):
        return iter(())


class _ExhaustingSeeks:
    """Makes chosen seek_to_display_time() calls run out of retries for real:
    for those calls no retry is allowed and the container decodes nothing
    after the seek, so the seek takes its retries-exhausted path (and logs)
    mid-stream, wherever the time is."""

    def __init__(self, monkeypatch, fail):
        self.fail = fail  # t -> whether this call should exhaust
        self.calls = []   # (t, result)
        real = PyAVCapture.seek_to_display_time
        exhausting = self

        def seek(cap, t):
            if not exhausting.fail(t):
                result = real(cap, t)
            else:
                container = cap.container
                cap.container = _NothingDecodingContainer(container, cap.stream, late_seconds=0.0)
                cap._SEEK_MAX_RETRIES = 0
                try:
                    result = real(cap, t)
                finally:
                    cap.container = container
                    del cap._SEEK_MAX_RETRIES
            exhausting.calls.append((t, result))
            return result

        monkeypatch.setattr(PyAVCapture, "seek_to_display_time", seek)


def _scan_for_end_with(scanner, path, discovery):
    """(label end, detections) from a forward scan whose label is on screen
    throughout."""
    detections = []

    class _Counting(_WholeRegionDetector):
        def predict(self, frame):
            detections.append(1)
            return super().predict(frame)

    box = _region_box(scanner)
    with PyAVCapture(str(path)) as cap:
        end = scanner._scan_for_end(cap, _Counting(), box, discovery, ref_box=box)
    return end, len(detections)


@pytest.mark.parametrize("failures", [1, 2], ids=["one-failed-seek", "two-consecutive-failed-seeks"])
def test_phase4_forward_scan_skips_a_seek_that_runs_out_of_retries(clips, monkeypatch, caplog, failures):
    """A seek that runs out of retries mid-label is not the end of the stream:
    the forward scan skips that time (without counting it as an absence, so
    two in a row do not end the label either) and the label ends where it
    would have without the failures."""
    path = _clip(clips, "zero-start-h264")
    scanner = _phase34_scanner(path)
    discovery = 0.5
    clean_end, clean_detections = _scan_for_end_with(scanner, path, discovery)
    assert clean_end > discovery + 10 * scanner.TIMING_SCAN_INTERVAL

    failing = [discovery + (3 + i) * scanner.TIMING_SCAN_INTERVAL for i in range(failures)]
    caplog.set_level(logging.WARNING, logger="videocr.pyav_adapter")
    seeks = _ExhaustingSeeks(monkeypatch, lambda t: any(abs(t - f) < 1e-9 for f in failing))
    end, detections = _scan_for_end_with(scanner, path, discovery)

    assert [result for t, result in seeks.calls if any(abs(t - f) < 1e-9 for f in failing)] == [False] * failures
    assert sum(r.levelno == logging.WARNING for r in caplog.records) == failures, "the injected seeks did not run out of retries"
    assert end == clean_end, f"the label ended at {end} after {failures} failed seek(s), not at {clean_end}"
    assert detections == clean_detections - failures


def test_phase4_forward_scan_ends_at_its_time_bound_when_every_seek_runs_out_of_retries(clips, monkeypatch, caplog):
    """Seeks that keep failing never end the label early, and the scan still
    stops: at its time bound, having tried each time once."""
    path = _clip(clips, "zero-start-h264")
    scanner = _phase34_scanner(path)
    discovery, step = 0.5, scanner.TIMING_SCAN_INTERVAL
    times, t = [], discovery + step
    while t <= discovery + scanner.TIMING_SCAN_MAX_DURATION:
        times.append(t)
        t += step
    caplog.set_level(logging.WARNING, logger="videocr.pyav_adapter")
    seeks = _ExhaustingSeeks(monkeypatch, lambda t: True)

    end, detections = _scan_for_end_with(scanner, path, discovery)

    assert all(result is False for _, result in seeks.calls)
    assert sum(r.levelno == logging.WARNING for r in caplog.records) == len(seeks.calls)
    assert [t for t, _ in seeks.calls] == times, "the scan did not try each time up to its bound exactly once"
    assert end == discovery and detections == 0
