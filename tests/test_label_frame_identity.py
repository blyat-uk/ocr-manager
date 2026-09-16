"""Label phase 1.5 must OCR exactly the frames phase 1 detected text in.

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
import subprocess

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
}
SIZES = {"zero-start-h264-960p": "1280x960"}

# Phase 1's own sampling (every 0.5 s), and every frame.
SAMPLING = [pytest.param(None, id="every-0.5s"), pytest.param("every-frame", id="every-frame")]

# The second range starts before the offset clips' first frame (1.5 s); the
# third starts inside them, mid-GOP.
RANGES = [
    pytest.param(None, None, id="whole-clip"),
    pytest.param("0:01", "0:03.5", id="from-1s"),
    pytest.param("0:02.3", "0:03.9", id="from-2.3s"),
]


def _encode(path, pix_fmt, codec, rate, extra, size="320x240"):
    gop = (["-x265-params", "keyint=10:min-keyint=10:log-level=error"]
           if codec == "libx265" else ["-g", "10"])
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}",
         "-frames:v", str(FRAMES), "-pix_fmt", pix_fmt, "-c:v", codec,
         *gop, *extra, str(path)],
        check=True, capture_output=True,
    )


def _start_time(path):
    container = av.open(str(path))
    try:
        return (container.start_time or 0) / 1_000_000
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
    root = tmp_path_factory.mktemp("label_frame_identity")
    out = {}
    for cid, (ext, pix_fmt, codec, rate, extra) in CLIPS.items():
        path = root / f"{cid}.{ext}"
        _encode(path, pix_fmt, codec, rate, extra, SIZES.get(cid, "320x240"))
        expected_start = 1.5 if extra else 0.0
        assert _start_time(path) == pytest.approx(expected_start, abs=1e-3), cid
        assert _assert_adjacent_frames_differ(path) == FRAMES, cid
        out[cid] = path
    return out


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


@pytest.mark.parametrize("time_start,time_end", RANGES)
@pytest.mark.parametrize("sampling", SAMPLING)
@pytest.mark.parametrize("clip_id", list(CLIPS))
def test_phase15_reads_the_frames_phase1_sampled(clips, monkeypatch, clip_id, sampling, time_start, time_end):
    scanner = _scanner(clips[clip_id], sampling)
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


@pytest.mark.parametrize("sampling", SAMPLING)
@pytest.mark.parametrize("clip_id", ["zero-start-h264", "offset-h264"])
def test_ffmpeg_fallback_phase15_reads_the_frames_phase1_sampled(clips, monkeypatch, clip_id, sampling):
    """The subprocess backend positions by frame ordinal and estimates PTS
    as ordinal / fps + start time -- a different frame-index meaning from
    PyAVCapture's. Phase 1.5 must still land on phase 1's frames through it."""
    if not pyav_adapter.FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg CLI not available")
    monkeypatch.setattr("videocr.label_scanner.Capture", CpuFFmpegCapture)
    scanner = _scanner(clips[clip_id], sampling)
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

    scanner = _scanner(clips["zero-start-h264"], None)
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


@pytest.mark.parametrize("clip_id", list(CLIPS))
def test_seek_to_pts_lands_on_exactly_that_frame(clips, clip_id):
    frames = _every_frame(PyAVCapture, clips[clip_id])
    with PyAVCapture(str(clips[clip_id])) as cap:
        _assert_seek_lands(cap, frames, _visit_order(len(frames)))


@pytest.mark.parametrize("clip_id", ["zero-start-h264", "offset-h264"])
def test_ffmpeg_fallback_seek_to_pts_lands_on_exactly_that_frame(clips, clip_id):
    if not pyav_adapter.FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg CLI not available")
    frames = _every_frame(CpuFFmpegCapture, clips[clip_id])
    with CpuFFmpegCapture(str(clips[clip_id])) as cap:
        # Every seek restarts ffmpeg, so visit a subset, including frame 0
        # and consecutive frames (which must not restart it).
        _assert_seek_lands(cap, frames, _visit_order(len(frames), stride=7) + [40, 41, 42, 0])


@pytest.mark.parametrize("clip_id", ["zero-start-h264", "offset-h264", "offset-h265-10bit"])
def test_seek_to_pts_between_and_outside_frames(clips, clip_id):
    frames = _every_frame(PyAVCapture, clips[clip_id])
    pts = [p for p, _ in frames]
    with PyAVCapture(str(clips[clip_id])) as cap:
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
    frames = _every_frame(PyAVCapture, clips[clip_id])
    with PyAVCapture(str(clips[clip_id])) as cap:
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


@pytest.mark.parametrize("sampling", SAMPLING)
@pytest.mark.parametrize("clip_id", list(CLIPS))
def test_retained_crops_are_what_phase15_would_have_fetched(clips, monkeypatch, clip_id, sampling):
    from videocr.label_scanner import _RetainedCrops

    scanner = _masked_scanner(clips[clip_id], sampling)
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

    scanner = _masked_scanner(clips[clip_id], "every-frame")
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
    scanner = _masked_scanner(clips["offset-h264"], None)
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
