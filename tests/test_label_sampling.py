"""Label phase 1 must sample exactly the frames a full decode would.

Phase 1 of the label scanner looks at one frame every
`SAMPLE_INTERVAL_SECONDS`. It used to decode *and convert to BGR* every
frame and throw away all but every Nth; it now only converts the frames it
samples. That is a speedup only if the frames it keeps -- their pixels,
their frame index and their PTS, which label timing is derived from -- are
exactly the ones the every-frame loop kept. These tests pin that.

References, from most to least independent of the code under test:

* a raw PyAV decode of every frame from the start of the file, keeping
  every Nth (no capture class involved at all);
* the pre-change phase 1 loop -- seek once, `read()` every frame, keep every
  Nth by counting -- re-implemented here on `PyAVCapture`, which is what the
  label goldens were captured with, and the only way to cover the seek.
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

# id -> (pix_fmt, codec, rate, extra ffmpeg args, tone mapped)
CLIPS = {
    "8bit-h264": ("yuv420p", "libx264", "25", [], False),
    # The 4K reference media is 10-bit HEVC: the BGR conversion of a >8-bit
    # source is the one that differs between PyAV's reformatter and a graph
    # format node, so it gets its own clip.
    "10bit-h265": ("yuv420p10le", "libx265", "25", [], False),
    # int(23.976 * 0.5) == 11, so the interval is not always 12.
    "8bit-h264-23.976fps": ("yuv420p", "libx264", "24000/1001", [], False),
    # Container start_time 1.5 s: every PTS is offset.
    "8bit-h264-offset": ("yuv420p", "libx264", "25", ["-output_ts_offset", "1.5"], False),
    # PQ-tagged: PyAVCapture pushes frames through a tone-map filter graph.
    "10bit-h264-pq": (
        "yuv420p10le", "libx264", "25",
        ["-vf", "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc",
         "-color_trc", "smpte2084", "-color_primaries", "bt2020", "-colorspace", "bt2020nc"],
        True,
    ),
}
SDR_CLIPS = [cid for cid, spec in CLIPS.items() if not spec[4]]

# (time_start, time_end). The second starts mid-GOP at a frame index that is
# not a multiple of the interval (0.72 s -> frame 18 at 25 fps), so the seek
# and the counting origin are both exercised, and ends before the clip does.
RANGES = [
    pytest.param(None, None, id="whole-clip"),
    pytest.param("0:00.72", "0:03.5", id="mid-gop-range"),
]


def _encode(path, pix_fmt, codec, rate, extra):
    gop = (["-x265-params", "keyint=10:min-keyint=10:log-level=error"]
           if codec == "libx265" else ["-g", "10"])
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate={rate}",
         "-frames:v", str(FRAMES), "-pix_fmt", pix_fmt, "-c:v", codec,
         *gop, *extra, str(path)],
        check=True, capture_output=True,
    )


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    root = tmp_path_factory.mktemp("label_sampling")
    out = {}
    for cid, (pix_fmt, codec, rate, extra, tone_mapped) in CLIPS.items():
        path = root / f"{cid}.mp4"
        _encode(path, pix_fmt, codec, rate, extra)
        with PyAVCapture(str(path)) as cap:
            # Guard the fixture itself: an untagged "HDR" clip would quietly
            # turn the tone-map case into a second SDR case.
            assert cap._needs_tonemap == tone_mapped, cid
            # ...and every frame must differ from the next inside the region
            # phase 1 hands to detection (the top 80%), so a sample taken one
            # frame early or late can never compare equal by accident.
            previous = None
            for _ in range(FRAMES):
                ok, frame = cap.read()
                assert ok, cid
                top = frame[: int(frame.shape[0] * 0.8)].copy()
                assert previous is None or not np.array_equal(previous, top), cid
                previous = top
        out[cid] = path
    return out


def _scanner(path):
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        width, height, num_frames = stream.width, stream.height, stream.frames
    finally:
        container.close()
    assert num_frames == FRAMES
    return LabelScanner(str(path), fps, width, height, num_frames, None, None, None, None)


def _index_range(scanner, time_start, time_end):
    """The same frame-index arithmetic phase 1 uses."""
    from videocr import utils
    start_idx = utils.get_frame_index(time_start, scanner.fps) if time_start else 0
    end_idx = utils.get_frame_index(time_end, scanner.fps) if time_end else scanner.num_frames
    interval = max(1, int(scanner.fps * scanner.SAMPLE_INTERVAL_SECONDS))
    return start_idx, end_idx, interval


class _RecordingDetector:
    """Stands in for the detection engine: records every frame phase 1
    hands it and reports one box, so every sample lands in text_frames."""

    def __init__(self):
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame.copy())
        return [{"dt_polys": [[[0, 0], [8, 0], [8, 8], [0, 8]]]}]


def _run_phase1(path, time_start, time_end):
    scanner = _scanner(path)
    det = _RecordingDetector()
    text_frames = scanner._phase1_find_text_frames(det, time_start, time_end)
    return scanner, det, text_frames


def _every_frame_loop(path, start_idx, end_idx, interval, capture=PyAVCapture):
    """The pre-change phase 1 decode loop: read (decode + convert) every
    frame after the seek, keep every Nth by counting."""
    kept = []
    with capture(str(path)) as cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for offset in range(end_idx - start_idx):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if offset % interval == 0:
                kept.append((start_idx + offset, cap.get_last_pts(), frame.copy()))
    return kept


def _raw_decode_every_nth(path, end_idx, interval):
    """Every frame decoded and converted by PyAV itself, every Nth kept."""
    kept = []
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        for ordinal, frame in enumerate(container.decode(video=0)):
            if ordinal >= end_idx:
                break
            if ordinal % interval == 0:
                kept.append((ordinal, float(frame.pts * stream.time_base),
                             frame.to_ndarray(format="bgr24")))
    finally:
        container.close()
    return kept


def _assert_phase1_matches(scanner, det, text_frames, reference):
    # Non-vacuous: several samples (the clips fixture separately guarantees
    # that adjacent frames differ in the region compared below).
    assert len(reference) >= 5
    cutoff = scanner.dialogue_cutoff_y

    assert [(idx, pts) for idx, pts, _ in text_frames] == \
        [(idx, pts) for idx, pts, _ in reference], "frame index / PTS differ"
    assert len(det.frames) == len(reference)
    for (idx, _, ref_frame), got in zip(reference, det.frames):
        expected, _ = scanner._downscale(ref_frame[:cutoff, :], scanner.SCAN_HEIGHT)
        assert got.dtype == expected.dtype and got.shape == expected.shape, idx
        assert np.array_equal(got, expected), f"pixels differ at frame {idx}"


@pytest.mark.parametrize("time_start,time_end", RANGES)
@pytest.mark.parametrize("clip_id", list(CLIPS))
def test_phase1_samples_match_the_every_frame_loop(clips, clip_id, time_start, time_end):
    scanner, det, text_frames = _run_phase1(clips[clip_id], time_start, time_end)
    start_idx, end_idx, interval = _index_range(scanner, time_start, time_end)
    reference = _every_frame_loop(clips[clip_id], start_idx, end_idx, interval)
    _assert_phase1_matches(scanner, det, text_frames, reference)


@pytest.mark.parametrize("clip_id", SDR_CLIPS)
def test_phase1_samples_match_a_raw_pyav_decode(clips, clip_id):
    scanner, det, text_frames = _run_phase1(clips[clip_id], None, None)
    _, end_idx, interval = _index_range(scanner, None, None)
    reference = _raw_decode_every_nth(clips[clip_id], end_idx, interval)
    _assert_phase1_matches(scanner, det, text_frames, reference)


def test_phase1_converts_only_the_frames_it_samples(clips, monkeypatch):
    """The speedup itself: frames phase 1 does not sample are never
    converted to BGR. read() is the only PyAVCapture method that converts."""
    reads = []
    real_read = PyAVCapture.read

    def counting_read(self):
        reads.append(1)
        return real_read(self)

    monkeypatch.setattr(PyAVCapture, "read", counting_read)
    scanner, _, text_frames = _run_phase1(clips["8bit-h264"], "0:00.72", "0:03.5")
    start_idx, end_idx, interval = _index_range(scanner, "0:00.72", "0:03.5")

    samples = len(range(start_idx, end_idx, interval))
    assert len(text_frames) == samples
    assert len(reads) == samples, (
        f"{len(reads)} frames converted for {samples} samples "
        f"({end_idx - start_idx} frames in range)"
    )


# --- grab(): the capture-level contract phase 1 relies on --------------------

def _make_graph_clip(path, pix_fmt, size):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25",
         "-frames:v", "40", "-pix_fmt", pix_fmt, "-c:v", "libx264", "-g", "10", str(path)],
        check=True, capture_output=True,
    )


# Every graph configuration PyAVCapture can build: none, tone map, downscale,
# downscale + in-graph crop. Skipping graph pushes for grabbed frames is only
# safe if none of them carries state from one frame to the next.
GRAB_CONFIGS = [
    pytest.param("8bit", None, None, id="no-graph-8bit"),
    pytest.param("10bit", None, None, id="no-graph-10bit"),
    pytest.param("pq", None, None, id="tonemap-graph"),
    pytest.param("8bit-640", 240, None, id="downscale-graph"),
    pytest.param("8bit-640", 240, (64, 60, 192, 64), id="downscale-crop-graph"),
]


# Which offset after the seek is read() rather than grab()ed. With 1, the
# very first call after the seek is a grab(), which must consume the frame
# set() already decoded and parked, exactly as read() would.
FIRST_READ = [pytest.param(0, id="read-first"), pytest.param(1, id="grab-first")]


@pytest.mark.parametrize("first_read", FIRST_READ)
@pytest.mark.parametrize("kind,target_height,crop", GRAB_CONFIGS)
def test_grab_then_read_matches_reading_every_frame(clips, tmp_path, kind, target_height, crop, first_read):
    if kind == "pq":
        path = clips["10bit-h264-pq"]
    else:
        path = tmp_path / f"{kind}.mp4"
        pix_fmt = "yuv420p10le" if kind == "10bit" else "yuv420p"
        size = "640x480" if kind.endswith("640") else "320x240"
        _make_graph_clip(path, pix_fmt, size)

    def open_cap():
        return PyAVCapture(str(path), decode_target_height=target_height, crop_rect=crop)

    interval, seek_to, count = 3, 7, 25

    with open_cap() as cap:
        if crop is not None:
            assert cap._crop_graph_active, "fixture did not engage the in-graph crop"
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
        every = []
        for _ in range(count):
            ok, frame = cap.read()
            assert ok
            every.append((cap.get_last_pts(), cap.get(cv2.CAP_PROP_POS_FRAMES), frame.copy()))

    with open_cap() as cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
        for offset in range(count):
            if (offset - first_read) % interval:
                assert cap.grab()
                # grab() advances the same position/PTS state read() does.
                assert cap.get_last_pts() == every[offset][0]
                assert cap.get(cv2.CAP_PROP_POS_FRAMES) == every[offset][1]
                continue
            ok, frame = cap.read()
            assert ok
            pts, pos, expected = every[offset]
            assert cap.get_last_pts() == pts
            assert cap.get(cv2.CAP_PROP_POS_FRAMES) == pos
            assert np.array_equal(frame, expected), f"pixels differ at offset {offset}"

        # And at the end of the stream grab() reports failure like read().
        while cap.grab():
            pass
        assert cap.read() == (False, None)


def test_ffmpeg_fallback_phase1_matches_its_every_frame_loop(clips, monkeypatch):
    """The non-reference subprocess backend has no cheap skip, but phase 1
    must still sample the same frames through it."""

    class CpuFFmpegCapture(FFmpegNVDECCapture):
        def __init__(self, video_path, **kwargs):
            super().__init__(video_path, use_gpu=False, **kwargs)

    if not pyav_adapter.FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg CLI not available")
    monkeypatch.setattr("videocr.label_scanner.Capture", CpuFFmpegCapture)

    path = clips["8bit-h264"]
    scanner, det, text_frames = _run_phase1(path, "0:00.72", "0:03.5")
    start_idx, end_idx, interval = _index_range(scanner, "0:00.72", "0:03.5")
    reference = _every_frame_loop(path, start_idx, end_idx, interval, capture=CpuFFmpegCapture)
    _assert_phase1_matches(scanner, det, text_frames, reference)
