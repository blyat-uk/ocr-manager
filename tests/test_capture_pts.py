import json
import subprocess

import cv2
import pytest

from videocr.pyav_adapter import Capture, FFmpegNVDECCapture


def ffprobe_pts(path, count):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-read_intervals", f"%+#{count}",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    frames = json.loads(out.stdout)["frames"]
    return [float(f["pts_time"]) for f in frames]


def test_pyav_pts_matches_ffprobe(synthetic_video):
    expected = ffprobe_pts(synthetic_video, 10)
    got = []
    with Capture(str(synthetic_video)) as cap:
        for _ in range(10):
            ok, _frame = cap.read()
            assert ok
            got.append(cap.get_last_pts())
    assert got == pytest.approx(expected, abs=1e-6)


def test_fallback_first_frame_pts_is_zero_based(synthetic_video):
    expected = ffprobe_pts(synthetic_video, 5)
    cap = FFmpegNVDECCapture(str(synthetic_video), use_gpu=False)
    with cap as c:
        got = []
        for _ in range(5):
            ok, _frame = c.read()
            assert ok
            got.append(c.get_last_pts())
    assert got == pytest.approx(expected, abs=1e-3)


def test_container_start_time_is_reported(synthetic_video):
    with Capture(str(synthetic_video)) as cap:
        assert cap.get_stream_start_time() >= 0.0


def test_fallback_reseek_to_same_frame_is_not_a_noop(synthetic_video):
    """set() must compare against the absolute position (seek_pos + pos),
    not the frames-read-since-last-seek counter. Otherwise a second seek to
    a frame number that happens to equal the current relative offset (or a
    repeat seek to the same absolute frame) silently no-ops and the reader
    is left at the wrong position."""
    expected = ffprobe_pts(synthetic_video, 5)
    cap = FFmpegNVDECCapture(str(synthetic_video), use_gpu=False)
    with cap as c:
        c.set(cv2.CAP_PROP_POS_FRAMES, 3)
        ok, _frame = c.read()
        assert ok
        first_pts = c.get_last_pts()
        assert first_pts == pytest.approx(expected[3], abs=1e-3)
        assert c.get(cv2.CAP_PROP_POS_FRAMES) == 4

        # Seek back to the same absolute frame and read again.
        c.set(cv2.CAP_PROP_POS_FRAMES, 3)
        ok, _frame = c.read()
        assert ok
        second_pts = c.get_last_pts()

        assert second_pts == pytest.approx(first_pts, abs=1e-6)
        assert second_pts == pytest.approx(expected[3], abs=1e-3)
