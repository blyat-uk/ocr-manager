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
    """set() must compare the seek target against the absolute position
    (seek_pos + pos), not against the frames-read-since-last-seek counter.

    Seek to frame 2, read twice (so the relative "frames since seek"
    counter reaches 2 -- the same number as the frame we're about to
    re-seek to), then seek back to frame 2 again. A guard that compares
    the target against the relative counter sees 2 == 2 and wrongly
    treats this as a no-op, leaving the reader positioned past frame 2
    instead of seeking back to it.
    """
    expected = ffprobe_pts(synthetic_video, 5)
    cap = FFmpegNVDECCapture(str(synthetic_video), use_gpu=False)
    with cap as c:
        c.set(cv2.CAP_PROP_POS_FRAMES, 2)
        ok, _frame = c.read()
        assert ok
        assert c.get_last_pts() == pytest.approx(expected[2], abs=1e-3)
        ok, _frame = c.read()
        assert ok
        assert c.get_last_pts() == pytest.approx(expected[3], abs=1e-3)
        assert c.get(cv2.CAP_PROP_POS_FRAMES) == 4

        # Seek back to frame 2. The relative "since last seek" counter is
        # also 2 at this point, which is what makes the stale guard fail.
        c.set(cv2.CAP_PROP_POS_FRAMES, 2)
        ok, _frame = c.read()
        assert ok

        assert c.get_last_pts() == pytest.approx(expected[2], abs=1e-3)
