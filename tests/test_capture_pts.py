import json
import subprocess

import cv2
import pytest

from videocr.pyav_adapter import Capture, FFmpegNVDECCapture, PyAVCapture


def ffprobe_pts(path, count):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-read_intervals", f"%+#{count}",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    frames = json.loads(out.stdout)["frames"]
    return [float(f["pts_time"]) for f in frames]


def ffprobe_format_start_time(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=start_time",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["start_time"])


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


def test_container_start_time_is_reported(offset_video):
    """Both backends must report a genuine non-zero start, not a hardcoded
    0.0. offset_video's first frame, like its container, starts at ~1.5 s
    (-output_ts_offset), so a backend that always returns 0.0 fails this
    test."""
    expected_start = ffprobe_format_start_time(offset_video)
    assert expected_start == pytest.approx(1.5, abs=1e-3)
    assert ffprobe_pts(offset_video, 1)[0] == pytest.approx(expected_start, abs=1e-3)

    with PyAVCapture(str(offset_video)) as pyav_cap:
        assert pyav_cap.get_stream_start_time() == pytest.approx(expected_start, abs=1e-3)

    ffmpeg_cap = FFmpegNVDECCapture(str(offset_video), use_gpu=False)
    with ffmpeg_cap as c:
        assert c.get_stream_start_time() == pytest.approx(expected_start, abs=1e-3)


@pytest.mark.parametrize("backend", ["pyav", "fallback"])
def test_start_time_is_the_container_not_the_first_video_frame(delayed_video_subtitle_clip, backend):
    """Subtitle times count from the player's zero, so a frame's time in the
    output is its PTS minus get_stream_start_time(). The player's zero is the
    container start (libavformat's format start_time, the earliest of all the
    streams) -- what mpv rebases to with --rebase-start-time=yes, its default.

    In this clip the audio, and with it the container, starts at 0 while the
    first frame starts at 0.080 s -- the layout of Jinwu Guard episodes 07,
    08, 09, 12 and 15. Counting from the first video frame instead puts the
    bar's first frame out at 1.00 s, two frames before the 1.080 s the bar is
    actually painted at, which is the 80 ms the line then shows early.
    Rendering this very clip through mpv confirms it: with 1.00 the line is on
    screen two frames before the bar, with 1.08 they appear together.
    """
    if backend == "pyav":
        cap = PyAVCapture(str(delayed_video_subtitle_clip))
    else:
        cap = FFmpegNVDECCapture(str(delayed_video_subtitle_clip), use_gpu=False)
    with cap as c:
        assert c.get_stream_start_time() == pytest.approx(0.0, abs=1e-9)
        bar_pts = None
        for _ in range(40):
            ok, frame = c.read()
            assert ok
            if frame[315, 320].min() > 200:
                bar_pts = c.get_last_pts()
                break
        assert bar_pts is not None
        assert bar_pts - c.get_stream_start_time() == pytest.approx(1.08, abs=1e-3)


def test_fallback_pts_includes_container_start_time(offset_video):
    """The container start-time offset must reach the reported PTS values,
    not just get_stream_start_time(). Read the first frame of offset_video
    and confirm its PTS matches ffprobe's true (offset) pts_time -- this
    fails if the offset is computed but never added in read()."""
    expected = ffprobe_pts(offset_video, 1)
    cap = FFmpegNVDECCapture(str(offset_video), use_gpu=False)
    with cap as c:
        ok, _frame = c.read()
        assert ok
        assert c.get_last_pts() == pytest.approx(expected[0], abs=1e-3)


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
