import subprocess

import cv2
import pytest

# Pixel region the drawbox in tests/conftest.py fills: x=160,y=300,w=320,h=30.
BAR_REGION = (160, 300, 480, 330)  # x1, y1, x2, y2
BAR_PRESENT_MIN_BRIGHTNESS = 200  # white bar ~255
BAR_ABSENT_MAX_BRIGHTNESS = 100  # background #202020 ~32


def _probe_frames(path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def _extract_frame(video, n: int, out_path) -> None:
    """Decode and save the nth (0-indexed) frame of video as a PNG."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(video), "-vf", f"select=eq(n\\,{n})",
         "-vframes", "1", "-fps_mode", "vfr", str(out_path)],
        capture_output=True, text=True, check=True,
    )


def _bar_region_mean_brightness(png_path) -> float:
    """Mean grayscale brightness of the pixel region the subtitle bar occupies."""
    image = cv2.imread(str(png_path))
    x1, y1, x2, y2 = BAR_REGION
    region = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def test_synthetic_video_has_ten_frames(synthetic_video):
    assert synthetic_video.exists()
    assert _probe_frames(synthetic_video) == 10


def test_synthetic_subtitle_video_has_75_frames(synthetic_subtitle_video):
    assert _probe_frames(synthetic_subtitle_video) == 75


@pytest.mark.parametrize("frame_n, bar_expected", [
    (24, False),  # just before the window
    (25, True),   # window start (inclusive)
    (49, True),   # window end (inclusive)
    (50, False),  # just after the window
])
def test_subtitle_bar_boundary_frames(
    synthetic_subtitle_video, tmp_path, frame_n, bar_expected
):
    """The white bar must appear in exactly frames 25-49 (inclusive), in the
    exact pixel region the drawbox targets - not just for 75 total frames."""
    out_path = tmp_path / f"frame_{frame_n}.png"
    _extract_frame(synthetic_subtitle_video, frame_n, out_path)
    brightness = _bar_region_mean_brightness(out_path)
    if bar_expected:
        assert brightness > BAR_PRESENT_MIN_BRIGHTNESS, (
            f"frame {frame_n}: expected bar present, "
            f"got mean brightness {brightness}"
        )
    else:
        assert brightness < BAR_ABSENT_MAX_BRIGHTNESS, (
            f"frame {frame_n}: expected bar absent, "
            f"got mean brightness {brightness}"
        )
