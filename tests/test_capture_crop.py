import subprocess

import numpy as np
import pytest

from videocr.pyav_adapter import PyAVCapture

CROP = (64, 120, 192, 64)  # x, y, w, h inside a 320x240 frame


def _read_all(cap, n):
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        assert ok
        frames.append(frame.copy())
    return frames


def _make_clip(path, width, height, frames=8, rate=25):
    duration = frames / rate
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={rate}:duration={duration}",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", str(path)],
        check=True, capture_output=True,
    )


def test_crop_in_graph_matches_numpy_slice(synthetic_video):
    x, y, w, h = CROP
    with PyAVCapture(str(synthetic_video)) as cap:
        full = _read_all(cap, 10)
    with PyAVCapture(str(synthetic_video), crop_rect=CROP) as cap:
        # Prove the mechanism, not a coincidence: this must actually run
        # through the filter-graph crop, not silently fall back to slicing
        # the full frame (which would trivially match the reference too).
        assert cap._crop_graph_active
        cropped = _read_all(cap, 10)

    assert len(full) == len(cropped) == 10
    for i, (f, c) in enumerate(zip(full, cropped)):
        expected = f[y:y + h, x:x + w]
        assert c.shape == expected.shape, f"frame {i} shape {c.shape} != {expected.shape}"
        assert np.array_equal(c, expected), (
            f"frame {i} differs, max abs diff "
            f"{int(np.abs(c.astype(int) - expected.astype(int)).max())}"
        )


def test_crop_pts_unchanged(synthetic_video):
    with PyAVCapture(str(synthetic_video)) as cap:
        _read_all(cap, 5)
        full_pts = cap.get_last_pts()
    with PyAVCapture(str(synthetic_video), crop_rect=CROP) as cap:
        assert cap._crop_graph_active
        _read_all(cap, 5)
        crop_pts = cap.get_last_pts()
    assert full_pts == crop_pts


def test_crop_matches_slice_with_decode_downscale(tmp_path):
    src = tmp_path / "uhd.mp4"
    _make_clip(src, 3840, 2160, frames=5)
    crop = (400, 900, 1024, 96)  # in 1080p output space
    with PyAVCapture(str(src), decode_target_height=1080) as cap:
        full = _read_all(cap, 5)
    with PyAVCapture(str(src), decode_target_height=1080, crop_rect=crop) as cap:
        assert cap._crop_graph_active
        cropped = _read_all(cap, 5)
    x, y, w, h = crop
    for i, (f, c) in enumerate(zip(full, cropped)):
        expected = f[y:y + h, x:x + w]
        assert np.array_equal(c, expected), (
            f"frame {i}: max abs diff "
            f"{int(np.abs(c.astype(int) - expected.astype(int)).max())}"
        )


# (source width, source height, decode_target_height, crop_rect in output space)
CROP_INVARIANT_MATRIX = [
    pytest.param(320, 240, None, (64, 60, 192, 64), id="320x240-no-downscale"),
    pytest.param(3840, 2160, 1080, (400, 900, 1024, 96), id="3840x2160-to-1080-ratio2"),
    pytest.param(1920, 1200, 1080, (64, 120, 192, 64), id="1920x1200-to-1080-ratio10-9-reproducer"),
    pytest.param(2704, 1520, 1080, (64, 120, 192, 64), id="2704x1520-to-1080"),
    pytest.param(1920, 1088, 1080, (64, 120, 192, 64), id="1920x1088-to-1080"),
]


@pytest.mark.parametrize(
    "width, height, decode_target_height, crop", CROP_INVARIANT_MATRIX
)
def test_crop_invariant_holds_across_decode_ratios(
    tmp_path, width, height, decode_target_height, crop
):
    """The only two acceptable outcomes for any (source size, decode target,
    crop) combination: the graph crop is active and pixel-identical to the
    full-frame reference slice, or the crop stage was refused (video.py is
    then responsible for slicing in Python). A graph crop that merely
    approximates the reference must be unreachable by construction.
    """
    src = tmp_path / "clip.mp4"
    _make_clip(src, width, height, frames=5)

    with PyAVCapture(str(src), decode_target_height=decode_target_height) as cap:
        full = _read_all(cap, 5)
    with PyAVCapture(str(src), decode_target_height=decode_target_height,
                      crop_rect=crop) as cap:
        graph_active = cap._crop_graph_active
        cropped = _read_all(cap, 5)

    if not graph_active:
        # Refused: video.py's `not getattr(v, '_crop_slice', None)` guard
        # (videocr/video.py) depends on `_crop_slice` being exactly None
        # here -- assert that directly. Then prove end-to-end that the
        # frame PyAVCapture handed back, sliced in Python the way video.py
        # would, reproduces the reference exactly (reusing the clip and
        # frames already generated above; no extra decode needed).
        assert cap._crop_slice is None
        x, y, w, h = crop
        for i, (f, c) in enumerate(zip(full, cropped)):
            expected = f[y:y + h, x:x + w]
            got = c[y:y + h, x:x + w]
            assert np.array_equal(got, expected), (
                f"{width}x{height}->{decode_target_height} frame {i}: "
                "Python-slice fallback on the refused path diverged from the reference"
            )
        return

    x, y, w, h = crop
    for i, (f, c) in enumerate(zip(full, cropped)):
        expected = f[y:y + h, x:x + w]
        assert np.array_equal(c, expected), (
            f"{width}x{height}->{decode_target_height} frame {i}: "
            f"max abs diff {int(np.abs(c.astype(int) - expected.astype(int)).max())}"
        )
