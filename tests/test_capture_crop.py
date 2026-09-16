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


def _make_clip(path, width, height, frames=8, rate=25,
               pix_fmt="yuv420p", codec="libx264"):
    duration = frames / rate
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={rate}:duration={duration}",
         "-pix_fmt", pix_fmt, "-c:v", codec, str(path)],
        check=True, capture_output=True,
    )


# A source that needs a downscale, hence a filter graph, hence an in-graph
# crop -- but small enough to encode in a test. 480 -> 240 is an exact 2:1
# ratio on both axes, so _crop_axis_plan admits it.
_GRAPH_SRC = (640, 480)
_GRAPH_TARGET_HEIGHT = 240
_GRAPH_CROP = (64, 60, 192, 64)  # x, y, w, h inside the 320x240 output


@pytest.fixture(scope="module")
def downscaled_clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("crop") / "downscale.mp4"
    _make_clip(out, *_GRAPH_SRC, frames=10)
    return out


# (pix_fmt, codec) pairs for a source needing neither tone map nor downscale.
NO_GRAPH_FORMATS = [
    pytest.param("yuv420p", "libx264", id="8bit-h264"),
    pytest.param("yuv420p10le", "libx264", id="10bit-h264"),
    pytest.param("yuv420p10le", "libx265", id="10bit-h265"),
]


@pytest.mark.parametrize("pix_fmt, codec", NO_GRAPH_FORMATS)
def test_crop_refused_when_no_filter_graph_is_needed(tmp_path, pix_fmt, codec):
    """A crop must never be the sole reason a filter graph exists.

    With no graph, read() converts via `frame.to_ndarray(format='bgr24')`;
    with one, via a `format=bgr24` filter node. The two disagree for any
    source deeper than 8 bits (measured max abs diff 148 on yuv420p10le),
    so accepting the crop here would silently change every pixel. The
    capture must refuse, hand back the untouched reference frame, and leave
    `_crop_slice` None for video.py to slice in Python.
    """
    src = tmp_path / "clip.mp4"
    _make_clip(src, 320, 240, frames=5, pix_fmt=pix_fmt, codec=codec)

    with PyAVCapture(str(src)) as cap:
        reference = _read_all(cap, 5)
    with PyAVCapture(str(src), crop_rect=CROP) as cap:
        assert cap._crop_slice is None
        assert not cap._crop_graph_active
        assert cap._filter_graph is None
        got = _read_all(cap, 5)

    for i, (ref, g) in enumerate(zip(reference, got)):
        assert np.array_equal(g, ref), (
            f"frame {i}: asking for a crop perturbed the frame "
            f"(max abs diff {int(np.abs(g.astype(int) - ref.astype(int)).max())})"
        )


def test_crop_in_graph_matches_numpy_slice(downscaled_clip):
    x, y, w, h = _GRAPH_CROP
    with PyAVCapture(str(downscaled_clip),
                     decode_target_height=_GRAPH_TARGET_HEIGHT) as cap:
        full = _read_all(cap, 10)
    with PyAVCapture(str(downscaled_clip),
                     decode_target_height=_GRAPH_TARGET_HEIGHT,
                     crop_rect=_GRAPH_CROP) as cap:
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


def test_crop_pts_unchanged(downscaled_clip):
    with PyAVCapture(str(downscaled_clip),
                     decode_target_height=_GRAPH_TARGET_HEIGHT) as cap:
        _read_all(cap, 5)
        full_pts = cap.get_last_pts()
    with PyAVCapture(str(downscaled_clip),
                     decode_target_height=_GRAPH_TARGET_HEIGHT,
                     crop_rect=_GRAPH_CROP) as cap:
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


# (source width, source height, decode_target_height, crop in output space,
#  pix_fmt, codec)
CROP_INVARIANT_MATRIX = [
    pytest.param(320, 240, None, (64, 60, 192, 64), "yuv420p", "libx264",
                 id="320x240-no-downscale"),
    pytest.param(3840, 2160, 1080, (400, 900, 1024, 96), "yuv420p", "libx264",
                 id="3840x2160-to-1080-ratio2"),
    pytest.param(1920, 1200, 1080, (64, 120, 192, 64), "yuv420p", "libx264",
                 id="1920x1200-to-1080-ratio10-9-reproducer"),
    pytest.param(2704, 1520, 1080, (64, 120, 192, 64), "yuv420p", "libx264",
                 id="2704x1520-to-1080"),
    pytest.param(1920, 1088, 1080, (64, 120, 192, 64), "yuv420p", "libx264",
                 id="1920x1088-to-1080"),
    # 10-bit rows. The 8-bit-only matrix above is what let the "graph exists
    # only because a crop was asked for" bug ship: it is invisible at 8 bits
    # because to_ndarray and the format filter agree there.
    pytest.param(320, 240, None, (64, 60, 192, 64), "yuv420p10le", "libx264",
                 id="10bit-h264-no-downscale"),
    pytest.param(320, 240, None, (64, 60, 192, 64), "yuv420p10le", "libx265",
                 id="10bit-h265-no-downscale"),
    pytest.param(1280, 720, 360, (64, 120, 192, 64), "yuv420p10le", "libx264",
                 id="10bit-h264-to-360"),
    pytest.param(1280, 720, 360, (64, 120, 192, 64), "yuv420p10le", "libx265",
                 id="10bit-h265-to-360"),
]


@pytest.mark.parametrize(
    "width, height, decode_target_height, crop, pix_fmt, codec",
    CROP_INVARIANT_MATRIX,
)
def test_crop_invariant_holds_across_decode_ratios(
    tmp_path, width, height, decode_target_height, crop, pix_fmt, codec
):
    """The only two acceptable outcomes for any (source size, pixel format,
    decode target, crop) combination: the graph crop is active and
    pixel-identical to the full-frame reference slice, or the crop stage was
    refused (video.py is then responsible for slicing in Python). A graph
    crop that merely approximates the reference must be unreachable by
    construction.
    """
    src = tmp_path / "clip.mp4"
    _make_clip(src, width, height, frames=5, pix_fmt=pix_fmt, codec=codec)

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
                f"{width}x{height}({pix_fmt})->{decode_target_height} frame {i}: "
                "Python-slice fallback on the refused path diverged from the reference"
            )
        return

    x, y, w, h = crop
    for i, (f, c) in enumerate(zip(full, cropped)):
        expected = f[y:y + h, x:x + w]
        assert np.array_equal(c, expected), (
            f"{width}x{height}({pix_fmt})->{decode_target_height} frame {i}: "
            f"max abs diff {int(np.abs(c.astype(int) - expected.astype(int)).max())}"
        )
