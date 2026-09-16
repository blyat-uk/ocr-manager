"""HDR tone mapping and the no-silent-degradation contract for PyAVCapture."""
import subprocess

import av
import av.filter
import numpy as np
import pytest

from videocr import pyav_adapter
from videocr.pyav_adapter import PyAVCapture

# Tagging the *frames* via setparams is what makes the encoder write the PQ
# VUI; the stream-level -color_* flags alone are dropped by this ffmpeg build
# (measured: color_trc stayed 2/unspecified for both libx264 and libx265).
PQ_FLAGS = [
    "-vf", "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc",
    "-color_trc", "smpte2084",
    "-color_primaries", "bt2020",
    "-colorspace", "bt2020nc",
]


def _make_clip(path, width, height, frames=5, rate=25, hdr=False,
               pix_fmt="yuv420p10le", codec="libx264"):
    duration = frames / rate
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", f"testsrc2=size={width}x{height}:rate={rate}:duration={duration}",
        "-pix_fmt", pix_fmt, "-c:v", codec,
    ]
    if hdr:
        cmd += PQ_FLAGS
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True)

    if hdr:
        # Verify the PQ transfer actually reached the bitstream before any
        # test relies on it -- an untagged clip would quietly turn every
        # HDR assertion below into a test of the SDR path.
        container = av.open(str(path))
        try:
            trc = int(container.streams.video[0].codec_context.color_trc)
        finally:
            container.close()
        assert trc == pyav_adapter._TRC_SMPTE2084, (
            f"HDR fixture was not tagged smpte2084 (color_trc={trc}); "
            "this ffmpeg build needs a different tagging mechanism here."
        )


def _raw_first_frame(path):
    """First frame converted the plain way: no tone map, no filter graph."""
    container = av.open(str(path))
    try:
        frame = next(container.decode(video=0))
        return frame.to_ndarray(format="bgr24")
    finally:
        container.close()


class _BrokenGraph:
    """Real filter graph that refuses to instantiate one named filter.

    Models the shipped failure exactly: av's binary wheel has no `zscale`,
    so `graph.add('zscale', ...)` raises ValueError and the whole graph --
    tone map *and* downscale -- fails to build.
    """

    def __init__(self, inner, failing_filter):
        self._inner = inner
        self._failing_filter = failing_filter

    def add(self, name, *args, **kwargs):
        if name == self._failing_filter:
            raise ValueError(f"no filter {name}")
        return self._inner.add(name, *args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _break_filter(monkeypatch, name):
    real = av.filter.Graph
    monkeypatch.setattr(av.filter, "Graph", lambda: _BrokenGraph(real(), name))


# --- (a) HDR sources decode correctly through PyAVCapture -------------------

def test_pyav_zscale_probe_reads_pyavs_own_registry():
    """The probe that picks the tone-map chain must ask PyAV, not the CLI.

    `_has_zscale()` shells out to the system ffmpeg, which on this machine
    reports zscale while av's bundled build does not have it. Consulting the
    wrong one is what made every PQ/HLG source fail to build a graph.
    """
    pyav_adapter._PYAV_ZSCALE_AVAILABLE = None
    try:
        assert pyav_adapter._pyav_has_zscale() == (
            "zscale" in av.filter.filters_available
        )
    finally:
        pyav_adapter._PYAV_ZSCALE_AVAILABLE = None


def test_hdr_pq_source_is_tone_mapped(tmp_path):
    """A PQ source must come back tone mapped, not passed through.

    Before the fix the graph could not be built (no zscale in the wheel),
    the exception handler swallowed it, and HDR was silently delivered
    untouched -- a regression against the subprocess backend this one
    replaced.
    """
    src = tmp_path / "pq.mp4"
    _make_clip(src, 320, 240, hdr=True)

    with PyAVCapture(str(src)) as cap:
        assert cap._needs_tonemap, "fixture is not being detected as HDR"
        assert cap._filter_graph is not None, "tone-map graph was not built"
        ok, frame = cap.read()
    assert ok
    assert frame.shape == (240, 320, 3)

    untouched = _raw_first_frame(src)
    assert untouched.shape == frame.shape
    max_diff = int(np.abs(frame.astype(int) - untouched.astype(int)).max())
    assert max_diff > 16, (
        "frame is indistinguishable from the un-tone-mapped conversion "
        f"(max abs diff {max_diff}); the tone-map chain did not run"
    )


def test_hdr_pq_source_honours_decode_target_height(tmp_path):
    """Tone map and downscale must both happen, in one graph.

    This is the exact configuration that previously returned
    native-resolution frames: the graph failed on zscale, the handler reset
    `_scale_factor` to 1.0, and video.py went on slicing with coordinates it
    had already rescaled into 360-high output space.
    """
    src = tmp_path / "pq_hd.mp4"
    _make_clip(src, 1280, 720, hdr=True)

    with PyAVCapture(str(src), decode_target_height=360) as cap:
        assert cap._filter_graph is not None
        assert cap._scale_factor == pytest.approx(0.5)
        ok, frame = cap.read()
    assert ok
    assert frame.shape == (360, 640, 3), (
        f"decode_target_height was not honoured: got {frame.shape}"
    )


def test_hdr_pq_crop_matches_the_full_frame_slice(tmp_path):
    """The in-graph crop stays exact inside the tone-map chain."""
    src = tmp_path / "pq_crop.mp4"
    _make_clip(src, 1280, 720, hdr=True)
    crop = (64, 120, 192, 64)  # x, y, w, h in 640x360 output space

    with PyAVCapture(str(src), decode_target_height=360) as cap:
        ok, full = cap.read()
        assert ok
        full = full.copy()
    with PyAVCapture(str(src), decode_target_height=360, crop_rect=crop) as cap:
        graph_active = cap._crop_graph_active
        ok, cropped = cap.read()
        assert ok

    x, y, w, h = crop
    expected = full[y:y + h, x:x + w]
    got = cropped if graph_active else cropped[y:y + h, x:x + w]
    assert np.array_equal(got, expected), (
        f"max abs diff {int(np.abs(got.astype(int) - expected.astype(int)).max())}"
    )


# --- (b) A capture that cannot honour its geometry must not degrade --------

@pytest.mark.parametrize("hdr, broken", [
    pytest.param(False, "scale", id="sdr-downscale-unbuildable"),
    pytest.param(True, "tonemap", id="hdr-tonemap-unbuildable"),
    pytest.param(True, "scale", id="hdr-downscale-unbuildable"),
])
def test_unbuildable_graph_refuses_instead_of_returning_native_frames(
    tmp_path, monkeypatch, hdr, broken
):
    src = tmp_path / "clip.mp4"
    _make_clip(src, 1280, 720, hdr=hdr)
    _break_filter(monkeypatch, broken)

    with pytest.raises(RuntimeError) as exc_info:
        with PyAVCapture(str(src), decode_target_height=360) as cap:
            cap.read()

    message = str(exc_info.value)
    assert "1280x720" in message and "640x360" in message, message
    assert f"no filter {broken}" in message


def test_unbuildable_graph_never_leaves_a_stale_crop_plan(tmp_path, monkeypatch):
    """video.py slices with output-space coordinates whenever `_crop_slice`
    is None, so a failed capture must not be usable at all -- and must not
    leave a half-built plan behind for anything that catches the error."""
    src = tmp_path / "clip.mp4"
    _make_clip(src, 1280, 720)
    _break_filter(monkeypatch, "scale")

    cap = PyAVCapture(str(src), decode_target_height=360,
                      crop_rect=(64, 120, 192, 64))
    with pytest.raises(RuntimeError):
        cap.__enter__()

    assert cap._crop_slice is None
    assert cap._crop_graph_active is False
    assert cap._filter_graph is None
    assert cap.container is None, "the demuxer was leaked by the failed __enter__"
