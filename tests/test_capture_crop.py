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


def test_configure_crop_called_after_enter_matches_slice(downscaled_clip):
    """The crop can be configured *after* __enter__ has already returned,
    not only supplied to the constructor. Every other crop test in this
    file (and in tests/test_capture_hdr.py) passes crop_rect= to the
    constructor, so none of them exercise this path - which is exactly
    what videocr/video.py's run_ocr() relies on: it can't know a crop
    pre-scaled to decode-output coordinates before opening, since that
    scaling needs the native height this same capture is the one place
    to learn.
    """
    x, y, w, h = _GRAPH_CROP
    with PyAVCapture(str(downscaled_clip),
                     decode_target_height=_GRAPH_TARGET_HEIGHT) as cap:
        full = _read_all(cap, 10)
    with PyAVCapture(str(downscaled_clip),
                     decode_target_height=_GRAPH_TARGET_HEIGHT) as cap:
        # No crop_rect at construction this time - the graph exists
        # already (downscale needs one), but with no crop yet.
        assert cap._crop_slice is None
        assert not cap._crop_graph_active
        cap.configure_crop(_GRAPH_CROP)
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


def test_reconfiguring_crop_clears_stale_state(downscaled_clip, monkeypatch):
    """Re-configuring from an active crop to no crop, or to one
    _plan_crop ends up refusing, must not leave the previous call's crop
    state behind: a stale _crop_slice/_crop_graph_active would make
    _frame_producer's `not getattr(v, '_crop_slice', None)` guard
    (videocr/video.py) skip its own Python-level slice, believing the
    graph already cropped when it did not -- OCRing the wrong region of
    the frame.
    """
    import videocr.pyav_adapter as pyav_adapter_mod

    with PyAVCapture(str(downscaled_clip),
                     decode_target_height=_GRAPH_TARGET_HEIGHT,
                     crop_rect=_GRAPH_CROP) as cap:
        assert cap._crop_graph_active
        assert cap._crop_slice is not None

        # Re-configure to no crop at all.
        cap.configure_crop(None)
        assert cap._crop_request is None
        assert cap._crop_slice is None
        assert cap._crop_native is None
        assert cap._crop_scaled_size is None
        assert not cap._crop_graph_active

        # Back to an admitted crop, then to one _plan_crop refuses.
        # _crop_axis_plan (see its own docstring) is a function of the
        # capture's native/output *dimensions* alone, never of the crop
        # rectangle -- so on one capture (fixed dimensions) a crop is
        # either always admissible or always refused, and a real refused
        # geometry can't be reached by changing only the crop rect on an
        # already-admitting capture the way this test needs to. Forcing
        # the refusal at its actual decision point exercises exactly the
        # code path test_crop_invariant_holds_across_decode_ratios's own
        # refused matrix entries take, without depending on hunting for a
        # specific width/height/target combination that happens to
        # refuse.
        cap.configure_crop(_GRAPH_CROP)
        assert cap._crop_graph_active

        monkeypatch.setattr(pyav_adapter_mod, "_crop_axis_plan", lambda *a, **k: None)
        cap.configure_crop(_GRAPH_CROP)
        assert cap._crop_request == _GRAPH_CROP  # still recorded, just refused
        assert cap._crop_slice is None
        assert cap._crop_native is None
        assert cap._crop_scaled_size is None
        assert not cap._crop_graph_active


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


# --------------------------------------------------------------------------
# Pipeline level: Video.run_ocr() itself, not just PyAVCapture in isolation
# --------------------------------------------------------------------------

def test_run_ocr_turns_on_the_graph_crop_for_a_downscaled_source(tmp_path, monkeypatch):
    """The test that would have caught finding 1 (crop_rect only ever fed
    by tests, so Video.run_ocr() never actually turned the graph crop on
    in production - see task-3-report.md). Every crop test above (and in
    tests/test_capture_hdr.py) drives PyAVCapture directly with
    crop_rect= at the constructor, so none of them would notice run_ocr()
    itself never calling configure_crop(). Confirmed by reverting the
    configure_crop() call at videocr/video.py:423-428 in a scratch copy:
    every other test in this suite and all four golden digests still
    pass; only this test fails.

    No GPU/OCR needed: OCR engine construction and the producer thread's
    own decode work are stubbed out, since this only has to observe what
    run_ocr() did to the capture before any frame is actually processed.
    """
    from videocr import engine_registry
    from videocr.video import Video

    src = tmp_path / "uhd.mp4"
    _make_clip(src, 3840, 2160, frames=3)
    # Native-space crop (crop_x/y/width/height are native, per
    # videocr/video.py's own inference block, clamped against self.width/
    # self.height before any decode-scale is applied). Scaled by
    # run_ocr()'s own decode_height/height factor of 0.5 for this
    # 2160-tall source, this lands on (400, 900, 1024, 96) in the
    # 1920x1080 decode-target output space - the exact geometry
    # test_crop_matches_slice_with_decode_downscale already proves the
    # graph accepts.
    crop_x, crop_y, crop_w, crop_h = 800, 1800, 2048, 192

    seen = {}

    def spying_frame_producer(self, v, queue, *_a, **_kw):
        # Record capture state before consuming anything, then end the
        # run immediately - this test only needs to know what run_ocr()
        # did to the capture, not decode the clip.
        seen['crop_graph_active'] = getattr(v, '_crop_graph_active', False)
        seen['crop_slice'] = getattr(v, '_crop_slice', None)
        self._producer_time = 0.0
        queue.put(("done",))

    monkeypatch.setattr(Video, "_frame_producer", spying_frame_producer)

    class _DummyEngine:
        def predict(self, batch):
            return [[] for _ in batch]

    monkeypatch.setattr(engine_registry, "get_ocr_engine",
                        lambda *a, **k: _DummyEngine())

    v = Video(str(src), None, None)
    v.run_ocr(
        use_gpu=False, lang='ch', time_start='0:00', time_end='',
        conf_threshold=95, use_fullframe=False, brightness_threshold=150,
        similar_image_threshold=0.3, similar_pixel_threshold=25, frames_to_skip=0,
        crop_x=crop_x, crop_y=crop_y, crop_width=crop_w, crop_height=crop_h,
    )

    assert seen['crop_graph_active'] is True, (
        "Video.run_ocr() did not turn the graph crop on for a source "
        "that needs a filter graph anyway - the exact regression "
        "finding 1 was."
    )
    assert seen['crop_slice'] is not None
