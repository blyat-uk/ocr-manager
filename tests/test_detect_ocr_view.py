"""Frames as the OCR pass sees them (core/detect/ocr_view.py).

The mirror tests drive the real `videocr.video.Video.run_ocr` (with a fake
capture, or a real one on encoded clips) and compare what it hands the OCR
engine against ocr_view's own view of the same frames.
"""
import subprocess

import cv2
import numpy as np
import pytest

from core.detect import ocr_view as OV

H, W = 54, 1344
CENTRE_X0 = (W - H) // 2  # the gate's centre square, as videocr/video.py computes it


def _empty_ocr_item():
    return {"rec_texts": [], "rec_scores": [], "rec_polys": []}


# --------------------------------------------------------------------------
# Sampling times and keep ranges
# --------------------------------------------------------------------------

def test_sample_times_skip_the_first_and_last_tenth_without_ranges():
    times = OV.sample_times(1000.0, None, 24)
    assert len(times) == 24
    assert all(100.0 <= t <= 900.0 for t in times)
    assert times == sorted(times)
    gaps = np.diff(times)
    assert gaps.min() > 0.8 * (800.0 / 24)
    assert times[0] < 150 and times[-1] > 850


def test_sample_times_stay_inside_the_keep_ranges_in_proportion():
    ranges = [("1:00", "2:00"), ("5:00", "5:30")]
    times = OV.sample_times(600.0, ranges, 24)
    assert len(times) == 24
    first = [t for t in times if 60.0 <= t <= 120.0]
    second = [t for t in times if 300.0 <= t <= 330.0]
    assert len(first) + len(second) == 24
    assert len(first) == 16 and len(second) == 8


def test_later_sampling_rounds_interleave_with_the_first():
    first = OV.sample_times(1000.0, None, 24)
    second = OV.sample_times(1000.0, None, 24, phase=0.0)
    third = OV.sample_times(1000.0, None, 24, phase=0.25)
    merged = sorted(first + second + third)
    assert len(set(merged)) == 72
    assert all(100.0 <= t <= 900.0 for t in merged)
    # every second-round time sits between two first-round times
    for a, b in zip(first, first[1:]):
        assert sum(a < t < b for t in second) == 1


def test_sample_times_open_ended_ranges_run_to_the_file_edges():
    times = OV.sample_times(600.0, [(None, "1:00"), ("9:00", "")], 12)
    assert all(t <= 60.0 or t >= 540.0 for t in times)
    assert sum(t <= 60.0 for t in times) == 6


def test_keep_ranges_stored_as_mappings_are_read_like_pairs():
    assert OV.keep_spans(600.0, [{"start": "1:00", "end": "2:00"}]) == [(60.0, 120.0)]


def test_overlapping_keep_ranges_are_merged_before_sampling():
    # Overlap counted twice would put twice the samples in 1:30-2:00.
    assert OV.keep_spans(600.0, [("1:30", "2:30"), ("1:00", "2:00")]) == [(60.0, 150.0)]
    times = OV.sample_times(600.0, [("1:00", "2:00"), ("1:30", "2:30")], 18)
    assert np.allclose(np.diff(times), 90.0 / 18)


def test_keep_ranges_that_select_nothing_never_fall_back_to_the_whole_file():
    ranges = [("12:00", "13:00"), ("5:00", "4:00")]
    assert OV.keep_spans(600.0, ranges) == []
    assert OV.sample_times(600.0, ranges, 24) == []


# --------------------------------------------------------------------------
# Mirror of videocr/video.py: downscale, mask and gate, pinned against the
# real run_ocr with a fake capture (no media, no models).
# --------------------------------------------------------------------------

class _RecordingOCR:
    def __init__(self):
        self.frames = []

    def predict(self, frames):
        self.frames.extend(f.copy() for f in frames)
        return [_empty_ocr_item() for _ in frames]


def _run_ocr_on_frames(monkeypatch, frames, threshold, crop=None):
    """Feed `frames` through the real Video.run_ocr and return what it hands
    the OCR engine."""
    from videocr import utils
    from videocr import video as V

    h, w = frames[0].shape[:2]

    class FakeCapture:
        def __init__(self, path, use_gpu=True, decode_target_height=None, crop_rect=None):
            self._i = 0
            self._crop_slice = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_COUNT: len(frames), cv2.CAP_PROP_FPS: 25.0,
                cv2.CAP_PROP_FRAME_HEIGHT: h, cv2.CAP_PROP_FRAME_WIDTH: w,
            }.get(prop, 0)

        def set(self, prop, value):
            return True

        def read(self):
            if self._i >= len(frames):
                return False, None
            self._i += 1
            return True, frames[self._i - 1].copy()

        def get_last_pts(self):
            return (self._i - 1) / 25.0

        def get_stream_start_time(self):
            return 0.0

    recorder = _RecordingOCR()
    monkeypatch.setattr(V, "Capture", FakeCapture)
    monkeypatch.setattr(utils, "create_ocr_engine", lambda *a, **k: recorder)
    video = V.Video("fake.mp4", None, None)
    cx, cy, cw, ch = crop if crop else (None, None, None, None)
    video.run_ocr(False, "ch", "", "", 95, crop is None, threshold, 0, 25, 0, cx, cy, cw, ch)
    return recorder.frames


def _separated(frames):
    """Two blank frames after each test frame end any tracking state, so a
    frame reaches OCR exactly when it trips the gate by itself."""
    blank = np.zeros_like(frames[0])
    out = []
    for f in frames:
        out += [f, blank, blank]
    return out


def _blocky_frame(rng, h, w, block=16):
    """Grey blocks with per-channel noise: bright blocks survive an area
    downscale (plain noise would average out and never trip the gate), and
    the noise makes the min-channel mask cut inside them."""
    grey = rng.integers(0, 256, (h // block + 1, w // block + 1))
    img = np.repeat(np.repeat(grey, block, 0), block, 1)[:h, :w]
    noise = rng.integers(-12, 13, (h, w, 3))
    return np.clip(img[..., None] + noise, 0, 255).astype(np.uint8)


# 1142: int(1142 * (720 / 1142)) is 719 -- the copy must keep run_ocr's float
# arithmetic, not "fix" it to 720.
@pytest.mark.parametrize("h, w", [(53, 1344), (720, 900), (721, 900), (1080, 1920), (1142, 700), (2000, 400)])
def test_ocr_view_and_mask_match_what_run_ocr_hands_the_engine(monkeypatch, h, w):
    rng = np.random.default_rng(h * 7 + w)
    frames = [_blocky_frame(rng, h, w) for _ in range(3)]
    threshold = 200

    seen = _run_ocr_on_frames(monkeypatch, _separated(frames), threshold)

    expected = [OV.mask(OV.to_ocr_view(f), threshold) for f in frames]
    assert all(OV.gate_fires(e) for e in expected)
    assert len(seen) == len(expected)
    for got, want in zip(seen, expected):
        assert got.shape == want.shape
        assert np.array_equal(got, want)


def test_gate_fires_at_exactly_the_minimum_variance_like_run_ocr(monkeypatch):
    # Ten isolated pixels of 27 in the 54x54 centre square give a Laplacian
    # variance of exactly 145800 / 2916 = 50.0: the OCR gate's ">=" fires.
    exact = np.zeros((H, W, 3), dtype=np.uint8)
    nine = np.zeros((H, W, 3), dtype=np.uint8)
    spots = [(5 + 3 * i, CENTRE_X0 + 5 + 3 * i) for i in range(10)]
    for k, (r, c) in enumerate(spots):
        exact[r, c] = 27
        if k < 9:
            nine[r, c] = 27

    seen = _run_ocr_on_frames(monkeypatch, _separated([exact, nine]), 20)

    assert OV.gate_fires(OV.mask(exact, 20))
    assert not OV.gate_fires(OV.mask(nine, 20))
    assert len(seen) == 1 and np.array_equal(seen[0], OV.mask(exact, 20))


def test_gate_matches_run_ocr_at_the_variance_boundary(monkeypatch):
    # One bright pixel on black: the centre square's Laplacian variance is
    # 20*v^2/h^2, which crosses MIN_LAPLACIAN_VARIANCE between v=85 and 86
    # for h=54. A pixel just outside the centre square must never count.
    frames = []
    for v, col in [(85, CENTRE_X0 + 20), (86, CENTRE_X0 + 20), (255, CENTRE_X0 - 3),
                   (120, CENTRE_X0 + 1), (90, CENTRE_X0 + H + 2)]:
        f = np.zeros((H, W, 3), dtype=np.uint8)
        f[27, col] = v
        frames.append(f)

    seen = _run_ocr_on_frames(monkeypatch, _separated(frames), 80)

    fired = [f for f in frames if OV.gate_fires(OV.mask(OV.to_ocr_view(f), 80))]
    assert [int(f.max()) for f in fired] == [86, 120]
    assert len(seen) == len(fired)
    for got, want in zip(seen, fired):
        assert np.array_equal(got, OV.mask(OV.to_ocr_view(want), 80))


# --------------------------------------------------------------------------
# Frame source: grab_ocr_strips must return exactly the pixels the OCR pass
# sees, through the same capture chain (decode downscale, in-graph crop,
# 10-bit conversion, HDR tone map).
# --------------------------------------------------------------------------

_HDR_ARGS = ["-vf", "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc",
             "-color_trc", "smpte2084", "-color_primaries", "bt2020", "-colorspace", "bt2020nc"]


def _encode_cmd(path, size, pix_fmt, frames, gop, hdr):
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
            "-i", f"testsrc2=size={size}:rate=25:duration={frames / 25}",
            "-pix_fmt", pix_fmt, "-c:v", "libx264", "-g", str(gop), "-preset", "ultrafast",
            *(_HDR_ARGS if hdr else []), str(path)]


def _stream_is_pq(path):
    import av
    from videocr import pyav_adapter

    container = av.open(str(path))
    try:
        return int(container.streams.video[0].codec_context.color_trc) == pyav_adapter._TRC_SMPTE2084
    finally:
        container.close()


@pytest.fixture(scope="session")
def encoder(tmp_path_factory):
    """encoder(path, size, pix_fmt, frames=40, gop=15, hdr=False) -> path.

    Probes this ffmpeg build once per (pix_fmt, hdr) with a tiny clip and
    SKIPS -- visibly, with the reason -- when libx264 cannot write that pixel
    format or the PQ tag does not reach the stream, instead of failing on a
    missing build feature."""
    probe_dir = tmp_path_factory.mktemp("encoder-probe")
    problems = {}

    def problem(pix_fmt, hdr):
        key = (pix_fmt, hdr)
        if key not in problems:
            probe = probe_dir / f"{pix_fmt}-{int(hdr)}.mp4"
            result = subprocess.run(_encode_cmd(probe, "64x64", pix_fmt, 2, 1, hdr), capture_output=True, text=True)
            if result.returncode != 0:
                last_line = (result.stderr.strip().splitlines() or ["no error output"])[-1]
                problems[key] = (f"this ffmpeg cannot encode {pix_fmt}{' with PQ setparams' if hdr else ''} "
                                 f"via libx264: {last_line}")
            elif hdr and not _stream_is_pq(probe):
                problems[key] = "this ffmpeg build does not write the PQ transfer tag via setparams"
            else:
                problems[key] = None
        return problems[key]

    def encode(path, size, pix_fmt, frames=40, gop=15, hdr=False):
        reason = problem(pix_fmt, hdr)
        if reason:
            pytest.skip(reason)
        subprocess.run(_encode_cmd(path, size, pix_fmt, frames, gop, hdr), check=True, capture_output=True)
        return path

    return encode


def _run_ocr_frames_by_pts(monkeypatch, path, crop, time_start="", time_end=""):
    """Every frame the real run_ocr hands its OCR engine, keyed by PTS, with
    the brightness filter off so each decoded frame is handed over as-is."""
    from videocr import utils
    from videocr import video as V

    recorder = _RecordingOCR()
    monkeypatch.setattr(utils, "create_ocr_engine", lambda *a, **k: recorder)
    video = V.Video(str(path), None, None)
    x, y, w, h = crop
    video.run_ocr(False, "ch", time_start, time_end, 95, False, 0, 0, 25, 0, x, y, w, h)
    assert len(recorder.frames) == len(video.pred_frames)
    return {round(p.pts_start, 6): f for p, f in zip(video.pred_frames, recorder.frames)}


def _assert_strips_match_run_ocr(monkeypatch, path, crop, time_start="", time_end="", picks=6):
    by_pts = _run_ocr_frames_by_pts(monkeypatch, path, crop, time_start, time_end)
    pts = sorted(by_pts)
    chosen = pts[1::max(1, len(pts) // picks)][:picks]
    strips = OV.grab_ocr_strips(str(path), crop, chosen)
    assert len(strips) == len(chosen)
    for t, strip in zip(chosen, strips):
        assert strip.shape == by_pts[t].shape, f"t={t}"
        assert np.array_equal(strip, by_pts[t]), f"t={t}: strip differs from the OCR pass"


@pytest.mark.parametrize("name, size, pix_fmt, hdr, crop", [
    # no filter graph: decoded as-is, sliced in Python
    ("sdr8_360p", "640x360", "yuv420p", False, (100, 290, 400, 40)),
    # PQ tone map in the graph, crop planned inside it
    ("pq10_720p", "1280x720", "yuv420p10le", True, (100, 640, 1000, 60)),
    # 4:3 decode downscale: the capture refuses the in-graph crop, Python
    # slices. The box's far edges scale to x 1575.75 / y 1041.75, which
    # run_ocr truncates.
    ("sdr10_1440p", "2560x1440", "yuv420p10le", False, (401, 1300, 1700, 89)),
])
def test_grab_ocr_strips_returns_the_pixels_run_ocr_sees(monkeypatch, tmp_path, encoder, name, size, pix_fmt, hdr, crop):
    path = encoder(tmp_path / f"{name}.mp4", size, pix_fmt, hdr=hdr)
    # Otherwise the HDR case would silently exercise the SDR chain.
    assert _stream_is_pq(path) == hdr
    _assert_strips_match_run_ocr(monkeypatch, path, crop)


@pytest.mark.parametrize("crop", [
    (576, 1892, 2688, 108),     # a subtitle band: in-graph crop, no Python downscale
    (400, 200, 3000, 1600),     # 800 rows after decode downscale: run_ocr shrinks it to 720
])
def test_grab_ocr_strips_matches_run_ocr_on_10bit_4k(monkeypatch, tmp_path, encoder, crop):
    path = encoder(tmp_path / "sdr10_4k.mp4", "3840x2160", "yuv420p10le", frames=30)
    _assert_strips_match_run_ocr(monkeypatch, path, crop)


@pytest.mark.needs_media
@pytest.mark.slow
def test_grab_ocr_strips_matches_run_ocr_on_the_real_10bit_4k_reference(monkeypatch, reference_media, detector_truth):
    entry = reference_media.get("xwz")
    if entry is None:
        pytest.skip("xwz reference project not present")
    crop = tuple(detector_truth["xwz"]["files"][entry["video"].name]["crop"])
    _assert_strips_match_run_ocr(monkeypatch, entry["video"], crop, "5:00", "5:02")


# --------------------------------------------------------------------------
# Failures drop frames, they do not raise
# --------------------------------------------------------------------------

def test_a_file_that_cannot_be_opened_yields_no_strips(tmp_path):
    assert OV.grab_ocr_strips(str(tmp_path / "missing.mp4"), (0, 0, 100, 40), [1.0, 2.0]) == []


def _fake_capture_class(fail_seek_at=None, fail_open_decoder=False):
    """640x360 @ 25 fps; frame k is filled with k % 256 so a strip names the
    frame it came from."""

    class FakeCapture:
        def __init__(self, path, use_gpu=True, decode_target_height=None, crop_rect=None):
            self._decoder = crop_rect is not None
            self._next = 0
            self._crop_slice = None

        def __enter__(self):
            if self._decoder and fail_open_decoder:
                raise OSError("decoder refused")
            return self

        def __exit__(self, *exc):
            return False

        def get(self, prop):
            return {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 360,
                    cv2.CAP_PROP_FPS: 25.0, cv2.CAP_PROP_FRAME_COUNT: 2500}.get(prop, 0)

        def set(self, prop, value):
            if value == fail_seek_at:
                import av
                raise av.error.FFmpegError(5, "seek failed")
            self._next = int(value)
            return True

        def read(self):
            frame = np.full((360, 640, 3), self._next % 256, dtype=np.uint8)
            self._next += 1
            return True, frame

    return FakeCapture


def test_a_frame_that_fails_to_seek_is_dropped_and_the_rest_keep_their_order(monkeypatch):
    monkeypatch.setattr(OV, "Capture", _fake_capture_class(fail_seek_at=50))
    # Eight frames over four containers: the failing frame (t=2 -> 50) shares
    # its container with t=6, which must still be read.
    times = [3.0, 1.0, 2.0, 4.0, 8.0, 5.0, 7.0, 6.0]

    strips = OV.grab_ocr_strips("fake.mp4", (0, 300, 640, 40), times)

    assert [int(s[0, 0, 0]) for s in strips] == [75, 25, 100, 200, 125, 175, 150]
    assert all(s.shape == (40, 640, 3) for s in strips)


def test_a_decoder_that_cannot_open_drops_its_frames_instead_of_raising(monkeypatch):
    monkeypatch.setattr(OV, "Capture", _fake_capture_class(fail_open_decoder=True))
    assert OV.grab_ocr_strips("fake.mp4", (0, 300, 640, 40), [1.0, 2.0]) == []
