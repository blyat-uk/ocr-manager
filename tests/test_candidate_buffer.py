"""The candidate buffer is bounded, and bounded deterministically.

Drives `Video.run_ocr`'s real producer/consumer loop over a synthetic frame
stream and a stand-in OCR engine: no media, no GPU, milliseconds.
"""
import numpy as np
import pytest

from videocr import video as video_mod
from videocr.video import Video

FPS = 25.0
FRAME_SIZE = 64            # 64x64x3 BGR = 12,288 bytes a frame
FRAMES_PER_SUBTITLE = 60   # stride is round(fps/4) = 6, so 8 candidates fit
SUBTITLES = 10

BOX = [[0, 0], [100, 0], [100, 20], [0, 20]]


def _frame(value: int) -> np.ndarray:
    return np.full((FRAME_SIZE, FRAME_SIZE, 3), value, dtype=np.uint8)


def _stream() -> list:
    """SUBTITLES blocks of identical frames. Consecutive blocks differ by far
    more than similar_pixel_threshold, so the producer starts a new subtitle
    at every boundary and treats everything within a block as the same one."""
    frames = []
    for s in range(SUBTITLES):
        frames.extend(_frame(10 if s % 2 == 0 else 240)
                      for _ in range(FRAMES_PER_SUBTITLE))
    return frames


class CountingOCR:
    """Returns one fixed low-confidence reading per frame and records the
    size of every batch it is handed."""

    def __init__(self):
        self.batch_sizes = []

    def ocr(self, frames):
        self.batch_sizes.append(len(frames))
        return [[[BOX, ("文", 0.5)]] for _ in frames]

    def predict(self, frames):
        return self.ocr(frames)


class FakeCapture:
    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_stream_start_time(self):
        return 0.0

    def set(self, prop, value):
        return True

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        frame = self.frames[self.i]
        self.i += 1
        return True, frame

    def get_last_pts(self):
        return (self.i - 1) / FPS


@pytest.fixture(autouse=True)
def stub_pipeline(monkeypatch):
    """Pin the OCR result format and swap in the fake capture/engine."""
    monkeypatch.setattr(video_mod.utils, "needs_conversion", lambda: False)
    frames = _stream()
    monkeypatch.setattr(video_mod, "Capture",
                        lambda path, **kwargs: FakeCapture(frames))
    return frames


def _run(monkeypatch, cap_bytes=None):
    if cap_bytes is not None:
        monkeypatch.setattr(video_mod, "MAX_CANDIDATE_BUFFER_BYTES", cap_bytes)

    ocr = CountingOCR()
    monkeypatch.setattr(video_mod.utils, "create_ocr_engine",
                        lambda *a, **k: ocr)

    v = Video.__new__(Video)
    v.path = "synthetic"
    v.det_model_dir = v.rec_model_dir = None
    v.fps = FPS
    v.num_frames = SUBTITLES * FRAMES_PER_SUBTITLE
    v.height = v.width = FRAME_SIZE
    v.run_ocr(
        use_gpu=False, lang="ch", time_start="0:00", time_end="",
        conf_threshold=95, use_fullframe=True, brightness_threshold=0,
        similar_image_threshold=0.3, similar_pixel_threshold=25,
        frames_to_skip=0, crop_x=None, crop_y=None,
        crop_width=None, crop_height=None,
    )
    # The first batch is the subtitles themselves; every later batch is one
    # slot's candidates, resolved because the stand-in reads at 0.5 (<0.95).
    return v, sum(ocr.batch_sizes[1:])


def test_all_candidates_are_buffered_under_the_cap(monkeypatch):
    """The production cap must not clip an ordinary run: this stream offers
    the full MAX_CANDIDATES for every one of its subtitles."""
    v, candidates = _run(monkeypatch)
    assert len(v.pred_frames) == SUBTITLES
    assert candidates == SUBTITLES * video_mod.MAX_CANDIDATES


def test_candidate_buffer_respects_the_byte_budget(monkeypatch):
    """Past the budget, candidates are dropped rather than buffered."""
    frame_bytes = FRAME_SIZE * FRAME_SIZE * 3
    admitted = 25
    _, candidates = _run(monkeypatch, cap_bytes=frame_bytes * admitted)
    assert candidates == admitted


def test_candidate_admission_is_deterministic(monkeypatch):
    """Same stream, same cap, same decisions -- twice over, and the clipped
    run must still produce the same subtitles as the unclipped one (these
    candidates all read identically, so dropping some changes nothing)."""
    frame_bytes = FRAME_SIZE * FRAME_SIZE * 3
    runs = [_run(monkeypatch, cap_bytes=frame_bytes * 25) for _ in range(2)]
    assert runs[0][1] == runs[1][1]
    texts = [[f.text for f in v.pred_frames] for v, _ in runs]
    assert texts[0] == texts[1]

    full, _ = _run(monkeypatch)
    assert [f.text for f in full.pred_frames] == texts[0]


def test_candidate_fits_is_a_pure_function_of_bytes():
    frame = _frame(0)
    cap = video_mod.MAX_CANDIDATE_BUFFER_BYTES
    assert video_mod._candidate_fits(0, frame)
    assert video_mod._candidate_fits(cap - frame.nbytes, frame)
    assert not video_mod._candidate_fits(cap - frame.nbytes + 1, frame)


def test_reference_crop_workload_is_entirely_under_the_cap():
    """The cap is chosen to sit above the worst case of the geometry the
    golden cases (and production) use, so it can never alter their output."""
    crop_frame_bytes = 1344 * 53 * 3
    worst_case = video_mod.MAX_CANDIDATES * video_mod.BATCH_SIZE * crop_frame_bytes
    assert worst_case <= video_mod.MAX_CANDIDATE_BUFFER_BYTES

    fullframe_bytes = 1280 * 720 * 3
    unbounded = video_mod.MAX_CANDIDATES * video_mod.BATCH_SIZE * fullframe_bytes
    assert unbounded > 5 * 1024**3, "the bound this test guards is not needed"
    assert video_mod.MAX_CANDIDATE_BUFFER_BYTES < unbounded
