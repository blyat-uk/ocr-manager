"""The candidate buffer is bounded, and bounded independently of BATCH_SIZE.

Drives `Video.run_ocr`'s real producer/consumer loop over a synthetic frame
stream and a stand-in OCR engine: no media, no GPU, milliseconds.
"""
import numpy as np
import pytest

from videocr import engine_registry, video as video_mod
from videocr.video import Video

FPS = 25.0
FRAME_SIZE = 64            # 64x64x3 BGR = 12,288 bytes a frame
FRAME_BYTES = FRAME_SIZE * FRAME_SIZE * 3
FRAMES_PER_SUBTITLE = 60   # stride is round(fps/4) = 6, so 8 candidates fit
SUBTITLES = 10
OFFERED = SUBTITLES * 8    # what the producer offers with nothing in its way

# Budget that admits every offered candidate, and one that binds hard.
SLOT_BUDGET_OPEN = FRAME_BYTES * 8
SLOT_BUDGET_TIGHT = FRAME_BYTES * 3

BOX = [[0, 0], [100, 0], [100, 20], [0, 20]]


def _frame(value: int, index: int) -> np.ndarray:
    """A flat frame carrying its own global index in two corner pixels.

    The tag makes each frame individually identifiable in the OCR batches, so
    tests can compare the admitted *set* and not merely its size. Two pixels
    is far below the similarity threshold (0.3% of 4,096 px = 12 px), so
    tagging does not disturb the producer's subtitle segmentation.
    """
    frame = np.full((FRAME_SIZE, FRAME_SIZE, 3), value, dtype=np.uint8)
    frame[0, 0, 0] = index // 256
    frame[0, 0, 1] = index % 256
    return frame


def _tag(frame) -> int:
    return int(frame[0, 0, 0]) * 256 + int(frame[0, 0, 1])


def _stream() -> list:
    """SUBTITLES blocks of near-identical frames. Consecutive blocks differ by
    far more than similar_pixel_threshold, so the producer starts a new
    subtitle at every boundary and treats everything within a block as one."""
    frames = []
    for index in range(SUBTITLES * FRAMES_PER_SUBTITLE):
        block = index // FRAMES_PER_SUBTITLE
        frames.append(_frame(10 if block % 2 == 0 else 240, index))
    return frames


def _is_candidate_tag(tag: int) -> bool:
    """A subtitle's first frame opens a block; everything else is a candidate."""
    return tag % FRAMES_PER_SUBTITLE != 0


class TaggingOCR:
    """Returns one fixed low-confidence reading per frame and records the tag
    of every frame it is handed, in order."""

    def __init__(self):
        self.seen = []

    def ocr(self, frames):
        self.seen.extend(_tag(f) for f in frames)
        return [[[BOX, ("文", 0.5)]] for _ in frames]

    def predict(self, frames):
        return self.ocr(frames)

    @property
    def candidate_tags(self):
        return sorted(t for t in self.seen if _is_candidate_tag(t))


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


# Captured at import, before any test can patch it.
PRODUCTION_SLOT_BUDGET = video_mod.MAX_CANDIDATE_BYTES_PER_SUBTITLE
PRODUCTION_BATCH_SIZE = video_mod.BATCH_SIZE


@pytest.fixture(autouse=True)
def stub_pipeline(monkeypatch):
    """Pin the OCR result format and swap in the fake capture."""
    monkeypatch.setattr(video_mod.utils, "needs_conversion", lambda: False)
    frames = _stream()
    monkeypatch.setattr(video_mod, "Capture",
                        lambda path, **kwargs: FakeCapture(frames))
    return frames


def _run(monkeypatch, slot_budget=PRODUCTION_SLOT_BUDGET,
         batch_size=PRODUCTION_BATCH_SIZE):
    # Always set both, never merely leave them: monkeypatch unwinds at the end
    # of a test, not the end of a call, so a run that skipped this would
    # silently inherit whatever a previous run in the same test installed.
    monkeypatch.setattr(video_mod, "MAX_CANDIDATE_BYTES_PER_SUBTITLE", slot_budget)
    monkeypatch.setattr(video_mod, "BATCH_SIZE", batch_size)

    ocr = TaggingOCR()
    monkeypatch.setattr(video_mod.utils, "create_ocr_engine",
                        lambda *a, **k: ocr)
    # Every call here uses the same (lang, det, rec, gpu) key, but each
    # needs its OWN fresh TaggingOCR to track just that run's frames -- the
    # process-wide engine registry would otherwise hand back a stale engine
    # from an earlier call in this same test session instead of the one
    # just monkeypatched in above.
    engine_registry.reset_registry()

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
    return v, ocr.candidate_tags


BATCH_SIZES = (2, 5, 10, 256)


@pytest.mark.parametrize("slot_budget, expected_per_subtitle", [
    pytest.param(SLOT_BUDGET_TIGHT, 3, id="budget-binds"),
    pytest.param(SLOT_BUDGET_OPEN, 8, id="budget-open"),
])
def test_admitted_candidates_are_identical_across_batch_sizes(
    monkeypatch, slot_budget, expected_per_subtitle
):
    """The rule this whole branch exists to establish: for a fixed video and
    fixed settings the output cannot depend on BATCH_SIZE.

    A running byte total across the batch broke it. Flushes happen every
    BATCH_SIZE subtitles and refill such a total, so a smaller BATCH_SIZE
    refills more often and admits strictly more candidates - measured 80 /
    50 / 25 / 25 at BATCH_SIZE 2 / 5 / 10 / 256 against a 25-frame budget.
    _resolve_candidates can rewrite a subtitle's text, so a different
    admitted set is a different .ass.

    Compares the admitted SET, not its size: two runs could coincidentally
    keep the same number of different frames.
    """
    admitted = {bs: _run(monkeypatch, slot_budget, bs)[1] for bs in BATCH_SIZES}

    reference = admitted[BATCH_SIZES[0]]
    assert len(reference) == SUBTITLES * expected_per_subtitle
    for batch_size, tags in admitted.items():
        assert tags == reference, (
            f"BATCH_SIZE={batch_size} admitted a different candidate set "
            f"({len(tags)} frames vs {len(reference)}); OCR output is a "
            "function of BATCH_SIZE"
        )


def test_budget_is_spent_per_subtitle_not_per_batch(monkeypatch):
    """Every subtitle gets its own allowance, so a stream of N subtitles
    admits N x allowance regardless of how the batches are cut."""
    _, tags = _run(monkeypatch, SLOT_BUDGET_TIGHT, batch_size=256)
    per_subtitle = {}
    for tag in tags:
        per_subtitle.setdefault(tag // FRAMES_PER_SUBTITLE, []).append(tag)
    assert len(per_subtitle) == SUBTITLES
    assert all(len(v) == 3 for v in per_subtitle.values())


def test_all_candidates_are_admitted_under_the_production_budget(monkeypatch):
    """This stream's frames are 12,288 bytes, far under the per-frame
    threshold the production budget implies, so nothing is clipped."""
    v, tags = _run(monkeypatch)
    assert len(v.pred_frames) == SUBTITLES
    assert len(tags) == OFFERED
    assert FRAME_BYTES * video_mod.MAX_CANDIDATES < PRODUCTION_SLOT_BUDGET


def test_candidates_admitted_is_a_pure_function_of_frame_size():
    """The sizing table in videocr/video.py's comment, asserted.

    (pixels, expected candidates) at the production budget. These are the
    geometries the cap does and does not clip; if the budget moves, this
    fails and the comment gets corrected with it.
    """
    budget = video_mod.MAX_CANDIDATE_BYTES_PER_SUBTITLE
    threshold_px = budget // video_mod.MAX_CANDIDATES // 3
    assert threshold_px == 87381

    cases = [
        ((1344, 53), 8),    # the golden / reference crop
        ((1344, 66), 7),    # 13 rows taller
        ((1920, 45), 8),    # a wide, thin band
        ((1920, 46), 7),
        ((1920, 106), 3),   # a two-line subtitle region
        ((1280, 720), 0),   # use_fullframe
    ]
    for (w, h), expected in cases:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        assert video_mod._candidates_admitted(frame) == expected, (
            f"{w}x{h} ({w * h} px)"
        )


def test_the_bound_is_actually_needed():
    """Without a bound the batch holds MAX_CANDIDATES x BATCH_SIZE frames."""
    fullframe_bytes = 1280 * 720 * 3
    unbounded = video_mod.MAX_CANDIDATES * video_mod.BATCH_SIZE * fullframe_bytes
    assert unbounded > 5 * 1024**3
    bounded = video_mod.MAX_CANDIDATE_BYTES_PER_SUBTITLE * video_mod.BATCH_SIZE
    assert bounded == 512 * 1024**2
