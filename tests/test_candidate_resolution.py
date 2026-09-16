"""Unit tests for the deterministic best-of-N candidate path.

These lock in the two invariants the slow determinism test cannot see:
the candidate path never touches timing, and each batch slot's candidates
resolve against that slot's own subtitle. No GPU, no media, milliseconds.
"""
import pytest

from videocr import video as video_mod
from videocr.models import PredictedFrames
from videocr.video import Video

# A wide, short box, so PredictedFrames does not decide the coords are rotated.
BOX = [[0, 0], [100, 0], [100, 20], [0, 20]]

CONF_THRESHOLD = 95           # 0-100, as run_ocr receives it
CONF_THRESHOLD_PERCENT = 0.95  # 0-1, as _process_batch passes it on


def reading(*words):
    """Build one OCR result in PaddleOCR's pre-3.0.3 format.

    Each word is a (text, confidence) pair; boxes are laid out left to right
    on one line.
    """
    out = []
    for i, (text, conf) in enumerate(words):
        box = [[x + i * 110, y] for x, y in BOX]
        out.append([box, (text, conf)])
    return out


class FakeOCR:
    """Maps each stand-in 'frame' (a plain hashable key) to a fixed reading."""

    def __init__(self, table):
        self.table = table
        self.batches = []

    def ocr(self, frames):
        self.batches.append(list(frames))
        return [self.table[f] for f in frames]

    def predict(self, frames):
        return self.ocr(frames)


@pytest.fixture(autouse=True)
def no_format_conversion(monkeypatch):
    """Pin the OCR result format so the fixtures above are what the code sees."""
    monkeypatch.setattr(video_mod.utils, "needs_conversion", lambda: False)


def make_video(pred_frames):
    v = Video.__new__(Video)
    v.lang = "ch"
    v.pred_frames = list(pred_frames)
    v._ocr_time = 0.0
    return v


def make_target(words, *, start=5, end=999, pts_start=1.25, pts_end=9.75):
    pred = PredictedFrames(start, [reading(*words)], CONF_THRESHOLD_PERCENT, "ch",
                           pts=pts_start)
    pred.end_index = end
    pred.pts_end = pts_end
    return pred


def timing_of(pred):
    return (pred.start_index, pred.end_index, pred.pts_start, pred.pts_end)


def test_winning_resolution_never_touches_timing():
    """A candidate that actually wins replaces text only, never the clock."""
    target = make_target([("你好世界", 0.60)])
    before = timing_of(target)
    v = make_video([target])

    # Three independent frames agree on a different reading: strictly modal.
    ocr = FakeOCR({f"c{i}": reading(("你好新世界", 0.99)) for i in range(3)})
    v._resolve_candidates(ocr, ["c0", "c1", "c2"], target,
                          CONF_THRESHOLD, CONF_THRESHOLD_PERCENT)

    assert target.text == "你好新世界", "the modal candidate should have won"
    assert target.confidence == pytest.approx(0.99)
    assert timing_of(target) == before


def test_empty_candidates_never_blank_out_a_subtitle():
    """An empty reading scores the sentinel 100; it must not win."""
    target = make_target([("你好", 0.60)])
    before = timing_of(target)
    v = make_video([target])

    ocr = FakeOCR({f"c{i}": [] for i in range(3)})
    v._resolve_candidates(ocr, ["c0", "c1", "c2"], target,
                          CONF_THRESHOLD, CONF_THRESHOLD_PERCENT)

    assert target.text == "你好"
    assert target.confidence == pytest.approx(0.60)
    assert timing_of(target) == before


def test_each_batch_slot_resolves_against_its_own_subtitle():
    """Slot i's candidates must reach pred_frames[base + i], nothing else."""
    a = make_target([("第一句", 0.60)])
    b = make_target([("第二句", 0.60)])
    c = make_target([("第三句", 0.60)])
    # A previous batch's subtitle sits in front of this batch's three slots.
    earlier = make_target([("上一批", 0.60)])
    v = make_video([earlier, a, b, c])
    base = 1

    ocr = FakeOCR({
        "b1": reading(("第二句改", 0.99)),
        "b2": reading(("第二句改", 0.99)),
        "c1": reading(("第三句改", 0.99)),
        "c2": reading(("第三句改", 0.99)),
    })
    batch_candidates = [[], ["b1", "b2"], ["c1", "c2"]]
    v._resolve_batch_candidates(ocr, batch_candidates, base,
                                CONF_THRESHOLD, CONF_THRESHOLD_PERCENT)

    assert b.text == "第二句改"
    assert c.text == "第三句改"
    # Neither the empty slot's subtitle nor the earlier batch was touched.
    assert a.text == "第一句"
    assert earlier.text == "上一批"
    # Each slot was OCR'd on its own frames only.
    assert ocr.batches == [["b1", "b2"], ["c1", "c2"]]


def test_agreement_beats_a_much_more_confident_dissenter():
    """Confidence is certainty, not correctness: the majority reading wins.

    The dissenter's lead here is far beyond CANDIDATE_CONFIDENCE_MARGIN, so
    only the agreement rule can be what saves the original - exactly the shape
    that produced the 万 -> 方 regression under highest-confidence-wins.
    """
    target = make_target([("光芒万文", 0.900)])
    v = make_video([target])

    ocr = FakeOCR({
        "c0": reading(("光芒方文", 0.990)),
        "c1": reading(("光芒万文", 0.899)),
        "c2": reading(("光芒万文", 0.898)),
    })
    v._resolve_candidates(ocr, ["c0", "c1", "c2"], target,
                          CONF_THRESHOLD, CONF_THRESHOLD_PERCENT)

    assert target.text == "光芒万文"


def test_agreement_wins_even_when_it_is_less_confident():
    """A modal candidate replaces the original despite scoring lower."""
    target = make_target([("老吴", 0.940)])
    v = make_video([target])

    ocr = FakeOCR({f"c{i}": reading(("老吕", 0.800)) for i in range(3)})
    v._resolve_candidates(ocr, ["c0", "c1", "c2"], target,
                          CONF_THRESHOLD, CONF_THRESHOLD_PERCENT)

    assert target.text == "老吕"
    assert target.confidence == pytest.approx(0.800)


def test_a_lone_dissenter_inside_the_margin_is_refused():
    """With no majority either way, a sub-margin lead is noise, not evidence."""
    target = make_target([("老吴", 0.900)])
    v = make_video([target])

    lead = video_mod.CANDIDATE_CONFIDENCE_MARGIN / 200.0  # half the margin
    ocr = FakeOCR({"c0": reading(("老吳", 0.900 + lead))})
    v._resolve_candidates(ocr, ["c0"], target,
                          CONF_THRESHOLD, CONF_THRESHOLD_PERCENT)

    assert target.text == "老吴"


def test_a_lone_dissenter_beyond_the_margin_is_accepted():
    """A frame that is decisively clearer may still win on its own."""
    target = make_target([("糊掉的", 0.600)])
    v = make_video([target])

    lead = video_mod.CANDIDATE_CONFIDENCE_MARGIN / 50.0  # twice the margin
    ocr = FakeOCR({"c0": reading(("清楚的", 0.600 + lead))})
    v._resolve_candidates(ocr, ["c0"], target,
                          CONF_THRESHOLD, CONF_THRESHOLD_PERCENT)

    assert target.text == "清楚的"


def test_an_uncorroborated_shorter_winner_is_refused():
    """Dropping a word raises the mean for free; one frame's word is not enough.

    The lone dissenter here clears the confidence margin comfortably, so only
    the length guard can be what refuses it.
    """
    target = make_target([("跟温祈墨顺利汇合", 0.60), ("7元", 0.60)])
    v = make_video([target])

    ocr = FakeOCR({"c0": reading(("跟温祈墨顺利汇合", 0.99))})
    v._resolve_candidates(ocr, ["c0"], target,
                          CONF_THRESHOLD, CONF_THRESHOLD_PERCENT)

    assert target.text == "跟温祈墨顺利汇合7元"


def test_a_modal_shorter_winner_is_accepted():
    """Several frames agreeing the trailing glyphs are absent is the evidence.

    This is the OCR-artifact case ('...7元'): one frame hallucinates trailing
    characters, the rest do not see them, and the majority must win.
    """
    target = make_target([("全员已待命", 0.77), ("7元", 0.60)])
    v = make_video([target])

    ocr = FakeOCR({f"c{i}": reading(("全员已待命", 0.95)) for i in range(3)})
    v._resolve_candidates(ocr, ["c0", "c1", "c2"], target,
                          CONF_THRESHOLD, CONF_THRESHOLD_PERCENT)

    assert target.text == "全员已待命"


def test_a_winner_may_still_lengthen_a_subtitle():
    """The length guard is one-sided - it must not block genuine recoveries."""
    target = make_target([("跟温祈墨", 0.60)])
    v = make_video([target])

    ocr = FakeOCR({
        f"c{i}": reading(("跟温祈墨顺利汇合", 0.95)) for i in range(3)
    })
    v._resolve_candidates(ocr, ["c0", "c1", "c2"], target,
                          CONF_THRESHOLD, CONF_THRESHOLD_PERCENT)

    assert target.text == "跟温祈墨顺利汇合"


def test_a_high_confidence_subtitle_is_never_re_ocrd():
    """Above the threshold the candidate path must not spend a single OCR call."""
    target = make_target([("已经很清楚了", 0.99)])
    v = make_video([target])

    ocr = FakeOCR({"c0": reading(("完全不同", 0.99))})
    v._resolve_candidates(ocr, ["c0"], target,
                          CONF_THRESHOLD, CONF_THRESHOLD_PERCENT)

    assert ocr.batches == []
    assert target.text == "已经很清楚了"
