"""Label phase 1.5 must report progress while it runs.

Phase 1.5 (OCR on every phase-1 detection box) re-reads and OCRs one frame
per text frame, which on 4K takes long enough that a progress bar stuck at
the end of phase 1 looks like a hang. It reports through the same
ProgressTracker weights as the other label phases.
"""
import cv2
import pytest

from videocr.label_scanner import LabelScanner
from videocr.progress import ProgressTracker
from videocr.pyav_adapter import PyAVCapture

LABEL_PHASES = ["label_p1", "label_p1_5", "label_p2", "label_p3", "label_p4"]


class _WholeRegionDetector:
    def predict(self, frame):
        h, w = frame.shape[:2]
        return [{"dt_polys": [[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]]}]


class _NoTextOCR:
    def __init__(self):
        self.calls = 0

    def predict(self, crop):
        self.calls += 1
        return []


def _scanner(path):
    with PyAVCapture(str(path)) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    return LabelScanner(str(path), fps, width, height, num_frames, None, None, None, None)


def test_label_weights_cover_every_phase_and_sum_to_100():
    weights = ProgressTracker()._label_weights
    assert list(weights) == LABEL_PHASES
    assert all(w > 0 for w in weights.values())
    assert sum(weights.values()) == 100


def test_phase15_reports_progress_while_it_runs(synthetic_subtitle_video, monkeypatch):
    scanner = _scanner(synthetic_subtitle_video)
    weights = ProgressTracker()._label_weights

    in_phase15 = {"now": False}
    real_phase15 = LabelScanner._batch_ocr_text_frames
    text_frame_count = {}

    def tracking_phase15(self, text_frames, *args, **kwargs):
        text_frame_count["n"] = len(text_frames)
        in_phase15["now"] = True
        try:
            return real_phase15(self, text_frames, *args, **kwargs)
        finally:
            in_phase15["now"] = False

    monkeypatch.setattr(LabelScanner, "_batch_ocr_text_frames", tracking_phase15)
    # Stop the scan after phase 2: this test is about phases 1 and 1.5.
    monkeypatch.setattr(LabelScanner, "_phase2_group_by_position", lambda self, text_frames, progress=None: [])

    reports = []
    tracker = ProgressTracker(include_dialogue=False, include_labels=True,
                              progress_callback=lambda name, pct: reports.append((in_phase15["now"], name, pct)))
    ocr = _NoTextOCR()
    scanner.scan(_WholeRegionDetector(), ocr, "", "", 0.0, progress=tracker)

    n = text_frame_count["n"]
    assert n >= 5 and ocr.calls == n  # non-vacuous: phase 1.5 did real work

    during = [pct for now, _, pct in reports if now]
    assert len(during) >= n, (
        f"phase 1.5 processed {n} text frames but reported progress {len(during)} times"
    )
    assert all(name == "Extracting labels" for _, name, _ in reports)
    # It continues from where phase 1 ended and finishes at its own weight...
    assert during[-1] == weights["label_p1"] + weights["label_p1_5"]
    assert min(during) >= weights["label_p1"]
    # ...and the whole sequence never goes backwards.
    percents = [pct for _, _, pct in reports]
    assert percents == sorted(percents)
    assert percents[-1] <= 100
