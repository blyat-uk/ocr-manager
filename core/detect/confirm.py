"""OCR-confirmation of one file's brightness threshold.

`core.detect.brightness` measures a threshold and, when it doubts the
reading, flags it: the file lands in FLAGGED and the queue badges it "check
brightness". Most of those doubts are answerable without the user -- mask one
strip the file is known to hold a subtitle on and ask the OCR engine to read
it. If the engine reads text at the folder's own confidence threshold, the
stored value is good enough for a run and the doubt is retired. If it does
not, step the threshold down and ask again; the first step that reads
confidently becomes the file's value.

The ladder (`ladder`)
    `start_value`, then STEP below it, and so on while the next rung would
    stay above FLOOR, and FLOOR itself as the last rung -- so the hard stop is
    always probed even when the start value is not a multiple of STEP (237
    walks 237, 227, ... 147, 140). A start at or below FLOOR has one rung, its
    own. Lower thresholds keep DIMMER pixels (the OCR pass masks to
    min(B,G,R) >= t), which is why the ladder only ever goes down: a reading
    that failed is a reading the mask was too strict for.

The pass rule
    A rung passes when the engine reads non-empty text AND that reading's
    confidence clears `conf_threshold` -- the folder's own
    `FolderSettings.conf_threshold`, on the same 0-100 scale the OCR pass
    compares against (`videocr.video.Video._run_ocr_with_engine` divides it by
    100). The reading itself is `brightness._reading`, which is what the OCR
    pass would emit for that frame: words joined as subtitles are written,
    garbage dropped, and "PaddleOCR returned nothing" scored 0 rather than the
    100 sentinel `PredictedFrames` uses -- so an empty frame can never confirm
    anything.

    A rung whose masked strip does not trip the OCR pass's Laplacian gate
    (`ocr_view.gate_fires`) fails WITHOUT being sent to OCR. The run would
    never OCR that frame either, so the question "would the run read this?"
    is answered by the run's own rule instead of being guessed at.

Two engine calls at most
    The first rung is probed on its own: a stored value that is simply fine
    costs one image and one call, which is the common case. Only when it fails
    do the remaining rungs go out as ONE batch, the way
    `brightness._verify` batches its grid, and the HIGHEST passing rung wins
    (the ladder is descending, so that is the first passing rung in order).
    A brighter threshold lets less burned-in scenery through, so among
    thresholds that read the line equally well the brightest is the safest.

What this is not
    It is not a measurement and it never widens a plateau: it reads one strip
    and answers yes or no. It can lower a file's value, never raise it, and a
    ladder that passes nowhere leaves the value exactly as the detector left
    it -- 140 was not shown to be better than the detector's pick, so the
    user still reviews the file by hand.

    One frame is one frame. A file flagged "dim-text?" because one particular
    line was lost at the pick can pass here on a different, brighter line;
    that is the trade this stage was asked for (2026-09-20), and the detector's
    original doubt stays readable in `evidence["brightness"]["flagged"]`.

No Qt imports (core/detect/ is Qt-free).
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from core.detect import brightness as _brightness
from core.detect import ocr_view
from core.detect.flags import is_cancelled as _is_cancelled

logger = logging.getLogger(__name__)

# How far one rung steps down, and the threshold the ladder stops at.
# Both are policy, not measurements: the brief of 2026-09-20 ("reduce
# brightness by 10 ... a brightness value of 140 is the hard stop").
STEP = 10
FLOOR = 140


@dataclass(frozen=True)
class Rung:
    """One threshold tried.

    threshold: the value masked at.
    gated: the masked strip tripped the OCR pass's Laplacian gate, i.e. the
        run would have OCR'd this frame. False means the rung was never sent
        to the engine.
    text: what the engine read there, as the OCR pass would emit it
        (`brightness._reading`). "" when nothing was read, or not gated.
    confidence: that reading's confidence, 0.0-1.0. 0.0 when text is "".
    passed: `text` is non-empty and confidence * 100 >= conf_threshold.
    """
    threshold: int
    gated: bool
    text: str
    confidence: float
    passed: bool

    def to_evidence(self) -> dict:
        return {"threshold": int(self.threshold), "gated": bool(self.gated), "text": self.text,
                "confidence": float(self.confidence), "passed": bool(self.passed)}


@dataclass(frozen=True)
class ConfirmResult:
    """The outcome of one file's ladder.

    value: the threshold that passed -- the highest one, the ladder being
        descending -- or None when none did. None is not a failure to
        measure: it means the file still needs the user.
    probe_time: the time of the strip probed; None when no strip could be
        read at all (the file would not open, or the cache and the decoder
        both came back empty).
    rungs: one Rung per threshold actually tried, in ladder order. Stops at
        the rung that passed: the ones below it were never probed.
    cancelled: cancel_check fired. Nothing here is to be applied.
    """
    value: int | None
    probe_time: float | None
    rungs: tuple[Rung, ...] = ()
    cancelled: bool = False

    @property
    def confirmed(self) -> bool:
        """The ladder found a threshold the engine reads the strip at."""
        return self.value is not None and not self.cancelled


def ladder(start_value: int) -> list[int]:
    """The thresholds to try for a file whose stored value is `start_value`,
    descending: the value itself, STEP below it while that stays above FLOOR,
    and FLOOR last. A `start_value` at or below FLOOR yields only itself --
    there is nothing below the hard stop to try."""
    start = int(start_value)
    rungs = [start]
    threshold = start
    while threshold - STEP > FLOOR:
        threshold -= STEP
        rungs.append(threshold)
    if FLOOR < start:
        rungs.append(FLOOR)
    return rungs


def _probe(strip: np.ndarray, thresholds: list[int], ocr_engine, conf_threshold: int) -> list[Rung]:
    """One Rung per threshold, in the order given, in ONE engine call.

    Every threshold is masked and gated first; only the gated masks are sent
    to the engine, and a threshold whose mask does not trip the gate reads as
    empty -- exactly as it would in the OCR pass, which never OCRs a frame
    that fails the gate.
    """
    minimum = float(conf_threshold) / 100.0
    masks, gated = [], []
    for threshold in thresholds:
        masked = ocr_view.mask(strip, threshold)
        if ocr_view.gate_fires(masked):
            masks.append(masked)
            gated.append(threshold)
    readings: dict[int, tuple[str, float]] = {}
    if masks:
        for threshold, pred in zip(gated, _brightness._ocr_predict(ocr_engine, masks)):
            readings[threshold] = _brightness._reading(pred)

    # `gated` reports whether the OCR pass's gate fired, NOT whether a reading
    # came back: an engine that returns fewer predictions than it was given
    # images must leave the thresholds it skipped reading as empty, not
    # relabel them as frames the run would have passed over.
    fired = set(gated)
    rungs = []
    for threshold in thresholds:
        text, confidence = readings.get(threshold, ("", 0.0))
        rungs.append(Rung(threshold=int(threshold), gated=threshold in fired, text=text,
                          confidence=float(confidence), passed=bool(text) and confidence >= minimum))
    return rungs


def _first_passing(rungs: list[Rung]) -> int | None:
    for rung in rungs:
        if rung.passed:
            return rung.threshold
    return None


def confirm_brightness(video_path: str, crop_box, probe_time: float | None, start_value: int,
                       conf_threshold: int, ocr_engine, *, strip: np.ndarray | None = None,
                       cancel_check: Callable[[], bool] | None = None) -> ConfirmResult:
    """Walk one file's ladder on one strip and return the threshold that
    reads it confidently, or None.

    `strip` is the strip's pixels when the caller already has them --
    `core.jobs.view_cache` holds the gallery's and the tiles' strips
    losslessly on disk, so a warmed file costs no decode at all. Without it
    the strip is grabbed at `probe_time` exactly as the OCR pass sees it
    (`ocr_view.grab_ocr_strips`); a file that will not open, or a time that
    yields no strip, returns value None with no rungs.

    `ocr_engine` is an OCR engine the caller holds an exclusive lease on for
    the whole call (videocr.engine_registry.lease_ocr_engine) -- engines are
    not thread-safe and every reading here is consumed inside that lease.

    `cancel_check` is polled before each engine call; a cancelled call
    returns cancelled=True and nothing about it is to be applied.
    """
    if probe_time is None:
        return ConfirmResult(value=None, probe_time=None)
    probe_time = float(probe_time)

    if strip is None:
        try:
            grabbed = ocr_view.grab_ocr_strips(video_path, crop_box, [probe_time])
        except ocr_view.FETCH_ERRORS as exc:
            logger.warning("%s: cannot grab the confirm strip (%s: %s)", video_path, type(exc).__name__, exc)
            return ConfirmResult(value=None, probe_time=probe_time)
        strip = grabbed[0] if grabbed else None
    if strip is None or getattr(strip, "size", 0) == 0:
        return ConfirmResult(value=None, probe_time=probe_time)

    thresholds = ladder(start_value)
    if _is_cancelled(cancel_check):
        return ConfirmResult(value=None, probe_time=probe_time, cancelled=True)

    # The stored value on its own first: a value that is simply fine costs one
    # image and one call, which is what most files are.
    rungs = _probe(strip, thresholds[:1], ocr_engine, conf_threshold)
    value = _first_passing(rungs)
    if value is None and thresholds[1:]:
        if _is_cancelled(cancel_check):
            return ConfirmResult(value=None, probe_time=probe_time, cancelled=True)
        rungs += _probe(strip, thresholds[1:], ocr_engine, conf_threshold)
        value = _first_passing(rungs)

    # The rungs below the one that passed were never the file's answer: drop
    # them, so the evidence reads as the ladder that was actually walked.
    if value is not None:
        cut = next(i for i, rung in enumerate(rungs) if rung.passed)
        rungs = rungs[:cut + 1]
    return ConfirmResult(value=value, probe_time=probe_time, rungs=tuple(rungs))


# --------------------------------------------------------------------------
# The evidence record
# --------------------------------------------------------------------------

def record(result: ConfirmResult, *, start_value: int, conf_threshold: int, crop_box) -> dict:
    """`result` as JSON-able data for `evidence["brightness"]["confirm"]`.

    It carries everything `matches` compares, because its job is twofold: it
    is what the Brightness tab can quote, and it is the memo that stops a
    folder re-probing the same failed ladder on every open."""
    return {
        "start_value": int(start_value),
        "conf_threshold": int(conf_threshold),
        "crop_box": [int(v) for v in crop_box],
        "probe_time": None if result.probe_time is None else float(result.probe_time),
        "value": None if result.value is None else int(result.value),
        "rungs": [rung.to_evidence() for rung in result.rungs],
    }


def matches(stored: dict | None, *, start_value: int, conf_threshold: int, crop_box,
            probe_time: float | None) -> bool:
    """Whether `stored` is the record of exactly this probe -- same starting
    value, same confidence threshold, same crop box, same strip.

    The caller uses it to leave a file alone that has been probed already.
    Evidence is a disposable cache that may come back partial or junk (the
    same rule the views read it by), so anything malformed simply does not
    match: probing again is safe, skipping wrongly is not.
    """
    if not isinstance(stored, dict):
        return False
    try:
        if int(stored["start_value"]) != int(start_value):
            return False
        if int(stored["conf_threshold"]) != int(conf_threshold):
            return False
        if [int(v) for v in stored["crop_box"]] != [int(v) for v in crop_box]:
            return False
        recorded = stored["probe_time"]
        if (recorded is None) != (probe_time is None):
            return False
        return recorded is None or float(recorded) == float(probe_time)
    except (KeyError, TypeError, ValueError):
        return False
