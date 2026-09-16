"""Automatic brightness threshold detection.

The OCR pass (videocr/video.py, `_frame_producer`) masks every cropped frame
to the pixels where min(B,G,R) >= t, then only OCRs frames whose masked
centre square trips a Laplacian-variance gate. On sampled subtitle frames OCR
text accuracy is flat across a wide plateau of t, and a brighter t lets less
burned-in scenery (HUDs, sky, highlights) through. But the OCR pass must also
catch each subtitle through its dimmer fade-in/fade-out frames, and short
lines are lost first as t nears the plateau's top -- so the pick keeps a
margin below the top. See docs/superpowers/specs/2026-09-16-ocr-manager-
revamp-design.md section 7.2 and task-4-report.md.

Full detection (`detect_brightness` without `folder_plateau`):

1. Sample 24 frames across the keep ranges (else the file minus its first
   and last 10%), producing each strip exactly as the OCR pass would see it
   before masking -- same capture chain, same crop, same downscale
   (`grab_ocr_strips`). While fewer than 16 hold text, sample up to two more
   interleaved rounds of 24 (see SAMPLE_ROUND_PHASES).
2. Run text detection on the unmasked strips: strips with polygons are text
   strips, strips without are empty strips.
3. `analytic_seed`: Otsu inside the eroded polygons on the min-channel, the
   median over frames of each frame's median glyph level, rounded to 5,
   minus 8.
4. `gate_floor`: the lowest t from which up, >=95% of empty strips stay below
   the OCR pass's Laplacian gate.
5. `verify_with_ocr`: OCR up to 16 text strips at seed +/- 25 step 5 in one
   batch, score each t by agreement with each strip's modal text and by mean
   confidence, take the widest valid run and pick 20 below its top (never
   below its start). A gate floor the pick does not clear is flagged, not
   applied.

Deviation from the task brief, on end-to-end evidence: the brief picked one
step below the top and raised the pick to gate_floor + 5 ("bias high": the
threshold mainly controls clutter). Running the real OCR pass at those picks
lost short subtitle lines on 4 of 13 reference projects, and every loss sat
0-10 below the plateau top; see PICK_BELOW_TOP.

Cheap path (`folder_plateau` given): 6 frames, analytic seed only. A seed
inside the folder plateau is accepted, capped at the plateau's own pick
(top - 20); otherwise the result is flagged "escalate" for the caller to
re-run full detection.

Everything the OCR pass does to a frame is mirrored here from
videocr/video.py rather than approximated, and the constants are imported
from there, so a threshold is tuned on the pixels OCR will actually see. The
mirrors are pinned against the real `Video.run_ocr` by
tests/test_detect_brightness.py.

No Qt imports (core/detect/ is Qt-free).
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import fields as _dataclass_fields

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

from core.config import Config as _Config
from videocr import utils as _videocr_utils
from videocr.models import PredictedFrames
from videocr.pyav_adapter import DECODE_TARGET_HEIGHT, Capture
from videocr.video import MIN_CROP_HEIGHT, MIN_LAPLACIAN_VARIANCE, TARGET_VIDEO_HEIGHT

logger = logging.getLogger(__name__)


def _config_default(name: str, fallback):
    """core.config.Config's own default for `name`, so this module follows the
    app's defaults instead of keeping twins of them."""
    for f in _dataclass_fields(_Config):
        if f.name == name:
            return f.default
    return fallback


# Value reported when nothing could be measured (no crop, no text found). Such
# results are always flagged; the number is only a placeholder the caller
# must not auto-apply.
DEFAULT_BRIGHTNESS = int(_config_default("brightness", 230))
# Language the OCR pass joins words for (no spaces for 'ch').
OCR_LANG = _config_default("ocr_lang", "ch")

# --- Sampling
FULL_SAMPLE_FRAMES = 24
CHEAP_SAMPLE_FRAMES = 6
# Full detection samples 24 frames per round and tops up with further,
# interleaved rounds while fewer than VERIFY_STRIPS strips hold text. Measured
# on the reference corpus, 24 uniform frames held only 2-15 text strips, and
# verifying on 5-8 of them put the plateau's top edge 5-10 levels higher than
# 14-20 strips did (In Search of Gods 252 -> 242, Stay Low Profile 237 -> 232):
# the dimmer subtitles that bound the plateau from above are the ones a small
# sample misses. Each round's slot offset puts its frames between the earlier
# rounds' frames.
SAMPLE_ROUND_PHASES = (0.5, 0.0, 0.25)
SAMPLE_EDGE_EXCLUDE_FRAC = 0.10   # without keep ranges, skip the first/last 10%
SAMPLE_WORKERS = 4                # capture containers opened in parallel

# --- Analytic seed
SEED_ROUND = 5
SEED_MARGIN = 8
SEED_MIN, SEED_MAX = 100, 245
POLY_ERODE_KERNEL = 3             # 1px erosion keeps polygon-edge background out
MIN_GLYPH_REGION_PIXELS = 50      # fewer eroded-polygon pixels than this: no Otsu
IMPLAUSIBLE_SEED = 150            # below this the min-channel model is broken (coloured text)

# --- Gate floor
GATE_QUIET_PERCENT = 95
GATE_FLOOR_MARGIN = 5             # a pick below gate_floor + 5 is flagged "no-clean-threshold"

# --- OCR verification
VERIFY_STRIPS = 16
VERIFY_HALF_WINDOW = 25
VERIFY_STEP = 5
# A threshold is valid when its agreement is within this of the best
# threshold's agreement AND the mean confidence clears MIN_MEAN_CONFIDENCE.
AGREEMENT_TOLERANCE = 0.02
MIN_MEAN_CONFIDENCE = 0.97
# The pick sits this far below the plateau's top (never below its start).
# The plateau is measured on sampled mid-subtitle frames; the OCR pass also
# has to catch short lines through their fades, which are dimmer. Measured
# end to end (real OCR pass, 150 s per project, against the output at the
# hand-tuned value): picks 0-10 below the top lost real lines on Jinwu Guard
# ("什么人" at 242, top 247), Legendary Twins ("好", "芊芊" at 237 and 242, top
# 247), Stay Low Profile ("罢了" at 237, top 237) and Legend of Soldier
# (countdown "10", "1" at 237, top 237); 20 below lost none on any of them.
# See task-4-report.md.
PICK_BELOW_TOP = 20

# --- BrightnessResult.flagged reasons (composed with "+", like core.detect.crop)
FLAG_NEEDS_CROP = "needs-crop"              # no crop box: nothing was measured
FLAG_COLOURED_TEXT = "coloured-text?"       # implausibly low seed; not verified, do not auto-apply
FLAG_NO_CLEAN_THRESHOLD = "no-clean-threshold"  # the pick is below gate_floor + 5 (or no floor exists)
FLAG_ESCALATE = "escalate"                  # cheap path: seed outside the folder plateau (or no text)
FLAG_NO_TEXT = "no-text"                    # no sampled strip had text: nothing to seed from
FLAG_NO_PLATEAU = "no-plateau?"             # OCR verification found no valid run; value is the seed


@dataclass
class BrightnessResult:
    """Outcome of one file's brightness detection.

    value: the threshold to use. When `flagged` contains anything other than
        "no-clean-threshold" the caller must not auto-apply it: for
        "needs-crop" / "no-text" it is DEFAULT_BRIGHTNESS (nothing was
        measured); for "coloured-text?" it is the implausible seed itself;
        for "no-plateau?" it is the unverified seed; for "escalate" it is the
        cheap seed that failed the folder check. An unflagged cheap-path result
        is the file's own seed, capped at the folder plateau's top minus
        PICK_BELOW_TOP. "no-clean-threshold" alone
        is informational: the value is still OCR-verified, but frames without
        subtitles keep clutter that trips the OCR gate at this value (a single
        saturated pixel in the centre square is enough), and raising the value
        past it would cost subtitle text.
    plateau: inclusive (lo, hi) range of thresholds OCR verified as reading
        the same text confidently -- or, on a cheap-path hit, the folder
        plateau the seed was checked against. None when nothing was verified.
    seed: the analytic seed (DEFAULT_BRIGHTNESS when it could not be computed).
    gate_floor: see gate_floor(); None when undefined or not measured.
    flagged: None, or one or more FLAG_* reasons joined by "+". The cheap path
        marks a result for escalation with flagged == "escalate": the caller
        re-runs full detection for that file.
    curve: (t, agreement * mean confidence) for every threshold OCR'd, in
        ascending t; empty when verification did not run.
    """
    value: int
    plateau: tuple[int, int] | None
    seed: int
    gate_floor: int | None
    flagged: str | None
    curve: list[tuple[int, float]]


def _compose_flag(existing: str | None, new: str) -> str:
    if not existing:
        return new
    parts = existing.split("+")
    return existing if new in parts else f"{existing}+{new}"


# --------------------------------------------------------------------------
# Mirrors of the OCR pass (videocr/video.py, Video._frame_producer)
# --------------------------------------------------------------------------

def _ocr_view(frame: np.ndarray) -> np.ndarray:
    """The frame as the OCR pass holds it just before masking.

    Verbatim mirror of the "Downscale for faster OCR" block in
    videocr/video.py `_frame_producer` (the `frame_h > TARGET_VIDEO_HEIGHT`
    branch just above "Apply brightness filter"), including its float
    arithmetic -- int(1142 * (720 / 1142)) is 719, not 720. It lives inline
    there, and videocr/ is off limits to detector work, so it is copied
    rather than shared; test_ocr_view_and_mask_match_what_run_ocr_hands_the_
    engine pins the copy against the real run_ocr.
    """
    frame_h, frame_w = frame.shape[:2]
    if frame_h > TARGET_VIDEO_HEIGHT:
        target_scale = TARGET_VIDEO_HEIGHT / frame_h
        min_scale = MIN_CROP_HEIGHT / frame_h
        scale = max(target_scale, min_scale)
        if scale < 1.0:
            new_h = max(MIN_CROP_HEIGHT, int(frame_h * scale))
            new_w = max(1, int(frame_w * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def _mask(frame: np.ndarray, t: int) -> np.ndarray:
    """The OCR pass's brightness filter (videocr/video.py, "Apply brightness
    filter"): keep pixels whose every channel is >= t."""
    return cv2.bitwise_and(frame, frame, mask=cv2.inRange(frame, (t,) * 3, (255,) * 3))


def _gate_fires(masked: np.ndarray) -> bool:
    """The OCR pass's text gate on an already-masked frame (videocr/video.py,
    "Center square check" / "Use Laplacian variance"): grey, the h x h square
    horizontally centred, Laplacian variance >= MIN_LAPLACIAN_VARIANCE."""
    grey = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape
    center_x_start = (w - h) // 2
    center_x_end = center_x_start + h
    center = grey[:, center_x_start:center_x_end]
    return bool(cv2.Laplacian(center, cv2.CV_64F).var() >= MIN_LAPLACIAN_VARIANCE)


def _ocr_crop_geometry(width: int, height: int, crop_box) -> tuple[int | None, tuple[int, int, int, int]] | None:
    """(decode_target_height, (x0, y0, x1, y1) in decode-output pixels) for
    `crop_box` on a width x height source, exactly as Video.run_ocr derives
    them (the "infer missing crop parameters" / "clamp" / "Scale crop
    coordinates from native to decode resolution" blocks). None when the box
    has no area -- run_ocr would then fall back to the bottom third, which is
    not a crop this detector can tune for."""
    crop_x, crop_y, crop_width, crop_height = crop_box
    inferred_x = 0 if crop_x is None else crop_x
    inferred_y = 0 if crop_y is None else crop_y
    inferred_width = (width - inferred_x) if crop_width is None else crop_width
    inferred_height = (height - inferred_y) if crop_height is None else crop_height
    inferred_x = max(0, min(int(inferred_x), width))
    inferred_y = max(0, min(int(inferred_y), height))
    inferred_width = max(0, min(int(inferred_width), width - inferred_x))
    inferred_height = max(0, min(int(inferred_height), height - inferred_y))
    if inferred_width <= 0 or inferred_height <= 0:
        return None
    x0, y0 = inferred_x, inferred_y
    x1, y1 = inferred_x + inferred_width, inferred_y + inferred_height

    decode_height = DECODE_TARGET_HEIGHT if height > DECODE_TARGET_HEIGHT else None
    if decode_height is not None:
        scale_factor = decode_height / height
        x0, y0 = int(x0 * scale_factor), int(y0 * scale_factor)
        x1, y1 = int(x1 * scale_factor), int(y1 * scale_factor)
    return decode_height, (x0, y0, x1, y1)


def _ocr_predict(ocr_engine, images: list[np.ndarray]) -> list:
    """One OCR batch, called the way Video._process_batch calls it."""
    if _videocr_utils.needs_conversion():
        return list(ocr_engine.predict(images))
    return ocr_engine.ocr(images)


# --------------------------------------------------------------------------
# Frame source
# --------------------------------------------------------------------------

def _parse_time(value) -> float | None:
    """Seconds from a keep-range endpoint: None/'' (open), a number of
    seconds, or "MM:SS" / "H:MM:SS" as videocr.utils.get_frame_index reads
    them."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parts = [float(p) for p in str(value).split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    raise ValueError(f"time {value!r} is not MM:SS or H:MM:SS")


def _keep_spans(duration: float, time_ranges) -> list[tuple[float, float]]:
    default = [(duration * SAMPLE_EDGE_EXCLUDE_FRAC, duration * (1.0 - SAMPLE_EDGE_EXCLUDE_FRAC))]
    if not time_ranges:
        return default
    spans = []
    for pair in time_ranges:
        # (start, end) as FileConfig holds them, or {"start", "end"} as .ocr.json stores them.
        start, end = (pair.get("start"), pair.get("end")) if isinstance(pair, dict) else pair
        s = _parse_time(start)
        e = _parse_time(end)
        s = 0.0 if s is None else max(0.0, min(s, duration))
        e = duration if e is None else max(0.0, min(e, duration))
        if e > s:
            spans.append((s, e))
    return sorted(spans) or default


def sample_times(duration: float, time_ranges, n: int, phase: float = 0.5) -> list[float]:
    """`n` times spread evenly over the keep ranges (proportionally to their
    lengths), or over the file minus its first and last 10% when there are
    none. Ascending. The concatenated spans are cut into n equal slots and
    each time sits `phase` of the way into its slot: the default 0.5 is the
    slot middle, and other phases give a later sampling round times that fall
    between an earlier round's."""
    spans = _keep_spans(duration, time_ranges)
    total = sum(e - s for s, e in spans)
    if n <= 0 or total <= 0:
        return []
    times = []
    for k in range(n):
        pos = (k + phase) / n * total
        for i, (s, e) in enumerate(spans):
            if pos <= e - s or i == len(spans) - 1:
                times.append(s + min(pos, e - s))
                break
            pos -= e - s
    return times


def grab_ocr_strips(video_path: str, crop_box, times: list[float]) -> list[np.ndarray]:
    """Crop strips at `times`, pixel-identical to what the OCR pass hands its
    brightness filter for the same frames: opened through the same `Capture`
    with the same decode_target_height and in-graph crop request as
    Video.run_ocr (so the same HDR tone map, 10-bit conversion and 4K decode
    downscale apply), sliced the same way when the capture could not crop in
    its graph, and downscaled with _ocr_view().

    Why not core.detect.crop.grab_frames(): it grabs a full-width bottom band
    through a different filter order (crop -> scale -> bgr24, or the system
    ffmpeg CLI), so it does not reproduce the OCR pass's pixels. Measured on
    real files: identical on Slay the Gods (1080p 8-bit), but 89% of pixels
    off by up to 13 levels on XWZ (4K 10-bit), and a different frame
    altogether on Jinwu Guard (h264 MKV). See task-4-report.md.

    `times` are in the OCR pass's own position domain (a time maps to frame
    index round(t * fps), as `time_start` does). Results are in the order of
    `times`; a frame that cannot be read is dropped and logged.
    """
    if not times:
        return []
    with Capture(video_path) as probe:
        width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = probe.get(cv2.CAP_PROP_FPS)
    geometry = _ocr_crop_geometry(width, height, crop_box)
    if geometry is None:
        logger.warning("%s: crop box %s has no area", video_path, crop_box)
        return []
    decode_height, (x0, y0, x1, y1) = geometry
    graph_crop = (x0, y0, x1 - x0, y1 - y0)

    order = sorted(range(len(times)), key=lambda i: times[i])
    workers = min(SAMPLE_WORKERS, len(order))
    results: list[np.ndarray | None] = [None] * len(times)

    def work(chunk: list[int]) -> None:
        # Each worker seeks forward through its own ascending share of the
        # times; one container per thread (PyAV releases the GIL to decode).
        last_index, last_strip = None, None
        with Capture(video_path, decode_target_height=decode_height, crop_rect=graph_crop) as cap:
            for i in chunk:
                index = max(0, int(round(times[i] * fps)))
                if index == last_index and last_strip is not None:
                    results[i] = last_strip.copy()
                    continue
                if index > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    logger.warning("%s: could not read a frame at t=%.3f", video_path, times[i])
                    continue
                if not getattr(cap, "_crop_slice", None):
                    frame = frame[y0:y1, x0:x1]
                strip = np.ascontiguousarray(_ocr_view(frame))
                results[i] = strip
                last_index, last_strip = index, strip

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="brightness-grab") as pool:
        list(pool.map(work, [order[k::workers] for k in range(workers)]))
    return [strip for strip in results if strip is not None]


def _video_duration(video_path: str) -> float:
    """Duration as the OCR pass counts it: frame count / fps from Capture."""
    with Capture(video_path) as cap:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
    return frames / fps if fps else 0.0


def _sample_strips(video_path: str, crop_box, time_ranges, n: int, phase: float = 0.5) -> list[np.ndarray]:
    duration = _video_duration(video_path)
    return grab_ocr_strips(video_path, crop_box, sample_times(duration, time_ranges, n, phase))


def _detect_text_polys(det_engine, strips: list[np.ndarray]) -> list[list[np.ndarray]]:
    """Every polygon the detection engine returns, per strip.

    Deliberately NOT filtered by core.detect.crop.DT_SCORE_THRESHOLD (0.9).
    That value was set on 480px-high bands; on a thin subtitle crop the
    same engine scores real subtitles lower -- 0.84-0.98 on One Hundred
    Thousand Years of Qi Refining 298, half of them under 0.9. Filtering
    sorted those into the EMPTY strips, where their saturated glyphs trip the
    gate at every threshold and erase the gate floor.
    """
    if not strips:
        return []
    out = []
    for item in det_engine.predict(strips):
        polys = item.get("dt_polys")
        out.append([] if polys is None else [np.asarray(p) for p in polys])
    return out


# --------------------------------------------------------------------------
# Seed and gate floor
# --------------------------------------------------------------------------

def _glyph_level(strip: np.ndarray, polys) -> float | None:
    """Median min-channel level of one strip's glyph pixels: Otsu inside the
    eroded polygons splits glyph fill from the background they enclose. None
    when the strip offers too little to split."""
    if polys is None or len(polys) == 0:
        return None
    h, w = strip.shape[:2]
    region = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(region, [pts], 255)
    region = cv2.erode(region, np.ones((POLY_ERODE_KERNEL, POLY_ERODE_KERNEL), np.uint8))
    values = strip.min(axis=2)[region > 0]
    if values.size < MIN_GLYPH_REGION_PIXELS:
        return None
    otsu, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    glyph = values[values > otsu]
    if glyph.size == 0:
        return None
    return float(np.median(glyph))


def _round_half_up(value: float, step: int) -> int:
    return int(math.floor(value / step + 0.5)) * step


def analytic_seed(strips: list[np.ndarray], polys_per_strip) -> int:
    """round5(median over frames of each frame's median glyph level) - 8,
    clamped to [100, 245]. Strips without polygons are ignored.

    The MEDIAN across frames is load-bearing: a detector that also boxes a
    burned-in HUD drags that frame's glyph level down to the HUD's, and a low
    quantile across frames followed it there on the reference project that
    has one. Raises ValueError when no strip yields glyph pixels.
    """
    levels = [level for level in (_glyph_level(s, p) for s, p in zip(strips, polys_per_strip))
              if level is not None]
    if not levels:
        raise ValueError("no strip has text polygons with glyph pixels to seed from")
    seed = _round_half_up(float(np.median(levels)), SEED_ROUND) - SEED_MARGIN
    return int(min(SEED_MAX, max(SEED_MIN, seed)))


def gate_floor(empty_strips: list[np.ndarray]) -> int | None:
    """The lowest t such that, at t and at every threshold above it up to 254,
    at least 95% of the text-free strips stay below the OCR pass's Laplacian
    gate -- i.e. clutter no longer makes OCR fire on frames without
    subtitles. None when even t=254 leaves more than 5% tripping, or when
    there are no empty strips to measure.

    "At every threshold above it" matters: masking is not monotone in the
    gate. A smooth bright region is quiet at a low t (kept whole, no edges),
    loud in the middle (cut into hard edges) and quiet again once t clears
    it, so the first quiet threshold from below is meaningless.

    Only the centre square is masked and gated: masking and grey conversion
    are per-pixel and the gate already sees the centre as its own image, so
    cropping first is exact and ~25x cheaper on a subtitle strip.
    """
    if not empty_strips:
        return None
    centres = []
    for strip in empty_strips:
        h, w = strip.shape[:2]
        x0 = (w - h) // 2
        centres.append(strip[:, x0:x0 + h] if 0 <= x0 else strip)
    n = len(centres)
    floor = None
    for t in range(254, 0, -1):
        quiet = sum(not _gate_fires(_mask(c, t)) for c in centres)
        if quiet * 100 < GATE_QUIET_PERCENT * n:
            break
        floor = t
    return floor


# --------------------------------------------------------------------------
# OCR verification
# --------------------------------------------------------------------------

def _spread_pick(items: list, k: int) -> list:
    if len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def _threshold_grid(seed: int) -> list[int]:
    return [t for t in range(seed - VERIFY_HALF_WINDOW, seed + VERIFY_HALF_WINDOW + 1, VERIFY_STEP)
            if 1 <= t <= 255]


def _reading(pred) -> tuple[str, float]:
    """(text, confidence) exactly as the OCR pass would emit this result:
    PredictedFrames joins words into lines the way subtitles are written, and
    drops words under its garbage floor. An empty reading scores 0, not the
    100 sentinel PredictedFrames uses for "nothing detected"."""
    frame = PredictedFrames(0, [pred], 0, OCR_LANG)
    if not frame.lines:
        return "", 0.0
    return frame.text, float(frame.confidence)


def _modal_text(readings: list[tuple[str, float]]) -> str:
    """The reading most thresholds agree on, empty readings included, ties
    broken by mean confidence (an empty reading scores 0) and then by the
    lowest threshold. An empty modal means the strip mostly reads nothing."""
    counts: Counter = Counter()
    conf_sums: dict[str, float] = {}
    for text, conf in readings:
        counts[text] += 1
        conf_sums[text] = conf_sums.get(text, 0.0) + conf
    return max(counts, key=lambda t: (counts[t], conf_sums[t] / counts[t]))


def _similarity(text: str, modal: str) -> float:
    return 1.0 - min(1.0, Levenshtein.distance(text, modal) / len(modal))


def verify_with_ocr(strips: list[np.ndarray], seed: int, ocr_engine
                    ) -> tuple[int, tuple[int, int] | None, list[tuple[int, float]]]:
    """OCR-verify thresholds around `seed` and pick one inside the plateau,
    a safe margin below its top. Returns (value, plateau, curve).

    Up to 16 text strips are masked at every t in seed +/- 25 (step 5, within
    1..255) and OCR'd in ONE batch. For each strip, its modal text is the
    reading most thresholds agree on; a strip whose modal reading is EMPTY
    (a logo or speck read at one threshold, a fade that is mostly
    unreadable) carries no subtitle evidence and is left out. Each t scores:
      agreement = mean over strips of 1 - CER(reading at t, strip's modal text)
      confidence = mean over strips of the reading's confidence (0 if empty)
    A t is valid when agreement >= best agreement - 0.02 and confidence >=
    0.97. A single invalid t with valid thresholds on both sides is bridged:
    masking only ever removes glyph pixels as t rises, so text lost to a
    high threshold stays lost above it, and clutter kept by a low one is
    kept below it -- a lone dip is one frame's speck read as a stroke, not a
    property of t. The widest run of consecutive (bridged) valid thresholds
    is the plateau (ties go to the higher run), and the pick is
    PICK_BELOW_TOP below its top, or its start if that is higher. The curve
    records agreement * confidence per t, unbridged.

    When no t is valid the plateau is None and the value is the seed.
    """
    grid = _threshold_grid(seed)
    chosen = _spread_pick(list(strips), VERIFY_STRIPS)
    if not grid or not chosen:
        return seed, None, [(t, 0.0) for t in grid]

    batch = [_mask(strip, t) for strip in chosen for t in grid]
    results = _ocr_predict(ocr_engine, batch)
    readings = [[_reading(results[i * len(grid) + j]) for j in range(len(grid))]
                for i in range(len(chosen))]

    agreement = [0.0] * len(grid)
    confidence = [0.0] * len(grid)
    counted = 0
    for per_t in readings:
        modal = _modal_text(per_t)
        if not modal:
            continue
        counted += 1
        for j, (text, conf) in enumerate(per_t):
            agreement[j] += _similarity(text, modal)
            confidence[j] += conf
    if counted == 0:
        return seed, None, [(t, 0.0) for t in grid]
    agreement = [a / counted for a in agreement]
    confidence = [c / counted for c in confidence]
    curve = [(t, agreement[j] * confidence[j]) for j, t in enumerate(grid)]

    best = max(agreement)
    measured = [agreement[j] >= best - AGREEMENT_TOLERANCE and confidence[j] >= MIN_MEAN_CONFIDENCE
                for j in range(len(grid))]
    valid = [ok or (0 < j < len(grid) - 1 and measured[j - 1] and measured[j + 1])
             for j, ok in enumerate(measured)]
    runs, current = [], []
    for j, ok in enumerate(valid):
        if ok:
            current.append(j)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        return seed, None, curve

    run = max(reversed(runs), key=len)   # reversed: equal widths go to the higher run
    lo, hi = grid[run[0]], grid[run[-1]]
    return max(lo, hi - PICK_BELOW_TOP), (lo, hi), curve


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def detect_brightness(video_path: str, crop_box, time_ranges, det_engine, ocr_engine,
                      folder_plateau: tuple[int, int] | None = None) -> BrightnessResult:
    """Pick a brightness threshold for one file. See the module docstring.

    `crop_box` is the file's (x, y, w, h) in native pixels, as the OCR pass
    takes it. `time_ranges` are the file's keep ranges as (start, end) pairs
    ("MM:SS" strings, seconds, or None for open ends), or None. Engines are
    passed in (detection-only and full OCR) so a process-wide engine cache
    can supply them.

    With `folder_plateau` this is the cheap path: 6 frames, analytic seed
    only. A seed inside the plateau becomes the value, capped the way full
    detection picks (top - PICK_BELOW_TOP, never below the plateau's start);
    otherwise the result carries flagged == "escalate" and the caller must run
    full detection (call again without `folder_plateau`).
    """
    if crop_box is None:
        return BrightnessResult(DEFAULT_BRIGHTNESS, None, DEFAULT_BRIGHTNESS, None, FLAG_NEEDS_CROP, [])

    cheap = folder_plateau is not None
    strips, polys = [], []
    for phase in SAMPLE_ROUND_PHASES[:1] if cheap else SAMPLE_ROUND_PHASES:
        batch = _sample_strips(video_path, crop_box, time_ranges,
                               CHEAP_SAMPLE_FRAMES if cheap else FULL_SAMPLE_FRAMES, phase)
        strips += batch
        polys += _detect_text_polys(det_engine, batch)
        if sum(1 for p in polys if p) >= VERIFY_STRIPS:
            break
    text_strips = [s for s, p in zip(strips, polys) if p]
    text_polys = [p for p in polys if p]
    empty_strips = [s for s, p in zip(strips, polys) if not p]
    try:
        seed = analytic_seed(text_strips, text_polys)
    except ValueError:
        seed = None

    if cheap:
        lo, hi = folder_plateau
        if seed is not None and lo <= seed <= hi:
            return BrightnessResult(min(seed, max(lo, hi - PICK_BELOW_TOP)), (lo, hi), seed, None, None, [])
        fallback = DEFAULT_BRIGHTNESS if seed is None else seed
        return BrightnessResult(fallback, None, fallback, None, FLAG_ESCALATE, [])

    floor = gate_floor(empty_strips)
    flagged = None if floor is not None else FLAG_NO_CLEAN_THRESHOLD
    if seed is None:
        return BrightnessResult(DEFAULT_BRIGHTNESS, None, DEFAULT_BRIGHTNESS, floor,
                                _compose_flag(flagged, FLAG_NO_TEXT), [])
    if seed < IMPLAUSIBLE_SEED:
        # The min-channel mask erases coloured (e.g. yellow) glyphs at any
        # useful threshold; OCR around a meaningless seed proves nothing.
        return BrightnessResult(seed, None, seed, floor, _compose_flag(flagged, FLAG_COLOURED_TEXT), [])

    value, plateau, curve = verify_with_ocr(text_strips, seed, ocr_engine)
    if plateau is None:
        flagged = _compose_flag(flagged, FLAG_NO_PLATEAU)
    if floor is not None and value < floor + GATE_FLOOR_MARGIN:
        flagged = _compose_flag(flagged, FLAG_NO_CLEAN_THRESHOLD)
    return BrightnessResult(int(value), plateau, seed, floor, flagged, curve)
