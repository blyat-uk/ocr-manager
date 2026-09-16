"""Automatic brightness threshold detection.

The OCR pass (videocr/video.py, `_frame_producer`) masks every cropped frame
to the pixels where min(B,G,R) >= t, then only OCRs frames whose masked
centre square trips a Laplacian-variance gate (mirrored in
core.detect.ocr_view). On sampled subtitle frames OCR text accuracy is flat
across a wide plateau of t, and a brighter t lets less burned-in scenery
(HUDs, sky, highlights) through. But the OCR pass must also catch each
subtitle through its dimmer fade frames, and short lines are lost first as t
nears the plateau's top -- so the pick keeps a margin below the top. See
docs/superpowers/specs/2026-09-16-ocr-manager-revamp-design.md section 7.2
and task-4-report.md.

Full detection (`detect_brightness` without `folder_plateau`):

1. Sample 24 frames across the keep ranges (else the file minus its first
   and last 10%), each strip exactly as the OCR pass would see it before
   masking (`ocr_view.grab_ocr_strips`). While fewer than 16 hold text,
   sample up to two more interleaved rounds of 24 (SAMPLE_ROUND_PHASES).
2. Run text detection on the unmasked strips: strips with polygons are text
   strips, strips without are empty strips.
3. `analytic_seed`: Otsu inside the eroded polygons on the min-channel, the
   median over frames of each frame's median glyph level, rounded to 5,
   minus 8.
4. `gate_floor`: the lowest t from which up, >=95% of empty strips stay below
   the OCR pass's Laplacian gate.
5. `verify_with_ocr`: OCR up to 16 text strips at seed +/- 25 step 5 in one
   batch (a masked strip that does not trip the gate reads as empty, as in
   the OCR pass), score each t by agreement with each strip's modal text and
   by mean confidence, take the widest valid run and pick 20 below its top
   (never below its start). A gate floor the pick does not clear is flagged,
   not applied.

Deviations from the task brief, on end-to-end evidence (task-4-report.md):
the brief picked one step below the top and raised the pick to
gate_floor + 5 ("bias high"). Running the real OCR pass at those picks lost
short subtitle lines on 4 of 13 reference projects, every loss 0-10 below the
plateau top; see PICK_BELOW_TOP.

Dim-text check: the most complete reading of a line must be readable at the
pick, on its own frame or a nearby one. For every verified text strip --
evidence or not -- its line is its reading at the lowest threshold where it
reads anything (or, when it reads nowhere in the window, its reading at its
own seed threshold). When the strip's reading at the pick is not that same
line (_same_line), it was caught mid-fade -- lost at any threshold,
including a hand-tuned one -- or it is a line the pick loses for good: a
dimmer style, a short line, or a fragment of it that won the modal. Its
neighbour frames (+/- 0.4 and 0.8 s), masked at the pick, tell them apart:
if any reads the same line, the line survives; otherwise the result is
flagged "dim-text?". The check only adds a flag; it never moves the value.

Cheap path (`folder_plateau` given): 6 frames, analytic seed only. A seed
inside the folder plateau is accepted and picked PICK_BELOW_TOP below itself
(a file's own seed sits at or just under its own plateau top), never below
the plateau's start; otherwise the result is flagged "escalate" for the
caller to re-run full detection.

Which results a caller may apply without review: `BrightnessResult.
auto_applicable`. No Qt imports (core/detect/ is Qt-free).
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import fields as _dataclass_fields

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

from core.config import Config as _Config
from core.detect import ocr_view
from videocr import utils as _videocr_utils
from videocr.models import PredictedFrames

logger = logging.getLogger(__name__)


def _config_default(name: str, fallback):
    """core.config.Config's own default for `name`, so this module follows the
    app's defaults instead of keeping twins of them."""
    for f in _dataclass_fields(_Config):
        if f.name == name:
            return f.default
    return fallback


# Value reported when nothing could be measured (no crop, no text, cancelled).
# Such results are always flagged and never auto-applicable.
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
# Fewer strips than this that verification could use as evidence (see
# _modal_text): the plateau's top is not trustworthy (see above), so the result
# is flagged "thin-evidence?".
THIN_EVIDENCE_STRIPS = 8

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
# A strip is evidence only when its modal reading is supported by this many
# thresholds' readings, counting near matches (_near_reading): a logo or speck
# read once ('A', ':', 'MX' on the reference corpus) is noise, while a dim
# line readable at two or three of the lowest thresholds is exactly the
# evidence that must shrink the plateau -- even though an eroding line rarely
# reads the same string twice (CrossFire: 苍蝇 at 220, 苍绳 at 225; Body
# Refining: 才也会有半分心动 then 半分).
MIN_READING_REPEATS = 2
# The pick sits this far below the plateau's top (never below its start).
# The plateau is measured on sampled mid-subtitle frames; the OCR pass also
# has to catch short lines through their fades, which are dimmer. Measured
# end to end (real OCR pass, 150 s per project, against the output at the
# hand-tuned value): picks 0-10 below the top lost real lines on Jinwu Guard
# ("什么人" at 242, top 247), Legendary Twins ("好", "芊芊" at 237 and 242, top
# 247), Stay Low Profile ("罢了" at 237, top 237) and Legend of Soldier
# (countdown "10", "1" at 237, top 237); 20 below lost none on any of them.
# A plateau narrower than this cannot hold that margin: "narrow-plateau?".
PICK_BELOW_TOP = 20

# --- BrightnessResult.flagged reasons, composed with "+" (like core.detect.crop).
# Only FLAG_NO_CLEAN_THRESHOLD on its own leaves a result auto-applicable.
FLAG_NO_CLEAN_THRESHOLD = "no-clean-threshold"  # informational: empty frames trip the gate at the pick
FLAG_NEEDS_CROP = "needs-crop"              # no crop box: nothing measured
FLAG_RANGES_EMPTY = "ranges-empty?"         # keep ranges select nothing in the file: nothing measured
FLAG_NO_TEXT = "no-text"                    # no sampled strip had text: nothing to seed from
FLAG_THIN_EVIDENCE = "thin-evidence?"       # fewer than THIN_EVIDENCE_STRIPS strips usable as evidence
FLAG_COLOURED_TEXT = "coloured-text?"       # implausibly low seed; not verified
FLAG_NO_PLATEAU = "no-plateau?"             # OCR verification found no valid run; value is the seed
FLAG_NARROW_PLATEAU = "narrow-plateau?"     # plateau narrower than PICK_BELOW_TOP: no safe margin
FLAG_DIM_TEXT = "dim-text?"                 # a strip's most complete line is not read at the pick, nor nearby
FLAG_ESCALATE = "escalate"                  # cheap path: seed outside the folder plateau (or no text)
FLAG_CANCELLED = "cancelled"                # cancel_check fired: result incomplete

# --- Dim-text check
# Neighbour frames checked around a strip whose line the pick loses, in seconds.
NEIGHBOUR_OFFSETS_SEC = (-0.8, -0.4, 0.4, 0.8)
# Neighbour frames fetched per grab; cancel_check is polled before each, so a
# cancel waits for at most one grab (~1.5 s at 4K). 8 gives each of the
# SAMPLE_WORKERS containers two frames: 16 frames of XWZ 170 (4K 10-bit,
# load 12-21) took 3.0-3.2 s in grabs of 8, 2.9-3.0 s in one grab and
# 3.4-3.5 s in grabs of 4.
NEIGHBOUR_FETCH_CHUNK = 8


@dataclass
class BrightnessResult:
    """Outcome of one file's brightness detection.

    value: the threshold. Use it without review only when `auto_applicable`.
        For "needs-crop" / "ranges-empty?" / "no-text" / "cancelled" it is
        DEFAULT_BRIGHTNESS (nothing measured); for "coloured-text?" and
        "no-plateau?" it is the unverified seed; for "escalate" it is the
        cheap seed that failed the folder check.
    plateau: inclusive (lo, hi) range of thresholds OCR verified as reading
        the same text confidently -- or, on a cheap-path hit, the folder
        plateau the seed was checked against. None when nothing was verified.
    seed: the analytic seed (DEFAULT_BRIGHTNESS when it could not be computed).
    gate_floor: see gate_floor(). None with FLAG_NO_CLEAN_THRESHOLD: measured,
        and no threshold keeps the empty frames quiet. None without that flag:
        not measured (no empty strips, or the path never got that far).
    flagged: None, or FLAG_* reasons joined by "+". The cheap path marks a
        result for escalation with flagged == "escalate": the caller re-runs
        full detection for that file.
    curve: (t, agreement * mean confidence) for every threshold verified, in
        ascending t; empty when verification did not run.
    """
    value: int
    plateau: tuple[int, int] | None
    seed: int
    gate_floor: int | None
    flagged: str | None
    curve: list[tuple[int, float]]

    @property
    def auto_applicable(self) -> bool:
        """True only for a clean result, or one whose sole flag is the
        informational "no-clean-threshold" (the value is OCR-verified with its
        safety margin; frames without subtitles merely keep clutter that
        trips the gate). Every other flag -- alone or composed -- means the
        value was not measured, not verified, or not safe: needs-crop,
        ranges-empty?, no-text, thin-evidence?, coloured-text?, no-plateau?,
        narrow-plateau?, dim-text?, escalate, cancelled."""
        return self.flagged is None or self.flagged == FLAG_NO_CLEAN_THRESHOLD


def _compose_flag(existing: str | None, new: str) -> str:
    if not existing:
        return new
    parts = existing.split("+")
    return existing if new in parts else f"{existing}+{new}"


def _is_cancelled(cancel_check: Callable[[], bool] | None) -> bool:
    return cancel_check is not None and bool(cancel_check())


class _Cancelled(Exception):
    """cancel_check fired inside a step of full detection; detect_brightness
    returns its "cancelled" result."""


def _ocr_predict(ocr_engine, images: list[np.ndarray]) -> list:
    """One OCR batch, called the way Video._process_batch calls it."""
    if _videocr_utils.needs_conversion():
        return list(ocr_engine.predict(images))
    return ocr_engine.ocr(images)


# --------------------------------------------------------------------------
# Frame source and detection
# --------------------------------------------------------------------------

def _ranges_select_nothing(video_path: str, time_ranges) -> bool:
    """True when keep ranges were given but cover no part of the file. A file
    that cannot be opened is not judged here: sampling then yields no strips
    and the no-text path reports it."""
    if not time_ranges:
        return False
    try:
        duration = ocr_view.video_duration(video_path)
    except ocr_view.FETCH_ERRORS:
        return False
    return duration > 0 and not ocr_view.keep_spans(duration, time_ranges)


def _sample_strips(video_path: str, crop_box, time_ranges, n: int,
                   phase: float = 0.5) -> list[tuple[float, np.ndarray]]:
    """(time, strip) pairs for one sampling round."""
    try:
        duration = ocr_view.video_duration(video_path)
    except ocr_view.FETCH_ERRORS as exc:
        logger.warning("%s: cannot open (%s: %s)", video_path, type(exc).__name__, exc)
        return []
    return ocr_view.grab_ocr_strips_at(video_path, crop_box, ocr_view.sample_times(duration, time_ranges, n, phase))


def _neighbour_times(t: float, duration: float, fps: float, spans) -> list[float]:
    """Times NEIGHBOUR_OFFSETS_SEC around `t`, clamped to the file (its last
    frame) and, when keep-range `spans` are given, to the span holding `t`.
    Times that land on t's own frame or repeat another neighbour's frame
    after clamping are dropped."""
    lo = 0.0
    hi = max(0.0, duration - 1.0 / fps) if fps else duration
    if spans:
        s, e = min(spans, key=lambda span: 0.0 if span[0] <= t <= span[1] else min(abs(t - span[0]), abs(t - span[1])))
        lo, hi = max(lo, s), min(hi, e)
    frame = (lambda x: round(x * fps)) if fps else (lambda x: x)
    seen = {frame(t)}
    out = []
    for offset in NEIGHBOUR_OFFSETS_SEC:
        nt = min(max(t + offset, lo), hi)
        if frame(nt) not in seen:
            seen.add(frame(nt))
            out.append(nt)
    return out


def _neighbour_strips(video_path: str, crop_box, time_ranges, centres: list[float],
                      cancel_check: Callable[[], bool] | None = None
                      ) -> dict[float, list[tuple[float, np.ndarray]]]:
    """(time, strip) neighbours (see _neighbour_times) for each centre time,
    clamped to the keep ranges, fetched through the OCR pass's own capture
    chain NEIGHBOUR_FETCH_CHUNK frames at a time. A frame that cannot be read
    is simply missing. Raises _Cancelled when cancel_check fires before a
    grab."""
    try:
        duration, fps = ocr_view.video_timing(video_path)
    except ocr_view.FETCH_ERRORS as exc:
        logger.warning("%s: cannot open for neighbour frames (%s: %s)", video_path, type(exc).__name__, exc)
        return {t: [] for t in centres}
    spans = ocr_view.keep_spans(duration, time_ranges) if time_ranges else None
    wanted = {t: _neighbour_times(t, duration, fps, spans) for t in centres}
    ordered = sorted({nt for ts in wanted.values() for nt in ts})
    fetched = {}
    for k in range(0, len(ordered), NEIGHBOUR_FETCH_CHUNK):
        if _is_cancelled(cancel_check):
            raise _Cancelled
        fetched.update(ocr_view.grab_ocr_strips_at(video_path, crop_box, ordered[k:k + NEIGHBOUR_FETCH_CHUNK]))
    return {t: [(nt, fetched[nt]) for nt in ts if nt in fetched] for t, ts in wanted.items()}


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


def _seed_from_level(level: float) -> int:
    seed = _round_half_up(level, SEED_ROUND) - SEED_MARGIN
    return int(min(SEED_MAX, max(SEED_MIN, seed)))


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
    return _seed_from_level(float(np.median(levels)))


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
        quiet = sum(not ocr_view.gate_fires(ocr_view.mask(c, t)) for c in centres)
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


def _is_subsequence(short: str, long: str) -> bool:
    remaining = iter(long)
    return all(ch in remaining for ch in short)


def _near_reading(a: str, b: str) -> bool:
    """Two readings of the same line: at most one edit per three characters of
    the longer (3 * Levenshtein <= its length), or the shorter -- at least two
    characters -- is the longer with characters dropped (erosion removes
    glyphs: 才也会有半分心动 -> 半分). Symmetric. Single-character readings
    only match themselves, so one-character junk never supports anything.
    For modal support within one strip only; across frames the dim-text
    check uses the stricter _same_line."""
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if 3 * Levenshtein.distance(short, long) <= len(long):
        return True
    return len(short) >= 2 and _is_subsequence(short, long)


def _same_line(a: str, b: str) -> bool:
    """Whether two readings show the same subtitle line, for the dim-text
    check: 3 * Levenshtein(a, b) <= max(len(a), len(b)) -- at most one edit
    per three characters of the longer -- after removing spaces, as the OCR
    pass does when it compares subtitle texts (PredictedSubtitle.
    is_similar_to; a no-op on 'ch' readings, which PredictedFrames already
    joins without spaces). Newlines stay, as they do there: a clutter line is
    a real difference, so "你好世界\n1" is still 你好世界 (2 edits in 6) but
    "你好世界\n12:30" is not (6 in 10). An empty reading never matches.

    Deliberately stricter than _near_reading, which also accepts a shorter
    reading contained in order in the longer: that suits one strip eroding
    across thresholds, but across frames it pairs different lines (师父 and
    是我害了师父们; 叶辰 and 他自称叶辰 on the reference corpus) and a line
    with its own fragment (样式 of 旁白第二种样式文本)."""
    a, b = a.replace(" ", ""), b.replace(" ", "")
    if not a or not b:
        return False
    return 3 * Levenshtein.distance(a, b) <= max(len(a), len(b))


def _modal_text(readings: list[tuple[str, float]]) -> str | None:
    """The strip's modal text over its NON-EMPTY readings -- or None when the
    strip is no evidence at all.

    The modal is the distinct reading with the most support, where a
    threshold's reading supports a candidate when it is a near match
    (_near_reading) -- an eroding line reads slightly differently at each of
    its last thresholds, and exact repeats would drop it. Ties go to the most
    exact repeats, then the higher mean confidence, then the reading seen at
    the lowest threshold.

    None when:
    - the modal's support is under MIN_READING_REPEATS (a logo or speck OCR'd
      once), or
    - the readable thresholds do not form one band: a subtitle's legibility
      runs unbroken between where clutter stops drowning it and where erosion
      starts eating it, so readings scattered across the grid (two or more
      unreadable thresholds between readable ones, or more than one hole) are
      noise. A single one-threshold hole is tolerated.

    Empty readings of an evidential strip are disagreement: a dim line
    readable only at the lowest thresholds shrinks the plateau to them.
    """
    readable = [j for j, (text, _) in enumerate(readings) if text]
    if not readable:
        return None
    texts = [readings[j][0] for j in readable]
    first_seen: dict[str, int] = {}
    conf_sums: dict[str, float] = {}
    exact = Counter(texts)
    for j in readable:
        text, conf = readings[j]
        first_seen.setdefault(text, j)
        conf_sums[text] = conf_sums.get(text, 0.0) + conf
    support = {c: sum(_near_reading(t, c) for t in texts) for c in exact}
    modal = max(exact, key=lambda c: (support[c], exact[c], conf_sums[c] / exact[c], -first_seen[c]))
    if support[modal] < MIN_READING_REPEATS:
        return None
    holes = [b - a - 1 for a, b in zip(readable, readable[1:]) if b - a > 1]
    if any(h > 1 for h in holes) or len(holes) > 1:
        return None
    return modal


def _similarity(text: str, modal: str) -> float:
    return 1.0 - min(1.0, Levenshtein.distance(text, modal) / len(modal))


def _widest_run(valid: list[bool]) -> tuple[int, int] | None:
    """(first, last) index of the widest run of valid thresholds, where a run
    may bridge ONE isolated invalid threshold between two valid stretches:
    masking only ever removes glyph pixels as t rises, so text lost to a high
    threshold stays lost above it and clutter kept by a low one is kept below
    it -- one lone dip is a speck read as a stroke, but repeated dips are not
    noise. Equal widths go to the higher run."""
    stretches, start = [], None
    for j, ok in enumerate(valid + [False]):
        if ok and start is None:
            start = j
        elif not ok and start is not None:
            stretches.append((start, j - 1))
            start = None
    candidates = list(stretches)
    for (a1, b1), (a2, b2) in zip(stretches, stretches[1:]):
        if a2 - b1 == 2:
            candidates.append((a1, b2))
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r[1] - r[0], r[1]))


@dataclass
class _Verification:
    value: int
    plateau: tuple[int, int] | None
    curve: list[tuple[int, float]]
    grid: list[int]
    chosen: list[int]                 # indices into the strips verified
    modals: list[str | None]          # per chosen strip; None = not evidence
    readable: list[list[int]]         # per chosen strip: grid indices that read text
    readings: list[list[tuple[str, float]]]  # per chosen strip, per grid threshold

    @property
    def evidence(self) -> int:
        return sum(m is not None for m in self.modals)


def _verify(strips: list[np.ndarray], seed: int, ocr_engine) -> _Verification:
    grid = _threshold_grid(seed)
    picks = _spread_pick(list(range(len(strips))), VERIFY_STRIPS)
    if not grid or not picks:
        return _Verification(seed, None, [(t, 0.0) for t in grid], grid, picks, [None] * len(picks),
                             [[] for _ in picks], [[] for _ in picks])

    empty = ("", 0.0)
    readings = [[empty] * len(grid) for _ in picks]
    batch, slots = [], []
    for i, index in enumerate(picks):
        for j, t in enumerate(grid):
            masked = ocr_view.mask(strips[index], t)
            if ocr_view.gate_fires(masked):
                batch.append(masked)
                slots.append((i, j))
    if batch:
        for (i, j), pred in zip(slots, _ocr_predict(ocr_engine, batch)):
            readings[i][j] = _reading(pred)

    modals = [_modal_text(per_t) for per_t in readings]
    readable = [[j for j, (text, _) in enumerate(per_t) if text] for per_t in readings]

    def result(value, plateau, curve):
        return _Verification(value, plateau, curve, grid, picks, modals, readable, readings)

    agreement = [0.0] * len(grid)
    confidence = [0.0] * len(grid)
    counted = 0
    for per_t, modal in zip(readings, modals):
        if modal is None:
            continue
        counted += 1
        for j, (text, conf) in enumerate(per_t):
            agreement[j] += _similarity(text, modal)
            confidence[j] += conf
    if counted == 0:
        return result(seed, None, [(t, 0.0) for t in grid])
    agreement = [a / counted for a in agreement]
    confidence = [c / counted for c in confidence]
    curve = [(t, agreement[j] * confidence[j]) for j, t in enumerate(grid)]

    best = max(agreement)
    valid = [agreement[j] >= best - AGREEMENT_TOLERANCE and confidence[j] >= MIN_MEAN_CONFIDENCE
             for j in range(len(grid))]
    run = _widest_run(valid)
    if run is None:
        return result(seed, None, curve)
    lo, hi = grid[run[0]], grid[run[1]]
    return result(max(lo, hi - PICK_BELOW_TOP), (lo, hi), curve)


def verify_with_ocr(strips: list[np.ndarray], seed: int, ocr_engine
                    ) -> tuple[int, tuple[int, int] | None, list[tuple[int, float]]]:
    """OCR-verify thresholds around `seed` and pick one inside the plateau,
    a safe margin below its top. Returns (value, plateau, curve).

    Up to 16 text strips are masked at every t in seed +/- 25 (step 5, within
    1..255). A masked strip that does not trip the OCR pass's gate is never
    OCR'd there, so it reads as empty here; the rest are OCR'd in ONE batch.
    Each strip's modal text and whether it counts as evidence: _modal_text().
    Each t scores:
      agreement = mean over evidential strips of 1 - CER(reading at t, the
                  strip's modal text)
      confidence = mean over evidential strips of the reading's confidence
                  (0 if empty)
    A t is valid when agreement >= best agreement - 0.02 and confidence >=
    0.97. The plateau is the widest run of valid thresholds (_widest_run()),
    and the pick is PICK_BELOW_TOP below its top, or its start if that is
    higher. The curve records agreement * confidence per t.

    When no t is valid the plateau is None and the value is the seed.
    """
    v = _verify(strips, seed, ocr_engine)
    return v.value, v.plateau, v.curve


def _dim_text_check(video_path: str, crop_box, time_ranges, strips: list[np.ndarray], times: list[float],
                    polys_per_strip, verification: _Verification, ocr_engine,
                    cancel_check: Callable[[], bool] | None = None) -> list[dict]:
    """Check that every verified strip's most complete line is read at the pick.

    For each strip verification OCR'd (evidence or not):
    - its line is its reading at the lowest grid threshold that reads
      anything. A strip that reads nowhere in the window is masked at its own
      threshold (the seed formula applied to that strip alone) and, when that
      trips the gate, re-read there -- all such strips in ONE OCR batch. A
      strip still without a reading carries no text and is dropped.
    - its reading at the pick is verification's own (gated) reading there.
    - when that is _same_line as its line, the line survives on its own frame.
    - otherwise the strip is a candidate: its neighbour frames are fetched,
      masked at the PICK and, where they trip the gate, OCR'd -- all
      candidates' neighbours in ONE batch. The line survives when any
      neighbour reads the same line.

    cancel_check is polled before each OCR batch and each neighbour grab;
    raises _Cancelled when it fires.

    Returns one record per strip with a line: {index, time, threshold (where
    the line was read), text (the line), at_pick, candidate, neighbours:
    [(time, text read at the pick)], survives}. A record that does not
    survive is a line the pick loses.
    """
    grid, pick = verification.grid, verification.value
    # The pick is always a grid threshold (the seed, or a plateau edge or
    # 20 below its top); were it not, nothing would count as read there.
    pick_j = grid.index(pick) if pick in grid else None
    records, rereads = [], []
    for k, index in enumerate(verification.chosen):
        per_t = verification.readings[k]
        record = dict(index=index, time=times[index], threshold=None, text="", at_pick="",
                      candidate=False, neighbours=[], survives=True)
        if pick_j is not None and per_t:
            record["at_pick"] = per_t[pick_j][0]
        lowest = next((j for j, (text, _) in enumerate(per_t) if text), None)
        if lowest is not None:
            record.update(threshold=grid[lowest], text=per_t[lowest][0])
            records.append(record)
            continue
        level = _glyph_level(strips[index], polys_per_strip[index])
        if level is None:
            continue
        own = _seed_from_level(level)
        masked = ocr_view.mask(strips[index], own)
        if ocr_view.gate_fires(masked):
            record["threshold"] = own
            records.append(record)
            rereads.append((record, masked))
    if rereads:
        if _is_cancelled(cancel_check):
            raise _Cancelled
        for (record, _), pred in zip(rereads, _ocr_predict(ocr_engine, [masked for _, masked in rereads])):
            record["text"] = _reading(pred)[0]
    records = [record for record in records if record["text"]]

    candidates = [record for record in records if not _same_line(record["at_pick"], record["text"])]
    if not candidates:
        return records
    neighbours = _neighbour_strips(video_path, crop_box, time_ranges, [record["time"] for record in candidates],
                                   cancel_check=cancel_check)
    batch, slots = [], []
    for record in candidates:
        record["candidate"] = True
        for neighbour_time, strip in neighbours.get(record["time"], []):
            masked = ocr_view.mask(strip, pick)
            if ocr_view.gate_fires(masked):
                batch.append(masked)
                slots.append((record, neighbour_time))
    if batch:
        if _is_cancelled(cancel_check):
            raise _Cancelled
        for (record, neighbour_time), pred in zip(slots, _ocr_predict(ocr_engine, batch)):
            record["neighbours"].append((neighbour_time, _reading(pred)[0]))
    for record in candidates:
        record["survives"] = any(_same_line(text, record["text"]) for _, text in record["neighbours"])
    return records


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def detect_brightness(video_path: str, crop_box, time_ranges, det_engine, ocr_engine,
                      folder_plateau: tuple[int, int] | None = None,
                      cancel_check: Callable[[], bool] | None = None) -> BrightnessResult:
    """Pick a brightness threshold for one file. See the module docstring.

    `crop_box` is the file's (x, y, w, h) in native pixels, as the OCR pass
    takes it. `time_ranges` are the file's keep ranges as (start, end) pairs
    ("MM:SS" strings, seconds, or None for open ends) or {"start", "end"}
    mappings, or None. Engines are passed in (detection-only and full OCR) so
    a process-wide engine cache can supply them.

    With `folder_plateau` this is the cheap path: 6 frames, analytic seed
    only. A seed inside the plateau gives the value seed - PICK_BELOW_TOP,
    never below the plateau's start ("narrow-plateau?" when that clamp bites):
    a file's own seed sits at or just under its own plateau top, which the
    folder plateau's top does not bound. Otherwise the result carries
    flagged == "escalate" and the caller must run full detection (call again
    without `folder_plateau`).

    `cancel_check`: a zero-argument callable polled before each sampling
    round, before OCR verification, and before each OCR batch and neighbour
    grab of the dim-text check; once it returns truthy the result is returned
    flagged "cancelled".
    """
    if crop_box is None:
        return BrightnessResult(DEFAULT_BRIGHTNESS, None, DEFAULT_BRIGHTNESS, None, FLAG_NEEDS_CROP, [])
    if _ranges_select_nothing(video_path, time_ranges):
        return BrightnessResult(DEFAULT_BRIGHTNESS, None, DEFAULT_BRIGHTNESS, None, FLAG_RANGES_EMPTY, [])

    def cancelled(seed=None, floor=None):
        fallback = DEFAULT_BRIGHTNESS if seed is None else seed
        return BrightnessResult(DEFAULT_BRIGHTNESS, None, fallback, floor, FLAG_CANCELLED, [])

    cheap = folder_plateau is not None
    strips, times, polys = [], [], []
    for phase in SAMPLE_ROUND_PHASES[:1] if cheap else SAMPLE_ROUND_PHASES:
        if _is_cancelled(cancel_check):
            return cancelled()
        pairs = _sample_strips(video_path, crop_box, time_ranges,
                               CHEAP_SAMPLE_FRAMES if cheap else FULL_SAMPLE_FRAMES, phase)
        strips += [strip for _, strip in pairs]
        times += [t for t, _ in pairs]
        polys += _detect_text_polys(det_engine, [strip for _, strip in pairs])
        if sum(1 for p in polys if p) >= VERIFY_STRIPS:
            break
    text_strips = [s for s, p in zip(strips, polys) if p]
    text_times = [t for t, p in zip(times, polys) if p]
    text_polys = [p for p in polys if p]
    empty_strips = [s for s, p in zip(strips, polys) if not p]
    try:
        seed = analytic_seed(text_strips, text_polys)
    except ValueError:
        seed = None

    if cheap:
        lo, hi = folder_plateau
        if seed is not None and seed < IMPLAUSIBLE_SEED:
            # Same verdict full detection would reach; escalating cannot help.
            return BrightnessResult(seed, None, seed, None, FLAG_COLOURED_TEXT, [])
        if seed is None or not lo <= seed <= hi:
            fallback = DEFAULT_BRIGHTNESS if seed is None else seed
            return BrightnessResult(fallback, None, fallback, None, FLAG_ESCALATE, [])
        flagged = FLAG_NARROW_PLATEAU if seed - PICK_BELOW_TOP < lo else None
        return BrightnessResult(max(lo, seed - PICK_BELOW_TOP), (lo, hi), seed, None, flagged, [])

    floor = gate_floor(empty_strips)
    # No empty strips: the floor was not measured, which is not evidence of clutter.
    flagged = FLAG_NO_CLEAN_THRESHOLD if floor is None and empty_strips else None
    if seed is None:
        return BrightnessResult(DEFAULT_BRIGHTNESS, None, DEFAULT_BRIGHTNESS, floor,
                                _compose_flag(flagged, FLAG_NO_TEXT), [])
    if seed < IMPLAUSIBLE_SEED:
        # The min-channel mask erases coloured (e.g. yellow) glyphs at any
        # useful threshold; OCR around a meaningless seed proves nothing.
        return BrightnessResult(seed, None, seed, floor, _compose_flag(flagged, FLAG_COLOURED_TEXT), [])

    if _is_cancelled(cancel_check):
        return cancelled(seed, floor)
    verification = _verify(text_strips, seed, ocr_engine)
    value, plateau = verification.value, verification.plateau
    if verification.evidence < THIN_EVIDENCE_STRIPS:
        flagged = _compose_flag(flagged, FLAG_THIN_EVIDENCE)
    if plateau is None:
        flagged = _compose_flag(flagged, FLAG_NO_PLATEAU)
    elif plateau[1] - plateau[0] < PICK_BELOW_TOP:
        flagged = _compose_flag(flagged, FLAG_NARROW_PLATEAU)
    try:
        dim = _dim_text_check(video_path, crop_box, time_ranges, text_strips, text_times, text_polys,
                              verification, ocr_engine, cancel_check=cancel_check)
    except _Cancelled:
        return cancelled(seed, floor)
    if not all(record["survives"] for record in dim):
        flagged = _compose_flag(flagged, FLAG_DIM_TEXT)
    if floor is not None and value < floor + GATE_FLOOR_MARGIN:
        flagged = _compose_flag(flagged, FLAG_NO_CLEAN_THRESHOLD)
    return BrightnessResult(int(value), plateau, seed, floor, flagged, verification.curve)
