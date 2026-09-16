"""Audio-guided subtitle crop detection.

`detect_crop()` replaces the old brute-force 40-60% / 0.5s crop scan with a
speech-guided one: it asks `core.detect.vad.probe_times()` for timestamps
ranked by likelihood of carrying dialogue, grabs just those frames (already
cropped to the bottom band and downscaled), runs the text-detection engine
on them, and stops as soon as it has enough agreeing hits -- instead of
walking the whole 40-60% window at a fixed step.

Two correctness rules this module fixes relative to the old detector
(`core/subtitle_detector.py`'s `_compute_crop_from_polys`), both pinned by
tests in `tests/test_detect_crop.py`:

1. Union, not tightest box. The old code kept a running min/max across all
   *accepted* polys but picked the *single* candidate frame with the
   smallest resulting height as "the" answer (see `_pick_best_candidate`
   there). A frame with a two-line subtitle visible produces a taller box
   than a frame that happens to show only the bottom line; picking the
   tightest single frame systematically clips the second line. This module
   unions extents *across* hit frames instead of choosing one frame.

2. Padding. `aggregate_box()` pads the raw union by 0.3% of frame height,
   top and bottom, before applying the existing 5% minimum-height floor,
   25% ceiling and 70% centred-width rules -- the old code's default
   padding was 0, and the raw (unpadded) box measurably sat a few pixels
   short of what users accepted (see task-2-brief.md's background section).

No Qt imports here (core/detect/ is Qt-free by project convention) -- this
module only touches numpy, subprocess (ffmpeg/ffprobe) and the detection
engine object it's handed.
"""
from __future__ import annotations

import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from dataclasses import fields as _dataclass_fields
from statistics import median

import numpy as np

from core.config import Config as _Config
from core.detect import vad

logger = logging.getLogger(__name__)

# --- Geometry defaults, mirroring core/subtitle_detector.py's `automation`
# settings keys so a settings dict built for the old detector still works
# here. `CROP_VERTICAL_PADDING` is the one deliberate change: the old
# default was 0 (no padding); this module's default is 0.3% of frame
# height, per the brief's rule 2 above.
CROP_WIDTH_FRACTION = 0.70
CROP_VERTICAL_PADDING = 0.003
CROP_MIN_HEIGHT_FRACTION = 0.05
BOTTOM_HALF_CUTOFF = 0.55
MAX_CROP_HEIGHT_FRAC = 0.25

# Detection score threshold - real text typically scores ~0.97.
DT_SCORE_THRESHOLD = 0.9

def _label_max_duration_default() -> float:
    """Read core.config.Config's own label_max_duration default rather than
    hardcoding a twin of it: the label pipeline already encodes "how long a
    single subtitle can plausibly stay on screen" as this value, and if
    that assumption ever changes, this module's watermark-vs-repeated-line
    boundary (below) should track it automatically instead of silently
    diverging. core/config.py has no Qt imports, so this import doesn't
    violate core/detect/'s Qt-free convention.
    """
    for f in _dataclass_fields(_Config):
        if f.name == "label_max_duration":
            return float(f.default)
    return 5.0  # matches Config's own fallback, in case the field is ever renamed


# Watermark rejection: a box present in every sampled frame, to within this
# fraction of frame height, is *suspected* static content (a logo/watermark)
# rather than a subtitle, but only once there are enough samples for "every
# frame" to mean something (2 identical frames could just be 2 lucky
# probes). Expressed as a fraction of frame height, not a fixed pixel count,
# so it doesn't get twice as strict at 4K as at 1080p (4px at 1080p was the
# original value this reproduces).
WATERMARK_TOLERANCE_FRAC = 4.0 / 1080.0
WATERMARK_MIN_SAMPLES = 3
# Extent identity alone cannot distinguish a real watermark from the same
# subtitle line sampled several times within its own display: vad.probe_times()
# only guarantees 0.75s minimum separation between picks, so N picks can
# span as little as (N-1)*0.75s, which is well inside a single subtitle's
# lifetime. The span requirement therefore has to exceed the longest a
# single subtitle could plausibly last, not an arbitrary round number --
# WATERMARK_MIN_SPAN_SEC is set above _label_max_duration_default() (the
# label pipeline's own ceiling, 5.0s by default) with an explicit margin,
# so a span that clears it is provably longer than any one subtitle could
# have lasted. Below that span the two cases are genuinely indistinguishable
# by extent alone -- see _watermark_status()'s "uncertain" outcome, which
# does NOT reject (a wrong box is visible/correctable in a review UI; a
# missing box with no explanation is not).
WATERMARK_SPAN_MARGIN_SEC = 1.0
WATERMARK_MIN_SPAN_SEC = _label_max_duration_default() + WATERMARK_SPAN_MARGIN_SEC

# Frame grab defaults.
TARGET_HEIGHT = 480
GRAB_POOL_SIZE = 10  # max concurrent ffmpeg subprocesses inside grab_frames()

# detect_crop() orchestration.
PROBE_BATCH_SIZE = 5          # probes fetched per batch before re-checking the stop condition
CONSENSUS_MIN_ENTRIES = 3     # consensus list must have at least this many entries to shortcut
CONSENSUS_STOP_HITS = 2       # ...at which point this many agreeing hits is enough
# Convergence stop (replaces a fixed hit-count stop -- see _run_round()):
# stop once the raw union hasn't grown for this many consecutive batches.
# 2 ("a couple") means the *earliest* a stop can happen is after 3 batches
# (15 probes at PROBE_BATCH_SIZE=5): batch 1 always resets the counter (it
# has no prior union to match), batch 2 must repeat batch 1's union
# (1 stable round), batch 3 must repeat that again (2 stable rounds ->
# stop). Requiring 1 repeat alone would let a single lucky coincidence
# stop early; requiring 2 means the union has now demonstrably stopped
# growing across two independent additional looks, not just one.
CONVERGENCE_STABLE_ROUNDS = 2
# Hard ceiling on probes fetched in one _run_round() call, so content
# whose union never stabilizes (or that never converges within the
# candidate list) still has bounded latency. Set to roughly the old
# brute-force detector's own worst case (30-35 probes, see the brief's
# background section) so this path's worst case is no worse than what it
# replaces, while the measured common case (see task-2-report.md) stays
# far under it.
MAX_PROBES_PER_ROUND = 30
CONSENSUS_Y_DEVIATION_FRAC = 0.10
CONSENSUS_MAX_HEIGHT_RATIO = 1.5
# Baseline clustering (round 5, replacing round 4's Y-centre outlier
# rejection -- see _cluster_by_baseline()'s docstring for why Y-centre
# alone provably rejects genuine content, e.g. a subtitle deliberately
# repositioned to avoid on-screen graphics, not just noise).
#
# Subtitles share a BOTTOM edge (baseline); a second line extends the box
# upward from that same baseline. A spurious detection elsewhere in the
# accepted band has a genuinely different baseline. So: group accepted
# extents by bottom edge within BASELINE_CLUSTER_TOLERANCE_FRAC, and treat
# the largest cluster as the real subtitle position -- but only once that
# cluster has recurred at least BASELINE_CLUSTER_MIN_DOMINANT_SIZE times.
# That minimum-size gate (not just "largest of whatever's seen so far")
# is what keeps this safe at small sample counts: 2 unrelated noise
# detections that happen to cluster together must not out-vote 1 genuine
# hit just by being a 2 vs 1 majority -- real subtitle text recurs at the
# identical baseline far more reliably than noise does, so requiring the
# SAME baseline to reappear this many times before trusting it is what
# actually rules out "2 noise + 1 real" style coincidences, not the
# clustering step by itself. Value chosen to match this module's other
# small-sample-distrust constants (WATERMARK_MIN_SAMPLES, etc.), all 3.
#
# Tolerance: real per-frame OCR bounding boxes for the SAME baseline jitter
# by only a few px in practice (observed ~2-5px on the reference corpus);
# 0.03 of frame height (~32px at 1080p) is generous headroom above that
# jitter while staying well under any deliberate reposition worth flagging
# (reproduced case: a 180px shift, ~6x this tolerance at 1080p).
BASELINE_CLUSTER_TOLERANCE_FRAC = 0.03
BASELINE_CLUSTER_MIN_DOMINANT_SIZE = 3
UNIFORM_START_FRAC = 0.40
UNIFORM_END_FRAC = 0.60
UNIFORM_STEP_SEC = 0.5
LOW_AGREEMENT_HITS = 3        # fewer accepted hits than this and we still flag the result

# CropResult.flagged reason strings.
FLAG_NO_SPEECH = "no-speech"                        # true silence / no audio stream at all
FLAG_SPEECH_PROBES_EXHAUSTED = "speech-probes-exhausted"  # audio has speech, but no text found there
FLAG_TOP_POSITIONED = "top-positioned?"
FLAG_CEILING_EXCEEDED = "ceiling-exceeded"
FLAG_LOW_AGREEMENT = "low-agreement"
FLAG_STATIC_CONTENT = "static-content"              # watermark/logo rejection (confirmed)
FLAG_WATERMARK_UNCERTAIN = "static-content?"        # same extent every sample, but not enough
                                                     # temporal spread to confirm -- box is kept
FLAG_MULTIPLE_POSITIONS = "multiple-positions?"     # a second baseline cluster with >=2 members
                                                     # was included in the union, not just the
                                                     # dominant one -- e.g. a genuinely repositioned
                                                     # subtitle
FLAG_OUTLIER_DISCARDED = "outlier-discarded?"       # a single hit at a baseline nothing else shared
                                                     # was excluded from the union -- never silently


@dataclass
class CropResult:
    box: tuple[int, int, int, int] | None
    sample_pts: list[float] = field(default_factory=list)
    envelope: tuple[int, int, int, int] | None = None
    agreed: int = 0
    probes_used: int = 0
    flagged: str | None = None


# --------------------------------------------------------------------------
# ffprobe / ffmpeg plumbing
# --------------------------------------------------------------------------

def _probe_dimensions(video_path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, check=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"no video stream found in {video_path}")
    return int(streams[0]["width"]), int(streams[0]["height"])


def _crop_geometry(orig_w: int, orig_h: int, band_frac: float,
                    target_height: int) -> tuple[int, int, int, int, int, int]:
    """Compute the integer crop+scale geometry for grabbing the bottom
    `band_frac` of a frame and downscaling it to `target_height`.

    Returns (crop_w, crop_h, crop_x, crop_y, out_w, out_h). Explicit
    integers rather than ffmpeg filter expressions (`ih*B`, `scale=-2:H`)
    because a raw-video pipe carries no header describing the frame it
    sends -- the caller has to already know width/height to reshape the
    byte buffer, so that arithmetic has to happen on our side, not inside
    ffmpeg's filter graph. See grab_frames()'s docstring.
    """
    band_frac = min(max(band_frac, 0.0), 1.0)
    crop_h = max(1, int(round(orig_h * band_frac)))
    crop_h = min(crop_h, orig_h)
    crop_y = orig_h - crop_h
    crop_w = orig_w
    out_h = max(1, min(target_height, crop_h))
    out_w = max(2, int(round(crop_w * out_h / crop_h)))
    if out_w % 2:
        out_w -= 1
    return crop_w, crop_h, 0, crop_y, out_w, out_h


def _grab_one(video_path: str, t: float, crop_w: int, crop_h: int, crop_x: int,
               crop_y: int, out_w: int, out_h: int) -> np.ndarray | None:
    """Grab a single frame at time `t`, already cropped+scaled. None on failure.

    Every failure mode of the ffmpeg invocation itself (missing binary,
    permission error, timeout, ...) is caught here and turned into a
    logged None, matching grab_frames()'s documented "failures are dropped,
    not raised" contract -- only `subprocess.run()`'s own raise sites need
    catching; a short/garbled read is handled separately below via a
    length check, not an exception.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, t):.3f}", "-i", video_path,
        "-frames:v", "1",
        "-vf", f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={out_w}:{out_h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(
            "grab_frames: failed to grab t=%.3f from %s (%s: %s)",
            t, video_path, type(exc).__name__, exc,
        )
        return None
    expected = out_w * out_h * 3
    if result.returncode != 0 or len(result.stdout) != expected:
        logger.warning(
            "grab_frames: failed to grab t=%.3f from %s (rc=%s, got %d/%d bytes)",
            t, video_path, result.returncode, len(result.stdout), expected,
        )
        return None
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(out_h, out_w, 3)


def _grab_frames_with_times(video_path: str, times: list[float], band_frac: float,
                             target_height: int, known_dims: tuple[int, int] | None = None,
                             ) -> tuple[list[tuple[float, np.ndarray]], tuple]:
    """Implementation behind grab_frames(): parallel one-shot ffmpeg grabs,
    bounded to GRAB_POOL_SIZE concurrent processes, returning (time, frame)
    pairs in request order with failed grabs dropped. Also returns the
    crop geometry used, so callers (detect_crop) can map detection results
    in the returned frames' coordinate space back to full-frame pixels.

    `known_dims`: (width, height) if the caller already knows it (detect_crop
    always does, from its own opening _probe_dimensions() call) -- skips
    the ffprobe re-spawn this function would otherwise do on every batch.
    grab_frames() itself never passes this, so it stays independently
    callable/testable exactly as documented.
    """
    if not times:
        return [], (0, 0, 0, 0, 0, 0, 0, 0)

    orig_w, orig_h = known_dims if known_dims is not None else _probe_dimensions(video_path)
    geometry = _crop_geometry(orig_w, orig_h, band_frac, target_height)
    crop_w, crop_h, crop_x, crop_y, out_w, out_h = geometry

    results: list[np.ndarray | None] = [None] * len(times)
    with ThreadPoolExecutor(max_workers=min(GRAB_POOL_SIZE, len(times))) as pool:
        futures = {
            pool.submit(_grab_one, video_path, t, crop_w, crop_h, crop_x, crop_y, out_w, out_h): i
            for i, t in enumerate(times)
        }
        for future in futures:
            i = futures[future]
            results[i] = future.result()

    pairs = [(t, frame) for t, frame in zip(times, results) if frame is not None]
    return pairs, (orig_w, orig_h) + geometry


def grab_frames(video_path: str, times: list[float], band_frac: float = 0.55,
                 target_height: int = TARGET_HEIGHT) -> list[np.ndarray]:
    """Parallel one-shot ffmpeg grabs, cropped to the bottom `band_frac` of
    the frame and scaled to `target_height`, in the same order as `times`.

    Failed grabs (ffmpeg error, timeout, short read) are dropped rather
    than raising -- a handful of unreadable probe timestamps shouldn't
    fail the whole detection pass -- and logged via the `core.detect.crop`
    logger.
    """
    pairs, _ = _grab_frames_with_times(video_path, times, band_frac, target_height)
    return [frame for _, frame in pairs]


# --------------------------------------------------------------------------
# Pure aggregation
# --------------------------------------------------------------------------

def _poly_extent(poly) -> tuple[float, float, float, float, float] | None:
    """(min_x, min_y, max_x, max_y, center_y) for one polygon, or None if malformed."""
    pts = np.asarray(poly, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return None
    xs, ys = pts[:, 0], pts[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), float(ys.mean())


def _per_frame_extents(polys_per_frame, frame_h: float, cutoff_frac: float,
                        ) -> list[tuple[float, float, float, float] | None]:
    """Per probed frame: the union extent of its in-band polys (center_y at
    or below `cutoff_frac` of frame height), or None if that frame had no
    in-band poly. Shared by aggregate_box() and detect_crop() -- both the
    final box computation AND the "did this probe frame count as a hit"
    decision use this exact same acceptance test, so a detection that
    won't contribute to the box can never be counted as a hit either.
    """
    cutoff_y = frame_h * cutoff_frac
    extents: list[tuple[float, float, float, float] | None] = []
    for polys in polys_per_frame:
        extent = None
        for poly in polys:
            parsed = _poly_extent(poly)
            if parsed is None:
                continue
            min_x, min_y, max_x, max_y, center_y = parsed
            if center_y < cutoff_y:
                continue
            if extent is None:
                extent = [min_x, min_y, max_x, max_y]
            else:
                extent[0] = min(extent[0], min_x)
                extent[1] = min(extent[1], min_y)
                extent[2] = max(extent[2], max_x)
                extent[3] = max(extent[3], max_y)
        extents.append(tuple(extent) if extent is not None else None)
    return extents


def _watermark_status(accepted: list[tuple[float, float, float, float]], total_frames: int,
                       frame_h: float, contributing_times: list[float] | None = None,
                       ) -> str | None:
    """Tri-state watermark judgement for a set of per-frame extents:

    - None: not watermark-shaped at all (too few samples, or the extents
      actually differ across frames). Ordinary detection.
    - "confirmed": the same extent (within WATERMARK_TOLERANCE_FRAC of
      frame height) appears in every one of `total_frames` sampled frames,
      AND the contributing samples span more than WATERMARK_MIN_SPAN_SEC --
      long enough that no single subtitle could plausibly have still been
      the same displayed line across the whole span. Reject.
    - "uncertain": the extent matches, but either no timing information was
      supplied, or the samples that do have timing span too little to rule
      out "the same line, sampled repeatedly". Do NOT reject: return the
      box as usual and let the caller attach an explanatory flag instead.
      A wrong box is visible and correctable in a review UI; a missing box
      with no explanation is neither.

    `contributing_times` is optional: aggregate_box()'s public, pure API
    doesn't require callers to supply sample timestamps (the 7 brief-locked
    unit tests in tests/test_detect_crop.py call it without any). Absent
    any timing information at all, this preserves the original
    extent-identity-only behaviour those tests pin (always "confirmed" when
    the extents match) rather than inventing a judgement from nothing.
    detect_crop() (the actual orchestration this rule protects) always
    supplies real probe timestamps, so the temporal-spread requirement is
    live on the path that matters.
    """
    if len(accepted) != total_frames or total_frames < WATERMARK_MIN_SAMPLES:
        return None
    tolerance = frame_h * WATERMARK_TOLERANCE_FRAC
    first = accepted[0]
    same_extent = all(
        all(abs(a - b) <= tolerance for a, b in zip(e, first))
        for e in accepted[1:]
    )
    if not same_extent:
        return None
    if contributing_times is None:
        return "confirmed"
    if len(contributing_times) < 2:
        return "uncertain"
    span = max(contributing_times) - min(contributing_times)
    return "confirmed" if span >= WATERMARK_MIN_SPAN_SEC else "uncertain"


def _bounding_union(extents: list[tuple[float, float, float, float] | None],
                     ) -> tuple[float, float, float, float] | None:
    """Plain min/max bounding box over whichever entries of `extents` are
    not None, with no watermark judgement involved -- used both by
    _union_extent() (the final, watermark-aware box) and by _run_round()'s
    convergence check (just "has the raw union grown", a strictly earlier
    and simpler question than "is it static content").
    """
    accepted = [e for e in extents if e is not None]
    if not accepted:
        return None
    min_x = min(e[0] for e in accepted)
    min_y = min(e[1] for e in accepted)
    max_x = max(e[2] for e in accepted)
    max_y = max(e[3] for e in accepted)
    return (min_x, min_y, max_x, max_y)


def _cluster_by_baseline(indexed_baselines: list[tuple[int, float]], tolerance: float,
                          ) -> list[list[int]]:
    """Single-linkage gap clustering of (index, baseline) pairs on the
    baseline value: sort ascending, start a new cluster whenever the gap
    to the previous value exceeds `tolerance`. Returns clusters as lists
    of the original indices, in baseline-ascending order within each.
    """
    if not indexed_baselines:
        return []
    ordered = sorted(indexed_baselines, key=lambda pair: pair[1])
    clusters: list[list[int]] = [[ordered[0][0]]]
    prev_baseline = ordered[0][1]
    for idx, baseline in ordered[1:]:
        if baseline - prev_baseline <= tolerance:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
        prev_baseline = baseline
    return clusters


def _baseline_cluster_union(extents: list[tuple[float, float, float, float] | None],
                             accepted_idx: list[int], frame_h: float,
                             ) -> tuple[tuple[float, float, float, float], int, str | None]:
    """Cluster accepted per-frame extents by BOTTOM EDGE (baseline), not
    Y-centre, and union the dominant cluster's extents -- admitting a
    genuine two-line frame by construction, since it shares the same
    baseline as a one-line frame from the same track and only differs in
    top edge.

    An earlier version of this rejected Y-CENTRE outliers directly instead
    of clustering by baseline. Reproduced failure: a subtitle legitimately
    repositioned to avoid on-screen graphics is *also* a Y-centre-position
    minority, indistinguishable by that signal from real noise -- that
    version silently dropped it (union (288,797,1344,236) -> (288,977,
    1344,56)). Baseline clustering tells the two cases apart: a
    repositioned subtitle's frames share a DIFFERENT baseline with EACH
    OTHER (forming their own cluster), where noise's baseline is
    essentially uncorrelated frame to frame (staying singletons).

    Returns (union, agreed_count, position_flag). `position_flag` is
    FLAG_MULTIPLE_POSITIONS if a second cluster with >=2 members got
    folded into the union, FLAG_OUTLIER_DISCARDED if any singleton
    cluster got excluded, both (composed) if both happened, or None.
    Never discards silently.

    Below BASELINE_CLUSTER_MIN_DOMINANT_SIZE members in the largest
    cluster, nothing is confidently "the" subtitle yet -- union
    everything unfiltered rather than guess. This is what keeps 2
    coincidentally-clustered noise detections from out-voting 1 genuine
    hit: the pair alone (size 2) can't clear the dominant-size gate any
    more safely than the singleton can, so neither gets excluded until
    real evidence (the SAME baseline recurring this many times) exists.
    """
    baseline_items = [(i, extents[i][3]) for i in accepted_idx]  # extents[i][3] == max_y (bottom edge)
    tolerance = frame_h * BASELINE_CLUSTER_TOLERANCE_FRAC
    clusters = _cluster_by_baseline(baseline_items, tolerance)
    clusters.sort(key=len, reverse=True)
    dominant = clusters[0]

    if len(dominant) < BASELINE_CLUSTER_MIN_DOMINANT_SIZE:
        # Not enough repeated evidence for any single position yet --
        # union everything rather than confidently exclude on a guess.
        kept_idx = accepted_idx
        position_flag = None
    else:
        kept_idx = list(dominant)
        position_flag = None
        for cluster in clusters[1:]:
            if len(cluster) >= 2:
                kept_idx.extend(cluster)
                position_flag = _compose_flag(position_flag, FLAG_MULTIPLE_POSITIONS)
            else:
                position_flag = _compose_flag(position_flag, FLAG_OUTLIER_DISCARDED)

    union = _bounding_union([extents[i] for i in kept_idx])
    return union, len(kept_idx), position_flag


def _union_extent(polys_per_frame, frame_h: float, cutoff_frac: float,
                   sample_times: list[float] | None = None,
                   ) -> tuple[tuple[float, float, float, float] | None, int, str | None, str | None]:
    """Returns (union extent, count of contributing frames, watermark
    status, position status). The union is None only when there are no
    in-band polys at all, OR the watermark status is "confirmed" (see
    _watermark_status()) -- an "uncertain" watermark status, or any
    position status, still returns a real union, since insufficient or
    conflicting evidence must not silently discard a detection.

    `sample_times`, if given, must align 1:1 with `polys_per_frame`; only
    the timestamps of frames that actually contributed an in-band poly are
    used (see _watermark_status()).

    Watermark judgement runs first, over ALL accepted extents/timestamps
    regardless of baseline clustering -- a genuine watermark is, by
    definition, one cluster with identical extent (including baseline),
    so clustering doesn't change that judgement; it only matters for
    telling apart the position of a REAL, moving subtitle from noise once
    the content isn't static.
    """
    extents = _per_frame_extents(polys_per_frame, frame_h, cutoff_frac)
    accepted_idx = [i for i, e in enumerate(extents) if e is not None]
    if not accepted_idx:
        return None, 0, None, None

    accepted = [extents[i] for i in accepted_idx]
    contributing_times = None
    if sample_times is not None and len(sample_times) == len(extents):
        contributing_times = [sample_times[i] for i in accepted_idx]

    watermark_status = _watermark_status(accepted, len(extents), frame_h, contributing_times)
    if watermark_status == "confirmed":
        return None, len(accepted_idx), watermark_status, None

    union, agreed, position_flag = _baseline_cluster_union(extents, accepted_idx, frame_h)
    return union, agreed, watermark_status, position_flag


def aggregate_box(polys_per_frame, frame_size: tuple[int, int], band_frac: float = 0.55,
                   settings: dict | None = None,
                   sample_times: list[float] | None = None) -> tuple[int, int, int, int] | None:
    """Union of accepted text-detection polygons across sampled frames, into
    one padded, clamped crop box -- or None if there's nothing to build a
    box from (no in-band polys, the box exceeds the height ceiling, or the
    result is a *confirmed* static watermark -- see _watermark_status()).
    An extent that merely *looks* identical across samples but lacks
    enough temporal spread to confirm that ("uncertain") still returns a
    real box; callers that want to surface the distinction should call
    _union_extent() directly for the watermark status, as detect_crop()
    does.

    `polys_per_frame`: one entry per sampled frame, each a list of polygons
    (each polygon an array-like of (x, y) points) in `frame_size` pixel
    coordinates.
    `frame_size`: (width, height) of the coordinate space the polygons are in.
    `band_frac`: fraction of frame height below which a poly's center must
    sit to be accepted -- overridable via settings['bottom_half_cutoff'].
    `settings`: optional dict mirroring core/subtitle_detector.py's
    `automation` keys (crop_width_fraction, crop_vertical_padding,
    crop_min_height_fraction, bottom_half_cutoff). None means the defaults
    at the top of this module.
    `sample_times`: optional, one timestamp per entry of `polys_per_frame`
    (same length, same order) -- lets the watermark check require real
    temporal spread among the contributing samples instead of only extent
    identity. Optional and keyword-friendly precisely so the plain
    positional calls in tests/test_detect_crop.py's pure tests keep working
    unchanged; see _watermark_status()'s docstring.
    """
    frame_w, frame_h = frame_size
    s = settings or {}
    width_frac = float(s.get("crop_width_fraction", CROP_WIDTH_FRACTION))
    vert_pad_frac = float(s.get("crop_vertical_padding", CROP_VERTICAL_PADDING))
    min_height_frac = float(s.get("crop_min_height_fraction", CROP_MIN_HEIGHT_FRACTION))
    cutoff_frac = float(s.get("bottom_half_cutoff", band_frac))

    union, _agreed, _watermark_status_, _position_flag_ = _union_extent(
        polys_per_frame, frame_h, cutoff_frac, sample_times,
    )
    if union is None:
        return None

    min_x, min_y, max_x, max_y = union

    pad = frame_h * vert_pad_frac
    top = max(0.0, min_y - pad)
    bottom = min(float(frame_h), max_y + pad)

    min_height = frame_h * min_height_frac
    if bottom - top < min_height:
        center = (top + bottom) / 2.0
        top = max(0.0, center - min_height / 2.0)
        bottom = min(float(frame_h), top + min_height)

    width = int(frame_w * width_frac)
    x = (frame_w - width) // 2
    y = int(round(top))
    h = int(round(bottom)) - y

    if h > frame_h * MAX_CROP_HEIGHT_FRAC:
        return None

    return (x, y, width, h)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _uniform_probe_times(duration_sec: float, start_frac: float = UNIFORM_START_FRAC,
                          end_frac: float = UNIFORM_END_FRAC,
                          step_sec: float = UNIFORM_STEP_SEC) -> list[float]:
    start = duration_sec * start_frac
    end = duration_sec * end_frac
    if end <= start:
        return []
    n_steps = int((end - start) / step_sec) + 1
    return [start + i * step_sec for i in range(n_steps) if start + i * step_sec <= end]


def _consistent_with_consensus(y_frac: float, h_frac: float,
                                consensus: list[tuple[float, float]]) -> bool:
    if not consensus:
        return True
    med_y = median(c[0] for c in consensus)
    med_h = median(c[1] for c in consensus)
    if med_h > 0 and h_frac > med_h * CONSENSUS_MAX_HEIGHT_RATIO:
        return False
    if abs(y_frac - med_y) > CONSENSUS_Y_DEVIATION_FRAC:
        return False
    return True


def _compose_flag(existing: str | None, new: str) -> str:
    """Combine flag reasons instead of one silently clobbering another --
    e.g. "speech probes found nothing" AND "had to widen past the bottom
    band" can both be true of the same result, and both are useful to a
    reviewer deciding whether to trust the box."""
    if not existing:
        return new
    parts = existing.split("+")
    if new in parts:
        return existing
    return f"{existing}+{new}"


def _spread_order(times: list[float]) -> list[float]:
    """Reorder timestamps by greedy farthest-point sampling: the earliest
    entries visited are maximally spread apart, rather than adjacent in
    the original (chronological) list. Doesn't add, drop, or rank
    candidates -- vad.probe_times() already picked all of them -- only
    changes visitation order.

    Why this matters: whichever candidates get visited first determine
    which samples the watermark check (see WATERMARK_MIN_SPAN_SEC) has
    temporal spread to judge with early on. Walking candidates in strict
    chronological order means a local cluster of nearby hits could
    otherwise dominate the early batches, giving the watermark check
    nothing but closely-spaced samples to reason about for longer than
    necessary. Note what this does NOT do: it doesn't decide which hits
    end up contributing to the union -- that's _run_round()'s convergence
    stop, which keeps probing (regardless of visitation order) until the
    union itself stops growing, precisely so that spread-ordering the
    *fetch* order can never cause a batch containing a genuine second
    subtitle line to simply never be looked at.
    """
    if len(times) <= 2:
        return list(times)
    remaining = list(times)
    ordered = [remaining.pop(0), remaining.pop(-1)]
    while remaining:
        best_idx, best_dist = 0, -1.0
        for i, t in enumerate(remaining):
            dist = min(abs(t - o) for o in ordered)
            if dist > best_dist:
                best_idx, best_dist = i, dist
        ordered.append(remaining.pop(best_idx))
    return ordered


def _map_poly_to_full_frame(poly, geometry: tuple) -> np.ndarray:
    """Map a polygon from a grabbed (cropped+scaled) frame's coordinate
    space back to full original-frame pixel coordinates."""
    orig_w, orig_h, crop_w, crop_h, crop_x, crop_y, out_w, out_h = geometry
    scale_x = crop_w / out_w
    scale_y = crop_h / out_h
    pts = np.asarray(poly, dtype=np.float64)
    mapped = np.empty_like(pts)
    mapped[:, 0] = pts[:, 0] * scale_x + crop_x
    mapped[:, 1] = pts[:, 1] * scale_y + crop_y
    return mapped


def _run_round(video_path: str, times: list[float], det_engine, band_frac: float,
                consensus: list[tuple[float, float]] | None,
                frame_size: tuple[int, int], settings: dict | None,
                known_dims: tuple[int, int] | None = None,
                ) -> tuple[list, list[float], int, list[float]]:
    """Fetch+detect `times` in PROBE_BATCH_SIZE-sized batches, stopping
    once the raw in-band union has stopped growing for CONVERGENCE_STABLE_ROUNDS
    consecutive batches (or CONSENSUS_STOP_HITS is reached with an
    in-tolerance consensus, or MAX_PROBES_PER_ROUND is hit, or `times` is
    exhausted) -- NOT once an arbitrary hit count is reached.

    This replaces an earlier fixed-count stop (`raw_hits >= 5`) that, once
    combined with _spread_order() and PROBE_BATCH_SIZE == 5, let whichever
    5 probes happened to land in the very first batch become the ENTIRE
    contributing set: if all 5 scored hits, the loop stopped before ever
    looking at later batches, silently dropping a genuine second subtitle
    line if it wasn't among that first five -- exactly the tightest-box
    clipping the union rule exists to prevent, but now with no signal that
    anything was missed (see test_convergence_captures_a_two_line_subtitle_
    regardless_of_probe_order). A union-based estimator has to stop when
    the *estimate* stops growing, not when it has seen N samples: the
    union here always consumes every accepted hit found before that point,
    never just the first batch.

    A hit that wouldn't survive aggregate_box()'s own filtering (e.g. text
    detected in the sliver between the grabbed band's top edge and the
    cutoff line) must not count toward growth either, or detect_crop can
    "converge" on hits that end up contributing nothing, landing on
    box=None with neither fallback triggered.

    Deliberately NOT outlier-filtered here. An earlier version ran outlier
    rejection inside this loop, before the convergence check -- which let
    discarding evidence manufacture false stability: the FILTERED union
    could stop growing (and probing stop) while the RAW union was still
    changing, because the very hit that would have kept it growing had
    already been thrown away first. Reproduced directly by a re-reviewer.
    Baseline clustering (the actual outlier-vs-real-content judgement) now
    runs once, only in _union_extent()/aggregate_box(), AFTER probing has
    already concluded based on the unfiltered evidence -- see
    _baseline_cluster_union()'s docstring. This function always returns
    every accepted hit, filtered or not; detect_crop() decides what to do
    with them.

    `times` is walked in _spread_order(), not the order it's given in --
    spread is still useful here for getting temporal span (needed by the
    watermark check) established early, but it no longer decides WHICH
    hits contribute: it only decides the order batches are looked at in,
    and the convergence stop (not a hit count) decides when to stop
    looking, so a late-arriving batch is never silently skipped just for
    arriving late.

    Returns (polys_per_frame, sample_pts_used, raw_hit_count, frame_times).
    Polygons are already mapped to full-frame pixel coordinates.
    `frame_times` has one entry per entry of `polys_per_frame`, the probe
    timestamp that produced it (for the watermark temporal-spread check).
    """
    polys_per_frame: list[list] = []
    frame_times: list[float] = []
    sample_pts: list[float] = []
    consensus = consensus or []
    _, frame_h = frame_size
    cutoff_frac = float((settings or {}).get("bottom_half_cutoff", band_frac))
    raw_hits = 0
    prior_union: tuple[float, float, float, float] | None = None
    stable_rounds = 0

    times = _spread_order(times)

    i = 0
    while i < len(times):
        if len(sample_pts) >= MAX_PROBES_PER_ROUND:
            break

        chunk = times[i:i + PROBE_BATCH_SIZE]
        i += len(chunk)

        pairs, geometry = _grab_frames_with_times(
            video_path, chunk, band_frac, TARGET_HEIGHT, known_dims=known_dims,
        )
        sample_pts.extend(chunk)
        if not pairs:
            continue
        orig_w, orig_h, crop_w, crop_h, crop_x, crop_y, out_w, out_h = geometry
        full_geometry = (orig_w, orig_h, crop_w, crop_h, crop_x, crop_y, out_w, out_h)

        frames = [frame for _, frame in pairs]
        results = list(det_engine.predict(frames))

        for (t, _frame), item in zip(pairs, results):
            scores = item.get("dt_scores")
            if scores is None:
                scores = []
            polys = item.get("dt_polys")
            if polys is None:
                polys = []
            accepted = [
                _map_poly_to_full_frame(poly, full_geometry)
                for poly, score in zip(polys, scores)
                if float(score) >= DT_SCORE_THRESHOLD
            ]
            polys_per_frame.append(accepted)
            frame_times.append(t)

        # Everything below reads the UNFILTERED accumulated data -- see
        # this function's docstring for why filtering must not happen
        # before the stop decision.
        extents = _per_frame_extents(polys_per_frame, frame_h, cutoff_frac)
        raw_hits = sum(e is not None for e in extents)

        # Consensus fast path: an independent, externally-validated reason
        # to trust an early result, unrelated to the convergence check
        # below. Still count-gated (CONSENSUS_STOP_HITS), but the
        # consistency check against already-resolved files' median
        # shape is what actually protects it from stopping on a
        # not-yet-complete union.
        if len(consensus) >= CONSENSUS_MIN_ENTRIES and raw_hits >= CONSENSUS_STOP_HITS:
            provisional = aggregate_box(polys_per_frame, frame_size, band_frac, settings, frame_times)
            if provisional is not None:
                _, py, _, ph = provisional
                if _consistent_with_consensus(py / frame_h, ph / frame_h, consensus):
                    break

        # Convergence stop: has the raw union grown since the last batch?
        # A None union (no in-band hits yet at all) is never "stable" --
        # there's nothing to converge on, so keep probing until either a
        # real union appears or the probe budget/candidate list runs out.
        current_union = _bounding_union(extents)
        if current_union is not None and current_union == prior_union:
            stable_rounds += 1
        else:
            stable_rounds = 0
        prior_union = current_union

        if current_union is not None and stable_rounds >= CONVERGENCE_STABLE_ROUNDS:
            break

    extents = _per_frame_extents(polys_per_frame, frame_h, cutoff_frac)
    raw_hits = sum(e is not None for e in extents)
    return polys_per_frame, sample_pts, raw_hits, frame_times


def detect_crop(video_path: str, duration_sec: float, det_engine,
                 consensus: list[tuple[float, float]] | None = None,
                 settings: dict | None = None) -> CropResult:
    """Detect the subtitle crop box for `video_path`.

    Orchestration: vad.probe_times() picks candidate timestamps ranked by
    likelihood of carrying dialogue -> grab_frames() fetches them in
    batches, already cropped to the bottom band and downscaled ->
    det_engine.predict() scores each frame's text polygons -> polys
    scoring >= 0.9 that also fall in-band are kept and mapped back to
    full-frame coordinates -> stop once 5 probe frames agree, or 2 frames
    agree when `consensus` (a list of (y_frac, h_frac) from already-resolved
    files) has at least 3 entries and the box is consistent with it ->
    aggregate_box() unions the accepted polys into one padded, clamped crop.

    Fallbacks, in order (flags compose rather than overwrite -- see
    _compose_flag()):
    - vad.probe_times() returns [] (true digital silence / no audio
      stream): fall back to uniform 0.5s probing over the same 40-60%
      window immediately. flagged="no-speech".
    - Speech-guided probes are exhausted with zero in-band hits (audio
      exists and has speech, but no visible text at any sampled
      timestamp): fall back to the same uniform probing.
      flagged="speech-probes-exhausted" (NOT "no-speech" -- there was
      speech, just no detected text at those timestamps).
    - The bottom band yields nothing at all, even after the uniform
      fallback: retry once on full frames (band_frac=1.0, no position
      cutoff). flagged gains "top-positioned?".
    A result can also be flagged "static-content" (watermark CONFIRMED --
    same extent every sample, spanning more than WATERMARK_MIN_SPAN_SEC;
    box is None), "static-content?" (same extent every sample, but not
    enough temporal spread to confirm; box is still returned -- see
    _watermark_status()), "multiple-positions?" (a second baseline
    cluster with >=2 members got folded into the union alongside the
    dominant one -- e.g. a genuinely repositioned subtitle), "outlier-
    discarded?" (a single hit at a baseline nothing else shared was
    excluded from the union -- see _baseline_cluster_union()),
    "ceiling-exceeded" (in-band hits exist but the resulting box is too
    tall), or "low-agreement" (a box was built, but from fewer than
    LOW_AGREEMENT_HITS contributing frames).
    """
    known_dims = _probe_dimensions(video_path)
    orig_w, orig_h = known_dims
    frame_size = (orig_w, orig_h)

    sample_pts: list[float] = []
    flagged: str | None = None

    times = vad.probe_times(video_path, duration_sec, window_frac=(0.40, 0.60))
    speech_probing_available = bool(times)
    if not speech_probing_available:
        times = _uniform_probe_times(duration_sec)
        flagged = _compose_flag(flagged, FLAG_NO_SPEECH)

    polys_per_frame, used, raw_hits, frame_times = _run_round(
        video_path, times, det_engine, band_frac=BOTTOM_HALF_CUTOFF,
        consensus=consensus, frame_size=frame_size, settings=settings, known_dims=known_dims,
    )
    sample_pts.extend(used)

    if raw_hits == 0 and speech_probing_available:
        uniform_times = _uniform_probe_times(duration_sec)
        polys_per_frame, used, raw_hits, frame_times = _run_round(
            video_path, uniform_times, det_engine, band_frac=BOTTOM_HALF_CUTOFF,
            consensus=consensus, frame_size=frame_size, settings=settings,
            known_dims=known_dims,
        )
        sample_pts.extend(used)
        flagged = _compose_flag(flagged, FLAG_SPEECH_PROBES_EXHAUSTED)

    used_full_frame_retry = False
    if raw_hits == 0:
        retry_times = _uniform_probe_times(duration_sec) if not sample_pts else sample_pts
        # Reuse the already-attempted timestamps for the full-frame retry
        # rather than probing new ones -- we already know these timestamps
        # exist in the video; band_frac=1.0 just widens what we look at.
        retry_times = sorted(set(retry_times))
        polys_per_frame, used, raw_hits, frame_times = _run_round(
            video_path, retry_times, det_engine, band_frac=1.0,
            consensus=consensus, frame_size=frame_size,
            settings={**(settings or {}), "bottom_half_cutoff": 0.0},
            known_dims=known_dims,
        )
        sample_pts.extend(used)
        flagged = _compose_flag(flagged, FLAG_TOP_POSITIONED)
        used_full_frame_retry = True

    cutoff_frac = 0.0 if used_full_frame_retry else float(
        (settings or {}).get("bottom_half_cutoff", BOTTOM_HALF_CUTOFF)
    )
    envelope_extent, agreed, watermark_status, position_flag = _union_extent(
        polys_per_frame, orig_h, cutoff_frac, frame_times,
    )
    envelope = None
    if envelope_extent is not None:
        ex_min_x, ex_min_y, ex_max_x, ex_max_y = envelope_extent
        envelope = (
            int(round(ex_min_x)), int(round(ex_min_y)),
            int(round(ex_max_x - ex_min_x)), int(round(ex_max_y - ex_min_y)),
        )

    box_settings = settings if not used_full_frame_retry else {
        **(settings or {}), "bottom_half_cutoff": 0.0,
    }
    box_band_frac = 1.0 if used_full_frame_retry else BOTTOM_HALF_CUTOFF
    box = aggregate_box(polys_per_frame, frame_size, box_band_frac, box_settings, frame_times)

    if box is None and agreed > 0:
        # watermark_status can be "confirmed" (aggregate_box's underlying
        # union was already None -- box=None follows directly), "uncertain"
        # (the union WAS real, but aggregate_box's own ceiling check
        # rejected the padded/floored box built from it -- box=None here
        # comes from the ceiling, not the watermark judgement), or None
        # (no watermark involvement at all, just a too-tall box). Only the
        # "confirmed" case is a watermark rejection; the ternary below
        # resolves both other cases to FLAG_CEILING_EXCEEDED correctly,
        # but that's the ceiling check's doing, not an "uncertain implies
        # ceiling" guarantee -- an uncertain union that happens to fit
        # under the ceiling never reaches this branch at all (box is not
        # None then; see the "uncertain" flag composition below instead).
        flagged = _compose_flag(
            flagged, FLAG_STATIC_CONTENT if watermark_status == "confirmed" else FLAG_CEILING_EXCEEDED,
        )
    if box is not None and watermark_status == "uncertain":
        # Same extent every sample, but not enough temporal spread among
        # the contributing probes to tell a real watermark apart from the
        # same subtitle line sampled repeatedly -- kept the box rather
        # than guess, per the review ruling; flag it so a human can.
        flagged = _compose_flag(flagged, FLAG_WATERMARK_UNCERTAIN)
    if position_flag is not None:
        # FLAG_MULTIPLE_POSITIONS and/or FLAG_OUTLIER_DISCARDED, already
        # composed together by _baseline_cluster_union() if both applied.
        for part in position_flag.split("+"):
            flagged = _compose_flag(flagged, part)
    if box is not None and 0 < agreed < LOW_AGREEMENT_HITS:
        flagged = _compose_flag(flagged, FLAG_LOW_AGREEMENT)

    return CropResult(
        box=box,
        sample_pts=sample_pts,
        envelope=envelope,
        agreed=agreed,
        probes_used=len(sample_pts),
        flagged=flagged,
    )
