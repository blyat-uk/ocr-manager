"""Audio-guided subtitle crop detection.

`detect_crop()` replaces the old brute-force 40-60% / 0.5s crop scan with a
speech-guided one: it asks `core.detect.vad.probe_times()` for timestamps
ranked by likelihood of carrying dialogue, grabs just those frames (already
cropped to the bottom band and downscaled), runs the text-detection engine
on them, and stops once the raw (unfiltered) in-band union has stopped
growing across consecutive probe batches -- see _run_round()'s
convergence stop -- instead of walking the whole 40-60% window at a fixed
step, or stopping once an arbitrary count of probes has agreed.

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

Probe frames are fetched two ways, chosen per source by resolution (see
_prefers_persistent_fetch()): one-shot `ffmpeg -ss` processes, or a pool of
persistent PyAV containers (_PersistentFrameFetcher) that seek accurately and
decode forward. Both return the SAME frame for the same requested time, pixel
for pixel; only the cost differs. Neither snaps probes to keyframes -- see
_PersistentFrameFetcher's docstring for the measurement that ruled that out.

No Qt imports here (core/detect/ is Qt-free by project convention) -- this
module only touches numpy, PyAV, subprocess (ffmpeg/ffprobe) and the
detection engine object it's handed.
"""
from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from dataclasses import fields as _dataclass_fields
from statistics import median

import av
import numpy as np

from core.config import Config as _Config
from core.detect import vad
from core.detect.flags import compose_flag as _compose_flag
from core.detect.flags import is_cancelled as _is_cancelled
from core.detect.flags import only_informational
from videocr.pyav_adapter import _TRC_ARIB_STD_B67, _TRC_SMPTE2084, PyAVCapture, _pyav_has_zscale

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
# Convergence stop (replaces a fixed hit-count stop -- see _run_round()):
# stop once the raw union hasn't grown (beyond CONVERGENCE_VERTICAL_TOLERANCE_FRAC,
# below) for this many consecutive batches.
# 2 ("a couple") means the *earliest* a stop can happen is after 3 batches
# (15 probes at PROBE_BATCH_SIZE=5): batch 1 always resets the counter (it
# has no prior union to match), batch 2 must match batch 1's union
# (1 stable round), batch 3 must still match it (2 stable rounds ->
# stop). Requiring 1 repeat alone would let a single lucky coincidence
# stop early; requiring 2 means the union has now demonstrably stopped
# growing across two independent additional looks, not just one.
CONVERGENCE_STABLE_ROUNDS = 2
# Consensus (>=CONSENSUS_MIN_ENTRIES already-resolved files, see
# _run_round()) RELAXES this requirement to just 1 stable round when the
# RAW union's shape already agrees with it -- legitimate evidence this
# file's band matches the series, but never a substitute for stability
# itself: a stop still requires the raw union to have been observed
# stable (within tolerance) across a batch boundary, not merely "enough
# hits seen".
CONSENSUS_STABLE_ROUNDS = 1
# Convergence tolerance (Task 2b). "Stable" compares only the raw union's
# VERTICAL edges, and allows them to have moved by up to this fraction of
# frame height since the start of the current stable streak -- see
# _union_is_stable().
#
# Horizontal edges are not compared at all: the returned box's width is a
# fixed, centred CROP_WIDTH_FRACTION of the frame (aggregate_box() never reads
# the union's x-extent), so a wider dialogue line cannot change the box, and
# comparing it is what kept exact-equality convergence from ever firing on
# real dialogue (slay hit MAX_PROBES_PER_ROUND on every run; batch-to-batch
# horizontal growth measured up to 185px there).
#
# The vertical value is derived from union variation measured on the two
# reference files (every speech-guided probe, see task-2b-report.md):
# - Jitter, the growth that must count as stable: batch-to-batch vertical
#   growth of the raw union without a new line measured at most 0.57% of
#   frame height (XWZ 169 12.4px/2160, XWZ 168 7.4px/2160, slay 3px/888),
#   and the full spread of any single edge across ALL frames of the subtitle
#   cluster -- a bound on how far jitter alone can ever move that edge -- at
#   most 1.14% (slay's bottom edge, 10.2px/888; XWZ at most 0.92%).
# - Real growth, which must never count as stable: a second subtitle line
#   grows the union by at least one line height, and the smallest single-line
#   height measured was 4.03% (xwz, 87px/2160).
# 1.5% clears the largest jitter bound with ~30% headroom while staying below
# 0.4x the smallest line height, so a second line always exceeds it by more
# than 2.5x.
CONVERGENCE_VERTICAL_TOLERANCE_FRAC = 0.015
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
# An isolated (singleton-cluster) hit is kept, not discarded, if it sits
# within this many detected line heights of the dominant cluster's own
# union -- see _baseline_cluster_union()'s docstring. "About one line
# height" per the review ruling: a genuine missing line (OCR caught only
# one line of a two-line subtitle on that probe) sits directly against
# the kept union with little to no gap; real noise measured on the
# reference corpus sat ~2.4 line heights away, comfortably outside this.
ADJACENT_SINGLETON_MAX_GAP_LINE_HEIGHTS = 1.0
# Probe fetching (Task 2b). Sources with MORE than this many pixels fetch
# probes through persistent PyAV containers; at or below it, through one-shot
# ffmpeg processes. Measured crossover, see _prefers_persistent_fetch().
PERSISTENT_FETCH_MIN_PIXELS = 1920 * 1080
PERSISTENT_POOL_SIZE = PROBE_BATCH_SIZE   # one container per probe of a batch
# Decoders for which skipping non-reference frames (AVDISCARD_NONREF) on the
# way to an accurate-seek target was MEASURED to return the same frame as a
# full decode: pixel-identical to one-shot grabs for h264 and hevc on the six
# reference sources (x265 HEVC, 8- and 10-bit, 4K and 1080p) and on synthetic
# x264 (B-frames, 25 and 24000/1001 fps) and x265 (B-pyramid, temporal layers,
# 10-bit) clips -- see
# test_persistent_fetcher_frames_match_one_shot_and_recorded_times_refetch_them.
# That is evidence for these encodes, not a proof for every h264/hevc stream:
# - HEVC `_N` NAL unit types only promise "not referenced by pictures of the
#   same temporal sub-layer"; a picture in a HIGHER sub-layer may still
#   reference one, and FFmpeg would skip it anyway. x265 marks only truly
#   unreferenced pictures `_N`, so the temporal-layers clip cannot exercise
#   that case.
# - h264 without a VUI bitstream_restriction makes FFmpeg infer the reorder
#   depth from POC gaps as it decodes; skipped frames change those gaps, so
#   output order near the target could differ from a full decode.
# libdav1d (AV1) is the measured counter-example -- with skipping on it
# returned a later frame for 14 of 17 requested times -- so any codec not
# listed here decodes every frame.
NONREF_SKIP_EXACT_CODECS = frozenset({"h264", "hevc"})
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
FLAG_MULTIPLE_POSITIONS = "multiple-positions?"     # a second baseline cluster was folded into the
                                                     # union alongside the dominant one -- either
                                                     # >=2 members (e.g. a genuinely repositioned
                                                     # subtitle) or a single hit kept because it sits
                                                     # adjacent to the dominant cluster (a likely
                                                     # missing line -- see _baseline_cluster_union())
FLAG_OUTLIER_DISCARDED = "outlier-discarded?"       # a single hit at a baseline nothing else shared
                                                     # was excluded from the union -- never silently
FLAG_CANCELLED = "cancelled"                        # cancel_check cut a round short -- composed
                                                     # on every round it happens to, alongside
                                                     # whatever that round's own flag was (if any);
                                                     # never inferred from an absent flag elsewhere
FLAG_UNKNOWN_REJECTION = "unknown-rejection"        # structural safety net (see detect_crop()'s
                                                     # post-condition, just before CropResult is
                                                     # built): box is None but nothing above composed
                                                     # a flag -- a gap in this function's own flag
                                                     # coverage, not a real detection outcome. Should
                                                     # never appear for any KNOWN no-box path; if it
                                                     # does, a specific flag is missing above it.


# Which flags leave a box safe to apply without review -- see
# CropResult.auto_applicable for the classification of every flag.
INFORMATIONAL_FLAGS = frozenset({FLAG_NO_SPEECH, FLAG_SPEECH_PROBES_EXHAUSTED})
BLOCKING_FLAGS = frozenset({
    FLAG_TOP_POSITIONED, FLAG_CEILING_EXCEEDED, FLAG_LOW_AGREEMENT, FLAG_STATIC_CONTENT,
    FLAG_WATERMARK_UNCERTAIN, FLAG_MULTIPLE_POSITIONS, FLAG_OUTLIER_DISCARDED, FLAG_CANCELLED,
    FLAG_UNKNOWN_REJECTION,
})


# CropSample.lines: two boxes sit on the same text row when their vertical
# extents overlap by at least this fraction of the smaller box's height.
TEXT_ROW_MIN_OVERLAP_FRAC = 0.5


@dataclass(frozen=True)
class CropSample:
    """One analysed probe frame, as evidence for reviewing a CropResult.

    `time`: the entry of CropResult.sample_pts this frame was recorded under
    (same position, same value) -- re-fetch it with grab_frames().
    `boxes`: (x, y, w, h) of every text polygon detect_crop() accepted on
    this frame -- score >= DT_SCORE_THRESHOLD and centre inside the band its
    round judged (the bottom-band cutoff; the whole frame on the full-frame
    retry) -- in the source frame's native pixels like CropResult.box, NOT in
    the cropped, downscaled band grab_frames() returns. Rounded like
    CropResult.envelope, sorted by (y, x). Empty when nothing was accepted.
    `kept`: this frame contributed to the kept union (the frames `agreed`
    counts and `hit_pts` lists). Decided per frame, not per time: the
    full-frame retry re-probes times the bottom-band rounds already probed,
    and only the retry's frame at such a time is kept, although the time is
    in hit_pts.
    `lines`: distinct text rows among `boxes` (see _count_text_rows()), 0
    when there are none.
    """

    time: float
    boxes: tuple[tuple[int, int, int, int], ...]
    kept: bool
    lines: int

    def to_evidence(self) -> dict:
        """JSON-able: {"time", "boxes" (lists), "kept", "lines"}."""
        return {
            "time": float(self.time),
            "boxes": [[int(v) for v in box] for box in self.boxes],
            "kept": bool(self.kept),
            "lines": int(self.lines),
        }


def _int_list(values) -> list[int] | None:
    return None if values is None else [int(v) for v in values]


@dataclass
class CropResult:
    """One file's crop detection.

    Apply `box` without review only when `auto_applicable`. `flagged` is
    None or FLAG_* reasons joined by "+". Cancellation (FLAG_CANCELLED) does
    not raise: the result keeps whatever box the evidence gathered so far
    gives, which may be clipped. The times in `sample_pts` / `hit_pts`
    re-fetch their frames through THIS module's fetch layer (grab_frames),
    whose pixels are not the OCR pass's, and on some sources not even its
    frames -- see grab_frames() and core/detect/__init__.py.
    """

    box: tuple[int, int, int, int] | None
    # One entry per frame actually fetched and analysed, in probing order:
    # the probe's requested time, which is a time that re-fetches exactly the
    # analysed frame through core.detect.crop's fetch layer (at-or-after,
    # ms-rounded -- see _seek_seconds()), on either fetch path. It may precede
    # that frame's true PTS by up to one frame. Never the frame's own PTS:
    # on a time base that is not whole milliseconds (e.g. 1/24000 at
    # 24000/1001 fps) a PTS such as 0.917583 rounds up to 0.918 and would
    # re-fetch the NEXT frame. A grab that failed returned no frame and is
    # not listed (it still spends MAX_PROBES_PER_ROUND).
    # probes_used == len(sample_pts).
    sample_pts: list[float] = field(default_factory=list)
    envelope: tuple[int, int, int, int] | None = None
    agreed: int = 0
    probes_used: int = 0
    flagged: str | None = None
    # The timestamps whose detections actually contributed to the kept
    # union (survived both the in-band filter and baseline-cluster
    # admission -- see _baseline_cluster_union()), in chronological order.
    # NOT the same as sample_pts: sample_pts is the full probe history in
    # probing order, which after a fallback round is guaranteed to start
    # with a no-text frame (round 1 only falls back because it found zero
    # hits). Callers that need "a frame where a subtitle was actually
    # seen" (e.g. seeding a review UI's initial frame) must use hit_pts,
    # not sample_pts[0].
    hit_pts: list[float] = field(default_factory=list)
    # (width, height) of the original video frame, as already probed by
    # detect_crop() -- lets callers convert `box` into (y_frac, h_frac)
    # without a second, redundant dimension probe of their own.
    frame_size: tuple[int, int] | None = None
    # One CropSample per sample_pts entry, same order: what each analysed
    # frame showed and whether it contributed -- evidence for review only;
    # nothing above is derived from it.
    samples: list[CropSample] = field(default_factory=list)
    # The bottom_half_cutoff the kept samples were judged with, and `box` was
    # built with: the settings' band (default BOTTOM_HALF_CUTOFF), or 0.0 after
    # the full-frame retry. Re-running aggregate_box over the samples ("fit to
    # all samples") must use it. Evidence for review only, like `samples`;
    # None for a result detect_crop() did not build.
    cutoff_frac: float | None = None

    def to_evidence(self) -> dict:
        """The result as JSON-able evidence for review: box, envelope,
        agreed, probes_used, flagged, hit_pts, frame_size, samples as dicts
        (CropSample.to_evidence()) and cutoff_frac. Tuples become lists."""
        return {
            "box": _int_list(self.box),
            "envelope": _int_list(self.envelope),
            "agreed": int(self.agreed),
            "probes_used": int(self.probes_used),
            "flagged": self.flagged,
            "hit_pts": [float(t) for t in self.hit_pts],
            "frame_size": _int_list(self.frame_size),
            "samples": [sample.to_evidence() for sample in self.samples],
            "cutoff_frac": None if self.cutoff_frac is None else float(self.cutoff_frac),
        }

    @property
    def auto_applicable(self) -> bool:
        """True only when there is a box and every flag on it is
        informational: the flag says how the probes were chosen, not that
        the box is in doubt. An unrecognised flag blocks.

        | flag                    | box     | class         | why |
        |-------------------------|---------|---------------|-----|
        | no-speech               | kept    | informational | No audio stream or digital silence, so probes are uniform 0.5 s steps over 40-60% -- the old detector's own probing, whose boxes were applied unreviewed. The box passes every rule below. |
        | speech-probes-exhausted | kept    | informational | Speech-guided probes found no text, so the same uniform probing ran; same reasoning. |
        | top-positioned?         | kept    | blocking      | From the full-frame retry, with no bottom-band cutoff: signs, titles or scene text anywhere in frame qualify. The old bottom-half detector could never return such a box. |
        | low-agreement           | kept    | blocking      | Under LOW_AGREEMENT_HITS contributing frames: the union needs several frames to catch a second line, so the box may clip one. |
        | static-content?         | kept    | blocking      | Same extent in every sample over too short a span to rule out a logo or watermark. |
        | multiple-positions?     | kept    | blocking      | A second baseline cluster (a repositioned subtitle, or a lone adjacent line) was folded into the union: the box may be inflated or span two positions. |
        | outlier-discarded?      | kept    | blocking      | A hit at a baseline nothing else shared was left out. Noise or a once-seen subtitle elsewhere -- the code cannot tell, and if it was a subtitle the box misses it. |
        | cancelled               | partial | blocking      | Probing was cut short; the box is from partial evidence and may be clipped. |
        | static-content          | None    | blocking      | Confirmed watermark; no box. |
        | ceiling-exceeded        | None    | blocking      | Union taller than MAX_CROP_HEIGHT_FRAC; no box. |
        | unknown-rejection       | None    | blocking      | Safety net: no box and no specific flag. |
        """
        return self.box is not None and only_informational(self.flagged, INFORMATIONAL_FLAGS)


# --------------------------------------------------------------------------
# ffprobe / ffmpeg plumbing
# --------------------------------------------------------------------------

def _probe_source(video_path: str) -> tuple[int, int, str | None]:
    """(width, height, color_transfer) of the first video stream via ffprobe.
    color_transfer is ffprobe's name for it ("smpte2084", "bt709", ...) or
    None when the stream does not signal one."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,color_transfer", "-of", "json", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, check=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"no video stream found in {video_path}")
    stream = streams[0]
    return int(stream["width"]), int(stream["height"]), stream.get("color_transfer")


def _probe_dimensions(video_path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream via ffprobe."""
    width, height, _transfer = _probe_source(video_path)
    return width, height


# Probe frames are tone-mapped exactly when, and exactly as, the OCR pass
# tone-maps: PQ and HLG sources, through PyAVCapture._add_tonemap_chain()
# (videocr/pyav_adapter.py), ahead of the crop. Spec section 11: detection
# sampling uses the OCR pass's chain.
#
# Measured before this (I2; synthetic 30 s clips, burned-in white subtitles
# over colour, light-gradient and dark scenes, converted SDR -> PQ at 100, 203
# and 1000 nits, PQ tag-only, HLG; real TextDetection; final-fix-report.md):
# - pixels: un-tone-mapped probes were 74-114 levels off the tone-mapped ones
#   on over 99% of pixels (now 0.01-0.03 levels off a graph that scales
#   before bgr24, 0.02-0.07 off PyAVCapture + cv2 resize);
# - detect_crop() boxes moved 1-7 px on all 6 HDR clips. On the 203-nit PQ
#   clip the box bottom sat 3 px short at 1080p and 7 px short at 2160p (the
#   padding there is 3.2 / 6.5 px), from 9 and 8 contributing frames instead
#   of 12. For scale: resampling differences of ~1.5 levels alone move a
#   box 1-3 px on the SDR control, so the 1080p shifts are near the
#   detector's own jitter; the 2160p shift and the pixel gap are not.
# - cost: detect_crop() on the PQ clips went 0.82 -> 1.21 s (1080p) and
#   0.73 -> 0.94 s (2160p), medians of 3 at load 6-11; SDR sources unchanged.
_TONE_MAPPED_TRANSFERS = {"smpte2084": _TRC_SMPTE2084, "arib-std-b67": _TRC_ARIB_STD_B67}


def _tone_map_filters(transfer: str | None) -> list[str]:
    """ffmpeg CLI filters for the OCR pass's tone map of a `transfer` source,
    in order; [] when the OCR pass does not tone-map it. Mirrors
    PyAVCapture._add_tonemap_chain() filter for filter -- including its
    choice of chain by what PyAV's bundled FFmpeg supports, not by what the
    system ffmpeg does -- so one-shot grabs and the persistent path (which
    calls that method itself) produce the same pixels."""
    if transfer not in _TONE_MAPPED_TRANSFERS:
        return []
    if _pyav_has_zscale():
        linear = "zscale=t=linear:npl=100" if transfer == "smpte2084" else "zscale=t=linear"
        return [linear, "format=gbrpf32le", "tonemap=hable", "zscale=t=bt709"]
    return ["format=gbrpf32le", "tonemap=hable"]


def _probe_filters(crop_w: int, crop_h: int, crop_x: int, crop_y: int, out_w: int, out_h: int,
                   transfer: str | None) -> str:
    """The -vf chain of a one-shot grab. SDR: crop, scale (bgr24 at the
    output). Tone-mapped: tone map, crop, bgr24, scale -- the OCR pass's
    order (tone map ahead of the crop, scaling on bgr24)."""
    tone_map = _tone_map_filters(transfer)
    if not tone_map:
        return f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={out_w}:{out_h}"
    return ",".join(tone_map + [f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}", "format=bgr24",
                                f"scale={out_w}:{out_h}"])


# Default for `transfer` arguments below: probe the source for it.
_PROBE = object()


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


def _seek_seconds(t: float) -> float:
    """The container-relative time an accurate seek to `t` actually targets.

    Millisecond precision, because that is what the one-shot path hands
    `ffmpeg -ss`. The persistent path (see _PersistentDecoder.grab()) targets
    exactly the same value, so for any requested time both paths decode the
    same frame -- the first one at or after this time -- and a probe's frame
    cannot depend on which fetch strategy happened to run. This rounding is
    also why callers record the REQUESTED time, not the decoded frame's PTS:
    the requested time maps back to the same frame by construction, a PTS
    that is not a whole millisecond may not (see CropResult.sample_pts).
    """
    return float(f"{max(0.0, t):.3f}")


def _grab_one(video_path: str, t: float, crop_w: int, crop_h: int, crop_x: int,
               crop_y: int, out_w: int, out_h: int, transfer: str | None = None) -> np.ndarray | None:
    """Grab a single frame at time `t`, already cropped+scaled (and
    tone-mapped when `transfer` is PQ or HLG, see _tone_map_filters()). None
    on failure.

    Every failure mode of the ffmpeg invocation itself (missing binary,
    permission error, timeout, ...) is caught here and turned into a
    logged None, matching grab_frames()'s documented "failures are dropped,
    not raised" contract -- only `subprocess.run()`'s own raise sites need
    catching; a short/garbled read is handled separately below via a
    length check, not an exception.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{_seek_seconds(t):.3f}", "-i", video_path,
        "-frames:v", "1",
        "-vf", _probe_filters(crop_w, crop_h, crop_x, crop_y, out_w, out_h, transfer),
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
                             transfer=_PROBE,
                             ) -> tuple[list[tuple[float, np.ndarray]], tuple]:
    """Implementation behind grab_frames(): parallel one-shot ffmpeg grabs,
    bounded to GRAB_POOL_SIZE concurrent processes, returning (requested
    time, frame) pairs in request order with failed grabs dropped. Also returns the
    crop geometry used, so callers (detect_crop) can map detection results
    in the returned frames' coordinate space back to full-frame pixels.

    `known_dims`: (width, height) if the caller already knows it (detect_crop
    always does, from its own opening _probe_dimensions() call) -- skips
    the ffprobe re-spawn this function would otherwise do on every batch.
    grab_frames() itself never passes this, so it stays independently
    callable/testable exactly as documented.

    `transfer`: the source's color_transfer as _probe_source() reports it
    (None: not signalled), which decides the tone map; probed when not given.
    """
    if not times:
        return [], (0, 0, 0, 0, 0, 0, 0, 0)

    if known_dims is None or transfer is _PROBE:
        probed_w, probed_h, probed_transfer = _probe_source(video_path)
        if known_dims is None:
            known_dims = (probed_w, probed_h)
        if transfer is _PROBE:
            transfer = probed_transfer
    orig_w, orig_h = known_dims
    geometry = _crop_geometry(orig_w, orig_h, band_frac, target_height)
    crop_w, crop_h, crop_x, crop_y, out_w, out_h = geometry

    results: list[np.ndarray | None] = [None] * len(times)
    with ThreadPoolExecutor(max_workers=min(GRAB_POOL_SIZE, len(times))) as pool:
        futures = {
            pool.submit(_grab_one, video_path, t, crop_w, crop_h, crop_x, crop_y, out_w, out_h, transfer): i
            for i, t in enumerate(times)
        }
        for future in futures:
            i = futures[future]
            results[i] = future.result()

    pairs = [(t, frame) for t, frame in zip(times, results) if frame is not None]
    return pairs, (orig_w, orig_h) + geometry


# Tolerance when comparing a decoded frame's timestamp against the seek
# target: absorbs float rounding in pts * time_base, far below one frame.
_PTS_EPSILON_SEC = 1e-6


class _PersistentDecoder:
    """One open PyAV container plus the crop/scale/bgr24 filter graphs built
    for it. Used by exactly one worker thread at a time (see
    _PersistentFrameFetcher.fetch()); PyAV releases the GIL while demuxing
    and decoding, so several of these decode genuinely in parallel.
    """

    def __init__(self, video_path: str):
        self._container = av.open(video_path)
        try:
            if not self._container.streams.video:
                raise ValueError(f"no video stream found in {video_path}")
            self._stream = self._container.streams.video[0]
            self._time_base = self._stream.time_base
            start = self._container.start_time
            # Probe times, like `ffmpeg -ss`, are relative to the container's
            # start; stream timestamps are not.
            self._start_sec = start / av.time_base if start is not None else 0.0
            self._skip_nonref = self._stream.codec_context.name in NONREF_SKIP_EXACT_CODECS
            self._graphs: dict[tuple, av.filter.Graph] = {}
        except BaseException:
            self._container.close()
            raise

    def _graph_for(self, geometry: tuple) -> av.filter.Graph:
        """crop -> scale -> bgr24, the same filters as a one-shot grab
        (_probe_filters()). A PQ or HLG stream -- decided from the stream's
        own transfer characteristic, as PyAVCapture._detect_tonemap() does
        for the OCR pass -- gets the OCR pass's tone map first, through
        PyAVCapture._add_tonemap_chain() itself, then crop -> bgr24 -> scale.
        See _TONE_MAPPED_TRANSFERS for the measurement behind it."""
        graph = self._graphs.get(geometry)
        if graph is None:
            _orig_w, _orig_h, crop_w, crop_h, crop_x, crop_y, out_w, out_h = geometry
            graph = av.filter.Graph()
            buffer = graph.add_buffer(template=self._stream)
            trc = int(self._stream.codec_context.color_trc)
            if trc in _TONE_MAPPED_TRANSFERS.values():
                head = [PyAVCapture._add_tonemap_chain(graph, buffer, trc)]
                tail = [graph.add("crop", f"{crop_w}:{crop_h}:{crop_x}:{crop_y}"),
                        graph.add("format", "bgr24"),
                        graph.add("scale", f"{out_w}:{out_h}")]
            else:
                head = [buffer]
                tail = [graph.add("crop", f"{crop_w}:{crop_h}:{crop_x}:{crop_y}"),
                        graph.add("scale", f"{out_w}:{out_h}"),
                        graph.add("format", "bgr24")]
            chain = head + tail + [graph.add("buffersink")]
            for upstream, downstream in zip(chain, chain[1:]):
                upstream.link_to(downstream)
            graph.configure()
            self._graphs[geometry] = graph
        return graph

    def grab(self, t: float, geometry: tuple) -> np.ndarray:
        """Accurate seek: the first frame at or after _seek_seconds(t) --
        the same frame `ffmpeg -ss` returns -- cropped and scaled the same
        way _grab_one() does. Raises on any failure; the caller drops and
        logs.
        """
        target = _seek_seconds(t) + self._start_sec
        time_base = self._time_base
        codec_context = self._stream.codec_context
        self._container.seek(int(target / time_base), stream=self._stream, backward=True)
        frame = None
        skipping = self._skip_nonref
        try:
            if skipping:
                # Frames no other frame references can be skipped on the way
                # to the target -- but only BEFORE it: skipping stops at the
                # first packet whose pts reaches the target, and every packet
                # holding a frame at or after the target reaches it too, so
                # the target and everything after it are always decoded.
                codec_context.skip_frame = "NONREF"
            for packet in self._container.demux(self._stream):
                if skipping and (packet.pts is None or packet.pts * time_base >= target - _PTS_EPSILON_SEC):
                    codec_context.skip_frame = "DEFAULT"
                    skipping = False
                for decoded in codec_context.decode(packet):
                    if decoded.pts is not None and decoded.pts * time_base >= target - _PTS_EPSILON_SEC:
                        frame = decoded
                        break
                if frame is not None:
                    break
        finally:
            if self._skip_nonref:
                codec_context.skip_frame = "DEFAULT"
        if frame is None:
            raise EOFError(f"no frame at or after {t:.3f}s")

        graph = self._graph_for(geometry)
        graph.push(frame)
        image = graph.pull().to_ndarray()
        _orig_w, _orig_h, _cw, _ch, _cx, _cy, out_w, out_h = geometry
        if image.shape != (out_h, out_w, 3):
            raise ValueError(f"filtered frame has shape {image.shape}, expected {(out_h, out_w, 3)}")
        return image

    def close(self) -> None:
        self._graphs.clear()
        self._container.close()


class _PersistentFrameFetcher:
    """Probe fetching through a pool of persistent PyAV containers, one per
    probe in a batch, each seeking accurately and decoding forward.

    Why accurate seeks, and not the keyframe snapping the Task 2b brief asked
    for: a keyframe-only fetch is ~15x cheaper per probe at 4K, but the
    reference encodes place keyframes at scene cuts, where burned-in
    subtitles are absent or mid-fade. Measured with the detection engine on
    every speech-guided probe, a subtitle was detected at the requested
    times for 50.5% (XWZ 168) / 47.5% (XWZ 169) / 74% (slay) of probes, but
    in only 3% / 3% / 22% of frames sitting exactly on a keyframe; snapping
    each probe to the nearest keyframe inside the speech segment containing
    it cut those hit rates to 40% / 29% / 50%, and an end-to-end run
    halved the contributing hits. Snapping to keyframes removes the very
    evidence the audio guidance selects for. See task-2b-report.md.

    What the pool buys instead is the per-probe process and container
    start-up that one-shot grabs pay, plus parallel decoding across
    containers; decode-forward skips non-reference frames where that is
    exact (NONREF_SKIP_EXACT_CODECS). Frames are pixel-identical to
    _grab_one() at the same requested time.

    Contract, same as _grab_frames_with_times(): (requested time, image)
    pairs in request order, failures dropped and logged, never raised. The
    requested time is what callers record -- it re-fetches exactly this frame
    through either fetch path (see CropResult.sample_pts).

    If EVERY probe of a non-empty batch fails -- PyAV opened the file but
    cannot decode it, e.g. "cannot decode unknown codec" -- the pool is
    released and that batch and every later one for this file go through
    one-shot ffmpeg grabs instead, with a warning. Without that, every probe
    would be dropped and the file misreported as having no text
    (speech-probes-exhausted, top-positioned?).
    """

    def __init__(self, video_path: str, known_dims: tuple[int, int],
                 pool_size: int = PERSISTENT_POOL_SIZE):
        self._video_path = video_path
        self._dims = known_dims
        self._decoders: list[_PersistentDecoder] = []
        self._executor: ThreadPoolExecutor | None = None
        self._one_shot_fallback = False
        try:
            for _ in range(max(1, pool_size)):
                self._decoders.append(_PersistentDecoder(video_path))
        except BaseException:
            self.close()
            raise
        self._executor = ThreadPoolExecutor(max_workers=len(self._decoders),
                                            thread_name_prefix="crop-probe-fetch")

    def fetch(self, times: list[float], band_frac: float,
              target_height: int) -> tuple[list[tuple[float, np.ndarray]], tuple]:
        """(pairs, geometry) exactly as _grab_frames_with_times() returns them."""
        if self._one_shot_fallback:
            return _grab_frames_with_times(self._video_path, times, band_frac, target_height,
                                           known_dims=self._dims)
        orig_w, orig_h = self._dims
        geometry = (orig_w, orig_h) + _crop_geometry(orig_w, orig_h, band_frac, target_height)
        if not times:
            return [], geometry

        results: list[np.ndarray | None] = [None] * len(times)
        n_decoders = len(self._decoders)

        def work(k: int) -> None:
            # Decoder k owns every k-th request: no decoder is ever touched
            # by two threads at once.
            decoder = self._decoders[k]
            for i in range(k, len(times), n_decoders):
                results[i] = self._grab(decoder, times[i], geometry)

        list(self._executor.map(work, range(min(n_decoders, len(times)))))
        pairs = [(t, image) for t, image in zip(times, results) if image is not None]
        if not pairs:
            logger.warning(
                "%s: persistent decoders returned no frame for a batch of %d probes; "
                "using one-shot ffmpeg grabs for the rest of this file",
                self._video_path, len(times),
            )
            self._one_shot_fallback = True
            self._release()
            return _grab_frames_with_times(self._video_path, times, band_frac, target_height,
                                           known_dims=self._dims)
        return pairs, geometry

    def _grab(self, decoder: _PersistentDecoder, t: float,
              geometry: tuple) -> np.ndarray | None:
        try:
            return decoder.grab(t, geometry)
        except (av.error.FFmpegError, OSError, ValueError, EOFError) as exc:
            logger.warning(
                "grab_frames: failed to grab t=%.3f from %s (%s: %s)",
                t, self._video_path, type(exc).__name__, exc,
            )
            return None

    def _release(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        decoders, self._decoders = self._decoders, []
        for decoder in decoders:
            decoder.close()

    def close(self) -> None:
        self._release()


def _prefers_persistent_fetch(frame_size: tuple[int, int]) -> bool:
    """Fetch-strategy policy: persistent containers only where they
    measurably beat one-shot grabs.

    Measured with this module's real detect_crop() on the reference files,
    fetch time for the same 15 probes (and the same box) with the policy
    forced each way, medians of 4 interleaved runs on a shared, loaded
    machine: XWZ 168 (3840x2160, 10-bit HEVC) one-shot 3.33s vs persistent
    1.09s; XWZ 169 3.47s vs 1.03s; slay (1920x888, HEVC) one-shot 0.91s vs
    persistent 1.20s. A resolution sweep re-encoded from one XWZ scene
    (720p..2160p, same keyframe positions and bits per pixel) had
    persistent winning at every size, so the crossover is content-dependent
    -- slay's longer GOPs and costlier-per-pixel decode favour one-shot
    processes, whose ffmpeg decoders frame-thread each long decode run --
    and the only real 1080p-class source measured favours one-shot grabs.
    So the line sits just above 1080p: nothing at or below 1920x1080
    changes.
    """
    width, height = frame_size
    return width * height > PERSISTENT_FETCH_MIN_PIXELS


def _open_frame_fetcher(video_path: str, known_dims: tuple[int, int],
                        pool_size: int = PERSISTENT_POOL_SIZE) -> _PersistentFrameFetcher | None:
    """The persistent fetcher for this source if the policy wants one, else
    None (one-shot grabs). If the containers cannot be opened, logs a warning
    and returns None: one-shot grabs still work wherever the ffmpeg CLI can
    read the file."""
    if not _prefers_persistent_fetch(known_dims):
        return None
    try:
        return _PersistentFrameFetcher(video_path, known_dims, pool_size=pool_size)
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        logger.warning(
            "%s: could not open persistent decoders (%s: %s); falling back to one-shot ffmpeg grabs",
            video_path, type(exc).__name__, exc,
        )
        return None


def grab_frames(video_path: str, times: list[float], band_frac: float = 0.55,
                 target_height: int = TARGET_HEIGHT) -> list[np.ndarray]:
    """Frames at `times`, cropped to the bottom `band_frac` of the frame and
    scaled to `target_height`, in the same order as `times`.

    Fetched by parallel one-shot ffmpeg grabs, or -- above the resolution
    crossover, see _prefers_persistent_fetch() -- by a pool of persistent
    containers; the frames are identical either way.

    These are NOT the pixels the OCR pass sees, and on some sources not even
    its frames: a full-width band through crop -> scale -> bgr24 (or the
    system ffmpeg CLI), where the OCR pass decodes through
    videocr.pyav_adapter.Capture with its own decode downscale, crop and
    brightness mask. Measured: identical on Slay the Gods (1080p 8-bit), but
    87-90% of pixels off by up to 13 levels on XWZ (4K 10-bit) and a
    different frame on Jinwu Guard (h264 MKV) -- see task-4-report.md. Use
    core.detect.ocr_view for anything that must match OCR (brightness
    tuning or previews), and see core/detect/__init__.py for which times
    re-fetch through which function.

    Failed grabs (decode error, timeout, short read, past the end) are
    dropped rather than raising -- a handful of unreadable probe timestamps
    shouldn't fail the whole detection pass -- and logged via the
    `core.detect.crop` logger.
    """
    if not times:
        return []
    orig_w, orig_h, transfer = _probe_source(video_path)
    known_dims = (orig_w, orig_h)
    # No more containers than frames asked for: each persistent 4K decoder
    # holds several decoded reference frames.
    fetcher = _open_frame_fetcher(video_path, known_dims,
                                  pool_size=min(PERSISTENT_POOL_SIZE, len(times)))
    if fetcher is None:
        pairs, _ = _grab_frames_with_times(video_path, times, band_frac, target_height,
                                           known_dims=known_dims, transfer=transfer)
    else:
        try:
            pairs, _ = fetcher.fetch(times, band_frac, target_height)
        finally:
            fetcher.close()
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


def _union_is_stable(current: tuple[float, float, float, float],
                     anchor: tuple[float, float, float, float], frame_h: float) -> bool:
    """Has the raw union stayed put since `anchor` -- the raw union at the
    start of the current stable streak?

    Only the vertical edges are compared, each within
    CONVERGENCE_VERTICAL_TOLERANCE_FRAC of frame height (see that constant
    for the measured derivation). Horizontal edges are ignored because the
    box's width never depends on them. Growth is measured against the
    streak's START, not the previous batch, so it is cumulative: several
    batches each growing a little less than the tolerance cannot add up to
    more than it and still count as stable.
    """
    tolerance = frame_h * CONVERGENCE_VERTICAL_TOLERANCE_FRAC
    return (abs(current[1] - anchor[1]) <= tolerance
            and abs(current[3] - anchor[3]) <= tolerance)


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

    Returns (union, kept_idx, position_flag) -- `kept_idx` is the list of
    indices (into `extents`/`accepted_idx`) that actually contributed to
    `union`; callers that only need the count use len(kept_idx).
    `position_flag` is FLAG_MULTIPLE_POSITIONS if a second cluster with
    >=2 members (or an adjacent singleton -- see below) got folded into
    the union, FLAG_OUTLIER_DISCARDED if a non-adjacent singleton cluster got
    excluded, both (composed) if both happened, or None. Never discards
    silently.

    A singleton cluster is NOT automatically discarded: if it sits
    ADJACENT to the dominant cluster's own union -- within about one
    detected line height of its nearest edge -- it's kept and folded in
    instead, because a lone hit touching the known-good region is almost
    always a missing line (OCR caught only one line of a two-line
    subtitle on that particular probe), not noise. "Line height" is the
    median individual-extent height within the dominant cluster itself
    (not the union's own height, which can already include admitted
    multi-line members), and the gap is measured relative to it, not in
    absolute pixels, so this scales correctly across resolutions the same
    way BASELINE_CLUSTER_TOLERANCE_FRAC does. Reproduced: an isolated
    upper-line-only frame sitting ~10px above a ~50px-tall dominant
    cluster (gap ~0.2 line heights) was wrongly discarded before this,
    clipping the box back down to one line; a genuinely unrelated
    detection (reproduced: ~2.4 line heights away) still isn't adjacent
    and stays discarded -- see
    test_slay_stray_hit_geometry_stays_rejected_after_adjacency_fix.

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
        dominant_top = min(extents[i][1] for i in dominant)
        dominant_bottom = max(extents[i][3] for i in dominant)
        line_height = median(extents[i][3] - extents[i][1] for i in dominant)
        adjacency_limit = line_height * ADJACENT_SINGLETON_MAX_GAP_LINE_HEIGHTS
        for cluster in clusters[1:]:
            if len(cluster) >= 2:
                kept_idx.extend(cluster)
                position_flag = _compose_flag(position_flag, FLAG_MULTIPLE_POSITIONS)
                continue
            j = cluster[0]
            s_top, s_bottom = extents[j][1], extents[j][3]
            if s_bottom <= dominant_top:
                gap = dominant_top - s_bottom
            elif s_top >= dominant_bottom:
                gap = s_top - dominant_bottom
            else:
                gap = 0.0  # overlaps the dominant union already
            if gap <= adjacency_limit:
                kept_idx.append(j)
                position_flag = _compose_flag(position_flag, FLAG_MULTIPLE_POSITIONS)
            else:
                position_flag = _compose_flag(position_flag, FLAG_OUTLIER_DISCARDED)

    union = _bounding_union([extents[i] for i in kept_idx])
    return union, kept_idx, position_flag


def _union_extent_detailed(polys_per_frame, frame_h: float, cutoff_frac: float,
                            sample_times: list[float] | None = None,
                            ) -> tuple[tuple[float, float, float, float] | None, list[int],
                                       str | None, str | None]:
    """As _union_extent(), but returns the KEPT INDICES (into
    `polys_per_frame`/`sample_times`) that actually contributed to the
    union, instead of just their count -- detect_crop() uses this to build
    CropResult.hit_pts (see its docstring). Split out from _union_extent()
    so that function's existing 4-tuple (union, agreed_count,
    watermark_status, position_flag) contract -- unpacked by every caller
    in tests/test_detect_crop.py -- doesn't have to change just to plumb
    this one extra detail through to detect_crop().

    The union is None only when there are no in-band polys at all, OR the
    watermark status is "confirmed" (see _watermark_status()) -- an
    "uncertain" watermark status, or any position status, still returns a
    real union, since insufficient or conflicting evidence must not
    silently discard a detection.

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
        return None, [], None, None

    accepted = [extents[i] for i in accepted_idx]
    contributing_times = None
    if sample_times is not None and len(sample_times) == len(extents):
        contributing_times = [sample_times[i] for i in accepted_idx]

    watermark_status = _watermark_status(accepted, len(extents), frame_h, contributing_times)
    if watermark_status == "confirmed":
        return None, [], watermark_status, None

    union, kept_idx, position_flag = _baseline_cluster_union(extents, accepted_idx, frame_h)
    return union, kept_idx, watermark_status, position_flag


def _union_extent(polys_per_frame, frame_h: float, cutoff_frac: float,
                   sample_times: list[float] | None = None,
                   ) -> tuple[tuple[float, float, float, float] | None, int, str | None, str | None]:
    """Returns (union extent, count of contributing frames, watermark
    status, position status). Thin wrapper over _union_extent_detailed()
    that collapses its kept-indices list down to a count, preserving this
    function's existing public contract -- see _union_extent_detailed()
    for the full behaviour and for the kept indices themselves.
    """
    union, kept_idx, watermark_status, position_flag = _union_extent_detailed(
        polys_per_frame, frame_h, cutoff_frac, sample_times,
    )
    return union, len(kept_idx), watermark_status, position_flag


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
    """Two-sided shape agreement check: consensus may only relax the
    convergence stop (see _run_round()) when the raw union's height agrees
    with the series in BOTH directions, not just "not too tall".

    An earlier version only rejected a union that was too TALL relative to
    consensus (h_frac > med_h * CONSENSUS_MAX_HEIGHT_RATIO); a union that
    was too SHORT -- e.g. a one-line union checked against a two-line
    consensus -- passed silently, because "shorter than expected" cleared
    that same one-sided ratio test trivially. Reproduced: consensus
    (917/1080, 116/1080) (a two-line series) against a one-line union
    (977/1080, 56/1080) returned True, RELAXING the stop requirement at
    exactly the moment the consensus is evidence the box should be
    TALLER -- the opposite of what "consistent" should license. See
    test_consensus_never_relaxes_toward_a_union_shorter_than_the_series.
    """
    if not consensus:
        return True
    med_y = median(c[0] for c in consensus)
    med_h = median(c[1] for c in consensus)
    if med_h > 0 and h_frac > med_h * CONSENSUS_MAX_HEIGHT_RATIO:
        return False
    if med_h > 0 and h_frac < med_h / CONSENSUS_MAX_HEIGHT_RATIO:
        return False
    if abs(y_frac - med_y) > CONSENSUS_Y_DEVIATION_FRAC:
        return False
    return True


def consistent_with_consensus(box: tuple[int, int, int, int], frame_size: tuple[int, int],
                              consensus: list[tuple[float, float]]) -> bool:
    """Whether a finished crop `box` agrees with `consensus` under the rule
    detect_crop() applies before consensus may relax its convergence stop
    (_consistent_with_consensus(): height within CONSENSUS_MAX_HEIGHT_RATIO
    of the series median both ways, top within CONSENSUS_Y_DEVIATION_FRAC).

    `box` is (x, y, w, h) in the native pixels of a frame `frame_size`
    (width, height), as CropResult.box / CropResult.frame_size give them;
    `consensus` holds (y_frac, h_frac) entries, as detect_crop() takes them.
    For judging a result after the fact, e.g. a hint re-detection against
    its hint (core.jobs.apply). detect_crop() itself does not call this: it
    checks its raw union, so detection is unchanged. Raises ValueError
    without a frame height.
    """
    _width, height = frame_size
    if not height or height <= 0:
        raise ValueError(f"frame_size {frame_size!r} has no height")
    return _consistent_with_consensus(box[1] / height, box[3] / height, list(consensus))


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
                cancel_check: Callable[[], bool] | None = None,
                fetcher: _PersistentFrameFetcher | None = None,
                transfer=_PROBE,
                ) -> tuple[list, list[float], int, list[float]]:
    """Fetch+detect `times` in PROBE_BATCH_SIZE-sized batches, stopping
    once the raw in-band union has stopped growing -- its vertical edges
    within CONVERGENCE_VERTICAL_TOLERANCE_FRAC of where they stood when the
    stable streak began, see _union_is_stable() -- for
    CONVERGENCE_STABLE_ROUNDS consecutive batches (or just
    CONSENSUS_STABLE_ROUNDS, when an in-tolerance consensus is available --
    see the "Consensus RELAXES..." comment at the stop check below), or
    MAX_PROBES_PER_ROUND is hit, or `times` is exhausted, or `cancel_check`
    (a zero-argument callable returning truthy once cancellation has been
    requested -- stdlib only, e.g. wrapping a threading.Event or a plain
    flag; this module stays Qt-free, see the module docstring) starts
    returning True -- NEVER once an arbitrary hit count is reached, with
    or without consensus.

    `cancel_check` is polled once per iteration, BEFORE fetching the next
    batch -- so at most one already-in-flight batch (already-dispatched
    ffmpeg grabs + one det_engine.predict() call) still completes after
    cancellation is requested, not an unbounded number of them. A caller
    driving this from a background thread (see core/subtitle_detector.py)
    can therefore expect cancellation to take effect within roughly one
    batch, not only once the whole candidate list or MAX_PROBES_PER_ROUND
    is exhausted.

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
    with them. Nothing in this loop -- including the consensus check
    below -- may call aggregate_box() (or anything else that runs
    clustering) while probing is still in progress: an earlier version's
    consensus check did exactly that, evaluating a filtered/clustered
    provisional box instead of the raw evidence, which is a second way
    the same bug (a judged view manufacturing a stop decision) can creep
    back in even after this loop itself stopped filtering directly.

    `times` is walked in _spread_order(), not the order it's given in --
    spread is still useful here for getting temporal span (needed by the
    watermark check) established early, but it no longer decides WHICH
    hits contribute: it only decides the order batches are looked at in,
    and the convergence stop (not a hit count) decides when to stop
    looking, so a late-arriving batch is never silently skipped just for
    arriving late.

    `transfer`: the source's color_transfer (see _probe_source()), which
    decides whether one-shot grabs tone-map; probed when not given.

    `fetcher`: the persistent-container fetcher detect_crop() opened for
    this source (see _prefers_persistent_fetch()), or None for one-shot
    grabs. Either way each fetched frame is recorded -- in sample_pts and
    frame_times alike -- under its probe's requested time: a time that
    re-fetches exactly the analysed frame through this module's fetch layer
    (at-or-after, ms-rounded), which may precede the frame's true PTS by up
    to one frame (see CropResult.sample_pts for why not the PTS). Grabs that
    fail return no frame and are not recorded, but the probe budget
    (MAX_PROBES_PER_ROUND) counts requested probes, so they still spend it.

    Returns (polys_per_frame, sample_pts_used, raw_hit_count, frame_times).
    Polygons are already mapped to full-frame pixel coordinates.
    `sample_pts_used` lists every frame actually fetched, in probing order.
    `frame_times` has one entry per entry of `polys_per_frame`, that frame's
    recorded time (for the watermark temporal-spread check).
    """
    polys_per_frame: list[list] = []
    frame_times: list[float] = []
    sample_pts: list[float] = []
    consensus = consensus or []
    _, frame_h = frame_size
    cutoff_frac = float((settings or {}).get("bottom_half_cutoff", band_frac))
    raw_hits = 0
    stability_anchor: tuple[float, float, float, float] | None = None
    stable_rounds = 0
    attempted = 0

    times = _spread_order(times)

    i = 0
    while i < len(times):
        if attempted >= MAX_PROBES_PER_ROUND:
            break
        if cancel_check is not None and cancel_check():
            break

        chunk = times[i:i + PROBE_BATCH_SIZE]
        i += len(chunk)
        attempted += len(chunk)

        if fetcher is None:
            pairs, geometry = _grab_frames_with_times(
                video_path, chunk, band_frac, TARGET_HEIGHT, known_dims=known_dims,
                transfer=transfer,
            )
        else:
            pairs, geometry = fetcher.fetch(chunk, band_frac, TARGET_HEIGHT)
        # Only the probes that returned a frame, each under its requested
        # time: re-requesting that time (the UI slider, the filmstrip, the
        # full-frame retry below) lands on exactly the frame analysed here
        # (Task 2b requirement 4, see CropResult.sample_pts).
        sample_pts.extend(t for t, _frame in pairs)
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

        # Convergence stop: has the raw union grown -- beyond the tolerance,
        # measured from the start of the current stable streak -- since the
        # streak began? Any growth beyond it restarts the streak from the
        # grown union (see _union_is_stable()). A None union (no in-band
        # hits yet at all) is never "stable" -- there's nothing to converge
        # on, so keep probing until either a real union appears or the
        # probe budget/candidate list runs out.
        current_union = _bounding_union(extents)
        if current_union is None:
            stability_anchor = None
            stable_rounds = 0
        elif stability_anchor is not None and _union_is_stable(current_union, stability_anchor, frame_h):
            stable_rounds += 1
        else:
            stability_anchor = current_union
            stable_rounds = 0

        # Consensus RELAXES the convergence requirement -- it never
        # replaces it. A trustworthy cross-file consensus (>=3 already-
        # resolved files) whose median shape agrees with the RAW union
        # lowers how many consecutive stable batches are needed, from
        # CONVERGENCE_STABLE_ROUNDS down to CONSENSUS_STABLE_ROUNDS. It
        # can NEVER stop on a hit count alone (that was the original bug:
        # 2 hits plus a matching provisional box could stop probing after
        # a single batch, with zero stability confirmed at all), and the
        # check MUST read `current_union` -- the same raw, unclustered
        # union the plain convergence check above uses -- never a box
        # built via aggregate_box()/_baseline_cluster_union(), which
        # would evaluate a filtered/judged view instead of the evidence
        # actually gathered so far. Same principle as the convergence
        # stop itself: nothing may stop probing on a filtered view.
        required_stable_rounds = CONVERGENCE_STABLE_ROUNDS
        if current_union is not None and len(consensus) >= CONSENSUS_MIN_ENTRIES:
            y_frac = current_union[1] / frame_h
            h_frac = (current_union[3] - current_union[1]) / frame_h
            if _consistent_with_consensus(y_frac, h_frac, consensus):
                required_stable_rounds = min(required_stable_rounds, CONSENSUS_STABLE_ROUNDS)

        if current_union is not None and stable_rounds >= required_stable_rounds:
            break

    extents = _per_frame_extents(polys_per_frame, frame_h, cutoff_frac)
    raw_hits = sum(e is not None for e in extents)
    return polys_per_frame, sample_pts, raw_hits, frame_times


# --------------------------------------------------------------------------
# Evidence (CropResult.samples) -- read-only over what detection already
# decided; nothing here feeds back into the box, flags or times.
# --------------------------------------------------------------------------

def _count_text_rows(boxes) -> int:
    """Distinct text rows among (x, y, w, h) boxes. Two boxes share a row
    when their vertical extents overlap by at least
    TEXT_ROW_MIN_OVERLAP_FRAC of the smaller box's height; rows chain, so a
    box sharing a row with each of two others joins all three."""
    row_of = list(range(len(boxes)))

    def row(i: int) -> int:
        while row_of[i] != i:
            i = row_of[i]
        return i

    for i, (_xi, yi, _wi, hi) in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            _xj, yj, _wj, hj = boxes[j]
            overlap = min(yi + hi, yj + hj) - max(yi, yj)
            if overlap >= TEXT_ROW_MIN_OVERLAP_FRAC * min(hi, hj):
                row_of[row(j)] = row(i)
    return sum(1 for i in range(len(boxes)) if row(i) == i)


def _accepted_boxes(polys, frame_h: float, cutoff_frac: float) -> tuple[tuple[int, int, int, int], ...]:
    """(x, y, w, h) of each of one frame's polygons that the union accepts
    (see CropSample.boxes), sorted by (y, x). Each polygon goes through
    _per_frame_extents() itself, so a box is shown exactly when the
    aggregation counts that polygon. `polys` already passed the score
    threshold and are full-frame (see _run_round()).

    A polygon that cannot be parsed or rounded is left out rather than
    raised: evidence must never fail a detection whose result stands."""
    boxes = []
    for poly in polys:
        try:
            extent = _per_frame_extents([[poly]], frame_h, cutoff_frac)[0]
            if extent is not None:
                min_x, min_y, max_x, max_y = extent
                boxes.append((int(round(min_x)), int(round(min_y)),
                              int(round(max_x - min_x)), int(round(max_y - min_y))))
        except (TypeError, ValueError, OverflowError):
            pass
    return tuple(sorted(boxes, key=lambda b: (b[1], b[0], b[2], b[3])))


def _frame_of_sample(sample_times: list[float], frame_times: list[float]) -> dict[int, int]:
    """For one _run_round() call: which analysed frame (index into its
    polys_per_frame / frame_times) each recorded sample (index into its
    sample_pts) is. Both lists are in probing order; frame_times has one
    entry per frame the engine answered, so it equals sample_times unless
    the engine returned fewer results than frames. Matched in order, so a
    sample the engine did not answer gets no frame instead of the next
    sample's."""
    frame_of_sample: dict[int, int] = {}
    i = 0
    for j, t in enumerate(frame_times):
        while i < len(sample_times) and sample_times[i] != t:
            i += 1
        if i == len(sample_times):
            break
        frame_of_sample[i] = j
        i += 1
    return frame_of_sample


def _crop_samples(rounds: list[tuple[list[float], list, list[float], bool]], frame_h: float,
                  band_cutoff_frac: float, kept_idx: list[int]) -> list[CropSample]:
    """CropResult.samples from every _run_round() call detect_crop() made,
    in order, each as (sample_pts, polys_per_frame, frame_times, full-frame
    retry?). Each round's frames are judged against the band that round
    probed: the whole frame for the full-frame retry, `band_cutoff_frac`
    otherwise. `kept_idx` indexes the LAST round's frames, the only ones the
    union was built from."""
    samples: list[CropSample] = []
    for k, (sample_times, polys_per_frame, frame_times, full_frame) in enumerate(rounds):
        cutoff_frac = 0.0 if full_frame else band_cutoff_frac
        kept_frames = set(kept_idx) if k == len(rounds) - 1 else set()
        frame_of_sample = _frame_of_sample(sample_times, frame_times[:len(polys_per_frame)])
        for i, t in enumerate(sample_times):
            j = frame_of_sample.get(i)
            boxes = () if j is None else _accepted_boxes(polys_per_frame[j], frame_h, cutoff_frac)
            samples.append(CropSample(time=t, boxes=boxes, kept=j is not None and j in kept_frames,
                                      lines=_count_text_rows(boxes)))
    return samples


def detect_crop(video_path: str, duration_sec: float, det_engine,
                 consensus: list[tuple[float, float]] | None = None,
                 settings: dict | None = None,
                 cancel_check: Callable[[], bool] | None = None) -> CropResult:
    """Detect the subtitle crop box for `video_path`.

    `det_engine` must be an engine the caller holds a lease on
    (videocr.engine_registry) for the whole call: an engine must never serve
    two threads at once, and OCR workers run as threads in the same process.

    `cancel_check`, if given, is a zero-argument callable polled while the
    audio window is extracted (see vad.extract_audio_window(); cancelled
    there, the result is just "cancelled", with nothing probed), between
    probe batches (see _run_round()) AND between the fallback rounds
    below, so a caller driving several files from a background thread
    (core/subtitle_detector.py) can make Cancel take effect within roughly
    one batch of one file, not only between whole files.

    Orchestration: vad.probe_times() picks candidate timestamps ranked by
    likelihood of carrying dialogue -> they are fetched in batches (one-shot
    grabs, or persistent containers above the resolution crossover -- see
    _prefers_persistent_fetch()), already cropped to the bottom band and
    downscaled ->
    det_engine.predict() scores each frame's text polygons -> polys
    scoring >= 0.9 that also fall in-band are kept and mapped back to
    full-frame coordinates -> _run_round() stops once the raw (unfiltered)
    union's vertical extent has held steady, within
    CONVERGENCE_VERTICAL_TOLERANCE_FRAC, for CONVERGENCE_STABLE_ROUNDS (2) consecutive
    probe batches, or for just CONSENSUS_STABLE_ROUNDS (1) when
    `consensus` (a list of (y_frac, h_frac) from already-resolved files)
    has at least CONSENSUS_MIN_ENTRIES (3) entries and the raw union's own
    shape agrees with it in BOTH directions -- neither too tall nor too
    short relative to the series median (see _consistent_with_consensus())
    -> aggregate_box() unions the accepted polys into one padded, clamped
    crop.

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
    cluster got folded into the union alongside the dominant one -- either
    >=2 members, e.g. a genuinely repositioned subtitle, or a single hit
    kept because it sits adjacent to the dominant cluster, a likely
    missing line), "outlier-discarded?" (a single hit at a baseline
    nothing else shared, and not adjacent to the dominant cluster, was
    excluded from the union -- see _baseline_cluster_union()),
    "ceiling-exceeded" (in-band hits exist but the resulting box is too
    tall), or "low-agreement" (a box was built, but from fewer than
    LOW_AGREEMENT_HITS contributing frames).
    """
    orig_w, orig_h, transfer = _probe_source(video_path)
    known_dims = (orig_w, orig_h)
    frame_size = (orig_w, orig_h)

    sample_pts: list[float] = []
    flagged: str | None = None

    try:
        times = vad.probe_times(video_path, duration_sec, window_frac=(0.40, 0.60),
                                cancel_check=cancel_check)
    except vad.AudioExtractionCancelled:
        # Cancelled before a single probe: say so, and nothing else --
        # falling through would compose no-speech onto a file that was
        # never listened to.
        return CropResult(box=None, flagged=FLAG_CANCELLED, frame_size=frame_size,
                          cutoff_frac=float((settings or {}).get("bottom_half_cutoff", BOTTOM_HALF_CUTOFF)))
    speech_probing_available = bool(times)
    if not speech_probing_available:
        times = _uniform_probe_times(duration_sec)
        flagged = _compose_flag(flagged, FLAG_NO_SPEECH)

    # Opened once per file and shared by every round below; closed however
    # detection ends (including an exception from the engine), since each
    # container holds several decoded reference frames.
    fetcher = _open_frame_fetcher(video_path, known_dims)
    used_full_frame_retry = False
    # Every round's (sample_pts, polys_per_frame, frame_times, full-frame
    # retry?), for CropResult.samples only.
    rounds: list[tuple[list[float], list, list[float], bool]] = []
    try:
        polys_per_frame, used, raw_hits, frame_times = _run_round(
            video_path, times, det_engine, band_frac=BOTTOM_HALF_CUTOFF,
            consensus=consensus, frame_size=frame_size, settings=settings, known_dims=known_dims,
            cancel_check=cancel_check, fetcher=fetcher, transfer=transfer,
        )
        sample_pts.extend(used)
        rounds.append((used, polys_per_frame, frame_times, False))
        # Checked (and reused, not re-polled) once per round, right after that
        # round returns: cancel_check() cutting a round short is a real,
        # distinct reason a round found little or nothing -- composed
        # explicitly here (FLAG_CANCELLED) rather than left to be inferred
        # from an absent flag downstream, which is exactly how round 1's
        # cancellation went unflagged before this (see task-3 review round 3):
        # the round-2/round-3 entry gates below skip on `not cancelled`, so a
        # file cancelled during round 1 with zero hits never triggered
        # FLAG_SPEECH_PROBES_EXHAUSTED or FLAG_TOP_POSITIONED either, and
        # nothing else was there to explain the resulting box=None.
        cancelled = _is_cancelled(cancel_check)
        if cancelled:
            flagged = _compose_flag(flagged, FLAG_CANCELLED)

        if raw_hits == 0 and speech_probing_available and not cancelled:
            uniform_times = _uniform_probe_times(duration_sec)
            polys_per_frame, used, raw_hits, frame_times = _run_round(
                video_path, uniform_times, det_engine, band_frac=BOTTOM_HALF_CUTOFF,
                consensus=consensus, frame_size=frame_size, settings=settings,
                known_dims=known_dims, cancel_check=cancel_check, fetcher=fetcher,
                transfer=transfer,
            )
            sample_pts.extend(used)
            rounds.append((used, polys_per_frame, frame_times, False))
            flagged = _compose_flag(flagged, FLAG_SPEECH_PROBES_EXHAUSTED)
            cancelled = _is_cancelled(cancel_check)
            if cancelled:
                flagged = _compose_flag(flagged, FLAG_CANCELLED)

        if raw_hits == 0 and not cancelled:
            retry_times = _uniform_probe_times(duration_sec) if not sample_pts else sample_pts
            # Reuse the already-attempted timestamps for the full-frame retry
            # rather than probing new ones -- we already know these timestamps
            # exist in the video; band_frac=1.0 just widens what we look at.
            retry_times = sorted(set(retry_times))
            polys_per_frame, used, raw_hits, frame_times = _run_round(
                video_path, retry_times, det_engine, band_frac=1.0,
                consensus=consensus, frame_size=frame_size,
                settings={**(settings or {}), "bottom_half_cutoff": 0.0},
                known_dims=known_dims, cancel_check=cancel_check, fetcher=fetcher,
                transfer=transfer,
            )
            sample_pts.extend(used)
            rounds.append((used, polys_per_frame, frame_times, True))
            flagged = _compose_flag(flagged, FLAG_TOP_POSITIONED)
            used_full_frame_retry = True
            if _is_cancelled(cancel_check):
                flagged = _compose_flag(flagged, FLAG_CANCELLED)
    finally:
        if fetcher is not None:
            fetcher.close()

    cutoff_frac = 0.0 if used_full_frame_retry else float(
        (settings or {}).get("bottom_half_cutoff", BOTTOM_HALF_CUTOFF)
    )
    envelope_extent, kept_idx, watermark_status, position_flag = _union_extent_detailed(
        polys_per_frame, orig_h, cutoff_frac, frame_times,
    )
    agreed = len(kept_idx)
    # Chronological order, not discovery order: kept_idx follows
    # _cluster_by_baseline()'s baseline-sorted order, not probe order.
    hit_pts = sorted(frame_times[i] for i in kept_idx)
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

    if box is None:
        # Gated on watermark_status directly, NOT on `agreed > 0`: a
        # CONFIRMED watermark has evidence (every sampled frame matched
        # it), but _union_extent_detailed() deliberately returns an empty
        # kept_idx for it (nothing is "kept" into a union -- see its
        # docstring), so `agreed` (== len(kept_idx)) is always 0 in this
        # branch. Inferring "was this a watermark" from `agreed > 0` was
        # wrong the moment agreed stopped being a raw evidence count and
        # became a kept-union count -- see task-3 review round 2.
        if watermark_status == "confirmed":
            flagged = _compose_flag(flagged, FLAG_STATIC_CONTENT)
        elif agreed > 0:
            # Real hits exist (a genuine, non-watermark union was found),
            # but aggregate_box()'s own ceiling check rejected the
            # resulting padded/floored box as too tall. An "uncertain"
            # watermark union that happens to fit under the ceiling never
            # reaches this branch at all (box is not None then; see the
            # "uncertain" flag composition below instead).
            flagged = _compose_flag(flagged, FLAG_CEILING_EXCEEDED)
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

    # Structural post-condition, not a specific-case fix: every box=None
    # exit above is expected to have composed its own specific flag by
    # this point. If one didn't -- because this function grows a new
    # no-box path later and whoever adds it forgets to flag it, not
    # because of any case currently known -- this is the last line of
    # defence against silently returning box=None with no explanation at
    # all. See test_no_known_scenario_reaches_the_fallback_flag() and
    # test_safety_net_fallback_flag_fires_for_a_truly_unanticipated_no_box_path()
    # in tests/test_detect_crop.py: FLAG_UNKNOWN_REJECTION must never
    # appear for any currently-known scenario.
    if box is None and flagged is None:
        logger.warning(
            "%s: detect_crop() rejected the box with no flag composed -- "
            "this function's flag coverage is incomplete for whatever path "
            "just ran; falling back to %r so the result is never silently "
            "unexplained",
            video_path, FLAG_UNKNOWN_REJECTION,
        )
        flagged = _compose_flag(flagged, FLAG_UNKNOWN_REJECTION)

    # Evidence only, built once every field above is final.
    samples = _crop_samples(
        rounds, orig_h, float((settings or {}).get("bottom_half_cutoff", BOTTOM_HALF_CUTOFF)), kept_idx,
    )

    return CropResult(
        box=box,
        sample_pts=sample_pts,
        envelope=envelope,
        agreed=agreed,
        probes_used=len(sample_pts),
        flagged=flagged,
        hit_pts=hit_pts,
        frame_size=frame_size,
        samples=samples,
        cutoff_frac=cutoff_frac,
    )
