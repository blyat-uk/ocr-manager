"""Frames exactly as the OCR pass sees them.

videocr/video.py's `Video.run_ocr` / `_frame_producer` decodes through
`videocr.pyav_adapter.Capture`, crops, downscales tall crops, masks every
frame to the pixels whose channels are all >= the brightness threshold, and
only OCRs frames whose masked centre square trips a Laplacian-variance gate.
Anything that tunes or previews a brightness threshold has to look at those
same pixels, or it is tuned for frames OCR never sees. This module mirrors
each step verbatim (constants imported, never copied) and is pinned against
the real `run_ocr` by tests/test_detect_ocr_view.py:

- `crop_geometry`   run_ocr's crop inference, clamping and decode-scale maths
- `grab_ocr_strips` the same Capture, decode height and in-graph crop request
- `to_ocr_view`     the producer's downscale of tall crops
- `mask`            the brightness filter
- `gate_fires`      the text gate on a masked frame

plus the sampling helpers brightness detection uses (`keep_spans`,
`sample_times`, `video_duration`). Stage 3's brightness preview tiles build
on the same functions.

The logic stays inline in videocr/video.py and is copied here rather than
shared, so the OCR pass itself stays byte-identical. No Qt imports.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import av
import cv2
import numpy as np

from videocr.pyav_adapter import DECODE_TARGET_HEIGHT, Capture
from videocr.video import MIN_CROP_HEIGHT, MIN_LAPLACIAN_VARIANCE, TARGET_VIDEO_HEIGHT

logger = logging.getLogger(__name__)

# Failures that drop a frame (or every frame of a file that cannot be opened)
# instead of raising -- the same families core/detect/crop.py treats as a
# failed grab. A RuntimeError from Capture's filter graph is deliberately NOT
# here: that refusal means the OCR pass itself cannot decode the file.
FETCH_ERRORS = (av.error.FFmpegError, OSError, ValueError, EOFError)

SAMPLE_EDGE_EXCLUDE_FRAC = 0.10   # without keep ranges, skip the first/last 10%
SAMPLE_WORKERS = 4                # capture containers opened in parallel


class _OpenCapture:
    """`with _OpenCapture(path, **kwargs) as cap:` is `with Capture(path,
    **kwargs) as cap:`, except that a FETCH_ERRORS failure while CLOSING the
    container is logged instead of raised. By then every frame the block
    needed has been read; a container that fails to close must not throw
    those frames away or abort detection. Failures while opening propagate
    unchanged, and an exception raised inside the block still propagates."""

    def __init__(self, video_path: str, **kwargs):
        self._video_path = video_path
        self._capture = Capture(video_path, **kwargs)

    def __enter__(self):
        return self._capture.__enter__()

    def __exit__(self, *exc_info):
        try:
            return self._capture.__exit__(*exc_info)
        except FETCH_ERRORS as exc:
            logger.warning("%s: capture failed to close (%s: %s)", self._video_path, type(exc).__name__, exc)
            return False


def to_ocr_view(frame: np.ndarray) -> np.ndarray:
    """The frame as the OCR pass holds it just before masking.

    Verbatim mirror of the "Downscale for faster OCR" block in
    videocr/video.py `_frame_producer` (the `frame_h > TARGET_VIDEO_HEIGHT`
    branch just above "Apply brightness filter"), including its float
    arithmetic -- int(1142 * (720 / 1142)) is 719, not 720.
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


def mask(frame: np.ndarray, t: int) -> np.ndarray:
    """The OCR pass's brightness filter (videocr/video.py, "Apply brightness
    filter"): keep pixels whose every channel is >= t."""
    return cv2.bitwise_and(frame, frame, mask=cv2.inRange(frame, (t,) * 3, (255,) * 3))


def gate_fires(masked: np.ndarray) -> bool:
    """The OCR pass's text gate on an already-masked frame (videocr/video.py,
    "Center square check" / "Use Laplacian variance"): grey, the h x h square
    horizontally centred, Laplacian variance >= MIN_LAPLACIAN_VARIANCE. A
    frame this returns False for is never OCR'd."""
    grey = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape
    center_x_start = (w - h) // 2
    center_x_end = center_x_start + h
    center = grey[:, center_x_start:center_x_end]
    return bool(cv2.Laplacian(center, cv2.CV_64F).var() >= MIN_LAPLACIAN_VARIANCE)


def crop_geometry(width: int, height: int, crop_box) -> tuple[int | None, tuple[int, int, int, int]] | None:
    """(decode_target_height, (x0, y0, x1, y1) in decode-output pixels) for
    `crop_box` (x, y, w, h in native pixels) on a width x height source,
    exactly as Video.run_ocr derives them (the "infer missing crop
    parameters" / "clamp" / "Scale crop coordinates from native to decode
    resolution" blocks, truncating with int()). None when the box has no
    area -- run_ocr would then fall back to the bottom third."""
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


# --------------------------------------------------------------------------
# Sampling
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


def keep_spans(duration: float, time_ranges) -> list[tuple[float, float]]:
    """The (start, end) seconds to sample from, ascending, overlaps merged.

    No ranges (None or empty) means the whole file minus its first and last
    10%. Ranges are (start, end) pairs as FileConfig holds them, or
    {"start", "end"} mappings as .ocr.json stores them; open ends run to the
    file's edges. Ranges that resolve to nothing inside the file return []
    -- never a silent fallback to the whole file, whose intros and outros can
    carry differently styled text (measured on Martial Master's opening
    lyrics)."""
    if not time_ranges:
        return [(duration * SAMPLE_EDGE_EXCLUDE_FRAC, duration * (1.0 - SAMPLE_EDGE_EXCLUDE_FRAC))]
    spans = []
    for pair in time_ranges:
        start, end = (pair.get("start"), pair.get("end")) if isinstance(pair, dict) else pair
        s = _parse_time(start)
        e = _parse_time(end)
        s = 0.0 if s is None else max(0.0, min(s, duration))
        e = duration if e is None else max(0.0, min(e, duration))
        if e > s:
            spans.append((s, e))
    merged: list[tuple[float, float]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def sample_times(duration: float, time_ranges, n: int, phase: float = 0.5) -> list[float]:
    """`n` times spread evenly over keep_spans(duration, time_ranges), in
    proportion to the spans' lengths. Ascending. The concatenated spans are
    cut into n equal slots and each time sits `phase` of the way into its
    slot: the default 0.5 is the slot middle, and other phases give a later
    sampling round times that fall between an earlier round's. [] when the
    ranges select nothing."""
    spans = keep_spans(duration, time_ranges)
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


def video_timing(video_path: str) -> tuple[float, float]:
    """(duration, fps) as the OCR pass counts them: frame count / fps from
    Capture. Raises FETCH_ERRORS when the file cannot be opened."""
    with _OpenCapture(video_path) as cap:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
    return (frames / fps if fps else 0.0), fps


def video_duration(video_path: str) -> float:
    """Duration as the OCR pass counts it (see video_timing)."""
    return video_timing(video_path)[0]


def grab_ocr_strips(video_path: str, crop_box, times: list[float]) -> list[np.ndarray]:
    """Strips only; see grab_ocr_strips_at."""
    return [strip for _, strip in grab_ocr_strips_at(video_path, crop_box, times)]


def grab_ocr_strips_at(video_path: str, crop_box, times: list[float]) -> list[tuple[float, np.ndarray]]:
    """Crop strips at `times`, pixel-identical to what the OCR pass hands its
    brightness filter for the same frames: opened through the same `Capture`
    with the same decode_target_height and in-graph crop request as
    Video.run_ocr (so the same HDR tone map, 10-bit conversion and 4K decode
    downscale apply), sliced the same way when the capture could not crop in
    its graph, and downscaled with to_ocr_view().

    Why not core.detect.crop.grab_frames(): it grabs a full-width bottom band
    through a different filter order (crop -> scale -> bgr24, or the system
    ffmpeg CLI), so it does not reproduce the OCR pass's pixels. Measured:
    identical on Slay the Gods (1080p 8-bit), but 87-90% of pixels off by up
    to 13 levels on XWZ (4K 10-bit) and a different frame altogether on Jinwu
    Guard (h264 MKV). See task-4-report.md.

    `times` map to frame index round(t * fps), which is then sought exactly
    as the OCR pass seeks its range start (Capture.set(CAP_PROP_POS_FRAMES)).
    The OCR pass itself TRUNCATES a range start (int(t * fps), see
    videocr.utils.get_frame_index), so for a time that is not a whole frame
    this can return the frame after the one the OCR pass would start on.
    Sampling does not care which of two adjacent frames it gets; a preview of
    a range boundary would. Returns (requested time,
    strip) pairs in the order of `times`. A frame that cannot be read, seeked
    to or opened (FETCH_ERRORS) is dropped and logged, never raised; a file
    that cannot be opened at all returns [].
    """
    if not times:
        return []
    try:
        with _OpenCapture(video_path) as probe:
            width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = probe.get(cv2.CAP_PROP_FPS)
    except FETCH_ERRORS as exc:
        logger.warning("%s: cannot open for strip sampling (%s: %s)", video_path, type(exc).__name__, exc)
        return []
    geometry = crop_geometry(width, height, crop_box)
    if geometry is None:
        logger.warning("%s: crop box %s has no area", video_path, crop_box)
        return []
    decode_height, (x0, y0, x1, y1) = geometry
    graph_crop = (x0, y0, x1 - x0, y1 - y0)

    order = sorted(range(len(times)), key=lambda i: times[i])
    workers = min(SAMPLE_WORKERS, len(order))
    results: list[np.ndarray | None] = [None] * len(times)

    def read_chunk(cap, chunk: list[int]) -> None:
        last_index, last_strip = None, None
        for i in chunk:
            index = max(0, int(round(times[i] * fps)))
            if index == last_index and last_strip is not None:
                results[i] = last_strip.copy()
                continue
            try:
                if index > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
            except FETCH_ERRORS as exc:
                logger.warning("%s: dropped t=%.3f (%s: %s)", video_path, times[i], type(exc).__name__, exc)
                continue
            if not ok or frame is None:
                logger.warning("%s: could not read a frame at t=%.3f", video_path, times[i])
                continue
            if not getattr(cap, "_crop_slice", None):
                frame = frame[y0:y1, x0:x1]
            strip = np.ascontiguousarray(to_ocr_view(frame))
            results[i] = strip
            last_index, last_strip = index, strip

    def work(chunk: list[int]) -> None:
        # Each worker seeks forward through its own ascending share of the
        # times; one container per thread (PyAV releases the GIL to decode).
        # Only OPENING the container is guarded here: seek/read failures are
        # handled per frame in read_chunk, and anything else raised there is a
        # bug to surface, not a capture failure to log away. A failure to
        # CLOSE it is logged by _OpenCapture: the chunk's frames are read.
        with ExitStack() as stack:
            try:
                cap = stack.enter_context(
                    _OpenCapture(video_path, decode_target_height=decode_height, crop_rect=graph_crop))
            except FETCH_ERRORS as exc:
                logger.warning("%s: dropped %d frame(s), capture failed to open (%s: %s)",
                               video_path, len(chunk), type(exc).__name__, exc)
                return
            read_chunk(cap, chunk)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocr-view-grab") as pool:
        list(pool.map(work, [order[k::workers] for k in range(workers)]))
    return [(times[i], strip) for i, strip in enumerate(results) if strip is not None]
