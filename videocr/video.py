from __future__ import annotations
from typing import List
from collections import Counter
from queue import Queue, Empty
from threading import Thread
import cv2
import numpy as np
import time

from . import engine_registry, utils
from .models import PredictedFrames, PredictedSubtitle

from .pyav_adapter import Capture, DECODE_TARGET_HEIGHT

# Batch size for OCR processing - higher values = faster but more GPU memory.
# Smaller batches flush more often, so OCR inference on an early batch
# overlaps with the producer thread still decoding later frames, instead of
# (at a batch size this clip never fills mid-run) all inference happening
# serially after decode is already done. Measured on the reference episode:
# this pipelining keeps total run_ocr wall time roughly flat to mildly
# better at BATCH_SIZE 32 vs 256, NOT because the producer does less work -
# the producer thread's own wall time (self._producer_time) actually goes
# UP at the smaller batch size, on both the old and new code, because
# queue.put() blocks more often while the consumer is busy inside a
# just-triggered OCR call instead of draining the queue. Output is
# byte-identical across batch sizes - see tests/test_retry_determinism.py -
# because nothing in the pipeline keeps batch-shaped state;
# _candidates_admitted() (above) is what makes that true for the candidate
# buffer specifically.
BATCH_SIZE = 32

# Maximum RAM to use for frame buffer (10GB)
MAX_BUFFER_BYTES = 10 * 1024**3

# Target video height for downscaling (720p for speed, maintains OCR accuracy)
TARGET_VIDEO_HEIGHT = 720

# Minimum crop height after scaling to maintain OCR accuracy
# Text smaller than this becomes difficult for OCR to read reliably
MIN_CROP_HEIGHT = 150

# Maximum alternative frames OCR'd for one low-confidence subtitle
MAX_CANDIDATES = 8

# Ceiling on the alternative-reading frames buffered for ONE subtitle.
#
# A candidate is held from the moment the producer offers it until the batch
# it belongs to has been OCR'd and resolved, so the standing worst case is
# MAX_CANDIDATES x BATCH_SIZE = 256 frames on top of the batch's own 32 and
# the frame queue. At the reference crop geometry (1344x53 BGR, 213,696 bytes
# a frame) that is ~52 MB; with use_fullframe the candidates are whole
# 1280x720 pictures (2,764,800 bytes) and it reaches ~675 MB -- per worker,
# and OCRManager runs several workers at once. (These figures move with
# BATCH_SIZE, which is exactly the point of the budget below: nothing else
# in the pipeline should.) 32 subtitles each holding a couple of seconds of
# similar frames is only ~1 minute of dialogue, so this is a routine batch
# boundary, not a pathological one.
#
# The budget is PER SUBTITLE, deliberately, not a running total across the
# batch. A running total is refilled at the batch flush, and flushes happen
# every BATCH_SIZE subtitles, so a smaller BATCH_SIZE refills it more often
# and admits strictly more candidates - measured on a synthetic stream
# offering 80 candidates against a 25-frame budget: BATCH_SIZE 2 -> 80
# admitted, 5 -> 50, 10 -> 25, 256 -> 25. Since _resolve_candidates can
# rewrite a subtitle's text, that makes the .ass a function of BATCH_SIZE,
# which is exactly what this branch exists to rule out. A per-subtitle
# budget has no batch state in it at all, so where the flush boundaries
# fall cannot matter.
#
# 2 MiB a subtitle bounds the batch at 2 MiB x BATCH_SIZE = 64 MiB (at the
# current BATCH_SIZE of 32; this scales with it, same as the worst case
# above). Spread over MAX_CANDIDATES it admits frames up to 262,144 bytes
# (87,381 px) at full strength; past that a subtitle simply gets fewer
# candidates, via _candidates_admitted(). This DOES clip real geometries,
# and saying so is the point:
#
#   1344x53   (the golden/reference crop)  71,232 px -> all 8 candidates
#   1344x66   (13 rows taller)             88,704 px -> 7
#   1920x45   (a wide, thin band)          86,400 px -> 8
#   1920x46                                88,320 px -> 7
#   1920x106  (a two-line subtitle region) 203,520 px -> 3
#   1280x720  (use_fullframe)              921,600 px -> 0
#
# So the reference crop clears the threshold with about 23% to spare, but
# crop mode in general does not, and use_fullframe loses the best-of-N path
# entirely rather than spending ~675 MB a worker to keep it (at BATCH_SIZE
# 32; ~5.3 GiB at the old 256). Whatever a geometry gets, it gets the same
# amount at every BATCH_SIZE - per-subtitle bytes, not per-batch frames, is
# what makes that true.
MAX_CANDIDATE_BYTES_PER_SUBTITLE = 2 * 1024**2


def _candidates_admitted(frame) -> int:
    """How many candidates one subtitle may buffer, given its frame size.

    A pure function of the frame and two module constants - no batch state,
    no queue depth, no elapsed time. That is what makes the admitted set,
    and therefore the OCR output, identical for a given video and settings
    at every BATCH_SIZE, on a loaded machine or an idle one.
    """
    return min(MAX_CANDIDATES,
               MAX_CANDIDATE_BYTES_PER_SUBTITLE // frame.nbytes)


# Confidence advantage (on the 0-100 scale) a candidate reading must hold over
# the original before it may replace it *without* agreement from other frames.
#
# Chosen from the noise floor of the population this path actually sees, which
# is only subtitles scoring *below* conf_threshold. Measured over a reference
# episode: two frames of one subtitle that read the exact same text - so the
# whole difference is measurement noise - disagree by a median of 0.85 points,
# 7.59 at the 90th percentile and 8.07 at worst. A margin under that band lets
# noise rewrite text. 10.0 clears the worst observed spread with headroom,
# while still admitting a decisively clearer frame (the recoveries this path
# is for lead by 20 points and more).
CANDIDATE_CONFIDENCE_MARGIN = 10.0

# Maximum gap between frames to merge into same subtitle (in seconds)
# 0.3s allows for occasional OCR failures without splitting subtitles
SUBTITLE_MERGE_GAP_SECONDS = 0.3

# Minimum Laplacian variance in center region to consider as having text
# Laplacian measures edge sharpness - text strokes create strong variance
# After brightness filtering, empty frames have variance ≈ 0, any text >> 0
MIN_LAPLACIAN_VARIANCE = 50

# Minimum subtitle duration in seconds to filter out noise
# Real subtitles are at least 0.2-0.5 seconds
MIN_SUBTITLE_DURATION = 0.15

# --- Exact center-mask pre-gate (LOOKING state only) --------------------
#
# The brightness filter's mask is cv2.inRange(frame, (t,t,t), (255,255,255)):
# a pixel survives only if EVERY channel is >= t, i.e. min(B,G,R) >= t.
# cv2.inRange is pointwise (per-pixel), so it commutes with slicing: running
# it on just the center square gives exactly the same result, there, as
# running it on the whole frame and then slicing the center out. So if the
# center square's own mask is empty, that region of the REAL, full-frame
# mask is empty too - not approximately, exactly, no reconstruction or
# numerical margin involved - so the greyscale conversion and Laplacian
# variance the full pipeline would compute over that region are 0 (a
# uniformly-zero array has zero variance), and raw_detection is False.
# _center_mask_is_empty() below checks only the small center square instead
# of paying for the mask/greyscale/Laplacian pipeline over the whole frame
# to find that out.
#
# (An earlier version of this gate reconstructed a BT.709 luma value from
# the center square and compared it to a derived floor. That was strictly
# worse on every axis: reconstructing Y' needs three float32 copies of a
# non-contiguous view plus a multiply-add-divide pass, where inRange over
# the same region is a single, already-existing primitive; the floor is a
# NECESSARY-only bound with a numerical margin, where the center mask is
# the EXACT condition with no margin needed at all, so it also rejects
# strictly more frames. It bought nothing and cost more, including at
# use_fullframe where it net-lost to the full pipeline it was meant to
# avoid - see tests/test_luma_gate.py.)
#
# Only valid while LOOKING for a subtitle to start. TRACKING an existing
# one still needs the full masked grey frame for its similarity check
# against the previous frame, so the gate is never applied there.


def _center_mask_is_empty(center_bgr: np.ndarray, threshold: int) -> bool:
    """True iff no pixel in `center_bgr` has every channel >= threshold.

    Exactly cv2.inRange(center_bgr, (threshold,)*3, (255,255,255)) having
    no nonzero entries - i.e. the same per-channel min-threshold mask the
    brightness filter itself builds, restricted to just the small center
    square instead of the whole (possibly much larger) frame.
    """
    return cv2.countNonZero(cv2.inRange(
        center_bgr, (threshold,) * 3, (255,) * 3)) == 0


class Video:
    path: str
    lang: str
    use_fullframe: bool
    det_model_dir: str
    rec_model_dir: str
    num_frames: int
    fps: float
    height: int
    width: int
    pred_frames: List[PredictedFrames]
    pred_subs: List[PredictedSubtitle]

    def __init__(self, path: str, det_model_dir: str, rec_model_dir: str):
        self.path = path
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        # Backing fields for the num_frames/fps/height/width properties
        # below. None means "not probed yet". run_ocr() fills these in
        # from the one container open it already does for real decoding
        # (see there), instead of from a second, metadata-only Capture
        # this constructor used to open just to read them - one container
        # open per run_ocr() call instead of two. A caller that reads one
        # of these before run_ocr() has run (videocr/api.py's only_labels
        # path never calls run_ocr() at all) still gets a correct value,
        # via _probe_metadata()'s own on-demand, cached-thereafter open.
        self._num_frames = None
        self._fps = None
        self._height = None
        self._width = None

    def _probe_metadata(self) -> None:
        """Fill in whichever of num_frames/fps/height/width are still
        unset, from their own, dedicated, metadata-only container open.

        Only touches fields that are still None, and only opens a
        container at all if at least one is: a caller that has already
        set some of these directly (e.g. tests/test_candidate_buffer.py
        assigns all four before calling run_ocr(); a caller could just as
        well set only some) keeps what it set instead of a property read
        for the one it didn't silently overwriting it with the real
        file's values as a side effect.

        Only reached when something reads one of the properties below
        while at least one of the four is still unset, before run_ocr()
        has already populated them from the capture it opens for real
        decoding - e.g. videocr/api.py's only_labels path.
        """
        if None not in (getattr(self, '_num_frames', None), getattr(self, '_fps', None),
                        getattr(self, '_height', None), getattr(self, '_width', None)):
            return
        with Capture(self.path) as v:
            if getattr(self, '_num_frames', None) is None:
                self._num_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            if getattr(self, '_fps', None) is None:
                self._fps = v.get(cv2.CAP_PROP_FPS)
            if getattr(self, '_height', None) is None:
                self._height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if getattr(self, '_width', None) is None:
                self._width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def num_frames(self) -> int:
        if getattr(self, '_num_frames', None) is None:
            self._probe_metadata()
        return self._num_frames

    @num_frames.setter
    def num_frames(self, value) -> None:
        self._num_frames = value

    @property
    def fps(self) -> float:
        if getattr(self, '_fps', None) is None:
            self._probe_metadata()
        return self._fps

    @fps.setter
    def fps(self, value) -> None:
        self._fps = value

    @property
    def height(self) -> int:
        if getattr(self, '_height', None) is None:
            self._probe_metadata()
        return self._height

    @height.setter
    def height(self, value) -> None:
        self._height = value

    @property
    def width(self) -> int:
        if getattr(self, '_width', None) is None:
            self._probe_metadata()
        return self._width

    @width.setter
    def width(self, value) -> None:
        self._width = value

    def run_ocr(self, use_gpu: bool, lang: str, time_start: str, time_end: str, conf_threshold: int, use_fullframe: bool, brightness_threshold: int, similar_image_threshold: float, similar_pixel_threshold: int, frames_to_skip: int, crop_x: int, crop_y: int, crop_width: int, crop_height: int, progress=None, subtitle_callback=None, cancel_event=None, ocr_engine=None) -> None:
        """OCR the dialogue region of one time range into `self.pred_frames`.

        `ocr_engine`, when given, must be an engine the caller holds a lease
        on (`videocr.engine_registry`) for at least the whole of this call --
        videocr/api.py passes the one it leases for a file's every range and
        label pass. Without one, this call leases its own and returns it when
        it finishes. Engines are never shared between concurrent callers, and
        nothing here keeps the engine (or any result from it) past the call.
        """
        args = (use_gpu, lang, time_start, time_end, conf_threshold, use_fullframe,
                brightness_threshold, similar_image_threshold, similar_pixel_threshold,
                frames_to_skip, crop_x, crop_y, crop_width, crop_height)
        options = dict(progress=progress, subtitle_callback=subtitle_callback, cancel_event=cancel_event)
        if ocr_engine is not None:
            self._run_ocr_with_engine(ocr_engine, *args, **options)
            return
        with engine_registry.lease_ocr_engine(lang, self.det_model_dir, self.rec_model_dir, use_gpu) as ocr:
            self._run_ocr_with_engine(ocr, *args, **options)

    def _run_ocr_with_engine(self, ocr, use_gpu: bool, lang: str, time_start: str, time_end: str, conf_threshold: int, use_fullframe: bool, brightness_threshold: int, similar_image_threshold: float, similar_pixel_threshold: int, frames_to_skip: int, crop_x: int, crop_y: int, crop_width: int, crop_height: int, progress=None, subtitle_callback=None, cancel_event=None) -> None:
        conf_threshold_percent = float(conf_threshold / 100)
        self.lang = lang
        self.use_fullframe = use_fullframe
        self.pred_frames = []
        self._last_emitted_idx = 0

        # Batch accumulation for efficient OCR processing
        batch_frames = []
        batch_start_indices = []
        batch_end_indices = []
        batch_pts_start = []  # PTS values for batch timing
        batch_pts_end = []
        # Alternative readings buffered per batch slot, parallel to batch_frames.
        # batch_candidates[i] holds the candidate frames offered for the
        # subtitle whose first frame is batch_frames[i].
        batch_candidates = []
        # Candidates for the subtitle currently being tracked by the producer;
        # attached to the tail batch slot when that subtitle ends.
        pending_candidates = []

        # Profiling variables
        self._ocr_time = 0.0
        self._queue_wait_time = 0.0
        self._producer_time = 0.0  # Will be set by producer thread
        profiling_start = time.perf_counter()

        # Single container open for this range. decode_target_height is
        # always requested at the module's standard target: PyAVCapture
        # only actually downscales when the source is taller than it (see
        # its __enter__), so this is a no-op for <=1080p sources and still
        # gets 4K+ sources their decode-time downscale. Container metadata
        # (num_frames/fps/height/width) is read from this same open capture
        # right after opening it, instead of from the separate metadata-only
        # Capture Video.__init__ used to open - one container open instead
        # of two.
        #
        # The crop can't be passed to the constructor any more, though:
        # doing so needs it pre-scaled to decode-output coordinates, which
        # needs self.height known *before* opening - exactly the
        # chicken-and-egg this change removes. Instead it's configured
        # below, after opening, via Capture.configure_crop() (PyAVCapture
        # exposes this so a caller can plan+build the crop into the filter
        # graph once it knows one, as long as that happens before the
        # first read()/set() - see pyav_adapter.py). That keeps the graph
        # -level crop optimisation working for every source that needs a
        # filter graph anyway (tone-mapped and/or downscaled), which is
        # the only case PyAVCapture._plan_crop ever plans a graph crop for
        # in the first place; the golden/determinism cases (1080p SDR, no
        # downscale) never engaged it even before this file existed, so
        # they see no difference either way.
        with Capture(self.path, decode_target_height=DECODE_TARGET_HEIGHT) as v:
            # First use of this Video (or a caller that pre-populated some
            # or all of these itself, e.g. tests/test_candidate_buffer.py
            # sets all four before calling run_ocr()): read whichever of
            # num_frames/fps/height/width are still unset from the capture
            # just opened for decoding, rather than a second, metadata-only
            # one. Each field is guarded independently so a caller that set
            # only some of these keeps what it set - a single all-or-
            # nothing guard here would silently overwrite the rest of a
            # partial pre-population with this file's real values.
            if getattr(self, '_num_frames', None) is None:
                self.num_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            if getattr(self, '_fps', None) is None:
                self.fps = v.get(cv2.CAP_PROP_FPS)
            if getattr(self, '_height', None) is None:
                self.height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if getattr(self, '_width', None) is None:
                self.width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))

            ocr_start = utils.get_frame_index(time_start, self.fps) if time_start else 0
            ocr_end = utils.get_frame_index(time_end, self.fps) if time_end else self.num_frames

            if ocr_end < ocr_start:
                raise ValueError("time_start is later than time_end")
            num_ocr_frames = ocr_end - ocr_start

            crop_x_start = None
            crop_y_start = None
            crop_x_end = None
            crop_y_end = None

            if not self.use_fullframe:
                if not all(p is None for p in [crop_x, crop_y, crop_width, crop_height]):
                    # infer missing crop parameters
                    inferred_x = 0 if crop_x is None else crop_x
                    inferred_y = 0 if crop_y is None else crop_y
                    inferred_width = (self.width - inferred_x) if crop_width is None else crop_width
                    inferred_height = (self.height - inferred_y) if crop_height is None else crop_height

                    # clamp to valid ranges
                    inferred_x = max(0, min(int(inferred_x), self.width))
                    inferred_y = max(0, min(int(inferred_y), self.height))
                    inferred_width = max(0, min(int(inferred_width), self.width - inferred_x))
                    inferred_height = max(0, min(int(inferred_height), self.height - inferred_y))
                    if inferred_width > 0 and inferred_height > 0:
                        crop_x_start = inferred_x
                        crop_y_start = inferred_y
                        crop_x_end = inferred_x + inferred_width
                        crop_y_end = inferred_y + inferred_height

            # Decode-level downscaling for 4K+ videos. Mirrors the same
            # self.height > DECODE_TARGET_HEIGHT check PyAVCapture just made
            # internally with the same constant, so this always agrees with
            # what the capture above actually did.
            decode_height = DECODE_TARGET_HEIGHT if self.height > DECODE_TARGET_HEIGHT else None

            # Scale crop coordinates from native to decode resolution
            if decode_height is not None:
                scale_factor = decode_height / self.height
                if crop_x_start is not None:
                    crop_x_start = int(crop_x_start * scale_factor)
                    crop_y_start = int(crop_y_start * scale_factor)
                    crop_x_end = int(crop_x_end * scale_factor)
                    crop_y_end = int(crop_y_end * scale_factor)

            # get frames from ocr_start to ocr_end using producer-consumer pattern
            modulo = frames_to_skip + 1
            frames_to_process = (num_ocr_frames + modulo - 1) // modulo

            # Set up progress tracking via unified tracker
            if progress is not None:
                progress.set_phase('dialogue', frames_to_process)

            # Create queue for producer-consumer communication
            # Dynamic buffer sizing based on frame size and RAM budget
            if decode_height is not None:
                buf_w = int(self.width * (decode_height / self.height))
                frame_bytes = buf_w * decode_height * 3
            else:
                frame_bytes = self.width * self.height * 3  # BGR24
            buffer_frames = max(BATCH_SIZE * 2, MAX_BUFFER_BYTES // frame_bytes)
            frame_queue = Queue(maxsize=buffer_frames)

            adjusted_ocr_start = ocr_start

            # Bake the (already decode-output-scaled) crop into the filter
            # graph if this capture supports it and one was actually
            # inferred; must happen before get_stream_start_time()/set()
            # below touch anything read()-adjacent (see configure_crop()'s
            # own precondition). _frame_producer still receives
            # crop_x_start/... regardless: if the graph didn't take the
            # crop (v._crop_slice stays None - no filter graph existed for
            # this source, or PyAV isn't available), it slices in Python
            # instead, off the identical values either path would use.
            graph_crop = None
            if crop_x_end is not None and crop_y_end is not None:
                graph_crop = (crop_x_start, crop_y_start,
                              crop_x_end - crop_x_start, crop_y_end - crop_y_start)
            if hasattr(v, 'configure_crop'):
                v.configure_crop(graph_crop)

            # Get the container-level start_time for PTS normalization.
            # Players offset PTS only by container start_time (0 for MKV,
            # possibly non-zero for MP4). Stream start_time is not used.
            self._stream_start_time = v.get_stream_start_time() if hasattr(v, 'get_stream_start_time') else 0.0

            # Seek to the actual frame we want
            v.set(cv2.CAP_PROP_POS_FRAMES, ocr_start)

            # Start producer thread for frame reading
            producer = Thread(target=self._frame_producer, args=(v, frame_queue, adjusted_ocr_start, num_ocr_frames, modulo, crop_x_start, crop_y_start, crop_x_end, crop_y_end, brightness_threshold, similar_image_threshold, similar_pixel_threshold, progress, self.fps, cancel_event))
            producer.start()

            # Consumer loop - process messages from queue
            try:
                while True:
                    # Check cancellation before blocking on queue
                    if cancel_event is not None and cancel_event.is_set():
                        break

                    wait_start = time.perf_counter()
                    try:
                        msg = frame_queue.get(timeout=0.5)
                    except Empty:
                        # Timeout — loop back to check cancel_event
                        self._queue_wait_time += time.perf_counter() - wait_start
                        continue
                    self._queue_wait_time += time.perf_counter() - wait_start
                    msg_type = msg[0]

                    if msg_type == "done":
                        # The last tracked subtitle is finished - hand its
                        # candidates to the batch slot they belong to.
                        if batch_candidates:
                            batch_candidates[-1].extend(pending_candidates)
                        pending_candidates = []
                        break
                    elif msg_type == "extend":
                        # Similar frame - extend end_index and pts_end of previous
                        frame_idx, pts = msg[1], msg[2]
                        if batch_end_indices:
                            batch_end_indices[-1] = frame_idx
                            batch_pts_end[-1] = pts
                        elif self.pred_frames:
                            self.pred_frames[-1].end_index = frame_idx
                            self.pred_frames[-1].pts_end = pts
                    elif msg_type == "candidate":
                        # Buffer an alternative reading for the CURRENT subtitle.
                        # Whether it gets OCR'd is decided when the subtitle's
                        # batch is processed, from that subtitle's own
                        # first-reading confidence - never from queue depth.
                        #
                        # Admission is bounded by THIS subtitle's own byte
                        # budget, which carries no batch state, so neither
                        # consumer timing nor where the batch flushes fall can
                        # change which candidates are kept.
                        candidate = msg[1]
                        if len(pending_candidates) < _candidates_admitted(candidate):
                            pending_candidates.append(candidate)
                    elif msg_type == "frame":
                        # A new subtitle starts here, so the previous one is
                        # complete: hand its candidates to its batch slot.
                        if batch_candidates:
                            batch_candidates[-1].extend(pending_candidates)
                        pending_candidates = []

                        # Flush a full batch only at this subtitle boundary, so
                        # a slot is never closed while candidates are still
                        # arriving for it.
                        if len(batch_frames) >= BATCH_SIZE:
                            batch_base = len(self.pred_frames)
                            self._process_batch(ocr, batch_frames, batch_start_indices, batch_end_indices, batch_pts_start, batch_pts_end, conf_threshold_percent)
                            self._resolve_batch_candidates(ocr, batch_candidates, batch_base, conf_threshold, conf_threshold_percent)
                            self._emit_pending_subtitles(subtitle_callback)
                            batch_frames = []
                            batch_start_indices = []
                            batch_end_indices = []
                            batch_pts_start = []
                            batch_pts_end = []
                            batch_candidates = []

                        frame, frame_idx, pts = msg[1], msg[2], msg[3]
                        batch_frames.append(frame)
                        batch_start_indices.append(frame_idx)
                        batch_end_indices.append(frame_idx)
                        batch_pts_start.append(pts)
                        batch_pts_end.append(pts)
                        batch_candidates.append([])
            finally:
                producer.join()

            # Process remaining frames in batch
            if batch_frames:
                batch_base = len(self.pred_frames)
                self._process_batch(ocr, batch_frames, batch_start_indices, batch_end_indices, batch_pts_start, batch_pts_end, conf_threshold_percent)
                self._resolve_batch_candidates(ocr, batch_candidates, batch_base, conf_threshold, conf_threshold_percent)

            # Emit any remaining subtitles (all frames finalized)
            self._emit_pending_subtitles(subtitle_callback, emit_last=True)

    def get_subtitles(self, sim_threshold: int) -> str:
        """Generate ASS format subtitles."""
        self._generate_subtitles(sim_threshold)

        # Generate ASS header with video resolution
        header = utils.format_ass_header(self.width, self.height)

        # Generate dialogue lines - split multi-line text and merge persistent lines
        # When two characters talk simultaneously (one persistent, one changing),
        # the persistent line appears in consecutive subtitles. Instead of emitting
        # duplicates, we extend the persistent line's end time.
        entries = []  # list of [start, end, text]
        prev_lines = {}  # text -> index in entries
        frame_duration = 1.0 / self.fps  # Duration of one frame in seconds

        # Get container-level start_time offset (set during run_ocr)
        # This normalizes PTS values relative to playback time
        stream_offset = getattr(self, '_stream_start_time', 0.0)

        for sub in self.pred_subs:
            # Prefer PTS-based timestamps (canonical) over frame-index-based
            if sub.pts_start is not None and sub.pts_end is not None:
                # Normalize PTS by subtracting container start_time offset
                adjusted_start = max(0, sub.pts_start - stream_offset)
                adjusted_end = max(0, sub.pts_end - stream_offset) + frame_duration
                start = utils.get_ass_timestamp_from_seconds(adjusted_start)
                end = utils.get_ass_timestamp_from_seconds(adjusted_end)
            else:
                # Fallback to frame-index-based timestamps
                start = utils.get_ass_timestamp(sub.index_start, self.fps)
                end = utils.get_ass_timestamp(sub.index_end + 1, self.fps)

            current_lines = {}
            for line in reversed(sub.text.split('\n')):
                line = line.strip()
                if not line or line in current_lines:
                    continue
                if line in prev_lines:
                    # Line persists from previous subtitle - extend end time
                    idx = prev_lines[line]
                    entries[idx][1] = end
                    current_lines[line] = idx
                else:
                    # New line - add entry
                    current_lines[line] = len(entries)
                    entries.append([start, end, line])
            prev_lines = current_lines

        dialogues = [utils.format_ass_dialogue(s, e, t) for s, e, t in entries]
        return header + "".join(dialogues)

    def _generate_subtitles(self, sim_threshold: int) -> None:
        self.pred_subs = []

        if self.pred_frames is None:
            raise AttributeError("Please call self.run_ocr() first to perform ocr on frames")

        max_frame_merge_diff = int(SUBTITLE_MERGE_GAP_SECONDS * self.fps)
        min_duration_frames = int(MIN_SUBTITLE_DURATION * self.fps)
        for frame in self.pred_frames:
            self._append_sub(PredictedSubtitle([frame], sim_threshold), max_frame_merge_diff)
        # Filter out empty subtitles and those too short (noise)
        self.pred_subs = [sub for sub in self.pred_subs
                         if len(sub.frames[0].lines) > 0
                         and (sub.index_end - sub.index_start) >= min_duration_frames]

    def _append_sub(self, sub: PredictedSubtitle, max_frame_merge_diff: int) -> None:
        if len(sub.frames) == 0:
            return

        # merge new sub to the last subs if they are not empty, similar and within merge window
        if self.pred_subs:
            last_sub = self.pred_subs[-1]
            gap = sub.index_start - last_sub.index_end
            # Also check temporal gap to distinguish OCR retries from genuinely different subtitles
            # Retries on same subtitle will have small gap between previous end and new start
            start_gap = sub.frames[0].start_index - last_sub.frames[-1].end_index if last_sub.frames else gap
            is_similar = last_sub.is_similar_to(sub) if len(last_sub.frames[0].lines) > 0 else False

            # Max start gap for considering same subtitle (1.5 seconds at video fps)
            max_start_gap = int(1.5 * self.fps)

            # Merge if: end-to-start gap is small AND texts are similar AND start indices are close
            if len(last_sub.frames[0].lines) > 0 and gap <= max_frame_merge_diff and is_similar and start_gap <= max_start_gap:
                del self.pred_subs[-1]
                sub = PredictedSubtitle(last_sub.frames + sub.frames, sub.sim_threshold)

        self.pred_subs.append(sub)

    def _process_batch(self, ocr, frames: list, start_indices: list, end_indices: list, pts_start_list: list, pts_end_list: list, conf_threshold: float) -> None:
        """Process a batch of frames through OCR in a single call."""
        if not frames:
            return

        # Call OCR on entire batch at once - much more efficient than per-frame
        # Use predict() for PaddleOCR 3.x (ocr() is deprecated)
        ocr_start = time.perf_counter()
        if utils.needs_conversion():
            results = list(ocr.predict(frames))
        else:
            results = ocr.ocr(frames)
        self._ocr_time += time.perf_counter() - ocr_start

        # Create PredictedFrames for each result
        for pred_data, start_idx, end_idx, pts_start, pts_end in zip(results, start_indices, end_indices, pts_start_list, pts_end_list):
            # Wrap in list since PredictedFrames expects pred_data[0] to work
            pred_frame = PredictedFrames(start_idx, [pred_data], conf_threshold, self.lang, pts=pts_start)
            pred_frame.end_index = end_idx
            pred_frame.pts_end = pts_end
            self.pred_frames.append(pred_frame)

    def _resolve_batch_candidates(self, ocr, batch_candidates: list, base: int, conf_threshold: int, conf_threshold_percent: float) -> None:
        """Resolve buffered candidates for every slot of the batch just OCR'd.

        `batch_candidates[i]` belongs to the subtitle that `_process_batch`
        turned into `self.pred_frames[base + i]`, where `base` is the length
        of `pred_frames` captured before that batch was appended. Binding the
        slot index this way keeps the mapping exact and independent of
        BATCH_SIZE, which is what makes the result reproducible.
        """
        for offset, candidates in enumerate(batch_candidates):
            if not candidates:
                continue
            index = base + offset
            if index >= len(self.pred_frames):
                break
            self._resolve_candidates(ocr, candidates, self.pred_frames[index],
                                     conf_threshold, conf_threshold_percent)

    def _resolve_candidates(self, ocr, candidates: list, target, conf_threshold: int, conf_threshold_percent: float) -> None:
        """Improve one subtitle's text using its buffered alternative frames.

        Runs only when the subtitle's own first reading scored below
        conf_threshold. Timing is never modified: only `text`, `lines` and
        `confidence` of the existing PredictedFrames are replaced.

        Selection is agreement-first. Every candidate is an independent frame
        of the *same* subtitle, so several frames reading the same string is
        much stronger evidence than any single frame's confidence score --
        which measures the model's certainty, not its correctness, and which a
        confident misreading can win outright. A reading therefore replaces the
        original only when it is strictly modal, or when it is more confident
        by a real margin; and never when it merely drops a word the original
        was confident about.
        """
        if not candidates:
            return
        if self._confidence_pct(target) >= conf_threshold:
            return

        ocr_start_time = time.perf_counter()
        if utils.needs_conversion():
            results = list(ocr.predict(candidates))
        else:
            results = ocr.ocr(candidates)
        self._ocr_time += time.perf_counter() - ocr_start_time

        # The original reading votes alongside the candidates.
        readings = [target]
        for pred_data in results:
            alt = PredictedFrames(target.start_index, [pred_data],
                                  conf_threshold_percent, self.lang,
                                  pts=target.pts_start)
            # An empty reading carries the sentinel confidence of 100; it must
            # never be allowed to win and blank out a subtitle that has text.
            if not alt.lines:
                continue
            readings.append(alt)
        if len(readings) < 2:
            return

        votes = Counter(r.text for r in readings)

        def mean_conf(text: str) -> float:
            scores = [self._confidence_pct(r) for r in readings if r.text == text]
            return sum(scores) / len(scores)

        # Most agreed-upon text, ties broken by mean confidence. Counter keeps
        # insertion order and max() keeps the first maximum, so this is stable.
        winner_text = max(votes, key=lambda t: (votes[t], mean_conf(t)))
        if winner_text == target.text:
            return

        winner = max((r for r in readings if r.text == winner_text),
                     key=self._confidence_pct)

        if votes[winner_text] <= votes[target.text]:
            # Nothing corroborates this reading, so demand a real confidence
            # advantage...
            best_conf = max(self._confidence_pct(r) for r in readings
                            if r.text == winner_text)
            if best_conf < self._confidence_pct(target) + CANDIDATE_CONFIDENCE_MARGIN:
                return
            # ...and refuse it outright if it is shorter. Dropping a word
            # raises the mean for free, so on one frame's opinion alone a
            # shorter reading is indistinguishable from a frame that merely
            # failed to detect the word. The guard is unconditional, with no
            # carve-out for words the original scored poorly on. It is
            # tempting to read MIN_WORD_CONFIDENCE as such a carve-out, but
            # it is a garbage floor (0.5), not a confidence bar: surviving
            # the filter means a word was not obvious noise, not that the
            # original was confident about it. Nothing here can tell a
            # deliberate deletion from a missed detection, so nothing here
            # gets to make an exception.
            if len(winner.text) < len(target.text):
                return
        # A strictly modal winner needs no length guard: several independent
        # frames of this subtitle agreeing that the trailing glyphs are absent
        # is exactly the evidence the guard would be asking for. One frame
        # hallucinating extra characters must not outvote the frames that do
        # not see them.

        target.lines = winner.lines
        target.text = winner.text
        target.confidence = winner.confidence

    @staticmethod
    def _confidence_pct(pred) -> float:
        """PredictedFrames.confidence on a 0-100 scale.

        `confidence` is the mean word score (0-1) when the frame has text, and
        exactly 100 when PaddleOCR returned nothing at all. Normalise the
        former; leave the sentinel alone so empty frames never trigger work.
        """
        conf = pred.confidence
        return conf if conf > 1.0 else conf * 100.0

    def _emit_pending_subtitles(self, subtitle_callback, emit_last=False):
        """Emit pending subtitle detections via callback."""
        if subtitle_callback is None:
            return

        limit = len(self.pred_frames) if emit_last else len(self.pred_frames) - 1
        container_offset = getattr(self, '_stream_start_time', 0.0)
        frame_duration = 1.0 / self.fps

        for i in range(self._last_emitted_idx, limit):
            frame = self.pred_frames[i]
            if not frame.text:
                continue
            if frame.pts_start is None or frame.pts_end is None:
                continue
            start = max(0, frame.pts_start - container_offset)
            end = max(0, frame.pts_end - container_offset) + frame_duration
            if (end - start) < MIN_SUBTITLE_DURATION:
                continue
            subtitle_callback(start, end, frame.text)

        self._last_emitted_idx = limit

    def _frame_producer(self, v, queue: Queue, ocr_start: int, num_ocr_frames: int, modulo: int, crop_x_start, crop_y_start, crop_x_end, crop_y_end, brightness_threshold: int, similar_image_threshold: int, similar_pixel_threshold: int, progress, fps: float, cancel_event=None) -> None:
        """Producer thread: reads frames and puts eligible ones in queue.

        Uses a two-state approach for efficiency:
        - LOOKING: Fast Laplacian variance check for text edges (no text = quick skip)
        - TRACKING: Similarity check to efficiently track subtitle duration

        Temporal smoothing requires 2/3 consecutive frames to confirm text presence,
        reducing false positives from scene edges and flashes.
        """
        prev_grey = None
        tracking_subtitle = False  # State: are we currently tracking a subtitle?
        detection_history = []  # Track last 3 detection results for END smoothing only
        producer_start = time.perf_counter()
        current_pts = None  # Track PTS of current frame

        # Deterministic best-of-N: while tracking one subtitle, every
        # candidate_stride-th frame is offered as an alternative reading,
        # up to MAX_CANDIDATES. This schedule is a function of the frame
        # stream alone, never of consumer state.
        candidate_stride = max(1, int(round(fps / 4.0)))  # ~4 per second
        frames_in_subtitle = 0
        candidates_sent = 0

        for i in range(num_ocr_frames):
            if cancel_event is not None and cancel_event.is_set():
                break

            if i % modulo == 0:
                ret, frame = v.read()
                # Get canonical PTS timestamp (in seconds)
                current_pts = v.get_last_pts() if hasattr(v, 'get_last_pts') else (i + ocr_start) / fps
                if frame is None:
                    if progress is not None:
                        progress.update(1)
                    continue

                # Apply crop (coordinates pre-scaled if decode downscaling is active).
                # If the capture already cropped inside its filter graph
                # (v._crop_slice is set), skip the redundant Python slice.
                if not self.use_fullframe and not getattr(v, '_crop_slice', None):
                    if crop_x_end is not None and crop_y_end is not None:
                        frame = frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                    else:
                        # only use bottom third of the frame by default
                        frame = frame[2 * frame.shape[0] // 3 :, :]

                # Downscale for faster OCR, but ensure minimum height for accuracy
                # Scale is based on crop region size, not original video size
                frame_h, frame_w = frame.shape[:2]
                if frame_h > TARGET_VIDEO_HEIGHT:
                    # Calculate scale to reach target height
                    target_scale = TARGET_VIDEO_HEIGHT / frame_h
                    # Ensure we don't go below minimum readable height
                    min_scale = MIN_CROP_HEIGHT / frame_h
                    scale = max(target_scale, min_scale)

                    if scale < 1.0:
                        new_h = max(MIN_CROP_HEIGHT, int(frame_h * scale))
                        new_w = max(1, int(frame_w * scale))
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Apply brightness filter
                if brightness_threshold:
                    # Center square geometry is fixed by frame shape alone
                    # (not by content), so it can be computed once up front
                    # and reused whether or not the pre-gate below fires.
                    frame_h, frame_w = frame.shape[:2]
                    center_x_start = (frame_w - frame_h) // 2
                    center_x_end = center_x_start + frame_h

                    gated = False
                    if not tracking_subtitle:
                        # LOOKING STATE exact pre-gate: see
                        # _center_mask_is_empty()'s derivation above. Never
                        # applied in TRACKING: that state still needs the
                        # full masked grey below for its similarity check
                        # against the previous frame.
                        center_bgr = frame[:, center_x_start:center_x_end]
                        if _center_mask_is_empty(center_bgr, brightness_threshold):
                            gated = True

                    if gated:
                        raw_detection = False
                    else:
                        frame = cv2.bitwise_and(frame, frame, mask=cv2.inRange(frame, (brightness_threshold,) * 3, (255,) * 3))

                        # Convert to grayscale for checks
                        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # Center square check: h×h pixels, horizontally centered
                        center = grey[:, center_x_start:center_x_end]

                        # Use Laplacian variance for text detection (measures edge sharpness)
                        laplacian = cv2.Laplacian(center, cv2.CV_64F)
                        raw_detection = laplacian.var() >= MIN_LAPLACIAN_VARIANCE

                    # Update detection history (used for END smoothing only)
                    detection_history.append(raw_detection)
                    if len(detection_history) > 3:
                        detection_history.pop(0)

                    if not tracking_subtitle:
                        # LOOKING STATE: Instant detection for subtitle START
                        # Use raw_detection (1 frame) - no smoothing delay
                        if raw_detection:
                            # Text found - start tracking immediately
                            tracking_subtitle = True
                            prev_grey = grey
                            # Reset history to current detection for clean END tracking
                            detection_history = [True]
                            queue.put(("frame", frame, i + ocr_start, current_pts))
                            frames_in_subtitle = 0
                            candidates_sent = 0
                        if progress is not None:
                            progress.update(1)
                        continue
                    else:
                        # TRACKING STATE: Use 2/3 smoothing for END detection (prevents flicker)
                        text_still_present = sum(detection_history) >= 2

                        if not text_still_present:
                            # Text disappeared (confirmed by 2/3 rule) - back to looking state
                            tracking_subtitle = False
                            prev_grey = None
                            detection_history = []
                            if progress is not None:
                                progress.update(1)
                            continue

                        # Text still present - use similarity check with total-pixel threshold
                        if prev_grey is not None and similar_image_threshold:
                            _, absdiff = cv2.threshold(cv2.absdiff(prev_grey, grey), similar_pixel_threshold, 255, cv2.THRESH_BINARY)
                            # Use total pixels for threshold (same as original logic)
                            total_pixels = grey.shape[0] * grey.shape[1]
                            pixel_threshold = int(total_pixels * similar_image_threshold / 100.0)

                            if np.count_nonzero(absdiff) < pixel_threshold:
                                # Similar frame - extend, and offer an
                                # alternative reading on the fixed schedule.
                                frames_in_subtitle += 1
                                if (candidates_sent < MAX_CANDIDATES
                                        and frames_in_subtitle % candidate_stride == 0):
                                    queue.put(("candidate", frame.copy(), i + ocr_start, current_pts))
                                    candidates_sent += 1
                                queue.put(("extend", i + ocr_start, current_pts))
                                prev_grey = grey
                                if progress is not None:
                                    progress.update(1)
                                continue

                        # Different text - OCR new subtitle
                        prev_grey = grey
                        queue.put(("frame", frame, i + ocr_start, current_pts))
                        frames_in_subtitle = 0
                        candidates_sent = 0
                        if progress is not None:
                            progress.update(1)
                        continue

                # Non-brightness-filtered path: use original similarity check
                if similar_image_threshold:
                    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_grey is not None:
                        _, absdiff = cv2.threshold(cv2.absdiff(prev_grey, grey), similar_pixel_threshold, 255, cv2.THRESH_BINARY)
                        total_pixels = grey.shape[0] * grey.shape[1]
                        pixel_threshold = int(total_pixels * similar_image_threshold / 100.0)
                        if np.count_nonzero(absdiff) < pixel_threshold:
                            frames_in_subtitle += 1
                            if (candidates_sent < MAX_CANDIDATES
                                    and frames_in_subtitle % candidate_stride == 0):
                                queue.put(("candidate", frame.copy(), i + ocr_start, current_pts))
                                candidates_sent += 1
                            queue.put(("extend", i + ocr_start, current_pts))
                            prev_grey = grey
                            if progress is not None:
                                progress.update(1)
                            continue
                    prev_grey = grey

                # Different frame - send for OCR
                queue.put(("frame", frame, i + ocr_start, current_pts))
                frames_in_subtitle = 0
                candidates_sent = 0
                if progress is not None:
                    progress.update(1)
            else:
                v.read()

        # Signal end of frames and send producer timing
        self._producer_time = time.perf_counter() - producer_start
        queue.put(("done", None, None, None))
