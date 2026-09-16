from __future__ import annotations
from typing import List
from collections import Counter
from queue import Queue, Empty
from threading import Thread
import cv2
import numpy as np
import time

from . import utils
from .models import PredictedFrames, PredictedSubtitle

from .pyav_adapter import Capture, DECODE_TARGET_HEIGHT

# Batch size for OCR processing - higher values = faster but more GPU memory
# 64 is optimal for most cases
BATCH_SIZE = 256

# Maximum RAM to use for frame buffer (10GB)
MAX_BUFFER_BYTES = 10 * 1024**3

# Target video height for downscaling (720p for speed, maintains OCR accuracy)
TARGET_VIDEO_HEIGHT = 720

# Minimum crop height after scaling to maintain OCR accuracy
# Text smaller than this becomes difficult for OCR to read reliably
MIN_CROP_HEIGHT = 150

# Maximum alternative frames OCR'd for one low-confidence subtitle
MAX_CANDIDATES = 8

# Ceiling on the alternative-reading frames buffered for one batch, per worker.
#
# A candidate is held from the moment the producer offers it until the batch
# it belongs to has been OCR'd and resolved, so the standing worst case is
# MAX_CANDIDATES x BATCH_SIZE = 2048 frames on top of the batch's own 256 and
# the frame queue. At the reference crop geometry (1344x53 BGR, 213,696 bytes
# a frame) that is ~437 MB; with use_fullframe the candidates are whole
# 1280x720 pictures (2,764,800 bytes) and it reaches ~5.7 GB -- per worker,
# and OCRManager runs several workers at once. 256 subtitles each holding a
# couple of seconds of similar frames is only ~8.5 minutes of dialogue, so
# this is an ordinary episode, not a pathological one.
#
# 512 MiB sits just above the crop-mode worst case, so the mode production
# actually runs in is never clipped at all (and the golden cases, which OCR
# two minutes at that geometry, are nowhere near it); use_fullframe is held
# to ~194 frames instead of 2048.
MAX_CANDIDATE_BUFFER_BYTES = 512 * 1024**2


def _candidate_fits(buffered_bytes: int, frame) -> bool:
    """Whether one more candidate frame fits in a batch's byte budget.

    A pure function of the running total and the frame itself. That is the
    whole point: admission must depend only on the frame stream and the
    configuration, so the buffered set - and therefore the output - is the
    same on a loaded machine as on an idle one, at any BATCH_SIZE and any
    worker count. Anything consulting queue depth or elapsed time here would
    make the OCR result a function of scheduling.
    """
    return buffered_bytes + frame.nbytes <= MAX_CANDIDATE_BUFFER_BYTES


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
        with Capture(path) as v:
            self.num_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = v.get(cv2.CAP_PROP_FPS)
            self.height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))

    def run_ocr(self, use_gpu: bool, lang: str, time_start: str, time_end: str, conf_threshold: int, use_fullframe: bool, brightness_threshold: int, similar_image_threshold: float, similar_pixel_threshold: int, frames_to_skip: int, crop_x: int, crop_y: int, crop_width: int, crop_height: int, progress=None, subtitle_callback=None, cancel_event=None):
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
        # Bytes of candidate frames currently held for this batch: those
        # already attached to batch_candidates plus pending_candidates. Reset
        # with the batch at each flush, when they are all released.
        batch_candidate_bytes = 0

        ocr = utils.create_ocr_engine(self.lang, self.det_model_dir, self.rec_model_dir, use_gpu)

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

        # Decode-level downscaling for 4K+ videos
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

        # Profiling variables
        self._ocr_time = 0.0
        self._queue_wait_time = 0.0
        self._producer_time = 0.0  # Will be set by producer thread
        profiling_start = time.perf_counter()

        adjusted_ocr_start = ocr_start

        graph_crop = None
        if crop_x_end is not None and crop_y_end is not None:
            graph_crop = (crop_x_start, crop_y_start,
                          crop_x_end - crop_x_start, crop_y_end - crop_y_start)

        with Capture(self.path, decode_target_height=decode_height,
                     crop_rect=graph_crop) as v:
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
                        # Admission is bounded by the batch's byte budget and
                        # evaluated in producer order, so the cap cannot make
                        # the buffered set depend on consumer timing: the same
                        # video always drops the same candidates.
                        candidate = msg[1]
                        if _candidate_fits(batch_candidate_bytes, candidate):
                            pending_candidates.append(candidate)
                            batch_candidate_bytes += candidate.nbytes
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
                            # Every buffered candidate belonged to the batch
                            # just resolved, so the budget is free again.
                            batch_candidate_bytes = 0

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

        return ocr

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
            # failed to detect the word. The guard is unconditional here:
            # PredictedFrames already discards every word below
            # MIN_WORD_CONFIDENCE before assembling `lines`, so a dropped word
            # the original was *not* confident about cannot occur.
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
                    frame = cv2.bitwise_and(frame, frame, mask=cv2.inRange(frame, (brightness_threshold,) * 3, (255,) * 3))

                    # Convert to grayscale for checks
                    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    h, w = grey.shape

                    # Center square check: h×h pixels, horizontally centered
                    center_x_start = (w - h) // 2
                    center_x_end = center_x_start + h
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
