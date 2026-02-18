from __future__ import annotations
from typing import List
from queue import Queue, Empty
from threading import Thread, Event
import cv2
import numpy as np
import time

from . import utils
from .models import PredictedFrames, PredictedSubtitle

from .pyav_adapter import Capture

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

# Maximum OCR retry attempts when confidence is below threshold
MAX_OCR_RETRIES = 10

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

        # get frames from ocr_start to ocr_end using producer-consumer pattern
        modulo = frames_to_skip + 1
        frames_to_process = (num_ocr_frames + modulo - 1) // modulo

        # Set up progress tracking via unified tracker
        if progress is not None:
            progress.set_phase('dialogue', frames_to_process)

        # Create queue for producer-consumer communication
        # Dynamic buffer sizing based on frame size and RAM budget
        frame_bytes = self.width * self.height * 3  # BGR24
        buffer_frames = max(BATCH_SIZE * 2, MAX_BUFFER_BYTES // frame_bytes)
        frame_queue = Queue(maxsize=buffer_frames)

        # Profiling variables
        self._ocr_time = 0.0
        self._queue_wait_time = 0.0
        self._producer_time = 0.0  # Will be set by producer thread
        profiling_start = time.perf_counter()

        # Retry state - shared between producer and consumer threads
        # When confidence is below threshold, consumer sets retry_needed to trigger
        # producer to send subsequent similar frames for OCR retry
        retry_needed = Event()
        retry_count = [0]  # Mutable container for thread-safe counter
        retry_best = [None]  # Best (confidence, PredictedFrames) seen during retry

        # Note: We no longer apply manual frame offset corrections based on container start_time.
        # PyAV's PTS values are already normalized and provide accurate timing.
        # Using PTS directly eliminates the need for manual offset calculations
        # which could cause double-correction issues.
        adjusted_ocr_start = ocr_start

        with Capture(self.path) as v:
            # Get the stream's start_time offset for PTS normalization
            # Different videos have different start_times (e.g., 0.042s vs 0.080s)
            # which must be subtracted from PTS to get correct timestamps
            self._stream_start_time = v.get_stream_start_time() if hasattr(v, 'get_stream_start_time') else 0.0

            # Seek to the actual frame we want
            v.set(cv2.CAP_PROP_POS_FRAMES, ocr_start)

            # Start producer thread for frame reading
            producer = Thread(target=self._frame_producer, args=(v, frame_queue, adjusted_ocr_start, num_ocr_frames, modulo, crop_x_start, crop_y_start, crop_x_end, crop_y_end, brightness_threshold, similar_image_threshold, similar_pixel_threshold, progress, retry_needed, retry_count, self.fps, cancel_event))
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
                        # Finalize any pending retry before exiting
                        if retry_best[0] is not None:
                            self.pred_frames[-1] = retry_best[0][1]
                            retry_needed.clear()
                            retry_best[0] = None
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
                    elif msg_type == "retry":
                        # Retry frame - OCR single frame and check confidence
                        frame, frame_idx, pts = msg[1], msg[2], msg[3]
                        ocr_start_time = time.perf_counter()
                        if utils.needs_conversion():
                            result = list(ocr.predict([frame]))[0]
                        else:
                            result = ocr.ocr([frame])[0]
                        self._ocr_time += time.perf_counter() - ocr_start_time

                        # Create temp PredictedFrames to check confidence (preserve original PTS)
                        original_pts_start = self.pred_frames[-1].pts_start
                        temp_pred = PredictedFrames(self.pred_frames[-1].start_index, [result], conf_threshold_percent, self.lang, pts=original_pts_start)
                        temp_pred.end_index = frame_idx
                        temp_pred.pts_end = pts

                        # Track best result
                        if retry_best[0] is None or temp_pred.confidence > retry_best[0][0]:
                            retry_best[0] = (temp_pred.confidence, temp_pred)

                        # Check if good enough or max retries reached
                        if temp_pred.confidence >= conf_threshold:
                            # Good confidence - use best result and exit retry mode
                            self.pred_frames[-1] = retry_best[0][1]
                            retry_needed.clear()
                            retry_count[0] = 0
                            retry_best[0] = None
                        elif retry_count[0] >= MAX_OCR_RETRIES:
                            # Max retries - use best we found
                            self.pred_frames[-1] = retry_best[0][1]
                            retry_needed.clear()
                            retry_count[0] = 0
                            retry_best[0] = None
                    elif msg_type == "frame":
                        # New different frame - finalize any pending retry first
                        if retry_best[0] is not None:
                            self.pred_frames[-1] = retry_best[0][1]
                            retry_needed.clear()
                            retry_count[0] = 0
                            retry_best[0] = None

                        frame, frame_idx, pts = msg[1], msg[2], msg[3]
                        batch_frames.append(frame)
                        batch_start_indices.append(frame_idx)
                        batch_end_indices.append(frame_idx)
                        batch_pts_start.append(pts)
                        batch_pts_end.append(pts)

                        # Process batch when full
                        if len(batch_frames) >= BATCH_SIZE:
                            self._process_batch(ocr, batch_frames, batch_start_indices, batch_end_indices, batch_pts_start, batch_pts_end, conf_threshold_percent)
                            self._emit_pending_subtitles(subtitle_callback)
                            batch_frames = []
                            batch_start_indices = []
                            batch_end_indices = []
                            batch_pts_start = []
                            batch_pts_end = []
                            # Check if last processed frame needs retry
                            if self.pred_frames and self.pred_frames[-1].confidence < conf_threshold:
                                retry_needed.set()
                                retry_best[0] = (self.pred_frames[-1].confidence, self.pred_frames[-1])
            finally:
                producer.join()

            # Process remaining frames in batch
            if batch_frames:
                self._process_batch(ocr, batch_frames, batch_start_indices, batch_end_indices, batch_pts_start, batch_pts_end, conf_threshold_percent)
                # Check if last processed frame needs retry
                if self.pred_frames and self.pred_frames[-1].confidence < conf_threshold:
                    retry_needed.set()
                    retry_best[0] = (self.pred_frames[-1].confidence, self.pred_frames[-1])

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

        # Get stream start_time offset (set during run_ocr)
        # This normalizes PTS values so frame 0 starts at time 0
        stream_offset = getattr(self, '_stream_start_time', 0.0)

        for sub in self.pred_subs:
            # Prefer PTS-based timestamps (canonical) over frame-index-based
            if sub.pts_start is not None and sub.pts_end is not None:
                # Normalize PTS by subtracting stream start_time offset
                # This converts from absolute PTS to relative time from video start
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

    def _emit_pending_subtitles(self, subtitle_callback, emit_last=False):
        """Emit pending subtitle detections via callback."""
        if subtitle_callback is None:
            return

        limit = len(self.pred_frames) if emit_last else len(self.pred_frames) - 1
        stream_offset = getattr(self, '_stream_start_time', 0.0)
        frame_duration = 1.0 / self.fps

        for i in range(self._last_emitted_idx, limit):
            frame = self.pred_frames[i]
            if not frame.text:
                continue
            if frame.pts_start is None or frame.pts_end is None:
                continue
            start = max(0, frame.pts_start - stream_offset)
            end = max(0, frame.pts_end - stream_offset) + frame_duration
            if (end - start) < MIN_SUBTITLE_DURATION:
                continue
            subtitle_callback(start, end, frame.text)

        self._last_emitted_idx = limit

    def _frame_producer(self, v, queue: Queue, ocr_start: int, num_ocr_frames: int, modulo: int, crop_x_start, crop_y_start, crop_x_end, crop_y_end, brightness_threshold: int, similar_image_threshold: int, similar_pixel_threshold: int, progress, retry_needed: Event, retry_count: list, fps: float, cancel_event=None) -> None:
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
        # Buffer for retro-check: store previous frame data for instant start detection
        prev_frame_data = None  # (frame, grey, frame_idx, pts)

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

                # Apply crop at native resolution
                if not self.use_fullframe:
                    if crop_x_end is not None and crop_y_end is not None:
                        frame = frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                    else:
                        # only use bottom third of the frame by default
                        frame = frame[2 * self.height // 3 :, :]

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
                            retry_count[0] = 0
                            # Reset history to current detection for clean END tracking
                            detection_history = [True]
                            queue.put(("frame", frame, i + ocr_start, current_pts))
                        else:
                            # No text - buffer this frame for potential retro-check
                            prev_frame_data = (frame.copy(), grey.copy(), i + ocr_start, current_pts)
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
                            prev_frame_data = None
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
                                # Similar frame
                                if retry_needed.is_set() and retry_count[0] < MAX_OCR_RETRIES:
                                    # Low confidence - retry OCR on this frame
                                    queue.put(("retry", frame.copy(), i + ocr_start, current_pts))
                                    retry_count[0] += 1
                                else:
                                    # Normal extend
                                    queue.put(("extend", i + ocr_start, current_pts))
                                prev_grey = grey
                                if progress is not None:
                                    progress.update(1)
                                continue

                        # Different text - OCR new subtitle
                        prev_grey = grey
                        retry_count[0] = 0
                        queue.put(("frame", frame, i + ocr_start, current_pts))
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
                            if retry_needed.is_set() and retry_count[0] < MAX_OCR_RETRIES:
                                queue.put(("retry", frame.copy(), i + ocr_start, current_pts))
                                retry_count[0] += 1
                            else:
                                queue.put(("extend", i + ocr_start, current_pts))
                            prev_grey = grey
                            if progress is not None:
                                progress.update(1)
                            continue
                    prev_grey = grey

                # Different frame - send for OCR
                retry_count[0] = 0
                queue.put(("frame", frame, i + ocr_start, current_pts))
                if progress is not None:
                    progress.update(1)
            else:
                v.read()

        # Signal end of frames and send producer timing
        self._producer_time = time.perf_counter() - producer_start
        queue.put(("done", None, None, None))
