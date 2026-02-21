"""Background subtitle detection worker for finding frames with hardcoded subtitles."""
import logging

import cv2
from PyQt6.QtCore import QObject, QThread, pyqtSignal

logger = logging.getLogger(__name__)

# Detection score threshold - real text typically scores ~0.97
DT_SCORE_THRESHOLD = 0.9

# Probe range: 40%-60% of video, stepping 0.5s
PROBE_START_FRAC = 0.40
PROBE_END_FRAC = 0.60
PROBE_STEP_SEC = 0.5

# Downscale target for fast detection
TARGET_HEIGHT = 480

# Default batch size for processing files (limits simultaneous open video handles)
DEFAULT_BATCH_SIZE = 10

# Auto-crop constants (defaults, overridable via automation settings)
CROP_WIDTH_FRACTION = 0.70       # 70% of video width
CROP_VERTICAL_PADDING = 0        # No padding above/below text
CROP_MIN_HEIGHT_FRACTION = 0.05  # Minimum crop height = 5% of video height
BOTTOM_HALF_CUTOFF = 0.50        # Ignore boxes in the top half of the frame


def _compute_crop_from_polys(polys, scores, orig_width, orig_height, downscaled_height,
                             crop_width_frac=CROP_WIDTH_FRACTION,
                             vert_padding=CROP_VERTICAL_PADDING,
                             min_height_frac=CROP_MIN_HEIGHT_FRACTION,
                             bottom_cutoff=BOTTOM_HALF_CUTOFF):
    """Compute a crop box from detected text polygons.

    Args:
        polys: List of polygon arrays from PaddleOCR detection.
        scores: List of detection confidence scores.
        orig_width: Original video frame width.
        orig_height: Original video frame height.
        downscaled_height: Height of the downscaled frame used for detection.
        crop_width_frac: Fraction of video width to use for crop.
        vert_padding: Fraction of video height for vertical padding.
        min_height_frac: Minimum crop height as fraction of video height.
        bottom_cutoff: Fraction of frame height; ignore boxes above this.

    Returns:
        (crop_x, crop_y, crop_w, crop_h) in original resolution, or None if no valid boxes.
    """
    if polys is None or len(polys) == 0:
        return None

    scale = orig_height / downscaled_height

    # Filter polys by score and bottom-half position
    min_y = None
    max_y = None

    for i, poly in enumerate(polys):
        # Check score threshold
        if scores is not None and i < len(scores):
            if float(scores[i]) < DT_SCORE_THRESHOLD:
                continue

        # Get Y coordinates from polygon (array of [x, y] points)
        try:
            ys = [float(pt[1]) for pt in poly]
        except (IndexError, TypeError, ValueError):
            continue

        y_center = sum(ys) / len(ys)

        # Filter: only keep boxes in bottom half of frame
        if y_center < downscaled_height * bottom_cutoff:
            continue

        poly_min_y = min(ys)
        poly_max_y = max(ys)

        if min_y is None or poly_min_y < min_y:
            min_y = poly_min_y
        if max_y is None or poly_max_y > max_y:
            max_y = poly_max_y

    if min_y is None or max_y is None:
        return None

    # Scale Y coordinates back to original resolution
    min_y_orig = min_y * scale
    max_y_orig = max_y * scale

    # Add vertical padding
    pad = orig_height * vert_padding
    crop_y = max(0, int(min_y_orig - pad))
    crop_bottom = min(orig_height, int(max_y_orig + pad))
    crop_h = crop_bottom - crop_y

    # Enforce minimum crop height
    min_h = int(orig_height * min_height_frac)
    if crop_h < min_h:
        center_y = (crop_y + crop_bottom) / 2
        crop_y = max(0, int(center_y - min_h / 2))
        crop_h = min(min_h, orig_height - crop_y)

    # Width = fraction of original, centered horizontally
    crop_w = int(orig_width * crop_width_frac)
    crop_x = int((orig_width - crop_w) / 2)

    return (crop_x, crop_y, crop_w, crop_h)


class SubtitleDetectionWorker(QObject):
    """Worker that scans video files to find frames containing subtitles.

    Uses PaddleOCR TextDetection (detection-only, no OCR) to efficiently
    locate a frame with visible text in each video file.

    Runs in a QThread. Probes the 40-60% range of each video in rounds,
    batching frames across files for GPU throughput.

    Signals:
        file_detected(str, int, int, int, int, int): filename, slider_position, crop_x, crop_y, crop_w, crop_h
        progress(int, int): (files_resolved, total_files)
        finished(): all files processed
        error(str): error message
    """

    file_detected = pyqtSignal(str, int, int, int, int, int)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, video_files: list[tuple[str, str, float]],
                 automation_settings: dict | None = None):
        """Initialize worker.

        Args:
            video_files: list of (filename, full_path, duration_seconds) tuples
            automation_settings: optional dict with auto-crop overrides
        """
        super().__init__()
        self._video_files = video_files
        self._auto_settings = automation_settings
        self._cancel_requested = False
        self._thread: QThread | None = None

    def start(self):
        """Start the worker in a new thread."""
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.start()

    def cancel(self):
        """Request cancellation."""
        self._cancel_requested = True

    def cleanup(self):
        """Stop the thread and clean up."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None

    def _run(self):
        """Execute the detection scan using a sliding window of open captures.

        Maintains up to `batch_size` open video handles at once. When a file
        resolves or exhausts its probe range, its capture is closed immediately
        and the next queued file is opened in its place.
        """
        # Extract auto-crop overrides from settings
        s = self._auto_settings or {}
        crop_width_frac = float(s.get('crop_width_fraction', CROP_WIDTH_FRACTION))
        vert_padding = float(s.get('crop_vertical_padding', CROP_VERTICAL_PADDING))
        min_height_frac = float(s.get('crop_min_height_fraction', CROP_MIN_HEIGHT_FRACTION))
        bottom_cutoff = float(s.get('bottom_half_cutoff', BOTTOM_HALF_CUTOFF))

        batch_size = int(s.get('detection_batch_size', DEFAULT_BATCH_SIZE))
        if batch_size < 1:
            batch_size = DEFAULT_BATCH_SIZE

        det_engine = None
        active = []  # currently open file info dicts (each has 'cap' key)

        try:
            from videocr.utils import create_detection_engine, suppress_output
            from videocr.pyav_adapter import Capture

            # Load detection engine once
            with suppress_output():
                det_engine = create_detection_engine(None, True)

            total = len(self._video_files)
            resolved_count = 0
            self.progress.emit(resolved_count, total)

            next_file_idx = 0

            def fill_window():
                """Open files from the queue until the window is full."""
                nonlocal next_file_idx
                while len(active) < batch_size and next_file_idx < total:
                    if self._cancel_requested:
                        return
                    filename, full_path, duration = self._video_files[next_file_idx]
                    next_file_idx += 1
                    try:
                        cap = Capture(full_path)
                        cap.__enter__()
                        active.append({
                            'filename': filename,
                            'duration': duration,
                            'cap': cap,
                            'fps': cap.get(cv2.CAP_PROP_FPS) or 25.0,
                            'orig_width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            'orig_height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            'step_idx': 0,
                        })
                    except Exception as e:
                        logger.warning(f"Cannot open {filename} for subtitle detection: {e}")

            def close_file(info):
                """Close a file's capture and drop all references to free memory."""
                try:
                    info['cap'].__exit__(None, None, None)
                except Exception:
                    pass
                info.clear()

            # Fill initial window
            fill_window()

            # Probe loop — each iteration is one round across all active files
            while active and not self._cancel_requested:
                frames = []
                frame_info = []
                exhausted = []

                for info in active:
                    start_sec = info['duration'] * PROBE_START_FRAC
                    probe_sec = start_sec + info['step_idx'] * PROBE_STEP_SEC
                    end_sec = info['duration'] * PROBE_END_FRAC

                    if probe_sec > end_sec:
                        exhausted.append(info)
                        continue

                    cap = info['cap']
                    target_frame = int(probe_sec * info['fps'])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue

                    # Downscale to TARGET_HEIGHT for fast detection
                    h, w = frame.shape[:2]
                    if h > TARGET_HEIGHT:
                        scale = TARGET_HEIGHT / h
                        new_w = max(1, int(w * scale))
                        frame = cv2.resize(frame, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

                    frames.append(frame)
                    frame_info.append((info, probe_sec))

                # Detect text in all collected frames
                newly_resolved = []
                if frames:
                    results = list(det_engine.predict(frames))

                    for idx, item in enumerate(results):
                        info, probe_sec = frame_info[idx]
                        scores = item.get("dt_scores", [])
                        polys = item.get("dt_polys", [])

                        has_text = False
                        if scores is not None and len(scores) > 0:
                            for score in scores:
                                if float(score) >= DT_SCORE_THRESHOLD:
                                    has_text = True
                                    break
                        elif polys is not None and len(polys) > 0:
                            has_text = True

                        if has_text:
                            duration = info['duration']
                            slider_pos = int((probe_sec / duration) * 10000) if duration > 0 else 5000
                            slider_pos = max(0, min(10000, slider_pos))

                            orig_w = info['orig_width']
                            orig_h = info['orig_height']
                            downscaled_h = min(TARGET_HEIGHT, orig_h)
                            crop_result = _compute_crop_from_polys(
                                polys, scores, orig_w, orig_h, downscaled_h,
                                crop_width_frac=crop_width_frac,
                                vert_padding=vert_padding,
                                min_height_frac=min_height_frac,
                                bottom_cutoff=bottom_cutoff,
                            )
                            if crop_result:
                                cx, cy, cw, ch = crop_result
                            else:
                                cx, cy, cw, ch = 0, 0, 0, 0

                            self.file_detected.emit(info['filename'], slider_pos, cx, cy, cw, ch)
                            newly_resolved.append(info)
                            resolved_count += 1
                            self.progress.emit(resolved_count, total)

                # Close and remove resolved/exhausted files
                for info in newly_resolved + exhausted:
                    close_file(info)
                    active.remove(info)

                # Advance probe step for remaining active files
                for info in active:
                    info['step_idx'] += 1

                # Refill the window with new files
                fill_window()

            self.finished.emit()

        except Exception as e:
            logger.exception("Subtitle detection failed")
            self.error.emit(str(e))
        finally:
            del det_engine
            for info in active:
                close_file(info)
            active.clear()
