"""Label/nameplate scanner for detecting positioned text overlays in video.

4-phase pipeline that trusts PaddleOCR:
  Phase 1: Detection Scan — sparse sampling at 720p to find frames with text
  Phase 2: Position Grouping — cluster detection boxes spatially across frames
  Phase 3: Crop, Clean, Recognize — dual OCR at regular intervals, detect content changes
  Phase 4: Timing Refinement — detection scan to find precise start/end

Key principles:
  - Trust Paddle — no pixel heuristics, no color analysis, no scene cut detection
  - Detection at 720p — catches smaller nameplates
  - No brightness filter at detection stage — only at recognition (Phase 3 dual OCR)
  - Dual OCR — brightness-filtered + raw image, pick higher confidence
  - Back-to-back via content comparison — OCR at regular intervals, detect text changes
  - Position grouping is purely spatial — no time gap thresholds
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import cv2
import numpy as np
from thefuzz import fuzz

from . import utils
from .pyav_adapter import Capture


@dataclass
class LabelResult:
    """Final label with timing and position."""

    start_pts: float
    end_pts: float
    text: str
    pos_x: int
    pos_y: int
    bbox_x_min: float = None
    bbox_y_min: float = None
    bbox_x_max: float = None
    bbox_y_max: float = None


class _RetainedCrops:
    """Phase 1's OCR crops, kept for phase 1.5 up to a byte budget.

    Phase 1 already holds each text frame at native resolution when it
    finds boxes in it, and phase 1.5 only ever OCRs
    `_crop_box_region(frame[:cutoff], box)` for each of those boxes. Keeping
    just those crops (copies, so the frame itself is released) lets phase
    1.5 skip re-fetching the frame -- a seek plus a decode from the previous
    keyframe, ~0.18-0.25 s per text frame at 4K -- without holding whole
    4K frames (~22 MB each once sliced to the dialogue cutoff).

    Admission is per text frame and all-or-nothing, since phase 1.5 needs
    the frame anyway if any of its crops is missing. A frame whose crops do
    not fit in what is left of the budget is not kept, and phase 1.5 fetches
    it by PTS instead. Phase 1.5 takes each entry once, releasing its bytes.
    """

    def __init__(self, budget_bytes):
        self.budget_bytes = budget_bytes
        self.nbytes = 0
        self.peak_nbytes = 0
        self.refused = 0
        self._entries = {}

    @staticmethod
    def _size(crops):
        return sum(crop.nbytes for crop in crops if crop is not None)

    def offer(self, frame_idx, pts, boxes, crops):
        size = self._size(crops)
        if self.nbytes + size > self.budget_bytes:
            self.refused += 1
            return False
        self._entries[(frame_idx, pts)] = (boxes, crops)
        self.nbytes += size
        self.peak_nbytes = max(self.peak_nbytes, self.nbytes)
        return True

    def take(self, frame_idx, pts, boxes):
        """The crops kept for this text frame, or None. Only returned for the
        very `boxes` list phase 1 recorded them against."""
        entry = self._entries.pop((frame_idx, pts), None)
        if entry is None:
            return None
        kept_boxes, crops = entry
        self.nbytes -= self._size(crops)
        return crops if kept_boxes is boxes else None


class LabelScanner:
    SCAN_HEIGHT = 720  # Detection resolution (up from 480)
    RECOGNIZE_HEIGHT = 720  # Max dimension for OCR crops
    SAMPLE_INTERVAL_SECONDS = 0.5  # Sparse sampling interval
    TIMING_SCAN_INTERVAL = 0.2  # 200ms between timing scan samples
    TIMING_SCAN_MAX_DURATION = 5.0  # Max seconds to search in each direction
    OCR_CONFIDENCE_THRESHOLD = 0.95
    TEXT_SIMILARITY_THRESHOLD = 0.85  # For back-to-back label detection
    MERGE_TEXT_SIMILARITY = 0.80  # Text match threshold for Phase 2 merge
    # Most phase 1 may keep in crops for phase 1.5 per scan (see _RetainedCrops).
    # Measured on 4K label runs: 212 MB for 5 min (136 text frames, 178 boxes)
    # and 152 MB for 7 min (151 frames, 226 boxes), so this covers about ten
    # minutes of label-dense 4K; frames past it are fetched by PTS instead.
    PHASE15_RETAIN_BUDGET_BYTES = 512 * 1024 * 1024

    def __init__(self, video_path, fps, width, height, num_frames, crop_x, crop_y, crop_width, crop_height, label_min_duration=1.0, label_max_duration=5.0, conf_threshold=95, conf_threshold_min=75, brightness_threshold=None, label_mask_crops=None):
        self.video_path = video_path
        self.fps = fps
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.label_min_duration = label_min_duration
        self.label_max_duration = label_max_duration
        self.conf_threshold = conf_threshold / 100.0
        self.conf_threshold_min = conf_threshold_min / 100.0
        self.brightness_threshold = brightness_threshold
        self.label_mask_crops = label_mask_crops

        # Dialogue cutoff: slice off bottom region where subtitles appear
        self.dialogue_cutoff_y = crop_y if crop_y is not None else int(height * 0.8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_label_masks(self, frame):
        """Black out label mask regions to hide watermarks from detection."""
        if self.label_mask_crops:
            for (x, y, w, h) in self.label_mask_crops:
                frame[y:y+h, x:x+w] = 0
        return frame

    def _apply_brightness_filter(self, frame):
        """Apply brightness threshold filter if configured.

        Blacks out pixels below brightness_threshold to help with
        white text on varied backgrounds.
        """
        if self.brightness_threshold:
            frame = cv2.bitwise_and(frame, frame, mask=cv2.inRange(frame, (self.brightness_threshold,) * 3, (255,) * 3))
        return frame

    def _downscale(self, frame, target_height):
        """Scale frame to target height. Returns (scaled_frame, scale_factor)."""
        h = frame.shape[0]
        if h <= target_height:
            return frame, 1.0
        scale = target_height / h
        new_h = target_height
        new_w = max(1, int(frame.shape[1] * scale))
        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return scaled, scale

    def _resize_max_dimension(self, frame, max_dim):
        """Resize frame so max dimension is at most max_dim."""
        h, w = frame.shape[:2]
        if max(h, w) <= max_dim:
            return frame, 1.0
        scale = max_dim / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _run_detection(self, det_engine, frame):
        """Run text detection. Returns list of 4x2 box arrays."""
        results = det_engine.predict(frame)
        boxes = []
        for item in results:
            polys = item.get("dt_polys", [])
            for poly in polys:
                arr = np.array(poly, dtype=np.float32)
                if arr.shape == (4, 2):
                    boxes.append(arr)
        return boxes

    @staticmethod
    def _box_centroid(box):
        """Compute centroid of a 4x2 bounding box."""
        return float(box[:, 0].mean()), float(box[:, 1].mean())

    @staticmethod
    def _box_bounds(box):
        """Get (x_min, y_min, x_max, y_max) from a 4x2 box."""
        return (float(box[:, 0].min()), float(box[:, 1].min()), float(box[:, 0].max()), float(box[:, 1].max()))

    def _boxes_overlap(self, box1, box2, threshold=0.5):
        """Check if two boxes are close enough to be the same label.

        Checks X and Y proximity independently, each against its own
        average dimension. This prevents a wide box (e.g. dialogue) from
        absorbing a narrow box (e.g. label) that happens to be nearby
        vertically.
        """
        cx1, cy1 = self._box_centroid(box1)
        cx2, cy2 = self._box_centroid(box2)

        # Get box sizes
        x1_min, y1_min, x1_max, y1_max = self._box_bounds(box1)
        x2_min, y2_min, x2_max, y2_max = self._box_bounds(box2)

        avg_w = ((x1_max - x1_min) + (x2_max - x2_min)) / 2
        avg_h = ((y1_max - y1_min) + (y2_max - y2_min)) / 2

        dx = abs(cx1 - cx2)
        dy = abs(cy1 - cy2)

        return dx < avg_w * threshold and dy < avg_h * threshold

    @staticmethod
    def _deduplicate_text(text):
        """Fix doubled OCR text like '神界神界' → '神界'.

        PaddleOCR sometimes reads a region twice, producing doubled text
        as a single result.  Only triggers for text of 4+ characters where
        the first half exactly equals the second half.
        """
        if len(text) >= 4 and len(text) % 2 == 0:
            half = len(text) // 2
            if text[:half] == text[half:]:
                return text[:half]
        return text

    @staticmethod
    def _normalize_text(text):
        """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
        if not text:
            return ""
        import re

        # Lowercase
        t = text.lower()
        # Remove punctuation
        t = re.sub(r"[^\w\s]", "", t)
        # Collapse whitespace
        t = " ".join(t.split())
        return t

    def _texts_similar(self, text1, text2, threshold=None):
        """Check if two texts are similar enough to be the same label.

        Args:
            text1: First text to compare.
            text2: Second text to compare.
            threshold: Optional similarity threshold (0.0-1.0). Defaults to
                      TEXT_SIMILARITY_THRESHOLD if not provided.

        Returns:
            True if texts are similar enough.
        """
        if threshold is None:
            threshold = self.TEXT_SIMILARITY_THRESHOLD

        t1 = self._normalize_text(text1)
        t2 = self._normalize_text(text2)

        if not t1 or not t2:
            return False

        # Exact match
        if t1 == t2:
            return True

        # Substring match - but only if shorter is >= 50% of longer
        # This prevents single CJK characters from matching longer strings
        if t1 in t2 or t2 in t1:
            shorter, longer = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
            if len(shorter) >= len(longer) * 0.5:
                return True

        # Fuzzy match
        ratio = fuzz.ratio(t1, t2) / 100.0
        return ratio >= threshold

    def _get_encompassing_box(self, boxes):
        """Get a box that encompasses all given boxes."""
        all_points = np.vstack(boxes)
        x_min = all_points[:, 0].min()
        y_min = all_points[:, 1].min()
        x_max = all_points[:, 0].max()
        y_max = all_points[:, 1].max()
        return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)

    def _cluster_boxes_in_frame(self, boxes):
        """Group boxes that are vertically close into clusters.

        Uses adaptive threshold based on box size: cluster boxes if the
        edge-to-edge vertical gap is less than the average height of both boxes.

        Args:
            boxes: List of 4x2 box arrays from one frame

        Returns:
            List of clusters, each cluster is a list of boxes
        """
        if not boxes:
            return []

        def get_box_y_bounds(box):
            _, y_min, _, y_max = self._box_bounds(box)
            return y_min, y_max

        # Sort boxes by Y min (top edge)
        boxes_with_bounds = [(box, *get_box_y_bounds(box)) for box in boxes]
        boxes_with_bounds.sort(key=lambda x: x[1])  # Sort by y_min

        clusters = []
        current_cluster = [boxes_with_bounds[0][0]]
        _, prev_y_min, prev_y_max = boxes_with_bounds[0]
        prev_h = prev_y_max - prev_y_min

        for box, y_min, y_max in boxes_with_bounds[1:]:
            h = y_max - y_min
            # Edge-to-edge gap: top of current box minus bottom of previous box
            edge_gap = y_min - prev_y_max
            avg_height = (prev_h + h) / 2

            if edge_gap <= avg_height:
                # Close enough, add to current cluster
                current_cluster.append(box)
            else:
                # Too far, start new cluster
                clusters.append(current_cluster)
                current_cluster = [box]
            prev_y_min, prev_y_max, prev_h = y_min, y_max, h

        # Don't forget the last cluster
        clusters.append(current_cluster)

        return clusters

    def _merge_vertical_fragments(self, boxes):
        """Merge vertically-stacked boxes near a vertical anchor.

        Some Chinese donghua use large vertical text with big character spacing.
        Detection at 720p may split a vertical label into separate per-character
        boxes (e.g. "神尊" → "神" + "尊"). A nearby vertical anchor (h >= 2*w)
        confirms the area uses vertical text, so any non-anchor boxes that are
        X-aligned with each other and near the anchor get merged.

        Args:
            boxes: List of 4x2 box arrays (in original frame coords).

        Returns:
            List of boxes with vertically-stacked fragments merged.
        """
        if len(boxes) < 2:
            return boxes

        frame_width = self.width
        ANCHOR_X_PROXIMITY = 0.1  # non-anchor must be within 10% frame width of anchor X
        ALIGN_X_THRESHOLD = 0.02  # non-anchors share X centroid within 2% frame width

        anchor_x_proximity_px = frame_width * ANCHOR_X_PROXIMITY
        align_x_threshold_px = frame_width * ALIGN_X_THRESHOLD

        # Step 1: Find vertical anchors (h >= 2*w)
        anchor_indices = set()
        anchor_x_centroids = []
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = self._box_bounds(box)
            w = x_max - x_min
            h = y_max - y_min
            if w > 0 and h >= 2 * w:
                anchor_indices.add(i)
                anchor_x_centroids.append(self._box_centroid(box)[0])

        if not anchor_x_centroids:
            return boxes

        # Step 2: Collect non-anchor boxes that are roughly square and
        # X-proximate to any anchor.  Horizontal text (w >= 2*h) is excluded
        # so disclaimer lines etc. are never treated as vertical fragments.
        candidates = []  # (index, box, cx, cy)
        for i, box in enumerate(boxes):
            if i in anchor_indices:
                continue
            x_min, y_min, x_max, y_max = self._box_bounds(box)
            w = x_max - x_min
            h = y_max - y_min
            if w <= 0 or h <= 0:
                continue
            if max(w, h) / min(w, h) > 2.0:
                continue
            cx, cy = self._box_centroid(box)
            near_anchor = any(
                abs(cx - acx) < anchor_x_proximity_px
                for acx in anchor_x_centroids
            )
            if near_anchor:
                candidates.append((i, box, cx, cy))

        if len(candidates) < 2:
            return boxes

        # Step 3: Group candidates by X alignment
        candidates.sort(key=lambda c: c[2])
        groups = []
        current_group = [candidates[0]]
        for cand in candidates[1:]:
            if abs(cand[2] - current_group[-1][2]) <= align_x_threshold_px:
                current_group.append(cand)
            else:
                groups.append(current_group)
                current_group = [cand]
        groups.append(current_group)

        # Step 4: Merge groups of 2+
        merged_indices = set()
        merged_boxes = []
        for group in groups:
            if len(group) < 2:
                continue
            group_boxes = [c[1] for c in group]
            merged_boxes.append(self._get_encompassing_box(group_boxes))
            for c in group:
                merged_indices.add(c[0])

        if not merged_indices:
            return boxes

        result = [box for i, box in enumerate(boxes) if i not in merged_indices]
        result.extend(merged_boxes)

        return result

    def _crop_cluster_region(self, frame, box, padding=50):
        """Crop region around box with fixed pixel padding.

        Args:
            frame: Source frame
            box: 4x2 box array
            padding: Fixed pixel padding around box

        Returns:
            (crop, (offset_x, offset_y)) or (None, (0, 0)) if invalid
        """
        h, w = frame.shape[:2]
        x_min, y_min, x_max, y_max = self._box_bounds(box)

        crop_x1 = max(0, int(x_min - padding))
        crop_y1 = max(0, int(y_min - padding))
        crop_x2 = min(w, int(x_max + padding))
        crop_y2 = min(h, int(y_max + padding))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None, (0, 0)

        return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy(), (crop_x1, crop_y1)

    def _box_contained_in(self, inner_box, outer_box, max_outside=0.1):
        """Check if inner_box is contained within outer_box.

        At most max_outside (fraction) of the inner box's width/height
        may extend beyond the outer box edges.  This handles the case
        where Phase 3 OCR boxes are slightly larger than the Phase 1
        detection boundary.

        Returns True if the inner box is essentially inside the outer box.
        """
        ix1, iy1, ix2, iy2 = self._box_bounds(inner_box)
        ox1, oy1, ox2, oy2 = self._box_bounds(outer_box)

        iw = ix2 - ix1
        ih = iy2 - iy1
        if iw <= 0 or ih <= 0:
            return False

        # How far each edge of inner extends beyond outer (0 if inside)
        overshoot_left = max(0, ox1 - ix1)
        overshoot_right = max(0, ix2 - ox2)
        overshoot_top = max(0, oy1 - iy1)
        overshoot_bottom = max(0, iy2 - oy2)

        return (
            overshoot_left <= iw * max_outside
            and overshoot_right <= iw * max_outside
            and overshoot_top <= ih * max_outside
            and overshoot_bottom <= ih * max_outside
        )

    def _merge_ocr_by_detection_boxes(self, ocr_lines, detection_boxes):
        """Merge OCR results that are contained within the same detection box.

        Phase 1 detection at 720p gives perfect per-label bounding boxes.
        Phase 3 OCR on a zoomed crop may split a single label into multiple
        character-level boxes.  Any OCR box that is contained within a
        Phase 1 detection box (max 10% outside allowed) belongs to that
        label and gets merged.

        Args:
            ocr_lines: List of {'text', 'confidence', 'box'} from OCR
            detection_boxes: Phase 1 detection boxes (in frame coords)

        Returns:
            List of merged results: one per detection box that had OCR matches,
            plus any OCR results that didn't match any detection box.
        """
        merged = []
        matched_ocr_indices = set()

        for di, det_box in enumerate(detection_boxes):
            # Find all OCR lines contained within this detection box
            matching_lines = []
            for i, line in enumerate(ocr_lines):
                if self._box_contained_in(line["box"], det_box, max_outside=0.1):
                    matching_lines.append(line)
                    matched_ocr_indices.add(i)

            if matching_lines:
                # Deduplicate: PaddleOCR sometimes detects the same text
                # twice at overlapping positions.  Keep the higher-confidence
                # result when two lines have similar text.
                unique_lines = []
                for line in matching_lines:
                    is_dup = False
                    for ui, existing in enumerate(unique_lines):
                        if self._texts_similar(line["text"], existing["text"]):
                            if line["confidence"] > existing["confidence"]:
                                unique_lines[ui] = line
                            is_dup = True
                            break
                    if not is_dup:
                        unique_lines.append(line)
                matching_lines = unique_lines

                # Sort by reading order based on detection box orientation:
                # vertical text (taller than wide) → top to bottom by Y
                # horizontal text → left to right by X
                det_x_min, det_y_min, det_x_max, det_y_max = self._box_bounds(det_box)
                if (det_y_max - det_y_min) > (det_x_max - det_x_min):
                    matching_lines.sort(key=lambda l: self._box_centroid(l["box"])[1])
                else:
                    matching_lines.sort(key=lambda l: self._box_centroid(l["box"])[0])

                # Merge text (concatenate without space for CJK text)
                merged_text = "".join(l["text"] for l in matching_lines)
                # Use max confidence (detection box validates grouping)
                max_conf = max(l["confidence"] for l in matching_lines)

                # Use the actual OCR positions for the result box
                ocr_box = self._get_encompassing_box([l["box"] for l in matching_lines])

                merged.append(
                    {
                        "text": merged_text,
                        "confidence": max_conf,
                        "box": ocr_box,
                    }
                )

        # Also include OCR results that didn't match any detection box
        for i, line in enumerate(ocr_lines):
            if i not in matched_ocr_indices:
                merged.append(line)

        return merged

    def _run_ocr_on_cluster(self, ocr, crop, crop_offset, detection_boxes=None):
        """Run OCR on a cluster crop and return all detected lines.

        Args:
            ocr: PaddleOCR engine
            crop: Image crop containing multiple text lines
            crop_offset: (x, y) offset of crop in original frame
            detection_boxes: Optional list of detection boxes
                            (in original frame coords) to use for merging

        Returns:
            List of dicts, each with:
            - 'text': str
            - 'confidence': float
            - 'box': 4x2 array in original frame coords
        """
        if crop is None or crop.size == 0:
            return []

        # Resize to max 720px, track scale
        crop_resized, scale = self._resize_max_dimension(crop, self.RECOGNIZE_HEIGHT)

        ocr_result = list(ocr.predict(crop_resized))
        pred_data = utils.convert_pred_data_to_old_format(ocr_result)

        if not pred_data or not pred_data[0]:
            return []

        results = []
        offset_x, offset_y = crop_offset

        for item in pred_data[0]:
            if len(item) < 2:
                continue

            # item[0] is the box (4x2 array), item[1] is (text, confidence)
            box_scaled = np.array(item[0], dtype=np.float32)
            text = item[1][0]
            confidence = item[1][1]

            if not text or confidence < 0.5:
                continue

            # Scale box back to crop coords (reverse the resize)
            box_crop = box_scaled / scale

            # Add crop offset to get frame coords
            box_frame = box_crop.copy()
            box_frame[:, 0] += offset_x
            box_frame[:, 1] += offset_y

            results.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "box": box_frame,
                }
            )

        # Merge OCR results by detection boxes
        if detection_boxes and results:
            merged = self._merge_ocr_by_detection_boxes(results, detection_boxes)
            return merged

        return results

    def _crop_box_region(self, frame, box, padding_ratio=0.1):
        """Crop a region around a box with padding."""
        h, w = frame.shape[:2]
        x_min, y_min, x_max, y_max = self._box_bounds(box)

        box_w = x_max - x_min
        box_h = y_max - y_min
        pad_x = int(box_w * padding_ratio)
        pad_y = int(box_h * padding_ratio)

        crop_x1 = max(0, int(x_min - pad_x))
        crop_y1 = max(0, int(y_min - pad_y))
        crop_x2 = min(w, int(x_max + pad_x))
        crop_y2 = min(h, int(y_max + pad_y))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None, (0, 0)

        return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy(), (crop_x1, crop_y1)

    def _run_ocr_on_crop(self, ocr, crop):
        """Run full OCR on a crop and return (text, confidence)."""
        if crop is None or crop.size == 0:
            return None, 0.0

        # Resize to max 720px
        crop_resized, _ = self._resize_max_dimension(crop, self.RECOGNIZE_HEIGHT)

        ocr_result = list(ocr.predict(crop_resized))
        pred_data = utils.convert_pred_data_to_old_format(ocr_result)

        if not pred_data or not pred_data[0]:
            return None, 0.0

        # Combine all text, weighted by confidence
        texts = []
        total_conf = 0.0
        count = 0

        for item in pred_data[0]:
            if len(item) < 2:
                continue
            word_text = item[1][0]
            word_conf = item[1][1]
            if word_text and word_conf > 0.5:
                texts.append(word_text)
                total_conf += word_conf
                count += 1

        if not texts:
            return None, 0.0

        combined_text = " ".join(texts)
        avg_conf = total_conf / count if count > 0 else 0.0

        return combined_text, avg_conf

    # ------------------------------------------------------------------
    # Phase 1: Detection Scan
    # ------------------------------------------------------------------

    def _phase1_find_text_frames(self, det_engine, time_start, time_end, progress=None, cancel_event=None, retain=None):
        """Sparse sampling at 720p to identify frames containing text outside dialogue region.

        Every frame in the range is decoded, but only every
        `sample_interval`-th one is converted to BGR (`read()`); the rest are
        skipped with `grab()`. The sampled frames, their indices and their
        PTS are identical to reading every frame and keeping every Nth
        (pinned by tests/test_label_sampling.py). Decoding stays at native
        resolution.

        If `retain` (a _RetainedCrops) is given, the native-resolution crop
        phase 1.5 will OCR for each box is offered to it, so phase 1.5 need
        not fetch the frame again.

        Returns list of (frame_idx, pts, boxes) where boxes are in original (cropped frame) coords.
        """
        sample_interval = max(1, int(self.fps * self.SAMPLE_INTERVAL_SECONDS))

        start_idx = utils.get_frame_index(time_start, self.fps) if time_start else 0
        end_idx = utils.get_frame_index(time_end, self.fps) if time_end else self.num_frames

        total_samples = (end_idx - start_idx + sample_interval - 1) // sample_interval

        if progress is not None:
            progress.set_phase("label_p1", total_samples)

        text_frames = []
        sample_counter = 0

        with Capture(self.video_path) as cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

            frame_idx = start_idx

            while frame_idx < end_idx:
                if cancel_event is not None and cancel_event.is_set():
                    return text_frames

                # Only process at sample intervals. Frames in between are
                # decoded but never converted to BGR, which at 4K is most of
                # the cost of reading them.
                if (frame_idx - start_idx) % sample_interval != 0:
                    cap.grab()
                    frame_idx += 1
                    continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    frame_idx += 1
                    continue

                pts = cap.get_last_pts() if hasattr(cap, "get_last_pts") else frame_idx / self.fps

                # No brightness filter at detection stage
                self._apply_label_masks(frame)

                # SLICE off dialogue region (don't mask, just slice)
                frame_cropped = frame[: self.dialogue_cutoff_y, :]

                # Downscale for detection at 720p
                scaled, scale = self._downscale(frame_cropped, self.SCAN_HEIGHT)

                # Run detection. At or below SCAN_HEIGHT `scaled` is this
                # frame, which the crops for phase 1.5 are cut from below, so
                # detection gets it read-only: an engine that wrote to its
                # input would raise here instead of changing those crops.
                # (The real engine accepts read-only input and returns the
                # same boxes.)
                if scaled is frame_cropped:
                    scaled = frame_cropped.view()
                    scaled.flags.writeable = False
                boxes = self._run_detection(det_engine, scaled)

                if boxes:
                    # Scale boxes back to original (cropped frame) coords
                    orig_boxes = [box / scale for box in boxes]
                    orig_boxes = self._merge_vertical_fragments(orig_boxes)
                    text_frames.append((frame_idx, pts, orig_boxes))
                    if retain is not None:
                        retain.offer(frame_idx, pts, orig_boxes,
                                     [self._crop_box_region(frame_cropped, box)[0] for box in orig_boxes])

                if progress is not None:
                    progress.update(1)

                sample_counter += 1
                frame_idx += 1

        return text_frames

    # ------------------------------------------------------------------
    # Phase 1.5: Batch OCR to annotate boxes with text
    # ------------------------------------------------------------------

    @staticmethod
    def _read_frame_at_pts(cap, pts):
        """Read the frame whose PTS get_last_pts() reported as `pts`.

        Returns the native-resolution BGR frame, or None if it cannot be
        read or the capture lands on a frame with a different PTS: a
        different frame is never substituted for the one asked for.
        """
        if not cap.seek_to_pts(pts):
            return None
        ret, frame = cap.read()
        if not ret or frame is None or cap.get_last_pts() != pts:
            return None
        return frame

    @staticmethod
    def _read_frame_on_screen(cap, t):
        """Read the frame on screen at time `t`: the last frame whose PTS is
        at most `t`, or the first frame if `t` precedes it (see
        seek_to_display_time). None if `t` is past the last frame or the
        frame cannot be read.

        Phases 3 and 4 record what they find in this frame under `t`.
        set(CAP_PROP_POS_FRAMES, int(t * fps)) could read a frame one or more
        frames away from it (tests/test_label_frame_identity.py).
        """
        if not cap.seek_to_display_time(t):
            return None
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return frame

    def _batch_ocr_text_frames(self, text_frames, ocr, progress=None, retained=None, cancel_event=None):
        """Run OCR on every Phase 1 detection box to annotate with text.

        Replaces each bare np.array box with {"box": np.array, "text": str|None}
        so Phase 2 can use text similarity in addition to spatial proximity.

        Crops phase 1 kept in `retained` are OCR'd as they are. Any other
        frame is fetched by the PTS phase 1 recorded, not by its frame_idx:
        phase 1 counts frame_idx from where its own read started, which is
        not the position set(CAP_PROP_POS_FRAMES) seeks to once the file has
        a container start time (tests/test_label_frame_identity.py). The
        video is only opened if some frame has to be fetched.

        Stops before the next text frame once `cancel_event` is set, returning
        the text frames annotated so far.

        Returns augmented text_frames: [(frame_idx, pts, [{"box": ..., "text": ...}, ...]), ...]
        """
        if progress is not None:
            progress.set_phase("label_p1_5", len(text_frames))

        augmented = []

        with contextlib.ExitStack() as stack:
            cap = None
            for frame_idx, pts, boxes in text_frames:
                if cancel_event is not None and cancel_event.is_set():
                    break
                crops = retained.take(frame_idx, pts, boxes) if retained is not None else None
                if crops is None:
                    if cap is None:
                        cap = stack.enter_context(Capture(self.video_path))
                    crops = self._fetch_crops(cap, pts, boxes)

                if crops is None:
                    # Keep boxes without text if frame unreadable
                    augmented.append((frame_idx, pts, [{"box": box, "text": None} for box in boxes]))
                else:
                    annotated_boxes = []
                    for box, crop in zip(boxes, crops):
                        text, conf = self._run_ocr_on_crop(ocr, crop)
                        annotated_boxes.append({"box": box, "text": text})
                    augmented.append((frame_idx, pts, annotated_boxes))

                if progress is not None:
                    progress.update(1)

        return augmented

    def _fetch_crops(self, cap, pts, boxes):
        """Re-read the frame phase 1 sampled at `pts` and crop `boxes` out of
        it exactly as phase 1 would have; None if the frame cannot be read."""
        frame = self._read_frame_at_pts(cap, pts)
        if frame is None:
            return None

        self._apply_label_masks(frame)

        # Slice to dialogue cutoff (same as Phase 1)
        frame_cropped = frame[: self.dialogue_cutoff_y, :]
        return [self._crop_box_region(frame_cropped, box)[0] for box in boxes]

    # ------------------------------------------------------------------
    # Phase 2: Position Grouping
    # ------------------------------------------------------------------

    def _phase2_group_by_position(self, text_frames, progress=None):
        """Group detection boxes spatially across frames.

        Uses centroid-based overlap with a time gap constraint to prevent
        transitive chaining across the video.  Each group represents a
        screen position where text appears across nearby frames.

        Returns list of groups:
        {
            'encompassing_box': np.array,  # union of all boxes in group
            'observations': [(pts, box), ...],  # sorted by time
            'first_pts': float,
            'last_pts': float,
        }
        """
        if progress is not None:
            progress.set_phase("label_p2", len(text_frames))

        # Max time gap between observations in the same group.
        # Labels separated by more than this are always separate groups.
        max_time_gap = max(5.0, self.label_max_duration * 1.5)

        groups = []

        for frame_idx, pts, boxes in text_frames:
            for entry in boxes:
                box = entry["box"]
                box_text = entry["text"]
                bx_min, by_min, bx_max, by_max = self._box_bounds(box)
                box_w = bx_max - bx_min
                box_h = by_max - by_min
                box_cx, box_cy = self._box_centroid(box)

                merged = False
                for group in groups:
                    # Time proximity: reject if gap exceeds threshold
                    if pts - group["latest_pts"] > max_time_gap:
                        continue

                    # Centroid-based spatial overlap (prevents transitive chaining)
                    gcx, gcy = group["centroid"]
                    g_avg_w, g_avg_h = group["avg_dims"]

                    # Use the SMALLER of (relative threshold, absolute cap)
                    # This prevents huge boxes from absorbing unrelated smaller boxes
                    # Absolute cap: ~3% of frame width/height for 4K video
                    MAX_X_THRESHOLD = self.width * 0.03  # ~115px on 4K
                    MAX_Y_THRESHOLD = self.height * 0.03  # ~65px on 4K

                    avg_w = (g_avg_w + box_w) / 2
                    avg_h = (g_avg_h + box_h) / 2

                    x_threshold = min(avg_w * 0.5, MAX_X_THRESHOLD)
                    y_threshold = min(avg_h * 0.5, MAX_Y_THRESHOLD)

                    dx = abs(box_cx - gcx)
                    dy = abs(box_cy - gcy)

                    if dx < x_threshold and dy < y_threshold:
                        # Text must match to merge (skip check if either has no text)
                        if box_text and group.get("text"):
                            if not self._texts_similar(box_text, group["text"], threshold=self.MERGE_TEXT_SIMILARITY):
                                continue
                        group["boxes"].append(box)
                        group["observations"].append((pts, box))

                        # Update running centroid
                        n = len(group["boxes"])
                        group["centroid"] = (
                            gcx + (box_cx - gcx) / n,
                            gcy + (box_cy - gcy) / n,
                        )
                        # Update running average dimensions
                        group["avg_dims"] = (
                            g_avg_w + (box_w - g_avg_w) / n,
                            g_avg_h + (box_h - g_avg_h) / n,
                        )
                        group["latest_pts"] = max(group["latest_pts"], pts)
                        if box_text and not group.get("text"):
                            group["text"] = box_text
                        merged = True
                        break

                if not merged:
                    groups.append(
                        {
                            "boxes": [box],
                            "observations": [(pts, box)],
                            "centroid": (box_cx, box_cy),
                            "avg_dims": (box_w, box_h),
                            "latest_pts": pts,
                            "text": box_text,
                        }
                    )

            if progress is not None:
                progress.update(1)

        # Finalize groups: compute encompassing box, sort observations, get time range
        result_groups = []
        for group in groups:
            observations = sorted(group["observations"], key=lambda x: x[0])
            encompassing = self._get_encompassing_box(group["boxes"])
            result_groups.append(
                {
                    "encompassing_box": encompassing,
                    "observations": observations,
                    "first_pts": observations[0][0],
                    "last_pts": observations[-1][0],
                }
            )

        # Filter out moving/transient groups
        result_groups = self._filter_moving_groups(result_groups)
        result_groups = self._merge_overlapping_groups(result_groups)

        return result_groups

    def _filter_moving_groups(self, groups):
        """Filter out groups with moving text or insufficient observations.

        Static labels/nameplates should:
        1. Be detected in multiple frames (≥2 observations)
        2. Have consistent positions across frames (centroid drift <50px)
        3. Have sufficient observation density (detected in most samples over their time span)

        Animated text (scrolls, transitions) fails these criteria.

        Args:
            groups: List of position groups from Phase 2.

        Returns:
            Filtered list of groups.
        """
        MIN_OBSERVATIONS = 3  # Must be seen in at least 3 frames
        MAX_CENTROID_DRIFT = 50  # Max allowed centroid movement in pixels
        MIN_OBSERVATION_DENSITY = 0.3  # Must be detected in at least 30% of expected samples

        filtered = []

        for group in groups:
            observations = group["observations"]
            num_obs = len(observations)
            first_pts = group["first_pts"]
            last_pts = group["last_pts"]
            time_span = last_pts - first_pts

            # Filter 1: Minimum observations
            if num_obs < MIN_OBSERVATIONS:
                continue

            # Filter 2: Observation density
            # A static label should be detected consistently over its visible period
            if time_span > self.SAMPLE_INTERVAL_SECONDS:
                expected_obs = time_span / self.SAMPLE_INTERVAL_SECONDS
                density = num_obs / expected_obs
                if density < MIN_OBSERVATION_DENSITY:
                    continue

            # Filter 3: Position stability (calculate centroid drift)
            centroids_x = []
            centroids_y = []
            for _, box in observations:
                cx, cy = self._box_centroid(box)
                centroids_x.append(cx)
                centroids_y.append(cy)

            x_drift = max(centroids_x) - min(centroids_x)
            y_drift = max(centroids_y) - min(centroids_y)
            max_drift = max(x_drift, y_drift)

            if max_drift > MAX_CENTROID_DRIFT:
                continue

            filtered.append(group)

        return filtered

    def _merge_overlapping_groups(self, groups):
        """Merge Phase 2 groups that overlap in both time and space.

        PaddleOCR inconsistently detects long labels — one frame sees the full
        box, another sees just the top fragment, another just the bottom.
        Phase 2 creates separate groups for each fragment because their
        centroids are far apart. This method merges groups whose encompassing
        boxes overlap spatially (≥50% of the smaller box's area) during
        overlapping time ranges, so that Phase 3 OCRs the full region once.

        Uses union-find (same pattern as _merge_split_labels).

        Args:
            groups: List of position groups from Phase 2.

        Returns:
            List of groups with overlapping ones merged.
        """
        if len(groups) < 2:
            return groups

        n = len(groups)
        initial_count = n
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        def _box_rect(box):
            """Extract (x_min, y_min, x_max, y_max) from a 4x2 box array."""
            xs = box[:, 0]
            ys = box[:, 1]
            return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

        for i in range(n):
            for j in range(i + 1, n):
                gi, gj = groups[i], groups[j]

                # Check time overlap: ranges must intersect
                if gi["first_pts"] > gj["last_pts"] or gj["first_pts"] > gi["last_pts"]:
                    continue

                # Check spatial overlap of encompassing boxes
                x1_min, y1_min, x1_max, y1_max = _box_rect(gi["encompassing_box"])
                x2_min, y2_min, x2_max, y2_max = _box_rect(gj["encompassing_box"])

                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)

                if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                    continue  # No spatial intersection

                inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                area_i = (x1_max - x1_min) * (y1_max - y1_min)
                area_j = (x2_max - x2_min) * (y2_max - y2_min)
                smaller_area = min(area_i, area_j)

                if smaller_area > 0 and inter_area / smaller_area >= 0.5:
                    union(i, j)

        # Collect union-find groups
        merged_map = {}
        for i in range(n):
            merged_map.setdefault(find(i), []).append(i)

        result = []
        for indices in merged_map.values():
            if len(indices) == 1:
                result.append(groups[indices[0]])
            else:
                # Merge: union encompassing boxes, combine observations, extend time
                all_boxes = []
                all_observations = []
                for idx in indices:
                    g = groups[idx]
                    all_boxes.append(g["encompassing_box"])
                    all_observations.extend(g["observations"])

                all_observations.sort(key=lambda x: x[0])
                encompassing = self._get_encompassing_box(all_boxes)

                result.append({
                    "encompassing_box": encompassing,
                    "observations": all_observations,
                    "first_pts": all_observations[0][0],
                    "last_pts": all_observations[-1][0],
                })

        return result

    # ------------------------------------------------------------------
    # Phase 3: Crop, Clean, Recognize
    # ------------------------------------------------------------------

    def _dual_ocr_crop(self, ocr, raw_crop, detection_boxes=None):
        """Run OCR on both brightness-filtered and raw crop, return best.

        Args:
            ocr: PaddleOCR engine.
            raw_crop: Raw crop from full-res frame.
            detection_boxes: Optional detection boxes for merging.

        Returns:
            (text, confidence, winner, cleaned_crop, ocr_input)
            where winner is "cleaned" or "raw".
        """
        # Attempt A: cleaned (brightness-filtered)
        cleaned_crop = self._apply_brightness_filter(raw_crop.copy()) if self.brightness_threshold else None
        cleaned_text, cleaned_conf = None, 0.0

        if cleaned_crop is not None:
            cleaned_text, cleaned_conf = self._run_ocr_on_crop(ocr, cleaned_crop)

        # Attempt B: raw
        raw_text, raw_conf = self._run_ocr_on_crop(ocr, raw_crop)

        # Pick winner by max confidence
        if cleaned_text and cleaned_conf > raw_conf:
            winner = "cleaned"
            text, confidence = cleaned_text, cleaned_conf
            ocr_input, _ = self._resize_max_dimension(cleaned_crop, self.RECOGNIZE_HEIGHT)
        else:
            winner = "raw"
            text, confidence = raw_text, raw_conf
            ocr_input, _ = self._resize_max_dimension(raw_crop, self.RECOGNIZE_HEIGHT)

        return text, confidence, winner, cleaned_crop, ocr_input

    def _cluster_groups_for_ocr(self, groups):
        """Cluster Phase 2 groups that are spatially close for batch OCR.

        Groups whose encompassing boxes are near each other (in either X or Y)
        get batched into one crop so PaddleOCR processes them in a single call.
        Uses edge-to-edge gap vs average dimension, same logic as in-frame
        clustering but applied in both directions.

        Returns list of clusters, each cluster is a list of groups.
        """
        if len(groups) <= 1:
            return [groups]

        # Greedy spatial clustering: merge group into a cluster if its box
        # is close to any box already in that cluster.
        clusters = []  # Each: list of groups

        for group in groups:
            gbox = group["encompassing_box"]
            gx_min, gy_min, gx_max, gy_max = self._box_bounds(gbox)
            gw, gh = gx_max - gx_min, gy_max - gy_min

            merged = False
            for cluster in clusters:
                for existing in cluster:
                    ebox = existing["encompassing_box"]
                    ex_min, ey_min, ex_max, ey_max = self._box_bounds(ebox)
                    ew, eh = ex_max - ex_min, ey_max - ey_min

                    # Must overlap in time to batch together
                    g_first, g_last = group["first_pts"], group["last_pts"]
                    e_first, e_last = existing["first_pts"], existing["last_pts"]
                    if g_first > e_last or e_first > g_last:
                        continue

                    # Edge-to-edge gaps (negative means overlap)
                    x_gap = max(gx_min - ex_max, ex_min - gx_max)
                    y_gap = max(gy_min - ey_max, ey_min - gy_max)

                    min_h = min(gh, eh)
                    min_w = min(gw, ew)

                    # Tight threshold: gap must be less than the smaller
                    # box's dimension. Only clusters truly adjacent boxes
                    # (e.g. stacked disclaimer lines).
                    x_close = x_gap < min_w
                    y_close = y_gap < min_h

                    if x_close and y_close:
                        cluster.append(group)
                        merged = True
                        break
                if merged:
                    break

            if not merged:
                clusters.append([group])

        return clusters

    def _assign_ocr_lines_to_groups(self, ocr_lines, cluster_groups):
        """Assign OCR result lines to groups by closest centroid with max distance.

        Each OCR line is assigned to the closest group, but only if the line's
        centroid is within 1x the group's box height (Y) and width (X).
        Lines that don't match any group are discarded.

        Returns dict mapping group index -> list of OCR line dicts.
        """
        assigned = {gi: [] for gi in range(len(cluster_groups))}

        for line in ocr_lines:
            line_cx, line_cy = self._box_centroid(line["box"])

            best_gi = None
            best_dist = float("inf")

            for gi, group in enumerate(cluster_groups):
                gbox = group["encompassing_box"]
                g_cx, g_cy = self._box_centroid(gbox)
                gx_min, gy_min, gx_max, gy_max = self._box_bounds(gbox)
                g_h = gy_max - gy_min
                g_w = gx_max - gx_min

                dx = abs(line_cx - g_cx)
                dy = abs(line_cy - g_cy)

                # Must be within 1x group dimensions to be considered
                if dy > g_h or dx > g_w:
                    continue

                dist = dy  # Primary sort by vertical distance
                if dist < best_dist:
                    best_dist = dist
                    best_gi = gi

            if best_gi is not None:
                assigned[best_gi].append(line)

        return assigned

    def _merge_group_lines(self, lines):
        """Merge multiple OCR lines assigned to one group into (text, confidence, box).

        Sorts lines left-to-right by X centroid and concatenates text.
        Returns (text, max_confidence, encompassing_box) or (None, 0.0, None) if no lines.
        """
        if not lines:
            return None, 0.0, None

        lines.sort(key=lambda l: self._box_centroid(l["box"])[0])
        text = "".join(l["text"] for l in lines)
        conf = max(l["confidence"] for l in lines)
        box = self._get_encompassing_box([l["box"] for l in lines])
        return text, conf, box

    def _phase3_ocr_and_segment(self, groups, ocr, progress=None, cancel_event=None):
        """OCR at regular intervals, detect content changes to split back-to-back labels.

        Clusters spatially close groups and sends one combined crop per cluster
        to PaddleOCR, then maps per-line results back to individual groups.

        For each cluster of groups:
        1. Crop one image covering all groups in the cluster
        2. Run dual OCR (cleaned + raw) once per sample
        3. Assign OCR lines to individual groups by Y position
        4. Per-group content change detection to split back-to-back labels

        Returns list of label segments:
        {
            'box': encompassing_box,
            'text': str,
            'confidence': float,
            'start_pts': float,
            'end_pts': float,
        }
        """
        # Cluster groups for batch OCR, tracking original indices
        ocr_clusters = self._cluster_groups_for_ocr(groups)
        # Map each group to its original index via identity
        group_to_idx = {id(g): i for i, g in enumerate(groups)}

        # Count total OCR samples for progress
        total_samples = 0
        for cluster in ocr_clusters:
            first_pts = min(g["first_pts"] for g in cluster)
            last_pts = max(g["last_pts"] for g in cluster)
            span = last_pts - first_pts
            total_samples += max(1, int(span / self.SAMPLE_INTERVAL_SECONDS) + 1)

        if progress is not None:
            progress.set_phase("label_p3", total_samples)

        all_segments = []

        with Capture(self.video_path) as cap:
            for ci, cluster in enumerate(ocr_clusters):
                if cancel_event is not None and cancel_event.is_set():
                    return all_segments

                # Combined encompassing box for the cluster
                combined_box = self._get_encompassing_box([g["encompassing_box"] for g in cluster])

                # Union time range across all groups in cluster
                first_pts = min(g["first_pts"] for g in cluster)
                last_pts = max(g["last_pts"] for g in cluster)

                # Determine sample points
                sample_pts_list = []
                pts = first_pts
                while pts <= last_pts:
                    sample_pts_list.append(pts)
                    pts += self.SAMPLE_INTERVAL_SECONDS
                if not sample_pts_list:
                    sample_pts_list = [first_pts]

                # Per-group OCR result tracking
                group_ocr_results = [[] for _ in cluster]

                for si, sample_pts in enumerate(sample_pts_list):
                    # Sample times never pass last_pts, a frame phase 1 read,
                    # so there is no end of stream to check for here.
                    frame = self._read_frame_on_screen(cap, sample_pts)
                    if frame is None:
                        if progress is not None:
                            progress.update(1)
                        continue

                    self._apply_label_masks(frame)

                    # Crop combined region covering all groups in cluster
                    raw_crop, offset = self._crop_cluster_region(frame, combined_box, padding=50)
                    if raw_crop is None:
                        if progress is not None:
                            progress.update(1)
                        continue

                    # Use Phase 1 detection boundaries (group encompassing boxes)
                    # as merge containers.  Phase 1 at 720p gives perfect
                    # per-label boxes; Phase 3 on zoomed crops may split a
                    # label into character-level fragments.  Any OCR box
                    # contained within a Phase 1 boundary is the same label.
                    det_boxes = [g["encompassing_box"] for g in cluster]

                    # Run cleaned OCR first (primary pass — handles most white text)
                    cleaned_crop = None
                    cleaned_lines = []
                    if self.brightness_threshold:
                        cleaned_crop = self._apply_brightness_filter(raw_crop.copy())
                        cleaned_lines = self._run_ocr_on_cluster(ocr, cleaned_crop, offset, detection_boxes=det_boxes)

                    cleaned_assigned = self._assign_ocr_lines_to_groups(cleaned_lines, cluster)

                    # Check if all groups have high confidence from cleaned pass
                    all_high_conf = bool(self.brightness_threshold)
                    if all_high_conf:
                        for gi, group in enumerate(cluster):
                            _, cleaned_conf, _ = self._merge_group_lines(cleaned_assigned[gi])
                            if cleaned_conf < 0.92:
                                all_high_conf = False
                                break

                    # Only run raw OCR if cleaned confidence is low (colored text fallback)
                    raw_lines = []
                    if not all_high_conf:
                        raw_lines = self._run_ocr_on_cluster(ocr, raw_crop, offset, detection_boxes=det_boxes)

                    raw_assigned = self._assign_ocr_lines_to_groups(raw_lines, cluster)

                    # Per group: merge lines, pick best between raw and cleaned
                    for gi, group in enumerate(cluster):
                        raw_text, raw_conf, raw_box = self._merge_group_lines(raw_assigned[gi])
                        cleaned_text, cleaned_conf, cleaned_box = self._merge_group_lines(cleaned_assigned[gi])

                        # Pick winner
                        if cleaned_text and cleaned_conf > raw_conf:
                            text, confidence, text_box = cleaned_text, cleaned_conf, cleaned_box
                        else:
                            text, confidence, text_box = raw_text, raw_conf, raw_box

                        if text and confidence >= self.conf_threshold_min:
                            text = self._deduplicate_text(text)
                            group_ocr_results[gi].append((sample_pts, text, confidence, text_box))

                    if progress is not None:
                        progress.update(1)

                # Segment by content per group
                for gi, group in enumerate(cluster):
                    global_gi = group_to_idx[id(group)]
                    segments = self._best_single_reading(group_ocr_results[gi], group["encompassing_box"])
                    all_segments.extend(segments)

        return all_segments

    def _segment_by_content(self, ocr_results, fallback_box):
        """Walk through OCR results chronologically, detect text changes.

        Args:
            ocr_results: [(pts, text, confidence, box), ...] sorted by time.
                box is the actual OCR text position (or None).
            fallback_box: Group encompassing box, used when per-reading box is unavailable.

        Returns:
            List of segments: {'box', 'text', 'confidence', 'start_pts', 'end_pts'}
        """
        if not ocr_results:
            return []

        segments = []
        current_readings = []  # [(pts, text, confidence, box)]

        for pts, text, confidence, text_box in ocr_results:
            if not current_readings:
                # Start new segment
                current_readings.append((pts, text, confidence, text_box))
                continue

            # Compare with current segment text (use the best reading so far)
            _, best_text, _, _ = max(current_readings, key=lambda x: x[2])

            if self._texts_similar(text, best_text):
                # Same text, extend segment
                current_readings.append((pts, text, confidence, text_box))
            else:
                # Text changed — finalize current segment
                seg = self._finalize_segment(current_readings, fallback_box)
                if seg:
                    segments.append(seg)
                # Start new segment
                current_readings = [(pts, text, confidence, text_box)]

        # Finalize last segment
        if current_readings:
            seg = self._finalize_segment(current_readings, fallback_box)
            if seg:
                segments.append(seg)

        return segments

    def _finalize_segment(self, readings, fallback_box):
        """Create a segment from a list of similar readings.

        Picks the reading with highest confidence as canonical text.
        Uses the best reading's OCR box for position; falls back to
        the group encompassing box if no per-reading box is available.

        Args:
            readings: [(pts, text, confidence, box), ...]
            fallback_box: Group encompassing box.

        Returns:
            Segment dict or None if invalid.
        """
        if not readings:
            return None

        # Pick best by confidence
        best_pts, best_text, best_conf, best_box = max(readings, key=lambda x: x[2])

        # Use actual OCR box if available, otherwise fall back to group box
        seg_box = best_box if best_box is not None else fallback_box

        return {
            "box": seg_box,
            "text": best_text,
            "confidence": best_conf,
            "start_pts": readings[0][0],
            "end_pts": readings[-1][0],
        }

    def _best_single_reading(self, ocr_results, fallback_box):
        """Pick the single best OCR reading from all results for a group.

        After merging overlapping groups, each group corresponds to one
        spatial label.  Instead of segmenting by content changes, we just
        pick the longest text above conf_threshold_min (ties broken by
        highest confidence) and return a single segment spanning the full
        time range.

        Args:
            ocr_results: [(pts, text, confidence, box), ...] sorted by time.
            fallback_box: Group encompassing box.

        Returns:
            List with zero or one segment dicts.
        """
        if not ocr_results:
            return []

        # Filter to readings above minimum confidence
        valid = [(pts, text, conf, box) for pts, text, conf, box in ocr_results
                 if text and conf >= self.conf_threshold_min]

        if not valid:
            return []

        # Pick longest text; break ties by highest confidence
        best_pts, best_text, best_conf, best_box = max(
            valid, key=lambda x: (len(x[1]), x[2])
        )

        best_text = self._deduplicate_text(best_text)
        seg_box = best_box if best_box is not None else fallback_box

        return [{
            "box": seg_box,
            "text": best_text,
            "confidence": best_conf,
            "start_pts": ocr_results[0][0],
            "end_pts": ocr_results[-1][0],
        }]

    # ------------------------------------------------------------------
    # Phase 4: Timing Refinement
    # ------------------------------------------------------------------

    def _get_reference_box(self, cap, det_engine, encompassing_box, ref_pts):
        """Get the actual detection box at a known PTS for size matching.

        Runs detection at ref_pts and returns the detected box closest to
        the encompassing box centroid. Falls back to encompassing_box if
        detection fails.
        """
        frame = self._read_frame_on_screen(cap, ref_pts)
        if frame is None:
            return encompassing_box

        self._apply_label_masks(frame)
        frame_cropped = frame[: self.dialogue_cutoff_y, :]

        # Detect in cropped ROI around target for efficiency
        orig_boxes = self._detect_in_roi(det_engine, frame_cropped, encompassing_box)

        if not orig_boxes:
            return encompassing_box

        target_cx, target_cy = self._box_centroid(encompassing_box)
        best_box = None
        best_dist = float("inf")

        for orig_box in orig_boxes:
            cx, cy = self._box_centroid(orig_box)
            dist = ((cx - target_cx) ** 2 + (cy - target_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_box = orig_box

        return best_box if best_box is not None else encompassing_box

    def _crop_roi_for_detection(self, frame_cropped, target_box):
        """Extract a padded ROI around target_box for efficient detection.

        Returns (roi, roi_x1, roi_y1) or (None, 0, 0) if invalid.
        The ROI is NOT downscaled — caller handles that.
        """
        tx_min, ty_min, tx_max, ty_max = self._box_bounds(target_box)
        tw, th = tx_max - tx_min, ty_max - ty_min
        pad_x = max(int(tw * 0.5), 50)
        pad_y = max(int(th * 1.0), 50)

        roi_y1 = max(0, int(ty_min) - pad_y)
        roi_y2 = min(frame_cropped.shape[0], int(ty_max) + pad_y)
        roi_x1 = max(0, int(tx_min) - pad_x)
        roi_x2 = min(frame_cropped.shape[1], int(tx_max) + pad_x)

        if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
            return None, 0, 0

        roi = frame_cropped[roi_y1:roi_y2, roi_x1:roi_x2]
        return roi, roi_x1, roi_y1

    def _detect_in_roi(self, det_engine, frame_cropped, target_box):
        """Run detection on a cropped ROI and return boxes in original frame coords."""
        roi, roi_x1, roi_y1 = self._crop_roi_for_detection(frame_cropped, target_box)
        if roi is None:
            return []

        # Only downscale if ROI is taller than SCAN_HEIGHT
        roi_h = roi.shape[0]
        if roi_h > self.SCAN_HEIGHT:
            scaled, scale = self._downscale(roi, self.SCAN_HEIGHT)
        else:
            scaled, scale = roi, 1.0

        boxes = self._run_detection(det_engine, scaled)

        # Map boxes back to original frame_cropped coords
        orig_boxes = []
        for box in boxes:
            b = box / scale
            offset = np.array([[roi_x1, roi_y1]] * 4, dtype=np.float32)
            orig_boxes.append(b + offset)

        orig_boxes = self._merge_vertical_fragments(orig_boxes)
        return orig_boxes

    def _box_detected_at_position(self, det_engine, frame, target_box, ref_box=None):
        """Check if a box is detected near target position with matching size.

        When ref_box is provided, also checks that the detected box has similar
        width to the reference — this distinguishes back-to-back labels at the
        same position (e.g. 4-char vs 12-char text).
        """
        # Slice off dialogue region
        frame_cropped = frame[: self.dialogue_cutoff_y, :]

        # Detect in cropped ROI around target for efficiency
        orig_boxes = self._detect_in_roi(det_engine, frame_cropped, target_box)

        target_cx, target_cy = self._box_centroid(target_box)
        target_x_min, target_y_min, target_x_max, target_y_max = self._box_bounds(target_box)
        target_height = target_y_max - target_y_min

        # Reference box dimensions for size matching
        ref_width = None
        if ref_box is not None:
            ref_x_min, _, ref_x_max, _ = self._box_bounds(ref_box)
            ref_width = ref_x_max - ref_x_min

        # Very strict threshold: centroid must be within 15% of box height or 20px
        y_threshold = max(20, target_height * 0.15)

        for orig_box in orig_boxes:
            cx, cy = self._box_centroid(orig_box)

            # Check Y centroid is very close (strict)
            if abs(cy - target_cy) > y_threshold:
                continue

            # Check X centroid is reasonably close (within 20% of frame width)
            if abs(cx - target_cx) > self.width * 0.2:
                continue

            # Size matching: width must be within 40% of reference
            if ref_width is not None and ref_width > 0:
                det_x_min, _, det_x_max, _ = self._box_bounds(orig_box)
                det_width = det_x_max - det_x_min
                if abs(det_width - ref_width) / ref_width > 0.4:
                    continue

            return True

        return False

    def _scan_for_start(self, cap, det_engine, box, discovery_pts, ref_box=None, lower_bound=None):
        """Scan backward to find where label starts appearing."""
        step = self.TIMING_SCAN_INTERVAL
        consecutive_absent = 0
        last_present_pts = discovery_pts

        pts = discovery_pts - step
        min_pts = max(0, discovery_pts - self.TIMING_SCAN_MAX_DURATION)
        if lower_bound is not None:
            min_pts = max(min_pts, lower_bound)

        while pts >= min_pts:
            if int(pts * self.fps) < 0:
                break

            frame = self._read_frame_on_screen(cap, pts)
            if frame is None:
                pts -= step
                continue

            # No brightness filter at detection stage
            self._apply_label_masks(frame)
            if self._box_detected_at_position(det_engine, frame, box, ref_box):
                last_present_pts = pts
                consecutive_absent = 0
            else:
                consecutive_absent += 1
                if consecutive_absent >= 2:  # Require 2 consecutive absences
                    break

            pts -= step

        return last_present_pts

    def _scan_for_end(self, cap, det_engine, box, discovery_pts, ref_box=None, upper_bound=None):
        """Scan forward to find where label stops appearing.

        Ends at the time bound, after two consecutive absences, or at the
        first time past the last frame: the end of the stream is where the
        display-time seek finds no frame, not where int(pts * fps) reaches the
        frame count, which on a file whose first frame is after time 0 comes
        up to that offset early. (A seek whose every retry lands late also
        finds no frame; that is logged, and ends the scan the same way.)
        """
        step = self.TIMING_SCAN_INTERVAL
        consecutive_absent = 0
        last_present_pts = discovery_pts

        pts = discovery_pts + step
        max_pts = discovery_pts + self.TIMING_SCAN_MAX_DURATION
        if upper_bound is not None:
            max_pts = min(max_pts, upper_bound)

        while pts <= max_pts:
            if not cap.seek_to_display_time(pts):
                break  # past the last frame, and so is every later time
            ret, frame = cap.read()
            if not ret or frame is None:
                pts += step
                continue

            # No brightness filter at detection stage
            self._apply_label_masks(frame)
            if self._box_detected_at_position(det_engine, frame, box, ref_box):
                last_present_pts = pts
                consecutive_absent = 0
            else:
                consecutive_absent += 1
                if consecutive_absent >= 2:  # Require 2 consecutive absences
                    break

            pts += step

        return last_present_pts

    def _phase4_find_timing(self, segments, det_engine, progress=None, cancel_event=None):
        """Refine timing for each label segment using detection scanning.

        For back-to-back segments at the same position, constrains the timing
        scan so segments don't bleed into each other. Uses the midpoint between
        adjacent segments as the boundary.

        Returns list of LabelResult.
        """
        if progress is not None:
            progress.set_phase("label_p4", len(segments))

        results = []

        with Capture(self.video_path) as cap:
            for li, seg in enumerate(segments):
                if cancel_event is not None and cancel_event.is_set():
                    return results

                box = seg["box"]
                text = seg["text"]
                start_pts = seg["start_pts"]
                end_pts = seg["end_pts"]

                # Get reference detection box at segment midpoint for size matching
                ref_pts = (start_pts + end_pts) / 2
                ref_box = self._get_reference_box(cap, det_engine, box, ref_pts)

                # Compute bounds from adjacent segments at the same position
                # as fallback safety net (size matching is the primary discriminator)
                lower_bound = None
                upper_bound = None
                if li > 0:
                    prev = segments[li - 1]
                    if self._boxes_overlap(box, prev["box"]):
                        lower_bound = (prev["end_pts"] + start_pts) / 2
                if li < len(segments) - 1:
                    nxt = segments[li + 1]
                    if self._boxes_overlap(box, nxt["box"]):
                        upper_bound = (end_pts + nxt["start_pts"]) / 2

                # Scan backward/forward to find precise boundaries
                refined_start = self._scan_for_start(cap, det_engine, box, start_pts, ref_box, lower_bound)
                refined_end = self._scan_for_end(cap, det_engine, box, end_pts, ref_box, upper_bound)

                # Check duration
                duration = refined_end - refined_start
                if duration < self.label_min_duration or duration > self.label_max_duration:
                    if progress is not None:
                        progress.update(1)
                    continue

                # Create result
                cx, cy = self._box_centroid(box)
                x_min, y_min, x_max, y_max = self._box_bounds(box)
                pos_x = int(cx)
                pos_y = int(y_max) + 40

                results.append(
                    LabelResult(
                        start_pts=refined_start,
                        end_pts=refined_end,
                        text=text,
                        pos_x=pos_x,
                        pos_y=pos_y,
                        bbox_x_min=x_min,
                        bbox_y_min=y_min,
                        bbox_x_max=x_max,
                        bbox_y_max=y_max,
                    )
                )

                if progress is not None:
                    progress.update(1)

        return results

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _remove_duplicates(self, labels):
        """Remove duplicate labels with overlapping time, similar position, and similar text.

        Requires text similarity for all merges — different text means different labels,
        even at nearby positions (e.g. multi-line disclaimers).
        """
        if len(labels) < 2:
            return labels

        to_remove = set()

        for i, label_i in enumerate(labels):
            if i in to_remove:
                continue

            for j, label_j in enumerate(labels):
                if i >= j or j in to_remove:
                    continue

                # Check time overlap first
                overlap_start = max(label_i.start_pts, label_j.start_pts)
                overlap_end = min(label_i.end_pts, label_j.end_pts)
                if overlap_end <= overlap_start:
                    continue

                # Must have similar text to be considered duplicates
                if not self._texts_similar(label_i.text, label_j.text):
                    continue

                # Check position proximity (within 10% of frame dimensions)
                dx = abs(label_i.pos_x - label_j.pos_x)
                dy = abs(label_i.pos_y - label_j.pos_y)

                if dx < self.width * 0.1 and dy < self.height * 0.1:
                    # Keep the one with longer duration
                    dur_i = label_i.end_pts - label_i.start_pts
                    dur_j = label_j.end_pts - label_j.start_pts
                    if dur_i >= dur_j:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break

        return [l for i, l in enumerate(labels) if i not in to_remove]

    def _merge_split_labels(self, labels):
        """Merge labels that are fragments of a single visual label split by OCR.

        PaddleOCR sometimes splits a text overlay into multiple detection boxes.
        This uses orientation-aware rules (bbox shape) and edge-to-edge gap
        measurements (not centroid distance) to decide what to merge.

        Pass 1: Merge horizontal fragments (width > height, same Y row, close
                 horizontally).
        Pass 2: Merge vertical fragments (height > width, same X column, close
                 vertically).

        Args:
            labels: List of LabelResult objects.

        Returns:
            List of LabelResult with split fragments merged.
        """
        if len(labels) < 2:
            return labels

        initial_count = len(labels)

        def _bbox_wh(label):
            """Return (width, height) of label bbox, or None if bbox missing."""
            if (label.bbox_x_min is None or label.bbox_x_max is None or
                    label.bbox_y_min is None or label.bbox_y_max is None):
                return None
            return (label.bbox_x_max - label.bbox_x_min,
                    label.bbox_y_max - label.bbox_y_min)

        def _centroid(label):
            """Return (cx, cy) of label bbox."""
            cx = (label.bbox_x_min + label.bbox_x_max) / 2
            cy = (label.bbox_y_min + label.bbox_y_max) / 2
            return cx, cy

        def _time_overlaps(li, lj):
            """Check >50% time overlap relative to shorter duration."""
            overlap_start = max(li.start_pts, lj.start_pts)
            overlap_end = min(li.end_pts, lj.end_pts)
            overlap = max(0, overlap_end - overlap_start)
            shorter_dur = min(li.end_pts - li.start_pts,
                              lj.end_pts - lj.start_pts)
            return shorter_dur > 0 and overlap / shorter_dur >= 0.5

        def _run_union_find(labels, pair_test):
            """Run union-find over labels using pair_test(i, j) -> bool."""
            n = len(labels)
            parent = list(range(n))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for i in range(n):
                for j in range(i + 1, n):
                    if pair_test(i, j):
                        union(i, j)

            groups = {}
            for i in range(n):
                groups.setdefault(find(i), []).append(i)
            return groups

        def _merge_group(members, sort_key):
            """Merge a list of LabelResult members into one."""
            members.sort(key=sort_key)
            text = " ".join(m.text for m in members)
            start_pts = min(m.start_pts for m in members)
            end_pts = max(m.end_pts for m in members)
            bx_min = min(m.bbox_x_min for m in members if m.bbox_x_min is not None)
            by_min = min(m.bbox_y_min for m in members if m.bbox_y_min is not None)
            bx_max = max(m.bbox_x_max for m in members if m.bbox_x_max is not None)
            by_max = max(m.bbox_y_max for m in members if m.bbox_y_max is not None)
            cx = (bx_min + bx_max) / 2
            pos_x = int(cx)
            pos_y = int(by_max) + 40
            return LabelResult(
                start_pts=start_pts, end_pts=end_pts, text=text,
                pos_x=pos_x, pos_y=pos_y,
                bbox_x_min=bx_min, bbox_y_min=by_min,
                bbox_x_max=bx_max, bbox_y_max=by_max,
            )

        def _collect_groups(labels, groups, sort_key):
            """Collect merged results from union-find groups."""
            result = []
            for indices in groups.values():
                if len(indices) == 1:
                    result.append(labels[indices[0]])
                else:
                    members = [labels[i] for i in indices]
                    result.append(_merge_group(members, sort_key))
            return result

        # Precompute bbox info; labels without bbox can't participate
        wh_cache = {i: _bbox_wh(l) for i, l in enumerate(labels)}

        # --- Pass 1: Merge horizontal fragments ---
        def horiz_pair(i, j):
            wh_i, wh_j = wh_cache.get(i), wh_cache.get(j)
            if wh_i is None or wh_j is None:
                return False
            wi, hi = wh_i
            wj, hj = wh_j
            # Both must be horizontal
            if wi <= hi or wj <= hj:
                return False
            if not _time_overlaps(labels[i], labels[j]):
                return False
            min_h = min(hi, hj)
            # Same Y row: centroid Y difference <= min bbox height
            _, cy_i = _centroid(labels[i])
            _, cy_j = _centroid(labels[j])
            if abs(cy_i - cy_j) > min_h:
                return False
            # Edge-to-edge horizontal gap <= min bbox height
            li, lj = labels[i], labels[j]
            if li.bbox_x_min <= lj.bbox_x_min:
                gap = lj.bbox_x_min - li.bbox_x_max
            else:
                gap = li.bbox_x_min - lj.bbox_x_max
            return gap <= min_h

        groups = _run_union_find(labels, horiz_pair)
        labels = _collect_groups(labels, groups,
                                 sort_key=lambda l: l.bbox_x_min)
        # Rebuild cache after merge
        wh_cache = {i: _bbox_wh(l) for i, l in enumerate(labels)}

        # --- Pass 2: Merge vertical fragments ---
        def vert_pair(i, j):
            wh_i, wh_j = wh_cache.get(i), wh_cache.get(j)
            if wh_i is None or wh_j is None:
                return False
            wi, hi = wh_i
            wj, hj = wh_j
            # Both must be vertical
            if hi <= wi or hj <= wj:
                return False
            if not _time_overlaps(labels[i], labels[j]):
                return False
            min_w = min(wi, wj)
            # Same X column: centroid X difference <= min bbox width
            cx_i, _ = _centroid(labels[i])
            cx_j, _ = _centroid(labels[j])
            if abs(cx_i - cx_j) > min_w:
                return False
            # Edge-to-edge vertical gap <= min bbox width
            li, lj = labels[i], labels[j]
            if li.bbox_y_min <= lj.bbox_y_min:
                gap = lj.bbox_y_min - li.bbox_y_max
            else:
                gap = li.bbox_y_min - lj.bbox_y_max
            return gap <= min_w

        groups = _run_union_find(labels, vert_pair)
        labels = _collect_groups(labels, groups,
                                 sort_key=lambda l: l.bbox_y_min)

        return labels

    def _merge_adjacent_labels(self, labels):
        """Merge adjacent labels at the same position with similar text.

        Labels don't flash on/off - they appear, stay, and disappear.
        If we have similar text at the same position with a small time gap,
        it's the same label with OCR variations or detection gaps.

        Args:
            labels: List of LabelResult objects.

        Returns:
            List of LabelResult with adjacent similar labels merged.
        """
        if len(labels) < 2:
            return labels

        MAX_MERGE_GAP = 1.0  # Max gap between labels to consider merging
        ADJACENT_SIMILARITY_THRESHOLD = 0.70  # Relaxed for OCR errors

        # Sort by position (Y then X), then by start time
        # This groups labels at the same position together
        pos_threshold_x = self.width * 0.05
        pos_threshold_y = self.height * 0.05

        def pos_key(lbl):
            # Quantize position to group nearby labels
            qx = int(lbl.pos_x / pos_threshold_x)
            qy = int(lbl.pos_y / pos_threshold_y)
            return (qy, qx, lbl.start_pts)

        labels = sorted(labels, key=pos_key)

        merged = []
        i = 0
        while i < len(labels):
            current = labels[i]
            # Create mutable copy for merging
            merged_label = LabelResult(
                start_pts=current.start_pts,
                end_pts=current.end_pts,
                text=current.text,
                pos_x=current.pos_x,
                pos_y=current.pos_y,
                bbox_x_min=current.bbox_x_min,
                bbox_y_min=current.bbox_y_min,
                bbox_x_max=current.bbox_x_max,
                bbox_y_max=current.bbox_y_max,
            )

            # Try to merge consecutive labels
            while i + 1 < len(labels):
                next_label = labels[i + 1]

                # Check position proximity first (within 5% of frame dimensions)
                dx = abs(merged_label.pos_x - next_label.pos_x)
                dy = abs(merged_label.pos_y - next_label.pos_y)
                if dx > pos_threshold_x or dy > pos_threshold_y:
                    break

                # Time gap (positive = gap, negative = overlap)
                time_gap = next_label.start_pts - merged_label.end_pts

                # Must be adjacent or overlapping, not too far apart
                if time_gap > MAX_MERGE_GAP:
                    break

                # Check text similarity with relaxed threshold
                if not self._texts_similar(merged_label.text, next_label.text,
                                           ADJACENT_SIMILARITY_THRESHOLD):
                    break

                # Merge: extend timing, keep the longer text (likely more complete)
                # If equal length, prefer the later label (middle of visibility
                # typically has better OCR than edges)
                merged_label.end_pts = max(merged_label.end_pts, next_label.end_pts)
                merged_label.start_pts = min(merged_label.start_pts, next_label.start_pts)
                if len(next_label.text) >= len(merged_label.text):
                    merged_label.text = next_label.text
                    merged_label.pos_x = next_label.pos_x
                    merged_label.pos_y = next_label.pos_y

                # Update union bbox
                if merged_label.bbox_x_min is not None and next_label.bbox_x_min is not None:
                    merged_label.bbox_x_min = min(merged_label.bbox_x_min, next_label.bbox_x_min)
                    merged_label.bbox_y_min = min(merged_label.bbox_y_min, next_label.bbox_y_min)
                    merged_label.bbox_x_max = max(merged_label.bbox_x_max, next_label.bbox_x_max)
                    merged_label.bbox_y_max = max(merged_label.bbox_y_max, next_label.bbox_y_max)
                elif next_label.bbox_x_min is not None:
                    merged_label.bbox_x_min = next_label.bbox_x_min
                    merged_label.bbox_y_min = next_label.bbox_y_min
                    merged_label.bbox_x_max = next_label.bbox_x_max
                    merged_label.bbox_y_max = next_label.bbox_y_max

                i += 1

            merged.append(merged_label)
            i += 1

        return merged

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def scan(self, det_engine, ocr, time_start, time_end, stream_start_time, progress=None, cancel_event=None):
        """Run the 4-phase label scanning pipeline.

        Args:
            det_engine: TextDetection engine for detection-only calls.
            ocr: Full PaddleOCR engine for recognition.
            time_start: Start time string (e.g., "2:30") or empty.
            time_end: End time string or empty.
            stream_start_time: Container-level start_time offset to subtract for ASS timestamps.
            progress: Optional ProgressTracker for unified progress reporting.
            cancel_event: Optional threading.Event for cooperative cancellation.

        Returns:
            List of LabelResult with normalized PTS values.
        """
        # Phase 1: Detection scan, keeping the crops phase 1.5 will OCR
        retained = _RetainedCrops(self.PHASE15_RETAIN_BUDGET_BYTES)
        text_frames = self._phase1_find_text_frames(det_engine, time_start, time_end, progress, cancel_event=cancel_event, retain=retained)
        if not text_frames or (cancel_event is not None and cancel_event.is_set()):
            return []

        # Phase 1.5: Batch OCR to annotate boxes with text
        text_frames = self._batch_ocr_text_frames(text_frames, ocr, progress, retained=retained, cancel_event=cancel_event)
        del retained
        if cancel_event is not None and cancel_event.is_set():
            return []

        # Phase 2: Position grouping (text-aware)
        groups = self._phase2_group_by_position(text_frames, progress)
        if not groups or (cancel_event is not None and cancel_event.is_set()):
            return []

        # Phase 3: Crop, clean, recognize
        segments = self._phase3_ocr_and_segment(groups, ocr, progress, cancel_event=cancel_event)
        if not segments or (cancel_event is not None and cancel_event.is_set()):
            return []

        # Phase 4: Timing refinement
        labels = self._phase4_find_timing(segments, det_engine, progress, cancel_event=cancel_event)

        # Post-processing: Remove duplicates
        labels = self._remove_duplicates(labels)

        # Post-processing: Merge split label fragments (same label split by OCR)
        labels = self._merge_split_labels(labels)

        # Post-processing: Merge adjacent labels at same position
        labels = self._merge_adjacent_labels(labels)

        # Adjust PTS by subtracting container start_time offset
        for label in labels:
            label.start_pts = max(0, label.start_pts - stream_start_time)
            label.end_pts = max(0, label.end_pts - stream_start_time)

        return labels
