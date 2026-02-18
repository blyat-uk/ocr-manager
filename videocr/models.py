from __future__ import annotations
from typing import List
from dataclasses import dataclass
from thefuzz import fuzz

from . import utils

@dataclass
class PredictedText:
    __slots__ = 'bounding_box', 'confidence', 'text'
    bounding_box: list
    confidence: float
    text: str


class PredictedFrames:
    start_index: int  # 0-based index of the frame
    end_index: int
    words: List[PredictedText]
    confidence: float  # total confidence of all words
    text: str
    # PTS-based timestamps (in seconds) - canonical timing source
    pts_start: float  # Presentation timestamp of first frame (seconds)
    pts_end: float    # Presentation timestamp of last frame (seconds)

    def __init__(self, index: int, pred_data: list[list], conf_threshold: float, lang: str = 'ch', pts: float = None):
        # Fix for PaddleOCR versions greater than 2.7.0.2
        if not pred_data or pred_data[0] is None:
            pred_data = [[]]

        # Fix for PaddleOCR versions greater or equal than 3.0.3
        if utils.needs_conversion():
            pred_data = utils.convert_pred_data_to_old_format(pred_data)

        self.start_index = index
        self.end_index = index
        self.pts_start = pts  # Can be None if PTS not available
        self.pts_end = pts    # Will be updated when end_index changes
        self.lines = []
        self.lang = lang

        # Collect all words, merging them first before applying confidence threshold
        # This ensures "你竟掌握了" (93%) + "鲲鹏道法" (97%) becomes one line with avg 95%
        # rather than filtering "你竟掌握了" and keeping only "鲲鹏道法"
        MIN_WORD_CONFIDENCE = 0.5  # Filter obvious garbage (like "2" at 18%)

        words = []
        total_conf = 0
        word_count = 0

        for l in pred_data[0]:
            if len(l) < 2:
                continue
            bounding_box = l[0]
            text = l[1][0]
            conf = l[1][1]

            # Remove spaces from Chinese text (OCR artifacts)
            if lang == 'ch':
                text = text.replace(' ', '')

            # Only filter obvious garbage, not legitimate low-confidence words
            if conf >= MIN_WORD_CONFIDENCE:
                total_conf += conf
                word_count += 1
                words.append(PredictedText(bounding_box, conf, text))

        # Detect if PaddleOCR returned rotated coordinates (common for landscape images)
        # When rotated, bbox "height" (Y range) is larger than "width" (X range)
        coords_rotated = self._detect_rotated_coords(words)

        # Group words into lines based on overlap (use correct axis based on rotation)
        self.lines = self._group_words_into_lines(words, use_x_axis=coords_rotated)

        # Sort lines and words within lines
        if coords_rotated:
            # Rotated: Y represents horizontal position (but inverted - higher Y = left)
            # Sort words by descending Y to get left-to-right order
            for line in self.lines:
                line.sort(key=lambda word: self._get_word_center_y(word), reverse=True)
            self.lines.sort(key=lambda line: self._get_line_center_x(line))
        else:
            # Normal: sort words by X (horizontal position), lines by Y (vertical position)
            for line in self.lines:
                line.sort(key=lambda word: word.bounding_box[0][0])
            self.lines.sort(key=lambda line: self._get_line_center_y(line))

        # Calculate confidence
        if self.lines:
            self.confidence = total_conf / word_count
        elif len(pred_data[0]) == 0:
            self.confidence = 100
        else:
            self.confidence = 0

        # Join words: no spaces for Chinese, spaces for other languages
        word_separator = '' if self.lang == 'ch' else ' '
        self.text = '\n'.join(word_separator.join(word.text for word in line) for line in self.lines)

    def _detect_rotated_coords(self, words: List[PredictedText]) -> bool:
        """Detect if PaddleOCR returned rotated coordinates.

        PaddleOCR sometimes rotates landscape images internally, causing
        bounding box coordinates to be in rotated space where:
        - Y coordinates represent horizontal position
        - X coordinates represent vertical position

        Detection: if most words have bbox height > width, coords are likely rotated.
        """
        if not words:
            return False

        rotated_count = 0
        for word in words:
            width = self._get_word_width(word)
            height = self._get_word_height(word)
            if height > width * 1.5:  # Significant rotation indicator
                rotated_count += 1

        # If majority of words appear rotated, assume rotated coordinates
        return rotated_count > len(words) / 2

    def _get_word_center_x(self, word: PredictedText) -> float:
        """Get the horizontal center of a word's bounding box."""
        xs = [point[0] for point in word.bounding_box]
        return (min(xs) + max(xs)) / 2

    def _get_word_width(self, word: PredictedText) -> float:
        """Get the width of a word's bounding box."""
        xs = [point[0] for point in word.bounding_box]
        return max(xs) - min(xs)

    def _get_word_center_y(self, word: PredictedText) -> float:
        """Get the vertical center of a word's bounding box."""
        ys = [point[1] for point in word.bounding_box]
        return (min(ys) + max(ys)) / 2

    def _get_word_height(self, word: PredictedText) -> float:
        """Get the height of a word's bounding box."""
        ys = [point[1] for point in word.bounding_box]
        return max(ys) - min(ys)

    def _get_line_center_x(self, line: List[PredictedText]) -> float:
        """Get the average center X of all words in a line."""
        if not line:
            return 0
        return sum(self._get_word_center_x(w) for w in line) / len(line)

    def _get_line_center_y(self, line: List[PredictedText]) -> float:
        """Get the average center Y of all words in a line."""
        if not line:
            return 0
        return sum(self._get_word_center_y(w) for w in line) / len(line)

    def _words_on_same_line(self, word1: PredictedText, word2: PredictedText, use_x_axis: bool = False) -> bool:
        """Check if two words are on the same line.

        Args:
            use_x_axis: If True, use X coordinates (for rotated images).
                        If False, use Y coordinates (normal case).
        """
        if use_x_axis:
            center1 = self._get_word_center_x(word1)
            center2 = self._get_word_center_x(word2)
            size1 = self._get_word_width(word1)
            size2 = self._get_word_width(word2)
        else:
            center1 = self._get_word_center_y(word1)
            center2 = self._get_word_center_y(word2)
            size1 = self._get_word_height(word1)
            size2 = self._get_word_height(word2)

        # Words are on the same line if their centers are within half the average size
        avg_size = (size1 + size2) / 2
        return abs(center1 - center2) <= avg_size * 0.5

    def _group_words_into_lines(self, words: List[PredictedText], use_x_axis: bool = False) -> List[List[PredictedText]]:
        """Group words into lines based on overlap using greedy clustering.

        Args:
            use_x_axis: If True, group by X-axis overlap (for rotated coords).
                        If False, group by Y-axis overlap (normal case).
        """
        if not words:
            return []

        lines = []

        for word in words:
            merged = False
            for line in lines:
                # Check if word overlaps with any word in this line
                if any(self._words_on_same_line(word, existing, use_x_axis) for existing in line):
                    line.append(word)
                    merged = True
                    break

            if not merged:
                lines.append([word])

        return lines

    def is_similar_to(self, other: PredictedFrames, threshold=70) -> bool:
        return fuzz.ratio(self.text, other.text) >= threshold


class PredictedSubtitle:
    frames: List[PredictedFrames]
    sim_threshold: int
    text: str

    def __init__(self, frames: List[PredictedFrames], sim_threshold: int):
        self.frames = [f for f in frames if f.confidence > 0]
        self.frames.sort(key=lambda frame: frame.start_index)
        self.sim_threshold = sim_threshold

        if self.frames:
            self.text = max(self.frames, key=lambda f: f.confidence).text
        else:
            self.text = ''

    @property
    def index_start(self) -> int:
        if self.frames:
            return self.frames[0].start_index
        return 0

    @property
    def index_end(self) -> int:
        if self.frames:
            return self.frames[-1].end_index
        return 0

    @property
    def pts_start(self) -> float:
        """Get the PTS-based start time in seconds (canonical timing)."""
        if self.frames and self.frames[0].pts_start is not None:
            return self.frames[0].pts_start
        return None

    @property
    def pts_end(self) -> float:
        """Get the PTS-based end time in seconds (canonical timing)."""
        if self.frames and self.frames[-1].pts_end is not None:
            return self.frames[-1].pts_end
        return None

    def is_similar_to(self, other: PredictedSubtitle) -> bool:
        text1 = self.text.replace(' ', '')
        text2 = other.text.replace(' ', '')

        # Primary: fuzzy ratio comparison
        if fuzz.ratio(text1, text2) >= self.sim_threshold:
            return True

        # Secondary: handle OCR returning same characters in different order
        # This happens when OCR detects words with inconsistent bounding box positions
        if len(text1) > 0 and len(text2) > 0:
            # Only apply when texts are similar length (within 10%)
            len_diff = abs(len(text1) - len(text2)) / max(len(text1), len(text2))
            if len_diff <= 0.1 and sorted(text1) == sorted(text2):
                return True

        return False

    def __repr__(self):
        return '{} - {}. {}'.format(self.index_start, self.index_end, self.text)
