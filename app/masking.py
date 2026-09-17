"""The OCR pass's own pixels, per strip, for the review views.

`app/views/*` may not import `core` (tests/ui/test_main_window.py's
`test_views_import_no_core_modules`), so the Brightness tab reaches
`core.detect.ocr_view.mask` / `gate_fires` through this module instead --
the same two functions, called by name on the module so a test can spy on
them, never re-implemented. Nothing here changes a pixel the OCR pass would
see; it only measures and slices what `ocr_view` returns.

`StripPixels` holds one OCR-exact strip (as `ocr_view.grab_ocr_strips_at`
produced it) plus the detector's polygon boxes for it, and answers the
questions the tiles ask at a threshold `t`:

    masked(t)        the strip after the OCR pass's brightness filter
    lost(t)          the glyph pixels that filter throws away
    lost_percent(t)  how many of them, as a percentage
    gate(t)          whether the OCR pass would still OCR this frame

Everything that does not depend on `t` -- the min-channel, the glyph split
(Otsu inside the boxes) and the glyph mask -- is computed once per strip.
The `t`-dependent answers cache the last threshold only, because the
threshold moves under the user's finger and the previous value is never
wanted again: one `mask()` per strip per threshold change (~0.07 ms on a
1344x55 strip).

`StripSample.boxes` may extend past the strip (the detector boxes a polygon
on the unmasked strip and rounds outwards), so every box is clipped here
before it selects a pixel.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.detect import ocr_view
from core.detect.brightness import DEFAULT_BRIGHTNESS

__all__ = ["DEFAULT_BRIGHTNESS", "LOST_ALERT_PERCENT", "MAX_T", "MIN_T", "StripPixels",
           "clip_boxes", "gate_fires", "mask"]

MIN_T = 100                # the thresholds the curve spans; a subtitle threshold
MAX_T = 255                # is never picked outside them (core/detect/brightness.py)
LOST_ALERT_PERCENT = 10    # at or above this, a tile's status turns bad
MIN_SPLIT_PIXELS = 16      # too few pixels inside the boxes to split: no glyph mask


def mask(strip: np.ndarray, t: int) -> np.ndarray:
    """`core.detect.ocr_view.mask`, resolved at call time."""
    return ocr_view.mask(strip, int(t))


def gate_fires(masked: np.ndarray) -> bool:
    """`core.detect.ocr_view.gate_fires`, resolved at call time."""
    return ocr_view.gate_fires(masked)


def clip_boxes(boxes, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """`boxes` as (x, y, w, h), clipped to a width x height strip; boxes that
    keep no area are dropped."""
    clipped = []
    for box in boxes or ():
        x, y, box_width, box_height = (int(value) for value in box)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + box_width), min(height, y + box_height)
        if x1 > x0 and y1 > y0:
            clipped.append((x0, y0, x1 - x0, y1 - y0))
    return clipped


def _region(boxes, height: int, width: int) -> np.ndarray:
    inside = np.zeros((height, width), dtype=bool)
    for x, y, box_width, box_height in boxes:
        inside[y:y + box_height, x:x + box_width] = True
    return inside


class StripPixels:
    """One OCR-exact strip and what a threshold does to it."""

    def __init__(self, strip: np.ndarray, boxes=()):
        self.strip = strip
        self.height, self.width = strip.shape[:2]
        self.boxes = clip_boxes(boxes, self.width, self.height)
        self._min_channel = strip.min(axis=2)
        self._split: int | None = None
        self._glyph: np.ndarray | None = None
        self._glyph_count = 0
        self._measure()
        self._t: int | None = None
        self._masked: np.ndarray | None = None

    # --- threshold-independent -------------------------------------------------

    def _measure(self) -> None:
        """The glyph split (Otsu on the min-channel inside the boxes) and the
        glyph mask: inside the boxes, ABOVE the split.

        Above, not at: `core.detect.brightness._glyph_pixels` defines a glyph
        pixel as `min_channel > otsu`, which is also OpenCV's own
        THRESH_BINARY convention (`> thresh` becomes maxval). It matters on a
        strip with only two levels, where every threshold between them splits
        equally well and cv2 returns the lowest -- the background's own
        level. Taking pixels AT the split would then call the background a
        glyph.

        `_glyph_pixels` splits inside ERODED polygons; evidence keeps the
        polygons' bounding boxes only, so the boxes are used as they are --
        the split moves by a level or two against the detector's, which
        changes no decision here (the tiles compare the same pixels before
        and after one mask)."""
        if not self.boxes:
            return
        inside = _region(self.boxes, self.height, self.width)
        values = self._min_channel[inside]
        if values.size < MIN_SPLIT_PIXELS or int(values.max()) == int(values.min()):
            return
        split, _binary = cv2.threshold(values.reshape(-1, 1), 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        glyph = inside & (self._min_channel > int(split))
        if not glyph.any():
            return
        self._split, self._glyph, self._glyph_count = int(split), glyph, int(glyph.sum())

    @property
    def glyph_split(self) -> int | None:
        """Otsu inside the boxes, on the min-channel; None when the strip
        offers too little to split (no boxes, too few pixels, one level)."""
        return self._split

    @property
    def glyph_count(self) -> int:
        return self._glyph_count

    def has_glyphs(self) -> bool:
        return self._glyph is not None

    # --- threshold-dependent ---------------------------------------------------

    def masked(self, t: int) -> np.ndarray:
        """The strip after the OCR pass's brightness filter at `t`."""
        t = int(t)
        if self._t != t or self._masked is None:
            self._masked = mask(self.strip, t)
            self._t = t
        return self._masked

    def lost(self, t: int) -> np.ndarray | None:
        """The glyph pixels `t` throws away: above the glyph split in the
        raw strip, inside the boxes, and zero after masking. Masking
        zeroes a pixel exactly when its min-channel is below `t`, so that
        last test is `min_channel < t` -- the same pixels, without a second
        pass over the masked strip. None without a glyph mask."""
        if self._glyph is None:
            return None
        return self._glyph & (self._min_channel < int(t))

    def lost_percent(self, t: int) -> float | None:
        """What share of the glyph pixels `t` throws away, 0-100. None
        without a glyph mask."""
        lost = self.lost(t)
        if lost is None or self._glyph_count == 0:
            return None
        return 100.0 * int(lost.sum()) / self._glyph_count

    def gate(self, t: int) -> bool:
        """Whether the OCR pass's text gate still fires on this strip at `t`
        -- whether it would OCR the frame."""
        return gate_fires(self.masked(t))

    def first_losing_threshold(self, start: int, limit: float = LOST_ALERT_PERCENT,
                               stop: int = MAX_T) -> int | None:
        """The lowest whole threshold in [start, stop] at which `limit` % or
        more of the glyph pixels are lost, or None when none is.

        Binary search: raising the threshold can only drop more pixels, so
        `lost_percent` never decreases in `t`."""
        if self._glyph is None:
            return None
        low, high = max(MIN_T, int(start)), int(stop)
        if low > high or (self.lost_percent(high) or 0.0) < limit:
            return None
        while low < high:
            middle = (low + high) // 2
            if (self.lost_percent(middle) or 0.0) >= limit:
                high = middle
            else:
                low = middle + 1
        return low
