"""The views' bridge to `core/detect`: the detector helpers they call, and
the pixels they measure with.

`tests/ui/test_main_window.py::test_views_import_no_core_modules` rejects any
`core.*` import in `app/views/*.py`, so a view that needs one of the Qt-free,
pure detector helpers reaches it through here. This is the whole of that
bridge -- the Crop tab's box aggregation and mask preview, the Brightness
tab's strip measurements and the Time ranges tab's speech-in-a-skip check --
so there is one place to look for what the views take from `core`, and one
place a detector rename has to reach.

Every wrapper calls through the module object (`_crop.` / `ocr_view.` /
`_audio.`) rather than a name bound at import time, so a test that
monkeypatches `core.detect.crop.aggregate_box` or `core.detect.ocr_view.mask`
still sees the call. Nothing here holds state beyond one strip's
measurements, decides anything, or imports Qt.

`StripPixels` holds one OCR-exact strip (as `ocr_view.grab_ocr_strips_at`
produced it) plus the detector's polygon boxes for it, and answers the
questions the Brightness tiles ask at a threshold `t`:

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

`app/state_text.py` stays what its name says -- pure badge and caption text.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.detect import audio_profile as _audio
from core.detect import crop as _crop
from core.detect import ocr_view
from core.detect.brightness import DEFAULT_BRIGHTNESS, MIN_GLYPH_REGION_PIXELS

__all__ = ["DEFAULT_BRIGHTNESS", "LOST_ALERT_PERCENT", "LOST_RISE_POINTS", "MAX_T", "MIN_T",
           "StripPixels",
           "aggregate_crop_box", "clip_boxes", "gate_fires", "mask", "mask_region",
           "normalise_boxes", "speech_in_skips"]

MIN_T = 100                # the thresholds the curve spans; a subtitle threshold
MAX_T = 255                # is never picked outside them (core/detect/brightness.py)
LOST_ALERT_PERCENT = 10    # at or above this, a tile's status turns bad
LOST_RISE_POINTS = 10      # a tile this far above its own baseline is losing strokes
# Too few pixels inside the boxes to split. The detector's own floor
# (core.detect.brightness.MIN_GLYPH_REGION_PIXELS), so a strip this view
# calls unmeasurable is one the detector could not measure either.
MIN_SPLIT_PIXELS = MIN_GLYPH_REGION_PIXELS


def mask(strip: np.ndarray, t: int) -> np.ndarray:
    """`core.detect.ocr_view.mask`, resolved at call time."""
    return ocr_view.mask(strip, int(t))


def gate_fires(masked: np.ndarray) -> bool:
    """`core.detect.ocr_view.gate_fires`, resolved at call time."""
    return ocr_view.gate_fires(masked)


def normalise_boxes(boxes) -> tuple[tuple[int, int, int, int], ...]:
    """`boxes` as a tuple of (x, y, w, h) int tuples.

    Evidence stores them as lists of ints (core/detect/brightness.py's
    `to_evidence`), and a list never equals the tuple it was cached as -- so
    everything that compares boxes (the StripPixels cache key) normalises
    through here first, on both sides."""
    return tuple(tuple(int(value) for value in box) for box in boxes or ())


# --------------------------------------------------------------------------
# The Crop tab
# --------------------------------------------------------------------------

def _rectangle(box) -> list[tuple[int, int]]:
    """(x, y, width, height) as the four corner points of its rectangle --
    the polygon shape `aggregate_box` reads."""
    x, y, width, height = (int(value) for value in box)
    return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]


def aggregate_crop_box(boxes_per_sample, frame_size, settings: dict | None = None,
                       sample_times=None) -> tuple[int, int, int, int] | None:
    """`core.detect.crop.aggregate_box` over one list of (x, y, w, h) boxes
    per sampled frame -- the Crop tab's "⤢ fit to all N samples".

    `settings` mirrors the folder's detector settings and must carry the
    `bottom_half_cutoff` the detection itself used (`evidence["crop"]
    ["cutoff_frac"]`, 0.0 after a full-frame retry) -- judging the samples
    with another band would keep or drop different text than the box being
    replaced. None when there is nothing to build a box from.
    """
    polygons = [[_rectangle(box) for box in boxes] for boxes in boxes_per_sample]
    box = _crop.aggregate_box(polygons, tuple(int(value) for value in frame_size), settings=settings,
                              sample_times=None if sample_times is None else [float(t) for t in sample_times])
    return None if box is None else tuple(int(value) for value in box)


def speech_in_skips(speech, skips, min_overlap_sec: float = 2.0) -> list[tuple[float, float]]:
    """`core.detect.audio_profile.speech_in_skips`, resolved at call time --
    the Time ranges tab's speech-in-a-skipped-span warnings.

    `skips` are the timeline's own (start_sec, end_sec) keep-complement
    spans, not the detected blocks: what matters is what the OCR run will
    really skip, which is whatever the keep ranges leave out."""
    return _audio.speech_in_skips(speech, skips, min_overlap_sec)


def mask_region(region, threshold: int):
    """The OCR pass's brightness filter over a BGR region -- the Crop tab's
    "masked" preview. `mask` by another name, because the Crop tab masks a
    region of a frame rather than a strip; one implementation, so the two
    previews can never drift apart."""
    return mask(region, threshold)


# --------------------------------------------------------------------------
# The Brightness tab
# --------------------------------------------------------------------------


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
        self.given_boxes = normalise_boxes(boxes)
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
        more of the glyph pixels are lost, or None when none is. `limit` is
        the caller's bar -- the Brightness tab passes the tile's own lost %
        at the detector's value plus LOST_RISE_POINTS (see its `_measure`).

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
