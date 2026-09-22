"""numpy frames -> QImage, and the cache the review views draw from.

The jobs that decode frames are Qt-free (core/jobs/view_jobs.py,
core/jobs/detect_jobs.py): they hand back numpy arrays. This module is the
one place those become QImages, and the one place the window keeps them.

bgr_to_qimage always copies. A QImage built on a numpy buffer borrows it,
so the image would show garbage (or crash) the moment the array is freed --
and these arrays are owned by job results the controller drops.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from PyQt6.QtGui import QImage

# ~70 frames of 1280x720 BGR (2.8 MB each), on top of whatever PaddleOCR is
# holding. A crop view shows one canvas frame and about a dozen filmstrip
# frames, so this keeps the last five or so files hot -- enough to walk back
# and forth through a stretch of the queue without decoding again -- while
# browsing a hundred-file folder can no longer grow the process by half a
# gigabyte of frames nobody is looking at any more.
DEFAULT_MAX_BYTES = 192 * 1024**2


def bgr_to_qimage(image) -> QImage | None:
    """A QImage that owns a copy of a BGR (or grayscale) uint8 frame; None
    for no frame or one it cannot convert.

    Takes 3-channel BGR (as OpenCV, PyAV and every frame source here produce
    it) and 1-channel grayscale, contiguous or not.
    """
    if image is None:
        return None
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.size == 0:
        return None
    if array.ndim == 2:
        gray = np.ascontiguousarray(array)
        height, width = gray.shape
        return QImage(gray.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
    if array.ndim == 3 and array.shape[2] == 3:
        rgb = np.ascontiguousarray(array[:, :, ::-1])
        height, width = rgb.shape[:2]
        return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
    return None


class FrameCache:
    """The frames and strips the views draw, keyed by tuples that start with
    the file name: ("a.mp4", "frame", time) and ("a.mp4", "strip", crop_box,
    time). Least-recently-used entries are evicted once the total exceeds
    `max_bytes`; the entry just put is never the one evicted.

    Arrays are stored as given, not copied: they come straight from a job
    result nobody else holds. Callers must not mutate what they get back.

    A key can also be marked unavailable: the frame at that time was asked
    for and could not be read. `get` still answers None (a view draws its
    placeholder either way), but `knows` is True, so the caller does not ask
    for it again. Markers hold no pixels: they are outside the byte budget
    and never evict a frame. They are also not capped -- there is one per
    time that failed, and they live until `clear_file` or `clear` drops them
    (a file removed, its crop box changed, the folder closed), which in a
    session bounds them by the times the views actually asked for.

    Evicting an entry is not the same as marking it unavailable: an evicted
    frame is simply unknown again, and asking for it fetches it once more.

    Entries also carry their provenance, because not every reader can use
    every frame. Whole frames are kept on disk as WebP at quality 90
    (core/jobs/view_cache.py): a fine canvas to look at, and ~84 KB instead
    of 571 KB, but the round trip moves pixel levels by up to ~58 at the text
    edges that matter. The crop view's "masked" toggle previews the OCR
    brightness filter, which thresholds on those very levels, so it may only
    draw a frame that came from the decoder -- while the canvas, the
    filmstrip and everything else on screen are happy with the disk copy.
    `get(key, exact=True)` is how that one reader says so: a lossy entry
    answers None, exactly as if it were not held, and the caller fetches the
    time again from the decoder.

    Provenance is bookkeeping about entries, not a fact about times, so it
    lives and dies with the entry: eviction, `clear_file` and `clear` drop it,
    and a key that comes back starts from whatever that put says. That is the
    opposite of the unavailable markers, which outlive any entry on purpose.
    """

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES):
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._missing: set[tuple] = set()
        self._lossy: set[tuple] = set()
        self._bytes = 0

    def get(self, key, exact: bool = False) -> np.ndarray | None:
        """The cached array for `key`, which becomes the most recent entry.

        `exact=True` treats a frame that came from the lossy disk cache as a
        miss, so the caller re-fetches the decoder's own pixels: the crop
        view's "masked" preview filters on real pixel levels, and a WebP
        round trip at quality 90 moves them by up to ~58.

        That miss leaves the cache exactly as it was -- the entry stays, and
        it does not become the most recent one. The frame is still the right
        thing for every other reader, and an exact reader that could not use
        it read nothing, so it has no claim to be keeping anything alive.
        """
        image = self._items.get(key)
        if image is None:
            return None
        if exact and key in self._lossy:
            return None
        self._items.move_to_end(key)
        return image

    def put(self, key, image: np.ndarray, lossy: bool = False) -> None:
        """Hold `image` for `key`, as the most recent entry.

        `lossy=True` says these pixels came back out of the on-disk WebP
        cache rather than the decoder, so an exact reader must not be given
        them. Provenance only ever improves: an exact put over a lossy entry
        upgrades it (the re-decode the exact reader asked for has arrived),
        and a lossy put over an exact entry is ignored rather than allowed to
        downgrade it. An exact frame in hand is strictly better than the disk
        copy of the same frame, and re-marking it lossy would throw it away
        and send the crop view off to decode a time it already had.
        """
        if lossy and key in self._items and key not in self._lossy:
            self._items.move_to_end(key)        # the better pixels stay; only recency moves
            return
        self._drop(key)
        self._missing.discard(key)              # it could be read after all
        self._items[key] = image
        self._bytes += int(image.nbytes)
        if lossy:
            self._lossy.add(key)
        while self._bytes > self.max_bytes and len(self._items) > 1:
            self._drop(next(iter(self._items)))

    def mark_unavailable(self, key) -> None:
        """Remember that this frame was asked for and could not be read."""
        if key not in self._items:
            self._missing.add(key)

    def is_unavailable(self, key) -> bool:
        return key in self._missing

    def knows(self, key) -> bool:
        """True once this frame has been fetched, successfully or not: it is
        cached, or known to be unreadable. Either way, do not ask again."""
        return key in self._items or key in self._missing

    def clear_file(self, file: str, kind: str | None = None) -> None:
        """Forget `file`'s entries and markers: every kind, or only "frame" /
        "strip".

        Whole file: the entry is gone (a removed video, a closed folder), and
        a file of the same name that comes back starts from nothing.
        Strips only: the file's crop box changed, so strips grabbed with the
        old box can never be drawn again -- they are keyed by that box, so
        they would otherwise sit in the cache until they aged out.
        """
        def matches(key) -> bool:
            return key[0] == file and (kind is None or key[1] == kind)

        for key in [key for key in self._items if matches(key)]:
            self._drop(key)
        self._missing.difference_update([key for key in self._missing if matches(key)])

    def clear(self) -> None:
        self._items.clear()
        self._missing.clear()
        self._lossy.clear()
        self._bytes = 0

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __contains__(self, key) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def _drop(self, key) -> None:
        """Forget one entry, pixels and provenance together: the single way
        an entry leaves, so the lossy markers cannot outlive what they
        describe, whether the entry was evicted, cleared or replaced."""
        image = self._items.pop(key, None)
        self._lossy.discard(key)
        if image is not None:
            self._bytes -= int(image.nbytes)
