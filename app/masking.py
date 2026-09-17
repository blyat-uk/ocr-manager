"""Thin wrappers over core/detect for views (views may not import `core`).

`tests/ui/test_main_window.py::test_views_import_no_core_modules` rejects any
`core.*` import in `app/views/*.py`, so a view that needs one of the Qt-free,
pure detector helpers reaches it through here. Nothing in this module holds
state, decides anything or imports Qt: each function is the detector call the
view would otherwise make, with the argument shuffling the view would
otherwise do.

Every wrapper calls through the module object (`_crop.` / `_ocr_view.`)
rather than a name bound at import time, so a test that monkeypatches
`core.detect.crop.aggregate_box` or `core.detect.ocr_view.mask` still sees
the call.

`app/state_text.py` stays what its name says -- pure badge and caption text.
"""
from __future__ import annotations

from core.detect import crop as _crop
from core.detect import ocr_view as _ocr_view


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


def mask_region(region, threshold: int):
    """`core.detect.ocr_view.mask`: the OCR pass's brightness filter over a
    BGR region -- keep the pixels whose every channel is >= `threshold`."""
    return _ocr_view.mask(region, int(threshold))
