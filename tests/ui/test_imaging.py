"""app/imaging.py: numpy -> QImage conversion and the review views' cache.

bgr_to_qimage is the ONE place a job's numpy frame becomes a QImage (the
jobs themselves are Qt-free), and it always copies: the QImage outlives the
array it was built from. FrameCache holds the frames and strips the views
draw, evicting by bytes.

A QImage needs a QApplication in this process, hence the `qapp` fixture.
"""
from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtGui import QImage

from app.imaging import FrameCache, bgr_to_qimage

RED_BGR = (0, 0, 255)
GREEN_BGR = (0, 255, 0)


def _bgr(rows: list[list[tuple[int, int, int]]]) -> np.ndarray:
    return np.array(rows, dtype=np.uint8)


# --------------------------------------------------------------------------
# bgr_to_qimage
# --------------------------------------------------------------------------

def test_bgr_becomes_rgb(qapp):
    image = bgr_to_qimage(_bgr([[RED_BGR, GREEN_BGR]]))

    assert image.format() == QImage.Format.Format_RGB888
    assert (image.width(), image.height()) == (2, 1)
    assert image.pixelColor(0, 0).getRgb()[:3] == (255, 0, 0)
    assert image.pixelColor(1, 0).getRgb()[:3] == (0, 255, 0)


def test_the_qimage_owns_its_pixels(qapp):
    array = _bgr([[RED_BGR, GREEN_BGR]])
    image = bgr_to_qimage(array)

    array[:] = 0
    del array

    assert image.pixelColor(0, 0).getRgb()[:3] == (255, 0, 0)
    assert image.pixelColor(1, 0).getRgb()[:3] == (0, 255, 0)


def test_a_grayscale_frame_becomes_a_grayscale_qimage(qapp):
    image = bgr_to_qimage(np.array([[0, 128, 255]], dtype=np.uint8))

    assert image.format() == QImage.Format.Format_Grayscale8
    assert (image.width(), image.height()) == (3, 1)
    assert [image.pixelColor(x, 0).getRgb()[0] for x in range(3)] == [0, 128, 255]


def test_a_non_contiguous_frame_converts_correctly(qapp):
    full = _bgr([[RED_BGR, GREEN_BGR],
                 [GREEN_BGR, GREEN_BGR],
                 [GREEN_BGR, RED_BGR]])
    view = full[::2]                                  # every other row: strided, not contiguous
    assert not view.flags["C_CONTIGUOUS"]

    image = bgr_to_qimage(view)

    assert (image.width(), image.height()) == (2, 2)
    assert image.pixelColor(0, 0).getRgb()[:3] == (255, 0, 0)
    assert image.pixelColor(1, 0).getRgb()[:3] == (0, 255, 0)
    assert image.pixelColor(1, 1).getRgb()[:3] == (255, 0, 0)


def test_a_reversed_column_view_converts_correctly(qapp):
    full = _bgr([[RED_BGR, GREEN_BGR]])

    image = bgr_to_qimage(full[:, ::-1])              # negative stride

    assert image.pixelColor(0, 0).getRgb()[:3] == (0, 255, 0)
    assert image.pixelColor(1, 0).getRgb()[:3] == (255, 0, 0)


@pytest.mark.parametrize("value", [
    None,
    np.zeros((0, 4, 3), dtype=np.uint8),
    np.zeros((2, 2, 3), dtype=np.float32),
    np.zeros((2, 2, 4), dtype=np.uint8),
    np.zeros((2, 2, 2, 3), dtype=np.uint8),
])
def test_what_cannot_be_converted_is_none(qapp, value):
    assert bgr_to_qimage(value) is None


# --------------------------------------------------------------------------
# FrameCache
# --------------------------------------------------------------------------

def _frame(value: int, pixels: int = 100) -> np.ndarray:
    return np.full((pixels,), value, dtype=np.uint8)      # nbytes == pixels


def test_the_cache_gives_back_what_was_put():
    cache = FrameCache()
    frame = _frame(1)

    cache.put(("a.mp4", "frame", 1.5), frame)

    assert cache.get(("a.mp4", "frame", 1.5)) is frame
    assert cache.get(("a.mp4", "frame", 2.0)) is None
    assert ("a.mp4", "frame", 1.5) in cache


def test_the_cache_evicts_the_least_recently_used_by_bytes():
    cache = FrameCache(max_bytes=250)
    for index in range(3):
        cache.put(("a.mp4", "frame", float(index)), _frame(index))

    assert cache.nbytes == 200
    assert cache.get(("a.mp4", "frame", 0.0)) is None          # the oldest went first
    assert cache.get(("a.mp4", "frame", 1.0)) is not None
    assert cache.get(("a.mp4", "frame", 2.0)) is not None


def test_reading_an_entry_makes_it_recent():
    cache = FrameCache(max_bytes=250)
    cache.put(("a.mp4", "frame", 0.0), _frame(0))
    cache.put(("a.mp4", "frame", 1.0), _frame(1))

    cache.get(("a.mp4", "frame", 0.0))                         # 1.0 is now the oldest
    cache.put(("a.mp4", "frame", 2.0), _frame(2))

    assert cache.get(("a.mp4", "frame", 0.0)) is not None
    assert cache.get(("a.mp4", "frame", 1.0)) is None


def test_putting_a_key_again_replaces_it_without_double_counting():
    cache = FrameCache(max_bytes=250)
    cache.put(("a.mp4", "frame", 0.0), _frame(0))
    fresh = _frame(9)

    cache.put(("a.mp4", "frame", 0.0), fresh)

    assert cache.nbytes == 100
    assert cache.get(("a.mp4", "frame", 0.0)) is fresh


def test_the_newest_entry_survives_even_when_it_alone_is_over_budget():
    cache = FrameCache(max_bytes=50)
    cache.put(("a.mp4", "frame", 0.0), _frame(0))
    big = _frame(1, pixels=400)

    cache.put(("a.mp4", "frame", 1.0), big)

    assert cache.get(("a.mp4", "frame", 0.0)) is None
    assert cache.get(("a.mp4", "frame", 1.0)) is big


def test_clear_file_drops_that_files_frames_and_strips_only():
    cache = FrameCache()
    box = (288, 786, 1344, 53)
    cache.put(("a.mp4", "frame", 1.0), _frame(1))
    cache.put(("a.mp4", "strip", box, 1.0), _frame(2))
    cache.put(("b.mp4", "frame", 1.0), _frame(3))

    cache.clear_file("a.mp4")

    assert cache.get(("a.mp4", "frame", 1.0)) is None
    assert cache.get(("a.mp4", "strip", box, 1.0)) is None
    assert cache.get(("b.mp4", "frame", 1.0)) is not None
    assert cache.nbytes == 100


def test_clear_file_can_drop_one_kind():
    cache = FrameCache()
    box = (288, 786, 1344, 53)
    cache.put(("a.mp4", "frame", 1.0), _frame(1))
    cache.put(("a.mp4", "strip", box, 1.0), _frame(2))

    cache.clear_file("a.mp4", "strip")

    assert cache.get(("a.mp4", "frame", 1.0)) is not None
    assert cache.get(("a.mp4", "strip", box, 1.0)) is None


def test_clearing_the_cache_empties_it():
    cache = FrameCache()
    cache.put(("a.mp4", "frame", 1.0), _frame(1))

    cache.clear()

    assert cache.nbytes == 0 and len(cache) == 0
    assert cache.get(("a.mp4", "frame", 1.0)) is None
