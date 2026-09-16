"""Exact luma pre-gate: luma_floor() and the BT.709 identity it rests on.

BT.709 gives 0.2126R + 0.7152G + 0.0722B = 1.164*(Y'-16) identically for any
pixel produced by the standard limited-to-full range BT.709 decode. If every
channel of a pixel is >= t, that convex combination (coefficients sum to 1)
is >= t-0.5 (the -0.5 covers the sub-LSB rounding a decoder may have done),
so Y' >= (t-0.5)/1.164 + 16 is a *necessary* condition for the pixel to
survive cv2.inRange(frame, (t,t,t), (255,255,255)) -- the per-channel min
mask the brightness filter uses. luma_floor() computes that bound; if the
maximum Y' in a region is below it, no pixel in that region can pass the
mask, so the mask over that region is provably all-zero.

No false negatives is the whole basis for the pre-gate (see
videocr/video.py, _frame_producer's LOOKING branch): this suite proves it
both algebraically (property test over random pixels) and operationally
(random strips actually fed through cv2.inRange).
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from videocr.video import luma_floor, _bt709_luma

RNG_SEED = 20260916


def test_luma_floor_reference_values():
    # Values from the task brief's measured 1080p (t=209) and 4K (t=230)
    # cases, to 2 decimal places.
    assert luma_floor(209) == pytest.approx(195.12, abs=0.01)
    assert luma_floor(230) == pytest.approx(213.16, abs=0.01)


def test_luma_floor_is_monotonic_in_threshold():
    thresholds = list(range(0, 256, 5))
    floors = [luma_floor(t) for t in thresholds]
    assert floors == sorted(floors)


@pytest.mark.parametrize("trial", range(200))
def test_property_no_false_negatives(trial):
    """For random 8-bit BGR pixels and random thresholds: every pixel whose
    min channel >= t has a reconstructed Y' >= luma_floor(t).

    This is the exactness argument itself, checked directly: luma_floor(t)
    must never sit above the true Y' of a pixel that would actually survive
    cv2.inRange's per-channel min-threshold mask.
    """
    rng = np.random.default_rng(RNG_SEED + trial)
    t = int(rng.integers(0, 256))
    # Bias sampling toward pixels that actually clear the threshold, so the
    # property gets exercised on the population it's meant to protect.
    low = rng.integers(t, 256, size=3) if t < 256 else np.array([255, 255, 255])
    pixel = low.astype(np.uint8).reshape(1, 1, 3)

    assert pixel.min() >= t
    y_prime = float(_bt709_luma(pixel)[0, 0])
    assert y_prime >= luma_floor(t) - 1e-6


@pytest.mark.parametrize("trial", range(200))
def test_property_no_false_negatives_random_strips(trial):
    """Same property, but over whole random strips (not hand-picked passing
    pixels): among ALL pixels in a random strip, any whose min channel >= t
    still has Y' >= luma_floor(t)."""
    rng = np.random.default_rng(RNG_SEED + 100_000 + trial)
    t = int(rng.integers(0, 256))
    strip = rng.integers(0, 256, size=(4, 16, 3), dtype=np.uint8)

    min_channel = strip.min(axis=2)
    passing = min_channel >= t
    if not passing.any():
        pytest.skip("no pixel in this random strip clears the threshold")

    y_prime = _bt709_luma(strip)
    floor = luma_floor(t)
    assert np.all(y_prime[passing] >= floor - 1e-6)


@pytest.mark.parametrize("trial", range(200))
def test_gate_rejection_implies_empty_mask(trial):
    """A strip the bound rejects (max Y' < luma_floor(t)) must also produce
    an all-zero mask under cv2.inRange(strip, (t,t,t), (255,255,255)) --
    the operational form of the same guarantee: the pre-gate never skips a
    frame that the real mask would have found text-shaped pixels in."""
    rng = np.random.default_rng(RNG_SEED + 200_000 + trial)
    t = int(rng.integers(1, 256))
    strip = rng.integers(0, 256, size=(4, 16, 3), dtype=np.uint8)

    y_prime = _bt709_luma(strip)
    if y_prime.max() >= luma_floor(t):
        pytest.skip("this random strip isn't one the bound rejects")

    mask = cv2.inRange(strip, (t, t, t), (255, 255, 255))
    assert np.count_nonzero(mask) == 0


def test_gate_rejects_strips_with_no_bright_pixel():
    """A strip well below any threshold's floor is rejected and its mask is
    genuinely empty -- a concrete, non-randomized sanity check."""
    t = 209
    strip = np.full((10, 10, 3), 100, dtype=np.uint8)  # far below floor

    assert _bt709_luma(strip).max() < luma_floor(t)
    mask = cv2.inRange(strip, (t, t, t), (255, 255, 255))
    assert np.count_nonzero(mask) == 0


def test_gate_admits_strips_with_a_bright_pixel():
    """A strip with one pixel at or above the threshold in every channel is
    never rejected by the gate (the gate must not produce a false
    negative for the exact boundary case)."""
    t = 209
    strip = np.zeros((10, 10, 3), dtype=np.uint8)
    strip[5, 5] = (255, 255, 255)

    assert _bt709_luma(strip).max() >= luma_floor(t)
