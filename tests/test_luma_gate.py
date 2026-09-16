"""Exact center-mask pre-gate: _center_mask_is_empty().

The brightness filter's mask is cv2.inRange(frame, (t,t,t), (255,255,255)):
a pixel survives only if every channel is >= t. inRange is pointwise, so it
commutes with slicing - running it on just the center square gives exactly
the same result there as running it on the whole frame and slicing the
center out afterward. So if the center square's own mask is empty, that
slice of the *real* mask is empty too, exactly, with no reconstruction and
no numerical margin - see videocr/video.py's module comment above
_center_mask_is_empty() for the full argument.

This suite checks that claim two ways: against an oracle computed by a
completely different code path (plain numpy min/compare, never calling
cv2.inRange or anything _center_mask_is_empty itself calls - a mutation
that broke the production function, e.g. a swapped channel or a flipped
comparison, would not also break this oracle, so it would actually be
caught), and against concrete boundary pixels. Trials are constructed, not
drawn uniformly at random and filtered after the fact: an unconditioned
uniform draw over 200 random thresholds mostly lands far from where real
brightness_threshold values (209, 230 - see tests/fixtures/media_manifest
.json and tools/bench.py's XWZ_4K_CASE) actually sit, so it barely
exercises the threshold this gate runs at in production. Conditioning the
strips on the outcome we want to check, at thresholds drawn from a
realistic band around the real ones, means every trial is doing work.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from videocr.video import _center_mask_is_empty

RNG_SEED = 20260916

# Real brightness_threshold values this gate actually runs at in
# production (see tests/fixtures/media_manifest.json's "slay" project and
# tools/bench.py's XWZ_4K_CASE), plus a realistic band around them - not
# the full 0-255 range, which a real crop's brightness threshold never
# spans.
PRODUCTION_THRESHOLDS = (209, 230)
REALISTIC_THRESHOLD_RANGE = (150, 245)


def _independent_oracle_rejects(strip: np.ndarray, threshold: int) -> bool:
    """True iff no pixel in `strip` has every channel >= threshold.

    Computed via plain numpy min/compare - a different code path from
    _center_mask_is_empty's cv2.inRange/cv2.countNonZero, so a bug in
    either (wrong channel order, off-by-one on the threshold, a flipped
    comparison) shows up as a mismatch instead of both sides agreeing by
    construction.
    """
    min_channel = strip.min(axis=-1)
    return bool(np.max(min_channel) < threshold)


def _guaranteed_reject_strip(rng, shape, threshold: int) -> np.ndarray:
    """Every channel of every pixel is < threshold, so the real mask over
    this strip is empty and the gate must reject it. (threshold=0 has no
    valid reject strip at all - min(B,G,R) >= 0 always - so this is only
    meaningful for threshold >= 1, which is all this suite ever calls it
    with.)"""
    hi = max(1, threshold)  # rng.integers' high is exclusive; hi=1 -> all zeros
    return rng.integers(0, hi, size=shape, dtype=np.uint8)


def _guaranteed_admit_strip(rng, shape, threshold: int) -> np.ndarray:
    """A guaranteed-reject strip with exactly one pixel planted at or
    above threshold in every channel, so the real mask is non-empty and
    the gate must admit it."""
    strip = _guaranteed_reject_strip(rng, shape, threshold)
    y = int(rng.integers(0, shape[0]))
    x = int(rng.integers(0, shape[1]))
    strip[y, x] = [int(rng.integers(threshold, 256)) for _ in range(shape[-1])]
    return strip


def _organic_random_strip(rng, shape) -> np.ndarray:
    """Fully unconditioned draw, for some unbiased coverage alongside the
    conditioned trials above."""
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_center_mask_matches_independent_oracle():
    """_center_mask_is_empty() agrees with a numpy-only oracle across many
    constructed trials at realistic thresholds, including the real
    production values.

    One test id, looping internally (not parametrize), so this is one
    collected test doing real work on every run - not hundreds of
    parametrize ids where most either duplicate each other or silently
    skip.
    """
    rng = np.random.default_rng(RNG_SEED)
    shape = (4, 16, 3)
    failures = []

    trials = []
    # Every production threshold, both outcomes, several strip shapes'
    # worth of random placement.
    for t in PRODUCTION_THRESHOLDS:
        for _ in range(20):
            trials.append((t, "reject"))
            trials.append((t, "admit"))
    # A realistic band around them, same split.
    for _ in range(300):
        t = int(rng.integers(*REALISTIC_THRESHOLD_RANGE))
        trials.append((t, "reject"))
        trials.append((t, "admit"))
    # Some unconditioned coverage too.
    for _ in range(100):
        t = int(rng.integers(0, 256))
        trials.append((t, "organic"))

    for t, mode in trials:
        if mode == "reject":
            strip = _guaranteed_reject_strip(rng, shape, t)
        elif mode == "admit":
            strip = _guaranteed_admit_strip(rng, shape, t)
        else:
            strip = _organic_random_strip(rng, shape)

        want = _independent_oracle_rejects(strip, t)
        got = _center_mask_is_empty(strip, t)
        if got != want:
            failures.append((t, mode, want, got))

    assert not failures, (
        f"{len(failures)}/{len(trials)} trials disagreed with the "
        f"independent oracle (threshold, mode, oracle_rejects, "
        f"gate_rejects): {failures[:10]}"
    )
    # Sanity: the construction actually exercised both outcomes near the
    # real thresholds, not just the "organic" fallback.
    reject_trials = [tr for tr in trials if tr[1] == "reject"]
    admit_trials = [tr for tr in trials if tr[1] == "admit"]
    assert len(reject_trials) >= 300
    assert len(admit_trials) >= 300
    assert any(t in PRODUCTION_THRESHOLDS for t, _ in trials)


def test_center_mask_matches_full_frame_slice():
    """cv2.inRange is pointwise, so it commutes with slicing: computing it
    on just the center square must equal computing it on the whole frame
    and slicing the center out afterward - the exact claim the module
    comment above _center_mask_is_empty() makes. Checked directly against
    cv2.inRange itself (not the oracle above), at the real production
    thresholds, over both a rejecting and an admitting frame.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    h, w = 20, 60
    center_x_start, center_x_end = 20, 40  # a 20x20 center square

    for threshold in PRODUCTION_THRESHOLDS:
        for mode in ("reject", "admit"):
            frame = (_guaranteed_reject_strip(rng, (h, w, 3), threshold)
                      if mode == "reject"
                      else _guaranteed_admit_strip(rng, (h, w, 3), threshold))
            # Force the planted bright pixel (if any) outside the center
            # square isn't guaranteed by _guaranteed_admit_strip, so also
            # plant one directly inside it for the "admit" case, ensuring
            # the interesting pixel is actually within the slice under
            # test.
            if mode == "admit":
                frame[5, 25] = [255, 255, 255]

            full_mask_center_slice = cv2.inRange(
                frame, (threshold,) * 3, (255,) * 3
            )[:, center_x_start:center_x_end]
            full_mask_empty = cv2.countNonZero(full_mask_center_slice) == 0

            center_bgr = frame[:, center_x_start:center_x_end]
            assert _center_mask_is_empty(center_bgr, threshold) == full_mask_empty


def test_boundary_exact_threshold_and_one_below():
    """The real boundary case: a pixel with every channel exactly equal to
    the threshold must be admitted (inRange's bounds are inclusive), and a
    pixel one below the threshold in every channel, as the only non-black
    pixel present, must be rejected. (255,255,255) is not a boundary case
    at a threshold of 209 - it's 46 above it - so this replaces that
    non-boundary check.
    """
    t = 209
    admitted = np.zeros((10, 10, 3), dtype=np.uint8)
    admitted[5, 5] = (t, t, t)
    assert not _center_mask_is_empty(admitted, t)

    rejected = np.zeros((10, 10, 3), dtype=np.uint8)
    rejected[5, 5] = (t - 1, t - 1, t - 1)
    assert _center_mask_is_empty(rejected, t)
