"""Zoom tiles for the Brightness review tab.

choose_tiles() picks, from the strips brightness detection sampled
(core.detect.brightness.BrightnessResult.strips), the frames a threshold is
most likely to break on: the darkest scene behind text, the brightest one,
the thinnest strokes, a subtitle of two or more lines, and a text-free
frame whose clutter still trips the OCR pass's gate. Pure: it reads
StripSample fields only, so the review tab re-fetches each tile's strip with
ocr_view.grab_ocr_strips_at(video, crop_box, [time]). No Qt imports.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.detect.brightness import StripSample

TILE_KINDS = ("dark", "bright", "thin", "two_line", "leaking")


def choose_tiles(strips: list[StripSample], value: int) -> dict[str, float]:
    """kind -> sample time. dark = text strip with the lowest background_level (the darkest scene); bright =
    text strip with the highest background_level; thin = text strip with the lowest stroke_px; two_line = a
    text strip with lines >= 2 (the one with the lowest glyph_level among them); leaking = the strip whose
    masked gate fires at `value` with the highest background_level among EMPTY strips. A kind with no
    candidate is omitted. Ties break by earlier time. Deterministic.

    "dark" goes by the scene, not the glyphs: the strip with the lowest glyph
    level is often not a subtitle at all (on a 1080p reference file it was a
    HUD the detector boxed). StripSample carries no pixels, so the leaking
    choice uses each empty strip's gate_at_value, which detect_brightness
    computes at the result's value (BrightnessResult.value): pass that value.
    Text strips without a glyph level (too small to split) are never "thin";
    among two-line strips they rank after every strip that has one. Kinds come
    back in TILE_KINDS order.
    """
    text = [s for s in strips if s.is_text]
    ranked = {
        "dark": [((s.background_level,), s) for s in text],
        "bright": [((-s.background_level,), s) for s in text],
        "thin": [((s.stroke_px,), s) for s in text if s.stroke_px is not None],
        "two_line": [((s.glyph_level is None, s.glyph_level or 0), s) for s in text if s.lines >= 2],
        "leaking": [((-s.background_level,), s) for s in strips if not s.is_text and s.gate_at_value],
    }
    return {kind: min(ranked[kind], key=lambda c: (c[0], c[1].time))[1].time
            for kind in TILE_KINDS if ranked[kind]}
