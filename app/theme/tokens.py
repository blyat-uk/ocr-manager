"""Design tokens for the new window's dark theme (ruling C9).

Every colour below is verbatim from `docs/superpowers/stage3-research/
ui-spec.md` §2.1 -- the `.hf` custom-property block (the "Colour tokens"
table) plus its "Additional literal (non-tokenized) colours used inline"
list, both of which the digest confirms are byte-for-byte identical between
`workbench-hifi.html` and `tabs-hifi.html`. No value here is invented; where
Task 1's widgets need a colour the mockups never rendered (Badge's "bad"
tone: no hi-fi figure shows a failed/error badge), the widget reuses one of
these tokens rather than adding a new hex -- see `app/widgets/base.py`.

Sizes (RAIL_WIDTH, INSPECTOR_WIDTH, FONT_SIZE_*, RADIUS_*) are transcribed
from the literal CSS of the two hi-fi files themselves (not just ui-spec's
prose tables), so each one matches an exact rule like `.badge { font-
size:9px; ... }`; the comment on each names the CSS class(es) it came from.

Single dark theme only: no light variant, no `prefers-color-scheme`
equivalent, no qt-material (ruling C9).
"""

import os

# --------------------------------------------------------------------------
# UI scale
# --------------------------------------------------------------------------
# The mockups were drawn for a browser at a comfortable reading distance;
# on a real desktop at 1440p their 9-11 px type is too small to read. Every
# size below is therefore the mockup's own value multiplied by UI_SCALE, so
# the whole window grows in proportion -- type, chrome, rails, radii and the
# painted geometry in the views alike. The `*_BASE` constants keep the
# untouched mockup values, which is what the tests compare against ui-spec.
#
# Override per run without editing code:
#     OCR_MANAGER_UI_SCALE=1.4 .venv/bin/python main.py
UI_SCALE_DEFAULT = 1.25
UI_SCALE_MIN = 1.0
UI_SCALE_MAX = 3.0


def _scale_from_env() -> float:
    """UI_SCALE_DEFAULT, or $OCR_MANAGER_UI_SCALE clamped to [1.0, 3.0].
    An unreadable value is ignored rather than raising at import time: a
    typo in a shell profile must not stop the app from starting."""
    raw = os.environ.get("OCR_MANAGER_UI_SCALE", "").strip()
    if not raw:
        return UI_SCALE_DEFAULT
    try:
        value = float(raw)
    except ValueError:
        return UI_SCALE_DEFAULT
    return min(UI_SCALE_MAX, max(UI_SCALE_MIN, value))


UI_SCALE = _scale_from_env()


def px(value: float) -> int:
    """A mockup pixel length at the current scale, rounded to a whole pixel
    (widths, heights, margins, radii -- anything Qt wants as an int)."""
    return int(round(value * UI_SCALE))


def pt(value: float) -> float:
    """A mockup font size at the current scale, keeping its fraction (Qt
    takes fractional point sizes, and 9.5 -> 11.875 reads better than 12)."""
    return value * UI_SCALE


# --------------------------------------------------------------------------
# Colour tokens (ui-spec.md §2.1, the `.hf { --bg:...; }` custom-property
# block)
# --------------------------------------------------------------------------
BG = "#111317"          # window/canvas background (darkest layer)
PANEL = "#171a20"        # top bar, rail, inspector, activity strip
PANEL2 = "#1d2129"       # recessed surfaces: kv rows, chips, active tab, on-segment
LINE = "#282d37"         # primary hairline borders
LINE2 = "#333a46"        # brighter borders on interactive chrome
TXT = "#e6e9ef"          # primary text
DIM = "#98a0ae"          # secondary text
DIM2 = "#69707d"         # tertiary text (paths, timestamps, captions, notes)
ACC = "#ffc247"          # accent/amber
ACC_DIM = "#6b5426"      # dimmed accent
OK = "#6fd48a"           # success/green
WARN = "#f0a742"         # warning/amber-orange
BAD = "#f4707d"          # error/red
BLUE = "#6aa9ff"         # info/blue

# --------------------------------------------------------------------------
# Additional literal (non-tokenized) colours (ui-spec.md §2.1's second list)
# --------------------------------------------------------------------------
PRIMARY_TEXT = "#20180a"      # .btn.primary text (dark text on amber fill)
ROW_SELECTED = "#20252f"      # .frow.sel background (selected queue row)
ROW_HOVER = "#1a1e26"         # .frow:not(.sel):hover background
BADGE_BG = "#232935"          # default .badge background; also .pbar track
BADGE_WARN_BG = "#3a2c14"     # .badge.w background
BADGE_GOOD_BG = "#17301f"     # .badge.g background
TRACK_BG = "#252b35"          # .bar / .mini track background
KV_WARN_BORDER = "#4a3a1c"    # warn-tinted kv row border
TAG_BAD_BORDER = "#5b2f34"    # bad-tinted tag / .ztile.bad border
TAG_BLUE_BORDER = "#2c4666"   # blue-tinted tag border (envelope legend chip)
WAVEFORM = "#5a6472"          # waveform stroke colour
THUMB_TOP = "#243044"         # .thumb/.sthumb gradient start: linear-gradient(160deg,#243044,#121820 70%)
THUMB_BOTTOM = "#121820"      # .thumb/.sthumb gradient end (at 70%)
THUMB_PENDING = "#161b22"     # a pending (still detecting) queue row's thumbnail background
SPOTLIGHT = (8, 10, 14, 115)  # rgba(8,10,14,.45) crop-box spotlight dimming
CANVAS_TOP = "#2b3d52"        # .canvas radial-gradient(120% 90% at 30% 25%, #2b3d52 0%, #16202c 55%, #0a0e14 100%)
CANVAS_MID = "#16202c"
CANVAS_BOTTOM = "#0a0e14"
CANVAS_MID_STOP = 0.55        # the gradient's middle stop
ENVELOPE_ALPHA = 0.9          # .envelope border: 1px dashed rgba(106,169,255,.9) (tabs-hifi's crop tab)
SAMPLE_BAR_ALPHA = 0.75       # .sthumb i: the white "a line was found here" bar, rgba(255,255,255,.75)
# `.hf .tag` background: rgba(10,12,16,.82). ui-spec §2.1's second list does not carry it (it lists the
# tag BORDER colours only), so it is transcribed from the literal CSS of both hi-fi files, as the radii
# below are.
TAG_BG = (10, 12, 16, 209)

# --------------------------------------------------------------------------
# Typography (ui-spec.md §2.2)
# --------------------------------------------------------------------------
# Font stack: the mockups use "-apple-system, ... system-ui, sans-serif" --
# macOS/web keywords Qt cannot resolve -- so ruling C9 ("falls back to the
# system sans with CJK coverage") replaces them with a real family list,
# ending in a CJK-capable face so Chinese subtitle text in the canvas/
# glyph tiles always has coverage.
FONT_STACK = ["Inter", "Segoe UI", "Noto Sans", "Noto Sans CJK SC", "sans-serif"]

FONT_SIZE_XS_BASE = 9
FONT_SIZE_XS = pt(FONT_SIZE_XS_BASE)            # .badge, rrow header labels, lane "speech" label
FONT_SIZE_SCOPE_BASE = 9.5
FONT_SIZE_SCOPE = pt(FONT_SIZE_SCOPE_BASE)       # .insp-scope, .sec-h, canvas-tag/.tag, .blk text
FONT_SIZE_SM_BASE = 10
FONT_SIZE_SM = pt(FONT_SIZE_SM_BASE)           # .fsub, footer hint text, .conf caption text, .note
FONT_SIZE_BTN_SM_BASE = 10.5
FONT_SIZE_BTN_SM = pt(FONT_SIZE_BTN_SM_BASE)     # .seg span, .btn.sm, samples label, activity-strip text
FONT_SIZE_BODY_BASE = 11.5
FONT_SIZE_BODY = pt(FONT_SIZE_BODY_BASE)       # .btn, .tab, .fname, .kv
FONT_SIZE_MD_BASE = 12.5
FONT_SIZE_MD = pt(FONT_SIZE_MD_BASE)         # .insp-file
FONT_SIZE_PROJ_BASE = 13.5
FONT_SIZE_PROJ = pt(FONT_SIZE_PROJ_BASE)       # .proj

FONT_WEIGHT_FNAME = 550     # .fname
FONT_WEIGHT_PROJ = 600      # .proj, .insp-file
FONT_WEIGHT_PRIMARY = 650   # .btn.primary

LETTER_SPACING_SCOPE_EM = 0.07   # .sec-h; .insp-scope uses .08em, see below
LETTER_SPACING_INSP_SCOPE_EM = 0.08  # .insp-scope

# --------------------------------------------------------------------------
# Layout sizes (px)
# --------------------------------------------------------------------------
RAIL_WIDTH_BASE = 246
RAIL_WIDTH = px(RAIL_WIDTH_BASE)         # .rail
INSPECTOR_WIDTH_BASE = 322
INSPECTOR_WIDTH = px(INSPECTOR_WIDTH_BASE)    # .insp
THUMB_WIDTH_BASE = 56
THUMB_WIDTH = px(THUMB_WIDTH_BASE)         # .thumb
THUMB_HEIGHT_BASE = 32
THUMB_HEIGHT = px(THUMB_HEIGHT_BASE)

# --------------------------------------------------------------------------
# Radii (ui-spec.md §2.4, transcribed from the literal CSS this task's
# widgets use)
# --------------------------------------------------------------------------
# ui-spec.md's own §2.4 summary table groups `.mini` under its "4px" row,
# but the literal CSS (`.hf .mini { ... border-radius:2px; ... }`,
# `.hf .bar { ... border-radius:2px; ... }`) says 2px for both -- read
# directly from workbench-hifi.html/tabs-hifi.html, which is what
# _BarTrack (app/widgets/base.py) actually paints.
RADIUS_THUMB_BOX_BASE = 1
RADIUS_THUMB_BOX = px(RADIUS_THUMB_BOX_BASE)   # .thumb i (the crop box drawn on a queue thumbnail)
RADIUS_XS_BASE = 2
RADIUS_XS = px(RADIUS_XS_BASE)      # .bar, .mini
RADIUS_THUMB_BASE = 3
RADIUS_THUMB = px(RADIUS_THUMB_BASE)   # .thumb, .sthumb
RADIUS_TAG_BASE = 4
RADIUS_TAG = px(RADIUS_TAG_BASE)     # .badge, .tag, .pbar
RADIUS_SEG_BASE = 5
RADIUS_SEG = px(RADIUS_SEG_BASE)     # .seg span, .track, .ztile
RADIUS_BTN_BASE = 6
RADIUS_BTN = px(RADIUS_BTN_BASE)     # .btn, .kv, .canvas
RADIUS_ROW_BASE = 7
RADIUS_ROW = px(RADIUS_ROW_BASE)     # .frow
RADIUS_CHIP_BASE = 20
RADIUS_CHIP = px(RADIUS_CHIP_BASE)   # .chip (pill)
# What the QSS uses for .chip: CSS clamps 20px to half the height (a pill), but Qt draws square
# corners when a radius exceeds half the widget, so the ~22 px chip gets half its height instead.
RADIUS_CHIP_QT_BASE = 10
RADIUS_CHIP_QT = px(RADIUS_CHIP_QT_BASE)
RADIUS_DOT = 50    # .dot -- CSS 50%; widgets translate this to width/2 px

# --------------------------------------------------------------------------
# Fixed component sizes (px), from the literal CSS
# --------------------------------------------------------------------------
BAR_WIDTH_BASE = 74
BAR_WIDTH = px(BAR_WIDTH_BASE)     # .bar (ConfBar's track)
BAR_HEIGHT_BASE = 3
BAR_HEIGHT = px(BAR_HEIGHT_BASE)
MINI_WIDTH_BASE = 90
MINI_WIDTH = px(MINI_WIDTH_BASE)    # .mini (MiniProgress's track)
MINI_HEIGHT_BASE = 4
MINI_HEIGHT = px(MINI_HEIGHT_BASE)
DOT_SIZE_BASE = 7
DOT_SIZE = px(DOT_SIZE_BASE)       # .dot
