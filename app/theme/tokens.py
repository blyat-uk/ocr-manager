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
SPOTLIGHT = (8, 10, 14, 115)  # rgba(8,10,14,.45) crop-box spotlight dimming

# --------------------------------------------------------------------------
# Typography (ui-spec.md §2.2)
# --------------------------------------------------------------------------
# Font stack: the mockups use "-apple-system, ... system-ui, sans-serif" --
# macOS/web keywords Qt cannot resolve -- so ruling C9 ("falls back to the
# system sans with CJK coverage") replaces them with a real family list,
# ending in a CJK-capable face so Chinese subtitle text in the canvas/
# glyph tiles always has coverage.
FONT_STACK = ["Inter", "Segoe UI", "Noto Sans", "Noto Sans CJK SC", "sans-serif"]

FONT_SIZE_XS = 9            # .badge, rrow header labels, lane "speech" label
FONT_SIZE_SCOPE = 9.5       # .insp-scope, .sec-h, canvas-tag/.tag, .blk text
FONT_SIZE_SM = 10           # .fsub, footer hint text, .conf caption text, .note
FONT_SIZE_BTN_SM = 10.5     # .seg span, .btn.sm, samples label, activity-strip text
FONT_SIZE_BODY = 11.5       # .btn, .tab, .fname, .kv
FONT_SIZE_MD = 12.5         # .insp-file
FONT_SIZE_PROJ = 13.5       # .proj

FONT_WEIGHT_FNAME = 550     # .fname
FONT_WEIGHT_PROJ = 600      # .proj, .insp-file
FONT_WEIGHT_PRIMARY = 650   # .btn.primary

LETTER_SPACING_SCOPE_EM = 0.07   # .sec-h; .insp-scope uses .08em, see below
LETTER_SPACING_INSP_SCOPE_EM = 0.08  # .insp-scope

# --------------------------------------------------------------------------
# Layout sizes (px)
# --------------------------------------------------------------------------
RAIL_WIDTH = 246         # .rail
INSPECTOR_WIDTH = 322    # .insp

# --------------------------------------------------------------------------
# Radii (ui-spec.md §2.4, transcribed from the literal CSS this task's
# widgets use)
# --------------------------------------------------------------------------
RADIUS_TAG = 4     # .badge, .mini, .tag, .pbar
RADIUS_SEG = 5     # .seg span, .track, .ztile
RADIUS_BTN = 6     # .btn, .kv, .canvas
RADIUS_ROW = 7     # .frow
RADIUS_CHIP = 20   # .chip (pill)
RADIUS_DOT = 50    # .dot -- CSS 50%; widgets translate this to width/2 px

# --------------------------------------------------------------------------
# Fixed component sizes (px), from the literal CSS
# --------------------------------------------------------------------------
BAR_WIDTH = 74     # .bar (ConfBar's track)
BAR_HEIGHT = 3
MINI_WIDTH = 90    # .mini (MiniProgress's track)
MINI_HEIGHT = 4
DOT_SIZE = 7       # .dot
