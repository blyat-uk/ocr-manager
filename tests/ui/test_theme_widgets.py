"""Task 1: theme tokens, the generated stylesheet, and the base widgets.

Tokens are checked against ui-spec.md §2.1's hex values verbatim (the
digest confirms the two hi-fi mockups define byte-for-byte identical
tokens). Widgets are constructed offscreen (QT_QPA_PLATFORM=offscreen) and
checked for the dynamic properties app/theme/qss.py selects on -- pytest-qt
is not installed, so click behaviour uses PyQt6.QtTest.QTest directly (see
tests/ui/conftest.py).
"""
import contextlib
import importlib
import os
import re
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QLabel, QPushButton

import app
from app.theme import qss, tokens
from app.widgets import base
from app.widgets.base import (
    Badge,
    Button,
    Chip,
    ConfBar,
    Dot,
    KvRow,
    MiniProgress,
    SectionHeader,
    SegmentedControl,
)

# --------------------------------------------------------------------------
# Tokens (ui-spec.md §2.1: colour tokens + "Additional literal (non-
# tokenized) colours used inline")
# --------------------------------------------------------------------------

COLOUR_TOKENS = [
    # ui-spec.md §2.1's `.hf { --bg:...; }` custom-property block.
    ("BG", "#111317"),
    ("PANEL", "#171a20"),
    ("PANEL2", "#1d2129"),
    ("LINE", "#282d37"),
    ("LINE2", "#333a46"),
    ("TXT", "#e6e9ef"),
    ("DIM", "#98a0ae"),
    ("DIM2", "#69707d"),
    ("ACC", "#ffc247"),
    ("ACC_DIM", "#6b5426"),
    ("OK", "#6fd48a"),
    ("WARN", "#f0a742"),
    ("BAD", "#f4707d"),
    ("BLUE", "#6aa9ff"),
    # ui-spec.md §2.1's "Additional literal (non-tokenized) colours" list.
    ("PRIMARY_TEXT", "#20180a"),
    ("ROW_SELECTED", "#20252f"),
    ("ROW_HOVER", "#1a1e26"),
    ("BADGE_BG", "#232935"),
    ("BADGE_WARN_BG", "#3a2c14"),
    ("BADGE_GOOD_BG", "#17301f"),
    ("TRACK_BG", "#252b35"),
    ("KV_WARN_BORDER", "#4a3a1c"),
    ("TAG_BAD_BORDER", "#5b2f34"),
    ("TAG_BLUE_BORDER", "#2c4666"),
    ("WAVEFORM", "#5a6472"),
    ("THUMB_TOP", "#243044"),
    ("THUMB_BOTTOM", "#121820"),
    ("THUMB_PENDING", "#161b22"),
]


@pytest.mark.parametrize("name, expected_hex", COLOUR_TOKENS)
def test_colour_token_matches_ui_spec(name, expected_hex):
    assert getattr(tokens, name) == expected_hex


def test_spotlight_is_rgba_45_percent_of_255():
    # rgba(8,10,14,.45) -- ui-spec.md §2.1.
    assert tokens.SPOTLIGHT == (8, 10, 14, 115)


def test_font_stack_is_cjk_capable_fallback_chain():
    # Ruling C9: "The font stack falls back to the system sans with CJK
    # coverage" -- the exact list from the Task 1 brief, not ui-spec's
    # literal (mac/web-only) "-apple-system, system-ui" stack.
    assert tokens.FONT_STACK == ["Inter", "Segoe UI", "Noto Sans", "Noto Sans CJK SC", "sans-serif"]


# ui-spec.md's own numbers, which are the mockup's -- so they are pinned
# against the unscaled `*_BASE` constants. `UI_SCALE` (and with it every
# scaled token) is a property of the running window, not of the spec:
# test_every_base_token_has_a_scaled_partner below is what ties the two
# together.
SIZE_TOKENS_FROM_UI_SPEC = [
    ("RAIL_WIDTH_BASE", 246),          # §2.3 ".rail -- width:246px"
    ("INSPECTOR_WIDTH_BASE", 322),     # §2.3 ".insp -- width:322px"
    ("THUMB_WIDTH_BASE", 56),          # §5 ".thumb (56x32px)"
    ("THUMB_HEIGHT_BASE", 32),
    ("BAR_WIDTH_BASE", 74),            # §5 "a 74x3px .bar"
    ("BAR_HEIGHT_BASE", 3),
    ("MINI_WIDTH_BASE", 90),           # §5 "a .mini progress bar (90x4px)"
    ("MINI_HEIGHT_BASE", 4),
    ("RADIUS_THUMB_BOX_BASE", 1),      # §2.4 radii table
    ("RADIUS_XS_BASE", 2),
    ("RADIUS_THUMB_BASE", 3),
    ("RADIUS_TAG_BASE", 4),
    ("RADIUS_SEG_BASE", 5),
    ("RADIUS_BTN_BASE", 6),
    ("RADIUS_ROW_BASE", 7),
    ("RADIUS_CHIP_BASE", 20),
    # §2.2's type scale.
    ("FONT_SIZE_XS_BASE", 9),
    ("FONT_SIZE_SCOPE_BASE", 9.5),
    ("FONT_SIZE_SM_BASE", 10),
    ("FONT_SIZE_BTN_SM_BASE", 10.5),
    ("FONT_SIZE_BODY_BASE", 11.5),
    ("FONT_SIZE_MD_BASE", 12.5),
    ("FONT_SIZE_PROJ_BASE", 13.5),
    # Not in ui-spec's prose: read from the mockups' literal CSS, as
    # tokens.py records. Pinned here so a later edit cannot drift them.
    ("DOT_SIZE_BASE", 7),
    ("RADIUS_CHIP_QT_BASE", 10),
]


@pytest.mark.parametrize("name, expected", SIZE_TOKENS_FROM_UI_SPEC)
def test_size_token_base_matches_ui_spec(name, expected):
    assert getattr(tokens, name) == expected


def test_every_base_token_has_a_scaled_partner():
    """The other half of the pin above: each `*_BASE` is the mockup's value
    and its partner is that value at `UI_SCALE`, so the window can be made
    readable without any number here drifting from the spec. A size token
    added later without going through `px()`/`pt()` fails here."""
    bases = sorted(name for name in vars(tokens) if name.endswith("_BASE"))
    assert len(bases) >= len(SIZE_TOKENS_FROM_UI_SPEC)
    for name in bases:
        scaled_name = name[: -len("_BASE")]
        base_value = getattr(tokens, name)
        scale = tokens.pt if scaled_name.startswith("FONT_SIZE_") else tokens.px
        assert getattr(tokens, scaled_name) == scale(base_value), scaled_name


def test_rail_and_inspector_widths():
    assert tokens.RAIL_WIDTH == tokens.px(246)
    assert tokens.INSPECTOR_WIDTH == tokens.px(322)


def test_app_package_resolves_to_real_source_package():
    # This test tree lives at tests/ui (not tests/app) specifically so that
    # pytest's import machinery never has to choose between this test
    # package and the real top-level `app/` package under the same dotted
    # name -- `app.__file__` must point at the repo's app/__init__.py, not
    # anything under tests/, and `python -m app` (plan 3B's smoke test)
    # must not resolve through this test tree either.
    repo_root = Path(__file__).resolve().parents[2]
    assert Path(app.__file__).resolve() == repo_root / "app" / "__init__.py"
    assert "tests" not in Path(app.__file__).resolve().parts


# --------------------------------------------------------------------------
# Stylesheet
# --------------------------------------------------------------------------

def test_stylesheet_sets_primary_button_variant_selector():
    assert 'QPushButton[variant="primary"]' in qss.build_stylesheet()


TOKENS_USED_IN_STYLESHEET = [
    # PANEL is not among these: no Task 1 base widget uses it directly (it's
    # the top bar/rail/inspector/activity-strip background, all later-task
    # view containers) -- apply_theme()'s QPalette still sets it (the Base
    # role), see test_apply_theme_sets_palette_and_font below.
    "BG", "PANEL2", "LINE", "LINE2", "TXT", "DIM", "DIM2", "ACC",
    "OK", "WARN", "BAD", "PRIMARY_TEXT", "ROW_HOVER", "BADGE_BG",
    "BADGE_WARN_BG", "BADGE_GOOD_BG", "KV_WARN_BORDER", "TAG_BAD_BORDER",
]


@pytest.mark.parametrize("name", TOKENS_USED_IN_STYLESHEET)
def test_stylesheet_contains_every_token_hex_it_uses(name):
    assert getattr(tokens, name) in qss.build_stylesheet()


def test_stylesheet_is_nonempty_and_has_no_obvious_placeholder():
    sheet = qss.build_stylesheet()
    assert len(sheet) > 500
    assert "{{" not in sheet and "}}" not in sheet  # no un-substituted f-string braces


def test_apply_theme_sets_stylesheet_palette_and_font(qapp):
    from PyQt6.QtGui import QPalette

    qss.apply_theme(qapp)
    assert qapp.styleSheet() == qss.build_stylesheet()
    palette = qapp.palette()
    assert palette.color(QPalette.ColorRole.Window).name() == tokens.BG
    assert palette.color(QPalette.ColorRole.Base).name() == tokens.PANEL
    assert palette.color(QPalette.ColorRole.Text).name() == tokens.TXT
    assert palette.color(QPalette.ColorRole.Highlight).name() == tokens.ACC
    assert qapp.font().families() == list(tokens.FONT_STACK)


# --------------------------------------------------------------------------
# Widgets -- construct offscreen, check dynamic properties
# --------------------------------------------------------------------------

def test_button_default_variant(qapp):
    button = Button("Start")
    assert button.property("variant") == "default"
    assert button.property("small") is False
    assert button.property("toggled") is False


def test_button_primary_small_toggled_properties(qapp):
    button = Button("masked", variant="primary", small=True, toggled_on=True)
    assert button.property("variant") == "primary"
    assert button.property("small") is True
    assert button.property("toggled") is True


def test_button_ghost_variant(qapp):
    button = Button("Logs", variant="ghost")
    assert button.property("variant") == "ghost"


def test_button_set_variant_and_set_toggled(qapp):
    button = Button("envelope")
    button.set_variant("ghost")
    assert button.property("variant") == "ghost"
    button.set_toggled(True)
    assert button.property("toggled") is True


def test_dot_tone_property_and_size(qapp):
    dot = Dot("warn")
    assert dot.property("tone") == "warn"
    assert dot.width() == tokens.DOT_SIZE
    assert dot.height() == tokens.DOT_SIZE
    dot.set_tone("ok")
    assert dot.property("tone") == "ok"


def test_chip_exposes_dot_tone_and_count(qapp):
    chip = Chip(dot="warn", count=1, label="needs you")
    assert chip.findChild(QLabel).text() in {"1", "needs you"}  # sanity: labels exist
    assert chip._dot.property("tone") == "warn"
    assert chip._count_label.text() == "1"
    assert chip._label.text() == "needs you"
    chip.set_count(3)
    assert chip._count_label.text() == "3"
    chip.set_tone("ok")
    assert chip._dot.property("tone") == "ok"


def test_badge_exposes_badge_property(qapp):
    badge = Badge("reviewed", tone="good")
    assert badge.property("badge") == "good"
    assert badge.text() == "reviewed"


@pytest.mark.parametrize("tone", ["default", "warn", "good", "bad"])
def test_badge_every_tone_sets_the_badge_property(qapp, tone):
    badge = Badge("x", tone=tone)
    assert badge.property("badge") == tone


def test_badge_set_state_updates_text_and_tone(qapp):
    badge = Badge("waiting", tone="default")
    badge.set_state("check crop", "warn")
    assert badge.text() == "check crop"
    assert badge.property("badge") == "warn"


def test_kvrow_default_tone(qapp):
    row = KvRow("Crop", "288, 784 · 1344 × 55")
    assert row.property("tone") == ""
    assert row._value_label.property("tone") == ""
    assert row._value_label.text() == "288, 784 · 1344 × 55"
    assert row._key_label.text() == "Crop"


@pytest.mark.parametrize("tone", ["warn", "ok", "bad", "acc"])
def test_kvrow_tone_variants(qapp, tone):
    row = KvRow("Brightness", "211", tone=tone)
    assert row.property("tone") == tone
    assert row._value_label.property("tone") == tone


def test_kvrow_set_value_updates_text_and_tone(qapp):
    row = KvRow("Brightness", "209", tone=None)
    row.set_value("211", "acc")
    assert row._value_label.text() == "211"
    assert row.property("tone") == "acc"


def test_kvrow_only_repolishes_when_the_tone_actually_changes(qapp, monkeypatch):
    """Every hot refresh path rewrites every row. An unpolish/polish pass per
    row per refresh -- two of them, row and value -- buys nothing when the
    tone is what it already was; only the text changed, and a QSS rule does
    not select on text."""
    calls = []
    monkeypatch.setattr(base, "_repolish", lambda widget: calls.append(widget))
    row = KvRow("Brightness", "209")
    calls.clear()

    row.set_value("211")                       # same tone, new text
    assert calls == []
    assert row.value() == "211"

    row.set_value("211", "warn")               # the tone changed: both repolish
    assert calls == [row._value_label, row]
    calls.clear()

    row.set_value("214", "warn")
    assert calls == []
    assert row.value_tone() == "warn" and row.property("tone") == "warn"


def test_section_header_uppercases_and_holds_trailing_widget(qapp):
    trailing = QPushButton("re-detect")
    header = SectionHeader("Detected", trailing=trailing)
    assert header._label.text() == "DETECTED"
    assert header._trailing is trailing


def test_conf_bar_tone_property_and_caption(qapp):
    bar = ConfBar(0.92, tone="ok", caption="12 of 12 samples agree")
    assert bar.property("tone") == "ok"
    assert bar._caption.text() == "12 of 12 samples agree"
    assert bar._track._fraction == pytest.approx(0.92)


def test_conf_bar_set_value_updates_fraction_and_tone(qapp):
    bar = ConfBar(0.92, tone="ok", caption="")
    bar.set_value(0.54, "warn", "narrow safe range (203–216)")
    assert bar.property("tone") == "warn"
    assert bar._track._fraction == pytest.approx(0.54)
    assert bar._caption.text() == "narrow safe range (203–216)"


def test_mini_progress_tone_and_fraction(qapp):
    mini = MiniProgress(0.62)
    assert mini.property("tone") == "run"
    assert mini._track._fraction == pytest.approx(0.62)
    mini.set_value(1.0)
    assert mini._track._fraction == pytest.approx(1.0)


# --------------------------------------------------------------------------
# SegmentedControl
# --------------------------------------------------------------------------

def test_segmented_control_builds_one_button_per_item(qapp):
    seg = SegmentedControl(["All 5", "Needs you 1", "Reviewed 3"])
    assert len(seg._buttons) == 3
    assert seg._buttons[0].property("on") is True
    assert seg._buttons[1].property("on") is False
    assert seg.current() == 0


def test_segmented_control_click_emits_current_changed(qapp):
    seg = SegmentedControl(["All 5", "Needs you 1", "Reviewed 3"])
    received = []
    seg.current_changed.connect(received.append)
    QTest.mouseClick(seg._buttons[1], Qt.MouseButton.LeftButton)
    assert received == [1]
    assert seg.current() == 1
    assert seg._buttons[1].property("on") is True
    assert seg._buttons[0].property("on") is False


def test_segmented_control_click_on_current_segment_does_not_emit(qapp):
    seg = SegmentedControl(["All 5", "Needs you 1"])
    received = []
    seg.current_changed.connect(received.append)
    QTest.mouseClick(seg._buttons[0], Qt.MouseButton.LeftButton)
    assert received == []


def test_segmented_control_set_current_does_not_emit(qapp):
    seg = SegmentedControl(["All 5", "Needs you 1"])
    received = []
    seg.current_changed.connect(received.append)
    seg.set_current(1)
    assert received == []
    assert seg.current() == 1
    assert seg._buttons[1].property("on") is True


def test_segmented_control_set_labels_replaces_segments(qapp):
    seg = SegmentedControl(["A", "B"])
    seg.set_current(1)
    seg.set_labels(["X", "Y", "Z"])
    assert len(seg._buttons) == 3
    assert [b.text() for b in seg._buttons] == ["X", "Y", "Z"]


# --------------------------------------------------------------------------
# Task 3 additions: read accessors, set_texts, indeterminate MiniProgress,
# ElidedLabel
# --------------------------------------------------------------------------

def test_read_accessors(qapp):
    chip = Chip(dot="warn", count=1, label="needs you")
    assert chip.text() == "1 needs you" and chip.tone() == "warn"
    assert KvRow("Crop", "288, 784 · 1344 × 55").value() == "288, 784 · 1344 × 55"
    row = KvRow("Brightness", "211")
    row.set_value("211", "warn", tint_border=False)
    assert (row.value_tone(), row.property("tone")) == ("warn", "")
    assert ConfBar(0.5, caption="12 of 12 samples agree").caption() == "12 of 12 samples agree"
    assert SegmentedControl(["All 5", "Needs you 1"]).labels() == ["All 5", "Needs you 1"]


def test_segmented_control_set_texts_keeps_buttons_and_current(qapp):
    seg = SegmentedControl(["All 5", "Needs you 1", "Reviewed 3"])
    seg.set_current(2)
    buttons = list(seg._buttons)
    seg.set_texts(["All 5", "Needs you 0", "Reviewed 4"])
    assert seg._buttons == buttons
    assert seg.labels() == ["All 5", "Needs you 0", "Reviewed 4"]
    assert seg.current() == 2
    seg.set_texts(["A", "B"])                       # another count: rebuilt
    assert seg.labels() == ["A", "B"]


def test_mini_progress_indeterminate_animates_only_while_shown(qapp):
    mini = MiniProgress()
    mini.set_indeterminate(True)
    assert mini.is_indeterminate()
    assert not mini._timer.isActive()               # not shown yet
    mini.show()
    assert mini._timer.isActive()
    before = (mini._track._offset, mini._track._fraction)
    mini._advance()
    mini._advance()
    assert (mini._track._offset, mini._track._fraction) != before
    mini.hide()
    assert not mini._timer.isActive()
    mini.show()
    mini.set_value(0.4)
    assert not mini.is_indeterminate() and not mini._timer.isActive()
    assert mini._track._fraction == pytest.approx(0.4) and mini._track._offset == 0.0
    mini.close()


def test_elided_label_keeps_the_full_text(qapp):
    from app.widgets.base import ElidedLabel

    label = ElidedLabel("ZS2_-_12_[1080p]TXHBR.mp4 with a very long tail that cannot fit")
    label.resize(60, 20)
    label.show()                                    # resize events reach a shown widget
    assert label.full_text() == "ZS2_-_12_[1080p]TXHBR.mp4 with a very long tail that cannot fit"
    assert label.toolTip() == label.full_text()
    assert label.text() != label.full_text() and label.text().endswith("…")
    label.resize(2000, 20)
    assert label.text() == label.full_text()
    assert label.sizeHint().width() >= label.fontMetrics().horizontalAdvance(label.full_text())
    assert label.minimumSizeHint().width() == 0
    label.close()


def test_stylesheet_restates_the_disabled_look_for_button_variants():
    sheet = qss.build_stylesheet()
    assert 'QPushButton[variant="primary"]:disabled' in sheet
    assert 'QPushButton[variant="ghost"]:disabled' in sheet
    # The disabled rule must come after the variant rule it overrides.
    assert sheet.index('QPushButton[variant="primary"]:disabled') > sheet.index('QPushButton[variant="primary"] {')


def test_stylesheet_shows_keyboard_focus_on_every_button():
    """With a stylesheet installed Qt draws no default focus rectangle, so
    without a rule of its own a focused button is indistinguishable from an
    unfocused one -- while Space and T step aside for whatever widget has the
    focus. The sheet paints the same 1 px accent border the folder-settings
    spin boxes already use."""
    sheet = qss.build_stylesheet()
    assert "QPushButton:focus" in sheet
    rule = sheet[sheet.index("QPushButton:focus"):]
    assert f"border: 1px solid {tokens.ACC};" in rule[:rule.index("}")]
    # ... and after the plain QPushButton rule it overrides.
    assert sheet.index("QPushButton:focus") > sheet.index("QPushButton {")
    # An ID selector outranks a bare pseudo-class, so the buttons that have
    # one are named too, or they would keep their own border when focused.
    for name in ("StageTab", "LogHeader", "SegmentItem"):
        assert f"QPushButton#{name}:focus" in sheet
    # The primary variant's fill IS the accent, so its ring is the dark ink.
    primary = sheet[sheet.index('QPushButton[variant="primary"]:focus'):]
    assert f"border: 1px solid {tokens.PRIMARY_TEXT};" in primary[:primary.index("}")]


def test_chip_radius_is_the_qt_adjusted_token():
    # CSS clamps `.chip`'s 20px to a pill; Qt draws square corners for a radius above half the height.
    assert tokens.RADIUS_CHIP == tokens.px(20)
    assert tokens.RADIUS_CHIP_QT == tokens.px(10)
    sheet = qss.build_stylesheet()
    chip_rule = sheet[sheet.index("QWidget#Chip {"):]
    assert f"border-radius: {tokens.RADIUS_CHIP_QT}px;" in chip_rule[:chip_rule.index("}")]
    assert not hasattr(qss, "CHIP_QT_RADIUS")


# --------------------------------------------------------------------------
# UI scale: nothing may be pinned to a literal pixel
# --------------------------------------------------------------------------
# The window is drawn at `tokens.UI_SCALE` (the mockups' 9-11 px type is
# unreadable on a real desktop). These are the tests that keep it that way:
# a length written as a literal instead of `tokens.px(...)` would simply not
# move between the two scales below, and every one of them says so.


@contextlib.contextmanager
def ui_scale(value: float):
    """Re-read the tokens at `value`, as a fresh process with
    `$OCR_MANAGER_UI_SCALE=value` would.

    Only `app.theme.tokens` is reloaded, and `importlib.reload` updates a
    module in place: `app.theme.qss` and `app.widgets.base` both hold the
    module object (`from app.theme import tokens`), so they see the new
    sizes without being reloaded themselves -- no class object is replaced,
    so widgets built inside and outside this block stay the same types."""
    previous = os.environ.get("OCR_MANAGER_UI_SCALE")
    os.environ["OCR_MANAGER_UI_SCALE"] = str(value)
    importlib.reload(tokens)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OCR_MANAGER_UI_SCALE", None)
        else:
            os.environ["OCR_MANAGER_UI_SCALE"] = previous
        importlib.reload(tokens)


@contextlib.contextmanager
def themed_ui_scale(qapp, value: float):
    """`ui_scale()`, with the stylesheet that goes with it installed on the
    session QApplication -- the only way to measure what a widget actually
    lays out to. The ambient theme is restored however the block exits, so a
    failing assertion cannot leak a scale into the tests that follow."""
    try:
        with ui_scale(value):
            qss.apply_theme(qapp)
            yield
    finally:
        qss.apply_theme(qapp)


def _stylesheet_lengths(sheet: str) -> list[float]:
    return [float(value) for value in re.findall(r"([0-9]+(?:\.[0-9]+)?)px", sheet)]


def test_ui_scale_reads_the_environment_and_clamps_it():
    ambient = tokens.UI_SCALE                # whatever this run was started at
    with ui_scale(1.0):
        assert tokens.UI_SCALE == 1.0
        assert tokens.px(246) == 246 and tokens.pt(11.5) == 11.5
    with ui_scale(2.0):
        assert tokens.UI_SCALE == 2.0
        assert tokens.px(246) == 492 and tokens.pt(11.5) == 23.0
    with ui_scale(9.0):                      # out of range: clamped, not raised
        assert tokens.UI_SCALE == tokens.UI_SCALE_MAX
    assert tokens.UI_SCALE == ambient        # and restored afterwards


def test_every_stylesheet_length_follows_the_ui_scale():
    """A regression guard against re-introducing a literal: at twice the
    scale every length in the generated QSS must have grown with it. The
    only lengths allowed to stay put are 1 px hairlines -- borders and the
    menu separator, whose width is part of Qt's box model."""
    with ui_scale(1.0):
        single = _stylesheet_lengths(qss.build_stylesheet())
    with ui_scale(2.0):
        double = _stylesheet_lengths(qss.build_stylesheet())

    assert single and len(single) == len(double)
    scaled = 0
    for one, two in zip(single, double, strict=True):
        if one == 1:
            assert two in (1.0, 2.0)         # a hairline stays; a 1 px padding doubles
            continue
        assert two > one, f"{one}px did not grow with the scale"
        assert abs(two - 2 * one) <= 1, f"{one}px -> {two}px is not the scale"
        scaled += 1
    assert scaled >= 40                      # the sheet really is mostly lengths


def test_stylesheet_font_sizes_are_whole_pixels_from_the_scaled_tokens():
    """Qt's QSS parser truncates a fractional `font-size:...px`, so a token
    emitted raw would lose most of a pixel (9.5 at 1.25 is 11.875 -> 11,
    +16% where every other size gets +25%)."""
    sheet = qss.build_stylesheet()
    sizes = re.findall(r"font-size: ([0-9.]+)px", sheet)
    assert sizes
    for size in sizes:
        assert "." not in size, f"font-size: {size}px is fractional"
    assert f"font-size: {round(tokens.FONT_SIZE_BODY)}px" in sheet
    assert f"font-size: {round(tokens.FONT_SIZE_XS)}px" in sheet
    # ... and they are the scaled tokens, not the mockup's own numbers.
    with ui_scale(1.0):
        single = qss.build_stylesheet()
    with ui_scale(2.0):
        doubled = qss.build_stylesheet()
    assert "font-size: 12px" in single and "font-size: 12px" not in doubled   # .btn, 11.5
    assert "font-size: 23px" in doubled and "font-size: 23px" not in single


def test_hairline_borders_stay_one_pixel_at_every_scale():
    """The deliberate exception. A border is part of Qt's box model: at 2 px
    the content box shrinks by a pixel on each side, which moves every label
    inside a button, and the focus ring would resize its own button. A 1 px
    rule still reads at any scale."""
    for scale in (1.0, 2.0, 3.0):
        with ui_scale(scale):
            sheet = qss.build_stylesheet()
        assert f"border: 1px solid {tokens.ACC};" in sheet     # the focus ring
        assert "border-bottom: 1px solid" in sheet             # panel separators
        assert "2px solid" not in sheet


def _widget_metrics() -> dict[str, int]:
    """Every length the base widgets lay out with, measured off real
    widgets (not read back from the tokens they were built from)."""
    chip = Chip(dot="ok", count=3, label="reviewed")
    kv = KvRow("Crop", "288, 784 · 1344 × 55")
    header = SectionHeader("Detected")
    conf = ConfBar(0.5, caption="12 of 12 samples agree")
    mini = MiniProgress(0.5)
    toggle = base.Toggle(True)
    horizontal = SegmentedControl(["All 5"])
    vertical = SegmentedControl(["All 5"], orientation=Qt.Orientation.Vertical)
    chip_margins = chip.layout().contentsMargins()
    kv_margins = kv.layout().contentsMargins()
    conf_margins = conf.layout().contentsMargins()
    return {
        "dot": Dot().width(),
        "dot_height": Dot().height(),
        "chip_margin_x": chip_margins.left(),
        "chip_margin_y": chip_margins.top(),
        "chip_spacing": chip.layout().spacing(),
        "chip_min_height": chip.minimumHeight(),
        "kv_margin_x": kv_margins.left(),
        "kv_margin_y": kv_margins.top(),
        "kv_spacing": kv.layout().spacing(),
        "header_spacing": header.layout().spacing(),
        "conf_margin_top": conf_margins.top(),
        "conf_margin_bottom": conf_margins.bottom(),
        "conf_spacing": conf.layout().spacing(),
        "conf_track_width": conf._track.width(),
        "conf_track_height": conf._track.height(),
        "mini_track_width": mini._track.width(),
        "mini_track_height": mini._track.height(),
        "segment_spacing_h": horizontal._layout.spacing(),
        "segment_spacing_v": vertical._layout.spacing(),
        "toggle_track_width": toggle.TRACK_WIDTH,
        "toggle_track_height": toggle.TRACK_HEIGHT,
        "toggle_knob": toggle.KNOB,
        "toggle_gap": toggle.GAP,
    }


def test_base_widget_lengths_follow_the_ui_scale(qapp):
    with ui_scale(1.0):
        single = _widget_metrics()
    with ui_scale(2.0):
        double = _widget_metrics()
    # At scale 1 every one of them is the mockup's own number ...
    assert single == {
        "dot": 7, "dot_height": 7,
        "chip_margin_x": 9, "chip_margin_y": 3, "chip_spacing": 6, "chip_min_height": 20,
        "kv_margin_x": 8, "kv_margin_y": 5, "kv_spacing": 8,
        "header_spacing": 6,
        "conf_margin_top": 2, "conf_margin_bottom": 8, "conf_spacing": 6,
        "conf_track_width": 74, "conf_track_height": 3,
        "mini_track_width": 90, "mini_track_height": 4,
        "segment_spacing_h": 4, "segment_spacing_v": 2,
        "toggle_track_width": 24, "toggle_track_height": 14, "toggle_knob": 8, "toggle_gap": 7,
    }
    # ... and at scale 2, exactly twice it. A literal would sit still here.
    assert double == {name: 2 * value for name, value in single.items()}


def test_the_dot_stays_round_and_the_bar_track_radius_stays_proportional(qapp):
    """A scale is not a licence to change a shape. `Dot` is square, so it
    paints a circle; `_BarTrack`'s radius is clamped to half its height, so
    the 3-4 px bar keeps rounded ends instead of square-cut ones (the token
    alone, 2 px at scale 1, already outgrows half of a 3 px bar)."""
    radii = {}
    for scale in (1.0, 1.25, 2.0, 3.0):
        with ui_scale(scale):
            dot = Dot("ok")
            assert dot.width() == dot.height() == tokens.px(tokens.DOT_SIZE_BASE)
            conf, mini = ConfBar(1.0), MiniProgress(1.0)
            for name, track in (("conf", conf._track), ("mini", mini._track)):
                assert 0 < track.radius() <= track.height() / 2
                radii.setdefault(name, []).append(track.radius())
    for values in radii.values():
        assert values[-1] == 3 * values[0]          # ... and it scaled with the rest


def test_the_chip_stays_a_pill_at_every_scale(qapp):
    """Qt does not clamp a border-radius to half the box the way CSS does --
    it squares the corners off instead -- and font metrics do not grow in
    exact step with the scaled padding, so the chip carries a floor."""
    for scale in (1.0, 1.25, 1.75, 2.0, 3.0):
        with themed_ui_scale(qapp, scale):
            chip = Chip(dot="warn", count=1, label="needs you")
            chip.ensurePolished()
            chip.adjustSize()
            assert chip.height() >= 2 * tokens.RADIUS_CHIP_QT, f"squared corners at {scale}"


def test_scaled_padding_never_clips_a_button_label(qapp):
    """The padding grew; so must the button. Qt measures a styled button as
    text + padding + border, so this fails the moment a padding is scaled
    past a font size that is not."""
    for scale in (1.0, 1.25, 2.0, 3.0):
        with themed_ui_scale(qapp, scale):
            for button in (Button("Mark reviewed"), Button("re-detect", variant="ghost", small=True)):
                button.ensurePolished()
                text = button.fontMetrics().horizontalAdvance(button.text())
                assert button.sizeHint().width() >= text + 2 * tokens.px(3)
                assert button.sizeHint().height() > button.fontMetrics().height()
