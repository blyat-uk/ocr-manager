"""Task 1: theme tokens, the generated stylesheet, and the base widgets.

Tokens are checked against ui-spec.md §2.1's hex values verbatim (the
digest confirms the two hi-fi mockups define byte-for-byte identical
tokens). Widgets are constructed offscreen (QT_QPA_PLATFORM=offscreen) and
checked for the dynamic properties app/theme/qss.py selects on -- pytest-qt
is not installed, so click behaviour uses PyQt6.QtTest.QTest directly (see
tests/app/conftest.py).
"""
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QLabel, QPushButton

from app.theme import qss, tokens
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


def test_rail_and_inspector_widths():
    assert tokens.RAIL_WIDTH == 246
    assert tokens.INSPECTOR_WIDTH == 322


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
