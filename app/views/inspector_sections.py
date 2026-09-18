"""The inspector's sections (`.sec`, ui-spec §3.7), split out of
`inspector.py` to keep each file single-purpose:

- `DetectedSection`: "DETECTED" -- Crop, Brightness and OCR window rows with
  confidence bars (`state_text` captions); a click on a row asks for its tab.
  Its note carries the series-median brightness sentence.
- `ProofSection`: "PROOF · REAL OCR OF 30 S" (ruling C4) -- "running on
  MM:SS-MM:SS…" while the proof runs, then up to PROOF_LINES_SHOWN "MM:SS
  text" lines ("show all" reveals the rest), and "{n} lines · took {s} s".
  A result whose settings have since changed keeps its lines and says so.
- `ChangeOffer`: "IF YOU CHANGE SOMETHING HERE" -- the hint re-detect offer
  (ruling C3), one button per kind the user edited in this session.

These widgets hold no state about which file they describe and never call
the controller: `Inspector` decides what to show and when.
"""
from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, pyqtSignal
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.state_text import (
    brightness_caption,
    brightness_text,
    clock,
    crop_caption,
    crop_text,
    field_blocking,
    ranges_caption,
    ranges_text,
)
from app.theme import tokens
from app.widgets.base import Button, ConfBar, KvRow, SectionHeader, repolish

PROOF_LINES_SHOWN = 6             # more recognised lines than this hide behind "show all"
PULSE_MS = 1400
PULSE_LOW_OPACITY = 0.35

SECTION_MARGIN_X = 12             # `.sec` padding, left and right (mockup px, scaled at every use)
BUTTON_SM_CHROME = 18             # `.btn.sm`: 8 px padding and a 1 px border each side (mockup px)

VALUES_NOTE = "Values belong to this file."
SHOW_ALL_TEXT = "show all"
STALE_TEXT = "settings changed — run again"
NO_LINES_TEXT = "No subtitles recognised in this window — check crop and brightness."
RUNNING_TEXT = "running on {window}…"
RUNNING_UNKNOWN_TEXT = "running…"            # defensive: run_proof refuses an unknown duration
HINT_TEXT = "↻ re-detect the other {count} using this {what} as a hint"
# Count 0: the offer is still shown, disabled, so the user can see there is
# nothing else to re-detect -- but "the other 0" reads as a template with a
# hole in it, and there is no ↻ to press.
HINT_NONE_TEXT = "no other file to re-detect with this {what}"
REDETECTING_TEXT = "re-detecting {count} files…"
HINT_KINDS = ("crop", "brightness")          # the two kinds ruling C3 offers, in inspector order


def button_text_budget() -> int:
    """What a `.btn.sm` label may measure inside a section of the inspector
    column before the column has to grow and clip. A function, not a
    constant: it is the scaled inspector width less the scaled section
    padding and button chrome, so it follows `tokens.UI_SCALE` instead of
    freezing whatever the scale was at import time. The bare 1 px is the
    inspector's own border, the stylesheet's hairline."""
    return tokens.INSPECTOR_WIDTH - 1 - 2 * tokens.px(SECTION_MARGIN_X) - tokens.px(BUTTON_SM_CHROME)


def hint_text(count: int, what: str) -> str:
    """ui-spec §3.7's hint button label, verbatim and on one line -- or, with
    nothing to re-detect, the sentence for that. What the button shows is
    this text wrapped to the column (see `wrap_to_width`)."""
    if count <= 0:
        return HINT_NONE_TEXT.format(what=what)
    return HINT_TEXT.format(count=count, what=what)


def _greedy_wrap(text: str, metrics, width: int) -> list[str]:
    """`text` broken at spaces so no line measures wider than `width` in
    `metrics`' font. Greedy, like every word wrap: a word too wide on its own
    still gets its line."""
    lines: list[str] = []
    line = ""
    for word in text.split(" "):
        candidate = f"{line} {word}" if line else word
        if line and metrics.horizontalAdvance(candidate) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def wrap_to_width(text: str, metrics, width: int) -> str:
    """`text` wrapped to `width` and joined by newlines, which a QPushButton
    paints (it never breaks a line itself).

    Measuring, rather than breaking the copy at a hard-coded word, is what
    keeps the label a single verbatim string: reword it and it wraps wherever
    it then has to, instead of quietly wrapping in the wrong place or not at
    all. A wrapped label is balanced afterwards -- re-wrapped to the narrowest
    width that still fills the same number of lines -- so the last line is
    never left holding one word, as CSS `text-wrap: balance` does it."""
    lines = _greedy_wrap(text, metrics, width)
    if len(lines) > 1:
        low = max(metrics.horizontalAdvance(word) for word in text.split(" "))
        high = width
        while low < high:                          # the narrowest width with as few lines
            middle = (low + high) // 2
            if len(_greedy_wrap(text, metrics, middle)) <= len(lines):
                high = middle
            else:
                low = middle + 1
        lines = _greedy_wrap(text, metrics, low)
    return "\n".join(lines)


def small_button(text: str, variant: str = "default") -> Button:
    """A `.btn.sm` that does not take focus from the queue on click (Tab still
    reaches it), so ↑/↓/Space/T keep working after a click."""
    button = Button(text, variant, small=True)
    button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
    return button


def note_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("Note")
    label.setWordWrap(True)
    return label


class Section(QWidget):
    """`.sec`: padding 10 12 (mockup px, scaled), a hairline below."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("InspectorSection")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.body = QVBoxLayout(self)
        margin_x = tokens.px(SECTION_MARGIN_X)
        self.body.setContentsMargins(margin_x, tokens.px(10), margin_x, tokens.px(10))
        self.body.setSpacing(0)


class DetectedRow(KvRow):
    """A Detected summary row; clicking it switches the stage tab."""

    clicked = pyqtSignal()

    def __init__(self, key: str, parent: QWidget | None = None):
        super().__init__(key, "—", parent=parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class DetectedSection(Section):
    # row key -> the stage tab it opens
    TAB_FOR_ROW = {"Crop": "Crop", "Brightness": "Brightness", "OCR window": "Time ranges"}

    tab_requested = pyqtSignal(str)
    redetect_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.redetect_button = small_button("↻ re-detect", "ghost")
        self.redetect_button.clicked.connect(self.redetect_requested)
        header = SectionHeader("Detected", trailing=self.redetect_button)
        self.body.addWidget(header)
        self.body.addSpacing(tokens.px(8))
        self.crop_row, self.crop_conf = self._add_row("Crop")
        self.brightness_row, self.brightness_conf = self._add_row("Brightness")
        self.window_row, self.window_conf = self._add_row("OCR window")
        self.note_label = note_label(VALUES_NOTE)
        self.body.addWidget(self.note_label)

    def _add_row(self, key: str) -> tuple[DetectedRow, ConfBar]:
        row = DetectedRow(key)
        row.clicked.connect(lambda: self.tab_requested.emit(self.TAB_FOR_ROW[key]))
        conf = ConfBar(0.0, "ok", "")
        self.body.addWidget(row)
        self.body.addSpacing(tokens.px(3))
        self.body.addWidget(conf)
        return row, conf

    def set_entry(self, entry, median_note: str = "") -> None:
        """`median_note`: the series-median sentence (`state_text.
        series_median_note`), appended to the section's note; "" leaves the
        note at its first sentence."""
        self.note_label.setText(" ".join(part for part in (VALUES_NOTE, median_note) if part))
        for field, row, conf, value, caption in (
                ("crop", self.crop_row, self.crop_conf, crop_text(entry.crop), crop_caption),
                ("brightness", self.brightness_row, self.brightness_conf, brightness_text(entry.brightness),
                 brightness_caption),
                ("ranges", self.window_row, self.window_conf, ranges_text(entry.time_ranges), ranges_caption)):
            stored = getattr(entry, "time_ranges" if field == "ranges" else field)
            blocking = field_blocking(entry, field)
            text, fraction, tone = caption(entry.evidence.get(field), None if stored is None else stored.source,
                                           blocking=blocking)
            warn = tone == "warn" or blocking             # a flagged value reads warn even without evidence
            row.set_value(value, "warn" if warn else None, tint_border=False)
            conf.set_value(fraction, "warn" if warn else "ok", text)


class _OcrLine(QWidget):
    """`.ocrline`: a tabular time and the recognised text, dashed rule below."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("OcrLine")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, tokens.px(4), 0, tokens.px(4))
        layout.setSpacing(tokens.px(8))
        self.time_label = QLabel()
        self.time_label.setObjectName("OcrTime")
        self.text_label = QLabel()
        self.text_label.setObjectName("OcrText")
        layout.addWidget(self.time_label, 0, Qt.AlignmentFlag.AlignBaseline)
        layout.addWidget(self.text_label, 1, Qt.AlignmentFlag.AlignBaseline)

    def polish_width(self) -> None:
        """Times line up like tabular figures: every time gets the width of
        the widest "00:00"-shaped text in the label's (styled) font."""
        metrics = self.time_label.fontMetrics()
        self.time_label.setFixedWidth(max(metrics.horizontalAdvance(f"{d}{d}:{d}{d}") for d in "0123456789"))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.polish_width()

    def set_line(self, start: float, text: str) -> None:
        self.time_label.setText(clock(start))
        self.text_label.setText(text)

    def text(self) -> str:
        return f"{self.time_label.text()} {self.text_label.text()}"


class ProofSection(Section):
    """"Proof · real OCR of 30 s" (ruling C4). Four presentations, each set by
    the Inspector from the controller's proof state: nothing yet, running,
    a result (fresh or stale), and a refusal ("Can't run yet: ...").

    The "T run" button is disabled exactly while that file's proof runs, or
    while the file has not been scanned and so has no window to run on
    (`set_runnable`), and so is the T key that shares its command (ruling 5)."""

    run_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.run_button = small_button("T run")
        self.run_button.clicked.connect(self.run_requested)
        self.body.addWidget(SectionHeader("Proof · real OCR of 30 s", trailing=self.run_button))
        self.body.addSpacing(tokens.px(8))
        self.status_label = note_label(RUNNING_UNKNOWN_TEXT)
        self.body.addWidget(self.status_label)
        self._pulse_effect = QGraphicsOpacityEffect(self.status_label)
        self.status_label.setGraphicsEffect(self._pulse_effect)
        self._pulse = QPropertyAnimation(self._pulse_effect, b"opacity", self)
        self._pulse.setDuration(PULSE_MS)
        self._pulse.setStartValue(1.0)
        self._pulse.setKeyValueAt(0.5, PULSE_LOW_OPACITY)
        self._pulse.setEndValue(1.0)
        self._pulse.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._pulse.setLoopCount(-1)
        self._lines: list[_OcrLine] = []
        self._lines_box = QVBoxLayout()
        self._lines_box.setContentsMargins(0, 0, 0, 0)
        self._lines_box.setSpacing(0)
        self.body.addLayout(self._lines_box)
        self.show_all_button = small_button(SHOW_ALL_TEXT, "ghost")
        self.show_all_button.clicked.connect(self._show_all)
        self.body.addSpacing(tokens.px(4))
        self.body.addWidget(self.show_all_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.body.addSpacing(tokens.px(6))
        self.note_label = note_label()
        self.body.addWidget(self.note_label)
        self._result = None
        self._expanded = False
        self._runnable = True
        self.show_nothing()

    def set_runnable(self, runnable: bool, reason: str = "") -> None:
        """Whether this file can be proved at all (`state_text.can_run_proof`
        -- the predicate T and the queue's menu item use). Independent of
        whether a proof is running: `show_running` disables the button on top
        of this, and `_reset` restores it to this."""
        self._runnable = runnable
        self.run_button.setToolTip("" if runnable else reason)
        if not self._pulse.state():                  # not mid-run: apply it now
            self.run_button.setEnabled(runnable)

    # --- the four presentations ------------------------------------------------

    def show_nothing(self) -> None:
        """No proof has run for this file yet (or none is remembered)."""
        self._reset()
        self._set_note("")

    def show_running(self, window_text: str | None) -> None:
        """`window_text`: "09:38–10:08", the window OCR runs on."""
        self._reset()
        self._set_note("")
        self.run_button.setEnabled(False)
        self.status_label.setText(RUNNING_TEXT.format(window=window_text) if window_text
                                  else RUNNING_UNKNOWN_TEXT)
        self.status_label.show()
        self._pulse.start()

    def show_result(self, result, *, stale: bool = False) -> None:
        """`stale`: the file's crop, brightness or time ranges changed since
        this proof ran, so its lines describe settings the file no longer
        has. They stay on screen -- they are still the last thing real OCR
        saw -- under a note asking for another run."""
        collapse = result is not self._result
        self._reset()
        if collapse:
            self._expanded = False
        self._result = result
        self._show_lines(result.lines)
        if stale:
            self._set_note(STALE_TEXT)
        elif not result.lines:
            self._set_note(NO_LINES_TEXT, "warn")
        else:
            count = len(result.lines)
            self._set_note(f"{count} {'line' if count == 1 else 'lines'} · took {result.seconds:.1f} s")

    def show_error(self, message: str) -> None:
        """The proof could not be asked for at all (an unknown duration)."""
        self.show_nothing()
        self._set_note(message, "warn")

    def texts(self) -> list[str]:
        return [line.text() for line in self._lines if not line.isHidden()]

    # --- internals -----------------------------------------------------------------

    def _reset(self) -> None:
        self._result = None
        self._pulse.stop()
        self._pulse_effect.setOpacity(1.0)
        self.status_label.hide()
        self.run_button.setEnabled(self._runnable)
        self._show_lines([])

    def _show_all(self) -> None:
        self._expanded = True
        if self._result is not None:
            self._show_lines(self._result.lines)

    def _show_lines(self, lines) -> None:
        shown = lines if self._expanded else lines[:PROOF_LINES_SHOWN]
        while len(self._lines) < len(shown):
            line = _OcrLine()
            line.hide()
            self._lines.append(line)
            self._lines_box.addWidget(line)
        for index, widget in enumerate(self._lines):
            if index < len(shown):
                start, _end, text = shown[index]
                widget.set_line(start, text)
                widget.show()
            else:
                widget.hide()
        self.show_all_button.setVisible(len(shown) < len(lines))

    def _set_note(self, text: str, tone: str = "") -> None:
        self.note_label.setText(text)
        self.note_label.setVisible(bool(text))
        if self.note_label.property("tone") != tone:
            self.note_label.setProperty("tone", tone)
            repolish(self.note_label)


class ChangeOffer(Section):
    """Shown after a manual crop or brightness edit (see Inspector), one
    button per edited kind (ruling C3). A queued hint re-detect replaces the
    offer text with "re-detecting N files…" until those jobs end."""

    NOTE = "Corrections are never copied verbatim to other episodes. Instead the app offers:"

    hint_requested = pyqtSignal(str)             # "crop" | "brightness"
    dismissed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.body.addWidget(SectionHeader("If you change something here"))
        self.body.addSpacing(tokens.px(8))
        self.note_label = note_label(self.NOTE)
        self.body.addWidget(self.note_label)
        self.body.addSpacing(tokens.px(7))
        self.hint_buttons: dict[str, Button] = {}
        for what in HINT_KINDS:
            button = small_button("")
            button.clicked.connect(lambda _checked=False, kind=what: self.hint_requested.emit(kind))
            self.hint_buttons[what] = button
            self.body.addWidget(button, 0, Qt.AlignmentFlag.AlignLeft)
            self.body.addSpacing(tokens.px(6))
        self.this_file_only_button = small_button("apply to this file only", "ghost")
        self.this_file_only_button.clicked.connect(self.dismissed)
        self.body.addWidget(self.this_file_only_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.body.addSpacing(tokens.px(6))
        self.status_label = note_label()
        self.body.addWidget(self.status_label)

    def set_targets(self, counts: dict[str, int]) -> None:
        """`counts`: the kinds to offer, mapped to how many other files the
        re-detect would submit (`controller.hint_targets`). A kind that is
        missing is not offered; a count of 0 is offered but disabled, so the
        user can see there is nothing else to re-detect."""
        for what, button in self.hint_buttons.items():
            count = counts.get(what)
            button.setVisible(count is not None)
            button.setText(wrap_to_width(hint_text(count or 0, what), button.fontMetrics(),
                                         button_text_budget()))
            button.setEnabled(bool(count))
        offering = bool(counts)
        self.note_label.setVisible(offering)
        self.this_file_only_button.setVisible(offering)

    def set_redetecting(self, count: int) -> None:
        """`count` files are being re-detected with this file's value as a
        hint; 0 clears the line."""
        self.status_label.setText(REDETECTING_TEXT.format(count=count) if count else "")
        self.status_label.setVisible(bool(count))

    def is_empty(self) -> bool:
        """Nothing to offer and nothing running: the section has no reason to
        be on screen."""
        return self.status_label.isHidden() and all(button.isHidden() for button in self.hint_buttons.values())
