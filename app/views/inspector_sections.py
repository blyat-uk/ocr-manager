"""The inspector's sections (`.sec`, ui-spec §3.7), split out of
`inspector.py` to keep each file single-purpose:

- `DetectedSection`: "DETECTED" -- Crop, Brightness and OCR window rows with
  confidence bars (`state_text` captions); a click on a row asks for its tab.
- `ProofSection`: "PROOF · REAL OCR OF 30 S" -- "running…" while the proof
  runs, then up to three "MM:SS text" lines and "{n} lines · took {s} s".
- `ChangeOffer`: "IF YOU CHANGE SOMETHING HERE" -- the hint re-detect offer.
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
from app.widgets.base import Button, ConfBar, KvRow, SectionHeader, repolish

PROOF_LINES_SHOWN = 3
PULSE_MS = 1400
PULSE_LOW_OPACITY = 0.35


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
    """`.sec`: padding 10 12, a hairline below."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("InspectorSection")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(12, 10, 12, 10)
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
        self.body.addSpacing(8)
        self.crop_row, self.crop_conf = self._add_row("Crop")
        self.brightness_row, self.brightness_conf = self._add_row("Brightness")
        self.window_row, self.window_conf = self._add_row("OCR window")
        self.body.addWidget(note_label("Values belong to this file."))

    def _add_row(self, key: str) -> tuple[DetectedRow, ConfBar]:
        row = DetectedRow(key)
        row.clicked.connect(lambda: self.tab_requested.emit(self.TAB_FOR_ROW[key]))
        conf = ConfBar(0.0, "ok", "")
        self.body.addWidget(row)
        self.body.addSpacing(3)
        self.body.addWidget(conf)
        return row, conf

    def set_entry(self, entry) -> None:
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
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(8)
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
    run_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.run_button = small_button("T run")
        self.run_button.clicked.connect(self.run_requested)
        self.body.addWidget(SectionHeader("Proof · real OCR of 30 s", trailing=self.run_button))
        self.body.addSpacing(8)
        self.status_label = note_label("running…")
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
        for _ in range(PROOF_LINES_SHOWN):
            line = _OcrLine()
            self._lines.append(line)
            self.body.addWidget(line)
        self.body.addSpacing(6)
        self.note_label = note_label()
        self.body.addWidget(self.note_label)
        self.show_nothing()

    def show_nothing(self) -> None:
        self._set_running(False)
        self._show_lines([])
        self._set_note("")

    def show_running(self) -> None:
        self._show_lines([])
        self._set_note("")
        self._set_running(True)

    def show_result(self, result) -> None:
        self._set_running(False)
        self._show_lines(result.lines[:PROOF_LINES_SHOWN])
        count = len(result.lines)
        self._set_note(f"{count} {'line' if count == 1 else 'lines'} · took {result.seconds:.1f} s")

    def show_error(self, message: str) -> None:
        self._set_running(False)
        self._show_lines([])
        self._set_note(message, "warn")

    def texts(self) -> list[str]:
        return [line.text() for line in self._lines if not line.isHidden()]

    def _set_running(self, running: bool) -> None:
        self.status_label.setVisible(running)
        if running:
            self._pulse.start()
        else:
            self._pulse.stop()
            self._pulse_effect.setOpacity(1.0)

    def _show_lines(self, lines) -> None:
        for index, widget in enumerate(self._lines):
            if index < len(lines):
                start, _end, text = lines[index]
                widget.set_line(start, text)
                widget.show()
            else:
                widget.hide()

    def _set_note(self, text: str, tone: str = "") -> None:
        self.note_label.setText(text)
        self.note_label.setVisible(bool(text))
        if self.note_label.property("tone") != tone:
            self.note_label.setProperty("tone", tone)
            repolish(self.note_label)


class ChangeOffer(Section):
    """Shown after a manual crop or brightness edit (see Inspector)."""

    NOTE = "Corrections are never copied verbatim to other episodes. Instead the app offers:"

    hint_requested = pyqtSignal()
    dismissed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.body.addWidget(SectionHeader("If you change something here"))
        self.body.addSpacing(8)
        self.note_label = note_label(self.NOTE)
        self.body.addWidget(self.note_label)
        self.body.addSpacing(7)
        self.hint_button = small_button("")
        self.hint_button.clicked.connect(self.hint_requested)
        self.this_file_only_button = small_button("apply to this file only", "ghost")
        self.this_file_only_button.clicked.connect(self.dismissed)
        for button in (self.hint_button, self.this_file_only_button):
            self.body.addWidget(button, 0, Qt.AlignmentFlag.AlignLeft)
            self.body.addSpacing(6)

    def set_targets(self, count: int) -> None:
        self.hint_button.setText(f"↻ re-detect the other {count} using this as a hint")
        self.hint_button.setEnabled(count > 0)
