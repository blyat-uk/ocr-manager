"""The Folder settings sheet's field table and row widgets (ruling B9).

Split out of `folder_settings.py` to keep each file single-purpose (the same
rule as `inspector_sections.py`): `SECTIONS` is the whole of B9 -- every
`FolderSettings` field the sheet shows, its row key, its one-line
explanation and the editor it needs -- and `Section`/`SettingRow` are the
`.sec` and `.kv` blocks it is laid out in (workbench-hifi figure 3).

Three `FolderSettings` fields have no editor on purpose: `frames_to_skip`
and `use_gpu`, which B9 does not name and no version of the app has ever
exposed, and `label_conf_threshold`, which the label scanner ignores (see
the comment on the Labels section). All three keep their stored values.
"""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget

from app.theme import tokens
from app.widgets.base import ElidedLabel, SectionHeader

LANGUAGE_NAMES = {"ch": "Chinese"}
LABELS_NOTE = "With labels off, the label mask regions and their thresholds are hidden entirely."
AUTOPILOT_NOTE = "These tune detections that start after a change; ↻ re-detect redoes a file."
EXTRACTION_FIELDS = ("dialogue_enabled", "labels_enabled")
LABELS_SECTION = "Labels"
MASK_KEY = "Mask regions"
MASK_NOTE = "blacked out before labels are searched, e.g. a channel logo"


def language_label(code: str) -> str:
    name = LANGUAGE_NAMES.get(code)
    return f"{name} ({code})" if name else code


def language_code(text: str) -> str:
    text = text.strip()
    for code in LANGUAGE_NAMES:
        if text == language_label(code):
            return code
    return text


@dataclass(frozen=True)
class Field:
    """One FolderSettings field: its row, explanation and editor.

    `kind`: "toggle" | "number" | "language". A number with `decimals` 0 is a
    QSpinBox, otherwise a QDoubleSpinBox. `percent`: the model stores a
    fraction, the editor shows it × 100. `integer`: the model stores an int."""

    name: str
    key: str
    note: str
    kind: str = "number"
    minimum: float = 0
    maximum: float = 100
    decimals: int = 0
    step: float = 1
    suffix: str = ""
    singular_suffix: str = ""
    percent: bool = False
    integer: bool = True

    def to_editor(self, value):
        shown = value * 100 if self.percent else value
        return round(float(shown), self.decimals) if self.decimals else int(round(shown))

    def to_model(self, shown):
        if self.percent:
            return round(shown / 100, self.decimals + 2)
        if self.integer:
            return int(shown)
        return round(float(shown), self.decimals)

    def suffix_for(self, shown) -> str:
        return self.singular_suffix if self.singular_suffix and shown == 1 else self.suffix


def _percent(name: str, key: str, note: str, **kwargs) -> Field:
    return Field(name, key, note, suffix=" %", **kwargs)


def _seconds(name: str, key: str, note: str, **kwargs) -> Field:
    return Field(name, key, note, decimals=1, step=0.1, suffix=" s", integer=False, **kwargs)


SECTIONS: tuple[tuple[str, tuple[Field, ...], str], ...] = (
    ("What to extract", (
        Field("dialogue_enabled", "Dialogue subtitles", "the subtitle lines, read inside each file's crop box",
              kind="toggle"),
        Field("labels_enabled", "Positioned labels / nameplates",
              "short captions and name cards placed elsewhere in the frame", kind="toggle"),
    ), LABELS_NOTE),
    ("OCR engine", (
        Field("ocr_lang", "Language", "the OCR model's language code, e.g. ch, en, japan", kind="language"),
        _percent("conf_threshold", "Confidence threshold", "retry until this confident"),
        _percent("sim_threshold", "Merge similar lines above",
                 "neighbouring readings at least this alike become one line"),
        Field("similar_image", "Similar-frame threshold",
              "a frame with fewer pixels changed than this is not read again",
              maximum=100, decimals=2, step=0.05, suffix=" %", integer=False),
    ), ""),
    # B9 also lists a Labels "Confidence threshold (%)" for `label_conf_threshold`.
    # It has no editor on purpose: `videocr/label_scanner.py` stores it (line 153)
    # and never reads it again -- only `conf_threshold_min` filters readings
    # (lines 1474 and 1586) -- so the control would change nothing. The model
    # field, its migration and `ocr_kwargs` still carry it unchanged.
    (LABELS_SECTION, (
        _seconds("label_min_duration", "Minimum duration", "labels on screen for less time are dropped", maximum=120),
        _seconds("label_max_duration", "Maximum duration", "labels on screen for longer are dropped", maximum=120),
        _percent("label_conf_threshold_min", "Minimum confidence", "label readings below this are discarded"),
    ), ""),
    ("Performance", (
        Field("ocr_parallel", "Parallel files", "episodes OCR'd at the same time during a run", minimum=1, maximum=8),
    ), ""),
    ("Auto-pilot", (
        Field("autopilot_enabled", "Run detections when a folder opens",
              "find crop, brightness and intro/outro ranges without being asked", kind="toggle"),
        Field("brightness_full_detect_files", "Full brightness detection on the first",
              "later files start from their shared plateau, which is quicker", minimum=1, maximum=20,
              suffix=" files", singular_suffix=" file"),
        _seconds("min_segment_length", "Minimum repeating segment",
                 "shorter audio repeats across files are not treated as intro or outro", minimum=5, maximum=300),
        Field("merge_repeating_silences", "Merge repeating silences",
              "bridge silent gaps that sit at the same spot in several files", kind="toggle"),
        _percent("crop_width_fraction", "Crop width (% of frame)",
                 "detected crop boxes span this much of the width, centred", minimum=10, percent=True),
        _percent("crop_vertical_padding", "Crop vertical padding (% of frame height)",
                 "room added above and below the detected text", maximum=10, decimals=1, step=0.1, percent=True),
        _percent("crop_min_height_fraction", "Crop minimum height (% of frame height)",
                 "shorter boxes grow around their centre to this height", minimum=1, maximum=50, percent=True),
        _percent("bottom_half_cutoff", "Subtitle band starts at (% from top)",
                 "text above this line is not taken for subtitles", percent=True),
    ), AUTOPILOT_NOTE),
)


# --------------------------------------------------------------------------
# Editors
# --------------------------------------------------------------------------

class _WheelNeedsFocus:
    """The wheel changes the value only while the editor has focus; otherwise
    it scrolls the sheet (no value changes by scrolling past a field)."""

    def wheelEvent(self, event) -> None:
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


class SpinBox(_WheelNeedsFocus, QSpinBox):
    pass


class DoubleSpinBox(_WheelNeedsFocus, QDoubleSpinBox):
    pass


class LanguageCombo(_WheelNeedsFocus, QComboBox):
    pass


class SettingRow(QWidget):
    """`.kv` with an editor as its value (styled as `#KvRow`)."""

    def __init__(self, key: str, value: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("KvRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(tokens.px(8), tokens.px(4), tokens.px(5), tokens.px(4))
        layout.setSpacing(tokens.px(8))
        self.key_label = QLabel(key)
        self.key_label.setProperty("kvRole", "key")
        layout.addWidget(self.key_label)
        layout.addStretch(1)
        layout.addWidget(value)
        self.setMinimumHeight(tokens.px(30))


class Section(QWidget):
    """`.sec` inside the sheet: the first has no rule above it, the others a
    hairline with 10 px (scaled) above and below (figure 3)."""

    def __init__(self, title: str, first: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.title = title
        self.keys: list[str] = []
        self.note_text = ""
        self.setObjectName("FolderSection")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("first", first)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(0, 0 if first else tokens.px(10), 0, tokens.px(10))
        self.body.setSpacing(0)
        self.body.addWidget(SectionHeader(title))
        self.body.addSpacing(tokens.px(8))

    def add_row(self, key: str, value: QWidget, note: str) -> None:
        self.keys.append(key)
        self.body.addWidget(SettingRow(key, value))
        self.body.addSpacing(tokens.px(2))
        explanation = ElidedLabel(note)
        explanation.setObjectName("Note")
        explanation.setContentsMargins(tokens.px(8), 0, 0, 0)
        self.body.addWidget(explanation)
        self.body.addSpacing(tokens.px(8))

    def add_note(self, text: str) -> QLabel:
        self.note_text = text
        label = QLabel(text)
        label.setObjectName("Note")
        label.setWordWrap(True)
        self.body.addWidget(label)
        return label
