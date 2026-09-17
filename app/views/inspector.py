"""The file inspector (`.insp`, ruling B4, ui-spec §3.7): one 322 px
column about the selected episode only, scrolling vertically. Top to bottom:

1. header -- "◆ THIS EPISODE ONLY", the file name, its media line;
2. the active stage tab's `inspector_panel()`, swapped on `tab_changed`;
3. "DETECTED" -- crop, brightness and OCR window with confidence bars;
4. "PROOF · REAL OCR OF 30 S";
5. "IF YOU CHANGE SOMETHING HERE" -- after a manual crop or brightness edit
   in this session, the offer to re-detect the other files with it as a
   hint (`controller.hint_targets` counts them);
6. footer (pinned below the scroll area) -- "✓ Mark reviewed (Space)" /
   "Mark not reviewed" (disabled while the file is PENDING) and "skip file" /
   "include file".

A manual edit is noticed from `file_changed`: the file's crop box or
brightness value differs from the last one seen and is now MANUAL. So an
edit from anywhere (the tabs, the queue's paste) raises the offer for that
file, while accepting a flagged value (same value, MANUAL) does not.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from app.state_text import REVIEW_WAIT_TOOLTIP, can_mark_reviewed, is_manual, is_reviewed, media_text
from app.theme import tokens
from app.views.inspector_sections import ChangeOffer, DetectedSection, ProofSection, Section
from app.widgets.base import Button, ElidedLabel

SCOPE_TEXT = "◆ THIS EPISODE ONLY"
REVIEW_TEXT = "✓ Mark reviewed (Space)"
UNREVIEW_TEXT = "Mark not reviewed"


def _edit_keys(entry) -> dict[str, object]:
    crop = entry.crop
    return {"crop": None if crop is None else (crop.x, crop.y, crop.width, crop.height),
            "brightness": None if entry.brightness is None else entry.brightness.value}


def _is_manual(entry, kind: str) -> bool:
    return is_manual(entry.crop if kind == "crop" else entry.brightness)


class Inspector(QWidget):
    tab_requested = pyqtSignal(str)             # a stage tab title

    def __init__(self, controller, stage, parent: QWidget | None = None):
        super().__init__(parent)
        self._controller = controller
        self._stage = stage
        self._file: str | None = None
        self._seen: dict[str, dict[str, object]] = {}       # file -> last crop box / brightness seen
        self._edited: dict[str, str] = {}                    # file -> kind of its latest manual edit
        self.setObjectName("Inspector")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(tokens.INSPECTOR_WIDTH)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 0, 0, 0)                 # the 1 px left border
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("InspectorScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        content = QWidget()
        content.setObjectName("InspectorContent")
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        head = Section()
        self.scope_label = QLabel(SCOPE_TEXT)
        self.scope_label.setObjectName("InspectorScope")
        font = self.scope_label.font()
        font.setLetterSpacing(font.SpacingType.AbsoluteSpacing,
                              tokens.FONT_SIZE_SCOPE * tokens.LETTER_SPACING_INSP_SCOPE_EM)
        self.scope_label.setFont(font)
        self.file_label = ElidedLabel(mode=Qt.TextElideMode.ElideMiddle)
        self.file_label.setObjectName("InspectorFile")
        self.media_label = QLabel()
        self.media_label.setObjectName("InspectorSub")
        head.body.addWidget(self.scope_label)
        head.body.addSpacing(3)
        head.body.addWidget(self.file_label)
        head.body.addSpacing(2)
        head.body.addWidget(self.media_label)
        column.addWidget(head)

        self._panel_host = Section()
        self._panels: list[QWidget] = []
        column.addWidget(self._panel_host)

        self.detected = DetectedSection()
        self.detected.tab_requested.connect(self.tab_requested)
        self.detected.redetect_requested.connect(self._redetect)
        self.proof = ProofSection()
        self.proof.run_requested.connect(lambda: self.run_proof_for(self._file))
        self.offer = ChangeOffer()
        self.offer.hint_requested.connect(self._redetect_others)
        self.offer.dismissed.connect(self._dismiss_offer)
        for section in (self.detected, self.proof, self.offer):
            column.addWidget(section)
        column.addStretch(1)

        footer = QWidget()
        footer.setObjectName("InspectorFooter")
        footer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(12, 10, 12, 10)
        footer_layout.setSpacing(6)
        self.review_button = Button(REVIEW_TEXT, "primary")
        self.skip_button = Button("skip file", "ghost")
        for button in (self.review_button, self.skip_button):
            button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.review_button.clicked.connect(self.toggle_reviewed)
        self.skip_button.clicked.connect(self._toggle_skipped)
        footer_layout.addWidget(self.review_button, 1)
        footer_layout.addWidget(self.skip_button)
        outer.addWidget(footer)
        self._footer = footer

        # Short names for the parts tests and later tasks reach for.
        self.crop_row, self.crop_conf = self.detected.crop_row, self.detected.crop_conf
        self.brightness_row, self.brightness_conf = self.detected.brightness_row, self.detected.brightness_conf
        self.window_row, self.window_conf = self.detected.window_row, self.detected.window_conf
        self.redetect_button = self.detected.redetect_button
        self.proof_button, self.proof_status, self.proof_note = (
            self.proof.run_button, self.proof.status_label, self.proof.note_label)
        self.offer_section, self.offer_note = self.offer, self.offer.note_label
        self.hint_button, self.this_file_only_button = self.offer.hint_button, self.offer.this_file_only_button
        self.proof_texts = self.proof.texts

        stage.tab_changed.connect(self._show_panel)
        self._show_panel(stage.current_index())
        controller.project_opened.connect(self._on_project_opened)
        controller.project_closed.connect(self._on_project_closed)
        controller.files_changed.connect(self._on_files_changed)
        controller.file_changed.connect(self._on_file_changed)
        controller.folder_changed.connect(self.refresh)
        controller.proof_started.connect(self._on_proof_event)
        controller.proof_finished.connect(self._on_proof_event)
        self.set_file(None)

    # --- the file and the tab panel -------------------------------------------------------

    def current_file(self) -> str | None:
        return self._file

    def current_panel(self) -> QWidget | None:
        return next((panel for panel in self._panels if not panel.isHidden()), None)

    def _show_panel(self, index: int) -> None:
        panel = self._stage.tabs()[index].inspector_panel()
        if panel not in self._panels:
            self._panels.append(panel)
            self._panel_host.body.addWidget(panel)
        for other in self._panels:
            other.setVisible(other is panel)

    def set_file(self, name: str | None) -> None:
        self._file = name
        self.refresh()
        self._show_proof()

    def refresh(self, *_args) -> None:
        controller = self._controller
        name = self._file
        has_file = self._has_file()
        for section in (self._panel_host, self.detected, self.proof):
            section.setVisible(has_file)
        self.review_button.setEnabled(has_file)
        self.review_button.setToolTip("")
        self.skip_button.setEnabled(has_file)
        if not has_file:
            self.file_label.set_full_text("No file selected" if controller.project is not None else "")
            self.media_label.setText("")
            self.offer.hide()
            return
        entry = controller.entry(name)
        self.file_label.set_full_text(name)
        self.media_label.setText(media_text(entry.media))
        self.detected.set_entry(entry)
        reviewed = is_reviewed(entry)
        reviewable = can_mark_reviewed(entry)
        self.review_button.setText(UNREVIEW_TEXT if reviewed else REVIEW_TEXT)
        self.review_button.set_variant("default" if reviewed else "primary")
        self.review_button.setEnabled(reviewable)
        self.review_button.setToolTip("" if reviewable else REVIEW_WAIT_TOOLTIP)
        self.skip_button.setText("include file" if entry.skipped else "skip file")
        self._refresh_offer()

    # --- commands -----------------------------------------------------------------------------

    def run_proof_for(self, name: str | None) -> None:
        """Real OCR of 30 s of `name` (the queue's T, the "T run" button)."""
        if name is None or name not in self._controller.names():
            return
        try:
            self._controller.run_proof(name)
        except ValueError as exc:                      # the duration is not known yet
            if name == self._file:
                self.proof.show_error(f"Can't run yet: {exc}")

    def _has_file(self) -> bool:
        return self._file is not None and self._file in self._controller.names()

    def _redetect(self) -> None:
        if self._has_file():
            self._controller.redetect(self._file)

    def toggle_reviewed(self) -> None:
        """Mark the file reviewed, or not reviewed (the footer button, Space);
        nothing while it is PENDING."""
        if self._has_file():
            entry = self._controller.entry(self._file)
            if can_mark_reviewed(entry):
                self._controller.mark_reviewed(self._file, not is_reviewed(entry))

    def _toggle_skipped(self) -> None:
        if self._has_file():
            self._controller.set_skipped(self._file, not self._controller.entry(self._file).skipped)

    # --- proof ----------------------------------------------------------------------------------

    def _on_proof_event(self, name: str) -> None:
        if name == self._file:
            self._show_proof()

    def _show_proof(self) -> None:
        name = self._file
        if name is None or name not in self._controller.names():
            self.proof.show_nothing()
        elif self._controller.proof_pending(name):
            self.proof.show_running()
        elif (result := self._controller.proof_result(name)) is not None:
            self.proof.show_result(result)
        else:
            self.proof.show_nothing()

    # --- the change offer --------------------------------------------------------------------

    def adopt_open_project(self) -> None:
        """Start tracking edits in a folder opened before this view existed."""
        self._on_project_opened(self._controller.project.path)

    def _on_project_opened(self, _path: str) -> None:
        self._edited.clear()
        self._seen = {name: _edit_keys(self._controller.entry(name)) for name in self._controller.names()}

    def _on_project_closed(self) -> None:
        self._edited.clear()
        self._seen.clear()
        self.set_file(None)

    def _on_files_changed(self) -> None:
        names = set(self._controller.names())
        self._seen = {name: keys for name, keys in self._seen.items() if name in names}
        self._edited = {name: kind for name, kind in self._edited.items() if name in names}
        for name in names - set(self._seen):
            self._seen[name] = _edit_keys(self._controller.entry(name))
        self.refresh()

    def _on_file_changed(self, name: str) -> None:
        if name not in self._controller.names():
            return
        entry = self._controller.entry(name)
        keys = _edit_keys(entry)
        before = self._seen.get(name, keys)
        for kind in ("crop", "brightness"):
            if keys[kind] != before[kind] and _is_manual(entry, kind):
                self._edited[name] = kind
        self._seen[name] = keys
        if name == self._file:
            self.refresh()
        elif self._file in self._edited:
            self._refresh_offer()                      # another file's change can change the count

    def _refresh_offer(self) -> None:
        name = self._file
        kind = self._edited.get(name) if name is not None else None
        if kind is None:
            self.offer.hide()
            return
        self.offer.set_targets(len(self._controller.hint_targets(name, kind)))
        self.offer.show()

    def _redetect_others(self) -> None:
        name = self._file
        kind = self._edited.pop(name, None) if self._has_file() else None
        if kind is not None:
            self._controller.redetect_others_with_hint(name, kind)
        self._refresh_offer()

    def _dismiss_offer(self) -> None:
        if self._file is not None:
            self._edited.pop(self._file, None)
        self._refresh_offer()
