"""The file inspector (`.insp`, ruling B4, ui-spec §3.7): one
`tokens.INSPECTOR_WIDTH` column (the mockup's 322 px at the current UI
scale) about the selected episode only, scrolling vertically. Top to
bottom:

1. header -- "◆ THIS EPISODE ONLY", the file name, its media line;
2. the active stage tab's `inspector_panel()`, swapped on `tab_changed`;
3. "DETECTED" -- crop, brightness and OCR window with confidence bars;
4. "PROOF · REAL OCR OF 30 S" (ruling C4);
5. "IF YOU CHANGE SOMETHING HERE" -- after a manual crop or brightness edit
   in this session, the offer to re-detect the other files with it as a
   hint (`controller.hint_targets` counts them, ruling C3);
6. footer (pinned below the scroll area) -- "✓ Mark reviewed (Space)" /
   "Mark not reviewed" (disabled while the file is PENDING) and "skip file" /
   "include file".

A manual edit is noticed from `file_changed`: the file's crop box or
brightness value differs from the last one seen and is now MANUAL. So an
edit from anywhere (the tabs, the queue's paste) raises the offer for that
file, while accepting a flagged value (same value, MANUAL) does not. Each
edited kind keeps its own button; taking one offer leaves the other
standing, and "apply to this file only" puts the whole offer away until the
next edit.

Proof results live in the controller for the session. Whether one still
describes its file is this view's business: the crop, brightness and time
ranges the file had when the proof was asked for are remembered here
(`_proof_keys`) and compared on every `file_changed`. Evidence is never
consulted -- a detection that only wrote evidence changed nothing OCR would
see, and a proof must not go stale because of it.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from app.state_text import (
    PROOF_WAIT_TOOLTIP,
    can_mark_reviewed,
    can_run_proof,
    is_manual,
    is_reviewed,
    mark_reviewed_tooltip,
    media_text,
    proof_window_clock,
    series_median_brightness,
    series_median_note,
)
from app.theme import tokens
from app.views.inspector_sections import HINT_KINDS, ChangeOffer, DetectedSection, ProofSection, Section
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


def _proof_key(entry) -> tuple:
    """Everything a proof OCR of `entry` depends on per file: its crop, its
    brightness and its keep ranges (the folder's own settings are not this
    view's to watch -- a folder change rebuilds every section anyway)."""
    keys = _edit_keys(entry)
    ranges = entry.time_ranges
    return (keys["crop"], keys["brightness"],
            None if ranges is None else tuple((item.start, item.end) for item in ranges.ranges))


class Inspector(QWidget):
    tab_requested = pyqtSignal(str)             # a stage tab title

    def __init__(self, controller, stage, parent: QWidget | None = None):
        super().__init__(parent)
        self._controller = controller
        self._stage = stage
        self._file: str | None = None
        self._seen: dict[str, dict[str, object]] = {}       # file -> last crop box / brightness seen
        self._edited: dict[str, set[str]] = {}              # file -> kinds it was manually edited in
        self._proof_keys: dict[str, tuple] = {}             # file -> settings its proof was asked for with
        self._stale_proofs: set[str] = set()                # files whose proof no longer matches them
        self._redetecting: dict[tuple[str, str], list[str]] = {}   # (source, kind) -> files still re-detecting
        self.setObjectName("Inspector")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(tokens.INSPECTOR_WIDTH)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 0, 0, 0)                 # room for the stylesheet's 1 px left border
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
        head.body.addSpacing(tokens.px(3))
        head.body.addWidget(self.file_label)
        head.body.addSpacing(tokens.px(2))
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
        footer_layout.setContentsMargins(tokens.px(12), tokens.px(10), tokens.px(12), tokens.px(10))
        footer_layout.setSpacing(tokens.px(6))
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
        self.detected_note = self.detected.note_label
        self.proof_button, self.proof_status, self.proof_note = (
            self.proof.run_button, self.proof.status_label, self.proof.note_label)
        self.proof_show_all = self.proof.show_all_button
        self.offer_section, self.offer_note = self.offer, self.offer.note_label
        self.offer_status = self.offer.status_label
        self.hint_buttons, self.this_file_only_button = self.offer.hint_buttons, self.offer.this_file_only_button
        self.proof_texts = self.proof.texts

        stage.tab_changed.connect(self._show_panel)
        self._show_panel(stage.current_index())
        controller.project_opened.connect(self._on_project_opened)
        controller.project_closed.connect(self._on_project_closed)
        controller.files_changed.connect(self._on_files_changed)
        controller.file_changed.connect(self._on_file_changed)
        controller.folder_changed.connect(self.refresh)
        controller.proof_started.connect(self._on_proof_started)
        controller.proof_finished.connect(self._on_proof_event)
        controller.activity_changed.connect(self._on_activity_changed)
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
        self.detected.set_entry(entry, self._median_note(entry))
        reviewed = is_reviewed(entry)
        reviewable = can_mark_reviewed(entry)
        self.review_button.setText(UNREVIEW_TEXT if reviewed else REVIEW_TEXT)
        self.review_button.set_variant("default" if reviewed else "primary")
        self.review_button.setEnabled(reviewable)
        # Not one sentence: "Mark reviewed" is refused either because the
        # detections are still running or because a value the file needs is
        # missing, and only the first is a wait (`mark_reviewed_tooltip`).
        self.review_button.setToolTip(mark_reviewed_tooltip(entry))
        self.skip_button.setText("include file" if entry.skipped else "skip file")
        self._refresh_offer()

    # --- commands -----------------------------------------------------------------------------

    def run_proof_for(self, name: str | None) -> None:
        """Real OCR of 30 s of `name` -- the one command behind the "T run"
        button, the T key and the queue's "Test OCR (T)" (ruling 5). Ignored
        while that file's proof is already running: a second run would only
        queue the same window again."""
        if name is None or name not in self._controller.names() or self._controller.proof_pending(name):
            return
        try:
            self._controller.run_proof(name)
        except ValueError as exc:                      # the duration is not known yet
            if name == self._file:
                self.proof.show_error(f"Can't run yet: {exc}")

    def _median_note(self, entry) -> str:
        """The series-median sentence for the Detected note -- only for a file
        that has a brightness of its own to keep."""
        if entry.brightness is None:
            return ""
        controller = self._controller
        return series_median_note(series_median_brightness(controller.entry(name) for name in controller.names()))

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

    def _on_proof_started(self, name: str) -> None:
        """Remember what the file looked like when its proof was asked for:
        ProofOcrJob froze the same values at construction, so a later edit
        makes whatever comes back stale."""
        if name in self._controller.names():
            self._proof_keys[name] = _proof_key(self._controller.entry(name))
            self._stale_proofs.discard(name)
        self._on_proof_event(name)

    def _on_proof_event(self, name: str) -> None:
        if name == self._file:
            self._show_proof()

    def _show_proof(self) -> None:
        name = self._file
        entry = None if name is None or name not in self._controller.names() \
            else self._controller.entry(name)
        # Ruling C4: the button, T and the queue's menu item are one command,
        # so they are one predicate. Set before the presentation, which reads
        # it back through `_reset`.
        self.proof.set_runnable(entry is not None and can_run_proof(entry), PROOF_WAIT_TOOLTIP)
        if entry is None:
            self.proof.show_nothing()
        elif self._controller.proof_pending(name):
            self.proof.show_running(proof_window_clock(entry.sample_time, entry.media.duration))
        elif (result := self._controller.proof_result(name)) is not None:
            self.proof.show_result(result, stale=name in self._stale_proofs)
        else:
            self.proof.show_nothing()

    # --- what this session knows about the folder --------------------------------------

    def adopt_open_project(self) -> None:
        """Start tracking edits in a folder opened before this view existed."""
        self._on_project_opened(self._controller.project.path)

    def _on_project_opened(self, _path: str) -> None:
        self._forget_session()
        self._seen = {name: _edit_keys(self._controller.entry(name)) for name in self._controller.names()}

    def _on_project_closed(self) -> None:
        self._forget_session()
        self._seen.clear()
        self.set_file(None)

    def _forget_session(self) -> None:
        """Edits, proof staleness and re-detects belong to one open folder."""
        self._edited.clear()
        self._proof_keys.clear()
        self._stale_proofs.clear()
        self._redetecting.clear()

    def _on_files_changed(self) -> None:
        names = set(self._controller.names())
        self._seen = {name: keys for name, keys in self._seen.items() if name in names}
        self._edited = {name: kinds for name, kinds in self._edited.items() if name in names}
        self._proof_keys = {name: key for name, key in self._proof_keys.items() if name in names}
        self._stale_proofs &= names
        redetecting = {}
        for key, targets in self._redetecting.items():     # a vanished source or target ends its wait
            kept = [target for target in targets if target in names]
            if key[0] in names and kept:
                redetecting[key] = kept
        self._redetecting = redetecting
        for name in names - set(self._seen):
            self._seen[name] = _edit_keys(self._controller.entry(name))
        self.refresh()

    def _on_file_changed(self, name: str) -> None:
        if name not in self._controller.names():
            return
        entry = self._controller.entry(name)
        keys = _edit_keys(entry)
        before = self._seen.get(name, keys)
        for kind in HINT_KINDS:
            if keys[kind] != before[kind] and _is_manual(entry, kind):
                self._edited.setdefault(name, set()).add(kind)
        self._seen[name] = keys
        if name in self._proof_keys and _proof_key(entry) != self._proof_keys[name]:
            self._stale_proofs.add(name)               # the proof ran on settings the file no longer has
        if name == self._file:
            self.refresh()
            self._show_proof()
        elif self._file in self._edited:
            self._refresh_offer()                      # another file's change can change the count

    # --- the change offer -----------------------------------------------------------------

    def _on_activity_changed(self) -> None:
        """Auto-pilot's queue moved: a hint re-detect may be over."""
        if self._redetecting:
            self._refresh_offer()

    def _prune_redetecting(self) -> None:
        """A hint re-detect is over (ruling C3's "re-detecting N files…") once
        none of the files it covers has a detection of that kind still to
        come. "To come" is auto-pilot's own pending map, not the runner's
        queue: a brightness job waits for the folder's ranges analysis and has
        no job of its own until that ends, and a queued job has not started."""
        if not self._redetecting:
            return
        pending = self._controller.pending_detectors()
        self._redetecting = {key: targets for key, targets in self._redetecting.items()
                             if any(key[1] in pending.get(name, ()) for name in targets)}

    def _refresh_offer(self) -> None:
        self._prune_redetecting()
        name = self._file
        kinds = self._edited.get(name, set()) if name is not None else set()
        self.offer.set_targets({kind: len(self._controller.hint_targets(name, kind))
                                for kind in HINT_KINDS if kind in kinds})
        # "re-detecting {n} files…" counts distinct FILES, not jobs: a file
        # covered by both this file's crop and brightness hints counts once.
        targets = {target for (source, _kind), files in self._redetecting.items()
                   if source == name for target in files}
        self.offer.set_redetecting(len(targets))
        self.offer.setVisible(not self.offer.is_empty())

    def _redetect_others(self, kind: str) -> None:
        name = self._file
        if self._has_file() and kind in self._edited.get(name, set()):
            self._edited[name].discard(kind)
            if not self._edited[name]:
                del self._edited[name]
            targets = self._controller.hint_targets(name, kind)
            self._controller.redetect_others_with_hint(name, kind)
            if targets:
                self._redetecting[(name, kind)] = targets
        self._refresh_offer()

    def _dismiss_offer(self) -> None:
        """"apply to this file only": the edit stands, nothing else is
        re-detected, and the offer waits for the next edit of that file."""
        if self._file is not None:
            self._edited.pop(self._file, None)
        self._refresh_offer()
