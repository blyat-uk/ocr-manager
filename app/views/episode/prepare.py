"""The Preparing screen (episode-view-v2.html, screens 1a/1b/1c; copy set
"B"): what the window shows the moment an episode opens.

Nothing new runs here. The checklist reads the file's entry and what
AutoPilot still has to do for it (`pending_detectors`, `running_detectors`)
and ticks off as the ordinary detection jobs finish:

    Read the video                  media duration > 0            (metadata)
    Listened for speech             evidence["audio"]             (audio_profile)
    Finding the subtitle area       a crop that is not pending    (crop)
    Measuring subtitle brightness   a brightness value, or labels-only  (brightness)

An item whose job is running shows a spinner; one still to come, a hollow
circle; one that ended without a value (a video with no audio track), a
dash, so the list never hangs on a job that is not coming.

Once the review state settles, the screen says so:

- all green (PROPOSED or REVIEWED, not skipped): a real frame with the box
  drawn in green, "Your episode is good to go." and the time the run should
  take on this computer (left out while no speed is remembered). Start OCR
  is the primary button, and Enter presses it.
- needs a look (FLAGGED, or skipped): the box dashed amber, "Nearly good to
  go. Take a quick look first." and the reason. Review is the primary
  button; Start OCR is disabled, since a flagged file is never startable.

Start OCR is enabled exactly when the file is in
`startable_files(include_done=True)` -- whatever the screen says, so an
episode whose only unfinished item is the speech profile can already start.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from app.episode_feed import estimate_words
from app.imaging import bgr_to_qimage
from app.state_text import (
    brightness_flag_text,
    crop_flag_summary,
    field_blocking,
    missing_required_values,
)
from app.theme import tokens
from app.views.deferred import Deferred
from app.views.episode.common import (
    BODY_SIZE,
    HEADLINE_SIZE,
    SMALL_SIZE,
    FillBar,
    episode_label,
    text_label,
)
from app.views.episode.slideshow import FramePreview
from app.widgets.base import Button

HEADLINE_WORKING = "Looking at your episode…"
HEADLINE_READY = "Your episode is good to go."
HEADLINE_FLAGGED = "Nearly good to go. Take a quick look first."
READY_DIALOGUE = "Found the subtitles and tuned the brightness"
READY_LABELS = "Ready to read the on-screen labels"
ESTIMATE = "{estimate} on this computer"
SEED_NOTE = "Starting from {episode}'s settings"
REVIEW = "Review"
START = "▶ Start OCR"

# (key, label, the detection kind that produces it)
CHECKLIST = (
    ("media", "Read the video", "metadata"),
    ("audio", "Listened for speech", "audio_profile"),
    ("crop", "Finding the subtitle area", "crop"),
    ("brightness", "Measuring subtitle brightness", "brightness"),
)
DONE, ACTIVE, WAITING, MISSING = "done", "active", "waiting", "missing"
MARKS = {DONE: "✓", ACTIVE: "◌", WAITING: "○", MISSING: "–"}
MARK_COLOURS = {DONE: tokens.OK, ACTIVE: tokens.ACC, WAITING: tokens.DIM2, MISSING: tokens.DIM2}
SPIN_FRAMES = ("◜", "◝", "◞", "◟")
SPIN_MS = 150
PREVIEW_RESERVED_HEIGHT = 260      # mockup px under the preview: headline, sublines, buttons, margins

MODE_WORKING, MODE_READY, MODE_FLAGGED = "working", "ready", "flagged"


# --------------------------------------------------------------------------
# Pure rules (the entry is read, never changed)
# --------------------------------------------------------------------------

def checklist_states(entry, pending: set[str], running: set[str], *, labels_only: bool) -> list[tuple[str, str]]:
    """[(label, state)] in CHECKLIST order; state is DONE | ACTIVE | WAITING | MISSING."""
    has = {
        "media": entry.media.duration > 0,
        "audio": bool(entry.evidence.get("audio")),
        "crop": "crop" not in pending and (labels_only or entry.crop is not None),
        "brightness": labels_only or (entry.brightness is not None and "brightness" not in pending),
    }
    settled = entry.review != "pending"
    states = []
    for key, label, kind in CHECKLIST:
        if has[key]:
            state = DONE
        elif kind in running:
            state = ACTIVE
        elif kind in pending or not settled:
            state = WAITING
        else:
            state = MISSING
        states.append((label, state))
    return states


def screen_mode(entry, states: list[tuple[str, str]]) -> str:
    """MODE_WORKING until detection has settled and every item is ticked
    (or has nothing coming); then MODE_FLAGGED or MODE_READY."""
    if entry.review == "pending" or any(state in (ACTIVE, WAITING) for _label, state in states):
        return MODE_WORKING
    if entry.review == "flagged" or entry.skipped:
        return MODE_FLAGGED
    return MODE_READY


def flag_reasons(entry) -> list[str]:
    """Why a flagged episode needs a look, one line a field, in review copy."""
    reasons = []
    if entry.skipped:
        reasons.append("This episode is marked skipped")
    missing = missing_required_values(entry)
    if "crop" in missing:
        reasons.append("Subtitle area: none found yet")
    elif field_blocking(entry, "crop"):
        summary = crop_flag_summary(entry.flags.get("crop"))
        reasons.append(f"Subtitle area: {summary[0] if summary else 'needs a look'}")
    if "brightness" in missing and "crop" not in missing:
        reasons.append("Subtitle brightness: not measured")
    elif field_blocking(entry, "brightness"):
        reasons.append(f"Subtitle brightness: {brightness_flag_text(entry.flags.get('brightness')) or 'needs a look'}")
    if field_blocking(entry, "ranges"):
        reasons.append("Time ranges: need a look")
    return reasons or ["Something needs a look"]


def flagged_field(entry) -> str | None:
    """The Stage tab a flagged episode's review should open on: "crop",
    "brightness" or "ranges"; None when nothing is flagged."""
    if entry.review != "flagged":
        return None
    missing = missing_required_values(entry)
    for field in ("crop", "brightness", "ranges"):
        if field in missing or field_blocking(entry, field):
            return field
    return None


def preview_time(entry) -> float | None:
    """The frame the Preparing screen shows: the crop's first hit, else the
    middle of the video; None while the duration is unknown."""
    if entry.sample_time is not None:
        return float(entry.sample_time)
    if entry.media.duration > 0:
        return entry.media.duration / 2
    return None


# --------------------------------------------------------------------------
# The view
# --------------------------------------------------------------------------

class CheckItem(QWidget):
    def __init__(self, label: str, parent: QWidget | None = None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(tokens.px(8))
        self.mark = text_label(MARKS[WAITING], color=tokens.DIM2, size=BODY_SIZE)
        self.mark.setFixedWidth(tokens.px(14))
        self.text = text_label(label, color=tokens.DIM2, size=BODY_SIZE)
        row.addWidget(self.mark)
        row.addWidget(self.text, 1)
        self.state = WAITING

    def set_state(self, state: str, spin_frame: int = 0) -> None:
        mark = SPIN_FRAMES[spin_frame % len(SPIN_FRAMES)] if state == ACTIVE else MARKS[state]
        if (state, mark) == (self.state, self.mark.text()):
            return
        self.state = state
        self.mark.setText(mark)
        self.mark.setStyleSheet(f"color: {MARK_COLOURS[state]}; background: transparent;"
                                f" font-size: {round(tokens.pt(BODY_SIZE))}px;")
        colour = tokens.DIM2 if state in (WAITING, MISSING) else tokens.TXT
        self.text.setStyleSheet(f"color: {colour}; background: transparent;"
                                f" font-size: {round(tokens.pt(BODY_SIZE))}px;")


class PrepareView(QWidget):
    review_requested = pyqtSignal()
    start_requested = pyqtSignal()

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__(parent)
        self._controller = controller
        self._name: str | None = None
        self._mode = MODE_WORKING
        self._frame_time: float | None = None
        self._spin = 0
        self.setObjectName("EpisodePrepare")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(tokens.px(24), tokens.px(24), tokens.px(24), tokens.px(24))
        outer.addStretch(1)
        column = QVBoxLayout()
        column.setSpacing(tokens.px(10))
        column.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.preview = FramePreview(width=520)
        column.addWidget(self.preview, 0, Qt.AlignmentFlag.AlignHCenter)
        self.headline = text_label(HEADLINE_WORKING, size=HEADLINE_SIZE, weight=600, center=True, wrap=True)
        column.addWidget(self.headline)

        self.progress = FillBar()
        column.addWidget(self.progress, 0, Qt.AlignmentFlag.AlignHCenter)
        self.checklist_box = QWidget()
        checks = QVBoxLayout(self.checklist_box)
        checks.setContentsMargins(0, tokens.px(2), 0, 0)
        checks.setSpacing(tokens.px(5))
        self.items = [CheckItem(label) for _key, label, _kind in CHECKLIST]
        for item in self.items:
            checks.addWidget(item)
        column.addWidget(self.checklist_box, 0, Qt.AlignmentFlag.AlignHCenter)

        self.subline = text_label(color=tokens.DIM, size=BODY_SIZE, center=True, wrap=True)
        self.reason = text_label(color=tokens.WARN, size=BODY_SIZE, center=True, wrap=True)
        self.seed_note = text_label(color=tokens.DIM2, size=SMALL_SIZE, center=True, wrap=True)
        for label in (self.subline, self.reason, self.seed_note):
            column.addWidget(label)

        buttons = QHBoxLayout()
        buttons.setSpacing(tokens.px(8))
        buttons.addStretch(1)
        self.review_button = Button(REVIEW)
        self.start_button = Button(START, "primary")
        self.review_button.clicked.connect(self.review_requested)
        self.start_button.clicked.connect(self.start_requested)
        buttons.addWidget(self.review_button)
        buttons.addWidget(self.start_button)
        buttons.addStretch(1)
        column.addSpacing(tokens.px(4))
        column.addLayout(buttons)

        outer.addLayout(column)
        outer.addStretch(1)

        self._spinner = QTimer(self)
        self._spinner.setInterval(SPIN_MS)
        self._spinner.timeout.connect(self._spin_once)
        self._refresh_later = Deferred(self.refresh, self)
        for signal in (controller.files_changed, controller.activity_changed, controller.run_changed):
            signal.connect(self._refresh_later.schedule)
        controller.file_changed.connect(self._on_file_changed)
        controller.frame_ready.connect(self._on_frame_ready)
        if hasattr(controller, "folder_changed"):
            controller.folder_changed.connect(self._refresh_later.schedule)
        self.refresh()

    # --- reading (window, tests) --------------------------------------------------------

    def file(self) -> str | None:
        return self._name

    def mode(self) -> str:
        return self._mode

    def checklist(self) -> list[tuple[str, str]]:
        return [(item.text.text(), item.state) for item in self.items]

    def flagged_field(self) -> str | None:
        entry = self._entry()
        return None if entry is None else flagged_field(entry)

    def primary_button(self) -> Button:
        return self.review_button if self._mode == MODE_FLAGGED else self.start_button

    # --- refreshing ---------------------------------------------------------------------

    def set_file(self, name: str | None) -> None:
        if name != self._name:
            self._name = name
            self._frame_time = None
            self.preview.set_image(None)
        self.refresh()

    def _entry(self):
        if self._name is None or self._controller.project is None:
            return None
        try:
            return self._controller.entry(self._name)
        except KeyError:
            return None

    def _labels_only(self) -> bool:
        project = self._controller.project
        return project is not None and not project.folder.dialogue_enabled

    def refresh(self, *_args) -> None:
        self._refresh_later.cancel()
        entry = self._entry()
        if entry is None:
            self._show_nothing()
            return
        name = self._name
        pending = self._controller.pending_detectors().get(name, set())
        running = self._controller.running_detectors(name)
        states = checklist_states(entry, pending, running, labels_only=self._labels_only())
        for item, (_label, state) in zip(self.items, states, strict=True):
            item.set_state(state, self._spin)
        self._mode = screen_mode(entry, states)
        working = self._mode == MODE_WORKING
        finished = sum(1 for _label, state in states if state in (DONE, MISSING))
        self.progress.set_fraction(finished / len(states))
        self.progress.setVisible(working)
        self.checklist_box.setVisible(working)
        if any(state == ACTIVE for _label, state in states):
            if not self._spinner.isActive():
                self._spinner.start()
        else:
            self._spinner.stop()

        self.headline.setText({MODE_WORKING: HEADLINE_WORKING, MODE_READY: HEADLINE_READY,
                               MODE_FLAGGED: HEADLINE_FLAGGED}[self._mode])
        self._refresh_subline(entry)
        self.reason.setText("⚠ " + " · ".join(flag_reasons(entry)) if self._mode == MODE_FLAGGED else "")
        self.reason.setVisible(self._mode == MODE_FLAGGED)
        seed = self._controller.episode_seed_source()
        self.seed_note.setText(SEED_NOTE.format(episode=episode_label(seed)) if seed else "")
        self.seed_note.setVisible(bool(seed))
        self._refresh_preview(entry)
        self._refresh_buttons(name)

    def _show_nothing(self) -> None:
        self._mode = MODE_WORKING
        self._spinner.stop()
        for item in self.items:
            item.set_state(WAITING)
        self.headline.setText(HEADLINE_WORKING)
        for label in (self.subline, self.reason, self.seed_note):
            label.setText("")
            label.setVisible(False)
        self.preview.setVisible(False)
        self.start_button.setEnabled(False)

    def _refresh_subline(self, entry) -> None:
        if self._mode != MODE_READY:
            self.subline.setText("")
            self.subline.setVisible(False)
            return
        parts = [READY_LABELS if self._labels_only() else READY_DIALOGUE]
        estimate = self._controller.episode_estimate_seconds()
        if estimate is not None and estimate > 0:
            parts.append(ESTIMATE.format(estimate=estimate_words(estimate)))
        self.subline.setText(" · ".join(parts))
        self.subline.setVisible(True)

    def _refresh_preview(self, entry) -> None:
        show = self._mode != MODE_WORKING
        self.preview.setVisible(show)
        if not show:
            return
        crop = entry.crop
        box = None if crop is None else (crop.x, crop.y, crop.width, crop.height)
        self.preview.set_box(box, (entry.media.width, entry.media.height),
                             "warn" if self._mode == MODE_FLAGGED else "ok")
        time_value = preview_time(entry)
        if time_value is None:
            return
        if time_value != self._frame_time:
            self._frame_time = time_value
            self.preview.set_image(None)
        if not self.preview.has_image():
            frame = self._controller.frame(self._name, time_value)
            if frame is not None:
                self.preview.set_image(bgr_to_qimage(frame))
            else:
                self._controller.request_frames(self._name, [time_value])

    def _refresh_buttons(self, name: str) -> None:
        startable = name in self._controller.startable_files(include_done=True)
        self.start_button.setEnabled(startable)
        flagged = self._mode == MODE_FLAGGED
        if self.review_button.property("variant") != ("primary" if flagged else "default"):
            self.review_button.set_variant("primary" if flagged else "default")
        if self.start_button.property("variant") != ("default" if flagged else "primary"):
            self.start_button.set_variant("default" if flagged else "primary")

    # --- events ---------------------------------------------------------------------------

    def _on_file_changed(self, name: str) -> None:
        if name == self._name:
            self._refresh_later.schedule()

    def _on_frame_ready(self, name: str, time_value: float) -> None:
        if name != self._name or self._frame_time is None or abs(time_value - self._frame_time) > 1e-6:
            return
        frame = self._controller.frame(name, self._frame_time)
        if frame is not None:
            self.preview.set_image(bgr_to_qimage(frame))

    def _spin_once(self) -> None:
        self._spin += 1
        for item in self.items:
            if item.state == ACTIVE:
                item.set_state(ACTIVE, self._spin)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            button = self.primary_button()
            if button.isEnabled() and not button.isHidden():
                button.click()
                return
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._spinner.stop()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Half the width, and whatever height the words and buttons leave.
        room = self.height() - tokens.px(PREVIEW_RESERVED_HEIGHT)
        self.preview.fit(min(self.width() // 2, int(room * FramePreview.ASPECT)))
