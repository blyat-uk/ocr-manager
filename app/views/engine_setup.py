"""The first-run "Set up the OCR engine" dialog (portable bundles only).

A bundle ships without paddle; this dialog installs it once, for this
computer. It shows the detected hardware and the offers `app.engine.
EngineSetup` makes -- the recommended GPU build when one fits (with its
download size), and the CPU build -- then runs the install off the GUI
thread and shows its current step, the downloaded bytes and, on request,
pip's full output. When a GPU build does not work the installer falls back
to the CPU build, and the dialog says so and why.

States: choose -> installing -> done | failed | cancelled (back to choose).
Closing the dialog while it installs cancels the install first. The dialog
is accepted only after an engine installed; `exec()` returning Rejected
means the user left without one (the app then quits unless an older engine
is still installed).

The service is duck-typed (signals step/progress/log/notice/finished and
hardware_text/offers/recommendation_reason/installed_text/start/cancel/
running): tests drive the dialog with a fake that installs nothing.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from app.theme import tokens
from app.widgets.base import Button, KvRow, SectionHeader, repolish

TITLE = "Set up the OCR engine"
SUBTITLE = ("OCR Manager reads subtitles with PaddleOCR. Its engine is downloaded once, "
            "for this computer's hardware.")
INSTALL_TEXT, RETRY_TEXT, CONTINUE_TEXT = "Install", "Try again", "Continue"
QUIT_TEXT, CANCEL_TEXT, CANCELLING_TEXT, CLOSE_TEXT = "Quit", "Cancel", "Cancelling…", "Close"
DETAILS_SHOW, DETAILS_HIDE = "▸ Details", "▾ Details"
PROGRESS_STEPS = 1000
MAX_LOG_LINES = 5000
CPU_BLURB = "works on every computer; slower"
GPU_BLURB = "fastest; uses the NVIDIA GPU"


def format_bytes(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1e9:.2f} GB"
    return f"{value / 1e6:.0f} MB"


class EngineSetupDialog(QDialog):
    def __init__(self, setup, parent: QWidget | None = None, *, have_engine: bool = False):
        """`have_engine`: an engine is already installed (a reinstall), so
        leaving the dialog keeps it and the button says Close, not Quit."""
        super().__init__(parent)
        self._setup = setup
        self._have_engine = have_engine
        self._state = "choose"
        self._outcome = None
        self._selected = None
        self.setObjectName("EngineSetup")
        self.setWindowTitle(TITLE)
        self.setModal(True)
        self.setMinimumWidth(tokens.px(560))

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        head = QWidget()
        head.setObjectName("EngineSetupHead")
        head.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        head_layout = QVBoxLayout(head)
        head_layout.setContentsMargins(tokens.px(18), tokens.px(14), tokens.px(18), tokens.px(12))
        head_layout.setSpacing(tokens.px(4))
        self.title_label = QLabel(TITLE)
        self.title_label.setObjectName("EngineSetupTitle")
        self.subtitle_label = QLabel(SUBTITLE)
        self.subtitle_label.setObjectName("EngineSetupScope")
        self.subtitle_label.setWordWrap(True)
        head_layout.addWidget(self.title_label)
        head_layout.addWidget(self.subtitle_label)
        column.addWidget(head)

        body = QWidget()
        body.setObjectName("EngineSetupBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(tokens.px(18), tokens.px(14), tokens.px(18), tokens.px(14))
        layout.setSpacing(tokens.px(8))
        self.hardware_row = KvRow("Hardware", setup.hardware_text())
        layout.addWidget(self.hardware_row)
        installed = setup.installed_text()
        self.installed_row = KvRow("Installed", installed or "nothing yet", "ok" if installed else None)
        layout.addWidget(self.installed_row)

        layout.addSpacing(tokens.px(4))
        layout.addWidget(SectionHeader("Engine"))
        self.option_buttons: dict[str, Button] = {}
        for offer in setup.offers():
            blurb = CPU_BLURB if offer.variant == "cpu" else GPU_BLURB
            head_text = offer.title + ("  ·  recommended" if offer.recommended else "")
            button = Button(f"{head_text}\n{offer.size_text} — {blurb}")
            button.setObjectName("EngineOption")
            button.setCheckable(False)
            button.clicked.connect(lambda _checked=False, v=offer.variant: self.select(v))
            self.option_buttons[offer.variant] = button
            layout.addWidget(button)
            if offer.recommended and self._selected is None:
                self._selected = offer.variant
        if self._selected is None and self.option_buttons:
            self._selected = next(iter(self.option_buttons))
        self.reason_label = QLabel(setup.recommendation_reason())
        self.reason_label.setObjectName("EngineSetupNote")
        self.reason_label.setWordWrap(True)
        layout.addWidget(self.reason_label)

        self.progress_area = QWidget()
        progress = QVBoxLayout(self.progress_area)
        progress.setContentsMargins(0, tokens.px(6), 0, 0)
        progress.setSpacing(tokens.px(5))
        self.step_label = QLabel()
        self.step_label.setObjectName("EngineSetupStep")
        self.step_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("EngineProgress")
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, PROGRESS_STEPS)
        self.progress_bar.setFixedHeight(tokens.px(6))
        self.bytes_label = QLabel()
        self.bytes_label.setObjectName("EngineSetupNote")
        progress.addWidget(self.step_label)
        progress.addWidget(self.progress_bar)
        progress.addWidget(self.bytes_label)
        self.progress_area.hide()
        layout.addWidget(self.progress_area)

        self.notice_label = QLabel()
        self.notice_label.setObjectName("EngineSetupMessage")
        self.notice_label.setWordWrap(True)
        self.notice_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.notice_label.hide()
        layout.addWidget(self.notice_label)

        self.details_button = Button(DETAILS_SHOW, "ghost", small=True)
        self.details_button.setObjectName("LogHeader")
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self._toggle_details)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("LogBody")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(MAX_LOG_LINES)
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.log_view.setMinimumHeight(tokens.px(160))
        self.log_view.hide()
        self.details_button.hide()                          # nothing to show before an install
        layout.addWidget(self.details_button, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.log_view, 1)
        column.addWidget(body, 1)

        foot = QWidget()
        foot.setObjectName("EngineSetupFoot")
        foot.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        buttons = QHBoxLayout(foot)
        buttons.setContentsMargins(tokens.px(18), tokens.px(10), tokens.px(18), tokens.px(10))
        buttons.setSpacing(tokens.px(8))
        buttons.addStretch(1)
        self.cancel_button = Button(CLOSE_TEXT if have_engine else QUIT_TEXT, "ghost")
        self.cancel_button.clicked.connect(self._cancel_clicked)
        self.install_button = Button(INSTALL_TEXT, "primary")
        self.install_button.clicked.connect(self._install_clicked)
        self.install_button.setDefault(True)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.install_button)
        column.addWidget(foot)

        setup.step.connect(self._on_step)
        setup.progress.connect(self._on_progress)
        setup.log.connect(self._on_log)
        setup.notice.connect(self._on_notice)
        setup.finished.connect(self._on_finished)
        self._paint_selection()

    # -- state ------------------------------------------------------------

    def state(self) -> str:
        return self._state

    def selected(self) -> str | None:
        return self._selected

    def outcome(self):
        return self._outcome

    def select(self, variant: str) -> None:
        if self._state in ("installing", "done") or variant not in self.option_buttons:
            return
        self._selected = variant
        self._paint_selection()

    def _paint_selection(self) -> None:
        for variant, button in self.option_buttons.items():
            button.set_toggled(variant == self._selected)

    def _set_state(self, state: str) -> None:
        self._state = state
        installing = state == "installing"
        for button in self.option_buttons.values():
            button.setEnabled(not installing and state != "done")
        self.install_button.setVisible(True)
        self.install_button.setEnabled(not installing)
        if state == "done":
            self.install_button.setText(CONTINUE_TEXT)
            self.cancel_button.hide()
        elif state == "failed":
            self.install_button.setText(RETRY_TEXT)
        else:
            self.install_button.setText(INSTALL_TEXT)
        if installing:
            self.cancel_button.setText(CANCEL_TEXT)
            self.cancel_button.setEnabled(True)
        elif state != "done":
            self.cancel_button.setText(CLOSE_TEXT if self._have_engine else QUIT_TEXT)
            self.cancel_button.setEnabled(True)
            self.cancel_button.show()

    def _message(self, text: str, tone: str) -> None:
        self.notice_label.setText(text)
        self.notice_label.setProperty("tone", tone)
        repolish(self.notice_label)
        self.notice_label.setVisible(bool(text))

    # -- buttons ----------------------------------------------------------

    def _install_clicked(self) -> None:
        if self._state == "done":
            self.accept()
            return
        if self._state == "installing" or self._selected is None:
            return
        self._message("", "warn")
        self.log_view.clear()
        self.details_button.show()
        self.progress_area.show()
        self.step_label.setText("Starting")
        self.bytes_label.setText("")
        self.progress_bar.setRange(0, 0)
        self._set_state("installing")
        self._setup.start(self._selected)

    def _cancel_clicked(self) -> None:
        if self._state == "installing":
            self.cancel_button.setText(CANCELLING_TEXT)
            self.cancel_button.setEnabled(False)
            self._setup.cancel()
            return
        self.reject()

    def _toggle_details(self, shown: bool) -> None:
        self.log_view.setVisible(shown)
        self.details_button.setText(DETAILS_HIDE if shown else DETAILS_SHOW)
        self._fit_height()

    def _fit_height(self) -> None:
        """Shrink back when the log or a message goes away (a hidden widget's
        room would otherwise be shared out among the rows). Deferred: the
        nested layouts only drop a hidden widget on their next pass."""
        QTimer.singleShot(0, self._fit_height_now)

    def _fit_height_now(self) -> None:
        layout = self.layout()
        layout.invalidate()
        layout.activate()
        self.setMinimumHeight(layout.totalMinimumSize().height())     # the log's minimum is gone
        self.resize(self.width(), self.sizeHint().height())

    # -- installer events -----------------------------------------------------

    def _on_step(self, text: str, done: int, total: int) -> None:
        self.step_label.setText(text)
        self._on_progress(done, total)

    def _on_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self.progress_bar.setRange(0, 0)                 # busy: no byte count for this step
            self.bytes_label.setText("")
            return
        self.progress_bar.setRange(0, PROGRESS_STEPS)
        self.progress_bar.setValue(min(PROGRESS_STEPS, int(PROGRESS_STEPS * done / total)))
        self.bytes_label.setText(f"{format_bytes(done)} of about {format_bytes(total)}")

    def _on_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    def _on_notice(self, text: str) -> None:
        self._message(text, "warn")

    def _on_finished(self, outcome) -> None:
        self._outcome = outcome
        self.progress_bar.setRange(0, PROGRESS_STEPS)
        if outcome.ok:
            self.progress_bar.setValue(PROGRESS_STEPS)
            self.bytes_label.setText("")
            self.step_label.setText(f"Installed: {outcome.title}")
            for button in self.option_buttons.values():
                button.set_toggled(False)                   # what was picked may not be what installed
            notes = []
            if outcome.fell_back:
                notes.append(f"The GPU build did not work on this computer, so the CPU build was installed "
                             f"instead. OCR will be slower.\nReason: {outcome.fallback_reason}")
            notes.extend(outcome.warnings)
            self._message("\n\n".join(notes), "warn")
            self._set_state("done")
        elif outcome.cancelled:
            self.progress_area.hide()
            self._message("Cancelled. Nothing was changed.", "warn")
            self._set_state("cancelled")
            self._fit_height()
        else:
            self.progress_bar.setValue(0)
            self.bytes_label.setText("")
            self.step_label.setText("The install failed")
            self._message(outcome.error + ("\n\nDetails has pip's full output." if outcome.error else ""), "bad")
            self._set_state("failed")

    # -- closing ------------------------------------------------------------

    def reject(self) -> None:
        """Esc / the window's close button: an install is cancelled first,
        and the dialog closes when it has stopped."""
        if self._state == "installing":
            self._cancel_clicked()
            return
        super().reject()

    def closeEvent(self, event) -> None:
        if self._state == "installing":
            event.ignore()
            self._cancel_clicked()
            return
        super().closeEvent(event)
