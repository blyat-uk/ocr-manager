"""The workbench window (plan 3B, ruling C10): top bar, banners, the review
queue / stage / inspector body (or the open-folder empty state), and the
activity strip.

Kept from today's window (current-app inventory): the folder picker,
QSettings("OCRManager", "OCRTool") "window/geometry", Ctrl+Q, and the
startup dependency check -- shown as a non-modal banner rather than a
message box, with today's texts. New: drag-and-drop of a folder, the
remembered last folder ("project/last_path"), and folders that cannot be
opened reported in place (the open folder stays open).

Space (mark reviewed / not reviewed) and T (test OCR) are window shortcuts
for the selected file. They act while focus is on the queue, a stage page or
a non-interactive area, and step aside while the focused widget uses keys
itself (buttons, check boxes, text and number inputs, combo boxes, sliders,
item views): Space then presses the focused button, and a settings sheet can
never flip a hidden file's review. ↑/↓ stay with the queue (plan 3C's crop
canvas nudges with the arrows).

`report_unexpected_error` is where `python -m app`'s excepthook sends an
exception raised in a slot: the Pipeline log and a dismissible banner.

"⚙ Folder settings" opens the sheet of plan 3B Task 4
(`open_folder_settings`). It belongs to reviewing, so it never covers the
Run view: opening it switches back to Review mode, and switching to Run
closes it. During a run the top bar hides its button anyway (B12), while
"⤓ Logs" stays.

Run (plan 3B Task 5, rulings B12, C5): "▶ Start" collects the startable
files, done ones included; when some already have `chi/` output, one Yes/No
question (default No, the window's only modal) decides whether they are
re-run -- nothing is deleted either way. A start the controller refuses
(e.g. two files writing the same output) is shown beside Start and the mode
stays. Otherwise the window switches to Run mode: the Run view replaces the
stage and inspector, the queue stays, and the top bar's Review · Run switch
goes back and forth. "⤓ Logs" and a queue row's "Open logs" open the
non-modal logs window.
"""
from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QAction, QGuiApplication, QKeySequence
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.controller import ProjectController, UnsupportedProjectVersion
from app.logbook import PIPELINE_LOG
from app.state_text import can_mark_reviewed, can_run_proof
from app.views.activity import ActivityStrip
from app.views.banner import Banner
from app.views.folder_settings import FolderSettingsSheet
from app.views.inspector import Inspector
from app.views.logs import LogsWindow
from app.views.open_folder import (
    FolderPicker,
    OpenFolderView,
    app_settings,
    folder_from_mime,
    last_path,
    remember_path,
)
from app.views.queue import QueueView
from app.views.run_view import RunView
from app.views.stage import Stage, StageTab, placeholder_tabs
from app.views.topbar import APP_NAME, TopBar

logger = logging.getLogger(__name__)

GEOMETRY_KEY = "window/geometry"
DEFAULT_SIZE = (1440, 900)
REQUIRED_TOOLS = ("ffmpeg",)
OPEN_FAILED_TITLE = "Could not open this folder"
SAVE_FAILED_TITLE = "Not saved"
CRASH_TITLE = "Something went wrong — details are in Logs (Pipeline)."
NEWER_VERSION_TEXT = ("This folder was saved by a newer version of OCR Manager (project version {version}). "
                      "Update the app to open it.")
EXPECTED_OPEN_ERRORS = (UnsupportedProjectVersion, OSError, ValueError)
MODE_REVIEW, MODE_RUN = 0, 1
OVERWRITE_TITLE = "Replace existing subtitles?"
OVERWRITE_TEXT = ("{n} file(s) already have subtitles in chi/. Re-run and replace them when their new output "
                  "is ready?")
NOTHING_TO_RUN = "Nothing to run — the files you picked already have subtitles."
RUN_STOP_TITLE = "A run is in progress"
RUN_OPEN_TEXT = "Stop it and open the other folder?"
RUN_QUIT_TEXT = "Stop it and quit?"
BLOCKED_SAVE_TITLE = "Settings were not saved"
BLOCKED_SAVE_TEXT = ("This folder's .ocr.json could not be written, so nothing you changed in this session is "
                     "on disk. Close anyway?")


def dependency_problems() -> list[tuple[str, str]]:
    """(title, text) per missing dependency, in today's words (main.py
    check_dependencies)."""
    problems = []
    missing = [tool for tool in REQUIRED_TOOLS if not shutil.which(tool)]
    if missing:
        problems.append(("Missing Dependencies",
                         f"Required tools not found in PATH:\n{', '.join(missing)}\n\n"
                         f"Please install them before using this application."))
    from videocr import pyav_adapter
    if not pyav_adapter.PYAV_AVAILABLE:
        problems.append(("Video backend degraded",
                         "PyAV is not available, so video will be decoded by a fallback "
                         "backend that is not bit-exact and estimates timestamps.\n\n"
                         "Fix it with:\n"
                         "    .venv/bin/pip install -U --only-binary=:all: av\n\n"
                         f"Import error: {pyav_adapter.PYAV_IMPORT_ERROR}"))
    return problems


KEY_CONSUMERS = (QAbstractButton, QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox,
                 QAbstractSlider, QAbstractItemView)


def consumes_keys(widget: QWidget | None) -> bool:
    """A focused widget that handles Space or letters itself: the Space and T
    shortcuts step aside for it."""
    return isinstance(widget, KEY_CONSUMERS)


def _open_error_text(path: str, exc: Exception) -> str:
    if isinstance(exc, UnsupportedProjectVersion):
        return NEWER_VERSION_TEXT.format(version=exc.version)
    if isinstance(exc, NotADirectoryError):
        return f"{path} is not a folder."
    return str(exc) or f"{type(exc).__name__} while opening {path}"


class MainWindow(QMainWindow):
    def __init__(self, controller: ProjectController | None = None,
                 tabs_factory: Callable[[ProjectController], list[StageTab]] = placeholder_tabs):
        super().__init__()
        self.controller = controller if controller is not None else ProjectController(parent=self)
        self._closing = False               # the close was confirmed: never ask its questions twice
        self.setWindowTitle(APP_NAME)
        self.setAcceptDrops(True)
        self._picker = FolderPicker(self)
        self._picker.chosen.connect(self.open_folder)

        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self.topbar = TopBar(self.controller)
        column.addWidget(self.topbar)
        self.dependency_banner = Banner()
        self.error_banner = Banner()
        self.crash_banner = Banner()
        for banner in (self.dependency_banner, self.error_banner, self.crash_banner):
            column.addWidget(banner)

        self.centre = QStackedWidget()
        self.open_view = OpenFolderView()
        self.queue = QueueView(self.controller)
        self.stage = Stage(self.controller, tabs_factory(self.controller))
        self.inspector = Inspector(self.controller, self.stage)
        self.run_view = RunView(self.controller)
        self.logs_window: LogsWindow | None = None
        workbench = QWidget()
        body = QHBoxLayout(workbench)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self.queue)
        self.review_area = QWidget()
        review = QHBoxLayout(self.review_area)
        review.setContentsMargins(0, 0, 0, 0)
        review.setSpacing(0)
        review.addWidget(self.stage, 1)
        review.addWidget(self.inspector)
        self.modes = QStackedWidget()                    # Review · Run: what sits right of the queue
        self.modes.addWidget(self.review_area)
        self.modes.addWidget(self.run_view)
        body.addWidget(self.modes, 1)
        self.workbench = workbench
        self.centre.addWidget(self.open_view)
        self.centre.addWidget(workbench)
        column.addWidget(self.centre, 1)
        self.activity_strip = ActivityStrip(self.controller)
        column.addWidget(self.activity_strip)
        self.setCentralWidget(central)
        self.folder_settings = FolderSettingsSheet(self.controller, central, anchor=self.centre)

        self._connect()
        self.quit_action = self._shortcut("Quit", "Ctrl+Q", self.close)
        self.open_action = self._shortcut("Open folder…", "Ctrl+O", self.choose_folder)
        self.review_action = self._shortcut("Mark reviewed", "Space", self.inspector.toggle_reviewed)
        self.proof_action = self._shortcut("Test OCR", "T", lambda: self.inspector.run_proof_for(self.queue.selected()))
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._sync_actions)
        self._restore_geometry()
        self._check_dependencies()
        if self.controller.project is not None:          # a controller that already has a folder open
            self.inspector.adopt_open_project()
            self.queue.rebuild()
            self._on_project_opened(self.controller.project.path)
        self._sync_actions()

    def _shortcut(self, text: str, keys: str, slot) -> QAction:
        action = QAction(text, self)
        action.setShortcut(QKeySequence(keys))
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        action.triggered.connect(lambda _checked=False: slot())
        self.addAction(action)
        return action

    def _connect(self) -> None:
        controller = self.controller
        controller.project_opened.connect(self._on_project_opened)
        controller.project_closed.connect(self._on_project_closed)
        controller.save_failed.connect(lambda message: self.error_banner.show_message(SAVE_FAILED_TITLE, message,
                                                                                      "bad"))
        controller.project_saved.connect(self._on_project_saved)
        for signal in (controller.project_opened, controller.project_closed, controller.files_changed,
                       controller.file_changed, controller.proof_started, controller.proof_finished):
            signal.connect(self._sync_actions)
        self.queue.selection_changed.connect(self._on_selection_changed)
        self.queue.proof_requested.connect(self.inspector.run_proof_for)
        self.queue.logs_requested.connect(self.open_logs)
        self.inspector.tab_requested.connect(self._on_tab_requested)
        self.topbar.open_requested.connect(self.choose_folder)
        self.topbar.folder_settings_requested.connect(self.open_folder_settings)
        self.folder_settings.closed.connect(lambda: self.queue.setFocus(Qt.FocusReason.OtherFocusReason))
        self.topbar.logs_requested.connect(lambda: self.open_logs(None))
        self.topbar.start_requested.connect(self.start_run)
        self.topbar.mode_changed.connect(self.set_mode)
        self.open_view.choose_requested.connect(self.choose_folder)

    # --- folders --------------------------------------------------------------------------

    def _run_in_progress(self) -> bool:
        snapshot = self.controller.run_snapshot()
        return snapshot is not None and not snapshot.finished

    def _ask(self, title: str, text: str) -> bool:
        """One Yes/No question, default No, in the style of the overwrite
        prompt (the window's only modals)."""
        answers = QMessageBox.StandardButton
        return QMessageBox.question(self, title, text, answers.Yes | answers.No,
                                    answers.No) == answers.Yes

    def _may_stop_the_run(self, text: str) -> bool:
        """True to go ahead: no run is on, or the user agreed to stop it. The
        run is then stopped cooperatively and the stop is logged, rather than
        vanishing with the folder's jobs (close_folder cancels everything)."""
        if not self._run_in_progress():
            return True
        if not self._ask(RUN_STOP_TITLE, text):
            return False
        self.controller.append_log(PIPELINE_LOG, f"\nRun stopped: {text}\n")
        logger.info("stopping the run: %s", text)
        self.controller.stop_run()
        return True

    def open_folder(self, path: str) -> None:
        """Open `path`. A folder that cannot be opened (an unsupported project
        version, not a directory, ...) is reported in place: inline in the
        empty state, or as a banner while another folder stays open.

        Opening a folder closes the open one, which cancels its jobs -- the
        running OCR among them. The project block is a one-click folder
        picker sitting exactly where the run status prints, and Ctrl+O and a
        dropped folder do the same, so a run is never lost to one click:
        stopping it takes one question first."""
        if not self._may_stop_the_run(RUN_OPEN_TEXT):
            return
        try:
            self.controller.open_folder(path)
        except Exception as exc:                     # reported to the user, never raised into Qt
            if not isinstance(exc, EXPECTED_OPEN_ERRORS):
                logger.exception("could not open %s", path)
            message = _open_error_text(path, exc)
            if self.controller.project is None:
                self.open_view.show_error(message)
            else:
                self.error_banner.show_message(OPEN_FAILED_TITLE, message, "bad")

    def choose_folder(self) -> None:
        self._picker.pick(last_path())

    def _on_project_opened(self, path: str) -> None:
        remember_path(path)
        self.setWindowTitle(f"{APP_NAME} — {os.path.basename(path) or path}")
        self.open_view.clear_error()
        if self.error_banner.title() != SAVE_FAILED_TITLE:
            self.error_banner.hide()      # an open failure is over; a warning about unsaved work is not
        self.centre.setCurrentWidget(self.workbench)
        self.set_mode(MODE_REVIEW)
        self.queue.setFocus(Qt.FocusReason.OtherFocusReason)

    def _on_project_saved(self) -> None:
        """The next successful save takes the "Not saved" banner down: a
        banner nothing but ✕ can clear goes on saying the folder is unsaved
        long after it was written."""
        if self.error_banner.title() == SAVE_FAILED_TITLE:
            self.error_banner.hide()

    def _on_project_closed(self) -> None:
        self.setWindowTitle(APP_NAME)
        self.centre.setCurrentWidget(self.open_view)
        self.set_mode(MODE_REVIEW)

    def _on_selection_changed(self, name) -> None:
        # Before the views ask for their pixels: the frames and strips of the
        # file being left are not worth a worker any more (see
        # ProjectController.set_view_file).
        self.controller.set_view_file(name)
        self.stage.set_file(name)
        self.inspector.set_file(name)
        self._sync_actions()

    def _sync_actions(self, *_args) -> None:
        """Space and T act on the selected file; neither while the focused
        widget consumes keys, Space not while the file is PENDING or a value
        it needs is missing, and T not while that file's proof is already
        running or its metadata has not been read (ruling C4: T and the
        inspector's "T run" button are one command)."""
        name = self.queue.selected()
        has_file = name is not None and name in self.controller.names()
        focused = QApplication.focusWidget()
        editing = consumes_keys(focused) or self.folder_settings.contains_focus(focused)
        entry = self.controller.entry(name) if has_file else None
        self.proof_action.setEnabled(has_file and not editing and not self.controller.proof_pending(name)
                                     and can_run_proof(entry))
        self.review_action.setEnabled(has_file and not editing and can_mark_reviewed(entry))

    def report_unexpected_error(self, text: str) -> None:
        """An exception nothing handled (see app/__main__.py's excepthook): its
        traceback goes to the Pipeline log and a dismissible banner says so."""
        self.controller.append_log(PIPELINE_LOG, f"Unexpected error:\n{text}")
        self.crash_banner.show_message(CRASH_TITLE, "", "bad")

    def _on_tab_requested(self, title: str) -> None:
        index = self.stage.index_of(title)
        if index is not None:
            self.stage.set_current(index)

    # --- later tasks ----------------------------------------------------------------------

    def open_folder_settings(self) -> None:
        """The Folder settings sheet (plan 3B Task 4), over the review
        layout: it never covers the Run view."""
        self.set_mode(MODE_REVIEW)
        self.folder_settings.open()

    # --- run and logs -----------------------------------------------------------------------

    def mode(self) -> int:
        return self.modes.currentIndex()

    def set_mode(self, mode: int) -> None:
        """MODE_REVIEW (stage + inspector) or MODE_RUN (the Run view). The
        Folder settings sheet belongs to Review, so Run closes it."""
        if mode == MODE_RUN:
            self.folder_settings.close_sheet()
        self.modes.setCurrentIndex(mode)
        self.topbar.set_mode(mode)

    def start_run(self) -> None:
        """Start the startable files. Done files are asked about first; on No
        they stay out of the run (ruling C5: nothing is deleted)."""
        controller = self.controller
        if controller.project is None:
            return
        self.topbar.clear_start_error()
        names = controller.startable_files(include_done=True)
        replace = controller.files_needing_overwrite(names)
        if replace:
            answers = QMessageBox.StandardButton
            reply = QMessageBox.question(self, OVERWRITE_TITLE, OVERWRITE_TEXT.format(n=len(replace)),
                                         answers.Yes | answers.No, answers.No)
            if reply != answers.Yes:
                names = [name for name in names if name not in set(replace)]
        if not names:
            if replace:                                  # every file was declined: say so instead of nothing
                self.topbar.show_start_error(NOTHING_TO_RUN)
            return
        try:
            controller.start_run(names)
        except (ValueError, RuntimeError) as exc:        # e.g. two files write the same chi/ output
            self.topbar.show_start_error(str(exc))
            return
        self.set_mode(MODE_RUN)

    def open_logs(self, key: str | None = None) -> None:
        """Show the logs window (one per window), at `key`'s section when given."""
        if self.logs_window is None:
            self.logs_window = LogsWindow(self.controller, self)
        self.logs_window.show()
        self.logs_window.raise_()
        self.logs_window.activateWindow()
        if key is not None:
            self.logs_window.show_key(key)

    # --- drag and drop --------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:
        if folder_from_mime(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        self.dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        folder = folder_from_mime(event.mimeData())
        if folder is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.open_folder(folder)

    # --- window state ---------------------------------------------------------------------

    def _restore_geometry(self) -> None:
        geometry = app_settings().value(GEOMETRY_KEY)
        if isinstance(geometry, QByteArray) and self.restoreGeometry(geometry):
            return
        width, height = DEFAULT_SIZE
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width, height = min(width, available.width()), min(height, available.height())
        self.resize(width, height)

    def _check_dependencies(self) -> None:
        problems = dependency_problems()
        if problems:
            self.dependency_banner.show_message(" · ".join(title for title, _text in problems),
                                                "\n\n".join(text for _title, text in problems), "warn")

    def closeEvent(self, event) -> None:
        """Quitting stops whatever is running, so it asks the same questions
        opening another folder does: once, and only while there is something
        to lose."""
        if not self._closing:
            if not self._may_stop_the_run(RUN_QUIT_TEXT):
                event.ignore()
                return
            if self.controller.save_blocked and not self._ask(BLOCKED_SAVE_TITLE, BLOCKED_SAVE_TEXT):
                event.ignore()
                return
            self._closing = True
        app_settings().setValue(GEOMETRY_KEY, self.saveGeometry())
        if self.logs_window is not None:
            self.logs_window.close()
        self.run_view.shutdown()
        self.controller.shutdown()
        event.accept()
