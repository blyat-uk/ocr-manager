"""`python -m app [folder]`: the workbench window (ruling C10).

Startup order (`main`): `app.bootstrap.boot()` first -- stdout/stderr to
the log file when there is no console, then the bundle's OCR engine
activated before anything imports paddle -- then the flags:

- `--version` prints the version and exits.
- `--self-test`, `--install-engine {auto,cpu,gpu}`, `--ocr-smoke`: headless
  CI modes, see `app/cli.py` for what they do and their exit codes.
- `--setup-engine` opens the engine setup dialog even when an engine is
  installed (bundles only; a note and the window in developer mode).
- `--quit-after SECONDS` closes the window after that long (the smoke
  test's hook).

In a bundle without an installed engine the setup dialog
(`app/views/engine_setup.py`) comes first; leaving it without an engine
quits. The main window is imported only after that.

PyQt6 aborts the process when a slot raises and `sys.excepthook` is Python's
default, so `main` installs a hook for as long as the window runs: the
traceback goes to stderr and to the window (`report_unexpected_error`: the
Pipeline log and a banner), and the app keeps running.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QDialog

from app.bootstrap import Boot, boot, reactivate
from app.theme.qss import apply_theme
from app.version import __version__

if TYPE_CHECKING:
    from app.main_window import MainWindow

APP_NAME = "OCR Manager"
ORGANIZATION = "OCRManager"             # the organisation app_settings() already uses
ICON_FILE = Path(__file__).resolve().parent.parent / "resources" / "app-icon.png"


def _program_name() -> str:
    """How this run was started, for --help: "python -m app", or the script
    name when it came through main.py."""
    script = sys.argv[0] if sys.argv else ""
    name = os.path.basename(script)
    return "python -m app" if name in ("", "__main__.py") else name


def _parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(prog=_program_name(), description="OCR Manager")
    parser.add_argument("folder", nargs="?", help="a folder of episodes to open")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument("--setup-engine", action="store_true",
                        help="open the OCR engine setup (install, reinstall or switch GPU/CPU)")
    parser.add_argument("--self-test", action="store_true", help="headless checks, JSON report (CI)")
    parser.add_argument("--install-engine", choices=("auto", "cpu", "gpu"), default=None,
                        help="install the OCR engine without a window (CI)")
    parser.add_argument("--ocr-smoke", action="store_true", help="OCR one rendered line and exit (CI)")
    parser.add_argument("--quit-after", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_known_args(argv)


def _describe_app(app: QApplication) -> None:
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(ORGANIZATION)
    if ICON_FILE.is_file():
        app.setWindowIcon(QIcon(str(ICON_FILE)))


def run_engine_setup(started: Boot) -> bool:
    """The engine setup dialog; True when an engine is installed afterwards."""
    from app.engine import EngineSetup
    from app.views.engine_setup import EngineSetupDialog

    have_engine = started.state is not None
    setup = EngineSetup(started.bundle)
    dialog = EngineSetupDialog(setup, have_engine=have_engine)
    accepted = dialog.exec() == QDialog.DialogCode.Accepted
    setup.cancel()
    setup.wait(60)
    dialog.deleteLater()
    if accepted:
        reactivate(started)
    return started.state is not None


def install_excepthook(window: MainWindow) -> Callable:
    """Report unhandled exceptions through `window` instead of aborting;
    returns the hook it replaced (restore it when the window is gone)."""
    previous = sys.excepthook

    def hook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):         # Ctrl+C in the terminal: quit
            previous(exc_type, exc, tb)
            QApplication.quit()
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            sys.stderr.write(text)
        except Exception:
            pass
        try:
            window.report_unexpected_error(text)
        except Exception:                                   # e.g. the window is already deleted
            pass

    sys.excepthook = hook
    return previous


def main(argv: list[str] | None = None) -> int:
    started = boot()
    args, qt_args = _parse(sys.argv[1:] if argv is None else list(argv))
    if args.version:
        print(__version__, flush=True)
        return 0
    if args.self_test or args.install_engine or args.ocr_smoke:
        from app import cli
        if args.self_test:
            return cli.self_test(started)
        if args.install_engine:
            return cli.install_engine(started, args.install_engine)
        return cli.ocr_smoke(started)

    app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
    _describe_app(app)
    apply_theme(app)
    if started.bundled and (started.needs_engine or args.setup_engine):
        if not run_engine_setup(started):
            return 1
    elif args.setup_engine:
        print("--setup-engine: developer mode (no $OCR_MANAGER_BUNDLE), paddle comes from this environment",
              file=sys.stderr)

    from app.main_window import MainWindow
    from app.views.tabs import evidence_tabs

    window = MainWindow(tabs_factory=evidence_tabs)
    window.show()
    if args.folder:
        window.open_folder(args.folder)
    if args.quit_after is not None:
        def quit_now() -> None:
            window.close()
            app.quit()

        QTimer.singleShot(max(0, int(args.quit_after * 1000)), quit_now)
    previous_hook = install_excepthook(window)
    try:
        code = app.exec()
        window.close()              # a no-op when already closed; shuts the controller down otherwise
    finally:
        sys.excepthook = previous_hook
    window.deleteLater()
    return code


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
