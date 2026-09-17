"""`python -m app [folder]`: the workbench window (ruling C10).

`--quit-after SECONDS` closes the window after that long (the smoke test's
hook).

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

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.theme.qss import apply_theme


def _program_name() -> str:
    """How this run was started, for --help: "python -m app", or the script
    name when it came through main.py."""
    script = sys.argv[0] if sys.argv else ""
    name = os.path.basename(script)
    return "python -m app" if name in ("", "__main__.py") else name


def _parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(prog=_program_name(), description="OCR Manager")
    parser.add_argument("folder", nargs="?", help="a folder of episodes to open")
    parser.add_argument("--quit-after", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_known_args(argv)


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
    args, qt_args = _parse(sys.argv[1:] if argv is None else list(argv))
    app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
    apply_theme(app)
    window = MainWindow()
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
