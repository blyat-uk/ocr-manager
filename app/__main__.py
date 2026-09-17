"""`python -m app [folder]`: the workbench window (ruling C10).

`--quit-after SECONDS` closes the window after that long (the smoke test's
hook).
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.theme.qss import apply_theme


def _parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(prog="python -m app", description="OCR Manager")
    parser.add_argument("folder", nargs="?", help="a folder of episodes to open")
    parser.add_argument("--quit-after", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_known_args(argv)


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
    code = app.exec()
    window.close()                  # a no-op when already closed; shuts the controller down otherwise
    window.deleteLater()
    return code


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
