#!/usr/bin/env python3
"""OCR Manager's entry point: `.venv/bin/python main.py [folder]`.

The window itself lives in `app/` (ruling C10) and `python -m app` starts
exactly the same thing; this file only exists so the app can still be started
by its script name, which is how the desktop entry and PyInstaller's
`ocr-manager.spec` launch it. (The installed console script goes straight to
`app.__main__:main` -- see `[project.scripts]` in pyproject.toml -- because
`main.py` is a loose module, not part of the shipped packages.)

`main` takes today's optional folder argument and the `--quit-after SECONDS`
test hook -- see app/__main__.py, which parses them.
"""
import multiprocessing
import sys

from app.__main__ import main

if __name__ == "__main__":
    # Required for a frozen (PyInstaller) build: without it, the
    # forkserver/resource-tracker launch commands core.detect.ranges' process
    # pool uses on the first fingerprinting job aren't recognised, and the
    # frozen executable re-launches the whole GUI instead (twice), blocking
    # forever. No-op in a normal (non-frozen) run.
    multiprocessing.freeze_support()
    sys.exit(main())
