"""OCR Manager's new PyQt6 window (plan 3B).

This package is the only Qt layer of the revamp: everything under
`core/` (`core/project/`, `core/jobs/`, `core/detect/`) stays Qt-free
(ruling C8). `python -m app` and `main.py` both start it; the v1 window,
its widgets and its config code were deleted with plan 3B's Task 6
(ruling C10).
"""
