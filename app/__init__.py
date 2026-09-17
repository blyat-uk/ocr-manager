"""OCR Manager's new PyQt6 window (plan 3B).

This package is the only Qt layer of the revamp: everything under
`core/` (`core/project/`, `core/jobs/`, `core/detect/`) stays Qt-free
(ruling C8). The old window under `widgets/` and `main.py` keeps working
until plan 3B's parity task lands (ruling C10).
"""
