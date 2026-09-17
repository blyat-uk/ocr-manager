"""Fixtures for tests/ui (the new PyQt6 window under `app/`).

pytest-qt is NOT installed (task brief) -- this file provides the one
fixture it would otherwise give us (a session-scoped QApplication), and
tests drive widgets directly with PyQt6.QtTest.QTest instead of pytest-qt's
`qtbot`. tests/conftest.py's autouse `_reset_engine_registry` fixture still
applies here (pytest collects parent conftests for a subdirectory), which
is harmless: nothing under tests/ui touches the OCR engine registry.

This directory is named `tests/ui`, not `tests/app`, specifically so it
never shares a name with the real top-level `app/` package -- see
test_theme_widgets.py's test_app_package_resolves_to_real_source_package.
"""
import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """The one QApplication instance for the whole test session -- Qt does
    not allow more than one per process, and offscreen tests still need one
    to construct any QWidget."""
    app = QApplication.instance() or QApplication([])
    return app
