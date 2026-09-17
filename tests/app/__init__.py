"""tests/app -- tests for the new PyQt6 window under `app/`.

There is no `tests/__init__.py` anywhere else in this repo, so pytest's
default ("prepend") import mode walks up from any file in this directory
only as long as it finds `__init__.py` files: this directory has one, but
its parent `tests/` does not, so pytest inserts `tests/` into `sys.path`
and imports everything under here as the dotted package `app` -- the same
name as the real source package at the repo root (`app/`). Whichever one
gets imported first would otherwise win `sys.modules["app"]` for the rest
of the test session, making the other unreachable.

Fix: extend this package's own `__path__` to also search the real `app/`
directory, after this one -- the same multi-root technique namespace
packages and `pkgutil.extend_path()` use, not a private import-system
detail. Submodules that live in this test directory (`conftest.py`,
`test_*.py`) resolve here first; anything else (`app.theme`, `app.widgets`,
`app.state_text`, ...) falls through to the real source package. Safe even
if a future `tests/__init__.py` removes the collision entirely (dotted name
becomes `tests.app`, and this append is simply unused).
"""
from pathlib import Path

_REAL_APP_DIR = Path(__file__).resolve().parents[2] / "app"
if str(_REAL_APP_DIR) not in __path__:
    __path__.append(str(_REAL_APP_DIR))
