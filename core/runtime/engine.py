"""The OCR engine directory: where a bundle's paddle lives, and turning it on.

A bundle ships everything except paddle. The app installs paddle on first
run into ONE directory per interpreter minor and paddle pin,

    <data>/engine/py312-paddle3.3.0/        (engine_dir())
        state.json                          (EngineState)
        lib/python3.12/site-packages/       (Linux/macOS; Windows: Python312/site-packages)
            paddle/  nvidia/  ...

using `pip install --user` with PYTHONUSERBASE=<that dir>, so `paddle/` and
the `nvidia/*` CUDA libraries it loads relative to itself land side by side
in one site dir. `activate()` makes that dir the process's *user site*
(site.USER_BASE / site.USER_SITE plus site.addsitedir) before anything
imports paddle: paddle's own library search (`paddle/base/core.py`) looks in
site.getsitepackages() and site.USER_SITE only.

An install goes into `<tag>.partial` and is renamed to `<tag>` only once it
verified, so a directory named `<tag>` with a verified state is always a
complete engine.
"""
from __future__ import annotations

import json
import os
import shutil
import site
import sys
import sysconfig
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from core.runtime.paths import data_dir

PADDLE_VERSION = "3.3.0"
STATE_FILE = "state.json"
STATE_SCHEMA = 1
PARTIAL_SUFFIX = ".partial"
OLD_SUFFIX = ".old"


def engine_tag(version_info=sys.version_info, paddle: str = PADDLE_VERSION) -> str:
    return f"py{version_info[0]}{version_info[1]}-paddle{paddle}"


def engines_root(data: Path | None = None) -> Path:
    return (data_dir() if data is None else data) / "engine"


def engine_dir(data: Path | None = None) -> Path:
    return engines_root(data) / engine_tag()


def partial_dir(data: Path | None = None) -> Path:
    final = engine_dir(data)
    return final.with_name(final.name + PARTIAL_SUFFIX)


def user_scheme() -> str:
    """The sysconfig scheme `pip install --user` uses on this interpreter."""
    try:
        return sysconfig.get_preferred_scheme("user")
    except (AttributeError, KeyError):                          # pragma: no cover - < 3.10
        return f"{os.name}_user"


def site_dir(base: Path) -> Path:
    """The site-packages `pip install --user` fills when PYTHONUSERBASE=base."""
    return Path(sysconfig.get_path("purelib", user_scheme(), vars={"userbase": str(base)}))


# --------------------------------------------------------------------------
# state.json
# --------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class EngineState:
    """`<engine dir>/state.json`. `variant` is "cpu" or the CUDA build
    ("cu129", "cu126", "cu118"); `verified` means the paddle import and a
    small op on that device succeeded in a separate process; `models_ready`
    that the OCR models were downloaded and both engines built once.
    `fallback_reason` is set when a GPU build was tried and CPU installed
    instead."""
    variant: str
    verified: bool = False
    models_ready: bool = False
    gpu_name: str | None = None
    driver: str | None = None
    paddle: str = PADDLE_VERSION
    python: str = field(default_factory=lambda: f"{sys.version_info[0]}.{sys.version_info[1]}")
    installed_at: str = field(default_factory=utc_now)
    fallback_reason: str | None = None
    schema: int = STATE_SCHEMA

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> EngineState:
        known = {name for name in cls.__dataclass_fields__}
        values = {key: value for key, value in data.items() if key in known}
        if not isinstance(values.get("variant"), str):
            raise ValueError("state.json has no variant")
        return cls(**values)

    @property
    def is_gpu(self) -> bool:
        return self.variant != "cpu"


def read_state(directory: Path) -> EngineState | None:
    """The engine's state, or None when missing or unreadable."""
    try:
        data = json.loads((directory / STATE_FILE).read_text(encoding="utf-8"))
        return EngineState.from_json(data) if isinstance(data, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def write_state(directory: Path, state: EngineState) -> None:
    """Atomically (tmp + fsync + replace), like the project store."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / STATE_FILE
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(state.to_json(), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)


def installed_state(data: Path | None = None) -> EngineState | None:
    """The state of a complete, verified engine for this interpreter, else None."""
    directory = engine_dir(data)
    state = read_state(directory)
    if state is None or not state.verified or not site_dir(directory).is_dir():
        return None
    return state


# --------------------------------------------------------------------------
# Activation
# --------------------------------------------------------------------------

def activate(directory: Path) -> Path:
    """Make `directory` this process's user site (see the module docstring)
    and return its site-packages. Idempotent. Must run before anything
    imports paddle. Also exports PYTHONUSERBASE for child processes started
    without -E/-s."""
    directory = Path(directory)
    packages = site_dir(directory)
    site.ENABLE_USER_SITE = True
    site.USER_BASE = str(directory)
    site.USER_SITE = str(packages)
    os.environ["PYTHONUSERBASE"] = str(directory)
    if str(packages) not in sys.path:
        site.addsitedir(str(packages))
    return packages


# --------------------------------------------------------------------------
# Directory housekeeping
# --------------------------------------------------------------------------

def remove_tree(path: Path) -> bool:
    """Delete a directory tree; True when it is gone. Never raises: a file
    held open (Windows) leaves the tree behind for the next start."""
    if not path.exists():
        return True
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


def promote(partial: Path, final: Path) -> None:
    """Replace `final` with the verified `partial` install."""
    if final.exists():
        aside = final.with_name(final.name + OLD_SUFFIX)
        remove_tree(aside)
        os.replace(final, aside)
        remove_tree(aside)
    os.replace(partial, final)


def remove_other_engines(keep: Path) -> list[Path]:
    """Engine dirs for other interpreter/paddle versions, and leftovers of
    interrupted installs, once `keep` verified. Returns what was removed."""
    removed = []
    root = keep.parent
    if not root.is_dir():
        return removed
    for child in root.iterdir():
        if child == keep or not child.is_dir():
            continue
        if remove_tree(child):
            removed.append(child)
    return removed
