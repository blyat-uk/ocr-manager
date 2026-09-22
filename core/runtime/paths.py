"""Where OCR Manager keeps its own files, and whether it runs from a bundle.

- **Data dir** (`data_dir()`): the per-user directory for the engine, logs
  and anything else the app owns outside project folders.

  | OS      | default                                             |
  |---------|-----------------------------------------------------|
  | Linux   | `$XDG_DATA_HOME/ocr-manager`, else `~/.local/share/ocr-manager` |
  | Windows | `%LOCALAPPDATA%\\ocr-manager`                        |
  | macOS   | `~/Library/Application Support/ocr-manager`         |

  `$OCR_MANAGER_DATA_DIR` overrides all of them (tests and CI use it).

- **Bundle** (`current_bundle()`): a portable build sets
  `$OCR_MANAGER_BUNDLE=<root>`; `<root>/bundle.json` says which build it is
  and `<root>/constraints.txt` pins what the bundled interpreter already has.
  With the variable unset the app runs in developer mode: paddle comes from
  the environment and nothing is ever installed.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

APP_DIR_NAME = "ocr-manager"
DATA_DIR_ENV = "OCR_MANAGER_DATA_DIR"
BUNDLE_ENV = "OCR_MANAGER_BUNDLE"
BUNDLE_FILE = "bundle.json"
CONSTRAINTS_FILE = "constraints.txt"
LOG_FILE = "ocr-manager.log"


def host_os(platform_name: str | None = None) -> str:
    """"linux", "win" or "mac" -- the names bundle.json uses."""
    name = sys.platform if platform_name is None else platform_name
    if name.startswith("win"):
        return "win"
    if name == "darwin":
        return "mac"
    return "linux"


def host_arch(machine: str | None = None) -> str:
    """"x86_64" or "arm64" (anything else is returned lower-cased as is)."""
    value = (platform.machine() if machine is None else machine).lower()
    if value in ("x86_64", "amd64", "x64"):
        return "x86_64"
    if value in ("arm64", "aarch64"):
        return "arm64"
    return value


def data_dir(env: Mapping[str, str] | None = None, os_name: str | None = None,
             home: Path | None = None) -> Path:
    """The app's per-user data directory (not created here)."""
    env = os.environ if env is None else env
    override = env.get(DATA_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    os_name = host_os() if os_name is None else os_name
    home = Path.home() if home is None else home
    if os_name == "win":
        base = env.get("LOCALAPPDATA", "").strip()
        return (Path(base) if base else home / "AppData" / "Local") / APP_DIR_NAME
    if os_name == "mac":
        return home / "Library" / "Application Support" / APP_DIR_NAME
    xdg = env.get("XDG_DATA_HOME", "").strip()
    return (Path(xdg) if xdg else home / ".local" / "share") / APP_DIR_NAME


def logs_dir(env: Mapping[str, str] | None = None) -> Path:
    return data_dir(env) / "logs"


def log_file(env: Mapping[str, str] | None = None) -> Path:
    return logs_dir(env) / LOG_FILE


def source_root() -> Path:
    """The directory holding main.py, app/, core/ and videocr/ (`<root>/src`
    in a bundle, the checkout in developer mode)."""
    return Path(__file__).resolve().parent.parent.parent


class BundleError(RuntimeError):
    """$OCR_MANAGER_BUNDLE is set but does not name a usable bundle."""


@dataclass(frozen=True)
class Bundle:
    root: Path
    version: str
    os: str
    arch: str

    @property
    def constraints(self) -> Path:
        return self.root / CONSTRAINTS_FILE


def current_bundle(env: Mapping[str, str] | None = None) -> Bundle | None:
    """The bundle this process runs from, or None in developer mode.

    Raises BundleError when the variable is set but its directory is missing
    or its bundle.json is unreadable: a broken bundle must say so rather
    than quietly behave like a developer checkout."""
    env = os.environ if env is None else env
    raw = env.get(BUNDLE_ENV, "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise BundleError(f"{BUNDLE_ENV}={raw} is not a directory")
    path = root / BUNDLE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BundleError(f"{path} is missing") from None
    except (OSError, ValueError) as exc:
        raise BundleError(f"{path} is unreadable: {exc}") from None
    if not isinstance(data, dict):
        raise BundleError(f"{path} is not a JSON object")
    return Bundle(root=root,
                  version=str(data.get("version", "")),
                  os=str(data.get("os") or host_os()),
                  arch=str(data.get("arch") or host_arch()))
