"""The pip command that installs paddle into the engine dir, and reading its output.

    <python> -X utf8 -m pip install --user --no-cache-dir --only-binary=:all:
        --no-warn-script-location --disable-pip-version-check --no-input
        --progress-bar raw -c <engine dir>/constraints.txt <requirement>

run with PYTHONUSERBASE=<engine dir> (and without -s/-E, which would turn
the user site off). The constraints are the bundle's own `pip freeze`, so
every package the bundle already has counts as satisfied and nothing in it
is shadowed or upgraded: only paddle (and, for a GPU build, the nvidia-*
CUDA libraries) land in the engine dir. `--no-cache-dir` keeps pip from
leaving a second copy of a multi-gigabyte wheel in its cache.

The GPU wheel is not on PyPI; it is required by direct URL and its
dependencies resolve from the default index (no extra index is added).

`--progress-bar raw` (pip >= 24.1) prints "Progress <cur> of <total>" lines
per download; older pips get `off` and the dialog shows steps only.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from core.runtime.engine import PADDLE_VERSION
from core.runtime.gpu import CPU, build_for

GPU_WHEEL_URL = ("https://paddle-whl.cdn.bcebos.com/stable/{cuda}/paddlepaddle-gpu/"
                 "paddlepaddle_gpu-{version}-cp{py}-cp{py}-{platform}.whl")
WHEEL_PLATFORM = {"linux": "linux_x86_64", "win": "win_amd64"}
RAW_PROGRESS_PIP = (24, 1)
BREAK_SYSTEM_PACKAGES_PIP = (23, 0, 1)
# Lines of the bundle's `pip freeze` that must not constrain the engine
# install: paddle itself and CUDA packages (the GPU wheel pins its own), and
# anything pip cannot use as a constraint (editables, direct references).
_DROP_CONSTRAINT = re.compile(r"^\s*(paddlepaddle|paddle-|nvidia-|cuda-)", re.IGNORECASE)


def requirement(variant: str, os_name: str, py: tuple[int, int]) -> str:
    """"paddlepaddle==3.3.0", or the GPU wheel by direct URL."""
    if variant == CPU:
        return f"paddlepaddle=={PADDLE_VERSION}"
    build = build_for(variant)
    if build is None:
        raise ValueError(f"unknown engine variant {variant!r}")
    platform = WHEEL_PLATFORM.get(os_name)
    if platform is None:
        raise ValueError(f"paddle has no GPU wheel for {os_name}")
    url = GPU_WHEEL_URL.format(cuda=build.cuda, version=PADDLE_VERSION, py=f"{py[0]}{py[1]}", platform=platform)
    return f"paddlepaddle-gpu @ {url}"


def parse_pip_version(text: str) -> tuple[int, ...]:
    """`pip --version` output ("pip 24.2 from ...") -> (24, 2)."""
    match = re.search(r"pip\s+(\d+(?:\.\d+)*)", text)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def install_command(python: str, req: str, constraints: Path | None,
                    pip_version: tuple[int, ...] = ()) -> list[str]:
    command = [python, "-X", "utf8", "-m", "pip", "install", "--user", "--no-cache-dir",
               "--only-binary=:all:", "--no-warn-script-location", "--disable-pip-version-check",
               "--no-input"]
    if pip_version >= RAW_PROGRESS_PIP:
        command += ["--progress-bar", "raw"]
    else:
        command += ["--progress-bar", "off"]
    if pip_version >= BREAK_SYSTEM_PACKAGES_PIP:
        command.append("--break-system-packages")   # a user-site install into our own dir, never the system's
    if constraints is not None:
        command += ["-c", str(constraints)]
    command.append(req)
    return command


def install_env(base: Mapping[str, str], engine_dir: Path) -> dict[str, str]:
    """The environment pip runs in: the engine dir as the user base, and none
    of the variables that would turn the user site off or move it."""
    env = {key: value for key, value in base.items()
           if key not in ("PYTHONNOUSERSITE", "PYTHONHOME", "PYTHONPATH", "PIP_USER", "PIP_TARGET",
                          "PIP_PREFIX", "PIP_REQUIRE_VIRTUALENV", "VIRTUAL_ENV")}
    env["PYTHONUSERBASE"] = str(engine_dir)
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def filter_constraints(lines: Sequence[str]) -> list[str]:
    """The bundle's freeze minus what must not constrain the engine install."""
    kept = []
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("-") or " @ " in text or "://" in text:
            continue
        if _DROP_CONSTRAINT.match(text):
            continue
        kept.append(text)
    return kept


def write_constraints(source: Path | None, target: Path) -> Path | None:
    """Copy the bundle's constraints (filtered) next to the install; None
    when the bundle has none."""
    if source is None or not source.is_file():
        return None
    lines = filter_constraints(source.read_text(encoding="utf-8", errors="replace").splitlines())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

_PROGRESS = re.compile(r"^Progress (\d+) of (\d+)\s*$")
_DOWNLOADING = re.compile(r"^\s*Downloading (\S+)")
_COLLECTING = re.compile(r"^\s*Collecting (\S+)")
_INSTALLING = re.compile(r"^\s*Installing collected packages")
_DONE = re.compile(r"^\s*Successfully installed")


@dataclass
class PipStatus:
    step: str = ""                 # human text for the current step
    done_bytes: int = 0            # bytes downloaded so far, all files
    changed_step: bool = False
    changed_bytes: bool = False
    is_progress_line: bool = False


class PipOutput:
    """Turns pip's output lines into a step text and a running byte count."""

    def __init__(self) -> None:
        self.finished_bytes = 0            # files downloaded before the current one
        self.current_done = 0
        self.current_total = 0
        self.step = "Preparing"

    @property
    def done_bytes(self) -> int:
        return self.finished_bytes + self.current_done

    def feed(self, line: str) -> PipStatus:
        status = PipStatus(step=self.step, done_bytes=self.done_bytes)
        match = _PROGRESS.match(line)
        if match:
            self.current_done, self.current_total = int(match.group(1)), int(match.group(2))
            status.done_bytes = self.done_bytes
            status.changed_bytes = True
            status.is_progress_line = True
            return status
        step = None
        if (match := _DOWNLOADING.match(line)):
            name = match.group(1).rsplit("/", 1)[-1]
            if not name.endswith(".metadata"):             # PEP 658 metadata: still resolving
                self._close_file()
                step = f"Downloading {name}"
        elif (match := _COLLECTING.match(line)):
            step = f"Resolving {match.group(1)}"
        elif _INSTALLING.match(line):
            self._close_file()
            step = "Installing (unpacking the downloaded packages)"
        elif _DONE.match(line):
            step = "Installed"
        if step is not None:
            self.step = step
            status.step = step
            status.changed_step = True
            status.done_bytes = self.done_bytes
        return status

    def _close_file(self) -> None:
        self.finished_bytes += self.current_done
        self.current_done = self.current_total = 0


def python_for_subprocess(executable: str) -> str:
    """pythonw.exe has no console streams; start children with the console
    interpreter next to it (they get CREATE_NO_WINDOW instead)."""
    path = Path(executable)
    if path.name.lower() == "pythonw.exe":
        console = path.with_name("python.exe")
        if console.exists():
            return str(console)
    return executable


def default_env() -> dict[str, str]:
    return dict(os.environ)
