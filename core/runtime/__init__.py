"""The app's runtime environment (Qt-free, stdlib only): the data dir, the
portable bundle, and the OCR engine (paddle) a bundle installs on first run.

| Module     | Role |
|------------|------|
| `paths`    | data dir, log file, `$OCR_MANAGER_BUNDLE` and bundle.json |
| `gpu`      | the nvidia-smi probe and the pure, table-driven variant choice |
| `engine`   | the engine dir, its state.json, and `activate()` |
| `pip`      | the pip command, its environment, and reading its output |
| `procs`    | running a child process with streamed output and a cancel |
| `checks`   | `python -m core.runtime.checks verify|prefetch` (child processes) |
| `install`  | `EngineInstaller`: pip -> verify -> prefetch, GPU -> CPU fallback |
| `logfile`  | sending stdout/stderr to a log file when there is no console |

Nothing here imports paddle; `activate()` must run before anything does.
"""
from __future__ import annotations

from pathlib import Path

from core.runtime.engine import EngineState, activate, engine_dir, installed_state
from core.runtime.gpu import CPU, Choice, GpuProbe, select_variant
from core.runtime.paths import Bundle

REQUESTS = ("auto", "cpu", "gpu")


def activate_installed(data: Path | None = None) -> EngineState | None:
    """Activate this interpreter's verified engine; its state, or None when
    there is none (the setup dialog / --install-engine is needed)."""
    state = installed_state(data)
    if state is not None:
        activate(engine_dir(data))
    return state


def resolve_request(request: str, bundle: Bundle, probe: GpuProbe) -> tuple[str, Choice]:
    """The variant to install for "auto" | "cpu" | "gpu", and the machine's
    recommendation. "gpu" and "auto" both take the recommendation: on a
    machine no GPU build fits, "gpu" yields "cpu" and the caller reports
    `Choice.reason`."""
    if request not in REQUESTS:
        raise ValueError(f"unknown engine request {request!r}")
    choice = select_variant(bundle.os, bundle.arch, probe)
    if request == "cpu":
        return CPU, choice
    return choice.variant, choice
