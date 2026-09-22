"""Installing the OCR engine (paddle) into the engine dir, with a CPU fallback.

    installer = EngineInstaller(InstallContext.for_bundle(bundle), emit=on_event)
    result = installer.install(variant, gpu=choice.gpu, cancel=cancel_event)

One attempt is: pip install into `<tag>.partial` -> `verify` (paddle imports
and runs a small conv2d on the device, in a child process) -> `prefetch`
(build both OCR engines once so the models download, and predict once).
When the attempt was a GPU build and it fails at any step, the partial dir
is wiped and the CPU build is installed instead; the result says so and
why. Only a verified install is promoted to `<tag>` and gets a state.json
with `verified: true`; engine dirs of other versions are removed after that.

A model download that fails (no network) does not fail the install: the
engine is verified, `models_ready` is false, and paddlex downloads the models
on first use. A prefetch that fails with the models on disk is a device
failure, which for a GPU build means the CPU fallback.

Cancel (`cancel` event) ends the running child process (see procs.py) and
deletes the partial dir; an engine that was installed before stays.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from core.runtime import checks, pip
from core.runtime.engine import (
    EngineState,
    engine_dir,
    engines_root,
    partial_dir,
    promote,
    remove_other_engines,
    remove_tree,
    write_state,
)
from core.runtime.gpu import CPU, GpuInfo, download_bytes, variant_label
from core.runtime.paths import Bundle, data_dir, host_arch, host_os, source_root
from core.runtime.procs import Cancelled, ProcessResult, ProcessRunner, SubprocessRunner

PIP_VERSION_TIMEOUT = 120.0
VERIFY_TIMEOUT = 300.0
PREFETCH_TIMEOUT = 1800.0
# Peak disk use of an install against its download: pip keeps the downloaded
# wheels in its temp dir until it has unpacked all of them (cu129 on Linux:
# 5.4 GB downloaded + 8.2 GB unpacked).
DISK_FACTOR = 2.6


@dataclass
class InstallEvent:
    kind: str                   # "step" | "log" | "progress" | "notice"
    text: str = ""
    done_bytes: int = 0
    total_bytes: int = 0


@dataclass
class InstallResult:
    ok: bool
    variant: str | None = None
    state: EngineState | None = None
    engine_dir: Path | None = None
    cancelled: bool = False
    fell_back: bool = False
    fallback_reason: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class InstallContext:
    os_name: str
    arch: str
    python: str                                 # the interpreter pip and the checks run with
    data: Path
    constraints: Path | None
    source: Path = field(default_factory=source_root)
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    py: tuple[int, int] = (sys.version_info[0], sys.version_info[1])

    @classmethod
    def for_bundle(cls, bundle: Bundle, data: Path | None = None) -> InstallContext:
        return cls(os_name=bundle.os or host_os(), arch=bundle.arch or host_arch(),
                   python=pip.python_for_subprocess(sys.executable),
                   data=data_dir() if data is None else data, constraints=bundle.constraints)


class _AttemptFailed(Exception):
    pass


class EngineInstaller:
    def __init__(self, context: InstallContext, runner: ProcessRunner | None = None,
                 emit: Callable[[InstallEvent], None] | None = None):
        self.context = context
        self.runner = runner if runner is not None else SubprocessRunner()
        self._emit = emit or (lambda event: None)

    # -- public ----------------------------------------------------------

    def install(self, variant: str, gpu: GpuInfo | None = None,
                cancel: threading.Event | None = None) -> InstallResult:
        cancel = cancel or threading.Event()
        partial = partial_dir(self.context.data)
        final = engine_dir(self.context.data)
        try:
            pip_version = self._pip_version(cancel)
            fallback_reason = None
            result = None
            if variant != CPU:
                try:
                    result = self._attempt(variant, gpu, partial, pip_version, cancel)
                except _AttemptFailed as exc:
                    fallback_reason = str(exc)
                    self._notice(f"The {variant_label(variant)} did not work: {fallback_reason}. "
                                 f"Installing the CPU build instead.")
            if result is None:
                try:
                    result = self._attempt(CPU, None, partial, pip_version, cancel)
                except _AttemptFailed as exc:
                    self._cleanup(partial)
                    return InstallResult(ok=False, error=str(exc), fell_back=fallback_reason is not None,
                                         fallback_reason=fallback_reason)
            state, warnings = result
            if fallback_reason is not None:
                state.fallback_reason = fallback_reason
                if gpu is not None:
                    state.gpu_name, state.driver = gpu.name, gpu.driver
            write_state(partial, state)
            self._step("Finishing")
            promote(partial, final)
            for removed in remove_other_engines(final):
                self._log(f"Removed the old engine {removed.name}")
            self._cleanup_tmp()
            return InstallResult(ok=True, variant=state.variant, state=state, engine_dir=final,
                                 fell_back=fallback_reason is not None, fallback_reason=fallback_reason,
                                 warnings=warnings)
        except Cancelled:
            self._cleanup(partial)
            return InstallResult(ok=False, cancelled=True, error="Cancelled")
        except _AttemptFailed as exc:                           # pip itself is unusable
            self._cleanup(partial)
            return InstallResult(ok=False, error=str(exc))
        except OSError as exc:
            self._cleanup(partial)
            return InstallResult(ok=False, error=f"{type(exc).__name__}: {exc}")

    # -- one attempt -----------------------------------------------------

    def _attempt(self, variant: str, gpu: GpuInfo | None, partial: Path, pip_version: tuple[int, ...],
                 cancel: threading.Event) -> tuple[EngineState, list[str]]:
        ctx = self.context
        label = variant_label(variant)
        device = "cpu" if variant == CPU else "gpu"
        if not remove_tree(partial):
            raise _AttemptFailed(f"could not clear {partial}")
        partial.mkdir(parents=True)
        total = download_bytes(variant, ctx.os_name)
        self._check_disk(total)
        constraints = pip.write_constraints(ctx.constraints, partial / "constraints.txt")
        requirement = pip.requirement(variant, ctx.os_name, ctx.py)
        command = pip.install_command(ctx.python, requirement, constraints, pip_version)
        env = pip.install_env(ctx.env, partial)
        tmp = self._tmp_dir()
        for key in ("TMPDIR", "TEMP", "TMP"):
            env[key] = str(tmp)

        self._step(f"Downloading the {label}", 0, total)
        self._log("$ " + " ".join(command))
        output = pip.PipOutput()

        def on_pip_line(line: str) -> None:
            status = output.feed(line)
            if status.is_progress_line:
                self._progress(status.done_bytes, max(total, status.done_bytes))
                return
            self._log(line)
            if status.changed_step:
                self._step(status.step, status.done_bytes, max(total, status.done_bytes))

        done = self.runner.run(command, env=env, cwd=ctx.source, on_line=on_pip_line, cancel=cancel)
        if not done.ok:
            raise _AttemptFailed(f"pip failed ({_describe_exit(done)})")

        self._step(f"Checking the {label}")
        verify = self._check("verify", partial, device, VERIFY_TIMEOUT, cancel)
        if not verify.get("ok"):
            raise _AttemptFailed(f"the check failed: {verify.get('error') or 'no result'}")

        self._step("Downloading the OCR models (about 170 MB) and building the engines")
        warnings = []
        prefetch = self._check("prefetch", partial, device, PREFETCH_TIMEOUT, cancel)
        models_ready = bool(prefetch.get("ok"))
        if not models_ready:
            error = prefetch.get("error") or "no result"
            if prefetch.get("models"):                          # models are there: the device failed
                raise _AttemptFailed(f"building the OCR engine failed: {error}")
            warnings.append(f"The OCR models could not be downloaded now ({error}); "
                            f"they will download the first time OCR runs.")
            self._notice(warnings[-1])
        state = EngineState(variant=variant, verified=True, models_ready=models_ready,
                            gpu_name=gpu.name if gpu else None, driver=gpu.driver if gpu else None)
        return state, warnings

    def _check(self, name: str, partial: Path, device: str, timeout: float, cancel: threading.Event) -> dict:
        ctx = self.context
        command = [ctx.python, "-s", "-E", "-X", "utf8", "-m", "core.runtime.checks", name,
                   "--engine", str(partial), "--device", device]
        env = {key: value for key, value in ctx.env.items() if key != "PYTHONUSERBASE"}
        self._log("$ " + " ".join(command))
        done = self.runner.run(command, env=env, cwd=ctx.source, on_line=self._log, cancel=cancel,
                               timeout=timeout)
        result = checks.parse_result(done.tail) or {}
        if done.timed_out:
            result = {**result, "ok": False, "error": f"timed out after {int(timeout)} s"}
        elif not done.ok and result.get("ok"):
            result["ok"] = False
        if not result.get("ok") and not result.get("error"):
            result["error"] = _describe_exit(done)
        return result

    # -- helpers ---------------------------------------------------------

    def _pip_version(self, cancel: threading.Event) -> tuple[int, ...]:
        seen: list[str] = []
        done = self.runner.run([self.context.python, "-m", "pip", "--version"], env=dict(self.context.env),
                               on_line=seen.append, cancel=cancel, timeout=PIP_VERSION_TIMEOUT)
        version = pip.parse_pip_version("\n".join(seen))
        if not done.ok or not version:
            raise _AttemptFailed(f"pip is not available in {self.context.python} ({_describe_exit(done)})")
        self._log(seen[-1] if seen else f"pip {'.'.join(map(str, version))}")
        return version

    def _check_disk(self, download: int) -> None:
        root = engines_root(self.context.data)
        root.mkdir(parents=True, exist_ok=True)
        need = int(download * DISK_FACTOR)
        free = shutil.disk_usage(root).free
        if free < need:
            raise _AttemptFailed(f"not enough disk space in {root}: {_gb(need)} needed, {_gb(free)} free")

    def _tmp_dir(self) -> Path:
        tmp = engines_root(self.context.data) / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp

    def _cleanup_tmp(self) -> None:
        remove_tree(engines_root(self.context.data) / "tmp")

    def _cleanup(self, partial: Path) -> None:
        remove_tree(partial)
        self._cleanup_tmp()

    def _step(self, text: str, done: int = 0, total: int = 0) -> None:
        self._emit(InstallEvent("step", text, done, total))

    def _progress(self, done: int, total: int) -> None:
        self._emit(InstallEvent("progress", "", done, total))

    def _log(self, text: str) -> None:
        self._emit(InstallEvent("log", text))

    def _notice(self, text: str) -> None:
        self._emit(InstallEvent("notice", text))


def _describe_exit(done: ProcessResult) -> str:
    if done.timed_out:
        return "timed out"
    last = next((line.strip() for line in reversed(done.tail) if line.strip()), "")
    return f"exit code {done.returncode}" + (f": {last}" if last else "")


def _gb(value: int) -> str:
    return f"{value / 1e9:.1f} GB"
