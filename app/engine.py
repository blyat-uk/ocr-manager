"""The OCR engine as the window sees it: an app-level helper over
`core.runtime`, so views never import `core` themselves.

- `EngineSetup` (QObject) drives the first-run install for the setup
  dialog (`app/views/engine_setup.py`): the hardware line, the offers (the
  recommended GPU build when there is one, and CPU), and one install at a
  time on a plain worker thread whose events arrive as Qt signals (queued
  onto the GUI thread).
- `engine_problem()` is the main window's startup check: in a bundle, a
  missing or unverified engine is a dependency problem.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from core.runtime import gpu as gpu_mod
from core.runtime.engine import engine_dir, installed_state
from core.runtime.install import EngineInstaller, InstallContext, InstallEvent, InstallResult
from core.runtime.paths import Bundle, BundleError, current_bundle


@dataclass(frozen=True)
class Offer:
    variant: str                   # "cpu" | "cu129" | ...
    title: str                     # "GPU build (CUDA 12.9)"
    size_text: str                 # "about 5.4 GB to download"
    recommended: bool = False


@dataclass(frozen=True)
class SetupOutcome:
    ok: bool
    cancelled: bool = False
    title: str = ""                # the installed build, "CPU build"
    fell_back: bool = False
    fallback_reason: str = ""
    error: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)


def size_text(size: int) -> str:
    """"about 5.4 GB to download", "about 210 MB to download"."""
    if size >= 1_000_000_000:
        return f"about {size / 1e9:.1f} GB to download"
    return f"about {size / 1e6:.0f} MB to download"


def installed_text(data: Path | None = None) -> str:
    """"GPU build (CUDA 12.9), installed 2026-09-23" or "" when none."""
    state = installed_state(data)
    if state is None:
        return ""
    text = f"{gpu_mod.variant_label(state.variant)}, installed {state.installed_at[:10]}"
    if state.fallback_reason:
        text += " (the GPU build did not work)"
    return text


class EngineSetup(QObject):
    # bytes as `object`: a Qt int is 32-bit and a GPU download is over 2 GB
    step = pyqtSignal(str, object, object)      # text, done bytes, total bytes
    progress = pyqtSignal(object, object)       # done bytes, total bytes
    log = pyqtSignal(str)
    notice = pyqtSignal(str)
    finished = pyqtSignal(object)               # SetupOutcome

    def __init__(self, bundle: Bundle, data: Path | None = None, probe: gpu_mod.GpuProbe | None = None,
                 installer_factory: Callable[[Callable[[InstallEvent], None]], EngineInstaller] | None = None,
                 parent: QObject | None = None):
        super().__init__(parent)
        self._bundle = bundle
        self._data = data
        self._probe = probe if probe is not None else gpu_mod.probe_gpus(bundle.os)
        self._choice = gpu_mod.select_variant(bundle.os, bundle.arch, self._probe)
        if installer_factory is None:
            def installer_factory(emit):
                return EngineInstaller(InstallContext.for_bundle(bundle, data), emit=emit)
        self._installer = installer_factory(self._on_event)
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    # -- what the dialog shows ------------------------------------------

    def hardware_text(self) -> str:
        return gpu_mod.describe(self._probe)

    def recommendation_reason(self) -> str:
        return self._choice.reason

    def offers(self) -> list[Offer]:
        offers = []
        if self._choice.is_gpu:
            offers.append(self._offer(self._choice.variant, recommended=True))
        offers.append(self._offer(gpu_mod.CPU, recommended=not self._choice.is_gpu))
        return offers

    def installed_text(self) -> str:
        return installed_text(self._data)

    def _offer(self, variant: str, recommended: bool) -> Offer:
        return Offer(variant, gpu_mod.variant_label(variant),
                     size_text(gpu_mod.download_bytes(variant, self._bundle.os)), recommended)

    # -- running ----------------------------------------------------------

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, variant: str) -> None:
        if self.running():
            return
        self._cancel = threading.Event()
        gpu = self._choice.gpu if variant != gpu_mod.CPU else None
        self._thread = threading.Thread(target=self._work, args=(variant, gpu, self._cancel),
                                        name="engine-install", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _work(self, variant: str, gpu, cancel: threading.Event) -> None:
        try:
            result = self._installer.install(variant, gpu, cancel)
        except Exception as exc:                        # never leave the dialog waiting
            result = InstallResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        self.finished.emit(_outcome(result))

    def _on_event(self, event: InstallEvent) -> None:
        if event.kind == "step":
            self.step.emit(event.text, event.done_bytes, event.total_bytes)
        elif event.kind == "progress":
            self.progress.emit(event.done_bytes, event.total_bytes)
        elif event.kind == "notice":
            self.notice.emit(event.text)
            self.log.emit(event.text)
        else:
            self.log.emit(event.text)


def _outcome(result: InstallResult) -> SetupOutcome:
    return SetupOutcome(ok=result.ok, cancelled=result.cancelled,
                        title=gpu_mod.variant_label(result.variant) if result.variant else "",
                        fell_back=result.fell_back, fallback_reason=result.fallback_reason or "",
                        error=result.error or "", warnings=tuple(result.warnings))


def engine_problem() -> tuple[str, str] | None:
    """(title, text) when this is a bundle without a working engine; None
    in developer mode or when the engine is installed."""
    try:
        bundle = current_bundle()
    except BundleError as exc:
        return ("Broken installation", f"{exc}\n\nReinstall OCR Manager.")
    if bundle is None:
        return None
    if installed_state() is None:
        return ("OCR engine missing",
                f"The OCR engine (PaddleOCR) is not installed in {engine_dir()}.\n\n"
                f"Restart OCR Manager to set it up, or run it with --setup-engine.")
    return None
