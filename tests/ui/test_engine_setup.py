"""The first-run engine setup dialog (app/views/engine_setup.py), driven by a
fake setup service, and the real EngineSetup service over a fake installer."""
from __future__ import annotations

import json
import time

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog

from app.engine import EngineSetup, Offer, SetupOutcome, size_text
from app.views.engine_setup import (
    CANCELLING_TEXT,
    CLOSE_TEXT,
    CONTINUE_TEXT,
    INSTALL_TEXT,
    QUIT_TEXT,
    RETRY_TEXT,
    EngineSetupDialog,
)
from core.runtime.gpu import GpuInfo, GpuProbe
from core.runtime.install import InstallEvent, InstallResult
from core.runtime.paths import Bundle


class FakeSetup(QObject):
    step = pyqtSignal(str, object, object)
    progress = pyqtSignal(object, object)
    log = pyqtSignal(str)
    notice = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, gpu: bool = True, installed: str = ""):
        super().__init__()
        self.gpu = gpu
        self.installed = installed
        self.started: list[str] = []
        self.cancelled = 0

    def hardware_text(self):
        return "NVIDIA GeForce RTX 4090 · driver 580.65.06 · compute 8.9" if self.gpu else "no NVIDIA driver found"

    def recommendation_reason(self):
        return "RTX 4090 with driver 580.65.06 runs the CUDA 12.9 build" if self.gpu else "no NVIDIA driver found"

    def offers(self):
        cpu = Offer("cpu", "CPU build", "about 210 MB to download", recommended=not self.gpu)
        if not self.gpu:
            return [cpu]
        return [Offer("cu129", "GPU build (CUDA 12.9)", "about 5.4 GB to download", recommended=True), cpu]

    def installed_text(self):
        return self.installed

    def start(self, variant):
        self.started.append(variant)

    def cancel(self):
        self.cancelled += 1

    def running(self):
        return False


@pytest.fixture
def make_dialog(qapp):
    dialogs = []

    def make(**kwargs):
        have_engine = kwargs.pop("have_engine", False)
        setup = FakeSetup(**kwargs)
        dialog = EngineSetupDialog(setup, have_engine=have_engine)
        dialog.show()
        dialogs.append(dialog)
        return dialog, setup

    yield make
    for dialog in dialogs:
        dialog._state = "choose"                    # let it close
        dialog.close()
        dialog.deleteLater()


def test_shows_hardware_and_recommends_the_gpu_build(make_dialog):
    dialog, _setup = make_dialog()
    assert dialog.hardware_row.value().startswith("NVIDIA GeForce RTX 4090")
    assert dialog.installed_row.value() == "nothing yet"
    assert list(dialog.option_buttons) == ["cu129", "cpu"]
    assert dialog.selected() == "cu129"
    gpu_text = dialog.option_buttons["cu129"].text()
    assert "GPU build (CUDA 12.9)" in gpu_text and "recommended" in gpu_text and "5.4 GB" in gpu_text
    assert "210 MB" in dialog.option_buttons["cpu"].text()
    assert dialog.option_buttons["cu129"].property("toggled") is True
    assert dialog.install_button.text() == INSTALL_TEXT and dialog.cancel_button.text() == QUIT_TEXT
    assert dialog.progress_area.isHidden() and dialog.log_view.isHidden()


def test_cpu_only_machine_offers_cpu(make_dialog):
    dialog, setup = make_dialog(gpu=False)
    assert list(dialog.option_buttons) == ["cpu"] and dialog.selected() == "cpu"
    dialog.install_button.click()
    assert setup.started == ["cpu"]


def test_picking_cpu_and_installing(make_dialog):
    dialog, setup = make_dialog()
    dialog.option_buttons["cpu"].click()
    assert dialog.selected() == "cpu"
    assert dialog.option_buttons["cpu"].property("toggled") is True
    assert dialog.option_buttons["cu129"].property("toggled") is False
    dialog.install_button.click()
    assert setup.started == ["cpu"] and dialog.state() == "installing"
    assert not dialog.install_button.isEnabled()
    assert not any(button.isEnabled() for button in dialog.option_buttons.values())
    assert dialog.cancel_button.text() == "Cancel" and not dialog.progress_area.isHidden()
    dialog.select("cu129")                                  # ignored while installing
    assert dialog.selected() == "cpu"


def test_progress_step_and_log(make_dialog):
    dialog, setup = make_dialog()
    dialog.install_button.click()
    setup.step.emit("Downloading paddlepaddle_gpu-3.3.0.whl", 0, 5_400_000_000)
    setup.progress.emit(2_700_000_000, 5_400_000_000)
    assert dialog.step_label.text() == "Downloading paddlepaddle_gpu-3.3.0.whl"
    assert dialog.progress_bar.maximum() == 1000 and dialog.progress_bar.value() == 500
    assert dialog.bytes_label.text() == "2.70 GB of about 5.40 GB"
    setup.step.emit("Checking the GPU build", 0, 0)
    assert dialog.progress_bar.maximum() == 0               # busy
    setup.log.emit("Successfully installed paddlepaddle-gpu-3.3.0")
    assert "Successfully installed" in dialog.log_view.toPlainText()
    dialog.details_button.click()
    assert not dialog.log_view.isHidden() and dialog.details_button.text().startswith("▾")


def test_success_with_gpu_fallback_says_so_and_continue_accepts(make_dialog):
    dialog, setup = make_dialog()
    dialog.install_button.click()
    setup.notice.emit("The GPU build (CUDA 12.9) did not work: the check failed. Installing the CPU build instead.")
    assert not dialog.notice_label.isHidden()
    setup.finished.emit(SetupOutcome(ok=True, title="CPU build", fell_back=True,
                                     fallback_reason="the check failed: paddle sees no CUDA device"))
    assert dialog.state() == "done"
    assert dialog.step_label.text() == "Installed: CPU build"
    assert "CPU build was installed instead" in dialog.notice_label.text()
    assert "paddle sees no CUDA device" in dialog.notice_label.text()
    assert dialog.install_button.text() == CONTINUE_TEXT and dialog.cancel_button.isHidden()
    dialog.install_button.click()
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_failure_offers_a_retry(make_dialog):
    dialog, setup = make_dialog()
    dialog.install_button.click()
    setup.finished.emit(SetupOutcome(ok=False, error="pip failed (exit code 1: ERROR: no space left)"))
    assert dialog.state() == "failed" and dialog.install_button.text() == RETRY_TEXT
    assert dialog.notice_label.property("tone") == "bad" and "no space left" in dialog.notice_label.text()
    assert all(button.isEnabled() for button in dialog.option_buttons.values())
    dialog.install_button.click()
    assert setup.started == ["cu129", "cu129"]


def test_cancel_while_installing(make_dialog):
    dialog, setup = make_dialog()
    dialog.install_button.click()
    dialog.cancel_button.click()
    assert setup.cancelled == 1 and dialog.cancel_button.text() == CANCELLING_TEXT
    assert not dialog.cancel_button.isEnabled() and dialog.isVisible()
    setup.finished.emit(SetupOutcome(ok=False, cancelled=True, error="Cancelled"))
    assert dialog.state() == "cancelled" and "Nothing was changed" in dialog.notice_label.text()
    assert dialog.install_button.text() == INSTALL_TEXT and dialog.cancel_button.text() == QUIT_TEXT


def test_closing_while_installing_cancels_first(make_dialog):
    dialog, setup = make_dialog()
    dialog.install_button.click()
    dialog.close()
    assert dialog.isVisible() and setup.cancelled == 1
    dialog.reject()                                         # Esc
    assert dialog.isVisible()
    setup.finished.emit(SetupOutcome(ok=False, cancelled=True))
    dialog.close()
    assert not dialog.isVisible() and dialog.result() == QDialog.DialogCode.Rejected


def test_quit_rejects_and_a_reinstall_says_close(make_dialog):
    dialog, _setup = make_dialog()
    dialog.cancel_button.click()
    assert dialog.result() == QDialog.DialogCode.Rejected
    again, _setup = make_dialog(have_engine=True, installed="GPU build (CUDA 12.9), installed 2026-09-20")
    assert again.cancel_button.text() == CLOSE_TEXT
    assert again.installed_row.value().startswith("GPU build (CUDA 12.9)")


def test_size_text():
    assert size_text(5_400_000_000) == "about 5.4 GB to download"
    assert size_text(210_000_000) == "about 210 MB to download"


# --------------------------------------------------------------------------
# The real service, over a fake installer
# --------------------------------------------------------------------------

class FakeInstaller:
    def __init__(self, emit, result=None):
        self.emit = emit
        self.result = result or InstallResult(ok=True, variant="cu129")
        self.calls = []

    def install(self, variant, gpu, cancel):
        self.calls.append((variant, gpu))
        self.emit(InstallEvent("step", "Downloading", 0, 100))
        self.emit(InstallEvent("progress", "", 40, 100))
        self.emit(InstallEvent("log", "pip says hi"))
        self.emit(InstallEvent("notice", "heads up"))
        return self.result


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        QTest.qWait(10)
    return predicate()


def test_the_service_runs_the_installer_off_the_gui_thread(qapp, tmp_path):
    bundle = Bundle(tmp_path, "1.0.0", "linux", "x86_64")
    probe = GpuProbe((GpuInfo("NVIDIA GeForce RTX 4090", "580.65.06", "8.9"),))
    made = []

    def factory(emit):
        made.append(FakeInstaller(emit))
        return made[-1]

    setup = EngineSetup(bundle, data=tmp_path / "data", probe=probe, installer_factory=factory)
    assert [offer.variant for offer in setup.offers()] == ["cu129", "cpu"]
    assert setup.offers()[0].recommended and setup.installed_text() == ""
    assert "RTX 4090" in setup.hardware_text()
    seen = {"step": [], "progress": [], "log": [], "notice": [], "finished": []}

    class Sink(QObject):
        def on_step(self, *args): seen["step"].append(args)
        def on_progress(self, *args): seen["progress"].append(args)
        def on_log(self, line): seen["log"].append(line)
        def on_notice(self, text): seen["notice"].append(text)
        def on_finished(self, outcome): seen["finished"].append(outcome)

    sink = Sink()
    setup.step.connect(sink.on_step)
    setup.progress.connect(sink.on_progress)
    setup.log.connect(sink.on_log)
    setup.notice.connect(sink.on_notice)
    setup.finished.connect(sink.on_finished)
    setup.start("cu129")
    assert wait_until(lambda: seen["finished"])
    setup.wait(5)
    assert made[0].calls == [("cu129", probe.primary)]
    assert seen["step"] == [("Downloading", 0, 100)] and seen["progress"] == [(40, 100)]
    assert seen["log"] == ["pip says hi", "heads up"] and seen["notice"] == ["heads up"]
    outcome = seen["finished"][0]
    assert outcome.ok and outcome.title == "GPU build (CUDA 12.9)"


def test_the_service_on_a_mac_offers_cpu_only(qapp, tmp_path):
    setup = EngineSetup(Bundle(tmp_path, "1.0.0", "mac", "arm64"), data=tmp_path, probe=GpuProbe(),
                        installer_factory=lambda emit: FakeInstaller(emit))
    assert [offer.variant for offer in setup.offers()] == ["cpu"]
    assert setup.recommendation_reason() == "paddle has no GPU build for macOS"


def test_the_installed_text_reads_the_state(qapp, tmp_path):
    from core.runtime import engine
    from core.runtime.engine import EngineState

    directory = engine.engine_dir(tmp_path)
    engine.site_dir(directory).mkdir(parents=True)
    engine.write_state(directory, EngineState(variant="cpu", verified=True, installed_at="2026-09-20T10:00:00Z",
                                              fallback_reason="pip failed"))
    setup = EngineSetup(Bundle(tmp_path, "1.0.0", "linux", "x86_64"), data=tmp_path, probe=GpuProbe(),
                        installer_factory=lambda emit: FakeInstaller(emit))
    assert setup.installed_text() == "CPU build, installed 2026-09-20 (the GPU build did not work)"
    assert json.loads((directory / "state.json").read_text())["variant"] == "cpu"
