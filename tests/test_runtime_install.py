"""core.runtime.install: pip -> verify -> prefetch, the GPU -> CPU fallback,
cancel and cleanup -- with a fake process runner (no network, no paddle)."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from core.runtime import engine
from core.runtime.engine import EngineState
from core.runtime.gpu import GpuInfo
from core.runtime.install import EngineInstaller, InstallContext
from core.runtime.procs import Cancelled, ProcessResult, SubprocessRunner

GPU = GpuInfo("NVIDIA GeForce RTX 4090", "580.65.06", "8.9")


class FakeRunner:
    """Answers each kind of command the installer runs.

    `script` maps a kind ("pip-version", "pip-gpu", "pip-cpu", "verify-gpu",
    "verify-cpu", "prefetch-gpu", "prefetch-cpu") to a list of answers used
    in order; an answer is a dict: {"code": int, "lines": [...],
    "result": {...} (a RESULT line), "cancel": True (raise Cancelled),
    "timeout": True}. A successful pip answer creates the site dir, like pip."""

    def __init__(self, script: dict | None = None):
        self.script = {key: list(value) for key, value in (script or {}).items()}
        self.calls: list[tuple[str, list[str], dict]] = []

    @staticmethod
    def kind(command: list[str]) -> str:
        if "pip" in command and "--version" in command:
            return "pip-version"
        if "pip" in command and "install" in command:
            return "pip-gpu" if command[-1].startswith("paddlepaddle-gpu") else "pip-cpu"
        check = command[command.index("core.runtime.checks") + 1]
        return f"{check}-{command[command.index('--device') + 1]}"

    def run(self, command, *, env=None, cwd=None, on_line=lambda line: None, cancel=None, timeout=None):
        kind = self.kind(list(command))
        self.calls.append((kind, list(command), {"env": env, "cwd": cwd, "timeout": timeout}))
        answers = self.script.get(kind)
        answer = answers.pop(0) if answers else {}
        if answer.get("cancel"):
            raise Cancelled()
        lines = list(answer.get("lines", []))
        if kind == "pip-version":
            lines = lines or ["pip 26.0.1 from /bundle/python/lib/python3.12/site-packages/pip (python 3.12)"]
        code = answer.get("code", 0)
        if kind.startswith("pip-") and kind != "pip-version" and code == 0:
            engine.site_dir(Path(env["PYTHONUSERBASE"])).mkdir(parents=True, exist_ok=True)
            marker = Path(env["PYTHONUSERBASE"]) / f"installed-{kind}"
            marker.write_text("x")
            lines = lines or ["Downloading https://x/paddle.whl (100 MB)", "Progress 50 of 100",
                              "Progress 100 of 100", "Successfully installed paddle"]
        if kind.startswith(("verify", "prefetch")):
            result = answer.get("result", {"ok": code == 0})
            lines.append("RESULT " + json.dumps(result))
        for line in lines:
            on_line(line)
        return ProcessResult(code, lines[-40:], bool(answer.get("timeout")))

    def kinds(self) -> list[str]:
        return [kind for kind, _command, _extra in self.calls]


@pytest.fixture
def context(tmp_path):
    constraints = tmp_path / "bundle" / "constraints.txt"
    constraints.parent.mkdir()
    constraints.write_text("numpy==2.2.6\npaddlepaddle==3.3.0\nnvidia-cudnn-cu12==9.9.0.52\n")
    return InstallContext(os_name="linux", arch="x86_64", python="/bundle/python/bin/python3",
                          data=tmp_path / "data", constraints=constraints, source=tmp_path / "src",
                          env={"PATH": "/bin", "PYTHONNOUSERSITE": "1"}, py=(3, 12))


def install(context, runner, variant, gpu=None, cancel=None):
    events = []
    result = EngineInstaller(context, runner, emit=events.append).install(variant, gpu, cancel)
    return result, events


def test_cpu_install_runs_pip_verify_prefetch_and_promotes(context):
    runner = FakeRunner()
    (context.data / "engine" / "py311-paddle3.2.0").mkdir(parents=True)          # an older engine
    result, events = install(context, runner, "cpu")
    assert result.ok and result.variant == "cpu" and not result.fell_back
    assert runner.kinds() == ["pip-version", "pip-cpu", "verify-cpu", "prefetch-cpu"]
    final = engine.engine_dir(context.data)
    assert result.engine_dir == final and (final / "installed-pip-cpu").exists()
    assert not engine.partial_dir(context.data).exists()
    assert sorted(p.name for p in (context.data / "engine").iterdir()) == [final.name]
    state = engine.installed_state(context.data)
    assert state.variant == "cpu" and state.verified and state.models_ready and state.fallback_reason is None
    kinds = {event.kind for event in events}
    assert {"step", "log", "progress"} <= kinds
    progress = [event for event in events if event.kind == "progress"]
    assert progress[-1].done_bytes == 100 and progress[-1].total_bytes >= 100


def test_pip_runs_as_a_user_install_into_the_partial_dir(context):
    runner = FakeRunner()
    install(context, runner, "cpu")
    _kind, command, extra = runner.calls[1]
    partial = engine.partial_dir(context.data)
    assert command[0] == "/bundle/python/bin/python3" and "--user" in command
    assert command[-1] == "paddlepaddle==3.3.0"
    assert extra["env"]["PYTHONUSERBASE"] == str(partial)
    assert "PYTHONNOUSERSITE" not in extra["env"]
    assert extra["env"]["TMPDIR"] == str(context.data / "engine" / "tmp")
    constraints = Path(command[command.index("-c") + 1])
    assert constraints == partial / "constraints.txt"


def test_checks_run_isolated_in_the_source_root(context):
    runner = FakeRunner()
    install(context, runner, "cpu")
    for kind, command, extra in runner.calls[2:]:
        assert command[:5] == ["/bundle/python/bin/python3", "-s", "-E", "-X", "utf8"]
        assert command[5:7] == ["-m", "core.runtime.checks"]
        assert command[command.index("--engine") + 1] == str(engine.partial_dir(context.data))
        assert extra["cwd"] == context.source and extra["timeout"]


def test_gpu_install_success(context):
    runner = FakeRunner()
    result, _events = install(context, runner, "cu129", GPU)
    assert result.ok and result.variant == "cu129" and not result.fell_back
    assert runner.kinds() == ["pip-version", "pip-gpu", "verify-gpu", "prefetch-gpu"]
    assert runner.calls[1][1][-1].startswith("paddlepaddle-gpu @ https://paddle-whl.cdn.bcebos.com/stable/cu129/")
    state = engine.installed_state(context.data)
    assert (state.variant, state.gpu_name, state.driver) == ("cu129", GPU.name, GPU.driver)


@pytest.mark.parametrize("failure", [
    {"pip-gpu": [{"code": 1, "lines": ["ERROR: HTTP error 404"]}]},
    {"verify-gpu": [{"code": 1, "result": {"ok": False, "error": "RuntimeError: paddle sees no CUDA device"}}]},
    {"verify-gpu": [{"code": -11, "lines": ["Segmentation fault"]}]},
    {"verify-gpu": [{"timeout": True, "code": -9}]},
    {"prefetch-gpu": [{"code": 1, "result": {"ok": False, "error": "cuDNN error", "models": True}}]},
], ids=["pip", "verify-error", "verify-crash", "verify-timeout", "prefetch-device"])
def test_a_failing_gpu_build_falls_back_to_cpu(context, failure):
    runner = FakeRunner(failure)
    result, events = install(context, runner, "cu129", GPU)
    assert result.ok and result.variant == "cpu" and result.fell_back and result.fallback_reason
    assert runner.kinds()[-3:] == ["pip-cpu", "verify-cpu", "prefetch-cpu"]
    final = engine.engine_dir(context.data)
    assert (final / "installed-pip-cpu").exists()
    assert not (final / "installed-pip-gpu").exists()                    # the GPU attempt was wiped
    state = engine.installed_state(context.data)
    assert state.variant == "cpu" and state.fallback_reason == result.fallback_reason
    assert state.gpu_name == GPU.name
    notices = [event.text for event in events if event.kind == "notice"]
    assert any("Installing the CPU build instead" in text for text in notices)


def test_a_model_download_failure_does_not_fail_the_install(context):
    runner = FakeRunner({"prefetch-gpu": [{"code": 1, "result": {"ok": False, "error": "ConnectionError",
                                                                  "models": False}}]})
    result, _events = install(context, runner, "cu129", GPU)
    assert result.ok and result.variant == "cu129" and not result.fell_back
    assert result.warnings and "ConnectionError" in result.warnings[0]
    assert engine.installed_state(context.data).models_ready is False


def test_when_cpu_fails_too_nothing_is_promoted_and_the_old_engine_stays(context):
    final = engine.engine_dir(context.data)
    engine.site_dir(final).mkdir(parents=True)
    engine.write_state(final, EngineState(variant="cu126", verified=True))
    runner = FakeRunner({"pip-gpu": [{"code": 1}], "pip-cpu": [{"code": 1, "lines": ["ERROR: no space"]}]})
    result, _events = install(context, runner, "cu129", GPU)
    assert not result.ok and "pip failed" in result.error and result.fell_back
    assert not engine.partial_dir(context.data).exists()
    assert engine.installed_state(context.data).variant == "cu126"


def test_cancel_removes_the_partial_dir_and_keeps_the_old_engine(context):
    final = engine.engine_dir(context.data)
    engine.site_dir(final).mkdir(parents=True)
    engine.write_state(final, EngineState(variant="cpu", verified=True))
    runner = FakeRunner({"verify-gpu": [{"cancel": True}]})
    result, _events = install(context, runner, "cu129", GPU)
    assert not result.ok and result.cancelled
    assert runner.kinds() == ["pip-version", "pip-gpu", "verify-gpu"]          # no fallback after a cancel
    assert not engine.partial_dir(context.data).exists()
    assert not (context.data / "engine" / "tmp").exists()
    assert engine.installed_state(context.data).variant == "cpu"


def test_no_pip_is_a_clear_error(context):
    runner = FakeRunner({"pip-version": [{"code": 1, "lines": ["No module named pip"]}]})
    result, _events = install(context, runner, "cpu")
    assert not result.ok and "pip is not available" in result.error
    assert runner.kinds() == ["pip-version"]


def test_not_enough_disk_space(context, monkeypatch):
    import core.runtime.install as install_module

    class Usage:
        free = 10_000_000

    monkeypatch.setattr(install_module.shutil, "disk_usage", lambda path: Usage)
    result, _events = install(context, FakeRunner(), "cpu")
    assert not result.ok and "not enough disk space" in result.error


def test_old_pip_gets_no_raw_progress(context):
    runner = FakeRunner({"pip-version": [{"lines": ["pip 23.0 from /x (python 3.12)"]}]})
    install(context, runner, "cpu")
    command = runner.calls[1][1]
    assert command[command.index("--progress-bar") + 1] == "off"


# --------------------------------------------------------------------------
# The real subprocess runner
# --------------------------------------------------------------------------

def test_subprocess_runner_streams_lines_and_returns_the_code():
    seen = []
    code = "import sys; print('one'); print('two', flush=True); sys.stderr.write('three\\n'); sys.exit(3)"
    result = SubprocessRunner().run([sys.executable, "-c", code], on_line=seen.append)
    assert sorted(seen) == ["one", "three", "two"] and result.returncode == 3 and not result.ok


def test_subprocess_runner_cancel_ends_the_child():
    cancel = threading.Event()
    seen = []

    def on_line(line):
        seen.append(line)
        cancel.set()

    code = "import time; print('started', flush=True); time.sleep(60)"
    with pytest.raises(Cancelled):
        SubprocessRunner().run([sys.executable, "-c", code], on_line=on_line, cancel=cancel)
    assert seen == ["started"]


def test_subprocess_runner_timeout():
    result = SubprocessRunner().run([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.5)
    assert result.timed_out and not result.ok
