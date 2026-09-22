"""Startup: stdout/stderr redirection without a console, the bootstrap's
engine activation, the CLI flags, and the device fallback in videocr.utils."""
from __future__ import annotations

import json
import os
import site
import subprocess
import sys
import types
from pathlib import Path

import pytest

from core.runtime import engine
from core.runtime.engine import EngineState

REPO_ROOT = Path(__file__).resolve().parent.parent


def _child(code: str, env: dict | None = None, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                          env=dict(os.environ, **(env or {})), timeout=120, check=False, **kwargs)


# --------------------------------------------------------------------------
# core.runtime.logfile (in child processes: it dup2()s fds 1 and 2)
# --------------------------------------------------------------------------

PYTHONW = """
import os, sys
os.close(1); os.close(2)
sys.stdout = None; sys.stderr = None          # what pythonw.exe gives a program
from core.runtime.logfile import needs_redirect, redirect_if_needed
assert needs_redirect()
path = redirect_if_needed(__import__('pathlib').Path(sys.argv[1]))
print('python-level stdout')
sys.stderr.write('python-level stderr\\n')
os.write(1, b'fd 1\\n'); os.write(2, b'fd 2\\n')
from videocr.utils import suppress_output
with suppress_output():                      # os.dup(1)/os.dup(2) must work now
    os.write(1, b'swallowed\\n')
print('after suppress_output', path is not None)
"""


def test_no_console_output_goes_to_the_log_file(tmp_path):
    log = tmp_path / "logs" / "ocr-manager.log"
    log.parent.mkdir()
    log.write_text("previous run\n")
    done = subprocess.run([sys.executable, "-c", PYTHONW, str(log)], cwd=REPO_ROOT, capture_output=True,
                          text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    text = log.read_text()
    for expected in ("python-level stdout", "python-level stderr", "fd 1", "fd 2", "after suppress_output True"):
        assert expected in text
    assert "swallowed" not in text
    assert (tmp_path / "logs" / "ocr-manager.log.1").read_text() == "previous run\n"


def test_devnull_output_counts_as_nowhere_only_when_asked():
    code = ("from core.runtime.logfile import needs_redirect; "
            "import sys; sys.exit(10 * needs_redirect() + needs_redirect(discard_devnull=True))")
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, timeout=60, check=False)
    assert done.returncode == 1
    assert _child(code).returncode == 0                  # pipes are real output


# --------------------------------------------------------------------------
# app.bootstrap
# --------------------------------------------------------------------------

@pytest.fixture
def bundle_env(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "bundle.json").write_text(json.dumps({"version": "1.0.0", "os": "linux", "arch": "x86_64"}))
    (root / "constraints.txt").write_text("numpy==2.2.6\n")
    data = tmp_path / "data"
    monkeypatch.setenv("OCR_MANAGER_BUNDLE", str(root))
    monkeypatch.setenv("OCR_MANAGER_DATA_DIR", str(data))
    saved_path = list(sys.path)
    saved_site = (site.ENABLE_USER_SITE, site.USER_BASE, site.USER_SITE)
    saved_base = os.environ.get("PYTHONUSERBASE")
    yield root, data
    sys.path[:] = saved_path
    site.ENABLE_USER_SITE, site.USER_BASE, site.USER_SITE = saved_site
    if saved_base is None:
        os.environ.pop("PYTHONUSERBASE", None)
    else:
        os.environ["PYTHONUSERBASE"] = saved_base


def _install_fake_engine(data: Path, variant: str = "cpu") -> Path:
    directory = engine.engine_dir(data)
    engine.site_dir(directory).mkdir(parents=True)
    engine.write_state(directory, EngineState(variant=variant, verified=True, models_ready=True))
    return directory


def test_boot_in_developer_mode_does_nothing(monkeypatch):
    from app.bootstrap import boot

    monkeypatch.delenv("OCR_MANAGER_BUNDLE", raising=False)
    started = boot()
    assert not started.bundled and started.state is None and started.log_path is None
    assert not started.needs_engine


def test_boot_in_a_bundle_activates_the_installed_engine(bundle_env):
    from app.bootstrap import boot

    _root, data = bundle_env
    started = boot()
    assert started.bundled and started.needs_engine
    directory = _install_fake_engine(data)
    started = boot()
    assert started.state.variant == "cpu" and not started.needs_engine
    assert site.USER_SITE == str(engine.site_dir(directory))


def test_boot_reports_a_broken_bundle(monkeypatch, tmp_path):
    from app.bootstrap import boot

    monkeypatch.setenv("OCR_MANAGER_BUNDLE", str(tmp_path / "nowhere"))
    started = boot()
    assert not started.bundled and "not a directory" in started.bundle_error


def test_engine_problem_in_a_bundle_without_an_engine(bundle_env):
    from app.engine import engine_problem

    _root, data = bundle_env
    title, text = engine_problem()
    assert title == "OCR engine missing" and "--setup-engine" in text
    _install_fake_engine(data)
    assert engine_problem() is None


def test_engine_problem_in_developer_mode(monkeypatch):
    from app.engine import engine_problem

    monkeypatch.delenv("OCR_MANAGER_BUNDLE", raising=False)
    assert engine_problem() is None


# --------------------------------------------------------------------------
# CLI flags (own processes)
# --------------------------------------------------------------------------

def test_version_flag():
    done = subprocess.run([sys.executable, "main.py", "--version"], cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=120, check=False)
    from app.version import __version__
    assert done.returncode == 0 and done.stdout.strip() == __version__


def test_self_test_flag_reports_json():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env.pop("OCR_MANAGER_BUNDLE", None)
    done = subprocess.run([sys.executable, "-m", "app", "--self-test"], cwd=REPO_ROOT, capture_output=True,
                          text=True, timeout=300, check=False, env=env)
    report = json.loads(done.stdout)
    assert report["imports"]["failed"] == {} and report["imports"]["count"] > 50
    assert set(report["tools"]) == {"ffmpeg", "ffprobe"}
    assert report["bundle"] is None and report["engine"] == {"bundled": False}
    assert done.returncode == (0 if report["ok"] else 1)


def test_install_engine_needs_a_bundle():
    env = dict(os.environ)
    env.pop("OCR_MANAGER_BUNDLE", None)
    done = subprocess.run([sys.executable, "main.py", "--install-engine", "cpu"], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=120, check=False, env=env)
    assert done.returncode == 2 and "needs a portable bundle" in done.stdout


def test_install_engine_gpu_without_a_gpu_build_exits_3(bundle_env, monkeypatch, capsys):
    from app import cli
    from app.bootstrap import boot
    from core.runtime import gpu

    monkeypatch.setattr(gpu, "probe_gpus", lambda os_name: gpu.GpuProbe(error="no NVIDIA driver found"))
    assert cli.install_engine(boot(), "gpu") == cli.EXIT_NO_GPU
    out = capsys.readouterr().out
    assert "no NVIDIA driver found" in out and out.strip().splitlines()[-1].startswith("RESULT ")


def test_ocr_smoke_without_an_engine_in_a_bundle_fails_fast(bundle_env, capsys):
    from app import cli
    from app.bootstrap import boot

    assert cli.ocr_smoke(boot()) == cli.EXIT_FAIL
    assert "no OCR engine is installed" in capsys.readouterr().out


def test_smoke_line_renders_white_text_on_black(qapp):
    from app.cli import SMOKE_TEXT_CJK, SMOKE_TEXT_LATIN, render_line

    image, text, _family = render_line()
    assert image.shape == (128, 1280, 3) and image.dtype.name == "uint8"
    assert text in (SMOKE_TEXT_CJK, SMOKE_TEXT_LATIN)
    assert image.max() > 200 and (image == 0).mean() > 0.8


@pytest.fixture
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------
# videocr.utils: the device an engine is built on
# --------------------------------------------------------------------------

@pytest.fixture
def fresh_cuda_check(monkeypatch):
    from videocr import utils
    monkeypatch.setattr(utils, "_cuda_usable", None)
    return utils


def _fake_paddle(monkeypatch, compiled: bool, count: int):
    cuda = types.SimpleNamespace(device_count=lambda: count)
    device = types.SimpleNamespace(is_compiled_with_cuda=lambda: compiled, cuda=cuda)
    monkeypatch.setitem(sys.modules, "paddle", types.SimpleNamespace(device=device))


@pytest.mark.parametrize("compiled, count, use_gpu, expected", [
    (True, 1, True, "gpu"),
    (True, 1, False, "cpu"),
    (False, 0, True, "cpu"),          # a CPU paddle
    (True, 0, True, "cpu"),           # a CUDA paddle with no device
])
def test_resolve_device(fresh_cuda_check, monkeypatch, compiled, count, use_gpu, expected):
    _fake_paddle(monkeypatch, compiled, count)
    assert fresh_cuda_check.resolve_device(use_gpu) == expected


def test_a_broken_paddle_resolves_to_cpu(fresh_cuda_check, monkeypatch):
    monkeypatch.setitem(sys.modules, "paddle", None)             # import paddle raises ImportError
    assert fresh_cuda_check.resolve_device(True) == "cpu"


def test_the_cuda_check_runs_once(fresh_cuda_check, monkeypatch):
    calls = []
    cuda = types.SimpleNamespace(device_count=lambda: calls.append(1) or 1)
    monkeypatch.setitem(sys.modules, "paddle", types.SimpleNamespace(
        device=types.SimpleNamespace(is_compiled_with_cuda=lambda: True, cuda=cuda)))
    assert [fresh_cuda_check.resolve_device(True) for _ in range(3)] == ["gpu"] * 3
    assert calls == [1]


def test_engines_are_built_with_the_resolved_device(fresh_cuda_check, monkeypatch):
    """GPU: exactly `device="gpu"` as before (fidelity). CPU: oneDNN off."""
    built = []

    class Fake:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=Fake, TextDetection=Fake))
    _fake_paddle(monkeypatch, True, 1)
    utils = fresh_cuda_check
    utils.create_ocr_engine("ch", None, None, True)
    utils.create_detection_engine(None, True)
    assert built[0]["device"] == "gpu" and "enable_mkldnn" not in built[0]
    assert built[1] == {"model_name": "PP-OCRv5_server_det", "model_dir": None, "device": "gpu"}
    built.clear()
    monkeypatch.setattr(utils, "_cuda_usable", False)
    utils.create_ocr_engine("ch", None, None, True)
    utils.create_detection_engine(None, True)
    assert all(kwargs["device"] == "cpu" and kwargs["enable_mkldnn"] is False for kwargs in built)
