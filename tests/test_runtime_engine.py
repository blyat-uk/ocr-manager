"""core.runtime: data dir, bundle detection, the engine dir, activation,
state.json, the pip command and reading pip's output."""
from __future__ import annotations

import json
import os
import site
import subprocess
import sys
from pathlib import Path

import pytest

from core.runtime import engine, pip
from core.runtime.checks import models_present, parse_result
from core.runtime.engine import EngineState
from core.runtime.paths import (
    BUNDLE_ENV,
    DATA_DIR_ENV,
    BundleError,
    current_bundle,
    data_dir,
    host_arch,
    host_os,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def test_data_dir_per_os(tmp_path):
    home = tmp_path / "home"
    assert data_dir({}, "linux", home) == home / ".local" / "share" / "ocr-manager"
    assert data_dir({"XDG_DATA_HOME": "/xdg"}, "linux", home) == Path("/xdg/ocr-manager")
    assert data_dir({"LOCALAPPDATA": "C:/Users/u/AppData/Local"}, "win", home) == \
        Path("C:/Users/u/AppData/Local/ocr-manager")
    assert data_dir({}, "win", home) == home / "AppData" / "Local" / "ocr-manager"
    assert data_dir({}, "mac", home) == home / "Library" / "Application Support" / "ocr-manager"
    for os_name in ("linux", "win", "mac"):
        assert data_dir({DATA_DIR_ENV: str(tmp_path / "d")}, os_name, home) == tmp_path / "d"


def test_host_names():
    assert host_os("win32") == "win" and host_os("darwin") == "mac" and host_os("linux") == "linux"
    assert host_arch("AMD64") == "x86_64" and host_arch("aarch64") == "arm64" and host_arch("arm64") == "arm64"


def test_current_bundle(tmp_path):
    assert current_bundle({}) is None
    (tmp_path / "bundle.json").write_text(json.dumps({"version": "1.0.0", "os": "win", "arch": "x86_64"}))
    bundle = current_bundle({BUNDLE_ENV: str(tmp_path)})
    assert (bundle.root, bundle.version, bundle.os, bundle.arch) == (tmp_path, "1.0.0", "win", "x86_64")
    assert bundle.constraints == tmp_path / "constraints.txt"
    with pytest.raises(BundleError, match="not a directory"):
        current_bundle({BUNDLE_ENV: str(tmp_path / "missing")})
    (tmp_path / "bundle.json").write_text("{nope")
    with pytest.raises(BundleError, match="unreadable"):
        current_bundle({BUNDLE_ENV: str(tmp_path)})
    (tmp_path / "bundle.json").unlink()
    with pytest.raises(BundleError, match="missing"):
        current_bundle({BUNDLE_ENV: str(tmp_path)})


# --------------------------------------------------------------------------
# engine dir, state.json
# --------------------------------------------------------------------------

def test_engine_dir_is_versioned_by_python_and_paddle(tmp_path):
    assert engine.engine_tag((3, 12)) == "py312-paddle3.3.0"
    assert engine.engine_dir(tmp_path) == tmp_path / "engine" / engine.engine_tag()
    assert engine.partial_dir(tmp_path).name == engine.engine_tag() + ".partial"


def test_site_dir_is_where_pip_user_installs(tmp_path):
    """The same path pip computes for `--user` with PYTHONUSERBASE set."""
    base = tmp_path / "base"
    code = "import sysconfig; print(sysconfig.get_path('purelib', sysconfig.get_preferred_scheme('user')))"
    shown = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env=dict(os.environ, PYTHONUSERBASE=str(base))).stdout.strip()
    assert engine.site_dir(base) == Path(shown)
    assert str(engine.site_dir(base)).startswith(str(base))


def test_state_round_trip(tmp_path):
    state = EngineState(variant="cu129", verified=True, models_ready=True, gpu_name="RTX 4090", driver="580.65.06")
    engine.write_state(tmp_path, state)
    data = json.loads((tmp_path / "state.json").read_text())
    assert data["variant"] == "cu129" and data["paddle"] == "3.3.0" and data["schema"] == 1
    assert data["installed_at"].endswith("Z")
    assert engine.read_state(tmp_path) == state
    assert not (tmp_path / "state.json.tmp").exists()


def test_unreadable_or_unknown_state_is_no_state(tmp_path):
    assert engine.read_state(tmp_path) is None
    (tmp_path / "state.json").write_text("[1, 2]")
    assert engine.read_state(tmp_path) is None
    (tmp_path / "state.json").write_text("{\"verified\": true}")
    assert engine.read_state(tmp_path) is None
    (tmp_path / "state.json").write_text(json.dumps({"variant": "cpu", "verified": True, "future_field": 1}))
    assert engine.read_state(tmp_path).variant == "cpu"


def test_installed_state_needs_verified_state_and_site_dir(tmp_path):
    directory = engine.engine_dir(tmp_path)
    engine.write_state(directory, EngineState(variant="cpu", verified=False))
    assert engine.installed_state(tmp_path) is None
    engine.write_state(directory, EngineState(variant="cpu", verified=True))
    assert engine.installed_state(tmp_path) is None                 # no site-packages
    engine.site_dir(directory).mkdir(parents=True)
    assert engine.installed_state(tmp_path).variant == "cpu"


def test_promote_replaces_the_old_engine_and_old_versions_are_removed(tmp_path):
    root = tmp_path / "engine"
    final, partial = root / "py312-paddle3.3.0", root / "py312-paddle3.3.0.partial"
    (final / "old").mkdir(parents=True)
    (partial / "new").mkdir(parents=True)
    (root / "py311-paddle3.2.0" / "x").mkdir(parents=True)
    (root / "tmp").mkdir()
    (root / "notes.txt").write_text("a file is not an engine")
    engine.promote(partial, final)
    assert (final / "new").exists() and not (final / "old").exists() and not partial.exists()
    removed = engine.remove_other_engines(final)
    assert sorted(path.name for path in removed) == ["py311-paddle3.2.0", "tmp"]
    assert sorted(path.name for path in root.iterdir()) == ["notes.txt", "py312-paddle3.3.0"]


# --------------------------------------------------------------------------
# activation
# --------------------------------------------------------------------------

@pytest.fixture
def restore_site(monkeypatch):
    saved_path = list(sys.path)
    saved = (site.ENABLE_USER_SITE, site.USER_BASE, site.USER_SITE)
    monkeypatch.delenv("PYTHONUSERBASE", raising=False)
    yield
    sys.path[:] = saved_path
    site.ENABLE_USER_SITE, site.USER_BASE, site.USER_SITE = saved
    os.environ.pop("PYTHONUSERBASE", None)


def test_activate_makes_the_engine_dir_the_user_site(tmp_path, restore_site):
    directory = tmp_path / "engine" / "py312-paddle3.3.0"
    packages = engine.site_dir(directory)
    (packages / "fakepaddle_for_test").mkdir(parents=True)
    (packages / "fakepaddle_for_test" / "__init__.py").write_text("VALUE = 42\n")
    assert engine.activate(directory) == packages
    assert site.ENABLE_USER_SITE is True
    assert site.USER_BASE == str(directory) and site.USER_SITE == str(packages)
    assert os.environ["PYTHONUSERBASE"] == str(directory)
    assert str(packages) in sys.path
    before = sys.path.count(str(packages))
    engine.activate(directory)                                  # idempotent
    assert sys.path.count(str(packages)) == before
    import fakepaddle_for_test
    assert fakepaddle_for_test.VALUE == 42
    sys.modules.pop("fakepaddle_for_test", None)


def test_activation_in_a_child_matches_what_paddle_searches(tmp_path):
    """paddle/base/core.py looks for its libs in site.getsitepackages() and
    site.USER_SITE; after activate() USER_SITE is the engine's site dir even
    under -s -E (how the bundle starts the interpreter)."""
    directory = tmp_path / "eng"
    code = ("import site, sys; from core.runtime.engine import activate; p = activate(sys.argv[1]); "
            "print(site.USER_SITE == str(p), str(p) in sys.path)")
    out = subprocess.run([sys.executable, "-s", "-E", "-c", code, str(directory)], capture_output=True, text=True,
                         cwd=REPO_ROOT, check=True).stdout.split()
    assert out == ["True", "True"]


def test_activate_installed_activates_only_a_verified_engine(tmp_path, restore_site):
    from core.runtime import activate_installed

    assert activate_installed(tmp_path) is None
    directory = engine.engine_dir(tmp_path)
    engine.site_dir(directory).mkdir(parents=True)
    engine.write_state(directory, EngineState(variant="cpu", verified=True))
    assert activate_installed(tmp_path).variant == "cpu"
    assert site.USER_SITE == str(engine.site_dir(directory))


# --------------------------------------------------------------------------
# pip
# --------------------------------------------------------------------------

def test_requirements():
    assert pip.requirement("cpu", "linux", (3, 12)) == "paddlepaddle==3.3.0"
    assert pip.requirement("cpu", "mac", (3, 12)) == "paddlepaddle==3.3.0"
    assert pip.requirement("cu129", "linux", (3, 12)) == (
        "paddlepaddle-gpu @ https://paddle-whl.cdn.bcebos.com/stable/cu129/paddlepaddle-gpu/"
        "paddlepaddle_gpu-3.3.0-cp312-cp312-linux_x86_64.whl")
    assert pip.requirement("cu118", "win", (3, 12)) == (
        "paddlepaddle-gpu @ https://paddle-whl.cdn.bcebos.com/stable/cu118/paddlepaddle-gpu/"
        "paddlepaddle_gpu-3.3.0-cp312-cp312-win_amd64.whl")
    with pytest.raises(ValueError):
        pip.requirement("cu129", "mac", (3, 12))
    with pytest.raises(ValueError):
        pip.requirement("cu130", "linux", (3, 12))


def test_install_command_for_a_current_pip(tmp_path):
    constraints = tmp_path / "constraints.txt"
    command = pip.install_command("/b/python/bin/python3", "paddlepaddle==3.3.0", constraints, (26, 0, 1))
    assert command[:6] == ["/b/python/bin/python3", "-X", "utf8", "-m", "pip", "install"]
    for flag in ("--user", "--no-cache-dir", "--only-binary=:all:", "--no-warn-script-location",
                 "--disable-pip-version-check", "--no-input", "--break-system-packages"):
        assert flag in command
    assert command[command.index("--progress-bar") + 1] == "raw"
    assert command[command.index("-c") + 1] == str(constraints)
    assert command[-1] == "paddlepaddle==3.3.0"
    assert "-s" not in command and "-E" not in command         # both would turn the user site off
    assert "--index-url" not in command and "--extra-index-url" not in command and "-i" not in command


def test_install_command_falls_back_for_old_pips():
    command = pip.install_command("python", "paddlepaddle==3.3.0", None, (23, 0))
    assert command[command.index("--progress-bar") + 1] == "off"
    assert "--break-system-packages" not in command and "-c" not in command
    assert pip.install_command("python", "x", None, (24, 1))[pip.install_command("python", "x", None, (24, 1))
                                                              .index("--progress-bar") + 1] == "raw"


def test_parse_pip_version():
    assert pip.parse_pip_version("pip 24.2 from /x/site-packages/pip (python 3.12)") == (24, 2)
    assert pip.parse_pip_version("pip 26.0.1 from ...") == (26, 0, 1)
    assert pip.parse_pip_version("No module named pip") == ()


def test_install_env(tmp_path):
    env = pip.install_env({"PATH": "/bin", "PYTHONNOUSERSITE": "1", "PYTHONPATH": "/x", "PIP_TARGET": "/t",
                           "VIRTUAL_ENV": "/v"}, tmp_path)
    assert env["PYTHONUSERBASE"] == str(tmp_path) and env["PATH"] == "/bin"
    for gone in ("PYTHONNOUSERSITE", "PYTHONPATH", "PIP_TARGET", "VIRTUAL_ENV"):
        assert gone not in env


def test_constraints_are_filtered(tmp_path):
    source = tmp_path / "freeze.txt"
    source.write_text("\n".join([
        "# comment", "", "numpy==2.2.6", "paddleocr==3.3.2", "paddlex==3.3.10", "paddlepaddle==3.3.0",
        "paddlepaddle-gpu==3.3.0", "nvidia-cudnn-cu12==9.9.0.52", "cuda-python==12.9.4",
        "-e git+https://example/x#egg=x", "mypkg @ file:///build/mypkg", "PyQt6==6.10.2"]))
    target = pip.write_constraints(source, tmp_path / "out" / "constraints.txt")
    assert target.read_text().split() == ["numpy==2.2.6", "paddleocr==3.3.2", "paddlex==3.3.10", "PyQt6==6.10.2"]
    assert pip.write_constraints(None, tmp_path / "x.txt") is None
    assert pip.write_constraints(tmp_path / "missing.txt", tmp_path / "x.txt") is None


def test_pip_output_counts_bytes_over_files_and_names_steps():
    out = pip.PipOutput()
    lines = [
        "Collecting paddlepaddle==3.3.0",
        "  Downloading paddlepaddle-3.3.0-cp312-cp312-manylinux1_x86_64.whl.metadata (8.9 kB)",
        "Downloading https://files/paddlepaddle-3.3.0-cp312-cp312-manylinux1_x86_64.whl (193.7 MB)",
        "Progress 0 of 193700000",
        "Progress 100000000 of 193700000",
        "Progress 193700000 of 193700000",
        "Downloading https://files/opt_einsum-3.3.0-py3-none-any.whl (65 kB)",
        "Downloading https://files/networkx-3.4-py3-none-any.whl (1.7 MB)",
        "Progress 1700000 of 1700000",
        "Installing collected packages: paddlepaddle",
        "Successfully installed paddlepaddle-3.3.0",
    ]
    statuses = [out.feed(line) for line in lines]
    assert statuses[0].step == "Resolving paddlepaddle==3.3.0"
    assert not statuses[1].changed_step                         # a .metadata download is still resolving
    assert statuses[2].step == "Downloading paddlepaddle-3.3.0-cp312-cp312-manylinux1_x86_64.whl"
    assert statuses[4].is_progress_line and statuses[4].done_bytes == 100_000_000
    assert statuses[8].done_bytes == 193_700_000 + 1_700_000
    assert statuses[9].step.startswith("Installing")
    assert statuses[10].step == "Installed" and out.done_bytes == 195_400_000


def test_python_for_subprocess(tmp_path):
    pythonw = tmp_path / "pythonw.exe"
    assert pip.python_for_subprocess(str(pythonw)) == str(pythonw)      # no python.exe next to it
    (tmp_path / "python.exe").write_text("")
    assert pip.python_for_subprocess(str(pythonw)) == str(tmp_path / "python.exe")
    assert pip.python_for_subprocess("/usr/bin/python3") == "/usr/bin/python3"


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def test_parse_result_takes_the_last_result_line():
    assert parse_result(["noise", 'RESULT {"ok": true, "a": 1}', "trailing"]) == {"ok": True, "a": 1}
    assert parse_result(['RESULT {"ok": false}', 'RESULT {"ok": true}']) == {"ok": True}
    assert parse_result(["RESULT not json"]) is None and parse_result([]) is None


def test_models_present(tmp_path):
    assert not models_present(tmp_path)
    for name in ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "inference.yml").write_text("x")
        (tmp_path / name / "inference.pdiparams").write_text("x")
    assert models_present(tmp_path)


def test_runtime_imports_no_qt():
    code = ("import sys, core.runtime, core.runtime.install, core.runtime.checks, core.runtime.logfile; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('PyQt6', 'PyQt5', 'PySide6', 'paddle')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
