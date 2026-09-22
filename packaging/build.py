#!/usr/bin/env python3
"""Build OCR Manager's release artifacts for the host OS.

    python packaging/build.py                 # bundle + artifacts into dist/
    python packaging/build.py --bundle-only   # just the runnable tree
    python packaging/build.py run -- --self-test   # run the built bundle (console)

Stdlib only; any Python 3.11+ runs it. What it builds:

    <root>/python/          python-build-standalone CPython, with every runtime
                            dependency in its own site-packages EXCEPT Paddle
    <root>/src/             main.py, app/, core/, videocr/, resources/
    <root>/bin/             static ffmpeg + ffprobe (put on PATH by the launcher)
    <root>/licenses/        third-party notices
    <root>/bundle.json      {"version", "os", "arch"}
    <root>/constraints.txt  pip freeze of the bundled interpreter: the first-run
                            Paddle install is constrained by it, so it never
                            shadows or upgrades a bundled package

Paddle (the CPU or a CUDA build, chosen for the machine) is installed by the
app on first run into the user's data folder -- see core/runtime/.

Artifacts, named ocr-manager-v<X.Y.Z>-<os>.<ext>:

    linux  .tar.gz (portable tree) and .AppImage
    win    .exe (per-user Inno Setup installer) and .zip (portable tree)
    mac    .dmg (OCR Manager.app, Apple Silicon, macOS 14.5+)

Everything downloaded is pinned by URL and SHA-256 and cached in --cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import traceback
import zipfile
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGING = REPO / "packaging"

APP_NAME = "OCR Manager"
BUNDLE_ID = "uk.blyat.ocr-manager"
MACOS_MIN = "14.5"          # Paddle's macOS arm64 wheel is built for 14.5

# python-build-standalone: https://github.com/astral-sh/python-build-standalone
PYTHON = "3.12.14"
PBS_RELEASE = "20260901"
PBS_ASSETS = {
    ("linux", "x86_64"): ("x86_64-unknown-linux-gnu",
                          "72748da13197c1fb161e3afeef20a6a385ff24f2165e6e2758e47008e7faba4c"),
    ("win", "x86_64"): ("x86_64-pc-windows-msvc",
                        "7c45c9622400d578709a9b2cddbe8124cc21d382409d9f13406d706d28e31b14"),
    ("mac", "arm64"): ("aarch64-apple-darwin",
                       "81a359f1cfadd4da11766534c5913791cea55f26e1bb902cacd2a531bb1e4b2b"),
}

# Static ffmpeg/ffprobe. Linux and Windows: BtbN's LGPL builds, a month-end
# autobuild (kept for two years). macOS arm64: martin-riedl.de's signed builds.
_BTBN = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-08-31-13-27/"
_RIEDL = "https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_9.0.2/"
FFMPEG = {
    "linux": [(_BTBN + "ffmpeg-n8.1.2-50-g1a748fe2cd-linux64-lgpl-8.1.tar.xz",
               "7d6d93e9c39e0e461feb13c118e91e4eec2515e4da3a01d4ad6790996731bbee")],
    "win": [(_BTBN + "ffmpeg-n8.1.2-50-g1a748fe2cd-win64-lgpl-8.1.zip",
             "f6274bbd9c247f9e90c1bbed066b03ed4a3907cece2fb91be6dd352393936365")],
    "mac": [(_RIEDL + "ffmpeg.zip", "c8ed4c4e6978a03c485edbfe4e0a5dc2380f8a30bba5150531b31b094492d924"),
            (_RIEDL + "ffprobe.zip", "fcbe839537485eaee7a7a8bc5cbc0f90d53617e80943e8a5b2e31cb851197ea6")],
}

APPIMAGETOOL = ("https://github.com/AppImage/appimagetool/releases/download/continuous/"
                "appimagetool-x86_64.AppImage")

# The MSVC runtime Paddle's Windows wheels link against, shipped app-local
# next to python.exe so no VC++ redistributable install is needed.
MSVC_RUNTIME = ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll",
                "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll", "vcomp140.dll")

TAIL_LINES = 80

SOURCES = ("main.py", "app", "core", "videocr", "resources")

# The interpreter flags every launcher uses: no user site (-s) and no PYTHON*
# variables (-E), so nothing on the user's machine leaks in; no bytecode
# writes (-B: the tree may be read-only, and on macOS it is signed); UTF-8 mode.
PY_FLAGS = ("-s", "-E", "-B", "-X", "utf8")


def log(msg: str) -> None:
    print(f"==> {msg}", flush=True)


class CommandFailed(Exception):
    """A build command exited non-zero; carries the end of its output."""


def stream(cmd: list, **kw) -> int:
    """Run `cmd`, echoing its output live and keeping the last lines: CI
    job logs need a token to read, so a failure is reported through a
    public annotation (see annotate()) that carries them."""
    if kw.get("stdout") is not None:
        return subprocess.run([str(c) for c in cmd], **kw).returncode
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", **kw)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        tail.append(line.rstrip())
    code = proc.wait()
    if code:
        stream.tail = list(tail)
    return code


stream.tail = []


def run(cmd: list, **kw) -> None:
    print("   $", " ".join(str(c) for c in cmd), flush=True)
    if code := stream(cmd, **kw):
        raise CommandFailed(f"exit code {code}: " + " ".join(str(c) for c in cmd)
                            + "\n" + "\n".join(stream.tail))


def annotate(title: str, text: str) -> None:
    """A GitHub Actions error annotation (readable without a token)."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    body = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error title={title}::{body}", flush=True)


def host() -> tuple[str, str]:
    system, machine = platform.system(), platform.machine().lower()
    os_name = {"Linux": "linux", "Windows": "win", "Darwin": "mac"}.get(system)
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    if (os_name, arch) not in PBS_ASSETS:
        sys.exit(f"unsupported build host: {system} {machine}")
    return os_name, arch


def app_version() -> str:
    text = (REPO / "app" / "version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"', text, re.M)
    if not match:
        sys.exit("app/version.py has no __version__")
    return match.group(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, cache: Path, expected: str | None = None) -> Path:
    """Download `url` into `cache` once; verify its SHA-256 when one is given."""
    name = url.rstrip("/").rsplit("/", 1)[-1].replace("%2B", "+")
    dest = cache / name
    if dest.exists() and (expected is None or sha256(dest) == expected):
        return dest
    log(f"downloading {name}")
    cache.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ocr-manager-build"})
    with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    if expected is not None and (actual := sha256(part)) != expected:
        part.unlink()
        sys.exit(f"{name}: SHA-256 {actual} != pinned {expected}")
    os.replace(part, dest)
    return dest


# -- the runnable tree ------------------------------------------------------

def python_exe(root: Path, os_name: str, gui: bool = False) -> Path:
    if os_name == "win":
        return root / "python" / ("pythonw.exe" if gui else "python.exe")
    return root / "python" / "bin" / "python3"


def install_python(root: Path, os_name: str, arch: str, cache: Path) -> None:
    triple, digest = PBS_ASSETS[(os_name, arch)]
    name = f"cpython-{PYTHON}+{PBS_RELEASE}-{triple}-install_only_stripped.tar.gz"
    url = (f"https://github.com/astral-sh/python-build-standalone/releases/download/"
           f"{PBS_RELEASE}/{name.replace('+', '%2B')}")
    archive = fetch(url, cache, digest)
    log("unpacking the interpreter")
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="tar")          # the archive's top dir is python/
    lib = root / "python" / ("Lib" if os_name == "win" else f"lib/python{PYTHON.rsplit('.', 1)[0]}")
    for unused in ("test", "idlelib", "turtledemo"):
        shutil.rmtree(lib / unused, ignore_errors=True)


def install_dependencies(root: Path, os_name: str) -> None:
    py = python_exe(root, os_name)
    pip = [py, "-s", "-E", "-m", "pip"]
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_WARN_SCRIPT_LOCATION": "1"}
    log("installing the runtime dependencies")
    run([*pip, "install", "--upgrade", "pip"], env=env)
    run([*pip, "install", "--only-binary=:all:",
         "-r", PACKAGING / "requirements-bundle.txt",
         "-c", PACKAGING / "constraints-bundle.txt"], env=env)
    # Paddle came in only so its own dependencies resolve with the rest; the
    # app installs the build that suits the machine on first run.
    run([*pip, "uninstall", "-y", "paddlepaddle"], env=env)
    frozen = subprocess.run([str(py), "-s", "-E", "-m", "pip", "freeze", "--all"], check=True,
                            capture_output=True, text=True, env=env).stdout
    lines = [ln for ln in frozen.splitlines() if ln and not ln.lower().startswith("paddlepaddle")]
    (root / "constraints.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if os_name == "win":
        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        for dll in MSVC_RUNTIME:
            if (system32 / dll).exists() and not (root / "python" / dll).exists():
                shutil.copy2(system32 / dll, root / "python" / dll)


# Qt modules the app never loads (it uses QtCore, QtGui and QtWidgets; the
# platform plugins need DBus, OpenGL, Network, Svg, Wayland and XcbQpa, which
# stay). The PyQt6-Qt6 wheel ships all of Qt, ~120 MB of it unused.
QT_UNUSED = ("3D", "Bluetooth", "Charts", "DataVisualization", "Designer", "FFmpegStub", "Graphs", "Help",
             "Labs", "Lottie", "Multimedia", "Nfc", "Pdf", "Positioning", "Qml", "Quick",
             "RemoteObjects", "Scxml", "Sensors", "SerialPort", "ShaderTools", "SpatialAudio",
             "Sql", "StateMachine", "Test", "TextToSpeech", "VirtualKeyboard", "WebChannel",
             "WebSockets", "WebView")
QT_UNUSED_DIRS = ("qml", "qsci")
QT_UNUSED_PLUGINS = ("assetimporters", "designer", "geometryloaders", "help", "multimedia",
                     "position", "qmllint", "qmlls", "qmltooling", "renderers", "renderplugins",
                     "sceneparsers", "scxmldatamodel", "sensors", "sqldrivers", "texttospeech",
                     "webview")


def prune_qt(root: Path, os_name: str) -> None:
    lib = root / "python" / ("Lib" if os_name == "win" else f"lib/python{PYTHON.rsplit('.', 1)[0]}")
    pyqt = lib / "site-packages" / "PyQt6"
    unused = re.compile(r"^(?:lib)?Qt6?(?:%s)" % "|".join(QT_UNUSED))
    # Qt Multimedia's own FFmpeg (PyAV carries the one the app decodes with).
    qt_ffmpeg = re.compile(r"^(?:lib)?(?:avcodec|avformat|avutil|swresample|swscale)[-.]")
    removed = 0
    for entry in pyqt.iterdir():
        if entry.name.startswith("Qt") and unused.match(entry.name):
            entry.unlink()
            removed += 1
    qt = pyqt / "Qt6"
    for sub in ("lib", "bin"):
        for entry in (qt / sub).iterdir() if (qt / sub).is_dir() else ():
            if unused.match(entry.name) or qt_ffmpeg.match(entry.name):
                shutil.rmtree(entry) if entry.is_dir() and not entry.is_symlink() else entry.unlink()
                removed += 1
    for name in QT_UNUSED_DIRS:
        shutil.rmtree(qt / name, ignore_errors=True)
    for name in QT_UNUSED_PLUGINS:
        shutil.rmtree(qt / "plugins" / name, ignore_errors=True)
    log(f"pruned {removed} unused Qt modules and libraries")


def copy_sources(root: Path) -> None:
    log("copying the application")
    src = root / "src"
    src.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    for item in SOURCES:
        path = REPO / item
        if path.is_dir():
            shutil.copytree(path, src / item, ignore=ignore)
        else:
            shutil.copy2(path, src / item)


def compile_bytecode(root: Path, os_name: str) -> None:
    """Precompile everything with hash-checked-free pycs: the launchers run
    with -B (nothing is written into the tree), and archive formats that round
    mtimes (zip) cannot invalidate them."""
    log("precompiling bytecode")
    py = python_exe(root, os_name)
    run([py, "-s", "-E", "-m", "compileall", "-q", "-j", "0",
         "--invalidation-mode", "unchecked-hash", root / "src", root / "python"],
        stdout=subprocess.DEVNULL)


def install_ffmpeg(root: Path, os_name: str, cache: Path) -> None:
    log("adding ffmpeg and ffprobe")
    bin_dir = root / "bin"
    bin_dir.mkdir()
    lic = root / "licenses"
    lic.mkdir(exist_ok=True)
    wanted = {"ffmpeg", "ffprobe", "ffmpeg.exe", "ffprobe.exe"}
    for url, digest in FFMPEG[os_name]:
        archive = fetch(url, cache, digest)
        if archive.name.endswith(".tar.xz"):
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    base = member.name.rsplit("/", 1)[-1]
                    if member.isfile() and (base in wanted and "/bin/" in member.name or base == "LICENSE.txt"):
                        data = tar.extractfile(member).read()
                        target = bin_dir / base if base in wanted else lic / "FFmpeg-LICENSE.txt"
                        target.write_bytes(data)
        else:
            with zipfile.ZipFile(archive) as zf:
                for name in zf.namelist():
                    base = name.rsplit("/", 1)[-1]
                    if base in wanted:
                        (bin_dir / base).write_bytes(zf.read(name))
                    elif base == "LICENSE.txt":
                        (lic / "FFmpeg-LICENSE.txt").write_bytes(zf.read(name))
    for tool in ("ffmpeg", "ffprobe"):
        path = bin_dir / (tool + (".exe" if os_name == "win" else ""))
        if not path.exists():
            sys.exit(f"{path.name} missing from the ffmpeg download")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_notices(root: Path, os_name: str) -> None:
    ffmpeg_src = ("BtbN/FFmpeg-Builds (LGPL build), https://github.com/BtbN/FFmpeg-Builds"
                  if os_name != "mac" else
                  "ffmpeg.martin-riedl.de (static build), https://ffmpeg.martin-riedl.de")
    text = f"""OCR Manager bundles third-party software. Each keeps its own license.

- CPython {PYTHON} (python-build-standalone {PBS_RELEASE}): PSF License,
  python/ (see the LICENSE files the interpreter carries).
- Python packages in python/: each package's license is in its
  *.dist-info directory under site-packages.
- FFmpeg (bin/ffmpeg, bin/ffprobe): from {ffmpeg_src}.
  FFmpeg is licensed under the LGPL v2.1+ (GPL v2+ for GPL builds); its
  source is available at https://ffmpeg.org/download.html and from the
  build project above.
- Paddle / PaddleOCR / PaddleX (installed on first run): Apache License 2.0.
"""
    (root / "licenses").mkdir(exist_ok=True)
    (root / "licenses" / "THIRD-PARTY.txt").write_text(text, encoding="utf-8")


def write_metadata(root: Path, version: str, os_name: str, arch: str) -> None:
    (root / "bundle.json").write_text(
        json.dumps({"version": version, "os": os_name, "arch": arch}, indent=2) + "\n", encoding="utf-8")


def build_tree(root: Path, version: str, os_name: str, arch: str, cache: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    install_python(root, os_name, arch, cache)
    install_dependencies(root, os_name)
    prune_qt(root, os_name)
    copy_sources(root)
    install_ffmpeg(root, os_name, cache)
    write_notices(root, os_name)
    write_metadata(root, version, os_name, arch)
    compile_bytecode(root, os_name)


# -- icons ------------------------------------------------------------------

def make_ico(py: Path, out: Path) -> None:
    run([py, "-s", "-E", "-c",
         "import sys; from PIL import Image; Image.open(sys.argv[1]).save(sys.argv[2], "
         "sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])",
         REPO / "resources" / "app-icon.png", out])


def make_icns(out: Path, work: Path) -> None:
    iconset = work / "app.iconset"
    shutil.rmtree(iconset, ignore_errors=True)
    iconset.mkdir(parents=True)
    png = REPO / "resources" / "app-icon.png"
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            px = size * scale
            name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
            run(["sips", "-z", px, px, png, "--out", iconset / name], stdout=subprocess.DEVNULL)
    run(["iconutil", "-c", "icns", iconset, "-o", out])


# -- per-OS layouts and artifacts ------------------------------------------

def linux_artifacts(tree: Path, version: str, dist: Path, cache: Path) -> list[Path]:
    launcher = tree / "ocr-manager"
    shutil.copy2(PACKAGING / "linux" / "ocr-manager", launcher)
    launcher.chmod(0o755)
    shutil.copy2(REPO / "resources" / "app-icon.png", tree / "ocr-manager.png")
    shutil.copy2(PACKAGING / "linux" / "ocr-manager.desktop", tree / "ocr-manager.desktop")

    stem = f"ocr-manager-v{version}-linux"
    tarball = dist / f"{stem}.tar.gz"
    log(f"writing {tarball.name}")
    with tarfile.open(tarball, "w:gz", compresslevel=6) as tar:
        tar.add(tree, arcname=stem)

    # AppImage: the same tree is the AppDir; AppRun is the launcher itself.
    (tree / "AppRun").symlink_to("ocr-manager")
    tool = fetch(APPIMAGETOOL, cache)
    tool.chmod(0o755)
    appimage = dist / f"{stem}.AppImage"
    log(f"writing {appimage.name}")
    run([tool, "--appimage-extract-and-run", "--no-appstream", tree, appimage],
        env={**os.environ, "ARCH": "x86_64"})
    (tree / "AppRun").unlink()
    return [tarball, appimage]


def windows_artifacts(tree: Path, version: str, dist: Path, work: Path) -> list[Path]:
    ico = work / "app.ico"
    make_ico(python_exe(tree, "win"), ico)
    shutil.copy2(ico, tree / "app.ico")

    log("compiling the launcher")
    build = work / "launcher"
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)
    shutil.copy2(PACKAGING / "windows" / "launcher.c", build)
    shutil.copy2(PACKAGING / "windows" / "launcher.rc", build)
    shutil.copy2(ico, build / "app.ico")
    run(["rc", "/nologo", "/fo", "launcher.res", "launcher.rc"], cwd=build)
    run(["cl", "/nologo", "/O2", "/W4", "/DUNICODE", "/D_UNICODE", "launcher.c", "launcher.res",
         "/link", "/SUBSYSTEM:WINDOWS", "user32.lib", "/OUT:launcher.exe"], cwd=build)
    shutil.copy2(build / "launcher.exe", tree / f"{APP_NAME}.exe")
    shutil.copy2(PACKAGING / "windows" / "ocr-manager-cli.cmd", tree / "ocr-manager-cli.cmd")

    stem = f"ocr-manager-v{version}-win"
    archive = dist / f"{stem}.zip"
    log(f"writing {archive.name}")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(tree.rglob("*")):
            if path.is_file():
                zf.write(path, f"{stem}/{path.relative_to(tree).as_posix()}")

    iscc = shutil.which("iscc") or r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    log(f"writing {stem}.exe")
    run([iscc, "/Qp", f"/DAppVersion={version}", f"/DSourceDir={tree}", f"/DOutputDir={dist}",
         f"/DOutputBaseFilename={stem}", f"/DIconFile={ico}",
         PACKAGING / "windows" / "installer.iss"])
    return [archive, dist / f"{stem}.exe"]


_MACHO_MAGIC = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xfe\xed\xfa\xcf"}


def is_macho(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(4) in _MACHO_MAGIC


def mac_artifacts(tree: Path, version: str, dist: Path, work: Path) -> list[Path]:
    app = work / f"{APP_NAME}.app"
    shutil.rmtree(app, ignore_errors=True)
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    shutil.move(str(tree), str(contents / "Resources"))
    shutil.copy2(PACKAGING / "mac" / "ocr-manager", contents / "MacOS" / "ocr-manager")
    (contents / "MacOS" / "ocr-manager").chmod(0o755)
    make_icns(contents / "Resources" / "app.icns", work)
    plist = (PACKAGING / "mac" / "Info.plist").read_text(encoding="utf-8")
    for key, value in {"VERSION": version, "BUNDLE_ID": BUNDLE_ID, "MACOS_MIN": MACOS_MIN}.items():
        plist = plist.replace(f"@{key}@", value)
    (contents / "Info.plist").write_text(plist, encoding="utf-8")
    # Ad-hoc signatures: Apple Silicon refuses unsigned code, and a sealed
    # bundle gets "Open Anyway" rather than "damaged". Nested Mach-O files
    # outside the standard places are not reached by --deep, so each one is
    # signed first, then the bundle seals them.
    machos = [p for p in (contents / "Resources").rglob("*")
              if p.is_file() and not p.is_symlink() and is_macho(p)]
    log(f"signing {len(machos)} binaries")
    for i in range(0, len(machos), 200):
        run(["codesign", "--force", "--sign", "-", *machos[i:i + 200]], stdout=subprocess.DEVNULL)
    run(["codesign", "--force", "--sign", "-", app])
    run(["codesign", "--verify", "--deep", "--strict", app])

    stage = work / "dmg"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    shutil.move(str(app), str(stage / app.name))
    (stage / "Applications").symlink_to("/Applications")
    dmg = dist / f"ocr-manager-v{version}-mac.dmg"
    log(f"writing {dmg.name}")
    run(["hdiutil", "create", "-volname", APP_NAME, "-srcfolder", stage, "-ov",
         "-format", "UDZO", "-fs", "HFS+", dmg])
    # Keep a runnable copy of the app for the smoke tests.
    shutil.move(str(stage / app.name), str(work / app.name))
    return [dmg]


def tree_path(work: Path, os_name: str) -> Path:
    """Where the runnable bundle root is after a build (the mac root lives
    inside the .app)."""
    if os_name == "mac":
        return work / f"{APP_NAME}.app" / "Contents" / "Resources"
    return work / "tree"


# -- running the built bundle -------------------------------------------------

def run_bundle(work: Path, args: list[str]) -> int:
    """Run the built bundle's app with a console interpreter, exactly as the
    launchers do (same env and flags): CI's smoke tests go through this."""
    os_name, _ = host()
    root = tree_path(work, os_name)
    if not (root / "bundle.json").exists():
        sys.exit(f"no bundle at {root}; build it first")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["OCR_MANAGER_BUNDLE"] = str(root)
    env["PATH"] = str(root / "bin") + os.pathsep + env.get("PATH", "")
    cmd = [str(python_exe(root, os_name)), *PY_FLAGS, str(root / "src" / "main.py"), *args]
    print("   $", " ".join(cmd), flush=True)
    code = stream(cmd, env=env)
    if code:
        annotate(f"bundle {' '.join(args)} failed", f"exit code {code}\n" + "\n".join(stream.tail))
    return code


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", default=None, help="defaults to app/version.py")
    parser.add_argument("--dist", type=Path, default=REPO / "dist")
    parser.add_argument("--work", type=Path, default=REPO / "build" / "bundle")
    parser.add_argument("--cache", type=Path, default=REPO / ".build-cache")
    parser.add_argument("--bundle-only", action="store_true", help="build the tree, no artifacts")
    if argv[:1] == ["run"]:
        args, rest = parser.parse_known_args(argv[1:])
        if rest[:1] == ["--"]:
            rest = rest[1:]
        return run_bundle(args.work.resolve(), rest)
    args = parser.parse_args(argv)

    os_name, arch = host()
    version = args.version or app_version()
    work, dist, cache = args.work.resolve(), args.dist.resolve(), args.cache.resolve()
    work.mkdir(parents=True, exist_ok=True)
    dist.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(work / f"{APP_NAME}.app", ignore_errors=True)

    tree = work / "tree"
    log(f"building OCR Manager {version} for {os_name}/{arch}")
    build_tree(tree, version, os_name, arch, cache)
    if args.bundle_only:
        if os_name == "mac":
            sys.exit("--bundle-only is not supported on macOS (the tree lives in the .app)")
        return 0

    if os_name == "linux":
        outputs = linux_artifacts(tree, version, dist, cache)
    elif os_name == "win":
        outputs = windows_artifacts(tree, version, dist, work)
    else:
        outputs = mac_artifacts(tree, version, dist, work)
    for path in outputs:
        print(f"    {path.name}  {path.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as exc:
        if exc.code not in (None, 0) and not isinstance(exc.code, int):
            annotate("build failed", str(exc.code))
        raise
    except BaseException:
        annotate("build failed", traceback.format_exc()[-6000:])
        raise
