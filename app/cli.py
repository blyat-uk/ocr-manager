"""The headless command-line modes of `python -m app` / main.py (CI hooks).

`--self-test`
    Imports every module under app/, core/ and videocr/; runs `ffmpeg
    -version` and `ffprobe -version` from PATH; imports av and calls
    `videocr.pyav_adapter.assert_reference_backend()`. Prints one JSON
    report (`"ok"` plus the details). Exit 0 when everything passed, else 1.
    The OCR engine is reported but not required.

`--install-engine {auto,cpu,gpu}`
    Installs the OCR engine into the data dir (bundles only). Progress as
    `[step] ...`, `[progress] <done> of <total> bytes`, `[notice] ...` and
    `  | <pip output>` lines; the last line is `RESULT <json>`. Exit 0
    installed; 1 failed; 2 not a bundle; 3 `gpu` was asked for but no GPU
    build fits this machine, or the GPU build failed and CPU was installed;
    130 interrupted (Ctrl+C).

`--ocr-smoke`
    Builds the OCR engine on the resolved device, OCRs a rendered line of
    Chinese (Latin when no CJK font is installed) and prints `RESULT
    <json>`. Exit 0 when any text was recognised, else 1.

Output goes to stdout (the log file under pythonw).
"""
from __future__ import annotations

import importlib
import json
import os
import pkgutil
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback

from app.bootstrap import Boot
from core.proc import TEXT_ENCODING, hidden_child
from app.version import __version__

PACKAGES = ("app", "core", "videocr")
TOOLS = ("ffmpeg", "ffprobe")
TOOL_TIMEOUT = 30
SMOKE_TEXT_CJK = "你好世界测试字幕"
SMOKE_TEXT_LATIN = "OCR SMOKE 12345"
PREFERRED_CJK_FONTS = ("Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "PingFang SC",
                       "WenQuanYi Micro Hei", "SimHei", "Hiragino Sans GB", "Noto Sans SC")
EXIT_OK, EXIT_FAIL, EXIT_NOT_BUNDLE, EXIT_NO_GPU, EXIT_INTERRUPTED = 0, 1, 2, 3, 130
_smoke_app = None                       # the QGuiApplication --ocr-smoke renders with, when none exists


def _emit(line: str) -> None:
    print(line, flush=True)


# --------------------------------------------------------------------------
# --self-test
# --------------------------------------------------------------------------

def import_all(packages=PACKAGES) -> tuple[list[str], dict[str, str]]:
    """Import every module of `packages`; (imported, {module: error})."""
    imported, failed = [], {}
    for name in packages:
        try:
            package = importlib.import_module(name)
        except Exception as exc:                            # noqa: BLE001 - reported, not raised
            failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        imported.append(name)
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{name}."):
            try:
                importlib.import_module(info.name)
                imported.append(info.name)
            except Exception as exc:                        # noqa: BLE001
                failed[info.name] = f"{type(exc).__name__}: {exc}"
    return imported, failed


def check_tool(tool: str) -> dict:
    path = shutil.which(tool)
    if path is None:
        return {"ok": False, "path": None, "error": f"{tool} is not on PATH"}
    try:
        done = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=TOOL_TIMEOUT,
                              check=False, **TEXT_ENCODING, **hidden_child())
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "path": path, "error": str(exc)}
    first = (done.stdout or "").splitlines()[0] if done.stdout else ""
    return {"ok": done.returncode == 0, "path": path, "version": first,
            **({} if done.returncode == 0 else {"error": f"exit code {done.returncode}"})}


def check_av() -> dict:
    try:
        import av
    except Exception as exc:                                # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    report = {"ok": True, "version": getattr(av, "__version__", "")}
    try:
        from videocr import pyav_adapter
        check = getattr(pyav_adapter, "assert_reference_backend", None)
        if check is not None:
            check()
            report["reference_backend"] = True
    except Exception as exc:                                # noqa: BLE001
        report.update(ok=False, reference_backend=False, error=f"{type(exc).__name__}: {exc}")
    return report


def engine_report(boot: Boot) -> dict:
    from core.runtime.engine import engine_dir
    report: dict = {"bundled": boot.bundled}
    if boot.bundled:
        report["dir"] = str(engine_dir())
        report["installed"] = boot.state is not None
        if boot.state is not None:
            report.update(variant=boot.state.variant, models_ready=boot.state.models_ready,
                          fallback_reason=boot.state.fallback_reason)
    return report


def self_test(boot: Boot) -> int:
    imported, failed = import_all()
    tools = {tool: check_tool(tool) for tool in TOOLS}
    av_report = check_av()
    ok = not failed and not boot.bundle_error and all(t["ok"] for t in tools.values()) and av_report["ok"]
    report = {
        "ok": ok,
        "version": __version__,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "bundle": None if boot.bundle is None else {
            "root": str(boot.bundle.root), "version": boot.bundle.version,
            "os": boot.bundle.os, "arch": boot.bundle.arch},
        "bundle_error": boot.bundle_error,
        "imports": {"count": len(imported), "failed": failed},
        "tools": tools,
        "av": av_report,
        "engine": engine_report(boot),
    }
    _emit(json.dumps(report, indent=2, ensure_ascii=False))
    return EXIT_OK if ok else EXIT_FAIL


# --------------------------------------------------------------------------
# --install-engine
# --------------------------------------------------------------------------

class _Printer:
    """InstallEvent -> lines; byte progress at most once a second."""

    def __init__(self) -> None:
        self._last = 0.0

    def __call__(self, event) -> None:
        if event.kind == "progress":
            now = time.monotonic()
            if now - self._last < 1.0 and event.done_bytes < event.total_bytes:
                return
            self._last = now
            _emit(f"[progress] {event.done_bytes} of {event.total_bytes} bytes")
        elif event.kind == "step":
            _emit(f"[step] {event.text}")
        elif event.kind == "notice":
            _emit(f"[notice] {event.text}")
        else:
            _emit(f"  | {event.text}")


def install_engine(boot: Boot, request: str) -> int:
    from core.runtime import resolve_request
    from core.runtime.gpu import CPU, describe, probe_gpus, variant_label
    from core.runtime.install import EngineInstaller, InstallContext

    if boot.bundle is None:
        _emit("--install-engine needs a portable bundle ($OCR_MANAGER_BUNDLE); in developer mode paddle "
              "comes from your environment." + (f" ({boot.bundle_error})" if boot.bundle_error else ""))
        return EXIT_NOT_BUNDLE
    probe = probe_gpus(boot.bundle.os)
    variant, choice = resolve_request(request, boot.bundle, probe)
    _emit(f"[hardware] {describe(probe)}")
    _emit(f"[choice] {variant_label(choice.variant)}: {choice.reason}")
    if request == "gpu" and variant == CPU:
        _emit(f"RESULT {json.dumps({'ok': False, 'error': 'no GPU build fits: ' + choice.reason})}")
        return EXIT_NO_GPU
    _emit(f"[install] {variant_label(variant)}")
    installer = EngineInstaller(InstallContext.for_bundle(boot.bundle), emit=_Printer())
    cancel = threading.Event()
    box: dict = {}

    def work() -> None:
        try:
            box["result"] = installer.install(variant, choice.gpu if variant != CPU else None, cancel)
        except BaseException as exc:                         # noqa: BLE001
            box["error"] = "".join(traceback.format_exception(exc))

    thread = threading.Thread(target=work, name="engine-install")
    thread.start()
    interrupted = False
    while thread.is_alive():
        try:
            thread.join(0.5)
        except KeyboardInterrupt:
            interrupted = True
            _emit("[step] Cancelling")
            cancel.set()
    if "error" in box:
        _emit(box["error"])
        _emit(f"RESULT {json.dumps({'ok': False, 'error': 'installer crashed'})}")
        return EXIT_FAIL
    result = box["result"]
    summary = {"ok": result.ok, "variant": result.variant, "fell_back": result.fell_back,
               "fallback_reason": result.fallback_reason, "error": result.error, "warnings": result.warnings,
               "engine_dir": str(result.engine_dir) if result.engine_dir else None,
               "state": result.state.to_json() if result.state else None}
    _emit(f"RESULT {json.dumps(summary, ensure_ascii=False)}")
    if result.cancelled or interrupted:
        return EXIT_INTERRUPTED
    if not result.ok:
        return EXIT_FAIL
    if request == "gpu" and result.fell_back:
        return EXIT_NO_GPU
    return EXIT_OK


# --------------------------------------------------------------------------
# --ocr-smoke
# --------------------------------------------------------------------------

def _cjk_family() -> str | None:
    from PyQt6.QtGui import QFontDatabase

    families = QFontDatabase.families(QFontDatabase.WritingSystem.SimplifiedChinese)
    for name in PREFERRED_CJK_FONTS:
        if name in families:
            return name
    return families[0] if families else None


def render_line(text: str | None = None):
    """(BGR numpy image, the text drawn, the font family) -- white text on
    black, drawn with Qt so any installed CJK font works."""
    import numpy as np
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter

    global _smoke_app
    if QGuiApplication.instance() is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        if sys.platform == "win32" and os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            # The offscreen platform looks for fonts in Qt's own folder, which
            # the wheel does not ship; Windows keeps them here.
            os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
        _smoke_app = QGuiApplication([sys.argv[0] if sys.argv else "ocr-manager"])
    family = _cjk_family()
    if text is None:
        text = SMOKE_TEXT_CJK if family else SMOKE_TEXT_LATIN
    width, height = 1280, 128
    image = QImage(width, height, QImage.Format.Format_RGB888)
    image.fill(QColor("black"))
    painter = QPainter(image)
    font = QFont(family) if family else QFont()
    font.setPixelSize(64)
    painter.setFont(font)
    painter.setPen(QColor("white"))
    painter.drawText(image.rect(), int(Qt.AlignmentFlag.AlignCenter), text)
    painter.end()
    buffer = image.constBits()
    buffer.setsize(image.sizeInBytes())
    rgb = np.frombuffer(buffer, np.uint8).reshape(height, image.bytesPerLine())[:, :width * 3]
    rgb = rgb.reshape(height, width, 3)
    if rgb.any():
        return np.ascontiguousarray(rgb[:, :, ::-1]), text, family or QFont().family()
    # Qt found no usable font (a headless platform without a font folder):
    # draw the line with Pillow from a system font file instead.
    return _render_with_pillow(width, height)


# System font files Pillow can draw the smoke line with: (path, has CJK).
FONT_FILES = (
    (r"C:\Windows\Fonts\msyh.ttc", True), (r"C:\Windows\Fonts\simhei.ttf", True),
    (r"C:\Windows\Fonts\simsun.ttc", True), (r"C:\Windows\Fonts\arial.ttf", False),
    ("/System/Library/Fonts/PingFang.ttc", True), ("/System/Library/Fonts/Hiragino Sans GB.ttc", True),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", False),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", True),
    ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", True),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", False), ("/usr/share/fonts/TTF/DejaVuSans.ttf", False),
)


def _render_with_pillow(width: int, height: int):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    for path, cjk in FONT_FILES:
        if os.path.exists(path):
            font = ImageFont.truetype(path, 64)
            text = SMOKE_TEXT_CJK if cjk else SMOKE_TEXT_LATIN
            break
    else:
        raise RuntimeError("no font to draw the smoke-test line with")
    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - (right - left)) / 2 - left, (height - (bottom - top)) / 2 - top), text,
              font=font, fill="white")
    return np.ascontiguousarray(np.asarray(image)[:, :, ::-1]), text, os.path.basename(path)


def ocr_smoke(boot: Boot) -> int:
    if boot.bundled and boot.state is None:
        _emit(f"RESULT {json.dumps({'ok': False, 'error': 'no OCR engine is installed (--install-engine)'})}")
        return EXIT_FAIL
    started = time.monotonic()
    try:
        image, expected, family = render_line()
        from videocr import utils

        device = utils.resolve_device(True)
        ocr = utils.create_ocr_engine("ch", None, None, True)
        texts, scores = [], []
        for item in ocr.predict(image):
            texts.extend(str(text) for text in item.get("rec_texts", []))
            scores.extend(float(score) for score in item.get("rec_scores", []))
    except Exception as exc:                                # noqa: BLE001
        traceback.print_exc()
        _emit(f"RESULT {json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'})}")
        return EXIT_FAIL
    recognized = "".join(texts).strip()
    report = {"ok": bool(recognized), "device": device, "font": family, "expected": expected,
              "recognized": recognized, "scores": [round(score, 4) for score in scores],
              "exact": recognized.replace(" ", "") == expected.replace(" ", ""),
              "seconds": round(time.monotonic() - started, 1)}
    _emit(f"RESULT {json.dumps(report, ensure_ascii=False)}")
    return EXIT_OK if recognized else EXIT_FAIL
