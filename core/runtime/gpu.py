"""Which paddle build fits this machine: the NVIDIA probe and the variant table.

`select_variant()` is pure and table-driven. It takes the newest GPU build
whose minimum driver the installed driver meets AND whose compiled
architectures cover the GPU's compute capability, then the next older one,
and otherwise the CPU build. macOS and non-x86_64 machines always get the
CPU build (paddle publishes no GPU wheel for them).

The table (`GPU_BUILDS`) is transcribed from the paddle 3.3.0 wheels:
compiled SASS per wheel and the CUDA toolkit's minimum driver. An arch is
covered when the wheel has exact SASS for it, or when it is listed in
`known_good` -- the Linux CUDA 12.9 wheel lacks sm_89 SASS yet runs on an
RTX 4090 (it JITs from the sm_86/sm_90 code), which the dev box verifies on
every fidelity check. Download sizes are the wheel plus its `nvidia-*` pip
dependencies, measured from the indexes (2026-09).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

NVIDIA_SMI = "nvidia-smi"
NVIDIA_SMI_TIMEOUT = 5.0
QUERY_FIELDS = "name,driver_version,compute_cap"
GB = 1_000_000_000


@dataclass(frozen=True)
class GpuInfo:
    name: str
    driver: str                     # as nvidia-smi prints it: "580.65.06", "576.02"
    compute_cap: str = ""           # "8.9", "12.0"; "" when this nvidia-smi cannot report it

    @property
    def driver_version(self) -> tuple[int, ...]:
        return version_tuple(self.driver)

    @property
    def sm(self) -> int | None:
        """Compute capability as an sm number: "8.9" -> 89, "12.0" -> 120."""
        try:
            major, minor = self.compute_cap.strip().split(".")
            return int(major) * 10 + int(minor)
        except ValueError:
            return None


@dataclass(frozen=True)
class GpuProbe:
    gpus: tuple[GpuInfo, ...] = ()
    error: str = ""                 # why no GPU was found ("" when some were)

    @property
    def primary(self) -> GpuInfo | None:
        """The GPU paddle's "gpu:0" is: nvidia-smi's first row."""
        return self.gpus[0] if self.gpus else None


@dataclass(frozen=True)
class GpuBuild:
    cuda: str                                   # "cu129": the wheel index path segment
    label: str                                  # "CUDA 12.9"
    min_driver: Mapping[str, tuple[int, ...]]   # os -> minimum driver version
    archs: Mapping[str, frozenset[int]]         # os -> sm numbers with SASS in the wheel
    download_bytes: Mapping[str, int]           # os -> wheel + nvidia-* deps
    known_good: Mapping[str, frozenset[int]] = field(default_factory=dict)

    def covers(self, os_name: str, sm: int) -> bool:
        return sm in self.archs.get(os_name, frozenset()) or sm in self.known_good.get(os_name, frozenset())


GPU_BUILDS: tuple[GpuBuild, ...] = (            # newest first
    GpuBuild(
        cuda="cu129", label="CUDA 12.9",
        min_driver={"linux": (575, 51), "win": (576, 2)},
        archs={"linux": frozenset({75, 80, 86, 90, 100, 120}), "win": frozenset({75, 80, 86, 89, 120})},
        known_good={"linux": frozenset({89})},
        download_bytes={"linux": int(5.4 * GB), "win": int(3.05 * GB)},
    ),
    GpuBuild(
        cuda="cu126", label="CUDA 12.6",
        min_driver={"linux": (560, 28), "win": (560, 76)},
        archs={"linux": frozenset({61, 70, 75, 80, 86, 89, 90}), "win": frozenset({75, 80, 86, 89})},
        download_bytes={"linux": int(3.93 * GB), "win": int(2.19 * GB)},
    ),
    GpuBuild(
        cuda="cu118", label="CUDA 11.8",
        min_driver={"linux": (520, 61), "win": (520, 6)},
        archs={"linux": frozenset({60, 61, 70, 75, 80, 86}), "win": frozenset({61, 70, 75, 80, 86})},
        download_bytes={"linux": int(3.15 * GB), "win": int(2.44 * GB)},
    ),
)
GPU_OSES = ("linux", "win")
GPU_ARCH = "x86_64"
CPU = "cpu"
CPU_DOWNLOAD_BYTES = {"linux": int(0.21 * GB), "win": int(0.12 * GB), "mac": int(0.12 * GB)}


def build_for(variant: str) -> GpuBuild | None:
    return next((build for build in GPU_BUILDS if build.cuda == variant), None)


def download_bytes(variant: str, os_name: str) -> int:
    build = build_for(variant)
    if build is None:
        return CPU_DOWNLOAD_BYTES.get(os_name, CPU_DOWNLOAD_BYTES["linux"])
    return build.download_bytes.get(os_name, 0)


def variant_label(variant: str) -> str:
    """"GPU build (CUDA 12.9)" or "CPU build"."""
    build = build_for(variant)
    return f"GPU build ({build.label})" if build else "CPU build"


def version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.strip().split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _format_version(version: tuple[int, ...]) -> str:
    return ".".join(f"{part:02d}" if i else str(part) for i, part in enumerate(version))


@dataclass(frozen=True)
class Choice:
    variant: str                    # "cpu" or a GpuBuild.cuda
    reason: str                     # one line: why this variant
    gpu: GpuInfo | None = None

    @property
    def is_gpu(self) -> bool:
        return self.variant != CPU


def select_variant(os_name: str, arch: str, probe: GpuProbe) -> Choice:
    """The build to install on this machine (pure; see the module docstring)."""
    if os_name not in GPU_OSES:
        return Choice(CPU, "paddle has no GPU build for macOS")
    if arch != GPU_ARCH:
        return Choice(CPU, f"paddle has no GPU build for {arch}")
    gpu = probe.primary
    if gpu is None:
        return Choice(CPU, probe.error or "no NVIDIA GPU found")
    driver = gpu.driver_version
    if not driver:
        return Choice(CPU, f"could not read the driver version of {gpu.name}", gpu)
    sm = gpu.sm
    fits_arch = [build for build in GPU_BUILDS if sm is None or build.covers(os_name, sm)]
    for build in fits_arch:
        if driver >= build.min_driver[os_name]:
            return Choice(build.cuda, f"{gpu.name} with driver {gpu.driver} runs the {build.label} build", gpu)
    if not fits_arch:
        return Choice(CPU, f"no paddle GPU build supports {gpu.name} (compute capability {gpu.compute_cap})", gpu)
    lowest = min(fits_arch, key=lambda build: build.min_driver[os_name])
    return Choice(CPU, f"{gpu.name} needs NVIDIA driver {_format_version(lowest.min_driver[os_name])} or newer "
                       f"for a GPU build ({lowest.label}); this driver is {gpu.driver}", gpu)


def parse_nvidia_smi(text: str) -> tuple[GpuInfo, ...]:
    """Rows of `--query-gpu=name,driver_version[,compute_cap] --format=csv,noheader`."""
    gpus = []
    for line in text.splitlines():
        fields = [piece.strip() for piece in line.split(",")]
        if len(fields) < 2 or not fields[0] or not version_tuple(fields[1]):
            continue
        cap = fields[2] if len(fields) > 2 and version_tuple(fields[2]) else ""
        gpus.append(GpuInfo(fields[0], fields[1], cap))
    return tuple(gpus)


def _no_window_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0


Run = Callable[..., "subprocess.CompletedProcess[str]"]


def probe_gpus(os_name: str, *, run: Run = subprocess.run,
               which: Callable[[str], str | None] = shutil.which) -> GpuProbe:
    """Ask nvidia-smi which NVIDIA GPUs this machine has. Never raises."""
    if os_name not in GPU_OSES:
        return GpuProbe(error="macOS has no NVIDIA GPU support")
    exe = which(NVIDIA_SMI)
    if not exe:
        return GpuProbe(error="no NVIDIA driver found (nvidia-smi is not on PATH)")
    last_error = ""
    for fields in (QUERY_FIELDS, "name,driver_version"):    # old drivers cannot report compute_cap
        try:
            done = run([exe, f"--query-gpu={fields}", "--format=csv,noheader"], capture_output=True, text=True,
                       timeout=NVIDIA_SMI_TIMEOUT, check=False, creationflags=_no_window_flags())
        except (OSError, subprocess.SubprocessError) as exc:
            return GpuProbe(error=f"nvidia-smi failed: {exc}")
        if done.returncode == 0:
            gpus = parse_nvidia_smi(done.stdout or "")
            if gpus:
                return GpuProbe(gpus)
            last_error = "nvidia-smi reported no GPU"
        else:
            last_error = (f"nvidia-smi exited with {done.returncode}: "
                          f"{(done.stdout or '').strip() or (done.stderr or '').strip()}").strip()
    return GpuProbe(error=last_error or "nvidia-smi reported no GPU")


def describe(probe: GpuProbe) -> str:
    """The hardware line the setup dialog shows."""
    gpu = probe.primary
    if gpu is None:
        return probe.error or "No NVIDIA GPU"
    return f"{gpu.name} · driver {gpu.driver}" + (f" · compute {gpu.compute_cap}" if gpu.compute_cap else "")


def candidates(os_name: str, arch: str, probe: GpuProbe) -> Sequence[str]:
    """Variants the user may pick: the recommended GPU build (if any), then CPU."""
    choice = select_variant(os_name, arch, probe)
    return (choice.variant, CPU) if choice.is_gpu else (CPU,)
