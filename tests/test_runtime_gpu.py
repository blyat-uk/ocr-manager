"""core.runtime.gpu: the nvidia-smi probe and the table-driven variant choice."""
from __future__ import annotations

import subprocess

import pytest

from core.runtime.gpu import (
    CPU,
    GPU_BUILDS,
    GpuInfo,
    GpuProbe,
    candidates,
    describe,
    download_bytes,
    parse_nvidia_smi,
    probe_gpus,
    select_variant,
    variant_label,
    version_tuple,
)


def probe(name="NVIDIA GeForce RTX 4090", driver="580.65.06", cap="8.9") -> GpuProbe:
    return GpuProbe((GpuInfo(name, driver, cap),))


# (case, os, arch, gpu name, driver, compute cap, expected variant)
CASES = [
    ("4090 linux driver 580 -> cu129 (sm_89 known good)", "linux", "x86_64", "RTX 4090", "580.65.06", "8.9", "cu129"),
    ("4090 windows driver 580 -> cu129 (exact sm_89)", "win", "x86_64", "RTX 4090", "580.88", "8.9", "cu129"),
    ("4090 linux driver 565 -> cu126", "linux", "x86_64", "RTX 4090", "565.57.01", "8.9", "cu126"),
    ("4090 windows driver 565 -> cu126", "win", "x86_64", "RTX 4090", "565.90", "8.9", "cu126"),
    ("5090 linux old driver 570 -> cpu", "linux", "x86_64", "RTX 5090", "570.86.10", "12.0", CPU),
    ("5090 windows old driver 572 -> cpu", "win", "x86_64", "RTX 5090", "572.16", "12.0", CPU),
    ("5090 linux driver at the cu129 minimum", "linux", "x86_64", "RTX 5090", "575.51.03", "12.0", "cu129"),
    ("5090 windows driver 576.02 exactly", "win", "x86_64", "RTX 5090", "576.02", "12.0", "cu129"),
    ("5090 windows driver 576.01 -> cpu", "win", "x86_64", "RTX 5090", "576.01", "12.0", CPU),
    ("GTX 1080 windows -> cu118 (only win build with sm_61)", "win", "x86_64", "GTX 1080", "580.88", "6.1", "cu118"),
    ("GTX 1080 linux -> cu126 (linux cu126 has sm_61)", "linux", "x86_64", "GTX 1080", "580.65.06", "6.1", "cu126"),
    ("RTX 3080 linux driver 550 -> cu118", "linux", "x86_64", "RTX 3080", "550.54.14", "8.6", "cu118"),
    ("RTX 3080 linux driver 560.35 -> cu126", "linux", "x86_64", "RTX 3080", "560.35.03", "8.6", "cu126"),
    ("T4 windows at the cu118 minimum 520.06", "win", "x86_64", "Tesla T4", "520.06", "7.5", "cu118"),
    ("T4 windows below every minimum -> cpu", "win", "x86_64", "Tesla T4", "520.05", "7.5", CPU),
    ("P100 linux (sm_60 only in cu118)", "linux", "x86_64", "Tesla P100", "580.65.06", "6.0", "cu118"),
    ("P100 windows (no build with sm_60) -> cpu", "win", "x86_64", "Tesla P100", "580.88", "6.0", CPU),
    ("Kepler -> cpu", "linux", "x86_64", "Tesla K80", "470.256.02", "3.7", CPU),
    ("H100 linux -> cu129", "linux", "x86_64", "H100", "580.65.06", "9.0", "cu129"),
    ("H100 windows (no win build with sm_90) -> cpu", "win", "x86_64", "H100", "580.88", "9.0", CPU),
    ("B200 linux sm_100 -> cu129", "linux", "x86_64", "B200", "580.65.06", "10.0", "cu129"),
    ("unknown compute cap: by driver only", "linux", "x86_64", "Some GPU", "580.65.06", "", "cu129"),
    ("macOS -> cpu whatever the probe", "mac", "arm64", "RTX 4090", "580.65.06", "8.9", CPU),
    ("linux arm64 -> cpu", "linux", "arm64", "RTX 4090", "580.65.06", "8.9", CPU),
]


@pytest.mark.parametrize("case, os_name, arch, name, driver, cap, expected", CASES, ids=[c[0] for c in CASES])
def test_select_variant_table(case, os_name, arch, name, driver, cap, expected):
    choice = select_variant(os_name, arch, probe(name, driver, cap))
    assert choice.variant == expected, choice.reason
    assert choice.reason
    assert choice.is_gpu == (expected != CPU)


def test_no_nvidia_smi_means_cpu_with_the_probe_error_as_reason():
    choice = select_variant("linux", "x86_64", GpuProbe(error="no NVIDIA driver found"))
    assert choice.variant == CPU and choice.reason == "no NVIDIA driver found"
    assert select_variant("win", "x86_64", GpuProbe()).reason == "no NVIDIA GPU found"


def test_old_driver_reason_names_the_driver_to_install():
    reason = select_variant("linux", "x86_64", probe("RTX 5090", "570.86.10", "12.0")).reason
    assert "575.51" in reason and "570.86.10" in reason
    reason = select_variant("win", "x86_64", probe("RTX 3080", "511.79", "8.6")).reason
    assert "520.06" in reason                        # the lowest bar among the builds that fit sm_86


def test_the_first_gpu_decides():
    two = GpuProbe((GpuInfo("GTX 1080", "580.88", "6.1"), GpuInfo("RTX 4090", "580.88", "8.9")))
    assert select_variant("win", "x86_64", two).variant == "cu118"


def test_candidates_offer_the_recommended_gpu_build_then_cpu():
    assert tuple(candidates("linux", "x86_64", probe())) == ("cu129", CPU)
    assert tuple(candidates("mac", "arm64", probe())) == (CPU,)


def test_table_is_newest_first_and_complete():
    assert [build.cuda for build in GPU_BUILDS] == ["cu129", "cu126", "cu118"]
    for build in GPU_BUILDS:
        for os_name in ("linux", "win"):
            assert build.min_driver[os_name] and build.archs[os_name] and build.download_bytes[os_name] > 0
    assert download_bytes(CPU, "mac") < download_bytes("cu118", "win")
    assert variant_label("cu129") == "GPU build (CUDA 12.9)" and variant_label(CPU) == "CPU build"


def test_version_tuple():
    assert version_tuple("580.65.06") == (580, 65, 6)
    assert version_tuple("576.02") == (576, 2)
    assert version_tuple("") == () and version_tuple("N/A") == ()


def test_parse_nvidia_smi_rows():
    text = "NVIDIA GeForce RTX 4090, 580.65.06, 8.9\nNVIDIA GeForce GTX 1080, 580.65.06, 6.1\n\n"
    assert parse_nvidia_smi(text) == (GpuInfo("NVIDIA GeForce RTX 4090", "580.65.06", "8.9"),
                                      GpuInfo("NVIDIA GeForce GTX 1080", "580.65.06", "6.1"))
    assert parse_nvidia_smi("Tesla K80, 470.82.01\n") == (GpuInfo("Tesla K80", "470.82.01", ""),)
    assert parse_nvidia_smi("No devices were found\n") == ()
    assert parse_nvidia_smi("RTX, [N/A], [N/A]") == ()
    assert GpuInfo("x", "1", "12.0").sm == 120 and GpuInfo("x", "1", "").sm is None


class FakeRun:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        code, out = answer
        return subprocess.CompletedProcess(command, code, out, "")


def test_probe_uses_nvidia_smi_with_a_timeout():
    run = FakeRun((0, "NVIDIA GeForce RTX 4090, 580.65.06, 8.9\n"))
    result = probe_gpus("linux", run=run, which=lambda name: "/usr/bin/nvidia-smi")
    assert result.primary == GpuInfo("NVIDIA GeForce RTX 4090", "580.65.06", "8.9")
    command, kwargs = run.calls[0]
    assert command == ["/usr/bin/nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]
    assert kwargs["timeout"] == 5.0 and kwargs["capture_output"] and not kwargs["check"]


def test_probe_retries_without_compute_cap_for_old_drivers():
    run = FakeRun((6, 'Field "compute_cap" is not a valid field to query.'), (0, "Tesla K80, 470.82.01\n"))
    result = probe_gpus("win", run=run, which=lambda name: "nvidia-smi.exe")
    assert result.primary == GpuInfo("Tesla K80", "470.82.01", "")
    assert run.calls[1][0][1] == "--query-gpu=name,driver_version"


def test_probe_failures_never_raise():
    assert "not on PATH" in probe_gpus("linux", which=lambda name: None).error
    assert probe_gpus("mac", run=FakeRun(), which=lambda name: "x").gpus == ()
    timed_out = probe_gpus("linux", run=FakeRun(subprocess.TimeoutExpired("nvidia-smi", 5)), which=lambda n: "x")
    assert timed_out.gpus == () and "nvidia-smi failed" in timed_out.error
    failing = probe_gpus("linux", run=FakeRun((9, "NVIDIA-SMI has failed"), (9, "NVIDIA-SMI has failed")),
                         which=lambda n: "x")
    assert failing.gpus == () and "NVIDIA-SMI has failed" in failing.error
    assert probe_gpus("linux", run=FakeRun(OSError("denied")), which=lambda n: "x").error


def test_describe():
    assert describe(probe()) == "NVIDIA GeForce RTX 4090 · driver 580.65.06 · compute 8.9"
    assert describe(GpuProbe(error="no NVIDIA driver found")) == "no NVIDIA driver found"
