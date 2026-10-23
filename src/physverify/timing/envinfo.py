"""Hardware and software of a timing run, and the rule that decides whether it is a measurement.

Only a run with a CUDA device, the ``precise`` profile and a GPU present is a cost measurement
(the university H200); anything else is a harness check.
"""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from subprocess import CompletedProcess
from types import ModuleType
from typing import Any, Final, Protocol

import psutil

from physverify.config.models import Settings
from physverify.runtime.environment import package_version
from physverify.timing.records import MIB

PACKAGES: Final = (
    "physverify",
    "numpy",
    "opencv-python-headless",
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "diffusers",
    "psutil",
)
SMI_TIMEOUT_S: Final = 30.0
DRIVER_QUERY: Final = ("--query-gpu=driver_version", "--format=csv,noheader")
PROCESSES_QUERY: Final = (
    "--query-compute-apps=pid,process_name,used_memory",
    "--format=csv,noheader",
)
CPUINFO: Final = Path("/proc/cpuinfo")

Runner = Callable[..., CompletedProcess[str]]


class CudaProbe(Protocol):
    def available(self) -> bool: ...

    def devices(self) -> list[dict[str, Any]]: ...  # name, total_mb, capability

    def versions(self) -> dict[str, str | None]: ...  # cuda, cudnn


def _torch() -> ModuleType | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


class TorchCudaProbe:
    """Imports torch inside its methods; reports no CUDA when torch is missing."""

    def available(self) -> bool:
        torch = _torch()
        return torch is not None and bool(torch.cuda.is_available())

    def devices(self) -> list[dict[str, Any]]:
        torch = _torch()
        if torch is None or not torch.cuda.is_available():
            return []
        return [_device_info(torch, index) for index in range(torch.cuda.device_count())]

    def versions(self) -> dict[str, str | None]:
        torch = _torch()
        if torch is None:
            return {"cuda": None, "cudnn": None}
        cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        return {"cuda": torch.version.cuda, "cudnn": None if cudnn is None else str(cudnn)}


def _device_info(torch: ModuleType, index: int) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(index)
    return {
        "index": index,
        "name": props.name,
        "total_mb": props.total_memory / MIB,
        "capability": f"{props.major}.{props.minor}",
    }


def nvidia_smi(query: Sequence[str], runner: Runner = subprocess.run) -> list[str] | None:
    """Non-empty output lines of ``nvidia-smi <query>``, or None when it is missing or fails."""
    try:
        done = runner(
            ["nvidia-smi", *query],
            capture_output=True,
            text=True,
            timeout=SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def cpu_model(cpuinfo: Path = CPUINFO) -> str | None:
    try:
        lines = cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return platform.processor() or None
    for line in lines:
        key, sep, value = line.partition(":")
        if sep and key.strip() == "model name":
            return value.strip()
    return platform.processor() or None


def _other_gpu_processes(runner: Runner) -> list[str] | None:
    lines = nvidia_smi(PROCESSES_QUERY, runner)
    if lines is None:
        return None
    own = str(os.getpid())  # this process may already hold a CUDA context
    return [line for line in lines if line.split(",", 1)[0].strip() != own]


def collect_environment(
    device: str,
    threads: int,
    probe: CudaProbe,
    *,
    profile: str | None = None,
    runner: Runner = subprocess.run,
    cpuinfo: Path = CPUINFO,
) -> dict[str, Any]:
    available = probe.available()
    versions = probe.versions() if available else {}
    driver = nvidia_smi(DRIVER_QUERY, runner) if available else None
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu_model(cpuinfo),
        "logical_cpus": os.cpu_count(),
        "usable_cpus": len(os.sched_getaffinity(0)),
        "ram_total_mb": psutil.virtual_memory().total / MIB,
        "threads": threads,
        "device": device,
        "profile": profile,
        "packages": {name: package_version(name) for name in PACKAGES},
        "cuda_available": available,
        "gpus": probe.devices() if available else [],
        "cuda": versions.get("cuda"),
        "cudnn": versions.get("cudnn"),
        "driver": driver[0] if driver else None,
        "gpu_processes_at_start": _other_gpu_processes(runner) if available else None,
    }


def run_environment(settings: Settings, probe: CudaProbe) -> dict[str, Any]:
    """The environment under the effective ``runtime.device``, ``runtime.threads`` and profile."""
    runtime = settings.runtime
    return collect_environment(runtime.device, runtime.threads, probe, profile=settings.profile)


def validity(
    device: str, profile: str, environment: Mapping[str, Any], valid_profile: str
) -> list[str]:
    """Why the run is not a cost measurement; empty means it is one."""
    reasons = []
    if not device.startswith("cuda"):
        reasons.append(f"device is {device!r}, not cuda")
    if profile != valid_profile:
        reasons.append(f"profile is {profile!r}, not {valid_profile!r}")
    if not environment.get("gpus"):
        reasons.append("no CUDA GPU detected")
    return reasons
