"""Environment of a timing run and the validity rule (cuda + precise + a GPU)."""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from physverify.config.models import Settings
from physverify.timing.environment import (
    DRIVER_QUERY,
    PACKAGES,
    PROCESSES_QUERY,
    collect_environment,
    cpu_model,
    nvidia_smi,
    run_environment,
    validity,
)
from physverify.timing.records import (
    SCHEMA_VERSION,
    MeasureStats,
    ResultsFile,
    TargetOutcome,
    TargetResult,
    TargetStatus,
)
from physverify.timing.settings import TargetRunSettings

SettingsFactory = Callable[..., Settings]

ENVIRONMENT_KEYS = {
    "os",
    "python",
    "cpu_model",
    "logical_cpus",
    "usable_cpus",
    "ram_total_mb",
    "threads",
    "device",
    "profile",
    "packages",
    "cuda_available",
    "gpus",
    "cuda",
    "cudnn",
    "driver",
    "gpu_processes_at_start",
}
H200 = {"index": 0, "name": "NVIDIA H200", "total_mb": 143771.0, "capability": "9.0"}
OTHER_PROCESS = "4242, python, 1000 MiB"


class FakeSmi:
    """Stands in for ``subprocess.run``; answers the driver and process queries."""

    def __init__(self, returncode: int = 0, error: BaseException | None = None) -> None:
        self.returncode = returncode
        self.error = error
        self.queries: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        query = tuple(argv[1:])
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        stdout = {
            DRIVER_QUERY: "570.86.15\n",
            PROCESSES_QUERY: f"{os.getpid()}, python, 500 MiB\n\n{OTHER_PROCESS}\n",
        }[query]
        return subprocess.CompletedProcess(list(argv), self.returncode, stdout, "")


@pytest.fixture
def cpuinfo(tmp_path: Path) -> Path:
    path = tmp_path / "cpuinfo"
    path.write_text("processor\t: 0\nmodel name\t: Fake CPU @ 1.00GHz\n", encoding="utf-8")
    return path


def test_environment_without_cuda_never_calls_nvidia_smi(
    fake_probe: Callable[..., Any], cpuinfo: Path
) -> None:
    smi = FakeSmi()

    env = collect_environment("cpu", 4, fake_probe(), profile="fast", runner=smi, cpuinfo=cpuinfo)

    assert set(env) == ENVIRONMENT_KEYS
    assert set(env["packages"]) == set(PACKAGES)
    assert (env["device"], env["threads"], env["profile"]) == ("cpu", 4, "fast")
    assert env["cpu_model"] == "Fake CPU @ 1.00GHz"
    assert env["cuda_available"] is False
    assert env["gpus"] == []
    assert env["cuda"] is env["cudnn"] is env["driver"] is env["gpu_processes_at_start"] is None
    assert env["usable_cpus"] >= 1
    assert env["ram_total_mb"] > 0
    assert smi.queries == []


def test_environment_with_cuda_reports_gpus_driver_and_other_processes(
    fake_probe: Callable[..., Any], cpuinfo: Path
) -> None:
    probe = fake_probe(True, [H200], {"cuda": "12.8", "cudnn": "90100"})

    env = collect_environment("cuda", 8, probe, runner=FakeSmi(), cpuinfo=cpuinfo)

    assert set(env) == ENVIRONMENT_KEYS
    assert env["gpus"] == [H200]
    assert (env["cuda"], env["cudnn"], env["driver"]) == ("12.8", "90100", "570.86.15")
    assert env["gpu_processes_at_start"] == [OTHER_PROCESS]


def test_environment_with_cuda_but_no_nvidia_smi(
    fake_probe: Callable[..., Any], cpuinfo: Path
) -> None:
    smi = FakeSmi(error=FileNotFoundError("nvidia-smi"))

    env = collect_environment("cuda", 8, fake_probe(True, [H200]), runner=smi, cpuinfo=cpuinfo)

    assert env["driver"] is None
    assert env["gpu_processes_at_start"] is None
    assert smi.queries == [DRIVER_QUERY, PROCESSES_QUERY]


@pytest.mark.parametrize(
    "smi",
    [
        FakeSmi(returncode=9),
        FakeSmi(error=FileNotFoundError("nvidia-smi")),
        FakeSmi(error=subprocess.TimeoutExpired("nvidia-smi", 30)),
    ],
)
def test_nvidia_smi_failures_give_none(smi: FakeSmi) -> None:
    assert nvidia_smi(DRIVER_QUERY, smi) is None


def test_nvidia_smi_drops_blank_lines() -> None:
    assert nvidia_smi(PROCESSES_QUERY, FakeSmi()) == [
        f"{os.getpid()}, python, 500 MiB",
        OTHER_PROCESS,
    ]


def test_cpu_model_without_the_field_falls_back(tmp_path: Path) -> None:
    empty = tmp_path / "cpuinfo"
    empty.write_text("processor\t: 0\n", encoding="utf-8")

    assert cpu_model(empty) == cpu_model(tmp_path / "missing")


def test_run_environment_uses_the_effective_settings(
    make_settings: SettingsFactory, fake_probe: Callable[..., Any]
) -> None:
    settings = make_settings("runtime.device=cuda:1", "runtime.threads=2", profile="precise")

    env = run_environment(settings, fake_probe())

    assert (env["device"], env["threads"], env["profile"]) == ("cuda:1", 2, "precise")


@pytest.mark.parametrize(
    ("device", "profile", "gpus", "reasons"),
    [
        ("cpu", "precise", [H200], ["device is 'cpu', not cuda"]),
        ("cuda", "fast", [H200], ["profile is 'fast', not 'precise'"]),
        ("cuda", "precise", [], ["no CUDA GPU detected"]),
        (
            "cpu",
            "fast",
            [],
            [
                "device is 'cpu', not cuda",
                "profile is 'fast', not 'precise'",
                "no CUDA GPU detected",
            ],
        ),
        ("cuda", "precise", [H200], []),
        ("cuda:0", "precise", [H200], []),
    ],
)
def test_validity(device: str, profile: str, gpus: list[Any], reasons: list[str]) -> None:
    assert validity(device, profile, {"gpus": gpus}, "precise") == reasons


def _results(valid: bool, reasons: list[str], targets: list[TargetResult]) -> ResultsFile:
    return ResultsFile(
        valid_measurement=valid,
        validity_reasons=reasons,
        run={"run_id": "r"},
        environment={"device": "cpu"},
        targets=targets,
    )


@pytest.mark.parametrize(("valid", "reasons"), [(True, ["no CUDA GPU detected"]), (False, [])])
def test_results_file_ties_validity_to_its_reasons(valid: bool, reasons: list[str]) -> None:
    with pytest.raises(ValidationError, match="valid_measurement"):
        _results(valid, reasons, [])


def test_results_file_round_trips_nan_statistics() -> None:
    outcome = TargetOutcome(
        name="fake",
        requires_gpu=False,
        status=TargetStatus.MEASURED,
        reason=None,
        run=TargetRunSettings(),
        clip_ids=("c0",),
        setup={"setup.prepare_s": 0.5},
        n_calls=1,
    )
    single = MeasureStats(
        target="fake",
        clip_id="c0",
        measure="fake.wall_s",
        unit="s",
        n=1,
        median=1.0,
        p25=1.0,
        p75=1.0,
        iqr=0.0,
        iqr_rel=0.0,
        mean=1.0,
        sd=math.nan,
        min=1.0,
        max=1.0,
        unstable=False,
    )
    results = _results(
        False, ["device is 'cpu', not cuda"], [TargetResult.from_outcome(outcome, [single])]
    )

    loaded = ResultsFile.model_validate_json(results.model_dump_json())

    assert loaded.schema_version == SCHEMA_VERSION
    assert math.isnan(loaded.targets[0].stats[0].sd)
    assert loaded.targets[0].outcome() == outcome
