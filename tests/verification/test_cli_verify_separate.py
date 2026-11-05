"""``physverify verify-separate`` over the fixture clip: R1_dyn rows only, never scored."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from cli_runner import CliResult, CliRunner
from synthetic_clip import SyntheticFixtureDataset

from physverify.app import plugins
from physverify.calibration.row_sigma import ROW_SIGMA_FILE, fit_row_sigma
from physverify.runtime.heavy import holder_path
from physverify.schema.checks import InputLevel, Verdict
from physverify.verification.runner import TABLE_FILE, RowStatus

COMMAND = "verify-separate"
DATASET = ("--dataset", SyntheticFixtureDataset.name)
EXIT_USAGE = 2


@pytest.fixture(autouse=True)
def fixture_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    kinds: Mapping[str, Any] = plugins.PLUGIN_KINDS
    datasets = kinds["datasets"]
    patched = {
        **kinds,
        "datasets": lambda: {**datasets(), SyntheticFixtureDataset.name: SyntheticFixtureDataset},
    }
    monkeypatch.setattr(plugins, "PLUGIN_KINDS", MappingProxyType(patched))


def run_dir(result: CliResult) -> Path:
    (line,) = [line for line in result.out.splitlines() if line.startswith("run: ")]
    return Path(line.removeprefix("run: "))


def table(result: CliResult) -> list[dict[str, str]]:
    with (run_dir(result) / TABLE_FILE).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verdicts(result: CliResult) -> set[str]:
    return {row["verdict"] for row in table(result)}


def test_the_oracle_run_writes_only_the_separate_rows(cli: CliRunner, lock_path: Path) -> None:
    result = cli(COMMAND, *DATASET, "--level", "oracle")
    assert result.code == 0, result.err
    assert [row["check_id"] for row in table(result)] == ["R1_dyn"]
    assert verdicts(result) <= {Verdict.UNSCORED.value, Verdict.ABSTAIN.value}
    assert not holder_path(lock_path).exists()


def test_a_run_with_ground_truth_stages_takes_the_substitution_path(cli: CliRunner) -> None:
    result = cli(COMMAND, *DATASET, "--level", "perceived", "--gt", "depth")
    assert result.code == 0, result.err
    assert {row["status"] for row in table(result)} <= {
        RowStatus.ABSTAINED.value,
        RowStatus.NO_EVENT.value,
        RowStatus.SCORED.value,
    }
    assert verdicts(result) <= {Verdict.UNSCORED.value, Verdict.ABSTAIN.value}


@pytest.mark.parametrize(
    "argv", [("--level", "oracle", "--gt", "depth"), ()], ids=["gt_on_oracle", "no_level"]
)
def test_usage_errors(cli: CliRunner, argv: tuple[str, ...]) -> None:
    assert cli(COMMAND, *DATASET, *argv).code == EXIT_USAGE


def test_the_row_sigma_is_not_fitted_on_oracle_inputs(cli: CliRunner) -> None:
    assert cli(COMMAND, *DATASET, "--level", "oracle", "--fit-row-sigma").code == EXIT_USAGE


def test_the_row_sigma_is_fitted_on_calib_clips_only(cli: CliRunner) -> None:
    result = cli(COMMAND, *DATASET, "--level", "perceived", "--gt", "depth", "--fit-row-sigma")
    assert result.code == EXIT_USAGE
    assert "calib clips only" in result.err


def row_sigma_file(tmp_path: Path, level: InputLevel, gt: tuple[str, ...]) -> str:
    fitted = fit_row_sigma(
        {f"c{j}": np.array([0.5, 1.0 + j, -2.0]) for j in range(5)},
        {f"c{j}": f"f-s{j:03d}" for j in range(5)},
        check_id="R1_dyn",
        level=level,
        gt_stages=gt,
    )
    path = tmp_path / ROW_SIGMA_FILE
    path.write_text(fitted.model_dump_json(by_alias=True), encoding="utf-8")
    return str(path)


def test_a_row_sigma_artifact_runs_at_its_own_level(cli: CliRunner, tmp_path: Path) -> None:
    artifact = row_sigma_file(tmp_path, InputLevel.MIXED, ("depth",))
    used = cli(COMMAND, *DATASET, "--level", "perceived", "--gt", "depth", "--row-sigma", artifact)
    assert used.code == 0, used.err
    assert {row["check_id"] for row in table(used)} == {"R1_dyn"}
    other = cli(COMMAND, *DATASET, "--level", "perceived", "--row-sigma", artifact)
    assert other.code == EXIT_USAGE
